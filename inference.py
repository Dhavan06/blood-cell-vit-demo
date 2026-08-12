"""
inference.py
-------------
Self-contained model loading + preprocessing for the web app (deliberately
does not import from the training scripts directory, so this folder can be
deployed on its own).
"""

import gc
from dataclasses import dataclass
from pathlib import Path

import torch

# Keep torch's intra-op thread pool small: on memory-constrained hosts
# (e.g. Render's free 512MB instances) extra OpenMP worker threads add
# allocator overhead for no real speedup on this tiny model.
torch.set_num_threads(1)

from PIL import Image
from torchvision import transforms
from transformers import ViTConfig, ViTForImageClassification

CLASSES = ["RBC", "WBC", "Platelets"]
IMAGE_SIZE = 96

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MODEL_PATH = Path(__file__).parent / "model" / "best_model.pt"

_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def _build_model():
    cfg = ViTConfig(
        image_size=IMAGE_SIZE,
        patch_size=8,
        num_channels=3,
        hidden_size=192,
        num_hidden_layers=6,
        num_attention_heads=3,
        intermediate_size=768,
        num_labels=len(CLASSES),
    )
    return ViTForImageClassification(cfg)


class CellClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _build_model()
        # Checkpoint is stored in fp16 to keep the repo small; cast back to
        # fp32 for CPU inference (fp16 matmul isn't well supported on CPU).
        state = torch.load(MODEL_PATH, map_location=self.device)
        state = {k: (v.float() if v.is_floating_point() else v) for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_image(self, pil_image: Image.Image):
        x = _transform(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(pixel_values=x).logits[0]
        probs = torch.softmax(logits, dim=0).cpu().tolist()
        pred_idx = int(torch.argmax(logits).item())
        return {
            "label": CLASSES[pred_idx],
            "confidence": probs[pred_idx],
            "probs": {c: p for c, p in zip(CLASSES, probs)},
        }

    @torch.no_grad()
    def predict_batch(self, pil_images, chunk_size=8):
        # Processed in small chunks (rather than one giant stacked tensor)
        # to keep peak memory bounded regardless of dataset size -- this
        # matters on memory-constrained deployments (e.g. Render's free
        # tier, 512MB RAM) where activation memory for a large batch can
        # push the process over the limit.
        if not pil_images:
            return []
        results = []
        for start in range(0, len(pil_images), chunk_size):
            chunk = pil_images[start : start + chunk_size]
            xs = torch.stack([_transform(im.convert("RGB")) for im in chunk]).to(self.device)
            logits = self.model(pixel_values=xs).logits
            probs = torch.softmax(logits, dim=1).cpu().tolist()
            for p in probs:
                pred_idx = int(max(range(len(p)), key=lambda i: p[i]))
                results.append(
                    {
                        "label": CLASSES[pred_idx],
                        "confidence": p[pred_idx],
                        "probs": {c: v for c, v in zip(CLASSES, p)},
                    }
                )
            del xs, logits
        gc.collect()
        return results


_classifier = None


def get_classifier() -> CellClassifier:
    global _classifier
    if _classifier is None:
        _classifier = CellClassifier()
    return _classifier
