"""
app.py
------
FastAPI backend + static frontend for the live Blood Cell ViT demo.

Routes:
  GET  /                    -> the single-page frontend
  GET  /demo_images/...     -> bundled demo crop images (static)
  GET  /api/demo            -> runs the model on the bundled demo dataset
                                (grouped by source microscopy field) and
                                returns predictions + Bland-Altman benchmark
                                against the bundled ground truth
  POST /api/predict_single  -> classify one uploaded image
  POST /api/predict_batch   -> classify a batch of uploaded images
                                (optionally with a labels.csv for live
                                benchmarking against the visitor's own
                                reference counts, i.e. "any dataset")
"""

import csv
import io
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from inference import get_classifier, CLASSES
from benchmarking import benchmark_predictions

BASE_DIR = Path(__file__).parent
DEMO_DIR = BASE_DIR / "demo_data"

app = FastAPI(title="Blood Cell ViT Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/demo_images", StaticFiles(directory=str(DEMO_DIR)), name="demo_images")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/")
def index():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))


def _load_demo_labels():
    rows = list(csv.DictReader(open(DEMO_DIR / "labels.csv")))
    return rows


@app.get("/api/demo")
def run_demo():
    clf = get_classifier()
    rows = _load_demo_labels()

    images = []
    for r in rows:
        img = Image.open(DEMO_DIR / r["filename"])
        images.append(img)

    preds = clf.predict_batch(images)

    per_file = []
    predicted_records = []
    reference_records = []
    correct = 0
    for r, p in zip(rows, preds):
        per_file.append(
            {
                "filename": r["filename"],
                "url": f"/demo_images/{r['filename']}",
                "true_label": r["label"],
                "group": r["group"],
                "predicted_label": p["label"],
                "confidence": p["confidence"],
            }
        )
        predicted_records.append({"group": r["group"], "label": p["label"]})
        reference_records.append({"group": r["group"], "label": r["label"]})
        if p["label"] == r["label"]:
            correct += 1

    accuracy = correct / len(rows) if rows else None
    benchmark = benchmark_predictions(predicted_records, reference_records, CLASSES)

    return JSONResponse(
        {
            "n_images": len(rows),
            "accuracy": accuracy,
            "classes": CLASSES,
            "per_file": per_file,
            "benchmark": benchmark,
        }
    )


@app.post("/api/predict_single")
async def predict_single(file: UploadFile = File(...)):
    clf = get_classifier()
    content = await file.read()
    img = Image.open(io.BytesIO(content))
    result = clf.predict_image(img)
    return JSONResponse(result)


@app.post("/api/predict_batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    labels_csv: UploadFile | None = File(None),
    group_by: str = Form("batch"),
):
    """
    files: any number of cell-crop images (the visitor's own dataset).
    labels_csv (optional): CSV with columns `filename,label[,group]`. If
       provided, the app benchmarks predictions against these reference
       labels/counts (Bland-Altman), exactly as in the notebook, on
       whatever dataset the visitor uploaded.
    group_by: if labels_csv has no `group` column, all images are treated
       as a single group named `batch` (still gives an overall count
       comparison) unless the CSV supplies its own groups.
    """
    clf = get_classifier()

    images = []
    filenames = []
    for f in files:
        content = await f.read()
        images.append(Image.open(io.BytesIO(content)))
        filenames.append(f.filename)

    preds = clf.predict_batch(images)

    per_file = [
        {
            "filename": fn,
            "predicted_label": p["label"],
            "confidence": p["confidence"],
        }
        for fn, p in zip(filenames, preds)
    ]

    class_distribution = {c: 0 for c in CLASSES}
    for p in preds:
        class_distribution[p["label"]] += 1

    response = {
        "n_images": len(images),
        "classes": CLASSES,
        "per_file": per_file,
        "class_distribution": class_distribution,
        "benchmark": None,
        "accuracy": None,
    }

    if labels_csv is not None:
        raw = (await labels_csv.read()).decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(raw))
        label_map = {}
        group_map = {}
        for row in reader:
            label_map[row["filename"]] = row["label"]
            group_map[row["filename"]] = row.get("group") or group_by

        predicted_records, reference_records = [], []
        correct, matched = 0, 0
        for fn, p in zip(filenames, preds):
            if fn not in label_map:
                continue
            matched += 1
            group = group_map[fn]
            predicted_records.append({"group": group, "label": p["label"]})
            reference_records.append({"group": group, "label": label_map[fn]})
            if p["label"] == label_map[fn]:
                correct += 1

        if matched > 0:
            response["accuracy"] = correct / matched
            response["n_matched_labels"] = matched
            response["benchmark"] = benchmark_predictions(
                predicted_records, reference_records, CLASSES
            )

    return JSONResponse(response)


@app.get("/api/health")
def health():
    return {"status": "ok"}
