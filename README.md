# Blood Cell ViT — Live Demo (deployable web app)

A FastAPI backend (serves the trained ViT + a REST API) plus a single-file
HTML/CSS/JS dashboard frontend. Three modes, all live model inference:

1. **Single image** — upload one cell crop, get the predicted class + confidence.
2. **Bring your own dataset** — upload a batch of images (+ optional
   `labels.csv` with `filename,label[,group]`) and get live accuracy +
   Bland-Altman count-agreement benchmarking against your own reference labels.
3. **Built-in demo** — runs on a bundled 102-crop sample (8 real BCCD
   microscopy fields, grouped by source image) with known ground truth, so
   visitors with no data of their own can still see the full pipeline.

This already includes the trained checkpoint (`model/best_model.pt`) and a
vendored copy of Chart.js (`static/vendor/chart.umd.js`, no external CDN
dependency at runtime).

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
# open http://localhost:8000
```

## Deploy — see `DEPLOY.md` in the parent folder for exact copy-paste steps
(push to GitHub, then connect to Render as a free web service).
