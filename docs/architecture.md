# Architecture Diagram

                   ┌──────────────────────┐
                   │   Streamlit App      │
                   │      (app.py)        │
                   └──────────┬───────────┘
                              │
                              │ imports
                              ▼
┌───────────────┐    ┌──────────────────┐    ┌────────────────┐
│  User input   │───▶│  predict.predict │───▶│   Verdict UI   │
└───────────────┘    └──────────┬───────┘    └────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    ▼                       ▼
         ┌────────────────────┐   ┌────────────────────┐
         │  TF-IDF Pipeline   │   │  Embeddings + LR   │
         │  (sklearn joblib)  │   │  (MiniLM + joblib) │
         └────────────────────┘   └────────────────────┘


Alternate access: FastAPI /predict wraps the same predict() function.

Data pipeline (offline, one-time):
  deepset/prompt-injections ───┐
                               ├──▶ dedup ──▶ split ──▶ CSV files
  jackhhao/jailbreak-classification ─┘


## Design notes

- Streamlit imports `predict()` directly rather than calling the FastAPI
  service over HTTP. keeps the HF Spaces deployment to one process, no
  networking between two containers.
- Models are loaded lazily and cached at module scope (see `features.py`'s
  `_EMBEDDER` pattern), so nothing gets reloaded per-request.
- The embedding model itself (`all-MiniLM-L6-v2`) is never committed:
  it's pulled from the HF Hub on first run and cached locally. Only the
  trained logistic regression head is serialized to `models/embed_lr.joblib`.