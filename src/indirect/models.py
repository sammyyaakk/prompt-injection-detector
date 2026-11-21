"""One interface for the six classifiers in the indirect study.

Each scorer maps a list of texts to an array of P(injection). Models trained
only on direct prompts are the originals; the `_ind` variants are retrained on
direct + indirect training data by `scripts/train_indirect.py`. Missing models
(e.g. DistilBERT weights not downloaded) return None so analyses can skip them.
"""
import os

import joblib
import numpy as np

MODELS_DIR = "models"
DOC_MAX_LENGTH = 512  # DistilBERT positional limit; benchmark docs fit inside it
BATCH_SIZE = 32

MODEL_FILES = {
    "tfidf": f"{MODELS_DIR}/tfidf_lr.joblib",
    "embedding": f"{MODELS_DIR}/embed_lr.joblib",
    "distilbert": f"{MODELS_DIR}/distilbert",
    "tfidf_ind": f"{MODELS_DIR}/tfidf_lr_indirect.joblib",
    "embedding_ind": f"{MODELS_DIR}/embed_lr_indirect.joblib",
    "distilbert_ind": f"{MODELS_DIR}/distilbert_indirect",
}
MODEL_LABELS = {
    "tfidf": "TF-IDF + LR",
    "embedding": "MiniLM + LR",
    "distilbert": "DistilBERT",
    "tfidf_ind": "TF-IDF + LR (indirect-trained)",
    "embedding_ind": "MiniLM + LR (indirect-trained)",
    "distilbert_ind": "DistilBERT (indirect-trained)",
}
MODEL_ORDER = list(MODEL_FILES)


def _distilbert_scorer(path):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()

    def score(texts):
        probs = []
        with torch.no_grad():
            for i in range(0, len(texts), BATCH_SIZE):
                enc = tokenizer(list(texts[i:i + BATCH_SIZE]), truncation=True, padding=True,
                                max_length=DOC_MAX_LENGTH, return_tensors="pt").to(device)
                probs.extend(torch.softmax(model(**enc).logits, dim=-1)[:, 1].cpu().tolist())
        return np.array(probs)

    score.tokenizer, score.model = tokenizer, model
    return score


def load_scorer(name):
    path = MODEL_FILES[name]
    if not os.path.exists(path):
        return None
    if name.startswith("tfidf"):
        pipe = joblib.load(path)
        scorer = lambda texts: pipe.predict_proba(list(texts))[:, 1]
        scorer.pipeline = pipe
        return scorer
    if name.startswith("embedding"):
        try:
            from src.features import get_embeddings
            clf = joblib.load(path)
            get_embeddings(["warm-up"], show_progress_bar=False)
        except Exception as exc:  # embedder download blocked, etc.
            print(f"[skip] {name}: {type(exc).__name__}: {exc}")
            return None
        return lambda texts: clf.predict_proba(get_embeddings(list(texts), show_progress_bar=False))[:, 1]
    try:
        return _distilbert_scorer(path)
    except Exception as exc:
        print(f"[skip] {name}: {type(exc).__name__}: {exc}")
        return None


def load_all(names=MODEL_ORDER):
    scorers = {}
    for name in names:
        s = load_scorer(name)
        if s is not None:
            scorers[name] = s
            print(f"[loaded] {MODEL_LABELS[name]}")
        else:
            print(f"[missing] {MODEL_LABELS[name]} ({MODEL_FILES[name]})")
    return scorers
