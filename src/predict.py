import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.features import get_embeddings

MODELS_DIR = "models"
DISTILBERT_DIR = "models/distilbert"
DISTILBERT_MAX_LENGTH = 128
_MODEL_CACHE = {}


def _load_tfidf():
    if "tfidf" not in _MODEL_CACHE:
        _MODEL_CACHE["tfidf"] = joblib.load(f"{MODELS_DIR}/tfidf_lr.joblib")
    return _MODEL_CACHE["tfidf"]


def _load_embedding_clf():
    if "embedding" not in _MODEL_CACHE:
        _MODEL_CACHE["embedding"] = joblib.load(f"{MODELS_DIR}/embed_lr.joblib")
    return _MODEL_CACHE["embedding"]


def _load_distilbert():
    if "distilbert" not in _MODEL_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_DIR)
        model.eval()
        _MODEL_CACHE["distilbert"] = (tokenizer, model)
    return _MODEL_CACHE["distilbert"]


def _predict_tfidf(text):
    pipeline = _load_tfidf()
    label = int(pipeline.predict([text])[0])
    confidence = float(pipeline.predict_proba([text])[0][label])
    return label, confidence


def _predict_embedding(text):
    clf = _load_embedding_clf()
    embedding = get_embeddings([text])
    label = int(clf.predict(embedding)[0])
    confidence = float(clf.predict_proba(embedding)[0][label])
    return label, confidence


def _predict_distilbert(text):
    tokenizer, model = _load_distilbert()
    inputs = tokenizer(
        [text], truncation=True, padding=True,
        max_length=DISTILBERT_MAX_LENGTH, return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    label = int(torch.argmax(probs).item())
    confidence = float(probs[label].item())
    return label, confidence


def predict(text, model="embedding"):
    if model == "tfidf":
        label, confidence = _predict_tfidf(text)
    elif model == "embedding":
        label, confidence = _predict_embedding(text)
    elif model == "distilbert":
        label, confidence = _predict_distilbert(text)
    else:
        raise ValueError(f"Unknown model: {model}")

    return {
        "text": text,
        "model": model,
        "label": "injection" if label == 1 else "benign",
        "confidence": round(confidence, 4),
    }