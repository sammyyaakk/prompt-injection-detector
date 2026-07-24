import joblib

from src.features import get_embeddings

MODELS_DIR = "models"
_MODEL_CACHE = {}


def _load_tfidf():
    if "tfidf" not in _MODEL_CACHE:
        _MODEL_CACHE["tfidf"] = joblib.load(f"{MODELS_DIR}/tfidf_lr.joblib")
    return _MODEL_CACHE["tfidf"]


def _load_embedding_clf():
    if "embedding" not in _MODEL_CACHE:
        _MODEL_CACHE["embedding"] = joblib.load(f"{MODELS_DIR}/embed_lr.joblib")
    return _MODEL_CACHE["embedding"]


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


def predict(text, model="embedding"):
    if model == "tfidf":
        label, confidence = _predict_tfidf(text)
    elif model == "embedding":
        label, confidence = _predict_embedding(text)
    else:
        raise ValueError(f"Unknown model: {model}")

    return {
        "text": text,
        "model": model,
        "label": "injection" if label == 1 else "benign",
        "confidence": round(confidence, 4),
    }