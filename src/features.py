import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedder once at module scope to prevent reloading overhead
_EMBEDDER = None


def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


def get_embeddings(texts, show_progress_bar=True):
    embedder = get_embedder()
    return embedder.encode(
        list(texts),
        batch_size=32,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True
    )