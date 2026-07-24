import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from src.data_loader import load_splits
from src.features import get_embeddings

MODELS_DIR = "models"


def train_tfidf_model(train_df, val_df):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            lowercase=True,
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])
    pipeline.fit(train_df["text"], train_df["label"])

    val_preds = pipeline.predict(val_df["text"])
    val_f1 = f1_score(val_df["label"], val_preds)

    joblib.dump(pipeline, f"{MODELS_DIR}/tfidf_lr.joblib")
    return val_f1


def train_embedding_model(train_df, val_df):
    train_embeddings = get_embeddings(train_df["text"])
    val_embeddings = get_embeddings(val_df["text"])

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    clf.fit(train_embeddings, train_df["label"])

    val_preds = clf.predict(val_embeddings)
    val_f1 = f1_score(val_df["label"], val_preds)

    joblib.dump(clf, f"{MODELS_DIR}/embed_lr.joblib")
    return val_f1


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    train_df, val_df, _ = load_splits()

    tfidf_f1 = train_tfidf_model(train_df, val_df)
    embed_f1 = train_embedding_model(train_df, val_df)

    print(f"TF-IDF + LR      val F1: {tfidf_f1:.3f}")
    print(f"Embeddings + LR  val F1: {embed_f1:.3f}")


if __name__ == "__main__":
    main()