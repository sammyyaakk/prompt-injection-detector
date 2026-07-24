import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.data_loader import load_splits
from src.features import get_embeddings

MODELS_DIR = "models"
HOLDOUT_PATH = "data/holdout/handcrafted.csv"


def _false_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) else 0.0


def _score(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "fpr": _false_positive_rate(y_true, y_pred),
    }


def evaluate_tfidf(pipeline, df):
    preds = pipeline.predict(df["text"])
    return _score(df["label"], preds)


def evaluate_embedding(clf, df):
    embeddings = get_embeddings(df["text"])
    preds = clf.predict(embeddings)
    return _score(df["label"], preds)


def cross_validate_tfidf(pipeline, train_df):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, train_df["text"], train_df["label"], cv=cv, scoring="f1")
    return scores.mean(), scores.std()


def cross_validate_embedding(clf, train_df):
    embeddings = get_embeddings(train_df["text"])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, embeddings, train_df["label"], cv=cv, scoring="f1")
    return scores.mean(), scores.std()


def _print_report(name, test_scores, holdout_scores, cv_mean, cv_std):
    print(name)
    print(f"  test:    {test_scores}")
    print(f"  holdout: {holdout_scores}")
    print(f"  5-fold CV F1: {cv_mean:.3f} +/- {cv_std:.3f}")


def main():
    train_df, _, test_df = load_splits()
    holdout_df = pd.read_csv(HOLDOUT_PATH)

    tfidf_pipeline = joblib.load(f"{MODELS_DIR}/tfidf_lr.joblib")
    embed_clf = joblib.load(f"{MODELS_DIR}/embed_lr.joblib")

    tfidf_cv_mean, tfidf_cv_std = cross_validate_tfidf(tfidf_pipeline, train_df)
    _print_report(
        "TF-IDF + LR",
        evaluate_tfidf(tfidf_pipeline, test_df),
        evaluate_tfidf(tfidf_pipeline, holdout_df),
        tfidf_cv_mean, tfidf_cv_std,
    )

    embed_cv_mean, embed_cv_std = cross_validate_embedding(embed_clf, train_df)
    _print_report(
        "Embeddings + LR",
        evaluate_embedding(embed_clf, test_df),
        evaluate_embedding(embed_clf, holdout_df),
        embed_cv_mean, embed_cv_std,
    )


if __name__ == "__main__":
    main()