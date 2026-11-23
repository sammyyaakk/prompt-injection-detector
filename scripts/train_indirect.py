"""Retrain the three classifiers on direct + indirect training data.

Training data = data/processed/train.csv (1,104 direct prompts) +
data/indirect/train.csv (1,500 documents). Same model configurations as the
direct study; DistilBERT reads up to 512 tokens instead of 128 so whole
documents fit. Validation F1 is reported on data/processed/val.csv.

Run:  python -m scripts.train_indirect               # all three
      python -m scripts.train_indirect --only tfidf  # just TF-IDF (no downloads)
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from src.indirect.models import MODEL_FILES

SEED = 42


def load_training_data():
    direct = pd.read_csv("data/processed/train.csv")[["text", "label"]]
    indirect = pd.read_csv("data/indirect/train.csv")[["text", "label"]]
    df = pd.concat([direct, indirect], ignore_index=True)
    df["text"] = df["text"].astype(str)
    val = pd.read_csv("data/processed/val.csv")[["text", "label"]]
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True), val


def train_tfidf(train, val):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000,
                                  lowercase=True, strip_accents="unicode")),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)),
    ])
    pipe.fit(train.text, train.label)
    joblib.dump(pipe, MODEL_FILES["tfidf_ind"])
    return f1_score(val.label, pipe.predict(val.text))


def train_embedding(train, val):
    from src.features import get_embeddings
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    clf.fit(get_embeddings(train.text), train.label)
    joblib.dump(clf, MODEL_FILES["embedding_ind"])
    return f1_score(val.label, clf.predict(get_embeddings(val.text)))


def train_distilbert(train, val, epochs=3):
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    name, out_dir, max_len = "distilbert-base-uncased", MODEL_FILES["distilbert_ind"], 512
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)

    class DS(torch.utils.data.Dataset):
        def __init__(self, df):
            self.enc = tokenizer(df.text.tolist(), truncation=True, padding="max_length",
                                 max_length=max_len, return_tensors="pt")
            self.labels = df.label.tolist()

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[i])
            return item

    def metrics(p):
        return {"f1": f1_score(p.label_ids, np.argmax(p.predictions, axis=-1))}

    args = TrainingArguments(
        output_dir="models/distilbert_indirect_checkpoints", num_train_epochs=epochs,
        per_device_train_batch_size=16, per_device_eval_batch_size=32,
        eval_strategy="epoch", save_strategy="no", logging_steps=50,
        learning_rate=2e-5, weight_decay=0.01, report_to=[], seed=SEED,
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=args, train_dataset=DS(train),
                      eval_dataset=DS(val), compute_metrics=metrics)
    trainer.train()
    f1 = trainer.evaluate()["eval_f1"]
    os.makedirs(out_dir, exist_ok=True)
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    return f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["tfidf", "embedding", "distilbert"], nargs="*")
    args = parser.parse_args()
    which = args.only or ["tfidf", "embedding", "distilbert"]
    train, val = load_training_data()
    print(f"training rows: {len(train)} ({train.label.sum()} injection)")
    fns = {"tfidf": train_tfidf, "embedding": train_embedding, "distilbert": train_distilbert}
    for name in which:
        print(f"{name:<11} (indirect-trained) val F1: {fns[name](train, val):.3f}")


if __name__ == "__main__":
    main()
