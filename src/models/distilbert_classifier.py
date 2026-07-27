import os

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.data_loader import load_splits

MODEL_NAME = "distilbert-base-uncased"
MODELS_DIR = "models/distilbert"
CHECKPOINTS_DIR = "models/distilbert_checkpoints"
MAX_LENGTH = 128


class _PromptDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def _tokenize(tokenizer, df):
    encodings = tokenizer(
        df["text"].tolist(), truncation=True, padding="max_length",
        max_length=MAX_LENGTH, return_tensors="pt",
    )
    return _PromptDataset(encodings, df["label"].tolist())


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"f1": f1_score(labels, preds)}


def train_distilbert_model(train_df, val_df):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    train_dataset = _tokenize(tokenizer, train_df)
    val_dataset = _tokenize(tokenizer, val_df)

    args = TrainingArguments(
        output_dir=CHECKPOINTS_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        learning_rate=2e-5,
        weight_decay=0.01,
        report_to=[],
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_compute_metrics,
    )
    trainer.train()

    val_f1 = trainer.evaluate()["eval_f1"]

    os.makedirs(MODELS_DIR, exist_ok=True)
    trainer.save_model(MODELS_DIR)
    tokenizer.save_pretrained(MODELS_DIR)

    return val_f1


def main():
    train_df, val_df, _ = load_splits()
    val_f1 = train_distilbert_model(train_df, val_df)
    print(f"DistilBERT       val F1: {val_f1:.3f}")


if __name__ == "__main__":
    main()
