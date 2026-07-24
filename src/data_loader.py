import os
import requests
import pandas as pd
from sklearn.model_selection import train_test_split

DATASETS_SERVER = "https://datasets-server.huggingface.co/rows"
PROCESSED_DIR = "data/processed"
ROWS_PER_PAGE = 100

TEST_FRACTION = 0.15
VAL_FRACTION_OF_REMAINDER = 0.1765


def _fetch_rows(dataset, config="default", split="train", max_rows=2000):
    all_rows = []
    offset = 0
    while offset < max_rows:
        params = {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": ROWS_PER_PAGE,
        }
        response = requests.get(DATASETS_SERVER, params=params, timeout=30)
        response.raise_for_status()
        rows = [item["row"] for item in response.json().get("rows", [])]
        if not rows:
            break
        all_rows.extend(rows)
        offset += ROWS_PER_PAGE
    return pd.DataFrame(all_rows)


def _load_prompt_injections():
    df = _fetch_rows("deepset/prompt-injections", max_rows=1000)
    if "prompt" in df.columns:
        df = df.rename(columns={"prompt": "text"})
    df["label"] = df["label"].astype(int)
    return df[["text", "label"]]


def _load_jailbreak_classification():
    df = _fetch_rows("jackhhao/jailbreak-classification", max_rows=2000)
    if "prompt" in df.columns:
        df = df.rename(columns={"prompt": "text"})

    injection_labels = {"jailbreak", "injection"}
    df["label"] = df["type"].str.lower().isin(injection_labels).astype(int)
    return df[["text", "label"]]


def _clean(df):
    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= 5]

    dedup_key = df["text"].str.lower()
    df = df.loc[~dedup_key.duplicated()]
    return df.reset_index(drop=True)


def build_dataset():
    injections_df = _load_prompt_injections()
    jailbreak_df = _load_jailbreak_classification()

    combined = pd.concat([injections_df, jailbreak_df], ignore_index=True)
    print(f"Combined rows before cleaning: {len(combined)}")

    combined = _clean(combined)
    print(f"Rows after cleaning + dedup: {len(combined)}")
    print(f"Class balance: {combined['label'].value_counts().to_dict()}")

    train_val, test = train_test_split(
        combined, test_size=TEST_FRACTION, stratify=combined["label"], random_state=42
    )
    train, val = train_test_split(
        train_val, test_size=VAL_FRACTION_OF_REMAINDER,
        stratify=train_val["label"], random_state=42
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train.to_csv(f"{PROCESSED_DIR}/train.csv", index=False)
    val.to_csv(f"{PROCESSED_DIR}/val.csv", index=False)
    test.to_csv(f"{PROCESSED_DIR}/test.csv", index=False)

    print(f"Saved splits -> train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train, val, test


def load_splits():
    paths = [f"{PROCESSED_DIR}/{name}.csv" for name in ("train", "val", "test")]
    if not all(os.path.exists(p) for p in paths):
        return build_dataset()
    return tuple(pd.read_csv(p) for p in paths)


if __name__ == "__main__":
    build_dataset()