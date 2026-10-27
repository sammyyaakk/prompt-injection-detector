"""Build a pilot indirect prompt injection benchmark: injection and benign
prompts from the test and holdout sets embedded inside short Wikipedia
paragraphs (SQuAD contexts), plus some unmodified paragraphs."""
import os
import random
import re

import pandas as pd
from transformers import AutoTokenizer

SEED = 42
SQUAD_PARQUET_URL = (
    "https://huggingface.co/datasets/rajpurkar/squad/resolve/main/"
    "plain_text/validation-00000-of-00001.parquet"
)
RAW_PATH = "data/raw/squad_validation_contexts.csv"
OUT_DIR = "data/indirect_pilot"
OUT_PATH = f"{OUT_DIR}/indirect_pilot.csv"

# Payloads come only from evaluation files -- never train.csv -- so the
# benchmark shares no text with anything the models were fit on.
PAYLOAD_SOURCES = {
    "test": "data/processed/test.csv",
    "holdout": "data/holdout/handcrafted.csv",
}

TOKENIZER_NAME = "distilbert-base-uncased"
N_HOSTS = 300
N_PLAIN = 50
MAX_HOST_TOKENS = 150
MAX_DOC_TOKENS = 400
POSITIONS = ["start", "middle", "end"]

# A sentence boundary needs a capital, digit or opening quote after it, and
# must not follow an abbreviation such as "U.S." or "Dr.".
_SENTENCE_END_RE = re.compile(
    r"(?<=[.!?])(?<!\b[A-Z]\.)(?<!\b(?:Mr|Dr|St|Mt|Jr|Sr|vs|No)\.)\s+(?=[\"'(\[A-Z0-9])"
)


def n_tokens(tokenizer, text):
    return len(tokenizer(text)["input_ids"])


def load_contexts():
    if os.path.exists(RAW_PATH):
        return pd.read_csv(RAW_PATH)

    # One parquet download instead of ~100 paged datasets-server requests,
    # which get rate-limited partway through.
    df = pd.read_parquet(SQUAD_PARQUET_URL)
    df = df.drop_duplicates("context")[["id", "title", "context"]].reset_index(drop=True)

    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    df.to_csv(RAW_PATH, index=False)
    return df


def select_hosts(contexts, tokenizer):
    contexts = contexts.copy()
    contexts["n_tokens"] = [n_tokens(tokenizer, t) for t in contexts["context"]]
    short = contexts[contexts["n_tokens"] <= MAX_HOST_TOKENS]
    print(f"SQuAD contexts: {len(contexts)} unique, {len(short)} with <= {MAX_HOST_TOKENS} tokens")
    return short.sample(n=N_HOSTS, random_state=SEED).reset_index(drop=True)


def load_payloads(tokenizer):
    frames = []
    for name, path in PAYLOAD_SOURCES.items():
        df = pd.read_csv(path)[["text", "label"]]
        df["text"] = df["text"].astype(str).str.strip()
        df["source_id"] = [f"{name}:{i}" for i in df.index]
        frames.append(df)
    payloads = pd.concat(frames, ignore_index=True)
    payloads["n_tokens"] = [n_tokens(tokenizer, t) for t in payloads["text"]]
    return payloads


def embed(host, payload, position):
    if position == "start":
        return f"{payload} {host}"
    if position == "end":
        return f"{host} {payload}"

    # Middle: insert at the sentence boundary closest to the halfway point,
    # falling back to the middle word for single-sentence paragraphs.
    sentences = _SENTENCE_END_RE.split(host)
    if len(sentences) > 1:
        cut = len(sentences) // 2
        return " ".join(sentences[:cut] + [payload] + sentences[cut:])
    words = host.split()
    cut = len(words) // 2
    return " ".join(words[:cut] + [payload] + words[cut:])


def embed_payloads(payloads, host_pool, tokenizer, rng):
    rows, skipped = [], []

    # Injections first (the scarcer class), longest payloads first within a
    # class, so long payloads get first pick of the short hosts they need.
    for label in (1, 0):
        subset = payloads[payloads["label"] == label]
        subset = subset.sample(frac=1, random_state=SEED).sort_values("n_tokens", ascending=False, kind="stable")

        placed = 0
        for payload in subset.itertuples(index=False):
            # Two special tokens are counted in both host and payload lengths.
            budget = MAX_DOC_TOKENS - payload.n_tokens + 2
            candidates = [h for h in host_pool if h.n_tokens <= budget]
            if not candidates:
                skipped.append(payload.source_id)
                continue

            host = rng.choice(candidates)
            position = POSITIONS[placed % len(POSITIONS)]
            text = embed(host.context, payload.text, position)
            if n_tokens(tokenizer, text) > MAX_DOC_TOKENS:
                skipped.append(payload.source_id)
                continue

            host_pool.remove(host)
            placed += 1
            rows.append({
                "text": text,
                "label": int(payload.label),
                "position": position,
                "source_id": f"{payload.source_id}|squad:{host.id}",
            })

    return rows, skipped


def plain_rows(host_pool, rng):
    plain = rng.sample(host_pool, min(N_PLAIN, len(host_pool)))
    return [
        {"text": h.context, "label": 0, "position": "none", "source_id": f"squad:{h.id}"}
        for h in plain
    ]


def main():
    rng = random.Random(SEED)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    hosts = select_hosts(load_contexts(), tokenizer)
    payloads = load_payloads(tokenizer)
    host_pool = list(hosts.itertuples(index=False))

    rows, skipped = embed_payloads(payloads, host_pool, tokenizer, rng)
    rows += plain_rows(host_pool, rng)

    pilot = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    pilot.to_csv(OUT_PATH, index=False)

    doc_tokens = [n_tokens(tokenizer, t) for t in pilot["text"]]
    print(f"Saved {OUT_PATH} ({len(pilot)} documents)")
    print(f"Payloads skipped (no host fits under {MAX_DOC_TOKENS} tokens): {len(skipped)}")
    print(f"Document length: max {max(doc_tokens)} tokens, median {int(pd.Series(doc_tokens).median())}")
    print(f"Class balance: {pilot['label'].value_counts().sort_index().to_dict()}")
    print("Position counts by label:")
    print(pd.crosstab(pilot["position"], pilot["label"], margins=True))


if __name__ == "__main__":
    main()
