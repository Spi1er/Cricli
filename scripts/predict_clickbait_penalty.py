#!/usr/bin/env python3
"""Predict clickbait penalty scores for one or more titles."""

from __future__ import annotations

import argparse
import json
import re
import zlib
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "clickbait_penalty_hash_lr"
TOKEN_RE = re.compile(r"[a-z0-9']+")


def stable_hash(text: str, dim: int) -> int:
    return 1 + (zlib.crc32(text.encode("utf-8")) % (dim - 1))


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def feature_indices(text: str, dim: int) -> np.ndarray:
    text = re.sub(r"\s+", " ", str(text).lower()).strip()
    tokens = tokenize(text)
    features = [0]
    for token in tokens:
        features.append(stable_hash(f"w1={token}", dim))
    for left, right in zip(tokens, tokens[1:]):
        features.append(stable_hash(f"w2={left}_{right}", dim))
    compact = f" {text} "
    for n in (3, 4, 5):
        if len(compact) >= n:
            for i in range(len(compact) - n + 1):
                features.append(stable_hash(f"c{n}={compact[i:i+n]}", dim))
    return np.unique(np.asarray(features, dtype=np.int32))


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -40, 40))))


def score_title(title: str, weights: np.ndarray, dim: int) -> float:
    idx = feature_indices(title, dim)
    return sigmoid(float(weights[idx].sum() / np.sqrt(len(idx))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("titles", nargs="*", help="Title strings to score.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--file", type=Path, help="Optional text file with one title per line.")
    args = parser.parse_args()

    metadata = json.loads((args.model / "metadata.json").read_text(encoding="utf-8"))
    weights = np.load(args.model / "model_weights.npz")["weights"]
    dim = int(metadata["feature_dim"])
    threshold = float(metadata["threshold"])

    titles = list(args.titles)
    if args.file:
        titles.extend(line.strip() for line in args.file.read_text(encoding="utf-8").splitlines() if line.strip())

    for title in titles:
        probability = score_title(title, weights, dim)
        output = {
            "title": title,
            "clickbait_probability": probability,
            "clickbait_penalty": probability,
            "predicted_label": "clickbait" if probability >= threshold else "non_clickbait",
            "threshold": threshold,
        }
        print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
