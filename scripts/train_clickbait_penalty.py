#!/usr/bin/env python3
"""Train a lightweight clickbait penalty model.

This intentionally avoids heavyweight dependencies so the first critic baseline
can run in a small course-project environment. It uses feature hashing over word
and character n-grams plus logistic regression trained with mini-batch SGD.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "clickbait_penalty_splits.csv"
DEFAULT_OUT = PROJECT_ROOT / "models" / "clickbait_penalty_hash_lr"


TOKEN_RE = re.compile(r"[a-z0-9']+")


def stable_hash(text: str, dim: int) -> int:
    # Reserve index 0 for the bias feature.
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


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def predict_proba_from_indices(weights: np.ndarray, indices: list[np.ndarray]) -> np.ndarray:
    logits = np.fromiter((weights[idx].sum() / math.sqrt(len(idx)) for idx in indices), dtype=np.float32)
    return sigmoid(logits).astype(np.float32)


def binary_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    y_pred = (probs >= threshold).astype(np.int32)
    y_true = y_true.astype(np.int32)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    accuracy = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    order = np.argsort(probs)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(probs) + 1)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos and n_neg:
        auc = (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    else:
        auc = float("nan")

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": float(auc),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "num_examples": int(len(y_true)),
    }


def choose_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, dict[str, float | int]]:
    best_threshold = 0.5
    best_metrics = binary_metrics(y_true, probs, best_threshold)
    best_f1 = float(best_metrics["f1"])
    for threshold in np.linspace(0.1, 0.9, 81):
        metrics = binary_metrics(y_true, probs, float(threshold))
        if float(metrics["f1"]) > best_f1:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_f1 = float(metrics["f1"])
    return best_threshold, best_metrics


def train(
    train_titles: list[str],
    train_labels: np.ndarray,
    val_titles: list[str],
    val_labels: np.ndarray,
    dim: int,
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    rng = np.random.default_rng(seed)
    weights = np.zeros(dim, dtype=np.float32)

    train_indices = [feature_indices(title, dim) for title in train_titles]
    val_indices = [feature_indices(title, dim) for title in val_titles]
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train_indices))
        epoch_lr = lr / math.sqrt(epoch)
        total_loss = 0.0

        for row_idx in order:
            idx = train_indices[row_idx]
            label = float(train_labels[row_idx])
            scale = math.sqrt(len(idx))
            logit = float(weights[idx].sum() / scale)
            prob = float(sigmoid(logit))
            error = prob - label
            total_loss += -(label * math.log(max(prob, 1e-7)) + (1 - label) * math.log(max(1 - prob, 1e-7)))

            weights[idx] *= 1.0 - epoch_lr * l2
            weights[idx] -= epoch_lr * error / scale

        val_probs = predict_proba_from_indices(weights, val_indices)
        _, val_metrics = choose_threshold(val_labels, val_probs)
        val_metrics = dict(val_metrics)
        val_metrics["epoch"] = epoch
        val_metrics["train_loss"] = total_loss / max(1, len(train_indices))
        history.append(val_metrics)
        print(
            f"epoch={epoch} loss={val_metrics['train_loss']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} val_auc={val_metrics['roc_auc']:.4f}"
        )

    return weights, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dim", type=int, default=2**18)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.12)
    parser.add_argument("--l2", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=5293)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    train_df = df[df["split"] == "train"].sample(frac=1, random_state=args.seed).reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    weights, history = train(
        train_titles=train_df["title"].astype(str).tolist(),
        train_labels=train_df["clickbait"].to_numpy(dtype=np.int32),
        val_titles=val_df["title"].astype(str).tolist(),
        val_labels=val_df["clickbait"].to_numpy(dtype=np.int32),
        dim=args.dim,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
    )

    val_indices = [feature_indices(title, args.dim) for title in val_df["title"].astype(str).tolist()]
    test_indices = [feature_indices(title, args.dim) for title in test_df["title"].astype(str).tolist()]
    val_probs = predict_proba_from_indices(weights, val_indices)
    threshold, val_metrics = choose_threshold(val_df["clickbait"].to_numpy(dtype=np.int32), val_probs)
    test_probs = predict_proba_from_indices(weights, test_indices)
    test_metrics = binary_metrics(test_df["clickbait"].to_numpy(dtype=np.int32), test_probs, threshold)

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "model_weights.npz", weights=weights)

    metadata = {
        "model_type": "hashed_ngram_logistic_regression",
        "feature_dim": args.dim,
        "features": ["bias", "word_unigram", "word_bigram", "char_3gram", "char_4gram", "char_5gram"],
        "threshold": threshold,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "l2": args.l2,
        "seed": args.seed,
        "data_path": str(args.data),
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.out / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    metrics = {"validation": val_metrics, "test": test_metrics}
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    pred_df = test_df[["title", "clickbait"]].copy()
    pred_df["clickbait_probability"] = test_probs
    pred_df["predicted_clickbait"] = (test_probs >= threshold).astype(int)
    pred_df.to_csv(args.out / "test_predictions.csv", index=False)

    report = [
        "# Clickbait Penalty Model",
        "",
        "Model: hashed n-gram logistic regression.",
        "",
        f"Decision threshold selected on validation split: `{threshold:.3f}`",
        "",
        "## Validation",
        "",
        json.dumps(val_metrics, indent=2),
        "",
        "## Test",
        "",
        json.dumps(test_metrics, indent=2),
        "",
    ]
    (args.out / "README.md").write_text("\n".join(report), encoding="utf-8")

    print("Saved model to", args.out)
    print("Validation metrics:", json.dumps(val_metrics, indent=2))
    print("Test metrics:", json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
