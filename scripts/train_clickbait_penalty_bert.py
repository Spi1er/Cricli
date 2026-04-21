#!/usr/bin/env python3
"""Fine-tune DistilBERT for clickbait penalty estimation.

Input:
  data/processed/clickbait_penalty_splits.csv

Output:
  models/clickbait_penalty_distilbert/

The model predicts P(clickbait | title). Use that probability as
`clickbait_penalty` in the headline reward function.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "clickbait_penalty_splits.csv"
DEFAULT_BASE_MODEL = PROJECT_ROOT / "models" / "base" / "distilbert-base-uncased-seqcls"
DEFAULT_OUT = PROJECT_ROOT / "models" / "clickbait_penalty_distilbert"


class TitleDataset(Dataset):
    def __init__(self, titles: list[str], labels: list[int], tokenizer, max_length: int) -> None:
        self.titles = titles
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.titles)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.titles[index],
            truncation=True,
            max_length=self.max_length,
        )
        encoded["labels"] = int(self.labels[index])
        return encoded


def infer_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def validate_device(device: str) -> str:
    if device == "auto":
        return infer_device()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is False.")
    if device == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_built()):
            raise RuntimeError("Requested --device mps, but this PyTorch build does not include MPS support.")
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "Requested --device mps, but torch.backends.mps.is_available() is False. "
                "Check Python/PyTorch install and macOS MPS support."
            )
        torch.ones(1, device="mps")
    if device not in {"cpu", "cuda", "mps"}:
        raise RuntimeError(f"Unsupported device: {device}")
    return device


def load_split(df: pd.DataFrame, split: str, max_rows: int | None, seed: int) -> pd.DataFrame:
    out = df[df["split"] == split].copy()
    if max_rows and len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=seed)
    return out.reset_index(drop=True)


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(labels, probs),
    }


def make_training_args(args: argparse.Namespace, device: str) -> TrainingArguments:
    kwargs = {
        "output_dir": str(args.out / "checkpoints"),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_total_limit": 2,
        "seed": args.seed,
        "report_to": [],
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "load_best_model_at_end": True,
        "save_strategy": "epoch",
    }

    # transformers renamed evaluation_strategy -> eval_strategy in newer releases.
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    if "use_mps_device" in params:
        kwargs["use_mps_device"] = device == "mps"
    if device == "cpu" and "no_cuda" in params:
        kwargs["no_cuda"] = True
    if device == "cpu" and "use_cpu" in params:
        kwargs["use_cpu"] = True

    return TrainingArguments(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=5293)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--max-train", type=int, help="Optional cap for quick smoke tests.")
    parser.add_argument("--max-val", type=int, help="Optional cap for quick smoke tests.")
    parser.add_argument("--max-test", type=int, help="Optional cap for quick smoke tests.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = validate_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    train_df = load_split(df, "train", args.max_train, args.seed)
    val_df = load_split(df, "val", args.max_val, args.seed)
    test_df = load_split(df, "test", args.max_test, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label={0: "non_clickbait", 1: "clickbait"},
        label2id={"non_clickbait": 0, "clickbait": 1},
    )

    train_ds = TitleDataset(train_df["title"].astype(str).tolist(), train_df["clickbait"].astype(int).tolist(), tokenizer, args.max_length)
    val_ds = TitleDataset(val_df["title"].astype(str).tolist(), val_df["clickbait"].astype(int).tolist(), tokenizer, args.max_length)
    test_ds = TitleDataset(test_df["title"].astype(str).tolist(), test_df["clickbait"].astype(int).tolist(), tokenizer, args.max_length)

    training_args = make_training_args(args, device)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    print(f"Device preference: {device}")
    print(f"Train/val/test rows: {len(train_ds):,}/{len(val_ds):,}/{len(test_ds):,}")
    trainer.train()

    val_metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="val")
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")

    pred = trainer.predict(test_ds)
    probs = torch.softmax(torch.tensor(pred.predictions), dim=-1).numpy()[:, 1]
    test_predictions = test_df[["title", "clickbait"]].copy()
    test_predictions["clickbait_probability"] = probs
    test_predictions["predicted_clickbait"] = (probs >= 0.5).astype(int)

    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    test_predictions.to_csv(args.out / "test_predictions.csv", index=False)

    metadata = {
        "model_type": "distilbert_sequence_classification",
        "base_model": str(args.base_model),
        "data_path": str(args.data),
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device_preference": device,
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "test_rows": len(test_ds),
    }
    metrics = {"validation": val_metrics, "test": test_metrics}
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    readme = [
        "# Clickbait Penalty DistilBERT",
        "",
        "Fine-tuned DistilBERT sequence classifier for estimating `P(clickbait | title)`.",
        "",
        "Use the clickbait probability as `clickbait_penalty` in the headline reward function.",
        "",
        "## Validation Metrics",
        "",
        "```json",
        json.dumps(val_metrics, indent=2),
        "```",
        "",
        "## Test Metrics",
        "",
        "```json",
        json.dumps(test_metrics, indent=2),
        "```",
        "",
    ]
    (args.out / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print("Saved model to", args.out)
    print("Validation metrics:", json.dumps(val_metrics, indent=2))
    print("Test metrics:", json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
