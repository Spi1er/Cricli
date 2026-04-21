#!/usr/bin/env python3
"""Train a small multi-dimensional headline reward critic.

This distills LLM-judge labels into a local DistilBERT critic.

Input:
  data/processed/headline_quality_reward_model_examples.jsonl

Output:
  models/headline_quality_reward_distilbert/

The model predicts six normalized reward dimensions:
faithfulness, clarity, specificity, attractiveness, non_clickbait, overall.
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.models.distilbert.modeling_distilbert import DistilBertModel, DistilBertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "headline_quality_reward_model_examples.jsonl"
DEFAULT_BASE_MODEL = PROJECT_ROOT / "models" / "base" / "distilbert-base-uncased-seqcls"
DEFAULT_OUT = PROJECT_ROOT / "models" / "headline_quality_reward_distilbert"

SCORE_FIELDS = ["faithfulness", "clarity", "specificity", "attractiveness", "non_clickbait", "overall"]


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def load_examples(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            dims = obj["reward_dimensions"]
            row = {
                "seed_id": int(obj["seed_id"]),
                "variant": obj["variant"],
                "summary": clean_text(obj["summary"]),
                "headline": clean_text(obj["headline"]),
                "category": clean_text(obj.get("category", "")),
                "clickbait_penalty": float(obj.get("clickbait_penalty", 0.0)),
            }
            for field in SCORE_FIELDS:
                row[field] = float(dims[field])
            rows.append(row)
    return pd.DataFrame(rows)


def split_by_seed(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    seed_ids = sorted(df["seed_id"].unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(seed_ids)

    n = len(seed_ids)
    train_ids = set(seed_ids[: int(n * 0.7)])
    val_ids = set(seed_ids[int(n * 0.7) : int(n * 0.85)])

    df = df.copy()
    df["split"] = "test"
    df.loc[df["seed_id"].isin(train_ids), "split"] = "train"
    df.loc[df["seed_id"].isin(val_ids), "split"] = "val"
    return df


class RewardDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        text = (
            f"Category: {row['category']}\n"
            f"Summary: {row['summary']}\n"
            f"Headline: {row['headline']}"
        )
        encoded = self.tokenizer(text, truncation=True, max_length=self.max_length)
        # Normalize 1-5 judge scores to 0-1 for stable regression.
        labels = np.asarray([(float(row[field]) - 1.0) / 4.0 for field in SCORE_FIELDS], dtype=np.float32)
        encoded["labels"] = labels
        return encoded


class DistilBertRewardRegressor(DistilBertPreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.num_labels = len(SCORE_FIELDS)
        self.distilbert = DistilBertModel(config)
        hidden_size = getattr(config, "dim", getattr(config, "hidden_size", 768))
        self.pre_classifier = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(getattr(config, "seq_classif_dropout", 0.2))
        self.regressor = nn.Linear(hidden_size, self.num_labels)
        self.loss_fn = nn.MSELoss()
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_state = outputs.last_hidden_state
        pooled = hidden_state[:, 0]
        pooled = torch.relu(self.pre_classifier(pooled))
        pooled = self.dropout(pooled)
        logits = torch.sigmoid(self.regressor(pooled))

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
        return SequenceClassifierOutput(loss=loss, logits=logits)


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
        raise RuntimeError("Requested --device cuda, but CUDA is unavailable.")
    if device == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("Requested --device mps, but MPS is unavailable.")
    return device


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
        "metric_for_best_model": "overall_mae",
        "greater_is_better": False,
        "load_best_model_at_end": True,
        "save_strategy": "epoch",
    }
    params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "epoch"
    if device == "cpu" and "use_cpu" in params:
        kwargs["use_cpu"] = True
    elif device == "cpu" and "no_cuda" in params:
        kwargs["no_cuda"] = True
    if "use_mps_device" in params:
        kwargs["use_mps_device"] = device == "mps"
    return TrainingArguments(**kwargs)


def compute_metrics(eval_pred) -> dict[str, float]:
    preds, labels = eval_pred
    preds_5 = preds * 4.0 + 1.0
    labels_5 = labels * 4.0 + 1.0
    metrics: dict[str, float] = {}
    for i, field in enumerate(SCORE_FIELDS):
        metrics[f"{field}_mae"] = mean_absolute_error(labels_5[:, i], preds_5[:, i])
        metrics[f"{field}_rmse"] = mean_squared_error(labels_5[:, i], preds_5[:, i]) ** 0.5
        try:
            metrics[f"{field}_r2"] = r2_score(labels_5[:, i], preds_5[:, i])
        except ValueError:
            metrics[f"{field}_r2"] = float("nan")
    metrics["macro_mae"] = float(np.mean([metrics[f"{field}_mae"] for field in SCORE_FIELDS]))
    return metrics


def predictions_dataframe(df: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    preds_5 = np.clip(preds * 4.0 + 1.0, 1.0, 5.0)
    for i, field in enumerate(SCORE_FIELDS):
        out[f"pred_{field}"] = preds_5[:, i]
        out[f"abs_error_{field}"] = (out[f"pred_{field}"] - out[field]).abs()
    out["pred_reward"] = (
        0.25 * out["pred_faithfulness"]
        + 0.15 * out["pred_clarity"]
        + 0.15 * out["pred_specificity"]
        + 0.20 * out["pred_attractiveness"]
        + 0.15 * out["pred_non_clickbait"]
        + 0.10 * out["pred_overall"]
        - 0.20 * out["clickbait_penalty"]
    )
    return out


def write_report(metrics: dict, out_dir: Path, df: pd.DataFrame) -> None:
    lines = [
        "# Headline Quality Reward Critic",
        "",
        "A DistilBERT-based multi-output regression critic distilled from LLM-judge labels.",
        "",
        "Important caveat: this is a tiny 300-example distillation set. Treat the model as a proof of concept, not a production-quality reward model.",
        "",
        "## Test Metrics",
        "",
        "| Dimension | MAE | RMSE | R2 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for field in SCORE_FIELDS:
        lines.append(
            f"| {field} | {metrics[f'test_{field}_mae']:.3f} | "
            f"{metrics[f'test_{field}_rmse']:.3f} | {metrics[f'test_{field}_r2']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"- Macro MAE: {metrics['test_macro_mae']:.3f}",
            "",
            "## Split Counts",
            "",
            "| Split | Rows |",
            "| --- | ---: |",
            *[f"| {split} | {count} |" for split, count in df["split"].value_counts().items()],
            "",
            "## Suggested Use",
            "",
            "Use this model as the first local multi-dimensional reward critic. For stronger results, expand the LLM-judge dataset beyond 100 seed examples and train a pairwise ranking critic as a second objective.",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=5293)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = validate_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    df = split_by_seed(load_examples(args.data), args.seed)
    df.to_csv(args.out / "training_data_with_splits.csv", index=False)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base = AutoModel.from_pretrained(args.base_model, attn_implementation="eager")
    base.config._attn_implementation = "eager"
    model = DistilBertRewardRegressor(base.config)
    model.distilbert.load_state_dict(base.state_dict(), strict=False)

    train_ds = RewardDataset(train_df, tokenizer, args.max_length)
    val_ds = RewardDataset(val_df, tokenizer, args.max_length)
    test_ds = RewardDataset(test_df, tokenizer, args.max_length)

    trainer_kwargs = {
        "model": model,
        "args": make_training_args(args, device),
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
    print(f"Train/val/test rows: {len(train_df):,}/{len(val_df):,}/{len(test_df):,}")
    trainer.train()

    val_metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="val")
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    pred = trainer.predict(test_ds)
    pred_df = predictions_dataframe(test_df, pred.predictions)

    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    pred_df.to_csv(args.out / "test_predictions.csv", index=False)

    metrics = {"validation": val_metrics, "test": test_metrics}
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metadata = {
        "model_type": "distilbert_multi_output_reward_regressor",
        "score_fields": SCORE_FIELDS,
        "base_model": str(args.base_model),
        "data": str(args.data),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "seed": args.seed,
        "device_preference": device,
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(test_metrics, args.out, df)

    print("Saved model to", args.out)
    print("Validation metrics:", json.dumps(val_metrics, indent=2))
    print("Test metrics:", json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
