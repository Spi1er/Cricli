#!/usr/bin/env python3
"""Train a pairwise headline reward scorer from LLM-judge preferences.

The scorer learns score(summary, headline), optimized with:
  loss = -log sigmoid(score(chosen) - score(rejected))

This is a compact RLHF-style reward modeling proof of concept.
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
from sklearn.metrics import accuracy_score, roc_auc_score
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
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.distilbert.modeling_distilbert import DistilBertModel, DistilBertPreTrainedModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "headline_quality_pairwise_preferences.jsonl"
DEFAULT_BASE_MODEL = PROJECT_ROOT / "models" / "base" / "distilbert-base-uncased-seqcls"
DEFAULT_OUT = PROJECT_ROOT / "models" / "headline_pairwise_reward_distilbert"


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def load_pairs(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows.append(
                {
                    "seed_id": int(obj["seed_id"]),
                    "summary": clean_text(obj["summary"]),
                    "category": clean_text(obj.get("category", "")),
                    "chosen_title": clean_text(obj["chosen_title"]),
                    "rejected_title": clean_text(obj["rejected_title"]),
                    "chosen_variant": clean_text(obj.get("chosen_variant", "")),
                    "rejected_variant": clean_text(obj.get("rejected_variant", "")),
                    "chosen_overall": int(obj["chosen_scores"]["overall"]),
                    "rejected_overall": int(obj["rejected_scores"]["overall"]),
                    "judge_model": clean_text(obj.get("judge_model", "")),
                }
            )
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


class PairwiseDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def encode(self, row: pd.Series, title: str) -> dict[str, list[int]]:
        text = (
            f"Category: {row['category']}\n"
            f"Summary: {row['summary']}\n"
            f"Headline: {title}"
        )
        return self.tokenizer(text, truncation=True, max_length=self.max_length)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        chosen = self.encode(row, row["chosen_title"])
        rejected = self.encode(row, row["rejected_title"])
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "labels": 1,
        }


class PairwiseDataCollator:
    def __init__(self, tokenizer) -> None:
        self.pad = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        chosen = [
            {
                "input_ids": feature["chosen_input_ids"],
                "attention_mask": feature["chosen_attention_mask"],
            }
            for feature in features
        ]
        rejected = [
            {
                "input_ids": feature["rejected_input_ids"],
                "attention_mask": feature["rejected_attention_mask"],
            }
            for feature in features
        ]
        chosen_batch = self.pad(chosen)
        rejected_batch = self.pad(rejected)
        return {
            "chosen_input_ids": chosen_batch["input_ids"],
            "chosen_attention_mask": chosen_batch["attention_mask"],
            "rejected_input_ids": rejected_batch["input_ids"],
            "rejected_attention_mask": rejected_batch["attention_mask"],
            "labels": torch.ones(len(features), dtype=torch.float32),
        }


class DistilBertPairwiseRewardModel(DistilBertPreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        hidden_size = getattr(config, "dim", getattr(config, "hidden_size", 768))
        self.pre_classifier = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(getattr(config, "seq_classif_dropout", 0.2))
        self.reward_head = nn.Linear(hidden_size, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.post_init()

    def score(self, input_ids, attention_mask) -> torch.Tensor:
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = torch.relu(self.pre_classifier(pooled))
        pooled = self.dropout(pooled)
        return self.reward_head(pooled).squeeze(-1)

    def forward(
        self,
        chosen_input_ids=None,
        chosen_attention_mask=None,
        rejected_input_ids=None,
        rejected_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        chosen_score = self.score(chosen_input_ids, chosen_attention_mask)
        rejected_score = self.score(rejected_input_ids, rejected_attention_mask)
        diff = chosen_score - rejected_score
        loss = None
        if labels is not None:
            loss = self.loss_fn(diff, torch.ones_like(diff))
        # Trainer expects logits; positive logit means chosen should win.
        return SequenceClassifierOutput(loss=loss, logits=diff.unsqueeze(-1))


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
        "metric_for_best_model": "accuracy",
        "greater_is_better": True,
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
    logits, labels = eval_pred
    logits = logits.reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    y = np.ones_like(preds)
    metrics = {
        "accuracy": accuracy_score(y, preds),
        "mean_margin": float(np.mean(logits)),
        "median_margin": float(np.median(logits)),
    }
    # AUC is not meaningful with only positive-oriented pairs, but keep a
    # symmetric diagnostic by adding flipped negative copies.
    auc_y = np.concatenate([np.ones_like(probs), np.zeros_like(probs)])
    auc_scores = np.concatenate([probs, 1.0 - probs])
    metrics["symmetric_auc"] = roc_auc_score(auc_y, auc_scores)
    return metrics


def prediction_dataframe(df: pd.DataFrame, logits: np.ndarray) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    logits = logits.reshape(-1)
    out["score_margin_chosen_minus_rejected"] = logits
    out["chosen_win_probability"] = 1.0 / (1.0 + np.exp(-logits))
    out["predicted_correct"] = out["chosen_win_probability"] >= 0.5
    return out


def write_report(metrics: dict, out_dir: Path, df: pd.DataFrame) -> None:
    lines = [
        "# Headline Pairwise Reward Critic",
        "",
        "A DistilBERT reward scorer trained from LLM-judge pairwise preferences.",
        "",
        "Important caveat: this is a tiny 167-pair proof-of-concept dataset. The model demonstrates the reward-modeling workflow but needs more judged pairs for robust generalization.",
        "",
        "## Test Metrics",
        "",
        f"- Accuracy: {metrics['test_accuracy']:.3f}",
        f"- Symmetric AUC: {metrics['test_symmetric_auc']:.3f}",
        f"- Mean margin, chosen minus rejected: {metrics['test_mean_margin']:.3f}",
        f"- Median margin, chosen minus rejected: {metrics['test_median_margin']:.3f}",
        "",
        "## Split Counts",
        "",
        "| Split | Rows |",
        "| --- | ---: |",
        *[f"| {split} | {count} |" for split, count in df["split"].value_counts().items()],
        "",
        "## Suggested Use",
        "",
        "Use this scorer for candidate reranking: generate several headlines, score each as `score(summary, headline)`, and choose the highest reward. Combine it with the clickbait penalty critic and the multi-dimensional reward critic for a more stable reward.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=10)
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

    df = split_by_seed(load_pairs(args.data), args.seed)
    df.to_csv(args.out / "training_pairs_with_splits.csv", index=False)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base = AutoModel.from_pretrained(args.base_model, attn_implementation="eager")
    base.config._attn_implementation = "eager"
    model = DistilBertPairwiseRewardModel(base.config)
    model.distilbert.load_state_dict(base.state_dict(), strict=False)

    train_ds = PairwiseDataset(train_df, tokenizer, args.max_length)
    val_ds = PairwiseDataset(val_df, tokenizer, args.max_length)
    test_ds = PairwiseDataset(test_df, tokenizer, args.max_length)

    trainer_kwargs = {
        "model": model,
        "args": make_training_args(args, device),
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": PairwiseDataCollator(tokenizer),
        "compute_metrics": compute_metrics,
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    print(f"Device preference: {device}")
    print(f"Train/val/test pairs: {len(train_df):,}/{len(val_df):,}/{len(test_df):,}")
    trainer.train()

    val_metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="val")
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    pred = trainer.predict(test_ds)
    pred_df = prediction_dataframe(test_df, pred.predictions)

    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    pred_df.to_csv(args.out / "test_pair_predictions.csv", index=False)

    metrics = {"validation": val_metrics, "test": test_metrics}
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metadata = {
        "model_type": "distilbert_pairwise_reward_scorer",
        "base_model": str(args.base_model),
        "data": str(args.data),
        "train_pairs": len(train_df),
        "val_pairs": len(val_df),
        "test_pairs": len(test_df),
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
