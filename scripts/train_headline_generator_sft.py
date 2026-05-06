#!/usr/bin/env python3
"""Supervised fine-tuning for a small headline generation model."""

from __future__ import annotations

import argparse
import inspect
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "google/flan-t5-small"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "headline_generator_flan_t5_small_sft"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "data" / "processed" / "headline_generator_sft_test_predictions.csv"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "headline_generator_sft_metadata.json"


class HeadlineSFTDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_source_length: int,
        max_target_length: int,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[idx]
        source = str(row["input_text"])
        target = str(row["target_text"])

        model_inputs = self.tokenizer(
            source,
            max_length=self.max_source_length,
            truncation=True,
        )
        labels = self.tokenizer(
            text_target=target,
            max_length=self.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def load_sft_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["input_text", "target_text"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df = df.copy()
    df["input_text"] = df["input_text"].map(clean_text)
    df["target_text"] = df["target_text"].map(clean_text)
    df = df[df["input_text"].ne("") & df["target_text"].ne("")].reset_index(drop=True)
    return df


def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is False.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return requested


def filtered_training_args(**kwargs: Any) -> Seq2SeqTrainingArguments:
    """Keep the script compatible across Transformers argument-name changes."""
    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    supported = set(signature.parameters)

    aliases = {
        "evaluation_strategy": "eval_strategy",
        "save_strategy": "save_strategy",
        "logging_strategy": "logging_strategy",
    }
    normalized = dict(kwargs)
    for old_name, new_name in aliases.items():
        if old_name in normalized and old_name not in supported and new_name in supported:
            normalized[new_name] = normalized.pop(old_name)

    return Seq2SeqTrainingArguments(**{k: v for k, v in normalized.items() if k in supported})


def make_trainer(**kwargs: Any) -> Seq2SeqTrainer:
    signature = inspect.signature(Seq2SeqTrainer.__init__)
    supported = set(signature.parameters)
    if "tokenizer" in kwargs and "tokenizer" not in supported and "processing_class" in supported:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
    return Seq2SeqTrainer(**{k: v for k, v in kwargs.items() if k in supported})


def make_training_args(args: argparse.Namespace, device: str) -> Seq2SeqTrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir / "checkpoints"),
        "overwrite_output_dir": True,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": args.save_total_limit,
        "predict_with_generate": True,
        "generation_max_length": args.max_target_length,
        "generation_num_beams": args.num_beams,
        "report_to": "none",
        "seed": args.seed,
        "data_seed": args.seed,
        "fp16": device == "cuda" and args.fp16,
    }
    if device == "mps":
        kwargs["use_mps_device"] = True
    return filtered_training_args(**kwargs)


def exact_match_rate(preds: list[str], refs: list[str]) -> float:
    if not preds:
        return 0.0
    matches = [clean_text(pred).lower() == clean_text(ref).lower() for pred, ref in zip(preds, refs)]
    return float(np.mean(matches))


def generate_predictions(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    df: pd.DataFrame,
    device: str,
    batch_size: int,
    max_source_length: int,
    max_target_length: int,
    num_beams: int,
    limit: int,
) -> pd.DataFrame:
    sample = df.head(limit).copy() if limit > 0 else df.copy()
    predictions: list[str] = []
    model.eval()
    model.to(device)
    for start in range(0, len(sample), batch_size):
        batch = sample.iloc[start : start + batch_size]
        encoded = tokenizer(
            batch["input_text"].tolist(),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_source_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_length=max_target_length,
                num_beams=num_beams,
            )
        predictions.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))

    sample["generated_title"] = predictions
    sample["exact_match"] = [
        clean_text(pred).lower() == clean_text(ref).lower()
        for pred, ref in zip(sample["generated_title"], sample["target_text"])
    ]
    return sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_DATA_DIR / "headline_sft_specificity_train.csv")
    parser.add_argument("--val-csv", type=Path, default=DEFAULT_DATA_DIR / "headline_sft_specificity_val.csv")
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_DATA_DIR / "headline_sft_specificity_test.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=32)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--prediction-limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=5293)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_df = load_sft_csv(args.train_csv)
    val_df = load_sft_csv(args.val_csv)
    test_df = load_sft_csv(args.test_csv)
    if args.smoke_test:
        train_df = train_df.head(64).copy()
        val_df = val_df.head(32).copy()
        test_df = test_df.head(32).copy()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

    train_dataset = HeadlineSFTDataset(train_df, tokenizer, args.max_source_length, args.max_target_length)
    val_dataset = HeadlineSFTDataset(val_df, tokenizer, args.max_source_length, args.max_target_length)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = make_training_args(args, device)
    trainer = make_trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    train_result = trainer.train()
    val_metrics = trainer.evaluate()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    pred_df = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        df=test_df,
        device=device,
        batch_size=args.eval_batch_size,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        num_beams=args.num_beams,
        limit=args.prediction_limit,
    )
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(args.predictions, index=False)

    metadata = {
        "model_name_or_path": args.model_name_or_path,
        "output_dir": str(args.output_dir),
        "train_csv": str(args.train_csv),
        "val_csv": str(args.val_csv),
        "test_csv": str(args.test_csv),
        "predictions": str(args.predictions),
        "device": device,
        "smoke_test": args.smoke_test,
        "rows": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
            "predictions": int(len(pred_df)),
        },
        "training": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_source_length": args.max_source_length,
            "max_target_length": args.max_target_length,
            "num_beams": args.num_beams,
        },
        "train_metrics": {k: float(v) for k, v in train_result.metrics.items() if isinstance(v, (int, float))},
        "val_metrics": {k: float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))},
        "test_prediction_exact_match": exact_match_rate(
            pred_df["generated_title"].tolist(),
            pred_df["target_text"].tolist(),
        ),
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved model to", args.output_dir)
    print("Wrote predictions to", args.predictions)
    print("Wrote metadata to", args.metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
