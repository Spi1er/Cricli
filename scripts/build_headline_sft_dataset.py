#!/usr/bin/env python3
"""Build generic and specificity-aware SFT datasets for headline generation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "mind_headline_pool_with_clickbait_penalty.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "headline_sft_dataset_profile.md"
DEFAULT_METADATA = DEFAULT_OUTPUT_DIR / "headline_sft_dataset_metadata.json"

SPLIT_RATIOS = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1,
}

TITLE_STOPWORDS = {
    "A",
    "An",
    "And",
    "Are",
    "As",
    "At",
    "Be",
    "By",
    "For",
    "From",
    "How",
    "In",
    "Into",
    "Is",
    "It",
    "New",
    "Of",
    "On",
    "Or",
    "The",
    "This",
    "To",
    "With",
    "Why",
}


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\u00a0", " ")
    return " ".join(text.split())


def word_count(text: str) -> int:
    return len(text.split())


def contains_specific_signal(title: str) -> bool:
    """Approximate whether a headline names concrete entities, dates, or quantities."""
    if re.search(r"\d", title):
        return True
    tokens = re.findall(r"\b[A-Z][A-Za-z0-9'&.-]{2,}\b", title)
    named_tokens = [token for token in tokens if token not in TITLE_STOPWORDS]
    return len(named_tokens) >= 2


def normalize_columns(df: pd.DataFrame, source_field: str) -> pd.DataFrame:
    required = ["title", "category", source_field]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    if "news_id" not in out.columns:
        out["news_id"] = out.index.map(lambda idx: f"row-{idx}")
    if "nid" not in out.columns:
        out["nid"] = out["news_id"]
    if "subvert" not in out.columns:
        out["subvert"] = ""
    if "clickbait_penalty" not in out.columns:
        out["clickbait_penalty"] = pd.NA
    if "predicted_clickbait" not in out.columns:
        out["predicted_clickbait"] = pd.NA
    if "url" not in out.columns:
        out["url"] = ""

    out["source_text"] = out[source_field].map(clean_text)
    out["target_title"] = out["title"].map(clean_text)
    out["category"] = out["category"].map(clean_text)
    out["subvert"] = out["subvert"].map(clean_text)
    out["news_id"] = out["news_id"].map(clean_text)
    out["nid"] = out["nid"].map(clean_text)
    out["url"] = out["url"].map(clean_text)
    out["source_word_count"] = out["source_text"].map(word_count)
    out["target_word_count"] = out["target_title"].map(word_count)
    out["has_specific_signal"] = out["target_title"].map(contains_specific_signal)
    out["clickbait_penalty"] = pd.to_numeric(out["clickbait_penalty"], errors="coerce")
    out["predicted_clickbait"] = pd.to_numeric(out["predicted_clickbait"], errors="coerce")
    return out


def assign_splits(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    out = df.sample(frac=1, random_state=random_state).reset_index(drop=True).copy()
    n = len(out)
    train_end = int(n * SPLIT_RATIOS["train"])
    val_end = train_end + int(n * SPLIT_RATIOS["val"])
    out["split"] = "test"
    out.loc[: train_end - 1, "split"] = "train"
    out.loc[train_end : val_end - 1, "split"] = "val"
    return out


def generic_prompt(row: pd.Series) -> str:
    return "\n".join(
        [
            "Generate a news headline.",
            f"Category: {row['category']}",
            f"Article summary: {row['source_text']}",
        ]
    )


def specificity_prompt(row: pd.Series) -> str:
    return "\n".join(
        [
            "Generate a faithful, specific, non-clickbait news headline.",
            "Use concrete names, numbers, places, or events only when supported by the article summary.",
            "Avoid curiosity-gap wording, exaggeration, and unsupported details.",
            f"Category: {row['category']}",
            f"Article summary: {row['source_text']}",
        ]
    )


def build_variant(df: pd.DataFrame, variant: str, prompt_fn) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out.insert(0, "sft_id", [f"{variant}-{idx + 1:06d}" for idx in range(len(out))])
    out["dataset_variant"] = variant
    out["input_text"] = out.apply(prompt_fn, axis=1)
    out["target_text"] = out["target_title"]
    columns = [
        "sft_id",
        "dataset_variant",
        "split",
        "nid",
        "news_id",
        "category",
        "subvert",
        "source_text",
        "target_title",
        "input_text",
        "target_text",
        "source_word_count",
        "target_word_count",
        "has_specific_signal",
        "clickbait_penalty",
        "predicted_clickbait",
        "url",
    ]
    return out[columns]


def filter_base(df: pd.DataFrame, min_source_words: int, min_title_words: int, max_title_words: int) -> pd.DataFrame:
    return df[
        df["source_text"].ne("")
        & df["target_title"].ne("")
        & df["category"].ne("")
        & df["source_word_count"].ge(min_source_words)
        & df["target_word_count"].between(min_title_words, max_title_words)
    ].copy()


def filter_specificity(
    df: pd.DataFrame,
    max_clickbait_penalty: float,
    require_specific_signal: bool,
) -> pd.DataFrame:
    out = df[df["clickbait_penalty"].le(max_clickbait_penalty)].copy()
    if "predicted_clickbait" in out.columns:
        out = out[(out["predicted_clickbait"].isna()) | (out["predicted_clickbait"].eq(0))].copy()
    if require_specific_signal:
        out = out[out["has_specific_signal"]].copy()
    return out


def write_split_files(output_dir: Path, prefix: str, df: pd.DataFrame) -> dict[str, str]:
    paths = {}
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split].reset_index(drop=True)
        path = output_dir / f"{prefix}_{split}.csv"
        split_df.to_csv(path, index=False)
        paths[split] = str(path)
        print("Wrote", path, len(split_df))
    return paths


def summarize_variant(df: pd.DataFrame) -> dict:
    by_split = df["split"].value_counts().reindex(["train", "val", "test"]).fillna(0).astype(int)
    by_category = df["category"].value_counts().head(12)
    summary = {
        "rows": int(len(df)),
        "split_counts": by_split.to_dict(),
        "mean_source_word_count": float(df["source_word_count"].mean()) if len(df) else 0.0,
        "mean_target_word_count": float(df["target_word_count"].mean()) if len(df) else 0.0,
        "specific_signal_rate": float(df["has_specific_signal"].mean()) if len(df) else 0.0,
        "category_counts_top12": {str(k): int(v) for k, v in by_category.items()},
    }
    if df["clickbait_penalty"].notna().any():
        summary["mean_clickbait_penalty"] = float(df["clickbait_penalty"].mean())
        summary["clickbait_rate"] = float((df["clickbait_penalty"] >= 0.5).mean())
    return summary


def markdown_table_from_counts(counts: dict[str, int]) -> list[str]:
    lines = ["| Value | Rows |", "| --- | ---: |"]
    for key, value in counts.items():
        lines.append(f"| {key} | {value:,} |")
    return lines


def build_report(
    args: argparse.Namespace,
    generic_df: pd.DataFrame,
    specificity_df: pd.DataFrame,
    metadata: dict,
) -> str:
    lines = [
        "# Headline SFT Dataset Profile",
        "",
        "This dataset is the first business-driven generator training layer. It creates one broad headline-generation SFT set and one specificity-aware SFT set for comparing whether targeted data and instructions improve title quality.",
        "",
        "## Configuration",
        "",
        f"- Input: `{args.input}`",
        f"- Source field: `{args.source_field}`",
        f"- Minimum source words: {args.min_source_words}",
        f"- Title word range: {args.min_title_words}-{args.max_title_words}",
        f"- Specificity max clickbait penalty: {args.specificity_max_clickbait_penalty}",
        f"- Require specificity signal: {args.require_specific_signal}",
        f"- Random state: {args.random_state}",
        "",
        "## Dataset Sizes",
        "",
        "| Variant | Rows | Train | Val | Test | Mean Clickbait Penalty | Clickbait Rate | Specific Signal Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for variant, df in [("generic", generic_df), ("specificity", specificity_df)]:
        summary = metadata["variants"][variant]
        split = summary["split_counts"]
        lines.append(
            "| {variant} | {rows:,} | {train:,} | {val:,} | {test:,} | {penalty:.4f} | {rate:.2%} | {signal:.2%} |".format(
                variant=variant,
                rows=summary["rows"],
                train=split.get("train", 0),
                val=split.get("val", 0),
                test=split.get("test", 0),
                penalty=summary.get("mean_clickbait_penalty", 0.0),
                rate=summary.get("clickbait_rate", 0.0),
                signal=summary["specific_signal_rate"],
            )
        )

    lines.extend(["", "## Generic Top Categories", ""])
    lines.extend(markdown_table_from_counts(metadata["variants"]["generic"]["category_counts_top12"]))
    lines.extend(["", "## Specificity Top Categories", ""])
    lines.extend(markdown_table_from_counts(metadata["variants"]["specificity"]["category_counts_top12"]))

    lines.extend(
        [
            "",
            "## Training Use",
            "",
            "- M1 generic SFT: train on `headline_sft_generic_train.csv`, validate on `headline_sft_generic_val.csv`, test on `headline_sft_generic_test.csv`.",
            "- M2 specificity-aware SFT: train on `headline_sft_specificity_train.csv`, validate on `headline_sft_specificity_val.csv`, test on `headline_sft_specificity_test.csv`.",
            "- Evaluate both models on the same held-out generation seed or the SFT test split using local critics and LLM judge.",
            "- Treat M2 as the first policy model aligned with the proposal goal: faithful, specific, non-clickbait headline generation.",
            "",
            "## Example Rows",
            "",
            "### Generic",
            "",
        ]
    )
    preview_cols = ["split", "category", "source_text", "target_title"]
    lines.append(generic_df[preview_cols].head(5).to_markdown(index=False))
    lines.extend(["", "### Specificity", ""])
    lines.append(specificity_df[preview_cols].head(5).to_markdown(index=False))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--source-field", choices=["summary", "abstract", "body"], default="summary")
    parser.add_argument("--min-source-words", type=int, default=20)
    parser.add_argument("--min-title-words", type=int, default=5)
    parser.add_argument("--max-title-words", type=int, default=18)
    parser.add_argument("--specificity-max-clickbait-penalty", type=float, default=0.2)
    parser.add_argument("--require-specific-signal", action="store_true")
    parser.add_argument("--random-state", type=int, default=5293)
    args = parser.parse_args()

    raw_df = pd.read_csv(args.input)
    normalized = normalize_columns(raw_df, args.source_field)
    split_df = assign_splits(normalized, args.random_state)

    base_df = filter_base(
        split_df,
        min_source_words=args.min_source_words,
        min_title_words=args.min_title_words,
        max_title_words=args.max_title_words,
    )
    generic_df = build_variant(base_df, "generic", generic_prompt)

    specificity_base = filter_specificity(
        base_df,
        max_clickbait_penalty=args.specificity_max_clickbait_penalty,
        require_specific_signal=args.require_specific_signal,
    )
    specificity_df = build_variant(specificity_base, "specificity", specificity_prompt)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_files = {
        "generic": write_split_files(args.output_dir, "headline_sft_generic", generic_df),
        "specificity": write_split_files(args.output_dir, "headline_sft_specificity", specificity_df),
    }

    metadata = {
        "input": str(args.input),
        "source_field": args.source_field,
        "output_files": output_files,
        "filters": {
            "min_source_words": args.min_source_words,
            "min_title_words": args.min_title_words,
            "max_title_words": args.max_title_words,
            "specificity_max_clickbait_penalty": args.specificity_max_clickbait_penalty,
            "require_specific_signal": args.require_specific_signal,
        },
        "random_state": args.random_state,
        "input_rows": int(len(raw_df)),
        "base_eligible_rows": int(len(base_df)),
        "variants": {
            "generic": summarize_variant(generic_df),
            "specificity": summarize_variant(specificity_df),
        },
    }
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = build_report(args, generic_df, specificity_df, metadata)
    args.report.write_text(report, encoding="utf-8")

    print("Wrote", args.metadata)
    print("Wrote", args.report)
    print(json.dumps(metadata["variants"], indent=2))


if __name__ == "__main__":
    main()
