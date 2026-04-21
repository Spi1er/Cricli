#!/usr/bin/env python3
"""Build first-stage processed datasets for headline optimization.

Outputs:
- data/processed/mind_headline_pool_sample.csv
- data/processed/mind_pairwise_preferences_sample.jsonl
- data/processed/clickbait_penalty_splits.csv
- data/processed/data_profile.md
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MIND_DIR = RAW_DIR / "mind_hf_rui98"
CLICKBAIT_DIR = RAW_DIR / "clickbait" / "marksverdhei_clickbait_title_classification"

RANDOM_SEED = 5293
HEADLINE_POOL_SIZE = 10_000
PREFERENCE_PAIR_SIZE = 20_000
MAX_NEGATIVES_PER_IMPRESSION = 3


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_count(text: str) -> int:
    return len(text.split())


def parse_id_list(value: object) -> list[int]:
    text = clean_text(value)
    if not text:
        return []
    ids: list[int] = []
    for token in text.split():
        try:
            ids.append(int(token))
        except ValueError:
            continue
    return ids


def build_headline_pool(news: pd.DataFrame) -> pd.DataFrame:
    pool = news.copy()
    for col in ["title", "abstract", "body", "category", "subvert"]:
        pool[col] = pool[col].map(clean_text)

    pool["title_word_count"] = pool["title"].map(word_count)
    pool["abstract_word_count"] = pool["abstract"].map(word_count)
    pool["body_word_count"] = pool["body"].map(word_count)

    pool = pool[
        (pool["title_word_count"].between(4, 30))
        & (pool["abstract_word_count"] >= 5)
        & (pool["body_word_count"] >= 30)
    ].copy()

    pool = pool.drop_duplicates(subset=["title", "abstract"])
    pool["summary"] = pool["abstract"]

    columns = [
        "nid",
        "news_id",
        "title",
        "summary",
        "abstract",
        "body",
        "category",
        "subvert",
        "title_word_count",
        "abstract_word_count",
        "body_word_count",
        "url",
    ]
    pool = pool[columns]

    if len(pool) > HEADLINE_POOL_SIZE:
        pool = (
            pool.groupby("category", group_keys=False)
            .apply(lambda x: x.sample(min(len(x), max(1, round(HEADLINE_POOL_SIZE * len(x) / len(pool)))), random_state=RANDOM_SEED))
            .sample(frac=1, random_state=RANDOM_SEED)
            .head(HEADLINE_POOL_SIZE)
        )

    return pool.reset_index(drop=True)


def build_pairwise_preferences(news: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    # train/dev/test files in this Hugging Face derivative reference the local
    # integer `nid`, while `news_id` stores the original MIND ID such as N55528.
    news_index = news.set_index("nid")
    records: list[dict[str, object]] = []
    rng = np.random.default_rng(RANDOM_SEED)

    shuffled = train.sample(frac=1, random_state=RANDOM_SEED)
    for row in shuffled.itertuples(index=False):
        positives = parse_id_list(row.positive)
        negatives = parse_id_list(row.negative)
        if not positives or not negatives:
            continue

        negatives = list(rng.choice(negatives, size=min(len(negatives), MAX_NEGATIVES_PER_IMPRESSION), replace=False))
        for pos_id in positives:
            if pos_id not in news_index.index:
                continue
            pos = news_index.loc[pos_id]
            pos_title = clean_text(pos["title"])
            pos_summary = clean_text(pos["abstract"]) or clean_text(pos["body"])[:1000]
            if not pos_title or not pos_summary:
                continue

            for neg_id in negatives:
                if neg_id not in news_index.index:
                    continue
                neg = news_index.loc[neg_id]
                neg_title = clean_text(neg["title"])
                neg_summary = clean_text(neg["abstract"]) or clean_text(neg["body"])[:1000]
                if not neg_title or not neg_summary:
                    continue

                records.append(
                    {
                        "uid": int(row.uid),
                        "impression_id": int(row.impression_id),
                        "chosen_nid": int(pos_id),
                        "rejected_nid": int(neg_id),
                        "chosen_news_id": clean_text(pos["news_id"]),
                        "rejected_news_id": clean_text(neg["news_id"]),
                        "chosen_title": pos_title,
                        "rejected_title": neg_title,
                        "chosen_summary": pos_summary,
                        "rejected_summary": neg_summary,
                        "chosen_category": clean_text(pos["category"]),
                        "rejected_category": clean_text(neg["category"]),
                        "chosen_subvert": clean_text(pos["subvert"]),
                        "rejected_subvert": clean_text(neg["subvert"]),
                        "same_category": clean_text(pos["category"]) == clean_text(neg["category"]),
                        "preference_source": "MIND_click_log",
                        "label_note": "clicked item is treated as chosen; non-clicked impression item is treated as rejected",
                    }
                )
                if len(records) >= PREFERENCE_PAIR_SIZE:
                    return pd.DataFrame.from_records(records)

    return pd.DataFrame.from_records(records)


def build_clickbait_splits(clickbait: pd.DataFrame) -> pd.DataFrame:
    df = clickbait.copy()
    df["title"] = df["title"].map(clean_text)
    df = df[df["title"].map(word_count) >= 3].copy()
    df = df.drop_duplicates(subset=["title"])
    df["clickbait"] = df["clickbait"].astype(int)
    df["title_word_count"] = df["title"].map(word_count)

    parts = []
    for label, group in df.groupby("clickbait"):
        group = group.sample(frac=1, random_state=RANDOM_SEED)
        n = len(group)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)
        group = group.copy()
        group["split"] = "train"
        group.iloc[train_end:val_end, group.columns.get_loc("split")] = "val"
        group.iloc[val_end:, group.columns.get_loc("split")] = "test"
        parts.append(group)

    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=RANDOM_SEED)


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def profile_markdown(
    news: pd.DataFrame,
    headline_pool: pd.DataFrame,
    preferences: pd.DataFrame,
    clickbait_splits: pd.DataFrame,
) -> str:
    missing_abstract = news["abstract"].isna().mean()
    missing_body = news["body"].isna().mean()

    category_counts = headline_pool["category"].value_counts().head(12)
    clickbait_counts = clickbait_splits.groupby(["split", "clickbait"]).size().unstack(fill_value=0)
    same_category_rate = preferences["same_category"].mean() if len(preferences) else 0

    lines = [
        "# Data Profile",
        "",
        "## MIND news",
        "",
        f"- Raw news rows: {len(news):,}",
        f"- Processed headline pool rows: {len(headline_pool):,}",
        f"- Raw abstract missing rate: {missing_abstract:.2%}",
        f"- Raw body missing rate: {missing_body:.2%}",
        f"- Headline pool title word count mean: {headline_pool['title_word_count'].mean():.2f}",
        f"- Headline pool abstract word count mean: {headline_pool['abstract_word_count'].mean():.2f}",
        "",
        "Top processed categories:",
        "",
    ]

    for category, count in category_counts.items():
        lines.append(f"- `{category}`: {count:,}")

    lines.extend(
        [
            "",
            "## MIND pairwise preferences",
            "",
            f"- Pairwise rows: {len(preferences):,}",
            f"- Unique users: {preferences['uid'].nunique():,}" if len(preferences) else "- Unique users: 0",
            f"- Unique chosen news: {preferences['chosen_news_id'].nunique():,}" if len(preferences) else "- Unique chosen news: 0",
            f"- Unique rejected news: {preferences['rejected_news_id'].nunique():,}" if len(preferences) else "- Unique rejected news: 0",
            f"- Same-category pair rate: {same_category_rate:.2%}",
            "",
            "Interpretation: these pairs are noisy implicit preferences from click logs. They are useful for engagement-oriented ranking, but they do not isolate headline quality because clicked and rejected items may have different article content.",
            "",
            "## Clickbait penalty data",
            "",
            f"- Processed rows: {len(clickbait_splits):,}",
            f"- Mean title word count: {clickbait_splits['title_word_count'].mean():.2f}",
            "",
            "Split counts:",
            "",
            "| Split | Non-clickbait | Clickbait |",
            "| --- | ---: | ---: |",
        ]
    )

    for split in ["train", "val", "test"]:
        row = clickbait_counts.loc[split] if split in clickbait_counts.index else pd.Series({0: 0, 1: 0})
        lines.append(f"| {split} | {int(row.get(0, 0)):,} | {int(row.get(1, 0)):,} |")

    lines.extend(
        [
            "",
            "## Processed files",
            "",
            "- `mind_headline_pool_sample.csv`: clean article/title pool for generation and evaluation.",
            "- `mind_pairwise_preferences_sample.jsonl`: clicked vs non-clicked preference pairs for reward/ranking experiments.",
            "- `clickbait_penalty_splits.csv`: binary clickbait labels with train/val/test split for penalty critic.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    news = pd.read_csv(MIND_DIR / "news_small.csv")
    train = pd.read_csv(MIND_DIR / "train_small.csv")
    clickbait = pd.read_csv(CLICKBAIT_DIR / "clickbait_title_classification.csv")

    headline_pool = build_headline_pool(news)
    preferences = build_pairwise_preferences(news, train)
    clickbait_splits = build_clickbait_splits(clickbait)

    headline_pool.to_csv(PROCESSED_DIR / "mind_headline_pool_sample.csv", index=False)
    write_jsonl(preferences, PROCESSED_DIR / "mind_pairwise_preferences_sample.jsonl")
    clickbait_splits.to_csv(PROCESSED_DIR / "clickbait_penalty_splits.csv", index=False)

    profile = profile_markdown(news, headline_pool, preferences, clickbait_splits)
    (PROCESSED_DIR / "data_profile.md").write_text(profile, encoding="utf-8")

    print("Wrote:")
    print(PROCESSED_DIR / "mind_headline_pool_sample.csv", len(headline_pool))
    print(PROCESSED_DIR / "mind_pairwise_preferences_sample.jsonl", len(preferences))
    print(PROCESSED_DIR / "clickbait_penalty_splits.csv", len(clickbait_splits))
    print(PROCESSED_DIR / "data_profile.md")


if __name__ == "__main__":
    main()
