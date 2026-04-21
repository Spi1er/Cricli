#!/usr/bin/env python3
"""Merge original and agentic LLM-judge labels into v2 reward datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OLD_REWARD = PROJECT_ROOT / "data" / "processed" / "headline_quality_reward_model_examples.jsonl"
DEFAULT_AGENTIC_REWARD = PROJECT_ROOT / "data" / "processed" / "headline_quality_agentic_reward_model_examples.jsonl"
DEFAULT_OLD_PAIRWISE = PROJECT_ROOT / "data" / "processed" / "headline_quality_pairwise_preferences.jsonl"
DEFAULT_AGENTIC_PAIRWISE = PROJECT_ROOT / "data" / "processed" / "headline_quality_agentic_pairwise_preferences.jsonl"
DEFAULT_REWARD_OUT = PROJECT_ROOT / "data" / "processed" / "headline_quality_reward_model_examples_v2.jsonl"
DEFAULT_PAIRWISE_OUT = PROJECT_ROOT / "data" / "processed" / "headline_quality_pairwise_preferences_v2.jsonl"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "headline_quality_reward_training_v2_manifest.json"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def reward_key(row: dict) -> tuple:
    return (
        int(row["seed_id"]),
        str(row["variant"]),
        str(row["headline"]).strip().lower(),
        str(row.get("comparison_set", "original_3way")),
    )


def pairwise_key(row: dict) -> tuple:
    return (
        int(row["seed_id"]),
        str(row["chosen_variant"]),
        str(row["rejected_variant"]),
        str(row["chosen_title"]).strip().lower(),
        str(row["rejected_title"]).strip().lower(),
        str(row.get("preference_source", "")),
    )


def dedupe(rows: list[dict], key_fn) -> list[dict]:
    out = []
    seen = set()
    for row in rows:
        key = key_fn(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def tag_rows(rows: list[dict], source: str) -> list[dict]:
    tagged = []
    for row in rows:
        obj = dict(row)
        obj["dataset_version"] = "v2"
        obj["source_file_group"] = source
        tagged.append(obj)
    return tagged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-reward", type=Path, default=DEFAULT_OLD_REWARD)
    parser.add_argument("--agentic-reward", type=Path, default=DEFAULT_AGENTIC_REWARD)
    parser.add_argument("--old-pairwise", type=Path, default=DEFAULT_OLD_PAIRWISE)
    parser.add_argument("--agentic-pairwise", type=Path, default=DEFAULT_AGENTIC_PAIRWISE)
    parser.add_argument("--reward-output", type=Path, default=DEFAULT_REWARD_OUT)
    parser.add_argument("--pairwise-output", type=Path, default=DEFAULT_PAIRWISE_OUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    old_reward = tag_rows(load_jsonl(args.old_reward), "original_3way")
    agentic_reward = tag_rows(load_jsonl(args.agentic_reward), "agentic_4way")
    old_pairwise = tag_rows(load_jsonl(args.old_pairwise), "original_3way")
    agentic_pairwise = tag_rows(load_jsonl(args.agentic_pairwise), "agentic_4way")

    reward_rows = dedupe(old_reward + agentic_reward, reward_key)
    pairwise_rows = dedupe(old_pairwise + agentic_pairwise, pairwise_key)

    args.reward_output.parent.mkdir(parents=True, exist_ok=True)
    args.pairwise_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.reward_output, reward_rows)
    write_jsonl(args.pairwise_output, pairwise_rows)

    manifest = {
        "reward_output": str(args.reward_output),
        "pairwise_output": str(args.pairwise_output),
        "sources": {
            "old_reward": {"path": str(args.old_reward), "rows": len(old_reward)},
            "agentic_reward": {"path": str(args.agentic_reward), "rows": len(agentic_reward)},
            "old_pairwise": {"path": str(args.old_pairwise), "rows": len(old_pairwise)},
            "agentic_pairwise": {"path": str(args.agentic_pairwise), "rows": len(agentic_pairwise)},
        },
        "merged": {
            "reward_rows": len(reward_rows),
            "pairwise_rows": len(pairwise_rows),
            "reward_duplicates_removed": len(old_reward) + len(agentic_reward) - len(reward_rows),
            "pairwise_duplicates_removed": len(old_pairwise) + len(agentic_pairwise) - len(pairwise_rows),
        },
        "intended_use": [
            "Train v2 multi-dimensional reward critic.",
            "Train v2 pairwise reward critic.",
            "Rerun best-of-N agentic headline selection with v2 critics.",
        ],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Wrote", args.reward_output)
    print("Wrote", args.pairwise_output)
    print("Wrote", args.manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
