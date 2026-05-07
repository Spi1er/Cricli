#!/usr/bin/env python3
"""Check Cricli project assets and reproducibility state."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PROCESSED = [
    "data/processed/clickbait_penalty_splits.csv",
    "data/processed/headline_generation_eval_seed_100.csv",
    "data/processed/headline_persona_calibrated_candidate_matrix.csv",
    "data/processed/headline_persona_calibrated_objective_selection.csv",
    "data/processed/headline_review_demo_cases.csv",
    "data/processed/headline_review_demo_cases_full.csv",
]
REQUIRED_RAW = [
    "data/raw/clickbait/marksverdhei_clickbait_title_classification/clickbait_title_classification.csv",
    "data/raw/mind_hf_rui98/news_small.csv",
    "data/raw/mind_hf_rui98/train_small.csv",
]
MODEL_ASSETS = {
    "base_distilbert_seqcls": "models/base/distilbert-base-uncased-seqcls",
    "clickbait_penalty_distilbert": "models/clickbait_penalty_distilbert",
    "headline_quality_reward_distilbert": "models/headline_quality_reward_distilbert",
    "headline_pairwise_reward_distilbert": "models/headline_pairwise_reward_distilbert",
    "headline_generator_generic_sft": "models/headline_generator_flan_t5_small_generic_sft",
    "headline_generator_specificity_sft": "models/headline_generator_flan_t5_small_specificity_sft",
}
DEMO_ASSETS = [
    "demo/headline_review_console.html",
    "demo/single_article_review.html",
]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def file_status(path: Path) -> dict[str, object]:
    exists = path.exists()
    return {
        "path": rel(path),
        "exists": exists,
        "bytes": path.stat().st_size if exists and path.is_file() else None,
    }


def model_status(path: Path) -> dict[str, object]:
    exists = path.exists()
    expected_files = ["config.json", "tokenizer.json", "model.safetensors", "pytorch_model.bin"]
    present = [name for name in expected_files if (path / name).exists()]
    return {
        "path": rel(path),
        "exists": exists,
        "present_files": present,
        "load_ready": exists and "config.json" in present and ("model.safetensors" in present or "pytorch_model.bin" in present),
    }


def import_status(python: str) -> dict[str, object]:
    code = """
import json
mods = ['torch','transformers','pandas','sklearn','requests']
out = {}
for mod_name in mods:
    try:
        mod = __import__(mod_name)
        out[mod_name] = getattr(mod, '__version__', 'ok')
    except Exception as exc:
        out[mod_name] = 'ERROR: ' + repr(exc)
print(json.dumps(out))
"""
    try:
        proc = subprocess.run([python, "-c", code], cwd=PROJECT_ROOT, check=False, text=True, capture_output=True, timeout=30)
        if proc.returncode != 0:
            return {"python": python, "ok": False, "error": proc.stderr.strip() or proc.stdout.strip()}
        return {"python": python, "ok": True, "packages": json.loads(proc.stdout)}
    except Exception as exc:  # noqa: BLE001
        return {"python": python, "ok": False, "error": repr(exc)}


def build_report(status: dict[str, object]) -> str:
    lines = ["# Project Asset Check", ""]
    lines.append(f"- Project root: `{status['project_root']}`")
    lines.append(f"- OpenAI API key configured: `{status['openai_api_key']}`")
    lines.append("")
    env = status["environment"]
    lines.extend(["## Environment", "", f"- Python: `{env.get('python')}`", f"- OK: `{env.get('ok')}`"])
    if env.get("packages"):
        for name, version in env["packages"].items():
            lines.append(f"- {name}: `{version}`")
    if env.get("error"):
        lines.append(f"- Error: `{env['error']}`")
    lines.append("")

    for title, key in [("Raw Data", "raw_data"), ("Processed Data", "processed_data"), ("Models", "models"), ("Demos", "demos")]:
        lines.extend([f"## {title}", ""])
        items = status[key]
        if isinstance(items, dict):
            iterator = items.items()
        else:
            iterator = [(item["path"], item) for item in items]
        for name, item in iterator:
            marker = "OK" if item.get("exists") else "MISSING"
            extra = ""
            if "load_ready" in item:
                extra = f", load_ready={item['load_ready']}"
            lines.append(f"- `{name}`: {marker}{extra}")
        lines.append("")

    lines.extend(["## Recommended Fixes", ""])
    for fix in status["recommended_fixes"]:
        lines.append(f"- {fix}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Cricli project assets and reproducibility state.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for dependency checks.")
    parser.add_argument("--json-output", type=Path, default=PROJECT_ROOT / "data" / "processed" / "project_asset_check.json")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "docs" / "PROJECT_ASSET_CHECK.md")
    args = parser.parse_args()

    raw = [file_status(PROJECT_ROOT / path) for path in REQUIRED_RAW]
    processed = [file_status(PROJECT_ROOT / path) for path in REQUIRED_PROCESSED]
    models = {name: model_status(PROJECT_ROOT / path) for name, path in MODEL_ASSETS.items()}
    demos = [file_status(PROJECT_ROOT / path) for path in DEMO_ASSETS]

    fixes = []
    if not models["clickbait_penalty_distilbert"]["load_ready"]:
        fixes.append("Restore or retrain `models/clickbait_penalty_distilbert`; current single-article risk scoring will otherwise use heuristic fallback.")
    if not models["base_distilbert_seqcls"]["load_ready"]:
        fixes.append("Prepare `models/base/distilbert-base-uncased-seqcls` before retraining DistilBERT critics.")
    if not all(item["exists"] for item in raw):
        fixes.append("Restore raw datasets under `data/raw/` before full end-to-end rebuilds.")
    if not os.environ.get("OPENAI_API_KEY"):
        fixes.append("Set `OPENAI_API_KEY` before API generation, LLM judging, or persona voting.")
    if not fixes:
        fixes.append("Core assets look ready. Next useful step is validating the single-article demo across all objectives.")

    status = {
        "project_root": ".",
        "openai_api_key": bool(os.environ.get("OPENAI_API_KEY")),
        "environment": import_status(args.python),
        "raw_data": raw,
        "processed_data": processed,
        "models": models,
        "demos": demos,
        "recommended_fixes": fixes,
    }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(status, indent=2), encoding="utf-8")
    args.report.write_text(build_report(status), encoding="utf-8")
    print("Wrote", rel(args.json_output))
    print("Wrote", rel(args.report))
    print(json.dumps({"recommended_fixes": fixes}, indent=2))


if __name__ == "__main__":
    main()
