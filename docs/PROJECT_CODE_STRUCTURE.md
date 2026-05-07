# Project Code Structure And Local Assets

This document describes the current Cricli local project structure after path cleanup, model retraining, SFT rerun, and demo updates.

## Canonical Project Root

Always run project commands from:

```text
Cricli
```

Current scripts derive paths from their own file location with `PROJECT_ROOT = Path(__file__).resolve().parents[1]`, so they should work from different current working directories, but using `Cricli` keeps commands and outputs easiest to read.

## Main Directories

```text
scripts/              runnable data, training, evaluation, demo, and asset-check scripts
data/raw/             local raw datasets; not intended for Git
data/processed/       processed datasets, score tables, judge outputs, and reports
docs/                 project summaries, analysis notes, and reproducibility checks
demo/                 generated local HTML demos
models/               local trained models and downloaded base checkpoints; excluded from Git
```

## Model Locations

Core model paths expected by scripts:

```text
models/base/distilbert-base-uncased-seqcls
models/clickbait_penalty_distilbert
models/headline_quality_reward_distilbert
models/headline_pairwise_reward_distilbert
models/headline_quality_reward_distilbert_v2
models/headline_pairwise_reward_distilbert_v2
models/headline_generator_flan_t5_small_generic_sft
models/headline_generator_flan_t5_small_specificity_sft
```

Current required local model for the single-article demo:

```text
models/clickbait_penalty_distilbert
```

Current restored local critic models:

```text
models/clickbait_penalty_distilbert
models/headline_quality_reward_distilbert
models/headline_pairwise_reward_distilbert
models/headline_quality_reward_distilbert_v2
models/headline_pairwise_reward_distilbert_v2
models/headline_generator_flan_t5_small_generic_sft
models/headline_generator_flan_t5_small_specificity_sft
```

The clickbait critic is used by the single-article and Gradio demos. The quality and pairwise reward critics support reward-guided reranking and objective selection. The SFT generator checkpoints are auxiliary candidate-source models; they are useful for evaluation and report evidence, but they are not the core product claim.

## Path And Output Rules

- Static review-console demo:
  - input: `data/processed/headline_review_demo_cases.csv`
  - output: `demo/headline_review_console.html`
- Single-article review demo:
  - default output CSV: `data/processed/single_article_review_candidates.csv`
  - default output HTML: `demo/single_article_review.html`
  - default output metadata: `data/processed/single_article_review_metadata.json`
- Gradio review demo:
  - app: `demo/gradio_app.py`
  - preferred input: `data/processed/headline_review_demo_cases_full.csv`
  - fallback input: `data/processed/headline_review_demo_cases.csv`
  - default URL: `http://127.0.0.1:7860/`
  - fallback port command: `python demo/gradio_app.py --port 7861`
- Use `--run-name` for single-article experiments to avoid overwriting previous outputs.
- Use `--objective all` to generate all four objective-specific recommendations in one run.

## Health Check

Run this before debugging anything else:

```bash
python scripts/check_project_assets.py --python .venv/bin/python
```

It writes:

```text
docs/PROJECT_ASSET_CHECK.md
data/processed/project_asset_check.json
```

The current core local state should show:

```text
base_distilbert_seqcls: OK, load_ready=True
clickbait_penalty_distilbert: OK, load_ready=True
headline_quality_reward_distilbert: OK, load_ready=True
headline_pairwise_reward_distilbert: OK, load_ready=True
headline_generator_generic_sft: OK, load_ready=True
headline_generator_specificity_sft: OK, load_ready=True
```

If `OPENAI_API_KEY` is not set, API generation, LLM judging, and persona voting will be unavailable. Local dry-run demos and local critic scoring can still run.

## Recommended Next Step

The current product-facing demo has been simplified around `scripts/run_product_demo.py`. The remaining project direction should stay business-facing:

1. Keep the simplified review-console path as the main product surface.
2. Use the compact 10-article HTML demo as the stable fallback and the full 100-seed Gradio explorer for live interaction.
3. Show a compact explanation for why the recommended headline wins under each objective.
4. Treat clickbait as one part of a fused risk/safety score, not as a separate product claim.
5. Keep the agentic/reward/SFT scripts as research evidence behind the review console, not as the main demo interface.
