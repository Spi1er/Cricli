# Demo Delivery Scope

This document defines the current delivery scope for the Cricli project after simplifying the workflow around the product-facing headline review console.

## Main Product Demo

Use this command from `Cricli`:

```bash
python scripts/run_product_demo.py --limit-seeds 10 --python .venv/bin/python
```

It runs the local asset check, builds a compact 10-article static HTML dataset, builds a full 100-seed Gradio case explorer dataset, and renders the self-contained HTML demo:

```text
demo/headline_review_console.html
```

The demo should be presented as a decision-support console, not as a claim that our generator beats GPT. The product value is selecting and explaining the best headline under a business objective.

Use `docs/PRODUCT_DEMO_SCENARIOS.md` as the presentation script. It defines five featured live-demo scenarios:

- objective changes recommendation;
- trust/safety keeps editorial control;
- specificity selects a concrete alternative;
- validated GenAI baseline;
- SFT adds specificity.

The first three scenarios are the recommended presentation path. The last two are useful backup scenarios for Q&A or longer demos.

## Files To Keep In Git

Core code:

```text
scripts/run_product_demo.py
scripts/build_headline_review_demo_cases.py
scripts/build_headline_review_demo_html.py
scripts/check_project_assets.py
scripts/review_single_article.py
demo/gradio_app.py
```

Main demo artifacts:

```text
data/processed/headline_review_demo_cases.csv
data/processed/headline_review_demo_metadata.json
data/processed/headline_review_demo_profile.md
data/processed/headline_review_demo_cases_full.csv
data/processed/headline_review_demo_cases_full_metadata.json
data/processed/headline_review_demo_cases_full_profile.md
demo/headline_review_console.html
```

Project docs:

```text
README.md
docs/PROJECT_CODE_STRUCTURE.md
docs/DEMO_DELIVERY_SCOPE.md
docs/SIMPLIFIED_PRODUCT_WORKFLOW.md
docs/PRODUCT_DEMO_SCENARIOS.md
docs/SETUP_NOTES.md
docs/REPRODUCIBILITY_CHECKLIST.md
```

## Files To Keep Local Only

These are generated during debugging or one-off checks and should not be committed:

```text
data/processed/project_asset_check.json
docs/PROJECT_ASSET_CHECK.md
data/processed/single_article_review_*.csv
data/processed/single_article_review_*.json
demo/single_article_review*.html
data/processed/*_model_check.*
data/processed/*_dryrun*
models/
data/raw/
.venv/
```

## Current Functional Story

The demo now follows this narrower structure:

```text
article summary
-> hidden candidate pool from human, GenAI, SFT/rewrite/agentic variants
-> unified scorecard: quality + risk/safety + audience fit + objective fit
-> objective-specific selector
-> recommended headline and explanation
```

The visible demo options are intentionally small:

```text
Human baseline
GenAI baseline
Low-risk alternative
Recommended
```

This keeps the product easy to understand while preserving the research pipeline behind the scenes.

## What Is Still Missing

- API key setup is still needed for fresh API generation, LLM-as-judge, and persona voting.
- SFT generator checkpoints are restored in the latest local rerun, but they remain excluded from GitHub and must be retrained on a fresh clone.
- The final report should use the simplified product framing and avoid overemphasizing agentic/RL claims.

## Gradio Live Demo

The static HTML console remains the stable fallback demo. The Gradio app is the live interactive demo and uses all 100 saved article seeds when `data/processed/headline_review_demo_cases_full.csv` is present:

```bash
python demo/gradio_app.py
```

The Gradio interface includes:

- five curated presentation scenarios;
- a 100-seed saved case explorer;
- a custom summary review mode;
- candidate source pool summaries;
- candidate-level quality, risk/safety, clickbait penalty, audience fit, objective fit, and decision scores;
- persona vote tables and objective-switch previews.

If port 7860 is busy:

```bash
python demo/gradio_app.py --port 7861
```

## Next Delivery Step

Before pushing to GitHub, stage only the files listed in "Files To Keep In Git". Do not stage `models/`, `data/raw/`, local single-article outputs, or temporary model-check files.
