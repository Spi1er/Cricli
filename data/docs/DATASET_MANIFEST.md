# Dataset Manifest

This manifest documents the data assets used by Cricli and separates files that are tracked in GitHub from files that must be restored or regenerated locally.

## Raw Data Sources

Raw datasets are intentionally not tracked in GitHub. Restore them under `data/raw/` only when rebuilding processed data from scratch.

### MIND Small Derivative

Expected local path:

```text
data/raw/mind_hf_rui98/
```

Source:

```text
Rui98/mind on Hugging Face, derived from the Microsoft News Dataset (MIND)
```

Expected files:

| File | Rows Including Header | Main Columns | Project Use |
| --- | ---: | --- | --- |
| `news_small.csv` | 65,239 | `nid`, `news_id`, `title`, `abstract`, `body`, `category`, `subvert`, `entity`, `ab_entity`, `url` | Article/title pool, summaries, categories, fixed evaluation seeds |
| `train_small.csv` | 236,345 | `impression_id`, `uid`, `positive`, `negative` | Optional noisy click/preference construction |
| `dev_small.csv` | 36,577 | `impression_id`, `uid`, `impressions` | Optional validation split for ranking experiments |
| `test_small.csv` | 36,577 | `impression_id`, `uid`, `impressions` | Optional test split for ranking experiments |
| `impressions_small.csv` | 230,118 | `impression_id`, `uid`, `user_id`, `division`, `history`, `impressions`, `time` | Optional user-behavior context |
| `user_interaction_small.csv` | 94,058 | `uid`, `history` | Optional user-history context |

Notes:

- Clicks are noisy implicit preference signals, not direct headline-quality labels.
- `category` and `subvert` are used as coarse content context.
- The current project uses a fixed 100-article evaluation seed from processed MIND headlines.

### Clickbait Title Classification

Expected local path:

```text
data/raw/clickbait/marksverdhei_clickbait_title_classification/
```

Source:

```text
marksverdhei/clickbait_title_classification on Hugging Face
```

Expected file:

| File | Rows Including Header | Main Columns | Project Use |
| --- | ---: | --- | --- |
| `clickbait_title_classification.csv` | 32,001 | `title`, `clickbait` | DistilBERT clickbait penalty critic |

Notes:

- This dataset contains titles and binary labels, not article bodies.
- It is used only as the clickbait/risk dimension.

## Processed Data Tracked In GitHub

The repository includes compact processed artifacts so teammates can inspect the project without restoring raw datasets or rerunning API calls.

| Asset Group | Main Files | Regeneration Script |
| --- | --- | --- |
| Clickbait critic data | `data/processed/clickbait_penalty_splits.csv` | `scripts/build_processed_datasets.py` |
| MIND headline pool | `data/processed/mind_headline_pool_sample.csv`, `data/processed/mind_headline_pool_with_clickbait_penalty.csv` | `scripts/build_processed_datasets.py`, `scripts/score_headline_clickbait_penalty.py` |
| Fixed evaluation seed | `data/processed/headline_generation_eval_seed_100.csv` | `scripts/build_headline_generation_seed.py` |
| OpenAI headline candidates | `headline_generation_zero_shot_100.csv`, rewrite outputs, agentic candidate outputs | API generation scripts in `scripts/` |
| Local critic scores | `*_scored_100.csv`, local evaluation CSVs | scoring and evaluation scripts in `scripts/` |
| LLM judge labels | `headline_quality_llm_judge*.csv`, reward JSONL files | `scripts/run_llm_judge_*.py` |
| Reward training data | `headline_quality_reward_model_examples*.jsonl`, `headline_quality_pairwise_preferences*.jsonl` | judge scripts and `scripts/build_reward_training_v2.py` |
| Persona votes | `headline_audience_persona_votes.csv` | `scripts/run_audience_persona_voting.py` |
| Objective selection | `headline_multi_agent_*`, `headline_persona_calibrated_*` | `scripts/build_multi_agent_objective_matrix.py`, `scripts/build_persona_calibrated_selector.py` |
| SFT datasets and predictions | `headline_sft_*`, `headline_generator_flan_t5_small_*_predictions.csv` | `scripts/build_headline_sft_dataset.py`, `scripts/train_headline_generator_sft.py` |
| Statistical support | `bootstrap_significance.md` | `scripts/bootstrap_significance.py` |
| Demo artifacts | `headline_review_demo_cases.csv`, `headline_review_demo_cases_full.csv`, `demo/headline_review_console.html` | `scripts/run_product_demo.py` |

## Local-Only Files Not Tracked In GitHub

These files are expected to be local and should not be committed:

| Local Asset | Why It Is Excluded |
| --- | --- |
| `data/raw/` | Raw datasets are external and larger than needed for the course repo |
| `models/` | Trained checkpoints and Hugging Face downloads are large and reproducible |
| `.venv/` | Local Python environment |
| API keys and shell exports | Secrets must not be committed |
| `data/processed/project_asset_check.json` | Local environment health-check output |
| `docs/PROJECT_ASSET_CHECK.md` | Local environment health-check output |
| `data/processed/single_article_review_*.csv` | Local one-off demo runs |
| `data/processed/single_article_review_*.json` | Local one-off demo metadata |
| `demo/single_article_review*.html` | Local one-off demo HTML |
| `__pycache__/` | Python cache files |

## Current Reproducibility Status

The latest local rerun completed both major reproducibility paths:

| Path | Status | Notes |
| --- | --- | --- |
| Reward / post-training line | Complete | DistilBERT clickbait, quality reward, pairwise reward, v2 reward critics, OpenAI generation/judging, persona voting, objective selection |
| SFT generator line | Complete | Generic and specificity-aware FLAN-T5-small SFT models trained locally and evaluated |
| Raw-data full rebuild | Requires raw data | Raw datasets are not included in GitHub |
| Model restoration from GitHub alone | Requires retraining | Checkpoints are excluded from GitHub |

## Reproducibility Note

Exact headline strings and LLM-judge rationales can differ across reruns because OpenAI generation/judging and local SFT training are stochastic. The project fixes the 100-example evaluation seed set and reports aggregate metrics so the main conclusions remain reproducible even when individual generated titles vary.
