# Reproducibility Checklist

This checklist documents the current reproducibility state for Cricli after the latest local rerun.

## Latest Rerun Status

| Component | Status | Verification Artifact |
| --- | --- | --- |
| Processed evaluation seed | Complete | `data/processed/headline_generation_eval_seed_100.csv` |
| Clickbait critic | Complete | `models/clickbait_penalty_distilbert` |
| Quality reward critic v1/v2 | Complete | `models/headline_quality_reward_distilbert`, `models/headline_quality_reward_distilbert_v2` |
| Pairwise reward critic v1/v2 | Complete | `models/headline_pairwise_reward_distilbert`, `models/headline_pairwise_reward_distilbert_v2` |
| OpenAI candidate generation and judging | Complete in latest rerun | `data/processed/headline_quality_llm_judge_agentic_v3_specificity_scores.csv` |
| Persona voting | Complete in latest rerun | `data/processed/headline_audience_persona_votes.csv` |
| Objective selection | Complete | `data/processed/headline_persona_calibrated_objective_selection.csv` |
| Generic FLAN-T5-small SFT | Complete locally | `models/headline_generator_flan_t5_small_generic_sft` |
| Specificity-aware FLAN-T5-small SFT | Complete locally | `models/headline_generator_flan_t5_small_specificity_sft` |
| Static HTML demo | Complete | `demo/headline_review_console.html` |
| Gradio live demo | Complete locally | `demo/gradio_app.py` |

## Environment Checklist

- Use Python 3.11 or 3.12.
- Create a local virtual environment in `.venv/`.
- Install core dependencies with `pip install -r requirements-clickbait-bert.txt`.
- Install demo dependencies with `pip install -r requirements-demo.txt`.
- On Apple Silicon, use `--device mps` for local training and scoring when available.
- Use `--device cpu` if MPS is unavailable.
- Run `python scripts/check_project_assets.py --python .venv/bin/python` after setup.

Expected local asset-check status after model retraining:

| Asset | Expected Status |
| --- | --- |
| `base_distilbert_seqcls` | OK |
| `clickbait_penalty_distilbert` | OK |
| `headline_quality_reward_distilbert` | OK |
| `headline_pairwise_reward_distilbert` | OK |
| `headline_generator_generic_sft` | OK |
| `headline_generator_specificity_sft` | OK |

## Data Checklist

Tracked in GitHub:

- `data/docs/DATASET_MANIFEST.md`
- `data/processed/clickbait_penalty_splits.csv`
- `data/processed/headline_generation_eval_seed_100.csv`
- `data/processed/headline_generation_*`
- `data/processed/headline_quality_llm_judge_*`
- `data/processed/headline_audience_persona_votes.csv`
- `data/processed/headline_multi_agent_*`
- `data/processed/headline_persona_calibrated_*`
- `data/processed/headline_sft_*`
- `data/processed/headline_review_demo_*`
- `data/processed/bootstrap_significance.md`

Not tracked in GitHub:

- `data/raw/`
- `models/`
- `.venv/`
- API keys
- local single-article demo outputs
- local asset-check outputs

## Minimum Reproduction Path

Use this path when a teammate or grader only needs to inspect the current project:

```bash
git clone https://github.com/Spi1er/Cricli.git
cd Cricli
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-clickbait-bert.txt
pip install -r requirements-demo.txt
python scripts/check_project_assets.py --python .venv/bin/python
python scripts/run_product_demo.py --limit-seeds 10 --python .venv/bin/python
```

This path does not require raw data, model retraining, or API access. It can inspect tracked reports and rebuild the static demo from existing processed artifacts.

## Local Model Reproduction Path

Use this path to reproduce the local model layer from processed artifacts:

1. Prepare `models/base/distilbert-base-uncased-seqcls`.
2. Train the clickbait penalty critic.
3. Train the v1 quality and pairwise reward critics.
4. Build v2 reward training data.
5. Train the v2 quality and pairwise reward critics.
6. Train the generic and specificity-aware FLAN-T5-small SFT generators.
7. Re-run local scoring and objective selection.

This path requires Hugging Face model downloads but does not require OpenAI API access until API generation or LLM judging is refreshed.

## API Reproduction Path

Use this path to refresh OpenAI-dependent artifacts:

1. Set `OPENAI_API_KEY`.
2. Run zero-shot generation.
3. Run critic-guided rewrite and round-2 rewrite.
4. Run LLM-as-judge for baseline, agentic, and SFT comparisons.
5. Run audience/persona voting.
6. Rebuild objective and persona-calibrated selection.
7. Rebuild demo cases and demo HTML.

API calls cost money and may produce slightly different headline strings, scores, and rationales across reruns.

## Current Result Anchors

The latest rerun produced the following aggregate anchors:

| Result Area | Current Value |
| --- | ---: |
| Fixed evaluation seeds | 100 |
| Persona-voted seeds | 100 |
| Persona vote rows | 1,816 |
| Agentic LLM overall, zero-shot | 4.77 |
| Agentic LLM overall, optimized rewrite | 4.73 |
| Agentic LLM overall, agentic selected v3 | 4.50 |
| Agentic selected clickbait penalty | 0.063 |
| SFT LLM overall, original | 4.23 |
| SFT LLM overall, generic SFT | 3.78 |
| SFT LLM overall, specificity SFT | 3.86 |
| Persona consensus wins, zero-shot | 51 |
| Persona consensus wins, original | 46 |

## Path Checklist

- Run commands from the repository root, `Cricli`.
- Use relative paths in documentation and tracked metadata.
- Do not commit local absolute paths such as user home directories.
- Do not commit `models/`, `data/raw/`, `.venv/`, or single-article scratch outputs.

## Reproducibility Note

The project is reproducible at the workflow and aggregate-metric level. Exact generated headline text can differ across runs because OpenAI generation/judging and local SFT training are stochastic. The fixed 100-example evaluation seed set, tracked processed artifacts, and documented commands make the main conclusions reproducible without requiring bit-for-bit identical outputs.
