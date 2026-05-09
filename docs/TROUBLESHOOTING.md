# Troubleshooting Guide

This guide covers common setup, reproduction, demo, and API issues for Cricli.

## 1. Run Commands From The Right Project Root

Use the canonical repository root:

```bash
cd Cricli
```

Earlier local work accidentally used a folder named `Circli`. The GitHub repository and current project root are `Cricli`. Most scripts derive paths from `PROJECT_ROOT = Path(__file__).resolve().parents[1]`, but running commands from `Cricli` avoids confusing relative outputs.

## 2. Dependencies Are Missing

Symptom examples:

```text
ModuleNotFoundError: No module named 'torch'
ModuleNotFoundError: No module named 'tabulate'
ModuleNotFoundError: No module named 'gradio'
```

Fix:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-clickbait-bert.txt
pip install -r requirements-demo.txt
```

Check core imports:

```bash
python - <<'PY'
import torch, transformers, pandas, sklearn, requests
print('ok')
PY
```

## 3. MPS Is Unavailable Or Slow

Most training and scoring scripts accept `--device auto`, `--device mps`, or `--device cpu`.

Use MPS on Apple Silicon when available:

```bash
python scripts/train_clickbait_penalty_bert.py --device mps
```

Fallback to CPU:

```bash
python scripts/train_clickbait_penalty_bert.py --device cpu
```

CPU works for correctness but can be slower for training and model scoring.

## 4. Model Weights Are Missing

GitHub intentionally excludes `models/` because model weights and checkpoints are large. If a script reports missing model paths, retrain or restore the models locally.

Start by checking assets:

```bash
python scripts/check_project_assets.py --python .venv/bin/python
```

Prepare the DistilBERT base checkpoint:

```bash
python - <<'PY'
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
out = Path('models/base/distilbert-base-uncased-seqcls')
out.mkdir(parents=True, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
tokenizer.save_pretrained(out)
model.save_pretrained(out)
print(f'saved base checkpoint to {out}')
PY
```

Then train the critics:

```bash
python scripts/train_clickbait_penalty_bert.py --device mps
python scripts/train_headline_quality_reward_critic.py --device mps
python scripts/train_headline_pairwise_reward_critic.py --device mps
```

Use `--device cpu` if MPS is not available.

## 5. OpenAI API Calls Fail

Symptoms:

```text
OPENAI_API_KEY is not set
400 Bad Request
429 Too Many Requests
OpenAI request failed after N attempts
```

Fix checklist:

1. Export the key:

```bash
export OPENAI_API_KEY='your_api_key_here'
export OPENAI_MODEL='gpt-4o-mini'
```

2. Use dry-run or fallback mode when API access is unavailable:

```bash
python scripts/run_zero_shot_headline_generation.py --dry-run
python scripts/review_single_article.py --summary 'short article summary' --force-fallback
```

3. If a partial API run failed, rerun the same script. Most API generation and judge scripts are resumable and keep `*_error` or `judge_error` columns.

4. If model parameters cause a 400 error, use the documented command defaults and avoid unsupported combinations of `temperature` and `reasoning_effort` for the selected model.

## 6. Gradio Port Is Busy

Default command:

```bash
python demo/gradio_app.py --server-name 127.0.0.1 --port 7860
```

If port 7860 is busy:

```bash
python demo/gradio_app.py --server-name 127.0.0.1 --port 7861
```

Live URL version:

```bash
python demo/gradio_app_live_url.py --server-name 127.0.0.1 --port 7860
```

Stop an old local Gradio process if needed:

```bash
pkill -f 'demo/gradio_app.py'
pkill -f 'demo/gradio_app_live_url.py'
```

## 7. Live URL Fetch Fails

The live URL demo uses best-effort HTML extraction. Some websites block automated requests, require JavaScript rendering, or hide article text behind paywalls.

Recommended presentation fallback:

1. Try `Fetch + Run Review` with a clean article URL.
2. If fetching fails, paste a short article summary into `Article Summary`.
3. Click `Run Review` and continue the same scoring/recommendation demo.

Sample URL used in documentation:

```text
https://www.npr.org/2025/06/12/nx-s1-5430893/cdc-employees-layoffs-revoked-hhs-hepatitis-lab
```

## 8. Raw Data Is Missing

`data/raw/` is not tracked in GitHub. This is expected. Current processed artifacts are tracked, so teammates can inspect results and rebuild demos without raw data.

For a full rebuild from raw files, restore the datasets described in:

```text
data/docs/DATASET_MANIFEST.md
```

Then run:

```bash
python scripts/build_processed_datasets.py
python scripts/build_headline_generation_seed.py
```

## 9. Results Do Not Match Exactly

OpenAI generation, LLM-as-judge scoring, persona voting, and SFT training are stochastic. The project is reproducible at the workflow and aggregate-metric level, not bit-for-bit identical headline text.

Use these anchors to verify that a rerun is in the expected range:

```text
Fixed evaluation seeds: 100
Agentic LLM overall, zero-shot: about 4.77
Agentic LLM overall, agentic selected v3: about 4.50
Persona-voted seeds: 100
```

## 10. Run The Smoke Tests

The repository includes lightweight tests that avoid model weights and API calls:

```bash
python -m unittest discover -s tests
```

These tests validate tracked data schemas, demo fallback scoring, live URL HTML extraction, and the asset-check integration path.
