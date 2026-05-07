# Setup Notes

These notes provide the practical setup path for running Cricli locally.

## Recommended Environment

- Python: 3.11 or 3.12.
- Hardware: Apple Silicon with MPS is supported; CPU also works.
- Project root: run commands from `Cricli`.
- Virtual environment: `.venv/`.

## Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-clickbait-bert.txt
pip install -r requirements-demo.txt
```

`requirements-clickbait-bert.txt` includes the core ML stack, including `torch`, `transformers`, `pandas`, `scikit-learn`, and `tabulate`. `requirements-demo.txt` installs Gradio for the live demo.

## API Keys

OpenAI API access is needed only for API generation, LLM-as-judge, persona voting, and live custom-summary candidate generation.

```bash
export OPENAI_API_KEY="your_api_key_here"
export OPENAI_MODEL="gpt-4o-mini"
```

Do not commit API keys.

## Hugging Face Downloads

The local critic and SFT scripts download public Hugging Face models the first time they run:

| Model | Use |
| --- | --- |
| `distilbert-base-uncased` | Base checkpoint for clickbait and reward critics |
| `google/flan-t5-small` | SFT headline generator baseline |

No OpenAI API key is needed for Hugging Face downloads. These are local model downloads and training runs.

## Prepare DistilBERT Base Checkpoint

```bash
python - <<'PY'
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

out = Path("models/base/distilbert-base-uncased-seqcls")
out.mkdir(parents=True, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
)
tokenizer.save_pretrained(out)
model.save_pretrained(out)
print(f"saved base checkpoint to {out}")
PY
```

## Train Local Critics

```bash
python scripts/train_clickbait_penalty_bert.py --device mps
python scripts/train_headline_quality_reward_critic.py --device mps
python scripts/train_headline_pairwise_reward_critic.py --device mps
```

After building v2 reward data:

```bash
python scripts/build_reward_training_v2.py

python scripts/train_headline_quality_reward_critic.py \
  --data data/processed/headline_quality_reward_model_examples_v2.jsonl \
  --out models/headline_quality_reward_distilbert_v2 \
  --device mps

python scripts/train_headline_pairwise_reward_critic.py \
  --data data/processed/headline_quality_pairwise_preferences_v2.jsonl \
  --out models/headline_pairwise_reward_distilbert_v2 \
  --device mps
```

Use `--device cpu` if MPS is unavailable.

## Train SFT Generator Baselines

SFT uses `google/flan-t5-small`, not DistilBERT. DistilBERT is used for critics and reward models; FLAN-T5-small is used for headline generation.

```bash
python scripts/build_headline_sft_dataset.py

python scripts/train_headline_generator_sft.py \
  --train-csv data/processed/headline_sft_generic_train.csv \
  --val-csv data/processed/headline_sft_generic_val.csv \
  --test-csv data/processed/headline_sft_generic_test.csv \
  --output-dir models/headline_generator_flan_t5_small_generic_sft \
  --predictions data/processed/headline_generator_flan_t5_small_generic_sft_predictions.csv \
  --metadata data/processed/headline_generator_flan_t5_small_generic_sft_metadata.json \
  --device mps

python scripts/train_headline_generator_sft.py \
  --train-csv data/processed/headline_sft_specificity_train.csv \
  --val-csv data/processed/headline_sft_specificity_val.csv \
  --test-csv data/processed/headline_sft_specificity_test.csv \
  --output-dir models/headline_generator_flan_t5_small_specificity_sft \
  --predictions data/processed/headline_generator_flan_t5_small_specificity_sft_predictions.csv \
  --metadata data/processed/headline_generator_flan_t5_small_specificity_sft_metadata.json \
  --device mps
```

Then evaluate:

```bash
python scripts/evaluate_sft_generators.py --device mps
python scripts/run_llm_judge_sft_comparison.py --model gpt-4o-mini --overwrite-existing
python scripts/analyze_sft_judge_errors.py
```

## Run Demos

Static HTML demo:

```bash
python scripts/run_product_demo.py --limit-seeds 10 --python .venv/bin/python
```

This creates a compact 10-article HTML fallback plus a full 100-seed Gradio case explorer.

Gradio live demo:

```bash
python demo/gradio_app.py
```

If port 7860 is busy:

```bash
python demo/gradio_app.py --port 7861
```

## What Is Included In GitHub

- Source code under `scripts/`.
- Static and Gradio demo code under `demo/`.
- Project documentation under `docs/`.
- Dataset manifest under `data/docs/`.
- Compact processed artifacts under `data/processed/`.
- Requirements files.

## What Is Not Included In GitHub

- `models/`
- `data/raw/`
- `.venv/`
- API keys
- local asset-check outputs
- local one-off single-article outputs

## Final Sanity Check

```bash
python scripts/check_project_assets.py --python .venv/bin/python
```

The latest local rerun showed all core models as load-ready. Raw data remains local-only and is required only for a full rebuild from raw files.
