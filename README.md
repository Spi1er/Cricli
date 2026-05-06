# Cricli: Audience-Aware Headline Optimization

This repository contains a course project for audience-aware headline optimization with local critics, LLM-as-judge evaluation, reward-model distillation, and multi-perspective selection.

The current project focus is aligned with the original proposal:

```text
article summary
-> headline candidate generation
-> multi-dimensional evaluation
-> audience/persona voting
-> small critic / reward model distillation
-> objective-specific headline selection
```

The project does not claim to be a full autonomous agent or online RL system. Instead, it implements a practical, reproducible headline optimization workflow inspired by post-training and RLHF systems: generate candidates, label preferences, train lightweight critics, compare evaluator disagreement, and select headlines under different objectives.

## Main Components

- **Candidate generation**
  - Original human headlines from MIND.
  - API zero-shot headlines.
  - Critic-guided rewrite baselines.
  - Agentic best-of-N candidate generation.
  - Small FLAN-T5 SFT generators used as auxiliary candidate sources.

- **Local critics and reward models**
  - DistilBERT clickbait penalty critic.
  - Multi-dimensional headline quality reward critic.
  - Pairwise preference reward critic.

- **LLM-as-judge and preference data**
  - Pointwise judge scores for faithfulness, clarity, specificity, attractiveness, non-clickbait, and overall quality.
  - Pairwise preference examples for reward modeling.
  - Reward misalignment analysis between local critics and LLM judge.

- **Audience/persona evaluation**
  - Trust-sensitive reader.
  - Growth-oriented reader.
  - Busy news reader.
  - Editorial reviewer.

- **Selection layer**
  - Multi-objective candidate matrix.
  - Objective-specific selectors for trust/safety, growth, editorial quality, and specificity.

## Repository Layout

```text
data/docs/                       Dataset manifest
data/processed/                  Processed datasets, judge outputs, reports, and selection matrices
docs/                            Project summary and structure notes
scripts/                         Data processing, training, judging, scoring, and analysis scripts
requirements-clickbait-bert.txt  Core dependencies used for critic training
```

Large model weights and checkpoints are intentionally excluded from Git. The repo keeps code, reports, metrics, and small processed artifacts needed to understand and reproduce the workflow.

## Recommended Reading Order

1. `docs/PROJECT_STRUCTURE.md`
2. `docs/WORK_SUMMARY.md`
3. `data/processed/headline_multi_agent_objective_profile.md`
4. `data/processed/headline_quality_llm_judge_agentic_v3_specificity_profile.md`
5. `data/processed/headline_sft_judge_error_analysis.md`

## Current Status

The strongest completed thread is the proposal-relevant critic and selection workflow:

```text
zero-shot baseline
-> critic-guided rewrite
-> LLM judge labels
-> local reward model training
-> best-of-N / agentic reranking
-> reward-model update
-> audience/persona voting
-> objective-specific selection
```

The FLAN-T5 SFT generator work is retained as an auxiliary candidate-source experiment, not as the main project objective.
