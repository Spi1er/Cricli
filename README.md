# Cricli: Headline Review & Selection Console

Cricli is a course project that turns headline generation into a controllable review and selection workflow for content teams.

The project is no longer framed as "our model writes better headlines than GenAI." Direct GenAI headline generation is already a strong baseline. The more realistic product value is:

```text
GenAI can generate headlines,
but editors and growth teams still need a decision layer
that compares candidates, checks risk, estimates audience fit,
and explains which headline should be published under a given objective.
```

## Product Use Case

Target users:

- News editors choosing titles for article pages or home feeds.
- Content operations teams reviewing AI-generated copy before publishing.
- Growth teams choosing titles for feeds, newsletters, push notifications, or campaign content.

Core problem:

- A single GenAI output may be fluent but not necessarily publishable.
- Different business goals prefer different headlines.
- A growth-oriented title may be engaging but risky.
- A trust-oriented title may be safer but less attractive.
- Editors need a clear reason for choosing one headline over another.

Cricli acts as a **headline review console**:

```text
article summary
-> candidate headlines from multiple sources
-> local critic scores
-> LLM-as-judge labels
-> audience/persona voting
-> objective-specific selector
-> recommended headline with score breakdown
```

## What The System Does

Given an article summary and a set of candidate headlines, the system evaluates each candidate from several perspectives:

- Clickbait risk
- Faithfulness to the article summary
- Clarity
- Specificity
- Attractiveness
- Non-clickbait quality
- Pairwise preference score
- Audience/persona preference
- Objective-specific fit

It then recommends a headline under different operating modes:

- **Trust / Safety:** prioritize faithful, clear, non-clickbait headlines.
- **Growth:** prioritize attractive and engaging headlines while controlling risk.
- **Editorial:** prioritize balanced, compact, publication-ready headlines.
- **Specificity:** prioritize concrete and supported details.

## Current Implementation

The current repository implements the research and data pipeline behind this console.

### Candidate Sources

- Original MIND human headlines.
- GPT zero-shot generated headlines.
- Critic-guided rewrite headlines.
- Best-of-N agentic candidate generation.
- Generic FLAN-T5-small SFT headline generator outputs.
- Specificity-aware FLAN-T5-small SFT headline generator outputs.

The SFT generator work is kept as an auxiliary candidate source. It is not the core product claim.

### Evaluators

- **Clickbait penalty critic:** DistilBERT classifier trained on clickbait/non-clickbait examples.
- **Quality reward critic:** DistilBERT reward regressor distilled from LLM-as-judge labels.
- **Pairwise reward critic:** local preference scorer trained from pairwise judge examples.
- **LLM-as-judge:** reference evaluator for faithfulness, clarity, specificity, attractiveness, non-clickbait, and overall quality.
- **Audience/persona voting:** simulates trust-sensitive, growth-oriented, busy-reader, and editorial-reviewer preferences.

### Selection Layer

- Multi-agent candidate matrix with 1,200 candidate actions over 100 fixed evaluation articles.
- Objective-specific selectors for trust/safety, growth, editorial quality, and specificity.
- Persona voting results over 90 evaluation articles.

## Key Findings So Far

1. **Direct GenAI is a strong headline generator.**

   Zero-shot GPT headlines are hard to beat on average LLM-judge quality. This is why the product should not claim to replace GenAI generation.

2. **Low clickbait does not guarantee a better headline.**

   Critic-guided rewrite lowered clickbait penalty, but sometimes reduced overall editorial quality.

3. **Local reward models are useful but biased.**

   The local reward critic can guide cheap reranking, but it sometimes overvalues formal, specific, summary-like titles. LLM judge and persona voting help reveal this misalignment.

4. **Headline quality is objective-dependent.**

   Trust, growth, editorial, and specificity objectives select different candidates. This supports the product idea of a review and selection console.

5. **Audience preference is not uniform.**

   Persona voting shows meaningful disagreement between trust-sensitive, growth, busy-reader, and editorial personas.

## Important Results

### Clickbait Penalty Critic

The trained DistilBERT clickbait critic achieved strong held-out performance:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9891 |
| F1 | 0.9890 |
| ROC-AUC | 0.9988 |

### Agentic / Reward-Guided Selection

Agentic v3 reduced clickbait risk and narrowed the LLM-judge gap to zero-shot:

| Variant | LLM Judge Overall | Clickbait Penalty |
| --- | ---: | ---: |
| Zero-shot | 4.71 | 0.088 |
| Optimized rewrite | 4.68 | 0.066 |
| Agentic selected v3 | 4.49 | 0.053 |

Interpretation: the system improves controllability and lowers clickbait risk, but direct zero-shot remains the strongest average generator.

### Audience Persona Voting

Persona voting was completed on 90 seed examples with 4 personas.

Consensus best counts:

| Variant | Count |
| --- | ---: |
| Zero-shot | 49 |
| Original human headline | 40 |
| Generic SFT | 1 |

Interpretation: zero-shot and human headlines remain strong, but the voting layer gives a structured way to compare audience preferences before publication.

## Repository Layout

```text
data/docs/                       Dataset manifest
data/processed/                  Processed datasets, judge outputs, reports, and selection matrices
docs/                            Project summary and structure notes
scripts/                         Data processing, training, judging, scoring, and analysis scripts
requirements-clickbait-bert.txt  Core dependencies used for critic training
```

Large model weights, checkpoints, raw data, and local virtual environments are intentionally excluded from Git.

## Recommended Reading Order

1. `docs/PROJECT_STRUCTURE.md`
2. `docs/WORK_SUMMARY.md`
3. `data/processed/headline_multi_agent_objective_profile.md`
4. `data/processed/headline_audience_persona_votes_profile.md`
5. `data/processed/headline_quality_llm_judge_agentic_v3_specificity_profile.md`
6. `data/processed/headline_sft_judge_error_analysis.md`

## Main Scripts

Candidate generation and rewriting:

- `scripts/run_zero_shot_headline_generation.py`
- `scripts/run_critic_guided_rewrite.py`
- `scripts/run_critic_guided_rewrite_round2.py`
- `scripts/run_agentic_headline_optimizer.py`

Critic and reward model training:

- `scripts/train_clickbait_penalty_bert.py`
- `scripts/train_headline_quality_reward_critic.py`
- `scripts/train_headline_pairwise_reward_critic.py`

Evaluation and selection:

- `scripts/run_llm_judge_headline_quality.py`
- `scripts/run_llm_judge_agentic_comparison.py`
- `scripts/build_multi_agent_objective_matrix.py`
- `scripts/run_audience_persona_voting.py`
- `scripts/analyze_audience_persona_votes.py`

Auxiliary SFT generator experiments:

- `scripts/build_headline_sft_dataset.py`
- `scripts/train_headline_generator_sft.py`
- `scripts/evaluate_sft_generators.py`

## Demo Direction

The final demo should be a lightweight headline review console:

```text
Select one article summary
-> show candidate headlines
-> show critic scores and persona votes
-> switch objective: trust / growth / editorial / specificity
-> show recommended headline and explanation
```

The demo should emphasize decision support, not raw generation quality.

## Project Positioning

Best short description:

> Cricli is an audience-aware headline review and selection system. It uses local critics, LLM-as-judge labels, persona voting, and objective-specific selectors to help content teams choose publishable headlines from GenAI and human-written candidates.

This framing keeps the project aligned with the original proposal while giving it a clearer practical use case.
