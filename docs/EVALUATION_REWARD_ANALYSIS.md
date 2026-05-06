# Evaluation And Reward Analysis

Owner workstream: Evaluation / Reward / Post-training

This document is the technical analysis for Cricli's evaluation and reward-modeling component. It is intended to become the core methodology and analysis section of the final report.

## 1. Objective

The project should not be evaluated as a pure headline generator that must always beat direct GenAI generation. Direct GenAI is already a strong baseline for short headline writing.

The more useful objective is:

```text
Given an article summary and multiple candidate headlines,
evaluate each candidate from several quality and audience perspectives,
then select a publishable headline under a chosen business objective.
```

This reframes the project as an offline preference optimization and decision-support system:

```text
candidate headline = policy output
evaluator / judge = preference labeler
local critic = learned reward model
objective selector = inference-time alignment / reranking policy
```

## 2. Why This Workstream Matters

This workstream provides the technical core of the project. It answers four questions:

1. What makes a headline good?
2. Can a small local model approximate expensive LLM evaluation signals?
3. Where do local rewards disagree with LLM or audience judgments?
4. How can multiple reward signals support different business objectives?

The main contribution is not a new large generator. The main contribution is a reproducible evaluation and selection workflow inspired by post-training systems:

```text
generate candidates
-> collect LLM judge and persona preference labels
-> train compact local critics
-> rerank candidates with reward signals
-> analyze reward misalignment
-> select headlines under explicit objectives
```

## 3. Definition Of A Good Headline

A good headline is not defined by one scalar score. It depends on the product surface and audience.

The project currently evaluates the following dimensions:

| Dimension | Meaning | Why It Matters |
| --- | --- | --- |
| Faithfulness | The headline is supported by the article summary. | Prevents misleading or hallucinated titles. |
| Clarity | The headline is easy to understand. | Supports fast reading and editorial quality. |
| Specificity | The headline includes concrete supported details. | Makes the title informative rather than generic. |
| Attractiveness | The headline is appealing enough to read. | Supports engagement and growth. |
| Non-clickbait | The headline avoids manipulative wording. | Protects trust and brand safety. |
| Audience fit | The headline fits a reader or business goal. | Different audiences prefer different titles. |

This is why the system uses multiple critics and selectors instead of only optimizing one score.

## 4. Evaluation Stack

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| Clickbait penalty critic | Headline | Probability of clickbait | Cheap risk signal. |
| LLM-as-judge | Summary + candidate headlines | 1-5 scores and rationales | Reference evaluator for quality. |
| Pointwise reward critic | Summary + headline | Predicted quality dimensions | Local approximation of LLM judge. |
| Pairwise reward critic | Summary + chosen/rejected headlines | Preference score | Local preference model. |
| Persona voting | Summary + candidate headlines + persona | Persona-specific scores and winner | Audience preference layer. |
| Objective selector | Candidate matrix + reward weights | Selected headline | Business-objective-specific policy. |

## 5. Clickbait Penalty Critic

The most mature local model is the clickbait penalty critic.

Model:

- DistilBERT sequence classifier.
- Input: headline.
- Output: `P(clickbait | headline)`.

Training data:

- `data/processed/clickbait_penalty_splits.csv`
- 31,986 clickbait/non-clickbait examples.

Held-out performance:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9891 |
| F1 | 0.9890 |
| ROC-AUC | 0.9988 |

MIND headline pool profile:

| Metric | Value |
| --- | ---: |
| Input rows | 10,000 |
| Mean clickbait penalty | 0.2328 |
| Median clickbait penalty | 0.0002 |
| Predicted clickbait titles | 2,315, or 23.15% |

Interpretation:

The clickbait critic is effective as a low-cost risk detector. It is useful as a negative reward component, but it does not measure factuality, attractiveness, or audience fit. This matters because a headline can be non-clickbait and still be weak.

## 6. Baseline LLM Judge Results

Initial LLM-as-judge compared three variants:

- `original`: human-written MIND headline.
- `zero_shot`: GPT-generated headline.
- `optimized`: round-2 clickbait-critic-guided rewrite.

Mean LLM judge scores:

| Variant | Faithfulness | Clarity | Specificity | Attractiveness | Non-clickbait | Overall | Clickbait Penalty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 4.09 | 4.42 | 3.61 | 3.53 | 4.48 | 3.83 | 0.2688 |
| zero_shot | 4.88 | 4.95 | 4.61 | 4.12 | 4.96 | 4.85 | 0.0879 |
| optimized | 4.86 | 4.94 | 4.56 | 4.07 | 4.96 | 4.82 | 0.0656 |

Judge winner counts:

| Variant | Best Count | Worst Count |
| --- | ---: | ---: |
| original | 24 | 75 |
| zero_shot | 71 | 11 |
| optimized | 5 | 14 |

Key finding:

Clickbait-guided rewriting reduced clickbait penalty from 0.0879 to 0.0656, but it did not improve average LLM-judge overall quality over zero-shot. This is the first major reward-design lesson:

```text
Lower clickbait risk is valuable, but clickbait reduction alone is not the same as headline quality.
```

## 7. Local Reward Critics

Two local reward critics were trained from LLM judge outputs.

### Pointwise Reward Critic

Model:

- DistilBERT multi-output reward regressor.

Input:

```text
category + summary + headline
```

Output:

```text
faithfulness, clarity, specificity, attractiveness, non_clickbait, overall
```

Training data:

| Version | Pointwise Examples |
| --- | ---: |
| v1 | 300 |
| v2 | 700 |

Test metrics:

| Metric | v1 | v2 |
| --- | ---: | ---: |
| Test macro MAE | 0.4921 | 0.4648 |
| Faithfulness MAE | 0.5244 | 0.4597 |
| Clarity MAE | 0.3913 | 0.3464 |
| Non-clickbait MAE | 0.3745 | 0.2738 |

Interpretation:

The v2 pointwise critic improved after adding agentic-vs-baseline judge data. This supports the idea of iterative reward-model improvement.

### Pairwise Reward Critic

Model:

- DistilBERT pairwise scorer.

Objective:

```text
loss = -log sigmoid(score(chosen) - score(rejected))
```

Training data:

| Version | Pairwise Examples |
| --- | ---: |
| v1 | 167 |
| v2 | 503 |

Test metrics:

| Metric | v1 | v2 |
| --- | ---: | ---: |
| Test accuracy | 0.8462 | 0.8193 |
| Symmetric AUC | 0.8462 | 0.8378 |

Interpretation:

The v2 pairwise task is broader and harder because it includes more candidate types. The slight drop does not necessarily mean the model is worse; the evaluation distribution became more diverse.

## 8. Agentic / Best-of-N Reward-Guided Selection

The agentic selection loop should be understood as best-of-N reranking, not as full autonomous agents or online RL.

Workflow:

```text
summary
-> generate multiple candidate headlines
-> score candidates with local critics
-> scalarize reward signals
-> select the highest-scoring candidate
-> compare selected candidate with baselines using LLM judge
```

Progression:

| Version | Main Change | LLM Overall | Best Count | Delta vs Zero-shot | Clickbait Penalty |
| --- | --- | ---: | ---: | ---: | ---: |
| Agentic v1 | Initial local reward reranking | 4.33 | 2 | -0.43 | 0.068 |
| Agentic v2 | Reward critic updated with more judge data | 4.45 | 8 | -0.35 | 0.069 |
| Agentic v3 | Specificity-focused generation and reward preset | 4.49 | 17 | -0.22 | 0.053 |

Final v3 comparison:

| Variant | Faithfulness | Clarity | Specificity | Attractiveness | Non-clickbait | Overall | Clickbait Penalty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 3.96 | 4.34 | 3.63 | 3.53 | 4.38 | 3.74 | 0.273 |
| zero_shot | 4.82 | 4.91 | 4.60 | 4.00 | 4.98 | 4.71 | 0.088 |
| optimized | 4.78 | 4.89 | 4.56 | 3.98 | 4.97 | 4.68 | 0.066 |
| agentic_selected | 4.73 | 4.71 | 4.52 | 4.03 | 4.90 | 4.49 | 0.053 |

Interpretation:

Agentic v3 does not beat zero-shot on average LLM-judge overall quality. However, it narrows the gap, increases best-count wins, and achieves the lowest clickbait penalty. This is useful for a controllable selection system, but it should not be overclaimed as a better general generator.

## 9. Reward Misalignment

The strongest technical insight is reward misalignment.

Local v3 evaluation slightly preferred agentic selection over zero-shot:

| Comparison | Local Reward Delta | LLM Judge Delta |
| --- | ---: | ---: |
| agentic_selected - zero_shot | +0.021 | -0.220 |

This means the local reward model and LLM judge disagree. The local critic often rewards concrete named entities or formal wording, but the judge may prefer headlines that preserve broader context.

Error case counts:

| Case Type | Count | Rate |
| --- | ---: | ---: |
| tie_or_mixed | 42 | 0.420 |
| zero_shot_beats_agentic | 22 | 0.220 |
| local_reward_overestimates_agentic | 17 | 0.170 |
| agentic_beats_zero_shot | 15 | 0.150 |
| local_reward_underestimates_agentic | 4 | 0.040 |

Dimensions where agentic loses to zero-shot:

| Dimension | Loss Count |
| --- | ---: |
| Specificity | 29 |
| Clarity | 28 |
| Faithfulness | 22 |
| Attractiveness | 19 |
| Non-clickbait | 10 |

### Example 1: Local Reward Overestimates Specific Names

```text
zero_shot:
Celtics' Javonte Green steps up during close game after Hayward's injury

agentic_selected:
Brad Stevens Surprises with Javonte Green's Minutes Against Mavericks
```

The agentic headline includes specific names, but loses the more important causal context: Hayward's injury and Green stepping up. This is a source-grounded specificity problem.

### Example 2: Local Reward Underweights Context Completeness

```text
zero_shot:
Fosun Acquires Thomas Cook Brand for $14.2 Million Following Bankruptcy

agentic_selected:
Fosun Acquires Thomas Cook Brand for $14.2 Million
```

The agentic headline is concise and specific, but drops the bankruptcy context. The local reward does not fully penalize that missing context.

### Lesson

Specificity should not mean adding details. It should mean preserving the most important supported details from the source summary.

## 10. Persona Preference Signals

Persona voting adds the audience-aware layer promised by the proposal.

Completed run:

| Metric | Value |
| --- | ---: |
| Completed seed count | 90 |
| Persona count | 4 |
| Vote rows | 1,676 |

Personas:

- `busy_news_reader`
- `editorial_reviewer`
- `growth_reader`
- `trust_sensitive_reader`

Consensus best counts:

| Variant | Consensus Best Count |
| --- | ---: |
| zero_shot | 49 |
| original | 40 |
| generic_sft | 1 |

Persona-specific pattern:

| Persona | Strongest Variant Pattern |
| --- | --- |
| busy_news_reader | Mostly zero-shot, with many original wins. |
| editorial_reviewer | Mostly zero-shot and original. |
| growth_reader | Strongest preference for zero-shot. |
| trust_sensitive_reader | Original and zero-shot are nearly tied. |

Persona disagreement:

| Distinct Best Variants Per Seed | Seed Count |
| --- | ---: |
| 1 | 46 |
| 2 | 43 |
| 3 | 1 |

Interpretation:

The same headline is not best for every audience. Persona voting supports the idea that headline quality should be conditioned on user and business objective, not collapsed into one universal score.

## 11. Objective-Specific Selection

The multi-agent objective matrix treats headline selection as an offline contextual decision problem.

Framing:

| RL-style Term | Project Equivalent |
| --- | --- |
| State | Article summary, category, context. |
| Action | Choose one candidate headline. |
| Reward vector | Critic scores, clickbait penalty, pairwise reward, style heuristics, judge labels. |
| Policy | Objective-specific selector. |

Candidate matrix:

- 1,200 candidate actions.
- 100 fixed evaluation articles.
- Multiple candidate sources.

Objective modes:

| Objective | Selection Goal |
| --- | --- |
| trust_safety | Faithful, clear, non-clickbait headlines. |
| growth | Attractive and engaging headlines with controlled risk. |
| editorial | Balanced, compact, publication-ready headlines. |
| specificity | Concrete and source-supported details. |

Objective mean selected scores:

| Objective | Mean Clickbait Penalty | Mean Faithfulness | Mean Specificity | Mean Attractiveness | Mean LLM Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| editorial | 0.055 | 4.818 | 4.610 | 4.111 | 4.554 |
| growth | 0.069 | 4.824 | 4.625 | 4.118 | 4.476 |
| specificity | 0.065 | 4.822 | 4.620 | 4.113 | 4.333 |
| trust_safety | 0.034 | 4.801 | 4.585 | 4.096 | 4.494 |

Interpretation:

Different objectives select different candidates. This supports the product framing: Cricli is not only a generator, but a control layer for editorial decisions.

## 12. SFT Generator Findings

SFT generator experiments are useful as auxiliary candidate sources, but they should not become the center of the project.

Main result:

The specificity-aware SFT model is slightly better than the generic SFT model, but both are weaker than the original human headlines according to LLM judge.

Mean judge deltas:

| Comparison | Overall Delta |
| --- | ---: |
| specificity_sft - generic_sft | +0.080 |
| specificity_sft - original | -0.370 |
| generic_sft - original | -0.450 |

Local critic vs LLM judge alignment:

| Comparison | Mean LLM Overall Delta | Mean Local Final Delta |
| --- | ---: | ---: |
| specificity_sft - original | -0.370 | +0.727 |
| generic_sft - original | -0.450 | +0.694 |

Interpretation:

The local critic overestimated SFT outputs. This is another reward-misalignment signal. The SFT models often generate summary-like, verbose, repetitive, or awkward headlines, while local critics sometimes score them highly because they appear specific or non-clickbait.

## 13. Post-training And RLHF Relevance

This project is relevant to post-training and RLHF, but the scope should be described accurately.

What the project does cover:

| Post-training Concept | Project Implementation |
| --- | --- |
| LLM-as-judge labeling | GPT-based pointwise and pairwise headline evaluation. |
| Reward modeling | DistilBERT pointwise and pairwise reward critics. |
| Preference data | Pairwise chosen/rejected headline examples. |
| Best-of-N / rejection sampling | Generate multiple candidates and select by local reward. |
| Reward iteration | v1 reward data -> v2 reward data -> improved reranking. |
| Reward misalignment analysis | Local reward vs LLM judge disagreement examples. |
| Multi-objective alignment | Trust, growth, editorial, and specificity selectors. |

What the project does not cover:

| Not Covered | Why It Matters |
| --- | --- |
| Online RL | No live environment or sequential feedback loop. |
| PPO / policy-gradient RL | The generator is not updated by RL. |
| DPO fine-tuning | Pairwise data exists, but no DPO generator training is implemented. |
| Human preference labels | Labels are LLM/persona simulated, not human annotator labels. |
| Production A/B testing | No CTR, dwell time, or user behavior feedback. |

Best phrasing:

```text
Cricli implements an offline, RLHF-inspired preference optimization workflow for headline selection.
It uses LLM and persona labels to train local reward critics and rerank candidate headlines under multiple objectives.
```

Avoid saying:

```text
Cricli is a full agentic RL system.
```

## 14. Main Technical Claims

The following claims are supported by current results:

1. Direct GenAI is a strong baseline for headline generation.
2. A small clickbait critic can provide reliable low-cost risk scoring.
3. Lower clickbait risk does not automatically imply better headline quality.
4. Local reward critics can guide candidate reranking, but they can be misaligned with LLM judge and audience preferences.
5. Reward-model iteration improved agentic selection from v1 to v3.
6. Persona voting shows that headline preference is audience-dependent.
7. Objective-specific selection is the most defensible product value of the system.

## 15. Limitations

Important limitations:

- The evaluation seed set has only 100 articles.
- Reward-model training data is small.
- LLM-as-judge can introduce its own bias.
- Persona voting is simulated, not based on real user studies.
- Local critics are sensitive to shortcut features such as formality, length, and named entities.
- SFT generator outputs are not yet strong enough to serve as the core product claim.
- The system has no online feedback loop, CTR data, or A/B testing.

These limitations should be stated clearly in the final report. They do not invalidate the project, but they define it as a proof-of-concept workflow rather than a production optimizer.

## 16. Functional Extension: Persona-Calibrated Selection

A first functional improvement has been implemented after the initial analysis: `scripts/build_persona_calibrated_selector.py`.

The script turns persona voting from a standalone analysis artifact into an operational selection signal:

```text
base objective score
+ persona target preference adjustment
+ consensus / persona-best bonus
= persona-calibrated objective score
```

New outputs:

- `data/processed/headline_persona_calibrated_candidate_matrix.csv`
- `data/processed/headline_persona_calibrated_objective_selection.csv`
- `data/processed/headline_persona_calibrated_objective_profile.md`
- `data/processed/headline_persona_calibrated_objective_metadata.json`

Default calibration strength is `0.50`, so persona signals adjust the local critic score without fully replacing it. The strength can be changed with:

```bash
python scripts/build_persona_calibrated_selector.py --calibration-strength 1.0
```

This extension makes the project closer to the target product: a headline review console where users can switch between trust, growth, editorial, and specificity objectives and receive different recommendations.

## 17. Recommended Next Steps For This Workstream

Highest-priority next steps:

1. Use `headline_persona_calibrated_objective_selection.csv` as the selection source for the demo interface.
2. Select 3 to 5 reward-misalignment examples for the final report.
3. Add a compact table comparing original, zero-shot, optimized, agentic v1, agentic v2, and agentic v3.
4. Compare base objective selections against persona-calibrated selections in the demo.
5. If time allows, calibrate local reward against LLM judge using held-out examples and report correlation or disagreement rate.

Work to avoid:

- More prompt variants without a clear evaluation question.
- More SFT generator versions unless they directly improve the review-console demo.
- Claims that local reward selection beats zero-shot generation in general.
- Overstating the system as full RL or autonomous multi-agent reasoning.

## 18. Report-Ready Summary

A concise version for the final report:

```text
We treat headline optimization as an offline preference-selection problem. For each article, multiple candidate headlines are generated or collected from human, zero-shot, rewrite, agentic, and SFT sources. We evaluate candidates with a DistilBERT clickbait critic, LLM-as-judge labels, distilled pointwise and pairwise reward critics, persona voting, and objective-specific selection rules. The results show that direct GenAI is a strong generator, but local critics provide useful controllability and risk scoring. The main technical challenge is reward misalignment: local reward models sometimes favor formal or specific headlines that LLM judges or personas do not prefer. This motivates Cricli's final product framing as a headline review and selection console rather than a standalone headline generator.
```
