# Project Structure

This document recenters the project around the proposal-related objective: audience-aware headline optimization with a fine-tuned small critic. The project should be understood as a modular evaluation and selection system, not as a full autonomous agent stack and not as a pure headline generator training project.

## 1. Core Framing

The task is:

```text
Given an article summary and an intended audience/objective,
generate or collect candidate headlines,
evaluate them from multiple perspectives,
and select the most suitable headline for that objective.
```

The project uses agent-inspired roles, but the implementation is intentionally lightweight:

- Candidate sources generate headline options.
- Evaluators score the options.
- Audience/persona evaluators estimate audience preference.
- Selection policies choose the final headline under different objectives.

This is closer to an RLHF-style and contextual bandit-style offline selection workflow than to full online RL or ReAct-style autonomous agents.

## 2. Current System Diagram

```text
MIND summaries
  |
  v
Candidate sources
  |-- original human headline
  |-- GPT zero-shot headline
  |-- critic-guided rewrite round 1 / round 2
  |-- agentic best-of-N candidates
  |-- generic SFT generator outputs
  |-- specificity-aware SFT generator outputs
  |
  v
Evaluation signals
  |-- clickbait penalty critic
  |-- local multi-dimensional reward critic
  |-- local pairwise reward critic
  |-- LLM-as-judge scores
  |-- audience/persona votes
  |-- simple style and support heuristics
  |
  v
Multi-perspective candidate matrix
  |
  v
Objective-specific selectors
  |-- trust/safety
  |-- growth
  |-- editorial quality
  |-- specificity
  |
  v
Selected headline + score breakdown
```

## 3. Module Inventory

### Candidate Sources

| Source | Role | Status |
| --- | --- | --- |
| Original MIND title | Human-written reference and candidate | Complete |
| GPT zero-shot | Strong API baseline | Complete |
| Critic-guided rewrite | Low-clickbait refinement baseline | Complete |
| Agentic candidate generator | Best-of-N candidate pool | Complete |
| Generic SFT generator | Auxiliary local generator source | Complete |
| Specificity-aware SFT generator | Auxiliary local generator source | Complete |

The SFT generators are not the main project objective. They provide extra candidate diversity and evidence about local generator limitations.

### Evaluators

| Evaluator | Role | Status |
| --- | --- | --- |
| Clickbait penalty critic | Low-cost clickbait risk model | Complete |
| Quality reward critic | Distilled multi-dimensional local judge | Complete v1/v2 |
| Pairwise reward critic | Preference-style local scorer | Complete v1/v2 |
| LLM-as-judge | High-quality reference evaluator | Complete for major comparisons |
| Audience/persona voting | Proposal-relevant audience preference layer | Complete on 100 fixed evaluation articles |
| Style/support heuristics | Length and summary-support features | Complete |

### Selection Policies

| Selector | Goal | Status |
| --- | --- | --- |
| Trust/safety | Faithful, clear, non-clickbait | Complete |
| Growth | Attractive but not manipulative | Complete |
| Editorial | Balanced, compact news headline | Complete |
| Specificity | Concrete and supported details | Complete |

## 4. Proposal Alignment

The strongest alignment is with:

- **RQ1:** iterative / agentic workflow versus zero-shot and rewrite baselines.
- **RQ3:** small critic models that approximate expensive LLM evaluation at lower inference cost.

The main caveat is:

- **RQ2:** audience persona voting is simulated with LLM personas rather than real human annotators. It is complete for the current project scope and has been summarized and integrated into the persona-calibrated selector.

## 5. What Changed From the Proposal

Good changes:

- Added explicit LLM-as-judge labels and pairwise preference datasets.
- Added local reward model distillation and reward-model iteration.
- Added reward misalignment analysis, which is highly relevant to post-training.
- Added objective-specific selection instead of collapsing headline quality into one fixed score.

Changes to keep bounded:

- FLAN-T5 SFT training is useful as an auxiliary candidate source, but should not become the project center.
- The project should not overclaim RAG, ReAct, tool-calling, or full agentic RL. These are not necessary for this task.
- "Agent" should mean role-based generation/evaluation/selection modules, especially audience/persona evaluators, not complex autonomous agents.

## 6. Current Priority

The next project work should focus on:

1. Use the completed persona voting results as audience-preference evidence in the final report.
2. Compare selector outputs with and without persona signals when space allows.
3. Add cost/latency/quality comparison between LLM judge and local critics if the final report needs another quantitative angle.
4. Prepare a final report around proposal questions RQ1, RQ2, and RQ3.

Work that should be deprioritized:

- More SFT generator versions.
- More prompt variants without a clear evaluation question.
- More reward models unless they directly support critic-quality or persona-alignment analysis.
