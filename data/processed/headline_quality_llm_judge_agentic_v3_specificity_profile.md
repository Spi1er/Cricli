# LLM Judge Agentic Comparison

- Scores: `data/processed/headline_quality_llm_judge_agentic_v3_specificity_scores.csv`
- Scored rows: 400
- Pairwise preference examples: 330
- Pointwise reward examples: 400

## Mean Scores By Variant

| variant | faithfulness | clarity | specificity | attractiveness | non_clickbait | overall | clickbait_penalty |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original | 4.08 | 4.39 | 3.64 | 3.57 | 4.43 | 3.81 | 0.281 |
| zero_shot | 4.87 | 4.94 | 4.57 | 4.02 | 4.99 | 4.77 | 0.087 |
| optimized | 4.84 | 4.93 | 4.53 | 3.98 | 4.98 | 4.73 | 0.060 |
| agentic_selected | 4.73 | 4.75 | 4.4 | 4.02 | 4.94 | 4.5 | 0.063 |

## Judge Winners

| variant | best_count | worst_count |
| --- | --- | --- |
| original | 26 | 51 |
| zero_shot | 60 | 1 |
| optimized | 1 | 22 |
| agentic_selected | 13 | 26 |

## Agentic Overall Deltas

| comparison | mean_overall_delta | median_overall_delta | agentic_win_rate |
| --- | --- | --- | --- |
| agentic_selected - original | 0.690 | 1.000 | 0.520 |
| agentic_selected - zero_shot | -0.270 | 0.000 | 0.110 |
| agentic_selected - optimized | -0.230 | 0.000 | 0.140 |

## Interpretation

This is the final LLM-judge check for whether the local critic selected headlines are actually preferred when compared directly against the original, zero-shot, and round-2 optimized baselines.

## Training Use

- `headline_quality_agentic_pairwise_preferences.jsonl` can extend the pairwise reward dataset with agentic-vs-baseline preferences.
- `headline_quality_agentic_reward_model_examples.jsonl` can extend the pointwise reward critic training set with a fourth policy-output variant.
- These labels are suitable for later best-of-N reranking, reward-model retraining, or policy optimization experiments.
