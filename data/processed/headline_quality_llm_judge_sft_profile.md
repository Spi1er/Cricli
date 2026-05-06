# LLM Judge SFT Generator Comparison

- Scores: `data/processed/headline_quality_llm_judge_sft_scores.csv`
- Scored rows: 300
- Pairwise preference examples: 189
- Pointwise reward examples: 300

## Mean Scores By Variant

| variant | faithfulness | clarity | specificity | attractiveness | non_clickbait | overall | clickbait_penalty |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original | 4.43 | 4.62 | 4.03 | 3.89 | 4.71 | 4.23 | 0.269 |
| generic_sft | 4.14 | 4.33 | 3.78 | 3.11 | 4.57 | 3.78 | 0.226 |
| specificity_sft | 4.19 | 4.31 | 4.02 | 3.1 | 4.61 | 3.86 | 0.211 |

## Judge Winners

| variant | best_count | worst_count |
| --- | --- | --- |
| original | 68 | 26 |
| generic_sft | 28 | 41 |
| specificity_sft | 4 | 33 |

## SFT Overall Deltas

| comparison | mean_overall_delta | median_overall_delta | specificity_win_rate |
| --- | --- | --- | --- |
| specificity_sft - original | -0.370 | -0.500 | 0.280 |
| specificity_sft - generic_sft | 0.080 | 0.000 | 0.120 |
| generic_sft - original | -0.450 | -1.000 |  |

## Interpretation

This is the LLM-judge check for whether specificity-aware SFT improves headline quality over generic SFT on the same fixed 100-example evaluation seed.

## Training Use

- `headline_quality_sft_pairwise_preferences.jsonl` can extend the pairwise reward dataset with SFT policy comparisons.
- `headline_quality_sft_reward_model_examples.jsonl` can extend pointwise reward critic training with outputs from actual small SFT generators.
- These labels are useful before moving to agentic generation, because they establish whether the base policy improved.
