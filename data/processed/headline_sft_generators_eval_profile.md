# SFT Headline Generator Evaluation

This report compares the generic SFT model and the specificity-aware SFT model on the same fixed 100-example headline generation seed set.

## Configuration

- Input: `/Users/pesun/STAT 5293 GenAI with LLM/Cricli/projects/data/processed/headline_generation_eval_seed_100.csv`
- Generic model: `/Users/pesun/STAT 5293 GenAI with LLM/Cricli/projects/models/headline_generator_flan_t5_small_generic_sft`
- Specificity model: `/Users/pesun/STAT 5293 GenAI with LLM/Cricli/projects/models/headline_generator_flan_t5_small_specificity_sft`
- Device: `mps`
- Output: `/Users/pesun/STAT 5293 GenAI with LLM/Cricli/projects/data/processed/headline_sft_generators_eval.csv`
- Clickbait weight: 0.5
- Quality weight: 1.0
- Pairwise weight: 0.25

## Variant Summary

| variant | rows | mean_clickbait_penalty | clickbait_rate | mean_quality_reward | mean_pairwise_reward | mean_final_score | mean_pred_overall | mean_reference_token_f1 | mean_summary_support_rate | mean_headline_words | specificity_signal_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | 100 | 0.2688 | 0.2700 | 4.0251 | 0.3824 | 3.9863 | 3.9128 | 1.0000 | 0.4521 | 10.6700 | 0.9000 |
| generic_sft | 100 | 0.2258 | 0.2300 | 4.5920 | 0.8051 | 4.6804 | 4.6334 | 0.2888 | 0.9084 | 12.3600 | 0.7700 |
| specificity_sft | 100 | 0.2114 | 0.2000 | 4.6159 | 0.8128 | 4.7134 | 4.6625 | 0.2704 | 0.9082 | 12.4300 | 0.8100 |

## Paired Final-Score Deltas

| comparison | mean_delta_final_score | median_delta_final_score | left_win_rate |
| --- | --- | --- | --- |
| specificity_sft - generic_sft | 0.0330 | 0.0000 | 0.2300 |
| specificity_sft - original | 0.7271 | 0.5201 | 0.8900 |
| generic_sft - original | 0.6941 | 0.4566 | 0.8700 |

## Best Variant By Local Final Score

| variant | best_count | best_rate |
| --- | --- | --- |
| original | 9 | 0.0900 |
| generic_sft | 70 | 0.7000 |
| specificity_sft | 21 | 0.2100 |

## Top SFT Examples

| seed_id | variant | category | headline | clickbait_penalty | quality_reward | pairwise_reward | final_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 65 | generic_sft | travel | Volunteers bring holiday cheer to Brevard County seniors as part of TRIAD's Senior Santa program | 0.0009 | 4.7073 | 1.1911 | 5.0046 |
| 65 | specificity_sft | travel | Volunteers bring holiday cheer to Brevard County seniors as part of TRIAD's Senior Santa program | 0.0009 | 4.7073 | 1.1911 | 5.0046 |
| 14 | generic_sft | finance | Critics say Procter & Gamble contributes to deforestation and should use recycled materials for paper products | 0.0007 | 4.6979 | 1.1769 | 4.9918 |
| 14 | specificity_sft | finance | Critics say Procter & Gamble contributes to deforestation and should use recycled materials for paper products | 0.0007 | 4.6979 | 1.1769 | 4.9918 |
| 6 | generic_sft | health | Airman returns home to Miami Valley, greeted by family and friends | 0.0001 | 4.7067 | 1.1079 | 4.9836 |
| 25 | generic_sft | lifestyle | Adam's Corner and Fisher House provide military families resources | 0.0001 | 4.7064 | 1.0990 | 4.9811 |
| 25 | specificity_sft | lifestyle | Adam's Corner and Fisher House provide military family resources | 0.0001 | 4.7065 | 1.0888 | 4.9786 |
| 60 | generic_sft | news | Students evacuated at Sky Valley Education Center after noxious fumes caught fire | 0.0003 | 4.7045 | 1.0865 | 4.9760 |
| 60 | specificity_sft | news | Students evacuated at Sky Valley Education Center after noxious fumes caught fire | 0.0003 | 4.7045 | 1.0865 | 4.9760 |
| 7 | specificity_sft | news | Hartford's new Weaver campus works to erase stark lines dividing the building's two school communities | 0.0002 | 4.7080 | 1.0720 | 4.9759 |
| 99 | generic_sft | news | Missouri State Auditor issued subpoena to Clay County officials pushing county to turn over documents related to citizen-mandated audit | 0.0001 | 4.6933 | 1.1237 | 4.9742 |
| 99 | specificity_sft | news | Missouri State Auditor issued subpoena to Clay County officials pushing county to turn over documents related to citizen-mandated audit | 0.0001 | 4.6933 | 1.1237 | 4.9742 |

## Interpretation

Use this as a local critic evaluation, not the final human-quality verdict. The most important next check is an LLM judge comparison between `generic_sft` and `specificity_sft`, because local critics can reward extractive or overly literal titles.
