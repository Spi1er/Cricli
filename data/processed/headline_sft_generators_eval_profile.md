# SFT Headline Generator Evaluation

This report compares the generic SFT model and the specificity-aware SFT model on the same fixed 100-example headline generation seed set.

## Configuration

- Input: `data/processed/headline_generation_eval_seed_100.csv`
- Generic model: `models/headline_generator_flan_t5_small_generic_sft`
- Specificity model: `models/headline_generator_flan_t5_small_specificity_sft`
- Device: `mps`
- Output: `data/processed/headline_sft_generators_eval.csv`
- Clickbait weight: 0.5
- Quality weight: 1.0
- Pairwise weight: 0.25

## Variant Summary

| variant | rows | mean_clickbait_penalty | clickbait_rate | mean_quality_reward | mean_pairwise_reward | mean_final_score | mean_pred_overall | mean_reference_token_f1 | mean_summary_support_rate | mean_headline_words | specificity_signal_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | 100 | 0.2806 | 0.2800 | 4.0340 | 0.3824 | 3.9893 | 3.9265 | 1.0000 | 0.4521 | 10.6700 | 0.9000 |
| generic_sft | 100 | 0.2218 | 0.2200 | 4.6196 | 0.8079 | 4.7106 | 4.6702 | 0.2863 | 0.9098 | 12.3700 | 0.7800 |
| specificity_sft | 100 | 0.2398 | 0.2400 | 4.6316 | 0.8068 | 4.7134 | 4.6842 | 0.2782 | 0.9072 | 12.3100 | 0.8100 |

## Paired Final-Score Deltas

| comparison | mean_delta_final_score | median_delta_final_score | left_win_rate |
| --- | --- | --- | --- |
| specificity_sft - generic_sft | 0.0028 | 0.0000 | 0.1500 |
| specificity_sft - original | 0.7241 | 0.4786 | 0.8900 |
| generic_sft - original | 0.7213 | 0.4832 | 0.8600 |

## Best Variant By Local Final Score

| variant | best_count | best_rate |
| --- | --- | --- |
| original | 10 | 0.1000 |
| generic_sft | 76 | 0.7600 |
| specificity_sft | 14 | 0.1400 |

## Top SFT Examples

| seed_id | variant | category | headline | clickbait_penalty | quality_reward | pairwise_reward | final_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 65 | specificity_sft | travel | Volunteers bring holiday cheer to Brevard County seniors as part of TRIAD's Senior Santa program | 0.0036 | 4.7187 | 1.1911 | 5.0147 |
| 65 | generic_sft | travel | Volunteers bring holiday cheer to Brevard County seniors as part of TRIAD's Senior Santa program | 0.0036 | 4.7187 | 1.1911 | 5.0147 |
| 6 | generic_sft | health | Airman returns home to Miami Valley, greeted by family and friends | 0.0002 | 4.7206 | 1.1079 | 4.9975 |
| 25 | generic_sft | lifestyle | Adam's Corner and Fisher House provide military families resources | 0.0002 | 4.7192 | 1.0990 | 4.9938 |
| 7 | generic_sft | news | Hartford's new Weaver campus works to erase stark lines dividing two school communities | 0.0007 | 4.7236 | 1.0717 | 4.9912 |
| 25 | specificity_sft | lifestyle | Adam's Corner and Fisher House provide military family resources | 0.0002 | 4.7190 | 1.0888 | 4.9911 |
| 60 | specificity_sft | news | Students evacuated at Sky Valley Education Center after noxious fumes caught fire | 0.0030 | 4.7209 | 1.0865 | 4.9910 |
| 60 | generic_sft | news | Students evacuated at Sky Valley Education Center after noxious fumes caught fire | 0.0030 | 4.7209 | 1.0865 | 4.9910 |
| 7 | specificity_sft | news | Hartford's new Weaver campus works to erase stark lines dividing the building's two school communities | 0.0028 | 4.7218 | 1.0720 | 4.9884 |
| 99 | generic_sft | news | Missouri State Auditor issued subpoena to Clay County officials pushing county to turn over documents related to citizen-mandated audit | 0.0001 | 4.7053 | 1.1237 | 4.9862 |
| 99 | specificity_sft | news | Missouri State Auditor issued subpoena to Clay County officials pushing county to turn over documents related to citizen-mandated audit | 0.0001 | 4.7053 | 1.1237 | 4.9862 |
| 64 | specificity_sft | lifestyle | St. Bernadette Catholic School cancels classes after fire causes major damage to primary building | 0.0005 | 4.7208 | 1.0438 | 4.9815 |

## Interpretation

Use this as a local critic evaluation, not the final human-quality verdict. The most important next check is an LLM judge comparison between `generic_sft` and `specificity_sft`, because local critics can reward extractive or overly literal titles.
