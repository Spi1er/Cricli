# Agentic vs Baselines Local Evaluation

This report re-scores all variants with the same local critics: clickbait penalty, multi-dimensional quality reward, and pairwise reward.

## Configuration

- Device: `mps`
- Clickbait weight: 0.5
- Quality weight: 1.3
- Pairwise weight: 0.4
- Reward preset: `faithfulness_specificity`
- Output: `data/processed/headline_agentic_v3_specificity_vs_baselines_eval.csv`

## Variant Summary

| variant | rows | mean_clickbait_penalty | clickbait_rate | mean_quality_reward | mean_pairwise_reward | mean_final_score | mean_pred_overall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original | 100 | 0.2807 | 0.2800 | 4.0313 | 0.3817 | 5.2553 | 3.9238 |
| zero_shot | 100 | 0.0872 | 0.0800 | 4.6279 | 0.8133 | 6.3643 | 4.6836 |
| round1_final | 100 | 0.0714 | 0.0600 | 4.6206 | 0.8067 | 6.3594 | 4.6735 |
| round2_final | 100 | 0.0603 | 0.0500 | 4.6156 | 0.8045 | 6.3573 | 4.6670 |
| agentic_selected | 100 | 0.0626 | 0.0600 | 4.6185 | 0.8033 | 6.3582 | 4.6679 |

## Paired Final-Score Deltas

| comparison | mean_delta_final_score | median_delta_final_score | agentic_win_rate |
| --- | --- | --- | --- |
| agentic_selected - original | 1.1029 | 0.8401 | 0.9100 |
| agentic_selected - zero_shot | -0.0061 | 0.0000 | 0.4700 |
| agentic_selected - round1_final | -0.0012 | 0.0000 | 0.4800 |
| agentic_selected - round2_final | 0.0009 | 0.0000 | 0.4800 |

## Best Variant by Local Final Score

| variant | best_count | best_rate |
| --- | --- | --- |
| zero_shot | 53 | 0.5300 |
| agentic_selected | 41 | 0.4100 |
| original | 4 | 0.0400 |
| round1_final | 1 | 0.0100 |
| round2_final | 1 | 0.0100 |

## Top Agentic Selected Examples

| seed_id | category | headline | clickbait_penalty | quality_reward | pairwise_reward | final_score |
| --- | --- | --- | --- | --- | --- | --- |
| 60 | news | Sky Valley Education Center Teachers Evacuate Students After Fluorescent Lights Catch Fire | 0.0002 | 4.7200 | 1.1294 | 6.6601 |
| 65 | travel | Brevard County TRIAD's Senior Santa Program Delivers Cheer to Seniors | 0.0015 | 4.7211 | 1.1246 | 6.6566 |
| 31 | autos | Duane Roots' 1,500hp Hemi Charger Hellcat Features E90 and Nitrous | 0.0001 | 4.7168 | 1.1341 | 6.6550 |
| 25 | lifestyle | Adam's Corner and Fisher House Support Military Families with Housing and Child Care | 0.0004 | 4.7176 | 1.1241 | 6.6527 |
| 42 | weather | Fort Collins Marks 20th Anniversary of Climate Action with Free Lincoln Center Event | 0.0001 | 4.7146 | 1.0597 | 6.6286 |
| 16 | travel | Halloween Film Festival at Tampa Theatre Includes Live Podcast and Rocky Horror Talk | 0.0002 | 4.7123 | 1.0614 | 6.6261 |
| 72 | health | Connor Murphy Highlights Photo Manipulation Strategies in 2016 YouTube Video | 0.0090 | 4.7171 | 1.0538 | 6.6237 |
| 76 | news | Steny Hoyer Discusses Impeachment Decision After Hearing with Kent and Taylor | 0.0001 | 4.7022 | 1.0745 | 6.6231 |
| 50 | foodanddrink | Vice Officers Raid Building Linked to Design District Massage Parlor | 0.0001 | 4.7089 | 1.0558 | 6.6214 |
| 62 | news | Family Discovers 2,246 Preserved Fetuses at Dr. Ulrich Klopfer's Illinois Home | 0.0001 | 4.7135 | 1.0247 | 6.6143 |
| 6 | health | Local Airman Returns Home to Miami Valley to Warm Welcome | 0.0001 | 4.7237 | 1.0133 | 6.6142 |
| 99 | news | Missouri Auditor Nicole Galloway Issues Subpoena to Clay County Officials | 0.0001 | 4.7029 | 1.0436 | 6.6108 |

## Interpretation

Use this as a local reward-model evaluation, not as the final human-quality verdict. The next step is to run the LLM judge on `agentic_selected` and compare those judge scores against the earlier original / zero-shot / optimized variants.
