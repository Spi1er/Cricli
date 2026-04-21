# Agentic vs Baselines Local Evaluation

This report re-scores all variants with the same local critics: clickbait penalty, multi-dimensional quality reward, and pairwise reward.

## Configuration

- Device: `mps`
- Clickbait weight: 1.0
- Quality weight: 1.0
- Pairwise weight: 0.25
- Output: `/Users/pesun/STAT 5293 GenAI with LLM/Cricli/projects/data/processed/headline_agentic_vs_baselines_eval.csv`

## Variant Summary

| variant | rows | mean_clickbait_penalty | clickbait_rate | mean_quality_reward | mean_pairwise_reward | mean_final_score | mean_pred_overall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original | 100 | 0.2733 | 0.2700 | 3.9955 | 0.0692 | 3.7395 | 3.9786 |
| zero_shot | 100 | 0.0879 | 0.0900 | 4.5613 | 0.1070 | 4.5002 | 4.6477 |
| round1_final | 100 | 0.0755 | 0.0700 | 4.5577 | 0.1064 | 4.5088 | 4.6432 |
| round2_final | 100 | 0.0656 | 0.0600 | 4.5503 | 0.1064 | 4.5114 | 4.6346 |
| agentic_selected | 100 | 0.0685 | 0.0700 | 4.5487 | 0.1013 | 4.5055 | 4.6312 |

## Paired Final-Score Deltas

| comparison | mean_delta_final_score | median_delta_final_score | agentic_win_rate |
| --- | --- | --- | --- |
| agentic_selected - original | 0.7660 | 0.4834 | 0.9100 |
| agentic_selected - zero_shot | 0.0054 | -0.0025 | 0.3900 |
| agentic_selected - round1_final | -0.0033 | -0.0008 | 0.4100 |
| agentic_selected - round2_final | -0.0058 | -0.0019 | 0.4000 |

## Best Variant by Local Final Score

| variant | best_count | best_rate |
| --- | --- | --- |
| zero_shot | 57 | 0.5700 |
| agentic_selected | 35 | 0.3500 |
| original | 5 | 0.0500 |
| round1_final | 2 | 0.0200 |
| round2_final | 1 | 0.0100 |

## Top Agentic Selected Examples

| seed_id | category | headline | clickbait_penalty | quality_reward | pairwise_reward | final_score |
| --- | --- | --- | --- | --- | --- | --- |
| 25 | lifestyle | Adam's Corner and Fisher House Support Military Families with Resources | 0.0001 | 4.6269 | 0.1751 | 4.6706 |
| 36 | health | Scientists Intentionally Infect Healthy Individuals to Study Virus Progression | 0.0003 | 4.6173 | 0.2062 | 4.6686 |
| 31 | autos | Duane Roots' 1,500hp Charger Hellcat Features E90 and Nitrous | 0.0002 | 4.6178 | 0.2019 | 4.6680 |
| 65 | travel | Brevard County TRIAD's Senior Santa Brings Cheer to Seniors | 0.0002 | 4.6258 | 0.1554 | 4.6645 |
| 78 | weather | Federal Forecasters Predict Milder Winter Across Much of the U.S | 0.0001 | 4.6229 | 0.1490 | 4.6600 |
| 26 | news | Sacramento Man Hosts Healdsburg Couple with 18 Dogs and Puppies | 0.0001 | 4.6235 | 0.1390 | 4.6581 |
| 58 | lifestyle | Mighty Cleaning Solutions Free of Harsh Chemicals and Scents | 0.0022 | 4.6058 | 0.2115 | 4.6565 |
| 45 | health | CBD Gains Popularity in Food, Drinks, and Skincare Products | 0.0001 | 4.6022 | 0.2150 | 4.6559 |
| 2 | foodanddrink | Bokisch Vineyards Hosts Fourth Annual Lodi Tempranillo Tour This Weekend | 0.0001 | 4.6208 | 0.1382 | 4.6552 |
| 14 | finance | Environmentalists Call for Recycled Materials in Procter & Gamble Products | 0.0001 | 4.6080 | 0.1829 | 4.6536 |
| 39 | sports | Hawaii Man Catches 100-Pound Ulua, Fulfilling Fishermen's Dream | 0.0001 | 4.6273 | 0.1048 | 4.6534 |
| 6 | health | Local Airman Welcomed Home by Family and Friends in Miami Valley | 0.0002 | 4.6258 | 0.1028 | 4.6514 |

## Interpretation

Use this as a local reward-model evaluation, not as the final human-quality verdict. The next step is to run the LLM judge on `agentic_selected` and compare those judge scores against the earlier original / zero-shot / optimized variants.
