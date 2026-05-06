# Agentic Headline Optimizer

This run generates multiple candidates per seed headline, scores them with local critics, and selects the best candidate by weighted reward.

## Configuration

- Device: `mps`
- Generator model: `gpt-4o-mini`
- Candidates per seed: 2
- Clickbait weight: 1.0
- Quality weight: 1.0
- Pairwise weight: 0.25
- Dry run: True

## Summary

- Candidate rows: 6
- Selected rows: 3
- Mean selected clickbait penalty: 0.0008
- Selected clickbait rate: 0.00%
- Mean selected quality reward: 4.1141
- Mean selected pairwise reward: 0.1295
- Mean selected final score: 4.1458
- Mean original clickbait penalty for selected seeds: 0.6673

## Selected Examples

| seed_id | category | original_title | agentic_selected_title | agentic_clickbait_penalty | agentic_quality_reward | agentic_pairwise_reward | agentic_final_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | news | NASA's Christina Koch got a little bit messy during first all-female spacewalk | News Report Details Main Developments | 0.0001 | 3.8897 | 0.1228 | 3.9203 |
| 2 | foodanddrink | Tour of Tempranillo offers taste of Lodi's take on the Spanish varietal | Tour of Tempranillo offers taste of Lodi's take on the Spanish varietal | 0.0021 | 4.6181 | 0.1311 | 4.6487 |
| 3 | travel | The Best Roller Coasters Around the World | Travel Report Details Main Developments | 0.0001 | 3.8347 | 0.1345 | 3.8682 |

## Next Training Use

- Use selected candidates as policy outputs for comparison against zero-shot and critic-guided rewrite baselines.
- Use candidate rankings as synthetic preference data: chosen = selected candidate, rejected = lower-scoring candidates from the same seed.
- Use the final score as a local reward for best-of-N sampling, rejection sampling, or later RL-style policy optimization.
