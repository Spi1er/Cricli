# Agentic Headline Optimizer

This run generates multiple candidates per seed headline, scores them with local critics, and selects the best candidate by weighted reward.

## Configuration

- Device: `mps`
- Generator model: `gpt-4o-mini`
- Candidates per seed: 5
- Clickbait weight: 0.5
- Quality weight: 1.3
- Pairwise weight: 0.4
- Reward preset: `faithfulness_specificity`
- Prompt style: `specificity`
- Dry run: False

## Summary

- Candidate rows: 500
- Selected rows: 100
- Mean selected clickbait penalty: 0.0535
- Selected clickbait rate: 5.00%
- Mean selected quality reward: 4.6087
- Mean selected pairwise reward: 0.8191
- Mean selected final score: 6.3552
- Mean original clickbait penalty for selected seeds: 0.2688

## Selected Examples

| seed_id | category | original_title | agentic_selected_title | agentic_clickbait_penalty | agentic_quality_reward | agentic_pairwise_reward | agentic_final_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | news | NASA's Christina Koch got a little bit messy during first all-female spacewalk | NASA Astronauts Face Challenges of Long Spacewalks on ISS | 0.0001 | 4.5808 | 0.6845 | 6.2886 |
| 2 | foodanddrink | Tour of Tempranillo offers taste of Lodi's take on the Spanish varietal | Bokisch Vineyards Hosts Fourth Annual Lodi Tour of Tempranillo | 0.0001 | 4.7044 | 0.8512 | 6.5218 |
| 3 | travel | The Best Roller Coasters Around the World | Modern Roller Coasters Break Speed Records and Defy Expectations | 0.0001 | 4.2899 | 0.6453 | 5.8416 |
| 4 | sports | 'We always believe': Win over 49ers proves Seattle Seahawks' mindset is more than just lip service | Mike Jones Highlights Seattle Seahawks' Victory Over 49ers | 0.0002 | 4.6814 | 0.9072 | 6.5239 |
| 5 | sports | Depth charts: Projected starters for Iowa State vs. No. 23 Texas | 2019 Big 12 Matchup: Cyclones and Longhorns Starters Announced | 0.0002 | 4.7003 | 0.7517 | 6.4871 |
| 6 | health | Local Airman returns from 6 Months in Afghanistan | Local Airman Returns Home to Miami Valley on Friday | 0.0001 | 4.7090 | 1.0259 | 6.5994 |
| 7 | news | Hartford's Weaver forging a new identity at campus shared by public, magnet students | Hartford's New Weaver Campus Seeks to Unite Public and Magnet Schools | 0.0001 | 4.6872 | 0.8721 | 6.5045 |
| 8 | sports | Louis Domingue changes agents amid attempt to play at NHL level again | Unrestricted Free Agent Domingue Looks for Playing Opportunities This Summer | 0.0002 | 4.5699 | 0.8349 | 6.3273 |
| 9 | news | Driver dies after being shot in St. Paul's Summit-U area | Driver Fatally Shot in St. Paul on Sunday Night | 0.0001 | 4.6963 | 0.6109 | 6.4232 |
| 10 | sports | High school football: First-round playoff pairings | Oklahoma Playoff Schedule: Class 6A and 5A First-Round Games Listed | 0.0001 | 4.6799 | 0.9414 | 6.5358 |
| 11 | lifestyle | 50+ Amazing Things That Happened in the '50s | Decade Features Multiple Princesses and New Royal Figures | 0.0072 | 4.5485 | 0.8022 | 6.2886 |
| 12 | sports | Javonte Green didn't expect to play as much as he did vs. Mavericks | Brad Stevens Surprises with Javonte Green's Minutes Against Mavericks | 0.0001 | 4.5285 | 0.5226 | 6.1661 |

## Next Training Use

- Use selected candidates as policy outputs for comparison against zero-shot and critic-guided rewrite baselines.
- Use candidate rankings as synthetic preference data: chosen = selected candidate, rejected = lower-scoring candidates from the same seed.
- Use the final score as a local reward for best-of-N sampling, rejection sampling, or later RL-style policy optimization.
