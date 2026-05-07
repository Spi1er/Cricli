# Agentic Headline Optimizer

This run generates multiple candidates per seed headline, scores them with local critics, and selects the best candidate by weighted reward.

## Configuration

- Device: `mps`
- Generator model: `gpt-4o-mini`
- Candidates per seed: 3
- Clickbait weight: 0.5
- Quality weight: 1.3
- Pairwise weight: 0.4
- Reward preset: `faithfulness_specificity`
- Prompt style: `specificity`
- Dry run: False

## Summary

- Candidate rows: 300
- Selected rows: 100
- Mean selected clickbait penalty: 0.0626
- Selected clickbait rate: 6.00%
- Mean selected quality reward: 4.6185
- Mean selected pairwise reward: 0.8033
- Mean selected final score: 6.3582
- Mean original clickbait penalty for selected seeds: 0.2688

## Selected Examples

| seed_id | category | original_title | agentic_selected_title | agentic_clickbait_penalty | agentic_quality_reward | agentic_pairwise_reward | agentic_final_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | news | NASA's Christina Koch got a little bit messy during first all-female spacewalk | NASA Astronauts Face Challenges During Spacewalks on the ISS | 0.0002 | 4.4788 | 0.6726 | 6.1435 |
| 2 | foodanddrink | Tour of Tempranillo offers taste of Lodi's take on the Spanish varietal | Lodi Hosts Fourth Annual Tour of Tempranillo This Weekend | 0.0001 | 4.7174 | 0.7589 | 6.5033 |
| 3 | travel | The Best Roller Coasters Around the World | Roller Coasters: Thrilling Speed Records and Innovative Designs | 0.0002 | 4.3738 | 0.9233 | 6.0740 |
| 4 | sports | 'We always believe': Win over 49ers proves Seattle Seahawks' mindset is more than just lip service | Mike Jones Highlights Seahawks' Impressive Victory Over 49ers | 0.0001 | 4.6641 | 0.8556 | 6.4817 |
| 5 | sports | Depth charts: Projected starters for Iowa State vs. No. 23 Texas | Cyclones and Longhorns Announce Starters for 2019 Big 12 Matchup | 0.0001 | 4.7142 | 0.8499 | 6.5468 |
| 6 | health | Local Airman returns from 6 Months in Afghanistan | Local Airman Returns Home to Miami Valley to Warm Welcome | 0.0001 | 4.7237 | 1.0133 | 6.6142 |
| 7 | news | Hartford's Weaver forging a new identity at campus shared by public, magnet students | Weaver Campus Students Address Divisions of Public and Magnet Arts Schools | 0.0004 | 4.6903 | 0.8792 | 6.5147 |
| 8 | sports | Louis Domingue changes agents amid attempt to play at NHL level again | Domingue Seeking Regular Playing Opportunity Ahead of Free Agency | 0.0001 | 4.3548 | 0.6927 | 5.9721 |
| 9 | news | Driver dies after being shot in St. Paul's Summit-U area | Police Investigate Fatal Shooting of Driver in St. Paul | 0.0001 | 4.6989 | 0.7218 | 6.4709 |
| 10 | sports | High school football: First-round playoff pairings | Oklahoma Playoffs: First-Round Pairings for All Classes Announced | 0.0001 | 4.7010 | 0.9706 | 6.5741 |
| 11 | lifestyle | 50+ Amazing Things That Happened in the '50s | Meet the New Queen and Future King of Pop This Decade | 1.0000 | 4.4466 | 0.8220 | 5.6619 |
| 12 | sports | Javonte Green didn't expect to play as much as he did vs. Mavericks | Javonte Green Contributes as Celtics Thrive Without Gordon Hayward | 0.0001 | 4.4743 | 0.4920 | 6.0788 |

## Next Training Use

- Use selected candidates as policy outputs for comparison against zero-shot and critic-guided rewrite baselines.
- Use candidate rankings as synthetic preference data: chosen = selected candidate, rejected = lower-scoring candidates from the same seed.
- Use the final score as a local reward for best-of-N sampling, rejection sampling, or later RL-style policy optimization.
