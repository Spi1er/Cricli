# Agentic Headline Optimizer

This run generates multiple candidates per seed headline, scores them with local critics, and selects the best candidate by weighted reward.

## Configuration

- Device: `mps`
- Generator model: `existing_candidates`
- Candidates per seed: None
- Clickbait weight: 1.0
- Quality weight: 1.0
- Pairwise weight: 0.25
- Dry run: False

## Summary

- Candidate rows: 298
- Selected rows: 100
- Mean selected clickbait penalty: 0.0783
- Selected clickbait rate: 8.00%
- Mean selected quality reward: 4.5797
- Mean selected pairwise reward: 0.7678
- Mean selected final score: 4.6933
- Mean original clickbait penalty for selected seeds: 0.2688

## Selected Examples

| seed_id | category | original_title | agentic_selected_title | agentic_clickbait_penalty | agentic_quality_reward | agentic_pairwise_reward | agentic_final_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | news | NASA's Christina Koch got a little bit messy during first all-female spacewalk | Hard Work Required for NASA Astronauts on Spacewalks | 0.0001 | 4.5678 | 0.7000 | 4.7427 |
| 2 | foodanddrink | Tour of Tempranillo offers taste of Lodi's take on the Spanish varietal | Bokisch Vineyards Hosts Fourth Annual Lodi Tempranillo Tour This Weekend | 0.0001 | 4.7076 | 0.8910 | 4.9302 |
| 3 | travel | The Best Roller Coasters Around the World | The Evolution of Roller Coasters: Adrenaline and Innovation | 0.0416 | 3.7539 | 0.4691 | 3.8295 |
| 4 | sports | 'We always believe': Win over 49ers proves Seattle Seahawks' mindset is more than just lip service | Mike Jones Reports Seahawks' Impressive Win Over 49ers | 0.0001 | 4.6471 | 0.7692 | 4.8393 |
| 5 | sports | Depth charts: Projected starters for Iowa State vs. No. 23 Texas | Cyclones and Longhorns Announce Starters for Big 12 Matchup | 0.0001 | 4.7033 | 0.8036 | 4.9041 |
| 6 | health | Local Airman returns from 6 Months in Afghanistan | Local Airman Welcomed Home by Family and Friends in Miami Valley | 0.0002 | 4.7054 | 0.9569 | 4.9444 |
| 7 | news | Hartford's Weaver forging a new identity at campus shared by public, magnet students | Weaver Campus Students Work to Integrate Two School Communities | 0.0002 | 4.6840 | 0.8338 | 4.8922 |
| 8 | sports | Louis Domingue changes agents amid attempt to play at NHL level again | Domingue Looks for Playing Opportunities Ahead of Free Agency | 0.0001 | 4.3468 | 0.6272 | 4.5035 |
| 9 | news | Driver dies after being shot in St. Paul's Summit-U area | Driver Fatally Shot in St. Paul on Sunday Night | 0.0001 | 4.6963 | 0.6109 | 4.8489 |
| 10 | sports | High school football: First-round playoff pairings | Oklahoma Playoffs Start Next Week with First-Round Matchups | 0.0001 | 4.6964 | 0.9564 | 4.9354 |
| 11 | lifestyle | 50+ Amazing Things That Happened in the '50s | A Decade of Princesses, New Queen, and Future King of Pop | 0.0267 | 4.5297 | 0.7826 | 4.6987 |
| 12 | sports | Javonte Green didn't expect to play as much as he did vs. Mavericks | {"headlines":["Celtics' Javonte Green Steps Up Amid Gordon Hayward's Injury","Celtics Thrive as Javonte Green Contributes After Last Year's Absence","Javonte Green Helps Celtics Tie Game Against Mavericks 86-86"} | 0.9528 | 4.1396 | 0.7620 | 3.3773 |

## Next Training Use

- Use selected candidates as policy outputs for comparison against zero-shot and critic-guided rewrite baselines.
- Use candidate rankings as synthetic preference data: chosen = selected candidate, rejected = lower-scoring candidates from the same seed.
- Use the final score as a local reward for best-of-N sampling, rejection sampling, or later RL-style policy optimization.
