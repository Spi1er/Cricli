# Agentic V3 Error Analysis

- Analysis CSV: `/Users/pesun/STAT 5293 GenAI with LLM/Cricli/projects/data/processed/headline_agentic_v3_error_analysis.csv`
- Seeds analyzed: 100

## Case Counts

| case_type | count | rate |
| --- | --- | --- |
| tie_or_mixed | 42 | 0.420 |
| zero_shot_beats_agentic | 22 | 0.220 |
| local_reward_overestimates_agentic | 17 | 0.170 |
| agentic_beats_zero_shot | 15 | 0.150 |
| local_reward_underestimates_agentic | 4 | 0.040 |

## Dimensions Where Agentic Loses To Zero-Shot

| dimension | loss_count |
| --- | --- |
|  | 55 |
| specificity | 29 |
| clarity | 28 |
| faithfulness | 22 |
| attractiveness | 19 |
| non_clickbait | 10 |

## Strong Agentic Wins Over Zero-Shot

| seed_id | category | agentic_vs_zero_overall_delta | agentic_vs_zero_local_delta | agentic_loss_dimensions_vs_zero | zero_shot_headline | agentic_selected_headline | zero_shot_rationale | agentic_selected_rationale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | sports | 1 | -0.043 |  | Cyclones and Longhorns Announce Starters for 2019 Big 12 Matchup | 2019 Big 12 Matchup: Cyclones and Longhorns Starters Announced | Good clarity, but 'Announce' feels slightly vague compared to 'Projected'. | Well-balanced and informative, keeping closely to the summary. |
| 55 | health | 1 | -0.837 |  | Becoming a dentist is a long and costly process in health care | Becoming a Dentist: A Long and Costly Journey in Health Care | Accurately summarizes the process of becoming a dentist. | Engaging and informative, encapsulates the essence of the journey with good language. |
| 99 | news | 1 | 0.015 |  | Missouri Auditor Issues Subpoena for Clay County Audit Documents | Missouri Auditor Nicole Galloway Issues Subpoena to Clay County Officials | Describes the action but lacks specific mention of the citizen mandate. | Includes the auditor's name and captures the context well. |
| 91 | news | 1 | 0.434 | clarity | Arcadia Man Expected to Plead Guilty to Falsifying Records and Illegal Campaign Contributions | Imaad Shah Zuberi Faces Guilty Plea Over $1 Million in Campaign Violations | Clear but lacks emphasis on the implications of the plea and prior donations. | Highlights the individual's name and key campaign violations effectively. |
| 83 | autos | 1 | 0.059 |  | Crashes near I-265 and I-65 in Louisville cause traffic backups | Traffic Delays Reported Thursday Morning on I-265 and I-65 in Louisville | Similar to the original, maintains focus on the crashes causing backups. | Provides clear details about the timing and location of the traffic delays, which enhances specificity. |
| 82 | news | 1 | 0.046 |  | Three former Louisville officers ordered to pay restitution for overtime fraud | Louisville Police Officers Plead Guilty to Federal Wire Fraud Charges | Clear and specific but lacks mention of the federal charges. | Accurate and detailed, mentioning federal wire fraud prominently. |
| 77 | sports | 1 | -0.036 |  | 49ers offense shows potential injuries ahead of game against Arizona | 49ers' Kittle, Sanders, and Breida Miss Practice Ahead of Sunday Matchup | Mentions injuries but is less specific about key players. | Clearly lists key players missing practice, making it very informative and engaging. |
| 74 | news | 1 | 0.209 |  | 3 On Your Side shares tips to identify scam phone calls | 3 On Your Side Warns Viewers About Social Security Phone Scams | Clear and informative, though slightly general. | Directly addresses the Social Security scams, very specific. |

## Strong Agentic Losses To Zero-Shot

| seed_id | category | agentic_vs_zero_overall_delta | agentic_vs_zero_local_delta | agentic_loss_dimensions_vs_zero | zero_shot_headline | agentic_selected_headline | zero_shot_rationale | agentic_selected_rationale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 97 | weather | -2 | -0.077 | faithfulness, clarity, specificity, attractiveness, non_clickbait | Minnesota farmer faces challenges with low crop production amid early November snow | Minnesota Farmer Drives Tractor Through Snow in Graceville Video | Directly mentions the low crop production and early November snow, clearly highlighting key details. | Describes a specific event but lacks context about the challenging conditions faced by farmers. |
| 12 | sports | -2 | 0.596 | faithfulness, clarity, specificity, non_clickbait | Celtics' Javonte Green steps up during close game after Hayward's injury | Brad Stevens Surprises with Javonte Green's Minutes Against Mavericks | This headline effectively captures the situation and highlights the impact of Green's performance amid Hayward's injury. | Although the focus on Stevens adds interest, it lacks specific details about the game and the broader context. |
| 48 | travel | -1 | 0.239 | faithfulness, clarity, attractiveness, non_clickbait | Hunting bobwhite quail remains a cherished tradition in Texas | Texas Parks and Wildlife Highlights Iconic Bobwhite Quail Hunting | Accurately reflects the significance of hunting bobwhite quail as a tradition in Texas. | Focuses on Texas Parks and Wildlife but less on the emotional aspect of hunting. |
| 49 | health | -1 | -0.744 | faithfulness, specificity | Maximizing Gym Workouts in 20 to 30 Minutes for Targeted Muscle Training | Maximize Your Gym Time: Efficient Workouts in 20 to 30 Minutes | Directly addresses the main idea of effective workouts within a limited timeframe. | Focuses on maximizing gym time but is less specific about targeting muscles compared to zero_shot and optimized. |
| 51 | news | -1 | -0.028 | faithfulness, specificity, attractiveness | Canada's Liberals Lead After Polls Close Amid Scandals and Conservative Challenge | Liberals Lead in Early Polls After Close of Voting in Four Provinces | Comprehensive, addressing scandals and opposition which adds to specificity and appeal. | Clear but lacks emphasis on the challenges facing Trudeau which reduces specificity. |
| 53 | finance | -1 | 0.380 | specificity, attractiveness | Fosun Acquires Thomas Cook Brand for $14.2 Million Following Bankruptcy | Fosun Acquires Thomas Cook Brand for $14.2 Million | Includes acquisition amount and context of bankruptcy, making it informative. | Lacks mention of bankruptcy but still clear and direct. |
| 56 | news | -1 | -0.196 | faithfulness, clarity, specificity, non_clickbait | Gilroy mother accused of drowning daughter says she "just snapped" in dispatcher call | Gilroy Mother Claims She 'Just Snapped' Before Drowning Daughter | Maintains the main claim and provides clarity and specificity. | While engaging, it slightly rephrases the act of drowning, which could imply interpretation. |
| 57 | news | -1 | -0.008 | clarity, attractiveness | Rep. Debbie Wasserman Schultz Defends Calling USCIS Director Cuccinelli a White Supremacist | Rep. Debbie Wasserman Schultz Defends Remarks on USCIS Director Ken Cuccinelli | Accurate and clear, maintaining the essence of the summary. | While clear, it lacks impact and specificity about the nature of the remarks. |

## Local Reward Overestimates Agentic

These are reward-misalignment examples: local reward prefers agentic, but the LLM judge gives lower overall score than zero-shot.

| seed_id | category | agentic_vs_zero_overall_delta | agentic_vs_zero_local_delta | agentic_loss_dimensions_vs_zero | zero_shot_headline | agentic_selected_headline | zero_shot_rationale | agentic_selected_rationale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | sports | -2 | 0.596 | faithfulness, clarity, specificity, non_clickbait | Celtics' Javonte Green steps up during close game after Hayward's injury | Brad Stevens Surprises with Javonte Green's Minutes Against Mavericks | This headline effectively captures the situation and highlights the impact of Green's performance amid Hayward's injury. | Although the focus on Stevens adds interest, it lacks specific details about the game and the broader context. |
| 53 | finance | -1 | 0.380 | specificity, attractiveness | Fosun Acquires Thomas Cook Brand for $14.2 Million Following Bankruptcy | Fosun Acquires Thomas Cook Brand for $14.2 Million | Includes acquisition amount and context of bankruptcy, making it informative. | Lacks mention of bankruptcy but still clear and direct. |
| 48 | travel | -1 | 0.239 | faithfulness, clarity, attractiveness, non_clickbait | Hunting bobwhite quail remains a cherished tradition in Texas | Texas Parks and Wildlife Highlights Iconic Bobwhite Quail Hunting | Accurately reflects the significance of hunting bobwhite quail as a tradition in Texas. | Focuses on Texas Parks and Wildlife but less on the emotional aspect of hunting. |
| 61 | foodanddrink | -1 | 0.199 | faithfulness, clarity, specificity, attractiveness | San Francisco Chefs Embrace Milk Bread Amid Sourdough Tradition | Anthony Strong Discusses Milk Bread's Appeal in Bay Area Food Scene | Well-rounded; accurately portrays the trend and context without exaggeration. | Focuses on one chef, missing broader appeal and context of the trend. |
| 4 | sports | -1 | 0.195 | faithfulness, clarity, specificity, attractiveness, non_clickbait | Seattle Seahawks Defeat Unbeaten 49ers in a Thrilling Comeback Victory | Mike Jones Highlights Seattle Seahawks' Victory Over 49ers | Clearly outlines the event and is engaging without sensationalism. | Highlights Mike Jones but lacks emphasis on the game's comeback aspect. |
| 11 | lifestyle | -1 | 0.071 | clarity | Decade Features Multiple Princesses, New Queen, King, and Future King of Pop | Decade Features Multiple Princesses and New Royal Figures | Accurately reflects the summary with clear details. | Good summary but slightly less clear than others. |
| 40 | autos | -1 | 0.061 | faithfulness | Head-on collision on U.S. 127 leaves one dead, two injured | State Patrol Investigates Fatal Head-On Crash South of Camden | Accurately summarizes the incident while being clear and specific. | Focuses on the investigation but less about the collision's details. |
| 85 | weather | -1 | 0.058 | clarity | Video Captures Snow Progression in Point Place, Ohio, Amid Frozen Clock | Twitter User Shares Snow Progression Video from Point Place, Ohio | This headline captures all the essential details and is engaging without being misleading. | The headline is informative but slightly less engaging than the zero_shot and optimized options. |

## Local Reward Underestimates Agentic

These are cases where the judge prefers agentic, but local reward does not. They may reveal missed reward features.

| seed_id | category | agentic_vs_zero_overall_delta | agentic_vs_zero_local_delta | agentic_loss_dimensions_vs_zero | zero_shot_headline | agentic_selected_headline | zero_shot_rationale | agentic_selected_rationale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 55 | health | 1 | -0.837 |  | Becoming a dentist is a long and costly process in health care | Becoming a Dentist: A Long and Costly Journey in Health Care | Accurately summarizes the process of becoming a dentist. | Engaging and informative, encapsulates the essence of the journey with good language. |
| 5 | sports | 1 | -0.043 |  | Cyclones and Longhorns Announce Starters for 2019 Big 12 Matchup | 2019 Big 12 Matchup: Cyclones and Longhorns Starters Announced | Good clarity, but 'Announce' feels slightly vague compared to 'Projected'. | Well-balanced and informative, keeping closely to the summary. |
| 77 | sports | 1 | -0.036 |  | 49ers offense shows potential injuries ahead of game against Arizona | 49ers' Kittle, Sanders, and Breida Miss Practice Ahead of Sunday Matchup | Mentions injuries but is less specific about key players. | Clearly lists key players missing practice, making it very informative and engaging. |
| 9 | news | 1 | -0.021 |  | Driver Fatally Shot in St. Paul, Police Investigate Incident | Driver Fatally Shot in St. Paul on Sunday Night | Clear and accurate, but no information about location specifics. | Includes specific timing and location, making it the strongest headline. |

## Takeaways

- V3 improved by generating more specific candidate headlines, but the main remaining risks are faithfulness and clarity.
- Reward misalignment still exists: the local v2 reward can favor detailed agentic titles that the LLM judge sees as less faithful or less clear.
- The next model-side improvement should emphasize source-grounded specificity: concrete details are useful only when they are explicitly supported by the summary.
