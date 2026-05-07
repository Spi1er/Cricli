# Agentic V3 Error Analysis

- Analysis CSV: `data/processed/headline_agentic_v3_error_analysis.csv`
- Seeds analyzed: 100

## Case Counts

| case_type | count | rate |
| --- | --- | --- |
| tie_or_mixed | 53 | 0.530 |
| zero_shot_beats_agentic | 20 | 0.200 |
| local_reward_overestimates_agentic | 16 | 0.160 |
| agentic_beats_zero_shot | 9 | 0.090 |
| local_reward_underestimates_agentic | 2 | 0.020 |

## Dimensions Where Agentic Loses To Zero-Shot

| dimension | loss_count |
| --- | --- |
|  | 59 |
| specificity | 28 |
| clarity | 21 |
| attractiveness | 20 |
| faithfulness | 20 |
| non_clickbait | 5 |

## Strong Agentic Wins Over Zero-Shot

| seed_id | category | agentic_vs_zero_overall_delta | agentic_vs_zero_local_delta | agentic_loss_dimensions_vs_zero | zero_shot_headline | agentic_selected_headline | zero_shot_rationale | agentic_selected_rationale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | travel | 1 | -0.208 |  | Roller Coasters Today: Speed Records and Thrilling Designs | Roller Coasters: Thrilling Speed Records and Innovative Designs | Captures the essence of the summary and provides specific elements. | Very engaging and covers all necessary aspects while remaining faithful. |
| 16 | travel | 1 | 0.061 |  | Tampa Theatre Hosts Halloween Film Festival with Live Shows and Talks | Halloween Film Festival at Tampa Theatre Includes Live Podcast and Rocky Horror Talk | Accurately captures the essence of the event but lacks specific details. | Most comprehensive and engaging, including key details about the events. |
| 21 | lifestyle | 1 | 0.010 |  | Worcester's School Committee election shows limited progress on racial representation | Worcester Voters Elect One Person of Color to School Committee | Clearly states the outcome of the election and its implications for racial representation. | Directly highlights the election of one person of color, making the outcome very clear and specific. |
| 41 | finance | 1 | 1.289 |  | Kevin Mooneyhan faces disappointment after Game 5 loss and losing tomahawk chop | Kevin Mooneyhan Reflects on Loss in Game 5 and Tomahawk Chop | Mixes Game loss and chop; clear but less focused. | Comprehensive and reflective; well-balanced in context. |
| 42 | weather | 1 | 0.334 |  | Fort Collins marks 20 years of climate action with free event | Fort Collins Marks 20th Anniversary of Climate Action with Free Lincoln Center Event | Clear and accurate, but lacks specific location detail. | Most specific and attractive, includes event location. |
| 59 | health | 1 | 0.238 |  | Details are limited following patient tragedy, name withheld | Patient's Identity Remains Confidential Amid Ongoing Investigation | Similar to the original but does not include the hospital name, maintaining clarity and focus on the tragedy. | Very clear and specific about the confidentiality aspect, appealing to readers interested in investigations. |
| 66 | sports | 1 | 0.120 |  | James Franklin Gives Penn State Shirt to Student Wearing Spartans Gear | Megan McNeely Receives Penn State T-Shirt from Coach Franklin on Campus | Successful clarity and specificity, well phrased. | Most specific and engaging while being fully faithful. |
| 90 | foodanddrink | 1 | 0.379 |  | Food52 Highlights Chopped Salad Popularized by Nancy Silverton at Pizzeria Mozza | Food52 Celebrates Enduring Popularity of Nancy Silverton's Chopped Salad Recipe | It provides clear, specific information about the salad and its creator. | Highlights the salad’s popularity and credits its origin effectively. |

## Strong Agentic Losses To Zero-Shot

| seed_id | category | agentic_vs_zero_overall_delta | agentic_vs_zero_local_delta | agentic_loss_dimensions_vs_zero | zero_shot_headline | agentic_selected_headline | zero_shot_rationale | agentic_selected_rationale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 68 | news | -2 | -0.491 | faithfulness, clarity, specificity, attractiveness, non_clickbait | Kentucky Representatives and Mitch McConnell Respond After Vote | Kentucky Representatives Respond to Recent Vote Outcomes | Clearly states who responded and maintains the context of the vote. | Lacks inclusion of McConnell, reducing the headline's impact. |
| 19 | health | -2 | 0.106 | faithfulness, clarity, specificity, non_clickbait | Study finds pre-breakfast exercise increases fat loss without affecting weight loss | Study Finds Morning Exercise More Effective for Fat Loss | Accurately reflects study findings and is clear and specific. | More effective for fat loss is vague; doesn't specify it doesn't affect weight. |
| 4 | sports | -1 | 0.124 | specificity, attractiveness | Seattle Seahawks Defeat Unbeaten 49ers in a Thrilling Comeback Victory | Mike Jones Highlights Seahawks' Impressive Victory Over 49ers | Clearly conveys the main event with excitement and detail. | Highlights Mike Jones but lacks the same excitement and detail about the event. |
| 47 | news | -1 | -0.055 | specificity | Unrepaired sinkhole in Lebanon causes stress for resident Debbie Carpenter | Unrepaired Sinkhole on Elm Street Causes Stress for Local Resident | Fully accurate, clear, and specific with the resident's name and highlights stress caused. | Highlights location on Elm Street, good specificity but less detail on resident's emotional state. |
| 48 | travel | -1 | 0.431 | faithfulness, clarity, specificity, non_clickbait | Hunting bobwhite quail remains a cherished tradition in Texas | Texas Parks and Wildlife Highlights Bobwhite Quail Hunting Experience | Accurately reflects the significance of quail hunting traditions and is engaging without being manipulative. | Highlights Texas Parks and Wildlife but lacks the personal and emotional touch of the hunting experience. |
| 51 | news | -1 | -0.052 | specificity | Canada's Liberals Lead After Polls Close Amid Scandals and Conservative Challenge | Liberals Lead in Four Provinces After Polls Close in Canada | Comprehensively captures the situation with mention of scandals and opposition. | Mentioned provinces but lacks context about scandals. |
| 53 | finance | -1 | 0.340 | attractiveness | Fosun Acquires Thomas Cook Brand for $14.2 Million Following Bankruptcy | Fosun Acquires Thomas Cook Brand for $14.2 Million | Informative and includes the important context of bankruptcy. | Accurate but lacks the crucial context on bankruptcy. |
| 55 | health | -1 | -0.857 | clarity | Becoming a dentist is a long and costly process in health care | Becoming a Dentist: A Long and Costly Journey in Health Care | Clear and concise summary of the process to become a dentist. | Good headline, slightly less clear due to length but still accurate. |

## Local Reward Overestimates Agentic

These are reward-misalignment examples: local reward prefers agentic, but the LLM judge gives lower overall score than zero-shot.

| seed_id | category | agentic_vs_zero_overall_delta | agentic_vs_zero_local_delta | agentic_loss_dimensions_vs_zero | zero_shot_headline | agentic_selected_headline | zero_shot_rationale | agentic_selected_rationale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 48 | travel | -1 | 0.431 | faithfulness, clarity, specificity, non_clickbait | Hunting bobwhite quail remains a cherished tradition in Texas | Texas Parks and Wildlife Highlights Bobwhite Quail Hunting Experience | Accurately reflects the significance of quail hunting traditions and is engaging without being manipulative. | Highlights Texas Parks and Wildlife but lacks the personal and emotional touch of the hunting experience. |
| 12 | sports | -1 | 0.361 | specificity | Celtics' Javonte Green steps up during close game after Hayward's injury | Javonte Green Contributes as Celtics Thrive Without Gordon Hayward | This headline is detailed, clear, and accurately presents Green's contributions in context of the game and Hayward's injury. | The headline conveys a positive spin on Green's contribution, though it's less specific about the game's context. |
| 53 | finance | -1 | 0.340 | attractiveness | Fosun Acquires Thomas Cook Brand for $14.2 Million Following Bankruptcy | Fosun Acquires Thomas Cook Brand for $14.2 Million | Informative and includes the important context of bankruptcy. | Accurate but lacks the crucial context on bankruptcy. |
| 38 | sports | -1 | 0.148 | clarity, specificity, attractiveness | Voting Open for Grand Rapids Area's Top Player from Postseason Openers | Vote for Grand Rapids' Top Player from Last Week's Playoff Openers | Clearly states the purpose and context of the voting, making it informative and engaging. | Accurate and clear but slightly less attractive than the zero-shot options. |
| 4 | sports | -1 | 0.124 | specificity, attractiveness | Seattle Seahawks Defeat Unbeaten 49ers in a Thrilling Comeback Victory | Mike Jones Highlights Seahawks' Impressive Victory Over 49ers | Clearly conveys the main event with excitement and detail. | Highlights Mike Jones but lacks the same excitement and detail about the event. |
| 19 | health | -2 | 0.106 | faithfulness, clarity, specificity, non_clickbait | Study finds pre-breakfast exercise increases fat loss without affecting weight loss | Study Finds Morning Exercise More Effective for Fat Loss | Accurately reflects study findings and is clear and specific. | More effective for fat loss is vague; doesn't specify it doesn't affect weight. |
| 40 | autos | -1 | 0.104 | clarity | Head-on collision on U.S. 127 leaves one dead, two injured | State Patrol Investigates Fatal Collision on U.S. 127 Near Camden | Covers the main facts of the collision clearly and accurately. | Highlights the investigation aspect, but less emphasis on the impact of the accident. |
| 92 | sports | -1 | 0.039 | faithfulness, clarity, specificity, attractiveness | Monty Williams praises Juwan Howard's coaching curiosity in first year at Michigan | Juwan Howard Seeks Coaching Advice from Monty Williams | This is a strong headline that encapsulates the essence of the summary while being engaging. | While relevant, it lacks the emphasis on praise for Howard's curiosity and is less comprehensive about the context. |

## Local Reward Underestimates Agentic

These are cases where the judge prefers agentic, but local reward does not. They may reveal missed reward features.

| seed_id | category | agentic_vs_zero_overall_delta | agentic_vs_zero_local_delta | agentic_loss_dimensions_vs_zero | zero_shot_headline | agentic_selected_headline | zero_shot_rationale | agentic_selected_rationale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | travel | 1 | -0.208 |  | Roller Coasters Today: Speed Records and Thrilling Designs | Roller Coasters: Thrilling Speed Records and Innovative Designs | Captures the essence of the summary and provides specific elements. | Very engaging and covers all necessary aspects while remaining faithful. |
| 96 | travel | 1 | -0.082 |  | Mopeds available for rent next month for personal exploration | Next Month, Explore the City on Your Own Moped | Accurate but lacks mention of Uber, reducing specificity and attractiveness. | Highlights personal exploration, making it very attractive while maintaining clarity and specificity. |

## Takeaways

- V3 improved by generating more specific candidate headlines, but the main remaining risks are faithfulness and clarity.
- Reward misalignment still exists: the local v2 reward can favor detailed agentic titles that the LLM judge sees as less faithful or less clear.
- The next model-side improvement should emphasize source-grounded specificity: concrete details are useful only when they are explicitly supported by the summary.
