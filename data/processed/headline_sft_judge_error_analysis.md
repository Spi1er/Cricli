# SFT Judge Error Analysis

- Input rows: 100
- Output: `data/processed/headline_sft_judge_error_analysis.csv`

## Main Result

The specificity-aware SFT model is slightly better than the generic SFT model on average, but both SFT generators are judged worse than the original human-written headlines. This means the current SFT step improves over a naive generator setup but has not yet matched the editorial target distribution.

## Mean Judge Deltas

| comparison | mean_faithfulness_delta | mean_clarity_delta | mean_specificity_delta | mean_attractiveness_delta | mean_non_clickbait_delta | mean_overall_delta |
| --- | --- | --- | --- | --- | --- | --- |
| specificity_minus_generic | 0.050 | -0.020 | 0.240 | -0.010 | 0.040 | 0.080 |
| specificity_minus_original | -0.240 | -0.310 | -0.010 | -0.790 | -0.100 | -0.370 |
| generic_minus_original | -0.290 | -0.290 | -0.250 | -0.780 | -0.140 | -0.450 |

## Local Critic vs LLM Judge Alignment

| comparison | mean_llm_overall_delta | mean_local_final_delta |
| --- | --- | --- |
| specificity_sft - generic_sft | 0.080 | 0.003 |
| specificity_sft - original | -0.370 | 0.724 |
| generic_sft - original | -0.450 | 0.721 |

## Failure Groups

| failure_group | rows |
| --- | --- |
| sft_tie | 43 |
| original_best_sft_tie | 40 |
| original_best_specificity_above_generic | 6 |
| specificity_best | 6 |
| generic_best | 4 |
| original_best_generic_above_specificity | 1 |

## Category Breakdown

| category | rows | mean_specificity_minus_generic | mean_specificity_minus_original | mean_generic_minus_original |
| --- | --- | --- | --- | --- |
| news | 25 | 0.000 | -0.520 | -0.520 |
| sports | 20 | 0.250 | -0.650 | -0.900 |
| finance | 10 | 0.000 | -0.800 | -0.800 |
| health | 10 | 0.000 | 0.100 | 0.100 |
| lifestyle | 10 | 0.000 | -0.200 | -0.200 |
| travel | 8 | 0.250 | -0.125 | -0.375 |
| foodanddrink | 7 | 0.143 | -0.286 | -0.429 |
| autos | 5 | 0.000 | -0.200 | -0.200 |
| weather | 5 | 0.000 | 0.400 | 0.400 |

## Where Specificity SFT Beats Generic SFT

| seed_id | category | specificity_minus_generic_overall | original_headline | generic_sft_headline | specificity_sft_headline | specificity_sft_rationale |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | sports | 2.000 | High school football: First-round playoff pairings | Oklahoma's first-round playoff pairings begin next week | Oklahoma's first-round playoff pairings: Class 6A-I Moore at Edmond Santa Fe Jenks at Mustang Norman at Owass | Highly specific with concrete details that engage readers without being manipulative. |
| 59 | health | 2.000 | Patient dies in ER at WellSpan York Hospital what we know now | What's a breakdown of what's going on in the death of a patient? | Patient's name has not been released, but there's a breakdown of what we do know | The headline provides good context and mentions the lack of patient name, but is slightly verbose. |
| 28 | sports | 1.000 | Frank Vogel says Lakers are 'coolest organization in the NBA,' calls coaching LeBron James and rest of team 'opportunity of a lifetime' | Lakers' Frank Vogel wants to make the best of it, and he wants to make the best of it | Lakers' Frank Vogel is aware of the great situation he's in with the Lakers, and wants to make the best of it | The headline holds a good level of detail but is lengthy and less engaging. |
| 75 | travel | 1.000 | Snow puts skids on scooters for day | Milwaukee's pilot scooter program not stopping snow for a day | Winter is not stopping Milwaukee's pilot scooter program even if snow does for a day | Very detailed but less punchy as a headline. |
| 35 | sports | 1.000 | James Paxton, Yankees know Astros can and will pick up on pitching signs | Astros are good at stealing signs and picking up on opposing pitchers' small tells | Astros' James Paxton gets hit hard, allowing five earned runs on eight hits in just four innings | Very specific but focuses on past performance rather than current implications. |
| 34 | sports | 1.000 | Do some couch scouting for Dolphins draft: Here are top prospects to watch this weekend | Miami Dolphins: Top players you can watch in South &lt;b&gt;...&lt;/b&gt | Miami Dolphins' top players in action against some of the best teams in the nation | Somewhat specific but misleads by suggesting results against top teams; lacks engagement. |
| 61 | foodanddrink | 1.000 | San Francisco chefs find nostalgia in Japanese milk bread | San Francisco chefs love sourdough bread | San Francisco chefs love white bread, but they love it | Confusing title as it repeats 'love it' without clarity and lacks precision. |
| 37 | lifestyle | 1.000 | These 12 exquisite wedding dresses are made from toilet paper | You won't believe these wedding gowns are made from only quilted Northern toilet paper, tape, glue, and a needle and thread | Wedding gowns made from quilted Northern toilet paper, tape, glue, and a needle and thread | Highly specific and clear without clickbait elements. |
| 8 | sports | 1.000 | Louis Domingue changes agents amid attempt to play at NHL level again | Domingue is a free agent, but he's not a regular at the NHL level | Domingue to be free agent this summer | It correctly states his status but doesn’t capture the full context of his career intentions. |
| 48 | travel | 1.000 | Texas quail rebounding after a dismal 2018 | 'Once people do it and experience it, it's something they'll never forget' | Wild bobwhite quail, a fading piece of Americana, are iconic in Texas | It is specific and clear, but not particularly engaging as a headline. |

## Worst Specificity SFT vs Original Examples

| seed_id | category | specificity_minus_original_overall | original_headline | specificity_sft_headline | specificity_sft_rationale |
| --- | --- | --- | --- | --- | --- |
| 41 | finance | -4.000 | Online petition to keep divisive Braves tomahawk chop nears 60,000 | Kevin Mooneyhan, Kevin Mooneyhan, Kevin Mooneyhan, Kevin Mooneyhan, Kevin Mooneyhan, Kevin Mooneyhan | Repetitive and provides no useful content. |
| 61 | foodanddrink | -3.000 | San Francisco chefs find nostalgia in Japanese milk bread | San Francisco chefs love white bread, but they love it | Confusing title as it repeats 'love it' without clarity and lacks precision. |
| 44 | finance | -3.000 | The most affordable apartments for rent in Zilker, Austin | Zilker's low-end rental price is a bit of a bargain, but it's worth it | Vague phrasing about bargains and lacks specific actionable information. |
| 38 | sports | -3.000 | Vote for Grand Rapids Player of the Week 10 | Grand Rapids' top player votes for his favorite | Incorrectly suggests a player is voting for others, which is not supported by the summary. |
| 81 | health | -3.000 | How This Guy Lost 40 Pounds and Got Shredded for His 40th Birthday | Aaron Archer loses 230 pounds and nearly 25 percent body fat | Same issue as generic, with incorrect weight loss figure. |
| 12 | sports | -3.000 | Javonte Green didn't expect to play as much as he did vs. Mavericks | Celtics are doing just the opposite, but they're doing just the opposite | Identical to the generic variant; it lacks clarity and specificity, making it ineffective. |
| 69 | lifestyle | -2.000 | 13 Things You Never Knew About Manatees | Manatees: What's the sweetest sea creature? | While it intrigues with a question, it lacks concrete facts and deviates from the original claim. |
| 73 | lifestyle | -2.000 | This Halloween Corn Maze Wedding Is Dark, Sophisticated, and Bone-Chillingly Beautiful | 'Spooky wedding shoot': 'It's a great way to get a little dark inspiration for your big day' | Similar to the generic variant, it is less engaging and specific compared to the original. |
| 74 | news | -2.000 | Don't be duped by 'phony' phone calls | The 3 On Your Side Team gets a lot of emails from viewers wondering if certain phone calls are legit or scams | Same as generic_sft; not suitable as a headline. |
| 29 | finance | -2.000 | R.I. unemployment rate stays at 3.6%, same as U.S. | Rhode Island's unemployment rage for October was 3.6 percent, the same as September | Similar issues as the generic version; lacks appeal. |

## Implications

- The SFT models often produce summary-like or overlong headlines, so the next SFT data pass should control headline style and length more tightly.
- Specificity-aware filtering helps specificity and non-clickbait scores, but it does not solve attractiveness or editorial sharpness.
- Local critics overestimated SFT outputs versus original titles, so the reward model should be updated with these SFT judge labels before being used for agentic reranking.
- The next model improvement should use SFT labels and judge feedback before adding more complex agents.
