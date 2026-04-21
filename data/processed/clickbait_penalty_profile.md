# MIND Headline Clickbait Penalty Profile

- Input rows: 10,000
- Model: `/Users/pesun/STAT 5293 GenAI with LLM/Circli/projects/models/clickbait_penalty_distilbert`
- Output: `/Users/pesun/STAT 5293 GenAI with LLM/Circli/projects/data/processed/mind_headline_pool_with_clickbait_penalty.csv`
- Decision threshold: 0.50
- Mean clickbait penalty: 0.2328
- Median clickbait penalty: 0.0002
- Predicted clickbait titles: 2,315 (23.15%)

## Category Profile

| Category | Rows | Mean penalty | Median penalty | High-penalty rate |
| --- | ---: | ---: | ---: | ---: |
| foodanddrink | 512 | 0.7003 | 0.9999 | 69.92% |
| entertainment | 108 | 0.6951 | 0.9997 | 68.52% |
| lifestyle | 475 | 0.6689 | 0.9999 | 66.74% |
| kids | 3 | 0.6666 | 0.9995 | 66.67% |
| music | 148 | 0.5932 | 0.9987 | 60.14% |
| movies | 102 | 0.5384 | 0.9976 | 52.94% |
| tv | 168 | 0.5289 | 0.9173 | 53.57% |
| health | 371 | 0.4966 | 0.3428 | 49.33% |
| travel | 472 | 0.2781 | 0.0002 | 27.75% |
| finance | 635 | 0.2073 | 0.0001 | 20.79% |
| autos | 309 | 0.1951 | 0.0002 | 19.42% |
| sports | 2,926 | 0.1543 | 0.0002 | 15.35% |
| video | 181 | 0.1480 | 0.0002 | 14.36% |
| news | 3,201 | 0.1029 | 0.0001 | 10.06% |
| weather | 387 | 0.0733 | 0.0001 | 7.24% |

## Highest Penalty Examples

| Penalty | Category | Title |
| ---: | --- | --- |
| 1.0000 | health | 13 People You Didn't Know Overcame Stuttering |
| 1.0000 | lifestyle | 21 adorable photos of dogs cuddling tigers, ducks, and other animals that prove they're not just man's best friend |
| 1.0000 | music | 17 Movie Soundtracks Every Kid from the '80s Loved |
| 1.0000 | lifestyle | 15 Household Mainstays from Your Childhood That You Won't Find in New Homes |
| 1.0000 | travel | 23 stunning photos I took in Sicily that show why it should be the next place on your bucket list |
| 1.0000 | lifestyle | 25 Unusual Animals People Actually Keep as Pets |
| 1.0000 | lifestyle | 15 Fascinating Facts About Dictionaries That Will Make You Want to Pick One Up |
| 1.0000 | health | 24 Things You Should Never Tell Someone Who Is Sick |
| 1.0000 | lifestyle | 8 Hero Dogs That We Don't Deserve |
| 1.0000 | travel | 16 places around the world locals don't want you to visit |
| 1.0000 | health | 50 things every woman over 50 should know about her health |
| 1.0000 | lifestyle | 14 reasons why you should do your Black Friday shopping online |

## Lowest Penalty Examples

| Penalty | Category | Title |
| ---: | --- | --- |
| 0.0001 | finance | Williamson Co. Commission approves tax break for Mitsubishi amid competitive market |
| 0.0001 | news | Providence schools takeover agreement spells out roles of state, city |
| 0.0001 | news | Evo Morales of Bolivia Accepts Asylum in Mexico |
| 0.0001 | news | Highway blockade reveals splits in Hong Kong protest movement |
| 0.0001 | finance | Topgolf Strikes Expansion Deal to Bring Indoor Golf to China and Beyond |
| 0.0001 | news | Biden expands edge in U.S. Democratic nomination race: Reuters/Ipsos poll |
| 0.0001 | finance | Boulder Veteran Touts Benefits Of CBD Oil For Veterans |
| 0.0001 | news | Biden gains support in Pennsylvania; new poll also shows majority favor impeachment inquiry |

## Interpretation

Use `clickbait_penalty` as a negative reward component. This score estimates whether a headline has clickbait-style wording; it does not directly measure factuality or audience alignment.
