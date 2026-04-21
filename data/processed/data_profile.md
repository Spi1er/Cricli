# Data Profile

## MIND news

- Raw news rows: 65,238
- Processed headline pool rows: 10,000
- Raw abstract missing rate: 5.23%
- Raw body missing rate: 0.25%
- Headline pool title word count mean: 10.89
- Headline pool abstract word count mean: 38.74

Top processed categories:

- `news`: 3,201
- `sports`: 2,926
- `finance`: 635
- `foodanddrink`: 512
- `lifestyle`: 475
- `travel`: 472
- `weather`: 387
- `health`: 371
- `autos`: 309
- `video`: 181
- `tv`: 168
- `music`: 148

## MIND pairwise preferences

- Pairwise rows: 20,000
- Unique users: 6,025
- Unique chosen news: 1,813
- Unique rejected news: 3,016
- Same-category pair rate: 16.39%

Interpretation: these pairs are noisy implicit preferences from click logs. They are useful for engagement-oriented ranking, but they do not isolate headline quality because clicked and rejected items may have different article content.

## Clickbait penalty data

- Processed rows: 31,986
- Mean title word count: 9.07

Split counts:

| Split | Non-clickbait | Clickbait |
| --- | ---: | ---: |
| train | 12,800 | 12,788 |
| val | 1,600 | 1,599 |
| test | 1,600 | 1,599 |

## Processed files

- `mind_headline_pool_sample.csv`: clean article/title pool for generation and evaluation.
- `mind_pairwise_preferences_sample.jsonl`: clicked vs non-clicked preference pairs for reward/ranking experiments.
- `clickbait_penalty_splits.csv`: binary clickbait labels with train/val/test split for penalty critic.
