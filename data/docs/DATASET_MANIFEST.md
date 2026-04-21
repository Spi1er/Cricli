# Dataset Manifest

This folder contains the first-stage datasets for the audience-aware multi-agent headline optimization project.

## 1. MIND small derivative

Local path:

`projects/data/raw/mind_hf_rui98/`

Source:

`Rui98/mind` on Hugging Face, derived from the Microsoft News Dataset (MIND).

Downloaded files:

| File | Rows incl. header | Main columns | Suggested use |
| --- | ---: | --- | --- |
| `news_small.csv` | 65,239 | `nid`, `news_id`, `title`, `abstract`, `body`, `category`, `subvert`, `entity`, `ab_entity`, `url` | Main article/title pool for headline generation and faithfulness evaluation |
| `train_small.csv` | 236,345 | `impression_id`, `uid`, `positive`, `negative` | Pairwise preference construction from clicked vs non-clicked news |
| `dev_small.csv` | 36,577 | `impression_id`, `uid`, `impressions` | Validation split for ranking/reward evaluation |
| `test_small.csv` | 36,577 | `impression_id`, `uid`, `impressions` | Test split for ranking/reward evaluation |
| `impressions_small.csv` | 230,118 | `impression_id`, `uid`, `user_id`, `division`, `history`, `impressions`, `time` | User click/impression logs for noisy implicit preference |
| `user_interaction_small.csv` | 94,058 | `uid`, `history` | Audience/persona modeling from user history |

Notes:

- Treat clicks as noisy implicit preference signals, not direct headline quality labels.
- `category` and `subvert` can be used as coarse audience or content segments.
- `history` can support personalized/audience-aware persona simulation.

## 2. Clickbait title classification

Local path:

`projects/data/raw/clickbait/marksverdhei_clickbait_title_classification/`

Source:

`marksverdhei/clickbait_title_classification` on Hugging Face.

Downloaded files:

| File | Rows incl. header | Main columns | Suggested use |
| --- | ---: | --- | --- |
| `clickbait_title_classification.csv` | 32,001 | `title`, `clickbait` | Train or evaluate a clickbait penalty dimension for the critic/reward model |

Notes:

- This dataset only contains headlines and binary labels, not article bodies.
- It is useful as an auxiliary signal for penalizing misleading or overly clickbait-style titles.

## Recommended first processed outputs

Create these under `projects/data/processed/`:

1. `mind_headline_pool_sample.csv`
   - `news_id`, `title`, `abstract`, `body`, `category`, `subvert`

2. `mind_pairwise_preferences_sample.jsonl`
   - `summary`, `chosen_title`, `rejected_title`, `uid`, `category`, `preference_source`

3. `clickbait_penalty_train.csv`
   - `title`, `clickbait`

