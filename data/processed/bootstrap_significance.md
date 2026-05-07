# Bootstrap Significance Report

- Judge file: `data/processed/headline_quality_llm_judge_agentic_v3_specificity_scores.csv`
- Bootstrap iterations: 1000
- Confidence level: 95%

Each row reports the paired difference `variant_a - variant_b`; `significant` is true when the interval excludes zero.

| variant_a        | variant_b        | dimension      |   n |   mean_a |   mean_b |   delta_a_minus_b |   ci_low |   ci_high | significant   |
|:-----------------|:-----------------|:---------------|----:|---------:|---------:|------------------:|---------:|----------:|:--------------|
| zero_shot        | optimized        | faithfulness   | 100 |     4.87 |     4.84 |              0.03 |     0    |      0.07 | False         |
| zero_shot        | optimized        | clarity        | 100 |     4.94 |     4.93 |              0.01 |     0    |      0.03 | False         |
| zero_shot        | optimized        | specificity    | 100 |     4.57 |     4.53 |              0.04 |     0    |      0.1  | False         |
| zero_shot        | optimized        | attractiveness | 100 |     4.02 |     3.98 |              0.04 |     0.01 |      0.08 | True          |
| zero_shot        | optimized        | non_clickbait  | 100 |     4.99 |     4.98 |              0.01 |     0    |      0.03 | False         |
| zero_shot        | optimized        | overall        | 100 |     4.77 |     4.73 |              0.04 |     0.01 |      0.08 | True          |
| zero_shot        | agentic_selected | faithfulness   | 100 |     4.87 |     4.73 |              0.14 |     0.04 |      0.24 | True          |
| zero_shot        | agentic_selected | clarity        | 100 |     4.94 |     4.75 |              0.19 |     0.1  |      0.28 | True          |
| zero_shot        | agentic_selected | specificity    | 100 |     4.57 |     4.4  |              0.17 |     0.02 |      0.32 | True          |
| zero_shot        | agentic_selected | attractiveness | 100 |     4.02 |     4.02 |              0    |    -0.14 |      0.15 | False         |
| zero_shot        | agentic_selected | non_clickbait  | 100 |     4.99 |     4.94 |              0.05 |     0.01 |      0.09 | True          |
| zero_shot        | agentic_selected | overall        | 100 |     4.77 |     4.5  |              0.27 |     0.13 |      0.41 | True          |
| zero_shot        | original         | faithfulness   | 100 |     4.87 |     4.08 |              0.79 |     0.57 |      1.01 | True          |
| zero_shot        | original         | clarity        | 100 |     4.94 |     4.39 |              0.55 |     0.39 |      0.73 | True          |
| zero_shot        | original         | specificity    | 100 |     4.57 |     3.64 |              0.93 |     0.67 |      1.18 | True          |
| zero_shot        | original         | attractiveness | 100 |     4.02 |     3.57 |              0.45 |     0.3  |      0.61 | True          |
| zero_shot        | original         | non_clickbait  | 100 |     4.99 |     4.43 |              0.56 |     0.4  |      0.73 | True          |
| zero_shot        | original         | overall        | 100 |     4.77 |     3.81 |              0.96 |     0.74 |      1.18 | True          |
| optimized        | agentic_selected | faithfulness   | 100 |     4.84 |     4.73 |              0.11 |     0.01 |      0.2  | True          |
| optimized        | agentic_selected | clarity        | 100 |     4.93 |     4.75 |              0.18 |     0.09 |      0.27 | True          |
| optimized        | agentic_selected | specificity    | 100 |     4.53 |     4.4  |              0.13 |    -0.01 |      0.28 | False         |
| optimized        | agentic_selected | attractiveness | 100 |     3.98 |     4.02 |             -0.04 |    -0.19 |      0.11 | False         |
| optimized        | agentic_selected | non_clickbait  | 100 |     4.98 |     4.94 |              0.04 |     0    |      0.09 | False         |
| optimized        | agentic_selected | overall        | 100 |     4.73 |     4.5  |              0.23 |     0.09 |      0.38 | True          |
| optimized        | original         | faithfulness   | 100 |     4.84 |     4.08 |              0.76 |     0.53 |      0.98 | True          |
| optimized        | original         | clarity        | 100 |     4.93 |     4.39 |              0.54 |     0.38 |      0.72 | True          |
| optimized        | original         | specificity    | 100 |     4.53 |     3.64 |              0.89 |     0.63 |      1.15 | True          |
| optimized        | original         | attractiveness | 100 |     3.98 |     3.57 |              0.41 |     0.25 |      0.57 | True          |
| optimized        | original         | non_clickbait  | 100 |     4.98 |     4.43 |              0.55 |     0.39 |      0.73 | True          |
| optimized        | original         | overall        | 100 |     4.73 |     3.81 |              0.92 |     0.69 |      1.14 | True          |
| agentic_selected | original         | faithfulness   | 100 |     4.73 |     4.08 |              0.65 |     0.44 |      0.88 | True          |
| agentic_selected | original         | clarity        | 100 |     4.75 |     4.39 |              0.36 |     0.19 |      0.54 | True          |
| agentic_selected | original         | specificity    | 100 |     4.4  |     3.64 |              0.76 |     0.5  |      1.01 | True          |
| agentic_selected | original         | attractiveness | 100 |     4.02 |     3.57 |              0.45 |     0.29 |      0.61 | True          |
| agentic_selected | original         | non_clickbait  | 100 |     4.94 |     4.43 |              0.51 |     0.35 |      0.67 | True          |
| agentic_selected | original         | overall        | 100 |     4.5  |     3.81 |              0.69 |     0.47 |      0.91 | True          |
