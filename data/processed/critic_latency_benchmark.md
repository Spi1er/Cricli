# Critic Latency Benchmark

- Device: `cpu`
- Batch size: 32
- JSON: `/Users/pesun/STAT 5293 GenAI with LLM/Circli/projects/data/processed/critic_latency_benchmark.json`

| Critic | Examples | Examples/sec | ms/example | Total measured sec |
| --- | ---: | ---: | ---: | ---: |
| clickbait_penalty_distilbert | 268 | 360.36 | 2.78 | 0.744 |
| headline_quality_reward_distilbert | 268 | 67.17 | 14.89 | 3.990 |
| headline_pairwise_reward_distilbert | 135 | 33.13 | 30.18 | 4.074 |

## Interpretation

These local critics run without API calls. Use the examples/sec numbers to compare against API-based LLM judging latency and cost. The pairwise critic processes a pair as one example, but internally runs two headline encodings.
