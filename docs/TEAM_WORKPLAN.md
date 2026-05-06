# Team Workplan

This document turns the current Cricli project direction into concrete workstreams that can be split across team members.

Current project goal:

```text
Build a headline review and selection workflow for content and growth teams.
The system should help users compare candidate headlines, understand risk and audience fit,
and choose a publishable headline under different business objectives.
```

The project should not be framed as a system that always writes better headlines than direct GenAI. The stronger and more realistic claim is that Cricli provides a structured decision layer on top of GenAI and human-written candidates.

## 1. Workstream Overview

| Workstream | Main Goal | Best Owner Profile | Priority |
| --- | --- | --- | --- |
| Product and use case | Make the business scenario clear and demo-ready | Product/storytelling, report writing | High |
| Evaluation and reward | Explain the post-training/RLHF-style technical core | Reward modeling, ML evaluation | High |
| Data and reproducibility | Make the repo understandable and reproducible | Careful engineering, documentation | Medium |
| Demo interface | Turn existing artifacts into a usable review console | Frontend, Streamlit/Gradio, visualization | High |
| Report and presentation | Convert the work into a polished final submission | Writing, slides, synthesis | High |

Recommended ownership if the team has four members:

| Member | Suggested Ownership | Notes |
| --- | --- | --- |
| Peiyan | Evaluation and reward | This is the strongest match for post-training, RLHF, reward modeling, and agentic selection interests. |
| Member A | Product and use case | Owns the business framing and demo story. |
| Member B | Demo interface | Builds the review console using existing processed artifacts. |
| Member C | Data, reproducibility, report integration | Keeps the repo clean and helps assemble final report/slides. |

If the team has three members, combine Product with Demo, and combine Data with Report.

## 2. Workstream A: Product And Use Case

### Goal

Define why this project matters for a real content or growth team.

The product framing should be:

```text
A content team already has candidate headlines from GenAI or human editors.
They need a review layer that compares candidates under trust, growth, editorial, and specificity objectives.
```

### Key Questions

- Who is the user: editor, growth manager, content operations reviewer, or newsletter/push notification operator?
- What decision are they making?
- What does the system show that a raw GenAI answer does not show?
- When would a user choose a safer headline over a more attractive headline?
- When would a user accept higher clickbait risk for growth?

### Tasks

1. Write 2 to 3 concrete user scenarios.
2. Define the four operating modes: trust/safety, growth, editorial, specificity.
3. Select 5 to 8 representative demo articles from the current 100 seed examples.
4. For each demo article, write a short explanation of why the recommended headline was selected.
5. Write the final report section: Problem, Users, Product Value.

### Inputs

- `README.md`
- `docs/PROJECT_STRUCTURE.md`
- `data/processed/headline_multi_agent_objective_profile.md`
- `data/processed/headline_audience_persona_votes_profile.md`
- `data/processed/headline_multi_agent_objective_selection.csv`
- `data/processed/headline_multi_agent_candidate_matrix.csv`

### Deliverables

- A short product specification section for the final report.
- A list of demo cases.
- Demo copy explaining each objective mode.
- One diagram showing the user workflow.

### Acceptance Criteria

This workstream is complete when a teammate can explain in one minute:

```text
Who uses Cricli, what problem it solves, and why it is useful even when GenAI can already generate headlines.
```

## 3. Workstream B: Evaluation And Reward

### Goal

Own the technical core: local critics, LLM-as-judge labels, reward modeling, pairwise preference modeling, persona votes, and objective-specific selection.

This workstream should connect the project to post-training and RLHF-style systems.

### Technical Framing

```text
candidate headline = policy output
LLM judge / persona vote = preference labeler
quality reward critic = learned reward model
pairwise reward critic = preference model
objective selector = inference-time alignment / reranking policy
```

### Tasks

1. Summarize the clickbait critic training result.
2. Summarize LLM-as-judge results across original, zero-shot, optimized, and agentic-selected variants.
3. Analyze why lower clickbait does not always produce better headlines.
4. Explain reward-model bias: local critics tend to prefer formal, specific, summary-like titles.
5. Compare local reward ranking with LLM judge ranking.
6. Explain persona voting and what it adds beyond a single judge.
7. Write the final technical methodology and evaluation sections.

### Inputs

- `data/processed/clickbait_penalty_profile.md`
- `data/processed/headline_quality_llm_judge_profile.md`
- `data/processed/headline_quality_llm_judge_agentic_v3_specificity_profile.md`
- `data/processed/headline_agentic_v3_error_analysis.md`
- `data/processed/headline_sft_judge_error_analysis.md`
- `data/processed/headline_audience_persona_votes_profile.md`
- `scripts/train_clickbait_penalty_bert.py`
- `scripts/train_headline_quality_reward_critic.py`
- `scripts/train_headline_pairwise_reward_critic.py`
- `scripts/build_multi_agent_objective_matrix.py`

### Deliverables

- Methodology section for reward and evaluation.
- Tables for main results.
- A short error analysis section.
- A clear paragraph explaining the post-training/RLHF relevance.

### Acceptance Criteria

This workstream is complete when the project can make a defensible technical claim:

```text
Cricli demonstrates an offline preference-optimization workflow: generate candidate actions,
collect judge and persona preferences, train compact local critics, and select headlines under explicit objectives.
```

## 4. Workstream C: Data And Reproducibility

### Goal

Make sure teammates and graders can understand what is included, what is excluded, and how to reproduce the main results.

### Tasks

1. Validate the README reproduction guide on a fresh clone or clean environment if possible.
2. Check that tracked reports use relative paths rather than local machine paths.
3. Keep `data/docs/DATASET_MANIFEST.md` aligned with the actual data files.
4. Document which files require API access to regenerate.
5. Document which model weights are excluded from Git and how to retrain them.
6. Prepare a short reproducibility checklist for the final report appendix.

### Inputs

- `README.md`
- `data/docs/DATASET_MANIFEST.md`
- `requirements-clickbait-bert.txt`
- `docs/WORK_SUMMARY.md`
- `data/processed/*.md`
- `data/processed/*.metadata.json`

### Deliverables

- Clean setup instructions.
- Updated dataset manifest if needed.
- A reproducibility appendix or checklist.
- Confirmation that no local absolute paths remain in tracked files.

### Acceptance Criteria

This workstream is complete when a teammate can clone the repo and know which reproduction level they are running:

| Level | Meaning |
| --- | --- |
| Level 1 | Read existing reports and artifacts. |
| Level 2 | Retrain local critics and rebuild local scores. |
| Level 3 | Regenerate API-based outputs and LLM/persona labels. |

## 5. Workstream D: Demo Interface

### Goal

Build a lightweight review console that shows the value of the system without requiring a production backend.

Recommended implementation: Streamlit or Gradio.

### Minimum Demo Flow

```text
Select an article
-> show article summary
-> show candidate headlines
-> show scores and persona votes
-> switch objective mode
-> show recommended headline and explanation
```

### Required Views

1. Article selector.
2. Candidate headline table.
3. Score breakdown by candidate.
4. Objective selector: trust/safety, growth, editorial, specificity.
5. Recommended headline panel.
6. Persona vote summary.

### Inputs

- `data/processed/headline_multi_agent_candidate_matrix.csv`
- `data/processed/headline_multi_agent_objective_selection.csv`
- `data/processed/headline_audience_persona_votes.csv`
- `data/processed/headline_generation_eval_seed_100.csv`

### Suggested File

```text
app.py
```

or:

```text
demo/headline_review_console.py
```

### Deliverables

- Runnable local demo.
- Screenshot or short demo walkthrough.
- A short README/demo section explaining how to launch it.

### Acceptance Criteria

This workstream is complete when a user can run one command and interact with a realistic headline selection case.

Example:

```bash
streamlit run app.py
```

## 6. Workstream E: Report And Presentation

### Goal

Turn the project into a coherent final submission.

### Recommended Report Structure

1. Introduction and motivation.
2. Product use case: headline review and selection.
3. Data and candidate sources.
4. Evaluation and reward modeling methodology.
5. Objective-specific selection and persona voting.
6. Main results.
7. Error analysis and limitations.
8. Demo description.
9. Conclusion and future work.

### Required Storyline

The final report should say:

```text
We started with headline optimization.
We found direct GenAI is a strong generator.
So we reframed the project as a decision-support system that evaluates and selects among candidate headlines.
This better matches real content workflows and gives the project a stronger practical value.
```

### Inputs

- `README.md`
- `docs/WORK_SUMMARY.md`
- `docs/PROJECT_STRUCTURE.md`
- This workplan.
- All main `data/processed/*_profile.md` reports.

### Deliverables

- Final report draft.
- Final slide deck.
- Architecture diagram.
- Demo screenshots.

### Acceptance Criteria

This workstream is complete when the report and slides make the system understandable without reading every script.

## 7. Immediate Next Steps

The next work should happen in this order:

1. Assign owners for the five workstreams.
2. Pick 5 to 8 demo articles from the existing 100-example seed set.
3. Build the lightweight demo interface from existing artifacts.
4. Write the product/use-case section and technical reward/evaluation section in parallel.
5. Run one final consistency check across README, docs, demo, and report.

## 8. Work To Avoid For Now

Avoid spending more time on:

- Additional SFT generator variants.
- New prompt variants without a clear evaluation question.
- More agent complexity such as RAG, ReAct, or tool-calling unless it directly improves the demo use case.
- Claims that the system beats direct GenAI generation in general.

The project should stay focused on review, evaluation, audience preference, and objective-specific selection.
