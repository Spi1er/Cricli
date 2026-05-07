# Live URL Demo

This optional Gradio entry point keeps the original saved-case demo intact while adding a live article URL path for presentations.

## What changed

- Original demo remains: `demo/gradio_app.py`
- Live URL demo is added as: `demo/gradio_app_live_url.py`
- The live demo can fetch an article URL, extract title/summary text, generate headline candidates, score them, and recommend a final headline.
- If fetching fails, use the same manual summary workflow as before.

## Run the original demo

```bash
cd "/Users/pesun/STAT 5293 GenAI with LLM/Cricli/projects"
.venv/bin/python demo/gradio_app.py --server-name 127.0.0.1 --port 7860
```

Open:

```text
http://127.0.0.1:7860
```

## Run the live URL demo

```bash
cd "/Users/pesun/STAT 5293 GenAI with LLM/Cricli/projects"
.venv/bin/python demo/gradio_app_live_url.py --server-name 127.0.0.1 --port 7860
```

Open:

```text
http://127.0.0.1:7860
```

## Suggested demo flow

1. Start with a curated saved case in the main demo view.
2. Open `Custom Review`.
3. Paste a news article URL into `Live Article URL`.
4. Click `Fetch + Run Review`.
5. Show the extracted article summary, generated candidates, score breakdown, and final recommended headline.

For a safer live presentation, keep a short article summary ready to paste in case the website blocks automated fetching.

## Sample URL

```text
https://www.npr.org/2025/06/12/nx-s1-5430893/cdc-employees-layoffs-revoked-hhs-hepatitis-lab
```

## Notes

- Live URL extraction is best-effort. Some sites block automated requests or render article text with JavaScript.
- No new dependency is required; the extractor uses Python standard-library HTML parsing.
- API generation is optional. If `OPENAI_API_KEY` is not set or API generation is disabled, the demo uses the deterministic fallback generator.
