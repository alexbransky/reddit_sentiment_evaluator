
# Reddit Entity & Sentiment

End-to-end pipeline to extract MOVIES and PEOPLE mentioned in a Reddit thread's comments, assign per-entity sentiment, deduplicate entity names, aggregate globally and by author, and explore results in a Streamlit UI.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm  # once
```

### Run pipeline (local, no AI API)
```python
from pipeline import run_pipeline, save_json
out = run_pipeline("https://www.reddit.com/r/HollywoodIndia/comments/.../...", use_openai=False)
save_json(out, "results.json")
```

### Run pipeline (OpenAI mode - optional)
Set `OPENAI_API_KEY` environment var and (optionally) model in `.env`:
```
cp .env.example .env  # edit with your key
```
Then:
```python
from pipeline import run_pipeline, save_json
out = run_pipeline("https://www.reddit.com/r/HollywoodIndia/comments/.../...", use_openai=True, openai_model="gpt-4.1-mini")
save_json(out, "results.json")
```

### Streamlit UI
```bash
streamlit run streamlit_app.py
```
Paste your Reddit URL and click **Fetch & Analyze**, or upload a `results.json`.

## Output JSON structure

```json
{
  "thread_url": "...",
  "generated_at": "2025-01-01T00:00:00Z",
  "mentions": [
    {
      "entity_type": "movie",
      "entity_name": "HP1",
      "canonical_name": "Harry Potter and the Sorcerer's Stone",
      "sentiment": "positive",
      "comment_id": "abc123",
      "author": "user42",
      "permalink": "https://...",
      "text": "I loved HP1..."
    }
  ],
  "dedupe_map": {
    "movie|HP1": "Harry Potter and the Sorcerer's Stone"
  },
  "aggregates": {
    "global_entity_sentiment": [
      {"entity_type":"movie","canonical_name":"The Beekeeper","counts":{"positive":7,"mixed":2,"negative":1},"total":10,"majority":"positive"}
    ],
    "author_entity_sentiment": [
      {"author":"user42","entities":[{"entity_type":"movie","canonical_name":"The Beekeeper","counts":{"positive":2},"total":2,"majority":"positive"}]}
    ]
  }
}
```

## Notes
- Local mode uses spaCy NER + simple title heuristics + VADER sentiment.
- OpenAI mode extracts entities and **per-entity** sentiment more robustly.
- Deduplication uses fuzzy token-set ratio clustering at 90% similarity; adjust as needed.
