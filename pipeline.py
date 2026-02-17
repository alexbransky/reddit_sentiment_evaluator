from datetime import datetime, timezone
import logging
import os
import re
import json
import time
import requests
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter

from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception
from openai import RateLimitError

from rapidfuzz import fuzz

# Local NLP (only used in non-OpenAI mode; keeps option available)
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---------- setup ----------
USER_AGENT = "reddit-entity-sentiment/0.3 by alexbransky"

# env
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()
TMDB_LANG = "en-US"

# Ensure NLTK VADER is ready
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

_SIA = SentimentIntensityAnalyzer()
_SPACY = None

def _get_spacy():
    global _SPACY
    if _SPACY is None:
        try:
            _SPACY = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            _SPACY = spacy.load("en_core_web_sm")
    return _SPACY

# ---------- Reddit fetch ----------

def _reddit_json_url(thread_url: str) -> str:
    u = thread_url if thread_url.endswith("/") else thread_url + "/"
    return u + ".json?limit=500&depth=10&raw_json=1"

def fetch_comments(thread_url: str) -> List[Dict[str, Any]]:
    r = requests.get(_reddit_json_url(thread_url), headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    data = r.json()
    comments: List[Dict[str, Any]] = []

    def walk(nodes):
        for n in nodes:
            kind = n.get("kind")
            d = n.get("data", {})
            if kind == "t1":
                body = d.get("body", "")
                if body and body not in ("[deleted]", "[removed]"):
                    comments.append({
                        "id": d.get("id"),
                        "author": d.get("author"),
                        "permalink": "https://www.reddit.com" + d.get("permalink", ""),
                        "score": d.get("score"),
                        "body": body
                    })
                replies = d.get("replies")
                if isinstance(replies, dict):
                    walk(replies.get("data", {}).get("children", []))

    if isinstance(data, list) and len(data) > 1:
        walk(data[1]["data"]["children"])
    return comments

# ---------- Heuristics & helpers ----------

# Titles to allow even though they are one word
ONE_WORD_TITLE_ALLOWLIST = {"Her", "Up", "Gravity", "Roma", "Titanic", "Amélie", "Joker", "Whiplash", "Tár", "Skyfall", "Mank"}

BAD_LEADS = re.compile(r"^(it'?s|im|i'?m|not|so|because|that|this|they|we|you|he|she)\b", re.I)
TITLE_SPAN = re.compile(r"(?:\b(?:The|A|An|[A-Z][a-z0-9’']+)\b(?:\s+|$)){1,12}")
QUOTED_TITLE = re.compile(r'[“"\']([^“"\']{2,80})[”"\']')

def expand_movie_name_from_text(name: str, text: str) -> str:
    n = (name or "").strip()
    if not n:
        return n
    if len(n.split()) >= 3:
        return n
    cands = [m.group(0).strip() for m in TITLE_SPAN.finditer(text) if n.lower() in m.group(0).lower()]
    return max(cands, key=len) if cands else n

def looks_like_title(name: str) -> bool:
    if not name:
        return False
    w = name.strip()
    if BAD_LEADS.match(w):
        return False
    parts = w.split()
    if len(parts) == 1 and w not in ONE_WORD_TITLE_ALLOWLIST:
        return False
    if len(parts) < 2 or len(parts) > 8:
        return False
    up_tokens = sum(1 for p in parts if p[:1].isupper())
    if up_tokens < max(1, len(parts) // 2):
        return False
    return True

# ---------- TMDB ----------

# Cache to avoid redundant TMDB lookups for the same query within a run
_tmdb_cache: dict[str, dict | None] = {}

def tmdb_search_movie(query: str) -> dict | None:
    if not TMDB_API_KEY or not query:
        return None
    if query in _tmdb_cache:
        return _tmdb_cache[query]
    try:
        r = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"query": query, "include_adult": "false", "language": TMDB_LANG, "page": 1, "api_key": TMDB_API_KEY},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json().get("results", [])
        if not data:
            _tmdb_cache[query] = None
            return None
        # choose highest fuzzy score then popularity
        best = None
        best_score = -1
        best_pop = -1
        for item in data:
            title = item.get("title") or item.get("original_title") or ""
            score = fuzz.token_set_ratio(query, title)
            pop = item.get("popularity", 0) or 0
            if (score, pop) > (best_score, best_pop):
                best = item
                best_score = score
                best_pop = pop
        _tmdb_cache[query] = best
        return best
    except Exception:
        _tmdb_cache[query] = None
        return None

def tmdb_canonical(item: dict) -> str:
    if not item:
        return ""
    title = item.get("title") or item.get("original_title") or ""
    year = (item.get("release_date") or "")[:4]
    return f"{title} ({year})" if year else title

# ---------- Sentiment (rules) ----------

NEGATIVE_WORDS = [
    "trash", "garbage", "awful", "terrible", "horrible", "bad", "worst", "hate", "hated",
    "sucks", "sucked", "crap", "pathetic", "boring", "dumb", "stupid"
]
POSITIVE_WORDS = [
    "amazing", "awesome", "great", "fantastic", "excellent", "masterpiece", "brilliant",
    "love", "loved", "good"
]
POSITIVE_NEGATED_IDIOMS = [
    r"\bnot\s+bad\b",
    r"\bnot\s+too\s+bad\b",
    r"\bnot\s+terrible\b",
    r"\bnot\s+the\s+worst\b",
]
NEGATORS = {"not", "no", "never", "hardly", "barely", "scarcely", "is", "are", "was", "were", "do", "does", "did", "can", "could", "should", "will"}

_WORD_RE = re.compile(r"[A-Za-z']+")
_POS_RE = [re.compile(rf"\b{re.escape(w)}\b", re.I) for w in POSITIVE_WORDS]
_NEG_RE = [re.compile(rf"\b{re.escape(w)}\b", re.I) for w in NEGATIVE_WORDS]
_POS_NEGATED_IDIOMS_RE = [re.compile(p, re.I) for p in POSITIVE_NEGATED_IDIOMS]

def _normalize_neg(text: str) -> str:
    t = text.lower().replace("\u2019", "'")
    t = (t.replace("isn't", "is not").replace("wasn't", "was not")
           .replace("aren't", "are not").replace("weren't", "were not")
           .replace("don't", "do not").replace("doesn't", "does not")
           .replace("didn't", "did not").replace("can't", "can not")
           .replace("couldn't", "could not").replace("shouldn't", "should not")
           .replace("won't", "will not"))
    return t

def _token_spans(text: str):
    return list(_WORD_RE.finditer(text))

def _has_negator_before(token_spans, hit_index, window=3) -> bool:
    start = max(0, hit_index - window)
    for i in range(start, hit_index):
        if token_spans[i].group(0) in NEGATORS or token_spans[i].group(0) == "not":
            return True
    return False

def rule_based_sentiment(text: str) -> str | None:
    if not text:
        return None
    t = _normalize_neg(text)
    # idioms like "not bad"
    for r in _POS_NEGATED_IDIOMS_RE:
        if r.search(t):
            return "positive"
    spans = _token_spans(t)

    # negative words
    for r in _NEG_RE:
        for m in r.finditer(t):
            idx = None
            for i, s in enumerate(spans):
                if s.start() == m.start():
                    idx = i
                    break
            if idx is None:
                continue
            if _has_negator_before(spans, idx):
                return "positive"
            return "negative"

    # positive words
    for r in _POS_RE:
        for m in r.finditer(t):
            idx = None
            for i, s in enumerate(spans):
                if s.start() == m.start():
                    idx = i
                    break
            if idx is None:
                continue
            if _has_negator_before(spans, idx):
                return "negative"  # "not great"
            return "positive"

    return None

def vader_sentiment(text: str) -> str:
    s = _SIA.polarity_scores(text or "")
    if s["compound"] >= 0.2:
        return "positive"
    elif s["compound"] <= -0.2:
        return "negative"
    else:
        return "mixed"

# ---------- OpenAI ----------

SYSTEM_PROMPT = """Extract entities and per-entity sentiment from each Reddit comment.

Rules:
- Only MOVIES and PEOPLE.
- Use the movie's full official English title (keep leading articles; e.g., "The Shape of Water"; do not truncate to common nouns like "Water").
- For PEOPLE, return the full name if present; do not append role words.
- Sentiment is from the author's perspective TOWARD EACH ENTITY: "positive", "negative", or "mixed".
- If wording is strongly negative (e.g., "is trash", "sucks", "garbage", "awful", "hate"), classify as "negative".
- Use "mixed" only when evidence is conflicting/ambiguous; do NOT default to mixed when text is clearly evaluative.
Return strict JSON following the schema.
"""

JSON_SCHEMA = {
  "type": "object",
  "additionalProperties": False,
  "properties": {
    "results": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
          "comment_id": {"type": "string"},
          "entities": {
            "type": "array",
            "items": {
              "type": "object",
              "additionalProperties": False,
              "properties": {
                "entity_type": {"type": "string", "enum": ["movie", "person"]},
                "name": {"type": "string"},
                "sentiment": {"type": "string", "enum": ["positive", "negative", "mixed"]}
              },
              "required": ["entity_type", "name", "sentiment"]
            }
          }
        },
        "required": ["comment_id", "entities"]
      }
    }
  },
  "required": ["results"]
}

def _is_rate_limited(e: BaseException) -> bool:
    # openai SDK: RateLimitError is a subclass of APIStatusError with status_code 429
    status = getattr(e, "status_code", None)
    return isinstance(e, RateLimitError) or status == 429


@retry(
    retry=retry_if_exception(_is_rate_limited),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    stop=stop_after_attempt(3),
)
def call_openai(batch: list[dict], model: str) -> dict:
    from openai import OpenAI
    client = OpenAI(max_retries=0)

    def _trim(txt: str, limit: int = 900) -> str:
        t = txt or ""
        return t if len(t) <= limit else t[:limit] + " …"

    items = [{"comment_id": c["id"], "text": _trim(c["body"])} for c in batch]

    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "entity_sentiment",
                "schema": JSON_SCHEMA,
                "strict": True,
            },
        },
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"items": items})},
        ],
    )

    content = completion.choices[0].message.content
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {"results": []}
    if "results" not in data:
        data["results"] = []
    return data


def batched(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]

# ---------- Deduplication & Aggregations ----------

def canonicalize_entities(mentions: List[Dict[str, Any]], similarity_threshold: int = 90) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Map raw names to canonical names (per type). Prefers prefilled canonical_name (e.g., from TMDB)."""
    by_type_values = defaultdict(list)
    raw_to_canonical: Dict[str, str] = {}
    for m in mentions:
        key = f'{m["entity_type"]}|{m["entity_name"]}'
        if m.get("canonical_name"):
            raw_to_canonical[key] = m["canonical_name"]
        by_type_values[m["entity_type"]].append(m["entity_name"].strip())

    for etype, names in by_type_values.items():
        uniq = []
        seen = set()
        for n in names:
            k = (etype, n.lower())
            if k not in seen:
                seen.add(k)
                uniq.append(n)
        clusters: List[List[str]] = []
        for name in sorted(uniq, key=lambda s: (len(s), s.lower()), reverse=True):
            if f"{etype}|{name}" in raw_to_canonical:
                continue
            placed = False
            for cl in clusters:
                anchor = cl[0]
                if fuzz.token_set_ratio(name, anchor) >= similarity_threshold:
                    cl.append(name)
                    placed = True
                    break
            if not placed:
                clusters.append([name])
        for cl in clusters:
            canonical = max(cl, key=len)
            for n in cl:
                key = f"{etype}|{n}"
                if key not in raw_to_canonical:
                    raw_to_canonical[key] = canonical

    for m in mentions:
        key = f'{m["entity_type"]}|{m["entity_name"]}'
        m["canonical_name"] = m.get("canonical_name") or raw_to_canonical.get(key, m["entity_name"])

    final_map = {}
    for m in mentions:
        final_map[f'{m["entity_type"]}|{m["entity_name"]}'] = m["canonical_name"]
    return mentions, final_map

def aggregate(mentions: List[Dict[str, Any]]) -> Dict[str, Any]:
    global_counts = defaultdict(lambda: Counter())
    author_counts = defaultdict(lambda: defaultdict(lambda: Counter()))
    for m in mentions:
        key = (m["entity_type"], m["canonical_name"])
        global_counts[key][m["sentiment"]] += 1
        author = m.get("author") or "unknown"
        author_counts[author][key][m["sentiment"]] += 1

    def summarize_counter(c: Counter):
        total = sum(c.values())
        if total == 0:
            return {"counts": dict(c), "total": 0, "majority": "unclear"}
        max_val = max(c.values())
        top = [k for k, v in c.items() if v == max_val]
        lead_share = max_val / total
        majority = "unclear" if (len(top) > 1 or lead_share < 0.5) else top[0]
        return {"counts": dict(c), "total": total, "majority": majority, "lead_share": round(lead_share, 3)}

    global_summary = [
        {"entity_type": et, "canonical_name": nm, **summarize_counter(cnt)}
        for (et, nm), cnt in sorted(global_counts.items(), key=lambda kv: (-sum(kv[1].values()), kv[0][1].lower()))
    ]

    author_summary = []
    for author, ents in author_counts.items():
        ent_list = [
            {"entity_type": et, "canonical_name": nm, **summarize_counter(cnt)}
            for (et, nm), cnt in ents.items()
        ]
        author_summary.append({"author": author, "entities": sorted(ent_list, key=lambda e: -e["total"])})

    return {
        "global_entity_sentiment": global_summary,
        "author_entity_sentiment": sorted(author_summary, key=lambda a: a["author"].lower())
    }

# ---------- Orchestrator ----------

def run_pipeline(
    thread_url: str,
    use_openai: bool = True,
    openai_model: str | None = None,
    batch_size: int = 10,
    require_tmdb: bool = True,
) -> Dict[str, Any]:
    comments = fetch_comments(thread_url)
    mentions: List[Dict[str, Any]] = []

    if use_openai:
        model = openai_model or os.getenv("MODEL", "gpt-4.1-mini")
        call_number = 0

        for batch in batched(comments, batch_size):
            call_number += 1
            comment_ids = [c["id"] for c in batch]

            logging.info(
                f"OpenAI call #{call_number} starting at {datetime.now(timezone.utc).isoformat()} "
                f"with {len(batch)} comments: {comment_ids}"
            )

            try:
                out = call_openai(batch, model=model)
            except Exception as e:
                logging.error(f"OpenAI call #{call_number} raised {type(e).__name__}: {e}")
                response = getattr(e, "response", None)
                headers = getattr(response, "headers", None)
                logging.error(f"Headers: {headers}")
                raise

            # FIX: was datetime.datetime.utcnow() which raised AttributeError
            # because datetime was imported as `from datetime import datetime`
            logging.info(
                f"OpenAI call #{call_number} completed at {datetime.now(timezone.utc).isoformat()}"
            )

            by_id = {c["id"]: c for c in batch}
            for r in out.get("results", []):
                comment_id = r.get("comment_id")
                comment = by_id.get(comment_id)
                if not comment:
                    continue
                for ent in r.get("entities", []):
                    name = (ent.get("name") or "").strip()
                    etype = ent.get("entity_type")
                    sentiment = ent.get("sentiment")
                    if not name or not etype or not sentiment:
                        continue

                    # Resolve movies against TMDB for a clean canonical title+year
                    canonical = None
                    if etype == "movie" and TMDB_API_KEY and require_tmdb:
                        tmdb_result = tmdb_search_movie(name)  # cached
                        canonical = tmdb_canonical(tmdb_result) if tmdb_result else None

                    mentions.append({
                        "entity_type": etype,
                        "entity_name": name,
                        "canonical_name": canonical,  # None falls back to fuzzy dedup
                        "sentiment": sentiment,
                        "comment_id": comment_id,
                        "author": comment.get("author"),
                        "permalink": comment.get("permalink"),
                        "text": comment.get("body"),
                    })

            # Throttle between batches to stay under rate limits
            time.sleep(2.0)

    else:
        nlp = _get_spacy()
        for c in comments:
            text = c["body"]
            sent = vader_sentiment(text)
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    mentions.append({
                        "entity_type": "person",
                        "entity_name": ent.text.strip(),
                        "canonical_name": None,
                        "sentiment": sent,
                        "comment_id": c["id"],
                        "author": c.get("author"),
                        "permalink": c.get("permalink"),
                        "text": text,
                    })
            movies = [m.strip() for m in QUOTED_TITLE.findall(text)]
            for m in TITLE_SPAN.findall(text):
                m = m.strip()
                if len(m.split()) <= 6 and len(m) >= 2 and not m.isupper():
                    movies.append(m)
            seen = set()
            for m in movies:
                if m.lower() in seen:
                    continue
                seen.add(m.lower())
                if looks_like_title(m):
                    canonical = None
                    if TMDB_API_KEY and require_tmdb:
                        tmdb_result = tmdb_search_movie(m)
                        canonical = tmdb_canonical(tmdb_result) if tmdb_result else None
                    mentions.append({
                        "entity_type": "movie",
                        "entity_name": m,
                        "canonical_name": canonical,
                        "sentiment": sent,
                        "comment_id": c["id"],
                        "author": c.get("author"),
                        "permalink": c.get("permalink"),
                        "text": text,
                    })

    mentions, raw_to_canonical = canonicalize_entities(mentions, similarity_threshold=90)
    aggregates = aggregate(mentions)

    return {
        "thread_url": thread_url,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mentions": mentions,
        "dedupe_map": raw_to_canonical,
        "aggregates": aggregates,
    }

def save_json(data: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)