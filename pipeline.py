import os
import re
import json
import time
import requests
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter

from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception
from rapidfuzz import fuzz

# Local NLP (only used in non-OpenAI mode; keeps option available)
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# ---------- setup ----------
USER_AGENT = "reddit-entity-sentiment/0.3 by yourname"

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
                if body and body not in ("[deleted]","[removed]"):
                    comments.append({
                        "id": d.get("id"),
                        "author": d.get("author"),
                        "permalink": "https://www.reddit.com" + d.get("permalink",""),
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
ONE_WORD_TITLE_ALLOWLIST = {"Her","Up","Gravity","Roma","Titanic","Amélie","Joker","Whiplash","Tár","Skyfall","Mank"}

BAD_LEADS = re.compile(r"^(it'?s|im|i'?m|not|so|because|that|this|they|we|you|he|she)\b", re.I)
TITLE_SPAN = re.compile(r"(?:\b(?:The|A|An|[A-Z][a-z0-9’']+)\b(?:\s+|$)){1,12}")
QUOTED_TITLE = re.compile(r"[“\"']([^“\"']{2,80})[”\"']")

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
    if up_tokens < max(1, len(parts)//2):
        return False
    return True

def tmdb_search_movie(query: str) -> dict | None:
    if not TMDB_API_KEY or not query:
        return None
    try:
        r = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"query": query, "include_adult": "false", "language": TMDB_LANG, "page": 1, "api_key": TMDB_API_KEY},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json().get("results", [])
        if not data:
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
        return best
    except Exception:
        return None

def tmdb_canonical(item: dict) -> str:
    if not item:
        return ""
    title = item.get("title") or item.get("original_title") or ""
    year = (item.get("release_date") or "")[:4]
    return f"{title} ({year})" if year else title

# ---------- Sentiment (rules) ----------

NEGATIVE_WORDS = [
    "trash","garbage","awful","terrible","horrible","bad","worst","hate","hated",
    "sucks","sucked","crap","pathetic","boring","dumb","stupid"
]
POSITIVE_WORDS = [
    "amazing","awesome","great","fantastic","excellent","masterpiece","brilliant",
    "love","loved","good"
]
POSITIVE_NEGATED_IDIOMS = [
    r"\bnot\s+bad\b",
    r"\bnot\s+too\s+bad\b",
    r"\bnot\s+terrible\b",
    r"\bnot\s+the\s+worst\b",
]
NEGATORS = {"not","no","never","hardly","barely","scarcely","is","are","was","were","do","does","did","can","could","should","will"}

_WORD_RE = re.compile(r"[A-Za-z']+")
_POS_RE = [re.compile(rf"\b{re.escape(w)}\b", re.I) for w in POSITIVE_WORDS]
_NEG_RE = [re.compile(rf"\b{re.escape(w)}\b", re.I) for w in NEGATIVE_WORDS]
_POS_NEGATED_IDIOMS_RE = [re.compile(p, re.I) for p in POSITIVE_NEGATED_IDIOMS]

def _normalize_neg(text: str) -> str:
    t = text.lower().replace("’","'")
    t = (t.replace("isn't","is not").replace("wasn't","was not")
           .replace("aren't","are not").replace("weren't","were not")
           .replace("don't","do not").replace("doesn't","does not")
           .replace("didn't","did not").replace("can't","can not")
           .replace("couldn't","could not").replace("shouldn't","should not")
           .replace("won't","will not"))
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
                    idx = i; break
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
                    idx = i; break
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

# ---------- OpenAI (Option B) ----------

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
  "properties": {
    "results": {
      "type": "array",
      "items": {
        "type":"object",
        "properties": {
          "comment_id": {"type":"string"},
          "entities": {
            "type":"array",
            "items": {
              "type":"object",
              "properties": {
                "entity_type": {"type":"string","enum":["movie","person"]},
                "name": {"type":"string"},
                "sentiment": {"type":"string","enum":["positive","negative","mixed"]}
              },
              "required":["entity_type","name","sentiment"]
            }
          }
        },
        "required":["comment_id","entities"]
      }
    }
  },
  "required":["results"]
}

def _is_rate_limited(e):
    typ = type(e).__name__
    status = getattr(e, "status_code", None)
    return typ in ("RateLimitError","OpenAIError","APIStatusError") and (status == 429 or status is None)

@retry(retry=retry_if_exception(_is_rate_limited), wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(6))
def call_openai(batch: List[Dict[str,Any]], model: str):
    from openai import OpenAI
    client = OpenAI()
    def _trim(txt, limit=900):
        t = txt or ""
        return t if len(t) <= limit else t[:limit] + " …"
    user_content = [{"type":"text","text":json.dumps({"comment_id":c["id"], "text":_trim(c["body"])})} for c in batch]
    resp = client.responses.create(
        model=model,
        temperature=0,
        response_format={"type":"json_schema","json_schema":{"name":"entity_sentiment","schema":JSON_SCHEMA,"strict":True}},
        input=[
            {"role":"system","content":[{"type":"text","text":SYSTEM_PROMPT}]},
            {"role":"user","content":user_content}
        ],
    )
    return resp.output_json

def batched(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]

# ---------- Deduplication & Aggregations ----------

def canonicalize_entities(mentions: List[Dict[str,Any]], similarity_threshold: int = 90) -> Tuple[List[Dict[str,Any]], Dict[str,str]]:
    """Map raw names to canonical names (per type). Prefers prefilled canonical_name (e.g., from TMDB)."""
    by_type_values = defaultdict(list)
    raw_to_canonical: Dict[str,str] = {}
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
                    cl.append(name); placed = True; break
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

def aggregate(mentions: List[Dict[str,Any]]) -> Dict[str,Any]:
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
        return {"counts": dict(c), "total": total, "majority": majority, "lead_share": round(lead_share,3)}

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
    require_tmdb: bool = True
) -> Dict[str,Any]:
    comments = fetch_comments(thread_url)
    mentions: List[Dict[str,Any]] = []

    if use_openai:
        model = openai_model or os.getenv("MODEL","gpt-4.1-mini")
        for batch in batched(comments, batch_size):
            out = call_openai(batch, model=model)
            by_id = {c["id"]: c for c in batch}
            for r in out.get("results", []):
                cid = r["comment_id"]
                source = by_id.get(cid, {})
                text = source.get("body") or ""
                rule_sent = rule_based_sentiment(text)
                for e in r.get("entities", []):
                    etype = e["entity_type"]
                    raw_name = (e["name"] or "").strip()

                    tmdb_item = None
                    if etype == "movie":
                        raw_name = expand_movie_name_from_text(raw_name, text)
                        if not looks_like_title(raw_name):
                            continue
                        if TMDB_API_KEY and require_tmdb:
                            tmdb_item = tmdb_search_movie(raw_name)
                            if not tmdb_item or fuzz.token_set_ratio(raw_name, (tmdb_item.get("title") or "")) < 85:
                                continue

                    sent = e["sentiment"]
                    if rule_sent and sent != rule_sent:
                        sent = rule_sent

                    mentions.append({
                        "entity_type": etype,
                        "entity_name": raw_name,
                        "sentiment": sent,
                        "comment_id": cid,
                        "author": source.get("author"),
                        "permalink": source.get("permalink"),
                        "text": text,
                        "canonical_name": tmdb_canonical(tmdb_item) if tmdb_item else None,
                    })
            time.sleep(0.5)
    else:
        nlp = _get_spacy()
        for c in comments:
            text = c["body"]
            sent = vader_sentiment(text)
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    mentions.append({
                        "entity_type":"person",
                        "entity_name": ent.text.strip(),
                        "sentiment": sent,
                        "comment_id": c["id"],
                        "author": c.get("author"),
                        "permalink": c.get("permalink"),
                        "text": text
                    })
            movies = [m.strip() for m in QUOTED_TITLE.findall(text)]
            for m in TITLE_SPAN.findall(text):
                m = m.strip()
                if len(m.split()) <= 6 and len(m) >= 2 and not m.isupper():
                    movies.append(m)
            seen = set()
            for m in movies:
                if m.lower() in seen: continue
                seen.add(m.lower())
                if looks_like_title(m):
                    mentions.append({
                        "entity_type":"movie",
                        "entity_name": m,
                        "sentiment": sent,
                        "comment_id": c["id"],
                        "author": c.get("author"),
                        "permalink": c.get("permalink"),
                        "text": text
                    })

    mentions, raw_to_canonical = canonicalize_entities(mentions, similarity_threshold=90)
    aggregates = aggregate(mentions)

    return {
        "thread_url": thread_url,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mentions": mentions,
        "dedupe_map": raw_to_canonical,
        "aggregates": aggregates
    }

def save_json(data: Dict[str,Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
