
import os
import re
import json
import time
import requests
from typing import List, Dict, Any, Tuple
from urllib.parse import urlparse
from collections import defaultdict, Counter

# Optional dependencies
from tenacity import retry, wait_exponential, stop_after_attempt

# Local NLP defaults
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from rapidfuzz import fuzz

# Ensure NLTK VADER is ready
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

# Lazy-load spaCy model to allow Streamlit to start fast
_SPACY = None
def get_spacy():
    global _SPACY
    if _SPACY is None:
        try:
            _SPACY = spacy.load("en_core_web_sm")
        except OSError:
            # Try to download if missing (download is provided in spacy.cli.download)
            try:
                from spacy.cli.download import download
            except Exception:
                # Fallback for older layouts where download may be exposed differently
                from spacy.cli.download import download
            download("en_core_web_sm")
            _SPACY = spacy.load("en_core_web_sm")
    return _SPACY

SIA = SentimentIntensityAnalyzer()

USER_AGENT = "reddit-entity-sentiment/0.2 by alexbransky"

def reddit_json_url(thread_url: str) -> str:
    u = thread_url if thread_url.endswith("/") else thread_url + "/"
    return u + ".json?limit=500&depth=10&raw_json=1"

def fetch_comments(thread_url: str) -> List[Dict[str, Any]]:
    url = reddit_json_url(thread_url)
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
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

# --- Extraction (local) -------------------------------------------------------

# Try to expand a short/partial movie name using the longest Title-Case span in the comment that contains it.
TITLE_SPAN = re.compile(r"(?:\b(?:The|A|An|[A-Z][a-z0-9’']+)\b(?:\s+|$)){1,12}")

def expand_movie_name_from_text(name: str, text: str) -> str:
    n = name.strip()
    if not n or len(n.split()) >= 3:
        return n
    cand = [m.group(0).strip() for m in TITLE_SPAN.finditer(text) if n.lower() in m.group(0).lower()]
    if cand:
        return max(cand, key=len)
    return n


QUOTED_TITLE = re.compile(r"[“\"']([^“\"']{2,80})[”\"']")
TITLE_CASE_RUN = re.compile(r"(?:\b[A-Z][a-z0-9’']+(?:\s+|$)){1,6}")

def extract_people_and_movies_local(text: str) -> Tuple[List[str], List[str]]:
    nlp = get_spacy()
    doc = nlp(text)
    people = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]

    movies = []
    # quoted fragments
    movies += [m.strip() for m in QUOTED_TITLE.findall(text)]
    # title case candidates
    for m in TITLE_CASE_RUN.findall(text):
        m = m.strip()
        if len(m.split()) <= 6 and len(m) >= 2 and not m.isupper():
            if m.lower() not in {"i","we","the","and","but","or"}:
                movies.append(m)

    def uniq(seq):
        seen = set()
        out = []
        for s in seq:
            k = s.lower()
            if k not in seen:
                seen.add(k)
                out.append(s)
        return out

    return uniq(people), uniq(movies)

def sentiment_label(text: str) -> str:
    s = SIA.polarity_scores(text)
    if s["compound"] >= 0.2:
        return "positive"
    elif s["compound"] <= -0.2:
        return "negative"
    else:
        return "mixed"

# --- Optional: OpenAI mode ----------------------------------------------------

def get_openai_client():
    from openai import OpenAI
    return OpenAI()

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

SYSTEM_PROMPT = """Extract entities and per-entity sentiment from each Reddit comment.

Rules:
- Only MOVIES and PEOPLE.
- Use the movie's full official English title (keep leading articles, e.g., "The Shape of Water"; do not truncate to common nouns like "Water").
- For PEOPLE, return the full name if present; do not append role words (director, actor).
- Sentiment is from the author's perspective TOWARD EACH ENTITY: "positive", "negative", or "mixed".
- If the language is clearly evaluative and strongly negative (e.g., "is trash", "sucks", "garbage", "awful", "hate"), classify as "negative".
- If evidence is conflicting or ambiguous, use "mixed" (do NOT default to mixed when the text is clearly evaluative).
- Return strict JSON for the provided schema.
"""

@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(4))
def call_openai(batch: List[Dict[str, Any]], model: str):
    client = get_openai_client()
    user_content = []
    for c in batch:
        user_content.append({"type":"text","text":json.dumps({"comment_id":c["id"], "text":c["body"]})})
    # Call the Responses API without the typed response_format (avoids signature/type mismatches)
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{
            "role": "system",
            "content": SYSTEM_PROMPT
        }, {
            "role": "user",
            "content": json.dumps(user_content)
        }],
    )
    # Extract content from the response
    text = resp.choices[0].message.content
    try:
        if text is None:
            return {}
        return json.loads(text)
    except Exception:
        # If parsing fails, return raw response for debugging purposes
        return resp

def batched(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]

# --- Deduplication ------------------------------------------------------------

def canonicalize_entities(mentions: List[Dict[str, Any]], similarity_threshold: int = 90) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Given raw mentions [{"entity_type","entity_name",...}], produce a mapping
    raw_name -> canonical_name (per entity_type), using fuzzy clustering.
    """
    by_type = defaultdict(list)
    for m in mentions:
        by_type[m["entity_type"]].append(m["entity_name"].strip())

    raw_to_canonical: Dict[str, str] = {}
    for etype, names in by_type.items():
        uniq_names = []
        seen_lower = set()
        for n in names:
            nl = n.lower()
            if nl not in seen_lower:
                seen_lower.add(nl)
                uniq_names.append(n)

        clusters: List[List[str]] = []
        for name in sorted(uniq_names, key=lambda s: (len(s), s.lower()), reverse=True):
            placed = False
            for cl in clusters:
                # compare against the first element of the cluster (its current canonical)
                anchor = cl[0]
                if fuzz.token_set_ratio(name, anchor) >= similarity_threshold:
                    cl.append(name)
                    placed = True
                    break
            if not placed:
                clusters.append([name])

        # pick canonical as the longest name in cluster (could swap to most frequent later)
        for cl in clusters:
            canonical = max(cl, key=len)
            for n in cl:
                raw_to_canonical[f"{etype}|{n}"] = canonical

    # apply mapping
    for m in mentions:
        key = f'{m["entity_type"]}|{m["entity_name"]}'
        m["canonical_name"] = raw_to_canonical.get(key, m["entity_name"])

    return mentions, raw_to_canonical

# --- Aggregations -------------------------------------------------------------

def aggregate(mentions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    mentions: list of dicts with keys
      - entity_type, entity_name, canonical_name, sentiment, comment_id, author, permalink, text
    Returns a dict with:
      - global_entity_sentiment
      - author_entity_sentiment
    """
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

        # If there is a tie (e.g., 2/2/2), mark as 'unclear'
        majority = "unclear" if len(top) > 1 else top[0]

        return {"counts": dict(c), "total": total, "majority": majority}

    global_summary = [
        {
            "entity_type": etype,
            "canonical_name": name,
            **summarize_counter(cnt)
        }
        for (etype, name), cnt in sorted(global_counts.items(), key=lambda kv: (-sum(kv[1].values()), kv[0][1].lower()))
    ]

    author_summary = []
    for author, entities in author_counts.items():
        ent_list = []
        for (etype, name), cnt in entities.items():
            ent_list.append({
                "entity_type": etype,
                "canonical_name": name,
                **summarize_counter(cnt)
            })
        author_summary.append({"author": author, "entities": sorted(ent_list, key=lambda e: -e["total"])})

    return {
        "global_entity_sentiment": global_summary,
        "author_entity_sentiment": sorted(author_summary, key=lambda a: a["author"].lower())
    }

# --- Orchestrator -------------------------------------------------------------

from typing import Optional

import re

# Cue lists (you can expand these over time)
NEGATIVE_WORDS = [
    "trash","garbage","awful","terrible","horrible","bad","worst","hate","hated",
    "sucks","sucked","crap","pathetic","boring","dumb","stupid"
]
POSITIVE_WORDS = [
    "amazing","awesome","great","fantastic","excellent","masterpiece","brilliant",
    "love","loved","good"
]

# Idioms that are actually positive though they contain a negation
POSITIVE_NEGATED_IDIOMS = [
    r"\bnot\s+bad\b",
    r"\bnot\s+too\s+bad\b",
    r"\bnot\s+terrible\b",
    r"\bnot\s+the\s+worst\b",
]

NEGATORS = {"not","no","never","hardly","barely","scarcely","isnt","isn't","wasnt","wasn't",
            "arent","aren't","werent","weren't","dont","don't","doesnt","doesn't","didnt","didn't",
            "cant","can't","couldnt","couldn't","shouldnt","shouldn't","wont","won't"}

# Precompile
_POS_RE = [re.compile(rf"\b{re.escape(w)}\b", re.I) for w in POSITIVE_WORDS]
_NEG_RE = [re.compile(rf"\b{re.escape(w)}\b", re.I) for w in NEGATIVE_WORDS]
_POS_NEGATED_IDIOMS_RE = [re.compile(p, re.I) for p in POSITIVE_NEGATED_IDIOMS]

_WORD_RE = re.compile(r"[A-Za-z']+")

def _normalize(text: str) -> str:
    # unify contractions to a form we can scan
    t = text.lower()
    t = t.replace("’", "'")
    # normalize common contractions to separate 'not'
    t = (t.replace("isn't","is not").replace("wasn't","was not")
           .replace("aren't","are not").replace("weren't","were not")
           .replace("don't","do not").replace("doesn't","does not")
           .replace("didn't","did not").replace("can't","can not")
           .replace("couldn't","could not").replace("shouldn't","should not")
           .replace("won't","will not"))
    return t

def _tokens_with_idx(text: str):
    return [(m.group(0), m.start()) for m in _WORD_RE.finditer(text)]

def _has_negator_before(tokens, hit_index, window=3) -> bool:
    """Return True if a negator token occurs within `window` tokens before hit_index."""
    start = max(0, hit_index - window)
    for i in range(start, hit_index):
        if tokens[i][0] in NEGATORS:
            return True
    return False

def rule_based_sentiment(text: str) -> str | None:
    """
    Negation-aware override:
      - If strong negative word present (and not negated) -> 'negative'
      - If strong positive word present (and not negated) -> 'positive'
      - If positive-negated idiom ('not bad', etc.) -> 'positive'
      - If positive word is negated ('not great') -> 'negative'
      - If negative word is negated ('not terrible') -> 'positive'
      - Otherwise None (let the model decide)
    """
    if not text:
        return None

    t = _normalize(text)
    toks = [w for (w, _) in _tokens_with_idx(t)]
    # build map from token index to token for window checks
    # simple scan using regex matches and token index math

    # quick pass for idioms like 'not bad'
    for r in _POS_NEGATED_IDIOMS_RE:
        if r.search(t):
            return "positive"

    # find all token positions so we can detect negation windows
    # Map word-start index -> token position
    token_spans = list(_WORD_RE.finditer(t))
    def token_pos_for_span(span_start):
        # binary search not needed; linear is fine at comment size
        for i, m in enumerate(token_spans):
            if m.start() == span_start:
                return i
        return None

    # Check negative words (un-negated)
    for r in _NEG_RE:
        for m in r.finditer(t):
            pos = token_pos_for_span(m.start())
            if pos is None:
                continue
            if not _has_negator_before([(m.group(0), m.start()) for m in token_spans], pos, window=3):
                return "negative"
            else:
                # 'not terrible' → positive
                return "positive"

    # Check positive words (un-negated)
    for r in _POS_RE:
        for m in r.finditer(t):
            pos = token_pos_for_span(m.start())
            if pos is None:
                continue
            if not _has_negator_before([(m.group(0), m.start()) for m in token_spans], pos, window=3):
                return "positive"
            else:
                # 'not great' → negative
                return "negative"

    return None


def run_pipeline(
    thread_url: str,
    use_openai: bool = False,
    openai_model: Optional[str] = None,
    batch_size: int = 10
) -> Dict[str, Any]:
    """
    Returns the final JSON structure with:
      - thread_url
      - generated_at
      - mentions (detailed per-entity mentions)
      - dedupe_map
      - aggregates (global + by author)
    """
    comments = fetch_comments(thread_url)

    mentions = []  # per-entity mention rows
    if use_openai:
        model = openai_model or os.getenv("MODEL", "gpt-4.1-mini")
        for batch in batched(comments, batch_size):
            # safety: do not slam the API
            time.sleep(0.4)
            out = call_openai(batch, model=model)
            # stitch
            by_id = {c["id"]: c for c in batch}
            response_data = {"results": []}
            try:
                if isinstance(out, str):
                    response_data = json.loads(out)
                elif isinstance(out, dict):
                    if "choices" in out and isinstance(out["choices"], list) and len(out["choices"]) > 0:
                        response_data = json.loads(out["choices"][0]["message"]["content"])
                elif hasattr(out, "choices") and out.choices:
                    # support different response object shapes
                    content = None
                    try:
                        content = out.choices[0].message.content
                    except Exception:
                        try:
                            content = out.choices[0].message.content
                        except Exception:
                            content = None
                    if content:
                        response_data = json.loads(content)
            except Exception:
                # keep default response_data on any parsing error
                response_data = {"results": []}
            for r in response_data.get("results", []):
                cid = r["comment_id"]
                source = by_id.get(cid, {})
                text = source.get("body") or ""

                # strong-rule sentiment from the raw comment
                rule_sent = rule_based_sentiment(text)

                for e in r.get("entities", []):
                    raw_name = e["name"].strip()
                    if e["entity_type"] == "movie":
                        raw_name = expand_movie_name_from_text(raw_name, text)

                    sent = e["sentiment"]
                    if rule_sent and sent != rule_sent:
                        sent = rule_sent  # e.g., "is trash" => negative

                    mentions.append({
                        "entity_type": e["entity_type"],
                        "entity_name": raw_name,
                        "sentiment": sent,
                        "comment_id": cid,
                        "author": source.get("author"),
                        "permalink": source.get("permalink"),
                        "text": text
                    })

    else:
        # local path
        for c in comments:
            sent = sentiment_label(c["body"])
            people, movies = extract_people_and_movies_local(c["body"])
            for p in people:
                mentions.append({
                    "entity_type": "person",
                    "entity_name": p,
                    "sentiment": sent,
                    "comment_id": c["id"],
                    "author": c.get("author"),
                    "permalink": c.get("permalink"),
                    "text": c.get("body")
                })
            for m in movies:
                mentions.append({
                    "entity_type": "movie",
                    "entity_name": m,
                    "sentiment": sent,
                    "comment_id": c["id"],
                    "author": c.get("author"),
                    "permalink": c.get("permalink"),
                    "text": c.get("body")
                })

    # Deduplicate
    mentions, raw_to_canonical = canonicalize_entities(mentions, similarity_threshold=90)

    # Aggregations
    aggregates = aggregate(mentions)

    out = {
        "thread_url": thread_url,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mentions": mentions,
        "dedupe_map": raw_to_canonical,
        "aggregates": aggregates
    }
    return out

def save_json(data: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
