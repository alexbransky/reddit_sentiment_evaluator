import os, json
import streamlit as st
import altair as alt
from dotenv import load_dotenv

from pipeline import run_pipeline

load_dotenv()

st.set_page_config(page_title="Reddit Entity & Sentiment Explorer", layout="wide")
st.title("Reddit Entity & Sentiment Explorer")

with st.sidebar:
    st.header("Run Pipeline")
    reddit_url = st.text_input("Reddit thread URL", placeholder="https://www.reddit.com/r/.../comments/<id>/<slug>/", value="")
    use_ai = st.toggle("Use OpenAI (more accurate, costs money)", value=True)
    model = st.text_input("OpenAI model", value=os.getenv("MODEL","gpt-4.1-mini"))
    batch_size = st.number_input("AI batch size (comments per call)", min_value=1, max_value=25, value=10)
    require_tmdb = st.toggle("Validate movie titles via TMDB (recommended)", value=True, help="Requires TMDB_API_KEY in your .env")
    run_btn = st.button("Fetch & Analyze")
    uploaded_json = st.file_uploader("Or upload a results JSON", type=["json"])

results = None

if run_btn and reddit_url:
    with st.spinner("Running pipeline..."):
        try:
            results = run_pipeline(
                reddit_url,
                use_openai=use_ai,
                openai_model=model,
                batch_size=int(batch_size),
                require_tmdb=require_tmdb
            )
        except Exception as e:
            st.error(f"Error: {e}")
            results = None
    if results:
        st.success("Analysis complete.")
        st.download_button("Download JSON", data=json.dumps(results, ensure_ascii=False, indent=2),
                           file_name="reddit_entity_sentiment.json", mime="application/json")
elif uploaded_json is not None:
    results = json.load(uploaded_json)
    st.info("Loaded uploaded JSON.")

if not results:
    st.stop()

st.subheader("Overview")
st.write(f"**Thread:** {results.get('thread_url')}")
st.write(f"**Generated at (UTC):** {results.get('generated_at')}")

# Filters
col1, col2, col3, col4 = st.columns(4)
with col1:
    etype = st.selectbox("Entity type", ["all","movie","person"], index=0)
with col2:
    sentiment = st.selectbox("Sentiment", ["all","positive","negative","mixed","unclear"], index=0)
with col3:
    author_query = st.text_input("Author contains", value="")
with col4:
    name_query = st.text_input("Entity name contains", value="")

mentions = results.get("mentions", [])

def apply_filters(rows):
    out = []
    for r in rows:
        if etype != "all" and r["entity_type"] != etype:
            continue
        if sentiment != "all":
            if sentiment != r["sentiment"]:
                continue
        if author_query and author_query.lower() not in (r.get("author") or "unknown").lower():
            continue
        if name_query and name_query.lower() not in r.get("canonical_name","").lower():
            continue
        out.append(r)
    return out

filtered_mentions = apply_filters(mentions)

st.subheader("Global Aggregates")
global_aggs = results["aggregates"]["global_entity_sentiment"]
if etype != "all":
    global_aggs = [g for g in global_aggs if g["entity_type"] == etype]
if name_query:
    global_aggs = [g for g in global_aggs if name_query.lower() in g["canonical_name"].lower()]

st.dataframe(global_aggs, use_container_width=True, hide_index=True)

# Chart of top entities by total
if global_aggs:
    max_n = max(1, len(global_aggs))
    topN = st.slider("Top N (by mentions)", 1, max_n, min(15, max_n))
    chart_data = [{"entity": f'{g["canonical_name"]} ({g["entity_type"][0]})',
                   "sentiment": s, "count": g["counts"].get(s,0)}
                  for g in global_aggs[:topN] for s in ["positive","mixed","negative"]]
    chart = alt.Chart(alt.Data(values=chart_data)).mark_bar().encode(
        x=alt.X("entity:N", sort="-y", title="Entity"),
        y=alt.Y("count:Q", title="Mentions"),
        color="sentiment:N",
        column=alt.Column("sentiment:N", header=alt.Header(title="Sentiment"))
    ).properties(height=260)
    st.altair_chart(chart, use_container_width=True)

st.subheader("By Author")
author_aggs = results["aggregates"]["author_entity_sentiment"]
if author_query:
    author_aggs = [a for a in author_aggs if author_query.lower() in a["author"].lower()]
st.dataframe(author_aggs, use_container_width=True, hide_index=True)

st.subheader("Mentions (detailed)")
st.caption("Each row is an entity mention tied to a single comment.")
st.dataframe(filtered_mentions, use_container_width=True, hide_index=True)
