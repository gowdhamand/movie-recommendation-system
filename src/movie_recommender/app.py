import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchText,
    MatchValue,
    Range,
    RecommendInput,
    RecommendQuery,
)
from sentence_transformers import SentenceTransformer

# shared config
COLLECTION_NAME = "movies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
st.set_page_config(
    page_title="Movie Recommender",
    page_icon=":movie_camera:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Cached Resources

@st.cache_resource
def load_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def get_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

@st.cache_resource
def get_llm_client(api_key: str):
    import anthropic
    return anthropic.Anthropic(api_key=api_key)

model = load_model()
client = get_client()

# Helper Functions

# Check collection exists
def ensure_collection():
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=model.get_sentence_embedding_dimension(),
                distance=Distance.COSINE,
            ),
        )

def format_collection_status(status) -> str:
    if hasattr(status, "value"):
        return str(status.value)
    text = str(status)
    if "." in text:
        return text.rsplit(".", 1)[-1]
    return text


def is_collection_healthy(status) -> bool:
    return format_collection_status(status).lower() in ("green", "ready")


def get_collection_stats():
    try:
        info = client.get_collection(COLLECTION_NAME)
        return {
            "points": info.points_count,
            "status": info.status,
        }
    except Exception as e:
        return {
            "points": 0,
            "status": "error",
            "error": str(e),
        }


# Convert a raw qdrant point to display dict
def point_to_dict(point: PointStruct) -> dict:
    p = point.payload or {}
    return {
        "movie_id": int(p.get("movie_id", 0)),
        "title": str(p.get("title", "")),
        "release_year": int(p.get("release_year", 0)),
        "genres": ", ".join(p["genres"]) if isinstance(p.get("genres"), list) else str(p.get("genres", "")),
        "rating": float(p.get("rating", 0.0)),
        "overview": str(p.get("overview", "")),
        "chunk_type": str(p.get("chunk_type", "")),
        "score": round(point.score, 4) if hasattr(point, "score") else None,
    }

# dedub a list of points
def dedup(points: list[PointStruct]) -> list[PointStruct]:
    seen: dict[int, PointStruct] = {}
    for p in points:
        mid = p.payload.get("movie_id")
        sc = getattr(p, "score", None) or 0.0
        if mid is None:
            continue
        if mid not in seen or sc > getattr(seen[mid], "score", 0.0):
            seen[mid] = p
    return sorted(seen.values(), key=lambda p: getattr(p, "score", 0) or 0, reverse=True)

# Render one card result card
def render_card(movie: dict, rank: int):
    score = f" score={movie['score']:.4f}" if movie['score'] is not None else "-"
    with st.container(border=True):
        col_rank, col_info, col_score = st.columns([0.5, 8, 1.5])
        with col_rank:
            st.markdown(f"### {rank}")
        with col_info:
            st.markdown(f"**{movie['title']}** ({movie['release_year']})")
            st.caption(f"{movie['genres']} | {movie['rating']:.1f} | {score} | {movie['chunk_type']}")
            st.write(movie['overview'][:200] + "..." if len(movie['overview']) > 200 else movie['overview'])
        with col_score:
            st.metric("Score", score)

with st.sidebar:
    st.title("Movie Recommender")
    st.caption("Powered by Qdrant + Sentence Transformers")
    st.divider()

    ## Live stats - refresh every rerun
    stats = get_collection_stats()
    status_label = format_collection_status(stats["status"])
    st.subheader("Collection Stats")
    col1, col2 = st.columns(2)
    col1.metric("Points", stats["points"])
    col2.metric("Status", status_label)
    status_color = "green" if is_collection_healthy(stats["status"]) else "red"
    st.markdown(
        f"Connection: <span style='color: {status_color}; font-weight: 600;'>{status_label}</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.subheader("Settings")
    chunk_strategy = st.selectbox(
        "Check Strategy (for Add)",
        options=["full", "field", "sliding"],
        help= (
            "full: 1 chunk per movie, field: 1 chunk per field, sliding: 10 chunks per movie"
            "field: chunk by title, genres, overview"
            "sliding: chunk by sliding window of 500 words with 100 word overlap"
        )
    )
    default_limit = st.slider("Default result limit", min_value=1, max_value=20, value=5, step=1)
    st.divider()

    st.subheader("RAG Settings")
    anthropic_key = st.text_input(
        "Anthropic API key",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Or set ANTHROPIC_API_KEY in the project root .env file.",
    )
    rag_top_k = st.slider("RAG top-k chunks", min_value=1, max_value=10, value=5, step=1)
    rag_model = st.text_input("Claude model", value="claude-sonnet-4-6")
    st.divider()
    st.caption("Makesure Qdrant is running \m 'docker run -d -p 6333:6333 qdrant/qdrant'")

# Main TABS
tab_add, tab_search, tab_rag = st.tabs(["Add", "Search", "RAG Chat"])

with tab_add:
    st.header("Add Movies to the Vector Database")
    st.caption(
        "Use the form below to add a single movie manually, or upload a CSV "
        "to bulk-import from Kaggle."
    )
 
    # ── Section A: Add a single movie ────────────────────────────────────────
    st.subheader("A · Add a single movie")
 
    # Organise the form into two columns so it doesn't stretch too wide
    col_left, col_right = st.columns(2)
 
    with col_left:
        title    = st.text_input("Title *",       placeholder="Inception")
        genres   = st.text_input("Genres *",      placeholder="Action, Sci-Fi, Thriller")
        overview = st.text_area( "Overview *",    placeholder="A thief who steals corporate secrets…", height=120)
 
    with col_right:
        year       = st.number_input("Release year",   min_value=1888, max_value=2100, value=2024)
        rating     = st.slider(     "Rating (0–10)",   0.0, 10.0, 7.0, 0.1)
        popularity = st.number_input("Popularity",     min_value=0.0, value=50.0)
        vote_count = st.number_input("Vote count",     min_value=0,   value=1000)
 
    add_clicked = st.button("⬆️ Add to Qdrant", type="primary", key="add_single")
 
    if add_clicked:
        # Validate required fields
        if not title.strip() or not genres.strip() or not overview.strip():
            st.error("Title, Genres and Overview are required.")
        else:
            ensure_collection()
 
            # Build a simple namespace object so chunk functions can use
            # dot-notation (row.title) just like they do with DataFrame rows
            class Row:
                pass
            row = Row()
            row.title        = title.strip()
            row.genres_clean = genres.strip()
            row.overview     = overview.strip()
            row.release_year = int(year)
            row.vote_average = float(rating)
            row.popularity   = float(popularity)
            row.vote_count   = int(vote_count)
            # Use a timestamp-based int as the movie_id for manual entries
            import time
            row.id = int(time.time())
 
            # Import chunking functions from ingest.py
            from ingest import CHUNKERS, chunk_id_to_point_id
            chunker = CHUNKERS[chunk_strategy]
            chunks  = chunker(row)
            texts   = [c["text"] for c in chunks]
            vectors = model.encode(texts, show_progress_bar=False)
 
            points = [
                PointStruct(
                    id=chunk_id_to_point_id(chunk["chunk_id"]),
                    vector=vec.tolist(),
                    payload={
                        "movie_id":   row.id,
                        "title":      row.title,
                        "year":       row.release_year,
                        "genres":     row.genres_clean,
                        "rating":     row.vote_average,
                        "vote_count": row.vote_count,
                        "popularity": row.popularity,
                        "overview":   row.overview,
                        "chunk_text": chunk["text"],
                        "chunk_type": chunk["chunk_type"],
                        "chunk_uuid": chunk["chunk_id"],
                    },
                )
                for chunk, vec in zip(chunks, vectors)
            ]
 
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            st.success(
                f"✅ **{row.title}** added — {len(points)} chunk(s) upserted "
                f"using **{chunk_strategy}** strategy."
            )
            # Show what was actually stored so the user can learn
            with st.expander("📦 See what was stored"):
                for c in chunks:
                    st.code(c["text"], language=None)
 
    st.divider()
 
    # ── Section B: Bulk upload CSV ────────────────────────────────────────────
    st.subheader("B · Bulk upload from CSV")
    st.caption(
        "Upload your Kaggle `tmdb_5000_movies.csv` here. "
        "The file is processed with the chunking strategy chosen in the sidebar."
    )
 
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
 
    if uploaded:
        import pandas as pd, ast
 
        @st.cache_data
        def parse_csv(file_bytes: bytes) -> pd.DataFrame:
            import io
            df = pd.read_csv(io.BytesIO(file_bytes))
 
            def extract_genres(val):
                try:
                    return ", ".join(g["name"] for g in ast.literal_eval(val))
                except Exception:
                    return val if isinstance(val, str) else ""
 
            # Handle both TMDB-style (JSON genres) and plain-string genres columns
            if "genres" in df.columns:
                df["genres_clean"] = df["genres"].apply(extract_genres)
            else:
                df["genres_clean"] = ""
 
            df = df.rename(columns={
                "vote_average": "vote_average",
                "vote_count":   "vote_count",
            })
            df = df.dropna(subset=["overview"])
            df = df[df["overview"].str.strip() != ""]
 
            if "release_date" in df.columns:
                df["release_year"] = (
                    pd.to_datetime(df["release_date"], errors="coerce")
                    .dt.year.fillna(0).astype(int)
                )
            elif "year" in df.columns:
                df["release_year"] = df["year"].fillna(0).astype(int)
            else:
                df["release_year"] = 0
 
            for col in ["popularity", "vote_average", "vote_count"]:
                if col not in df.columns:
                    df[col] = 0.0
 
            return df
 
        df = parse_csv(uploaded.read())
        st.info(f"Parsed **{len(df)} movies** from the CSV.")
        st.dataframe(df[["title", "release_year", "genres_clean", "vote_average"]].head(10))
 
        limit_rows = st.number_input(
            "How many movies to ingest? (0 = all)",
            min_value=0, max_value=len(df), value=min(100, len(df)),
        )
 
        if st.button("🚀 Ingest into Qdrant", type="primary", key="bulk_ingest"):
            ensure_collection()
            subset = df.head(limit_rows) if limit_rows > 0 else df
 
            from ingest import CHUNKERS, chunk_id_to_point_id
            chunker = CHUNKERS[chunk_strategy]
 
            progress_bar = st.progress(0, text="Ingesting…")
            all_points   = []
 
            for i, (_, row) in enumerate(subset.iterrows()):
                chunks  = chunker(row)
                texts   = [c["text"] for c in chunks]
                vectors = model.encode(texts, show_progress_bar=False)
 
                for chunk, vec in zip(chunks, vectors):
                    all_points.append(PointStruct(
                        id=chunk_id_to_point_id(chunk["chunk_id"]),
                        vector=vec.tolist(),
                        payload={
                            "movie_id":   int(row.get("id", i)),
                            "title":      str(row.get("title", "")),
                            "year":       int(row.get("release_year", 0)),
                            "genres":     str(row.get("genres_clean", "")),
                            "rating":     float(row.get("vote_average", 0)),
                            "vote_count": int(row.get("vote_count", 0)),
                            "popularity": float(row.get("popularity", 0)),
                            "overview":   str(row.get("overview", "")),
                            "chunk_text": chunk["text"],
                            "chunk_type": chunk["chunk_type"],
                            "chunk_uuid": chunk["chunk_id"],
                        },
                    ))
 
                progress_bar.progress((i + 1) / len(subset), text=f"Embedding: {row.get('title','')}")
 
            # Upsert in batches of 100
            BATCH = 100
            for i in range(0, len(all_points), BATCH):
                client.upsert(collection_name=COLLECTION_NAME, points=all_points[i:i+BATCH])
 
            progress_bar.progress(1.0, text="Done!")
            st.success(f"✅ Ingested **{len(subset)} movies** → **{len(all_points)} vectors**.")
            st.rerun()   # refresh sidebar stats
 
 
# ╔═════════════════════════════════════════════════════════════════════════════
# ║  TAB 2 — SEARCH
# ╚═════════════════════════════════════════════════════════════════════════════
with tab_search:
    st.header("Search the Movie Collection")
 
    # ── Search mode picker ────────────────────────────────────────────────────
    # radio() gives a clear one-of-four choice; each option shows a different
    # set of controls below, so the UI stays uncluttered.
    mode = st.radio(
        "Search mode",
        options=[
            "🧠 Semantic",
            "🔤 Keyword",
            "🎯 Semantic + Filters",
            "🎬 Similar Movies",
        ],
        horizontal=True,
    )
 
    st.divider()
 
    results_raw = []   # filled by whichever mode block runs
 
    # ── Mode 1: Pure semantic ─────────────────────────────────────────────────
    if mode == "🧠 Semantic":
        st.subheader("🧠 Semantic Search")
        st.caption(
            "Describe what you want in plain English. The model finds movies "
            "with similar *meaning*, even if the exact words don't appear."
        )
        query = st.text_input("Query", placeholder="space exploration and human survival")
        limit = st.slider("Number of results", 1, 20, default_limit, key="sem_limit")
 
        if st.button("Search", type="primary", key="sem_go"):
            if not query.strip():
                st.warning("Please enter a query.")
            else:
                with st.spinner("Searching…"):
                    vec = model.encode(query).tolist()
                    response = client.query_points(
                        collection_name=COLLECTION_NAME,
                        query=vec,
                        limit=limit * 3,
                        with_payload=True,
                        with_vectors=False,
                    )
                results_raw = dedup(response.points)[:limit]
 
    # ── Mode 2: Keyword ───────────────────────────────────────────────────────
    elif mode == "🔤 Keyword":
        st.subheader("🔤 Keyword Search")
        st.caption(
            "Finds movies where the keyword appears literally in the title or plot. "
            "No AI — fast and exact."
        )
        keyword = st.text_input("Keyword", placeholder="Batman")
        limit   = st.slider("Number of results", 1, 50, default_limit, key="kw_limit")
 
        if st.button("Search", type="primary", key="kw_go"):
            if not keyword.strip():
                st.warning("Please enter a keyword.")
            else:
                with st.spinner("Searching…"):
                    raw, _ = client.scroll(
                        collection_name=COLLECTION_NAME,
                        scroll_filter=Filter(should=[
                            FieldCondition(key="title",    match=MatchText(text=keyword)),
                            FieldCondition(key="overview", match=MatchText(text=keyword)),
                        ]),
                        limit=limit * 2,
                        with_payload=True,
                        with_vectors=False,
                    )
                results_raw = dedup(raw)[:limit]
 
    # ── Mode 3: Semantic + Filters ────────────────────────────────────────────
    elif mode == "🎯 Semantic + Filters":
        st.subheader("🎯 Semantic Search with Filters")
        st.caption(
            "Combines vector similarity with hard constraints on genre, "
            "rating, and release year. All filter fields are optional."
        )
 
        query = st.text_input("Query", placeholder="crime and betrayal")
 
        # Filters laid out in columns so they sit on one row
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            genre = st.text_input("Genre (partial match)", placeholder="Drama")
        with fc2:
            min_rating = st.slider("Min rating", 0.0, 10.0, 0.0, 0.1)
        with fc3:
            year_from = st.number_input("Released after year", min_value=0, max_value=2100, value=0)
 
        # Score threshold with a sensible default and explanation
        threshold = st.slider(
            "Score threshold (0 = off)",
            0.0, 1.0, 0.0, 0.05,
            help="Only return results above this cosine similarity. 0.65 is a good default.",
        )
        limit = st.slider("Number of results", 1, 20, default_limit, key="filt_limit")
 
        if st.button("Search", type="primary", key="filt_go"):
            if not query.strip():
                st.warning("Please enter a query.")
            else:
                # Build filter conditions from whatever the user filled in
                conditions = []
                if genre.strip():
                    conditions.append(FieldCondition(key="genres", match=MatchText(text=genre.strip())))
                if min_rating > 0:
                    conditions.append(FieldCondition(key="rating", range=Range(gte=min_rating)))
                if year_from > 0:
                    conditions.append(FieldCondition(key="release_year", range=Range(gte=year_from)))
 
                search_filter = Filter(must=conditions) if conditions else None
 
                with st.spinner("Searching…"):
                    vec = model.encode(query).tolist()
                    response = client.query_points(
                        collection_name=COLLECTION_NAME,
                        query=vec,
                        query_filter=search_filter,
                        limit=limit * 3,
                        score_threshold=threshold if threshold > 0 else None,
                        with_payload=True,
                        with_vectors=False,
                    )
                results_raw = dedup(response.points)[:limit]
 
    # ── Mode 4: Similar movies ────────────────────────────────────────────────
    elif mode == "🎬 Similar Movies":
        st.subheader("🎬 Find Similar Movies")
        st.caption(
            "Enter a movie title to find semantically similar films. "
            "We look up the movie's stored plot vector and run Qdrant's `recommend()` API."
        )
 
        title_query = st.text_input("Movie title (partial match OK)", placeholder="Inception")
        limit       = st.slider("Number of similar movies", 1, 20, default_limit, key="sim_limit")
 
        if st.button("Find Similar", type="primary", key="sim_go"):
            if not title_query.strip():
                st.warning("Please enter a movie title.")
            else:
                with st.spinner("Looking up movie…"):
                    # Step 1: find the movie by title using keyword scroll
                    lookup, _ = client.scroll(
                        collection_name=COLLECTION_NAME,
                        scroll_filter=Filter(must=[
                            FieldCondition(key="title",      match=MatchText(text=title_query.strip())),
                            FieldCondition(key="chunk_type", match=MatchValue(value="plot")),
                        ]),
                        limit=1,
                        with_payload=True,
                        with_vectors=False,
                    )
 
                    # Fallback to "full" chunk if "field" strategy wasn't used
                    if not lookup:
                        lookup, _ = client.scroll(
                            collection_name=COLLECTION_NAME,
                            scroll_filter=Filter(must=[
                                FieldCondition(key="title",      match=MatchText(text=title_query.strip())),
                                FieldCondition(key="chunk_type", match=MatchValue(value="full")),
                            ]),
                            limit=1,
                            with_payload=True,
                            with_vectors=False,
                        )
 
                if not lookup:
                    st.error(f"Movie **'{title_query}'** not found. Try a different title or add it first.")
                else:
                    source        = lookup[0]
                    source_title  = source.payload.get("title", "?")
                    source_uuid   = source.id   # ← the UUID Qdrant stored it under
 
                    st.info(f"Finding movies similar to **{source_title}** …")
 
                    # Step 2: recommend using the stored vector UUID
                    source_movie_id = source.payload.get("movie_id")
                    with st.spinner("Running recommend()…"):
                        response = client.query_points(
                            collection_name=COLLECTION_NAME,
                            query=RecommendQuery(
                                recommend=RecommendInput(
                                    positive=[source_uuid],
                                    negative=[],
                                )
                            ),
                            limit=(limit + 1) * 3,
                            with_payload=True,
                            with_vectors=False,
                        )
                    raw = [
                        r for r in response.points
                        if r.payload.get("movie_id") != source_movie_id
                    ]
                    results_raw = dedup(raw)[:limit]
 
    # ── Render results (shared across all modes) ──────────────────────────────
    if results_raw:
        st.divider()
        st.subheader(f"Results — {len(results_raw)} movie(s) found")
 
        # Summary table (collapsed by default so cards are the focus)
        with st.expander("📋 Show as table"):
            import pandas as pd
            rows = [point_to_dict(p) for p in results_raw]
            df_display = pd.DataFrame(rows)[["title", "release_year", "genres", "rating", "score", "chunk_type"]]
            st.dataframe(df_display, use_container_width=True)
 
        # Result cards
        for rank, point in enumerate(results_raw, 1):
            render_card(point_to_dict(point), rank)
 
    elif any([
        mode == "🧠 Semantic"            and "sem_go"   in st.session_state,
        mode == "🔤 Keyword"             and "kw_go"    in st.session_state,
        mode == "🎯 Semantic + Filters"  and "filt_go"  in st.session_state,
        mode == "🎬 Similar Movies"      and "sim_go"   in st.session_state,
    ]):
        st.info("No results found. Try a different query or lower the threshold.")


# ╔═════════════════════════════════════════════════════════════════════════════
# ║  TAB 3 — RAG Q&A
# ╚═════════════════════════════════════════════════════════════════════════════
with tab_rag:
    st.header("🤖 RAG — Ask Anything About Movies")
    st.caption(
        "Type a question in plain English. The app retrieves relevant movie chunks "
        "from Qdrant and passes them as context to Claude, which answers using "
        "**only** that retrieved information — no hallucination from training data."
    )
 
    # ── Guard: API key must be set ────────────────────────────────────────────
    # We check this early and block the rest of the UI until the key is provided.
    # This gives a clear error rather than a cryptic API failure later.
    if not anthropic_key.strip():
        st.warning(
            "⚠️ Enter your **Anthropic API key** in the sidebar to use this tab.\n\n"
            "Get one at [console.anthropic.com](https://console.anthropic.com)."
        )
        st.stop()   # halts rendering of this tab only — other tabs still work
 
    # ── How RAG works — visual explainer ─────────────────────────────────────
    # Shown collapsed so experienced users skip it, but learners can expand it.
    with st.expander("📖 How does RAG work here?", expanded=False):
        st.markdown("""
        **The 3-step pipeline — visible in this UI:**
 
        1. **Retrieve** — your question is embedded into a vector and Qdrant
           finds the most semantically similar movie chunks. You'll see exactly
           which chunks were retrieved and their similarity scores.
 
        2. **Augment** — those chunks are formatted into a prompt as numbered
           `[Source 1]`, `[Source 2]`... The LLM is instructed to answer
           *only* from those sources and cite them.
 
        3. **Generate** — Claude reads the context and writes a grounded answer.
           If the answer isn't in the retrieved chunks, it says so rather than
           guessing.
 
        **Why this matters:** The "Sources Used" section below every answer lets
        you verify *why* the LLM said what it said. If a source looks wrong,
        adjust `top-k` in the sidebar or rephrase your question.
        """)
 
    st.divider()
 
    # ── Conversation history ──────────────────────────────────────────────────
    # st.session_state persists values across Streamlit reruns (button clicks).
    # We use it to store the full conversation so users can scroll back through
    # previous questions and answers in the same session.
    if "rag_history" not in st.session_state:
        st.session_state.rag_history = []   # list of RAGTurn dicts
 
    # ── Suggested questions (quick-start chips) ───────────────────────────────
    st.markdown("**💡 Try one of these:**")
    suggestions = [
        "What is the main theme of Inception?",
        "Compare how Interstellar and Arrival deal with time.",
        "I loved Parasite — what similar movies are in the database?",
        "What are common themes in the sci-fi movies?",
        "What was the budget of The Dark Knight?",   # ← "I don't know" test
    ]
 
    # Render chips as inline buttons — clicking one pre-fills the text input
    chip_cols = st.columns(len(suggestions))
    for col, suggestion in zip(chip_cols, suggestions):
        if col.button(suggestion[:30] + "…", key=f"chip_{suggestion[:20]}"):
            st.session_state.rag_prefill = suggestion
 
    # ── Question input ────────────────────────────────────────────────────────
    prefill = st.session_state.pop("rag_prefill", "")   # consume after one use
    question = st.text_input(
        "Your question",
        value=prefill,
        placeholder="What psychological themes appear in mind-bending thrillers?",
        key="rag_question_input",
    )
 
    ask_col, clear_col = st.columns([2, 1])
    ask_clicked   = ask_col.button("🤖 Ask Claude", type="primary", key="rag_ask")
    clear_clicked = clear_col.button("🗑️ Clear history", key="rag_clear")
 
    if clear_clicked:
        st.session_state.rag_history = []
        st.rerun()
 
    # ── Run RAG pipeline ──────────────────────────────────────────────────────
    if ask_clicked:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            # ── Step 1: Retrieve ──────────────────────────────────────────────
            with st.spinner("🔍 Retrieving from Qdrant…"):
                import textwrap
                from rag import retrieve, build_prompt   # reuse rag.py functions
 
                chunks = retrieve(question, top_k=rag_top_k)
 
            if not chunks:
                st.error("No relevant movies found in the database. Try rephrasing or add more movies first.")
            else:
                # ── Step 2: Augment ───────────────────────────────────────────
                prompt = build_prompt(question, chunks)
 
                # ── Step 3: Generate ──────────────────────────────────────────
                with st.spinner("🤖 Claude is thinking…"):
                    import anthropic as ac
                    llm    = get_llm_client(anthropic_key.strip())
                    message = llm.messages.create(
                        model      = rag_model,
                        max_tokens = 1024,
                        messages   = [{"role": "user", "content": prompt}],
                    )
                    answer = message.content[0].text
 
                # Save to session history so it appears in the conversation log
                st.session_state.rag_history.append({
                    "question": question,
                    "answer":   answer,
                    "chunks":   chunks,
                    "prompt":   prompt,
                    "model":    rag_model,
                    "top_k":    rag_top_k,
                })
 
    # ── Render conversation history ───────────────────────────────────────────
    # Most recent turn first — reverse so the latest answer is always at the top.
    if st.session_state.rag_history:
        st.divider()
 
        for turn_idx, turn in enumerate(reversed(st.session_state.rag_history)):
            is_latest = (turn_idx == 0)
 
            # ── Question bubble ───────────────────────────────────────────────
            st.markdown(f"**❓ {turn['question']}**")
 
            # ── Answer box ────────────────────────────────────────────────────
            with st.container(border=True):
                st.markdown(turn["answer"])
 
            # ── Sources used ──────────────────────────────────────────────────
            # This is the most important learning section — shows exactly which
            # chunks were retrieved and passed to the LLM as context.
            # Expanded for the latest turn; collapsed for older ones.
            with st.expander(
                f"📚 Sources used ({len(turn['chunks'])} chunks retrieved, top_k={turn['top_k']})",
                expanded=is_latest,
            ):
                for i, chunk in enumerate(turn["chunks"], 1):
                    src_col_info, src_col_score = st.columns([8, 1])
                    with src_col_info:
                        st.markdown(f"**[Source {i}] {chunk.title}** ({chunk.year})")
                        st.caption(
                            f"🎭 {chunk.genres}  ·  "
                            f"chunk type: `{chunk.chunk_type}`"
                        )
                        # Show the exact chunk text that was embedded and matched
                        st.info(f"**Matched chunk:** {chunk.chunk_text}")
                    with src_col_score:
                        # Score tells you how relevant this chunk was to the question
                        st.metric("Score", f"{chunk.score:.3f}")
 
                    if i < len(turn["chunks"]):
                        st.divider()
 
            # ── Debug: full prompt ────────────────────────────────────────────
            # Hidden by default — expand to see exactly what was sent to Claude.
            # This is the best way to understand what "augmentation" actually does.
            with st.expander("🔬 Debug — full prompt sent to LLM", expanded=False):
                st.caption(
                    "This is exactly what was sent to Claude. "
                    "Notice how the retrieved chunks appear as numbered sources "
                    "inside the prompt — that's the augmentation step."
                )
                st.code(turn["prompt"], language=None)
 
            st.markdown(f"<div style='color:gray;font-size:11px'>model: {turn['model']}</div>", unsafe_allow_html=True)
            st.divider()