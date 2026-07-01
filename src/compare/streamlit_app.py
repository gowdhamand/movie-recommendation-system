"""
Streamlit UI for Neo4j vs Qdrant RAG comparison.

Run:
    streamlit run src/compare/streamlit_app.py
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from neo4j import GraphDatabase

_COMPARE_DIR = Path(__file__).resolve().parent
if str(_COMPARE_DIR) not in sys.path:
    sys.path.insert(0, str(_COMPARE_DIR))

import compare_rag as cr
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

st.set_page_config(
    page_title="RAG Compare",
    page_icon="🎬",
    layout="wide",
)


@st.cache_resource
def load_stores():
    embedder = HuggingFaceEmbeddings(
        model_name=f"sentence-transformers/{cr.EMBEDDING_MODEL}"
    )
    neo4j_store = cr.init_neo4j_store(embedder)
    qdrant_store = cr.init_qdrant_store(embedder)
    llm = (
        ChatAnthropic(model=cr.CLAD_MODEL, temperature=0.0)
        if os.getenv("ANTHROPIC_API_KEY")
        else None
    )
    return neo4j_store, qdrant_store, llm


@st.cache_resource
def get_neo4j_driver():
    return GraphDatabase.driver(
        cr.NEO4J_URI,
        auth=(cr.NEO4J_USER, cr.NEO4J_PASSWORD),
    )


GRAPH_EXTRAS_CYPHER = """
MATCH (m:Movie)
WHERE m.id = $movie_id
OPTIONAL MATCH (m)-[:IN_GENRE]->(g:Genre)
OPTIONAL MATCH (m)-[:IN_KEYWORD]->(k:Keyword)
OPTIONAL MATCH (m)-[:PRODUCED_BY]->(c:ProductionCompany)
OPTIONAL MATCH (m)-[:PRODUCED_IN]->(co:Country)
OPTIONAL MATCH (m)-[:SPOKEN_IN]->(l:Language)
RETURN m.id AS id,
       m.title AS title,
       m.overview AS overview,
       m.release_date AS release_date,
       m.vote_average AS vote_average,
       m.vote_count AS vote_count,
       m.popularity AS popularity,
       m.status AS status,
       m.budget AS budget,
       m.revenue AS revenue,
       m.runtime AS runtime,
       m.tagline AS tagline,
       m.homepage AS homepage,
       m.original_language AS original_language,
       m.original_title AS original_title,
       collect(DISTINCT g.name) AS genres,
       collect(DISTINCT k.name) AS keywords,
       collect(DISTINCT c.name) AS companies,
       collect(DISTINCT co.name) AS countries,
       collect(DISTINCT l.name) AS languages
"""

SIMILAR_BY_KEYWORDS_CYPHER = """
MATCH (m:Movie {id: $movie_id})-[:IN_KEYWORD]->(k:Keyword)<-[:IN_KEYWORD]-(other:Movie)
WHERE other.id <> m.id
WITH other, collect(DISTINCT k.name) AS shared_keyword_names, count(DISTINCT k) AS shared_keywords
RETURN other.id AS id,
       other.title AS title,
       other.release_date AS release_date,
       other.vote_average AS vote_average,
       other.overview AS overview,
       shared_keywords,
       shared_keyword_names
ORDER BY shared_keywords DESC
LIMIT 5
"""


def _movie_id_from_doc(doc: cr.RetrievedDoc) -> int | None:
    raw = doc.metadata.get("movie_id") or doc.metadata.get("id")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _release_year_from_date(release_date) -> int | None:
    if not release_date:
        return None
    try:
        return int(str(release_date)[:4])
    except (TypeError, ValueError):
        return None


def _clean_list(values) -> list[str]:
    return [value for value in (values or []) if value]


def _graph_record_to_dict(record, similar: list[dict]) -> dict:
    if not record:
        return {"similar_by_keywords": similar}

    release_date = record.get("release_date")
    return {
        "id": record.get("id"),
        "title": record.get("title"),
        "overview": record.get("overview"),
        "release_date": release_date,
        "release_year": _release_year_from_date(release_date),
        "vote_average": record.get("vote_average"),
        "vote_count": record.get("vote_count"),
        "popularity": record.get("popularity"),
        "status": record.get("status"),
        "original_title": record.get("original_title"),
        "budget": record.get("budget"),
        "revenue": record.get("revenue"),
        "runtime": record.get("runtime"),
        "tagline": record.get("tagline"),
        "homepage": record.get("homepage"),
        "original_language": record.get("original_language"),
        "genres": _clean_list(record.get("genres")),
        "keywords": _clean_list(record.get("keywords")),
        "companies": _clean_list(record.get("companies")),
        "countries": _clean_list(record.get("countries")),
        "languages": _clean_list(record.get("languages")),
        "similar_by_keywords": similar,
    }


def fetch_graph_extras(movie_id: int) -> dict:
    driver = get_neo4j_driver()
    with driver.session() as session:
        record = session.run(GRAPH_EXTRAS_CYPHER, movie_id=movie_id).single()
        similar = session.run(SIMILAR_BY_KEYWORDS_CYPHER, movie_id=movie_id).data()
    return _graph_record_to_dict(record.data() if record else None, similar)


def _fmt_money(value) -> str | None:
    if value is None:
        return None
    try:
        return f"${int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _neo4j_details_payload(doc: cr.RetrievedDoc, graph: dict | None = None) -> dict:
    graph = graph or {}
    release_year = graph.get("release_year") or _release_year_from_date(
        doc.metadata.get("release_date")
    )
    genres = graph.get("genres") or doc.metadata.get("genres")
    if isinstance(genres, list):
        genres = ", ".join(genres)

    return {
        "movie_id": graph.get("id") or doc.metadata.get("movie_id") or doc.metadata.get("id"),
        "title": graph.get("title") or doc.title,
        "score": doc.score,
        "release_year": release_year,
        "release_date": graph.get("release_date") or doc.metadata.get("release_date"),
        "genres": genres,
        "rating": graph.get("vote_average") or doc.metadata.get("vote_average") or doc.metadata.get("rating"),
        "vote_count": graph.get("vote_count") or doc.metadata.get("vote_count"),
        "popularity": graph.get("popularity") or doc.metadata.get("popularity"),
        "status": graph.get("status") or doc.metadata.get("status"),
        "original_title": graph.get("original_title") or doc.metadata.get("original_title"),
        "original_language": graph.get("original_language") or doc.metadata.get("original_language"),
        "runtime": graph.get("runtime") or doc.metadata.get("runtime"),
        "budget": graph.get("budget") or doc.metadata.get("budget"),
        "revenue": graph.get("revenue") or doc.metadata.get("revenue"),
        "tagline": graph.get("tagline") or doc.metadata.get("tagline"),
        "homepage": graph.get("homepage") or doc.metadata.get("homepage"),
    }


def render_neo4j_details(doc: cr.RetrievedDoc, graph: dict | None = None):
    payload = _neo4j_details_payload(doc, graph)
    st.json({key: value for key, value in payload.items() if value is not None})

    if graph and graph.get("overview"):
        st.markdown("**Overview (from graph)**")
        st.write(graph["overview"])


def _render_graph_body(extras: dict, *, show_similar: bool):
    st.markdown("**Only in Neo4j (not stored in Qdrant payload)**")

    col1, col2, col3 = st.columns(3)
    budget = _fmt_money(extras.get("budget"))
    revenue = _fmt_money(extras.get("revenue"))
    runtime = extras.get("runtime")
    if budget:
        col1.metric("Budget", budget)
    if revenue:
        col2.metric("Revenue", revenue)
    if runtime:
        col3.metric("Runtime (min)", f"{int(runtime)}")

    if extras.get("genres"):
        st.markdown("**Genres**")
        st.write(", ".join(extras["genres"]))

    if extras.get("tagline"):
        st.caption(f"Tagline: _{extras['tagline']}_")
    if extras.get("homepage"):
        st.markdown(f"[Homepage]({extras['homepage']})")

    if extras.get("keywords"):
        st.markdown("**Keywords**")
        st.write(", ".join(extras["keywords"][:15]))

    if extras.get("companies"):
        st.markdown("**Production companies**")
        st.write(", ".join(extras["companies"][:8]))

    if extras.get("countries"):
        st.markdown("**Production countries**")
        st.write(", ".join(extras["countries"]))

    if extras.get("languages"):
        st.markdown("**Languages**")
        st.write(", ".join(extras["languages"]))

    if not show_similar:
        return

    similar = extras.get("similar_by_keywords") or []
    if similar:
        st.markdown("**Similar via shared keywords (graph traversal)**")
        for row in similar:
            year = _release_year_from_date(row.get("release_date"))
            year_label = f" ({year})" if year else ""
            shared_names = _clean_list(row.get("shared_keyword_names"))
            shared_count = row.get("shared_keywords", len(shared_names))
            label = f"{row['title']}{year_label} — {shared_count} shared keyword(s)"
            with st.expander(label):
                if shared_names:
                    st.markdown("**Shared keywords**")
                    st.write(", ".join(shared_names))
                if row.get("vote_average") is not None:
                    st.caption(f"Rating: {row['vote_average']:.1f}")
                if row.get("overview"):
                    preview = row["overview"][:300] + (
                        "..." if len(row["overview"]) > 300 else ""
                    )
                    st.write(preview)
                similar_id = row.get("id")
                if similar_id is not None:
                    try:
                        nested = fetch_graph_extras(int(similar_id))
                        st.markdown("**Full graph profile**")
                        _render_graph_body(nested, show_similar=False)
                    except Exception as exc:
                        st.warning(f"Could not load details for {row['title']}: {exc}")


def run_search(store_name: str, store, llm, query: str, top_k: int, with_answer: bool):
    chain = cr.build_store_chain(store, llm if with_answer else None, top_k)
    t0 = time.time()
    result = chain.invoke(query)
    elapsed = time.time() - t0
    return cr._to_store_result(store_name, result, elapsed)


def run_compare(neo4j_store, qdrant_store, llm, query: str, top_k: int, with_answer: bool):
    chain = cr.build_compare_chain(
        neo4j_store,
        qdrant_store,
        llm if with_answer else None,
        top_k,
    )
    t0 = time.time()
    result = chain.invoke(query)
    elapsed = time.time() - t0
    return cr.CompareResponse(
        query=query,
        neo4j=cr._to_store_result("neo4j", result["neo4j"], elapsed),
        qdrant=cr._to_store_result("qdrant", result["qdrant"], elapsed),
    )


def render_store_result(label: str, result: cr.StoreResult, *, show_graph_extras: bool = False):
    st.subheader(label)
    st.caption(f"Retrieved in {result.elapsed_seconds:.2f}s")

    if result.answer:
        st.markdown("**Answer**")
        st.info(result.answer)

    if not result.documents:
        st.warning("No matching documents found.")
        return

    st.markdown("**Retrieved movies**")
    for rank, doc in enumerate(result.documents, start=1):
        with st.container(border=True):
            st.markdown(f"**{rank}. {doc.title or 'Unknown'}**")
            st.caption(f"Score: {doc.score:.4f}")
            overview = doc.content or doc.metadata.get("overview", "")
            if overview:
                preview = overview[:300] + ("..." if len(overview) > 300 else "")
                st.write(preview)

            graph_data = None
            if show_graph_extras:
                movie_id = _movie_id_from_doc(doc)
                if movie_id is not None:
                    try:
                        graph_data = fetch_graph_extras(movie_id)
                    except Exception as exc:
                        st.warning(f"Could not load Neo4j graph data: {exc}")

            with st.expander("Details"):
                if show_graph_extras:
                    render_neo4j_details(doc, graph_data)
                else:
                    st.json(
                        {
                            "title": doc.title,
                            "score": doc.score,
                            "release_year": doc.metadata.get("release_year"),
                            "genres": doc.metadata.get("genres"),
                            "rating": doc.metadata.get("rating"),
                            "chunk_type": doc.metadata.get("chunk_type"),
                            "movie_id": doc.metadata.get("movie_id") or doc.metadata.get("id"),
                        }
                    )

            if show_graph_extras:
                with st.expander("Graph extras (Neo4j only)"):
                    if graph_data:
                        _render_graph_body(graph_data, show_similar=True)
                    elif _movie_id_from_doc(doc) is None:
                        st.caption("Graph extras unavailable (no movie id).")


with st.sidebar:
    st.title("RAG Compare")
    st.caption("Neo4j graph RAG vs Qdrant vector RAG")
    mode = st.radio(
        "Search mode",
        ["Compare both", "Neo4j only", "Qdrant only"],
    )
    top_k = st.slider("Results (top-k)", min_value=1, max_value=10, value=cr.DEFAULT_TOP_K)
    with_answer = st.toggle("Generate Claude answer", value=True)
    if with_answer and not os.getenv("ANTHROPIC_API_KEY"):
        st.warning("Set ANTHROPIC_API_KEY in .env for generated answers.")

st.title("Movie RAG Search")
st.caption("Ask a question and compare retrieval from Neo4j and Qdrant.")

query = st.text_input(
    "Your question",
    placeholder="What sci-fi movies deal with survival in space?",
)

send = st.button("Send", type="primary", use_container_width=True)

if send:
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Loading stores and searching..."):
                neo4j_store, qdrant_store, llm = load_stores()

                if mode == "Compare both":
                    response = run_compare(
                        neo4j_store, qdrant_store, llm, query.strip(), top_k, with_answer
                    )
                    col_neo4j, col_qdrant = st.columns(2)
                    with col_neo4j:
                        render_store_result("Neo4j", response.neo4j, show_graph_extras=True)
                    with col_qdrant:
                        render_store_result("Qdrant", response.qdrant)
                elif mode == "Neo4j only":
                    result = run_search("neo4j", neo4j_store, llm, query.strip(), top_k, with_answer)
                    render_store_result("Neo4j", result, show_graph_extras=True)
                else:
                    result = run_search("qdrant", qdrant_store, llm, query.strip(), top_k, with_answer)
                    render_store_result("Qdrant", result)
        except Exception as exc:
            st.error(f"Search failed: {exc}")
            st.caption("Make sure Neo4j (7687), Qdrant (6333), and ingest data are available.")
