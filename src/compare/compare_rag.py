"""
TMDB + Qdrant RAG vs. Neo4j RAG
========================================
Compares the performance of the two RAG systems.

EndPoints:
    GET /health         - Health check- readiness check
    POST /search/neo4j - Search Neo4j RAG
    POST /search/qdrant - Search Qdrant RAG
    POST /search/compare - Compare the two RAG systems

Requirements:
    pip install fastapi uvicorn langchain langchain-neo4j langchain-qdrant \
        langchain-anthropic langchain-huggingface sentence-transformers \
        pandas neo4j qdrant-client pydantic

Prerequisites:
    - Neo4j running locally on default port (7687). Movie Embeddings Index must be created.
    - Qdrant Docker container running locally on port (6333). Movie Collection must be created.
    - Anthropic API key must be set in .env file.

Run:
    uvicorn compare.compare_rag:app --app-dir src --reload --port 8000

Docs:
    http://localhost:8000/docs
"""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from langchain_anthropic import ChatAnthropic

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

## =================================================
## 1 Configurtation
## =================================================

NEO4J_URI = os.getenv("NEO4J_URI") or os.getenv("Neo4j_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER") or os.getenv("Neo4j_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") or os.getenv("Neo4j_PASSWORD", "")
NEO4J_INDEX = "movieEmbeddingsIndex"
NEO4J_NODE_LABEL = "Movie"
NEO4J_EMBEDDING_PROP = "embedding"

QDRANT_URL = os.getenv("QDRANT_URL") or os.getenv("Qdrant_URL", "http://localhost:6333")
QDRANT_COLLECTION = "movies"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CLAD_MODEL = "claude-sonnet-4-6"
DEFAULT_TOP_K = 5


## =================================================
# Resource initialization (runs once at startup, stored on app.state)
## =================================================

def init_neo4j_store(embedder):
    return Neo4jVector.from_existing_index(
        embedding=embedder,
        index_name=NEO4J_INDEX,
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        node_label=NEO4J_NODE_LABEL,
        text_node_property="overview",
        embedding_node_property=NEO4J_EMBEDDING_PROP,
    )


class MovieQdrantVectorStore(QdrantVectorStore):
    """Map our flat Qdrant payloads (title, movie_id, ...) into LangChain metadata."""

    @classmethod
    def _document_from_point(
        cls,
        scored_point,
        collection_name,
        content_payload_key,
        metadata_payload_key,
    ):
        payload = scored_point.payload or {}
        nested = payload.get(metadata_payload_key) or {}
        if nested:
            metadata = dict(nested)
        else:
            metadata = {
                key: value
                for key, value in payload.items()
                if key != content_payload_key
            }

        content = (
            payload.get(content_payload_key)
            or payload.get("chunk_text")
            or payload.get("overview")
            or ""
        )
        metadata["_id"] = scored_point.id
        metadata["_collection_name"] = collection_name
        return Document(page_content=content, metadata=metadata)


def init_qdrant_store(embedder):
    client = QdrantClient(url=QDRANT_URL)
    existing = [c.name for c in client.get_collections().collections]

    if QDRANT_COLLECTION not in existing:
        raise RuntimeError(
            f"Qdrant collection '{QDRANT_COLLECTION}' not found. Please create it first."
        )

    return MovieQdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=embedder,
        content_payload_key="overview",
        validate_collection_config=False,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    
    print("starting up - loading embedder and connecting to vector stores")

    embedder = HuggingFaceEmbeddings(
        model_name=f"sentence-transformers/{EMBEDDING_MODEL}"
    )
    app.state.embedder = embedder
    app.state.neo4j_store = init_neo4j_store(embedder)
    app.state.qdrant_store = init_qdrant_store(embedder)

    if os.getenv("ANTHROPIC_API_KEY"):
        app.state.llm = ChatAnthropic(model=CLAD_MODEL,temperature=0.0)
    else:
        app.state.llm = None
        print(
            "Anthropic API key not found, LLM will be disabled"
            "retrivel results only, no grenerated answers"
        )

    print("startup complete - ready to serve")
    yield
    print("shutting down - cleaning up resources")
    print("shutdown complete")

app = FastAPI(
    title="Compare RAG API",
    description="API for comparing the performance of the two RAG systems",
    version="0.1.0",
    lifespan=lifespan,
)


##=========================================================
## Request and Response Models
##=========================================================

class SearchRequest(BaseModel):
    query: str = Field(..., description="The query string to search for", min_length=2)
    top_k: int = Field(default=DEFAULT_TOP_K, description="The number of results to return", ge=1, le=20)
    generated_answer: bool = Field(
        True, 
        description="Whether to generate an answer using the LLM")

class RetrievedDoc(BaseModel):
    title: Optional[str] = None
    score: float
    content: str
    metadata: dict

class StoreResult(BaseModel):
    store: str
    elapsed_seconds: float
    documents: list[RetrievedDoc]
    answer: Optional[str] = None

class CompareResponse(BaseModel):
    query: str
    neo4j: StoreResult
    qdrant: StoreResult


##=========================================================
## Core Langchain comparision chain
##=========================================================


RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "human",
        """Answer the question using only the context below. If the context does not
contain the answer, say "I don't have enough information to answer that question."

Context: {context}

Question: {question}

Answer:""",
    ),
])


def doc_title(document: Document) -> str:
    return document.metadata.get("title") or "Unknown"


def movie_key(document: Document):
    movie_id = document.metadata.get("movie_id")
    if movie_id is not None:
        return int(movie_id)
    return doc_title(document)


def dedup_scored_documents(
    scored_documents: list[tuple[Document, float]],
    top_k: int,
) -> list[tuple[Document, float]]:
    """Keep one best-scoring chunk per movie."""
    seen: dict[object, tuple[Document, float]] = {}
    for doc, score in scored_documents:
        key = movie_key(doc)
        if key not in seen or score > seen[key][1]:
            seen[key] = (doc, score)
    ranked = sorted(seen.values(), key=lambda item: item[1], reverse=True)
    return ranked[:top_k]


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{doc_title(d)}] {d.page_content}" for d in docs
    )


def build_store_chain(store, llm, top_k: int):

    def retrieve_with_score(query: str):
        raw = store.similarity_search_with_score(query, k=top_k * 3)
        return dedup_scored_documents(raw, top_k)

    retrieve_step = RunnableParallel(
        scored_documents = RunnableLambda(retrieve_with_score),
        question = RunnablePassthrough(),
    )

    if llm is not None:
        answer_step = RunnableParallel(
            scored_documents = lambda x: x['scored_documents'],
            answer = (
                RunnableParallel(
                    context=lambda x: format_docs([d for d, _ in x["scored_documents"]]),
                    question = lambda x: x['question'],
                )
                | RAG_PROMPT
                | llm
                | StrOutputParser()
            )
        )

        return retrieve_step | answer_step
    else:
        return retrieve_step | RunnableLambda(
            lambda x: {
                "scored_documents": x['scored_documents'],
                "answer" : None
            }
        )


def build_compare_chain(neo4j_store, qdrant_store, llm, top_k: int):

    neo4j_chain = build_store_chain(neo4j_store, llm, top_k)
    qdrant_chain = build_store_chain(qdrant_store, llm, top_k)

    return RunnableParallel(
        neo4j=neo4j_chain,
        qdrant=qdrant_chain,
    )

##=========================================================
## API Endpoints
##=========================================================

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["Info"])
async def health():
    return {
        "status": "ok",
        "neo4j_index": NEO4J_INDEX,
        "qdrant_collection": QDRANT_COLLECTION,
        "claude_enabled": app_state_has_llm()
    }

def app_state_has_llm():
    return getattr(app.state, "llm", None) is not None


def _to_store_result(store_name: str, chain_output: dict, elapsed: float) -> StoreResult:
    docs = [
        RetrievedDoc(
            title=doc_title(d),
            score=s,
            content=d.page_content,
            metadata=d.metadata,
        )
        for d, s in chain_output["scored_documents"]
    ]

    return StoreResult(
        store=store_name,
        elapsed_seconds=elapsed,
        documents=docs,
        answer=chain_output["answer"],
    )

@app.post("/search/neo4j", tags=["Search"], response_model=StoreResult)
async def search_neo4j(request: SearchRequest):
    llm = app.state.llm if request.generated_answer else None
    chain = build_store_chain(app.state.neo4j_store, llm, request.top_k)

    t0 = time.time()

    try:
        result = await chain.ainvoke(request.query)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"neo4j chain failed: {str(e)}")

    elapsed = time.time() - t0
    return _to_store_result("neo4j", result, elapsed)

@app.post("/search/qdrant", tags=["Search"], response_model=StoreResult)
async def search_qdrant(request: SearchRequest):
    llm = app.state.llm if request.generated_answer else None
    chain = build_store_chain(app.state.qdrant_store, llm, request.top_k)

    t0 = time.time()

    try:
        result = await chain.ainvoke(request.query)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"qdrant chain failed: {str(e)}")

    elapsed = time.time() - t0
    return _to_store_result("qdrant", result, elapsed)


@app.post("/search/compare", tags=["Search"], response_model=CompareResponse)
async def compare_search(request: SearchRequest):
    llm = app.state.llm if request.generated_answer else None
    comparision_chain = build_compare_chain(app.state.neo4j_store, app.state.qdrant_store, llm, request.top_k)

    t0 = time.time()

    try:
        result = await comparision_chain.ainvoke(request.query)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"comparison chain failed: {str(e)}")

    elapsed = time.time() - t0
    return CompareResponse(
        query=request.query,
        neo4j=_to_store_result("neo4j", result["neo4j"], elapsed),
        qdrant=_to_store_result("qdrant", result["qdrant"], elapsed),
    )

