import os
import textwrap
from dataclasses import dataclass
from pathlib import Path

import dotenv
import anthropic
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText
from sentence_transformers import SentenceTransformer

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
dotenv.load_dotenv()


# 1 Config
COLLECTION_NAME = "movies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

LLM_MODEL = "claude-sonnet-4-6"
TOP_K = 5
MAX_TOKENS = 1024

# Clients
embed_model = SentenceTransformer(EMBEDDING_MODEL)

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def get_llm() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to .env in the project root."
        )
    return anthropic.Anthropic(api_key=api_key)

print("Ready to RAG!\n")


@dataclass
class RetrievedChunk:
    title: str
    year: int
    genres: str
    overview: str
    chunk_text: str
    chunk_type: str
    score: float

def retrive(quetion: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
    query_vector = embed_model.encode(quetion).tolist()
    raw_hits = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k * 3,
        with_payload=True,
        with_vectors=False,
    )

    seen: dict[int, RetrievedChunk] = {}
    for hit in raw_hits.points:
        mid = hit.payload
        chunk = RetrievedChunk(
            title=hit.payload.get("title", ""),
            year=hit.payload.get("release_year", 0),
            genres=hit.payload.get("genres", ""),
            overview=hit.payload.get("overview", ""),
            chunk_text=hit.payload.get("chunk_text", ""),
            chunk_type=hit.payload.get("chunk_type", ""),
            score=hit.score,
        )
       
        if chunk.title not in seen or chunk.score > seen[chunk.title].score:
            seen[chunk.title] = chunk

    #sort by score and return top k
    deduped = sorted(seen.values(), key=lambda c: c.score, reverse=True)
    return deduped[:top_k]


retrieve = retrive


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:

    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        block = textwrap.dedent(f"""
            [Source {i}]
            Title: {chunk.title} ({chunk.year})
            Genres: {chunk.genres}
            Overview: {chunk.overview}
            Chunk Text: {chunk.chunk_text}
            Chunk Type: {chunk.chunk_type}
            Score: {chunk.score:.4f}
        """).strip()
        context_blocks.append(block)

    context_section = "\n\n".join(context_blocks)

    prompt = textwrap.dedent(f"""
        You are knowledgable movie export assistant.
        Your job is to answer questions about movies using only the context
        provided below. do not use any outside knowledge.

        If the answer cannot be found in the context, Say: 
        "I don't have enough information to answer that question."

        Always cite which source(s) you used to answer the question. e.g. "According to Source 1 and Source 2, the answer is..."

        --Context--------------------------
        {context_section}
        ------------------------------------

        Question: {question}

        Answer (cite your sources):
        """).strip()

    return prompt


def generate(prompt: str) -> str:

    message = get_llm().messages.create(
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return message.content[0].text


@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: list[RetrievedChunk]
    prompt: str

def ask(question: str, top_k: int = TOP_K) -> RAGResponse:

    print(f"f\n{'-'*60}")
    print(f"Question: {question}")

    #step 1: retrieve
    print(f"Retriveing top {top_k} chunks from Qdrant...")
    chunks = retrive(question, top_k)
    print(f"Retrieved {len(chunks)} unique movie chunks")
    for c in chunks:
        print(f"    {c.title} ({c.year}) | {c.score:.4f} [{c.chunk_type}]")

    #step 2: Autogenerate answer
    print(f"Building prompt...")
    prompt = build_prompt(question, chunks)


    #Step 3: Generate
    print("Calling LLM")
    answer = generate(prompt)
    print(f"Answer received from LLM")

    return RAGResponse(
        question=question,
        answer=answer,
        sources=chunks,
        prompt=prompt,
    )

def print_response(response: RAGResponse, show_prompt: bool = False):
    width = 60
 
    print(f"\n{'═'*width}")
    print(f"  QUESTION")
    print(f"{'─'*width}")
    print(f"  {response.question}")
 
    print(f"\n{'─'*width}")
    print(f"  ANSWER")
    print(f"{'─'*width}")
    # Wrap long lines for readable terminal output
    for line in response.answer.splitlines():
        print(textwrap.fill(line, width=width, initial_indent="  ", subsequent_indent="  "))
 
    print(f"\n{'─'*width}")
    print(f"  SOURCES USED  ({len(response.sources)} chunks retrieved)")
    print(f"{'─'*width}")
    for i, src in enumerate(response.sources, 1):
        print(f"  [{i}] {src.title} ({src.year})  score={src.score}  type={src.chunk_type}")
        print(f"       {src.genres}")
        # Show only a snippet of the matched chunk text
        snippet = src.chunk_text[:120] + ("…" if len(src.chunk_text) > 120 else "")
        print(f"       \"{snippet}\"")
 
    if show_prompt:
        print(f"\n{'─'*width}")
        print("  FULL PROMPT SENT TO LLM  (debug)")
        print(f"{'─'*width}")
        print(response.prompt)
 
    print(f"{'═'*width}\n")


if __name__ == "__main__":

    test_questions = [
        # ── Test 1: Factual question about a specific movie ───────────────────
        # Expected: LLM finds the right chunks and cites them clearly.
        "What is the main theme of Inception?",
 
        # ── Test 2: Comparison across multiple movies ─────────────────────────
        # Expected: LLM draws on multiple retrieved sources to compare.
        # Good test for whether dedup is working — should pull different movies.
        "Compare how Interstellar and Arrival deal with the concept of time.",
 
        # ── Test 3: Recommendation with reasoning ─────────────────────────────
        # Expected: LLM retrieves similar films and explains WHY they match.
        # Tests the "conversational recommender" use case.
        "I loved Parasite. What similar movies are in the database and why?",
 
        # ── Test 4: Honest "I don't know" ─────────────────────────────────────
        # Expected: LLM says it doesn't have enough info rather than hallucinating.
        # This is the most important RAG behaviour to verify.
        "What was the budget of The Dark Knight?",
 
        # ── Test 5: Genre-level question ──────────────────────────────────────
        # Expected: LLM synthesises patterns across multiple retrieved movies.
        "What are common themes in the sci-fi movies in the database?",
    
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"  TEST QUESTION: {q}")
        print(f"{'='*60}\n")
        response = ask(q)
        print_response(response)

    print("\nAll tests completed!\n")

