"""
search.py — Search Qdrant collection for movie recommendations

Run:
    pip install qdrant-client sentence-transformers pandas tqdm
    python search.py
"""

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchText,
    Range,
    RecommendInput,
    RecommendQuery,
    ScoredPoint,
)

# 1 config (must match ingest.py)
COLLECTION_NAME = "movies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
QDRANT_HOST = "localhost"

# Load once at module level - model loading in slow. Reuse it everywhere.
print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)
client = QdrantClient(host=QDRANT_HOST, port=6333)
print("Ready to search!\n")


def semantic_search(query: str, limit: int = 5) -> list[ScoredPoint]:
    query_vector = model.encode(query).tolist()

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return response.points

def semantic_search_with_filter(
        query: str, 
        genre: str = None, 
        min_rating: float = None, 
        max_rating : float = None,
        year_from : int = None,
        limit: int = 5
) -> list[ScoredPoint]:
    """
    Semantic search with optional filters: genre, rating, year
    """

    query_vector = model.encode(query).tolist()

    conditions = []

    if genre:
        conditions.append(FieldCondition(key="genres", match=MatchText(text=genre)))

    rating_range = {}
    if min_rating: rating_range["gte"] = min_rating
    if max_rating: rating_range["lte"] = max_rating
    if rating_range:
        conditions.append(
            FieldCondition(key="rating", range=Range(**rating_range))
        )

    if year_from:
        conditions.append(
            FieldCondition(key="release_year", range=Range(gte=year_from))
        )

    search_filter = Filter(must=conditions) if conditions else None

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True,
        with_vectors=False,
        query_filter=search_filter,
    )

    return response.points

def print_results(results, title: str ="Results"):
    print(f"'\n{'='*60}")
    print(f"    {title}")
    print(f"{'='*60}\n")

    if not results:
        print("    No results found")
        return
    
    for i, r in enumerate(results, start=1):
        p = r.payload
        score = f" score={r.score:.4f}" if hasattr(r, "score") else ""
        print(f"\n #{i} {p.get('title', '?')} ({p.get('release_year', '?')}) | {score}")
        print(f"    Genere : {p.get('genres', '?')}")
        print(f"    Rating : {p.get('rating', '?')} | chunks={p.get('chunk_type', '?')}")
        print(f"    Plot : {p.get('overview', '?')}")


def keyword_search(query: str, limit: int = 5) -> list[ScoredPoint]:

    results , _nextoffset = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            should=[ # IR - match in title OR in overview
                FieldCondition(key="title", match=MatchText(text=query)),
                FieldCondition(key="overview", match=MatchText(text=query)),
            ],
        ),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    return results


def semantic_search_with_threshold(query: str, threshold: float = 0.5, limit: int = 5) -> list[ScoredPoint]:
    query_vector = model.encode(query).tolist()
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return response


def find_similar_movies(movie_id: int, limit: int = 5) -> list[ScoredPoint]:

    results, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="movie_id", match=MatchValue(value=movie_id)),
                FieldCondition(key="chunk_type", match=MatchValue(value="plot")),
            ],
        ),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    if not results:
        print(f"No similar movies found for movie_id: {movie_id}")
        results, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="movie_id", match=MatchValue(value=movie_id)),
                    FieldCondition(key="chunk_type", match=MatchValue(value="full")),
                ],
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

    if not results:
        print(f"No similar movies found for movie_id: {movie_id}")
        return []


    # The point's.id IS the UUID Drant stored it under
    plot_point_id = results[0].id
    source_title = results[0].payload.get("title", "Unknown")
    print(f"Source movie: {source_title} (ID: {plot_point_id})")

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=RecommendQuery(
            recommend=RecommendInput(
                positive=[plot_point_id],
                negative=[],
            )
        ),
        limit=limit + 5,
        with_payload=True,
        with_vectors=False,
    )

    similar = [
        point
        for point in response.points
        if point.payload.get("movie_id") != movie_id
    ][:limit]
    return similar

def semantic_chucked_search(query: str, limit: int = 5) -> list[ScoredPoint]:
    query_vector = model.encode(query).tolist()
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit * 10,
        with_payload=True,
        with_vectors=False,
    )

    seen: dict[int, ScoredPoint] = {}
    for point in response.points:
        mid = point.payload.get("movie_id")
        if mid is None:
            continue
        if mid not in seen or point.score > seen[mid].score:
            seen[mid] = point

    return sorted(seen.values(), key=lambda p: p.score, reverse=True)[:limit]

if __name__ == "__main__":

    # 1. Pure semantic search
    # res = semantic_search("The Dark Knight")
    # print_results(res, "1. Semantic Search — 'The Dark Knight'")

    # 2. Semantic search with filters
    # res = semantic_search_with_filter(
    #     query="The Dark Knight",
    #     genre="Action",
    #     min_rating=8.0,
    #     max_rating=10.0,
    #     year_from=2008,
    # )
    # print_results(res, "2. Semantic Search with Filters — 'The Dark Knight'")


    # 3. Key word search
    # res = keyword_search("Batman")
    # print_results(res, "3. Keyword Search — 'Batman'")

    # 4. Semantic search with threshold
    # res = semantic_search_with_threshold("romantic musical in a big city", threshold=0.5)
    # print_results(res, "4. Semantic Search with Threshold — 'romantic musical in a big city'")

    # 5. Find similar movies
    # res = find_similar_movies(27205, limit=5)
    # print_results(res, "5. Find Similar Movies — 'Inception'")

    # 6. Semantic chucked search
    res = semantic_chucked_search("Tamil", limit=5)
    print_results(res, "6. Semantic Chucked Search — 'Tamil'")