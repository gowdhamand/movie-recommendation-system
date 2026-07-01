from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from movie_recommender import search
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    print("Embedding model and Qdrant client ready")
    yield
    print("Shutting down...")


app = FastAPI(
    title="Movie Recommender API",
    description= (
        "API for movie recommendations using Qdrant and Semantic Search"
    ),
    version="0.1.0",
    lifespan=lifespan,
)

class MovieResult(BaseModel):
    """A single movie returned by any search endpoint"""
    movie_id    : int
    title       : str
    release_year: int
    genres      : str
    rating      : float
    overview    : str
    chunk_type  : str           # metadata, plot, genre, full
    score       : Optional[float] = None # Only for semantic search


class SearchReponse(BaseModel):
    """Response for any search endpoint"""
    query       : str
    mode        : str        # keyword, semantic, chunked, similar
    count       : int        # number of results returned
    results     : list[MovieResult]

def _payload_dict(point: search.ScoredPoint) -> dict:
    payload = point.payload or {}
    return payload if isinstance(payload, dict) else dict(payload)


def _format_genres(genres) -> str:
    if genres is None:
        return ""
    if isinstance(genres, (list, tuple)):
        return ", ".join(str(g).strip() for g in genres if str(g).strip())
    return str(genres).strip()


def _to_movie_result(point: search.ScoredPoint, score: Optional[float] = None) -> MovieResult:
    """Convert a Qdrant point to a MovieResult"""
    p = _payload_dict(point)
    actual_score = score if score is not None else getattr(point, "score", None)
    return MovieResult(
        movie_id=int(p.get("movie_id", 0)),
        title=str(p.get("title", "")),
        release_year=int(p.get("release_year", 0)),
        genres=_format_genres(p.get("genres")),
        rating=float(p.get("rating", 0.0)),
        overview=str(p.get("overview", "")),
        chunk_type=str(p.get("chunk_type", "")),
        score=round(actual_score, 4) if actual_score is not None else None,
    )

def _dedup(points) -> list:
    """Remove duplicate movies from a list of points"""
    seen: dict[int, object] = {}
    for p in points:
        payload = _payload_dict(p)
        mid = payload.get("movie_id")
        if mid is None:
            continue
        sc = getattr(p, "score", None) or 0.0
        if mid not in seen or sc > getattr(seen[mid], "score", 0.0):
            seen[mid] = p
    return list(seen.values())


@app.get("/health", tags=["Info"])
def health():
    try:
        info = search.client.get_collection(search.COLLECTION_NAME)
        return {
            "status": "ok",
            "collection": search.COLLECTION_NAME,
            "points": info.points_count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/semantic", response_model=SearchReponse, tags=["Search"])
async def semantic_search(
    q : str = Query(..., description="The query string to search for", min_length=2),
    limit : int = Query(default=5, description="The number of results to return", ge=1, le=20)
):
    """Search for movies using semantic similarity"""
    try:
        raw = search.semantic_search(q, limit=limit * 3)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    deduped = _dedup(raw)[:limit]
    results = [_to_movie_result(p, getattr(p, "score", None)) for p in deduped]

    return SearchReponse(
        query=q,
        mode="semantic",
        count=len(results),
        results=results,
    )


@app.get("/search/keyword", response_model=SearchReponse, tags=["Search"])
async def keyword_search(
    q : str = Query(..., description="The query string to search for", min_length=2),
    limit : int = Query(default=5, description="The number of results to return", ge=1, le=20)
):
    """Search for movies using keyword matching"""
    try:
        raw = search.keyword_search(q, limit=limit * 3)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    deduped = _dedup(raw)[:limit]
    results = [_to_movie_result(p, getattr(p, "score", None)) for p in deduped]

    return SearchReponse(
        query=q,
        mode="keyword",
        count=len(results),
        results=results,
    )


class FilterSearchRequest(BaseModel):
    """Request for filtering search"""
    query : str = Field(..., description="The query string to search for", min_length=2)
    genre : Optional[str] = Field(default=None, description="Filter by genre eg action, comedy, drama, etc.")
    min_rating : Optional[float] = Field(default=None, description="Filter by minimum rating eg 7.0, 8.0, 9.0, etc.")
    max_rating : Optional[float] = Field(default=None, description="Filter by maximum rating eg 7.0, 8.0, 9.0, etc.")
    year_from : Optional[int] = Field(default=None, description="Filter by year from eg 2000, 2001, 2002, etc.")
    limit : int = Field(default=5, description="The number of results to return", ge=1, le=20)

    # example shown in /docs
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "crime and betrayal",
                    "genre": "Action",
                    "min_rating": 8.0,
                    "max_rating": 10.0,
                    "year_from": 2008,
                    "limit": 5
                }
            ]
        }
    }


@app.get("/search/filtered", response_model=SearchReponse, tags=["Search"])
async def filtered_search(
    request: FilterSearchRequest
):
    """Search for movies using filtering"""
    try:
        raw = search.semantic_search_with_filter(request.query, request.genre, request.min_rating, request.max_rating, request.year_from, request.limit * 3)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    deduped = _dedup(raw)[:request.limit]
    results = [_to_movie_result(p, getattr(p, "score", None)) for p in deduped]

    return SearchReponse(
        query=request.query,
        mode="filtered",
        count=len(results),
        results=results,
    )

@app.get("/search/similar", response_model=SearchReponse, tags=["Search"])
async def similar_search(
    movie_id: int = Query(..., description="The ID of the movie to find similar movies for"),
    limit: int = Query(default=5, description="The number of results to return", ge=1, le=20)
):
    """Search for similar movies"""
    try:
        raw = search.find_similar_movies(movie_id, limit=limit * 3)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    if not raw:
        raise HTTPException(status_code=404, detail="No similar movies found")

    results = [_to_movie_result(p, getattr(p, "score", None)) for p in raw]

    return SearchReponse(
        query=f"Similar to movie_id: {movie_id}",
        mode="similar",
        count=len(results),
        results=results,
    )

@app.get("/search/threshold", response_model=SearchReponse, tags=["Search"])
async def threshold_search(
    q : str = Query(..., description="The query string to search for", min_length=2),
    threshold : float = Query(default=0.5, description="The threshold for the search", ge=0.0, le=1.0),
    limit : int = Query(default=5, description="The number of results to return", ge=1, le=20)
):
    """Search for movies using semantic similarity with threshold"""
    try:
        raw = search.semantic_search_with_threshold(q, threshold=threshold, limit=limit * 3)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    deduped = _dedup(raw)[:limit]
    results = [_to_movie_result(p, getattr(p, "score", None)) for p in deduped]

    return SearchReponse(
        query=q,
        mode="threshold",
        count=len(results),
        results=results,
    )
        