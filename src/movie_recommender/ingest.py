"""
ingest.py — Load Kaggle movie dataset into Qdrant
Dataset: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
         (tmdb_5000_movies.csv)
 
Run:
    pip install qdrant-client sentence-transformers pandas tqdm
    python ingest.py
"""

import ast
import uuid

import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)


#1 Config
CSV_PATH = "data/tmdb_5000_movies.csv"
COLLECTION_NAME = "movies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # 384-dim, fast & good quality
CHUNK_STRATEGY = "full"  # aligned with Neo4j build_embedding_text (Option A)
BATCH_SIZE = 100
QDRANT_HOST = "localhost"


#2 Load Data
def load_movies(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    def extract_genres(val):
        try:
            return ", ".join([genre["name"] for genre in ast.literal_eval(val)])
        except (ValueError, SyntaxError, TypeError):
            return ""

    df["genres_clean"] = df["genres"].apply(extract_genres)

    df = df[
        [
            "id",
            "title",
            "release_date",
            "genres_clean",
            "overview",
            "popularity",
            "vote_average",
            "vote_count",
        ]
    ].copy()

    df = df.dropna(subset=["overview"])
    df = df[df["overview"].str.strip() != ""]
    df["release_year"] = (
        pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(0).astype(int)
    )

    print(f"Loaded {len(df)} movies with overview")

    return df


# 2 Chunking Strategy
def chunk_full(row) -> list[dict]:
    """1 Chunk per movie"""
    text = (
        f"{row.title    } ({row.release_year})"
        f"Genres: {row.genres_clean}:"
        f"{row.overview}"
    )
    return [{
        "chunk_id" : f"{row.id}-full",
        "text" : text,
        "chunk_type" : "full",
    }]


def chunk_by_field(row) -> list[dict]:
    """1 Chunk per field"""
    
    return [
        {
            "chunk_id" : f"{row.id}-metadata",
            "text" : (
                f"Title: {row.title} ({row.release_year}) |"
                f"Genres: {row.genres_clean} |"
                f"Rating: {row.vote_average} ({row.vote_count} votes)"
                f"Popularity: {row.popularity}"
            ),
            "chunk_type" : "metadata",
        },
        {
            "chunk_id" : f"{row.id}-plot",
            "text" : f"Plot of {row.title}: {row.overview}",
            "chunk_type" : "plot",
        },
        {
            "chunk_id" : f"{row.id}-genre",
            "text" : f"{row.title} is a {row.genres_clean} movie film in {row.release_year}.",
            "chunk_type" : "genre",
        }
    ]

def chunk_sliding(row, window: int = 500, overlap: int = 100) -> list[dict]:
    """Sliding window chunks"""

    chunks = []
    words = row.overview.split()
    step = window - overlap
    for i in range(0, max(1, len(words) - overlap), step):
        window_text = " ".join(words[i:i+window])
        chunks.append({
            "chunk_id" : f"{row.id}-sliding-{i//step}",
            "text" : window_text,
            "chunk_type" : "sliding",
        })
    return chunks


CHUNKERS = {
    "full" : chunk_full,
    "field" : chunk_by_field,
    "sliding" : chunk_sliding,
}


def chunk_id_to_point_id(chunk_id: str) -> str:
    """Qdrant point IDs must be an unsigned int or UUID, not arbitrary strings."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

#3. Build Qdrant points 
def build_points(df: pd.DataFrame, model: SentenceTransformer, strategy: str = "field") -> list[PointStruct]:
    """Build Qdrant points from dataframe"""
    checker = CHUNKERS[strategy]
    points = []

    print(f"\n Embedding with strategy: {strategy} ...\n")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        chunks = checker(row)
        texts = [c["text"] for c in chunks]

        # Encode all chunks for this movie in one batch (faster)
        vectors = model.encode(texts, show_progress_bar=False)

        for chunk, vec in zip(chunks, vectors):
            points.append(
                PointStruct(
                    id=chunk_id_to_point_id(chunk["chunk_id"]),
                    vector=vec.tolist(),
                    payload = {
                        # Movie meta data for filter and display
                        "chunk_id" : chunk["chunk_id"],
                        "movie_id" : int(row.id),
                        "title" : row.title,
                        "release_year" : int(row.release_year),
                        "genres" : row.genres_clean.split(","),
                        "rating" : float(row.vote_average),
                        "votes" : int(row.vote_count),
                        "popularity" : float(row.popularity),
                        "overview" : row.overview,
                        "chunk_type" : chunk["chunk_type"],
                        "chunk_text" : chunk["text"],
                    }
                )
            )


    print(f"\n Built {len(points)} points for {len(df)} movies\n")
    return points


#4 create collection & upsert
def create_collection(client: QdrantClient, vector_size: int = 384):

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"\n Collection '{COLLECTION_NAME}' already exists. Deleting...\n")
        client.delete_collection(COLLECTION_NAME)

    print(f"\n Creating collection '{COLLECTION_NAME}' with vector size {vector_size}...\n")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size, 
            distance=Distance.COSINE # Best for Sentence embeddings
        ),
    )
    print(f"\n Collection '{COLLECTION_NAME}' and (dim={vector_size} cosine) created successfully\n")
    

def upsert_batches(client: QdrantClient, points: list[PointStruct]):
    print(f"\n upserting {len(points)} points in batches of {BATCH_SIZE}...\n")
    for i in tqdm(range(0, len(points), BATCH_SIZE)):
        batch = points[i:i+BATCH_SIZE]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
        )
    print(f"\n Upserted {len(points)} points successfully\n")


# 5 Verify 
def verify_collection(client: QdrantClient):
    info = client.get_collection(COLLECTION_NAME)
    print(f"\n Collection '{COLLECTION_NAME}' info:\n")
    print(f"  - Number of points: {info.points_count}")
    # print(f"  - Number of vectors: {info.vectors_count}")
    print(f"  - Status: {info.status}")


#5 Main function
if __name__ == "__main__":
    ## Connect to Qdrant
    client = QdrantClient(host=QDRANT_HOST,port=6333)

    # Load model first so we know vector size before creating the collection
    print(f"\n Loading embedding model {EMBEDDING_MODEL}...\n")
    model = SentenceTransformer(EMBEDDING_MODEL)
    vector_size = model.get_sentence_embedding_dimension()
    print(f"  - Vector size: {vector_size}\n")


    # Pipelines 
    df      = load_movies(CSV_PATH)
    points  = build_points(df, model, CHUNK_STRATEGY)


    create_collection(client, vector_size)
    upsert_batches(client, points)
    verify_collection(client)

    print("\n Ingestion complete! You can now use the collection for recommendations.")