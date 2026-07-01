"""
TMDB + Neo4j Ingestion Script
========================================
Ingests tmdb_5000_movies.csv into a local Neo4j instance.
"""

import ast
import json
import math
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.indexes import create_vector_index as neo4j_create_vector_index

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

## 1 Connection Settings
NEO4J_URI = os.getenv("NEO4J_URI") or os.getenv("Neo4j_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER") or os.getenv("Neo4j_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") or os.getenv("Neo4j_PASSWORD", "")

## 2 Path CSV File
CSV_PATH = "data/tmdb_5000_movies.csv"

## Batch Size for Insertion
BATCH_SIZE = 100

# ══════════════════════════════════════════════════════════════════════════════
# Vector Index Settings
# ══════════════════════════════════════════════════════════════════════════════

VECTOR_INDEX_NAME = "movieEmbeddingsIndex"
EMBEDDING_PROPERTY = "embedding"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_BATCH = 64

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def _is_missing(val) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    if isinstance(val, str) and not val.strip():
        return True
    return False


def parse_json_col(val):
    """Parse a column that contains a JSON array string."""
    if _is_missing(val):
        return []
    if not isinstance(val, str):
        return list(val) if isinstance(val, list) else []
    try:
        parsed = json.loads(val)
    except (json.JSONDecodeError, TypeError, ValueError):
        try:
            parsed = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []
    return parsed if isinstance(parsed, list) else []


def safe_int(val, default=0) -> int:
    if _is_missing(val):
        return default
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


def safe_float(val, default=None):
    if _is_missing(val):
        return default
    try:
        result = float(val)
        return default if math.isnan(result) else result
    except (TypeError, ValueError):
        return default

def _genres_clean(movie) -> str:
    genre_names = [g["name"] for g in movie.get("genres", []) if g.get("name")]
    return ", ".join(genre_names)


def _release_year(movie) -> int:
    release_date = movie.get("release_date")
    if _is_missing(release_date):
        return 0
    try:
        return int(pd.to_datetime(release_date).year)
    except (TypeError, ValueError):
        return 0


def build_embedding_text(movie):
    """
    Match Qdrant `full` chunk text for fair vector comparison:
    title (year) + genres + overview
    """
    title = movie.get("title") or ""
    overview = movie.get("overview") or ""
    return (
        f"{title} ({_release_year(movie)})"
        f"Genres: {_genres_clean(movie)}:"
        f"{overview}"
    )



def clean(row):
    """Clean a dict for one movie row"""
    return {
        "id":                safe_int(row["id"]),
        "title":             str(row["title"]) if not _is_missing(row["title"]) else None,
        "original_title":    str(row["original_title"]) if not _is_missing(row["original_title"]) else None,
        "overview":          str(row["overview"]) if not _is_missing(row["overview"]) else None,
        "budget":            safe_int(row["budget"], 0),
        "revenue":           safe_int(row["revenue"], 0),
        "runtime":           safe_float(row["runtime"]),
        "popularity":        safe_float(row["popularity"]),
        "vote_average":      safe_float(row["vote_average"]),
        "vote_count":        safe_int(row["vote_count"], 0) if not _is_missing(row["vote_count"]) else None,
        "release_date":      str(row["release_date"]) if not _is_missing(row["release_date"]) else None,
        "original_language": str(row["original_language"]) if not _is_missing(row["original_language"]) else None,
        "status":            str(row["status"]) if not _is_missing(row["status"]) else None,
        "tagline":           str(row["tagline"]) if not _is_missing(row["tagline"]) else None,
        "homepage":          str(row["homepage"]) if not _is_missing(row["homepage"]) else None,
        "genres":            parse_json_col(row["genres"]),
        "keywords":          parse_json_col(row["keywords"]),
        "production_companies": parse_json_col(row["production_companies"]),
        "production_countries": parse_json_col(row["production_countries"]),
        "spoken_languages":  parse_json_col(row["spoken_languages"]),
    }

# Create (indexes & constraints)
SCHEMA_QUERIES = [
    "CREATE CONSTRAINT movie_id IF NOT EXISTS FOR (m:Movie) REQUIRE m.id IS UNIQUE",
    "CREATE CONSTRAINT genre_id IF NOT EXISTS FOR (g:Genre) REQUIRE g.id IS UNIQUE",
    "CREATE CONSTRAINT keyword_id IF NOT EXISTS FOR (k:Keyword) REQUIRE k.id IS UNIQUE",
    "CREATE CONSTRAINT company_id IF NOT EXISTS FOR (c:ProductionCompany) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT country_iso IF NOT EXISTS FOR (c:Country) REQUIRE c.iso_3166_1 IS UNIQUE",
    "CREATE CONSTRAINT language_iso IF NOT EXISTS FOR (l:Language) REQUIRE l.iso_639_1 IS UNIQUE",
]

def create_schema(session):
    """Create indexes and constraints"""
    print("Creating schema...")
    for query in SCHEMA_QUERIES:
        session.run(query)
    print("Schema created successfully")

## 3. Cypher for batch upsert
MOVIE_CYPHER = """
UNWIND $batch AS m

MERGE (movie:Movie {id: m.id})
SET 
    movie.title                     = m.title,
    movie.original_title            = m.original_title,
    movie.overview                  = m.overview,
    movie.budget                    = m.budget,
    movie.revenue                   = m.revenue,
    movie.runtime                   = m.runtime,
    movie.popularity                = m.popularity,
    movie.vote_average              = m.vote_average,
    movie.vote_count                = m.vote_count,
    movie.release_date              = m.release_date,
    movie.original_language         = m.original_language,
    movie.status                    = m.status,
    movie.tagline                   = m.tagline,
    movie.homepage                  = m.homepage

WITH movie, m
FOREACH (g IN m.genres |
    MERGE (genre:Genre {id: g.id})
    ON CREATE SET genre.name = g.name
    MERGE (movie)-[:IN_GENRE]->(genre)
)
FOREACH (k IN m.keywords |
    MERGE (keyword:Keyword {id: k.id})
    ON CREATE SET keyword.name = k.name
    MERGE (movie)-[:IN_KEYWORD]->(keyword)
)
FOREACH (c IN m.production_companies |
    MERGE (company:ProductionCompany {id: c.id})
    ON CREATE SET company.name = c.name
    MERGE (movie)-[:PRODUCED_BY]->(company)
)
FOREACH (c IN m.production_countries |
    MERGE (country:Country {iso_3166_1: c.iso_3166_1})
    ON CREATE SET country.name = c.name
    MERGE (movie)-[:PRODUCED_IN]->(country)
)
FOREACH (l IN m.spoken_languages |
    MERGE (language:Language {iso_639_1: l.iso_639_1})
    ON CREATE SET language.name = l.name
    MERGE (movie)-[:SPOKEN_IN]->(language)
)
"""

# Ingestion

def ingest(driver, movies):
    total = len(movies)
    batches = math.ceil(total / BATCH_SIZE)
    print(f"Ingesting {total} movies in {batches} batches of {BATCH_SIZE}...")

    with driver.session() as session:
        create_schema(session)

        for i in range(batches):
            batch       = movies[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            t0          = time.time()
            session.run(MOVIE_CYPHER, {"batch": batch})
            elapsed = time.time() - t0
            done = min((i+1)*BATCH_SIZE, total)
            print(f"  - Inserted {done} of {total} movies ({done/total*100:.2f}%) in {elapsed:.2f} seconds")

        print("\n Ingestion complete!")

# ══════════════════════════════════════════════════════════════════════════════
# Vector Index : Create + populate
# ══════════════════════════════════════════════════════════════════════════════

def ensure_vector_index(driver):
    print(
        f"Creating vector index '{VECTOR_INDEX_NAME}' on Movie.{EMBEDDING_PROPERTY} "
        f"with model '{EMBEDDING_MODEL}'..."
    )

    neo4j_create_vector_index(
        driver,
        VECTOR_INDEX_NAME,
        label="Movie",
        embedding_property=EMBEDDING_PROPERTY,
        dimensions=EMBEDDING_DIM,
        similarity_fn="cosine",
    )
    print(f"Vector index '{VECTOR_INDEX_NAME}' created successfully")


def populate_movie_embeddings(driver, movies):
    """
    Embed overiew + genres + keywords for each movie locally, then
    upsert the resulting vector into Neo4j
    """
    print(f"Embedding {len(movies)} movies... with model '{EMBEDDING_MODEL}'...")
    embedder = SentenceTransformerEmbeddings(model=EMBEDDING_MODEL)

    total = len(movies)
    batches = math.ceil(total / EMBEDDING_BATCH)

    with driver.session() as session:
        for i in range(batches):
            batch = movies[i*EMBEDDING_BATCH:(i+1)*EMBEDDING_BATCH]
            texts = [build_embedding_text(movie) for movie in batch]

            t0 = time.time()
            vectors = [embedder.embed_query(t) for t in texts]


            # Upsert_vectors matches Neo4j notdes by elementId by default when 
            # ids corresponds to internal Ids. Here we match by our own 'id'
            # property. instead via a email chyper pass for realiability.
            session.run(
                """
                UNWIND $rows AS row
                MATCH (movie:Movie {id: row.id})
                SET movie.embedding = row.embedding
                """,
                rows=[
                    {"id": movie["id"], "embedding": vec}
                    for movie, vec in zip(batch, vectors)
                ],
            )
            elapsed = time.time() - t0
            done = min((i+1)*EMBEDDING_BATCH, total)
            print(f"  - Inserted {done} of {total} embeddings ({done/total*100:.2f}%) in {elapsed:.2f} seconds")   


## Verify Query
VERIFY_QUERIES = {
    "movies":       "MATCH (m:Movie) RETURN COUNT(m) AS count",
    "genres":       "MATCH (g:Genre) RETURN COUNT(g) AS count",
    "keywords":     "MATCH (k:Keyword) RETURN COUNT(k) AS count",
    "companies":    "MATCH (c:ProductionCompany) RETURN COUNT(c) AS count",
    "countries":    "MATCH (c:Country) RETURN COUNT(c) AS count",
    "languages":    "MATCH (l:Language) RETURN COUNT(l) AS count",
    "IN_GENRE rels" : "MATCH ()-[r:IN_GENRE]->() RETURN COUNT(r) AS count",
    "IN_KEYWORD rels" : "MATCH ()-[r:IN_KEYWORD]->() RETURN COUNT(r) AS count",
    "PRODUCED_BY rels" : "MATCH ()-[r:PRODUCED_BY]->() RETURN COUNT(r) AS count",
    "PRODUCED_IN rels" : "MATCH ()-[r:PRODUCED_IN]->() RETURN COUNT(r) AS count",
    "SPOKEN_IN rels" : "MATCH ()-[r:SPOKEN_IN]->() RETURN COUNT(r) AS count",
}


def verify_vector_search(driver):
    """Run a simple vector similarity search"""
    print("Running a simple vector similarity search...")
    embedder = SentenceTransformerEmbeddings(model=EMBEDDING_MODEL)

    with driver.session() as session:
        record = session.run(
            """
            MATCH (m:Movie)
            WHERE m.embedding IS NOT NULL
            RETURN m.title AS title, m.embedding AS embedding
            LIMIT 1
            """
        ).single()
        if not record:
            print("No embeddings found in the database")
            return

        query_vector = record["embedding"]
        results = session.run(
            """
            CALL db.index.vector.queryNodes($index_name, $top_k, $vector)
            YIELD node, score
            WHERE node.title <> $title
            RETURN node.title AS title, score
            ORDER BY score DESC
            """,
            index_name=VECTOR_INDEX_NAME,
            top_k=4,
            vector=query_vector,
            title=record["title"],
        ).data()

        print(f"Query: {record['title']}")
        print("-" * 50)
        for result in results:
            print(f"{result['title']} (Score: {result['score']:.4f})")
        print("-" * 50)


def verify(driver):
    """Verify the ingestion process"""
    print("Node and relationships counts...")
    with driver.session() as session:
        for label, q in VERIFY_QUERIES.items():
            result = session.run(q).single()
            print(f"  {label:<25} {result['count']:>6}")
    print("-"*50)
    print("\n sample queries:")
    with driver.session() as session:
        for query_name, query in SIMPLE_QUERIES:
            result = session.run(query)
            print(f"{query_name}: {result.data()}")
    print("-"*50)
    print("Vector similarity search...")
    verify_vector_search(driver)
    print("Vector search complete!\n")
    print("Verification complete!")


SIMPLE_QUERIES = [
    ("Top 5 movies by revenue", "MATCH (m:Movie) RETURN m.title, m.revenue ORDER BY m.revenue DESC LIMIT 5"),
    ("Most Connected Genres", "MATCH (m:Movie)-[:IN_GENRE]->(g:Genre) RETURN g.name, COUNT(m) AS movie_count ORDER BY movie_count DESC LIMIT 5"),
    ("Movies sharing the most keywords with 'Avatar'", "MATCH (m:Movie)-[:IN_KEYWORD]->(k:Keyword) WHERE k.name = 'Avatar' RETURN m.title, COUNT(k) AS keyword_count ORDER BY keyword_count DESC LIMIT 5"),
]


def verify_ingestion(driver):
    with driver.session() as session:
        print("-"*50)
        print("Verify Queries:")
        for query_name, query in VERIFY_QUERIES.items():
            result = session.run(query)
            count = result.single()["count"]
            print(f"{query_name}: {count}")
            print("-"*50)
        print("-"*50)
        print("Simple queries:")
        for query_name, query in SIMPLE_QUERIES:
            result = session.run(query)
            print(f"{query_name}: {result.data()}")
            
def main():

    # 1 Load CSV
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} movies")

    #2 clean rows
    print("Cleaning rows...")
    movies = [clean(row) for _, row in df.iterrows()]
    print(f"Cleaned {len(movies)} movies")

    #3 Connect to Neo4j
    print("Connecting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("Connected to Neo4j successfully")

    #4 Ingest movies
    print("Ingesting movies...")
    ingest(driver, movies)
    print("Ingestion complete!")

    # #5 Verify ingestion
    # print("Verifying ingestion...")
    # verify_ingestion(driver)
    # print("Verification complete!")

    # #6 Close driver
    # print("Closing driver...")
    # driver.close()
    # print("Driver closed successfully")

    # 5 Create + populate vector index
    print("Creating + populating vector index...")
    ensure_vector_index(driver)
    print("Populating vector index...")
    populate_movie_embeddings(driver, movies)
    print("Vector index populated successfully")
    print("Verification...")
    verify(driver)

    print("Closing driver...")
    driver.close()
    print("Driver closed successfully")


if __name__ == "__main__":
    main()
