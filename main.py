# Generated from main.ipynb

# ========================================================================
# # RAG pipeline
#
# Load documents, split them into chunks, and create embeddings.
# ========================================================================

# ========================================================================
# ## Imports
# ========================================================================

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

# ========================================================================
# ## Load data
#
# This function reads PDF, text, Markdown, JSON, and CSV files from the data folder.
# ========================================================================

def load_data(data_dir: str | Path = "data") -> list[dict[str, Any]]:
    """Load supported files into records ready for RAG chunking."""
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")

    records: list[dict[str, Any]] = []
    supported_suffixes = {".pdf", ".txt", ".md", ".json", ".csv"}

    for file_path in sorted(data_path.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in supported_suffixes:
            continue

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(file_path))
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    records.append({"text": text, "source": str(file_path), "page": page_number})
            continue

        if suffix in {".txt", ".md"}:
            text = file_path.read_text(encoding="utf-8")
        elif suffix == ".json":
            content = json.loads(file_path.read_text(encoding="utf-8"))
            text = json.dumps(content, ensure_ascii=False, indent=2)
        else:
            with file_path.open("r", encoding="utf-8", newline="") as csv_file:
                text = "\n".join(json.dumps(row, ensure_ascii=False) for row in csv.DictReader(csv_file))

        if text.strip():
            records.append({"text": text, "source": str(file_path)})

    return records

documents = load_data("data")
print(f"Loaded {len(documents)} records")

# ========================================================================
# ## Split into chunks
#
# The chunks overlap so that context is not lost at chunk boundaries.
# ========================================================================

def split_into_chunks(
    documents: list[dict[str, Any]],
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[dict[str, Any]]:
    """Split loaded documents into overlapping character chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be at least 0 and smaller than chunk_size")

    chunks: list[dict[str, Any]] = []
    step = chunk_size - overlap

    for document in documents:
        text = document.get("text", "").strip()
        if not text:
            continue

        for start in range(0, len(text), step):
            chunk_text = text[start : start + chunk_size]
            chunk = {key: value for key, value in document.items() if key != "text"}
            chunk.update({"text": chunk_text, "chunk_start": start, "chunk_end": start + len(chunk_text)})
            chunks.append(chunk)

            if start + chunk_size >= len(text):
                break

    return chunks

chunks = split_into_chunks(documents, chunk_size=1000, overlap=200)
print(f"Created {len(chunks)} chunks")

# ========================================================================
# ## Create embeddings
#
# Sentence Transformers converts each text chunk into a vector for semantic search.
# ========================================================================

from sentence_transformers import SentenceTransformer


def embed_chunks(
    chunks: list[dict[str, Any]],
    model_name: str = "all-MiniLM-L6-v2",
) -> list[dict[str, Any]]:
    """Create a normalized embedding vector for each text chunk."""
    if not chunks:
        return []

    model = SentenceTransformer(model_name)
    texts = [chunk["text"] for chunk in chunks]
    vectors = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    embedded_chunks = []
    for chunk, vector in zip(chunks, vectors):
        embedded_chunk = chunk.copy()
        embedded_chunk["embedding"] = vector.tolist()
        embedded_chunks.append(embedded_chunk)

    return embedded_chunks

embedded_chunks = embed_chunks(chunks)
print(f"Created {len(embedded_chunks)} embeddings")
print(f"Embedding dimensions: {len(embedded_chunks[0]['embedding'])}")

# ========================================================================
# ## Save embeddings in ChromaDB
#
# ChromaDB stores the vectors, text, and source metadata locally.
# ========================================================================

import chromadb

def save_embeddings(embedded_chunks, db_path="chroma_db"):
    client = chromadb.PersistentClient(path=db_path)

    collection = client.get_or_create_collection(
        name="rag_documents",
        metadata={"hnsw:space": "cosine"},
    )

    collection.upsert(
        ids=[f"chunk-{i}" for i in range(len(embedded_chunks))],
        embeddings=[chunk["embedding"] for chunk in embedded_chunks],
        documents=[chunk["text"] for chunk in embedded_chunks],
        metadatas=[
            {
                key: value
                for key, value in chunk.items()
                if key not in {"text", "embedding"}
            }
            for chunk in embedded_chunks
        ],
    )

    return collection

collection = save_embeddings(embedded_chunks)
print(f"Saved {collection.count()} embeddings")

# ========================================================================
# ## Retrieve relevant chunks
#
# The query is embedded with the same model and matched against the stored vectors.
# ========================================================================

def retrieve_chunks(
    query: str,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    n_results: int = 5,
) -> list[dict[str, Any]]:
    """Return the most similar stored chunks for a user query."""
    if not query.strip():
        raise ValueError("query cannot be empty")
    if n_results <= 0:
        raise ValueError("n_results must be greater than 0")

    query_embedding = model.encode([query], normalize_embeddings=True)[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    return [
        {
            "text": text,
            "metadata": metadata,
            "distance": distance,
        }
        for text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]

query_model = SentenceTransformer("all-MiniLM-L6-v2")
results = retrieve_chunks(
    "What is object-oriented programming?",
    collection,
    query_model,
    n_results=3,
)

for index, result in enumerate(results, start=1):
    print(f"\nResult {index} - {result['metadata']}")
    print(result["text"][:300])
