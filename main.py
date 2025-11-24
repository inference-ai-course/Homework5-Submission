"""
FastAPI service for RAG-based paper search.
Provides a /search endpoint that accepts queries and returns relevant passages.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
from pydantic import BaseModel

# Configuration
INDEX_DIR = Path("data/index")
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# Initialize FastAPI app
app = FastAPI(
    title="arXiv Paper RAG Search",
    description="Retrieval-Augmented Generation system for arXiv cs.CL papers",
    version="1.0.0"
)

# Global variables for model and data
model = None
faiss_index = None
chunks = None
metadata = None


class SearchResult(BaseModel):
    """Schema for search results."""
    chunk_text: str
    paper_id: str
    paper_title: str
    chunk_index: int
    distance: float


class SearchResponse(BaseModel):
    """Schema for search response."""
    query: str
    num_results: int
    results: List[SearchResult]


def load_resources():
    """Load the FAISS index, chunks, and metadata."""
    global model, faiss_index, chunks, metadata

    print("Loading resources...")

    # Load embedding model
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"Loaded embedding model: {EMBEDDING_MODEL}")

    # Load FAISS index
    index_path = INDEX_DIR / "faiss_index.bin"
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    faiss_index = faiss.read_index(str(index_path))
    print(f"Loaded FAISS index with {faiss_index.ntotal} vectors")

    # Load chunks
    chunks_path = INDEX_DIR / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found at {chunks_path}")
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")

    # Load metadata
    metadata_path = INDEX_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f"Loaded metadata for {len(metadata)} chunks")

    print("Resources loaded successfully!")


@app.on_event("startup")
async def startup_event():
    """Load resources when the API starts."""
    try:
        load_resources()
    except Exception as e:
        print(f"Error loading resources: {e}")
        print("API will start but search functionality will not work.")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "arXiv Paper RAG Search API",
        "version": "1.0.0",
        "endpoints": {
            "/search": "Search for relevant paper passages",
            "/health": "Health check endpoint",
            "/stats": "Get statistics about the indexed data"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    is_ready = all([model is not None, faiss_index is not None,
                    chunks is not None, metadata is not None])

    return {
        "status": "healthy" if is_ready else "not ready",
        "model_loaded": model is not None,
        "index_loaded": faiss_index is not None,
        "chunks_loaded": chunks is not None,
        "metadata_loaded": metadata is not None
    }


@app.get("/stats")
async def get_stats():
    """Get statistics about the indexed data."""
    if not all([faiss_index, chunks, metadata]):
        raise HTTPException(status_code=503, detail="Resources not loaded")

    # Count unique papers
    unique_papers = len(set(m['paper_id'] for m in metadata))

    return {
        "total_chunks": len(chunks),
        "total_papers": unique_papers,
        "index_size": faiss_index.ntotal,
        "embedding_dimension": faiss_index.d
    }


@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="Search query", min_length=1),
    k: int = Query(3, description="Number of results to return", ge=1, le=20)
):
    """
    Search for relevant passages based on a query.

    Args:
        q: The search query string
        k: Number of top results to return (default: 3, max: 20)

    Returns:
        SearchResponse with query and list of matching passages
    """
    # Check if resources are loaded
    if not all([model, faiss_index, chunks, metadata]):
        raise HTTPException(
            status_code=503,
            detail="Search service not ready. Resources are still loading."
        )

    try:
        # Encode the query
        query_embedding = model.encode([q])[0]

        # Normalize the query embedding (if index was normalized)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Perform FAISS search
        query_vector = np.array([query_embedding]).astype('float32')
        distances, indices = faiss_index.search(query_vector, k)

        # Prepare results
        results = []
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx < len(chunks) and idx < len(metadata):
                results.append(SearchResult(
                    chunk_text=chunks[idx],
                    paper_id=metadata[idx]['paper_id'],
                    paper_title=metadata[idx]['paper_title'],
                    chunk_index=metadata[idx]['chunk_index'],
                    distance=float(distance)
                ))

        return SearchResponse(
            query=q,
            num_results=len(results),
            results=results
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.get("/paper/{paper_id}")
async def get_paper_chunks(paper_id: str):
    """
    Get all chunks for a specific paper.

    Args:
        paper_id: The arXiv paper ID

    Returns:
        List of chunks for the specified paper
    """
    if not all([chunks, metadata]):
        raise HTTPException(status_code=503, detail="Resources not loaded")

    # Find all chunks for this paper
    paper_chunks = []
    for i, meta in enumerate(metadata):
        if meta['paper_id'] == paper_id:
            paper_chunks.append({
                "chunk_index": meta['chunk_index'],
                "chunk_text": chunks[i],
                "paper_title": meta['paper_title']
            })

    if not paper_chunks:
        raise HTTPException(status_code=404, detail=f"Paper {paper_id} not found")

    # Sort by chunk index
    paper_chunks.sort(key=lambda x: x['chunk_index'])

    return {
        "paper_id": paper_id,
        "total_chunks": len(paper_chunks),
        "chunks": paper_chunks
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
