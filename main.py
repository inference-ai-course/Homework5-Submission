"""
FastAPI Service for Hybrid RAG Search
Provides REST API endpoints for vector search, keyword search, and hybrid search
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from hybrid_indexer import HybridIndexer

# Initialize FastAPI app
app = FastAPI(
    title="arXiv Hybrid RAG Search API",
    description="Search arXiv papers using semantic similarity, keywords, or hybrid search",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize indexer (will be loaded on startup)
indexer = None


# Response models
class SearchResult(BaseModel):
    rank: int
    distance: Optional[float] = None
    chunk_id: str
    paper_id: str
    text: str
    token_count: int
    hybrid_score: Optional[float] = None
    vector_score: Optional[float] = None
    keyword_score: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    num_results: int
    results: List[SearchResult]


class HybridSearchResponse(BaseModel):
    query: str
    num_results: int
    results: List[SearchResult]
    method: str
    alpha: Optional[float] = None


class IndexStats(BaseModel):
    total_chunks: int
    dimension: int
    total_papers: int
    avg_chunk_tokens: float
    documents_in_db: Optional[int] = None
    chunks_in_db: Optional[int] = None


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load the hybrid index on startup"""
    global indexer
    print("Loading Hybrid Index (FAISS + SQLite FTS5)...")
    
    indexer = HybridIndexer(model_name='all-MiniLM-L6-v2')
    
    try:
        indexer.load_index("faiss_index.bin", "index_metadata.pkl")
        print("✅ Hybrid index loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading index: {str(e)}")
        print("Please run hybrid_indexer.py first to build the index.")


# Routes
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "arXiv Hybrid RAG Search API",
        "version": "2.0.0",
        "endpoints": {
            "/search": "Vector search (GET with ?q=query&k=3)",
            "/keyword_search": "Keyword search using FTS5 (GET with ?q=query&k=3)",
            "/hybrid_search": "Hybrid search combining vector + keyword (GET with ?q=query&k=3&alpha=0.6)",
            "/stats": "Get index statistics",
            "/health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if indexer is None or indexer.index is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    return {
        "status": "healthy",
        "index_loaded": True,
        "total_chunks": indexer.index.ntotal,
        "database_connected": indexer.conn is not None
    }


@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="Search query", min_length=3),
    k: int = Query(3, description="Number of results to return", ge=1, le=20)
):
    """
    Vector search using FAISS (semantic similarity)
    
    Args:
        q: Search query string
        k: Number of results to return (default: 3, max: 20)
        
    Returns:
        SearchResponse with ranked results
    """
    if indexer is None or indexer.index is None:
        raise HTTPException(
            status_code=503, 
            detail="Index not loaded. Please build the index first."
        )
    
    try:
        # Perform vector search
        vector_results = indexer.vector_search(q, k=k)
        
        # Convert to response format
        results = []
        for rank, (chunk_id, distance) in enumerate(vector_results, 1):
            chunk_meta = next((m for m in indexer.metadata if m['chunk_id'] == chunk_id), None)
            if chunk_meta:
                results.append(SearchResult(
                    rank=rank,
                    distance=distance,
                    chunk_id=chunk_id,
                    paper_id=chunk_meta['paper_id'],
                    text=chunk_meta['text'],
                    token_count=chunk_meta.get('token_count', 0)
                ))
        
        return SearchResponse(
            query=q,
            num_results=len(results),
            results=results
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search error: {str(e)}"
        )


@app.get("/keyword_search", response_model=SearchResponse)
async def keyword_search(
    q: str = Query(..., description="Search query", min_length=3),
    k: int = Query(3, description="Number of results to return", ge=1, le=20)
):
    """
    Keyword search using SQLite FTS5
    
    Args:
        q: Search query string
        k: Number of results to return (default: 3, max: 20)
        
    Returns:
        SearchResponse with ranked results
    """
    if indexer is None or indexer.conn is None:
        raise HTTPException(
            status_code=503, 
            detail="Index not loaded. Please build the index first."
        )
    
    try:
        # Perform keyword search
        keyword_results = indexer.keyword_search(q, k=k)
        
        # Convert to response format
        results = []
        for rank, (chunk_id, score) in enumerate(keyword_results, 1):
            chunk_meta = next((m for m in indexer.metadata if m['chunk_id'] == chunk_id), None)
            if chunk_meta:
                results.append(SearchResult(
                    rank=rank,
                    distance=score,  # Using distance field for BM25 score
                    chunk_id=chunk_id,
                    paper_id=chunk_meta['paper_id'],
                    text=chunk_meta['text'],
                    token_count=chunk_meta.get('token_count', 0)
                ))
        
        return SearchResponse(
            query=q,
            num_results=len(results),
            results=results
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Keyword search error: {str(e)}"
        )


@app.get("/hybrid_search", response_model=HybridSearchResponse)
async def hybrid_search(
    q: str = Query(..., description="Search query", min_length=3),
    k: int = Query(3, description="Number of results to return", ge=1, le=20),
    alpha: float = Query(0.6, description="Weight for vector search (0-1)", ge=0.0, le=1.0),
    fusion_method: str = Query("weighted", description="Fusion method: 'weighted' or 'rrf'")
):
    """
    Hybrid search combining vector and keyword search
    
    Args:
        q: Search query string
        k: Number of results to return (default: 3, max: 20)
        alpha: Weight for vector search (0-1). 1.0 = vector only, 0.0 = keyword only
        fusion_method: 'weighted' (weighted sum) or 'rrf' (reciprocal rank fusion)
        
    Returns:
        HybridSearchResponse with ranked results and method info
    """
    if indexer is None or indexer.index is None or indexer.conn is None:
        raise HTTPException(
            status_code=503, 
            detail="Index not loaded. Please build the index first."
        )
    
    try:
        # Perform hybrid search
        results = indexer.hybrid_search(q, k=k, alpha=alpha, fusion_method=fusion_method)
        
        # Convert to response format
        search_results = []
        for r in results:
            search_results.append(SearchResult(
                rank=r['rank'],
                chunk_id=r['chunk_id'],
                paper_id=r['paper_id'],
                text=r['text'],
                token_count=r['token_count'],
                hybrid_score=r['hybrid_score'],
                vector_score=r['vector_score'],
                keyword_score=r['keyword_score']
            ))
        
        return HybridSearchResponse(
            query=q,
            num_results=len(search_results),
            results=search_results,
            method=fusion_method,
            alpha=alpha
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Hybrid search error: {str(e)}"
        )


@app.get("/stats", response_model=IndexStats)
async def get_stats():
    """Get statistics about the index"""
    if indexer is None or indexer.index is None:
        raise HTTPException(
            status_code=503,
            detail="Index not loaded"
        )
    
    stats = indexer.get_stats()
    return IndexStats(**stats)


@app.get("/paper/{paper_id}")
async def get_paper_chunks(paper_id: str):
    """Get all chunks for a specific paper"""
    if indexer is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    # Filter chunks by paper_id
    paper_chunks = [
        {
            'chunk_id': meta['chunk_id'],
            'chunk_index': meta['chunk_index'],
            'text': indexer.chunks[i],
            'token_count': meta.get('token_count', 0)
        }
        for i, meta in enumerate(indexer.metadata)
        if meta['paper_id'] == paper_id
    ]
    
    if not paper_chunks:
        raise HTTPException(
            status_code=404,
            detail=f"No chunks found for paper_id: {paper_id}"
        )
    
    return {
        'paper_id': paper_id,
        'num_chunks': len(paper_chunks),
        'chunks': paper_chunks
    }


def main():
    """Run the FastAPI server"""
    print("=" * 60)
    print("Starting arXiv Hybrid RAG Search API")
    print("=" * 60)
    print("\nMake sure you have:")
    print("1. Processed PDFs (run pdf_processor_pypdf.py)")
    print("2. Built hybrid index (run hybrid_indexer.py)")
    print("\nAPI will be available at: http://localhost:8000")
    print("Interactive docs at: http://localhost:8000/docs")
    print("=" * 60)
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

