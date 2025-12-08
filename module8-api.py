# modules/m8_api_service/__init__.py
"""Module 8: FastAPI Service."""

from .main import create_app

__all__ = ["create_app"]


# modules/m8_api_service/schemas.py
"""Pydantic schemas for API."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum


class ModelType(str, Enum):
    BASE = "base"
    FINETUNED = "finetuned"


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    search_type: str = Field(default="hybrid", pattern="^(vector|keyword|hybrid)$")


class SearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: Dict = {}


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    search_type: str
    total_results: int


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    model_type: ModelType = ModelType.FINETUNED
    use_rag: bool = True
    max_tokens: int = Field(default=512, ge=64, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    message: str
    model_used: str
    sources: List[SearchResult] = []
    latency_ms: float


class EvaluateRequest(BaseModel):
    questions: List[str]
    expected_answers: Optional[List[str]] = None


class ComparisonResponse(BaseModel):
    question: str
    base_answer: str
    finetuned_answer: str
    base_score: float
    finetuned_score: float
    winner: str


class PipelineStatus(BaseModel):
    status: str
    papers_collected: int = 0
    chunks_indexed: int = 0
    qa_pairs_generated: int = 0
    model_trained: bool = False
    last_updated: Optional[str] = None


# modules/m8_api_service/main.py
"""FastAPI application."""

import time
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import get_config, INDEX_DIR
from .schemas import (
    SearchRequest, SearchResponse, SearchResult,
    ChatRequest, ChatResponse,
    ComparisonResponse, PipelineStatus, ModelType
)


# Global state
class AppState:
    def __init__(self):
        self.config = get_config()
        self.embedder = None
        self.faiss_indexer = None
        self.sqlite_fts = None
        self.hybrid_retriever = None
        self.base_model = None
        self.base_tokenizer = None
        self.ft_model = None
        self.ft_tokenizer = None
        self.llm_loader = None
        self.initialized = False


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting Academic LLM API...")
    
    # Initialize components
    try:
        from modules.m3_rag_pipeline import EmbeddingGenerator, FAISSIndexer
        from modules.m4_hybrid_retrieval import SQLiteFTS, HybridRetriever
        from modules.m1_langchain_llama import LLMLoader
        
        # Load embedder
        state.embedder = EmbeddingGenerator()
        state.embedder.load_model()
        
        # Load FAISS index if exists
        faiss_path = INDEX_DIR / "faiss" / "academic_index.faiss"
        if faiss_path.exists():
            state.faiss_indexer = FAISSIndexer(state.embedder.get_dimension())
            state.faiss_indexer.load("academic_index")
            logger.info("Loaded FAISS index")
        
        # Load SQLite FTS
        state.sqlite_fts = SQLiteFTS()
        state.sqlite_fts.connect()
        
        # Setup hybrid retriever if both are available
        if state.faiss_indexer:
            state.hybrid_retriever = HybridRetriever(
                state.faiss_indexer,
                state.sqlite_fts,
                state.embedder
            )
        
        # Load LLM (lazy - only when needed)
        state.llm_loader = LLMLoader()
        
        state.initialized = True
        logger.info("API initialized successfully")
        
    except Exception as e:
        logger.error(f"Initialization error: {e}")
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")
    if state.sqlite_fts:
        state.sqlite_fts.close()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    
    app = FastAPI(
        title="Academic LLM API",
        description="RAG-enhanced Academic Q&A with Fine-tuned LLaMA",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Health check
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "initialized": state.initialized,
            "index_loaded": state.faiss_indexer is not None
        }
    
    # Search endpoint
    @app.post("/search", response_model=SearchResponse)
    async def search(request: SearchRequest):
        if not state.hybrid_retriever:
            raise HTTPException(503, "Search index not initialized")
        
        try:
            if request.search_type == "hybrid":
                results = state.hybrid_retriever.search(
                    request.query, 
                    top_k=request.top_k
                )
            elif request.search_type == "vector":
                query_emb = state.embedder.embed_text(request.query)
                results_raw = state.faiss_indexer.search(query_emb, request.top_k)
                results = [
                    SearchResult(
                        chunk_id=c.chunk_id,
                        doc_id=c.doc_id,
                        text=c.text,
                        score=s,
                        metadata=c.metadata
                    ) for c, s in results_raw
                ]
            else:  # keyword
                results_raw = state.sqlite_fts.search(request.query, request.top_k)
                results = [
                    SearchResult(
                        chunk_id=r["chunk_id"],
                        doc_id=r["doc_id"],
                        text=r["text"],
                        score=r["score"],
                        metadata={}
                    ) for r in results_raw
                ]
            
            # Convert to response format
            search_results = []
            for r in results:
                if hasattr(r, 'chunk_id'):
                    search_results.append(SearchResult(
                        chunk_id=r.chunk_id,
                        doc_id=r.doc_id,
                        text=r.text[:500],
                        score=r.score,
                        metadata=getattr(r, 'metadata', {})
                    ))
            
            return SearchResponse(
                query=request.query,
                results=search_results,
                search_type=request.search_type,
                total_results=len(search_results)
            )
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise HTTPException(500, str(e))
    
    # Chat endpoint
    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        start_time = time.time()
        
        try:
            # Load model if needed
            if request.model_type == ModelType.FINETUNED:
                if state.ft_model is None:
                    state.ft_model, state.ft_tokenizer = state.llm_loader.load_finetuned_model()
                model, tokenizer = state.ft_model, state.ft_tokenizer
                model_name = "finetuned"
            else:
                if state.base_model is None:
                    state.base_model, state.base_tokenizer = state.llm_loader.load_base_model()
                model, tokenizer = state.base_model, state.base_tokenizer
                model_name = "base"
            
            # Get context from RAG if enabled
            sources = []
            context = ""
            if request.use_rag and state.hybrid_retriever:
                results = state.hybrid_retriever.search(request.message, top_k=3)
                context = "\n\n".join([r.text for r in results[:3]])
                sources = [
                    SearchResult(
                        chunk_id=r.chunk_id,
                        doc_id=r.doc_id,
                        text=r.text[:200],
                        score=r.score,
                        metadata={}
                    ) for r in results[:3]
                ]
            
            # Format prompt
            if context:
                user_message = f"""Based on the following research context, answer the question.

Context:
{context}

Question: {request.message}"""
            else:
                user_message = request.message
            
            prompt = state.llm_loader.format_prompt(user_message)
            
            # Generate
            response = state.llm_loader.generate(
                prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            latency = (time.time() - start_time) * 1000
            
            return ChatResponse(
                message=response,
                model_used=model_name,
                sources=sources,
                latency_ms=latency
            )
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise HTTPException(500, str(e))
    
    # Compare models endpoint
    @app.post("/compare")
    async def compare_models(question: str):
        try:
            # Ensure both models loaded
            if state.base_model is None:
                state.base_model, state.base_tokenizer = state.llm_loader.load_base_model()
            
            # Load fresh base for comparison
            from modules.m1_langchain_llama import LLMLoader
            base_loader = LLMLoader()
            base_model, base_tok = base_loader.load_base_model()
            
            # Get fine-tuned response
            ft_loader = LLMLoader()
            ft_model, ft_tok = ft_loader.load_finetuned_model()
            
            prompt = state.llm_loader.format_prompt(question)
            
            base_response = base_loader.generate(prompt)
            ft_response = ft_loader.generate(prompt)
            
            return {
                "question": question,
                "base_response": base_response,
                "finetuned_response": ft_response
            }
            
        except Exception as e:
            logger.error(f"Compare error: {e}")
            raise HTTPException(500, str(e))
    
    # Pipeline status
    @app.get("/status", response_model=PipelineStatus)
    async def get_status():
        return PipelineStatus(
            status="ready" if state.initialized else "initializing",
            chunks_indexed=state.faiss_indexer.index.ntotal if state.faiss_indexer else 0,
            model_trained=Path(state.config.training.output_dir).exists()
        )
    
    return app


# Run with: uvicorn modules.m8_api_service.main:app --host 0.0.0.0 --port 8000
app = create_app()


if __name__ == "__main__":
    import uvicorn
    config = get_config()
    uvicorn.run(app, host=config.api.host, port=config.api.port)
