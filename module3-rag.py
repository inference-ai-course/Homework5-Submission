# modules/m3_rag_pipeline/__init__.py
"""Module 3: RAG Pipeline - Chunking, Embedding, and Indexing."""

from .chunker import DocumentChunker, Chunk
from .embedder import EmbeddingGenerator
from .faiss_indexer import FAISSIndexer

__all__ = ["DocumentChunker", "Chunk", "EmbeddingGenerator", "FAISSIndexer"]


# modules/m3_rag_pipeline/chunker.py
"""Document chunking strategies."""

from typing import List, Optional, Dict
from dataclasses import dataclass
import re
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import get_config


@dataclass
class Chunk:
    """A text chunk with metadata."""
    chunk_id: str
    doc_id: str
    text: str
    start_idx: int
    end_idx: int
    metadata: Dict
    token_count: int = 0


class DocumentChunker:
    """Splits documents into chunks for embedding."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.chunk_size = self.config.data.chunk_size
        self.chunk_overlap = self.config.data.chunk_overlap
        self.min_chunk_length = self.config.data.min_chunk_length
        
    def chunk_document(
        self, 
        doc_id: str,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        """Chunk a single document using sliding window."""
        metadata = metadata or {}
        chunks = []
        
        # Estimate tokens (rough approximation: 1 token ≈ 4 chars)
        char_chunk_size = self.chunk_size * 4
        char_overlap = self.chunk_overlap * 4
        
        # Split into sentences first for cleaner boundaries
        sentences = self._split_sentences(text)
        
        current_chunk = []
        current_length = 0
        chunk_start = 0
        chunk_idx = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            if current_length + sentence_len > char_chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= self.min_chunk_length:
                    chunk = Chunk(
                        chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                        doc_id=doc_id,
                        text=chunk_text,
                        start_idx=chunk_start,
                        end_idx=chunk_start + len(chunk_text),
                        metadata=metadata.copy(),
                        token_count=len(chunk_text) // 4
                    )
                    chunks.append(chunk)
                    chunk_idx += 1
                
                # Start new chunk with overlap
                overlap_chars = 0
                overlap_sentences = []
                for s in reversed(current_chunk):
                    if overlap_chars + len(s) < char_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_chars += len(s)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_chars
                chunk_start = chunk_start + len(chunk_text) - overlap_chars
            
            current_chunk.append(sentence)
            current_length += sentence_len
        
        # Don't forget last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_length:
                chunk = Chunk(
                    chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                    doc_id=doc_id,
                    text=chunk_text,
                    start_idx=chunk_start,
                    end_idx=chunk_start + len(chunk_text),
                    metadata=metadata.copy(),
                    token_count=len(chunk_text) // 4
                )
                chunks.append(chunk)
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_by_sections(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        """Chunk by detecting section headers."""
        metadata = metadata or {}
        chunks = []
        
        # Common section patterns in academic papers
        section_pattern = r'\n(?=(?:\d+\.?\s+)?(?:Abstract|Introduction|Related Work|Background|Method|Results|Discussion|Conclusion|References))'
        
        sections = re.split(section_pattern, text, flags=re.IGNORECASE)
        
        for idx, section in enumerate(sections):
            section = section.strip()
            if len(section) < self.min_chunk_length:
                continue
            
            # Extract section title if present
            lines = section.split('\n')
            section_title = lines[0][:100] if lines else f"Section {idx}"
            
            # If section is too long, sub-chunk it
            if len(section) > self.chunk_size * 4:
                sub_chunks = self.chunk_document(
                    doc_id=doc_id,
                    text=section,
                    metadata={**metadata, "section": section_title}
                )
                # Update chunk IDs
                for i, sc in enumerate(sub_chunks):
                    sc.chunk_id = f"{doc_id}_sec{idx}_chunk_{i}"
                chunks.extend(sub_chunks)
            else:
                chunk = Chunk(
                    chunk_id=f"{doc_id}_sec{idx}",
                    doc_id=doc_id,
                    text=section,
                    start_idx=0,
                    end_idx=len(section),
                    metadata={**metadata, "section": section_title},
                    token_count=len(section) // 4
                )
                chunks.append(chunk)
        
        return chunks
    
    def chunk_batch(
        self,
        documents: List[Dict],
        by_sections: bool = False
    ) -> List[Chunk]:
        """Chunk multiple documents."""
        all_chunks = []
        
        for doc in documents:
            doc_id = doc.get("arxiv_id", doc.get("id", "unknown"))
            text = doc.get("full_text", doc.get("text", ""))
            metadata = {
                "title": doc.get("title", ""),
                "authors": doc.get("authors", []),
                "arxiv_id": doc_id
            }
            
            if by_sections:
                chunks = self.chunk_by_sections(doc_id, text, metadata)
            else:
                chunks = self.chunk_document(doc_id, text, metadata)
            
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks


# modules/m3_rag_pipeline/embedder.py
"""Embedding generation using sentence-transformers."""

import numpy as np
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import get_config


class EmbeddingGenerator:
    """Generates embeddings for text chunks."""
    
    def __init__(self, model_name: Optional[str] = None, config=None):
        self.config = config or get_config()
        self.model_name = model_name or self.config.model.embedding_model
        self.model = None
        self.embedding_dim = None
        
    def load_model(self):
        """Load the embedding model."""
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text."""
        if self.model is None:
            self.load_model()
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(
        self, 
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """Embed multiple texts."""
        if self.model is None:
            self.load_model()
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def embed_chunks(
        self,
        chunks: List,  # List[Chunk]
        batch_size: int = 32
    ) -> np.ndarray:
        """Embed chunk objects."""
        texts = [chunk.text for chunk in chunks]
        return self.embed_batch(texts, batch_size)
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        if self.embedding_dim is None:
            self.load_model()
        return self.embedding_dim


# modules/m3_rag_pipeline/faiss_indexer.py
"""FAISS index management for vector search."""

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import get_config, INDEX_DIR
from .chunker import Chunk


class FAISSIndexer:
    """Manages FAISS index for vector similarity search."""
    
    def __init__(self, embedding_dim: int, config=None):
        self.config = config or get_config()
        self.embedding_dim = embedding_dim
        self.index = None
        self.chunks: List[Chunk] = []
        self.id_to_chunk: Dict[str, Chunk] = {}
        self.index_dir = INDEX_DIR / "faiss"
        
    def create_index(self, index_type: Optional[str] = None):
        """Create a new FAISS index."""
        index_type = index_type or self.config.rag.faiss_index_type
        
        if index_type == "IndexFlatIP":
            # Inner product (for normalized vectors = cosine similarity)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif index_type == "IndexFlatL2":
            # L2 distance
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        elif index_type == "IndexIVFFlat":
            # IVF index for larger datasets
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        logger.info(f"Created FAISS index: {index_type}")
        
    def add_vectors(
        self,
        embeddings: np.ndarray,
        chunks: List[Chunk],
        normalize: bool = True
    ):
        """Add vectors to the index."""
        if self.index is None:
            self.create_index()
        
        # Normalize for cosine similarity
        if normalize:
            faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store chunk mappings
        for chunk in chunks:
            self.chunks.append(chunk)
            self.id_to_chunk[chunk.chunk_id] = chunk
        
        logger.info(f"Added {len(chunks)} vectors to index (total: {self.index.ntotal})")
        
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: Optional[int] = None,
        normalize: bool = True
    ) -> List[Tuple[Chunk, float]]:
        """Search for similar chunks."""
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("Index is empty. Add vectors first.")
        
        top_k = top_k or self.config.rag.top_k_retrieval
        
        # Ensure 2D array
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query
        if normalize:
            faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Map results to chunks
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append((chunk, float(score)))
        
        return results
    
    def save(self, name: str = "academic_index"):
        """Save index and chunks to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = self.index_dir / f"{name}.faiss"
        faiss.write_index(self.index, str(index_path))
        
        # Save chunks
        chunks_path = self.index_dir / f"{name}_chunks.pkl"
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        logger.info(f"Saved index to {self.index_dir}")
        
    def load(self, name: str = "academic_index"):
        """Load index and chunks from disk."""
        # Load FAISS index
        index_path = self.index_dir / f"{name}.faiss"
        self.index = faiss.read_index(str(index_path))
        
        # Load chunks
        chunks_path = self.index_dir / f"{name}_chunks.pkl"
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        
        # Rebuild mapping
        self.id_to_chunk = {chunk.chunk_id: chunk for chunk in self.chunks}
        
        logger.info(f"Loaded index with {self.index.ntotal} vectors")
        
    def get_stats(self) -> Dict:
        """Get index statistics."""
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "embedding_dim": self.embedding_dim,
            "num_chunks": len(self.chunks),
            "index_type": type(self.index).__name__ if self.index else None
        }
