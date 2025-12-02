"""
Hybrid Search Module: Combines FAISS semantic search with SQLite FTS5 keyword search
Implements reciprocal rank fusion (RRF) and weighted score combination.
"""

import sqlite3
import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass

# Configuration
DATA_DIR = Path("data")
INDEX_DIR = DATA_DIR / "index"
DB_PATH = INDEX_DIR / "hybrid_index.db"
FAISS_INDEX_PATH = INDEX_DIR / "faiss_index.bin"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
METADATA_PATH = INDEX_DIR / "metadata.json"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'


@dataclass
class SearchResult:
    """A single search result."""
    chunk_id: int
    chunk_text: str
    paper_id: str
    paper_title: str
    chunk_index: int
    score: float
    rank: int
    source: str  # 'vector', 'keyword', or 'hybrid'


class HybridSearchEngine:
    """
    Hybrid search engine combining FAISS vector search with SQLite FTS5 keyword search.
    """

    def __init__(self, db_path: Path = DB_PATH,
                 faiss_index_path: Path = FAISS_INDEX_PATH,
                 chunks_path: Path = CHUNKS_PATH,
                 metadata_path: Path = METADATA_PATH):
        """
        Initialize the hybrid search engine.

        Args:
            db_path: Path to SQLite database
            faiss_index_path: Path to FAISS index
            chunks_path: Path to chunks JSON file
            metadata_path: Path to metadata JSON file
        """
        print("Initializing Hybrid Search Engine...")

        # Load SQLite database
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row

        # Load FAISS index
        self.faiss_index = faiss.read_index(str(faiss_index_path))
        print(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")

        # Load chunks and metadata
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)

        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        print(f"Loaded {len(self.chunks)} chunks")

        # Load embedding model
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"Loaded embedding model: {EMBEDDING_MODEL}")

        print("Hybrid Search Engine initialized successfully!")

    def vector_search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """
        Perform semantic vector search using FAISS.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of (chunk_index, distance) tuples
        """
        # Encode query
        query_embedding = self.model.encode([query])[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Search FAISS index
        query_vector = np.array([query_embedding]).astype('float32')
        distances, indices = self.faiss_index.search(query_vector, k)

        # Return results as (index, distance) tuples
        results = [(int(idx), float(dist)) for idx, dist in zip(indices[0], distances[0])]
        return results

    def keyword_search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """
        Perform keyword search using SQLite FTS5.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of (chunk_id, bm25_score) tuples
        """
        cursor = self.conn.cursor()

        # Use FTS5 MATCH for full-text search with BM25 ranking
        # The rank column is negative BM25 score (lower is better in FTS5)
        cursor.execute("""
            SELECT
                c.chunk_id,
                -rank as bm25_score,
                cfm.faiss_index
            FROM doc_chunks_fts f
            JOIN chunks c ON f.rowid = c.chunk_id
            JOIN chunk_faiss_mapping cfm ON c.chunk_id = cfm.chunk_id
            WHERE doc_chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, k))

        results = []
        for row in cursor.fetchall():
            chunk_id = row[0]
            bm25_score = row[1]
            faiss_index = row[2]
            results.append((faiss_index, bm25_score))

        return results

    def reciprocal_rank_fusion(self,
                               vector_results: List[Tuple[int, float]],
                               keyword_results: List[Tuple[int, float]],
                               k: int = 60) -> List[Tuple[int, float]]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).

        RRF score for document d: sum over all rankings R of 1/(k + rank_R(d))
        where k is typically 60.

        Args:
            vector_results: List of (chunk_index, score) from vector search
            keyword_results: List of (chunk_index, score) from keyword search
            k: Constant for RRF (default: 60)

        Returns:
            List of (chunk_index, rrf_score) tuples sorted by RRF score descending
        """
        rrf_scores = {}

        # Add vector search rankings
        for rank, (chunk_idx, _) in enumerate(vector_results, start=1):
            rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0) + 1 / (k + rank)

        # Add keyword search rankings
        for rank, (chunk_idx, _) in enumerate(keyword_results, start=1):
            rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0) + 1 / (k + rank)

        # Sort by RRF score descending
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    def weighted_fusion(self,
                       vector_results: List[Tuple[int, float]],
                       keyword_results: List[Tuple[int, float]],
                       alpha: float = 0.5) -> List[Tuple[int, float]]:
        """
        Combine results using weighted score fusion.

        Final score = alpha * normalized_vector_score + (1 - alpha) * normalized_keyword_score

        Args:
            vector_results: List of (chunk_index, distance) from vector search
            keyword_results: List of (chunk_index, bm25_score) from keyword search
            alpha: Weight for vector search (0-1, default: 0.5)

        Returns:
            List of (chunk_index, combined_score) tuples sorted by score descending
        """
        # Normalize vector distances to [0, 1] scores (lower distance = higher score)
        if vector_results:
            max_dist = max(dist for _, dist in vector_results)
            min_dist = min(dist for _, dist in vector_results)
            dist_range = max_dist - min_dist if max_dist > min_dist else 1.0

            vector_scores = {
                idx: 1 - (dist - min_dist) / dist_range
                for idx, dist in vector_results
            }
        else:
            vector_scores = {}

        # Normalize keyword BM25 scores to [0, 1]
        if keyword_results:
            max_score = max(score for _, score in keyword_results)
            min_score = min(score for _, score in keyword_results)
            score_range = max_score - min_score if max_score > min_score else 1.0

            keyword_scores = {
                idx: (score - min_score) / score_range if score_range > 0 else 1.0
                for idx, score in keyword_results
            }
        else:
            keyword_scores = {}

        # Combine scores
        all_indices = set(vector_scores.keys()) | set(keyword_scores.keys())
        combined_scores = {}

        for idx in all_indices:
            vec_score = vector_scores.get(idx, 0.0)
            key_score = keyword_scores.get(idx, 0.0)
            combined_scores[idx] = alpha * vec_score + (1 - alpha) * key_score

        # Sort by combined score descending
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    def hybrid_search(self,
                     query: str,
                     k: int = 3,
                     method: str = 'rrf',
                     alpha: float = 0.5,
                     k_retrieve: int = 20) -> List[SearchResult]:
        """
        Perform hybrid search combining vector and keyword search.

        Args:
            query: Search query
            k: Number of final results to return
            method: Fusion method ('rrf' or 'weighted')
            alpha: Weight for vector search if using weighted fusion
            k_retrieve: Number of results to retrieve from each search method

        Returns:
            List of SearchResult objects
        """
        # Perform vector search
        vector_results = self.vector_search(query, k=k_retrieve)

        # Perform keyword search
        keyword_results = self.keyword_search(query, k=k_retrieve)

        # Combine results
        if method == 'rrf':
            combined = self.reciprocal_rank_fusion(vector_results, keyword_results)
        elif method == 'weighted':
            combined = self.weighted_fusion(vector_results, keyword_results, alpha=alpha)
        else:
            raise ValueError(f"Unknown fusion method: {method}")

        # Take top k results
        top_results = combined[:k]

        # Build SearchResult objects
        results = []
        for rank, (chunk_idx, score) in enumerate(top_results, start=1):
            if chunk_idx < len(self.chunks) and chunk_idx < len(self.metadata):
                meta = self.metadata[chunk_idx]
                results.append(SearchResult(
                    chunk_id=chunk_idx,
                    chunk_text=self.chunks[chunk_idx],
                    paper_id=meta['paper_id'],
                    paper_title=meta['paper_title'],
                    chunk_index=meta['chunk_index'],
                    score=score,
                    rank=rank,
                    source='hybrid'
                ))

        return results

    def vector_only_search(self, query: str, k: int = 3) -> List[SearchResult]:
        """
        Perform vector-only search.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of SearchResult objects
        """
        vector_results = self.vector_search(query, k=k)

        results = []
        for rank, (chunk_idx, distance) in enumerate(vector_results, start=1):
            if chunk_idx < len(self.chunks) and chunk_idx < len(self.metadata):
                meta = self.metadata[chunk_idx]
                results.append(SearchResult(
                    chunk_id=chunk_idx,
                    chunk_text=self.chunks[chunk_idx],
                    paper_id=meta['paper_id'],
                    paper_title=meta['paper_title'],
                    chunk_index=meta['chunk_index'],
                    score=1.0 / (1.0 + distance),  # Convert distance to score
                    rank=rank,
                    source='vector'
                ))

        return results

    def keyword_only_search(self, query: str, k: int = 3) -> List[SearchResult]:
        """
        Perform keyword-only search.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of SearchResult objects
        """
        keyword_results = self.keyword_search(query, k=k)

        results = []
        for rank, (chunk_idx, bm25_score) in enumerate(keyword_results, start=1):
            if chunk_idx < len(self.chunks) and chunk_idx < len(self.metadata):
                meta = self.metadata[chunk_idx]
                results.append(SearchResult(
                    chunk_id=chunk_idx,
                    chunk_text=self.chunks[chunk_idx],
                    paper_id=meta['paper_id'],
                    paper_title=meta['paper_title'],
                    chunk_index=meta['chunk_index'],
                    score=bm25_score,
                    rank=rank,
                    source='keyword'
                ))

        return results

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __del__(self):
        """Cleanup."""
        self.close()


def demo_hybrid_search():
    """Demo function to test hybrid search."""
    print("=" * 70)
    print("Hybrid Search Demo")
    print("=" * 70)

    # Initialize search engine
    engine = HybridSearchEngine()

    # Test query
    query = "transformer attention mechanism"
    print(f"\nQuery: '{query}'")
    print("-" * 70)

    # Vector-only search
    print("\n1. Vector-only search:")
    vector_results = engine.vector_only_search(query, k=3)
    for i, result in enumerate(vector_results, 1):
        print(f"\n{i}. [{result.paper_title[:60]}...]")
        print(f"   Score: {result.score:.4f}")
        print(f"   Text: {result.chunk_text[:150]}...")

    # Keyword-only search
    print("\n2. Keyword-only search:")
    keyword_results = engine.keyword_only_search(query, k=3)
    for i, result in enumerate(keyword_results, 1):
        print(f"\n{i}. [{result.paper_title[:60]}...]")
        print(f"   Score: {result.score:.4f}")
        print(f"   Text: {result.chunk_text[:150]}...")

    # Hybrid search with RRF
    print("\n3. Hybrid search (RRF):")
    hybrid_results = engine.hybrid_search(query, k=3, method='rrf')
    for i, result in enumerate(hybrid_results, 1):
        print(f"\n{i}. [{result.paper_title[:60]}...]")
        print(f"   Score: {result.score:.4f}")
        print(f"   Text: {result.chunk_text[:150]}...")

    # Hybrid search with weighted fusion
    print("\n4. Hybrid search (Weighted, alpha=0.6):")
    hybrid_weighted = engine.hybrid_search(query, k=3, method='weighted', alpha=0.6)
    for i, result in enumerate(hybrid_weighted, 1):
        print(f"\n{i}. [{result.paper_title[:60]}...]")
        print(f"   Score: {result.score:.4f}")
        print(f"   Text: {result.chunk_text[:150]}...")

    engine.close()

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo_hybrid_search()
