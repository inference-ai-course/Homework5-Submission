"""
Hybrid Indexer: Combines FAISS (vector search) with SQLite FTS5 (keyword search)
Extends the Week 4 EmbeddingIndexer to add metadata storage and hybrid retrieval
"""

import numpy as np
import faiss
import pickle
import json
import sqlite3
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from pathlib import Path
from collections import defaultdict


class HybridIndexer:
    """
    Hybrid retrieval system combining:
    - FAISS for dense vector (semantic) search
    - SQLite FTS5 for sparse keyword search
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', db_path: str = "hybrid_index.db"):
        """
        Initialize the hybrid indexer
        
        Args:
            model_name: Sentence-transformers model name
            db_path: Path to SQLite database
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = []
        self.metadata = []
        self.db_path = db_path
        self.conn = None
        
        print(f"Model loaded. Embedding dimension: {self.dimension}")
    
    def _init_database(self):
        """Initialize SQLite database with documents and FTS5 tables"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        
        # Create documents table for metadata
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT UNIQUE NOT NULL,
                title TEXT,
                author TEXT,
                year INTEGER,
                keywords TEXT,
                abstract TEXT,
                url TEXT
            )
        """)
        
        # Create FTS5 virtual table for full-text search on chunk content
        # Store chunk_id and paper_id as UNINDEXED columns for retrieval
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks USING fts5(
                chunk_id UNINDEXED,
                paper_id UNINDEXED,
                content
            )
        """)
        
        # Create mapping table to link chunk_id to FAISS index position
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chunk_mapping (
                chunk_id TEXT PRIMARY KEY,
                faiss_idx INTEGER NOT NULL,
                paper_id TEXT NOT NULL,
                chunk_index INTEGER
            )
        """)
        
        self.conn.commit()
        print(f"Database initialized: {self.db_path}")
    
    def _load_metadata_from_json(self, metadata_file: str = "arxiv_clean.json") -> Dict[str, Dict]:
        """
        Load paper metadata from arxiv_clean.json if available
        
        Returns:
            Dictionary mapping paper_id to metadata
        """
        metadata_dict = {}
        if Path(metadata_file).exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    papers = json.load(f)
                    for paper in papers:
                        paper_id = paper.get('paper_id', '')
                        if paper_id:
                            metadata_dict[paper_id] = {
                                'title': paper.get('title', ''),
                                'authors': paper.get('authors', []),
                                'abstract': paper.get('abstract', ''),
                                'url': paper.get('url', ''),
                                'date': paper.get('date', '')
                            }
                print(f"Loaded metadata for {len(metadata_dict)} papers from {metadata_file}")
            except Exception as e:
                print(f"Warning: Could not load metadata from {metadata_file}: {e}")
        return metadata_dict
    
    def build_index(self, chunks_data: List[Dict], metadata_file: Optional[str] = None) -> faiss.Index:
        """
        Build hybrid index (FAISS + SQLite FTS5) from chunk data
        
        Args:
            chunks_data: List of chunk dictionaries with 'text', 'paper_id', 'chunk_id'
            metadata_file: Optional path to arxiv_clean.json for paper metadata
            
        Returns:
            FAISS index
        """
        # Initialize database
        self._init_database()
        
        # Load paper metadata if available
        paper_metadata = {}
        if metadata_file:
            paper_metadata = self._load_metadata_from_json(metadata_file)
        elif Path("arxiv_clean.json").exists():
            paper_metadata = self._load_metadata_from_json("arxiv_clean.json")
        
        # Extract texts and prepare data
        texts = [chunk['text'] for chunk in chunks_data]
        self.chunks = texts
        self.metadata = chunks_data
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Create FAISS index
        print(f"Building FAISS index (dimension: {self.dimension})...")
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        print(f"FAISS index built with {self.index.ntotal} vectors")
        
        # Store in SQLite: documents and chunks
        print("Storing data in SQLite database...")
        
        # Group chunks by paper_id to insert documents once per paper
        papers_seen = set()
        
        for i, chunk in enumerate(chunks_data):
            paper_id = chunk['paper_id']
            
            # Insert document metadata (once per paper)
            if paper_id not in papers_seen:
                meta = paper_metadata.get(paper_id, {})
                authors = meta.get('authors', [])
                author_str = ', '.join(authors) if isinstance(authors, list) else str(authors)
                
                # Extract year from date if available
                year = None
                date_str = meta.get('date', '')
                if date_str and len(date_str) >= 4:
                    try:
                        year = int(date_str[:4])
                    except:
                        pass
                
                self.conn.execute("""
                    INSERT OR REPLACE INTO documents 
                    (paper_id, title, author, year, keywords, abstract, url)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    paper_id,
                    meta.get('title', ''),
                    author_str,
                    year,
                    '',  # keywords - could be extracted from abstract
                    meta.get('abstract', ''),
                    meta.get('url', '')
                ))
                papers_seen.add(paper_id)
            
            # Insert chunk into FTS5 table
            chunk_id = chunk['chunk_id']
            self.conn.execute("""
                INSERT INTO doc_chunks (chunk_id, paper_id, content)
                VALUES (?, ?, ?)
            """, (chunk_id, paper_id, chunk['text']))
            
            # Store mapping from chunk_id to FAISS index position
            self.conn.execute("""
                INSERT OR REPLACE INTO chunk_mapping 
                (chunk_id, faiss_idx, paper_id, chunk_index)
                VALUES (?, ?, ?, ?)
            """, (
                chunk_id,
                i,
                paper_id,
                chunk.get('chunk_index', 0)
            ))
        
        self.conn.commit()
        print(f"Stored {len(chunks_data)} chunks in SQLite")
        print(f"Stored {len(papers_seen)} documents with metadata")
        
        return self.index
    
    def vector_search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Perform vector search using FAISS
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of (chunk_id, distance) tuples, sorted by distance (ascending)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Embed the query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Search FAISS
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Map FAISS indices to chunk_ids
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                chunk_id = self.metadata[idx]['chunk_id']
                results.append((chunk_id, float(dist)))
        
        return results
    
    def keyword_search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Perform keyword search using SQLite FTS5
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of (chunk_id, bm25_score) tuples, sorted by score (descending)
        """
        if self.conn is None:
            raise ValueError("Database not initialized. Call build_index() first.")
        
        # Sanitize query for FTS5
        # Replace hyphens with spaces and escape quotes
        sanitized_query = query.replace('-', ' ').replace('"', '""')
        
        # FTS5 MATCH query with BM25 ranking
        # Note: FTS5 provides bm25() function for ranking
        try:
            cursor = self.conn.execute("""
                SELECT chunk_id, bm25(doc_chunks) as score
                FROM doc_chunks
                WHERE doc_chunks MATCH ?
                ORDER BY score
                LIMIT ?
            """, (sanitized_query, k))
            
            results = [(row[0], float(row[1])) for row in cursor.fetchall()]
            
            # BM25 scores are negative (lower is better), so we negate them
            # to make higher scores better (for consistency with distance)
            results = [(chunk_id, -score) for chunk_id, score in results]
            
            return results
        except sqlite3.OperationalError as e:
            # If query still fails, try with individual words
            print(f"Warning: FTS5 query failed for '{query}': {e}")
            print(f"Falling back to simple word search...")
            
            # Split into words and search for each
            words = sanitized_query.split()
            if not words:
                return []
            
            # Use OR to match any word
            simple_query = ' OR '.join(words)
            
            try:
                cursor = self.conn.execute("""
                    SELECT chunk_id, bm25(doc_chunks) as score
                    FROM doc_chunks
                    WHERE doc_chunks MATCH ?
                    ORDER BY score
                    LIMIT ?
                """, (simple_query, k))
                
                results = [(row[0], float(row[1])) for row in cursor.fetchall()]
                results = [(chunk_id, -score) for chunk_id, score in results]
                
                return results
            except sqlite3.OperationalError as e2:
                print(f"Error: Keyword search failed completely: {e2}")
                return []
    
    def hybrid_search(self, query: str, k: int = 3, alpha: float = 0.6, 
                     fusion_method: str = "weighted") -> List[Dict]:
        """
        Perform hybrid search combining vector and keyword results
        
        Args:
            query: Search query string
            k: Number of final results to return
            alpha: Weight for vector search (0-1). 1.0 = vector only, 0.0 = keyword only
            fusion_method: 'weighted' or 'rrf' (reciprocal rank fusion)
            
        Returns:
            List of result dictionaries with combined scores
        """
        # Get results from both methods
        vector_results = self.vector_search(query, k=k*2)  # Get more candidates
        keyword_results = self.keyword_search(query, k=k*2)
        
        # Normalize scores
        def normalize_scores(results, is_distance=True):
            """Normalize scores to [0, 1] range"""
            if not results:
                return {}
            
            scores = [score for _, score in results]
            if is_distance:
                # For distances: lower is better, normalize to similarity
                min_score, max_score = min(scores), max(scores)
                if max_score == min_score:
                    return {chunk_id: 1.0 for chunk_id, _ in results}
                return {
                    chunk_id: 1.0 - (score - min_score) / (max_score - min_score)
                    for chunk_id, score in results
                }
            else:
                # For BM25: higher is better
                min_score, max_score = min(scores), max(scores)
                if max_score == min_score:
                    return {chunk_id: 1.0 for chunk_id, _ in results}
                return {
                    chunk_id: (score - min_score) / (max_score - min_score)
                    for chunk_id, score in results
                }
        
        vector_scores = normalize_scores(vector_results, is_distance=True)
        keyword_scores = normalize_scores(keyword_results, is_distance=False)
        
        # Combine scores
        if fusion_method == "weighted":
            # Weighted sum
            combined_scores = {}
            all_chunks = set(vector_scores.keys()) | set(keyword_scores.keys())
            
            for chunk_id in all_chunks:
                v_score = vector_scores.get(chunk_id, 0.0)
                k_score = keyword_scores.get(chunk_id, 0.0)
                combined_scores[chunk_id] = alpha * v_score + (1 - alpha) * k_score
        
        elif fusion_method == "rrf":
            # Reciprocal Rank Fusion
            # RRF score = sum(1 / (k + rank)) for each method
            k_rrf = 60  # RRF constant
            
            # Build rank dictionaries
            vector_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(vector_results, 1)}
            keyword_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(keyword_results, 1)}
            
            combined_scores = {}
            all_chunks = set(vector_ranks.keys()) | set(keyword_ranks.keys())
            
            for chunk_id in all_chunks:
                rrf_score = 0.0
                if chunk_id in vector_ranks:
                    rrf_score += 1.0 / (k_rrf + vector_ranks[chunk_id])
                if chunk_id in keyword_ranks:
                    rrf_score += 1.0 / (k_rrf + keyword_ranks[chunk_id])
                combined_scores[chunk_id] = rrf_score
        
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Sort by combined score and get top-k
        sorted_chunks = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # Build result dictionaries
        results = []
        for rank, (chunk_id, score) in enumerate(sorted_chunks, 1):
            # Find chunk metadata
            chunk_meta = next((m for m in self.metadata if m['chunk_id'] == chunk_id), None)
            if chunk_meta:
                result = {
                    'rank': rank,
                    'chunk_id': chunk_id,
                    'paper_id': chunk_meta['paper_id'],
                    'text': chunk_meta['text'],
                    'hybrid_score': score,
                    'vector_score': vector_scores.get(chunk_id, 0.0),
                    'keyword_score': keyword_scores.get(chunk_id, 0.0),
                    'token_count': chunk_meta.get('token_count', 0)
                }
                results.append(result)
        
        return results
    
    def save_index(self, index_path: str = "faiss_index.bin", 
                   metadata_path: str = "index_metadata.pkl"):
        """Save FAISS index and metadata (database is saved automatically)"""
        if self.index is None:
            raise ValueError("No index to save. Build index first.")
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        print(f"FAISS index saved to: {index_path}")
        
        # Save metadata
        metadata_dict = {
            'chunks': self.chunks,
            'metadata': self.metadata,
            'dimension': self.dimension
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata_dict, f)
        print(f"Metadata saved to: {metadata_path}")
        print(f"Database saved to: {self.db_path}")
    
    def load_index(self, index_path: str = "faiss_index.bin",
                   metadata_path: str = "index_metadata.pkl"):
        """Load FAISS index and metadata, connect to database"""
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        print(f"FAISS index loaded from: {index_path}")
        print(f"Index contains {self.index.ntotal} vectors")
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata_dict = pickle.load(f)
        
        self.chunks = metadata_dict['chunks']
        self.metadata = metadata_dict['metadata']
        self.dimension = metadata_dict['dimension']
        print(f"Metadata loaded from: {metadata_path}")
        
        # Connect to database
        self.conn = sqlite3.connect(self.db_path)
        print(f"Database connected: {self.db_path}")
    
    def get_stats(self) -> Dict:
        """Get statistics about the index"""
        if self.index is None:
            return {"error": "No index built"}
        
        stats = {
            'total_chunks': self.index.ntotal,
            'dimension': self.dimension,
            'total_papers': len(set(m['paper_id'] for m in self.metadata)),
            'avg_chunk_tokens': np.mean([m.get('token_count', 0) for m in self.metadata])
        }
        
        if self.conn:
            try:
                cursor = self.conn.execute("SELECT COUNT(*) FROM documents")
                stats['documents_in_db'] = cursor.fetchone()[0]
            except Exception as e:
                stats['documents_in_db'] = 0
                print(f"Warning: Could not count documents: {e}")
            
            try:
                # Count from chunk_mapping table instead of FTS5 table
                # (FTS5 requires MATCH clause which complicates COUNT queries)
                cursor = self.conn.execute("SELECT COUNT(*) FROM chunk_mapping")
                stats['chunks_in_db'] = cursor.fetchone()[0]
            except Exception as e:
                # Fallback to metadata length if table doesn't exist
                stats['chunks_in_db'] = len(self.metadata)
                print(f"Warning: Could not count chunks from DB: {e}, using metadata length")
        
        return stats


def main():
    """Example usage"""
    # Initialize hybrid indexer
    indexer = HybridIndexer(model_name='all-MiniLM-L6-v2')
    
    # Load processed chunks
    with open('processed_chunks.json', 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    print(f"Loaded {len(chunks_data)} chunks")
    
    # Build hybrid index
    indexer.build_index(chunks_data, metadata_file="arxiv_clean.json")
    
    # Save index
    indexer.save_index("faiss_index.bin", "index_metadata.pkl")
    
    # Test searches
    print("\n" + "=" * 60)
    print("Testing Hybrid Search")
    print("=" * 60)
    
    test_queries = [
        "attention mechanisms transformers",
        "reinforcement learning human feedback",
        "language models few-shot learning"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        
        # Vector search
        vector_results = indexer.vector_search(query, k=3)
        print(f"Vector search top result: {vector_results[0][0]} (distance={vector_results[0][1]:.4f})")
        
        # Keyword search
        keyword_results = indexer.keyword_search(query, k=3)
        if keyword_results:
            print(f"Keyword search top result: {keyword_results[0][0]} (score={keyword_results[0][1]:.4f})")
        
        # Hybrid search
        hybrid_results = indexer.hybrid_search(query, k=3, alpha=0.6)
        print(f"Hybrid search top result:")
        print(f"  Paper: {hybrid_results[0]['paper_id']}")
        print(f"  Hybrid Score: {hybrid_results[0]['hybrid_score']:.4f}")
        print(f"  Text: {hybrid_results[0]['text'][:150]}...")
    
    # Print stats
    print("\n" + "=" * 60)
    print("Index Statistics")
    print("=" * 60)
    stats = indexer.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()