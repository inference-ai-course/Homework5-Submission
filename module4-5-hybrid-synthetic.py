# modules/m4_hybrid_retrieval/__init__.py
"""Module 4: Hybrid Retrieval System."""

from .sqlite_fts import SQLiteFTS
from .vector_search import VectorSearcher
from .fusion import HybridRetriever, RRFFusion

__all__ = ["SQLiteFTS", "VectorSearcher", "HybridRetriever", "RRFFusion"]


# modules/m4_hybrid_retrieval/sqlite_fts.py
"""SQLite FTS5 for keyword search."""

import sqlite3
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import INDEX_DIR


class SQLiteFTS:
    """SQLite FTS5 full-text search engine."""
    
    def __init__(self, db_name: str = "academic_search.db"):
        self.db_path = INDEX_DIR / "sqlite" / db_name
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        
    def connect(self):
        """Connect to database."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        
    def create_tables(self):
        """Create FTS5 tables."""
        if self.conn is None:
            self.connect()
        
        cursor = self.conn.cursor()
        
        # Metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                arxiv_id TEXT,
                abstract TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Chunks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT,
                text TEXT,
                section TEXT,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
        """)
        
        # FTS5 virtual table for full-text search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id,
                doc_id,
                text,
                section,
                content='chunks',
                content_rowid='rowid'
            )
        """)
        
        # Triggers to keep FTS in sync
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, chunk_id, doc_id, text, section)
                VALUES (new.rowid, new.chunk_id, new.doc_id, new.text, new.section);
            END
        """)
        
        self.conn.commit()
        logger.info("Created SQLite FTS tables")
        
    def add_document(self, doc: Dict):
        """Add a document to the database."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO documents (doc_id, title, authors, arxiv_id, abstract)
            VALUES (?, ?, ?, ?, ?)
        """, (
            doc.get("doc_id", doc.get("arxiv_id")),
            doc.get("title", ""),
            ",".join(doc.get("authors", [])),
            doc.get("arxiv_id", ""),
            doc.get("abstract", "")
        ))
        
        self.conn.commit()
        
    def add_chunk(self, chunk):
        """Add a chunk to the database."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO chunks (chunk_id, doc_id, text, section)
            VALUES (?, ?, ?, ?)
        """, (
            chunk.chunk_id,
            chunk.doc_id,
            chunk.text,
            chunk.metadata.get("section", "")
        ))
        
        self.conn.commit()
        
    def add_chunks_batch(self, chunks: List):
        """Add multiple chunks."""
        cursor = self.conn.cursor()
        
        data = [
            (c.chunk_id, c.doc_id, c.text, c.metadata.get("section", ""))
            for c in chunks
        ]
        
        cursor.executemany("""
            INSERT OR REPLACE INTO chunks (chunk_id, doc_id, text, section)
            VALUES (?, ?, ?, ?)
        """, data)
        
        self.conn.commit()
        logger.info(f"Added {len(chunks)} chunks to SQLite")
        
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, str, float]]:
        """Search using FTS5."""
        cursor = self.conn.cursor()
        
        # FTS5 search with BM25 ranking
        cursor.execute("""
            SELECT chunk_id, doc_id, text, bm25(chunks_fts) as score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
        """, (query, top_k))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "text": row["text"],
                "score": -row["score"]  # BM25 returns negative scores, lower is better
            })
        
        return results
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


# modules/m4_hybrid_retrieval/fusion.py
"""Score fusion strategies for hybrid retrieval."""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import get_config


@dataclass
class SearchResult:
    """Unified search result."""
    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: Dict
    source: str  # "vector" or "keyword"


class RRFFusion:
    """Reciprocal Rank Fusion for combining search results."""
    
    def __init__(self, k: int = 60):
        self.k = k  # RRF constant
        
    def fuse(
        self,
        vector_results: List[SearchResult],
        keyword_results: List[SearchResult]
    ) -> List[SearchResult]:
        """Fuse results using RRF."""
        scores = {}
        results_map = {}
        
        # Score vector results
        for rank, result in enumerate(vector_results):
            rrf_score = 1.0 / (self.k + rank + 1)
            chunk_id = result.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score
            results_map[chunk_id] = result
        
        # Score keyword results
        for rank, result in enumerate(keyword_results):
            rrf_score = 1.0 / (self.k + rank + 1)
            chunk_id = result.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score
            if chunk_id not in results_map:
                results_map[chunk_id] = result
        
        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        fused_results = []
        for chunk_id in sorted_ids:
            result = results_map[chunk_id]
            result.score = scores[chunk_id]
            result.source = "hybrid"
            fused_results.append(result)
        
        return fused_results


class HybridRetriever:
    """Combines vector and keyword search."""
    
    def __init__(self, faiss_indexer, sqlite_fts, embedder, config=None):
        self.config = config or get_config()
        self.faiss_indexer = faiss_indexer
        self.sqlite_fts = sqlite_fts
        self.embedder = embedder
        self.fusion = RRFFusion()
        
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        vector_weight: Optional[float] = None,
        keyword_weight: Optional[float] = None
    ) -> List[SearchResult]:
        """Perform hybrid search."""
        top_k = top_k or self.config.rag.top_k_retrieval
        vector_weight = vector_weight or self.config.rag.vector_weight
        keyword_weight = keyword_weight or self.config.rag.keyword_weight
        
        # Vector search
        query_embedding = self.embedder.embed_text(query)
        vector_results_raw = self.faiss_indexer.search(query_embedding, top_k * 2)
        
        vector_results = [
            SearchResult(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                score=score,
                metadata=chunk.metadata,
                source="vector"
            )
            for chunk, score in vector_results_raw
        ]
        
        # Keyword search
        keyword_results_raw = self.sqlite_fts.search(query, top_k * 2)
        
        keyword_results = [
            SearchResult(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                text=r["text"],
                score=r["score"],
                metadata={},
                source="keyword"
            )
            for r in keyword_results_raw
        ]
        
        # Fuse results
        if self.config.rag.use_rrf:
            fused = self.fusion.fuse(vector_results, keyword_results)
        else:
            # Weighted score fusion
            fused = self._weighted_fusion(
                vector_results, keyword_results,
                vector_weight, keyword_weight
            )
        
        return fused[:top_k]
    
    def _weighted_fusion(
        self,
        vector_results: List[SearchResult],
        keyword_results: List[SearchResult],
        vector_weight: float,
        keyword_weight: float
    ) -> List[SearchResult]:
        """Simple weighted score fusion."""
        scores = {}
        results_map = {}
        
        # Normalize and weight vector scores
        if vector_results:
            max_v = max(r.score for r in vector_results)
            for r in vector_results:
                norm_score = (r.score / max_v) * vector_weight
                scores[r.chunk_id] = scores.get(r.chunk_id, 0) + norm_score
                results_map[r.chunk_id] = r
        
        # Normalize and weight keyword scores
        if keyword_results:
            max_k = max(r.score for r in keyword_results)
            for r in keyword_results:
                norm_score = (r.score / max_k) * keyword_weight
                scores[r.chunk_id] = scores.get(r.chunk_id, 0) + norm_score
                if r.chunk_id not in results_map:
                    results_map[r.chunk_id] = r
        
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        return [results_map[cid] for cid in sorted_ids]


# modules/m5_synthetic_data/__init__.py
"""Module 5: Synthetic Data Generation."""

from .qa_generator import QAGenerator
from .dataset_builder import DatasetBuilder

__all__ = ["QAGenerator", "DatasetBuilder"]


# modules/m5_synthetic_data/qa_generator.py
"""Generate synthetic Q&A pairs using GPT-4."""

import json
from typing import List, Dict, Optional
from openai import OpenAI
from tqdm import tqdm
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import get_config


class QAGenerator:
    """Generates Q&A pairs from academic papers using GPT-4."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.client = OpenAI(api_key=self.config.api.openai_api_key)
        
    def generate_qa_pairs(
        self,
        paper: Dict,
        num_pairs: int = 5
    ) -> List[Dict]:
        """Generate Q&A pairs for a single paper."""
        
        prompt = f"""You are a research assistant creating quiz questions from academic papers.

Paper Title: {paper.get('title', 'Unknown')}

Abstract:
{paper.get('abstract', '')}

Content (first 2000 chars):
{paper.get('full_text', '')[:2000]}

Generate {num_pairs} high-quality question-answer pairs that:
1. Cover key findings, methods, and concepts
2. Range from factual to conceptual questions
3. Have detailed, accurate answers based only on the provided text
4. Use appropriate academic terminology

Return as JSON array with format:
[{{"question": "...", "answer": "...", "type": "factual|conceptual|methodological"}}]

IMPORTANT: Base answers ONLY on the provided text. If something isn't mentioned, don't invent it."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            # Parse JSON
            result = json.loads(content)
            
            # Handle different response formats
            if isinstance(result, dict) and "questions" in result:
                qa_pairs = result["questions"]
            elif isinstance(result, list):
                qa_pairs = result
            else:
                qa_pairs = [result]
            
            # Add source metadata
            for qa in qa_pairs:
                qa["source_arxiv_id"] = paper.get("arxiv_id", "")
                qa["source_title"] = paper.get("title", "")
            
            return qa_pairs
            
        except Exception as e:
            logger.error(f"Failed to generate Q&A for {paper.get('arxiv_id')}: {e}")
            return []
    
    def generate_edge_cases(self, paper: Dict, num_cases: int = 1) -> List[Dict]:
        """Generate edge case Q&A (hallucination tests)."""
        
        prompt = f"""Create {num_cases} "trick" questions about this paper that test if a model hallucinates:

Paper Title: {paper.get('title', 'Unknown')}
Abstract: {paper.get('abstract', '')}

Generate questions that:
1. Ask about details NOT in the paper (fake statistics, made-up experiments)
2. Include plausible-sounding but incorrect premises
3. Have answers that CORRECTLY identify the misinformation

Format as JSON array:
[{{"question": "...", "answer": "The paper does not mention/contain...", "type": "edge_case"}}]"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            
            if isinstance(result, dict):
                edge_cases = result.get("questions", [result])
            else:
                edge_cases = result
            
            for ec in edge_cases:
                ec["source_arxiv_id"] = paper.get("arxiv_id", "")
                ec["type"] = "edge_case"
            
            return edge_cases
            
        except Exception as e:
            logger.error(f"Failed to generate edge cases: {e}")
            return []
    
    def generate_batch(
        self,
        papers: List[Dict],
        qa_per_paper: Optional[int] = None,
        include_edge_cases: bool = True
    ) -> List[Dict]:
        """Generate Q&A for multiple papers."""
        qa_per_paper = qa_per_paper or self.config.data.qa_pairs_per_paper
        all_qa = []
        
        for paper in tqdm(papers, desc="Generating Q&A pairs"):
            # Regular Q&A
            qa_pairs = self.generate_qa_pairs(paper, qa_per_paper)
            all_qa.extend(qa_pairs)
            
            # Edge cases
            if include_edge_cases:
                edge_cases = self.generate_edge_cases(paper, 1)
                all_qa.extend(edge_cases)
        
        logger.info(f"Generated {len(all_qa)} Q&A pairs from {len(papers)} papers")
        return all_qa


# modules/m5_synthetic_data/dataset_builder.py
"""Build training datasets in instruction-tuning format."""

import json
from typing import List, Dict, Optional
from pathlib import Path
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import get_config, DATA_DIR


class DatasetBuilder:
    """Builds instruction-tuning datasets."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.output_dir = DATA_DIR / "synthetic"
        
    def build_instruct_dataset(
        self,
        qa_pairs: List[Dict],
        system_prompt: Optional[str] = None
    ) -> List[Dict]:
        """Convert Q&A pairs to instruction format."""
        system_prompt = system_prompt or self.config.system_prompt
        
        dataset = []
        for qa in qa_pairs:
            # LLaMA 3 Instruct format
            formatted_text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{qa['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{qa['answer']}<|eot_id|>"""
            
            entry = {
                "text": formatted_text,
                "question": qa["question"],
                "answer": qa["answer"],
                "type": qa.get("type", "unknown"),
                "source": qa.get("source_arxiv_id", "")
            }
            dataset.append(entry)
        
        return dataset
    
    def save_jsonl(
        self,
        dataset: List[Dict],
        filename: str = "synthetic_qa.jsonl"
    ) -> Path:
        """Save dataset as JSONL."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for entry in dataset:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(dataset)} entries to {filepath}")
        return filepath
    
    def load_jsonl(self, filename: str = "synthetic_qa.jsonl") -> List[Dict]:
        """Load dataset from JSONL."""
        filepath = self.output_dir / filename
        
        dataset = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                dataset.append(json.loads(line))
        
        return dataset
    
    def split_dataset(
        self,
        dataset: List[Dict],
        train_ratio: float = 0.9
    ) -> Tuple[List[Dict], List[Dict]]:
        """Split dataset into train and validation."""
        import random
        random.shuffle(dataset)
        
        split_idx = int(len(dataset) * train_ratio)
        train = dataset[:split_idx]
        val = dataset[split_idx:]
        
        logger.info(f"Split: {len(train)} train, {len(val)} validation")
        return train, val
