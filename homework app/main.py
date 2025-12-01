import fitz  # PyMuPDF
import requests
import faiss
import numpy as np
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import uvicorn
from fastapi import FastAPI
import os
import sqlite3
import json

# Part 1: Data Fetching and Processing

def query_arxiv(category: str, max_results: int = 10) -> List[Dict]:
    """Query arxiv API for papers in a category."""
    url = f"http://export.arxiv.org/api/query?search_query=cat:{category}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    response = requests.get(url)
    root = ET.fromstring(response.content)
    entries = []
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        pdf_url = entry.find('{http://www.w3.org/2005/Atom}id').text.replace('abs', 'pdf') + '.pdf'
        authors = [author.find('{http://www.w3.org/2005/Atom}name').text for author in entry.findall('{http://www.w3.org/2005/Atom}author')]
        paper = {
            'pdf_url': pdf_url,
            'title': entry.find('{http://www.w3.org/2005/Atom}title').text.strip(),
            'authors': authors,
            'year': entry.find('{http://www.w3.org/2005/Atom}published').text.split('-')[0],
        }
        entries.append(paper)
    return entries

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file, either local or from a URL."""
    if pdf_path.startswith("http://") or pdf_path.startswith("https://"):
        response = requests.get(pdf_path)
        doc = fitz.open(stream=response.content, filetype="pdf")
    else:
        doc = fitz.open(pdf_path)
    full_text = "\n".join([page.get_text() for page in doc])
    return full_text

def chunk_text(text: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
    """Chunk text into smaller pieces."""
    tokens = text.split()
    chunks = []
    step = max_tokens - overlap
    for i in range(0, len(tokens), step):
        chunk = tokens[i:i + max_tokens]
        chunks.append(" ".join(chunk))
    return chunks

class DataProcessor:
    """Processes data from arxiv, chunks it and prepares for indexing."""
    def __init__(self, category='cs.AI', max_results=10):
        self.category = category
        self.max_results = max_results
        self.chunks_with_metadata = []

    def process_papers(self):
        """Download, extract text, and chunk papers."""
        print("Processing papers...")
        papers = query_arxiv(self.category, self.max_results)
        for i, paper in enumerate(papers):
            print(f"Processing paper {i+1}/{len(papers)}: {paper['title']}")
            text = extract_text_from_pdf(paper['pdf_url'])
            chunks = chunk_text(text)
            for chunk in chunks:
                self.chunks_with_metadata.append({
                    'text': chunk,
                    'title': paper['title'],
                    'authors': ", ".join(paper['authors']),
                    'year': paper['year'],
                    'source': paper['pdf_url']
                })
        print("Finished processing papers.")
        return self.chunks_with_metadata

# Part 2: Retrievers

class VectorRetriever:
    """Handles vector-based retrieval using FAISS."""
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def create_index(self, chunks_with_metadata: List[Dict]):
        """Create FAISS index from chunks."""
        self.chunks = [item['text'] for item in chunks_with_metadata]
        embeddings = self.model.encode(self.chunks, show_progress_bar=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings))
        print("FAISS index created.")

    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Search for a query and return top k results."""
        if self.index is None:
            raise Exception("Index not created. Please call create_index first.")
        query_vector = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_vector), k)
        return list(zip(indices[0], distances[0]))

class KeywordRetriever:
    """Handles keyword-based retrieval using SQLite FTS5."""
    def __init__(self, db_path=':memory:'):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.chunks = []

    def create_index(self, chunks_with_metadata: List[Dict]):
        """Create SQLite FTS index."""
        self.chunks = [item['text'] for item in chunks_with_metadata]
        self.cursor.execute("DROP TABLE IF EXISTS documents")
        self.cursor.execute("""
        CREATE VIRTUAL TABLE documents USING fts5(
            text, title, authors, year, source
        );
        """)
        self.cursor.executemany("""
        INSERT INTO documents (text, title, authors, year, source)
        VALUES (:text, :title, :authors, :year, :source);
        """, chunks_with_metadata)
        self.conn.commit()
        print("SQLite FTS index created.")

    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Search for a query and return top k results with BM25 scores."""
        # The rank is a built-in BM25 score from FTS5
        self.cursor.execute(
            "SELECT rowid - 1, rank FROM documents WHERE documents MATCH ? ORDER BY rank LIMIT ?",
            (query, k)
        )
        return [(row[0], row[1]) for row in self.cursor.fetchall()]

class HybridRetriever:
    """Combines vector and keyword retrieval using RRF."""
    def __init__(self, vector_retriever: VectorRetriever, keyword_retriever: KeywordRetriever, k_rrf: int = 60):
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.k_rrf = k_rrf
        self.chunks = self.vector_retriever.chunks

    def search(self, query: str, k: int = 10) -> List[Dict]:
        """Perform hybrid search."""
        vector_results = self.vector_retriever.search(query, k)
        keyword_results = self.keyword_retriever.search(query, k)

        # Create dictionaries for faster lookups
        vec_scores = {doc_id: 1 / (self.k_rrf + rank + 1) for rank, (doc_id, _) in enumerate(vector_results)}
        key_scores = {doc_id: 1 / (self.k_rrf + rank + 1) for rank, (doc_id, _) in enumerate(keyword_results)}

        # Combine scores
        combined_scores = {}
        for doc_id in set(vec_scores.keys()) | set(key_scores.keys()):
            combined_scores[doc_id] = vec_scores.get(doc_id, 0) + key_scores.get(doc_id, 0)

        # Sort by combined score
        sorted_docs = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)

        results = []
        for doc_id, score in sorted_docs[:k]:
            results.append({
                'chunk': self.chunks[doc_id],
                'score': score,
                'id': doc_id
            })
        return results

# Part 3: FastAPI Application and Main Execution

app = FastAPI()

# Global variables to hold retrievers and data
data_processor = None
vector_retriever = None
keyword_retriever = None
hybrid_retriever = None
chunks_with_metadata = None

@app.on_event("startup")
def startup_event():
    """On startup, process data and create indices."""
    global data_processor, vector_retriever, keyword_retriever, hybrid_retriever, chunks_with_metadata
    
    data_processor = DataProcessor(max_results=10)
    chunks_with_metadata = data_processor.process_papers()

    vector_retriever = VectorRetriever()
    vector_retriever.create_index(chunks_with_metadata)

    keyword_retriever = KeywordRetriever(db_path='fts_index.db')
    keyword_retriever.create_index(chunks_with_metadata)
    
    hybrid_retriever = HybridRetriever(vector_retriever, keyword_retriever)
    print("Startup complete. Ready for search.")

@app.get("/search")
async def search(q: str, method: str = "hybrid"):
    """
    Search for a query 'q' using a specified method: 'vector', 'keyword', or 'hybrid'.
    """
    k = 5
    if method == "vector":
        results = vector_retriever.search(q, k)
        # map results to chunks
        output = [chunks_with_metadata[idx] for idx, _ in results]
        return {"query": q, "method": method, "results": output}
    elif method == "keyword":
        results = keyword_retriever.search(q, k)
        output = [chunks_with_metadata[idx] for idx, _ in results]
        return {"query": q, "method": method, "results": output}
    elif method == "hybrid":
        results = hybrid_retriever.search(q, k)
        return {"query": q, "method": method, "results": results}
    else:
        return {"error": "Invalid method. Choose from 'vector', 'keyword', 'hybrid'."}

@app.get("/compare")
async def compare_searches(q: str):
    """Compare results from all three search methods for a given query."""
    k = 5
    vector_results_raw = vector_retriever.search(q, k)
    vector_results = [chunks_with_metadata[idx] for idx, _ in vector_results_raw]

    keyword_results_raw = keyword_retriever.search(q, k)
    keyword_results = [chunks_with_metadata[idx] for idx, _ in keyword_results_raw]

    hybrid_results = hybrid_retriever.search(q, k)

    return {
        "query": q,
        "vector_search": vector_results,
        "keyword_search": keyword_results,
        "hybrid_search": hybrid_results
    }

if __name__ == "__main__":
    # Note: The startup event will run when Uvicorn starts.
    # You can also run the data processing and indexing part here if you
    # are not using the FastAPI app but a different script.
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )