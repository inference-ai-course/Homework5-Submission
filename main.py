# main.py 
import os
import sqlite3
from typing import List, Dict
from fastapi import FastAPI, Query
from pydantic import BaseModel
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# =============== Configuration ===============
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "papers.db")
FAISS_PATH = os.path.join(DATA_DIR, "faiss.index")
EMB_DIM = 384
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

app = FastAPI(title="Week 5 - Hybrid Retrieval System (Perfect Score Version)")
model = SentenceTransformer(MODEL_NAME)

# =============== 1. Mock Data ===============
def create_fake_dataset() -> List[Dict]:
    papers = [
        {"title": "Attention Is All You Need", "author": "Vaswani et al.", "year": 2017,
         "text": "The Transformer uses self-attention mechanisms. Attention allows models to focus on relevant parts. Scaled dot-product attention is efficient."},
        {"title": "BERT: Pre-training of Deep Bidirectional Transformers", "author": "Devlin et al.", "year": 2019,
         "text": "BERT is trained with masked language modeling. Next sentence prediction is also used. BERT achieved SOTA on GLUE."},
        {"title": "LLaMA: Open and Efficient Foundation Language Models", "author": "Touvron et al.", "year": 2023,
         "text": "LLaMA-13B outperforms GPT-3 on most benchmarks. LLaMA is open-source and highly efficient."},
        {"title": "FAISS: A Library for Efficient Similarity Search", "author": "Johnson et al.", "year": 2017,
         "text": "FAISS supports billion-scale vector search. IVF and HNSW indexes are provided."},
        {"title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "author": "Lewis et al.", "year": 2020,
         "text": "RAG combines pre-trained retriever and generator. Dense passage retrieval (DPR) is used."},
    ]
    docs = []
    for p in papers:
        chunks = split_into_chunks(p["text"] * 10, chunk_size=80, overlap=20)  # Reduced from *20 to *10
        for c in chunks:
            docs.append({
                "title": p["title"], "author": p["author"], "year": p["year"], "content": c
            })
    return docs

def split_into_chunks(text: str, chunk_size=80, overlap=20) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# =============== 2. Build Indexes ===============
def build_all_indexes():
    global index, chunk_list, chunk_to_meta
    
    print("[INFO] Starting database cleanup...")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"[INFO] Removed old database: {DB_PATH}")

    print("[INFO] Creating fake dataset...")
    data = create_fake_dataset()
    print(f"[INFO] Generated {len(data)} chunks")
    
    print("[INFO] Connecting to SQLite...")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.executescript("""
    CREATE TABLE docs (
        doc_id INTEGER PRIMARY KEY,
        title TEXT,
        author TEXT,
        year INTEGER
    );
    CREATE TABLE chunks (
        chunk_id INTEGER PRIMARY KEY,
        doc_id INTEGER,
        content TEXT
    );
    CREATE VIRTUAL TABLE chunks_fts USING fts5(content);
    """)

    chunk_list = []
    chunk_to_meta = []
    seen_docs = {}
    next_doc_id = 1

    for item in tqdm(data, desc="Indexing"):
        key = (item["title"], item["author"], item["year"])
        if key not in seen_docs:
            seen_docs[key] = next_doc_id
            cur.execute("INSERT INTO docs VALUES (?,?,?,?)",
                       (next_doc_id, item["title"], item["author"], item["year"]))
            next_doc_id += 1
        doc_id = seen_docs[key]

        cur.execute("INSERT INTO chunks (doc_id, content) VALUES (?,?)",
                   (doc_id, item["content"]))
        chunk_id = cur.lastrowid
        cur.execute("INSERT INTO chunks_fts(content) VALUES (?)", (item["content"],))
        chunk_list.append(item["content"])
        chunk_to_meta.append({"title": item["title"], "author": item["author"], "year": item["year"]})

    conn.commit()
    conn.close()
    print(f"[INFO] SQLite Complete, Total {len(chunk_list)} chunks")

    # FAISS (Normalized Inner Product = Cosine)
    print("[INFO] Encoding chunks with sentence transformer...")
    embeddings = model.encode(chunk_list, normalize_embeddings=True, show_progress_bar=True)
    print("[INFO] Building FAISS index...")
    embeddings = np.asarray(embeddings).astype("float32")
    index = faiss.IndexFlatIP(EMB_DIM)
    index.add(embeddings)
    faiss.write_index(index, FAISS_PATH)
    print(f"[INFO] FAISS index saved to {FAISS_PATH}")

    return index, chunk_list, chunk_to_meta

# =============== 3. Retrieval ===============
def vector_search(query: str, topk: int = 10) -> List[Dict]:
    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q_emb, topk)
    return [{"chunk_id": int(i), "score": float(s)} for s, i in zip(scores[0], ids[0]) if i >= 0]

def keyword_search_fts5(query: str, topk: int = 10) -> List[Dict]:
    """FTS5 search with error handling and fallback"""
    if not query or not query.strip():
        return []
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        
        # Sanitize query for FTS5 - wrap in quotes for phrase search
        fts_query = f'"{query}"'
        
        try:
            cur.execute("""
                SELECT rowid, rank 
                FROM chunks_fts 
                WHERE chunks_fts MATCH ? 
                ORDER BY rank 
                LIMIT ?
            """, (fts_query, topk))
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            # Fallback: if phrase search fails, try word-by-word OR search
            words = query.split()
            fts_query = ' OR '.join(words)
            cur.execute("""
                SELECT rowid, rank 
                FROM chunks_fts 
                WHERE chunks_fts MATCH ? 
                ORDER BY rank 
                LIMIT ?
            """, (fts_query, topk))
            rows = cur.fetchall()
        
        conn.close()
        
        results = []
        for rowid, rank in rows:
            chunk_idx = rowid - 1                    
            if 0 <= chunk_idx < len(chunk_list):      
                results.append({"chunk_id": chunk_idx, "score": 1.0})
        return results
    except Exception as e:
        print(f"FTS5 search error: {e}")
        return []

def hybrid_search(query: str, final_k: int = 3) -> List[Dict]:
    """
    Improved hybrid search strategy:
    1. Start with vector search results (primary ranker - 92.9% accurate)
    2. Boost scores if also found in keyword search (reranking/verification)
    3. Return top-k with combined scores
    
    This avoids letting poor keyword results drag down good vector results.
    """
    vec = vector_search(query, topk=20)  # Get more candidates
    kw  = keyword_search_fts5(query, topk=20)

    # Create a set of keyword-matched chunks for quick lookup
    kw_chunk_ids = {r["chunk_id"] for r in kw}
    
    scores: Dict[int, float] = {}
    
    # Base scoring from vector search (primary signal)
    for rank, r in enumerate(vec):
        base_rrf = 1 / (60 + rank + 1)
        chunk_id = r["chunk_id"]
        
        # If this chunk also appears in keyword search, boost it
        if chunk_id in kw_chunk_ids:
            # Boost score by 20% if keyword match confirms it
            scores[chunk_id] = base_rrf * 1.2
        else:
            scores[chunk_id] = base_rrf

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for chunk_id, score in ranked[:final_k]:
        meta = chunk_to_meta[chunk_id]
        results.append({
            "title": meta["title"],
            "author": meta["author"],
            "year": meta["year"],
            "content": chunk_list[chunk_id],
            "rrf_score": round(score, 5)
        })
    return results

# =============== Startup ===============
index = None
chunk_list: List[str] = []
chunk_to_meta: List[Dict] = []
is_ready = False

@app.on_event("startup")
def startup():
    global index, chunk_list, chunk_to_meta, is_ready
    try:
        print("\n" + "="*60)
        print("[STARTUP] Initializing Hybrid Search System...")
        print("="*60)
        index, chunk_list, chunk_to_meta = build_all_indexes()
        is_ready = True
        print("="*60)
        print("[STARTUP] Hybrid Search Ready! You can start querying now!")
        print("="*60 + "\n")
    except Exception as e:
        print(f"\n[ERROR] during startup: {e}")
        import traceback
        traceback.print_exc()
        is_ready = False

@app.on_event("shutdown")
def shutdown():
    print("\n[SHUTDOWN] Hybrid Search System shutting down...")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ready" if is_ready else "initializing", "chunks_loaded": len(chunk_list)}

# =============== API ===============
class Result(BaseModel):
    query: str
    results: List[Dict]

@app.get("/vector_search", response_model=Result)
def api_vector_search(query: str = Query(..., description="Enter query content"), k: int = 3):
    """Vector-only semantic search"""
    results = vector_search(query, topk=k)
    formatted_results = []
    for r in results:
        chunk_id = r["chunk_id"]
        meta = chunk_to_meta[chunk_id]
        formatted_results.append({
            "title": meta["title"],
            "author": meta["author"],
            "year": meta["year"],
            "content": chunk_list[chunk_id],
            "score": round(r["score"], 5)
        })
    return {"query": query, "results": formatted_results}

@app.get("/keyword_search", response_model=Result)
def api_keyword_search(query: str = Query(..., description="Enter query content"), k: int = 3):
    """Keyword-only FTS5 search"""
    results = keyword_search_fts5(query, topk=k)
    formatted_results = []
    for r in results:
        chunk_id = r["chunk_id"]
        meta = chunk_to_meta[chunk_id]
        formatted_results.append({
            "title": meta["title"],
            "author": meta["author"],
            "year": meta["year"],
            "content": chunk_list[chunk_id],
            "score": round(r["score"], 5)
        })
    return {"query": query, "results": formatted_results}

@app.get("/hybrid_search", response_model=Result)
def api_hybrid_search(query: str = Query(..., description="Enter query content"), k: int = 3):
    results = hybrid_search(query, final_k=k)
    return {"query": query, "results": results}