# Week 5 Homework Submission  
**MLE in GenAI – Embedding Database Optimization & Hybrid Retrieval**  

**Author:** Wei Yang  
**Notebook:** `Wei_Yang_week5_submission.ipynb`

---

## Overview

This submission completes **Week 5: Embedding Database Optimization** for the MLE in GenAI course.

The goal of this homework is to improve a Retrieval-Augmented Generation (RAG) system by:

- Introducing a lightweight **database-backed embedding store**
- Combining **semantic vector search** with **keyword-based retrieval**
- Evaluating retrieval quality improvements
- Exposing the hybrid retriever via an API

The notebook builds directly on the Week 4 arXiv RAG pipeline.

---

## Objectives Completed

This notebook implements all required Week 5 components:

### 1. Reuse of Week 4 Artifacts
- Loads previously generated:
  - `chunks.pkl`
  - `meta.pkl`
  - `faiss.index`
- Uses the same SentenceTransformer embedding model for consistency.

### 2. Embedding Database Construction
- Builds a SQLite database:  
  **`data/week5_hybrid.db`**
- Creates a `chunks` table with:
  - `id` (chunk id)
  - `pdf_name` (source document)
  - `text` (chunk content)
- Inserts all document chunks into the database.

### 3. FAISS Semantic Search
- Implements `search_faiss_only()`:
  - Embeds queries
  - Retrieves top-k chunks from FAISS
  - Returns ranked semantic results

### 4. SQL Keyword Search
- Implements `search_sql_text()`:
  - Performs case-insensitive substring search
  - Queries the SQLite database directly
  - Acts as a keyword-based baseline

### 5. Hybrid Retrieval
- Implements `hybrid_search()`:
  - Uses FAISS to generate semantic candidates
  - Fetches candidate rows from SQLite
  - Optionally filters using a keyword
  - Ranks results by FAISS distance
- Demonstrates improved retrieval coverage compared to single-method approaches.

### 6. Evaluation
- Evaluates **FAISS-only**, **SQL-only**, and **Hybrid** retrieval
- Uses a small query set with expected keywords
- Reports **hit@3** metrics
- Shows that hybrid retrieval improves recall over pure semantic or pure keyword search.

### 7. (Optional) API Exposure
- Provides a FastAPI `/hybrid_search` endpoint
- Returns hybrid retrieval results as structured JSON
- Designed for reuse in later assignments (Week 6+) and final projects.

---

## Notebook Structure

| Section | Description |
|------|-------------|
| Step 1 | Load Week 4 chunks, metadata, FAISS index, and embedding model |
| Step 2 | Build SQLite embedding database (`week5_hybrid.db`) |
| Step 3 | FAISS-only semantic search helper |
| Step 4 | SQL keyword search helper |
| Step 5 | Hybrid semantic + keyword retrieval |
| Step 6 | Evaluation comparing retrieval strategies |
| Step 7 | FastAPI endpoint (optional) |
| Final | Markdown summary of results |

---

## How to Run

1. **Activate environment**
conda activate mle_genai

Launch Jupyter
jupyter notebook
Open and run

Copy code
Wei_Yang_week5_submission.ipynb
Run all cells from top to bottom.

(Optional) Run API
If the FastAPI code is extracted to a .py file:

bash
Copy code
uvicorn wei_yang_week5_api:app --host 0.0.0.0 --port 8000 --reload
Results Summary
FAISS semantic search performs well on broad conceptual queries.

SQL keyword search is effective for exact term matching.

The hybrid approach combines both strengths and achieves higher hit@3
on the evaluation query set.

The resulting retriever is suitable for integration into agent workflows
and voice-based systems in later weeks.

Notes
This notebook is designed to run end-to-end after a kernel restart.

All debug cells and redundant database builds were removed for clarity.

The database schema and API design intentionally mirror patterns used
in production RAG systems.







