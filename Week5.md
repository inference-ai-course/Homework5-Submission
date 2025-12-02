# Week 5 Homework Implementation Summary

## 1. SQLite + FAISS Hybrid Index

**File**: `hybrid_index.py`

The hybrid index combines:
- **SQLite database** with document metadata (title, authors, year, keywords)
- **FTS5 virtual table** for full-text keyword search
- **FAISS index** for semantic vector search
- **Mapping table** linking SQLite chunk IDs to FAISS index positions

Key features:
- Documents table stores paper metadata
- Chunks table stores text segments
- FTS5 table enables BM25-based keyword ranking
- Efficient lookups via indexed columns

## 2. Hybrid Search Implementation

**File**: `hybrid_search.py`

Implemented search methods:
- `vector_search()`: Semantic search using FAISS
- `keyword_search()`: Keyword search using SQLite FTS5
- `hybrid_search()`: Combines both methods with fusion strategies

Fusion strategies:
1. **Reciprocal Rank Fusion (RRF)**: Parameter-free method that combines rankings
2. **Weighted Score Fusion**: Weighted combination with adjustable alpha parameter

## 3. FastAPI Endpoint

**File**: `main.py`

Added `/hybrid_search` endpoint with parameters:
- `query`: Search query string
- `k`: Number of results to return (default: 3)
- `method`: Fusion method ('rrf' or 'weighted')
- `alpha`: Weight for vector search when using weighted fusion

The endpoint returns JSON with ranked results including scores and metadata.

## 4. Evaluation Notebook

**File**: `Week5_Evaluation.ipynb`

Comprehensive evaluation including:
- **10 test queries** covering diverse NLP/ML topics
- **Three search methods**: vector-only, keyword-only, hybrid
- **Evaluation metrics**: Recall@3, Hit Rate@3, Mean Reciprocal Rank (MRR)
- **Per-query analysis** showing which method performs best for each query
- **Aggregate statistics** comparing overall performance
- **Visualizations** of results
- **Detailed example queries** showing top-3 results from each method

## Files Included

```
.
├── hybrid_index.py           # SQLite + FAISS index builder
├── hybrid_search.py          # Hybrid search engine implementation
├── main.py                   # Updated FastAPI with /hybrid_search endpoint
├── Week5_Evaluation.ipynb    # Comprehensive evaluation notebook
├── Class 5 Homework.ipynb    # This homework notebook
├── data/
│   └── index/
│       ├── hybrid_index.db   # SQLite database with FTS5
│       ├── faiss_index.bin   # FAISS vector index
│       ├── chunks.json       # Text chunks
│       └── metadata.json     # Chunk metadata
└── README.md                 # Updated documentation
```

## How to Run

### 1. Build the Hybrid Index
```bash
python hybrid_index.py
```

### 2. Test Hybrid Search
```bash
TOKENIZERS_PARALLELISM=false python hybrid_search.py
```

### 3. Start FastAPI Server
```bash
TOKENIZERS_PARALLELISM=false python main.py
```

### 4. Test API Endpoint
```bash
curl 
"http://localhost:8000/hybrid_search?query=transformer%20attention&k=3&method=rrf"
```

### 5. Run Evaluation
Open and run `Week5_Evaluation.ipynb` in Jupyter

## Key Results

The evaluation demonstrates that:
- **Hybrid search** typically outperforms vector-only and keyword-only methods
- **RRF fusion** provides consistent performance without parameter tuning
- **Weighted fusion** allows fine-tuning the balance between semantic and keyword 
search
- Different query types benefit from different approaches, making hybrid search 
the most robust choice