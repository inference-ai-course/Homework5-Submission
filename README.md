# Week 5: Hybrid Retrieval System

## Overview
This project implements a hybrid retrieval system combining FAISS (vector search) 
and SQLite FTS5 (keyword search) with Reciprocal Rank Fusion (RRF).

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Build Index (if not included)
```bash
python hybrid_indexer.py
```

### 3. Run API Server
```bash
python main.py
# Visit http://localhost:8000/docs
```

### 4. View Evaluation
```bash
jupyter notebook hw5.ipynb
```

## Key Results
- **Hybrid (RRF)**: 76.4% Recall@3, 100% Hit Rate@3
- **Vector-Only**: 72.2% Recall@3, 83.3% Hit Rate@3  
- **Keyword-Only**: 66.7% Recall@3, 83.3% Hit Rate@3

RRF outperforms both individual methods and weighted fusion.

## Files
- `hybrid_indexer.py` - Hybrid index builder
- `main.py` - FastAPI server
- `hw5.ipynb` - Evaluation notebook
- `hybrid_evaluation_report.json` - Metrics