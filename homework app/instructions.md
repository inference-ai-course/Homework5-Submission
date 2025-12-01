# RAG System with Hybrid Search for arXiv Papers - Instructions

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system that indexes arXiv AI research papers and provides a search endpoint to retrieve relevant passages based on user queries. This version uses a hybrid search approach, combining vector-based and keyword-based retrieval for more accurate results.

## Prerequisites

### Required Python Packages
Install the following dependencies:

```bash
pip install pymupdf requests sentence-transformers faiss-cpu fastapi uvicorn numpy sqlite3
```

Or if you have a GPU and want to use FAISS with GPU acceleration:
```bash
pip install pymupdf requests sentence-transformers faiss-gpu fastapi uvicorn numpy sqlite3
```

### Python Version
- Python 3.8 or higher recommended

## Project Structure

```
.
├── main.py              # Main FastAPI application
├── instructions.md      # This file
├── fts_index.db         # SQLite database for keyword search
```

## How It Works

### 1. Data Collection
The system queries the arXiv API for recent papers in the `cs.AI` category (Artificial Intelligence).

### 2. Text Extraction
Each PDF is downloaded and processed using PyMuPDF (fitz) to extract raw text from all pages. Metadata such as title, authors, and year are also extracted.

### 3. Text Chunking
The extracted text is split into chunks of 500 tokens with a 50-token overlap to maintain context between chunks.

### 4. Hybrid Indexing
Two separate indexes are created:
- **Vector Index (FAISS)**: Each chunk is converted to a 384-dimensional vector using the `all-MiniLM-L6-v2` sentence transformer model and stored in a FAISS index for fast similarity search.
- **Keyword Index (SQLite FTS5)**: The text chunks and their associated metadata are stored in a SQLite database with a Full-Text Search (FTS5) index. This allows for efficient keyword-based search.

### 5. Hybrid Retrieval with Reciprocal Rank Fusion (RRF)
When a query is received, it is sent to both the vector and keyword indexes. The results from each are then combined using Reciprocal Rank Fusion (RRF) to produce a single, unified ranking.

### 6. Search and Comparison APIs
- A `/search` endpoint accepts queries and a search method (`vector`, `keyword`, or `hybrid`) to return the most relevant passages.
- A `/compare` endpoint allows for a side-by-side comparison of the three search methods for a given query.

## Running the Application

### Option 1: Direct Execution
```bash
python main.py
```

### Option 2: With Custom Configuration
You can set environment variables to customize the server:

```bash
HOST=127.0.0.1 PORT=8080 DEBUG=False python main.py
```

### Environment Variables
- `HOST`: Server host address (default: `0.0.0.0`)
- `PORT`: Server port (default: `8000`)
- `DEBUG`: Enable hot-reload during development (default: `True`)

## Using the API Endpoints

Once the server is running, you can query it in several ways:

### 1. Search Endpoint

#### Browser
Navigate to:
```
http://localhost:8000/search?q=What%20are%20the%20latest%20advances%20in%20AI?&method=hybrid
```
You can change the `method` parameter to `vector` or `keyword` to try different search strategies.

#### cURL
```bash
curl "http://localhost:8000/search?q=What%20are%20the%20latest%20advances%20in%20AI?&method=hybrid"
```

### 2. Compare Endpoint

#### Browser
Navigate to:
```
http://localhost:8000/compare?q=What%20are%20the%20latest%20advances%20in%20AI?
```

#### cURL
```bash
curl "http://localhost:8000/compare?q=What%20are%20the%20latest%20advances%20in%20AI?"
```

### 3. Interactive API Documentation
Visit `http://localhost:8000/docs` for an interactive Swagger UI where you can test the API directly.

## Response Formats

### `/search` Endpoint
```json
{
  "query": "What are the latest advances in AI?",
  "method": "hybrid",
  "results": [
    {
      "chunk": "First most relevant chunk text...",
      "score": 0.03278688524590164,
      "id": 123
    },
    ...
  ]
}
```

### `/compare` Endpoint
```json
{
  "query": "What are the latest advances in AI?",
  "vector_search": [ ... ],
  "keyword_search": [ ... ],
  "hybrid_search": [ ... ]
}
```

## Performance Notes

### First Run
- The first time you run `main.py`, it will:
  - Download 10 arXiv papers (can take 1-5 minutes depending on network speed)
  - Extract text and metadata from PDFs
  - Generate embeddings (takes 10-30 seconds depending on hardware)
  - Build the FAISS and SQLite FTS indexes

### Subsequent Runs
- The system currently rebuilds the indexes on each startup.

## Next Steps

1. Implement index persistence (save/load FAISS index and SQLite DB)
2. Add more query parameters (k for number of results, filters, etc.)
3. Implement reranking with a cross-encoder model
4. Create a web interface for easier interaction
5. Add logging and monitoring
6. Evaluate the performance of the different search methods using metrics like recall or Hit Rate.

## Contact & Support

For issues or questions about this homework assignment, please refer to the course materials or contact your instructor.
