# Week 4 Homework: RAG System for arXiv cs.CL Papers

This project implements a Retrieval-Augmented Generation (RAG) system for semantic search over 50 recent arXiv cs.CL (Computation and Language) papers.

## Overview

The system consists of:
1. **Data Collection Pipeline**: Downloads and processes 50 arXiv cs.CL papers
2. **Text Processing**: Extracts text from PDFs and chunks into searchable segments
3. **Embedding Generation**: Creates dense vector embeddings using sentence-transformers
4. **FAISS Index**: Builds a fast similarity search index
5. **FastAPI Service**: REST API for querying the knowledge base
6. **Demo Notebook**: Interactive Jupyter notebook for exploration

## Project Structure

```
.
├── rag_pipeline.py           # Main pipeline for data processing and indexing
├── main.py                   # FastAPI service
├── create_index.py           # Helper script to create FAISS index
├── generate_report.py        # Generate retrieval performance report
├── RAG_Demo.ipynb           # Interactive demo notebook
├── retrieval_report.txt      # Performance report with example queries
├── requirements.txt          # Python dependencies
├── data/
│   ├── papers_metadata.json  # Metadata for all papers
│   ├── pdfs/                 # Downloaded PDF files (50 papers)
│   └── index/
│       ├── chunks.json       # Text chunks (1078 chunks)
│       ├── metadata.json     # Chunk metadata
│       ├── embeddings.npy    # Dense vector embeddings
│       └── faiss_index.bin   # FAISS index file
└── README.md
```

## System Statistics

- **Total Papers**: 50 arXiv cs.CL papers
- **Total Chunks**: 1,078 text segments
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Chunk Size**: 512 tokens with 50 token overlap
- **Index Type**: FAISS IndexFlatL2 with L2 normalization

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Homework4-Submission
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Option 1: Use Pre-built Index (Recommended)

If the data has already been processed (as in this submission), you can directly use the FastAPI service or demo notebook.

#### Start the FastAPI Service

```bash
source venv/bin/activate
TOKENIZERS_PARALLELISM=false python main.py
```

The API will be available at `http://localhost:8000`

#### API Endpoints

1. **Root**: `GET /`
   - Returns API information

2. **Search**: `GET /search?q=<query>&k=<num_results>`
   - `q`: Search query (required)
   - `k`: Number of results (default: 3, max: 20)
   - Example:
     ```bash
     curl "http://localhost:8000/search?q=transformer%20models&k=3"
     ```

3. **Health Check**: `GET /health`
   - Returns service health status

4. **Statistics**: `GET /stats`
   - Returns index statistics

5. **Get Paper**: `GET /paper/{paper_id}`
   - Returns all chunks for a specific paper

#### Example API Usage

```bash
# Search for transformer models
curl "http://localhost:8000/search?q=transformer%20models&k=3"

# Get system statistics
curl "http://localhost:8000/stats"

# Health check
curl "http://localhost:8000/health"
```

### Option 2: Run the Full Pipeline

To rebuild the entire index from scratch:

```bash
source venv/bin/activate

# Run the full pipeline (downloads papers, processes, and indexes)
TOKENIZERS_PARALLELISM=false python rag_pipeline.py

# If the pipeline crashes during FAISS indexing, create the index separately
python create_index.py
```

**Note**: The pipeline downloads 50 papers from arXiv, which may take 5-10 minutes depending on network speed.

### Using the Demo Notebook

1. **Start Jupyter**
   ```bash
   source venv/bin/activate
   jupyter notebook RAG_Demo.ipynb
   ```

2. **Run the cells** to:
   - Load the FAISS index and data
   - Execute example queries
   - Visualize retrieval results
   - Analyze system performance

### Generate Retrieval Report

To generate a retrieval report with example queries:

```bash
source venv/bin/activate
TOKENIZERS_PARALLELISM=false python generate_report.py
```

This creates `retrieval_report.txt` with detailed results for 5 example queries.

## Example Queries

The system has been tested with the following queries (see `retrieval_report.txt` for full results):

1. "What are transformer models and how do they work?"
2. "Explain attention mechanisms in natural language processing"
3. "How do large language models learn from data?"
4. "What techniques are used for training language models?"
5. "How do we evaluate the performance of NLP models?"

## Implementation Details

### Text Chunking Strategy

- **Method**: Sliding window with overlap
- **Chunk size**: 512 tokens (split by whitespace)
- **Overlap**: 50 tokens between adjacent chunks
- **Rationale**: Balances context preservation with retrieval precision

### Embedding Model

- **Model**: `all-MiniLM-L6-v2` from sentence-transformers
- **Dimensions**: 384
- **Advantages**: Fast, efficient, good semantic understanding

### FAISS Index

- **Type**: IndexFlatL2 (exact L2 distance search)
- **Normalization**: Embeddings are L2-normalized for cosine similarity
- **Performance**: ~1078 vectors, sub-millisecond search times

## Deliverables

1. ✅ **Code**: Complete RAG pipeline (`rag_pipeline.py`, `main.py`)
2. ✅ **Data & Index**: FAISS index and processed chunks in `data/index/`
3. ✅ **Retrieval Report**: `retrieval_report.txt` with 5 example queries
4. ✅ **FastAPI Service**: Production-ready API with multiple endpoints
5. ✅ **Demo Notebook**: Interactive `RAG_Demo.ipynb`

## Performance Notes

- **Retrieval Speed**: Sub-second for top-k queries
- **Memory Usage**: ~200MB for embeddings and index
- **Coverage**: 50 recent cs.CL papers (as of November 2024)

## Troubleshooting

### Common Issues

1. **Segmentation Fault During Embedding Generation**
   - Set `TOKENIZERS_PARALLELISM=false` before running
   - Reduce batch size in `rag_pipeline.py`

2. **FAISS Not Installed**
   - Install with: `pip install faiss-cpu`

3. **PyMuPDF Issues**
   - Ensure both `PyMuPDF` and `PyMuPDFb` are installed

4. **API Not Loading Resources**
   - Ensure `data/index/` contains all required files:
     - `faiss_index.bin`
     - `chunks.json`
     - `metadata.json`

## Future Improvements

- Add hybrid search (keyword + semantic)
- Implement reranking with cross-encoder
- Add metadata filtering (by date, author, etc.)
- Support for more file formats
- Add caching for frequently asked queries
- Implement batch query processing

## Dependencies

- `fastapi`: Web framework
- `uvicorn`: ASGI server
- `sentence-transformers`: Embedding generation
- `faiss-cpu`: Vector similarity search
- `PyMuPDF`: PDF text extraction
- `numpy`: Numerical operations
- `requests`: HTTP requests
- `tqdm`: Progress bars

## License

See LICENSE file for details.

## Author

Homework 4 Submission - AI Agent Development Course
