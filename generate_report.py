"""
Generate retrieval report with example queries.
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List, Dict

# Configuration
INDEX_DIR = Path("data/index")
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

print("Loading resources...")

# Load model
model = SentenceTransformer(EMBEDDING_MODEL)

# Load FAISS index
index_path = INDEX_DIR / "faiss_index.bin"
faiss_index = faiss.read_index(str(index_path))

# Load chunks
chunks_path = INDEX_DIR / "chunks.json"
with open(chunks_path, 'r', encoding='utf-8') as f:
    chunks = json.load(f)

# Load metadata
metadata_path = INDEX_DIR / "metadata.json"
with open(metadata_path, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

print(f"Loaded {faiss_index.ntotal} vectors, {len(chunks)} chunks\n")


def search_papers(query: str, k: int = 3) -> List[Dict]:
    """Search for relevant passages."""
    # Encode query
    query_embedding = model.encode([query])[0]
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Search
    query_vector = np.array([query_embedding]).astype('float32')
    distances, indices = faiss_index.search(query_vector, k)

    # Format results
    results = []
    for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
        if idx < len(chunks):
            results.append({
                'rank': i + 1,
                'distance': float(distance),
                'paper_id': metadata[idx]['paper_id'],
                'paper_title': metadata[idx]['paper_title'],
                'chunk_index': metadata[idx]['chunk_index'],
                'text': chunks[idx]
            })

    return results


# Define test queries
queries = [
    "What are transformer models and how do they work?",
    "Explain attention mechanisms in natural language processing",
    "How do large language models learn from data?",
    "What techniques are used for training language models?",
    "How do we evaluate the performance of NLP models?",
]

# Generate report
report = []
report.append("=" * 100)
report.append("RETRIEVAL REPORT: arXiv cs.CL Papers RAG System")
report.append("=" * 100)
report.append("")

for query_num, query in enumerate(queries, 1):
    report.append(f"\n{'='*100}")
    report.append(f"QUERY {query_num}: {query}")
    report.append(f"{'='*100}\n")

    results = search_papers(query, k=3)

    for result in results:
        report.append(f"--- Result {result['rank']} ---")
        report.append(f"Paper Title: {result['paper_title']}")
        report.append(f"Paper ID: {result['paper_id']}")
        report.append(f"Chunk Index: {result['chunk_index']}")
        report.append(f"Distance Score: {result['distance']:.4f}")
        report.append(f"\nText Excerpt (first 400 characters):")
        text_preview = result['text'][:400] + "..." if len(result['text']) > 400 else result['text']
        report.append(text_preview)
        report.append(f"\n{'-'*100}\n")

# Add statistics
report.append(f"\n{'='*100}")
report.append("SYSTEM STATISTICS")
report.append(f"{'='*100}")
report.append(f"Total papers indexed: {len(set(m['paper_id'] for m in metadata))}")
report.append(f"Total chunks: {len(chunks)}")
report.append(f"Embedding dimension: {faiss_index.d}")
report.append(f"Embedding model: {EMBEDDING_MODEL}")
report.append("")

# Save report
report_text = "\n".join(report)
report_path = Path("retrieval_report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)

print(report_text)
print(f"\nReport saved to {report_path}")
