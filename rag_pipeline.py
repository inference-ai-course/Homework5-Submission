"""
RAG Pipeline for arXiv cs.CL Papers
This script downloads papers, processes them, creates embeddings, and builds a FAISS index.
"""

import json
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

# Configuration
DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "pdfs"
INDEX_DIR = DATA_DIR / "index"
NUM_PAPERS = 50
MAX_CHUNK_TOKENS = 512
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# Create directories
DATA_DIR.mkdir(exist_ok=True)
PDF_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)


def fetch_arxiv_papers(category: str = "cs.CL", max_results: int = 50) -> List[Dict]:
    """
    Fetch paper metadata from arXiv API.

    Args:
        category: arXiv category (default: cs.CL for Computation and Language)
        max_results: Number of papers to fetch

    Returns:
        List of dictionaries containing paper metadata
    """
    print(f"Fetching {max_results} papers from arXiv category {category}...")

    base_url = "http://export.arxiv.org/api/query?"
    query = f"search_query=cat:{category}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"

    response = requests.get(base_url + query)
    response.raise_for_status()

    # Parse XML response
    root = ET.fromstring(response.content)
    namespace = {'atom': 'http://www.w3.org/2005/Atom'}

    papers = []
    for entry in root.findall('atom:entry', namespace):
        paper_id = entry.find('atom:id', namespace).text.split('/abs/')[-1]
        title = entry.find('atom:title', namespace).text.strip().replace('\n', ' ')
        summary = entry.find('atom:summary', namespace).text.strip().replace('\n', ' ')

        # Get PDF link
        pdf_link = None
        for link in entry.findall('atom:link', namespace):
            if link.get('title') == 'pdf':
                pdf_link = link.get('href')
                break

        if pdf_link:
            papers.append({
                'id': paper_id,
                'title': title,
                'summary': summary,
                'pdf_url': pdf_link
            })

    print(f"Found {len(papers)} papers with PDF links")
    return papers


def download_pdf(paper: Dict, pdf_dir: Path) -> str:
    """
    Download PDF for a paper.

    Args:
        paper: Dictionary with paper metadata
        pdf_dir: Directory to save PDFs

    Returns:
        Path to downloaded PDF or empty string on error
    """
    pdf_filename = f"{paper['id'].replace('/', '_')}.pdf"
    pdf_path = pdf_dir / pdf_filename

    if pdf_path.exists():
        return str(pdf_path)

    try:
        response = requests.get(paper['pdf_url'], timeout=30)
        response.raise_for_status()

        with open(pdf_path, 'wb') as f:
            f.write(response.content)

        return str(pdf_path)
    except Exception as e:
        print(f"Error downloading {paper['id']}: {e}")
        return ""


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Open a PDF and extract all text as a single string.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Extracted text as string
    """
    try:
        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            page_text = page.get_text()
            # Basic cleaning: remove excessive whitespace
            page_text = ' '.join(page_text.split())
            pages.append(page_text)
        full_text = "\n".join(pages)
        doc.close()
        return full_text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""


def chunk_text(text: str, max_tokens: int = 512, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Input text
        max_tokens: Maximum tokens per chunk
        overlap: Number of overlapping tokens between chunks

    Returns:
        List of text chunks
    """
    tokens = text.split()
    chunks = []
    step = max_tokens - overlap

    for i in range(0, len(tokens), step):
        chunk = tokens[i:i + max_tokens]
        chunk_text = " ".join(chunk)
        # Only include chunks with sufficient content
        if len(chunk) > 50:  # Minimum 50 tokens
            chunks.append(chunk_text)

    return chunks


def process_papers(papers: List[Dict], pdf_dir: Path) -> Tuple[List[str], List[Dict]]:
    """
    Process papers: download PDFs, extract text, and chunk.

    Args:
        papers: List of paper metadata
        pdf_dir: Directory containing PDFs

    Returns:
        Tuple of (chunks list, metadata list)
    """
    all_chunks = []
    chunk_metadata = []

    print("\nProcessing papers...")
    for paper in tqdm(papers):
        # Download PDF
        pdf_path = download_pdf(paper, pdf_dir)
        if not pdf_path:
            continue

        # Extract text
        text = extract_text_from_pdf(pdf_path)
        if not text:
            continue

        # Chunk text
        chunks = chunk_text(text, max_tokens=MAX_CHUNK_TOKENS, overlap=CHUNK_OVERLAP)

        # Store chunks with metadata
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_metadata.append({
                'paper_id': paper['id'],
                'paper_title': paper['title'],
                'chunk_index': i,
                'total_chunks': len(chunks)
            })

    print(f"\nTotal chunks created: {len(all_chunks)}")
    return all_chunks, chunk_metadata


def create_embeddings(chunks: List[str], model_name: str = EMBEDDING_MODEL) -> Tuple[np.ndarray, SentenceTransformer]:
    """
    Generate embeddings for text chunks.

    Args:
        chunks: List of text chunks
        model_name: Name of sentence-transformers model

    Returns:
        Tuple of (numpy array of embeddings, model)
    """
    print(f"\nGenerating embeddings using {model_name}...")
    # Disable multiprocessing to avoid segfaults
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model = SentenceTransformer(model_name)

    # Encode in batches to handle large number of chunks
    batch_size = 16  # Reduced batch size
    embeddings = []

    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i + batch_size]
        try:
            batch_embeddings = model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                device='cpu'  # Force CPU to avoid any GPU issues
            )
            embeddings.append(batch_embeddings)
        except Exception as e:
            print(f"Error encoding batch {i}: {e}")
            continue

    if not embeddings:
        raise ValueError("No embeddings were generated")

    embeddings = np.vstack(embeddings)
    print(f"Generated embeddings shape: {embeddings.shape}")

    return embeddings, model


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Build FAISS index from embeddings.

    Args:
        embeddings: Numpy array of embeddings

    Returns:
        FAISS index
    """
    print("\nBuilding FAISS index...")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)

    # Normalize embeddings for better cosine similarity (optional)
    faiss.normalize_L2(embeddings)

    # Add embeddings to index
    index.add(embeddings.astype('float32'))

    print(f"Index built with {index.ntotal} vectors")
    return index


def save_index_and_data(index: faiss.IndexFlatL2, chunks: List[str],
                        metadata: List[Dict], index_dir: Path):
    """
    Save FAISS index and associated data.

    Args:
        index: FAISS index
        chunks: List of text chunks
        metadata: List of chunk metadata
        index_dir: Directory to save files
    """
    print("\nSaving index and data...")

    # Save FAISS index
    index_path = index_dir / "faiss_index.bin"
    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index to {index_path}")

    # Save chunks
    chunks_path = index_dir / "chunks.json"
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved chunks to {chunks_path}")

    # Save metadata
    metadata_path = index_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata to {metadata_path}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("RAG Pipeline for arXiv cs.CL Papers")
    print("=" * 60)

    # Step 1: Fetch papers from arXiv
    papers = fetch_arxiv_papers(category="cs.CL", max_results=NUM_PAPERS)

    # Save paper metadata
    papers_path = DATA_DIR / "papers_metadata.json"
    with open(papers_path, 'w', encoding='utf-8') as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    print(f"Saved paper metadata to {papers_path}")

    # Step 2: Process papers (download, extract, chunk)
    chunks, chunk_metadata = process_papers(papers, PDF_DIR)

    if len(chunks) == 0:
        print("ERROR: No chunks created. Exiting.")
        return

    # Save chunks and metadata immediately
    print("\nSaving chunks and metadata...")
    chunks_path = INDEX_DIR / "chunks.json"
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved chunks to {chunks_path}")

    metadata_path = INDEX_DIR / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(chunk_metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata to {metadata_path}")

    # Step 3: Generate embeddings
    embeddings, _ = create_embeddings(chunks)

    # Save embeddings immediately
    embeddings_path = INDEX_DIR / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings to {embeddings_path}")

    # Step 4: Build FAISS index
    index = build_faiss_index(embeddings)

    # Step 5: Save FAISS index
    print("\nSaving FAISS index...")
    index_path = INDEX_DIR / "faiss_index.bin"
    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index to {index_path}")

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print(f"Total papers processed: {len(papers)}")
    print(f"Total chunks created: {len(chunks)}")
    print(f"Index dimension: {embeddings.shape[1]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
