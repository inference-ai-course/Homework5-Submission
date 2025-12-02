"""
Hybrid Index Builder for Week 5: SQLite + FAISS
Creates a combined index with metadata storage and keyword search capabilities.
"""

import json
import sqlite3
import re
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import faiss

# Configuration
DATA_DIR = Path("data")
INDEX_DIR = DATA_DIR / "index"
DB_PATH = DATA_DIR / "index" / "hybrid_index.db"


def extract_year_from_id(paper_id: str) -> int:
    """
    Extract year from arXiv paper ID.
    Format: YYMM.NNNNN

    Args:
        paper_id: arXiv paper ID

    Returns:
        Year as integer (e.g., 2025)
    """
    # Extract YY from the ID (first 2 digits)
    match = re.match(r'(\d{2})\d{2}\.\d+', paper_id)
    if match:
        yy = int(match.group(1))
        # Convert to full year (assuming 20xx)
        year = 2000 + yy
        return year
    return 2024  # Default fallback


def extract_keywords(title: str, summary: str) -> str:
    """
    Extract keywords from title and summary.
    Simple approach: take important words from title.

    Args:
        title: Paper title
        summary: Paper summary

    Returns:
        Comma-separated keywords
    """
    # Common stop words to exclude
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                  'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'should', 'could', 'may', 'might', 'can'}

    # Extract words from title
    words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
    keywords = [w for w in words if w not in stop_words]

    # Take up to 10 keywords
    return ', '.join(keywords[:10])


def create_database_schema(db_path: Path):
    """
    Create SQLite database schema with documents table and FTS5 table.

    Args:
        db_path: Path to database file
    """
    print(f"Creating database at {db_path}...")

    # Remove existing database
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create documents table with metadata
    cursor.execute("""
        CREATE TABLE documents (
            doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT NOT NULL,
            title TEXT NOT NULL,
            year INTEGER,
            keywords TEXT,
            summary TEXT
        )
    """)

    # Create chunks table
    cursor.execute("""
        CREATE TABLE chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        )
    """)

    # Create FTS5 virtual table for full-text search
    cursor.execute("""
        CREATE VIRTUAL TABLE doc_chunks_fts USING fts5(
            chunk_text,
            content='chunks',
            content_rowid='chunk_id'
        )
    """)

    # Create mapping table to link chunk_id to faiss_index position
    cursor.execute("""
        CREATE TABLE chunk_faiss_mapping (
            chunk_id INTEGER PRIMARY KEY,
            faiss_index INTEGER NOT NULL,
            FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id)
        )
    """)

    # Create indexes for faster lookups
    cursor.execute("CREATE INDEX idx_paper_id ON documents(paper_id)")
    cursor.execute("CREATE INDEX idx_doc_id ON chunks(doc_id)")

    conn.commit()
    conn.close()

    print("Database schema created successfully!")


def populate_database(db_path: Path):
    """
    Populate the database with papers, chunks, and metadata.

    Args:
        db_path: Path to database file
    """
    print("\nPopulating database...")

    # Load existing data
    papers_path = DATA_DIR / "papers_metadata.json"
    chunks_path = INDEX_DIR / "chunks.json"
    metadata_path = INDEX_DIR / "metadata.json"

    with open(papers_path, 'r', encoding='utf-8') as f:
        papers = json.load(f)

    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    with open(metadata_path, 'r', encoding='utf-8') as f:
        chunk_metadata = json.load(f)

    print(f"Loaded {len(papers)} papers and {len(chunks)} chunks")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create paper_id to doc_id mapping
    paper_to_doc = {}

    # Insert documents (papers)
    print("Inserting documents...")
    for paper in papers:
        paper_id = paper['id']
        title = paper['title']
        summary = paper.get('summary', '')
        year = extract_year_from_id(paper_id)
        keywords = extract_keywords(title, summary)

        cursor.execute("""
            INSERT INTO documents (paper_id, title, year, keywords, summary)
            VALUES (?, ?, ?, ?, ?)
        """, (paper_id, title, year, keywords, summary))

        doc_id = cursor.lastrowid
        paper_to_doc[paper_id] = doc_id

    print(f"Inserted {len(papers)} documents")

    # Insert chunks and build FTS index
    print("Inserting chunks and building FTS index...")
    for i, (chunk, meta) in enumerate(zip(chunks, chunk_metadata)):
        paper_id = meta['paper_id']
        doc_id = paper_to_doc.get(paper_id)

        if doc_id is None:
            print(f"Warning: Paper {paper_id} not found in documents table")
            continue

        chunk_index = meta['chunk_index']

        # Insert chunk
        cursor.execute("""
            INSERT INTO chunks (doc_id, chunk_index, chunk_text)
            VALUES (?, ?, ?)
        """, (doc_id, chunk_index, chunk))

        chunk_id = cursor.lastrowid

        # Insert into FTS table
        cursor.execute("""
            INSERT INTO doc_chunks_fts (rowid, chunk_text)
            VALUES (?, ?)
        """, (chunk_id, chunk))

        # Map chunk_id to FAISS index position
        cursor.execute("""
            INSERT INTO chunk_faiss_mapping (chunk_id, faiss_index)
            VALUES (?, ?)
        """, (chunk_id, i))

    print(f"Inserted {len(chunks)} chunks")

    conn.commit()
    conn.close()

    print("Database populated successfully!")


def verify_database(db_path: Path):
    """
    Verify the database contents.

    Args:
        db_path: Path to database file
    """
    print("\nVerifying database...")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Count documents
    cursor.execute("SELECT COUNT(*) FROM documents")
    doc_count = cursor.fetchone()[0]
    print(f"Total documents: {doc_count}")

    # Count chunks
    cursor.execute("SELECT COUNT(*) FROM chunks")
    chunk_count = cursor.fetchone()[0]
    print(f"Total chunks: {chunk_count}")

    # Sample a few documents
    print("\nSample documents:")
    cursor.execute("""
        SELECT paper_id, title, year, keywords
        FROM documents
        LIMIT 3
    """)

    for row in cursor.fetchall():
        paper_id, title, year, keywords = row
        print(f"  - {paper_id} ({year}): {title[:60]}...")
        print(f"    Keywords: {keywords[:60]}...")

    # Test FTS search
    print("\nTesting FTS search for 'transformer':")
    cursor.execute("""
        SELECT c.chunk_id, c.chunk_text, d.title
        FROM doc_chunks_fts f
        JOIN chunks c ON f.rowid = c.chunk_id
        JOIN documents d ON c.doc_id = d.doc_id
        WHERE doc_chunks_fts MATCH 'transformer'
        LIMIT 3
    """)

    results = cursor.fetchall()
    print(f"Found {len(results)} results")
    for i, (chunk_id, chunk_text, title) in enumerate(results, 1):
        print(f"  {i}. Chunk {chunk_id} from '{title[:50]}...'")
        print(f"     {chunk_text[:100]}...")

    conn.close()
    print("\nDatabase verification complete!")


def main():
    """Main execution function."""
    print("=" * 70)
    print("Hybrid Index Builder: SQLite + FAISS")
    print("=" * 70)

    # Step 1: Create database schema
    create_database_schema(DB_PATH)

    # Step 2: Populate database
    populate_database(DB_PATH)

    # Step 3: Verify database
    verify_database(DB_PATH)

    print("\n" + "=" * 70)
    print("Hybrid index created successfully!")
    print(f"Database location: {DB_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
