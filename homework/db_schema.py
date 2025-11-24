import sqlite3
import os

def create_database_schema(db_path="my_db.db"):
    """
    Create the database schema for hybrid retrieval system.
    Includes documents table for metadata and FTS5 table for keyword search.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create documents table for metadata
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            doc_id INTEGER PRIMARY KEY,
            title TEXT,
            author TEXT,
            year INTEGER,
            keywords TEXT,
            source_url TEXT,
            file_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create FTS5 virtual table for chunk text search
    cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks USING fts5(
            content,
            content='documents',
            content_rowid='doc_id'
        )
    ''')

    # Create chunks table to store chunk metadata (linking to FAISS indices)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id INTEGER PRIMARY KEY,
            doc_id INTEGER,
            chunk_text TEXT,
            chunk_index INTEGER,  -- index in FAISS
            start_char INTEGER,
            end_char INTEGER,
            FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
        )
    ''')

    # Create evaluation queries table for testing
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS eval_queries (
            query_id INTEGER PRIMARY KEY,
            query_text TEXT,
            relevant_docs TEXT,  -- comma-separated doc_ids
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()
    print(f"Database schema created/updated at {db_path}")

def insert_sample_eval_queries(db_path="my_db.db"):
    """
    Insert sample evaluation queries for testing the hybrid retrieval system.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    sample_queries = [
        ("machine learning in economics", "1,2,3"),
        ("causal inference methods", "4,5"),
        ("time series analysis", "6,7,8"),
        ("econometric modeling", "9,10"),
        ("statistical learning theory", "1,3,5"),
        ("regression analysis", "2,4,6"),
        ("neural networks applications", "7,8,9"),
        ("bayesian statistics", "10,1"),
        ("experimental economics", "2,3,4"),
        ("financial econometrics", "5,6,7")
    ]

    cursor.executemany('''
        INSERT OR IGNORE INTO eval_queries (query_text, relevant_docs)
        VALUES (?, ?)
    ''', sample_queries)

    conn.commit()
    conn.close()
    print("Sample evaluation queries inserted")

if __name__ == "__main__":
    create_database_schema()
    insert_sample_eval_queries()
