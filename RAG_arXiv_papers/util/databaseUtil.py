import sqlite3
import json

class ArXivDatabase:
    def __init__(self, db_path='arxiv.db'):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Access columns by name
        self.cursor = self.conn.cursor()
    
    def create_tables(self):
        self.cursor.execute('DROP TABLE IF EXISTS documents;')
        create_command1 = '''
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id    TEXT, 
                title     TEXT,
                author    TEXT,
                year      INTEGER,
                keywords  TEXT,
                content   TEXT
            );
        '''
        self.cursor.execute(create_command1)

        self.cursor.execute('DROP TABLE IF EXISTS doc_chunks;')
        create_command2 = '''
            CREATE VIRTUAL TABLE doc_chunks USING fts5(
                content,
                content=documents,
                content_rowid=id
            );
        '''
        self.cursor.execute(create_command2)

        self.conn.commit()
        print("Tables and FTS5 index created successfully!")
    

    def insert_data(self, chunk_map, meta_data_file):
        #insert into main table:
        json_data = None
        try:
            with open(meta_data_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f) 
        except Exception as e:
            print(f"Error reading {meta_data_file}: {e}")

        count = 0
        if json_data:
            pdfs = json_data['metadata']
            for pdf in pdfs:
                doc_id = pdf.get('doc_id', '')
                chunks = chunk_map[doc_id]
                for chunk_text in chunks:
                    try:
                        self.cursor.execute('''
                            INSERT INTO documents (doc_id, title, author, year, keywords, content)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (
                            doc_id,
                            pdf.get('title', ''),
                            pdf.get('author', ''),
                            pdf.get('year', 0),
                            pdf.get('title', '').replace(' ', ', '),
                            chunk_text
                        ))
                        count += 1
                    except sqlite3.IntegrityError as e:
                        print(f"Error inserting {pdf.get('doc_id', 'Unknown')}: {e}")

        print(f"Inserted {count} rows into documents table")

        #Rebuild/populate the FTS5 index
        try:
            self.cursor.execute("INSERT INTO doc_chunks(doc_chunks) VALUES('rebuild')")
        except sqlite3.Error as e:
            print(f"Error populating the FTS5 index: {e}")
        print(f'FTS5 index table populated')

        self.conn.commit()
        

    def query_doc(self, query_str, limit):
        query_command = f'''
            SELECT documents.id, doc_id, title, documents.content, rank
            FROM documents
            JOIN doc_chunks ON documents.id = doc_chunks.rowid
            WHERE doc_chunks MATCH '{query_str}'
            ORDER BY rank
            LIMIT {limit};
        '''
        try:
            self.cursor.execute(query_command)
            self.conn.commit()
            results = self.cursor.fetchall()
            return results
        except sqlite3.Error as e:
            print(f"Error querying {query_str}: {e}")
            return []
        