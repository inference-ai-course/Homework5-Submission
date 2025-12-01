
# Start the fastAPI app:

uvicorn main:app --reload

You can either test the end points by:

curl http://127.0.0.1:8000/api/vector_search/?q=xxx

curl http://127.0.0.1:8000/api/keyword_search/?q=xxx

curl http://127.0.0.1:8000/api/hybrid_search/?q=xxx

curl http://127.0.0.1:8000/api/hybrid_search_top_k/?q=xxx


or go to http://127.0.0.1:8000 and type your question and click buttons for each type of search

Examples of responses from the endpoints:

Vector Search:

{
  "query": "Liars' Bench",
  "results": [
    "329|1.0959986|chunk_text...",
    "328|1.102493|chunk_text...",
    "373|1.1556057|cunnk_text..."
  ]
}

each item in 'results' list is a string with the format 'index|distance|chunk_text'


Keyword Search:

{
  "query": "Liars' Bench",
  "results": [
    "329|-12.288914612855855|chunk_text...",
    "338|-11.904468042515578|chunk_text...",
    "331|-11.901263479723546|cunnk_text..."
  ]
}

each item in 'results' list is a string with the format 'id|rank|chunk_text'


Hybrid Search:

{
  "query": "Liars' Bench",
  "results": [
    "vector|328|keyword|329|-4.2540703|chunk_text...",
    "keyword|331|-4.0641317|chunk_text...",
    "keyword|338|-3.861722|cunnk_text..."
  ]
}

each item in 'results' list is a string with the format 'index|distance|chunk_text' if the result is from Vector Search, 'id|rank|chunk_text' if the result is from Keyword Search, and "'vector'|index|'keyword'|id|hybrid_score" if the result is a merged from both. For a same chunk, the relationship between index and id is id = index + 1 since in faiss indexing the index starts from 0, and in FTS5 table the id column (primary key autoincrement) starts from 1

# Test the util functions and generate output files: json, and faiss index file

python test.py

# Evaluate Vector Search, Keyword Search, and Hybrid Search:

search_evaluation.ipynb