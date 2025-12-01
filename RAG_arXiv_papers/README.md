
# Start the fastAPI app:

uvicorn main:app --reload

You can either test the end points by:

curl http://127.0.0.1:8000/api/vector_search/?q=xxx
curl http://127.0.0.1:8000/api/keyword_search/?q=xxx
curl http://127.0.0.1:8000/api/hybrid_search/?q=xxx
curl http://127.0.0.1:8000/api/hybrid_search_top_k/?q=xxx

or go to http://127.0.0.1:8000 and type your question and click buttons for each type of search


# Test the util functions and generate output files: json, and faiss index file

python test.py

# Evaluate Vector Search, Keyword Search, and Hybrid Search:

search_evaluation.ipynb