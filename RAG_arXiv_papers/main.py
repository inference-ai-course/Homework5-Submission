from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path
import util.util as util


app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def startup_event():
    """Load pdfs and generate chunks and create vector DB on startup"""
    util.init_server()


@app.get("/")
async def root():
    """Serve the main UI"""
    return FileResponse("static/index.html")


@app.get("/api/vector_search/")
async def vector_search(q: str):
    print(f'user input: {q}')
    distances, indexes = util.query_faiss(q)
    print(f'indexes: {indexes}')
    print(f'distances: {distances}')
    results = []
    for i, idx in enumerate(indexes):
        results.append(str(idx) + '|' + str(distances[i]) + '|' + util.chunks[idx])
    return {"query": q, "results": results}


@app.get("/api/keyword_search/")
async def keyword_search(q: str):
    print(f'user input: {q}')
    rows = util.query_keyword(q)
    results = []
    for row in rows:
        results.append(str(row['id']) + '|' + str(row['rank']) + '|' + row['content'])
    return {"query": q, "results": results}


@app.get("/api/hybrid_search/")
async def hybrid_search(q: str):
    print(f'user input: {q}')
    score_results, results = util.query_hybrid(q)
    return {"query": q, "results": results}


@app.get("/api/hybrid_search_top_k/")
async def hybrid_search_top_k(q: str):
    print(f'user input: {q}')
    score_results, results = util.query_hybrid(q, top=3)
    return {"query": q, "results": results}
    
    