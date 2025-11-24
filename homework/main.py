# Initialize scraper with 2-second delay between requests
from homework.rag import RAGClass
from homework.web_scraper import ArxivScraper
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyMuPDFLoader
import pprint
import os
import re
from collections import Counter
from transformers import pipeline
import torch
from huggingface_hub import login
import sqlite3
from homework.db_schema import create_database_schema

# Attempt login using the HUGGINGFACE_TOKEN environment variable
with open('huggingface_token.txt', 'r') as f:
    token = f.read().strip()
    login(token=token)

model_id = "meta-llama/Llama-3.1-8B"
# keyword_generator = pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-8B-Instruct",  device_map="auto")
keyword_generator = pipeline(
    "text-generation", model=model_id, model_kwargs={"dtype": torch.bfloat16}, device_map="auto"
)
# keyword_extractor = pipeline("zero-shot-classification", model="facebook/bart-large-mnli",  device_map="auto")

app = FastAPI()
conn = sqlite3.connect("./homework/mydata.db")

list_urls = [
    "https://arxiv.org/list/econ.EM/recent"
]

@app.get("/search/")
async def search(query: str, k = 3):
    docs = []
    pdf_folder = "./pdfs"
    for fname in os.listdir(pdf_folder):
        if fname.lower().endswith('.pdf'):
            loader = PyMuPDFLoader(os.path.join(pdf_folder, fname))
            docs.extend(loader.load())

    rag = RAGClass(docs)
    rag.split_documents()
    rag.convert_text_to_embeddings_with_faiss();

    # Get FAISS results (vector similarity search)
    faissResults = rag.search_embeddings(query, k)

    # Get FTS results (full-text search)
    cursor = conn.execute('''
        SELECT 
            d.title,
            d.author,
            c.chunk_index,
            c.text_content
        FROM chunks_fts f
        JOIN chunks c ON f.rowid = c.id
        JOIN documents d ON c.doc_id = d.id
        WHERE chunks_fts MATCH ?
        ORDER BY f.rank
        LIMIT ?;
    ''', (query, k))

    ftsResults = []
    for row in cursor.fetchall():
        ftsResults.append({
            "title": row[0],
            "author": row[1], 
            "chunk_index": row[2],
            "text_content": row[3],
            "source": "fts"  # Tag the source
        })

    # Merge results: combine and deduplicate by chunk index
    merged_results = {}
    
    # Add FAISS results (vector similarity)
    for result in faissResults:
        chunk_idx = result["index"]
        merged_results[chunk_idx] = {
            "chunk_index": chunk_idx,
            "chunk_text": result["chunk_text"],
            "distance": result["distance"],
            "title": result.get("metadata", {}).get("title", ""),
            "author": result.get("metadata", {}).get("author", ""),
            "source": "faiss",
            "score": 1.0 / (1.0 + result["distance"])  # Convert distance to similarity score
        }
    
    # Add/update with FTS results (full-text search)
    for result in ftsResults:
        chunk_idx = result["chunk_index"]
        if chunk_idx in merged_results:
            # Merge: keep FAISS data but add FTS metadata
            merged_results[chunk_idx]["title"] = result.get("title", merged_results[chunk_idx].get("title", ""))
            merged_results[chunk_idx]["author"] = result.get("author", merged_results[chunk_idx].get("author", ""))
            merged_results[chunk_idx]["source"] = "hybrid"  # Found in both
            # Boost score if found in both
            merged_results[chunk_idx]["score"] = merged_results[chunk_idx].get("score", 0) + 0.5
        else:
            # New result from FTS only
            merged_results[chunk_idx] = {
                "chunk_index": chunk_idx,
                "chunk_text": result["text_content"],
                "title": result.get("title", ""),
                "author": result.get("author", ""),
                "source": "fts",
                "score": 0.5  # Lower score for FTS-only results
            }
    
    # Convert to list and sort by combined score
    final_results = list(merged_results.values())
    final_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # Limit to top k results
    final_results = final_results[:k]

    return JSONResponse(content={
        "query": query,
        "results": final_results,
        "faiss_count": len(faissResults),
        "fts_count": len(ftsResults),
        "merged_count": len(final_results)
    })


def auto_generate_keywords(text):
    """
    Generate 5 single-word keywords summarizing the given text using an LLM.
    Returns a list of up to 5 keywords (or an empty list if unsuccessful).
    """
    prompt = (
        "Analyze the following text and provide a list of 5 distinct academic keywords, each a single word (no explanations or punctuation), separated by commas:\n\n"
        f"{text.strip()}\n\nKeywords:"
    )
    llm_result = keyword_generator(prompt, max_new_tokens=12, num_return_sequences=1)
    # Get the output and try to extract the keyword list after 'Keywords:'
    output = llm_result[0]["generated_text"]
    if "Keywords:" in output:
        keyword_part = output.split("Keywords:", 1)[1].strip()
    else:
        keyword_part = output.strip()
    # Split by comma or whitespace, filter to single words, ignore empty
    import re
    raw_keywords = re.split(r'[, ]+', keyword_part)
    keywords = [kw.strip() for kw in raw_keywords if kw.strip()]
    # Take only first 5 and filter out multi-word entries
    single_word_keywords = [kw for kw in keywords if len(kw.split()) == 1][:5]
    return single_word_keywords

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)