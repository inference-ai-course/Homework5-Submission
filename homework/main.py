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
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)
# keyword_extractor = pipeline("zero-shot-classification", model="facebook/bart-large-mnli",  device_map="auto")

app = FastAPI()
conn = sqlite3.connect("./homework/mydata.db")
# if os.path.exists("./homework/mydata.db"):
#     os.remove("./homework/mydata.db")
# # Create database schema first
# create_database_schema("./homework/mydata.db")

list_urls = [
    "https://arxiv.org/list/econ.EM/recent"
]

@app.get("/search/")
async def search(query: str, k = 3):
    # Extract all text from docs
    # all_text = "\n".join(doc.page_content for doc in docs)
    # for doc in docs:
    #     extracted_keywords =  auto_generate_keywords(doc.page_content[:1000])
    #     doc.metadata["keywords"] = ", ".join(extracted_keywords)
    #     pprint.pp(extracted_keywords)

    cursor = conn.execute('''
        SELECT 
            d.title,
            d.author,
            c.chunk_index,
            snippet(chunks_fts, 0, '<b>', '</b>', '...', 10) as match_preview,
            c.text_content -- The full text to send to the LLM
        FROM chunks_fts f
        JOIN chunks c ON f.rowid = c.id     -- Join Index to Child
        JOIN documents d ON c.doc_id = d.id -- Join Child to Parent
        WHERE chunks_fts MATCH ?
        ORDER BY f.rank
        LIMIT ?;
    ''', (query, k))

    results = []
    for row in cursor.fetchall():
        results.append({
            "doc_id": row[0],
            "title": row[1], 
            "content": row[2],
            "highlighted_text": row[3]  # Shows matching terms in <b> tags
        })

    return JSONResponse(content={
        "query": query,
        "results": results
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