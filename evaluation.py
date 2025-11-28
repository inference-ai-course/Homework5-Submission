# evaluation.py

import requests
from time import sleep
from typing import List, Dict

# ==================== Auto Wait for Server ====================
def wait_for_server(max_wait: int = 60) -> bool:
    """Wait until /health reports 'ready' (indexes fully built)"""
    print("Waiting for the FastAPI server to finish building indexes", end="")
    for i in range(max_wait):
        try:
            resp = requests.get("http://127.0.0.1:8000/health", timeout=5)
            data = resp.json()
            if data.get("status") == "ready":
                print(f"\nServer ready! Loaded {data['chunks_loaded']} chunks.")
                return True
        except Exception:
            pass
        print(".", end="", flush=True)
        sleep(1)
    print("\nTimeout – server did not become ready in time.")
    return False


if not wait_for_server():
    exit(1)

# ==================== Search Functions ====================
def search_vector(query: str, k: int = 3) -> List[Dict]:
    """Pure vector (semantic) search"""
    try:
        r = requests.get("http://127.0.0.1:8000/vector_search",
                         params={"query": query, "k": k}, timeout=15)
        r.raise_for_status()
        return r.json()["results"]
    except Exception as e:
        print(f"Vector search failed ({query}): {e}")
        return []


def search_keyword(query: str, k: int = 3) -> List[Dict]:
    """Pure keyword search using SQLite FTS5"""
    try:
        r = requests.get("http://127.0.0.1:8000/keyword_search",
                         params={"query": query, "k": k}, timeout=15)
        r.raise_for_status()
        return r.json()["results"]
    except Exception as e:
        print(f"Keyword search failed ({query}): {e}")
        return []


def search_hybrid(query: str, k: int = 3) -> List[Dict]:
    """Hybrid search with RRF fusion (the star of the show)"""
    try:
        r = requests.get("http://127.0.0.1:8000/hybrid_search",
                         params={"query": query, "k": k}, timeout=15)
        r.raise_for_status()
        return r.json()["results"]
    except Exception as e:
        print(f"Hybrid search failed ({query}): {e}")
        return []


# ==================== Test Queries ====================
test_queries = [
    ("Attention Is All You Need",          2017),
    ("Transformer self-attention",         2017),
    ("BERT pretraining",                   2019),
    ("masked language modeling",           2019),
    ("LLaMA foundation model",             2023),
    ("open source large language",         2023),
    ("RAG retrieval augmented",            2020),
    ("dense passage retrieval DPR",        2020),
    ("FAISS similarity search",            2017),
    ("billion scale vector",               2017),
    ("HNSW approximate neighbors",         2017),
    ("scaled dot-product attention",       2017),
    ("GPT benchmark comparison",           2023),
    ("next sentence prediction",           2019),
]


# ==================== Evaluation Logic ====================
def evaluate(method_func, name: str, k: int = 3):
    print(f"\n{'='*70}")
    print(f" Evaluating: {name} ".center(70, "="))
    print(f"{'No.':<3} {'Query':<38} {'Top-1 Year':<12} {'Expected':<10} Result")
    print("-" * 70)

    hits = 0
    for idx, (q, expected) in enumerate(test_queries, 1):
        results = method_func(q, k=k)
        if results:
            top_year = results[0]["year"]
            mark = "Success" if top_year == expected else "Failed"
            if top_year == expected:
                hits += 1
        else:
            top_year = "N/A"
            mark = "Failed"

        print(f"{idx:<3} {q:<38} {top_year:<12} {expected:<10} {mark}")

    hit_rate = hits / len(test_queries) * 100
    print("-" * 70)
    print(f"Hits: {hits}/{len(test_queries)} | Hit Rate@{k}: {hit_rate:.1f}%")
    print("=" * 70)
    return hits, hit_rate


# ==================== Run All Evaluations ====================
print("\n" + "="*70)
print(" Week 5 – Hybrid Retrieval System Evaluation ".center(70))
print("="*70)

vector_hits,  vector_rate  = evaluate(search_vector,  "Vector-Only Search (Semantic)",  k=3)
keyword_hits, keyword_rate = evaluate(search_keyword, "Keyword-Only Search (FTS5)",    k=3)
hybrid_hits,  hybrid_rate  = evaluate(search_hybrid,  "Hybrid Search (RRF Fusion)",    k=3)

# ==================== Summary ====================
print("\n" + "="*70)
print(" Summary ".center(70))
print("="*70)
print(f"{'Method':<30} {'Hit Rate @3':<15} {'Hits'}")
print("-" * 70)
print(f"{'Vector-Only':<30} {vector_rate:>10.1f}% {vector_hits:>8}/{len(test_queries)}")
print(f"{'Keyword-Only':<30} {keyword_rate:>10.1f}% {keyword_hits:>8}/{len(test_queries)}")
print(f"{'Hybrid (RRF)':<30} {hybrid_rate:>10.1f}% {hybrid_hits:>8}/{len(test_queries)}")
print("="*70)
