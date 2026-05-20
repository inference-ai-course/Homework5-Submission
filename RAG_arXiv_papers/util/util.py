import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import os
import time
import re
from pathlib import Path
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss #faiss-cpu or faiss-gpu
import numpy as np
import json
from util.databaseUtil import ArXivDatabase


chunks = []
faiss_index = None
db = None
input_dir = 'arxiv_pdfs/'
meta_data_file = f"{input_dir}/metadata.json"

def init_server():
    try:
        # Fetch 50 cs.CL papers
        if not os.path.exists(meta_data_file):
            print('Fetch 50 cs.CL papers...')
            fetch_arxiv_papers(category='cs.CL')

        # Get all PDF files and generate chunks
        print('Get all PDF files and generate chunks...')
        chunk_map = {}
        with open(meta_data_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        metadata = json_data['metadata']
        for pdf_data in metadata:
            doc_id, text = extract_text_from_pdf(pdf_data)
            temp_chunks = chunk_text(text)
            chunk_map[doc_id] = temp_chunks
            chunks.extend(temp_chunks)
            print(f'Done processing {doc_id}')

        #Embedding and FAISS Indexing
        print('Embedding and FAISS Indexing...')
        embeddings = embedding(chunks)
        global faiss_index
        faiss_index = faiss_indexing(embeddings)

        # Set up SQLite and FTS5 tables
        print('SQLite and FTS5 tables setup...')
        global db
        db = ArXivDatabase()
        db.create_tables()
        db.insert_data(chunk_map, meta_data_file)

    except Exception as e: 
        print(f"Error in initializing: {e}")
        raise

        
def fetch_arxiv_papers(category='cs.CL', max_results=50):
    """
    Fetch metadata for arXiv papers from specified category
    
    Args:
        category: arXiv category (default: cs.CL for Computation and Language)
        max_results: Number of papers to fetch (default: 50)
    """
    # Create directory for PDFs
    Path(input_dir).mkdir(exist_ok=True)
    
    # arXiv API base URL
    base_url = 'http://export.arxiv.org/api/query?'
    
    # Build query parameters
    params = {
        'search_query': f'cat:{category}',
        'start': 0,
        'max_results': max_results,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending'
    }
    
    query_url = base_url + urllib.parse.urlencode(params)
    
    print(f"Fetching {max_results} papers from {category}...")
    print(f"Query URL: {query_url}\n")

    paper_info = {'total': 0, 'metadata': []}
    
    try:
        # Fetch the feed
        with urllib.request.urlopen(query_url) as response:
            data = response.read()
        
        # Parse XML
        root = ET.fromstring(data)
        
        # Define namespace
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        # Find all entries
        entries = root.findall('atom:entry', ns)
        
        print(f"Found {len(entries)} papers\n")

        paper_info['total'] = len(entries)
        
        # Get paper metadata and download each PDF
        for idx, entry in enumerate(entries, 1):
            # Get paper ID
            paper_id = entry.find('atom:id', ns).text.split('/abs/')[-1]
            #print(paper_id)
            # Get title
            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
            # Get publish year
            year = entry.find('atom:published', ns).text.strip()
            year = int(year[0:year.find('-')])
            # Get authors
            authors = ''
            author_elements = entry.findall('atom:author', ns)
            for author in author_elements:
                authors += author.find('atom:name', ns).text.strip() + ', '
            
            # Get PDF link
            pdf_link = None
            for link in entry.findall('atom:link', ns):
                if '/pdf/' in link.get('href'):
                    pdf_link = link.get('href')
                    break
            
            paper_info['metadata'].append({
                'doc_id': paper_id,
                'title': title,
                'author': authors[0:-2],
                'year': year,
                'pdf_link': pdf_link
            })
        # end of for loop 

        print(f"Successfully processed {len(entries)} papers")
        with open(meta_data_file, 'w', encoding='utf-8') as f:
            json.dump(paper_info, f, indent=2, ensure_ascii=False)
        
    except Exception as e:
        print(f"Error fetching papers: {e}")


def extract_text_from_pdf(pdf_data) -> str:
    """
    Open a PDF and extract all text as a single string.
    """

    pdf_bytes = urllib.request.urlopen(pdf_data['pdf_link']).read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page in doc:
        page_text = page.get_text()  # get raw text from page
        # (Optional) clean page_text here (remove headers/footers)
        pages.append(page_text)
    full_text = "\n".join(pages)
    
    return pdf_data['doc_id'], full_text


def chunk_text(text: str, max_tokens: int = 512, overlap: int = 50) -> list[str]:
    tokens = text.split()
    chunks = []
    step = max_tokens - overlap
    for i in range(0, len(tokens), step):
        chunk = tokens[i:i + max_tokens]
        chunks.append(" ".join(chunk))
    return chunks

model = SentenceTransformer('all-MiniLM-L6-v2')
    #all-mpnet-base-v2
    #paraphrase-MiniLM-L6-v2
def embedding(chunks: list[str]):
    embeddings = model.encode(chunks)  # embeds each text chunk into a 384-d vecto
    return embeddings

def faiss_indexing(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # using a simple L2 index
    index.add(np.array(embeddings))  # add all chunk vectors
    return index

def query_faiss(query_text, k=3, f_index=None):
    query_embedding = model.encode(query_text, normalize_embeddings=True)
    if not f_index:
        distances, indices = faiss_index.search(np.array([query_embedding]), k)
    else:
        distances, indices = f_index.search(np.array([query_embedding]), k)
    return distances[0], indices[0]

def query_keyword(query_text, k=3):
    return db.query_doc(
        query_text.replace('?', ' ')
                  .replace('-', ' ')
                  .replace('"', ' ')
                  .replace('\'', ' '), 
        limit=k)

def hybrid_score(vec_score, key_score, alpha=0.5):
    return alpha * vec_score + (1 - alpha) * key_score

# return the merged results from vector search and keyword search
# if top == -1, return the sorted combined results of all, in this case 3 + 3 = 6
# if top > 0, return the top n results without duplicates
def query_hybrid(query_text, top=-1):
    k = 3
    map_size = 200
    # vector search:
    distances, indexes = query_faiss(query_text, map_size)
    # keyword search:
    rows = query_keyword(query_text, map_size)
    # construct faiss_scores map and keyword_scores map
    faiss_results = [] # top k search results from faiss index
    faiss_scores = {} # a map of idx/score(distance) pair for faiss index search
    for i, idx in enumerate(indexes):
        if i < k:
            faiss_results.append((idx, distances[i]))
        faiss_scores[str(idx)] = distances[i]
    keyword_results = [] # top k search results from FTS5 indexed keyword search
    keyword_scores = {} # a map of id/score(rank) pair for FTS5 indexed keyword search
    for i, row in enumerate(rows):
        if i < k:
            keyword_results.append((row['id'], row['rank']))
        keyword_scores[str(row['id'])] = row['rank']

    # the relationship between the idx in faiss_scores map and the id in keyword_scores map is
    # id = idx + 1 since in faiss indexing the index starts from 0, and in FTS5 table
    # the id column (primary key autoincrement) starts from 1
    combined = []
    for idx, v_score in faiss_results:
        last_k_score = list(keyword_scores.items())[-1][1] if keyword_scores else 0.0 # score(rank) of least relevant record in keyword search
        k_score = keyword_scores.get(str(idx+1), last_k_score) # id=idx+1
        if str(idx+1) not in keyword_scores:
            print(f'k_score for ({idx}|{v_score}) not found in keyword_scores map, using {last_k_score}')
        combined_score = hybrid_score(v_score, k_score, alpha=0.6)
        combined.append((idx, combined_score, 'vector'))
    for id, k_score in keyword_results:
        last_v_score = list(faiss_scores.items())[-1][1] if faiss_scores else 0.0 # score(distance) of least relevant record in vector search
        v_score = faiss_scores.get(str(id-1), last_v_score) # idx=id-1
        if str(id-1) not in faiss_scores:
            print(f'v_score for ({id}|{k_score}) not found in faiss_scores map, using {last_v_score}')
        combined_score = hybrid_score(v_score, k_score, alpha=0.6)
        combined.append((id, combined_score, 'keyword'))
    combined.sort(key=lambda x: x[1])
    
    score_results = []
    results = []
    for id, score, search_type in combined:
        score_results.append(search_type + '|' + str(id) + '|' + str(score))
        chunk = ''
        if search_type == 'vector': 
            chunk = chunks[id]
        elif search_type == 'keyword': 
            for row in rows:
                if row['id'] == id:
                    chunk = row['content']
                    break
        results.append(search_type + '|' + str(id) + '|' + str(score) + '|' + chunk)
    if top < 0 or len(combined) == k:
        return score_results, results
    else:
        return top_k_hybrid_results(results, top)
    

def top_k_hybrid_results(hybrid_results, top):
    score_results = []
    results = []
    count = 0
    i = 0
    total = len(hybrid_results)
    while i < total: 
        if count == top:
            break
        if i < total-1 and same_chunk(hybrid_results[i], hybrid_results[i+1]):
            results.append(merge_chunk(hybrid_results[i], hybrid_results[i+1]))
            i += 2
        else:
            results.append(hybrid_results[i])
            i += 1
        count += 1
    #score_results = [s[0:s.rfind('|')] for s in results] #chunk text could contain '|'
    for result in results:
        if re.search(r'^vector\|[0-9]+\|keyword\|[0-9]+', result):
            idx = find_nth_occurrence(result, '|', 5)
        else:
            idx = find_nth_occurrence(result, '|', 3)
        score_results.append(result[0:idx])
    return score_results, results

def find_nth_occurrence(s, char, n):
    positions = [i for i, c in enumerate(s) if c == char]
    return positions[n - 1] if len(positions) >= n else -1

def same_chunk(result1, result2):
    result1_strs = result1.split('|')
    type1 = result1_strs[0]
    id1 = int(result1_strs[1])
    result2_strs = result2.split('|')
    type2 = result2_strs[0]
    id2 = int(result2_strs[1])
    same1 = type1=='vector' and type2=='keyword' and id1 == id2-1
    same2 = type1=='keyword' and type2=='vector' and id1 == id2+1
    return same1 or same2

def merge_chunk(result1, result2):
    result1_strs = result1.split('|')
    type1 = result1_strs[0]
    id1 = int(result1_strs[1])
    result2_strs = result2.split('|')
    type2 = result2_strs[0]
    id2 = int(result2_strs[1])
    score = result2_strs[2]
    chunk = result2_strs[3]
    return f'{type1}|{id1}|{type2}|{id2}|{score}|{chunk}'