import util
import os
import json
from pathlib import Path
import util.util as util
from util.databaseUtil import ArXivDatabase


input_dir = 'arxiv_pdfs/'
output_dir = 'output/'

def chunk_json(chunks: list):
    ret = []
    for idx, chunk in enumerate(chunks, 1):
        ret.append({'id': idx, 'chunk': chunk})
    return ret

def faiss_report(faiss_index):
    query_texts = [
        "How to do end to end speech translation?",
        "What is Semantic Matching of Documents?",
        "What are the top works on SQuAD leaderboards?",
        "Can you summarize what BERT-BiDAF hybrid architecture is?",
        "What is Unsupervised Data Augmentation for Consistency Training?"
        ]

    report = 'The following are the 5 questions asked and the top-3 retrieved passages for each.\n\n'
    for id, query_text in enumerate(query_texts, 1):
        query = f'\nQuery {id}: {query_text}\n\n'
        #print(query)
        report += query
        distances, indexes = util.query_faiss(query_text, f_index=faiss_index)
        answer = f'Answer: \n\n'
        #print(answer)
        report += answer
        for index in indexes:
            chunk = f'{chunks[index]}\n\n'
            #print(chunk)
            report += chunk

    with open(output_dir + 'report_faiss.txt', 'w') as f:
        f.write(report)
    print(f"report file for faiss index search is generated in '{output_dir}' directory")


if __name__ == "__main__":
    # Fetch 50 cs.CL papers
    
    if not os.path.exists(input_dir) or (not Path(input_dir).glob('*.pdf')):
        util.fetch_arxiv_papers(category='cs.CL', max_results=50)

    # Get all PDF files
    pdf_files = sorted(Path(input_dir).glob('*.pdf'))
    chunks = []
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    papers = []
    
    for pdf_file in pdf_files:
        id, text = util.extract_text_from_pdf(pdf_file)
        temp_chunks = util.chunk_text(text)
        papers.append({'paper': pdf_file.name, 'chunks': chunk_json(temp_chunks)})
        chunks.extend(temp_chunks)

    #Generate paper_chunks.json
    with open(output_dir + 'paper_chunks.json', 'w') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    print(f"json file is generated in '{output_dir}' directory")

    embeddings = util.embedding(chunks)
    faiss_index = util.faiss_indexing(embeddings)

    #Generate faiss index file my_faiss_index.index
    faiss.write_index(faiss_index, output_dir + "my_faiss_index.index")
    print(f"faiss index file is generated in '{output_dir}' directory")

    #faiss_report(faiss_index)

    