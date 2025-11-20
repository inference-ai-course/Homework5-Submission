# Initialize scraper with 2-second delay between requests
from src.homework.rag import RAGClass
from src.homework.web_scraper import ArxivScraper
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

list_urls = [
    "https://arxiv.org/list/econ.EM/recent"
]
    
@app.get("/search/")
async def search(query: str, k = 3):
    scraper = ArxivScraper(delay=2)
    article = scraper.scrape_article_content("https://arxiv.org/html/2511.10995v1")
    scraper.save_articles([article], "./src/homework/arxiv_articles.txt")
    # articles = scraper.scrape_articles_from_lists(list_urls, './src/homework/arxiv_articles.txt')
    rag = RAGClass(data_path="./src/homework/arxiv_articles.txt")

    rag.load_documents()
    rag.split_documents()
    rag.convert_text_to_embeddings_with_faiss()
    
    # Search for similar chunks
    results = rag.search_embeddings(query, k)
    
    # Return results as JSON
    return JSONResponse(content={
        "query": query,
        "results": results
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)