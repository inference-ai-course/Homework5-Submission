# modules/m2_data_collection/__init__.py
"""Module 2: Data Collection and Extraction."""

from .arxiv_scraper import ArxivScraper
from .pdf_extractor import PDFExtractor
from .data_cleaner import DataCleaner

__all__ = ["ArxivScraper", "PDFExtractor", "DataCleaner"]


# modules/m2_data_collection/arxiv_scraper.py
"""arXiv paper scraping and downloading."""

import arxiv
import requests
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from tqdm import tqdm
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import get_config, DATA_DIR


@dataclass
class PaperMetadata:
    """Metadata for an arXiv paper."""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published: str
    updated: str
    pdf_url: str
    local_pdf_path: Optional[str] = None


class ArxivScraper:
    """Scrapes and downloads papers from arXiv."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.raw_dir = DATA_DIR / "raw"
        self.metadata_file = DATA_DIR / "processed" / "papers_metadata.json"
        self.papers: List[PaperMetadata] = []
        
    def search_papers(
        self, 
        category: Optional[str] = None,
        query: Optional[str] = None,
        max_results: Optional[int] = None
    ) -> List[PaperMetadata]:
        """Search arXiv for papers."""
        category = category or self.config.data.arxiv_category
        max_results = max_results or self.config.data.num_papers
        
        # Build search query
        if query:
            search_query = f"cat:{category} AND ({query})"
        else:
            search_query = f"cat:{category}"
        
        logger.info(f"Searching arXiv: {search_query}, max_results={max_results}")
        
        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        client = arxiv.Client()
        results = list(client.results(search))
        
        self.papers = []
        for result in results:
            paper = PaperMetadata(
                arxiv_id=result.entry_id.split("/")[-1],
                title=result.title.replace("\n", " "),
                authors=[author.name for author in result.authors],
                abstract=result.summary.replace("\n", " "),
                categories=result.categories,
                published=result.published.isoformat(),
                updated=result.updated.isoformat(),
                pdf_url=result.pdf_url
            )
            self.papers.append(paper)
        
        logger.info(f"Found {len(self.papers)} papers")
        return self.papers
    
    def download_pdfs(self, papers: Optional[List[PaperMetadata]] = None) -> List[str]:
        """Download PDFs for all papers."""
        papers = papers or self.papers
        downloaded_paths = []
        
        logger.info(f"Downloading {len(papers)} PDFs...")
        
        for paper in tqdm(papers, desc="Downloading PDFs"):
            try:
                # Create filename from arxiv_id
                filename = f"{paper.arxiv_id.replace('/', '_')}.pdf"
                filepath = self.raw_dir / filename
                
                if filepath.exists():
                    logger.debug(f"Already exists: {filename}")
                    paper.local_pdf_path = str(filepath)
                    downloaded_paths.append(str(filepath))
                    continue
                
                # Download PDF
                response = requests.get(paper.pdf_url, timeout=60)
                response.raise_for_status()
                
                filepath.write_bytes(response.content)
                paper.local_pdf_path = str(filepath)
                downloaded_paths.append(str(filepath))
                
                logger.debug(f"Downloaded: {filename}")
                
            except Exception as e:
                logger.error(f"Failed to download {paper.arxiv_id}: {e}")
        
        logger.info(f"Downloaded {len(downloaded_paths)} PDFs")
        return downloaded_paths
    
    def save_metadata(self, filepath: Optional[Path] = None):
        """Save paper metadata to JSON."""
        filepath = filepath or self.metadata_file
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "scraped_at": datetime.now().isoformat(),
            "category": self.config.data.arxiv_category,
            "total_papers": len(self.papers),
            "papers": [asdict(p) for p in self.papers]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved metadata to {filepath}")
    
    def load_metadata(self, filepath: Optional[Path] = None) -> List[PaperMetadata]:
        """Load paper metadata from JSON."""
        filepath = filepath or self.metadata_file
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.papers = [PaperMetadata(**p) for p in data["papers"]]
        logger.info(f"Loaded {len(self.papers)} papers from {filepath}")
        return self.papers


# modules/m2_data_collection/pdf_extractor.py
"""PDF text extraction using PyMuPDF."""

import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
from tqdm import tqdm
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR


@dataclass
class ExtractedDocument:
    """Extracted document with text and metadata."""
    arxiv_id: str
    title: str
    full_text: str
    pages: List[str]
    num_pages: int
    source_path: str


class PDFExtractor:
    """Extracts text from PDF files."""
    
    def __init__(self):
        self.raw_dir = DATA_DIR / "raw"
        self.processed_dir = DATA_DIR / "processed"
        self.documents: List[ExtractedDocument] = []
        
    def extract_single(self, pdf_path: str, metadata: Optional[Dict] = None) -> ExtractedDocument:
        """Extract text from a single PDF."""
        pdf_path = Path(pdf_path)
        
        try:
            doc = fitz.open(pdf_path)
            pages = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                # Clean up text
                text = self._clean_page_text(text)
                pages.append(text)
            
            full_text = "\n\n".join(pages)
            arxiv_id = pdf_path.stem.replace("_", "/")
            
            extracted = ExtractedDocument(
                arxiv_id=arxiv_id,
                title=metadata.get("title", arxiv_id) if metadata else arxiv_id,
                full_text=full_text,
                pages=pages,
                num_pages=len(pages),
                source_path=str(pdf_path)
            )
            
            doc.close()
            return extracted
            
        except Exception as e:
            logger.error(f"Failed to extract {pdf_path}: {e}")
            raise
    
    def _clean_page_text(self, text: str) -> str:
        """Clean extracted page text."""
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def extract_batch(
        self, 
        pdf_paths: List[str], 
        metadata_list: Optional[List[Dict]] = None
    ) -> List[ExtractedDocument]:
        """Extract text from multiple PDFs."""
        self.documents = []
        
        for i, pdf_path in enumerate(tqdm(pdf_paths, desc="Extracting PDFs")):
            try:
                metadata = metadata_list[i] if metadata_list else None
                doc = self.extract_single(pdf_path, metadata)
                self.documents.append(doc)
            except Exception as e:
                logger.warning(f"Skipping {pdf_path}: {e}")
        
        logger.info(f"Extracted {len(self.documents)} documents")
        return self.documents
    
    def save_extracted(self, output_dir: Optional[Path] = None):
        """Save extracted documents to JSON files."""
        output_dir = output_dir or (self.processed_dir / "extracted")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for doc in self.documents:
            filename = doc.arxiv_id.replace("/", "_") + ".json"
            filepath = output_dir / filename
            
            data = {
                "arxiv_id": doc.arxiv_id,
                "title": doc.title,
                "full_text": doc.full_text,
                "num_pages": doc.num_pages,
                "source_path": doc.source_path
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.documents)} documents to {output_dir}")


# modules/m2_data_collection/data_cleaner.py
"""Data cleaning and deduplication."""

import re
from typing import List, Set
from langdetect import detect, LangDetectException
from datasketch import MinHash, MinHashLSH
from loguru import logger


class DataCleaner:
    """Cleans and deduplicates text data."""
    
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.lsh = MinHashLSH(threshold=similarity_threshold, num_perm=128)
        self.seen_hashes: Set[str] = set()
        
    def clean_text(self, text: str) -> str:
        """Apply all cleaning operations to text."""
        text = self._remove_html(text)
        text = self._remove_pii(text)
        text = self._remove_special_chars(text)
        text = self._normalize_whitespace(text)
        return text
    
    def _remove_html(self, text: str) -> str:
        """Remove HTML tags."""
        return re.sub(r'<[^>]+>', '', text)
    
    def _remove_pii(self, text: str) -> str:
        """Remove PII (emails, phone numbers, credit cards)."""
        # Email
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        # Phone numbers (various formats)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        # Credit card numbers
        text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD]', text)
        return text
    
    def _remove_special_chars(self, text: str) -> str:
        """Remove special characters but keep essential punctuation."""
        # Keep alphanumeric, common punctuation, and whitespace
        text = re.sub(r'[^\w\s.,!?;:\-\'\"()\[\]{}]', ' ', text)
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace."""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def detect_language(self, text: str) -> str:
        """Detect language of text."""
        try:
            return detect(text[:1000])  # Use first 1000 chars
        except LangDetectException:
            return "unknown"
    
    def is_english(self, text: str) -> bool:
        """Check if text is English."""
        return self.detect_language(text) == "en"
    
    def _get_minhash(self, text: str) -> MinHash:
        """Create MinHash from text."""
        m = MinHash(num_perm=128)
        words = text.lower().split()
        for word in words:
            m.update(word.encode('utf-8'))
        return m
    
    def is_duplicate(self, text: str, doc_id: str) -> bool:
        """Check if text is duplicate using MinHash LSH."""
        m = self._get_minhash(text)
        
        # Query for similar documents
        result = self.lsh.query(m)
        
        if result:
            return True
        
        # Add to LSH index
        self.lsh.insert(doc_id, m)
        return False
    
    def deduplicate_batch(self, documents: List[dict]) -> List[dict]:
        """Remove duplicate documents from a batch."""
        unique_docs = []
        
        for doc in documents:
            doc_id = doc.get("arxiv_id", str(len(unique_docs)))
            text = doc.get("full_text", doc.get("abstract", ""))
            
            if not self.is_duplicate(text, doc_id):
                unique_docs.append(doc)
        
        removed = len(documents) - len(unique_docs)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate documents")
        
        return unique_docs
    
    def process_batch(self, documents: List[dict]) -> List[dict]:
        """Full processing pipeline for documents."""
        processed = []
        
        for doc in documents:
            # Clean text
            if "full_text" in doc:
                doc["full_text"] = self.clean_text(doc["full_text"])
            if "abstract" in doc:
                doc["abstract"] = self.clean_text(doc["abstract"])
            
            # Language filter
            text = doc.get("full_text", doc.get("abstract", ""))
            if not self.is_english(text):
                logger.debug(f"Skipping non-English: {doc.get('arxiv_id', 'unknown')}")
                continue
            
            processed.append(doc)
        
        # Deduplicate
        processed = self.deduplicate_batch(processed)
        
        logger.info(f"Processed {len(processed)} documents (from {len(documents)})")
        return processed
