"""
PDF Processing Module for RAG Pipeline - Using pypdf (no compilation needed)
Alternative to PyMuPDF that works on all systems without build tools
"""

from pypdf import PdfReader
from typing import List, Dict
import re
import json
from pathlib import Path


class PDFProcessor:
    def __init__(self, max_tokens: int = 512, overlap: int = 50):
        """
        Initialize PDF processor with chunking parameters
        
        Args:
            max_tokens: Maximum tokens per chunk
            overlap: Number of overlapping tokens between chunks
        """
        self.max_tokens = max_tokens
        self.overlap = overlap
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Open a PDF and extract all text as a single string using pypdf.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Full text content of the PDF
        """
        try:
            reader = PdfReader(pdf_path)
            pages = []
            
            for page in reader.pages:
                page_text = page.extract_text()
                # Basic cleaning: remove excessive whitespace
                page_text = re.sub(r'\s+', ' ', page_text)
                pages.append(page_text)
            
            full_text = "\n".join(pages)
            return full_text
        
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing artifacts and normalizing
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned text
        """
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers (common patterns)
        text = re.sub(r'\n\d+\n', '\n', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks based on token count
        
        Args:
            text: Full text to chunk
            
        Returns:
            List of text chunks
        """
        # Simple tokenization by splitting on whitespace
        tokens = text.split()
        chunks = []
        step = self.max_tokens - self.overlap
        
        for i in range(0, len(tokens), step):
            chunk = tokens[i:i + self.max_tokens]
            chunk_text = " ".join(chunk)
            
            # Only add chunks with meaningful content (>50 tokens)
            if len(chunk) > 50:
                chunks.append(chunk_text)
        
        return chunks
    
    def process_pdf(self, pdf_path: str, paper_id: str = None) -> List[Dict]:
        """
        Complete pipeline: extract, clean, and chunk a PDF
        
        Args:
            pdf_path: Path to PDF file
            paper_id: Optional paper identifier (e.g., arXiv ID)
            
        Returns:
            List of dictionaries containing chunk info
        """
        # Extract paper ID from filename if not provided
        if paper_id is None:
            paper_id = Path(pdf_path).stem
        
        # Extract and clean text
        raw_text = self.extract_text_from_pdf(pdf_path)
        if not raw_text:
            return []
        
        clean_text = self.clean_text(raw_text)
        
        # Create chunks
        chunks = self.chunk_text(clean_text)
        
        # Package chunks with metadata
        chunk_data = []
        for idx, chunk_text in enumerate(chunks):
            chunk_data.append({
                'paper_id': paper_id,
                'chunk_id': f"{paper_id}_chunk_{idx}",
                'chunk_index': idx,
                'text': chunk_text,
                'token_count': len(chunk_text.split())
            })
        
        return chunk_data
    
    def process_directory(self, pdf_dir: str, output_file: str = "processed_chunks.json") -> List[Dict]:
        """
        Process all PDFs in a directory
        
        Args:
            pdf_dir: Directory containing PDF files
            output_file: Output JSON file for chunks
            
        Returns:
            List of all chunks from all PDFs
        """
        pdf_path = Path(pdf_dir)
        all_chunks = []
        
        pdf_files = list(pdf_path.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file.name}")
            chunks = self.process_pdf(str(pdf_file))
            all_chunks.extend(chunks)
            print(f"  -> Generated {len(chunks)} chunks")
        
        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
        print(f"\nTotal chunks: {len(all_chunks)}")
        print(f"Saved to: {output_file}")
        
        return all_chunks


def main():
    """Example usage"""
    processor = PDFProcessor(max_tokens=512, overlap=50)
    
    # Process a single PDF
    # chunks = processor.process_pdf("path/to/paper.pdf", paper_id="2401.12345")
    
    # Or process entire directory
    chunks = processor.process_directory("./arxiv_pdfs", "processed_chunks.json")
    
    # Print sample
    if chunks:
        print("\n" + "=" * 60)
        print("Sample Chunk:")
        print("=" * 60)
        print(f"Paper ID: {chunks[0]['paper_id']}")
        print(f"Chunk ID: {chunks[0]['chunk_id']}")
        print(f"Token Count: {chunks[0]['token_count']}")
        print(f"Text Preview: {chunks[0]['text'][:200]}...")
        print("=" * 60)


if __name__ == "__main__":
    main()