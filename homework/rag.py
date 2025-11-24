import numpy as np
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.agents import create_agent
from langchain_huggingface import HuggingFaceEmbeddings
import faiss


class RAGClass:
    def __init__(self, documents: list):
        """
        Initialize the RAGClass with the path to the data file.
        """
        self.documents = documents
        self.text_chunks = []
        self.faiss_index = None
        self.embedding_model = HuggingFaceEmbeddings(model="sentence-transformers/all-mpnet-base-v2");

    def split_documents(self, chunk_size=256, chunk_overlap=10):
        """
        Splits loaded documents into smaller chunks for processing.
        Returns the list of text chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
        self.text_chunks = text_splitter.split_documents(self.documents)
        print(f"Split documents into {len(self.text_chunks)} chunks.")
        for i, chunk in enumerate(self.text_chunks):
            formatted_text = chunk.page_content.replace('. ', '.')
            print(f"Chunk {i+1}:{formatted_text}")
        return self.text_chunks


    def convert_text_to_embeddings_with_faiss(self, index_type="flat"):
        """
        Converts text to embeddings and indexes them using FAISS.
        
        Args:
            texts: A list of text strings or a single text string to convert to embeddings
            embedding_model: An embedding model instance (e.g., OllamaEmbeddings). 
                            If None, uses OllamaEmbeddings with "llama3" model.
            index_type: Type of FAISS index to create. Options: "flat" (exact search) or "l2" (L2 distance).
                    Default is "flat".
        
        Returns:
            tuple: (faiss_index, embeddings) where:
                - faiss_index: A FAISS index containing the embeddings
                - embeddings: A numpy array of the embeddings
        
        Example:
            >>> texts = ["Hello world", "FAISS is great"]
            >>> index, embeddings = convert_text_to_embeddings_with_faiss(texts)
            >>> # Search for similar vectors
            >>> query_embedding = embedding_model.embed_query("Hello")
            >>> distances, indices = index.search(np.array([query_embedding]), k=2)
        """
        
        # Convert texts to embeddings
        # Extract text content from Document objects
        texts = [chunk.page_content for chunk in self.text_chunks]
        print(f"Converting {len(texts)} text(s) to embeddings...")
        embeddings = self.embedding_model.embed_documents(texts)
        embeddings = np.array(embeddings).astype('float32')
        
        # Get embedding dimension
        dimension = embeddings.shape[1]
        
        # Create FAISS index
        if index_type == "flat":
            # Flat index for exact search (L2 distance)
            index = faiss.IndexFlatL2(dimension)
        elif index_type == "l2":
            # Same as flat for L2 distance
            index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unknown index_type: {index_type}. Use 'flat' or 'l2'.")
        
        # Add embeddings to the index
        index.add(embeddings)
        
        # Store index for later use
        self.faiss_index = index
        
        print(f"Created FAISS index with {index.ntotal} vectors of dimension {dimension}")
        
        return index, embeddings
    
    def search_embeddings(self, query: str, k: int = 3):
        """
        Search for similar chunks using a query string.
        
        Args:
            query: The search query string
            k: Number of similar chunks to return (default: 5)
        
        Returns:
            list: List of dictionaries containing chunk text, distance, and index
        """
        k = int(k)
        if self.faiss_index is None:
            raise ValueError("FAISS index not initialized. Please call convert_text_to_embeddings_with_faiss() first.")
        
        if self.embedding_model is None:
            raise ValueError("Embedding model not initialized. Please call convert_text_to_embeddings_with_faiss() first.")
        
        if not self.text_chunks:
            raise ValueError("Text chunks not available. Please call split_documents() first.")
        
        # Convert query to embedding
        query_embedding = self.embedding_model.embed_query(query)
        query_vector = np.array([query_embedding]).astype('float32')
        
        # Search the index
        distances, indices = self.faiss_index.search(query_vector, k)
        
        # Get the matching chunks
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.text_chunks):
                chunk = self.text_chunks[idx]
                results.append({
                    "chunk_text": chunk.page_content,
                    "distance": float(distance),
                    "index": int(idx),
                    "metadata": chunk.metadata
                })
        
        return results


