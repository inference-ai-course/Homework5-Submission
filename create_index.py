"""
Create FAISS index from saved embeddings.
"""

import numpy as np
import faiss
from pathlib import Path

INDEX_DIR = Path("data/index")

# Load embeddings
embeddings_path = INDEX_DIR / "embeddings.npy"
print(f"Loading embeddings from {embeddings_path}...")
embeddings = np.load(embeddings_path)
print(f"Loaded embeddings shape: {embeddings.shape}")

# Build FAISS index
print("\nBuilding FAISS index...")
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)

# Normalize embeddings for better cosine similarity
faiss.normalize_L2(embeddings)

# Add embeddings to index
index.add(embeddings.astype('float32'))
print(f"Index built with {index.ntotal} vectors")

# Save FAISS index
index_path = INDEX_DIR / "faiss_index.bin"
faiss.write_index(index, str(index_path))
print(f"\nSaved FAISS index to {index_path}")
print("Done!")
