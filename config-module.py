# config/settings.py
"""Central configuration for the Academic LLM System."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
STORAGE_DIR = BASE_DIR / "storage"
DATA_DIR = STORAGE_DIR / "data"
INDEX_DIR = STORAGE_DIR / "indexes"
MODEL_DIR = STORAGE_DIR / "models"

# Ensure directories exist
for d in [DATA_DIR / "raw", DATA_DIR / "processed", DATA_DIR / "synthetic",
          INDEX_DIR / "faiss", INDEX_DIR / "sqlite", 
          MODEL_DIR / "base", MODEL_DIR / "finetuned"]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """LLM Model configuration."""
    base_model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    device_map: str = "auto"
    
    # Fine-tuning params
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])


@dataclass
class TrainingConfig:
    """Training configuration for fine-tuning."""
    output_dir: str = str(MODEL_DIR / "finetuned" / "llama3-academic-qa")
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_grad_norm: float = 0.3
    weight_decay: float = 0.001
    optim: str = "paged_adamw_32bit"
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 25
    save_strategy: str = "epoch"
    fp16: bool = False
    bf16: bool = True  # Better for newer GPUs
    seed: int = 42


@dataclass
class DataConfig:
    """Data collection and processing configuration."""
    arxiv_category: str = "cs.CL"  # Computational Linguistics
    num_papers: int = 50
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_length: int = 100
    
    # Synthetic data
    qa_pairs_per_paper: int = 5
    include_edge_cases: bool = True
    edge_case_ratio: float = 0.1


@dataclass
class RAGConfig:
    """RAG pipeline configuration."""
    faiss_index_type: str = "IndexFlatIP"  # Inner product for cosine sim
    top_k_retrieval: int = 5
    rerank_top_k: int = 3
    similarity_threshold: float = 0.7
    
    # Hybrid search
    vector_weight: float = 0.6
    keyword_weight: float = 0.4
    use_rrf: bool = True  # Reciprocal Rank Fusion


@dataclass
class APIConfig:
    """API and service configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    gradio_port: int = 7860
    workers: int = 1
    
    # External APIs
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))


@dataclass
class SystemConfig:
    """Main system configuration aggregating all configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    api: APIConfig = field(default_factory=APIConfig)
    
    # System prompt for the academic assistant
    system_prompt: str = """You are an expert academic research assistant specialized in computer science and NLP research. 
You provide accurate, well-structured answers based on scientific literature. 
When answering questions:
1. Be precise and cite specific concepts from papers when relevant
2. Acknowledge uncertainty when information is not available
3. Use appropriate academic terminology
4. Provide context and explanations suitable for researchers"""


# Global config instance
config = SystemConfig()


def get_config() -> SystemConfig:
    """Get the global configuration instance."""
    return config


def update_config(**kwargs) -> SystemConfig:
    """Update configuration with custom values."""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config
