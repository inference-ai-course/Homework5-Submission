# Academic LLM Fine-Tuning System - Project Structure

## Directory Structure

```
academic_llm_system/
│
├── config/
│   ├── __init__.py
│   ├── settings.py              # Central configuration
│   └── logging_config.py        # Logging setup
│
├── modules/
│   ├── __init__.py
│   │
│   ├── m1_langchain_llama/      # Module 1: LangChain + LLaMA Integration
│   │   ├── __init__.py
│   │   ├── llm_loader.py        # Model loading utilities
│   │   ├── chain_builder.py     # LangChain chains
│   │   └── memory_manager.py    # Conversation memory
│   │
│   ├── m2_data_collection/      # Module 2: Data Collection & Extraction
│   │   ├── __init__.py
│   │   ├── arxiv_scraper.py     # arXiv paper scraper
│   │   ├── pdf_extractor.py     # PDF text extraction
│   │   ├── ocr_processor.py     # Tesseract OCR
│   │   └── data_cleaner.py      # Text cleaning & dedup
│   │
│   ├── m3_rag_pipeline/         # Module 3: RAG Pipeline
│   │   ├── __init__.py
│   │   ├── chunker.py           # Document chunking
│   │   ├── embedder.py          # Embedding generation
│   │   └── faiss_indexer.py     # FAISS index management
│   │
│   ├── m4_hybrid_retrieval/     # Module 4: Hybrid Retrieval
│   │   ├── __init__.py
│   │   ├── sqlite_fts.py        # SQLite FTS5 search
│   │   ├── vector_search.py     # FAISS vector search
│   │   └── fusion.py            # Score fusion (RRF)
│   │
│   ├── m5_synthetic_data/       # Module 5: Synthetic Data Generation
│   │   ├── __init__.py
│   │   ├── qa_generator.py      # GPT-4 Q&A generation
│   │   ├── dataset_builder.py   # JSONL dataset creation
│   │   └── edge_cases.py        # Edge case examples
│   │
│   ├── m6_fine_tuning/          # Module 6: QLoRA Fine-Tuning
│   │   ├── __init__.py
│   │   ├── qlora_trainer.py     # QLoRA training logic
│   │   ├── model_loader.py      # Model loading with quantization
│   │   └── checkpoint_manager.py # Checkpoint handling
│   │
│   ├── m7_evaluation/           # Module 7: Model Evaluation
│   │   ├── __init__.py
│   │   ├── evaluator.py         # Evaluation metrics
│   │   ├── comparator.py        # Base vs Fine-tuned comparison
│   │   └── report_generator.py  # Generate eval reports
│   │
│   └── m8_api_service/          # Module 8: FastAPI Service
│       ├── __init__.py
│       ├── main.py              # FastAPI app
│       ├── routes/
│       │   ├── search.py        # Search endpoints
│       │   ├── chat.py          # Chat endpoints
│       │   └── admin.py         # Admin endpoints
│       └── schemas.py           # Pydantic models
│
├── storage/
│   ├── data/
│   │   ├── raw/                 # Raw downloaded PDFs
│   │   ├── processed/           # Extracted text
│   │   └── synthetic/           # Generated Q&A data
│   ├── indexes/
│   │   ├── faiss/               # FAISS index files
│   │   └── sqlite/              # SQLite databases
│   └── models/
│       ├── base/                # Base model cache
│       └── finetuned/           # Fine-tuned models
│
├── ui/
│   ├── __init__.py
│   └── gradio_app.py            # Gradio web interface
│
├── scripts/
│   ├── setup_environment.sh     # Environment setup
│   ├── run_pipeline.py          # Full pipeline runner
│   └── evaluate_models.py       # Evaluation script
│
├── tests/
│   ├── test_modules/
│   └── integration/
│
├── requirements.txt
├── setup.py
└── README.md
```

## Environment Setup Script (setup_environment.sh)

```bash
#!/bin/bash
# Run on GPU server after SSH connection

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install core dependencies
pip install -r requirements.txt

# Install Unsloth for efficient fine-tuning
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Verify GPU
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0)}')"

echo "Environment setup complete!"
```

## requirements.txt

```
# Core ML
torch>=2.0.0
transformers>=4.40.0
accelerate>=0.27.0
bitsandbytes>=0.43.0
peft>=0.10.0
datasets>=2.18.0
sentence-transformers>=2.5.0

# LangChain
langchain>=0.1.0
langchain-community>=0.0.20
langchain-huggingface>=0.0.1

# Vector DB & Search
faiss-cpu>=1.7.4
chromadb>=0.4.0

# Data Processing
arxiv>=2.1.0
PyMuPDF>=1.23.0
pdf2image>=1.16.0
pytesseract>=0.3.10
trafilatura>=1.8.0
langdetect>=1.0.9
datasketch>=1.6.0

# API & Web
fastapi>=0.110.0
uvicorn>=0.27.0
gradio>=4.19.0
pydantic>=2.6.0

# Database
sqlalchemy>=2.0.0

# Utilities
python-dotenv>=1.0.0
tqdm>=4.66.0
loguru>=0.7.0
pandas>=2.2.0
numpy>=1.26.0

# OpenAI (for synthetic data generation)
openai>=1.12.0
```
