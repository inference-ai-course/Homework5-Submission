# Academic LLM Fine-Tuning System

An end-to-end system for building an academic Q&A assistant using RAG (Retrieval-Augmented Generation) and QLoRA fine-tuning with LLaMA 3.

## 🚀 Quick Start

### Prerequisites
- GPU server with 16GB+ VRAM
- Python 3.10+
- CUDA support

## 📋 Pipeline

1. **Data Collection** - Scrape papers from arXiv
2. **RAG Indexing** - Build FAISS + SQLite hybrid search
3. **Synthetic Data** - Generate Q&A pairs using GPT-4
4. **Fine-Tuning** - QLoRA fine-tune LLaMA 3.1 8B
5. **Evaluation** - Compare base vs fine-tuned models

## 🎯 Usage

### Full Pipeline
```bash
python pipeline-runner.py --step all --papers 50 --epochs 2
```

### Individual Steps
```bash
python pipeline-runner.py --step collect   # Collect papers
python pipeline-runner.py --step index     # Build RAG index
python pipeline-runner.py --step synthetic # Generate Q&A data
python pipeline-runner.py --step train     # Fine-tune model
python pipeline-runner.py --step eval      # Evaluate models
```

### Web UI
```bash
python gradio-ui.py
```

## 📁 Project Structure

```
├── config-module.py              # Configuration
├── module1-langchain.py          # LangChain + LLaMA integration
├── module2-data.py               # Data collection & extraction
├── module3-rag.py                # RAG pipeline
├── module4-5-hybrid-synthetic.py # Hybrid retrieval + synthetic data
├── module6-7-finetune-eval.py    # QLoRA fine-tuning + evaluation
├── module8-api.py                # FastAPI service
├── pipeline-runner.py            # Main pipeline orchestrator
└── gradio-ui.py                  # Web interface
```

## 🔧 Key Technologies

- **Base Model:** Meta-Llama-3.1-8B-Instruct
- **Fine-Tuning:** QLoRA (4-bit quantization)
- **RAG:** FAISS + SQLite FTS5 (hybrid search)
- **Framework:** LangChain, PyTorch, Transformers

## 📝 Notes

- Project files location: `/home/jovyan/work/`
- Persistent storage: `/home/jovyan/work/` (50GB)
- GPU: inference-ai GPU cuda (16GB VRAM)
