# scripts/run_pipeline.py
"""Main pipeline runner - orchestrates the complete workflow."""

import argparse
from pathlib import Path
from loguru import logger
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config.settings import get_config, DATA_DIR, MODEL_DIR


def run_data_collection(config, num_papers: int = 50, category: str = "cs.CL"):
    """Step 1: Collect papers from arXiv."""
    logger.info("=" * 50)
    logger.info("STEP 1: Data Collection")
    logger.info("=" * 50)
    
    from modules.m2_data_collection import ArxivScraper, PDFExtractor, DataCleaner
    
    # Scrape papers
    scraper = ArxivScraper(config)
    papers = scraper.search_papers(category=category, max_results=num_papers)
    scraper.download_pdfs(papers)
    scraper.save_metadata()
    
    # Extract text
    extractor = PDFExtractor()
    pdf_paths = [p.local_pdf_path for p in papers if p.local_pdf_path]
    metadata = [{"title": p.title, "arxiv_id": p.arxiv_id, "abstract": p.abstract} for p in papers if p.local_pdf_path]
    
    documents = extractor.extract_batch(pdf_paths, metadata)
    extractor.save_extracted()
    
    # Clean data
    cleaner = DataCleaner()
    docs_dict = [{"arxiv_id": d.arxiv_id, "title": d.title, "full_text": d.full_text} for d in documents]
    cleaned_docs = cleaner.process_batch(docs_dict)
    
    logger.info(f"Collected and processed {len(cleaned_docs)} papers")
    return cleaned_docs


def run_rag_indexing(config, documents: list):
    """Step 2: Build RAG index."""
    logger.info("=" * 50)
    logger.info("STEP 2: RAG Indexing")
    logger.info("=" * 50)
    
    from modules.m3_rag_pipeline import DocumentChunker, EmbeddingGenerator, FAISSIndexer
    from modules.m4_hybrid_retrieval import SQLiteFTS
    
    # Chunk documents
    chunker = DocumentChunker(config)
    all_chunks = chunker.chunk_batch(documents)
    
    # Generate embeddings
    embedder = EmbeddingGenerator(config=config)
    embedder.load_model()
    embeddings = embedder.embed_chunks(all_chunks)
    
    # Build FAISS index
    indexer = FAISSIndexer(embedder.get_dimension(), config)
    indexer.create_index()
    indexer.add_vectors(embeddings, all_chunks)
    indexer.save("academic_index")
    
    # Build SQLite FTS
    fts = SQLiteFTS()
    fts.connect()
    fts.create_tables()
    fts.add_chunks_batch(all_chunks)
    fts.close()
    
    logger.info(f"Indexed {len(all_chunks)} chunks")
    return indexer, embedder


def run_synthetic_generation(config, num_papers: int = 50, qa_per_paper: int = 5):
    """Step 3: Generate synthetic Q&A data."""
    logger.info("=" * 50)
    logger.info("STEP 3: Synthetic Data Generation")
    logger.info("=" * 50)
    
    from modules.m2_data_collection import ArxivScraper
    from modules.m5_synthetic_data import QAGenerator, DatasetBuilder
    
    # Load paper metadata
    scraper = ArxivScraper(config)
    papers = scraper.load_metadata()[:num_papers]
    
    papers_dict = [
        {"arxiv_id": p.arxiv_id, "title": p.title, "abstract": p.abstract}
        for p in papers
    ]
    
    # Generate Q&A
    generator = QAGenerator(config)
    qa_pairs = generator.generate_batch(papers_dict, qa_per_paper=qa_per_paper)
    
    # Build dataset
    builder = DatasetBuilder(config)
    dataset = builder.build_instruct_dataset(qa_pairs)
    filepath = builder.save_jsonl(dataset)
    
    logger.info(f"Generated {len(qa_pairs)} Q&A pairs, saved to {filepath}")
    return filepath


def run_finetuning(config, data_path: str = None):
    """Step 4: QLoRA fine-tuning."""
    logger.info("=" * 50)
    logger.info("STEP 4: QLoRA Fine-Tuning")
    logger.info("=" * 50)
    
    from modules.m6_fine_tuning import QLoRATrainer
    
    trainer = QLoRATrainer(config)
    results = trainer.run_full_pipeline(data_path)
    
    logger.info(f"Fine-tuning complete! Loss: {results['train_loss']:.4f}")
    return results


def run_evaluation(config, test_questions: list = None):
    """Step 5: Model evaluation and comparison."""
    logger.info("=" * 50)
    logger.info("STEP 5: Evaluation")
    logger.info("=" * 50)
    
    from modules.m1_langchain_llama import LLMLoader
    from modules.m7_evaluation import ModelComparator, ReportGenerator
    
    # Default test questions
    if test_questions is None:
        test_questions = [
            {"question": "What is the transformer architecture?", 
             "answer": "The transformer is a neural network architecture based on self-attention mechanisms."},
            {"question": "How does BERT differ from GPT?",
             "answer": "BERT uses bidirectional training while GPT uses unidirectional left-to-right training."},
            {"question": "What is attention mechanism in neural networks?",
             "answer": "Attention allows models to focus on relevant parts of the input when producing output."},
            {"question": "Explain the concept of fine-tuning in NLP.",
             "answer": "Fine-tuning adapts a pre-trained model to a specific task using task-specific data."},
            {"question": "What are word embeddings?",
             "answer": "Word embeddings are dense vector representations of words that capture semantic meaning."},
            {"question": "How does RAG improve language models?",
             "answer": "RAG combines retrieval with generation to ground responses in external knowledge."},
            {"question": "What is the purpose of the softmax function?",
             "answer": "Softmax converts logits into probability distributions over classes."},
            {"question": "Explain cross-entropy loss.",
             "answer": "Cross-entropy measures the difference between predicted and true probability distributions."},
            {"question": "What is tokenization in NLP?",
             "answer": "Tokenization splits text into smaller units like words or subwords for processing."},
            {"question": "How do language models handle out-of-vocabulary words?",
             "answer": "Modern models use subword tokenization like BPE to handle unknown words."},
        ]
    
    # Load models
    base_loader = LLMLoader(config)
    base_model, base_tok = base_loader.load_base_model()
    
    ft_loader = LLMLoader(config)
    ft_model, ft_tok = ft_loader.load_finetuned_model()
    
    # Compare
    comparator = ModelComparator(config)
    comparator.setup_models(base_model, base_tok, ft_model, ft_tok)
    
    results, stats = comparator.compare_batch(test_questions)
    
    # Generate report
    reporter = ReportGenerator()
    report = reporter.generate_comparison_report(
        results, stats,
        model_info={
            "base_model": config.model.base_model_name,
            "finetuned_model": config.training.output_dir
        }
    )
    report_path = reporter.save_report(report)
    reporter.save_json_results(results, stats)
    
    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Total Questions: {stats['total_questions']}")
    logger.info(f"Fine-tuned Wins: {stats['finetuned_wins']} ({stats['finetuned_win_rate']:.1%})")
    logger.info(f"Base Model Wins: {stats['base_wins']}")
    logger.info(f"Ties: {stats['ties']}")
    logger.info(f"Improvement: {stats['improvement_percent']:.1f}%")
    logger.info(f"Report saved to: {report_path}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Academic LLM Pipeline Runner")
    parser.add_argument("--step", choices=["all", "collect", "index", "synthetic", "train", "eval"],
                       default="all", help="Pipeline step to run")
    parser.add_argument("--papers", type=int, default=50, help="Number of papers to collect")
    parser.add_argument("--category", default="cs.CL", help="arXiv category")
    parser.add_argument("--qa-per-paper", type=int, default=5, help="Q&A pairs per paper")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs")
    
    args = parser.parse_args()
    config = get_config()
    
    # Update config
    config.data.num_papers = args.papers
    config.data.arxiv_category = args.category
    config.data.qa_pairs_per_paper = args.qa_per_paper
    config.training.num_train_epochs = args.epochs
    
    logger.info("Starting Academic LLM Pipeline")
    logger.info(f"Step: {args.step}")
    
    if args.step in ["all", "collect"]:
        documents = run_data_collection(config, args.papers, args.category)
        
    if args.step in ["all", "index"]:
        if args.step != "all":
            # Load existing documents
            from modules.m2_data_collection import ArxivScraper
            scraper = ArxivScraper(config)
            papers = scraper.load_metadata()
            documents = [{"arxiv_id": p.arxiv_id, "title": p.title} for p in papers]
        run_rag_indexing(config, documents)
        
    if args.step in ["all", "synthetic"]:
        run_synthetic_generation(config, args.papers, args.qa_per_paper)
        
    if args.step in ["all", "train"]:
        run_finetuning(config)
        
    if args.step in ["all", "eval"]:
        run_evaluation(config)
    
    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()


# ============================================================
# scripts/quick_test.py - Quick test script for GPU verification
# ============================================================
"""
Quick test to verify GPU setup and model loading.
Run this first to ensure your environment is correctly configured.
"""

def quick_test():
    """Run quick verification tests."""
    import torch
    
    print("=" * 50)
    print("GPU & Environment Verification")
    print("=" * 50)
    
    # GPU Check
    print(f"\n1. GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Package imports
    print("\n2. Checking package imports...")
    packages = [
        "transformers", "peft", "bitsandbytes", "datasets",
        "sentence_transformers", "faiss", "langchain",
        "gradio", "fastapi", "arxiv"
    ]
    
    for pkg in packages:
        try:
            __import__(pkg.replace("-", "_"))
            print(f"   ✅ {pkg}")
        except ImportError:
            print(f"   ❌ {pkg} - Not installed")
    
    # Quick model test (optional - takes time)
    print("\n3. Testing embedding model...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embedding = model.encode("Test sentence")
        print(f"   ✅ Embedding model works! Dim: {len(embedding)}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("Verification complete!")
    print("=" * 50)


if __name__ == "__main__":
    quick_test()
