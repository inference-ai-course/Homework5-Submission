# modules/m6_fine_tuning/__init__.py
"""Module 6: QLoRA Fine-Tuning."""

from .qlora_trainer import QLoRATrainer
from .model_loader import FineTuneModelLoader

__all__ = ["QLoRATrainer", "FineTuneModelLoader"]


# modules/m6_fine_tuning/model_loader.py
"""Model loading for fine-tuning with quantization."""

import torch
from typing import Tuple, Optional
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType
)
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import get_config


class FineTuneModelLoader:
    """Loads models prepared for QLoRA fine-tuning."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.model = None
        self.tokenizer = None
        
    def load_for_training(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load model with 4-bit quantization for QLoRA training."""
        model_name = self.config.model.base_model_name
        logger.info(f"Loading model for fine-tuning: {model_name}")
        
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Add LoRA adapters
        lora_config = LoraConfig(
            r=self.config.model.lora_r,
            lora_alpha=self.config.model.lora_alpha,
            lora_dropout=self.config.model.lora_dropout,
            target_modules=self.config.model.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        return self.model, self.tokenizer


# modules/m6_fine_tuning/qlora_trainer.py
"""QLoRA training implementation."""

import torch
from pathlib import Path
from typing import Optional, Dict
from datasets import load_dataset, Dataset
from transformers import (
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from trl import SFTTrainer, SFTConfig
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import get_config, DATA_DIR, MODEL_DIR
from .model_loader import FineTuneModelLoader


class QLoRATrainer:
    """Handles QLoRA fine-tuning workflow."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.model_loader = FineTuneModelLoader(config)
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def setup(self):
        """Initialize model and tokenizer."""
        self.model, self.tokenizer = self.model_loader.load_for_training()
        
    def load_dataset(
        self,
        data_path: Optional[str] = None
    ) -> Dataset:
        """Load training dataset."""
        data_path = data_path or str(DATA_DIR / "synthetic" / "synthetic_qa.jsonl")
        
        logger.info(f"Loading dataset from {data_path}")
        dataset = load_dataset("json", data_files=data_path, split="train")
        
        logger.info(f"Dataset size: {len(dataset)} examples")
        return dataset
    
    def train(
        self,
        dataset: Dataset,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """Run fine-tuning."""
        if self.model is None:
            self.setup()
        
        output_dir = output_dir or self.config.training.output_dir
        
        # Training arguments
        training_args = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=kwargs.get("epochs", self.config.training.num_train_epochs),
            per_device_train_batch_size=kwargs.get("batch_size", self.config.training.per_device_train_batch_size),
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            warmup_ratio=self.config.training.warmup_ratio,
            max_grad_norm=self.config.training.max_grad_norm,
            weight_decay=self.config.training.weight_decay,
            optim=self.config.training.optim,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            logging_steps=self.config.training.logging_steps,
            save_strategy=self.config.training.save_strategy,
            bf16=self.config.training.bf16,
            fp16=self.config.training.fp16,
            seed=self.config.training.seed,
            max_seq_length=self.config.model.max_seq_length,
            dataset_text_field="text",
            packing=False
        )
        
        # Create trainer
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )
        
        logger.info("Starting fine-tuning...")
        
        # Train
        train_result = self.trainer.train()
        
        # Save model
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model saved to {output_dir}")
        
        return {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime"),
            "output_dir": output_dir
        }
    
    def run_full_pipeline(
        self,
        data_path: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Dict:
        """Run complete fine-tuning pipeline."""
        # Setup
        self.setup()
        
        # Load data
        dataset = self.load_dataset(data_path)
        
        # Train
        results = self.train(dataset, output_dir)
        
        return results


# modules/m7_evaluation/__init__.py
"""Module 7: Model Evaluation."""

from .evaluator import ModelEvaluator
from .comparator import ModelComparator
from .report_generator import ReportGenerator

__all__ = ["ModelEvaluator", "ModelComparator", "ReportGenerator"]


# modules/m7_evaluation/evaluator.py
"""Model evaluation utilities."""

import torch
from typing import List, Dict, Optional
from dataclasses import dataclass
import json
from tqdm import tqdm
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import get_config


@dataclass
class EvaluationResult:
    """Single evaluation result."""
    question: str
    expected_answer: str
    model_answer: str
    is_correct: bool
    relevance_score: float
    latency_ms: float


class ModelEvaluator:
    """Evaluates model responses."""
    
    def __init__(self, model, tokenizer, config=None):
        self.config = config or get_config()
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def generate_response(
        self,
        question: str,
        max_new_tokens: int = 256
    ) -> tuple[str, float]:
        """Generate response and measure latency."""
        import time
        
        # Format prompt
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{self.config.system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,  # Low for deterministic eval
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        latency = (time.time() - start_time) * 1000  # ms
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract assistant response
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        
        return response, latency
    
    def evaluate_single(
        self,
        question: str,
        expected_answer: str
    ) -> EvaluationResult:
        """Evaluate single question."""
        model_answer, latency = self.generate_response(question)
        
        # Simple relevance scoring (can be enhanced with semantic similarity)
        relevance = self._calculate_relevance(model_answer, expected_answer)
        is_correct = relevance > 0.5
        
        return EvaluationResult(
            question=question,
            expected_answer=expected_answer,
            model_answer=model_answer,
            is_correct=is_correct,
            relevance_score=relevance,
            latency_ms=latency
        )
    
    def _calculate_relevance(self, generated: str, expected: str) -> float:
        """Calculate relevance score between generated and expected."""
        # Simple word overlap (can be improved with embeddings)
        gen_words = set(generated.lower().split())
        exp_words = set(expected.lower().split())
        
        if not exp_words:
            return 0.0
        
        overlap = len(gen_words & exp_words)
        return overlap / len(exp_words)
    
    def evaluate_batch(
        self,
        test_data: List[Dict]
    ) -> List[EvaluationResult]:
        """Evaluate multiple questions."""
        results = []
        
        for item in tqdm(test_data, desc="Evaluating"):
            result = self.evaluate_single(
                question=item["question"],
                expected_answer=item["answer"]
            )
            results.append(result)
        
        return results
    
    def compute_metrics(self, results: List[EvaluationResult]) -> Dict:
        """Compute aggregate metrics."""
        if not results:
            return {}
        
        accuracy = sum(1 for r in results if r.is_correct) / len(results)
        avg_relevance = sum(r.relevance_score for r in results) / len(results)
        avg_latency = sum(r.latency_ms for r in results) / len(results)
        
        return {
            "accuracy": accuracy,
            "avg_relevance": avg_relevance,
            "avg_latency_ms": avg_latency,
            "total_evaluated": len(results)
        }


# modules/m7_evaluation/comparator.py
"""Compare base and fine-tuned models."""

from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import get_config, MODEL_DIR, DATA_DIR
from .evaluator import ModelEvaluator, EvaluationResult


@dataclass
class ComparisonResult:
    """Comparison between two models."""
    question: str
    base_answer: str
    finetuned_answer: str
    expected_answer: str
    base_relevance: float
    finetuned_relevance: float
    winner: str  # "base", "finetuned", or "tie"


class ModelComparator:
    """Compares base vs fine-tuned model performance."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.base_evaluator = None
        self.ft_evaluator = None
        
    def setup_models(
        self,
        base_model,
        base_tokenizer,
        finetuned_model,
        finetuned_tokenizer
    ):
        """Setup both evaluators."""
        self.base_evaluator = ModelEvaluator(
            base_model, base_tokenizer, self.config
        )
        self.ft_evaluator = ModelEvaluator(
            finetuned_model, finetuned_tokenizer, self.config
        )
        
    def compare_single(
        self,
        question: str,
        expected_answer: str
    ) -> ComparisonResult:
        """Compare models on single question."""
        base_result = self.base_evaluator.evaluate_single(question, expected_answer)
        ft_result = self.ft_evaluator.evaluate_single(question, expected_answer)
        
        # Determine winner
        if ft_result.relevance_score > base_result.relevance_score + 0.1:
            winner = "finetuned"
        elif base_result.relevance_score > ft_result.relevance_score + 0.1:
            winner = "base"
        else:
            winner = "tie"
        
        return ComparisonResult(
            question=question,
            base_answer=base_result.model_answer,
            finetuned_answer=ft_result.model_answer,
            expected_answer=expected_answer,
            base_relevance=base_result.relevance_score,
            finetuned_relevance=ft_result.relevance_score,
            winner=winner
        )
    
    def compare_batch(
        self,
        test_data: List[Dict]
    ) -> tuple[List[ComparisonResult], Dict]:
        """Compare models on multiple questions."""
        results = []
        
        for item in test_data:
            result = self.compare_single(
                question=item["question"],
                expected_answer=item["answer"]
            )
            results.append(result)
        
        # Aggregate stats
        stats = self._compute_comparison_stats(results)
        
        return results, stats
    
    def _compute_comparison_stats(self, results: List[ComparisonResult]) -> Dict:
        """Compute comparison statistics."""
        total = len(results)
        ft_wins = sum(1 for r in results if r.winner == "finetuned")
        base_wins = sum(1 for r in results if r.winner == "base")
        ties = sum(1 for r in results if r.winner == "tie")
        
        base_avg_rel = sum(r.base_relevance for r in results) / total
        ft_avg_rel = sum(r.finetuned_relevance for r in results) / total
        
        improvement = ((ft_avg_rel - base_avg_rel) / base_avg_rel * 100) if base_avg_rel > 0 else 0
        
        return {
            "total_questions": total,
            "finetuned_wins": ft_wins,
            "base_wins": base_wins,
            "ties": ties,
            "finetuned_win_rate": ft_wins / total,
            "base_avg_relevance": base_avg_rel,
            "finetuned_avg_relevance": ft_avg_rel,
            "improvement_percent": improvement
        }


# modules/m7_evaluation/report_generator.py
"""Generate evaluation reports."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import asdict
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR


class ReportGenerator:
    """Generates evaluation reports."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or (DATA_DIR / "reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_comparison_report(
        self,
        comparison_results: List,
        stats: Dict,
        model_info: Optional[Dict] = None
    ) -> str:
        """Generate markdown comparison report."""
        report = f"""# Model Comparison Report

Generated: {datetime.now().isoformat()}

## Model Information
{json.dumps(model_info or {}, indent=2)}

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Questions | {stats['total_questions']} |
| Fine-tuned Wins | {stats['finetuned_wins']} ({stats['finetuned_win_rate']:.1%}) |
| Base Model Wins | {stats['base_wins']} |
| Ties | {stats['ties']} |
| Base Avg Relevance | {stats['base_avg_relevance']:.3f} |
| Fine-tuned Avg Relevance | {stats['finetuned_avg_relevance']:.3f} |
| **Improvement** | **{stats['improvement_percent']:.1f}%** |

## Detailed Results

"""
        for i, result in enumerate(comparison_results, 1):
            r = result if isinstance(result, dict) else asdict(result)
            report += f"""### Question {i}
**Q:** {r['question']}

**Expected:** {r['expected_answer'][:200]}...

| Model | Response | Relevance |
|-------|----------|-----------|
| Base | {r['base_answer'][:150]}... | {r['base_relevance']:.3f} |
| Fine-tuned | {r['finetuned_answer'][:150]}... | {r['finetuned_relevance']:.3f} |

**Winner:** {r['winner']}

---

"""
        return report
    
    def save_report(
        self,
        report: str,
        filename: str = "comparison_report.md"
    ) -> Path:
        """Save report to file."""
        filepath = self.output_dir / filename
        filepath.write_text(report)
        logger.info(f"Report saved to {filepath}")
        return filepath
    
    def save_json_results(
        self,
        results: List,
        stats: Dict,
        filename: str = "evaluation_results.json"
    ) -> Path:
        """Save results as JSON."""
        filepath = self.output_dir / filename
        
        data = {
            "generated_at": datetime.now().isoformat(),
            "statistics": stats,
            "results": [asdict(r) if hasattr(r, '__dataclass_fields__') else r for r in results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
        return filepath
