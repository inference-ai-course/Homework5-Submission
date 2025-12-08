# modules/m1_langchain_llama/__init__.py
"""Module 1: LangChain + LLaMA 3 Integration."""

from .llm_loader import LLMLoader, load_base_model, load_finetuned_model
from .chain_builder import ChainBuilder, create_qa_chain, create_rag_chain
from .memory_manager import MemoryManager

__all__ = [
    "LLMLoader", "load_base_model", "load_finetuned_model",
    "ChainBuilder", "create_qa_chain", "create_rag_chain",
    "MemoryManager"
]


# modules/m1_langchain_llama/llm_loader.py
"""LLM loading utilities for LLaMA 3 models."""

import torch
from pathlib import Path
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import get_config, MODEL_DIR


class LLMLoader:
    """Handles loading and managing LLaMA models."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_base_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load the base LLaMA model with 4-bit quantization."""
        logger.info(f"Loading base model: {self.config.model.base_model_name}")
        
        # Quantization config for 4-bit loading
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.base_model_name,
            quantization_config=bnb_config,
            device_map=self.config.model.device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        logger.info(f"Base model loaded on {self.device}")
        return self.model, self.tokenizer
    
    def load_finetuned_model(
        self, 
        adapter_path: Optional[str] = None
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load a fine-tuned model with LoRA adapters."""
        adapter_path = adapter_path or self.config.training.output_dir
        
        logger.info(f"Loading fine-tuned model from: {adapter_path}")
        
        # First load base model if not loaded
        if self.model is None:
            self.load_base_model()
        
        # Load LoRA adapters
        self.model = PeftModel.from_pretrained(
            self.model,
            adapter_path,
            is_trainable=False
        )
        self.model = self.model.merge_and_unload()  # Merge for inference
        
        logger.info("Fine-tuned model loaded and merged")
        return self.model, self.tokenizer
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """Generate text from the loaded model."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_base_model() first.")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from response
        response = response[len(self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):]
        return response.strip()
    
    def format_prompt(self, user_message: str, system_prompt: Optional[str] = None) -> str:
        """Format prompt using LLaMA 3 chat template."""
        system_prompt = system_prompt or self.config.system_prompt
        
        # LLaMA 3 Instruct format
        formatted = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return formatted


# Convenience functions
def load_base_model(config=None):
    """Quick function to load base model."""
    loader = LLMLoader(config)
    return loader.load_base_model()

def load_finetuned_model(adapter_path=None, config=None):
    """Quick function to load fine-tuned model."""
    loader = LLMLoader(config)
    return loader.load_finetuned_model(adapter_path)


# modules/m1_langchain_llama/chain_builder.py
"""LangChain chain builders for various use cases."""

from typing import Optional, List
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import get_config


class ChainBuilder:
    """Builds LangChain chains for different tasks."""
    
    def __init__(self, model, tokenizer, config=None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or get_config()
        self.llm = self._create_hf_pipeline()
        
    def _create_hf_pipeline(self) -> HuggingFacePipeline:
        """Create HuggingFace pipeline for LangChain."""
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            return_full_text=False
        )
        return HuggingFacePipeline(pipeline=pipe)
    
    def create_qa_chain(self) -> LLMChain:
        """Create a simple Q&A chain."""
        template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        prompt = PromptTemplate(
            input_variables=["system_prompt", "question"],
            template=template
        )
        
        return LLMChain(llm=self.llm, prompt=prompt)
    
    def create_rag_chain(self, retriever) -> any:
        """Create a RAG chain with retrieval."""
        template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert academic research assistant. Answer the question based on the provided context from research papers. If the context doesn't contain relevant information, say so clearly.

Context from research papers:
{context}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
        
        def format_docs(docs):
            return "\n\n---\n\n".join(
                f"[Source: {doc.metadata.get('title', 'Unknown')}]\n{doc.page_content}" 
                for doc in docs
            )
        
        # LCEL chain
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def create_comparison_chain(self) -> LLMChain:
        """Create chain for comparing model outputs."""
        template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are evaluating two model responses to the same question. Analyze which response is better in terms of accuracy, relevance, and completeness.<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: {question}

Response A (Base Model):
{response_a}

Response B (Fine-tuned Model):
{response_b}

Provide a detailed comparison and declare a winner.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        prompt = PromptTemplate(
            input_variables=["question", "response_a", "response_b"],
            template=template
        )
        
        return LLMChain(llm=self.llm, prompt=prompt)


# Convenience functions
def create_qa_chain(model, tokenizer, config=None):
    builder = ChainBuilder(model, tokenizer, config)
    return builder.create_qa_chain()

def create_rag_chain(model, tokenizer, retriever, config=None):
    builder = ChainBuilder(model, tokenizer, config)
    return builder.create_rag_chain(retriever)


# modules/m1_langchain_llama/memory_manager.py
"""Conversation memory management."""

from typing import List, Dict, Optional
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ConversationTurn:
    """Single conversation turn."""
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


class MemoryManager:
    """Manages conversation history and context."""
    
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.history: List[ConversationTurn] = []
        self.langchain_memory = ConversationBufferWindowMemory(
            k=max_turns,
            return_messages=True
        )
        
    def add_turn(self, role: str, content: str):
        """Add a conversation turn."""
        turn = ConversationTurn(role=role, content=content)
        self.history.append(turn)
        
        # Trim if exceeds max
        if len(self.history) > self.max_turns * 2:  # *2 for user+assistant pairs
            self.history = self.history[-self.max_turns * 2:]
        
        # Update LangChain memory
        if role == "user":
            self.langchain_memory.chat_memory.add_user_message(content)
        else:
            self.langchain_memory.chat_memory.add_ai_message(content)
    
    def get_context_string(self) -> str:
        """Get conversation history as formatted string."""
        context_parts = []
        for turn in self.history:
            prefix = "User" if turn.role == "user" else "Assistant"
            context_parts.append(f"{prefix}: {turn.content}")
        return "\n".join(context_parts)
    
    def get_langchain_memory(self) -> ConversationBufferWindowMemory:
        """Get the LangChain memory object."""
        return self.langchain_memory
    
    def clear(self):
        """Clear all history."""
        self.history = []
        self.langchain_memory.clear()
    
    def to_dict(self) -> List[Dict]:
        """Export history as list of dicts."""
        return [
            {"role": t.role, "content": t.content, "timestamp": t.timestamp.isoformat()}
            for t in self.history
        ]
