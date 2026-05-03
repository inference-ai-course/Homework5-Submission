# finetune_llama31_qlora.py

import os
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, TaskType


# =========================
# Config
# =========================
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DATASET_NAME = "yahma/alpaca-cleaned"

OUTPUT_DIR = "Week_5_FineTuning/llama31-8b-alpaca-qlora"
MAX_SEQ_LEN = 1024
NUM_TRAIN_SAMPLES = 5000  # set to None to use full dataset

SYSTEM_PROMPT = "You are a helpful, concise AI assistant."

USE_BF16 = True  # if your GPU supports bfloat16; otherwise set to False and use fp16

# Make sure you did: export HF_TOKEN=hf_xxx...
HF_TOKEN = os.environ.get("HF_TOKEN", None)


# Decide device explicitly
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")


def prepare_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_auth_token=HF_TOKEN,  # or token=HF_TOKEN on newer transformers
        use_fast=True,
    )
    # Llama-style models usually have no pad token; use EOS as pad
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_and_prepare_dataset(tokenizer):
    # Alpaca-cleaned: columns = ["instruction", "input", "output"]
    ds = load_dataset(DATASET_NAME, split="train")

    if NUM_TRAIN_SAMPLES is not None:
        ds = ds.select(range(NUM_TRAIN_SAMPLES))

    # 1) Convert each row to chat messages
    def to_chatml(example):
        instruction = example["instruction"]
        inp = example.get("input", "")
        output = example["output"]

        if inp:
            user_content = f"{instruction}\n\nInput: {inp}"
        else:
            user_content = instruction

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]
        return {"messages": messages}

    ds = ds.map(to_chatml)

    # 2) Apply chat template and tokenize (FIXED VERSION)
    def tokenize_example(example):
        # Get a ChatML-formatted string from the messages
        chat_str = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,             # <- string, not tensors
            add_generation_prompt=False,
        )

        # Tokenize that string normally
        enc = tokenizer(
            chat_str,
            max_length=MAX_SEQ_LEN,
            truncation=True,
            padding="max_length",
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # Simple causal LM objective: predict every non-padding token
        labels = [
            (tid if mask == 1 else -100)
            for tid, mask in zip(input_ids, attention_mask)
        ]
        enc["labels"] = labels
        return enc

    tokenized = ds.map(
        tokenize_example,
        remove_columns=ds.column_names,
    )

    # Small train/val split
    split = tokenized.train_test_split(test_size=0.01, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    return train_ds, eval_ds


def load_qlora_model(tokenizer):
    # 4-bit quantization config for QLoRA
    compute_dtype = torch.bfloat16 if USE_BF16 else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        use_auth_token=HF_TOKEN,  # or token=HF_TOKEN
    )

    model.config.use_cache = False  # required for gradient checkpointing
    model.config.pad_token_id = tokenizer.pad_token_id

    # More memory savings
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    return model


def apply_lora(model):
    # Common target modules for LLaMA-style models (attn + MLP)
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def main():
    tokenizer = prepare_tokenizer()
    train_ds, eval_ds = load_and_prepare_dataset(tokenizer)

    model = load_qlora_model(tokenizer)
    model = apply_lora(model)

    data_collator = default_data_collator

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        weight_decay=0.0,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        bf16=USE_BF16,
        fp16=not USE_BF16,
        report_to="none",  # turn off W&B etc
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save only the adapter & tokenizer (base model stays separate)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"QLoRA adapter saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
