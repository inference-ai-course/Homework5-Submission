# finetune_llama31_full.py:

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

# =========================
# Config
# =========================
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# Official cleaned Alpaca dataset on HF
DATASET_NAME = "yahma/alpaca-cleaned"   # often referred to as "alpaca-cleaned"
OUTPUT_DIR = "Week_5_FineTuning/llama31-8b-alpaca-full"

MAX_SEQ_LEN = 1024       # alpaca-cleaned has many prompts >256 tokens :contentReference[oaicite:3]{index=3}
NUM_TRAIN_SAMPLES = 5000  # set to None for full 52k

SYSTEM_PROMPT = "You are a helpful, concise AI assistant."
HF_TOKEN = os.environ["HF_TOKEN"]


# Decide device explicitly
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")


def prepare_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    # Llama 3.1 uses EOS as pad for chat in many examples
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

def load_and_prepare_dataset(tokenizer):
    # Alpaca-cleaned: columns = instruction, input, output :contentReference[oaicite:4]{index=4}
    ds = load_dataset(DATASET_NAME, split="train")

    if NUM_TRAIN_SAMPLES is not None:
        ds = ds.select(range(NUM_TRAIN_SAMPLES))

    # 1) Convert each row into ChatML-style messages
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

    # 2) Turn messages -> ChatML string using the model's chat template,
    #    then tokenize and build labels for causal LM
    def tokenize_example(example):
        # apply_chat_template builds the correct prompt with special tokens
        chat_str = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,            # get a string, we’ll tokenize next
            add_generation_prompt=False,  # we already include assistant answer
        )

        tokenized = tokenizer(
            chat_str,
            max_length=MAX_SEQ_LEN,
            truncation=True,
            padding="max_length",
        )

        input_ids = tokenized["input_ids"]
        labels = input_ids.copy()

        # Ignore padding in loss
        labels = [
            (tid if tid != tokenizer.pad_token_id else -100)
            for tid in labels
        ]
        tokenized["labels"] = labels
        return tokenized

    tokenized_ds = ds.map(
        tokenize_example,
        remove_columns=ds.column_names,  # keep only model features
    )

    # Simple train/val split
    split = tokenized_ds.train_test_split(test_size=0.01, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    return train_ds, eval_ds

def load_model(tokenizer):
    # Llama 3.1 weights are BF16 on HF :contentReference[oaicite:5]{index=5}
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",      # Accelerate will spread across GPUs
    )

    model.config.use_cache = False  # needed for training
    model.config.pad_token_id = tokenizer.pad_token_id

    # Memory savings; full finetune still updates ALL weights
    model.gradient_checkpointing_enable()

    return model

def main():
    tokenizer = prepare_tokenizer()
    train_ds, eval_ds = load_and_prepare_dataset(tokenizer)
    model = load_model(tokenizer)

    data_collator = default_data_collator

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=1,     # increase if you have more VRAM
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-5,
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        bf16=True,                         # for modern GPUs with bf16 support
        fp16=False,
        report_to="none",                  # disable W&B etc
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

    # Save the full finetuned model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Finetuned model saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
