# finetune_llama31_lora.py

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
from peft import LoraConfig, get_peft_model, TaskType


# =========================
# Config
# =========================
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DATASET_NAME = "yahma/alpaca-cleaned"

OUTPUT_DIR = "Week_5_FineTuning/llama31-8b-alpaca-lora"
MAX_SEQ_LEN = 1024
NUM_TRAIN_SAMPLES = 5000  # set to None to use full dataset

SYSTEM_PROMPT = "You are a helpful, concise AI assistant."
USE_BF16 = True  # if your GPU supports bfloat16

HF_TOKEN = os.environ.get("HF_TOKEN", None)


def prepare_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_auth_token=HF_TOKEN,  # or token=HF_TOKEN on newer transformers
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_and_prepare_dataset(tokenizer):
    # Alpaca-cleaned: ["instruction", "input", "output"]
    ds = load_dataset(DATASET_NAME, split="train")

    if NUM_TRAIN_SAMPLES is not None:
        ds = ds.select(range(NUM_TRAIN_SAMPLES))

    # 1) Convert to chat messages
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

    # 2) Apply chat template and tokenize (NO return_tensors, NO [0])
    def tokenize_chat(example):
        # Get ChatML string
        chat_str = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

        # Tokenize normally
        enc = tokenizer(
            chat_str,
            max_length=MAX_SEQ_LEN,
            truncation=True,
            padding="max_length",
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # Build labels: ignore padding with -100
        labels = [
            (tid if mask == 1 else -100)
            for tid, mask in zip(input_ids, attention_mask)
        ]
        enc["labels"] = labels
        return enc

    tokenized = ds.map(
        tokenize_chat,
        remove_columns=ds.column_names,
    )

    split = tokenized.train_test_split(test_size=0.01, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    return train_ds, eval_ds


def load_lora_base_model(tokenizer):
    dtype = torch.bfloat16 if USE_BF16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto",
        use_auth_token=HF_TOKEN,
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()
    return model


def apply_lora(model):
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
    print("Starting LoRA finetuning script...")  # debug marker

    tokenizer = prepare_tokenizer()
    train_ds, eval_ds = load_and_prepare_dataset(tokenizer)

    model = load_lora_base_model(tokenizer)
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
        report_to="none",
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"LoRA adapter saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
