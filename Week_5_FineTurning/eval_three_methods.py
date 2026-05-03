# eval_three_methods.py

import math
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from datasets import load_dataset

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

FULL_DIR  = "llama31-8b-alpaca-full"
LORA_DIR  = "lora-llama31-8b-alpaca"
QLORA_DIR = "qlora-llama31-8b-alpaca"

BATCH_SIZE = 4
DEVICE = "cuda"


SYSTEM_PROMPT = "You are a helpful, concise AI assistant."
DATASET_NAME = "yahma/alpaca-cleaned"
MAX_SEQ_LEN = 512


def prepare_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def make_eval_dataset(tokenizer, num_samples=500):
    ds = load_dataset(DATASET_NAME, split="train")
    ds = ds.shuffle(seed=42).select(range(num_samples))

    def to_chatml(example):
        inst = example["instruction"]
        inp = example.get("input", "")
        out = example["output"]

        user_content = f"{inst}\n\nInput: {inp}" if inp else inst

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": out},
        ]
        return {"messages": messages}

    ds = ds.map(to_chatml)

    def tokenize(example):
        enc = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=True,
            add_generation_prompt=False,
            max_length=MAX_SEQ_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"][0]
        attn = enc["attention_mask"][0]

        labels = input_ids.clone()
        labels[attn == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
        }

    ds = ds.map(tokenize, remove_columns=ds.column_names)
    ds.set_format(type="torch")
    return ds


@torch.no_grad()
def eval_model(model, dataloader):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(DEVICE)
        attn = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attn,
            labels=labels,
        )
        loss = outputs.loss
        logits = outputs.logits  # [B, T, V]

        # accumulate loss
        # loss is mean over non-ignored tokens
        batch_loss = loss.item()
        total_loss += batch_loss * input_ids.size(0)  # approx

        # token-level accuracy (ignoring label == -100)
        preds = logits.argmax(dim=-1)
        mask = labels != -100
        correct = (preds == labels) & mask
        total_correct += correct.sum().item()
        total_tokens += mask.sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    ppl = math.exp(avg_loss)
    acc = total_correct / total_tokens if total_tokens > 0 else 0.0
    return {
        "loss": avg_loss,
        "perplexity": ppl,
        "token_accuracy": acc,
    }


def load_full_finetuned(tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        FULL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    return model


def load_lora(tokenizer):
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, LORA_DIR)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model


def load_qlora(tokenizer):
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, QLORA_DIR)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model


if __name__ == "__main__":
    tokenizer = prepare_tokenizer()
    eval_ds = make_eval_dataset(tokenizer)
    dl = DataLoader(eval_ds, batch_size=BATCH_SIZE)

    print("Evaluating Full FT...")
    model_full = load_full_finetuned(tokenizer)
    metrics_full = eval_model(model_full, dl)
    print("Full FT:", metrics_full)

    print("Evaluating LoRA...")
    model_lora = load_lora(tokenizer)
    metrics_lora = eval_model(model_lora, dl)
    print("LoRA:", metrics_lora)

    print("Evaluating QLoRA...")
    model_qlora = load_qlora(tokenizer)
    metrics_qlora = eval_model(model_qlora, dl)
    print("QLoRA:", metrics_qlora)