# build_eval_dataset.py

from datasets import load_dataset
from transformers import AutoTokenizer
import torch

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DATASET_NAME = "yahma/alpaca-cleaned"
MAX_SEQ_LEN = 512
SYSTEM_PROMPT = "You are a helpful, concise AI assistant."

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

    # set format to PyTorch
    ds.set_format(type="torch")
    return ds

if __name__ == "__main__":
    tok = prepare_tokenizer()
    eval_ds = make_eval_dataset(tok)
    print(eval_ds)
