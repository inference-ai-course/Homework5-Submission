# eval_on_new_prompts.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

FULL_DIR  = "llama31-8b-alpaca-full"
LORA_DIR  = "lora-llama31-8b-alpaca"
QLORA_DIR = "qlora-llama31-8b-alpaca"

SYSTEM_PROMPT = "You are a helpful, concise AI assistant."

NEW_PROMPTS = [
    "Explain QLoRA as if I’m 15.",
    "Write a short function in Python that reverses a string.",
    "Give three pros and cons of using LoRA instead of full finetuning.",
]


def load_full(tokenizer):
    m = AutoModelForCausalLM.from_pretrained(
        FULL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    m.eval()
    return m

def load_lora(tokenizer):
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    m = PeftModel.from_pretrained(base, LORA_DIR)
    m.eval()
    return m

def load_qlora(tokenizer):
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    m = PeftModel.from_pretrained(base, QLORA_DIR)
    m.eval()
    return m


def generate(model, tokenizer, user_prompt, max_new_tokens=256):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_full = load_full(tokenizer)
    model_lora = load_lora(tokenizer)
    model_qlora = load_qlora(tokenizer)

    for p in NEW_PROMPTS:
        print("=" * 80)
        print("PROMPT:", p)

        print("\n[Full FT]")
        print(generate(model_full, tokenizer, p))

        print("\n[LoRA]")
        print(generate(model_lora, tokenizer, p))

        print("\n[QLoRA]")
        print(generate(model_qlora, tokenizer, p))
