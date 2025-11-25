from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import torch

model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj"]
)
lora_model = get_peft_model(base_model, lora_config)
lora_model.print_trainable_parameters()

from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

# Prepare the dataset (must have a 'text' field)
demo_dataset = Dataset.from_list([
    {"text": "Human: What is Python?\nAssistant: Python is a programming language."},
    {"text": "Human: How do I learn coding?\nAssistant: Start with basic concepts and practice regularly."}
])

# Training arguments
training_args = TrainingArguments(
    output_dir="./trl_sft_demo",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    learning_rate=5e-4,
    logging_steps=1,
    save_steps=10,
    save_total_limit=1,
    fp16=False,
    report_to=None,
)

# ✅ Initialize SFTTrainer — no config, no extras
trainer = SFTTrainer(
    model=lora_model,
    args=training_args,
    train_dataset=demo_dataset
)

# ✅ Train the model
trainer.train()