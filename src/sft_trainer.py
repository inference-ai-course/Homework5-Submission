"""Week 5: SFT trainer supporting HF/TRL and MLX-LM backends."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


class SFTRunner:
    """Unified SFT trainer supporting 'hf' (TRL SFTTrainer) and 'mlx' (mlx-lm) backends."""

    def __init__(
        self,
        backend: str = "hf",
        base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    ):
        self.backend = backend.lower()
        self.base_model = base_model
      
        self._model = None
        self._tokenizer = None
        self._adapter_path: str | None = None
        print(f"[sft_trainer] Initialized SFTRunner backend={self.backend} model={self.base_model}")

    def train(
        self,
        dataset,
        lora_config=None,
        output_dir: str = "outputs/sft_adapter",
        num_epochs: int = 3,
        max_steps: int = -1,
        learning_rate: float = 1e-4,
        per_device_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        # max_seq_length: int = 512,
        use_4bit: bool = True,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
    ) -> dict:
        """Run SFT. Returns {"train_loss": float, "steps": int, "adapter_path": str}."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print(f"[sft_trainer] Starting training: backend={self.backend}, epochs={num_epochs}, max_steps={max_steps}")
        print(f"[sft_trainer] Output directory: {output_dir}")

        if self.backend == "mlx":
            return self._train_mlx(
                dataset=dataset,
                lora_config=lora_config,
                output_dir=output_dir,
                num_epochs=num_epochs,
                max_steps=max_steps,
                learning_rate=learning_rate,
                per_device_batch_size=per_device_batch_size,
                # max_seq_length=max_seq_length,
            )
        else:
            return self._train_hf(
                dataset=dataset,
                lora_config=lora_config,
                output_dir=output_dir,
                num_epochs=num_epochs,
                max_steps=max_steps,
                learning_rate=learning_rate,
                per_device_batch_size=per_device_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                # max_seq_length=max_seq_length,
                use_4bit=use_4bit,
                warmup_ratio=warmup_ratio,
                weight_decay=weight_decay,
            )

    def _train_hf(
        self,
        dataset,
        lora_config=None,
        output_dir: str = "outputs/sft_adapter",
        num_epochs: int = 3,
        max_steps: int = -1,
        learning_rate: float = 3e-4,
        per_device_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        # max_seq_length: int = 512,
        use_4bit: bool = True,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
    ) -> dict:
        """HF+TRL path using SFTTrainer and LoraConfig from peft."""
        print("[sft_trainer] Loading HF/TRL dependencies...")

        try:
            import torch
        except ImportError:
            raise ImportError("Install pytorch: pip install torch")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError:
            raise ImportError("Install transformers: pip install transformers")

        try:
            from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError("Install peft: pip install peft")

        try:
            from trl import SFTTrainer, SFTConfig
        except ImportError:
            raise ImportError("Install trl: pip install trl")

        is_mac = sys.platform == "darwin"
        if use_4bit and is_mac:
            print("[sft_trainer] Mac detected: disabling 4-bit quantization (bitsandbytes not supported on Mac)")
            use_4bit = False

        bnb_config = None
        if use_4bit:
            try:
                import bitsandbytes  # noqa: F401
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                print("[sft_trainer] 4-bit QLoRA enabled")
            except ImportError:
                print("[sft_trainer] bitsandbytes not available, disabling 4-bit. Install: pip install bitsandbytes")
                use_4bit = False

        print(f"[sft_trainer] Loading base model: {self.base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
    
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("[sft_trainer] Base model and tokenizer loaded")

        if lora_config is None:
            lora_config = PeftLoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                bias="none",
            )
            print("[sft_trainer] Using default LoRA config: r=16, alpha=32")

        if use_4bit:
            try:
                from peft import prepare_model_for_kbit_training
                model = prepare_model_for_kbit_training(model)
            except Exception as e:
                print(f"[sft_trainer] Warning: prepare_model_for_kbit_training failed: {e}")

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        sft_config = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            max_steps=max_steps,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            # max_seq_length=max_seq_length,
            logging_steps=10,
            save_strategy="epoch",
            fp16=not is_mac and not use_4bit,
            bf16=is_mac,
            report_to="none",
        )

     

        print("[sft_trainer] Starting SFTTrainer training loop...")
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
            # tokenizer=tokenizer,
        )

        train_result = trainer.train()
        train_loss = train_result.training_loss
        steps = train_result.global_step
        print(f"[sft_trainer] Training complete: loss={train_loss:.4f}, steps={steps}")

        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"[sft_trainer] Adapter saved to {output_dir}")

        self._model = model
        # self._tokenizer = tokenizer
        self._adapter_path = output_dir

        return {"train_loss": float(train_loss), "steps": int(steps), "adapter_path": output_dir}

    def _write_mlx_lora_config(self, lora_config, output_dir: str) -> str | None:
        """Write a YAML config for mlx_lm.lora with rank/scale from lora_config. Returns path or None."""
        try:
            import yaml  # type: ignore
        except ImportError:
            return None  # yaml not available; caller skips -c flag

        cfg: dict = {}
        if lora_config is not None:
            r = getattr(lora_config, "r", None) or getattr(lora_config, "rank", 8)
            alpha = getattr(lora_config, "lora_alpha", None) or getattr(lora_config, "alpha", r)
            cfg["lora_parameters"] = {"rank": int(r), "scale": int(alpha), "dropout": 0.0}

        cfg_path = os.path.join(output_dir, "_mlx_lora_train_config.yaml")
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)
        return cfg_path

    def _train_mlx(
        self,
        dataset,
        lora_config=None,
        output_dir: str = "outputs/sft_adapter",
        num_epochs: int = 3,
        max_steps: int = -1,
        learning_rate: float = 1e-4,
        per_device_batch_size: int = 2,
        # max_seq_length: int = 512,
    ) -> dict:
        """MLX path: tries Python API first, falls back to mlx_lm.lora CLI via subprocess."""
        print("[sft_trainer] MLX backend selected")

        import json

        data_dir = Path(output_dir) / "mlx_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        train_jsonl = data_dir / "train.jsonl"

        # Convert to messages format so --mask-prompt is supported (text format does not support it)
        records = dataset if isinstance(dataset, list) else list(dataset)
        chat_records = []
        for item in records:
            if not isinstance(item, dict):
                item = {"text": str(item)}
            if "messages" in item:
                chat_records.append(item)
            elif "text" in item:
                import re
                text = item["text"]
                sys_m = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", text, re.DOTALL)
                usr_m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", text, re.DOTALL)
                ast_m = re.search(r"<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)", text, re.DOTALL)
                if sys_m and usr_m and ast_m:
                    chat_records.append({"messages": [
                        {"role": "system", "content": sys_m.group(1).strip()},
                        {"role": "user", "content": usr_m.group(1).strip()},
                        {"role": "assistant", "content": ast_m.group(1).strip()},
                    ]})
                else:
                    # Fallback: keep as text (mask-prompt won't apply but training still runs)
                    chat_records.append(item)
            else:
                chat_records.append(item)

        with open(train_jsonl, "w") as f:
            for item in chat_records:
                f.write(json.dumps(item) + "\n")
        print(f"[sft_trainer] Wrote {len(chat_records)} messages-format records to {train_jsonl}")

        iters = max_steps if max_steps > 0 else (len(chat_records) * num_epochs // max(per_device_batch_size, 1))

        # Write LoRA config YAML so rank/scale from lora_config are honoured
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cfg_path = self._write_mlx_lora_config(lora_config, output_dir)

        base_args = [
            "--model", self.base_model,
            "--train",
            "--data", str(data_dir),
            "--adapter-path", output_dir,
            "--iters", str(iters),
            "--batch-size", str(per_device_batch_size),
            "--learning-rate", str(learning_rate),
            # "--max-seq-length", str(max_seq_length),
            "--val-batches", "0",
            "--mask-prompt",
        ]
        if cfg_path:
            base_args += ["-c", cfg_path]

        # Attempt Python API first
        try:
            import mlx_lm  # noqa: F401
            from mlx_lm import lora as mlx_lora

            print("[sft_trainer] Using mlx_lm Python API for training")
            import sys as _sys
            old_argv = _sys.argv
            _sys.argv = ["mlx_lm.lora"] + base_args
            try:
                mlx_lora.main()
            finally:
                _sys.argv = old_argv

            print(f"[sft_trainer] MLX training complete, adapter at {output_dir}")
            self._adapter_path = output_dir
            return {"train_loss": float("nan"), "steps": iters, "adapter_path": output_dir}

        except Exception as e:
            print(f"[sft_trainer] MLX Python API failed ({e}), falling back to subprocess CLI")

        # Subprocess fallback
        cmd = [sys.executable, "-m", "mlx_lm.lora"] + base_args
        print(f"[sft_trainer] Running: {' '.join(cmd)}")

        losses: list[float] = []
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert process.stdout is not None
            for line in process.stdout:
                line = line.rstrip()
                print(f"[sft_trainer][mlx] {line}")
                if "loss" in line.lower():
                    import re
                    match = re.search(r"loss[:\s=]+([0-9.]+)", line, re.IGNORECASE)
                    if match:
                        try:
                            losses.append(float(match.group(1)))
                        except ValueError:
                            pass
            process.wait()
            if process.returncode != 0:
                raise RuntimeError(f"mlx_lm.lora exited with code {process.returncode}")
        except FileNotFoundError:
            raise ImportError(
                "mlx_lm not found. Install with: pip install mlx-lm\n"
                "Requires Apple Silicon Mac with MLX support."
            )

        final_loss = losses[-1] if losses else float("nan")
        print(f"[sft_trainer] MLX training complete: final_loss={final_loss:.4f}, steps={iters}")
        self._adapter_path = output_dir
        return {"train_loss": final_loss, "steps": iters, "adapter_path": output_dir}

    def save_adapter(self, path: str) -> None:
        """Save the LoRA adapter (HF: save_pretrained; MLX: already saved during train)."""
        if self.backend == "mlx":
            print(f"[sft_trainer] MLX adapters are saved automatically during training at {self._adapter_path}")
            return

        if self._model is None:
            print("[sft_trainer] No model loaded. Run train() first.")
            return

        Path(path).mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(path)
        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(path)
        print(f"[sft_trainer] Adapter saved to {path}")

    def merge_and_save(self, adapter_path: str, output_path: str) -> None:
        """Merge LoRA weights into base model and save as safetensors. HF path only."""
        print(f"[sft_trainer] Merging adapter {adapter_path} into {self.base_model}")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("Install transformers: pip install transformers")

        try:
            from peft import PeftModel
        except ImportError:
            raise ImportError("Install peft: pip install peft")

        import torch

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)

        print("[sft_trainer] Loading adapter...")
        model = PeftModel.from_pretrained(base, adapter_path)

        print("[sft_trainer] Merging weights...")
        merged = model.merge_and_unload()

        Path(output_path).mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(output_path, safe_serialization=True)
        tokenizer.save_pretrained(output_path)
        print(f"[sft_trainer] Merged model saved to {output_path}")

    def load_adapter(self, path: str) -> None:
        """
        Point this runner at an existing trained adapter so .generate() can use it.
        Does not load the model into memory — that happens lazily on first .generate() call.
        Works for both 'hf' and 'mlx' backends.
        """
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(f"[sft_trainer] adapter path does not exist: {path}")
        self._adapter_path = path
        # Invalidate any previously-loaded model so .generate() re-loads with the new adapter
        self._model = None
        self._tokenizer = None
        print(f"[sft_trainer] Adapter path set to {path}; will be loaded on next generate() call.")

    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Run inference on the loaded (or merged) model for quick testing."""
        if self.backend == "mlx":
            return self._generate_mlx(prompt, max_new_tokens)

        if self._model is None or self._tokenizer is None:
            print("[sft_trainer] No model in memory. Loading from adapter path...")
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from peft import PeftModel
                import torch
            except ImportError as e:
                raise ImportError(f"Missing dependency: {e}")

            base = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
            if self._adapter_path:
                self._model = PeftModel.from_pretrained(base, self._adapter_path)
            else:
                self._model = base

        print(f"[sft_trainer] Generating response for prompt: {prompt[:80]}...")
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        import torch
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        response = self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"[sft_trainer] Generation complete ({len(response)} chars)")
        return response

    def _generate_mlx(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Generate text using mlx_lm."""
        try:
            from mlx_lm import load, generate

            print(f"[sft_trainer] Loading MLX model for inference from {self._adapter_path or self.base_model}")
            model, tokenizer = load(
                self.base_model,
                adapter_path=self._adapter_path,
            )
            response = generate(model, tokenizer, prompt=prompt, max_tokens=max_new_tokens, verbose=False)
            print(f"[sft_trainer] MLX generation complete ({len(response)} chars)")
            return response
        except ImportError:
            raise ImportError("Install mlx-lm: pip install mlx-lm")
