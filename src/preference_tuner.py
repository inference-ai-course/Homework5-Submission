"""Week 5: Preference tuning (DPO, KTO, GRPO) via TRL."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable


class PrefRunner:
    """Preference tuning runner supporting DPO, KTO, and GRPO methods via TRL."""

    def __init__(
        self,
        method: str = "dpo",
        base_model_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
    ):
        self.method = method.lower()
        self.base_model_path = base_model_path
        self._model = None
        self._tokenizer = None
        print(f"[preference_tuner] Initialized PrefRunner method={self.method} model={self.base_model_path}")

    def train(
        self,
        dataset,
        output_dir: str = "outputs/pref_adapter",
        num_epochs: int = 1,
        max_steps: int = -1,
        learning_rate: float = 1e-5,
        per_device_batch_size: int = 1,
        reward_funcs: list | None = None,
        use_4bit: bool = False,
    ) -> dict:
        """Run preference tuning. Returns metrics dict."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print(f"[preference_tuner] Starting {self.method.upper()} training, output={output_dir}")

        if self.method == "dpo":
            return self._train_dpo(dataset, output_dir, num_epochs, max_steps, learning_rate, per_device_batch_size, use_4bit)
        elif self.method == "kto":
            return self._train_kto(dataset, output_dir, num_epochs, max_steps, learning_rate, per_device_batch_size, use_4bit)
        elif self.method == "grpo":
            return self._train_grpo(dataset, output_dir, num_epochs, max_steps, learning_rate, per_device_batch_size, reward_funcs, use_4bit)
        else:
            raise ValueError(f"Unknown method '{self.method}'. Choose from: dpo, kto, grpo")

    @staticmethod
    def _tokenizer_kwarg(trainer_cls) -> str:
        """
        Return the right kwarg name for passing the tokenizer to a TRL trainer.

        TRL <= 0.11 uses `tokenizer=`. TRL >= 0.12 renamed it to `processing_class=`
        to align with the new transformers Trainer API.
        """
        import inspect
        try:
            params = inspect.signature(trainer_cls.__init__).parameters
        except (ValueError, TypeError):
            return "processing_class"
        if "processing_class" in params:
            return "processing_class"
        if "tokenizer" in params:
            return "tokenizer"
        return "processing_class"

    def _resolve_model_path(self) -> tuple[str, str | None]:
        """
        Inspect self.base_model_path and return (hf_base_model, peft_adapter_path).

        Cases handled:
        - HF Hub ID (e.g. "Qwen/Qwen2.5-0.5B-Instruct")     -> (path, None)
        - Local HF model dir (has config.json/model_type)    -> (path, None)
        - HF PEFT adapter (adapter_config.json with
          base_model_name_or_path)                           -> (base, path)
        - MLX-LoRA adapter (adapter_config.json with "model"
          key, no base_model_name_or_path)                   -> (base, None) + warn
          MLX adapters cannot be loaded by transformers, so we drop the adapter
          and continue from the base model. Educationally fine: DPO learns from
          the preference pairs, not from continuing the SFT trajectory.
        """
        import json
        import os

        path = self.base_model_path
        if not os.path.isdir(path):
            # Treat as HuggingFace Hub ID
            return path, None

        adapter_cfg_file = os.path.join(path, "adapter_config.json")
        config_file = os.path.join(path, "config.json")

        if os.path.isfile(config_file):
            return path, None

        if os.path.isfile(adapter_cfg_file):
            with open(adapter_cfg_file) as f:
                cfg = json.load(f)
            if "base_model_name_or_path" in cfg:
                return cfg["base_model_name_or_path"], path
            if "model" in cfg:
                # MLX-LoRA shape — can't load adapter, fall back to base
                base = cfg["model"]
                print(
                    f"[preference_tuner] WARNING: '{path}' is an MLX-format LoRA adapter "
                    f"(produced by mlx-lm.lora). HuggingFace transformers cannot load "
                    f"MLX adapters directly. Starting DPO from base model '{base}' instead. "
                    f"This is fine for the homework's pedagogical goal."
                )
                return base, None

        raise ValueError(
            f"[preference_tuner] Cannot determine base model from '{path}'. "
            f"Expected an HF model dir (config.json), HF PEFT adapter dir "
            f"(adapter_config.json with base_model_name_or_path), or an MLX-LoRA "
            f"adapter dir (adapter_config.json with 'model' key)."
        )

    @staticmethod
    def _has_gpu() -> bool:
        try:
            import torch
        except ImportError:
            return False

        return torch.cuda.is_available() or (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )

    def _load_base_model(self, use_4bit: bool):
        """Load base model and tokenizer with optional 4-bit quantization."""
        try:
            import torch
        except ImportError:
            raise ImportError("Install pytorch: pip install torch")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError:
            raise ImportError("Install transformers: pip install transformers")

        base_model, peft_adapter_path = self._resolve_model_path()

        is_gpu_available = self._has_gpu()
        is_mac = sys.platform == "darwin"
        if use_4bit and is_mac:
            print("[preference_tuner] Mac detected: disabling 4-bit quantization")
            use_4bit = False
        if use_4bit and not is_gpu_available:
            print("[preference_tuner] No GPU/MPS detected: disabling 4-bit quantization")
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
                print("[preference_tuner] 4-bit QLoRA enabled")
            except ImportError:
                print("[preference_tuner] bitsandbytes not available. Install: pip install bitsandbytes")
                use_4bit = False

        print(f"[preference_tuner] Loading base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if peft_adapter_path is not None:
            try:
                from peft import PeftModel
                print(f"[preference_tuner] Applying PEFT adapter: {peft_adapter_path}")
                model = PeftModel.from_pretrained(model, peft_adapter_path, is_trainable=True)
                # Merge adapter weights into base model

                print("[preference_tuner] Merging adapter into base model...")
                model = model.merge_and_unload()
            except ImportError:
                print("[preference_tuner] WARNING: peft not installed; continuing from base only.")

        print("[preference_tuner] Model and tokenizer loaded")
        return model, tokenizer

    def _train_dpo(
        self,
        dataset,
        output_dir: str,
        num_epochs: int,
        max_steps: int,
        learning_rate: float,
        per_device_batch_size: int,
        use_4bit: bool,
    ) -> dict:
        """DPO training path using TRL DPOTrainer."""
        print("[preference_tuner] DPO: loading TRL DPOTrainer...")
        try:
            from trl import DPOTrainer, DPOConfig
        except ImportError:
            raise ImportError("Install trl: pip install trl")

        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError:
            raise ImportError("Install peft: pip install peft")

        model, tokenizer = self._load_base_model(use_4bit)
        use_cpu = not self._has_gpu()
        # use_cpu=True

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )

        dpo_config = DPOConfig(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            max_steps=max_steps,
            per_device_train_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            logging_steps=5,
            beta=0.1,
            report_to="none",
            use_cpu=use_cpu,
            bf16=False,
        )

        print("[preference_tuner] Starting DPO training...")
        tk_kwarg = self._tokenizer_kwarg(DPOTrainer)
        trainer = DPOTrainer(
            model=model,
            args=dpo_config,
            train_dataset=dataset,
            peft_config=lora_config,
            **{tk_kwarg: tokenizer},
        )
        result = trainer.train()
        trainer.save_model(output_dir)
        print(f"[preference_tuner] DPO complete: loss={result.training_loss:.4f}")
        self._model = model
        self._tokenizer = tokenizer
        return {"method": "dpo", "train_loss": float(result.training_loss), "steps": int(result.global_step), "adapter_path": output_dir}

    def _train_kto(
        self,
        dataset,
        output_dir: str,
        num_epochs: int,
        max_steps: int,
        learning_rate: float,
        per_device_batch_size: int,
        use_4bit: bool,
    ) -> dict:
        """KTO training path using TRL KTOTrainer."""
        print("[preference_tuner] KTO: loading TRL KTOTrainer...")
        try:
            from trl import KTOTrainer, KTOConfig
        except ImportError:
            raise ImportError("Install trl: pip install trl")

        try:
            from peft import LoraConfig, TaskType
        except ImportError:
            raise ImportError("Install peft: pip install peft")

        model, tokenizer = self._load_base_model(use_4bit)
        use_cpu = not self._has_gpu()
        # use_cpu = True

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )

        kto_config = KTOConfig(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            max_steps=max_steps,
            per_device_train_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            logging_steps=5,
            report_to="none",
            use_cpu=use_cpu,
            bf16=False,
        )

        print("[preference_tuner] Starting KTO training...")
        tk_kwarg = self._tokenizer_kwarg(KTOTrainer)
        trainer = KTOTrainer(
            model=model,
            args=kto_config,
            train_dataset=dataset,
            peft_config=lora_config,
            **{tk_kwarg: tokenizer},
        )
        result = trainer.train()
        trainer.save_model(output_dir)
        print(f"[preference_tuner] KTO complete: loss={result.training_loss:.4f}")
        self._model = model
        self._tokenizer = tokenizer
        return {"method": "kto", "train_loss": float(result.training_loss), "steps": int(result.global_step), "adapter_path": output_dir}

    def _train_grpo(
        self,
        dataset,
        output_dir: str,
        num_epochs: int,
        max_steps: int,
        learning_rate: float,
        per_device_batch_size: int,
        reward_funcs: list | None,
        use_4bit: bool,
    ) -> dict:
        """GRPO training path using TRL GRPOTrainer."""
        print("[preference_tuner] GRPO: loading TRL GRPOTrainer...")
        try:
            from trl import GRPOTrainer, GRPOConfig
        except ImportError:
            raise ImportError("Install trl>=0.8.0 for GRPO: pip install trl")

        try:
            from peft import LoraConfig, TaskType
        except ImportError:
            raise ImportError("Install peft: pip install peft")

        if reward_funcs is None:
            print("[preference_tuner] No reward_funcs provided for GRPO. Using default GSM8K reward.")
            reward_funcs = [make_gsm8k_reward_fn()]

        model, tokenizer = self._load_base_model(use_4bit)
        use_cpu = not self._has_gpu()
        # use_cpu = True

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )

        grpo_config = GRPOConfig(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            max_steps=max_steps,
            per_device_train_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            logging_steps=5,
            report_to="none",
            use_cpu=use_cpu,
            bf16=False,
        )

        print("[preference_tuner] Starting GRPO training...")
        tk_kwarg = self._tokenizer_kwarg(GRPOTrainer)
        trainer = GRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=dataset,
            reward_funcs=reward_funcs,
            peft_config=lora_config,
            **{tk_kwarg: tokenizer},
        )
        result = trainer.train()
        trainer.save_model(output_dir)
        print(f"[preference_tuner] GRPO complete: loss={result.training_loss:.4f}")
        self._model = model
        self._tokenizer = tokenizer
        return {"method": "grpo", "train_loss": float(result.training_loss), "steps": int(result.global_step), "adapter_path": output_dir}

    def build_dpo_dataset_from_llm(
        self,
        questions: list[str],
        llm_client,
        n_chosen: int = 1,
        n_rejected: int = 1,
    ) -> list[dict]:
        """Use llm_client to generate chosen/rejected pairs for DPO."""
        print(f"[preference_tuner] Building DPO dataset from {len(questions)} questions...")
        results: list[dict] = []

        for i, question in enumerate(questions):
            print(f"[preference_tuner] Processing question {i + 1}/{len(questions)}: {question[:60]}...")

            try:
                chosen_prompt = (
                    f"Answer the following question in a detailed, specific, and thorough way. "
                    f"Include examples and explanations.\n\nQuestion: {question}\n\nAnswer:"
                )
                chosen_raw = llm_client.generate(chosen_prompt)

                rejected_prompt = (
                    f"Answer the following question very briefly in one short sentence.\n\n"
                    f"Question: {question}\n\nAnswer:"
                )
                rejected_raw = llm_client.generate(rejected_prompt)

                # Robustly extract text from whatever shape llm_client returned.
                # Handles: dict {"content": str}, dict {"content": [{"text": ...}]},
                # dict {"content": {"text": ...}}, raw Anthropic Message, or bare str.
                def _extract(resp) -> str:
                    if isinstance(resp, str):
                        return resp
                    if isinstance(resp, dict):
                        if "error" in resp:
                            raise RuntimeError(resp["error"])
                        content = resp.get("content", "")
                        if isinstance(content, str):
                            return content
                        if isinstance(content, list):
                            parts = []
                            for blk in content:
                                if isinstance(blk, dict):
                                    parts.append(blk.get("text", ""))
                                elif hasattr(blk, "text"):
                                    parts.append(blk.text)
                                else:
                                    parts.append(str(blk))
                            return "\n".join(p for p in parts if p)
                        if isinstance(content, dict):
                            return content.get("text") or content.get("content") or ""
                        return str(content)
                    if hasattr(resp, "content"):  # raw Anthropic Message
                        return _extract({"content": resp.content})
                    if hasattr(resp, "text"):
                        return resp.text
                    return str(resp)

                chosen_response = _extract(chosen_raw)
                rejected_response = _extract(rejected_raw)

                results.append({
                    "prompt": question,
                    "chosen": str(chosen_response).strip(),
                    "rejected": str(rejected_response).strip(),
                })
                print(f"[preference_tuner] Pair {i + 1}: chosen={len(chosen_response)} chars, rejected={len(rejected_response)} chars")

            except Exception as e:
                print(f"[preference_tuner] Warning: failed on question {i + 1}: {e}")
                continue

        print(f"[preference_tuner] DPO dataset built: {len(results)} pairs")
        return results


def make_gsm8k_reward_fn() -> Callable:
    """Return a GRPO reward function that checks GSM8K-style #### answer format."""

    def gsm8k_reward(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        """Score completions: 1.0 if contains #### <number>, else 0.0."""
        rewards: list[float] = []
        pattern = re.compile(r"####\s*-?\d+(?:\.\d+)?")

        for completion in completions:
            if pattern.search(completion):
                rewards.append(1.0)
            else:
                rewards.append(0.0)

        correct = sum(1 for r in rewards if r > 0.5)
        print(f"[preference_tuner] GRPO reward batch: {correct}/{len(completions)} correct format")
        return rewards

    return gsm8k_reward
