"""Configuration for Week 5 notebooks (SFT / Fine-Tuning)"""

import platform
import os

# Your selected path: "A" = MLX (Mac Apple Silicon), "B" = HF+TRL (GPU/CPU), "C" = Cloud GPU
PATH = "B"

# Default LLM models (for Claude API and Ollama inference)
CLAUDE_MODEL = "claude-sonnet-4-6"
OLLAMA_MODEL = "qwen3:8b"#"qwen3.5:27b"
GEMINI_MODEL = "gemini-2.5-pro:gemini-2.5-pro-8b-instruct"  # For Gemini API users

# Base model for fine-tuning (HuggingFace Hub ID)
BASE_MODEL_HF  = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_MODEL_MLX = "mlx-community/Qwen2.5-0.5B-Instruct-bf16"

# Auto-detect fine-tuning backend unless overridden by env var   "gemini-3-flash-preview"#
def _detect_backend() -> str:
    override = os.environ.get("FINETUNE_BACKEND", "").lower()
    if override in ("mlx", "hf"):
        return override
    is_apple = (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
    )
    if is_apple:
        try:
            import mlx  # noqa: F401
            return "mlx"
        except ImportError:
            pass
    return "hf"

FINETUNE_BACKEND = _detect_backend()
