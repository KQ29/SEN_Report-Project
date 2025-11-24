# qwen_integration.py
"""Lightweight helpers for working with Qwen2.5-0.5B-Instruct locally.

This module keeps the model weights inside ``models/Qwen2.5-0.5B-Instruct`` and
offers a simple CLI to try a prompt:

    python qwen_integration.py "Write a haiku about study skills."
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_DIR = Path("models") / "Qwen2.5-0.5B-Instruct"


def ensure_model_present(local_dir: Path = MODEL_DIR) -> Path:

    if local_dir.exists():
        return local_dir
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=MODEL_ID, local_dir=str(local_dir))
    return local_dir


def load_qwen(local_dir: Optional[Path] = None):
    """Load tokenizer + model for local CPU inference."""
    local_dir = ensure_model_present(local_dir or MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map=None,  # keep on CPU for widest compatibility
    )
    model.eval()
    return tokenizer, model


def generate_response(prompt: str, max_new_tokens: int = 128) -> str:
    tokenizer, model = load_qwen()
    encoded = tokenizer(prompt, return_tensors="pt")
    output_ids = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Ignore the prompt portion of the generation for clarity
    generated = output_ids[0][encoded["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python qwen_integration.py \"<prompt>\"")
        raise SystemExit(1)
    prompt = sys.argv[1]
    print(generate_response(prompt))
