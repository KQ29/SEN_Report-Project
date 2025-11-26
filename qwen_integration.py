# qwen_integration.py
"""Qwen 0.5B optimized integration
This version uses strict formatting constraints designed
specifically for small models (0.5B).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "Qwen2.5-0.5B-Instruct"

_TOKENIZER = None
_MODEL = None


def ensure_model_present(local_dir: Path = MODEL_DIR) -> Path:
    if local_dir.exists():
        return local_dir
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=MODEL_ID, local_dir=str(local_dir))
    return local_dir


def load_qwen(local_dir: Optional[Path] = None):
    global _TOKENIZER, _MODEL
    local_dir = ensure_model_present(local_dir or MODEL_DIR)
    if _TOKENIZER is not None:
        return _TOKENIZER, _MODEL

    tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map=None
    )
    model.eval()

    _TOKENIZER, _MODEL = tokenizer, model
    return tokenizer, model


def generate_response(prompt: str, max_new_tokens: int = 512) -> str:
    tokenizer, model = load_qwen()

    system_msg = (
        "You are a concise assistant. Follow instructions exactly. "
        "Never invent information. Finish every sentence."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]

    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    encoded = tokenizer(chat_text, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0][encoded["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _fmt(value: Any) -> str:
    if value in (None, "", "missed"):
        return "not provided"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def build_prompt_from_report(report: Dict[str, Any]) -> str:
    """Small-model-optimized report prompt.
    Very short rules + hard structure = reliable output.
    """

    student = report.get("student", {})
    period = report.get("period", {})
    usage = report.get("usage", {})
    focus = report.get("focus", {})
    routine = report.get("routine", {})
    ai = report.get("ai_literacy", {}) or {}
    independence = report.get("independence", {}) or {}
    communication = report.get("communication", {}) or {}
    emotional = report.get("emotional_regulation", {}) or {}
    attempts = report.get("activity_performance", {}) or {}

    student_name = student.get("name")
    if not student_name:
        student_name = "not provided"

    lines: List[str] = [
        "Write a factual report using ONLY the information inside the DATA BLOCK.",
        "Never guess or invent information. Never add new numbers.",
        "If something is missing, write 'not provided'.",
        "",
        "Use this exact format and nothing else:",
        "",
        "### Engagement",
        "1 short paragraph.",
        "",
        "### Learning Progress",
        "1 short paragraph.",
        "",
        "### Supports",
        "1 short paragraph.",
        "",
        "### Next Steps",
        "Exactly 3 bullet points. After the 3rd bullet, STOP. Do not write anything else.",
    ]

    # If we actually have a name, force the model to use it.
    if student_name != "not provided":
        lines.append(
            f"The student's name is {student_name}. "
            f"Use this exact name at least once in the Engagement section and once in the Learning Progress section."
        )

    lines.extend(
        [
            "",
            "DATA BLOCK:",
            f"Student name: {student_name}",
            f"Class: {student.get('class','not provided')}",
            f"Reporting window: {period.get('start','not provided')} to {period.get('end','not provided')}",
            f"Focus score: {_fmt(focus.get('focus_score'))}",
            f"Focus score change: {_fmt(focus.get('focus_score_delta'))}",
            f"Active days: {_fmt(usage.get('active_days'))}",
            f"Sessions: {_fmt(usage.get('sessions'))}",
            f"Total time (mins): {_fmt(usage.get('total_time_mins'))}",
            f"Completion %: {_fmt(usage.get('completion_pct'))}",
            f"Dropoff risk: {_fmt(routine.get('dropoff_risk'))}",
            f"Help requests: {_fmt(independence.get('help_requests'))}",
            f"Support rate %: {_fmt(independence.get('support_rate'))}",
            f"MCQ correct %: {_fmt(attempts.get('mcq_correct_pct'))}",
            f"MCQ first-try %: {_fmt(attempts.get('mcq_first_try_success_pct'))}",
            f"Latest zone: {_fmt(emotional.get('latest_zone'))}",
            f"Messages: {_fmt(communication.get('messages'))}",
            f"Interactions: {_fmt(communication.get('interactions'))}",
            "",
            "End of DATA BLOCK.",
        ]
    )

    return "\n".join(lines)


def generate_ai_report(report: Dict[str, Any], max_new_tokens: int = 512) -> str:
    tokenizer, model = load_qwen()

    user_prompt = build_prompt_from_report(report)

    system_msg = (
        "You are a special education teacher. "
        "Follow the exact structure and never add invented details."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_prompt},
    ]

    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    encoded = tokenizer(chat_text, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0][encoded["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python qwen_integration.py \"<prompt>\"")
        raise SystemExit(1)

    prompt = sys.argv[1]
    print(generate_response(prompt))
