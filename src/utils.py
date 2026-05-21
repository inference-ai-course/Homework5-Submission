"""
Utility Functions -- Week 5

Helper functions for token estimation, cost calculation, formatting,
and saving student outputs to the outputs/ directory.
"""

import os
from typing import Dict, Any, Optional


def _resolve_outputs_dir(output_dir: Optional[str]) -> str:
    """
    Pick the right outputs/ dir regardless of cwd.

    - If caller passed an explicit output_dir, honor it.
    - If running from a notebook (cwd ends in 'notebooks') and ../outputs exists, use that.
    - Otherwise fall back to 'outputs' (relative to cwd).
    """
    if output_dir is not None:
        return output_dir
    cwd = os.getcwd()
    if os.path.basename(cwd) == "notebooks" and os.path.isdir(os.path.join(cwd, "..", "outputs")):
        return os.path.normpath(os.path.join(cwd, "..", "outputs"))
    # if os.path.basename(cwd) == "notebooks":
    #     # notebooks/ exists but ../outputs doesn't — create it at the project level
    #     return os.path.normpath(os.path.join(cwd, "..", "outputs"))
    return "outputs"


def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ~ 4 characters in English)."""
    return len(text) // 4


def estimate_cost(
    input_text: str,
    output_tokens: int,
    model: str = "claude-sonnet-4-6"
) -> float:
    """Estimate API call cost in dollars."""
    pricing = {
        "claude-sonnet-4-6":          {"input": 3.0,  "output": 15.0},
        "claude-opus-4-7":            {"input": 15.0, "output": 75.0},
        "claude-haiku-4-5-20251001":  {"input": 1.0,  "output": 5.0},
        "claude-sonnet-4-5-20250929": {"input": 3.0,  "output": 15.0},
    }

    if model not in pricing:
        model = "claude-sonnet-4-6"

    input_tokens = estimate_tokens(input_text)
    input_cost   = (input_tokens  / 1_000_000) * pricing[model]['input']
    output_cost  = (output_tokens / 1_000_000) * pricing[model]['output']
    return input_cost + output_cost


def format_response(response: Dict[str, Any], verbose: bool = True) -> str:
    """Format an LLM response dict for display."""
    if "error" in response:
        return f"Error: {response['error']}"

    output = []
    if verbose:
        output.append("=" * 60)
        output.append(f"Model: {response['model']}")
        output.append(f"Tokens: {response['usage']['input_tokens']} in, "
                      f"{response['usage']['output_tokens']} out")
        output.append(f"Stop reason: {response.get('stop_reason', 'n/a')}")
        output.append("=" * 60)

    output.append(response['content'])

    if verbose:
        output.append("=" * 60)

    return "\n".join(output)


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def save_task_output(
    task_name: str,
    notebook: str,
    prompt: str,
    response: Dict[str, Any],
    system_prompt: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    observations: Optional[str] = None,
    output_dir: Optional[str] = None
) -> str:
    """Save a task's LLM exchange to a markdown file in outputs/."""
    from datetime import datetime

    output_dir = _resolve_outputs_dir(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    content = [
        f"# {task_name}",
        "",
        f"**Notebook:** {notebook}  ",
        f"**Completed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    if system_prompt:
        content.extend(["## System Prompt", "", "```", system_prompt.strip(), "```", ""])

    content.extend(["## User Prompt", "", "```", prompt.strip(), "```", ""])

    if "error" in response:
        content.extend(["## Response", "", f"**Error:** {response['error']}", ""])
    else:
        content.extend(["## Response", "", response['content'], ""])
        content.extend([
            "## Metadata",
            "",
            f"- **Model:** {response['model']}",
            f"- **Input tokens:** {response['usage']['input_tokens']:,}",
            f"- **Output tokens:** {response['usage']['output_tokens']:,}",
            f"- **Total tokens:** {response['usage']['input_tokens'] + response['usage']['output_tokens']:,}",
        ])
        if metadata:
            for key, value in metadata.items():
                content.append(f"- **{key.replace('_', ' ').title()}:** {value}")
        content.append("")

    if observations:
        content.extend(["## Your Observations", "", observations.strip(), ""])

    content.append("---")

    task_slug = task_name.lower().replace(' ', '_').replace(':', '').replace('#', '')
    task_slug = ''.join(c for c in task_slug if c.isalnum() or c == '_')
    filename  = f"notebook{notebook}_{task_slug}.md"
    filepath  = os.path.join(output_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content))

    return filepath


def append_to_reflection(
    notebook: str,
    section_title: str,
    reflection_content: str,
    output_dir: Optional[str] = None
) -> str:
    """Append a reflection section to outputs/homework_reflection.md."""
    from datetime import datetime

    output_dir = _resolve_outputs_dir(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    reflection_file = os.path.join(output_dir, 'homework_reflection.md')

    if not os.path.exists(reflection_file):
        with open(reflection_file, 'w', encoding='utf-8') as f:
            f.write("# Week 5: Fine-Tuning LLMs -- Homework Reflection\n\n")
            f.write("**Student Name:** [Your Name Here]\n\n")
            f.write("**Path Selected:** [A (MLX) / B (HF+TRL) / C (Cloud GPU)]\n\n")
            f.write("---\n\n")

    with open(reflection_file, 'a', encoding='utf-8') as f:
        f.write(f"\n## Notebook {notebook}: {section_title}\n\n")
        f.write(f"**Completed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(reflection_content.strip())
        f.write("\n\n---\n")

    return reflection_file
