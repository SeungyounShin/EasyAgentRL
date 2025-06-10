#!/usr/bin/env python3
"""
Minimal **multi‑turn tool‑calling** demo for the Qwen2.5‑3b‑it_searchR1‑like model

Key points
-----------
* Supplies the DuckDuckGo *search* tool schema via `tools=[…]` so the model emits JSON‑style calls.
* Detects `<tool_call>` → parses JSON `{name:…, arguments:{query_list:[…]}}` and runs DuckDuckGo for each query.
* Streams the results back inside `<tool_response>` so the model can reason again, up to `MAX_TURNS`.

Install once:
    pip install "duckduckgo_search>=6.3.5"

Run:
    python3 search_r1_infer.py "How is the weather in Seoul?"
"""

from __future__ import annotations

import json
import re
import sys
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from duckduckgo_search import DDGS

# ----------------------------------------------------------------------------
# Color codes for terminal output
# ----------------------------------------------------------------------------
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'

# ----------------------------------------------------------------------------
# Constants & Prompt Template
# ----------------------------------------------------------------------------
DEFAULT_SYSTEM_CONTENT = "You are a helpful and harmless assistant."
DEFAULT_USER_CONTENT_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <think> and "
    "</think> first every time you get new information. After reasoning, if you "
    "find you lack some knowledge, you can call a search engine by <tool_call> "
    "query </tool_call> and it will return the top searched results between "
    "<tool_response> and </tool_response>. You can search as many times as your "
    "want. If you find no further external knowledge needed, you can directly "
    "provide the answer inside <answer> and </answer>, without detailed "
    "illustrations. For example, <answer> Beijing </answer>. Question: "
)

MODEL_NAME = "Seungyoun/qwen2.5-3b-it_searchR1-like-multiturn"
MAX_TURNS = 4
MAX_RESPONSE_TOKENS = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------------------------------
# Tool schema (JSON mirror of search_tool_config.yaml)
# ----------------------------------------------------------------------------
SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Searches the web for relevant information based on the given query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "A list of fully‑formed semantic queries. The tool will return "
                        "search results for each query."
                    ),
                }
            },
            "required": ["query_list"],
        },
    },
}

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

def create_prompt(question: str) -> List[dict]:
    """Build the initial chat prompt."""
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_CONTENT},
        {"role": "user", "content": DEFAULT_USER_CONTENT_PREFIX + question},
    ]


def ddg_search_one(query: str, k: int = 5) -> str:
    """Return top‑k DuckDuckGo results joined by newlines."""
    with DDGS() as ddgs:
        hits = list(ddgs.text(query, safesearch="moderate", max_results=k))
    return "\n".join(
        f"{i+1}. {h['title']} – {h['body']} ({h['href']})" for i, h in enumerate(hits)
    )


def extract_queries(raw: str) -> List[str]:
    """Parse the JSON inside <tool_call> and return the `query_list`. Fallback to raw."""
    try:
        payload = json.loads(raw)
        if (
            isinstance(payload, dict)
            and payload.get("name") == "search"
            and isinstance(payload.get("arguments"), dict)
        ):
            qlist = payload["arguments"].get("query_list", [])
            return [q for q in qlist if isinstance(q, str)]
    except json.JSONDecodeError:
        pass  # raw is not JSON → treat as literal
    return [raw]


# ----------------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------------

def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "How is the weather in Seoul?"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )

    messages = create_prompt(question)
    chat_history = tokenizer.apply_chat_template(
        messages,
        tools=[SEARCH_SCHEMA],  # expose tool to the model
        add_generation_prompt=True,
        tokenize=False,
    )

    tool_call_pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)

    for turn in range(MAX_TURNS):
        enc = tokenizer(chat_history, return_tensors="pt").to(DEVICE)
        out = model.generate(
            **enc,
            max_new_tokens=MAX_RESPONSE_TOKENS,
            temperature=0.7,
            do_sample=True,
        )
        new_text = tokenizer.decode(out[0][enc.input_ids.shape[1] :], skip_special_tokens=True)
        print(f"\n===== Assistant (turn {turn+1}) =====\n{new_text}\n")
        chat_history += new_text

        m = tool_call_pattern.search(new_text)
        if not m:
            break  # finished – no tool call

        queries = extract_queries(m.group(1))
        all_results: list[str] = []
        for q in queries:
            print(f"{Colors.CYAN}{Colors.BOLD}[Tool Call] 검색 쿼리: {q}{Colors.RESET}")
            search_result = ddg_search_one(q, k=5)
            all_results.append(search_result)
            print(f"{Colors.GREEN}[Tool Response]{Colors.RESET}")
            print(f"{Colors.GREEN}{search_result}{Colors.RESET}")
            print(f"{Colors.GREEN}{'='*50}{Colors.RESET}\n")
        
        tool_response_block = "<tool_response>\n" + "\n---\n".join(all_results) + "\n</tool_response>"
        chat_history += tool_response_block  # feed back into next turn


if __name__ == "__main__":
    main()
