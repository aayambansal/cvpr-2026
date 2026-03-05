#!/usr/bin/env python3
"""
Run VLM inference on CounterBench pairs via OpenRouter.

Models tested:
  - GPT-4o (openai/gpt-4o-2024-11-20)
  - Claude 3.5 Sonnet (anthropic/claude-3.5-sonnet)
  - Gemini 2.0 Flash (google/gemini-2.0-flash-001)
  - Llama 3.2 90B Vision (meta-llama/llama-3.2-90b-vision-instruct)
  - Qwen2 VL 72B (qwen/qwen2-vl-72b-instruct)
  - InternVL2.5 78B (openai/internvl2.5-78b:free  -- or via another provider)
  - Gemini 2.5 Flash (google/gemini-2.5-flash-preview)

We send each image with a yes/no or short-answer question and parse the answer.
"""

import json
import os
import sys
import time
import base64
import re
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not API_KEY:
    print("ERROR: OPENROUTER_API_KEY not set")
    sys.exit(1)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMG_DIR = DATA_DIR / "images"
RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODELS = {
    "gpt4o": "openai/gpt-4o-2024-11-20",
    "claude35sonnet": "anthropic/claude-3.5-sonnet",
    "gemini2flash": "google/gemini-2.0-flash-001",
    "llama32_90b": "meta-llama/llama-3.2-90b-vision-instruct",
    "qwen2vl72b": "qwen/qwen2-vl-72b-instruct",
    "gemini25flash": "google/gemini-2.5-flash-preview",
}

SYSTEM_PROMPT = """You are evaluating a visual scene. Answer the question about the image as concisely as possible.
- For yes/no questions, respond with ONLY "yes" or "no".
- For counting questions, respond with ONLY the number.
- Do not explain your reasoning. Just give the direct answer."""


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query_vlm(model_id, image_path, question, max_retries=3):
    """Send a single image + question to a VLM via OpenRouter."""
    b64 = encode_image(image_path)
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://syntheticsciences.ai",
        "X-Title": "CounterBench",
    }
    
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": question},
                ],
            },
        ],
        "max_tokens": 20,
        "temperature": 0.0,
    }
    
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            answer = data["choices"][0]["message"]["content"].strip().lower()
            return answer
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"ERROR: {str(e)}"
    return "ERROR: max retries"


def normalize_answer(raw):
    """Normalize model output to canonical form."""
    raw = raw.strip().lower()
    raw = re.sub(r'[^a-z0-9]', '', raw)
    
    if raw in ("yes", "y", "true", "correct"):
        return "yes"
    if raw in ("no", "n", "false", "incorrect"):
        return "no"
    # Try to extract a number
    nums = re.findall(r'\d+', raw)
    if nums:
        return nums[0]
    return raw


def run_model(model_name, model_id, pairs):
    """Run a single model on all pairs."""
    results = []
    total = len(pairs) * 2  # original + intervened
    done = 0
    
    print(f"\n{'='*60}")
    print(f"Running {model_name} ({model_id})")
    print(f"{'='*60}")
    
    for i, pair in enumerate(pairs):
        # Query original
        orig_path = IMG_DIR / pair["original_image"]
        raw_orig = query_vlm(model_id, orig_path, pair["question"])
        ans_orig = normalize_answer(raw_orig)
        done += 1
        
        # Query intervened
        intv_path = IMG_DIR / pair["intervened_image"]
        raw_intv = query_vlm(model_id, intv_path, pair["question"])
        ans_intv = normalize_answer(raw_intv)
        done += 1
        
        result = {
            "id": pair["id"],
            "category": pair["category"],
            "subcategory": pair["subcategory"],
            "intervention": pair["intervention"],
            "question": pair["question"],
            "gt_original": pair["answer_original"],
            "gt_intervened": pair["answer_intervened"],
            "should_change": pair["should_change"],
            "pred_original": ans_orig,
            "pred_intervened": ans_intv,
            "raw_original": raw_orig,
            "raw_intervened": raw_intv,
        }
        results.append(result)
        
        if (i + 1) % 25 == 0:
            print(f"  [{model_name}] {i+1}/{len(pairs)} pairs done")
        
        # Small delay to avoid rate limits
        time.sleep(0.3)
    
    # Save results
    out_path = RESULTS_DIR / f"{model_name}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [{model_name}] Results saved to {out_path}")
    
    return model_name, results


def main():
    # Load metadata
    meta_path = DATA_DIR / "counterbench_metadata.json"
    with open(meta_path) as f:
        pairs = json.load(f)
    
    print(f"Loaded {len(pairs)} pairs")
    
    # Run which models?
    model_filter = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())
    
    all_results = {}
    for model_name in model_filter:
        if model_name not in MODELS:
            print(f"Unknown model: {model_name}")
            continue
        model_id = MODELS[model_name]
        name, results = run_model(model_name, model_id, pairs)
        all_results[name] = results
    
    # Save combined results
    combined_path = RESULTS_DIR / "all_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {combined_path}")


if __name__ == "__main__":
    main()
