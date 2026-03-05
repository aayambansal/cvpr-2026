#!/usr/bin/env python3
"""
Optimized batch VLM inference for CounterBench.
Uses concurrent requests and saves partial results for resilience.
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

# Cache encoded images
_img_cache = {}

def encode_image(path):
    path_str = str(path)
    if path_str not in _img_cache:
        with open(path, "rb") as f:
            _img_cache[path_str] = base64.b64encode(f.read()).decode("utf-8")
    return _img_cache[path_str]


def query_vlm(model_id, image_path, question, max_retries=3):
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
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
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
                headers=headers, json=payload, timeout=60,
            )
            if resp.status_code == 429:
                time.sleep(2 ** (attempt + 2))
                continue
            if resp.status_code != 200:
                time.sleep(1)
                continue
            data = resp.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"].strip().lower()
            return "ERROR: no choices"
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"ERROR: {str(e)[:80]}"
    return "ERROR: max retries"


def normalize_answer(raw):
    raw = raw.strip().lower()
    # For yes/no
    if "yes" in raw[:10]:
        return "yes"
    if "no" in raw[:10]:
        return "no"
    nums = re.findall(r'\d+', raw)
    if nums:
        return nums[0]
    clean = re.sub(r'[^a-z0-9]', '', raw)
    if clean in ("yes", "y", "true"): return "yes"
    if clean in ("no", "n", "false"): return "no"
    return clean[:20]


def process_pair(model_id, pair):
    """Process one pair (2 queries)."""
    orig_path = IMG_DIR / pair["original_image"]
    intv_path = IMG_DIR / pair["intervened_image"]
    
    raw_orig = query_vlm(model_id, orig_path, pair["question"])
    raw_intv = query_vlm(model_id, intv_path, pair["question"])
    
    return {
        "id": pair["id"],
        "category": pair["category"],
        "subcategory": pair["subcategory"],
        "intervention": pair["intervention"],
        "question": pair["question"],
        "gt_original": pair["answer_original"],
        "gt_intervened": pair["answer_intervened"],
        "should_change": pair["should_change"],
        "pred_original": normalize_answer(raw_orig),
        "pred_intervened": normalize_answer(raw_intv),
        "raw_original": raw_orig,
        "raw_intervened": raw_intv,
    }


def run_model(model_name, model_id, pairs):
    """Run model with concurrency."""
    print(f"\n{'='*60}")
    print(f"Running {model_name} ({model_id}) on {len(pairs)} pairs")
    print(f"{'='*60}")
    
    # Check for existing partial results
    out_path = RESULTS_DIR / f"{model_name}_results.json"
    existing = {}
    if out_path.exists():
        with open(out_path) as f:
            existing_list = json.load(f)
            existing = {r["id"]: r for r in existing_list}
        print(f"  Resuming: {len(existing)} already done")
    
    remaining = [p for p in pairs if p["id"] not in existing]
    results = list(existing.values())
    
    # Process with thread pool (3 concurrent to avoid rate limits)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}
        for pair in remaining:
            future = executor.submit(process_pair, model_id, pair)
            futures[future] = pair["id"]
        
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"  Error: {e}")
            
            if (i + 1) % 50 == 0 or (i + 1) == len(remaining):
                print(f"  [{model_name}] {len(existing) + i + 1}/{len(pairs)} done")
                # Save partial results
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)
    
    # Final save
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [{model_name}] Complete. Saved to {out_path}")
    return model_name, results


def main():
    meta_path = DATA_DIR / "counterbench_metadata.json"
    with open(meta_path) as f:
        pairs = json.load(f)
    print(f"Loaded {len(pairs)} pairs")
    
    model_filter = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())
    
    all_results = {}
    for model_name in model_filter:
        if model_name not in MODELS:
            print(f"Unknown model: {model_name}")
            continue
        model_id = MODELS[model_name]
        name, results = run_model(model_name, model_id, pairs)
        all_results[name] = results
    
    combined_path = RESULTS_DIR / "all_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {combined_path}")


if __name__ == "__main__":
    main()
