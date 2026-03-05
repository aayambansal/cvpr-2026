#!/usr/bin/env python3
"""Retry failed models with correct OpenRouter model IDs."""
import json, os, sys, time, base64, re, requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMG_DIR = DATA_DIR / "images"
RESULTS_DIR = DATA_DIR / "results"

MODELS = {
    "llama32_11b": "meta-llama/llama-3.2-11b-vision-instruct",
    "qwen25vl72b": "qwen/qwen2.5-vl-72b-instruct",
    "pixtral_large": "mistralai/pixtral-large-2411",
}

SYSTEM_PROMPT = """You are evaluating a visual scene. Answer the question about the image as concisely as possible.
- For yes/no questions, respond with ONLY "yes" or "no".
- For counting questions, respond with ONLY the number.
- Do not explain your reasoning. Just give the direct answer."""

_img_cache = {}
def encode_image(path):
    s = str(path)
    if s not in _img_cache:
        with open(path, "rb") as f:
            _img_cache[s] = base64.b64encode(f.read()).decode("utf-8")
    return _img_cache[s]

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
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": question},
            ]},
        ],
        "max_tokens": 20,
        "temperature": 0.0,
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post("https://openrouter.ai/api/v1/chat/completions",
                               headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                time.sleep(3 ** (attempt + 1))
                continue
            if resp.status_code != 200:
                time.sleep(2)
                continue
            data = resp.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"].strip().lower()
            if "error" in data:
                return f"ERROR: {data['error'].get('message', 'unknown')[:80]}"
            return "ERROR: no choices"
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"ERROR: {str(e)[:80]}"
    return "ERROR: max retries"

def normalize_answer(raw):
    raw = raw.strip().lower()
    if "yes" in raw[:10]: return "yes"
    if "no" in raw[:10]: return "no"
    nums = re.findall(r'\d+', raw)
    if nums: return nums[0]
    clean = re.sub(r'[^a-z0-9]', '', raw)
    if clean in ("yes", "y", "true"): return "yes"
    if clean in ("no", "n", "false"): return "no"
    return clean[:20]

def run_model(model_name, model_id, pairs):
    print(f"\n{'='*60}")
    print(f"Running {model_name} ({model_id}) on {len(pairs)} pairs")
    print(f"{'='*60}")
    
    out_path = RESULTS_DIR / f"{model_name}_results.json"
    existing = {}
    if out_path.exists():
        with open(out_path) as f:
            existing_list = json.load(f)
            existing = {r["id"]: r for r in existing_list if "error" not in r.get("raw_original", "").lower()}
        print(f"  Resuming: {len(existing)} valid results cached")
    
    remaining = [p for p in pairs if p["id"] not in existing]
    results = list(existing.values())
    
    for i, pair in enumerate(remaining):
        orig_path = IMG_DIR / pair["original_image"]
        intv_path = IMG_DIR / pair["intervened_image"]
        
        raw_orig = query_vlm(model_id, orig_path, pair["question"])
        raw_intv = query_vlm(model_id, intv_path, pair["question"])
        
        result = {
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
        results.append(result)
        
        if (i + 1) % 25 == 0 or (i + 1) == len(remaining):
            print(f"  [{model_name}] {len(existing) + i + 1}/{len(pairs)} done")
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
        time.sleep(0.4)
    
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [{model_name}] Complete.")
    return model_name, results

def main():
    with open(DATA_DIR / "counterbench_metadata.json") as f:
        pairs = json.load(f)
    print(f"Loaded {len(pairs)} pairs")
    
    model_filter = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())
    for model_name in model_filter:
        if model_name not in MODELS:
            print(f"Unknown: {model_name}"); continue
        run_model(model_name, MODELS[model_name], pairs)

if __name__ == "__main__":
    main()
