"""
Run remaining VLMs on CounterBench. GPT-4o already complete.
Replace Claude (400 errors) with Claude 3.5 Haiku or working model.
"""

import os
import json
import base64
import time
import sys
from pathlib import Path

import requests

API_KEY = os.environ.get("OPENROUTER_API_KEY")
DATA_DIR = Path(__file__).parent.parent / "data"
IMAGES_DIR = DATA_DIR / "images"
BENCHMARK_PATH = DATA_DIR / "benchmark.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Models that still need running
MODELS = {
    "gemini_flash": "google/gemini-2.0-flash-001",
    "llama_90b": "meta-llama/llama-3.2-90b-vision-instruct",
    "qwen_vl": "qwen/qwen2.5-vl-72b-instruct",
    "claude_haiku": "anthropic/claude-3.5-haiku-20241022",
}

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query_vlm(model_id, question, image_path, max_retries=3):
    b64 = encode_image(image_path)
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://counterbench.research",
        "X-Title": "CounterBench"
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": question + "\n\nIMPORTANT: Give only a short, direct answer. Do not explain your reasoning."}
        ]}],
        "max_tokens": 50,
        "temperature": 0.0
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                time.sleep(min(2 ** attempt * 3, 30))
                continue
            if resp.status_code >= 400:
                err_text = resp.text[:200]
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return f"ERROR:{resp.status_code}"
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"ERROR:{str(e)[:80]}"
    return "ERROR:max_retries"


def run_model(model_name, model_id, benchmark):
    results_path = RESULTS_DIR / f"{model_name}_results.json"
    
    existing = {}
    if results_path.exists():
        with open(results_path) as f:
            for r in json.load(f).get("results", []):
                if not r.get("orig_response", "").startswith("ERROR"):
                    existing[r["id"]] = r
    
    results = []
    items = benchmark["items"]
    errors = 0
    
    for idx, item in enumerate(items):
        if item["id"] in existing:
            results.append(existing[item["id"]])
            continue
        
        orig_path = IMAGES_DIR / item["original_image"]
        int_path = IMAGES_DIR / item["intervened_image"]
        
        orig_resp = query_vlm(model_id, item["question"], orig_path)
        time.sleep(0.25)
        int_resp = query_vlm(model_id, item["question"], int_path)
        time.sleep(0.25)
        
        if orig_resp.startswith("ERROR") or int_resp.startswith("ERROR"):
            errors += 1
        
        results.append({
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "original_answer_gt": item["original_answer"],
            "intervened_answer_gt": item["intervened_answer"],
            "should_flip": item["should_flip"],
            "orig_response": orig_resp,
            "int_response": int_resp,
        })
        
        if (idx + 1) % 50 == 0:
            print(f"  [{model_name}] {idx+1}/{len(items)} ({errors} errors)")
            with open(results_path, "w") as f:
                json.dump({"model": model_name, "model_id": model_id, "results": results}, f)
    
    with open(results_path, "w") as f:
        json.dump({"model": model_name, "model_id": model_id, "results": results}, f)
    
    print(f"  [{model_name}] Done! {len(results)} results, {errors} errors")


def main():
    with open(BENCHMARK_PATH) as f:
        benchmark = json.load(f)
    
    for name, mid in MODELS.items():
        print(f"\n=== {name} ({mid}) ===")
        try:
            run_model(name, mid, benchmark)
        except Exception as e:
            print(f"FATAL {name}: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
