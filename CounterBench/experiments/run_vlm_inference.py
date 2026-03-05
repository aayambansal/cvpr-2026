"""
CounterBench VLM Inference: Run 5 VLMs on all 550 paired images via OpenRouter.
"""

import os
import json
import base64
import time
import sys
import traceback
from pathlib import Path

try:
    import requests
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    print("ERROR: OPENROUTER_API_KEY not set")
    sys.exit(1)

DATA_DIR = Path(__file__).parent.parent / "data"
IMAGES_DIR = DATA_DIR / "images"
BENCHMARK_PATH = DATA_DIR / "benchmark.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Models to test
MODELS = {
    "gpt4o": "openai/gpt-4o",
    "claude_sonnet": "anthropic/claude-sonnet-4-20250514",
    "gemini_flash": "google/gemini-2.0-flash-001",
    "llama_90b": "meta-llama/llama-3.2-90b-vision-instruct",
    "qwen_vl": "qwen/qwen2.5-vl-72b-instruct",
}

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query_vlm(model_id, question, image_path, max_retries=3):
    """Query a VLM with an image and question via OpenRouter."""
    b64 = encode_image(image_path)
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://counterbench.research",
        "X-Title": "CounterBench VLM Evaluation"
    }
    
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": question + "\n\nIMPORTANT: Give only a short, direct answer. Do not explain your reasoning."
                    }
                ]
            }
        ],
        "max_tokens": 50,
        "temperature": 0.0
    }
    
    for attempt in range(max_retries):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                wait = min(2 ** attempt * 2, 30)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            answer = data["choices"][0]["message"]["content"].strip()
            return answer
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    Failed after {max_retries} retries: {e}")
                return f"ERROR: {str(e)}"
    return "ERROR: max retries"


def run_model(model_name, model_id, benchmark):
    """Run a single model on all benchmark items."""
    results_path = RESULTS_DIR / f"{model_name}_results.json"
    
    # Load existing results for resumption
    existing = {}
    if results_path.exists():
        with open(results_path) as f:
            existing_data = json.load(f)
            for r in existing_data.get("results", []):
                existing[r["id"]] = r
    
    results = []
    items = benchmark["items"]
    total = len(items)
    
    for idx, item in enumerate(items):
        item_id = item["id"]
        
        # Skip if already done
        if item_id in existing and "orig_response" in existing[item_id] and "int_response" in existing[item_id]:
            results.append(existing[item_id])
            if (idx + 1) % 50 == 0:
                print(f"  [{model_name}] {idx+1}/{total} (cached)")
            continue
        
        orig_path = IMAGES_DIR / item["original_image"]
        int_path = IMAGES_DIR / item["intervened_image"]
        question = item["question"]
        
        # Query original
        orig_response = query_vlm(model_id, question, orig_path)
        time.sleep(0.3)
        
        # Query intervened
        int_response = query_vlm(model_id, question, int_path)
        time.sleep(0.3)
        
        result = {
            "id": item_id,
            "category": item["category"],
            "question": question,
            "original_answer_gt": item["original_answer"],
            "intervened_answer_gt": item["intervened_answer"],
            "should_flip": item["should_flip"],
            "orig_response": orig_response,
            "int_response": int_response,
        }
        results.append(result)
        
        if (idx + 1) % 25 == 0:
            print(f"  [{model_name}] {idx+1}/{total} done")
            # Save checkpoint
            checkpoint = {"model": model_name, "model_id": model_id, "results": results}
            with open(results_path, "w") as f:
                json.dump(checkpoint, f, indent=2)
    
    # Final save
    output = {"model": model_name, "model_id": model_id, "results": results}
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"  [{model_name}] Complete! {len(results)} results saved.")
    return output


def main():
    print("Loading benchmark...")
    with open(BENCHMARK_PATH) as f:
        benchmark = json.load(f)
    
    print(f"Benchmark: {benchmark['total_pairs']} pairs")
    
    for model_name, model_id in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Running model: {model_name} ({model_id})")
        print(f"{'='*60}")
        try:
            run_model(model_name, model_id, benchmark)
        except Exception as e:
            print(f"ERROR with {model_name}: {e}")
            traceback.print_exc()
    
    print("\n\nAll models complete!")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
