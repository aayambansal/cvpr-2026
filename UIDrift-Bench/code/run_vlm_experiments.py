#!/usr/bin/env python3
"""
Run VLM experiments on UI-Drift benchmark.
Tests 3 models via OpenRouter: GPT-4o-mini, GPT-4o, Gemini-2.0-Flash
on all base + drifted images with grounded QA.
"""

import os
import json
import base64
import time
import sys
import traceback
from pathlib import Path
import requests

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
assert API_KEY, "OPENROUTER_API_KEY not set"

DATA_DIR = Path(__file__).parent.parent / "data"
IMG_DIR = DATA_DIR / "images"
RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK_FILE = DATA_DIR / "benchmark.json"

# Models to test
MODELS = {
    "gpt4o-mini": "openai/gpt-4o-mini",
    "gpt4o": "openai/gpt-4o-2024-11-20",
    "gemini-flash": "google/gemini-2.0-flash-001",
}

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query_vlm(model_id, image_path, question, retries=3):
    """Query a VLM with an image and question. Returns answer text."""
    img_b64 = encode_image(image_path)
    
    prompt = f"""You are a UI analysis assistant. Look at this UI screenshot and answer the question precisely.

Question: {question}

Instructions:
1. Answer with ONLY the exact value/text requested - no explanation needed.
2. Also specify the approximate bounding box [x1, y1, x2, y2] as pixel coordinates where you found the evidence in the image (image is 1280x900).
3. Format your response as JSON: {{"answer": "...", "bbox": [x1, y1, x2, y2]}}

Respond ONLY with valid JSON."""

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}",
                        },
                    },
                ],
            }
        ],
        "max_tokens": 200,
        "temperature": 0.0,
    }
    
    for attempt in range(retries):
        try:
            resp = requests.post(OPENROUTER_URL, headers=HEADERS, json=payload, timeout=60)
            if resp.status_code == 429:
                wait = min(30, 2 ** attempt * 5)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            
            # Parse JSON response
            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                # Try to extract answer from text
                return {"answer": content, "bbox": [0, 0, 0, 0]}
                
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  Error after {retries} retries: {e}")
                return {"answer": "ERROR", "bbox": [0, 0, 0, 0]}
    
    return {"answer": "ERROR", "bbox": [0, 0, 0, 0]}


def run_experiments():
    with open(BENCHMARK_FILE) as f:
        benchmark = json.load(f)
    
    total_queries = 0
    for page in benchmark["pages"]:
        n_qa = len(page["qa_pairs"])
        n_variants = 1 + len(page["variants"])  # base + drifted
        total_queries += n_qa * n_variants
    
    total_api_calls = total_queries * len(MODELS)
    print(f"Total queries per model: {total_queries}")
    print(f"Total API calls: {total_api_calls}")
    print(f"Estimated cost: ~${total_api_calls * 0.003:.2f} (GPT-4o-mini bulk)")
    print()
    
    # Sample strategy: run ALL QA on base, sample for drifted to manage cost
    # We'll run 2 QA per page for drifted variants
    MAX_QA_PER_DRIFT = 2
    
    for model_name, model_id in MODELS.items():
        result_file = RESULTS_DIR / f"results_{model_name}.json"
        
        # Check for existing partial results
        existing = {}
        if result_file.exists():
            with open(result_file) as f:
                existing = json.load(f)
            print(f"Resuming {model_name}: {len(existing)} pages already done")
        
        results = existing
        
        for pi, page in enumerate(benchmark["pages"]):
            page_key = str(page["page_id"])
            if page_key in results:
                continue
            
            page_result = {
                "page_id": page["page_id"],
                "page_type": page["page_type"],
                "base_results": [],
                "variant_results": [],
            }
            
            # Run on base image (ALL QA pairs)
            base_path = IMG_DIR / page["base_image"]
            for qa in page["qa_pairs"]:
                vlm_resp = query_vlm(model_id, base_path, qa["question"])
                page_result["base_results"].append({
                    "question": qa["question"],
                    "ground_truth": qa["answer"],
                    "vlm_answer": vlm_resp.get("answer", ""),
                    "vlm_bbox": vlm_resp.get("bbox", [0,0,0,0]),
                    "gt_bbox": qa["evidence_bbox"],
                    "qa_type": qa["type"],
                })
                time.sleep(0.3)  # Rate limit
            
            # Run on drifted variants (sampled QA)
            for variant in page["variants"]:
                variant_path = IMG_DIR / variant["image"]
                sampled_qa = page["qa_pairs"][:MAX_QA_PER_DRIFT]
                
                variant_result = {
                    "drift_type": variant["drift_type"],
                    "severity": variant["severity"],
                    "results": [],
                }
                
                for qa in sampled_qa:
                    vlm_resp = query_vlm(model_id, variant_path, qa["question"])
                    variant_result["results"].append({
                        "question": qa["question"],
                        "ground_truth": qa["answer"],
                        "vlm_answer": vlm_resp.get("answer", ""),
                        "vlm_bbox": vlm_resp.get("bbox", [0,0,0,0]),
                        "gt_bbox": qa["evidence_bbox"],
                        "gt_drift_bbox": list(variant["bboxes"].get(qa["evidence_key"], [0,0,0,0])),
                        "qa_type": qa["type"],
                    })
                    time.sleep(0.3)
                
                page_result["variant_results"].append(variant_result)
            
            results[page_key] = page_result
            
            # Save incrementally
            with open(result_file, "w") as f:
                json.dump(results, f, indent=2)
            
            if (pi + 1) % 5 == 0:
                print(f"  [{model_name}] {pi+1}/{len(benchmark['pages'])} pages done")
        
        print(f"Completed {model_name}: {len(results)} pages")
    
    print("\nAll experiments complete!")


if __name__ == "__main__":
    run_experiments()
