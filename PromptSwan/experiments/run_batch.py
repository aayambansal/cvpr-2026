#!/usr/bin/env python3
"""
Batch runner: Run a SINGLE (model, strategy) pair.
Usage: python run_batch.py <model_name> <strategy_name>
"""

import os
import json
import time
import random
import sys
import re
from collections import Counter, defaultdict
from datasets import load_dataset
import requests
import concurrent.futures

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY")
assert OPENROUTER_KEY, "OPENROUTER_API_KEY not set"

MODELS = {
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "claude-haiku": "anthropic/claude-3.5-haiku",
    "gemini-flash": "google/gemini-2.0-flash-001",
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SAMPLE_SIZE = 200


def stratified_sample(dataset, n, seed=42):
    random.seed(seed)
    groups = defaultdict(list)
    for ex in dataset:
        key = (ex["task"], ex["difficulty"])
        groups[key].append(ex)
    total = sum(len(v) for v in groups.values())
    sampled = []
    for key, items in groups.items():
        k = max(1, round(n * len(items) / total))
        sampled.extend(random.sample(items, min(k, len(items))))
    random.shuffle(sampled)
    return sampled[:n]


# ── Prompt Templates ─────────────────────────────────────────────────

def build_naive_prompt(example):
    task = example["task"]
    q = example["default_question"]
    opts = example["mcq_options"]
    ctx = ("You are watching a video. You can see the beginning and end, but the middle is hidden."
           if task == "Detective"
           else "You are watching a complete video showing an unexpected event.")
    options_str = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(opts))
    return f"{ctx}\n\n{q}\n\n{options_str}\n\nAnswer with ONLY the letter (A, B, or C)."


def build_cot_prompt(example):
    task = example["task"]
    q = example["default_question"]
    opts = example["mcq_options"]
    ctx = ("You are watching a video. You can see the beginning and end, but the middle is hidden."
           if task == "Detective"
           else "You are watching a complete video showing an unexpected event.")
    options_str = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(opts))
    return (f"{ctx}\n\n{q}\n\n{options_str}\n\n"
            f"Think step by step, then answer with a single letter (A, B, or C).\n\nReasoning:")


def build_abductive_cot_prompt(example):
    task = example["task"]
    q = example["default_question"]
    opts = example["mcq_options"]
    if task == "Detective":
        ctx = ("You are watching a video where the MIDDLE frames are hidden. "
               "Reason abductively: given before and after, infer the hidden cause.")
        scaffold = ("1. OBSERVATION: What do before/after tell you?\n"
                    "2. HYPOTHESIS: For each option, does it explain the transition?\n"
                    "3. BEST EXPLANATION: Which is most plausible?")
    else:
        ctx = ("You are watching a complete video of an unexpected event. "
               "Reason defeasibly: the surprise may override expectations.")
        scaffold = ("1. EXPECTATION: What would normally happen?\n"
                    "2. SURPRISE: What unexpected thing occurs?\n"
                    "3. BELIEF UPDATE: Which option matches reality?")
    options_str = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(opts))
    return (f"{ctx}\n\n{q}\n\n{options_str}\n\n{scaffold}\n\n"
            f"Answer with a single letter (A, B, or C) at the end.\n\nReasoning:")


def build_hyp_eliminate_prompt(example):
    task = example["task"]
    q = example["default_question"]
    opts = example["mcq_options"]
    ctx = ("You are watching a video where the middle is hidden."
           if task == "Detective"
           else "You are watching a complete video of a surprising event.")
    options_str = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(opts))
    return (f"{ctx}\n\n{q}\n\n{options_str}\n\n"
            f"For EACH option: give one reason FOR, one AGAINST, rate HIGH/MED/LOW.\n"
            f"Eliminate LOW options. Choose the best.\n"
            f"End with your final answer letter (A, B, or C).\n\nAnalysis:")


def build_counterfactual_prompt(example):
    task = example["task"]
    q = example["default_question"]
    opts = example["mcq_options"]
    if task == "Detective":
        ctx = "You are watching a video where the middle frames are hidden."
        cf = ("For each option, imagine that event happened during the hidden segment. "
              "Would the aftermath make sense? Pick the best fit.")
    else:
        ctx = "You are watching a complete video. Something unexpected happens."
        cf = ("Consider what you'd normally expect. Then for each option, "
              "does it match the surprising nature? Pick the actual outcome.")
    options_str = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(opts))
    return (f"{ctx}\n\n{q}\n\n{options_str}\n\n{cf}\n\n"
            f"Answer with a single letter (A, B, or C) at the end.\n\nReasoning:")


STRATEGIES = {
    "naive": build_naive_prompt,
    "cot": build_cot_prompt,
    "abductive_cot": build_abductive_cot_prompt,
    "hyp_eliminate": build_hyp_eliminate_prompt,
    "counterfactual": build_counterfactual_prompt,
}


def call_openrouter(model_id, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 512,
                    "temperature": 0.0,
                },
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt + random.random())
            else:
                return None


def extract_answer(response):
    if not response:
        return None
    response = response.strip()
    last_line = response.split("\n")[-1].strip()
    m = re.search(r'(?:final\s+)?answer[:\s]*\(?([A-C])\)?', response, re.IGNORECASE)
    if m: return m.group(1).upper()
    m = re.search(r'\b([A-C])\b\s*\.?\s*$', last_line)
    if m: return m.group(1).upper()
    m = re.search(r'\(([A-C])\)', last_line)
    if m: return m.group(1).upper()
    m = re.search(r'\b([A-C])\b', response)
    if m: return m.group(1).upper()
    return None


def process_one(args):
    """Process a single example. Used for concurrent execution."""
    ex, prompt_fn, model_id = args
    prompt = prompt_fn(ex)
    response = call_openrouter(model_id, prompt)
    answer = extract_answer(response)
    pred_idx = ord(answer) - ord('A') if answer else None
    return {
        "q_id": ex["q_id"],
        "task": ex["task"],
        "difficulty": ex["difficulty"],
        "gt_idx": ex["mcq_gt_option"],
        "pred_idx": pred_idx,
        "pred_letter": answer,
        "correct": pred_idx == ex["mcq_gt_option"],
        "response_snippet": response[:300] if response else None,
        "full_response": response,
    }


def main():
    model_name = sys.argv[1]
    strategy_name = sys.argv[2]

    key = f"{model_name}__{strategy_name}"
    outfile = os.path.join(RESULTS_DIR, f"{key}.json")

    if os.path.exists(outfile):
        print(f"SKIP {key} (exists)")
        return

    model_id = MODELS[model_name]
    prompt_fn = STRATEGIES[strategy_name]

    ds = load_dataset("UBC-ViL/BlackSwanSuite-MCQ", split="validation")
    samples = stratified_sample(ds, SAMPLE_SIZE)

    print(f"Running {key} on {len(samples)} samples...", flush=True)

    results = []
    correct = 0
    total = 0

    for i, ex in enumerate(samples):
        r = process_one((ex, prompt_fn, model_id))
        results.append(r)
        if r["correct"]:
            correct += 1
        total += 1
        if (i + 1) % 25 == 0:
            print(f"  [{key}] {i+1}/{len(samples)} — acc={correct/total*100:.1f}%", flush=True)
        time.sleep(0.08)

    accuracy = correct / total * 100
    print(f"  FINAL [{key}]: {accuracy:.1f}% ({correct}/{total})", flush=True)

    out = {
        "strategy": strategy_name,
        "model": model_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }
    with open(outfile, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
