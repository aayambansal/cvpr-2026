#!/usr/bin/env python3
"""
Batch runner v2: Fixed prompts to emphasize text-based reasoning.
Usage: python run_batch_v2.py <model_name> <strategy_name>
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


# ── Prompt Templates (v2: text-based reasoning emphasis) ──────────────

def make_context(example):
    """Create a context block from the MCQ options and question, 
    framing the task as text-based commonsense reasoning."""
    task = example["task"]
    if task == "Detective":
        return (
            "SCENARIO: A video has three segments: a pre-event, a hidden middle event, "
            "and a post-event. You cannot see the middle segment. "
            "Based on commonsense reasoning about the described scenario, "
            "determine what most likely happened during the hidden middle event."
        )
    else:  # Reporter
        return (
            "SCENARIO: A video shows an unexpected or surprising event from start to finish. "
            "Based on commonsense reasoning about the described scenario, "
            "determine which description best captures what happens."
        )


def build_naive_prompt(example):
    ctx = make_context(example)
    q = example["default_question"]
    opts = example["mcq_options"]
    options_str = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(opts))
    return (
        f"{ctx}\n\n{q}\n\n{options_str}\n\n"
        f"Based on commonsense reasoning, answer with ONLY the letter (A, B, or C)."
    )


def build_cot_prompt(example):
    ctx = make_context(example)
    q = example["default_question"]
    opts = example["mcq_options"]
    options_str = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(opts))
    return (
        f"{ctx}\n\n{q}\n\n{options_str}\n\n"
        f"Think step by step about which option is most plausible "
        f"given commonsense knowledge about physical events and causation.\n"
        f"Then provide your final answer as a single letter (A, B, or C).\n\nReasoning:"
    )


def build_abductive_cot_prompt(example):
    task = example["task"]
    ctx = make_context(example)
    q = example["default_question"]
    opts = example["mcq_options"]
    if task == "Detective":
        scaffold = (
            "Use ABDUCTIVE REASONING (inference to the best explanation):\n"
            "1. CLUES: What information do the answer options reveal about context?\n"
            "2. CAUSAL CHAIN: For each option, trace the causal chain — "
            "does it plausibly connect a setup to an outcome?\n"
            "3. BEST EXPLANATION: Which option provides the most coherent "
            "causal bridge for a hidden event?"
        )
    else:
        scaffold = (
            "Use DEFEASIBLE REASONING (updating beliefs given new evidence):\n"
            "1. DEFAULT EXPECTATION: What would typically happen in this scenario?\n"
            "2. ANOMALY: Which option describes something unexpected or surprising?\n"
            "3. BELIEF REVISION: Update your expectation — which option "
            "describes what actually happened?"
        )
    options_str = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(opts))
    return (
        f"{ctx}\n\n{q}\n\n{options_str}\n\n{scaffold}\n\n"
        f"Provide reasoning, then your final answer as a single letter (A, B, or C).\n\nReasoning:"
    )


def build_hyp_eliminate_prompt(example):
    ctx = make_context(example)
    q = example["default_question"]
    opts = example["mcq_options"]
    options_str = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(opts))
    return (
        f"{ctx}\n\n{q}\n\n{options_str}\n\n"
        f"ELIMINATION STRATEGY:\n"
        f"For EACH option (A, B, C):\n"
        f"  + One reason it COULD be correct\n"
        f"  - One reason it COULD be wrong\n"
        f"  Plausibility: HIGH / MEDIUM / LOW\n\n"
        f"Eliminate LOW options. Choose the best remaining option.\n"
        f"End with your final answer letter (A, B, or C).\n\nAnalysis:"
    )


def build_counterfactual_prompt(example):
    task = example["task"]
    ctx = make_context(example)
    q = example["default_question"]
    opts = example["mcq_options"]
    if task == "Detective":
        cf = (
            "COUNTERFACTUAL REASONING:\n"
            "For each option, imagine a world where that event occurred.\n"
            "- Would the described aftermath make sense?\n"
            "- Is the causal mechanism physically plausible?\n"
            "- Which counterfactual world is most consistent with reality?"
        )
    else:
        cf = (
            "COUNTERFACTUAL REASONING:\n"
            "First, consider the EXPECTED normal outcome.\n"
            "Then for each option: if this were what happened, "
            "does it represent a surprising/unexpected deviation?\n"
            "Which option represents the actual unexpected outcome?"
        )
    options_str = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(opts))
    return (
        f"{ctx}\n\n{q}\n\n{options_str}\n\n{cf}\n\n"
        f"Answer with a single letter (A, B, or C) at the end.\n\nReasoning:"
    )


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
                    "messages": [
                        {"role": "system", "content": "You are a commonsense reasoning assistant. Answer the multiple-choice question based on reasoning about physical events and causation. Always provide a clear answer letter."},
                        {"role": "user", "content": prompt},
                    ],
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
    # Explicit answer patterns
    m = re.search(r'(?:final\s+)?answer[:\s]*\(?([A-C])\)?', response, re.IGNORECASE)
    if m: return m.group(1).upper()
    # Standalone letter at end
    m = re.search(r'\b([A-C])\b\s*\.?\s*$', last_line)
    if m: return m.group(1).upper()
    # (X) pattern in last line
    m = re.search(r'\(([A-C])\)', last_line)
    if m: return m.group(1).upper()
    # **X** pattern
    m = re.search(r'\*\*([A-C])\*\*', response)
    if m: return m.group(1).upper()
    # First A-C
    m = re.search(r'\b([A-C])\b', response)
    if m: return m.group(1).upper()
    return None


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

    # Save sample IDs
    sid_file = os.path.join(RESULTS_DIR, "sample_ids.json")
    if not os.path.exists(sid_file):
        with open(sid_file, "w") as f:
            json.dump([ex["q_id"] for ex in samples], f)

    print(f"Running {key} on {len(samples)} samples...", flush=True)

    results = []
    correct = 0
    total = 0

    for i, ex in enumerate(samples):
        prompt = prompt_fn(ex)
        response = call_openrouter(model_id, prompt)
        answer = extract_answer(response)
        pred_idx = ord(answer) - ord('A') if answer else None
        is_correct = pred_idx == ex["mcq_gt_option"]
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "q_id": ex["q_id"],
            "task": ex["task"],
            "difficulty": ex["difficulty"],
            "gt_idx": ex["mcq_gt_option"],
            "pred_idx": pred_idx,
            "pred_letter": answer,
            "correct": is_correct,
            "response_snippet": response[:300] if response else None,
            "full_response": response,
        })

        if (i + 1) % 25 == 0:
            print(f"  [{key}] {i+1}/{len(samples)} — acc={correct/total*100:.1f}%", flush=True)
        time.sleep(0.08)

    accuracy = correct / total * 100
    print(f"  FINAL [{key}]: {accuracy:.1f}% ({correct}/{total})", flush=True)

    with open(outfile, "w") as f:
        json.dump({
            "strategy": strategy_name,
            "model": model_name,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results,
        }, f, indent=2)


if __name__ == "__main__":
    main()
