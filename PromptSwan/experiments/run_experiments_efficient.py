#!/usr/bin/env python3
"""
Efficient experiment runner: 300-sample stratified subset per (strategy, model).
Stratified by task (Detective/Reporter) and difficulty (easy/medium/hard).
Total: 5 strategies x 3 models x 300 samples = 4,500 API calls.
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

SAMPLE_SIZE = 300  # per experiment


def stratified_sample(dataset, n, seed=42):
    """Stratified sample preserving task x difficulty distribution."""
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

    # Adjust to exactly n
    random.shuffle(sampled)
    return sampled[:n]


# ── Prompt Templates ─────────────────────────────────────────────────

def build_naive_prompt(example):
    task = example["task"]
    q = example["default_question"]
    opts = example["mcq_options"]
    if task == "Detective":
        context = (
            "You are watching a video. You can see the beginning (pre-event) "
            "and the end (post-event), but the middle frames are hidden."
        )
    else:
        context = (
            "You are watching a complete video showing an unexpected event."
        )
    options_str = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(opts))
    return (
        f"{context}\n\n{q}\n\n{options_str}\n\n"
        f"Answer with ONLY the letter (A, B, or C)."
    )


def build_cot_prompt(example):
    task = example["task"]
    q = example["default_question"]
    opts = example["mcq_options"]
    if task == "Detective":
        context = (
            "You are watching a video. You can see the beginning (pre-event) "
            "and the end (post-event), but the middle frames are hidden."
        )
    else:
        context = (
            "You are watching a complete video showing an unexpected event."
        )
    options_str = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(opts))
    return (
        f"{context}\n\n{q}\n\n{options_str}\n\n"
        f"Think step by step. Consider what is described in each option "
        f"and how it relates to the video context.\n"
        f"Then provide your final answer as a single letter (A, B, or C).\n\n"
        f"Reasoning:"
    )


def build_abductive_cot_prompt(example):
    task = example["task"]
    q = example["default_question"]
    opts = example["mcq_options"]
    if task == "Detective":
        context = (
            "You are watching a video. You can see the beginning (pre-event) "
            "and the end (post-event), but the MIDDLE frames are hidden. "
            "You must reason abductively: given the before and after, "
            "infer the most likely hidden cause."
        )
        scaffold = (
            "Follow this abductive reasoning process:\n"
            "1. OBSERVATION: What do you know from the pre-event and post-event?\n"
            "2. HYPOTHESIS GENERATION: For each option, does it explain the "
            "transition from pre-event to post-event?\n"
            "3. BEST EXPLANATION: Which option is the most plausible hidden "
            "cause that bridges the gap between before and after?"
        )
    else:
        context = (
            "You are watching a complete video showing an unexpected event. "
            "You must reason defeasibly: the unexpected event may override "
            "your initial expectations. What actually happened?"
        )
        scaffold = (
            "Follow this defeasible reasoning process:\n"
            "1. INITIAL EXPECTATION: What would you normally expect to happen?\n"
            "2. SURPRISING EVIDENCE: What unexpected event occurs?\n"
            "3. BELIEF UPDATE: Which option correctly describes what happened, "
            "even if it defies expectations?"
        )
    options_str = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(opts))
    return (
        f"{context}\n\n{q}\n\n{options_str}\n\n{scaffold}\n\n"
        f"Provide your reasoning, then your final answer as a single letter (A, B, or C).\n\n"
        f"Reasoning:"
    )


def build_hypothesis_eliminate_prompt(example):
    task = example["task"]
    q = example["default_question"]
    opts = example["mcq_options"]
    if task == "Detective":
        context = (
            "You are watching a video where the middle (main event) frames "
            "are hidden. You see only before and after."
        )
    else:
        context = (
            "You are watching a complete video showing a surprising event."
        )
    options_str = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(opts))
    return (
        f"{context}\n\n{q}\n\n{options_str}\n\n"
        f"Use the following elimination strategy:\n"
        f"For EACH option:\n"
        f"  - State one reason it COULD be correct.\n"
        f"  - State one reason it COULD be wrong.\n"
        f"  - Rate plausibility: HIGH / MEDIUM / LOW.\n\n"
        f"Then eliminate the LOW-plausibility options and choose the best.\n"
        f"End with your final answer as a single letter (A, B, or C).\n\n"
        f"Analysis:"
    )


def build_counterfactual_prompt(example):
    task = example["task"]
    q = example["default_question"]
    opts = example["mcq_options"]
    if task == "Detective":
        context = (
            "You are watching a video where the middle frames are hidden. "
            "You can see the setup and the aftermath."
        )
        cf_scaffold = (
            "Think counterfactually:\n"
            "For each option, imagine a world where that event happened "
            "during the hidden middle segment.\n"
            "- If option X happened, would the aftermath you see make sense?\n"
            "- Which hypothetical world best explains the transition from "
            "setup to aftermath?"
        )
    else:
        context = (
            "You are watching a complete video. Something unexpected happens."
        )
        cf_scaffold = (
            "Think counterfactually:\n"
            "First, consider what you would EXPECT to happen normally.\n"
            "Then, for each option, consider: If this were what actually "
            "happened, does it match the surprising nature of the video?\n"
            "Which option represents the actual unexpected outcome?"
        )
    options_str = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(opts))
    return (
        f"{context}\n\n{q}\n\n{options_str}\n\n{cf_scaffold}\n\n"
        f"Provide your reasoning, then answer with a single letter (A, B, or C).\n\n"
        f"Reasoning:"
    )


STRATEGIES = {
    "naive": build_naive_prompt,
    "cot": build_cot_prompt,
    "abductive_cot": build_abductive_cot_prompt,
    "hyp_eliminate": build_hypothesis_eliminate_prompt,
    "counterfactual": build_counterfactual_prompt,
}


# ── LLM Calling ──────────────────────────────────────────────────────

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
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return content
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt + random.random()
                time.sleep(wait)
            else:
                return None


def extract_answer(response):
    if response is None:
        return None
    response = response.strip()
    last_line = response.strip().split("\n")[-1].strip()

    # Pattern: "Answer: X" or "Final answer: X"
    match = re.search(r'(?:final\s+)?answer[:\s]*\(?([A-C])\)?', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Standalone letter at end of last line
    match = re.search(r'\b([A-C])\b\s*\.?\s*$', last_line)
    if match:
        return match.group(1).upper()

    # (A), (B), (C) pattern in last line
    match = re.search(r'\(([A-C])\)', last_line)
    if match:
        return match.group(1).upper()

    # First A-C in response
    match = re.search(r'\b([A-C])\b', response)
    if match:
        return match.group(1).upper()

    return None


def letter_to_idx(letter):
    if letter is None:
        return None
    return ord(letter) - ord('A')


# ── Main ─────────────────────────────────────────────────────────────

def run_one(dataset_samples, strategy_name, model_name, model_id):
    prompt_fn = STRATEGIES[strategy_name]
    results = []
    correct = 0
    total = 0

    for i, ex in enumerate(dataset_samples):
        prompt = prompt_fn(ex)
        response = call_openrouter(model_id, prompt)
        answer = extract_answer(response)
        pred_idx = letter_to_idx(answer)
        gt_idx = ex["mcq_gt_option"]
        is_correct = (pred_idx == gt_idx)
        if is_correct:
            correct += 1
        total += 1

        result = {
            "q_id": ex["q_id"],
            "task": ex["task"],
            "difficulty": ex["difficulty"],
            "gt_idx": gt_idx,
            "pred_idx": pred_idx,
            "pred_letter": answer,
            "correct": is_correct,
            "response_snippet": response[:300] if response else None,
            "full_response": response if response else None,
        }
        results.append(result)

        if (i + 1) % 25 == 0:
            acc = correct / total * 100
            print(f"  [{model_name}|{strategy_name}] {i+1}/{len(dataset_samples)} — acc={acc:.1f}%",
                  flush=True)

        time.sleep(0.12)

    accuracy = correct / total * 100 if total > 0 else 0
    return {
        "strategy": strategy_name,
        "model": model_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


def main():
    print("Loading BlackSwanSuite-MCQ validation...", flush=True)
    ds = load_dataset("UBC-ViL/BlackSwanSuite-MCQ", split="validation")
    print(f"Loaded {len(ds)} examples", flush=True)

    # Create stratified sample
    samples = stratified_sample(ds, SAMPLE_SIZE)
    task_counts = Counter(ex["task"] for ex in samples)
    diff_counts = Counter(ex["difficulty"] for ex in samples)
    print(f"Sampled {len(samples)}: tasks={dict(task_counts)}, diffs={dict(diff_counts)}", flush=True)

    # Save sample IDs for reproducibility
    sample_ids = [ex["q_id"] for ex in samples]
    with open(os.path.join(RESULTS_DIR, "sample_ids.json"), "w") as f:
        json.dump(sample_ids, f)

    all_results = {}

    for model_name, model_id in MODELS.items():
        for strategy_name in STRATEGIES:
            key = f"{model_name}__{strategy_name}"
            outfile = os.path.join(RESULTS_DIR, f"{key}.json")

            if os.path.exists(outfile):
                print(f"SKIP {key} (exists)", flush=True)
                with open(outfile) as f:
                    all_results[key] = json.load(f)
                continue

            print(f"\n{'='*60}", flush=True)
            print(f"RUNNING: {key}", flush=True)
            print(f"{'='*60}", flush=True)

            result = run_one(samples, strategy_name, model_name, model_id)

            with open(outfile, "w") as f:
                json.dump(result, f, indent=2)
            all_results[key] = result

            acc = result["accuracy"]
            print(f"  DONE [{key}]: {acc:.1f}%", flush=True)

    # Print summary
    print("\n" + "=" * 70)
    print(f"{'Model':<18} {'Strategy':<20} {'Acc (%)':>8} {'N':>5}")
    print("-" * 55)
    for key in sorted(all_results.keys()):
        res = all_results[key]
        print(f"{res['model']:<18} {res['strategy']:<20} {res['accuracy']:>7.1f}% {res['total']:>5}")

    # Save summary
    summary = {}
    for k, v in all_results.items():
        summary[k] = {kk: vv for kk, vv in v.items() if kk != "results"}
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/", flush=True)


if __name__ == "__main__":
    main()
