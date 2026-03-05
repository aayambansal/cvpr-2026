#!/usr/bin/env python3
"""
BlackSwan Prompting Baselines: Abductive + Defeasible Video Reasoning
=====================================================================
Runs 5 prompting strategies x 3 LLMs on BlackSwanSuite-MCQ validation.

Prompting strategies:
  1. Naive        – direct question, no reasoning scaffold
  2. Chain-of-Thought (CoT) – "think step by step"
  3. Abductive CoT  – explicit abductive reasoning scaffold
  4. Hypothesis-Eliminate – generate + eliminate hypotheses
  5. Counterfactual  – consider alternative worlds then choose

Models:
  - GPT-4o-mini (gpt-4o-mini via OpenRouter)
  - Claude 3.5 Haiku (anthropic/claude-3.5-haiku via OpenRouter)
  - Gemini 2.0 Flash (google/gemini-2.0-flash-001 via OpenRouter)
"""

import os
import json
import time
import random
import sys
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


# ── Prompt Templates ─────────────────────────────────────────────────

def build_naive_prompt(example):
    """Strategy 1: Naive – direct question with options."""
    task = example["task"]
    q = example["default_question"]
    opts = example["mcq_options"]

    if task == "Detective":
        context = (
            "You are watching a video. You can see the beginning (pre-event) "
            "and the end (post-event), but the middle frames are hidden."
        )
    else:  # Reporter
        context = (
            "You are watching a complete video showing an unexpected event."
        )

    options_str = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(opts))
    return (
        f"{context}\n\n"
        f"{q}\n\n"
        f"{options_str}\n\n"
        f"Answer with ONLY the letter (A, B, or C)."
    )


def build_cot_prompt(example):
    """Strategy 2: Chain-of-Thought – generic step-by-step."""
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
        f"{context}\n\n"
        f"{q}\n\n"
        f"{options_str}\n\n"
        f"Think step by step. Consider what is described in each option "
        f"and how it relates to the video context.\n"
        f"Then provide your final answer as a single letter (A, B, or C).\n\n"
        f"Reasoning:"
    )


def build_abductive_cot_prompt(example):
    """Strategy 3: Abductive CoT – explicit abductive reasoning scaffold."""
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
    else:  # Reporter
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
        f"{context}\n\n"
        f"{q}\n\n"
        f"{options_str}\n\n"
        f"{scaffold}\n\n"
        f"Provide your reasoning, then your final answer as a single letter (A, B, or C).\n\n"
        f"Reasoning:"
    )


def build_hypothesis_eliminate_prompt(example):
    """Strategy 4: Hypothesis-Eliminate – generate & eliminate."""
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
        f"{context}\n\n"
        f"{q}\n\n"
        f"{options_str}\n\n"
        f"Use the following elimination strategy:\n"
        f"For EACH option:\n"
        f"  - State one reason it COULD be correct.\n"
        f"  - State one reason it COULD be wrong.\n"
        f"  - Rate plausibility: HIGH / MEDIUM / LOW.\n\n"
        f"Then eliminate the LOW-plausibility options and choose the best.\n\n"
        f"Analysis:\n"
    )


def build_counterfactual_prompt(example):
    """Strategy 5: Counterfactual – consider alternative worlds."""
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
            "setup to aftermath?\n"
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
            "Which option represents the actual unexpected outcome?\n"
        )

    options_str = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(opts))
    return (
        f"{context}\n\n"
        f"{q}\n\n"
        f"{options_str}\n\n"
        f"{cf_scaffold}\n\n"
        f"Provide your reasoning, then answer with a single letter (A, B, or C).\n\n"
        f"Reasoning:"
    )


STRATEGIES = {
    "naive": build_naive_prompt,
    "cot": build_cot_prompt,
    "abductive_cot": build_abductive_cot_prompt,
    "hypothesis_eliminate": build_hypothesis_eliminate_prompt,
    "counterfactual": build_counterfactual_prompt,
}


# ── LLM Calling ──────────────────────────────────────────────────────

def call_openrouter(model_id, prompt, max_retries=3):
    """Call OpenRouter API with retries."""
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
                print(f"  Retry {attempt+1} for {model_id}: {e}", flush=True)
                time.sleep(wait)
            else:
                print(f"  FAILED after {max_retries} attempts: {e}", flush=True)
                return None


def extract_answer(response):
    """Extract A/B/C from LLM response."""
    if response is None:
        return None

    # Look for standalone letter at the end or after "answer:"
    response = response.strip()

    # Try last line
    last_line = response.strip().split("\n")[-1].strip()

    # Pattern: "Answer: X" or "Final answer: X" or just "X"
    import re
    # Look for explicit "answer" patterns
    match = re.search(r'(?:final\s+)?answer[:\s]*\(?([A-C])\)?', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Look for standalone letter at end
    match = re.search(r'\b([A-C])\b\s*\.?\s*$', last_line)
    if match:
        return match.group(1).upper()

    # Look for (A), (B), (C) pattern
    match = re.search(r'\(([A-C])\)', last_line)
    if match:
        return match.group(1).upper()

    # First single letter A-C in the response
    match = re.search(r'\b([A-C])\b', response)
    if match:
        return match.group(1).upper()

    return None


def letter_to_idx(letter):
    """Convert A/B/C to 0/1/2."""
    if letter is None:
        return None
    return ord(letter) - ord('A')


# ── Main Experiment Loop ─────────────────────────────────────────────

def run_experiment(dataset, strategy_name, model_name, model_id, sample_size=None):
    """Run one experiment (strategy x model) on the dataset."""

    prompt_fn = STRATEGIES[strategy_name]
    results = []

    subset = list(dataset)
    if sample_size and sample_size < len(subset):
        random.seed(42)
        subset = random.sample(subset, sample_size)

    correct = 0
    total = 0
    errors_by_type = defaultdict(int)

    for i, ex in enumerate(subset):
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
            "response": response[:500] if response else None,
        }
        results.append(result)

        if (i + 1) % 50 == 0:
            acc = correct / total * 100
            print(f"  [{model_name}|{strategy_name}] {i+1}/{len(subset)} — acc={acc:.1f}%", flush=True)

        # Small delay to avoid rate limiting
        time.sleep(0.15)

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"  FINAL [{model_name}|{strategy_name}]: {accuracy:.1f}% ({correct}/{total})", flush=True)

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

    # Use full dataset for all experiments
    sample_size = len(ds)  # Full validation set

    all_results = {}

    for model_name, model_id in MODELS.items():
        for strategy_name in STRATEGIES:
            key = f"{model_name}__{strategy_name}"
            outfile = os.path.join(RESULTS_DIR, f"{key}.json")

            # Skip if already done
            if os.path.exists(outfile):
                print(f"SKIP {key} (already exists)", flush=True)
                with open(outfile) as f:
                    all_results[key] = json.load(f)
                continue

            print(f"\nRUNNING: {key}", flush=True)
            result = run_experiment(ds, strategy_name, model_name, model_id, sample_size)

            with open(outfile, "w") as f:
                json.dump(result, f, indent=2)
            all_results[key] = result

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'Strategy':<25} {'Accuracy':>10}")
    print("-" * 55)
    for key, res in sorted(all_results.items()):
        print(f"{res['model']:<20} {res['strategy']:<25} {res['accuracy']:>8.1f}%")

    # Save summary
    summary = {k: {kk: vv for kk, vv in v.items() if kk != "results"} for k, v in all_results.items()}
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll results saved to {RESULTS_DIR}/", flush=True)


if __name__ == "__main__":
    main()
