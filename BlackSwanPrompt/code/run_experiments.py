#!/usr/bin/env python3
"""
BlackSwan Prompt Baselines: Systematic evaluation of prompting strategies
for abductive and defeasible video reasoning on BlackSwanSuite-MCQ.

This script runs 5 prompt templates x 3 LLMs on the full validation set (1793 MCQs).
"""

import json
import os
import time
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
import requests

# ---- Configuration ----
VAL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--UBC-ViL--BlackSwanSuite-MCQ/"
    "snapshots/2e78b5d715fb8ce2c3c3e365c1c2c1be4ed12fc0/BlackSwanSuite_MCQ_Val.jsonl"
)
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Models to test
MODELS = {
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "claude-3.5-haiku": "anthropic/claude-3.5-haiku",
    "gemini-2.0-flash": "google/gemini-2.0-flash-001",
}

# ---- Prompt Templates ----

def make_direct_prompt(item):
    """P1: Direct MCQ (vanilla baseline) - no special reasoning guidance."""
    options_str = "\n".join(f"({i}) {opt}" for i, opt in enumerate(item["mcq_options"]))
    task_ctx = (
        "You are watching a video. Some frames in the middle are hidden."
        if item["task"] == "Detective"
        else "You are watching a complete video of an unexpected event."
    )
    return (
        f"{task_ctx}\n\n"
        f"Question: {item['default_question']}\n\n"
        f"Options:\n{options_str}\n\n"
        f"Reply with ONLY the option number (0, 1, or 2)."
    )


def make_cot_prompt(item):
    """P2: Chain-of-Thought - explicit step-by-step reasoning before answering."""
    options_str = "\n".join(f"({i}) {opt}" for i, opt in enumerate(item["mcq_options"]))
    task_ctx = (
        "You are watching a video. Some frames in the middle are hidden. "
        "You can see what happens before and after, but not the event itself."
        if item["task"] == "Detective"
        else "You are watching a complete video that contains an unexpected or surprising event."
    )
    return (
        f"{task_ctx}\n\n"
        f"Question: {item['default_question']}\n\n"
        f"Options:\n{options_str}\n\n"
        f"Think step by step:\n"
        f"1. What do the descriptions of events before and after suggest?\n"
        f"2. Which option is most physically plausible?\n"
        f"3. Which option best explains something unexpected?\n\n"
        f"After your reasoning, state your final answer as: ANSWER: <number>"
    )


def make_abductive_prompt(item):
    """P3: Abductive Reasoning - explicitly frames task as inference-to-best-explanation."""
    options_str = "\n".join(f"({i}) {opt}" for i, opt in enumerate(item["mcq_options"]))
    if item["task"] == "Detective":
        return (
            "You are a detective analyzing a video where the key event has been hidden. "
            "You can see what happens before the event and after the event, but the "
            "crucial middle segment is missing. Your task is ABDUCTIVE REASONING: "
            "infer the most plausible hidden cause that explains the transition from "
            "the before-state to the after-state.\n\n"
            "Consider: What hidden event would make the after-state a natural consequence "
            "of the before-state? Look for the option that provides the best causal "
            "explanation for the observed outcome.\n\n"
            f"Question: {item['default_question']}\n\n"
            f"Options:\n{options_str}\n\n"
            f"Which option is the best explanation? Reply: ANSWER: <number>"
        )
    else:
        return (
            "You are analyzing a complete video that contains an unexpected event. "
            "Your task is DEFEASIBLE REASONING: you may have formed an initial "
            "hypothesis about what would happen, but new visual evidence may "
            "contradict it. Be willing to revise your beliefs based on what "
            "actually occurs in the video.\n\n"
            "Consider: What actually happened that was surprising or unexpected? "
            "Which option correctly captures the surprising deviation from "
            "normal expectations?\n\n"
            f"Question: {item['default_question']}\n\n"
            f"Options:\n{options_str}\n\n"
            f"Which option best describes the unexpected event? Reply: ANSWER: <number>"
        )


def make_elimination_prompt(item):
    """P4: Process-of-Elimination - explicitly eliminate implausible options first."""
    options_str = "\n".join(f"({i}) {opt}" for i, opt in enumerate(item["mcq_options"]))
    task_ctx = (
        "a video where the key middle event is hidden (you see before and after)"
        if item["task"] == "Detective"
        else "a complete video containing an unexpected event"
    )
    return (
        f"You are analyzing {task_ctx}.\n\n"
        f"Question: {item['default_question']}\n\n"
        f"Options:\n{options_str}\n\n"
        f"Use process of elimination:\n"
        f"- For each option, assess whether it is physically plausible given the context.\n"
        f"- Eliminate options that are contradictory, physically impossible, or "
        f"do not explain the transition between what comes before and after.\n"
        f"- If an option describes something too expected/mundane, it likely isn't "
        f"the surprising event.\n\n"
        f"After eliminating implausible options, select the remaining one.\n"
        f"Reply: ANSWER: <number>"
    )


def make_counterfactual_prompt(item):
    """P5: Counterfactual Contrastive - compare 'what would normally happen' vs 'what did happen'."""
    options_str = "\n".join(f"({i}) {opt}" for i, opt in enumerate(item["mcq_options"]))
    if item["task"] == "Detective":
        return (
            "You are investigating a video with a hidden middle segment.\n\n"
            "STEP 1 - Normal expectation: Given what happens before the hidden event, "
            "what would you NORMALLY expect to happen?\n"
            "STEP 2 - Surprising outcome: The after-event shows something unexpected "
            "occurred. What hidden event explains this surprising outcome?\n"
            "STEP 3 - Contrastive selection: Which option describes an event that "
            "deviates from the normal expectation and explains the surprising outcome?\n\n"
            f"Question: {item['default_question']}\n\n"
            f"Options:\n{options_str}\n\n"
            f"Reply: ANSWER: <number>"
        )
    else:
        return (
            "You are watching a complete video of an unexpected event.\n\n"
            "STEP 1 - Initial belief: Based on how the video begins, what would "
            "you initially predict will happen?\n"
            "STEP 2 - Belief update: As you watch the full video, what actually "
            "happens that contradicts or revises your initial belief?\n"
            "STEP 3 - Select the option that correctly identifies the surprising "
            "deviation from your initial expectation.\n\n"
            f"Question: {item['default_question']}\n\n"
            f"Options:\n{options_str}\n\n"
            f"Reply: ANSWER: <number>"
        )


PROMPT_TEMPLATES = {
    "P1_Direct": make_direct_prompt,
    "P2_CoT": make_cot_prompt,
    "P3_Abductive": make_abductive_prompt,
    "P4_Elimination": make_elimination_prompt,
    "P5_Counterfactual": make_counterfactual_prompt,
}


# ---- API Call ----

def call_openrouter(model_id, prompt, max_retries=3):
    """Call OpenRouter API with retry logic."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 512,
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            elif resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  API error {resp.status_code}: {resp.text[:200]}")
                time.sleep(2)
        except Exception as e:
            print(f"  Exception: {e}")
            time.sleep(2)
    return None


def parse_answer(response_text):
    """Extract answer number from model response."""
    if response_text is None:
        return -1
    text = response_text.strip()
    
    # Try to find "ANSWER: X" pattern
    import re
    match = re.search(r'ANSWER:\s*(\d)', text)
    if match:
        return int(match.group(1))
    
    # Try to find just a single digit at the end
    match = re.search(r'\b([012])\b', text)
    if match:
        # If the text is very short (just the number), use it
        if len(text) <= 3:
            return int(match.group(1))
    
    # Try last digit
    matches = re.findall(r'[012]', text)
    if matches:
        return int(matches[-1])
    
    return -1


# ---- Main Experiment ----

def run_experiment(model_name, model_id, template_name, template_fn, data):
    """Run a single (model, template) experiment on all data."""
    results = []
    correct = 0
    total = 0
    det_correct = 0
    det_total = 0
    rep_correct = 0
    rep_total = 0

    save_path = RESULTS_DIR / f"{model_name}_{template_name}.jsonl"
    
    # Check for existing results to resume
    existing = {}
    if save_path.exists():
        with open(save_path, "r") as f:
            for line in f:
                r = json.loads(line)
                existing[r["q_id"]] = r
        print(f"  Resuming: found {len(existing)} existing results")

    with open(save_path, "a") as fout:
        for idx, item in enumerate(data):
            q_id = item["q_id"]
            
            # Skip if already processed
            if q_id in existing:
                r = existing[q_id]
                pred = r["predicted"]
                gt = r["ground_truth"]
            else:
                prompt = template_fn(item)
                response = call_openrouter(model_id, prompt)
                pred = parse_answer(response)
                gt = item["mcq_gt_option"]
                
                result = {
                    "q_id": q_id,
                    "task": item["task"],
                    "difficulty": item["difficulty"],
                    "ground_truth": gt,
                    "predicted": pred,
                    "response": response[:500] if response else None,
                    "prompt_template": template_name,
                    "model": model_name,
                }
                fout.write(json.dumps(result) + "\n")
                fout.flush()
                
                # Rate limiting
                if idx % 50 == 0 and idx > 0:
                    time.sleep(1)

            total += 1
            if pred == gt:
                correct += 1
            
            if item["task"] == "Detective":
                det_total += 1
                if pred == gt:
                    det_correct += 1
            else:
                rep_total += 1
                if pred == gt:
                    rep_correct += 1

            if total % 100 == 0:
                print(f"  Progress: {total}/{len(data)} | Acc: {correct/total:.3f}")

    overall_acc = correct / total if total > 0 else 0
    det_acc = det_correct / det_total if det_total > 0 else 0
    rep_acc = rep_correct / rep_total if rep_total > 0 else 0

    return {
        "model": model_name,
        "template": template_name,
        "overall_acc": overall_acc,
        "detective_acc": det_acc,
        "reporter_acc": rep_acc,
        "total": total,
        "correct": correct,
        "det_total": det_total,
        "det_correct": det_correct,
        "rep_total": rep_total,
        "rep_correct": rep_correct,
    }


def main():
    # Load data
    with open(VAL_PATH, "r") as f:
        data = [json.loads(line) for line in f]
    print(f"Loaded {len(data)} validation entries")
    print(f"  Detective: {sum(1 for d in data if d['task']=='Detective')}")
    print(f"  Reporter: {sum(1 for d in data if d['task']=='Reporter')}")
    print()

    # Also add random baseline
    random.seed(42)
    random_correct = sum(1 for d in data if random.randint(0, 2) == d["mcq_gt_option"])
    print(f"Random baseline: {random_correct/len(data):.3f}")
    print()

    all_results = []

    # Run all combinations
    for model_name, model_id in MODELS.items():
        for template_name, template_fn in PROMPT_TEMPLATES.items():
            print(f"Running {model_name} x {template_name}...")
            result = run_experiment(model_name, model_id, template_name, template_fn, data)
            all_results.append(result)
            print(f"  Overall: {result['overall_acc']:.3f} | "
                  f"Detective: {result['detective_acc']:.3f} | "
                  f"Reporter: {result['reporter_acc']:.3f}")
            print()

    # Save summary
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    # Print table
    print("\n" + "=" * 80)
    print(f"{'Model':<20} {'Template':<18} {'Overall':>8} {'Detective':>10} {'Reporter':>10}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['model']:<20} {r['template']:<18} {r['overall_acc']:>8.3f} "
              f"{r['detective_acc']:>10.3f} {r['reporter_acc']:>10.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
