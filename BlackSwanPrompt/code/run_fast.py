#!/usr/bin/env python3
"""
Fast concurrent experiment runner for BlackSwan Prompt Baselines.
Uses asyncio + aiohttp for parallel API calls.
"""

import json
import os
import asyncio
import aiohttp
import re
import time
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

VAL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--UBC-ViL--BlackSwanSuite-MCQ/"
    "snapshots/2e78b5d715fb8ce2c3c3e365c1c2c1be4ed12fc0/BlackSwanSuite_MCQ_Val.jsonl"
)
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = {
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "claude-3.5-haiku": "anthropic/claude-3.5-haiku",
    "gemini-2.0-flash": "google/gemini-2.0-flash-001",
}

# ---- Prompt Templates (same as before) ----

def make_direct_prompt(item):
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


def parse_answer(response_text):
    if response_text is None:
        return -1
    text = response_text.strip()
    match = re.search(r'ANSWER:\s*(\d)', text)
    if match:
        return int(match.group(1))
    if len(text) <= 3:
        match = re.search(r'[012]', text)
        if match:
            return int(match.group(0))
    matches = re.findall(r'\b([012])\b', text)
    if matches:
        return int(matches[-1])
    return -1


# Semaphore to limit concurrent requests
SEM = asyncio.Semaphore(20)


async def call_api(session, model_id, prompt, retries=3):
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
    async with SEM:
        for attempt in range(retries):
            try:
                async with session.post(
                    OPENROUTER_URL, headers=headers, json=payload, timeout=60
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"]
                    elif resp.status == 429:
                        await asyncio.sleep(2 ** (attempt + 1))
                    else:
                        text = await resp.text()
                        print(f"  API {resp.status}: {text[:100]}", flush=True)
                        await asyncio.sleep(2)
            except Exception as e:
                print(f"  Error: {e}", flush=True)
                await asyncio.sleep(2)
    return None


async def process_item(session, model_id, template_fn, item):
    prompt = template_fn(item)
    response = await call_api(session, model_id, prompt)
    pred = parse_answer(response)
    return {
        "q_id": item["q_id"],
        "task": item["task"],
        "difficulty": item["difficulty"],
        "ground_truth": item["mcq_gt_option"],
        "predicted": pred,
        "response": response[:500] if response else None,
    }


async def run_condition(model_name, model_id, template_name, template_fn, data):
    save_path = RESULTS_DIR / f"{model_name}_{template_name}.jsonl"
    
    # Load existing results
    existing = {}
    if save_path.exists():
        with open(save_path, "r") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    existing[r["q_id"]] = r
                except:
                    pass
    
    # Filter to items not yet processed
    todo = [item for item in data if item["q_id"] not in existing]
    
    if not todo:
        print(f"  {model_name} x {template_name}: already complete ({len(existing)} results)")
        # Load all results
        all_results = list(existing.values())
    else:
        print(f"  {model_name} x {template_name}: {len(existing)} done, {len(todo)} remaining")
        
        async with aiohttp.ClientSession() as session:
            # Process in batches of 50
            batch_size = 50
            new_results = []
            for batch_start in range(0, len(todo), batch_size):
                batch = todo[batch_start:batch_start + batch_size]
                tasks = [
                    process_item(session, model_id, template_fn, item)
                    for item in batch
                ]
                batch_results = await asyncio.gather(*tasks)
                new_results.extend(batch_results)
                
                done_so_far = len(existing) + len(new_results)
                print(f"    Progress: {done_so_far}/{len(data)}", flush=True)
        
        # Save new results
        with open(save_path, "a") as f:
            for r in new_results:
                r["prompt_template"] = template_name
                r["model"] = model_name
                f.write(json.dumps(r) + "\n")
        
        all_results = list(existing.values()) + new_results
    
    # Compute metrics
    correct = sum(1 for r in all_results if r["predicted"] == r["ground_truth"])
    det_results = [r for r in all_results if r["task"] == "Detective"]
    rep_results = [r for r in all_results if r["task"] == "Reporter"]
    det_correct = sum(1 for r in det_results if r["predicted"] == r["ground_truth"])
    rep_correct = sum(1 for r in rep_results if r["predicted"] == r["ground_truth"])
    
    return {
        "model": model_name,
        "template": template_name,
        "overall_acc": correct / len(all_results) if all_results else 0,
        "detective_acc": det_correct / len(det_results) if det_results else 0,
        "reporter_acc": rep_correct / len(rep_results) if rep_results else 0,
        "total": len(all_results),
        "correct": correct,
        "det_total": len(det_results),
        "det_correct": det_correct,
        "rep_total": len(rep_results),
        "rep_correct": rep_correct,
    }


async def main():
    with open(VAL_PATH, "r") as f:
        data = [json.loads(line) for line in f]
    print(f"Loaded {len(data)} validation entries")
    
    # Random baseline
    random.seed(42)
    random_correct = sum(1 for d in data if random.randint(0, 2) == d["mcq_gt_option"])
    print(f"Random baseline: {random_correct/len(data):.3f}\n")
    
    all_results = []
    
    for model_name, model_id in MODELS.items():
        for template_name, template_fn in PROMPT_TEMPLATES.items():
            print(f"\nRunning {model_name} x {template_name}...")
            result = await run_condition(model_name, model_id, template_name, template_fn, data)
            all_results.append(result)
            print(f"  => Overall: {result['overall_acc']:.3f} | "
                  f"Det: {result['detective_acc']:.3f} | "
                  f"Rep: {result['reporter_acc']:.3f}")
    
    # Save summary
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"{'Model':<20} {'Template':<18} {'Overall':>8} {'Detective':>10} {'Reporter':>10}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['model']:<20} {r['template']:<18} {r['overall_acc']:>8.3f} "
              f"{r['detective_acc']:>10.3f} {r['reporter_acc']:>10.3f}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
