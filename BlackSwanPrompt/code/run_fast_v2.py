#!/usr/bin/env python3
"""
Fast concurrent experiment runner for BlackSwan Prompt Baselines - V2.
Fixed Claude refusal issue by reframing prompts as text-only MCQ reasoning.
"""

import json
import os
import asyncio
import aiohttp
import re
import time
import random
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

# ---- SYSTEM MESSAGE to prevent refusals ----
SYSTEM_MSG = (
    "You are a helpful reasoning assistant. You will be given multiple-choice questions "
    "about events described in text. The questions describe scenarios from videos, but you "
    "should reason purely from the TEXT descriptions provided. Do NOT refuse to answer. "
    "Do NOT say you cannot see a video. The text contains all information you need. "
    "Always pick the best answer from the options."
)

# ---- Prompt Templates ----

def make_direct_prompt(item):
    """P1: Direct MCQ (vanilla baseline)."""
    options_str = "\n".join(f"({i}) {opt}" for i, opt in enumerate(item["mcq_options"]))
    if item["task"] == "Detective":
        ctx = ("In a video, certain frames in the middle are hidden. "
               "Based on what is described happening before and after the hidden segment:")
    else:
        ctx = ("In a video showing an unexpected event, "
               "based on the complete sequence of events described:")
    return (
        f"{ctx}\n\n"
        f"Question: {item['default_question']}\n\n"
        f"Options:\n{options_str}\n\n"
        f"Reply with ONLY the option number (0, 1, or 2)."
    )

def make_cot_prompt(item):
    """P2: Chain-of-Thought."""
    options_str = "\n".join(f"({i}) {opt}" for i, opt in enumerate(item["mcq_options"]))
    if item["task"] == "Detective":
        ctx = ("In a video, certain key frames in the middle are hidden. "
               "You know what happens before and after, but not the event itself.")
    else:
        ctx = ("In a video showing a complete unexpected event, you observe the "
               "full sequence including a surprising outcome.")
    return (
        f"{ctx}\n\n"
        f"Question: {item['default_question']}\n\n"
        f"Options:\n{options_str}\n\n"
        f"Think step by step:\n"
        f"1. What do the described events before and after suggest?\n"
        f"2. Which option is most physically plausible?\n"
        f"3. Which option best explains something unexpected?\n\n"
        f"After your reasoning, state your final answer as: ANSWER: <number>"
    )

def make_abductive_prompt(item):
    """P3: Abductive/Defeasible Reasoning prompt."""
    options_str = "\n".join(f"({i}) {opt}" for i, opt in enumerate(item["mcq_options"]))
    if item["task"] == "Detective":
        return (
            "TASK: Abductive Reasoning\n"
            "A detective is analyzing a video where the key event has been hidden. "
            "The before and after segments are visible, but the crucial middle is missing. "
            "Your task: infer the most plausible hidden cause that explains the transition "
            "from the before-state to the after-state.\n\n"
            "Consider: What hidden event would make the after-state a natural consequence "
            "of the before-state?\n\n"
            f"Question: {item['default_question']}\n\n"
            f"Options:\n{options_str}\n\n"
            f"Which option is the best explanation? Reply: ANSWER: <number>"
        )
    else:
        return (
            "TASK: Defeasible Reasoning\n"
            "You are analyzing a complete video that contains an unexpected event. "
            "Your task: you may have formed an initial hypothesis about what would happen, "
            "but new evidence may contradict it. Revise your beliefs.\n\n"
            "Consider: What actually happened that was surprising or unexpected?\n\n"
            f"Question: {item['default_question']}\n\n"
            f"Options:\n{options_str}\n\n"
            f"Which option best describes the unexpected event? Reply: ANSWER: <number>"
        )

def make_elimination_prompt(item):
    """P4: Process-of-Elimination."""
    options_str = "\n".join(f"({i}) {opt}" for i, opt in enumerate(item["mcq_options"]))
    if item["task"] == "Detective":
        ctx = "analyzing a scenario where the key middle event is hidden (before and after are known)"
    else:
        ctx = "analyzing a complete scenario containing an unexpected event"
    return (
        f"You are {ctx}.\n\n"
        f"Question: {item['default_question']}\n\n"
        f"Options:\n{options_str}\n\n"
        f"Use process of elimination:\n"
        f"- Assess each option for physical plausibility.\n"
        f"- Eliminate contradictory or impossible options.\n"
        f"- Eliminate options that are too mundane (the event is supposed to be surprising).\n\n"
        f"After elimination, select the remaining best option.\n"
        f"Reply: ANSWER: <number>"
    )

def make_counterfactual_prompt(item):
    """P5: Counterfactual Contrastive."""
    options_str = "\n".join(f"({i}) {opt}" for i, opt in enumerate(item["mcq_options"]))
    if item["task"] == "Detective":
        return (
            "Scenario: A video with a hidden middle segment.\n\n"
            "STEP 1 - Normal expectation: Given what happens before, "
            "what would you NORMALLY expect?\n"
            "STEP 2 - Surprising outcome: The after-event shows something unexpected. "
            "What hidden event explains this?\n"
            "STEP 3 - Select: Which option describes an event that deviates from "
            "normal expectations and explains the surprising outcome?\n\n"
            f"Question: {item['default_question']}\n\n"
            f"Options:\n{options_str}\n\n"
            f"Reply: ANSWER: <number>"
        )
    else:
        return (
            "Scenario: A complete video of an unexpected event.\n\n"
            "STEP 1 - Initial belief: What would you initially predict happens?\n"
            "STEP 2 - Belief update: What actually happens that contradicts "
            "your initial belief?\n"
            "STEP 3 - Select the option that correctly identifies the surprising deviation.\n\n"
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
    # Pattern: ANSWER: X
    match = re.search(r'ANSWER:\s*(\d)', text)
    if match:
        return int(match.group(1))
    # Short response = just a number
    if len(text) <= 5:
        match = re.search(r'[012]', text)
        if match:
            return int(match.group(0))
    # Look for "(X)" pattern
    matches = re.findall(r'\(([012])\)', text)
    if matches:
        return int(matches[-1])
    # Last occurrence of 0, 1, or 2 as word boundary
    matches = re.findall(r'\b([012])\b', text)
    if matches:
        return int(matches[-1])
    return -1


SEM = asyncio.Semaphore(25)

async def call_api(session, model_id, prompt, retries=3):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }
    async with SEM:
        for attempt in range(retries):
            try:
                async with session.post(
                    OPENROUTER_URL, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"]
                    elif resp.status == 429:
                        await asyncio.sleep(2 ** (attempt + 1))
                    else:
                        text = await resp.text()
                        if "rate" in text.lower() or "limit" in text.lower():
                            await asyncio.sleep(3)
                        else:
                            await asyncio.sleep(1)
            except Exception as e:
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
    save_path = RESULTS_DIR / f"{model_name}_{template_name}_v2.jsonl"
    
    existing = {}
    if save_path.exists():
        with open(save_path, "r") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    existing[r["q_id"]] = r
                except:
                    pass
    
    todo = [item for item in data if item["q_id"] not in existing]
    
    if not todo:
        print(f"  {model_name} x {template_name}: already complete ({len(existing)} results)", flush=True)
        all_results = list(existing.values())
    else:
        print(f"  {model_name} x {template_name}: {len(existing)} done, {len(todo)} remaining", flush=True)
        
        async with aiohttp.ClientSession() as session:
            batch_size = 50
            new_results = []
            for batch_start in range(0, len(todo), batch_size):
                batch = todo[batch_start:batch_start + batch_size]
                tasks = [process_item(session, model_id, template_fn, item) for item in batch]
                batch_results = await asyncio.gather(*tasks)
                new_results.extend(batch_results)
                done_so_far = len(existing) + len(new_results)
                print(f"    Progress: {done_so_far}/{len(data)}", flush=True)
        
        with open(save_path, "a") as f:
            for r in new_results:
                r["prompt_template"] = template_name
                r["model"] = model_name
                f.write(json.dumps(r) + "\n")
        
        all_results = list(existing.values()) + new_results
    
    correct = sum(1 for r in all_results if r["predicted"] == r["ground_truth"])
    det = [r for r in all_results if r["task"] == "Detective"]
    rep = [r for r in all_results if r["task"] == "Reporter"]
    det_c = sum(1 for r in det if r["predicted"] == r["ground_truth"])
    rep_c = sum(1 for r in rep if r["predicted"] == r["ground_truth"])
    
    return {
        "model": model_name, "template": template_name,
        "overall_acc": correct / len(all_results) if all_results else 0,
        "detective_acc": det_c / len(det) if det else 0,
        "reporter_acc": rep_c / len(rep) if rep else 0,
        "total": len(all_results), "correct": correct,
        "det_total": len(det), "det_correct": det_c,
        "rep_total": len(rep), "rep_correct": rep_c,
        "parse_failures": sum(1 for r in all_results if r["predicted"] == -1),
    }


async def main():
    with open(VAL_PATH, "r") as f:
        data = [json.loads(line) for line in f]
    print(f"Loaded {len(data)} validation entries", flush=True)
    
    # Only run Claude and Gemini (GPT already done in v1)
    models_to_run = {
        "claude-3.5-haiku": MODELS["claude-3.5-haiku"],
        "gemini-2.0-flash": MODELS["gemini-2.0-flash"],
    }
    
    all_results = []
    for model_name, model_id in models_to_run.items():
        for template_name, template_fn in PROMPT_TEMPLATES.items():
            print(f"\nRunning {model_name} x {template_name}...", flush=True)
            result = await run_condition(model_name, model_id, template_name, template_fn, data)
            all_results.append(result)
            print(f"  => Overall: {result['overall_acc']:.3f} | "
                  f"Det: {result['detective_acc']:.3f} | "
                  f"Rep: {result['reporter_acc']:.3f} | "
                  f"Failures: {result['parse_failures']}", flush=True)
    
    summary_path = RESULTS_DIR / "summary_v2.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 90, flush=True)
    print(f"{'Model':<20} {'Template':<18} {'Overall':>8} {'Detective':>10} {'Reporter':>10} {'Fail':>6}", flush=True)
    print("-" * 90, flush=True)
    for r in all_results:
        print(f"{r['model']:<20} {r['template']:<18} {r['overall_acc']:>8.3f} "
              f"{r['detective_acc']:>10.3f} {r['reporter_acc']:>10.3f} {r['parse_failures']:>6}", flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
