#!/usr/bin/env python3
"""
BlackSwan Prompt Baselines - Final Version.
Text-only MCQ evaluation: models reason from question text + option descriptions.
Key insight: without video, models rely on linguistic/commonsense priors in option text.
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

SYSTEM_MSG = (
    "You are an expert at multiple-choice reasoning. You will be given a question about "
    "a video scenario along with 3 answer options. You do NOT have access to the video -- "
    "you must reason ONLY from the text of the question and the answer options provided. "
    "Use commonsense reasoning, physical plausibility, and linguistic cues to pick the best answer. "
    "Always provide an answer -- never refuse. If uncertain, make your best guess."
)


# ---- Prompt Templates ----

def make_direct_prompt(item):
    """P1: Direct - minimal instruction, just answer."""
    options_str = "\n".join(f"  ({i}) {opt}" for i, opt in enumerate(item["mcq_options"]))
    task_note = ("[Context: The video has hidden middle frames -- you must infer the hidden event.]"
                 if item["task"] == "Detective"
                 else "[Context: The video shows a complete unexpected event.]")
    return (
        f"{task_note}\n\n"
        f"{item['default_question']}\n\n"
        f"Options:\n{options_str}\n\n"
        f"Answer with just the number (0, 1, or 2):"
    )


def make_cot_prompt(item):
    """P2: Chain-of-Thought - step-by-step reasoning."""
    options_str = "\n".join(f"  ({i}) {opt}" for i, opt in enumerate(item["mcq_options"]))
    task_note = ("[Context: Hidden middle frames -- infer the hidden event from text clues.]"
                 if item["task"] == "Detective"
                 else "[Context: Complete video with an unexpected event.]")
    return (
        f"{task_note}\n\n"
        f"{item['default_question']}\n\n"
        f"Options:\n{options_str}\n\n"
        f"Let's think step by step:\n"
        f"1. What scenario does each option describe?\n"
        f"2. Which is most physically plausible as an unexpected event?\n"
        f"3. Which has the most specific/concrete details (suggesting it's the real event)?\n\n"
        f"ANSWER: "
    )


def make_abductive_prompt(item):
    """P3: Abductive Reasoning frame."""
    options_str = "\n".join(f"  ({i}) {opt}" for i, opt in enumerate(item["mcq_options"]))
    if item["task"] == "Detective":
        frame = (
            "ABDUCTIVE REASONING TASK: A key event is hidden between a before-state and after-state. "
            "From the answer options below, select the hidden event that best explains the causal "
            "transition. The correct answer typically describes something unexpected -- a failure, "
            "accident, or surprising outcome that connects the before and after states.\n"
        )
    else:
        frame = (
            "DEFEASIBLE REASONING TASK: An unexpected event occurs in a video. "
            "From the answer options below, identify which correctly describes the surprising event. "
            "The correct answer typically involves a belief revision -- something that contradicts "
            "what you'd normally expect.\n"
        )
    return (
        f"{frame}\n"
        f"{item['default_question']}\n\n"
        f"Options:\n{options_str}\n\n"
        f"ANSWER: "
    )


def make_elimination_prompt(item):
    """P4: Process of Elimination."""
    options_str = "\n".join(f"  ({i}) {opt}" for i, opt in enumerate(item["mcq_options"]))
    task_note = ("hidden middle event" if item["task"] == "Detective" else "unexpected event")
    return (
        f"[Task: Identify the {task_note} from the options below.]\n\n"
        f"{item['default_question']}\n\n"
        f"Options:\n{options_str}\n\n"
        f"Process of elimination:\n"
        f"- Eliminate any option that describes a normal/expected outcome (the event is surprising).\n"
        f"- Eliminate any option that is physically implausible or self-contradictory.\n"
        f"- The correct answer usually involves something going wrong or an unexpected twist.\n\n"
        f"ANSWER: "
    )


def make_counterfactual_prompt(item):
    """P5: Counterfactual Contrastive reasoning."""
    options_str = "\n".join(f"  ({i}) {opt}" for i, opt in enumerate(item["mcq_options"]))
    if item["task"] == "Detective":
        return (
            "COUNTERFACTUAL REASONING:\n"
            "A video has hidden middle frames between a before-state and after-state.\n\n"
            "Step 1: For each option, ask: 'If this happened, would it lead to a surprising outcome?'\n"
            "Step 2: The correct answer is the event that DEVIATES from what you'd normally expect.\n"
            "Step 3: Look for the option describing a failure, accident, or unexpected twist.\n\n"
            f"{item['default_question']}\n\n"
            f"Options:\n{options_str}\n\n"
            f"ANSWER: "
        )
    else:
        return (
            "COUNTERFACTUAL REASONING:\n"
            "A complete video shows an unexpected event.\n\n"
            "Step 1: What would you normally expect to happen in this scenario?\n"
            "Step 2: Which option describes something that CONTRADICTS that expectation?\n"
            "Step 3: The correct answer is the surprising deviation from normalcy.\n\n"
            f"{item['default_question']}\n\n"
            f"Options:\n{options_str}\n\n"
            f"ANSWER: "
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
    # ANSWER: X
    match = re.search(r'ANSWER:\s*[\(\[]?(\d)[\)\]]?', text, re.IGNORECASE)
    if match:
        val = int(match.group(1))
        if val in (0, 1, 2):
            return val
    # Short response
    if len(text) <= 5:
        match = re.search(r'[012]', text)
        if match:
            return int(match.group(0))
    # "Option X" or "(X)" pattern
    match = re.search(r'(?:option|choice)\s*[\(\[]?(\d)[\)\]]?', text, re.IGNORECASE)
    if match:
        val = int(match.group(1))
        if val in (0, 1, 2):
            return val
    # Last standalone digit 0-2
    matches = re.findall(r'\b([012])\b', text)
    if matches:
        return int(matches[-1])
    # Any digit 0-2
    matches = re.findall(r'[012]', text)
    if matches:
        return int(matches[-1])
    return -1


SEM = asyncio.Semaphore(25)

async def call_api(session, model_id, prompt, retries=4):
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
        "max_tokens": 600,
    }
    async with SEM:
        for attempt in range(retries):
            try:
                async with session.post(
                    OPENROUTER_URL, headers=headers, json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"]
                    elif resp.status == 429:
                        await asyncio.sleep(2 ** (attempt + 1) + random.random())
                    else:
                        await asyncio.sleep(2)
            except:
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
        "response": response[:600] if response else None,
    }


async def run_condition(model_name, model_id, template_name, template_fn, data):
    save_path = RESULTS_DIR / f"{model_name}_{template_name}_final.jsonl"
    
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
        print(f"  {model_name} x {template_name}: complete ({len(existing)})", flush=True)
        all_results = list(existing.values())
    else:
        print(f"  {model_name} x {template_name}: {len(existing)} done, {len(todo)} todo", flush=True)
        async with aiohttp.ClientSession() as session:
            new_results = []
            for i in range(0, len(todo), 50):
                batch = todo[i:i+50]
                results = await asyncio.gather(*[
                    process_item(session, model_id, template_fn, item) for item in batch
                ])
                new_results.extend(results)
                print(f"    {len(existing)+len(new_results)}/{len(data)}", flush=True)
        
        with open(save_path, "a") as f:
            for r in new_results:
                r["prompt_template"] = template_name
                r["model"] = model_name
                f.write(json.dumps(r) + "\n")
        all_results = list(existing.values()) + new_results
    
    c = sum(1 for r in all_results if r["predicted"] == r["ground_truth"])
    det = [r for r in all_results if r["task"] == "Detective"]
    rep = [r for r in all_results if r["task"] == "Reporter"]
    dc = sum(1 for r in det if r["predicted"] == r["ground_truth"])
    rc = sum(1 for r in rep if r["predicted"] == r["ground_truth"])
    pf = sum(1 for r in all_results if r["predicted"] == -1)
    
    # By difficulty
    diff_acc = {}
    for diff in ["easy", "medium", "hard"]:
        d_items = [r for r in all_results if r["difficulty"] == diff]
        if d_items:
            diff_acc[diff] = sum(1 for r in d_items if r["predicted"] == r["ground_truth"]) / len(d_items)
    
    return {
        "model": model_name, "template": template_name,
        "overall_acc": c / len(all_results) if all_results else 0,
        "detective_acc": dc / len(det) if det else 0,
        "reporter_acc": rc / len(rep) if rep else 0,
        "total": len(all_results), "correct": c,
        "det_total": len(det), "det_correct": dc,
        "rep_total": len(rep), "rep_correct": rc,
        "parse_failures": pf,
        "diff_acc": diff_acc,
    }


async def main():
    with open(VAL_PATH, "r") as f:
        data = [json.loads(line) for line in f]
    print(f"Loaded {len(data)} entries", flush=True)
    
    all_results = []
    for model_name, model_id in MODELS.items():
        for template_name, template_fn in PROMPT_TEMPLATES.items():
            print(f"\n{model_name} x {template_name}:", flush=True)
            result = await run_condition(model_name, model_id, template_name, template_fn, data)
            all_results.append(result)
            print(f"  Overall={result['overall_acc']:.3f} Det={result['detective_acc']:.3f} "
                  f"Rep={result['reporter_acc']:.3f} Fail={result['parse_failures']}", flush=True)
    
    with open(RESULTS_DIR / "summary_final.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 95)
    print(f"{'Model':<20} {'Template':<18} {'Overall':>8} {'Det':>8} {'Rep':>8} {'Easy':>7} {'Med':>7} {'Hard':>7} {'Fail':>5}")
    print("-" * 95)
    for r in all_results:
        da = r.get("diff_acc", {})
        print(f"{r['model']:<20} {r['template']:<18} {r['overall_acc']:>8.3f} "
              f"{r['detective_acc']:>8.3f} {r['reporter_acc']:>8.3f} "
              f"{da.get('easy',0):>7.3f} {da.get('medium',0):>7.3f} {da.get('hard',0):>7.3f} "
              f"{r['parse_failures']:>5}")
    print("=" * 95)


if __name__ == "__main__":
    asyncio.run(main())
