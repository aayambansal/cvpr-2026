#!/usr/bin/env python3
"""
Run ALL remaining experiments with per-example checkpointing.
Processes sequentially but checkpoints every example so it can resume.
"""
import os
import json
import time
import random
import re
from collections import defaultdict
from datasets import load_dataset
import requests

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY")
assert OPENROUTER_KEY

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
        groups[(ex["task"], ex["difficulty"])].append(ex)
    total = sum(len(v) for v in groups.values())
    sampled = []
    for key, items in groups.items():
        k = max(1, round(n * len(items) / total))
        sampled.extend(random.sample(items, min(k, len(items))))
    random.shuffle(sampled)
    return sampled[:n]


def make_context(task):
    if task == "Detective":
        return ("SCENARIO: A video has three segments: pre-event, hidden middle, post-event. "
                "You cannot see the middle. Determine what most likely happened.")
    return ("SCENARIO: A video shows an unexpected event. "
            "Determine which description best captures what happens.")


def build_naive(ex):
    ctx = make_context(ex["task"])
    opts = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(ex["mcq_options"]))
    return f"{ctx}\n\n{ex['default_question']}\n\n{opts}\n\nAnswer with ONLY the letter (A, B, or C)."


def build_cot(ex):
    ctx = make_context(ex["task"])
    opts = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(ex["mcq_options"]))
    return (f"{ctx}\n\n{ex['default_question']}\n\n{opts}\n\n"
            "Think step by step, then answer (A, B, or C).\n\nReasoning:")


def build_abductive(ex):
    ctx = make_context(ex["task"])
    opts = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(ex["mcq_options"]))
    if ex["task"] == "Detective":
        s = ("ABDUCTIVE REASONING:\n1. CLUES from options\n"
             "2. CAUSAL CHAIN for each\n3. BEST EXPLANATION")
    else:
        s = ("DEFEASIBLE REASONING:\n1. EXPECTATION\n"
             "2. ANOMALY\n3. BELIEF REVISION")
    return f"{ctx}\n\n{ex['default_question']}\n\n{opts}\n\n{s}\n\nFinal answer (A, B, or C):\n\nReasoning:"


def build_hyp_elim(ex):
    ctx = make_context(ex["task"])
    opts = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(ex["mcq_options"]))
    return (f"{ctx}\n\n{ex['default_question']}\n\n{opts}\n\n"
            "For EACH option: one reason FOR, one AGAINST, rate HIGH/MED/LOW.\n"
            "Eliminate LOW. Choose best.\nFinal answer (A, B, or C):\n\nAnalysis:")


def build_counterfactual(ex):
    ctx = make_context(ex["task"])
    opts = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(ex["mcq_options"]))
    if ex["task"] == "Detective":
        cf = "For each option, imagine that event occurred. Would the aftermath make sense?"
    else:
        cf = "Consider what's expected normally. Which option represents the unexpected outcome?"
    return f"{ctx}\n\n{ex['default_question']}\n\n{opts}\n\n{cf}\n\nFinal answer (A, B, or C):\n\nReasoning:"


STRATEGIES = {
    "naive": build_naive,
    "cot": build_cot,
    "abductive_cot": build_abductive,
    "hyp_eliminate": build_hyp_elim,
    "counterfactual": build_counterfactual,
}


def call_llm(model_id, prompt, retries=3):
    for a in range(retries):
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"},
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": "You are a commonsense reasoning assistant. Answer the MCQ based on reasoning about physical events. Always provide a clear answer letter."},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 400,
                    "temperature": 0.0,
                },
                timeout=30,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except:
            if a < retries - 1:
                time.sleep(1.5 ** a + random.random() * 0.5)
    return None


def extract(resp):
    if not resp: return None
    resp = resp.strip()
    ll = resp.split("\n")[-1].strip()
    for pattern in [
        r'(?:final\s+)?answer[:\s]*\(?([A-C])\)?',
        r'\*\*([A-C])\*\*',
        r'\b([A-C])\b\s*\.?\s*$',
        r'\(([A-C])\)',
    ]:
        m = re.search(pattern, resp if 'answer' in pattern.lower() or '\\*' in pattern else ll, re.IGNORECASE)
        if m: return m.group(1).upper()
    m = re.search(r'\b([A-C])\b', resp)
    return m.group(1).upper() if m else None


def run_one_experiment(samples, strategy_name, model_name, model_id):
    key = f"{model_name}__{strategy_name}"
    outfile = os.path.join(RESULTS_DIR, f"{key}.json")
    ckpt = outfile + ".ckpt"

    if os.path.exists(outfile):
        print(f"  SKIP {key}", flush=True)
        with open(outfile) as f:
            return json.load(f)

    # Resume from checkpoint
    done = {}
    if os.path.exists(ckpt):
        with open(ckpt) as f:
            for line in f:
                r = json.loads(line)
                done[r["q_id"]] = r

    prompt_fn = STRATEGIES[strategy_name]
    results = []
    correct = 0
    total = 0

    for i, ex in enumerate(samples):
        if ex["q_id"] in done:
            r = done[ex["q_id"]]
        else:
            prompt = prompt_fn(ex)
            response = call_llm(model_id, prompt)
            answer = extract(response)
            pred_idx = ord(answer) - ord('A') if answer else None
            r = {
                "q_id": ex["q_id"],
                "task": ex["task"],
                "difficulty": ex["difficulty"],
                "gt_idx": ex["mcq_gt_option"],
                "pred_idx": pred_idx,
                "pred_letter": answer,
                "correct": pred_idx == ex["mcq_gt_option"],
                "full_response": response,
            }
            with open(ckpt, "a") as f:
                f.write(json.dumps(r) + "\n")
            time.sleep(0.05)

        results.append(r)
        if r["correct"]: correct += 1
        total += 1

        if (i + 1) % 50 == 0:
            print(f"    [{key}] {i+1}/{len(samples)} acc={correct/total*100:.1f}%", flush=True)

    accuracy = correct / total * 100
    out = {"strategy": strategy_name, "model": model_name,
           "accuracy": accuracy, "correct": correct, "total": total, "results": results}
    with open(outfile, "w") as f:
        json.dump(out, f, indent=2)

    # Clean up checkpoint
    if os.path.exists(ckpt):
        os.remove(ckpt)

    print(f"  DONE {key}: {accuracy:.1f}%", flush=True)
    return out


def main():
    ds = load_dataset("UBC-ViL/BlackSwanSuite-MCQ", split="validation")
    samples = stratified_sample(ds, SAMPLE_SIZE)
    print(f"Samples: {len(samples)}", flush=True)

    all_results = {}
    for model_name, model_id in MODELS.items():
        for strat in STRATEGIES:
            key = f"{model_name}__{strat}"
            print(f"\n>>> {key}", flush=True)
            all_results[key] = run_one_experiment(samples, strat, model_name, model_id)

    print("\n" + "=" * 60)
    print(f"{'Model':<18} {'Strategy':<20} {'Acc':>6}")
    print("-" * 48)
    for k in sorted(all_results):
        r = all_results[k]
        print(f"{r['model']:<18} {r['strategy']:<20} {r['accuracy']:>5.1f}%")

    summary = {k: {kk: vv for kk, vv in v.items() if kk != "results"} for k, v in all_results.items()}
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
