#!/usr/bin/env python3
"""Run remaining experiments incrementally, skipping what's already done."""
import os, sys, json, time, random, requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import (
    SCENARIOS, MODELS, RESULTS_DIR, get_prompts, 
    parse_answer, parse_confidence, parse_counterfactual, parse_what_changed,
    call_vlm, compute_metrics
)

def run_single(model_name, model_id, condition, scenarios):
    out_file = RESULTS_DIR / f"results_{model_name}_{condition}.json"
    if out_file.exists():
        print(f"  SKIP {model_name}/{condition} — already exists")
        return json.load(open(out_file))
    
    print(f"  Running {model_name}/{condition} ({len(scenarios)} scenarios)...")
    results = []
    
    for idx, scenario in enumerate(scenarios):
        sid = scenario["id"]
        question = scenario["question"]
        options = scenario["options"]
        gt_answer = scenario["answer"]
        pre_event = scenario["pre_event"]
        post_event = scenario["post_event"]
        
        # Phase A
        phase_a_prompt = (
            f"SCENE DESCRIPTION (Pre-Event Only):\n"
            f"You can see the following scene BEFORE a surprising event occurred:\n"
            f'"{pre_event}"\n\n'
            f"Based ONLY on this pre-event description, answer the following:\n\n"
            + get_prompts(question, options, "A", condition)
        )
        response_a = call_vlm(model_id, phase_a_prompt)
        answer_a = parse_answer(response_a)
        
        # Phase B
        phase_b_prompt = (
            f"SCENE DESCRIPTION (Pre-Event + Post-Event Evidence):\n"
            f"PRE-EVENT: {pre_event}\n\n"
            f"POST-EVENT (NEW EVIDENCE): {post_event}\n\n"
            + get_prompts(question, options, "B", condition, phase_a_answer=answer_a if answer_a >= 0 else None)
        )
        response_b = call_vlm(model_id, phase_b_prompt)
        answer_b = parse_answer(response_b)
        
        entry = {
            "scenario_id": sid,
            "category": scenario["category"],
            "gt_answer": gt_answer,
            "prior_bias": scenario.get("prior_bias", -1),
            "phase_a_answer": answer_a,
            "phase_b_answer": answer_b,
            "phase_a_correct": answer_a == gt_answer,
            "phase_b_correct": answer_b == gt_answer,
            "answer_changed": answer_a != answer_b,
            "phase_a_response": response_a[:500],
            "phase_b_response": response_b[:500],
        }
        if condition == "belief_state":
            entry["confidence_a"] = parse_confidence(response_a)
            entry["confidence_b"] = parse_confidence(response_b)
            entry["what_changed"] = parse_what_changed(response_b)
        if condition == "counterfactual":
            entry["counterfactual_aware"] = parse_counterfactual(response_b)
        
        results.append(entry)
        if (idx + 1) % 10 == 0:
            print(f"    {idx+1}/{len(scenarios)} done")
        time.sleep(0.3)
    
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {out_file.name}")
    return results


if __name__ == "__main__":
    scenarios = SCENARIOS
    conditions = ["baseline", "belief_state", "counterfactual"]
    
    all_results = {}
    for model_name, model_id in MODELS.items():
        print(f"\nModel: {model_name}")
        all_results[model_name] = {}
        for condition in conditions:
            all_results[model_name][condition] = run_single(model_name, model_id, condition, scenarios)
    
    # Save combined
    with open(RESULTS_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Compute and save metrics
    metrics = compute_metrics(all_results)
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 100)
    print(f"{'Model':<10} {'Condition':<16} {'PhA Acc':>8} {'PhB Acc':>8} {'Δ':>6} {'Stubborn':>9} {'Approp':>8} {'Regress':>8} {'Change':>8}")
    print("-" * 100)
    for mn in metrics:
        for cond in metrics[mn]:
            m = metrics[mn][cond]
            print(
                f"{mn:<10} {cond:<16} "
                f"{m['phase_a_accuracy']:>7.1%} {m['phase_b_accuracy']:>7.1%} "
                f"{m['accuracy_delta']:>+5.1%} {m['stubbornness_rate']:>8.1%} "
                f"{m['appropriate_update_rate']:>7.1%} {m['regression_rate']:>7.1%} "
                f"{m['change_rate']:>7.1%}"
            )
    print("=" * 100)
