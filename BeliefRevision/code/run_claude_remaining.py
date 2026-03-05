#!/usr/bin/env python3
"""Run Claude remaining experiments with incremental saves."""
import sys, json, time, os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from run_experiments import (
    SCENARIOS, MODELS, RESULTS_DIR, get_prompts,
    parse_answer, parse_confidence, parse_counterfactual, parse_what_changed, call_vlm
)

def run_incremental(model_name, model_id, condition, scenarios):
    out_file = RESULTS_DIR / f"results_{model_name}_{condition}.json"
    partial_file = RESULTS_DIR / f"partial_{model_name}_{condition}.json"
    
    if out_file.exists():
        print(f"  SKIP {model_name}/{condition}")
        return
    
    # Load partial progress
    results = []
    start_idx = 0
    if partial_file.exists():
        results = json.load(open(partial_file))
        start_idx = len(results)
        print(f"  Resuming {model_name}/{condition} from item {start_idx}")
    
    for idx in range(start_idx, len(scenarios)):
        scenario = scenarios[idx]
        question = scenario["question"]
        options = scenario["options"]
        gt_answer = scenario["answer"]
        
        phase_a_prompt = (
            f"SCENE DESCRIPTION (Pre-Event Only):\n"
            f'"{scenario["pre_event"]}"\n\n'
            f"Based ONLY on this pre-event description:\n\n"
            + get_prompts(question, options, "A", condition)
        )
        response_a = call_vlm(model_id, phase_a_prompt)
        answer_a = parse_answer(response_a)
        
        phase_b_prompt = (
            f"SCENE DESCRIPTION (Pre + Post Evidence):\n"
            f"PRE-EVENT: {scenario['pre_event']}\n\n"
            f"POST-EVENT (NEW): {scenario['post_event']}\n\n"
            + get_prompts(question, options, "B", condition, phase_a_answer=answer_a if answer_a >= 0 else None)
        )
        response_b = call_vlm(model_id, phase_b_prompt)
        answer_b = parse_answer(response_b)
        
        entry = {
            "scenario_id": scenario["id"], "category": scenario["category"],
            "gt_answer": gt_answer, "prior_bias": scenario.get("prior_bias", -1),
            "phase_a_answer": answer_a, "phase_b_answer": answer_b,
            "phase_a_correct": answer_a == gt_answer, "phase_b_correct": answer_b == gt_answer,
            "answer_changed": answer_a != answer_b,
            "phase_a_response": response_a[:500], "phase_b_response": response_b[:500],
        }
        if condition == "belief_state":
            entry["confidence_a"] = parse_confidence(response_a)
            entry["confidence_b"] = parse_confidence(response_b)
            entry["what_changed"] = parse_what_changed(response_b)
        if condition == "counterfactual":
            entry["counterfactual_aware"] = parse_counterfactual(response_b)
        
        results.append(entry)
        
        # Save partial every 5 items
        if (idx + 1) % 5 == 0:
            with open(partial_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"    {idx+1}/{len(scenarios)} saved")
        
        time.sleep(0.2)
    
    # Save final
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    if partial_file.exists():
        os.remove(partial_file)
    print(f"  DONE: {len(results)} → {out_file.name}")


if __name__ == "__main__":
    model_id = MODELS["claude"]
    for cond in ["belief_state", "counterfactual"]:
        run_incremental("claude", model_id, cond, SCENARIOS)
