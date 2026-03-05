#!/usr/bin/env python3
"""
Comprehensive analysis of CounterBench results.
Computes:
  - Counterfactual Consistency Score (CCS) per model, category, subcategory
  - Original accuracy, intervened accuracy
  - Breakdown by should_change vs should_not_change
  - Error type analysis
  - Saves all tables as JSON for figure generation
"""

import json
import os
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = DATA_DIR / "results"
ANALYSIS_DIR = DATA_DIR / "analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)

MODEL_NAMES = {
    "gpt4o": "GPT-4o",
    "claude35sonnet": "Claude 3.5 Sonnet",
    "gemini2flash": "Gemini 2.0 Flash",
    "llama32_90b": "Llama 3.2 90B",
    "qwen2vl72b": "Qwen2-VL 72B",
    "gemini25flash": "Gemini 2.5 Flash",
}


def load_results():
    """Load all per-model result files."""
    all_results = {}
    for model_key in MODEL_NAMES:
        path = RESULTS_DIR / f"{model_key}_results.json"
        if path.exists():
            with open(path) as f:
                all_results[model_key] = json.load(f)
    return all_results


def compute_metrics(results):
    """Compute comprehensive metrics for a single model's results."""
    metrics = {
        "total": len(results),
        "orig_correct": 0,
        "intv_correct": 0,
        "consistent": 0,  # CCS: answer changes iff should_change
        "true_positive": 0,  # should change AND did change correctly
        "true_negative": 0,  # should NOT change AND did NOT change
        "false_positive": 0,  # should NOT change BUT did change
        "false_negative": 0,  # should change BUT did NOT change correctly
        "should_change_total": 0,
        "should_not_change_total": 0,
    }
    
    error_types = defaultdict(int)
    
    for r in results:
        gt_orig = r["gt_original"].strip().lower()
        gt_intv = r["gt_intervened"].strip().lower()
        pred_orig = r["pred_original"].strip().lower()
        pred_intv = r["pred_intervened"].strip().lower()
        should_change = r["should_change"]
        
        # Original accuracy
        orig_correct = pred_orig == gt_orig
        if orig_correct:
            metrics["orig_correct"] += 1
        
        # Intervened accuracy
        intv_correct = pred_intv == gt_intv
        if intv_correct:
            metrics["intv_correct"] += 1
        
        # Counterfactual consistency
        if should_change:
            metrics["should_change_total"] += 1
            # Model should give different answer (matching new GT)
            answer_changed = pred_orig != pred_intv
            if answer_changed and intv_correct:
                metrics["true_positive"] += 1
                metrics["consistent"] += 1
            elif not answer_changed:
                metrics["false_negative"] += 1
                error_types["sticky_answer"] += 1
            else:
                # Changed but wrong
                metrics["false_negative"] += 1
                error_types["wrong_change"] += 1
        else:
            metrics["should_not_change_total"] += 1
            answer_changed = pred_orig != pred_intv
            if not answer_changed:
                metrics["true_negative"] += 1
                metrics["consistent"] += 1
            else:
                metrics["false_positive"] += 1
                error_types["spurious_change"] += 1
    
    # Derived metrics
    n = metrics["total"]
    metrics["orig_accuracy"] = metrics["orig_correct"] / n if n else 0
    metrics["intv_accuracy"] = metrics["intv_correct"] / n if n else 0
    metrics["ccs"] = metrics["consistent"] / n if n else 0
    
    sc = metrics["should_change_total"]
    snc = metrics["should_not_change_total"]
    metrics["sensitivity"] = metrics["true_positive"] / sc if sc else 0
    metrics["specificity"] = metrics["true_negative"] / snc if snc else 0
    
    metrics["error_types"] = dict(error_types)
    
    return metrics


def compute_by_category(results):
    """Compute metrics broken down by category."""
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
    
    cat_metrics = {}
    for cat, cat_results in by_cat.items():
        cat_metrics[cat] = compute_metrics(cat_results)
    return cat_metrics


def compute_by_subcategory(results):
    """Compute metrics broken down by subcategory."""
    by_subcat = defaultdict(list)
    for r in results:
        by_subcat[r["subcategory"]].append(r)
    
    subcat_metrics = {}
    for subcat, sub_results in by_subcat.items():
        subcat_metrics[subcat] = compute_metrics(sub_results)
    return subcat_metrics


def compute_by_intervention(results):
    """Compute metrics broken down by intervention type."""
    by_intv = defaultdict(list)
    for r in results:
        by_intv[r["intervention"]].append(r)
    
    intv_metrics = {}
    for intv, intv_results in by_intv.items():
        intv_metrics[intv] = compute_metrics(intv_results)
    return intv_metrics


def main():
    all_results = load_results()
    print(f"Loaded results for {len(all_results)} models")
    
    # ── Overall metrics ───────────────────────────────────────────────────
    overall = {}
    for model_key, results in all_results.items():
        m = compute_metrics(results)
        overall[model_key] = m
        name = MODEL_NAMES[model_key]
        print(f"\n{name}:")
        print(f"  Original Accuracy:  {m['orig_accuracy']:.3f}")
        print(f"  Intervened Accuracy: {m['intv_accuracy']:.3f}")
        print(f"  CCS (Counterfactual Consistency): {m['ccs']:.3f}")
        print(f"  Sensitivity (should change → did): {m['sensitivity']:.3f}")
        print(f"  Specificity (shouldn't change → didn't): {m['specificity']:.3f}")
        print(f"  Error types: {m['error_types']}")
    
    # ── By category ───────────────────────────────────────────────────────
    category_results = {}
    for model_key, results in all_results.items():
        category_results[model_key] = compute_by_category(results)
    
    print("\n\n" + "="*80)
    print("CCS BY CATEGORY")
    print("="*80)
    cats = ["spatial", "causal", "compositional", "counting", "occlusion"]
    header = f"{'Model':<25}" + "".join(f"{c:>14}" for c in cats)
    print(header)
    print("-" * len(header))
    for model_key in all_results:
        row = f"{MODEL_NAMES[model_key]:<25}"
        for cat in cats:
            if cat in category_results[model_key]:
                ccs = category_results[model_key][cat]["ccs"]
                row += f"{ccs:>14.3f}"
            else:
                row += f"{'N/A':>14}"
        print(row)
    
    # ── By subcategory ────────────────────────────────────────────────────
    subcategory_results = {}
    for model_key, results in all_results.items():
        subcategory_results[model_key] = compute_by_subcategory(results)
    
    # ── By intervention type ──────────────────────────────────────────────
    intervention_results = {}
    for model_key, results in all_results.items():
        intervention_results[model_key] = compute_by_intervention(results)
    
    # ── Save all analysis ─────────────────────────────────────────────────
    analysis = {
        "overall": overall,
        "by_category": category_results,
        "by_subcategory": subcategory_results,
        "by_intervention": intervention_results,
        "model_names": MODEL_NAMES,
    }
    
    with open(ANALYSIS_DIR / "full_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    
    # ── LaTeX tables ──────────────────────────────────────────────────────
    # Main results table
    print("\n\n" + "="*80)
    print("LATEX TABLE: Main Results")
    print("="*80)
    
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\caption{Main results on CounterBench. CCS = Counterfactual Consistency Score; Sens. = Sensitivity (true change rate); Spec. = Specificity (true no-change rate).}")
    print(r"\label{tab:main_results}")
    print(r"\small")
    print(r"\begin{tabular}{l c c c c c}")
    print(r"\toprule")
    print(r"Model & Orig. Acc. & Intv. Acc. & CCS $\uparrow$ & Sens. $\uparrow$ & Spec. $\uparrow$ \\")
    print(r"\midrule")
    
    # Sort by CCS
    sorted_models = sorted(overall.keys(), key=lambda k: overall[k]["ccs"], reverse=True)
    for model_key in sorted_models:
        m = overall[model_key]
        name = MODEL_NAMES[model_key]
        print(f"{name} & {m['orig_accuracy']:.3f} & {m['intv_accuracy']:.3f} & {m['ccs']:.3f} & {m['sensitivity']:.3f} & {m['specificity']:.3f} \\\\")
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")
    
    # Category breakdown table
    print("\n\n" + "="*80)
    print("LATEX TABLE: CCS by Category")
    print("="*80)
    
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\caption{Counterfactual Consistency Score broken down by reasoning category.}")
    print(r"\label{tab:category_results}")
    print(r"\small")
    print(r"\begin{tabular}{l c c c c c c}")
    print(r"\toprule")
    print(r"Model & Spatial & Causal & Compositional & Counting & Occlusion & Overall \\")
    print(r"\midrule")
    
    for model_key in sorted_models:
        name = MODEL_NAMES[model_key]
        row = f"{name}"
        for cat in cats:
            if cat in category_results[model_key]:
                ccs = category_results[model_key][cat]["ccs"]
                row += f" & {ccs:.3f}"
            else:
                row += " & ---"
        row += f" & {overall[model_key]['ccs']:.3f}"
        row += r" \\"
        print(row)
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")
    
    print("\n\nAnalysis complete. Results saved to", ANALYSIS_DIR)


if __name__ == "__main__":
    main()
