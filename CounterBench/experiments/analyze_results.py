"""
CounterBench: Comprehensive Analysis Pipeline
Computes counterfactual consistency scores, accuracy, and generates all figures.
"""

import os
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR = Path(__file__).parent.parent / "data"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Models to analyze (only those with good results)
MODELS = ["gpt4o", "gemini_flash", "qwen_vl", "gpt4o_mini"]
MODEL_DISPLAY = {
    "gpt4o": "GPT-4o",
    "gemini_flash": "Gemini 2.0 Flash",
    "qwen_vl": "Qwen2.5-VL-72B",
    "gpt4o_mini": "GPT-4o mini",
}

CATEGORY_DISPLAY = {
    "spatial": "Spatial",
    "attribute": "Attribute",
    "counting": "Counting",
    "containment": "Containment",
    "causal": "Causal",
    "negative_control": "Neg. Control"
}

# Color palette (colorblind-safe)
COLORS = {
    "gpt4o": "#0072B2",
    "gemini_flash": "#D55E00",
    "qwen_vl": "#009E73",
    "gpt4o_mini": "#CC79A7",
}


def normalize_answer(answer, category, gt_answer):
    """Normalize model response for comparison."""
    if not answer or answer.startswith("ERROR"):
        return None
    
    a = answer.lower().strip().rstrip('.').strip()
    
    # Remove common prefixes
    for prefix in ["the answer is ", "answer: ", "the ", "it is ", "it's "]:
        if a.startswith(prefix):
            a = a[len(prefix):]
    
    # Yes/no normalization
    if gt_answer.lower() in ["yes", "no"]:
        if "yes" in a and "no" not in a:
            return "yes"
        elif "no" in a and "yes" not in a:
            return "no"
        elif a.startswith("yes"):
            return "yes"
        elif a.startswith("no"):
            return "no"
        return a
    
    # Number normalization
    if gt_answer.isdigit():
        nums = re.findall(r'\d+', a)
        if nums:
            return nums[0]
        # Word to number
        word_to_num = {"one":"1","two":"2","three":"3","four":"4","five":"5",
                       "six":"6","seven":"7","eight":"8","nine":"9","zero":"0"}
        for w, n in word_to_num.items():
            if w in a:
                return n
        return a
    
    # Color normalization
    colors = ["red","blue","green","yellow","purple","orange","cyan","pink"]
    for c in colors:
        if c in a:
            return c
    
    # Shape normalization for causal arrow
    a = a.replace("triangle", "triangle").replace("triangles", "triangle")
    return a.strip()


def check_match(response, gt_answer, category):
    """Check if response matches ground truth."""
    norm = normalize_answer(response, category, gt_answer)
    if norm is None:
        return False
    
    gt = gt_answer.lower().strip()
    
    # Exact match
    if norm == gt:
        return True
    
    # For color+shape answers (causal arrow), check if both parts present
    if " " in gt:
        parts = gt.split()
        return all(p in norm for p in parts)
    
    return False


def compute_metrics(model_name):
    """Compute all metrics for a model."""
    rpath = RESULTS_DIR / f"{model_name}_results.json"
    with open(rpath) as f:
        data = json.load(f)
    
    results = data["results"]
    metrics = {
        "model": model_name,
        "total": 0,
        "valid": 0,
        "by_category": defaultdict(lambda: {
            "total": 0, "valid": 0,
            "orig_correct": 0, "int_correct": 0,
            "both_correct": 0, "consistency": 0,
            "flip_when_should": 0, "flip_when_shouldnt": 0,
            "should_flip_total": 0, "shouldnt_flip_total": 0,
        })
    }
    
    # Global metrics
    orig_correct_total = 0
    int_correct_total = 0
    both_correct_total = 0
    consistency_total = 0
    valid_total = 0
    
    for r in results:
        cat = r["category"]
        m = metrics["by_category"][cat]
        m["total"] += 1
        metrics["total"] += 1
        
        if r["orig_response"].startswith("ERROR") or r["int_response"].startswith("ERROR"):
            continue
        
        m["valid"] += 1
        valid_total += 1
        
        orig_match = check_match(r["orig_response"], r["original_answer_gt"], cat)
        int_match = check_match(r["int_response"], r["intervened_answer_gt"], cat)
        
        if orig_match:
            m["orig_correct"] += 1
            orig_correct_total += 1
        if int_match:
            m["int_correct"] += 1
            int_correct_total += 1
        if orig_match and int_match:
            m["both_correct"] += 1
            both_correct_total += 1
        
        # Consistency: did the answer change status as expected?
        orig_norm = normalize_answer(r["orig_response"], cat, r["original_answer_gt"])
        int_norm = normalize_answer(r["int_response"], cat, r["intervened_answer_gt"])
        
        answer_changed = (orig_norm != int_norm) if (orig_norm and int_norm) else False
        
        if r["should_flip"]:
            m["should_flip_total"] += 1
            if answer_changed:
                m["flip_when_should"] += 1
        else:
            m["shouldnt_flip_total"] += 1
            if not answer_changed:
                m["flip_when_shouldnt"] += 1  # Actually this is "stable when should be"
        
        # Counterfactual consistency: correct on both AND answer changed correctly
        if r["should_flip"]:
            if orig_match and int_match:
                m["consistency"] += 1
                consistency_total += 1
        else:
            if orig_match and int_match and not answer_changed:
                m["consistency"] += 1
                consistency_total += 1
    
    metrics["valid"] = valid_total
    metrics["orig_accuracy"] = orig_correct_total / valid_total if valid_total > 0 else 0
    metrics["int_accuracy"] = int_correct_total / valid_total if valid_total > 0 else 0
    metrics["both_accuracy"] = both_correct_total / valid_total if valid_total > 0 else 0
    metrics["consistency_score"] = consistency_total / valid_total if valid_total > 0 else 0
    
    # Per-category rates
    for cat, m in metrics["by_category"].items():
        v = m["valid"]
        m["orig_acc"] = m["orig_correct"] / v if v > 0 else 0
        m["int_acc"] = m["int_correct"] / v if v > 0 else 0
        m["both_acc"] = m["both_correct"] / v if v > 0 else 0
        m["consistency_rate"] = m["consistency"] / v if v > 0 else 0
        m["flip_rate"] = m["flip_when_should"] / m["should_flip_total"] if m["should_flip_total"] > 0 else 0
        m["stability_rate"] = m["flip_when_shouldnt"] / m["shouldnt_flip_total"] if m["shouldnt_flip_total"] > 0 else 0
    
    return metrics


def print_metrics(all_metrics):
    """Print summary tables."""
    print("\n" + "="*80)
    print("COUNTERBENCH RESULTS SUMMARY")
    print("="*80)
    
    # Overall table
    print(f"\n{'Model':25s} {'Orig Acc':>10s} {'Int Acc':>10s} {'Both Acc':>10s} {'CCS':>10s}")
    print("-"*65)
    for m in all_metrics:
        name = MODEL_DISPLAY.get(m["model"], m["model"])
        print(f"{name:25s} {m['orig_accuracy']:10.1%} {m['int_accuracy']:10.1%} {m['both_accuracy']:10.1%} {m['consistency_score']:10.1%}")
    
    # Per-category
    categories = ["spatial", "attribute", "counting", "containment", "causal", "negative_control"]
    
    print(f"\n\n{'':25s}", end="")
    for cat in categories:
        print(f" {CATEGORY_DISPLAY[cat]:>12s}", end="")
    print()
    
    print("\nCounterfactual Consistency Score (CCS):")
    print("-"*100)
    for m in all_metrics:
        name = MODEL_DISPLAY.get(m["model"], m["model"])
        print(f"{name:25s}", end="")
        for cat in categories:
            if cat in m["by_category"]:
                v = m["by_category"][cat]["consistency_rate"]
                print(f" {v:12.1%}", end="")
            else:
                print(f" {'N/A':>12s}", end="")
        print()
    
    print("\nOriginal Image Accuracy:")
    print("-"*100)
    for m in all_metrics:
        name = MODEL_DISPLAY.get(m["model"], m["model"])
        print(f"{name:25s}", end="")
        for cat in categories:
            if cat in m["by_category"]:
                v = m["by_category"][cat]["orig_acc"]
                print(f" {v:12.1%}", end="")
            else:
                print(f" {'N/A':>12s}", end="")
        print()
    
    print("\nAnswer Flip Rate (when should flip):")
    print("-"*100)
    for m in all_metrics:
        name = MODEL_DISPLAY.get(m["model"], m["model"])
        print(f"{name:25s}", end="")
        for cat in categories:
            if cat in m["by_category"]:
                v = m["by_category"][cat]["flip_rate"]
                print(f" {v:12.1%}", end="")
            else:
                print(f" {'N/A':>12s}", end="")
        print()


def plot_main_figure(all_metrics):
    """Figure 1: Main bar chart - CCS by category and model."""
    categories = ["spatial", "attribute", "counting", "containment", "causal"]
    n_cats = len(categories)
    n_models = len(all_metrics)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(n_cats)
    width = 0.18
    
    for i, m in enumerate(all_metrics):
        vals = [m["by_category"].get(cat, {}).get("consistency_rate", 0) * 100 for cat in categories]
        offset = (i - n_models/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=MODEL_DISPLAY[m["model"]], 
                      color=COLORS[m["model"]], edgecolor='white', linewidth=0.5)
        # Add value labels
        for bar, val in zip(bars, vals):
            if val > 5:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5,
                        f'{val:.0f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    ax.set_ylabel('Counterfactual Consistency Score (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_DISPLAY[c] for c in categories], fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_title('Counterfactual Consistency Score by Task Category', fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ccs_by_category.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'ccs_by_category.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: ccs_by_category.pdf")


def plot_accuracy_comparison(all_metrics):
    """Figure 2: Original vs Intervened accuracy comparison."""
    categories = ["spatial", "attribute", "counting", "containment", "causal"]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    n_cats = len(categories)
    n_models = len(all_metrics)
    x = np.arange(n_cats)
    width = 0.18
    
    for panel_idx, (metric_key, title) in enumerate([
        ("orig_acc", "Original Image Accuracy"),
        ("int_acc", "Intervened Image Accuracy")
    ]):
        ax = axes[panel_idx]
        for i, m in enumerate(all_metrics):
            vals = [m["by_category"].get(cat, {}).get(metric_key, 0) * 100 for cat in categories]
            offset = (i - n_models/2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=MODEL_DISPLAY[m["model"]], 
                   color=COLORS[m["model"]], edgecolor='white', linewidth=0.5)
        
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([CATEGORY_DISPLAY[c] for c in categories], fontsize=9, rotation=15)
        ax.set_ylim(0, 105)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        if panel_idx == 1:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'accuracy_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: accuracy_comparison.pdf")


def plot_gap_analysis(all_metrics):
    """Figure 3: Gap between original accuracy and CCS (the 'consistency gap')."""
    categories = ["spatial", "attribute", "counting", "containment", "causal"]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    n_models = len(all_metrics)
    x = np.arange(len(categories))
    width = 0.18
    
    for i, m in enumerate(all_metrics):
        orig_accs = [m["by_category"].get(cat, {}).get("orig_acc", 0) * 100 for cat in categories]
        ccs_vals = [m["by_category"].get(cat, {}).get("consistency_rate", 0) * 100 for cat in categories]
        gaps = [o - c for o, c in zip(orig_accs, ccs_vals)]
        
        offset = (i - n_models/2 + 0.5) * width
        ax.bar(x + offset, gaps, width, label=MODEL_DISPLAY[m["model"]], 
               color=COLORS[m["model"]], edgecolor='white', linewidth=0.5)
    
    ax.set_ylabel('Consistency Gap (Orig Acc - CCS) (pp)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_DISPLAY[c] for c in categories], fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_title('Consistency Gap: How Much Does Performance Drop Under Intervention?', fontsize=12, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'consistency_gap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'consistency_gap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: consistency_gap.pdf")


def plot_negative_control(all_metrics):
    """Figure 4: Negative control - stability when intervention is irrelevant."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    models = []
    stability_rates = []
    ccs_rates = []
    
    for m in all_metrics:
        nc = m["by_category"].get("negative_control", {})
        if nc.get("valid", 0) > 0:
            models.append(MODEL_DISPLAY[m["model"]])
            stability_rates.append(nc.get("consistency_rate", 0) * 100)
            ccs_rates.append(m["consistency_score"] * 100)
    
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, stability_rates, width, label='Neg. Control Stability', color='#56B4E9', edgecolor='white')
    ax.bar(x + width/2, ccs_rates, width, label='Overall CCS', color='#E69F00', edgecolor='white')
    
    ax.set_ylabel('Score (%)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9, rotation=15)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_title('Stability on Irrelevant Interventions vs Overall CCS', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'negative_control.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'negative_control.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: negative_control.pdf")


def plot_radar_chart(all_metrics):
    """Figure 5: Radar/spider chart of CCS across categories."""
    categories = ["spatial", "attribute", "counting", "containment", "causal"]
    labels = [CATEGORY_DISPLAY[c] for c in categories]
    
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    
    for m in all_metrics:
        vals = [m["by_category"].get(cat, {}).get("consistency_rate", 0) * 100 for cat in categories]
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', linewidth=2, label=MODEL_DISPLAY[m["model"]], 
                color=COLORS[m["model"]], markersize=6)
        ax.fill(angles, vals, alpha=0.1, color=COLORS[m["model"]])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=8)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.set_title('Counterfactual Consistency Score Profile', fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'radar_ccs.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'radar_ccs.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: radar_ccs.pdf")


def plot_overall_summary(all_metrics):
    """Figure 6: Overall summary bar chart."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    models = [MODEL_DISPLAY[m["model"]] for m in all_metrics]
    orig_accs = [m["orig_accuracy"] * 100 for m in all_metrics]
    int_accs = [m["int_accuracy"] * 100 for m in all_metrics]
    both_accs = [m["both_accuracy"] * 100 for m in all_metrics]
    ccs_vals = [m["consistency_score"] * 100 for m in all_metrics]
    
    x = np.arange(len(models))
    width = 0.2
    
    ax.bar(x - 1.5*width, orig_accs, width, label='Original Acc', color='#56B4E9', edgecolor='white')
    ax.bar(x - 0.5*width, int_accs, width, label='Intervened Acc', color='#009E73', edgecolor='white')
    ax.bar(x + 0.5*width, both_accs, width, label='Both Correct', color='#E69F00', edgecolor='white')
    ax.bar(x + 1.5*width, ccs_vals, width, label='CCS', color='#CC79A7', edgecolor='white')
    
    # Add CCS values on top
    for i, v in enumerate(ccs_vals):
        ax.text(x[i] + 1.5*width, v + 1, f'{v:.1f}%', ha='center', va='bottom', 
                fontsize=9, fontweight='bold', color='#CC79A7')
    
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_title('Overall Performance: Accuracy vs Counterfactual Consistency', fontsize=12, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'overall_summary.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'overall_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: overall_summary.pdf")


def create_example_figure():
    """Create example pairs figure for the paper."""
    from PIL import Image as PILImage
    
    # Select one example from each category
    examples = {
        "spatial": "spatial_005",
        "attribute": "attribute_010",
        "counting": "counting_003",
        "containment": "containment_007",
        "causal": "causal_arrow_002",
    }
    
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.5))
    
    for col, (cat, eid) in enumerate(examples.items()):
        orig_path = DATA_DIR / "images" / f"{eid}_orig.png"
        int_path = DATA_DIR / "images" / f"{eid}_int.png"
        
        if orig_path.exists() and int_path.exists():
            orig_img = plt.imread(str(orig_path))
            int_img = plt.imread(str(int_path))
            
            axes[0, col].imshow(orig_img)
            axes[0, col].set_title(f'{CATEGORY_DISPLAY[cat]}', fontsize=11, fontweight='bold')
            axes[0, col].axis('off')
            
            axes[1, col].imshow(int_img)
            axes[1, col].axis('off')
        else:
            axes[0, col].text(0.5, 0.5, 'N/A', ha='center', va='center')
            axes[1, col].text(0.5, 0.5, 'N/A', ha='center', va='center')
            axes[0, col].set_title(CATEGORY_DISPLAY[cat], fontsize=11, fontweight='bold')
    
    axes[0, 0].set_ylabel('Original', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Intervened', fontsize=12, fontweight='bold')
    
    plt.suptitle('Example CounterBench Pairs: Original (top) vs Intervened (bottom)', 
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'example_pairs.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'example_pairs.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: example_pairs.pdf")


def save_results_json(all_metrics):
    """Save computed metrics to JSON for the paper."""
    output = {}
    for m in all_metrics:
        model = m["model"]
        output[model] = {
            "display_name": MODEL_DISPLAY.get(model, model),
            "orig_accuracy": round(m["orig_accuracy"] * 100, 1),
            "int_accuracy": round(m["int_accuracy"] * 100, 1),
            "both_accuracy": round(m["both_accuracy"] * 100, 1),
            "ccs": round(m["consistency_score"] * 100, 1),
            "valid_pairs": m["valid"],
            "by_category": {}
        }
        for cat in ["spatial", "attribute", "counting", "containment", "causal", "negative_control"]:
            if cat in m["by_category"]:
                bc = m["by_category"][cat]
                output[model]["by_category"][cat] = {
                    "orig_acc": round(bc.get("orig_acc", 0) * 100, 1),
                    "int_acc": round(bc.get("int_acc", 0) * 100, 1),
                    "both_acc": round(bc.get("both_acc", 0) * 100, 1),
                    "ccs": round(bc.get("consistency_rate", 0) * 100, 1),
                    "flip_rate": round(bc.get("flip_rate", 0) * 100, 1),
                    "valid": bc.get("valid", 0),
                }
    
    with open(RESULTS_DIR / "analysis_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved analysis to: {RESULTS_DIR / 'analysis_results.json'}")


def main():
    print("Computing metrics for all models...")
    all_metrics = []
    
    for model in MODELS:
        rpath = RESULTS_DIR / f"{model}_results.json"
        if rpath.exists():
            print(f"  Analyzing {model}...")
            m = compute_metrics(model)
            all_metrics.append(m)
        else:
            print(f"  Skipping {model} (no results)")
    
    # Print tables
    print_metrics(all_metrics)
    
    # Generate figures
    print("\nGenerating figures...")
    plot_main_figure(all_metrics)
    plot_accuracy_comparison(all_metrics)
    plot_gap_analysis(all_metrics)
    plot_negative_control(all_metrics)
    plot_radar_chart(all_metrics)
    plot_overall_summary(all_metrics)
    create_example_figure()
    
    # Save JSON
    save_results_json(all_metrics)
    
    print("\nAll done!")


if __name__ == "__main__":
    main()
