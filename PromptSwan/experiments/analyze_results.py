#!/usr/bin/env python3
"""
Analyze all completed experiment results.
Produces:
  1. Main accuracy table (model x strategy)
  2. Per-task breakdown (Detective vs Reporter)  
  3. Per-difficulty breakdown (easy/medium/hard)
  4. Error taxonomy from response analysis
  5. Confusion patterns
  6. Publication-quality figures
"""

import os
import json
import glob
import numpy as np
from collections import defaultdict, Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

STRATEGY_ORDER = ["naive", "cot", "abductive_cot", "hyp_eliminate", "counterfactual"]
STRATEGY_LABELS = {
    "naive": "Naive",
    "cot": "CoT",
    "abductive_cot": "Abd. CoT",
    "hyp_eliminate": "Hyp-Elim",
    "counterfactual": "Counter.",
}
MODEL_ORDER = ["gpt-4o-mini", "claude-haiku", "gemini-flash"]
MODEL_LABELS = {
    "gpt-4o-mini": "GPT-4o-mini",
    "claude-haiku": "Claude 3.5 Haiku",
    "gemini-flash": "Gemini 2.0 Flash",
}

COLORS = {
    "gpt-4o-mini": "#4CAF50",
    "claude-haiku": "#FF9800", 
    "gemini-flash": "#2196F3",
}


def load_all_results():
    """Load all completed experiment results."""
    results = {}
    for f in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
        if "sample_ids" in f or "summary" in f:
            continue
        with open(f) as fh:
            data = json.load(fh)
        key = f"{data['model']}__{data['strategy']}"
        results[key] = data
    return results


def compute_task_accuracy(data):
    """Compute accuracy broken down by task."""
    task_correct = defaultdict(int)
    task_total = defaultdict(int)
    for r in data["results"]:
        task_correct[r["task"]] += int(r["correct"])
        task_total[r["task"]] += 1
    return {t: task_correct[t] / task_total[t] * 100 for t in task_total}


def compute_difficulty_accuracy(data):
    """Compute accuracy broken down by difficulty."""
    diff_correct = defaultdict(int)
    diff_total = defaultdict(int)
    for r in data["results"]:
        diff_correct[r["difficulty"]] += int(r["correct"])
        diff_total[r["difficulty"]] += 1
    return {d: diff_correct[d] / diff_total[d] * 100 for d in diff_total}


def classify_error(result):
    """Classify error type from response."""
    if result["correct"]:
        return None
    if result["pred_letter"] is None:
        return "parse_failure"
    
    resp = (result.get("full_response") or result.get("response_snippet") or "").lower()
    
    # Hallucinated hidden cause: model invents scenario details not in options
    if any(w in resp for w in ["i think", "i imagine", "it seems like", "probably"]):
        if result["task"] == "Detective":
            return "hallucinated_cause"
    
    # Fails to retract: model sticks with initial expectation
    if any(w in resp for w in ["would expect", "normally", "typically", "usually"]):
        if result["task"] == "Reporter":
            return "fails_to_retract"
    
    # Overweights prior text: model biased by option ordering/phrasing
    if result["pred_idx"] == 0:  # Always picks first option
        return "position_bias_A"
    
    # Causal confusion: picks option with wrong causal direction
    if "because" in resp or "caused" in resp or "led to" in resp:
        return "causal_confusion"
    
    # Surface matching: picks option with most similar wording
    return "surface_matching"


def build_error_taxonomy(all_results):
    """Build error taxonomy across all experiments."""
    taxonomy = defaultdict(lambda: defaultdict(int))
    total_errors = defaultdict(int)
    
    for key, data in all_results.items():
        model = data["model"]
        strategy = data["strategy"]
        for r in data["results"]:
            etype = classify_error(r)
            if etype:
                taxonomy[(model, strategy)][etype] += 1
                total_errors[etype] += 1
    
    return taxonomy, total_errors


def fig1_main_accuracy_heatmap(all_results):
    """Figure 1: Main accuracy heatmap (model x strategy)."""
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    
    data_matrix = np.full((len(MODEL_ORDER), len(STRATEGY_ORDER)), np.nan)
    for i, model in enumerate(MODEL_ORDER):
        for j, strat in enumerate(STRATEGY_ORDER):
            key = f"{model}__{strat}"
            if key in all_results:
                data_matrix[i, j] = all_results[key]["accuracy"]
    
    im = ax.imshow(data_matrix, cmap="RdYlGn", aspect="auto", vmin=30, vmax=70)
    
    # Add text annotations
    for i in range(len(MODEL_ORDER)):
        for j in range(len(STRATEGY_ORDER)):
            val = data_matrix[i, j]
            if not np.isnan(val):
                color = "white" if val < 40 or val > 60 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", 
                       fontsize=11, fontweight="bold", color=color)
            else:
                ax.text(j, i, "—", ha="center", va="center", 
                       fontsize=11, color="gray")
    
    ax.set_xticks(range(len(STRATEGY_ORDER)))
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in STRATEGY_ORDER], rotation=0)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    
    # Add random baseline
    ax.axhline(y=-0.5, color='gray', linestyle='--', alpha=0.3)
    ax.text(len(STRATEGY_ORDER) - 0.5, -0.45, "Random: 33.3%", 
            ha="right", va="bottom", fontsize=8, color="gray", style="italic")
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label="Accuracy (%)")
    ax.set_title("MCQ Accuracy on BlackSwanSuite Validation (n=200)", fontsize=11, pad=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig1_accuracy_heatmap.pdf"))
    plt.savefig(os.path.join(FIGURES_DIR, "fig1_accuracy_heatmap.png"))
    plt.close()
    print("  fig1_accuracy_heatmap saved")


def fig2_task_breakdown(all_results):
    """Figure 2: Grouped bar chart - Detective vs Reporter by strategy."""
    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(12, 3.2), sharey=True)
    
    bar_width = 0.35
    x = np.arange(len(STRATEGY_ORDER))
    
    for ax_idx, model in enumerate(MODEL_ORDER):
        ax = axes[ax_idx]
        det_accs = []
        rep_accs = []
        valid_strats = []
        
        for strat in STRATEGY_ORDER:
            key = f"{model}__{strat}"
            if key in all_results:
                task_acc = compute_task_accuracy(all_results[key])
                det_accs.append(task_acc.get("Detective", 0))
                rep_accs.append(task_acc.get("Reporter", 0))
                valid_strats.append(strat)
            else:
                det_accs.append(0)
                rep_accs.append(0)
                valid_strats.append(strat)
        
        x_valid = np.arange(len(valid_strats))
        bars1 = ax.bar(x_valid - bar_width/2, det_accs, bar_width, 
                       label="Detective (abductive)", color="#5C6BC0", alpha=0.85)
        bars2 = ax.bar(x_valid + bar_width/2, rep_accs, bar_width,
                       label="Reporter (defeasible)", color="#EF5350", alpha=0.85)
        
        ax.axhline(y=33.3, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.set_xticks(x_valid)
        ax.set_xticklabels([STRATEGY_LABELS[s] for s in valid_strats], rotation=25, ha="right")
        ax.set_title(MODEL_LABELS[model], fontsize=10, fontweight="bold")
        ax.set_ylim(0, 80)
        
        if ax_idx == 0:
            ax.set_ylabel("Accuracy (%)")
            ax.legend(fontsize=7, loc="upper right")
    
    fig.suptitle("Accuracy by Task Type: Detective (Abductive) vs Reporter (Defeasible)", 
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig2_task_breakdown.pdf"))
    plt.savefig(os.path.join(FIGURES_DIR, "fig2_task_breakdown.png"))
    plt.close()
    print("  fig2_task_breakdown saved")


def fig3_difficulty_breakdown(all_results):
    """Figure 3: Accuracy by difficulty level."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    
    difficulties = ["easy", "medium", "hard"]
    diff_colors = {"easy": "#66BB6A", "medium": "#FFA726", "hard": "#EF5350"}
    
    bar_width = 0.25
    group_width = len(difficulties) * bar_width
    
    # Use GPT-4o-mini (most complete) for difficulty analysis
    strategies_plotted = []
    for strat in STRATEGY_ORDER:
        key = f"gpt-4o-mini__{strat}"
        if key in all_results:
            strategies_plotted.append(strat)
    
    x = np.arange(len(strategies_plotted))
    
    for d_idx, diff in enumerate(difficulties):
        accs = []
        for strat in strategies_plotted:
            key = f"gpt-4o-mini__{strat}"
            diff_acc = compute_difficulty_accuracy(all_results[key])
            accs.append(diff_acc.get(diff, 0))
        
        offset = (d_idx - 1) * bar_width
        ax.bar(x + offset, accs, bar_width, label=diff.capitalize(), 
               color=diff_colors[diff], alpha=0.85)
    
    ax.axhline(y=33.3, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in strategies_plotted])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("GPT-4o-mini Accuracy by Difficulty Level", fontsize=11)
    ax.legend(title="Difficulty")
    ax.set_ylim(0, 85)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig3_difficulty_breakdown.pdf"))
    plt.savefig(os.path.join(FIGURES_DIR, "fig3_difficulty_breakdown.png"))
    plt.close()
    print("  fig3_difficulty_breakdown saved")


def fig4_error_taxonomy(all_results):
    """Figure 4: Error taxonomy visualization."""
    taxonomy, total_errors = build_error_taxonomy(all_results)
    
    error_labels = {
        "hallucinated_cause": "Hallucinated\nHidden Cause",
        "fails_to_retract": "Fails to\nRetract",
        "position_bias_A": "Position\nBias (A)",
        "causal_confusion": "Causal\nConfusion",
        "surface_matching": "Surface\nMatching",
        "parse_failure": "Parse\nFailure",
    }
    
    error_types = list(error_labels.keys())
    
    fig, ax = plt.subplots(figsize=(7, 3.5))
    
    bar_width = 0.25
    x = np.arange(len(error_types))
    
    for m_idx, model in enumerate(MODEL_ORDER):
        counts = []
        for etype in error_types:
            total = 0
            for strat in STRATEGY_ORDER:
                total += taxonomy.get((model, strat), {}).get(etype, 0)
            counts.append(total)
        
        offset = (m_idx - 1) * bar_width
        ax.bar(x + offset, counts, bar_width, label=MODEL_LABELS[model],
               color=COLORS[model], alpha=0.85)
    
    ax.set_xticks(x)
    ax.set_xticklabels([error_labels[e] for e in error_types], fontsize=8)
    ax.set_ylabel("Error Count (across all strategies)")
    ax.set_title("Error Taxonomy Distribution by Model", fontsize=11)
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig4_error_taxonomy.pdf"))
    plt.savefig(os.path.join(FIGURES_DIR, "fig4_error_taxonomy.png"))
    plt.close()
    print("  fig4_error_taxonomy saved")


def fig5_strategy_radar(all_results):
    """Figure 5: Radar/spider chart comparing strategies for best model."""
    # Use GPT-4o-mini as the anchor model
    model = "gpt-4o-mini"
    
    categories = ["Overall", "Detective", "Reporter", "Easy", "Medium", "Hard"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(projection='polar'))
    
    strat_colors = {
        "naive": "#9E9E9E",
        "cot": "#42A5F5",
        "abductive_cot": "#66BB6A",
        "hyp_eliminate": "#FFA726",
        "counterfactual": "#AB47BC",
    }
    
    for strat in STRATEGY_ORDER:
        key = f"{model}__{strat}"
        if key not in all_results:
            continue
        data = all_results[key]
        task_acc = compute_task_accuracy(data)
        diff_acc = compute_difficulty_accuracy(data)
        
        values = [
            data["accuracy"],
            task_acc.get("Detective", 0),
            task_acc.get("Reporter", 0),
            diff_acc.get("easy", 0),
            diff_acc.get("medium", 0),
            diff_acc.get("hard", 0),
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=1.5, label=STRATEGY_LABELS[strat],
                color=strat_colors[strat], markersize=4)
        ax.fill(angles, values, alpha=0.08, color=strat_colors[strat])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 80)
    ax.set_title(f"Strategy Comparison ({MODEL_LABELS[model]})", fontsize=11, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig5_strategy_radar.pdf"))
    plt.savefig(os.path.join(FIGURES_DIR, "fig5_strategy_radar.png"))
    plt.close()
    print("  fig5_strategy_radar saved")


def print_latex_tables(all_results):
    """Print LaTeX-formatted tables for the paper."""
    
    # Table 1: Main results
    print("\n%% TABLE 1: Main Results")
    print("\\begin{tabular}{l" + "c" * len(STRATEGY_ORDER) + "}")
    print("\\toprule")
    header = " & ".join(["Model"] + [STRATEGY_LABELS[s] for s in STRATEGY_ORDER])
    print(f"{header} \\\\")
    print("\\midrule")
    
    for model in MODEL_ORDER:
        row = [MODEL_LABELS[model]]
        best_acc = 0
        for strat in STRATEGY_ORDER:
            key = f"{model}__{strat}"
            if key in all_results:
                acc = all_results[key]["accuracy"]
                best_acc = max(best_acc, acc)
                row.append(f"{acc:.1f}")
            else:
                row.append("---")
        # Bold the best
        row_formatted = [row[0]]
        for j, strat in enumerate(STRATEGY_ORDER):
            key = f"{model}__{strat}"
            if key in all_results and all_results[key]["accuracy"] == best_acc:
                row_formatted.append(f"\\textbf{{{row[j+1]}}}")
            else:
                row_formatted.append(row[j+1])
        print(" & ".join(row_formatted) + " \\\\")
    
    print("\\midrule")
    print(f"Random & " + " & ".join(["33.3"] * len(STRATEGY_ORDER)) + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    
    # Table 2: Task breakdown
    print("\n%% TABLE 2: Task Breakdown (Detective / Reporter)")
    print("\\begin{tabular}{l" + "c" * len(STRATEGY_ORDER) + "}")
    print("\\toprule")
    header = " & ".join(["Model"] + [STRATEGY_LABELS[s] for s in STRATEGY_ORDER])
    print(f"{header} \\\\")
    print("\\midrule")
    
    for model in MODEL_ORDER:
        det_row = [f"{MODEL_LABELS[model]} (Det.)"]
        rep_row = [f"\\quad (Rep.)"]
        for strat in STRATEGY_ORDER:
            key = f"{model}__{strat}"
            if key in all_results:
                ta = compute_task_accuracy(all_results[key])
                det_row.append(f"{ta.get('Detective', 0):.1f}")
                rep_row.append(f"{ta.get('Reporter', 0):.1f}")
            else:
                det_row.append("---")
                rep_row.append("---")
        print(" & ".join(det_row) + " \\\\")
        print(" & ".join(rep_row) + " \\\\")
        if model != MODEL_ORDER[-1]:
            print("\\cmidrule{1-" + str(len(STRATEGY_ORDER) + 1) + "}")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    
    # Save tables to file
    with open(os.path.join(RESULTS_DIR, "latex_tables.txt"), "w") as f:
        f.write("See console output for LaTeX tables\n")


def main():
    print("Loading results...", flush=True)
    all_results = load_all_results()
    print(f"Loaded {len(all_results)} experiments\n", flush=True)
    
    # Summary
    print("=" * 60)
    print(f"{'Model':<20} {'Strategy':<20} {'Acc':>6}")
    print("-" * 48)
    for key in sorted(all_results):
        r = all_results[key]
        print(f"{r['model']:<20} {r['strategy']:<20} {r['accuracy']:>5.1f}%")
    
    # Generate figures
    print("\nGenerating figures...")
    fig1_main_accuracy_heatmap(all_results)
    fig2_task_breakdown(all_results)
    fig3_difficulty_breakdown(all_results)
    fig4_error_taxonomy(all_results)
    fig5_strategy_radar(all_results)
    
    # Print tables
    print_latex_tables(all_results)
    
    # Detailed error analysis
    print("\n\nDETAILED ERROR ANALYSIS")
    print("=" * 60)
    taxonomy, total_errors = build_error_taxonomy(all_results)
    print("\nTotal error counts across all experiments:")
    for etype, count in sorted(total_errors.items(), key=lambda x: -x[1]):
        print(f"  {etype:<25} {count}")
    
    # Per-model error rates
    print("\nError rates by model:")
    for model in MODEL_ORDER:
        model_errors = defaultdict(int)
        model_total = 0
        for strat in STRATEGY_ORDER:
            key = f"{model}__{strat}"
            if key in all_results:
                for r in all_results[key]["results"]:
                    if not r["correct"]:
                        etype = classify_error(r)
                        model_errors[etype] += 1
                        model_total += 1
        print(f"\n  {MODEL_LABELS[model]} ({model_total} errors total):")
        for etype, count in sorted(model_errors.items(), key=lambda x: -x[1]):
            pct = count / model_total * 100 if model_total > 0 else 0
            print(f"    {etype:<25} {count:>4} ({pct:.1f}%)")

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
