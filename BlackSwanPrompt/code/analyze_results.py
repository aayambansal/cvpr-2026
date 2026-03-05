#!/usr/bin/env python3
"""
Full analysis of BlackSwan Prompt Baseline experiments.
Generates figures and computes statistics for the paper.
"""

import json
import os
import re
import random
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

# Try importing visualization libs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

VAL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--UBC-ViL--BlackSwanSuite-MCQ/"
    "snapshots/2e78b5d715fb8ce2c3c3e365c1c2c1be4ed12fc0/BlackSwanSuite_MCQ_Val.jsonl"
)

# Load validation data
with open(VAL_PATH) as f:
    val_data = {item["q_id"]: item for item in (json.loads(l) for l in f)}

# ---- Collect all results ----
def load_results(suffix="_final.jsonl"):
    all_results = {}
    for fpath in sorted(RESULTS_DIR.glob(f"*{suffix}")):
        fname = fpath.stem.replace("_final", "")
        parts = fname.split("_", 1)
        model = parts[0]
        template = parts[1] if len(parts) > 1 else "unknown"
        
        with open(fpath) as f:
            entries = [json.loads(l) for l in f]
        
        key = (model, template)
        all_results[key] = entries
    return all_results

results = load_results()
print(f"Loaded {len(results)} conditions:")
for (model, template), entries in results.items():
    n = len(entries)
    c = sum(1 for e in entries if e["predicted"] == e["ground_truth"])
    pf = sum(1 for e in entries if e["predicted"] == -1)
    print(f"  {model} x {template}: {n} entries, acc={c/n:.3f}, parse_fail={pf}")

# ---- Compute comprehensive metrics ----
def compute_metrics(entries):
    total = len(entries)
    correct = sum(1 for e in entries if e["predicted"] == e["ground_truth"])
    failures = sum(1 for e in entries if e["predicted"] == -1)
    
    det = [e for e in entries if e["task"] == "Detective"]
    rep = [e for e in entries if e["task"] == "Reporter"]
    det_c = sum(1 for e in det if e["predicted"] == e["ground_truth"])
    rep_c = sum(1 for e in rep if e["predicted"] == e["ground_truth"])
    
    # By difficulty
    diff = {}
    for d in ["easy", "medium", "hard"]:
        d_items = [e for e in entries if e["difficulty"] == d]
        if d_items:
            diff[d] = sum(1 for e in d_items if e["predicted"] == e["ground_truth"]) / len(d_items)
    
    return {
        "overall": correct / total if total else 0,
        "detective": det_c / len(det) if det else 0,
        "reporter": rep_c / len(rep) if rep else 0,
        "easy": diff.get("easy", 0),
        "medium": diff.get("medium", 0),
        "hard": diff.get("hard", 0),
        "failures": failures,
        "total": total,
    }

# Compute metrics for all
metrics = {}
for (model, template), entries in results.items():
    metrics[(model, template)] = compute_metrics(entries)

# Random baseline
random.seed(42)
random_preds = {item["q_id"]: random.randint(0, 2) for item in val_data.values()}
random_entries = [{"predicted": random_preds[qid], "ground_truth": item["mcq_gt_option"],
                   "task": item["task"], "difficulty": item["difficulty"]}
                  for qid, item in val_data.items()]
metrics[("random", "baseline")] = compute_metrics(random_entries)

# Majority baseline (always pick 0)
majority_entries = [{"predicted": 0, "ground_truth": item["mcq_gt_option"],
                     "task": item["task"], "difficulty": item["difficulty"]}
                    for item in val_data.values()]
metrics[("majority", "baseline")] = compute_metrics(majority_entries)

# ---- Print summary table ----
print("\n" + "=" * 110)
print(f"{'Model':<18} {'Template':<18} {'Overall':>8} {'Det':>8} {'Rep':>8} {'Easy':>8} {'Med':>8} {'Hard':>8} {'Fail':>6}")
print("-" * 110)
for (model, template), m in sorted(metrics.items()):
    print(f"{model:<18} {template:<18} {m['overall']:>8.3f} {m['detective']:>8.3f} "
          f"{m['reporter']:>8.3f} {m['easy']:>8.3f} {m['medium']:>8.3f} {m['hard']:>8.3f} "
          f"{m['failures']:>6}")
print("=" * 110)

# ---- Save metrics for paper ----
metrics_serializable = {f"{m}_{t}": v for (m, t), v in metrics.items()}
with open(RESULTS_DIR / "all_metrics.json", "w") as f:
    json.dump(metrics_serializable, f, indent=2)

# ========================================
# FIGURE GENERATION
# ========================================

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
})

# Color scheme
COLORS = {
    'gpt-4o-mini': '#1f77b4',
    'claude-3.5-haiku': '#ff7f0e',
    'gemini-2.0-flash': '#2ca02c',
    'random': '#999999',
    'majority': '#cccccc',
}
TEMPLATE_LABELS = {
    'P1_Direct': 'Direct',
    'P2_CoT': 'CoT',
    'P3_Abductive': 'Abductive',
    'P4_Elimination': 'Elimination',
    'P5_Counterfactual': 'Counterfactual',
    'baseline': 'Baseline',
}

# ---- Figure 1: Main results bar chart (prompt x model) ----
models_available = sorted(set(m for m, t in metrics if m not in ('random', 'majority')))
templates = ['P1_Direct', 'P2_CoT', 'P3_Abductive', 'P4_Elimination', 'P5_Counterfactual']

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
metric_keys = ['overall', 'detective', 'reporter']
titles = ['Overall Accuracy', 'Detective (Abductive)', 'Reporter (Defeasible)']

for ax_idx, (mkey, title) in enumerate(zip(metric_keys, titles)):
    ax = axes[ax_idx]
    x = np.arange(len(templates))
    width = 0.8 / max(len(models_available), 1)
    
    for i, model in enumerate(models_available):
        vals = [metrics.get((model, t), {}).get(mkey, 0) for t in templates]
        offset = (i - len(models_available)/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=model, color=COLORS.get(model, '#666'),
                      edgecolor='white', linewidth=0.5)
    
    # Random baseline line
    rand_val = metrics[("random", "baseline")][mkey]
    ax.axhline(y=rand_val, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Random (33.3%)')
    
    ax.set_xlabel('Prompt Template')
    ax.set_ylabel('Accuracy' if ax_idx == 0 else '')
    ax.set_title(title, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([TEMPLATE_LABELS.get(t, t) for t in templates], rotation=25, ha='right')
    ax.set_ylim(0, 0.85)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(axis='y', alpha=0.3)
    
    if ax_idx == 0:
        ax.legend(loc='upper left', framealpha=0.9)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig1_main_results.pdf", bbox_inches='tight')
plt.savefig(FIGURES_DIR / "fig1_main_results.png", bbox_inches='tight', dpi=300)
print("\nSaved Figure 1: main results")

# ---- Figure 2: Difficulty breakdown ----
fig, ax = plt.subplots(figsize=(8, 5))
diff_levels = ['easy', 'medium', 'hard']
x = np.arange(len(diff_levels))
width = 0.15

for i, template in enumerate(templates):
    vals = []
    for d in diff_levels:
        # Average across models
        model_vals = [metrics.get((m, template), {}).get(d, 0) for m in models_available]
        vals.append(np.mean(model_vals) if model_vals else 0)
    ax.bar(x + i * width, vals, width, label=TEMPLATE_LABELS.get(template, template),
           edgecolor='white', linewidth=0.5)

ax.axhline(y=1/3, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Random')
ax.set_xlabel('Question Difficulty')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy by Difficulty Level (Averaged Across Models)', fontweight='bold')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(['Easy', 'Medium', 'Hard'])
ax.set_ylim(0, 0.85)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig2_difficulty.pdf", bbox_inches='tight')
plt.savefig(FIGURES_DIR / "fig2_difficulty.png", bbox_inches='tight', dpi=300)
print("Saved Figure 2: difficulty breakdown")

# ---- Figure 3: Detective vs Reporter gap ----
fig, ax = plt.subplots(figsize=(7, 5))
for model in models_available:
    det_vals = [metrics.get((model, t), {}).get('detective', 0) for t in templates]
    rep_vals = [metrics.get((model, t), {}).get('reporter', 0) for t in templates]
    gaps = [r - d for d, r in zip(det_vals, rep_vals)]
    ax.plot([TEMPLATE_LABELS.get(t, t) for t in templates], gaps, 'o-',
            color=COLORS.get(model, '#666'), label=model, linewidth=2, markersize=8)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Prompt Template')
ax.set_ylabel('Reporter - Detective Accuracy Gap')
ax.set_title('Task Gap: Reporter vs. Detective Accuracy', fontweight='bold')
ax.legend(loc='best', framealpha=0.9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig3_task_gap.pdf", bbox_inches='tight')
plt.savefig(FIGURES_DIR / "fig3_task_gap.png", bbox_inches='tight', dpi=300)
print("Saved Figure 3: task gap")

# ---- Figure 4: Error taxonomy ----
# Analyze errors for GPT-4o-mini across templates
error_categories = defaultdict(lambda: defaultdict(int))

for template in templates:
    key = ("gpt-4o-mini", template)
    if key not in results:
        continue
    entries = results[key]
    
    for entry in entries:
        if entry["predicted"] == entry["ground_truth"]:
            continue
        if entry["predicted"] == -1:
            error_categories[template]["Parse Failure"] += 1
            continue
        
        # Categorize based on task type and response content
        response = entry.get("response", "") or ""
        task = entry["task"]
        
        # Heuristic error classification
        if task == "Detective":
            # Check if model picked the "safe/expected" option vs the surprising one
            gt_opt = val_data[entry["q_id"]]["mcq_options"][entry["ground_truth"]]
            pred_opt = val_data[entry["q_id"]]["mcq_options"][entry["predicted"]] if entry["predicted"] in (0,1,2) else ""
            
            # Safe/mundane choice = overweights prior
            surprise_words = ['fall', 'crash', 'fail', 'break', 'slip', 'drop', 'hit', 'miss',
                            'lose', 'wrong', 'accident', 'unexpected', 'instead', 'not']
            gt_surprise = sum(1 for w in surprise_words if w in gt_opt.lower())
            pred_surprise = sum(1 for w in surprise_words if w in pred_opt.lower())
            
            if gt_surprise > pred_surprise:
                error_categories[template]["Overweights Prior\n(picks mundane option)"] += 1
            elif "not" in pred_opt.lower() or "fail" in pred_opt.lower():
                error_categories[template]["Hallucinated Cause\n(wrong mechanism)"] += 1
            else:
                error_categories[template]["Plausibility Confusion\n(both seem possible)"] += 1
        else:
            # Reporter task
            gt_opt = val_data[entry["q_id"]]["mcq_options"][entry["ground_truth"]]
            pred_opt = val_data[entry["q_id"]]["mcq_options"][entry["predicted"]] if entry["predicted"] in (0,1,2) else ""
            
            # Check similarity between pred and gt
            gt_words = set(gt_opt.lower().split())
            pred_words = set(pred_opt.lower().split())
            overlap = len(gt_words & pred_words) / max(len(gt_words | pred_words), 1)
            
            if overlap > 0.5:
                error_categories[template]["Fails to Retract\n(close but wrong detail)"] += 1
            else:
                error_categories[template]["Overweights Prior Text\n(ignores new evidence)"] += 1

# Plot error taxonomy
fig, ax = plt.subplots(figsize=(10, 5))
error_types = sorted(set(et for cats in error_categories.values() for et in cats))
x = np.arange(len(templates))
width = 0.8 / max(len(error_types), 1)

colors_err = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
for i, et in enumerate(error_types):
    vals = [error_categories.get(t, {}).get(et, 0) for t in templates]
    offset = (i - len(error_types)/2 + 0.5) * width
    ax.bar(x + offset, vals, width, label=et, color=colors_err[i % len(colors_err)],
           edgecolor='white', linewidth=0.5)

ax.set_xlabel('Prompt Template')
ax.set_ylabel('Number of Errors')
ax.set_title('Error Taxonomy Distribution Across Prompt Templates (GPT-4o-mini)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([TEMPLATE_LABELS.get(t, t) for t in templates], rotation=15, ha='right')
ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig4_error_taxonomy.pdf", bbox_inches='tight')
plt.savefig(FIGURES_DIR / "fig4_error_taxonomy.png", bbox_inches='tight', dpi=300)
print("Saved Figure 4: error taxonomy")

# ---- Figure 5: Heatmap of results ----
fig, ax = plt.subplots(figsize=(8, 4))
all_models = models_available + ['random']
heatmap_data = []
for model in all_models:
    row = []
    for t in templates:
        if model == 'random':
            row.append(metrics[("random", "baseline")]["overall"])
        else:
            row.append(metrics.get((model, t), {}).get("overall", 0))
    heatmap_data.append(row)

heatmap_data = np.array(heatmap_data)
im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0.25, vmax=0.65)

ax.set_xticks(range(len(templates)))
ax.set_xticklabels([TEMPLATE_LABELS.get(t, t) for t in templates], rotation=25, ha='right')
ax.set_yticks(range(len(all_models)))
ax.set_yticklabels(all_models)

for i in range(len(all_models)):
    for j in range(len(templates)):
        val = heatmap_data[i, j]
        color = 'white' if val > 0.5 else 'black'
        ax.text(j, i, f'{val:.1%}', ha='center', va='center', fontsize=9, color=color, fontweight='bold')

ax.set_title('Accuracy Heatmap: Model × Prompt Template', fontweight='bold')
plt.colorbar(im, ax=ax, label='Accuracy', format=mtick.PercentFormatter(1.0))

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig5_heatmap.pdf", bbox_inches='tight')
plt.savefig(FIGURES_DIR / "fig5_heatmap.png", bbox_inches='tight', dpi=300)
print("Saved Figure 5: heatmap")

# ---- Statistical significance (McNemar's test) ----
print("\n=== Statistical Comparisons (GPT-4o-mini) ===")
# Compare P2_CoT (best) vs P1_Direct (baseline)
if ("gpt-4o-mini", "P1_Direct") in results and ("gpt-4o-mini", "P2_CoT") in results:
    p1 = {e["q_id"]: e["predicted"] == e["ground_truth"] for e in results[("gpt-4o-mini", "P1_Direct")]}
    p2 = {e["q_id"]: e["predicted"] == e["ground_truth"] for e in results[("gpt-4o-mini", "P2_CoT")]}
    
    common_ids = set(p1.keys()) & set(p2.keys())
    # McNemar contingency
    b = sum(1 for qid in common_ids if p1[qid] and not p2[qid])  # P1 right, P2 wrong
    c = sum(1 for qid in common_ids if not p1[qid] and p2[qid])  # P1 wrong, P2 right
    a = sum(1 for qid in common_ids if p1[qid] and p2[qid])
    d = sum(1 for qid in common_ids if not p1[qid] and not p2[qid])
    
    print(f"P1_Direct vs P2_CoT:")
    print(f"  Both correct: {a}, Both wrong: {d}")
    print(f"  P1 right P2 wrong: {b}, P1 wrong P2 right: {c}")
    if (b + c) > 0:
        mcnemar_stat = (abs(b - c) - 1) ** 2 / (b + c)
        print(f"  McNemar chi2={mcnemar_stat:.2f} (b={b}, c={c})")
        # rough p-value from chi2 distribution
        from math import exp
        p_approx = exp(-mcnemar_stat / 2)
        print(f"  Approximate p-value: {p_approx:.4f}")

# ---- Summary stats for paper ----
print("\n=== Key Numbers for Paper ===")
best_overall = max((v["overall"], k) for k, v in metrics.items() if k[0] not in ("random", "majority"))
worst_overall = min((v["overall"], k) for k, v in metrics.items() if k[0] not in ("random", "majority"))
print(f"Best overall: {best_overall[1]} = {best_overall[0]:.1%}")
print(f"Worst overall: {worst_overall[1]} = {worst_overall[0]:.1%}")
print(f"Random baseline: {metrics[('random', 'baseline')]['overall']:.1%}")

# Best for Detective vs Reporter
for task_key in ["detective", "reporter"]:
    best = max((v[task_key], k) for k, v in metrics.items() if k[0] not in ("random", "majority"))
    print(f"Best {task_key}: {best[1]} = {best[0]:.1%}")

# Biggest gap between detective and reporter
for model in models_available:
    for template in templates:
        m = metrics.get((model, template), {})
        gap = m.get("reporter", 0) - m.get("detective", 0)
        if abs(gap) > 0.15:
            print(f"  Large gap: {model} {template}: Rep-Det = {gap:+.1%}")

print("\nDone! All figures saved to", FIGURES_DIR)
