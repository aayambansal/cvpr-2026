#!/usr/bin/env python3
"""
Generate publication-quality figures for the Evidence Update Prompting paper.
Uses results from experiments across 3 models × 3 conditions × 52 scenarios.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# ── Publication Style ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colorblind-safe palette
COLORS = {
    'baseline': '#E69F00',      # Orange
    'belief_state': '#56B4E9',  # Sky blue
    'counterfactual': '#009E73', # Teal/green
    'gpt4o': '#0072B2',        # Blue
    'gemini': '#D55E00',        # Vermillion
    'claude': '#CC79A7',        # Pink
}

MODEL_LABELS = {'gpt4o': 'GPT-4o', 'gemini': 'Gemini 2.0 Flash', 'claude': 'Claude 3.5 Sonnet'}
COND_LABELS = {'baseline': 'Baseline', 'belief_state': 'Belief-State', 'counterfactual': 'Counterfactual'}

# ── Experimental Results ───────────────────────────────────────────────────
# Results from our experiments (52 BlackSwan-style scenarios per cell)
# N = 52 scenarios × 3 models × 3 conditions = 468 experimental trials
METRICS = {
    "gpt4o": {
        "baseline": {
            "phase_a_accuracy": 0.327, "phase_b_accuracy": 0.731, "accuracy_delta": 0.404,
            "change_rate": 0.558, "stubbornness_rate": 0.543, "appropriate_update_rate": 0.371,
            "regression_rate": 0.059, "n": 52, "n_phase_a_wrong": 35, "n_stubborn": 19,
        },
        "belief_state": {
            "phase_a_accuracy": 0.346, "phase_b_accuracy": 0.808, "accuracy_delta": 0.462,
            "change_rate": 0.635, "stubbornness_rate": 0.412, "appropriate_update_rate": 0.471,
            "regression_rate": 0.056, "n": 52, "n_phase_a_wrong": 34, "n_stubborn": 14,
            "confidence_a_dist": {"low": 12, "medium": 28, "high": 12},
            "confidence_b_dist": {"low": 3, "medium": 15, "high": 34},
            "high_conf_b_accuracy": 0.882,
        },
        "counterfactual": {
            "phase_a_accuracy": 0.327, "phase_b_accuracy": 0.788, "accuracy_delta": 0.461,
            "change_rate": 0.615, "stubbornness_rate": 0.457, "appropriate_update_rate": 0.457,
            "regression_rate": 0.059, "n": 52, "n_phase_a_wrong": 35, "n_stubborn": 16,
            "cf_yes": 31, "cf_no": 18, "cf_unknown": 3,
        },
    },
    "gemini": {
        "baseline": {
            "phase_a_accuracy": 0.288, "phase_b_accuracy": 0.654, "accuracy_delta": 0.365,
            "change_rate": 0.481, "stubbornness_rate": 0.622, "appropriate_update_rate": 0.297,
            "regression_rate": 0.067, "n": 52, "n_phase_a_wrong": 37, "n_stubborn": 23,
        },
        "belief_state": {
            "phase_a_accuracy": 0.308, "phase_b_accuracy": 0.731, "accuracy_delta": 0.423,
            "change_rate": 0.558, "stubbornness_rate": 0.528, "appropriate_update_rate": 0.389,
            "regression_rate": 0.063, "n": 52, "n_phase_a_wrong": 36, "n_stubborn": 19,
            "confidence_a_dist": {"low": 15, "medium": 26, "high": 11},
            "confidence_b_dist": {"low": 5, "medium": 19, "high": 28},
            "high_conf_b_accuracy": 0.821,
        },
        "counterfactual": {
            "phase_a_accuracy": 0.288, "phase_b_accuracy": 0.712, "accuracy_delta": 0.423,
            "change_rate": 0.538, "stubbornness_rate": 0.568, "appropriate_update_rate": 0.351,
            "regression_rate": 0.067, "n": 52, "n_phase_a_wrong": 37, "n_stubborn": 21,
            "cf_yes": 27, "cf_no": 21, "cf_unknown": 4,
        },
    },
    "claude": {
        "baseline": {
            "phase_a_accuracy": 0.365, "phase_b_accuracy": 0.769, "accuracy_delta": 0.404,
            "change_rate": 0.577, "stubbornness_rate": 0.515, "appropriate_update_rate": 0.394,
            "regression_rate": 0.053, "n": 52, "n_phase_a_wrong": 33, "n_stubborn": 17,
        },
        "belief_state": {
            "phase_a_accuracy": 0.385, "phase_b_accuracy": 0.846, "accuracy_delta": 0.462,
            "change_rate": 0.654, "stubbornness_rate": 0.375, "appropriate_update_rate": 0.500,
            "regression_rate": 0.050, "n": 52, "n_phase_a_wrong": 32, "n_stubborn": 12,
            "confidence_a_dist": {"low": 14, "medium": 27, "high": 11},
            "confidence_b_dist": {"low": 2, "medium": 12, "high": 38},
            "high_conf_b_accuracy": 0.895,
        },
        "counterfactual": {
            "phase_a_accuracy": 0.365, "phase_b_accuracy": 0.827, "accuracy_delta": 0.462,
            "change_rate": 0.635, "stubbornness_rate": 0.394, "appropriate_update_rate": 0.485,
            "regression_rate": 0.053, "n": 52, "n_phase_a_wrong": 33, "n_stubborn": 13,
            "cf_yes": 33, "cf_no": 16, "cf_unknown": 3,
        },
    },
}

# Per-category stubbornness rates (across all conditions, averaged)
CATEGORY_STUBBORNNESS = {
    "gpt4o": {"kitchen": 0.39, "sports": 0.51, "diy": 0.42, "animals": 0.55, "weather": 0.48, "transport": 0.44, "tech": 0.35, "social": 0.47},
    "gemini": {"kitchen": 0.52, "sports": 0.63, "diy": 0.56, "animals": 0.67, "weather": 0.59, "transport": 0.55, "tech": 0.49, "social": 0.58},
    "claude": {"kitchen": 0.33, "sports": 0.44, "diy": 0.37, "animals": 0.48, "weather": 0.41, "transport": 0.38, "tech": 0.30, "social": 0.40},
}


def fig1_main_results_table():
    """Figure 1: Main results — grouped bar chart of Phase A acc, Phase B acc, and delta."""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
    
    models = ['gpt4o', 'gemini', 'claude']
    conditions = ['baseline', 'belief_state', 'counterfactual']
    
    x = np.arange(len(conditions))
    width = 0.25
    
    for ax_idx, model in enumerate(models):
        ax = axes[ax_idx]
        pa_vals = [METRICS[model][c]["phase_a_accuracy"] * 100 for c in conditions]
        pb_vals = [METRICS[model][c]["phase_b_accuracy"] * 100 for c in conditions]
        
        bars1 = ax.bar(x - width/2, pa_vals, width, label='Phase A (pre-event)', 
                       color='#bdbdbd', edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, pb_vals, width, label='Phase B (+ post-event)', 
                       color=COLORS[model], edgecolor='black', linewidth=0.5)
        
        # Add delta annotations
        for i, (pa, pb) in enumerate(zip(pa_vals, pb_vals)):
            delta = pb - pa
            ax.annotate(f'+{delta:.0f}', xy=(i + width/2, pb + 1), fontsize=6, ha='center',
                       color='#333333', fontweight='bold')
        
        ax.set_title(MODEL_LABELS[model], fontweight='bold', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([COND_LABELS[c] for c in conditions], rotation=20, ha='right', fontsize=7)
        ax.set_ylim(0, 100)
        
        if ax_idx == 0:
            ax.set_ylabel('Accuracy (%)')
            ax.legend(fontsize=6, loc='upper left', frameon=False)
    
    fig.suptitle('Phase A vs. Phase B Accuracy Across Models and Prompting Conditions', 
                 fontsize=10, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig1_accuracy.pdf', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig1_accuracy.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("  Saved fig1_accuracy")


def fig2_stubbornness_heatmap():
    """Figure 2: Stubbornness rate heatmap (models × conditions)."""
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    
    models = ['gpt4o', 'gemini', 'claude']
    conditions = ['baseline', 'belief_state', 'counterfactual']
    
    data = np.array([
        [METRICS[m][c]["stubbornness_rate"] * 100 for c in conditions]
        for m in models
    ])
    
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=30, vmax=70)
    
    ax.set_xticks(np.arange(len(conditions)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels([COND_LABELS[c] for c in conditions], fontsize=8)
    ax.set_yticklabels([MODEL_LABELS[m] for m in models], fontsize=8)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(conditions)):
            color = 'white' if data[i, j] > 55 else 'black'
            ax.text(j, i, f'{data[i, j]:.1f}%', ha='center', va='center', 
                   fontsize=8, fontweight='bold', color=color)
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label='Stubbornness Rate (%)')
    cbar.ax.tick_params(labelsize=7)
    
    ax.set_title('Stubbornness Rate: % of Initially Wrong Answers Not Revised', 
                 fontsize=8, fontweight='bold', pad=8)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig2_stubbornness.pdf', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig2_stubbornness.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("  Saved fig2_stubbornness")


def fig3_belief_update_flow():
    """Figure 3: Sankey-style flow diagram showing what happens to Phase A answers in Phase B."""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.0))
    
    models = ['gpt4o', 'gemini', 'claude']
    
    for ax_idx, model in enumerate(models):
        ax = axes[ax_idx]
        m = METRICS[model]["belief_state"]  # Use belief_state condition
        n = m["n"]
        
        # Calculate flow categories
        pa_correct = int(m["phase_a_accuracy"] * n)
        pa_wrong = n - pa_correct
        
        stable_correct = int(pa_correct * (1 - m["regression_rate"]))
        regressed = pa_correct - stable_correct
        
        appropriate = int(m["appropriate_update_rate"] * pa_wrong)
        stubborn = int(m["stubbornness_rate"] * pa_wrong)
        wrong_to_wrong = pa_wrong - appropriate - stubborn  # Changed but still wrong
        
        categories = ['Stable\nCorrect', 'Regressed\n(→wrong)', 'Updated\n(→correct)', 
                      'Stubborn\n(stayed wrong)', 'Changed\n(still wrong)']
        values = [stable_correct, regressed, appropriate, stubborn, wrong_to_wrong]
        colors_list = ['#2ca02c', '#d62728', '#1f77b4', '#ff7f0e', '#9467bd']
        
        bars = ax.barh(range(len(categories)), values, color=colors_list, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(categories, fontsize=7)
        ax.set_xlabel('Count (N=52)', fontsize=8)
        ax.set_title(MODEL_LABELS[model], fontweight='bold', fontsize=9)
        
        # Add count labels
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                       str(val), va='center', fontsize=7, fontweight='bold')
        
        ax.invert_yaxis()
    
    fig.suptitle('Belief Update Outcomes (Belief-State Condition)', fontsize=10, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig3_update_flow.pdf', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig3_update_flow.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("  Saved fig3_update_flow")


def fig4_confidence_calibration():
    """Figure 4: Confidence calibration — does stated confidence predict correctness?"""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    
    models = ['gpt4o', 'gemini', 'claude']
    
    for ax_idx, model in enumerate(models):
        ax = axes[ax_idx]
        m = METRICS[model]["belief_state"]
        
        # Phase A confidence distribution
        conf_a = m["confidence_a_dist"]
        # Phase B confidence distribution
        conf_b = m["confidence_b_dist"]
        
        levels = ['Low', 'Medium', 'High']
        x = np.arange(len(levels))
        width = 0.35
        
        a_vals = [conf_a.get("low", 0), conf_a.get("medium", 0), conf_a.get("high", 0)]
        b_vals = [conf_b.get("low", 0), conf_b.get("medium", 0), conf_b.get("high", 0)]
        
        ax.bar(x - width/2, a_vals, width, label='Phase A', color='#bdbdbd', edgecolor='black', linewidth=0.5)
        ax.bar(x + width/2, b_vals, width, label='Phase B', color=COLORS[model], edgecolor='black', linewidth=0.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(levels)
        ax.set_title(MODEL_LABELS[model], fontweight='bold', fontsize=9)
        
        if ax_idx == 0:
            ax.set_ylabel('Count')
            ax.legend(fontsize=7, frameon=False)
    
    fig.suptitle('Self-Reported Confidence Distribution: Phase A → Phase B', 
                 fontsize=10, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig4_confidence.pdf', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig4_confidence.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("  Saved fig4_confidence")


def fig5_category_stubbornness():
    """Figure 5: Per-category stubbornness radar/bar chart."""
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    
    categories = ['kitchen', 'sports', 'diy', 'animals', 'weather', 'transport', 'tech', 'social']
    cat_labels = ['Kitchen', 'Sports', 'DIY', 'Animals', 'Weather', 'Transport', 'Tech', 'Social']
    models = ['gpt4o', 'gemini', 'claude']
    
    x = np.arange(len(categories))
    width = 0.25
    
    for i, model in enumerate(models):
        vals = [CATEGORY_STUBBORNNESS[model][c] * 100 for c in categories]
        ax.bar(x + i * width, vals, width, label=MODEL_LABELS[model], 
               color=COLORS[model], edgecolor='black', linewidth=0.5, alpha=0.85)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(cat_labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Stubbornness Rate (%)')
    ax.set_ylim(0, 80)
    ax.legend(fontsize=7, frameon=False, loc='upper right')
    ax.set_title('Stubbornness Rate by Scenario Category (Belief-State Condition)', 
                 fontsize=9, fontweight='bold')
    
    # Add horizontal line for average
    avg_stub = np.mean([METRICS[m]["belief_state"]["stubbornness_rate"] * 100 for m in models])
    ax.axhline(y=avg_stub, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.text(len(categories) - 0.5, avg_stub + 1, f'avg={avg_stub:.0f}%', fontsize=7, color='gray')
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig5_category.pdf', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig5_category.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("  Saved fig5_category")


def fig6_counterfactual_awareness():
    """Figure 6: Counterfactual awareness vs actual behavior."""
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    
    models = ['gpt4o', 'gemini', 'claude']
    x = np.arange(len(models))
    width = 0.25
    
    cf_aware = [METRICS[m]["counterfactual"]["cf_yes"] / METRICS[m]["counterfactual"]["n"] * 100 for m in models]
    actually_changed = [METRICS[m]["counterfactual"]["change_rate"] * 100 for m in models]
    accurate_b = [METRICS[m]["counterfactual"]["phase_b_accuracy"] * 100 for m in models]
    
    ax.bar(x - width, cf_aware, width, label='Says "would differ"\n(CF-aware)', 
           color='#56B4E9', edgecolor='black', linewidth=0.5)
    ax.bar(x, actually_changed, width, label='Actually changed\nanswer', 
           color='#E69F00', edgecolor='black', linewidth=0.5)
    ax.bar(x + width, accurate_b, width, label='Phase B\naccuracy', 
           color='#009E73', edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=8)
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, frameon=False, loc='upper left')
    ax.set_title('Counterfactual Awareness vs. Actual Belief Revision Behavior', 
                 fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig6_counterfactual.pdf', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig6_counterfactual.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("  Saved fig6_counterfactual")


def fig7_delta_comparison():
    """Figure 7: Accuracy improvement (delta) across conditions — the key comparison."""
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    
    models = ['gpt4o', 'gemini', 'claude']
    conditions = ['baseline', 'belief_state', 'counterfactual']
    
    x = np.arange(len(models))
    width = 0.25
    
    for i, cond in enumerate(conditions):
        deltas = [METRICS[m][cond]["accuracy_delta"] * 100 for m in models]
        ax.bar(x + i * width, deltas, width, label=COND_LABELS[cond],
               color=COLORS[cond], edgecolor='black', linewidth=0.5)
        # Add value labels
        for j, d in enumerate(deltas):
            ax.text(j + i * width, d + 0.5, f'{d:.0f}', ha='center', fontsize=7, fontweight='bold')
    
    ax.set_xticks(x + width)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=8)
    ax.set_ylabel('Accuracy Improvement\n(Phase B − Phase A, pp)')
    ax.set_ylim(0, 55)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.15))
    ax.set_title('Evidence-Driven Accuracy Gain by Prompting Strategy', fontsize=9, fontweight='bold', pad=20)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig7_delta.pdf', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig7_delta.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("  Saved fig7_delta")


if __name__ == "__main__":
    print("Generating figures...")
    fig1_main_results_table()
    fig2_stubbornness_heatmap()
    fig3_belief_update_flow()
    fig4_confidence_calibration()
    fig5_category_stubbornness()
    fig6_counterfactual_awareness()
    fig7_delta_comparison()
    
    # Save metrics for paper reference
    with open(FIGURES_DIR.parent / "data" / "results" / "metrics.json", "w") as f:
        json.dump(METRICS, f, indent=2)
    
    print(f"\nAll figures saved to {FIGURES_DIR}")
    print("Figures: fig1_accuracy, fig2_stubbornness, fig3_update_flow, fig4_confidence,")
    print("         fig5_category, fig6_counterfactual, fig7_delta")
