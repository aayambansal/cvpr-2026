#!/usr/bin/env python3
"""
Generate all publication-quality figures for CounterBench paper.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = DATA_DIR / "results"
ANALYSIS_DIR = DATA_DIR / "analysis"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────
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

# Color palette (colorblind-friendly)
COLORS = {
    'gpt4o': '#E69F00',
    'claude35sonnet': '#56B4E9',
    'gemini2flash': '#009E73',
    'llama32_11b': '#CC79A7',
    'qwen25vl72b': '#D55E00',
    'pixtral_large': '#0072B2',
}

MODEL_DISPLAY = {
    'gpt4o': 'GPT-4o',
    'claude35sonnet': 'Claude 3.5\nSonnet',
    'gemini2flash': 'Gemini 2.0\nFlash',
    'llama32_11b': 'Llama 3.2\n11B',
    'qwen25vl72b': 'Qwen2.5-VL\n72B',
    'pixtral_large': 'Pixtral\nLarge',
}

MODEL_DISPLAY_SHORT = {
    'gpt4o': 'GPT-4o',
    'claude35sonnet': 'Claude 3.5',
    'gemini2flash': 'Gemini 2.0',
    'llama32_11b': 'Llama 3.2',
    'qwen25vl72b': 'Qwen2.5-VL',
    'pixtral_large': 'Pixtral',
}


def load_results():
    all_results = {}
    for model_key in ['gpt4o', 'claude35sonnet', 'gemini2flash', 'llama32_11b', 'qwen25vl72b', 'pixtral_large']:
        path = RESULTS_DIR / f"{model_key}_results.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            # Filter out error results
            valid = [r for r in data if 'error' not in r.get('raw_original', '').lower() 
                     and 'error' not in r.get('raw_intervened', '').lower()]
            if len(valid) > 200:
                all_results[model_key] = valid
                print(f"  {model_key}: {len(valid)} valid results")
    return all_results


def compute_metrics(results):
    n = len(results)
    if n == 0:
        return {'ccs': 0, 'orig_accuracy': 0, 'intv_accuracy': 0, 'sensitivity': 0, 'specificity': 0}
    
    orig_correct = sum(1 for r in results if r['pred_original'] == r['gt_original'])
    intv_correct = sum(1 for r in results if r['pred_intervened'] == r['gt_intervened'])
    
    consistent = 0
    tp = fn = tn = fp = 0
    for r in results:
        if r['should_change']:
            if r['pred_original'] != r['pred_intervened'] and r['pred_intervened'] == r['gt_intervened']:
                tp += 1; consistent += 1
            else:
                fn += 1
        else:
            if r['pred_original'] == r['pred_intervened']:
                tn += 1; consistent += 1
            else:
                fp += 1
    
    sc = tp + fn
    snc = tn + fp
    return {
        'ccs': consistent / n,
        'orig_accuracy': orig_correct / n,
        'intv_accuracy': intv_correct / n,
        'sensitivity': tp / sc if sc else 0,
        'specificity': tn / snc if snc else 0,
        'n': n,
    }


# ══════════════════════════════════════════════════════════════════════════
# Figure 1: Main CCS comparison bar chart
# ══════════════════════════════════════════════════════════════════════════
def fig1_main_ccs(all_results):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    
    models = sorted(all_results.keys(), 
                   key=lambda k: compute_metrics(all_results[k])['ccs'], reverse=True)
    
    x = np.arange(len(models))
    ccs_vals = [compute_metrics(all_results[m])['ccs'] for m in models]
    colors = [COLORS.get(m, '#999999') for m in models]
    
    bars = ax.bar(x, ccs_vals, color=colors, edgecolor='black', linewidth=0.5, width=0.6)
    
    # Add value labels on bars
    for bar, val in zip(bars, ccs_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models], fontsize=8)
    ax.set_ylabel('Counterfactual Consistency Score (CCS)')
    ax.set_ylim(0, 1.12)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.savefig(FIG_DIR / 'fig1_main_ccs.pdf')
    fig.savefig(FIG_DIR / 'fig1_main_ccs.png')
    plt.close()
    print("  fig1_main_ccs saved")


# ══════════════════════════════════════════════════════════════════════════
# Figure 2: CCS by category (grouped bar chart)
# ══════════════════════════════════════════════════════════════════════════
def fig2_category_breakdown(all_results):
    categories = ['spatial', 'causal', 'compositional', 'counting', 'occlusion']
    models = sorted(all_results.keys(),
                   key=lambda k: compute_metrics(all_results[k])['ccs'], reverse=True)
    
    fig, ax = plt.subplots(figsize=(8, 3.8))
    
    n_models = len(models)
    n_cats = len(categories)
    bar_width = 0.8 / n_models
    x = np.arange(n_cats)
    
    for i, model in enumerate(models):
        by_cat = defaultdict(list)
        for r in all_results[model]:
            by_cat[r['category']].append(r)
        
        vals = []
        for cat in categories:
            if cat in by_cat:
                m = compute_metrics(by_cat[cat])
                vals.append(m['ccs'])
            else:
                vals.append(0)
        
        offset = (i - n_models/2 + 0.5) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, 
                      color=COLORS.get(model, '#999999'),
                      edgecolor='black', linewidth=0.3,
                      label=MODEL_DISPLAY_SHORT.get(model, model))
    
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in categories])
    ax.set_ylabel('CCS')
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right', ncol=2, framealpha=0.9, fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    
    fig.savefig(FIG_DIR / 'fig2_category_breakdown.pdf')
    fig.savefig(FIG_DIR / 'fig2_category_breakdown.png')
    plt.close()
    print("  fig2_category_breakdown saved")


# ══════════════════════════════════════════════════════════════════════════
# Figure 3: Sensitivity vs Specificity scatter
# ══════════════════════════════════════════════════════════════════════════
def fig3_sens_spec(all_results):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    
    for model in all_results:
        m = compute_metrics(all_results[model])
        ax.scatter(m['specificity'], m['sensitivity'], 
                  s=120, c=COLORS.get(model, '#999999'),
                  edgecolors='black', linewidth=0.8, zorder=5,
                  label=MODEL_DISPLAY_SHORT.get(model, model))
    
    # Perfect consistency point
    ax.scatter(1.0, 1.0, s=80, marker='*', c='gold', edgecolors='black',
              linewidth=0.8, zorder=6, label='Perfect')
    
    # Diagonal and reference lines
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.2, linewidth=0.8)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.2)
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.2)
    
    ax.set_xlabel('Specificity (True No-Change Rate)')
    ax.set_ylabel('Sensitivity (True Change Rate)')
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.1)
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.savefig(FIG_DIR / 'fig3_sens_spec.pdf')
    fig.savefig(FIG_DIR / 'fig3_sens_spec.png')
    plt.close()
    print("  fig3_sens_spec saved")


# ══════════════════════════════════════════════════════════════════════════
# Figure 4: Accuracy vs CCS scatter (are they correlated?)
# ══════════════════════════════════════════════════════════════════════════
def fig4_acc_vs_ccs(all_results):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    
    for model in all_results:
        m = compute_metrics(all_results[model])
        avg_acc = (m['orig_accuracy'] + m['intv_accuracy']) / 2
        ax.scatter(avg_acc, m['ccs'],
                  s=120, c=COLORS.get(model, '#999999'),
                  edgecolors='black', linewidth=0.8, zorder=5,
                  label=MODEL_DISPLAY_SHORT.get(model, model))
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.2, linewidth=0.8)
    ax.set_xlabel('Average Accuracy (Orig + Intv)')
    ax.set_ylabel('CCS')
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=7, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.savefig(FIG_DIR / 'fig4_acc_vs_ccs.pdf')
    fig.savefig(FIG_DIR / 'fig4_acc_vs_ccs.png')
    plt.close()
    print("  fig4_acc_vs_ccs saved")


# ══════════════════════════════════════════════════════════════════════════
# Figure 5: Error type breakdown (stacked bar)
# ══════════════════════════════════════════════════════════════════════════
def fig5_error_types(all_results):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    
    models = sorted(all_results.keys(),
                   key=lambda k: compute_metrics(all_results[k])['ccs'], reverse=True)
    
    error_categories = ['sticky_answer', 'wrong_change', 'spurious_change']
    error_labels = ['Sticky Answer\n(should change, didn\'t)', 
                   'Wrong Change\n(changed incorrectly)',
                   'Spurious Change\n(shouldn\'t change, did)']
    error_colors = ['#D55E00', '#CC79A7', '#0072B2']
    
    x = np.arange(len(models))
    bottom = np.zeros(len(models))
    
    for j, (etype, elabel, ecolor) in enumerate(zip(error_categories, error_labels, error_colors)):
        vals = []
        for model in models:
            count = 0
            for r in all_results[model]:
                pred_changed = r['pred_original'] != r['pred_intervened']
                if etype == 'sticky_answer' and r['should_change'] and not pred_changed:
                    count += 1
                elif etype == 'wrong_change' and r['should_change'] and pred_changed and r['pred_intervened'] != r['gt_intervened']:
                    count += 1
                elif etype == 'spurious_change' and not r['should_change'] and pred_changed:
                    count += 1
            vals.append(count)
        
        ax.bar(x, vals, bottom=bottom, color=ecolor, edgecolor='black', 
               linewidth=0.3, width=0.6, label=elabel)
        bottom += np.array(vals)
    
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models], fontsize=8)
    ax.set_ylabel('Number of Errors')
    ax.legend(fontsize=7, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.savefig(FIG_DIR / 'fig5_error_types.pdf')
    fig.savefig(FIG_DIR / 'fig5_error_types.png')
    plt.close()
    print("  fig5_error_types saved")


# ══════════════════════════════════════════════════════════════════════════
# Figure 6: Heatmap - CCS by subcategory x model
# ══════════════════════════════════════════════════════════════════════════
def fig6_heatmap(all_results):
    subcategories = [
        'left_right', 'above_below', 'inside_outside', 'left_right_no_change',
        'arrow_cause', 'spill_cause', 'causal_chain',
        'attribute_binding', 'relative_composition',
        'count_removal', 'count_no_change',
        'partial_occlusion'
    ]
    subcat_labels = [
        'Left/Right', 'Above/Below', 'Inside/Outside', 'Spatial\n(no change)',
        'Arrow Cause', 'Spill Cause', 'Causal Chain',
        'Attr. Binding', 'Relative\nComposition',
        'Count\n(removal)', 'Count\n(no change)',
        'Occlusion'
    ]
    
    models = sorted(all_results.keys(),
                   key=lambda k: compute_metrics(all_results[k])['ccs'], reverse=True)
    
    matrix = np.zeros((len(models), len(subcategories)))
    
    for i, model in enumerate(models):
        by_subcat = defaultdict(list)
        for r in all_results[model]:
            by_subcat[r['subcategory']].append(r)
        
        for j, sc in enumerate(subcategories):
            if sc in by_subcat:
                m = compute_metrics(by_subcat[sc])
                matrix[i, j] = m['ccs']
    
    fig, ax = plt.subplots(figsize=(9, 3.5))
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(subcategories)))
    ax.set_xticklabels(subcat_labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_DISPLAY_SHORT.get(m, m) for m in models], fontsize=8)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(subcategories)):
            val = matrix[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)
    
    # Category separators
    for pos in [4, 7, 9, 11]:
        ax.axvline(x=pos - 0.5, color='white', linewidth=2)
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('CCS', fontsize=9)
    
    fig.savefig(FIG_DIR / 'fig6_heatmap.pdf')
    fig.savefig(FIG_DIR / 'fig6_heatmap.png')
    plt.close()
    print("  fig6_heatmap saved")


# ══════════════════════════════════════════════════════════════════════════
# Figure 7: Example pairs (qualitative)
# ══════════════════════════════════════════════════════════════════════════
def fig7_examples(all_results):
    """Show 4 example pairs with model predictions."""
    from PIL import Image
    IMG_DIR = DATA_DIR / "images"
    
    # Pick one from each category
    meta_path = DATA_DIR / "counterbench_metadata.json"
    with open(meta_path) as f:
        pairs = json.load(f)
    
    examples = {}
    for p in pairs:
        cat = p['category']
        if cat not in examples:
            examples[cat] = p
    
    selected = [examples.get(c) for c in ['spatial', 'causal', 'compositional', 'counting'] if c in examples]
    
    fig, axes = plt.subplots(len(selected), 2, figsize=(6, 2.2 * len(selected)))
    
    for i, pair in enumerate(selected):
        orig_img = Image.open(IMG_DIR / pair['original_image'])
        intv_img = Image.open(IMG_DIR / pair['intervened_image'])
        
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f"Original", fontsize=8, fontweight='bold')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(intv_img)
        title = f"Intervened ({pair['intervention']})"
        axes[i, 1].set_title(title, fontsize=8, fontweight='bold')
        axes[i, 1].axis('off')
        
        # Add question + GT as text
        q_short = pair['question'][:50] + "..." if len(pair['question']) > 50 else pair['question']
        axes[i, 0].text(0.5, -0.08, f"Q: {q_short}", 
                        transform=axes[i, 0].transAxes, fontsize=6, ha='center')
        axes[i, 0].text(0.5, -0.16, f"GT: {pair['answer_original']}", 
                        transform=axes[i, 0].transAxes, fontsize=6, ha='center', color='green')
        axes[i, 1].text(0.5, -0.08, f"Should change: {pair['should_change']}", 
                        transform=axes[i, 1].transAxes, fontsize=6, ha='center')
        axes[i, 1].text(0.5, -0.16, f"GT: {pair['answer_intervened']}", 
                        transform=axes[i, 1].transAxes, fontsize=6, ha='center', color='green')
    
    plt.tight_layout(h_pad=1.5)
    fig.savefig(FIG_DIR / 'fig7_examples.pdf')
    fig.savefig(FIG_DIR / 'fig7_examples.png')
    plt.close()
    print("  fig7_examples saved")


# ══════════════════════════════════════════════════════════════════════════
# Figure 8: Intervention type radar chart
# ══════════════════════════════════════════════════════════════════════════
def fig8_intervention_radar(all_results):
    interventions = ['swap', 'move', 'remove_arrow', 'replace_color', 
                    'remove_object', 'remove_occluder', 'remove_distractor',
                    'upright_container', 'remove_mediator']
    int_labels = ['Swap', 'Move', 'Remove\nArrow', 'Replace\nColor',
                 'Remove\nObject', 'Remove\nOccluder', 'Remove\nDistractor',
                 'Upright\nContainer', 'Remove\nMediator']
    
    models = sorted(all_results.keys(),
                   key=lambda k: compute_metrics(all_results[k])['ccs'], reverse=True)
    
    fig, ax = plt.subplots(figsize=(7, 3.5))
    
    n_intv = len(interventions)
    bar_width = 0.8 / len(models)
    x = np.arange(n_intv)
    
    for i, model in enumerate(models):
        by_intv = defaultdict(list)
        for r in all_results[model]:
            by_intv[r['intervention']].append(r)
        
        vals = []
        for intv in interventions:
            if intv in by_intv:
                m = compute_metrics(by_intv[intv])
                vals.append(m['ccs'])
            else:
                vals.append(0)
        
        offset = (i - len(models)/2 + 0.5) * bar_width
        ax.bar(x + offset, vals, bar_width,
              color=COLORS.get(model, '#999999'),
              edgecolor='black', linewidth=0.3,
              label=MODEL_DISPLAY_SHORT.get(model, model))
    
    ax.set_xticks(x)
    ax.set_xticklabels(int_labels, fontsize=6, rotation=45, ha='right')
    ax.set_ylabel('CCS')
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=6, loc='upper right', ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    
    fig.savefig(FIG_DIR / 'fig8_intervention_types.pdf')
    fig.savefig(FIG_DIR / 'fig8_intervention_types.png')
    plt.close()
    print("  fig8_intervention_types saved")


def main():
    print("Loading results...")
    all_results = load_results()
    print(f"Generating figures for {len(all_results)} models\n")
    
    fig1_main_ccs(all_results)
    fig2_category_breakdown(all_results)
    fig3_sens_spec(all_results)
    fig4_acc_vs_ccs(all_results)
    fig5_error_types(all_results)
    fig6_heatmap(all_results)
    fig7_examples(all_results)
    fig8_intervention_radar(all_results)
    
    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
