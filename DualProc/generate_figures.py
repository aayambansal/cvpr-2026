#!/usr/bin/env python3
"""
Generate all publication-quality figures for the DualProc paper.
Produces 8 figures in PDF format for LaTeX inclusion.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
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
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.8,
})

# Colorblind-safe palette (Okabe-Ito)
COLORS = {
    'baseline': '#E69F00',     # Orange
    'cot': '#56B4E9',          # Sky blue
    'dual_process': '#009E73', # Bluish green
    'deliberate_only': '#CC79A7',  # Reddish purple
}

MODEL_SHORT = {
    'GPT-4o-mini': 'GPT-4o-mini',
    'Gemini-2.0-Flash': 'Gemini Flash',
    'Claude-3.5-Sonnet': 'Claude 3.5',
}

CONDITION_LABELS = {
    'baseline': 'Direct',
    'cot': 'CoT',
    'dual_process': 'DualProc',
    'deliberate_only': 'Delib. Only',
}

# Load experiment results
script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_dir, 'experiment_results.json'), 'r') as f:
    data = json.load(f)

fig_dir = os.path.join(script_dir, 'figures')
os.makedirs(fig_dir, exist_ok=True)

models = data['metadata']['models']
conditions = data['metadata']['conditions']
categories = data['metadata']['categories']
cat_labels = data['metadata']['category_labels']


# ============================================================================
# Figure 1: Method Overview (conceptual diagram)
# ============================================================================
def fig1_method_overview():
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.5)
    ax.axis('off')

    # Stage boxes
    boxes = [
        (0.3, 1.2, 2.4, 1.6, '#FFF3CD', 'Stage 1: System 1\n(Fast Guess)',
         'Quick answer\n+ confidence $c_1$'),
        (3.5, 1.2, 2.4, 1.6, '#D4EDDA', 'Stage 2: Deliberation',
         '3 alternatives\n+ evidence check'),
        (6.8, 1.2, 2.4, 1.6, '#CCE5FF', 'Stage 3: System 2\n(Final Answer)',
         'Revised answer\n+ confidence $c_2$'),
    ]

    for x, y, w, h, color, title, subtitle in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1",
            facecolor=color, edgecolor='#333333', linewidth=1.2
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.68, title, ha='center', va='center',
                fontsize=9, fontweight='bold', color='#222222')
        ax.text(x + w/2, y + h*0.28, subtitle, ha='center', va='center',
                fontsize=7.5, color='#555555', style='italic')

    # Arrows
    arrow_style = dict(arrowstyle='->', color='#555555', lw=1.5,
                       connectionstyle='arc3,rad=0.0')
    ax.annotate('', xy=(3.4, 2.0), xytext=(2.8, 2.0), arrowprops=arrow_style)
    ax.annotate('', xy=(6.7, 2.0), xytext=(6.0, 2.0), arrowprops=arrow_style)

    # Image icon on the left
    ax.text(0.15, 2.9, '🖼', fontsize=14, ha='center', va='center')
    ax.text(0.15, 2.4, 'Image +\nQuestion', fontsize=7, ha='center', va='center', color='#555555')
    ax.annotate('', xy=(0.3, 2.0), xytext=(0.15, 2.2),
                arrowprops=dict(arrowstyle='->', color='#888', lw=1))

    # Key insight box at top
    ax.text(5.0, 3.2, 'DualProc: Deliberation selectively targets confident errors, '
            'improving calibration without sacrificing accuracy',
            ha='center', va='center', fontsize=8, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor='#009E73', linewidth=1))

    # Confidence annotation
    ax.text(1.5, 0.85, '$c_1 = 0.85$', ha='center', fontsize=8, color='#B8860B')
    ax.text(8.0, 0.85, '$c_2 = 0.62$', ha='center', fontsize=8, color='#2171B5')
    ax.text(4.7, 0.85, 'if wrong → revise ↓conf', ha='center', fontsize=7, color='#666')

    fig.savefig(os.path.join(fig_dir, 'fig1_method_overview.pdf'))
    plt.close()
    print("  ✓ fig1_method_overview.pdf")


# ============================================================================
# Figure 2: Main results bar chart (accuracy + confidence gap + confident errors)
# ============================================================================
def fig2_main_results():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))

    x = np.arange(len(models))
    width = 0.18

    metrics_list = [
        ('accuracy', 'Accuracy', (0.5, 0.9)),
        ('confidence_gap', 'Confidence Gap\n(conf - acc)', (-0.1, 0.15)),
        ('confident_error_rate', 'Confident Error Rate\n(wrong + conf>0.75)', (0, 0.18)),
    ]

    for ax_idx, (metric, ylabel, ylim) in enumerate(metrics_list):
        ax = axes[ax_idx]
        for c_idx, cond in enumerate(conditions):
            vals = [data['results'][m][cond][metric] for m in models]
            offset = (c_idx - 1.5) * width
            bars = ax.bar(x + offset, vals, width, label=CONDITION_LABELS[cond],
                         color=COLORS[cond], edgecolor='white', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_SHORT[m] for m in models], fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_ylim(ylim)
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if metric == 'confidence_gap':
            ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')

    axes[0].legend(loc='lower left', fontsize=6.5, ncol=2, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig2_main_results.pdf'))
    plt.close()
    print("  ✓ fig2_main_results.pdf")


# ============================================================================
# Figure 3: Reliability diagrams (calibration plots) for each model
# ============================================================================
def fig3_calibration():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))

    for m_idx, model_name in enumerate(models):
        ax = axes[m_idx]

        for cond in ['baseline', 'cot', 'dual_process']:
            bins = data['results'][model_name][cond]['ece_bins']
            confs = [b['avg_conf'] for b in bins if b['count'] > 0]
            accs = [b['avg_acc'] for b in bins if b['count'] > 0]
            ax.plot(confs, accs, 'o-', color=COLORS[cond], label=CONDITION_LABELS[cond],
                   markersize=4, linewidth=1.2)

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5, label='Perfect')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Confidence', fontsize=8)
        ax.set_aspect('equal')
        ax.set_title(MODEL_SHORT[model_name], fontsize=9, fontweight='bold')
        ax.grid(alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if m_idx == 0:
            ax.set_ylabel('Accuracy', fontsize=8)
            ax.legend(fontsize=6, loc='upper left')

        # ECE annotation
        ece_base = data['results'][model_name]['baseline']['ece']
        ece_dual = data['results'][model_name]['dual_process']['ece']
        gap_base = data['results'][model_name]['baseline']['confidence_gap']
        gap_dual = data['results'][model_name]['dual_process']['confidence_gap']
        ax.text(0.95, 0.05, f'Gap: {gap_base:.2f}→{gap_dual:.2f}',
               transform=ax.transAxes, fontsize=6.5, ha='right',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='#f0f0f0', alpha=0.8))

    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig3_calibration.pdf'))
    plt.close()
    print("  ✓ fig3_calibration.pdf")


# ============================================================================
# Figure 4: Flip analysis (sankey-style grouped bars)
# ============================================================================
def fig4_flip_analysis():
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.5))

    x = np.arange(len(models))
    width = 0.3

    flip_correct = [data['results'][m]['flip_analysis']['flip_to_correct_pct'] for m in models]
    flip_wrong = [data['results'][m]['flip_analysis']['flip_to_wrong_pct'] for m in models]

    bars1 = ax.bar(x - width/2, flip_correct, width, label='Flip to Correct ↑',
                   color='#2CA02C', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, flip_wrong, width, label='Flip to Wrong ↓',
                   color='#D62728', edgecolor='white', linewidth=0.5)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
               f'{bar.get_height():.1f}%', ha='center', fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
               f'{bar.get_height():.1f}%', ha='center', fontsize=7)

    # Net improvement annotation
    for i, m in enumerate(models):
        net = data['results'][m]['flip_analysis']['net_flip_pct']
        ratio = data['results'][m]['flip_analysis']['flip_ratio']
        ax.text(i, max(flip_correct[i], flip_wrong[i]) + 2.5,
               f'Net: +{net:.1f}%\n({ratio:.1f}× ratio)',
               ha='center', fontsize=6.5, fontweight='bold', color='#009E73')

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in models], fontsize=8)
    ax.set_ylabel('Items Flipped (%)', fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig4_flip_analysis.pdf'))
    plt.close()
    print("  ✓ fig4_flip_analysis.pdf")


# ============================================================================
# Figure 5: Per-category accuracy heatmap
# ============================================================================
def fig5_category_heatmap():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))

    for m_idx, model_name in enumerate(models):
        ax = axes[m_idx]

        # Build matrix: conditions × categories
        matrix = np.zeros((len(conditions), len(categories)))
        for c_idx, cond in enumerate(conditions):
            for cat_idx, cat in enumerate(categories):
                matrix[c_idx, cat_idx] = data['results'][model_name][cond]['per_category'][cat]['accuracy']

        im = ax.imshow(matrix, cmap='RdYlGn', vmin=0.35, vmax=0.95, aspect='auto')

        # Annotate cells
        for c_idx in range(len(conditions)):
            for cat_idx in range(len(categories)):
                val = matrix[c_idx, cat_idx]
                color = 'white' if val < 0.5 or val > 0.85 else 'black'
                ax.text(cat_idx, c_idx, f'{val:.2f}', ha='center', va='center',
                       fontsize=6.5, color=color)

        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(cat_labels, fontsize=6.5, rotation=30, ha='right')
        ax.set_yticks(range(len(conditions)))
        if m_idx == 0:
            ax.set_yticklabels([CONDITION_LABELS[c] for c in conditions], fontsize=7)
        else:
            ax.set_yticklabels([])
        ax.set_title(MODEL_SHORT[model_name], fontsize=9, fontweight='bold')

    fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label='Accuracy', pad=0.02)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig5_category_heatmap.pdf'))
    plt.close()
    print("  ✓ fig5_category_heatmap.pdf")


# ============================================================================
# Figure 6: Confidence distributions (violin/box comparison)
# ============================================================================
def fig6_confidence_distributions():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))

    for m_idx, model_name in enumerate(models):
        ax = axes[m_idx]
        cd = data['conf_distribution_analysis'][model_name]

        # Four bars: baseline correct, baseline wrong, dual correct, dual wrong
        positions = [0, 1, 2.5, 3.5]
        means = [cd['base_conf_correct_mean'], cd['base_conf_wrong_mean'],
                 cd['dual_conf_correct_mean'], cd['dual_conf_wrong_mean']]
        stds = [cd['base_conf_correct_std'], cd['base_conf_wrong_std'],
                cd['dual_conf_correct_std'], cd['dual_conf_wrong_std']]
        colors_local = ['#2CA02C', '#D62728', '#009E73', '#CC79A7']
        labels_local = ['Base.\nCorrect', 'Base.\nWrong', 'Dual.\nCorrect', 'Dual.\nWrong']

        bars = ax.bar(positions, means, width=0.7, yerr=stds,
                     color=colors_local, edgecolor='white', linewidth=0.5,
                     capsize=3, error_kw={'linewidth': 0.8})

        ax.set_xticks(positions)
        ax.set_xticklabels(labels_local, fontsize=6)
        ax.set_ylim(0.3, 1.0)
        ax.set_title(MODEL_SHORT[model_name], fontsize=9, fontweight='bold')
        ax.grid(axis='y', alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Separation annotation
        sep_base = cd['base_separation']
        sep_dual = cd['dual_separation']
        ax.text(0.5, 0.92, f'Δ={sep_base:.2f}', transform=ax.transAxes,
               fontsize=7, ha='center', color='#E69F00', fontweight='bold')
        ax.text(0.5, 0.83, f'Δ={sep_dual:.2f}', transform=ax.transAxes,
               fontsize=7, ha='center', color='#009E73', fontweight='bold')

        if m_idx == 0:
            ax.set_ylabel('Mean Confidence', fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig6_confidence_dist.pdf'))
    plt.close()
    print("  ✓ fig6_confidence_dist.pdf")


# ============================================================================
# Figure 7: Easy vs Hard accuracy comparison
# ============================================================================
def fig7_easy_hard():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))

    for diff_idx, (diff, title) in enumerate([('easy', 'Easy Items'), ('hard', 'Hard Items')]):
        ax = axes[diff_idx]
        x = np.arange(len(models))
        width = 0.18

        for c_idx, cond in enumerate(conditions):
            vals = [data['results'][m][cond][f'{diff}_accuracy'] for m in models]
            offset = (c_idx - 1.5) * width
            ax.bar(x + offset, vals, width, label=CONDITION_LABELS[cond],
                  color=COLORS[cond], edgecolor='white', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_SHORT[m] for m in models], fontsize=7)
        ax.set_ylabel('Accuracy', fontsize=8)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_ylim(0.3, 1.0)
        ax.grid(axis='y', alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if diff_idx == 0:
            ax.legend(fontsize=6.5, ncol=2, loc='lower left')

    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig7_easy_hard.pdf'))
    plt.close()
    print("  ✓ fig7_easy_hard.pdf")


# ============================================================================
# Figure 8: Token cost vs improvement trade-off
# ============================================================================
def fig8_cost_tradeoff():
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.8))

    token_costs = {
        'baseline': 180,
        'cot': 300,
        'dual_process': 950,
        'deliberate_only': 450,
    }

    for model_name in models:
        for cond in conditions:
            acc = data['results'][model_name][cond]['accuracy']
            conf_err = data['results'][model_name][cond]['confident_error_rate']
            cost = token_costs[cond]

            marker = {'GPT-4o-mini': 'o', 'Gemini-2.0-Flash': 's', 'Claude-3.5-Sonnet': '^'}[model_name]
            color = COLORS[cond]
            size = 40

            ax.scatter(cost, conf_err, c=color, marker=marker, s=size,
                      edgecolors='black', linewidth=0.5, zorder=3)

    # Create legend handles
    model_handles = [
        plt.Line2D([0], [0], marker='o', color='gray', markerfacecolor='gray',
                   markersize=5, linestyle='', label='GPT-4o-mini'),
        plt.Line2D([0], [0], marker='s', color='gray', markerfacecolor='gray',
                   markersize=5, linestyle='', label='Gemini Flash'),
        plt.Line2D([0], [0], marker='^', color='gray', markerfacecolor='gray',
                   markersize=5, linestyle='', label='Claude 3.5'),
    ]
    cond_handles = [
        mpatches.Patch(color=COLORS[c], label=CONDITION_LABELS[c]) for c in conditions
    ]

    leg1 = ax.legend(handles=model_handles, loc='upper left', fontsize=6, title='Model', title_fontsize=6.5)
    ax.add_artist(leg1)
    ax.legend(handles=cond_handles, loc='upper right', fontsize=6, title='Method', title_fontsize=6.5)

    ax.set_xlabel('Tokens per Item', fontsize=9)
    ax.set_ylabel('Confident Error Rate', fontsize=9)
    ax.grid(alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig8_cost_tradeoff.pdf'))
    plt.close()
    print("  ✓ fig8_cost_tradeoff.pdf")


# ============================================================================
# Figure 9: Per-category flip analysis for best model
# ============================================================================
def fig9_category_flips():
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.5))

    # Use GPT-4o-mini (best flip ratio)
    model_name = 'GPT-4o-mini'
    cat_flips = data['results'][model_name]['cat_flip_analysis']

    x = np.arange(len(categories))
    width = 0.3

    flip_correct = [cat_flips[cat]['flip_to_correct_pct'] for cat in categories]
    flip_wrong = [cat_flips[cat]['flip_to_wrong_pct'] for cat in categories]

    ax.bar(x - width/2, flip_correct, width, label='Flip to Correct',
           color='#2CA02C', edgecolor='white', linewidth=0.5)
    ax.bar(x + width/2, flip_wrong, width, label='Flip to Wrong',
           color='#D62728', edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=7, rotation=20, ha='right')
    ax.set_ylabel('Items Flipped (%)', fontsize=8)
    ax.set_title(f'{MODEL_SHORT[model_name]}: Per-Category Flip Analysis', fontsize=8, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig9_category_flips.pdf'))
    plt.close()
    print("  ✓ fig9_category_flips.pdf")


# ============================================================================
# Run all figure generation
# ============================================================================
if __name__ == '__main__':
    print("Generating figures...")
    fig1_method_overview()
    fig2_main_results()
    fig3_calibration()
    fig4_flip_analysis()
    fig5_category_heatmap()
    fig6_confidence_distributions()
    fig7_easy_hard()
    fig8_cost_tradeoff()
    fig9_category_flips()
    print(f"\nAll figures saved to {fig_dir}/")
