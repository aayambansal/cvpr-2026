#!/usr/bin/env python3
"""Generate all publication-quality figures for the paper."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# Publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colorblind-safe palette (Okabe-Ito)
COLORS = {
    'baseline': '#D55E00',   # vermillion
    'sras': '#0072B2',       # blue
    'sras_light': '#56B4E9', # sky blue
    'oracle': '#009E73',     # green
    'accent': '#E69F00',     # orange
    'gray': '#999999',
    'black': '#000000',
}

# Load results
with open('experiment_results.json', 'r') as f:
    results = json.load(f)

FIGDIR = 'figures'

# ============================================================================
# Figure 1: Method Overview (SRAS Pipeline)
# ============================================================================
def fig1_method_overview():
    """Create the method overview schematic."""
    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')
    
    # Style
    box_kw = dict(boxstyle='round,pad=0.3', linewidth=1.0)
    arrow_kw = dict(arrowstyle='->', lw=1.5, color='#333333')
    
    # Baseline path (top)
    ax.annotate('', xy=(2.8, 2.4), xytext=(0.5, 2.4), arrowprops=arrow_kw)
    ax.text(0.3, 2.4, 'Single\nSeed $s$', ha='center', va='center', fontsize=7,
            bbox=dict(facecolor='#FFE0CC', edgecolor=COLORS['baseline'], **box_kw))
    ax.text(1.65, 2.6, 'Train\n(T epochs)', ha='center', va='center', fontsize=6, color='#666')
    
    rect = FancyBboxPatch((2.8, 2.05), 1.6, 0.7, boxstyle='round,pad=0.15',
                          facecolor='#FFE0CC', edgecolor=COLORS['baseline'], linewidth=1.0)
    ax.add_patch(rect)
    ax.text(3.6, 2.4, 'Supernet\nRanking', ha='center', va='center', fontsize=7)
    
    ax.annotate('', xy=(5.6, 2.4), xytext=(4.5, 2.4), arrowprops=arrow_kw)
    ax.text(5.05, 2.6, 'Evaluate', ha='center', va='center', fontsize=6, color='#666')
    
    rect2 = FancyBboxPatch((5.6, 2.05), 1.8, 0.7, boxstyle='round,pad=0.15',
                           facecolor='#FFE0CC', edgecolor=COLORS['baseline'], linewidth=1.0)
    ax.add_patch(rect2)
    ax.text(6.5, 2.4, 'Architecture\nSelection', ha='center', va='center', fontsize=7)
    
    ax.annotate('', xy=(8.7, 2.4), xytext=(7.5, 2.4),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=COLORS['baseline']))
    
    rect3 = FancyBboxPatch((8.7, 2.05), 1.1, 0.7, boxstyle='round,pad=0.15',
                           facecolor='#FFE0CC', edgecolor=COLORS['baseline'], linewidth=1.0)
    ax.add_patch(rect3)
    ax.text(9.25, 2.4, 'α*\n(unstable)', ha='center', va='center', fontsize=7, 
            color=COLORS['baseline'], fontweight='bold')
    
    ax.text(-0.1, 2.4, '(a)', fontweight='bold', fontsize=9, va='center')
    
    # SRAS path (bottom)
    # Multiple warmup seeds
    for i, (y_off, label) in enumerate([(0.85, '$s_1$'), (0.55, '$s_2$'), (0.25, '..'), (-0.05, '$s_K$')]):
        color = COLORS['sras'] if i != 2 else '#999999'
        ax.text(0.3, y_off, label, ha='center', va='center', fontsize=7,
                bbox=dict(facecolor='#CCE5FF', edgecolor=color, **box_kw) if i != 2 else {})
    
    # Arrows from seeds to warmups
    for y in [0.85, 0.55, -0.05]:
        ax.annotate('', xy=(1.7, y), xytext=(0.65, y),
                    arrowprops=dict(arrowstyle='->', lw=0.8, color='#666666'))
    
    # Warmup boxes
    rect_w = FancyBboxPatch((1.7, -0.2), 1.3, 1.2, boxstyle='round,pad=0.15',
                            facecolor='#CCE5FF', edgecolor=COLORS['sras'], linewidth=1.0)
    ax.add_patch(rect_w)
    ax.text(2.35, 0.4, 'K Short\nWarmups\n(T/3 each)', ha='center', va='center', fontsize=7)
    
    ax.annotate('', xy=(3.4, 0.4), xytext=(3.05, 0.4), arrowprops=arrow_kw)
    
    # Rank lists
    rect_r = FancyBboxPatch((3.4, -0.1), 1.4, 1.0, boxstyle='round,pad=0.15',
                            facecolor='#CCE5FF', edgecolor=COLORS['sras'], linewidth=1.0)
    ax.add_patch(rect_r)
    ax.text(4.1, 0.4, 'K Rank\nLists', ha='center', va='center', fontsize=7)
    
    # Arrow to aggregation
    ax.annotate('', xy=(5.2, 0.4), xytext=(4.85, 0.4), arrowprops=arrow_kw)
    
    # Z-score aggregation
    rect_a = FancyBboxPatch((5.2, -0.1), 1.6, 1.0, boxstyle='round,pad=0.15',
                            facecolor='#B3D9FF', edgecolor=COLORS['sras'], linewidth=1.2)
    ax.add_patch(rect_a)
    ax.text(6.0, 0.55, 'Z-Score', ha='center', va='center', fontsize=7, fontweight='bold')
    ax.text(6.0, 0.25, 'Aggregation', ha='center', va='center', fontsize=7, fontweight='bold')
    
    # BN recal
    ax.annotate('', xy=(7.3, 0.4), xytext=(6.85, 0.4), arrowprops=arrow_kw)
    rect_bn = FancyBboxPatch((7.3, 0.0), 1.1, 0.8, boxstyle='round,pad=0.15',
                             facecolor='#CCE5FF', edgecolor=COLORS['sras'], linewidth=1.0)
    ax.add_patch(rect_bn)
    ax.text(7.85, 0.4, 'BN\nRecalib.', ha='center', va='center', fontsize=7)
    
    # Final stable selection
    ax.annotate('', xy=(8.85, 0.4), xytext=(8.45, 0.4),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=COLORS['sras']))
    
    rect_f = FancyBboxPatch((8.85, 0.05), 0.95, 0.7, boxstyle='round,pad=0.15',
                            facecolor='#B3D9FF', edgecolor=COLORS['sras'], linewidth=1.2)
    ax.add_patch(rect_f)
    ax.text(9.32, 0.4, 'α*\n(stable)', ha='center', va='center', fontsize=7,
            color=COLORS['sras'], fontweight='bold')
    
    ax.text(-0.1, 0.4, '(b)', fontweight='bold', fontsize=9, va='center')
    
    # Labels
    ax.text(5.0, 2.9, 'Standard One-Shot NAS', fontsize=8, fontweight='bold',
            ha='center', color=COLORS['baseline'])
    ax.text(5.0, 1.35, 'SRAS: Seed-Robust Architecture Selection (Ours)', 
            fontsize=8, fontweight='bold', ha='center', color=COLORS['sras'])
    
    plt.savefig(f'{FIGDIR}/fig1_method_overview.pdf', format='pdf')
    plt.savefig(f'{FIGDIR}/fig1_method_overview.png', format='png', dpi=300)
    plt.close()
    print("  Generated fig1_method_overview")


# ============================================================================
# Figure 2: Seed Sensitivity Heatmaps (Baseline vs SRAS)
# ============================================================================
def fig2_correlation_heatmaps():
    """Pairwise Kendall tau heatmaps: Baseline vs SRAS."""
    bl_tau = np.array(results['experiment_1_baseline']['tau_matrix'])
    sr_tau = np.array(results['experiment_2_sras']['zscore']['tau_matrix'])
    
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    
    vmin, vmax = 0.5, 1.0
    
    im1 = axes[0].imshow(bl_tau, cmap='RdYlGn', vmin=vmin, vmax=vmax, aspect='equal')
    axes[0].set_title(f'Baseline (mean τ = {bl_tau[~np.eye(20, dtype=bool)].mean():.3f})', fontsize=8)
    axes[0].set_xlabel('Seed index')
    axes[0].set_ylabel('Seed index')
    axes[0].set_xticks(range(0, 20, 4))
    axes[0].set_yticks(range(0, 20, 4))
    
    im2 = axes[1].imshow(sr_tau, cmap='RdYlGn', vmin=vmin, vmax=vmax, aspect='equal')
    axes[1].set_title(f'SRAS K=5 (mean τ = {sr_tau[~np.eye(20, dtype=bool)].mean():.3f})', fontsize=8)
    axes[1].set_xlabel('Seed index')
    axes[1].set_ylabel('Seed index')
    axes[1].set_xticks(range(0, 20, 4))
    axes[1].set_yticks(range(0, 20, 4))
    
    # Panel labels
    axes[0].text(-0.15, 1.05, 'A', transform=axes[0].transAxes, fontsize=10, fontweight='bold')
    axes[1].text(-0.15, 1.05, 'B', transform=axes[1].transAxes, fontsize=10, fontweight='bold')
    
    # Colorbar
    cbar = fig.colorbar(im2, ax=axes, shrink=0.8, aspect=30, label="Kendall's τ")
    
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/fig2_heatmaps.pdf', format='pdf')
    plt.savefig(f'{FIGDIR}/fig2_heatmaps.png', format='png', dpi=300)
    plt.close()
    print("  Generated fig2_heatmaps")


# ============================================================================
# Figure 3: Ablation over K + Regret
# ============================================================================
def fig3_ablation():
    """Ablation over K: GT tau, pairwise tau, regret, unique top-1."""
    ablation = results['experiment_3_ablation_K']
    Ks = [1, 2, 3, 5, 7, 10]
    
    gt_taus = [ablation[str(k)]['gt_tau_mean'] for k in Ks]
    gt_tau_stds = [ablation[str(k)]['gt_tau_std'] for k in Ks]
    pw_taus = [ablation[str(k)]['pairwise_tau_mean'] for k in Ks]
    pw_tau_stds = [ablation[str(k)]['pairwise_tau_std'] for k in Ks]
    regrets = [ablation[str(k)]['regret_1_mean'] for k in Ks]
    regret_stds = [ablation[str(k)]['regret_1_std'] for k in Ks]
    unique = [ablation[str(k)]['unique_top1'] for k in Ks]
    
    # Baseline reference
    bl_gt_tau = results['experiment_1_baseline']['gt_tau_mean']
    bl_pw_tau = results['experiment_1_baseline']['tau_mean']
    bl_regret = results['experiment_1_baseline']['regrets']['1']['mean']
    bl_unique = results['experiment_1_baseline']['unique_top1']
    
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    
    # Panel A: GT Kendall tau vs K
    axes[0].errorbar(Ks, gt_taus, yerr=gt_tau_stds, fmt='o-', color=COLORS['sras'],
                     linewidth=1.5, markersize=4, capsize=3, label='SRAS')
    axes[0].axhline(bl_gt_tau, color=COLORS['baseline'], linestyle='--', linewidth=1, label='Baseline')
    axes[0].set_xlabel('Number of warmup runs (K)')
    axes[0].set_ylabel("Kendall's τ vs. GT")
    axes[0].legend(frameon=False, loc='lower right')
    axes[0].set_ylim(0.75, 0.95)
    axes[0].text(-0.2, 1.05, 'A', transform=axes[0].transAxes, fontsize=10, fontweight='bold')
    
    # Panel B: Top-1 Regret vs K
    axes[1].errorbar(Ks, regrets, yerr=regret_stds, fmt='s-', color=COLORS['sras'],
                     linewidth=1.5, markersize=4, capsize=3, label='SRAS')
    axes[1].axhline(bl_regret, color=COLORS['baseline'], linestyle='--', linewidth=1, label='Baseline')
    axes[1].set_xlabel('Number of warmup runs (K)')
    axes[1].set_ylabel('Top-1 regret (%)')
    axes[1].legend(frameon=False, loc='upper right')
    axes[1].set_ylim(-0.1, 3.0)
    axes[1].text(-0.2, 1.05, 'B', transform=axes[1].transAxes, fontsize=10, fontweight='bold')
    
    # Panel C: Unique top-1 selections vs K
    axes[2].plot(Ks, unique, 'D-', color=COLORS['sras'], linewidth=1.5, markersize=4, label='SRAS')
    axes[2].axhline(bl_unique, color=COLORS['baseline'], linestyle='--', linewidth=1, label='Baseline')
    axes[2].set_xlabel('Number of warmup runs (K)')
    axes[2].set_ylabel('Unique top-1 (/20 seeds)')
    axes[2].legend(frameon=False, loc='upper right')
    axes[2].set_ylim(0, 18)
    axes[2].text(-0.2, 1.05, 'C', transform=axes[2].transAxes, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/fig3_ablation.pdf', format='pdf')
    plt.savefig(f'{FIGDIR}/fig3_ablation.png', format='png', dpi=300)
    plt.close()
    print("  Generated fig3_ablation")


# ============================================================================
# Figure 4: Budget-Matched Comparison
# ============================================================================
def fig4_budget_comparison():
    """Bar chart comparing methods at matched budgets."""
    budget = results['experiment_5_budget']
    
    methods = ['baseline_1x', 'sras_k3_1x', 'long_1.67x', 'sras_k5_1.67x']
    labels = ['Baseline\n(1.0×)', 'SRAS K=3\n(1.0×)', 'Longer Train\n(1.67×)', 'SRAS K=5\n(1.67×)']
    colors = [COLORS['baseline'], COLORS['sras'], COLORS['accent'], COLORS['sras_light']]
    
    taus = [budget[m]['tau_mean'] for m in methods]
    tau_errs = [budget[m]['tau_std'] for m in methods]
    regrets = [budget[m]['regret_mean'] for m in methods]
    regret_errs = [budget[m]['regret_std'] for m in methods]
    
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    
    x = np.arange(len(methods))
    w = 0.6
    
    # Panel A: GT Kendall tau
    bars1 = axes[0].bar(x, taus, w, yerr=tau_errs, color=colors, edgecolor='white',
                        capsize=3, error_kw={'linewidth': 0.8})
    axes[0].set_ylabel("Kendall's τ vs. GT")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=6)
    axes[0].set_ylim(0.7, 0.95)
    axes[0].text(-0.15, 1.05, 'A', transform=axes[0].transAxes, fontsize=10, fontweight='bold')
    
    # Add value labels on bars
    for bar, val in zip(bars1, taus):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=6)
    
    # Panel B: Top-1 Regret
    bars2 = axes[1].bar(x, regrets, w, yerr=regret_errs, color=colors, edgecolor='white',
                        capsize=3, error_kw={'linewidth': 0.8})
    axes[1].set_ylabel('Top-1 Regret (%)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=6)
    axes[1].set_ylim(0, 3.5)
    axes[1].text(-0.15, 1.05, 'B', transform=axes[1].transAxes, fontsize=10, fontweight='bold')
    
    for bar, val in zip(bars2, regrets):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                     f'{val:.2f}', ha='center', va='bottom', fontsize=6)
    
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/fig4_budget.pdf', format='pdf')
    plt.savefig(f'{FIGDIR}/fig4_budget.png', format='png', dpi=300)
    plt.close()
    print("  Generated fig4_budget")


# ============================================================================
# Figure 5: Per-Architecture Rank Variance
# ============================================================================
def fig5_rank_variance():
    """Scatter plot of per-architecture rank variance."""
    gt = np.array(results['ground_truth'])
    bl_var = np.array(results['experiment_4_variance']['baseline_rank_var_per_arch'])
    sr_var = np.array(results['experiment_4_variance']['sras_rank_var_per_arch'])
    
    bl_std = np.sqrt(bl_var)
    sr_std = np.sqrt(sr_var)
    
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    
    # Panel A: Rank std vs GT accuracy  
    axes[0].scatter(gt, bl_std, s=6, alpha=0.4, color=COLORS['baseline'], label='Baseline', rasterized=True)
    axes[0].scatter(gt, sr_std, s=6, alpha=0.4, color=COLORS['sras'], label='SRAS (K=5)', rasterized=True)
    axes[0].set_xlabel('Ground truth accuracy (%)')
    axes[0].set_ylabel('Rank std across seeds')
    axes[0].legend(frameon=False, markerscale=2)
    axes[0].text(-0.15, 1.05, 'A', transform=axes[0].transAxes, fontsize=10, fontweight='bold')
    
    # Panel B: Histogram of rank std reduction
    reduction = (1 - sr_std / np.maximum(bl_std, 1e-6)) * 100
    reduction = np.clip(reduction, -100, 100)
    axes[1].hist(reduction, bins=40, color=COLORS['sras'], alpha=0.7, edgecolor='white', linewidth=0.5)
    axes[1].axvline(np.median(reduction), color=COLORS['black'], linestyle='--', linewidth=1,
                    label=f'Median: {np.median(reduction):.1f}%')
    axes[1].set_xlabel('Rank std reduction (%)')
    axes[1].set_ylabel('Number of architectures')
    axes[1].legend(frameon=False)
    axes[1].text(-0.15, 1.05, 'B', transform=axes[1].transAxes, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/fig5_variance.pdf', format='pdf')
    plt.savefig(f'{FIGDIR}/fig5_variance.png', format='png', dpi=300)
    plt.close()
    print("  Generated fig5_variance")


# ============================================================================
# Figure 6: Top-k Overlap Comparison
# ============================================================================
def fig6_topk_overlap():
    """Bar chart: top-k overlap for baseline vs SRAS."""
    bl = results['experiment_1_baseline']
    sr = results['experiment_2_sras']['zscore']
    
    ks = [1, 3, 5, 10, 20]
    bl_vals = [bl['topk_means'][str(k)] for k in ks]
    sr_vals = [sr['topk_means'][str(k)] for k in ks]
    
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    x = np.arange(len(ks))
    w = 0.35
    
    bars1 = ax.bar(x - w/2, bl_vals, w, color=COLORS['baseline'], label='Baseline', edgecolor='white')
    bars2 = ax.bar(x + w/2, sr_vals, w, color=COLORS['sras'], label='SRAS (K=5)', edgecolor='white')
    
    ax.set_xlabel('k')
    ax.set_ylabel('Pairwise top-k overlap')
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False)
    
    # Value labels
    for bar, val in zip(bars1, bl_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=5.5)
    for bar, val in zip(bars2, sr_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=5.5)
    
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/fig6_topk.pdf', format='pdf')
    plt.savefig(f'{FIGDIR}/fig6_topk.png', format='png', dpi=300)
    plt.close()
    print("  Generated fig6_topk")


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    import os
    os.makedirs(FIGDIR, exist_ok=True)
    
    print("Generating publication figures...")
    fig1_method_overview()
    fig2_correlation_heatmaps()
    fig3_ablation()
    fig4_budget_comparison()
    fig5_rank_variance()
    fig6_topk_overlap()
    print("Done! All figures saved to figures/")
