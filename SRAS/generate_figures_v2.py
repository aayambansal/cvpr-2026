#!/usr/bin/env python3
"""Generate all publication-quality figures for the v2 paper (extended experiments)."""

import json
import os
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

COLORS = {
    'baseline': '#D55E00',
    'sras': '#0072B2',
    'sras_light': '#56B4E9',
    'oracle': '#009E73',
    'accent': '#E69F00',
    'gray': '#999999',
    'black': '#000000',
    'purple': '#CC79A7',
    'teal': '#009E73',
}

with open('experiment_results_v2.json', 'r') as f:
    results = json.load(f)

FIGDIR = 'figures'
LATEXDIR = 'latex'
os.makedirs(FIGDIR, exist_ok=True)

def save_fig(name):
    plt.savefig(f'{FIGDIR}/{name}.pdf', format='pdf')
    plt.savefig(f'{FIGDIR}/{name}.png', format='png', dpi=300)
    plt.savefig(f'{LATEXDIR}/{name}.pdf', format='pdf')
    plt.close()
    print(f"  Generated {name}")


# ============================================================================
# Fig 1: Method Overview (same as v1)
# ============================================================================
def fig1_method_overview():
    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis('off')
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
    ax.text(9.25, 2.4, r'$\alpha^*$'+'\n(unstable)', ha='center', va='center', fontsize=7,
            color=COLORS['baseline'], fontweight='bold')
    ax.text(-0.1, 2.4, '(a)', fontweight='bold', fontsize=9, va='center')
    
    # SRAS path (bottom)
    for i, (y_off, label) in enumerate([(0.85, '$s_1$'), (0.55, '$s_2$'), (0.25, '..'), (-0.05, '$s_K$')]):
        color = COLORS['sras'] if i != 2 else '#999999'
        ax.text(0.3, y_off, label, ha='center', va='center', fontsize=7,
                bbox=dict(facecolor='#CCE5FF', edgecolor=color, **box_kw) if i != 2 else {})
    for y in [0.85, 0.55, -0.05]:
        ax.annotate('', xy=(1.7, y), xytext=(0.65, y),
                    arrowprops=dict(arrowstyle='->', lw=0.8, color='#666666'))
    rect_w = FancyBboxPatch((1.7, -0.2), 1.3, 1.2, boxstyle='round,pad=0.15',
                            facecolor='#CCE5FF', edgecolor=COLORS['sras'], linewidth=1.0)
    ax.add_patch(rect_w)
    ax.text(2.35, 0.4, 'K Short\nWarmups\n(T/3 each)', ha='center', va='center', fontsize=7)
    ax.annotate('', xy=(3.4, 0.4), xytext=(3.05, 0.4), arrowprops=arrow_kw)
    rect_r = FancyBboxPatch((3.4, -0.1), 1.4, 1.0, boxstyle='round,pad=0.15',
                            facecolor='#CCE5FF', edgecolor=COLORS['sras'], linewidth=1.0)
    ax.add_patch(rect_r)
    ax.text(4.1, 0.4, 'K Rank\nLists', ha='center', va='center', fontsize=7)
    ax.annotate('', xy=(5.2, 0.4), xytext=(4.85, 0.4), arrowprops=arrow_kw)
    rect_a = FancyBboxPatch((5.2, -0.1), 1.6, 1.0, boxstyle='round,pad=0.15',
                            facecolor='#B3D9FF', edgecolor=COLORS['sras'], linewidth=1.2)
    ax.add_patch(rect_a)
    ax.text(6.0, 0.55, 'Z-Score', ha='center', va='center', fontsize=7, fontweight='bold')
    ax.text(6.0, 0.25, 'Aggregation', ha='center', va='center', fontsize=7, fontweight='bold')
    ax.annotate('', xy=(7.3, 0.4), xytext=(6.85, 0.4), arrowprops=arrow_kw)
    rect_bn = FancyBboxPatch((7.3, 0.0), 1.1, 0.8, boxstyle='round,pad=0.15',
                             facecolor='#CCE5FF', edgecolor=COLORS['sras'], linewidth=1.0)
    ax.add_patch(rect_bn)
    ax.text(7.85, 0.4, 'BN\nRecalib.', ha='center', va='center', fontsize=7)
    ax.annotate('', xy=(8.85, 0.4), xytext=(8.45, 0.4),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=COLORS['sras']))
    rect_f = FancyBboxPatch((8.85, 0.05), 0.95, 0.7, boxstyle='round,pad=0.15',
                            facecolor='#B3D9FF', edgecolor=COLORS['sras'], linewidth=1.2)
    ax.add_patch(rect_f)
    ax.text(9.32, 0.4, r'$\alpha^*$'+'\n(stable)', ha='center', va='center', fontsize=7,
            color=COLORS['sras'], fontweight='bold')
    ax.text(-0.1, 0.4, '(b)', fontweight='bold', fontsize=9, va='center')
    ax.text(5.0, 2.9, 'Standard One-Shot NAS', fontsize=8, fontweight='bold',
            ha='center', color=COLORS['baseline'])
    ax.text(5.0, 1.35, 'SRAS: Seed-Robust Architecture Selection (Ours)',
            fontsize=8, fontweight='bold', ha='center', color=COLORS['sras'])
    save_fig('fig1_method_overview')


# ============================================================================
# Fig 2: Heatmaps (same as v1)
# ============================================================================
def fig2_correlation_heatmaps():
    bl_tau = np.array(results['experiment_1_baseline']['tau_matrix'])
    sr_tau = np.array(results['experiment_2_sras']['zscore']['tau_matrix'])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    vmin, vmax = 0.5, 1.0
    im1 = axes[0].imshow(bl_tau, cmap='RdYlGn', vmin=vmin, vmax=vmax, aspect='equal')
    axes[0].set_title(f'Baseline (mean $\\tau$ = {bl_tau[~np.eye(20, dtype=bool)].mean():.3f})', fontsize=8)
    axes[0].set_xlabel('Seed index'); axes[0].set_ylabel('Seed index')
    axes[0].set_xticks(range(0, 20, 4)); axes[0].set_yticks(range(0, 20, 4))
    im2 = axes[1].imshow(sr_tau, cmap='RdYlGn', vmin=vmin, vmax=vmax, aspect='equal')
    axes[1].set_title(f'SRAS K=5 (mean $\\tau$ = {sr_tau[~np.eye(20, dtype=bool)].mean():.3f})', fontsize=8)
    axes[1].set_xlabel('Seed index'); axes[1].set_ylabel('Seed index')
    axes[1].set_xticks(range(0, 20, 4)); axes[1].set_yticks(range(0, 20, 4))
    axes[0].text(-0.15, 1.05, 'A', transform=axes[0].transAxes, fontsize=10, fontweight='bold')
    axes[1].text(-0.15, 1.05, 'B', transform=axes[1].transAxes, fontsize=10, fontweight='bold')
    fig.colorbar(im2, ax=axes, shrink=0.8, aspect=30, label="Kendall's $\\tau$")
    plt.tight_layout()
    save_fig('fig2_heatmaps')


# ============================================================================
# Fig 3: Ablation over K (same as v1)
# ============================================================================
def fig3_ablation():
    ablation = results['experiment_3_ablation_K']
    Ks = [1, 2, 3, 5, 7, 10]
    gt_taus = [ablation[str(k)]['gt_tau_mean'] for k in Ks]
    gt_tau_stds = [ablation[str(k)]['gt_tau_std'] for k in Ks]
    regrets = [ablation[str(k)]['regret_1_mean'] for k in Ks]
    regret_stds = [ablation[str(k)]['regret_1_std'] for k in Ks]
    unique = [ablation[str(k)]['unique_top1'] for k in Ks]
    bl_gt_tau = results['experiment_1_baseline']['gt_tau_mean']
    bl_regret = results['experiment_1_baseline']['regrets']['1']['mean']
    bl_unique = results['experiment_1_baseline']['unique_top1']
    
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    axes[0].errorbar(Ks, gt_taus, yerr=gt_tau_stds, fmt='o-', color=COLORS['sras'],
                     linewidth=1.5, markersize=4, capsize=3, label='SRAS')
    axes[0].axhline(bl_gt_tau, color=COLORS['baseline'], linestyle='--', linewidth=1, label='Baseline')
    axes[0].set_xlabel('Number of warmup runs (K)'); axes[0].set_ylabel("Kendall's $\\tau$ vs. GT")
    axes[0].legend(frameon=False, loc='lower right'); axes[0].set_ylim(0.75, 0.95)
    axes[0].text(-0.2, 1.05, 'A', transform=axes[0].transAxes, fontsize=10, fontweight='bold')
    
    axes[1].errorbar(Ks, regrets, yerr=regret_stds, fmt='s-', color=COLORS['sras'],
                     linewidth=1.5, markersize=4, capsize=3, label='SRAS')
    axes[1].axhline(bl_regret, color=COLORS['baseline'], linestyle='--', linewidth=1, label='Baseline')
    axes[1].set_xlabel('Number of warmup runs (K)'); axes[1].set_ylabel('Top-1 regret (%)')
    axes[1].legend(frameon=False, loc='upper right'); axes[1].set_ylim(-0.1, 3.0)
    axes[1].text(-0.2, 1.05, 'B', transform=axes[1].transAxes, fontsize=10, fontweight='bold')
    
    axes[2].plot(Ks, unique, 'D-', color=COLORS['sras'], linewidth=1.5, markersize=4, label='SRAS')
    axes[2].axhline(bl_unique, color=COLORS['baseline'], linestyle='--', linewidth=1, label='Baseline')
    axes[2].set_xlabel('Number of warmup runs (K)'); axes[2].set_ylabel('Unique top-1 (/20 seeds)')
    axes[2].legend(frameon=False, loc='upper right'); axes[2].set_ylim(0, 18)
    axes[2].text(-0.2, 1.05, 'C', transform=axes[2].transAxes, fontsize=10, fontweight='bold')
    plt.tight_layout()
    save_fig('fig3_ablation')


# ============================================================================
# Fig 4: Budget-Matched Comparison (same as v1)
# ============================================================================
def fig4_budget_comparison():
    budget = results['experiment_5_budget']
    methods = ['baseline_1x', 'sras_k3_1x', 'long_1.67x', 'sras_k5_1.67x']
    labels = ['Baseline\n(1.0x)', 'SRAS K=3\n(1.0x)', 'Longer\n(1.67x)', 'SRAS K=5\n(1.67x)']
    colors = [COLORS['baseline'], COLORS['sras'], COLORS['accent'], COLORS['sras_light']]
    taus = [budget[m]['tau_mean'] for m in methods]
    tau_errs = [budget[m]['tau_std'] for m in methods]
    regrets = [budget[m]['regret_mean'] for m in methods]
    regret_errs = [budget[m]['regret_std'] for m in methods]
    
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    x = np.arange(len(methods)); w = 0.6
    bars1 = axes[0].bar(x, taus, w, yerr=tau_errs, color=colors, edgecolor='white', capsize=3, error_kw={'linewidth': 0.8})
    axes[0].set_ylabel("Kendall's $\\tau$ vs. GT"); axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=6); axes[0].set_ylim(0.7, 0.95)
    axes[0].text(-0.15, 1.05, 'A', transform=axes[0].transAxes, fontsize=10, fontweight='bold')
    for bar, val in zip(bars1, taus):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008, f'{val:.3f}', ha='center', va='bottom', fontsize=6)
    
    bars2 = axes[1].bar(x, regrets, w, yerr=regret_errs, color=colors, edgecolor='white', capsize=3, error_kw={'linewidth': 0.8})
    axes[1].set_ylabel('Top-1 Regret (%)'); axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=6); axes[1].set_ylim(0, 3.5)
    axes[1].text(-0.15, 1.05, 'B', transform=axes[1].transAxes, fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, regrets):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08, f'{val:.2f}', ha='center', va='bottom', fontsize=6)
    plt.tight_layout()
    save_fig('fig4_budget')


# ============================================================================
# Fig 5: Per-Architecture Rank Variance (same as v1)
# ============================================================================
def fig5_rank_variance():
    gt = np.array(results['ground_truth'])
    bl_var = np.array(results['experiment_4_variance']['baseline_rank_var_per_arch'])
    sr_var = np.array(results['experiment_4_variance']['sras_rank_var_per_arch'])
    bl_std = np.sqrt(bl_var); sr_std = np.sqrt(sr_var)
    
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    axes[0].scatter(gt, bl_std, s=6, alpha=0.4, color=COLORS['baseline'], label='Baseline', rasterized=True)
    axes[0].scatter(gt, sr_std, s=6, alpha=0.4, color=COLORS['sras'], label='SRAS (K=5)', rasterized=True)
    axes[0].set_xlabel('Ground truth accuracy (%)'); axes[0].set_ylabel('Rank std across seeds')
    axes[0].legend(frameon=False, markerscale=2)
    axes[0].text(-0.15, 1.05, 'A', transform=axes[0].transAxes, fontsize=10, fontweight='bold')
    
    reduction = (1 - sr_std / np.maximum(bl_std, 1e-6)) * 100
    reduction = np.clip(reduction, -100, 100)
    axes[1].hist(reduction, bins=40, color=COLORS['sras'], alpha=0.7, edgecolor='white', linewidth=0.5)
    axes[1].axvline(np.median(reduction), color=COLORS['black'], linestyle='--', linewidth=1,
                    label=f'Median: {np.median(reduction):.1f}%')
    axes[1].set_xlabel('Rank std reduction (%)'); axes[1].set_ylabel('Number of architectures')
    axes[1].legend(frameon=False)
    axes[1].text(-0.15, 1.05, 'B', transform=axes[1].transAxes, fontsize=10, fontweight='bold')
    plt.tight_layout()
    save_fig('fig5_variance')


# ============================================================================
# Fig 6: Top-k Overlap (same as v1)
# ============================================================================
def fig6_topk_overlap():
    bl = results['experiment_1_baseline']
    sr = results['experiment_2_sras']['zscore']
    ks = [1, 3, 5, 10, 20]
    bl_vals = [bl['topk_means'][str(k)] for k in ks]
    sr_vals = [sr['topk_means'][str(k)] for k in ks]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.arange(len(ks)); w = 0.35
    bars1 = ax.bar(x - w/2, bl_vals, w, color=COLORS['baseline'], label='Baseline', edgecolor='white')
    bars2 = ax.bar(x + w/2, sr_vals, w, color=COLORS['sras'], label='SRAS (K=5)', edgecolor='white')
    ax.set_xlabel('k'); ax.set_ylabel('Pairwise top-k overlap')
    ax.set_xticks(x); ax.set_xticklabels([str(k) for k in ks]); ax.set_ylim(0, 1.0)
    ax.legend(frameon=False)
    for bar, val in zip(bars1, bl_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=5.5)
    for bar, val in zip(bars2, sr_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=5.5)
    plt.tight_layout()
    save_fig('fig6_topk')


# ============================================================================
# NEW Fig 7: Dumb-Ensemble Baselines Comparison
# ============================================================================
def fig7_dumb_ensembles():
    """Bar chart comparing all aggregation methods at K=5."""
    ens = results['experiment_6_dumb_ensembles']
    
    methods = ['zscore', 'avg_raw', 'median_raw', 'median_rank', 'majority_vote_5', 'majority_vote_10']
    labels = ['Z-Score\n(SRAS)', 'Avg Raw\nScores', 'Median\nRaw', 'Median\nRank', 'Maj. Vote\n(k=5)', 'Maj. Vote\n(k=10)']
    
    gt_taus = [ens[m]['gt_tau_mean'] for m in methods]
    gt_tau_stds = [ens[m]['gt_tau_std'] for m in methods]
    regrets = [ens[m]['regret_1_mean'] for m in methods]
    regret_stds = [ens[m]['regret_1_std'] for m in methods]
    pw_taus = [ens[m]['pairwise_tau_mean'] for m in methods]
    
    # Single-seed baseline for reference
    bl_tau = results['experiment_1_baseline']['gt_tau_mean']
    bl_regret = results['experiment_1_baseline']['regrets']['1']['mean']
    
    colors = [COLORS['sras'], COLORS['sras_light'], COLORS['teal'], COLORS['accent'], COLORS['purple'], COLORS['gray']]
    
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))
    x = np.arange(len(methods))
    w = 0.65
    
    # Panel A: GT tau
    bars = axes[0].bar(x, gt_taus, w, yerr=gt_tau_stds, color=colors, edgecolor='white', capsize=2, error_kw={'linewidth': 0.6})
    axes[0].axhline(bl_tau, color=COLORS['baseline'], linestyle='--', linewidth=1, label='Single-seed baseline')
    axes[0].set_ylabel("$\\tau_{GT}$"); axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=5.5); axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(frameon=False, fontsize=6)
    axes[0].text(-0.18, 1.05, 'A', transform=axes[0].transAxes, fontsize=10, fontweight='bold')
    
    # Panel B: Pairwise tau
    bars = axes[1].bar(x, pw_taus, w, color=colors, edgecolor='white')
    bl_pw = results['experiment_1_baseline']['tau_mean']
    axes[1].axhline(bl_pw, color=COLORS['baseline'], linestyle='--', linewidth=1, label='Single-seed baseline')
    axes[1].set_ylabel("$\\tau_{pair}$"); axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=5.5); axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(frameon=False, fontsize=6)
    axes[1].text(-0.18, 1.05, 'B', transform=axes[1].transAxes, fontsize=10, fontweight='bold')
    
    # Panel C: Regret
    bars = axes[2].bar(x, regrets, w, yerr=regret_stds, color=colors, edgecolor='white', capsize=2, error_kw={'linewidth': 0.6})
    axes[2].axhline(bl_regret, color=COLORS['baseline'], linestyle='--', linewidth=1, label='Single-seed baseline')
    axes[2].set_ylabel("Regret@1 (%)"); axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=5.5); axes[2].set_ylim(0, 2.5)
    axes[2].legend(frameon=False, fontsize=6)
    axes[2].text(-0.18, 1.05, 'C', transform=axes[2].transAxes, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_fig('fig7_dumb_ensembles')


# ============================================================================
# NEW Fig 8: BN Recalibration Ablation
# ============================================================================
def fig8_bn_ablation():
    """Effect of BN noise scale on SRAS and baseline."""
    bn = results['experiment_7_bn_ablation']
    
    bn_levels = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    sras_taus = [bn[str(b)]['gt_tau_mean'] for b in bn_levels]
    sras_tau_stds = [bn[str(b)]['gt_tau_std'] for b in bn_levels]
    sras_regrets = [bn[str(b)]['regret_1_mean'] for b in bn_levels]
    
    bl_taus_selected = {0.0: bn['baseline_bn_0.0']['gt_tau_mean'],
                        0.5: bn['baseline_bn_0.5']['gt_tau_mean'],
                        2.0: bn['baseline_bn_2.0']['gt_tau_mean']}
    
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    
    # Panel A: GT tau vs BN noise
    axes[0].errorbar(bn_levels, sras_taus, yerr=sras_tau_stds, fmt='o-', color=COLORS['sras'],
                     linewidth=1.5, markersize=5, capsize=3, label='SRAS (K=5)')
    for bn_val, tau_val in bl_taus_selected.items():
        axes[0].plot(bn_val, tau_val, 's', color=COLORS['baseline'], markersize=7)
    axes[0].plot([], [], 's', color=COLORS['baseline'], label='Baseline (single seed)')
    axes[0].set_xlabel('BN noise scale (0 = perfect recalib.)')
    axes[0].set_ylabel("$\\tau_{GT}$")
    axes[0].legend(frameon=False)
    axes[0].set_ylim(0.65, 0.95)
    axes[0].text(-0.18, 1.05, 'A', transform=axes[0].transAxes, fontsize=10, fontweight='bold')
    axes[0].annotate('Perfect\nrecalib.', xy=(0.0, sras_taus[0]), xytext=(0.5, sras_taus[0]+0.02),
                    fontsize=6, arrowprops=dict(arrowstyle='->', lw=0.5), ha='center')
    axes[0].annotate('No\nrecalib.', xy=(2.0, sras_taus[-1]), xytext=(1.5, sras_taus[-1]-0.03),
                    fontsize=6, arrowprops=dict(arrowstyle='->', lw=0.5), ha='center')
    
    # Panel B: Regret vs BN noise
    axes[1].plot(bn_levels, sras_regrets, 's-', color=COLORS['sras'], linewidth=1.5, markersize=5, label='SRAS (K=5)')
    axes[1].set_xlabel('BN noise scale')
    axes[1].set_ylabel('Regret@1 (%)')
    axes[1].legend(frameon=False)
    axes[1].set_ylim(0, 1.0)
    axes[1].text(-0.18, 1.05, 'B', transform=axes[1].transAxes, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_fig('fig8_bn_ablation')


# ============================================================================
# NEW Fig 9: Independence Check (noise scaling + seed correlation)
# ============================================================================
def fig9_independence():
    """Rank noise scaling vs K and degradation under correlated seeds."""
    ind = results['experiment_8_independence']
    
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    
    # Panel A: Noise scaling
    scaling = ind['scaling_independent']
    Ks = [d['K'] for d in scaling['data']]
    rmses = [d['mean_rmse'] for d in scaling['data']]
    rmse_stds = [d['std_rmse'] for d in scaling['data']]
    alpha = scaling['fit_alpha']
    A = scaling['fit_A']
    
    axes[0].errorbar(Ks, rmses, yerr=rmse_stds, fmt='o', color=COLORS['sras'],
                     markersize=5, capsize=3, label='Empirical', zorder=3)
    k_fine = np.linspace(1, 20, 100)
    axes[0].plot(k_fine, A * k_fine**(-alpha), '--', color=COLORS['sras'], linewidth=1,
                label=f'Fit: $K^{{-{alpha:.2f}}}$', alpha=0.8)
    axes[0].plot(k_fine, A * k_fine**(-0.5), ':', color=COLORS['gray'], linewidth=1,
                label='Theoretical: $K^{-0.50}$', alpha=0.8)
    axes[0].set_xlabel('Number of warmup runs (K)')
    axes[0].set_ylabel('Rank RMSE vs. ground truth')
    axes[0].legend(frameon=False, fontsize=6.5)
    axes[0].set_xlim(0.5, 22)
    axes[0].text(-0.18, 1.05, 'A', transform=axes[0].transAxes, fontsize=10, fontweight='bold')
    
    # Panel B: Seed correlation degradation
    corr_data = ind['seed_correlation']
    corrs = [d['seed_correlation'] for d in corr_data]
    taus = [d['gt_tau_mean'] for d in corr_data]
    tau_stds = [d['gt_tau_std'] for d in corr_data]
    
    axes[1].errorbar(corrs, taus, yerr=tau_stds, fmt='s-', color=COLORS['sras'],
                     linewidth=1.5, markersize=5, capsize=3, label='SRAS (K=5)')
    axes[1].axhline(results['experiment_1_baseline']['gt_tau_mean'],
                    color=COLORS['baseline'], linestyle='--', linewidth=1, label='Single-seed baseline')
    axes[1].set_xlabel('Seed correlation')
    axes[1].set_ylabel("$\\tau_{GT}$")
    axes[1].legend(frameon=False)
    axes[1].set_ylim(0.55, 0.85)
    axes[1].fill_between([0.5, 1.0], 0.55, 0.85, alpha=0.08, color='red')
    axes[1].text(0.75, 0.57, 'SRAS < Baseline', fontsize=6, ha='center', color='red', alpha=0.6)
    axes[1].text(-0.18, 1.05, 'B', transform=axes[1].transAxes, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_fig('fig9_independence')


# ============================================================================
# NEW Fig 10: Search-Space Difficulty
# ============================================================================
def fig10_difficulty():
    """SRAS gain vs search-space difficulty (GT std / top-gap)."""
    diff = results['experiment_9_difficulty']
    gt_stds = sorted([float(k) for k in diff.keys()])
    
    bl_taus = [diff[str(s)]['baseline_tau_mean'] for s in gt_stds]
    sras_taus = [diff[str(s)]['sras_tau_mean'] for s in gt_stds]
    gains = [diff[str(s)]['tau_gain'] for s in gt_stds]
    top_gaps = [diff[str(s)]['top_gap'] for s in gt_stds]
    bl_regrets = [diff[str(s)]['baseline_regret_mean'] for s in gt_stds]
    sras_regrets = [diff[str(s)]['sras_regret_mean'] for s in gt_stds]
    
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    
    # Panel A: tau vs GT std
    axes[0].plot(gt_stds, bl_taus, 'o--', color=COLORS['baseline'], linewidth=1.5, markersize=5, label='Baseline')
    axes[0].plot(gt_stds, sras_taus, 's-', color=COLORS['sras'], linewidth=1.5, markersize=5, label='SRAS (K=5)')
    axes[0].set_xlabel('Search space GT std')
    axes[0].set_ylabel("$\\tau_{GT}$")
    axes[0].legend(frameon=False)
    axes[0].axvline(5.2, color=COLORS['gray'], linestyle=':', linewidth=0.8, alpha=0.5)
    axes[0].text(5.4, 0.42, 'NB-201', fontsize=6, color=COLORS['gray'])
    axes[0].text(-0.18, 1.05, 'A', transform=axes[0].transAxes, fontsize=10, fontweight='bold')
    
    # Panel B: tau gain vs GT std
    axes[1].bar(range(len(gt_stds)), gains, color=COLORS['sras'], edgecolor='white', alpha=0.8)
    axes[1].set_xticks(range(len(gt_stds)))
    axes[1].set_xticklabels([f'{s:.0f}' for s in gt_stds], fontsize=7)
    axes[1].set_xlabel('Search space GT std')
    axes[1].set_ylabel("$\\Delta\\tau_{GT}$ (SRAS - Baseline)")
    axes[1].text(-0.18, 1.05, 'B', transform=axes[1].transAxes, fontsize=10, fontweight='bold')
    
    # Panel C: Regret comparison
    x = np.arange(len(gt_stds)); w = 0.35
    axes[2].bar(x - w/2, bl_regrets, w, color=COLORS['baseline'], edgecolor='white', label='Baseline')
    axes[2].bar(x + w/2, sras_regrets, w, color=COLORS['sras'], edgecolor='white', label='SRAS')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f'{s:.0f}' for s in gt_stds], fontsize=7)
    axes[2].set_xlabel('Search space GT std')
    axes[2].set_ylabel('Regret@1 (%)')
    axes[2].legend(frameon=False, fontsize=6)
    axes[2].text(-0.18, 1.05, 'C', transform=axes[2].transAxes, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_fig('fig10_difficulty')


# ============================================================================
# NEW Fig 11: Failure Modes
# ============================================================================
def fig11_failure_modes():
    """Three failure mode analyses."""
    fm = results['experiment_10_failure_modes']
    
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    
    # Panel A: Extreme warmup noise
    en = fm['extreme_warmup_noise']
    noise_scales = [d['warmup_noise_scale'] for d in en]
    taus = [d['gt_tau_mean'] for d in en]
    axes[0].plot(noise_scales, taus, 'o-', color=COLORS['sras'], linewidth=1.5, markersize=5)
    axes[0].axhline(results['experiment_1_baseline']['gt_tau_mean'],
                    color=COLORS['baseline'], linestyle='--', linewidth=1, label='Baseline')
    axes[0].set_xlabel('Warmup noise scale')
    axes[0].set_ylabel("$\\tau_{GT}$ (SRAS K=5)")
    axes[0].legend(frameon=False)
    axes[0].text(-0.18, 1.05, 'A', transform=axes[0].transAxes, fontsize=10, fontweight='bold')
    axes[0].set_title('Very short warmups', fontsize=8)
    
    # Panel B: Shared data ordering
    sd = fm['shared_data_order']
    shared_fracs = [d['shared_fraction'] for d in sd]
    taus = [d['gt_tau_mean'] for d in sd]
    axes[1].plot(shared_fracs, taus, 's-', color=COLORS['sras'], linewidth=1.5, markersize=5)
    axes[1].axhline(results['experiment_1_baseline']['gt_tau_mean'],
                    color=COLORS['baseline'], linestyle='--', linewidth=1, label='Baseline')
    axes[1].set_xlabel('Shared noise fraction')
    axes[1].set_ylabel("$\\tau_{GT}$ (SRAS K=5)")
    axes[1].legend(frameon=False)
    axes[1].text(-0.18, 1.05, 'B', transform=axes[1].transAxes, fontsize=10, fontweight='bold')
    axes[1].set_title('Correlated seeds', fontsize=8)
    
    # Panel C: Supernet collapse
    sc = fm['supernet_collapse']
    collapse_scales = [d['collapse_scale'] for d in sc]
    eff_stds = [d['effective_std'] for d in sc]
    taus = [d['gt_tau_mean'] for d in sc]
    axes[2].plot(eff_stds, taus, 'D-', color=COLORS['sras'], linewidth=1.5, markersize=5)
    axes[2].axhline(results['experiment_1_baseline']['gt_tau_mean'],
                    color=COLORS['baseline'], linestyle='--', linewidth=1, label='Baseline (normal)')
    axes[2].set_xlabel('Effective GT std (collapsed)')
    axes[2].set_ylabel("$\\tau_{GT}$ (SRAS K=5)")
    axes[2].legend(frameon=False)
    axes[2].invert_xaxis()
    axes[2].text(-0.18, 1.05, 'C', transform=axes[2].transAxes, fontsize=10, fontweight='bold')
    axes[2].set_title('Supernet collapse', fontsize=8)
    
    plt.tight_layout()
    save_fig('fig11_failure_modes')


# ============================================================================
# NEW Fig 12: Two-Stage SRAS
# ============================================================================
def fig12_two_stage():
    """Two-stage prescreening: regret and cost tradeoff."""
    ts = results['experiment_11_two_stage']
    M_values = sorted([int(k) for k in ts.keys()])
    
    gt_taus = [ts[str(m)]['gt_tau_mean'] for m in M_values]
    regrets = [ts[str(m)]['regret_1_mean'] for m in M_values]
    costs = [ts[str(m)]['cost_fraction'] for m in M_values]
    
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    
    # Panel A: GT tau and Regret vs M
    color2 = COLORS['accent']
    ax1 = axes[0]
    ax1.plot(M_values, gt_taus, 'o-', color=COLORS['sras'], linewidth=1.5, markersize=5, label="$\\tau_{GT}$")
    ax1.set_xlabel('Prescreen pool size M')
    ax1.set_ylabel("$\\tau_{GT}$", color=COLORS['sras'])
    ax1.set_ylim(0.83, 0.91)
    ax1.tick_params(axis='y', labelcolor=COLORS['sras'])
    ax1.set_xscale('log')
    
    ax1b = ax1.twinx()
    ax1b.plot(M_values, regrets, 's--', color=color2, linewidth=1.5, markersize=5, label='Regret@1')
    ax1b.set_ylabel('Regret@1 (%)', color=color2)
    ax1b.tick_params(axis='y', labelcolor=color2)
    ax1b.set_ylim(0, 0.5)
    ax1b.spines['top'].set_visible(False)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=6.5)
    axes[0].text(-0.18, 1.05, 'A', transform=axes[0].transAxes, fontsize=10, fontweight='bold')
    
    # Panel B: Cost vs Performance (Pareto)
    axes[1].scatter(costs, gt_taus, c=COLORS['sras'], s=60, zorder=3, edgecolors='white')
    for m, c, t in zip(M_values, costs, gt_taus):
        axes[1].annotate(f'M={m}', (c, t), fontsize=6, ha='left', va='bottom',
                        xytext=(5, 3), textcoords='offset points')
    axes[1].set_xlabel('Relative cost (vs. full SRAS)')
    axes[1].set_ylabel("$\\tau_{GT}$")
    axes[1].set_ylim(0.83, 0.91)
    
    # Highlight sweet spot
    axes[1].axhline(gt_taus[-1] * 0.99, color=COLORS['gray'], linestyle=':', linewidth=0.8, alpha=0.5)
    axes[1].text(0.52, gt_taus[-1] * 0.99 + 0.001, '99% of full SRAS', fontsize=6, color=COLORS['gray'])
    axes[1].text(-0.18, 1.05, 'B', transform=axes[1].transAxes, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_fig('fig12_two_stage')


# ============================================================================
# NEW Fig 13: Tau Calibration Sanity Check
# ============================================================================
def fig13_tau_calibration():
    """Sanity check: distribution of GT tau values."""
    cal = results['experiment_12_tau_calibration']
    
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    
    # Panel A: Distribution of tau_GT (full training and warmup)
    full_vals = cal['full_training_tau_gt']['values']
    warmup_vals = cal['warmup_tau_gt']['values']
    
    axes[0].hist(full_vals, bins=20, alpha=0.6, color=COLORS['baseline'], label='Full training', edgecolor='white')
    axes[0].hist(warmup_vals, bins=20, alpha=0.6, color=COLORS['sras_light'], label='Warmup', edgecolor='white')
    axes[0].axvline(np.mean(full_vals), color=COLORS['baseline'], linestyle='--', linewidth=1.5)
    axes[0].axvline(np.mean(warmup_vals), color=COLORS['sras'], linestyle='--', linewidth=1.5)
    axes[0].set_xlabel("$\\tau_{GT}$ (single seed vs. ground truth)")
    axes[0].set_ylabel('Count (100 seeds)')
    axes[0].legend(frameon=False)
    axes[0].set_title(f"Mean: full={np.mean(full_vals):.3f}, warmup={np.mean(warmup_vals):.3f}", fontsize=7)
    axes[0].text(-0.18, 1.05, 'A', transform=axes[0].transAxes, fontsize=10, fontweight='bold')
    
    # Panel B: Pairwise tau distribution
    pw_vals = cal['full_training_pairwise_tau']['values']
    axes[1].hist(pw_vals, bins=20, alpha=0.7, color=COLORS['accent'], edgecolor='white')
    axes[1].axvline(np.mean(pw_vals), color=COLORS['black'], linestyle='--', linewidth=1.5)
    axes[1].set_xlabel("$\\tau_{pair}$ (seed i vs. seed j)")
    axes[1].set_ylabel('Count')
    axes[1].set_title(f"Mean pairwise $\\tau$ = {np.mean(pw_vals):.3f}", fontsize=7)
    axes[1].text(-0.18, 1.05, 'B', transform=axes[1].transAxes, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_fig('fig13_tau_calibration')


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print("Generating publication figures (v2)...")
    fig1_method_overview()
    fig2_correlation_heatmaps()
    fig3_ablation()
    fig4_budget_comparison()
    fig5_rank_variance()
    fig6_topk_overlap()
    fig7_dumb_ensembles()
    fig8_bn_ablation()
    fig9_independence()
    fig10_difficulty()
    fig11_failure_modes()
    fig12_two_stage()
    fig13_tau_calibration()
    print(f"\nDone! All {13} figures saved to {FIGDIR}/ and {LATEXDIR}/")
