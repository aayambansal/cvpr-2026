#!/usr/bin/env python3
"""
Analyze real CIFAR-10-C evaluation results and generate figures + data
for the paper's Real Validation section.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import json
import os
import warnings
warnings.filterwarnings('ignore')

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
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.2,
    'patch.linewidth': 0.8,
})

# Colorblind-safe Okabe-Ito palette
COLORS = {
    'orange': '#E69F00',
    'sky_blue': '#56B4E9',
    'green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'vermillion': '#D55E00',
    'purple': '#CC79A7',
    'black': '#000000',
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


def load_real_results():
    """Load and clean real CIFAR-10-C results."""
    with open(os.path.join(DATA_DIR, 'real_results.json')) as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    
    # Separate functional (>10% clean acc) from degenerate architectures
    functional_mask = df['cifar10_clean'] > 15.0  # above chance
    df_func = df[functional_mask].copy()
    df_degen = df[~functional_mask].copy()
    
    print(f"Total architectures: {len(df)}")
    print(f"Functional (>15% clean acc): {len(df_func)}")
    print(f"Degenerate (<=15% clean acc): {len(df_degen)}")
    
    return df, df_func, df_degen


def compute_real_statistics(df_func):
    """Compute key statistics from functional real architectures."""
    analysis = {}
    
    # Basic statistics
    analysis['n_total'] = len(df_func)
    analysis['clean_acc_mean'] = float(df_func['cifar10_clean'].mean())
    analysis['clean_acc_std'] = float(df_func['cifar10_clean'].std())
    analysis['clean_acc_min'] = float(df_func['cifar10_clean'].min())
    analysis['clean_acc_max'] = float(df_func['cifar10_clean'].max())
    analysis['cg_mean'] = float(df_func['corruption_gap'].mean())
    analysis['cg_std'] = float(df_func['corruption_gap'].std())
    analysis['cg_min'] = float(df_func['corruption_gap'].min())
    analysis['cg_max'] = float(df_func['corruption_gap'].max())
    analysis['corrupted_acc_mean'] = float(df_func['mean_corrupted_acc'].mean())
    
    # Rank correlation: clean accuracy vs corruption gap
    rho_cg, pval_cg = stats.spearmanr(df_func['cifar10_clean'], df_func['corruption_gap'])
    analysis['rho_clean_vs_cg'] = float(rho_cg)
    analysis['pval_clean_vs_cg'] = float(pval_cg)
    
    # Rank correlation: clean accuracy vs mean corrupted accuracy
    rho_corr, pval_corr = stats.spearmanr(df_func['cifar10_clean'], df_func['mean_corrupted_acc'])
    analysis['rho_clean_vs_corrupted'] = float(rho_corr)
    analysis['pval_clean_vs_corrupted'] = float(pval_corr)
    
    # Per-category correlations with clean accuracy
    for cat in ['noise', 'blur', 'weather', 'digital']:
        rho, pval = stats.spearmanr(df_func['cifar10_clean'], df_func[f'{cat}_mean_acc'])
        analysis[f'rho_clean_vs_{cat}'] = float(rho)
    
    # Severity degradation profile for top architectures (top-5 by clean acc)
    top5 = df_func.nlargest(5, 'cifar10_clean')
    severity_profile = {}
    for sev in range(1, 6):
        sev_cols = [c for c in df_func.columns if c.endswith(f'_s{sev}')]
        severity_profile[f's{sev}_mean'] = float(top5[sev_cols].mean().mean())
    analysis['top5_severity_profile'] = severity_profile
    
    # Architecture properties of top vs bottom functional
    top5_clean = df_func.nlargest(5, 'cifar10_clean')
    # For "lowest CG" among functional archs with reasonable accuracy (>80%)
    high_acc = df_func[df_func['cifar10_clean'] > 80.0]
    if len(high_acc) >= 5:
        lowest_cg = high_acc.nsmallest(5, 'corruption_gap')
    else:
        lowest_cg = df_func.nsmallest(5, 'corruption_gap')
    
    analysis['top5_by_clean'] = [
        {
            'arch_id': int(r['arch_id']),
            'clean_acc': float(r['cifar10_clean']),
            'cg': float(r['corruption_gap']),
            'mean_corr_acc': float(r['mean_corrupted_acc']),
            'noise': float(r['noise_mean_acc']),
            'blur': float(r['blur_mean_acc']),
            'weather': float(r['weather_mean_acc']),
            'digital': float(r['digital_mean_acc']),
            'n_conv3': int(r['n_conv3']),
            'n_skip': int(r['n_skip']),
        }
        for _, r in top5_clean.iterrows()
    ]
    
    analysis['lowest_cg_high_acc'] = [
        {
            'arch_id': int(r['arch_id']),
            'clean_acc': float(r['cifar10_clean']),
            'cg': float(r['corruption_gap']),
            'mean_corr_acc': float(r['mean_corrupted_acc']),
        }
        for _, r in lowest_cg.iterrows()
    ]
    
    # Key finding: among top architectures, rank by clean != rank by corrupted
    if len(df_func) >= 10:
        top10_clean = set(df_func.nlargest(10, 'cifar10_clean')['arch_id'].values)
        top10_corr = set(df_func.nlargest(10, 'mean_corrupted_acc')['arch_id'].values)
        top10_overlap = len(top10_clean & top10_corr) / 10
        analysis['top10_clean_vs_corrupted_overlap'] = float(top10_overlap)
    
    print("\n=== Real CIFAR-10-C Statistics (functional architectures) ===")
    print(f"Clean accuracy: {analysis['clean_acc_mean']:.1f}% +/- {analysis['clean_acc_std']:.1f}%")
    print(f"  Range: [{analysis['clean_acc_min']:.1f}%, {analysis['clean_acc_max']:.1f}%]")
    print(f"Corruption Gap: {analysis['cg_mean']:.1f} +/- {analysis['cg_std']:.1f}")
    print(f"  Range: [{analysis['cg_min']:.1f}, {analysis['cg_max']:.1f}]")
    print(f"Spearman rho (clean vs CG): {analysis['rho_clean_vs_cg']:.3f} (p={analysis['pval_clean_vs_cg']:.2e})")
    print(f"Spearman rho (clean vs corrupted acc): {analysis['rho_clean_vs_corrupted']:.3f}")
    if 'top10_clean_vs_corrupted_overlap' in analysis:
        print(f"Top-10 overlap (clean vs corrupted): {analysis['top10_clean_vs_corrupted_overlap']:.0%}")
    
    print("\nTop-5 by clean accuracy:")
    for a in analysis['top5_by_clean']:
        print(f"  Arch {a['arch_id']}: clean={a['clean_acc']:.1f}%, CG={a['cg']:.1f}, "
              f"noise={a['noise']:.1f}, blur={a['blur']:.1f}, weather={a['weather']:.1f}, digital={a['digital']:.1f}")
    
    print(f"\nSeverity degradation (top-5 avg):")
    for sev, val in analysis['top5_severity_profile'].items():
        print(f"  {sev}: {val:.1f}%")
    
    return analysis


def create_real_scatter_plot(df_func, df_degen):
    """Create scatter plot: clean accuracy vs corruption gap with real data."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    
    # Panel A: Clean acc vs Corruption Gap
    ax = axes[0]
    ax.scatter(df_func['cifar10_clean'], df_func['corruption_gap'], 
               c=COLORS['vermillion'], alpha=0.7, s=25, edgecolors='black', 
               linewidth=0.3, zorder=5, label='Functional')
    if len(df_degen) > 0:
        ax.scatter(df_degen['cifar10_clean'], df_degen['corruption_gap'],
                   c=COLORS['black'], alpha=0.3, s=15, marker='x', 
                   zorder=3, label='Degenerate')
    
    rho = stats.spearmanr(df_func['cifar10_clean'], df_func['corruption_gap']).correlation
    ax.text(0.05, 0.95, f'$\\rho$ = {rho:.2f}\n(n={len(df_func)})', 
            transform=ax.transAxes, fontsize=7, va='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.9))
    
    ax.set_xlabel('Clean CIFAR-10 Accuracy (%)', fontsize=8)
    ax.set_ylabel('Corruption Gap ($\\Delta_{CG}$)', fontsize=8)
    ax.set_title('Real CIFAR-10-C: Accuracy vs Robustness', fontsize=8, fontweight='bold')
    ax.legend(fontsize=6, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Panel B: Per-corruption category breakdown for top architectures
    ax = axes[1]
    top_archs = df_func.nlargest(10, 'cifar10_clean')
    categories = ['noise', 'blur', 'weather', 'digital']
    cat_labels = ['Noise', 'Blur', 'Weather', 'Digital']
    cat_colors = [COLORS['vermillion'], COLORS['sky_blue'], COLORS['green'], COLORS['purple']]
    
    x = np.arange(len(categories))
    means = [top_archs[f'{cat}_mean_acc'].mean() for cat in categories]
    stds = [top_archs[f'{cat}_mean_acc'].std() for cat in categories]
    
    bars = ax.bar(x, means, yerr=stds, capsize=3, color=cat_colors, alpha=0.85,
                  edgecolor='black', linewidth=0.5, error_kw={'linewidth': 0.8})
    
    # Add clean accuracy reference line
    clean_mean = top_archs['cifar10_clean'].mean()
    ax.axhline(y=clean_mean, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.text(len(categories)-0.5, clean_mean + 1, f'Clean: {clean_mean:.1f}%', 
            fontsize=6, ha='right', va='bottom', color='gray')
    
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=7)
    ax.set_ylabel('Mean Corrupted Accuracy (%)', fontsize=8)
    ax.set_title('Top-10 Architectures: Per-Category', fontsize=8, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for idx, ax in enumerate(axes):
        ax.text(-0.15, 1.05, chr(65 + idx), transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top')
    
    plt.tight_layout(w_pad=2)
    fig.savefig(os.path.join(FIG_DIR, 'real_cifar10c_validation.pdf'), format='pdf')
    fig.savefig(os.path.join(FIG_DIR, 'real_cifar10c_validation.png'), format='png')
    plt.close(fig)
    print("  [done] Real CIFAR-10-C validation figure")


def create_severity_degradation_plot(df_func):
    """Create severity degradation plot for top architectures."""
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    # Group architectures
    top5 = df_func.nlargest(5, 'cifar10_clean')
    mid5 = df_func.iloc[len(df_func)//2 - 2 : len(df_func)//2 + 3] if len(df_func) > 10 else df_func.head(5)
    
    groups = {
        'Top-5 (clean)': (top5, COLORS['vermillion'], '-', 'o'),
    }
    
    # Also pick the architecture with lowest CG among high-acc
    high_acc = df_func[df_func['cifar10_clean'] > 85.0]
    if len(high_acc) >= 5:
        lowest_cg = high_acc.nsmallest(5, 'corruption_gap')
        groups['Lowest-CG (>85% acc)'] = (lowest_cg, COLORS['green'], '--', 's')
    
    for label, (group, color, ls, marker) in groups.items():
        sev_means = []
        for sev in range(1, 6):
            sev_cols = [c for c in df_func.columns if c.endswith(f'_s{sev}')]
            sev_means.append(group[sev_cols].mean().mean())
        ax.plot(range(1, 6), sev_means, color=color, linestyle=ls, marker=marker,
                markersize=4, linewidth=1.5, label=label)
    
    ax.set_xlabel('Corruption Severity', fontsize=8)
    ax.set_ylabel('Mean Accuracy (%)', fontsize=8)
    ax.set_title('Accuracy vs Severity (Real)', fontsize=9, fontweight='bold')
    ax.set_xticks(range(1, 6))
    ax.legend(fontsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'real_severity_degradation.pdf'), format='pdf')
    fig.savefig(os.path.join(FIG_DIR, 'real_severity_degradation.png'), format='png')
    plt.close(fig)
    print("  [done] Severity degradation figure")


def main():
    print("=" * 60)
    print("Analyzing Real CIFAR-10-C Results")
    print("=" * 60)
    
    # Load data
    df, df_func, df_degen = load_real_results()
    
    # Compute statistics
    analysis = compute_real_statistics(df_func)
    
    # Save analysis
    with open(os.path.join(DATA_DIR, 'real_analysis.json'), 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to {os.path.join(DATA_DIR, 'real_analysis.json')}")
    
    # Generate figures
    print("\nGenerating figures...")
    create_real_scatter_plot(df_func, df_degen)
    create_severity_degradation_plot(df_func)
    
    print("\nDone!")
    return analysis


if __name__ == '__main__':
    analysis = main()
