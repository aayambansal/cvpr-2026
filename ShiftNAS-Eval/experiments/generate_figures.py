#!/usr/bin/env python3
"""
Generate all figures for the Fair NAS paper.
Includes new figures for multi-K overlap, NAS algorithm eval, and SASC stability.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
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

STRATEGY_COLORS = {
    'Random': COLORS['black'],
    'Clean (CIFAR-10)': COLORS['blue'],
    'Clean (CIFAR-100)': COLORS['sky_blue'],
    'Worst-Case': COLORS['vermillion'],
    'SASC-Global': COLORS['green'],
    'SASC-Pool': COLORS['orange'],
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Load data
df = pd.read_csv(os.path.join(DATA_DIR, 'architecture_results.csv'))
corr_df = pd.read_csv(os.path.join(DATA_DIR, 'rank_correlations.csv'), index_col=0)
strategy_df = pd.read_csv(os.path.join(DATA_DIR, 'strategy_comparison.csv'))
sensitivity_df = pd.read_csv(os.path.join(DATA_DIR, 'sasc_sensitivity.csv'))
with open(os.path.join(DATA_DIR, 'ranking_overlap.json')) as f:
    overlap_results = json.load(f)
with open(os.path.join(DATA_DIR, 'overlap_multi_k.json')) as f:
    overlap_multi_k = json.load(f)
with open(os.path.join(DATA_DIR, 'nas_algorithm_eval.json')) as f:
    nas_eval = json.load(f)
stability_df = pd.read_csv(os.path.join(DATA_DIR, 'sasc_stability.csv'))


# ============================================================================
# Figure 1: Protocol Overview Diagram
# ============================================================================
def create_protocol_figure():
    fig = plt.figure(figsize=(7.0, 2.8))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.5)
    ax.axis('off')
    
    box_style = dict(boxstyle='round,pad=0.3', facecolor='#E8F4FD', edgecolor='#2C3E50', linewidth=1.2)
    eval_style = dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', edgecolor='#E65100', linewidth=1.2)
    result_style = dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=1.2)
    
    ax.text(5, 3.3, 'Shift-Aware NAS Evaluation Protocol (ShiftNAS-Eval)', 
            fontsize=10, fontweight='bold', ha='center', va='center')
    
    # Phase 1
    ax.text(1.2, 2.4, 'Phase 1: Search', fontsize=8, fontweight='bold', ha='center', bbox=box_style)
    ax.text(1.2, 1.7, 'NAS on\nCIFAR-10', fontsize=7, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#BBDEFB', edgecolor='#1565C0', linewidth=0.8))
    ax.annotate('', xy=(1.2, 1.35), xytext=(1.2, 2.05),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1))
    ax.text(1.2, 1.0, 'Candidate\npool', fontsize=6.5, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='#E3F2FD', edgecolor='#42A5F5', linewidth=0.6))
    
    ax.annotate('', xy=(3.0, 1.7), xytext=(2.2, 1.7),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.5))
    
    # Phase 2
    ax.text(4.8, 2.4, 'Phase 2: Multi-Shift Evaluation', fontsize=8, fontweight='bold', 
            ha='center', bbox=eval_style)
    eval_items = [
        ('CIFAR-10\n(clean)', 3.3, 1.55),
        ('CIFAR-100\n(cross-dataset)', 4.3, 1.55),
        ('ImageNet-16\n(cross-domain)', 5.3, 1.55),
        ('CIFAR-10-C\n(corruption)', 6.3, 1.55),
    ]
    for text, x, y in eval_items:
        ax.text(x, y, text, fontsize=5.5, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#FFF8E1', edgecolor='#F57F17', linewidth=0.6))
    
    ax.annotate('', xy=(7.8, 1.7), xytext=(7.0, 1.7),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.5))
    
    # Phase 3
    ax.text(8.8, 2.4, 'Phase 3: SASC', fontsize=8, fontweight='bold', 
            ha='center', bbox=result_style)
    ax.text(8.8, 1.65, 'Pool-Based\nSASC Score', fontsize=7, ha='center', va='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#C8E6C9', edgecolor='#388E3C', linewidth=0.8))
    ax.text(8.8, 1.0, r'z-score over pool' + '\n' + r'$\alpha$clean + $\beta$cross + $\gamma$robust',
            fontsize=6, ha='center', va='center', style='italic',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='#E8F5E9', edgecolor='#66BB6A', linewidth=0.6))
    
    ax.text(5, 0.3, 'Key finding: Top-10 overlap = 0% for ImageNet-16; Kendall-tau = -0.41 among top architectures',
            fontsize=6.5, ha='center', va='center', style='italic', color='#555555',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#F5F5F5', edgecolor='#BDBDBD', linewidth=0.5))
    
    fig.savefig(os.path.join(FIG_DIR, 'protocol_overview.pdf'), format='pdf')
    fig.savefig(os.path.join(FIG_DIR, 'protocol_overview.png'), format='png')
    plt.close(fig)
    print("  [done] Protocol overview figure")


# ============================================================================
# Figure 2: Rank Correlation Heatmap
# ============================================================================
def create_correlation_heatmap():
    metrics = ['cifar10_clean', 'cifar100_clean', 'imagenet16_clean', 
               'mean_corrupted_acc', 'noise_mean_acc', 'blur_mean_acc',
               'weather_mean_acc', 'digital_mean_acc']
    labels = ['CIFAR-10', 'CIFAR-100', 'ImgNet-16', 
              'Mean Corr.', 'Noise', 'Blur', 'Weather', 'Digital']
    corr_matrix = corr_df.loc[metrics, metrics].values
    
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='RdYlBu_r', center=0.85, vmin=0.65, vmax=1.0,
                xticklabels=labels, yticklabels=labels,
                square=True, linewidths=0.5, cbar_kws={'shrink': 0.8, 'label': 'Spearman $\\rho$'},
                ax=ax, annot_kws={'size': 5.5})
    ax.set_title('Architecture Ranking Correlations', fontsize=9, fontweight='bold', pad=8)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    fig.savefig(os.path.join(FIG_DIR, 'rank_correlation_heatmap.pdf'), format='pdf')
    fig.savefig(os.path.join(FIG_DIR, 'rank_correlation_heatmap.png'), format='png')
    plt.close(fig)
    print("  [done] Rank correlation heatmap")


# ============================================================================  
# Figure 3: Strategy Comparison Bar Chart (updated with SASC-Pool)
# ============================================================================
def create_strategy_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    
    strategies = ['Random', 'Clean (CIFAR-10)', 'Clean (CIFAR-100)', 'Worst-Case', 'SASC-Global', 'SASC-Pool']
    short_names = ['Rand', 'Clean\n(C10)', 'Clean\n(C100)', 'Worst\nCase', 'SASC\nGlobal', 'SASC\nPool']
    
    metrics_sets = [
        ('cifar10_clean', 'CIFAR-10 Clean Acc (%)'),
        ('cifar100_clean', 'CIFAR-100 Clean Acc (%)'),
        ('imagenet16_clean', 'ImageNet-16 Acc (%)'),
    ]
    
    for idx, (metric, title) in enumerate(metrics_sets):
        ax = axes[idx]
        means = []
        stds = []
        colors = []
        for s in strategies:
            row = strategy_df[strategy_df['strategy'] == s]
            if len(row) > 0:
                means.append(row[f'{metric}_mean'].values[0])
                stds.append(row[f'{metric}_std'].values[0])
            else:
                means.append(0)
                stds.append(0)
            colors.append(STRATEGY_COLORS.get(s, COLORS['black']))
        
        bars = ax.bar(range(len(strategies)), means, yerr=stds, capsize=2, 
                      color=colors, alpha=0.85, edgecolor='black', linewidth=0.5,
                      error_kw={'linewidth': 0.8})
        
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(short_names, fontsize=5.5)
        ax.set_title(title, fontsize=8, fontweight='bold')
        ax.set_ylabel('Accuracy (%)' if idx == 0 else '')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    for idx, ax in enumerate(axes):
        ax.text(-0.15, 1.05, chr(65 + idx), transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top')
    
    plt.tight_layout(w_pad=1.5)
    fig.savefig(os.path.join(FIG_DIR, 'strategy_comparison.pdf'), format='pdf')
    fig.savefig(os.path.join(FIG_DIR, 'strategy_comparison.png'), format='png')
    plt.close(fig)
    print("  [done] Strategy comparison figure")


# ============================================================================
# Figure 4: Multi-K Overlap Curves (NEW — addresses reviewer point 5)
# ============================================================================
def create_multi_k_overlap():
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    
    metrics_to_plot = {
        'cifar100_clean': ('CIFAR-100', COLORS['blue'], '-', 'o'),
        'imagenet16_clean': ('ImageNet-16', COLORS['vermillion'], '-', 's'),
        'mean_corrupted_acc': ('Mean Corrupted', COLORS['green'], '--', '^'),
    }
    
    for metric, (label, color, ls, marker) in metrics_to_plot.items():
        ks = sorted([int(k) for k in overlap_multi_k[metric].keys()])
        overlaps = [overlap_multi_k[metric][str(k)] * 100 for k in ks]
        ax.plot(ks, overlaps, color=color, linestyle=ls, marker=marker, 
                markersize=4, linewidth=1.5, label=label)
    
    ax.set_xlabel('Top-K', fontsize=8)
    ax.set_ylabel('Overlap with CIFAR-10 Top-K (%)', fontsize=8)
    ax.set_title('Ranking Overlap vs. K', fontsize=9, fontweight='bold')
    ax.axhline(y=50, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)
    ax.legend(fontsize=7, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(-5, 105)
    ax.set_xscale('log')
    ax.set_xticks([10, 25, 50, 100, 200, 500])
    ax.set_xticklabels(['10', '25', '50', '100', '200', '500'])
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'overlap_multi_k.pdf'), format='pdf')
    fig.savefig(os.path.join(FIG_DIR, 'overlap_multi_k.png'), format='png')
    plt.close(fig)
    print("  [done] Multi-K overlap curves")


# ============================================================================
# Figure 5: Ranking Overlap Bar Chart (original, K=100 only)
# ============================================================================
def create_ranking_overlap():
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    metrics = ['cifar100_clean', 'imagenet16_clean', 'mean_corrupted_acc',
               'noise_mean_acc', 'blur_mean_acc', 'weather_mean_acc', 'digital_mean_acc']
    labels = ['CIFAR-100', 'ImgNet-16', 'Mean Corr.', 'Noise', 'Blur', 'Weather', 'Digital']
    overlaps = [overlap_results[m] * 100 for m in metrics]
    
    bar_colors = [COLORS['blue'], COLORS['blue'], COLORS['orange'], 
                  COLORS['vermillion'], COLORS['sky_blue'], COLORS['green'], COLORS['purple']]
    
    bars = ax.barh(range(len(metrics)), overlaps, color=bar_colors, alpha=0.85,
                   edgecolor='black', linewidth=0.5, height=0.7)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Top-100 Overlap with CIFAR-10 (%)', fontsize=8)
    ax.set_title('Ranking Overlap: Clean CIFAR-10 vs Other Metrics', fontsize=8, fontweight='bold')
    ax.axvline(x=50, color='gray', linestyle='--', linewidth=0.6, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, overlaps):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{val:.0f}%', va='center', fontsize=6.5)
    ax.set_xlim(0, 105)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'ranking_overlap.pdf'), format='pdf')
    fig.savefig(os.path.join(FIG_DIR, 'ranking_overlap.png'), format='png')
    plt.close(fig)
    print("  [done] Ranking overlap (K=100) figure")


# ============================================================================
# Figure 6: NAS Algorithm End-to-End Evaluation (NEW — addresses reviewer point 3)
# ============================================================================
def create_nas_algorithm_eval():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    
    algorithms = list(nas_eval.keys())
    short_algs = ['RS', 'RE', 'DARTS', 'ENAS']
    metrics_plot = [
        ('cifar10', 'CIFAR-10 Acc (%)'),
        ('imagenet16', 'ImageNet-16 Acc (%)'),
        ('corruption_gap', 'Corruption Gap'),
    ]
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    for idx, (metric, title) in enumerate(metrics_plot):
        ax = axes[idx]
        clean_vals = [nas_eval[a]['clean_top10'][metric] for a in algorithms]
        sasc_vals = [nas_eval[a]['sasc_top10'][metric] for a in algorithms]
        
        bars1 = ax.bar(x - width/2, clean_vals, width, label='Clean Select', 
                       color=COLORS['blue'], alpha=0.85, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, sasc_vals, width, label='SASC-Pool', 
                       color=COLORS['green'], alpha=0.85, edgecolor='black', linewidth=0.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(short_algs, fontsize=7)
        ax.set_title(title, fontsize=8, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if idx == 0:
            ax.set_ylabel('Score', fontsize=7)
            ax.legend(fontsize=6, loc='lower right')
        
        ax.text(-0.15, 1.05, chr(65 + idx), transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top')
    
    plt.tight_layout(w_pad=1.5)
    fig.savefig(os.path.join(FIG_DIR, 'nas_algorithm_eval.pdf'), format='pdf')
    fig.savefig(os.path.join(FIG_DIR, 'nas_algorithm_eval.png'), format='png')
    plt.close(fig)
    print("  [done] NAS algorithm evaluation figure")


# ============================================================================
# Figure 7: Scatter plots
# ============================================================================
def create_scatter_plots():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    sample = df.sample(2000, random_state=42)
    
    scatter_configs = [
        ('cifar10_clean', 'cifar100_clean', 'CIFAR-100 Acc (%)', COLORS['blue']),
        ('cifar10_clean', 'imagenet16_clean', 'ImageNet-16 Acc (%)', COLORS['green']),
        ('cifar10_clean', 'mean_corrupted_acc', 'Mean Corrupted Acc (%)', COLORS['vermillion']),
    ]
    
    for idx, (x, y, ylabel, color) in enumerate(scatter_configs):
        ax = axes[idx]
        ax.scatter(sample[x], sample[y], c=color, alpha=0.15, s=3, rasterized=True)
        rho = stats.spearmanr(df[x], df[y]).correlation
        ax.text(0.05, 0.95, f'$\\rho$ = {rho:.3f}', transform=ax.transAxes, fontsize=7,
                va='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.8))
        ax.set_xlabel('CIFAR-10 Clean Acc (%)', fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(-0.15, 1.05, chr(65 + idx), transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top')
    
    plt.tight_layout(w_pad=1.5)
    fig.savefig(os.path.join(FIG_DIR, 'scatter_correlations.pdf'), format='pdf')
    fig.savefig(os.path.join(FIG_DIR, 'scatter_correlations.png'), format='png')
    plt.close(fig)
    print("  [done] Scatter plots")


# ============================================================================
# Figure 8: Per-corruption radar chart
# ============================================================================
def create_corruption_radar():
    fig, ax = plt.subplots(figsize=(3.5, 3.0), subplot_kw=dict(polar=True))
    
    categories = ['Noise', 'Blur', 'Weather', 'Digital']
    N = len(categories)
    
    strategy_data = {}
    for _, row in strategy_df.iterrows():
        name = row['strategy']
        if name in ['Clean (CIFAR-10)', 'Worst-Case', 'SASC-Global']:
            strategy_data[name] = [
                row['noise_mean_acc_mean'],
                row['blur_mean_acc_mean'],
                row['weather_mean_acc_mean'],
                row['digital_mean_acc_mean'],
            ]
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    strategy_styles = {
        'Clean (CIFAR-10)': {'color': COLORS['blue'], 'ls': '-', 'marker': 'o'},
        'Worst-Case': {'color': COLORS['vermillion'], 'ls': '--', 'marker': 's'},
        'SASC-Global': {'color': COLORS['green'], 'ls': '-', 'marker': '^'},
    }
    
    for name, values in strategy_data.items():
        values_plot = values + values[:1]
        style = strategy_styles[name]
        ax.plot(angles, values_plot, color=style['color'], linestyle=style['ls'],
                linewidth=1.5, marker=style['marker'], markersize=4, label=name)
        ax.fill(angles, values_plot, color=style['color'], alpha=0.08)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylim(45, 82)
    ax.set_yticks([50, 60, 70, 80])
    ax.set_yticklabels(['50', '60', '70', '80'], fontsize=6)
    ax.set_title('Per-Corruption Category\nAccuracy (%)', fontsize=8, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=6, bbox_to_anchor=(1.35, -0.05))
    
    fig.savefig(os.path.join(FIG_DIR, 'corruption_radar.pdf'), format='pdf')
    fig.savefig(os.path.join(FIG_DIR, 'corruption_radar.png'), format='png')
    plt.close(fig)
    print("  [done] Corruption radar chart")


# ============================================================================
# Figure 9: Architecture properties vs corruption gap
# ============================================================================
def create_arch_properties():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    
    properties = [
        ('n_skip', 'Skip Connections', COLORS['green']),
        ('n_conv3', '3x3 Convolutions', COLORS['blue']),
        ('n_none', 'None Operations', COLORS['vermillion']),
    ]
    
    for idx, (prop, label, color) in enumerate(properties):
        ax = axes[idx]
        grouped = df.groupby(prop).agg({'corruption_gap': ['mean', 'std', 'count']}).reset_index()
        grouped.columns = [prop, 'cg_mean', 'cg_std', 'count']
        grouped['cg_se'] = grouped['cg_std'] / np.sqrt(grouped['count'])
        
        ax.bar(grouped[prop], grouped['cg_mean'], yerr=grouped['cg_se'] * 1.96,
               color=color, alpha=0.8, edgecolor='black', linewidth=0.5, capsize=3,
               error_kw={'linewidth': 0.8})
        ax.set_xlabel(f'Number of {label}', fontsize=7)
        ax.set_ylabel('Corruption Gap' if idx == 0 else '', fontsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks(grouped[prop].values)
        
        rho = stats.spearmanr(df[prop], df['corruption_gap']).correlation
        ax.text(0.95, 0.95, f'$\\rho$ = {rho:.2f}', transform=ax.transAxes, fontsize=6.5,
                va='top', ha='right', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='gray', alpha=0.8))
        ax.text(-0.15, 1.05, chr(65 + idx), transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top')
    
    plt.tight_layout(w_pad=1.5)
    fig.savefig(os.path.join(FIG_DIR, 'arch_properties.pdf'), format='pdf')
    fig.savefig(os.path.join(FIG_DIR, 'arch_properties.png'), format='png')
    plt.close(fig)
    print("  [done] Architecture properties figure")


# ============================================================================
# Figure 10: SASC Sensitivity Heatmap
# ============================================================================
def create_sasc_sensitivity():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    
    sensitivity_df['beta_frac'] = sensitivity_df['beta'] / (1 - sensitivity_df['alpha'] + 1e-8)
    
    for idx, (metric, title, cmap) in enumerate([
        ('cifar10_mean', 'Clean CIFAR-10 Acc (%)', 'Blues'),
        ('corrupted_mean', 'Mean Corrupted Acc (%)', 'Greens'),
    ]):
        ax = axes[idx]
        pivot = sensitivity_df.pivot_table(values=metric, index='alpha', 
                                           columns=sensitivity_df['beta'].round(2),
                                           aggfunc='mean')
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap=cmap, ax=ax,
                    cbar_kws={'shrink': 0.8}, annot_kws={'size': 6},
                    linewidths=0.3)
        ax.set_title(title, fontsize=8, fontweight='bold')
        ax.set_xlabel('$\\beta$ (cross-dataset weight)', fontsize=7)
        ax.set_ylabel('$\\alpha$ (clean weight)' if idx == 0 else '', fontsize=7)
        ax.text(-0.15, 1.05, chr(65 + idx), transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top')
    
    plt.tight_layout(w_pad=2)
    fig.savefig(os.path.join(FIG_DIR, 'sasc_sensitivity.pdf'), format='pdf')
    fig.savefig(os.path.join(FIG_DIR, 'sasc_sensitivity.png'), format='png')
    plt.close(fig)
    print("  [done] SASC sensitivity heatmap")


# ============================================================================
# Figure 11: SASC Pool Stability (NEW — addresses reviewer point 4)
# ============================================================================
def create_sasc_stability():
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    ax.errorbar(stability_df['pool_size'], stability_df['overlap_mean'] * 100,
                yerr=stability_df['overlap_std'] * 100,
                color=COLORS['green'], marker='o', markersize=5,
                capsize=4, linewidth=1.5, capthick=1)
    
    ax.set_xlabel('Candidate Pool Size', fontsize=8)
    ax.set_ylabel('Overlap with Global\nSASC Top-100 (%)', fontsize=8)
    ax.set_title('SASC-Pool Stability', fontsize=9, fontweight='bold')
    ax.axhline(y=50, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xscale('log')
    ax.set_xticks(stability_df['pool_size'].values)
    ax.set_xticklabels([str(int(x)) for x in stability_df['pool_size'].values])
    ax.set_ylim(-5, 110)
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'sasc_stability.pdf'), format='pdf')
    fig.savefig(os.path.join(FIG_DIR, 'sasc_stability.png'), format='png')
    plt.close(fig)
    print("  [done] SASC pool stability figure")


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print("Generating all figures...")
    create_protocol_figure()
    create_correlation_heatmap()
    create_strategy_comparison()
    create_multi_k_overlap()
    create_ranking_overlap()
    create_nas_algorithm_eval()
    create_scatter_plots()
    create_corruption_radar()
    create_arch_properties()
    create_sasc_sensitivity()
    create_sasc_stability()
    print(f"\nAll figures saved to {FIG_DIR}")
