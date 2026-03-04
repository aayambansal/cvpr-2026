#!/usr/bin/env python3
"""
Generate V2 publication-quality figures for the MemoryNAS paper.
Uses real training results, measured GPU memory, and baseline comparisons.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from scipy.stats import pearsonr, spearmanr
import os

# Publication style (CVPR two-column)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Okabe-Ito colorblind-safe palette
C = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'red': '#D55E00',
    'purple': '#CC79A7',
    'yellow': '#F0E442',
    'cyan': '#56B4E9',
    'black': '#000000',
    'grey': '#999999',
}

BASE = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(os.path.dirname(BASE), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_results():
    """Load all experiment results. Falls back gracefully if training not yet done."""
    results = {}
    for name in ['training_results', 'memory_validation', 'baselines_ablations']:
        path = os.path.join(BASE, f'{name}.json')
        if os.path.exists(path):
            with open(path) as f:
                results[name] = json.load(f)
            print(f"  Loaded {name}.json")
        else:
            print(f"  WARNING: {name}.json not found")
            results[name] = None
    return results


def save_fig(fig, name):
    """Save as both PDF and PNG."""
    for ext in ['pdf', 'png']:
        path = os.path.join(FIGURES_DIR, f'{name}.{ext}')
        fig.savefig(path, format=ext)
    print(f"  Saved {name}")
    plt.close(fig)


# ============================================================================
# Figure 1: Memory Estimator Validation (measured vs analytical)
# ============================================================================

def fig_memory_validation(data):
    """Scatter: measured GPU memory vs analytical estimate for 50 random architectures."""
    if data is None:
        print("  Skipping fig_memory_validation (no data)")
        return

    results = data['results']
    summary = data['summary']

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # Panel (a): Total memory (activations + weights)
    ax = axes[0]
    meas = np.array([r['measured_total_mb'] for r in results])
    anal = np.array([r['analytical_total_mb'] for r in results])
    ax.scatter(anal, meas, c=C['blue'], s=18, alpha=0.7, edgecolors='white', linewidths=0.3, zorder=3)
    # Identity line
    lo, hi = min(anal.min(), meas.min()) * 0.8, max(anal.max(), meas.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], '--', color=C['grey'], linewidth=0.8, label='y = x')
    # Fit line
    m, b = np.polyfit(anal, meas, 1)
    x_fit = np.linspace(lo, hi, 100)
    ax.plot(x_fit, m * x_fit + b, '-', color=C['red'], linewidth=1.0,
            label=f'Fit (slope={m:.2f})')
    stats_t = summary['total_memory_stats']
    ax.set_xlabel('Analytical Estimate (MB)')
    ax.set_ylabel('Measured GPU Peak (MB)')
    ax.set_title(f'(a) Total Memory\n'
                 f'Pearson r={stats_t["pearson_r"]:.3f}, '
                 f'Spearman ρ={stats_t["spearman_r"]:.3f}')
    ax.legend(loc='upper left', framealpha=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    # Panel (b): Ranking accuracy (sorted by estimator vs sorted by measured)
    ax = axes[1]
    rank_anal = np.argsort(np.argsort(anal))
    rank_meas = np.argsort(np.argsort(meas))
    ax.scatter(rank_anal, rank_meas, c=C['green'], s=18, alpha=0.7, edgecolors='white', linewidths=0.3, zorder=3)
    ax.plot([0, 50], [0, 50], '--', color=C['grey'], linewidth=0.8)
    rho = summary.get('ranking_spearman', spearmanr(anal, meas).statistic)
    ax.set_xlabel('Rank by Analytical Estimate')
    ax.set_ylabel('Rank by Measured GPU Memory')
    ax.set_title(f'(b) Ranking Accuracy\nSpearman ρ = {rho:.4f}')
    ax.set_xlim(-2, 52)
    ax.set_ylim(-2, 52)
    ax.set_aspect('equal')

    fig.tight_layout()
    save_fig(fig, 'fig_memory_validation')


# ============================================================================
# Figure 2: Trained Architecture Comparison (CIFAR-10 and CIFAR-100)
# ============================================================================

def fig_training_results(data):
    """Bar chart: trained accuracy for FLOPs-Pareto vs MemoryNAS architectures."""
    if data is None:
        print("  Skipping fig_training_results (no data)")
        return

    # Separate groups
    fp_names = [n for n in data if n.startswith('FP-')]
    mn_names = [n for n in data if n.startswith('MN-')]
    bl_names = [n for n in data if n.startswith('MBv2')]

    all_names = fp_names + mn_names + bl_names
    n = len(all_names)

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.5), sharex=True)

    for idx, (dataset, key) in enumerate([('CIFAR-10', 'cifar10_acc'), ('CIFAR-100', 'cifar100_acc')]):
        ax = axes[idx]
        accs = [data[name][key] for name in all_names]
        mems = [data[name]['gpu_peak_total_bs32'] / 1e6 for name in all_names]

        # Color by group
        colors = []
        for name in all_names:
            if name.startswith('FP-'):
                colors.append(C['blue'])
            elif name.startswith('MN-'):
                colors.append(C['green'])
            else:
                colors.append(C['grey'])

        x = np.arange(n)
        bars = ax.bar(x, accs, color=colors, edgecolor='white', linewidth=0.5, width=0.7)

        # Add memory labels on top of bars
        for i, (acc, mem) in enumerate(zip(accs, mems)):
            ax.text(i, acc + 0.3, f'{mem:.0f}', ha='center', va='bottom', fontsize=5.5,
                    color=C['black'], rotation=45)

        ax.set_ylabel(f'{dataset} Accuracy (%)')
        ax.set_ylim(min(accs) - 3, max(accs) + 4)
        if idx == 0:
            ax.set_title('Trained Accuracy vs GPU Peak Memory (bs=32, numbers on bars in MB)')

        # Legend on first panel
        if idx == 0:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=C['blue'], label='FLOPs-Pareto'),
                Patch(facecolor=C['green'], label='MemoryNAS'),
                Patch(facecolor=C['grey'], label='Baseline (MBv2)'),
            ]
            ax.legend(handles=legend_elements, loc='lower right', framealpha=0.8)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(all_names, rotation=45, ha='right', fontsize=6)
    fig.tight_layout()
    save_fig(fig, 'fig_training_results')


# ============================================================================
# Figure 3: Accuracy vs Peak Memory Pareto Front (trained models)
# ============================================================================

def fig_pareto_trained(data):
    """Scatter: accuracy vs measured peak memory for trained architectures."""
    if data is None:
        print("  Skipping fig_pareto_trained (no data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    for idx, (dataset, key) in enumerate([('CIFAR-10', 'cifar10_acc'), ('CIFAR-100', 'cifar100_acc')]):
        ax = axes[idx]

        for name, info in data.items():
            acc = info[key]
            mem = info['gpu_peak_total_bs32'] / 1e6
            flops = info['flops_m']

            if name.startswith('FP-'):
                ax.scatter(mem, acc, c=C['blue'], s=40, marker='o', zorder=3, edgecolors='white', linewidths=0.5)
                ax.annotate(name, (mem, acc), fontsize=5, xytext=(3, 3), textcoords='offset points')
            elif name.startswith('MN-'):
                ax.scatter(mem, acc, c=C['green'], s=40, marker='s', zorder=3, edgecolors='white', linewidths=0.5)
                ax.annotate(name, (mem, acc), fontsize=5, xytext=(3, 3), textcoords='offset points')
            else:
                ax.scatter(mem, acc, c=C['grey'], s=40, marker='^', zorder=3, edgecolors='white', linewidths=0.5)
                ax.annotate(name, (mem, acc), fontsize=5, xytext=(3, 3), textcoords='offset points')

        ax.set_xlabel('GPU Peak Memory at bs=32 (MB)')
        ax.set_ylabel(f'{dataset} Accuracy (%)')
        ax.set_title(f'({chr(97+idx)}) {dataset}')

        if idx == 0:
            from matplotlib.lines import Line2D
            legend = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=C['blue'], markersize=6, label='FLOPs-Pareto'),
                Line2D([0], [0], marker='s', color='w', markerfacecolor=C['green'], markersize=6, label='MemoryNAS'),
                Line2D([0], [0], marker='^', color='w', markerfacecolor=C['grey'], markersize=6, label='Baseline'),
            ]
            ax.legend(handles=legend, loc='lower right', framealpha=0.8)

    fig.tight_layout()
    save_fig(fig, 'fig_pareto_trained')


# ============================================================================
# Figure 4: Search Space Correlation Analysis
# ============================================================================

def fig_correlation_analysis(mem_data):
    """Show FLOPs vs memory correlation from validation data."""
    if mem_data is None:
        print("  Skipping fig_correlation_analysis (no data)")
        return

    results = mem_data['results']

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.3))

    flops = np.array([r['flops_m'] for r in results])
    mem_total = np.array([r['measured_total_mb'] for r in results])
    params = np.array([r['params_m'] for r in results])

    # (a) FLOPs vs Peak Memory
    ax = axes[0]
    ax.scatter(flops, mem_total, c=C['blue'], s=12, alpha=0.6, edgecolors='none')
    r, _ = pearsonr(flops, mem_total)
    ax.set_xlabel('FLOPs (M)')
    ax.set_ylabel('Peak GPU Memory (MB)')
    ax.set_title(f'(a) FLOPs vs Memory\nr = {r:.3f}')

    # (b) Parameters vs Peak Memory
    ax = axes[1]
    ax.scatter(params, mem_total, c=C['orange'], s=12, alpha=0.6, edgecolors='none')
    r, _ = pearsonr(params, mem_total)
    ax.set_xlabel('Parameters (M)')
    ax.set_ylabel('Peak GPU Memory (MB)')
    ax.set_title(f'(b) Params vs Memory\nr = {r:.3f}')

    # (c) FLOPs vs Parameters
    ax = axes[2]
    ax.scatter(flops, params, c=C['green'], s=12, alpha=0.6, edgecolors='none')
    r, _ = pearsonr(flops, params)
    ax.set_xlabel('FLOPs (M)')
    ax.set_ylabel('Parameters (M)')
    ax.set_title(f'(c) FLOPs vs Params\nr = {r:.3f}')

    fig.tight_layout()
    save_fig(fig, 'fig_correlation')


# ============================================================================
# Figure 5: Ablation — Search Budget Sensitivity
# ============================================================================

def fig_ablation_budget(baselines_data):
    """Search budget sensitivity: accuracy vs number of generations."""
    if baselines_data is None:
        print("  Skipping fig_ablation_budget (no data)")
        return

    ablations = baselines_data['ablations']
    budget_keys = sorted([k for k in ablations if k.startswith('budget_sensitivity_')])

    gens = []
    evals = []
    best_accs = []
    n_sols = []

    for key in budget_keys:
        info = ablations[key]
        gens.append(info['n_generations'])
        evals.append(info['total_evals'])
        best_accs.append(info['best_accuracy'])
        n_sols.append(info['n_solutions'])

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5))

    # (a) Best accuracy vs evaluations
    ax = axes[0]
    ax.plot(evals, best_accs, 'o-', color=C['blue'], markersize=6, markerfacecolor='white',
            markeredgecolor=C['blue'], markeredgewidth=1.5)
    ax.set_xlabel('Total Evaluations')
    ax.set_ylabel('Best Accuracy (%)')
    ax.set_title('(a) Convergence: Best Accuracy')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # (b) Number of Pareto solutions vs evaluations
    ax = axes[1]
    ax.plot(evals, n_sols, 's-', color=C['green'], markersize=6, markerfacecolor='white',
            markeredgecolor=C['green'], markeredgewidth=1.5)
    ax.set_xlabel('Total Evaluations')
    ax.set_ylabel('Pareto Solutions')
    ax.set_title('(b) Convergence: Pareto Front Size')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, 'fig_ablation_budget')


# ============================================================================
# Figure 6: Ablation — Memory as Objective vs Constraint Only
# ============================================================================

def fig_ablation_mem_obj(baselines_data):
    """Compare 3-obj (memory as objective) vs 2-obj + constraint."""
    if baselines_data is None:
        print("  Skipping fig_ablation_mem_obj (no data)")
        return

    ablations = baselines_data['ablations']
    baselines = baselines_data['baselines']

    budgets = [3.0, 5.0, 10.0]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.3))

    for i, budget in enumerate(budgets):
        ax = axes[i]

        # 3-obj MemoryNAS constrained
        key3 = f'memorynasConstrained_{budget}'
        if key3 in baselines:
            sols3 = baselines[key3]
            acc3 = [s['accuracy'] for s in sols3]
            flops3 = [s['flops_g'] for s in sols3]
            ax.scatter(flops3, acc3, c=C['green'], s=10, alpha=0.5, label='3-obj (ours)', zorder=3)

        # 2-obj + constraint
        key2 = f'2obj_constrained_{budget}'
        if key2 in ablations:
            sols2 = ablations[key2]
            acc2 = [s['accuracy'] for s in sols2]
            flops2 = [s['flops_g'] for s in sols2]
            ax.scatter(flops2, acc2, c=C['orange'], s=10, alpha=0.5, marker='x', label='2-obj+constr', zorder=2)

        ax.set_xlabel('FLOPs (G)')
        if i == 0:
            ax.set_ylabel('Proxy Accuracy (%)')
        ax.set_title(f'Budget = {budget} MB')
        ax.legend(loc='lower right', fontsize=6, framealpha=0.7)

    fig.suptitle('Memory as Objective vs Constraint-Only', fontsize=9, y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig_ablation_mem_objective')


# ============================================================================
# Figure 7: Per-Architecture Memory Breakdown
# ============================================================================

def fig_memory_breakdown(training_data):
    """Show weight vs activation memory for each trained architecture."""
    if training_data is None:
        print("  Skipping fig_memory_breakdown (no data)")
        return

    names = list(training_data.keys())
    weight_mems = [training_data[n]['weight_memory'] / 1e6 for n in names]
    act_mems = [training_data[n]['gpu_peak_act_bs32'] / 1e6 for n in names]
    analytical = [training_data[n]['analytical_memory'] / 1e6 for n in names]

    fig, ax = plt.subplots(figsize=(6.5, 3.0))

    x = np.arange(len(names))
    width = 0.6

    bars_w = ax.bar(x, weight_mems, width, color=C['blue'], label='Model Weights')
    bars_a = ax.bar(x, act_mems, width, bottom=weight_mems, color=C['cyan'], label='Activations (measured)')
    ax.scatter(x, [w + a for w, a in zip(weight_mems, analytical)], marker='_', s=80, color=C['red'],
               linewidths=2, zorder=5, label='Analytical activation est.')

    ax.set_ylabel('GPU Memory (MB)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=6)
    ax.legend(loc='upper left', framealpha=0.8, fontsize=7)
    ax.set_title('Memory Breakdown: Weights + Activations (batch size = 32)')

    fig.tight_layout()
    save_fig(fig, 'fig_memory_breakdown')


# ============================================================================
# Figure 8: Overview / Method Pipeline (reuse V1 schematic)
# ============================================================================

def fig_overview():
    """Check if the existing method pipeline figure exists."""
    path = os.path.join(FIGURES_DIR, 'method_pipeline.png')
    if os.path.exists(path):
        print("  method_pipeline.png already exists (keeping V1 schematic)")
    else:
        print("  WARNING: method_pipeline.png not found — regenerate separately")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("MemoryNAS V2 Figure Generation")
    print("=" * 60)

    print("\nLoading results...")
    data = load_results()

    print("\nGenerating figures...")

    # Fig 1: Memory estimator validation
    fig_memory_validation(data['memory_validation'])

    # Fig 2: Trained accuracy comparison
    fig_training_results(data['training_results'])

    # Fig 3: Pareto front with trained accuracy
    fig_pareto_trained(data['training_results'])

    # Fig 4: Search space correlation analysis
    fig_correlation_analysis(data['memory_validation'])

    # Fig 5: Ablation — search budget sensitivity
    fig_ablation_budget(data['baselines_ablations'])

    # Fig 6: Ablation — memory as objective vs constraint
    fig_ablation_mem_obj(data['baselines_ablations'])

    # Fig 7: Memory breakdown (weights + activations)
    fig_memory_breakdown(data['training_results'])

    # Fig 8: Method pipeline schematic
    fig_overview()

    print("\n" + "=" * 60)
    print("Done! Figures saved to:", FIGURES_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()
