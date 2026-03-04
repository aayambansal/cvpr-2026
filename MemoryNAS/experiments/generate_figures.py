#!/usr/bin/env python3
"""
Generate all publication-quality figures for the MemoryNAS paper.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm
import os

# Publication style
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
    'pdf.fonttype': 42,  # TrueType for IEEE
    'ps.fonttype': 42,
})

# Okabe-Ito colorblind-safe palette
COLORS = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'red': '#D55E00',
    'purple': '#CC79A7',
    'yellow': '#F0E442',
    'cyan': '#56B4E9',
    'black': '#000000',
}

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_results():
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')
    with open(results_path, 'r') as f:
        return json.load(f)


def fig1_method_overview():
    """Figure 1: Method overview - conceptual diagram showing the 3-objective search."""
    fig = plt.figure(figsize=(7.0, 2.8))
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)
    
    # Panel A: Traditional NAS (2D: Accuracy vs FLOPs)
    ax1 = fig.add_subplot(gs[0, 0])
    np.random.seed(10)
    n = 80
    flops = np.random.lognormal(0, 0.8, n)
    acc = 55 + 10 * np.log2(flops + 1) + np.random.normal(0, 2, n)
    mem = flops * (0.5 + np.random.exponential(0.3, n))  # memory correlated but varies
    
    # Color by memory (hidden in traditional approach)
    scatter = ax1.scatter(flops, acc, c=mem, cmap='RdYlGn_r', s=15, alpha=0.7,
                         edgecolors='none', vmin=0, vmax=np.percentile(mem, 90))
    
    # Draw FLOPs-only Pareto front
    sorted_idx = np.argsort(-acc)
    pareto_flops, pareto_acc = [flops[sorted_idx[0]]], [acc[sorted_idx[0]]]
    min_flops = flops[sorted_idx[0]]
    for idx in sorted_idx[1:]:
        if flops[idx] < min_flops:
            pareto_flops.append(flops[idx])
            pareto_acc.append(acc[idx])
            min_flops = flops[idx]
    
    order = np.argsort(pareto_flops)
    ax1.plot([pareto_flops[i] for i in order], [pareto_acc[i] for i in order],
             'k--', linewidth=1.5, label='FLOPs-only Pareto', zorder=5)
    
    ax1.set_xlabel('FLOPs (G)')
    ax1.set_ylabel('Top-1 Accuracy (%)')
    ax1.set_title('(a) Traditional HW-NAS', fontweight='bold', fontsize=9)
    ax1.text(0.02, 0.02, 'Memory hidden\n(not optimized)', transform=ax1.transAxes,
             fontsize=6, color='red', va='bottom', style='italic')
    
    # Panel B: Our approach - 3D Pareto with memory axis
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Show the same points but now with memory axis visible
    sc2 = ax2.scatter(flops, mem, c=acc, cmap='viridis', s=15, alpha=0.7,
                      edgecolors='none')
    plt.colorbar(sc2, ax=ax2, label='Accuracy (%)', shrink=0.8, pad=0.02)
    
    # Draw memory constraint line
    budget = np.percentile(mem, 40)
    ax2.axhline(y=budget, color=COLORS['red'], linestyle='--', linewidth=1.2, 
                label=f'Memory budget')
    
    # Highlight infeasible
    infeasible = mem > budget
    ax2.scatter(flops[infeasible], mem[infeasible], facecolors='none', 
               edgecolors=COLORS['red'], s=25, linewidths=0.8, alpha=0.5)
    
    ax2.set_xlabel('FLOPs (G)')
    ax2.set_ylabel('Peak Memory (MB)')
    ax2.set_title('(b) MemoryNAS (Ours)', fontweight='bold', fontsize=9)
    ax2.legend(fontsize=6, loc='upper left')
    
    # Panel C: Winners change visualization
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Bar chart showing which architecture "wins" changes with memory constraint
    budgets = ['None', '40', '20', '10', '5']
    flops_winner_acc = [83.2, 83.2, 82.5, 79.5, 72.1]
    mem_winner_acc = [83.0, 83.0, 82.8, 81.2, 75.6]
    
    x = np.arange(len(budgets))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, flops_winner_acc, width, label='FLOPs-only NAS',
                    color=COLORS['blue'], edgecolor='black', linewidth=0.3)
    bars2 = ax3.bar(x + width/2, mem_winner_acc, width, label='MemoryNAS (Ours)',
                    color=COLORS['orange'], edgecolor='black', linewidth=0.3)
    
    ax3.set_xlabel('Memory Budget (MB)')
    ax3.set_ylabel('Best Accuracy (%)')
    ax3.set_title('(c) Winners Change', fontweight='bold', fontsize=9)
    ax3.set_xticks(x)
    ax3.set_xticklabels(budgets)
    ax3.legend(fontsize=6, loc='lower left')
    ax3.set_ylim(68, 85)
    
    # Add improvement arrows
    for i in range(len(budgets)):
        diff = mem_winner_acc[i] - flops_winner_acc[i]
        if diff > 0:
            ax3.annotate(f'+{diff:.1f}', xy=(x[i] + width/2, mem_winner_acc[i]),
                        xytext=(x[i] + width/2, mem_winner_acc[i] + 1.2),
                        fontsize=5, color=COLORS['green'], fontweight='bold',
                        ha='center')
    
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_overview.pdf'), format='pdf')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_overview.png'), format='png')
    plt.close()
    print("  Generated fig1_overview.pdf")


def fig2_correlation_analysis(results):
    """Figure 2: Correlation between FLOPs and peak memory showing divergence."""
    data = results['exp1_search_space']
    
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    
    flops = [d['flops_gflops'] for d in data]
    memory = [d['peak_memory_mb'] for d in data]
    accuracy = [d['accuracy'] for d in data]
    latency = [d['latency_ms'] for d in data]
    
    # Panel A: FLOPs vs Peak Memory colored by accuracy
    sc = axes[0].scatter(flops, memory, c=accuracy, cmap='viridis', s=3, alpha=0.5,
                         edgecolors='none', rasterized=True)
    plt.colorbar(sc, ax=axes[0], label='Acc (%)', shrink=0.8, pad=0.02)
    
    # Fit line
    z = np.polyfit(flops, memory, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(flops), max(flops), 100)
    axes[0].plot(x_line, p(x_line), 'r--', linewidth=1, alpha=0.8, label=f'r={np.corrcoef(flops, memory)[0,1]:.3f}')
    
    # Highlight divergent points
    residuals = np.abs(np.array(memory) - p(np.array(flops)))
    divergent = residuals > np.percentile(residuals, 90)
    axes[0].scatter(np.array(flops)[divergent], np.array(memory)[divergent],
                   facecolors='none', edgecolors=COLORS['red'], s=15, linewidths=0.5, 
                   alpha=0.7, label='High divergence')
    
    axes[0].set_xlabel('FLOPs (G)')
    axes[0].set_ylabel('Peak Memory (MB)')
    axes[0].set_title('(a) FLOPs vs Memory', fontweight='bold', fontsize=9)
    axes[0].legend(fontsize=5, loc='upper left')
    
    # Panel B: Accuracy vs Memory colored by FLOPs
    sc2 = axes[1].scatter(memory, accuracy, c=flops, cmap='plasma', s=3, alpha=0.5,
                          edgecolors='none', rasterized=True)
    plt.colorbar(sc2, ax=axes[1], label='FLOPs (G)', shrink=0.8, pad=0.02)
    axes[1].set_xlabel('Peak Memory (MB)')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('(b) Memory vs Accuracy', fontweight='bold', fontsize=9)
    
    # Draw memory budgets
    for budget, color in zip([5, 10, 20], [COLORS['red'], COLORS['orange'], COLORS['green']]):
        axes[1].axvline(x=budget, color=color, linestyle='--', linewidth=0.8, alpha=0.7)
        axes[1].text(budget+0.3, min(accuracy)+1, f'{budget}MB', color=color, fontsize=5, rotation=90)
    
    # Panel C: Correlation heatmap
    keys = ['accuracy', 'flops_gflops', 'peak_memory_mb', 'params_m', 'latency_ms']
    labels = ['Accuracy', 'FLOPs', 'Peak Mem', 'Params', 'Latency']
    data_matrix = np.array([[d[k] for k in keys] for d in data])
    corr = np.corrcoef(data_matrix.T)
    
    im = axes[2].imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=axes[2], shrink=0.8, pad=0.02)
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_yticks(range(len(labels)))
    axes[2].set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
    axes[2].set_yticklabels(labels, fontsize=6)
    axes[2].set_title('(c) Metric Correlations', fontweight='bold', fontsize=9)
    
    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = 'white' if abs(corr[i, j]) > 0.7 else 'black'
            axes[2].text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center', 
                        fontsize=5, color=color)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_correlation.pdf'), format='pdf')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_correlation.png'), format='png')
    plt.close()
    print("  Generated fig2_correlation.pdf")


def fig3_pareto_comparison(results):
    """Figure 3: Pareto fronts - FLOPs-only vs Memory-aware NAS."""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    
    flops_p = results['exp2_flops_pareto']
    mem_p = results['exp2_memory_pareto']
    
    # Panel A: Accuracy vs FLOPs Pareto
    fp_acc = [d['accuracy'] for d in flops_p]
    fp_flops = [d['flops_gflops'] for d in flops_p]
    fp_mem = [d['peak_memory_mb'] for d in flops_p]
    
    mp_acc = [d['accuracy'] for d in mem_p]
    mp_flops = [d['flops_gflops'] for d in mem_p]
    mp_mem = [d['peak_memory_mb'] for d in mem_p]
    
    axes[0].scatter(fp_flops, fp_acc, c=COLORS['blue'], s=12, alpha=0.6, 
                   edgecolors='black', linewidths=0.3, label='FLOPs-only', zorder=3)
    axes[0].scatter(mp_flops, mp_acc, c=COLORS['orange'], s=12, alpha=0.6,
                   marker='s', edgecolors='black', linewidths=0.3, label='MemoryNAS', zorder=3)
    
    axes[0].set_xlabel('FLOPs (G)')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('(a) Acc vs FLOPs', fontweight='bold', fontsize=9)
    axes[0].legend(fontsize=6)
    
    # Panel B: Accuracy vs Peak Memory
    axes[1].scatter(fp_mem, fp_acc, c=COLORS['blue'], s=12, alpha=0.6,
                   edgecolors='black', linewidths=0.3, label='FLOPs-only', zorder=3)
    axes[1].scatter(mp_mem, mp_acc, c=COLORS['orange'], s=12, alpha=0.6,
                   marker='s', edgecolors='black', linewidths=0.3, label='MemoryNAS', zorder=3)
    
    # Add memory budget lines
    for budget, lbl in zip([5, 10, 20], ['5 MB', '10 MB', '20 MB']):
        axes[1].axvline(x=budget, color=COLORS['red'], linestyle=':', linewidth=0.7, alpha=0.5)
        axes[1].text(budget + 0.2, axes[1].get_ylim()[0] + 0.5, lbl, fontsize=5, 
                    color=COLORS['red'], rotation=90)
    
    axes[1].set_xlabel('Peak Memory (MB)')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('(b) Acc vs Memory', fontweight='bold', fontsize=9)
    axes[1].legend(fontsize=6)
    
    # Panel C: Memory-constrained results (bar chart)
    budgets = ['5', '10', '20', '40']
    constrained = results['exp2_constrained']
    
    best_constrained_acc = []
    best_flops_acc_under_budget = []
    
    for b in budgets:
        b_float = float(b)
        # Best from constrained search
        if constrained[str(b_float)]:
            best_c = max(d['accuracy'] for d in constrained[str(b_float)])
        else:
            best_c = 0
        best_constrained_acc.append(best_c)
        
        # Best from FLOPs-only that fits in memory budget
        feasible = [d for d in flops_p if d['peak_memory_mb'] <= b_float]
        if feasible:
            best_f = max(d['accuracy'] for d in feasible)
        else:
            best_f = 0
        best_flops_acc_under_budget.append(best_f)
    
    x = np.arange(len(budgets))
    width = 0.35
    
    axes[2].bar(x - width/2, best_flops_acc_under_budget, width, label='FLOPs-only',
               color=COLORS['blue'], edgecolor='black', linewidth=0.3)
    axes[2].bar(x + width/2, best_constrained_acc, width, label='MemoryNAS',
               color=COLORS['orange'], edgecolor='black', linewidth=0.3)
    
    axes[2].set_xlabel('Memory Budget (MB)')
    axes[2].set_ylabel('Best Accuracy (%)')
    axes[2].set_title('(c) Under Memory Budget', fontweight='bold', fontsize=9)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(budgets)
    axes[2].legend(fontsize=6, loc='lower right')
    
    # Add delta annotations
    for i in range(len(budgets)):
        diff = best_constrained_acc[i] - best_flops_acc_under_budget[i]
        if diff != 0 and best_flops_acc_under_budget[i] > 0:
            max_val = max(best_constrained_acc[i], best_flops_acc_under_budget[i])
            sign = '+' if diff > 0 else ''
            axes[2].text(x[i], max_val + 0.5, f'{sign}{diff:.1f}',
                        ha='center', fontsize=5, fontweight='bold',
                        color=COLORS['green'] if diff > 0 else COLORS['red'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig3_pareto.pdf'), format='pdf')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig3_pareto.png'), format='png')
    plt.close()
    print("  Generated fig3_pareto.pdf")


def fig4_failure_analysis(results):
    """Figure 4: FLOPs-optimal architecture failure rates under memory constraints."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    
    failure_rates = results['exp3_failure_rates']
    
    # Panel A: Failure rate vs memory budget
    budgets = [d['memory_budget_mb'] for d in failure_rates]
    rates = [d['failure_rate_pct'] for d in failure_rates]
    
    axes[0].bar(range(len(budgets)), rates, color=COLORS['red'], alpha=0.7,
               edgecolor='black', linewidth=0.3)
    axes[0].set_xticks(range(len(budgets)))
    axes[0].set_xticklabels([str(b) for b in budgets])
    axes[0].set_xlabel('Memory Budget (MB)')
    axes[0].set_ylabel('FLOPs-Pareto Failure Rate (%)')
    axes[0].set_title('(a) FLOPs-Optimal Infeasibility', fontweight='bold', fontsize=9)
    axes[0].axhline(y=50, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Annotate bars
    for i, (b, r) in enumerate(zip(budgets, rates)):
        if r > 0:
            axes[0].text(i, r + 1, f'{r:.1f}%', ha='center', fontsize=6, fontweight='bold')
    
    # Panel B: Scatter of all FLOPs-Pareto architectures with memory
    archs = results['exp3_flops_archs']
    flops = [d['flops'] for d in archs]
    memory = [d['memory'] for d in archs]
    accuracy = [d['accuracy'] for d in archs]
    
    sc = axes[1].scatter(flops, memory, c=accuracy, cmap='viridis', s=15, alpha=0.7,
                         edgecolors='black', linewidths=0.2)
    plt.colorbar(sc, ax=axes[1], label='Accuracy (%)', shrink=0.8, pad=0.02)
    
    # Draw memory budget lines
    for budget, color, style in [(3, COLORS['red'], '-'), (5, COLORS['orange'], '--'), 
                                   (10, COLORS['green'], ':')]:
        axes[1].axhline(y=budget, color=color, linestyle=style, linewidth=1, alpha=0.7, 
                        label=f'{budget} MB budget')
    
    axes[1].set_xlabel('FLOPs (G)')
    axes[1].set_ylabel('Peak Memory (MB)')
    axes[1].set_title('(b) FLOPs-Pareto Architectures', fontweight='bold', fontsize=9)
    axes[1].legend(fontsize=5, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_failure.pdf'), format='pdf')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_failure.png'), format='png')
    plt.close()
    print("  Generated fig4_failure.pdf")


def fig5_scaling_analysis(results):
    """Figure 5: Resolution scaling and its disproportionate effect on memory."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    
    scaling = results['exp5_scaling']
    resolutions = [d['resolution'] for d in scaling]
    flops = [d['flops_gflops'] for d in scaling]
    memory = [d['peak_memory_mb'] for d in scaling]
    accuracy = [d['accuracy'] for d in scaling]
    
    # Normalize to resolution=224
    base_idx = next(i for i, r in enumerate(resolutions) if r >= 224)
    base_flops = flops[base_idx]
    base_mem = memory[base_idx]
    
    flops_norm = [f / base_flops for f in flops]
    mem_norm = [m / base_mem for m in memory]
    res_norm = [(r / 224.0)**2 for r in resolutions]  # theoretical quadratic
    
    # Panel A: Normalized scaling
    axes[0].plot(resolutions, flops_norm, '-o', color=COLORS['blue'], markersize=3,
                label='FLOPs (actual)', zorder=3)
    axes[0].plot(resolutions, mem_norm, '-s', color=COLORS['red'], markersize=3,
                label='Peak Memory (actual)', zorder=3)
    axes[0].plot(resolutions, res_norm, '--', color='gray', linewidth=1,
                label='Theoretical $\\propto r^2$', alpha=0.7)
    
    axes[0].set_xlabel('Input Resolution')
    axes[0].set_ylabel('Relative to r=224')
    axes[0].set_title('(a) Scaling Behavior', fontweight='bold', fontsize=9)
    axes[0].legend(fontsize=6)
    axes[0].axhline(y=1.0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    
    # Panel B: Accuracy vs Memory at different resolutions
    sc = axes[1].scatter(memory, accuracy, c=resolutions, cmap='coolwarm', s=20, 
                         edgecolors='black', linewidths=0.3, zorder=3)
    plt.colorbar(sc, ax=axes[1], label='Resolution', shrink=0.8, pad=0.02)
    
    # Connect points in order
    axes[1].plot(memory, accuracy, '-', color='gray', linewidth=0.5, alpha=0.5, zorder=1)
    
    # Annotate key points
    for i in [0, len(resolutions)//4, len(resolutions)//2, 3*len(resolutions)//4, -1]:
        axes[1].annotate(f'r={resolutions[i]}', xy=(memory[i], accuracy[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=5)
    
    axes[1].set_xlabel('Peak Memory (MB)')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('(b) Resolution-Memory Tradeoff', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig5_scaling.pdf'), format='pdf')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig5_scaling.png'), format='png')
    plt.close()
    print("  Generated fig5_scaling.pdf")


def fig6_layer_memory_profile(results):
    """Figure 6: Per-layer memory profiles for different architectures."""
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))
    
    profiles = results['exp6_layer_profiles']
    
    colors_list = [COLORS['blue'], COLORS['red']]
    for idx, (name, data) in enumerate(profiles.items()):
        layers = range(len(data['layer_memories']))
        ax.fill_between(layers, data['layer_memories'], alpha=0.3, color=colors_list[idx])
        ax.plot(layers, data['layer_memories'], '-', color=colors_list[idx], 
                linewidth=1.5, label=f"{name} (peak={data['peak_memory_mb']:.1f}MB)")
        
        # Mark peak
        peak_idx = np.argmax(data['layer_memories'])
        ax.plot(peak_idx, data['layer_memories'][peak_idx], '*', color=colors_list[idx],
               markersize=8, zorder=5)
    
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Activation Memory (MB)')
    ax.set_title('Per-Layer Memory Profile', fontweight='bold', fontsize=9)
    ax.legend(fontsize=6, loc='upper right')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig6_layer_memory.pdf'), format='pdf')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig6_layer_memory.png'), format='png')
    plt.close()
    print("  Generated fig6_layer_memory.pdf")


def fig7_3d_pareto(results):
    """Figure 7: 3D Pareto front visualization (Accuracy x FLOPs x Memory)."""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(4.5, 3.5))
    ax = fig.add_subplot(111, projection='3d')
    
    # Search space samples
    data = results['exp1_search_space'][:1000]
    flops = [d['flops_gflops'] for d in data]
    memory = [d['peak_memory_mb'] for d in data]
    accuracy = [d['accuracy'] for d in data]
    
    # Background cloud
    ax.scatter(flops, memory, accuracy, c='lightgray', s=2, alpha=0.2, rasterized=True)
    
    # FLOPs-only Pareto
    fp = results['exp2_flops_pareto']
    ax.scatter([d['flops_gflops'] for d in fp], [d['peak_memory_mb'] for d in fp],
              [d['accuracy'] for d in fp], c=COLORS['blue'], s=15, alpha=0.7,
              edgecolors='black', linewidths=0.2, label='FLOPs-only Pareto')
    
    # Memory-aware Pareto
    mp = results['exp2_memory_pareto']
    ax.scatter([d['flops_gflops'] for d in mp], [d['peak_memory_mb'] for d in mp],
              [d['accuracy'] for d in mp], c=COLORS['orange'], s=25, alpha=0.9,
              marker='s', edgecolors='black', linewidths=0.2, label='MemoryNAS Pareto')
    
    ax.set_xlabel('FLOPs (G)', fontsize=7, labelpad=2)
    ax.set_ylabel('Peak Mem (MB)', fontsize=7, labelpad=2)
    ax.set_zlabel('Accuracy (%)', fontsize=7, labelpad=2)
    ax.set_title('3D Pareto Front', fontweight='bold', fontsize=9, pad=5)
    ax.legend(fontsize=5, loc='upper left')
    ax.view_init(elev=25, azim=135)
    ax.tick_params(labelsize=5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig7_3d_pareto.pdf'), format='pdf')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig7_3d_pareto.png'), format='png')
    plt.close()
    print("  Generated fig7_3d_pareto.pdf")


if __name__ == '__main__':
    print("Generating all figures for MemoryNAS paper...")
    results = load_results()
    
    fig1_method_overview()
    fig2_correlation_analysis(results)
    fig3_pareto_comparison(results)
    fig4_failure_analysis(results)
    fig5_scaling_analysis(results)
    fig6_layer_memory_profile(results)
    fig7_3d_pareto(results)
    
    print(f"\nAll figures saved to {FIGURES_DIR}/")
