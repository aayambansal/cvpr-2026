#!/usr/bin/env python3
"""
Generate all publication-quality figures for the GreenNAS v2 paper.
Figures:
  1. Method overview diagram
  2. Pareto fronts (acc vs energy, cost, latency) with clear GreenNAS advantage
  3. Hypervolume comparison (bar chart across methods)
  4. Proxy correlation analysis (FLOPs/mem/power vs energy, with Spearman+Kendall)
  5. Reproducibility (variance across seeds, acc + energy)
  6. Search convergence curves
  7. Proxy validation scatter (measured vs predicted, if NVML data available)
  8. Proxy ablation bar chart
  9. Constraint-based selection heatmap
  10. Batch-size sensitivity
  11. Training vs inference energy Pareto comparison
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from scipy import stats

# Publication settings
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
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
})

# Okabe-Ito colorblind-safe palette
COLORS = {
    'ours': '#0072B2',
    'random': '#E69F00',
    'flops_only': '#CC79A7',
    'weighted_sum': '#009E73',
    'epsilon_constraint': '#D55E00',
    'filter_rank': '#999999',
    'train_energy': '#56B4E9',
    'inf_energy': '#F0E442',
    'dark': '#000000',
    'gray': '#999999',
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

def load_results():
    with open(os.path.join(BASE_DIR, 'results.json')) as f:
        return json.load(f)

def load_nvml():
    """Load NVML measurements if available."""
    path = os.path.join(BASE_DIR, 'nvml_measurements.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def savefig(fig, name):
    fig.savefig(os.path.join(FIG_DIR, f'{name}.pdf'), format='pdf')
    fig.savefig(os.path.join(FIG_DIR, f'{name}.png'), format='png')
    plt.close(fig)

# ============================================================
# FIGURE 1: METHOD OVERVIEW
# ============================================================
def fig1_overview():
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.5)
    ax.axis('off')
    
    boxes = [
        (0.3, 1.5, 1.8, 1.2, 'Search\nSpace\n(Cell-based)', '#E8F4FD'),
        (2.8, 1.5, 1.8, 1.2, 'Composite\nEnergy Proxy\n(4 surrogates)', '#FFF3E0'),
        (5.3, 1.5, 1.8, 1.2, 'NSGA-II\n4-Objective\nOptimizer', '#E8F5E9'),
        (7.8, 1.5, 1.8, 1.2, 'Pareto Fronts\n+ HV/IGD\nMetrics', '#F3E5F5'),
    ]
    
    for x, y, w, h, text, color in boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='#333333', linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=7.5, fontweight='bold')
    
    for x1, x2 in [(2.1, 2.8), (4.6, 5.3), (7.1, 7.8)]:
        ax.annotate('', xy=(x2, 2.1), xytext=(x1, 2.1),
                    arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))
    
    sublabels = [
        (1.2, 1.2, '5 ops × 6 edges\n15,625 archs', 6.5),
        (3.7, 1.2, 'Power · Time ·\nMemory · BS-sens', 6.5),
        (6.2, 1.2, 'Pop=40, 30 gens\n5 seeds', 6.5),
        (8.7, 1.2, 'CIFAR-10/100\n+ NVML validation', 6.5),
    ]
    for x, y, text, fs in sublabels:
        ax.text(x, y, text, ha='center', va='top', fontsize=fs, color='#666666')
    
    ax.text(5, 3.3, 'GreenNAS: Carbon- and Cost-Aware Neural Architecture Search',
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    savefig(fig, 'fig1_overview')
    print("  Fig 1: Method overview")

# ============================================================
# FIGURE 2: PARETO FRONTS
# ============================================================
def fig2_pareto():
    data = load_results()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    
    metrics_list = [
        ('train_energy_kwh', 'Training Energy (Wh)', 1e3),
        ('dollar_cost', 'Cloud Cost (USD)', 1.0),
        ('latency_ms', 'Latency (ms)', 1.0),
    ]
    
    for ax_idx, (metric, ylabel, scale) in enumerate(metrics_list):
        ax = axes[ax_idx]
        
        for seed in ['1', '2', '3', '4', '5']:
            sd = data['cifar10'][seed]
            
            # GreenNAS
            for p in sd['ours']['population']:
                ax.scatter(p['metrics']['accuracy'], p['metrics'][metric] * scale,
                          c=COLORS['ours'], alpha=0.2, s=10, marker='o', zorder=3)
            # Random
            for r in sd['random']['results']:
                ax.scatter(r['accuracy'], r[metric] * scale,
                          c=COLORS['random'], alpha=0.06, s=5, marker='s', zorder=1)
            # FLOPs-only
            for p in sd['flops_only']['population']:
                ax.scatter(p['metrics']['accuracy'], p['metrics'][metric] * scale,
                          c=COLORS['flops_only'], alpha=0.15, s=8, marker='^', zorder=2)
        
        ax.set_xlabel('Accuracy (%)')
        ax.set_ylabel(ylabel)
        ax.set_title(['(a)', '(b)', '(c)'][ax_idx], fontweight='bold', loc='left')
    
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['ours'], markersize=5, label='GreenNAS (Ours)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['random'], markersize=5, label='Random Search'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS['flops_only'], markersize=5, label='FLOPs-only NAS'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.08), frameon=False, fontsize=7)
    plt.tight_layout()
    savefig(fig, 'fig2_pareto')
    print("  Fig 2: Pareto fronts")

# ============================================================
# FIGURE 3: HYPERVOLUME COMPARISON
# ============================================================
def fig3_hypervolume():
    data = load_results()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    
    methods = ['ours', 'random', 'flops_only', 'weighted_sum', 'epsilon_constraint', 'filter_rank']
    method_names = ['GreenNAS\n(Ours)', 'Random', 'FLOPs\nOnly', 'Weighted\nSum', 'ε-Constr.', 'Filter\n+Rank']
    colors = [COLORS[m] for m in methods]
    
    for ax_idx, dataset in enumerate(['cifar10', 'cifar100']):
        ax = axes[ax_idx]
        
        hvs = {m: [] for m in methods}
        for seed in ['1', '2', '3', '4', '5']:
            sd = data[dataset][seed]
            if 'metrics' in sd and 'hypervolume' in sd['metrics']:
                for m in methods:
                    if m in sd['metrics']['hypervolume']:
                        hvs[m].append(sd['metrics']['hypervolume'][m])
        
        x_pos = np.arange(len(methods))
        means = [np.mean(hvs[m]) if hvs[m] else 0 for m in methods]
        stds = [np.std(hvs[m]) if hvs[m] else 0 for m in methods]
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=3, color=colors, alpha=0.85,
                      edgecolor='#333333', linewidth=0.6, width=0.7)
        
        # Individual seed points
        for i, m in enumerate(methods):
            for j, v in enumerate(hvs[m]):
                ax.scatter(i + (j-2)*0.08, v, c='black', s=8, zorder=5, alpha=0.5)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(method_names, fontsize=6)
        ax.set_ylabel('Hypervolume ↑')
        title = 'CIFAR-10' if dataset == 'cifar10' else 'CIFAR-100'
        ax.set_title(f'({["a","b"][ax_idx]}) {title}', fontweight='bold', loc='left')
    
    plt.tight_layout()
    savefig(fig, 'fig3_hypervolume')
    print("  Fig 3: Hypervolume comparison")

# ============================================================
# FIGURE 4: PROXY CORRELATIONS (FIXED - no more NaN)
# ============================================================
def fig4_correlations():
    data = load_results()
    fig, axes = plt.subplots(2, 2, figsize=(5.0, 4.5))
    
    # Collect ALL architectures from ALL seeds' caches for maximum diversity
    all_archs = {}
    for seed in ['1', '2', '3', '4', '5']:
        sd = data['cifar10'][seed]
        for method in ['ours', 'random', 'flops_only']:
            if method in sd and 'cache' in sd[method]:
                for key, val in sd[method]['cache'].items():
                    if key not in all_archs:
                        all_archs[key] = val
    
    arch_list = list(all_archs.values())
    print(f"    Correlation analysis: {len(arch_list)} unique architectures")
    
    accs = np.array([a['accuracy'] for a in arch_list])
    energies = np.array([a['train_energy_kwh'] * 1000 for a in arch_list])  # Wh
    flops = np.array([a['flops'] / 1e6 for a in arch_list])
    mem = np.array([a['memory_traffic_gb'] * 1000 for a in arch_list])  # MB
    powers = np.array([a['gpu_power_w'] for a in arch_list])
    params = np.array([a['params'] / 1000 for a in arch_list])
    latencies = np.array([a['latency_ms'] for a in arch_list])
    
    # (a) FLOPs vs Energy
    ax = axes[0, 0]
    sc = ax.scatter(flops, energies, c=accs, cmap='viridis', s=6, alpha=0.5, edgecolors='none')
    ax.set_xlabel('FLOPs (M)')
    ax.set_ylabel('Energy (Wh)')
    ax.set_title('(a) FLOPs vs Energy', fontweight='bold', loc='left')
    r_p, p_p = stats.pearsonr(flops, energies)
    r_s, _ = stats.spearmanr(flops, energies)
    ax.text(0.95, 0.05, f'r={r_p:.3f}\nρ={r_s:.3f}', transform=ax.transAxes, ha='right', fontsize=6,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # (b) Memory Traffic vs Energy
    ax = axes[0, 1]
    ax.scatter(mem, energies, c=accs, cmap='viridis', s=6, alpha=0.5, edgecolors='none')
    ax.set_xlabel('Memory Traffic (MB)')
    ax.set_ylabel('Energy (Wh)')
    ax.set_title('(b) Memory vs Energy', fontweight='bold', loc='left')
    r_p, _ = stats.pearsonr(mem, energies)
    r_s, _ = stats.spearmanr(mem, energies)
    ax.text(0.95, 0.05, f'r={r_p:.3f}\nρ={r_s:.3f}', transform=ax.transAxes, ha='right', fontsize=6,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # (c) Params vs Latency
    ax = axes[1, 0]
    ax.scatter(params, latencies, c=accs, cmap='viridis', s=6, alpha=0.5, edgecolors='none')
    ax.set_xlabel('Parameters (K)')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('(c) Params vs Latency', fontweight='bold', loc='left')
    r_p, _ = stats.pearsonr(params, latencies)
    r_s, _ = stats.spearmanr(params, latencies)
    ax.text(0.95, 0.05, f'r={r_p:.3f}\nρ={r_s:.3f}', transform=ax.transAxes, ha='right', fontsize=6,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # (d) Power vs Accuracy
    ax = axes[1, 1]
    sc = ax.scatter(powers, accs, c=energies, cmap='plasma', s=6, alpha=0.5, edgecolors='none')
    ax.set_xlabel('GPU Power (W)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('(d) Power vs Accuracy', fontweight='bold', loc='left')
    r_p, _ = stats.pearsonr(powers, accs)
    r_s, _ = stats.spearmanr(powers, accs)
    ax.text(0.95, 0.05, f'r={r_p:.3f}\nρ={r_s:.3f}', transform=ax.transAxes, ha='right', fontsize=6,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Colorbar for accuracy
    cbar_ax = fig.add_axes((1.02, 0.55, 0.02, 0.35))
    cb1 = fig.colorbar(sc, cax=cbar_ax)
    cb1.set_label('Energy (Wh)', fontsize=7)
    
    plt.tight_layout()
    savefig(fig, 'fig4_correlations')
    print("  Fig 4: Proxy correlations")

# ============================================================
# FIGURE 5: REPRODUCIBILITY
# ============================================================
def fig5_reproducibility():
    data = load_results()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    
    methods = ['ours', 'random', 'flops_only', 'weighted_sum', 'epsilon_constraint']
    method_names = ['GreenNAS', 'Random', 'FLOPs-only', 'W-Sum', 'ε-Constr.']
    colors_list = [COLORS[m] for m in methods]
    
    seed_accs = {m: [] for m in methods}
    seed_energies = {m: [] for m in methods}
    
    for seed in ['1', '2', '3', '4', '5']:
        sd = data['cifar10'][seed]
        for m in methods:
            if m in sd:
                if 'population' in sd[m]:
                    items = sd[m]['population']
                    seed_accs[m].append(max(p['metrics']['accuracy'] for p in items))
                    seed_energies[m].append(min(p['metrics']['train_energy_kwh'] for p in items) * 1000)
                elif 'results' in sd[m]:
                    items = sd[m]['results']
                    seed_accs[m].append(max(r['accuracy'] for r in items))
                    seed_energies[m].append(min(r['train_energy_kwh'] for r in items) * 1000)
    
    # (a) Accuracy
    ax = axes[0]
    x_pos = np.arange(len(methods))
    means = [np.mean(seed_accs[m]) for m in methods]
    stds_val = [np.std(seed_accs[m]) for m in methods]
    ax.bar(x_pos, means, yerr=stds_val, capsize=3, color=colors_list, alpha=0.8,
           edgecolor='#333333', linewidth=0.6, width=0.7)
    for i, m in enumerate(methods):
        for j, v in enumerate(seed_accs[m]):
            ax.scatter(i + (j-2)*0.08, v, c='black', s=8, zorder=5, alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_names, fontsize=6)
    ax.set_ylabel('Best Accuracy (%)')
    ax.set_title('(a) Accuracy', fontweight='bold', loc='left')
    
    # (b) Energy
    ax = axes[1]
    means_e = [np.mean(seed_energies[m]) for m in methods]
    stds_e = [np.std(seed_energies[m]) for m in methods]
    ax.bar(x_pos, means_e, yerr=stds_e, capsize=3, color=colors_list, alpha=0.8,
           edgecolor='#333333', linewidth=0.6, width=0.7)
    for i, m in enumerate(methods):
        for j, v in enumerate(seed_energies[m]):
            ax.scatter(i + (j-2)*0.08, v, c='black', s=8, zorder=5, alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_names, fontsize=6)
    ax.set_ylabel('Best Energy (Wh)')
    ax.set_title('(b) Energy', fontweight='bold', loc='left')
    
    plt.tight_layout()
    savefig(fig, 'fig5_reproducibility')
    print("  Fig 5: Reproducibility")

# ============================================================
# FIGURE 6: CONVERGENCE
# ============================================================
def fig6_convergence():
    data = load_results()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    
    for seed in ['1', '2', '3', '4', '5']:
        sd = data['cifar10'][seed]
        
        for method, color, label in [
            ('ours', COLORS['ours'], 'GreenNAS'),
            ('random', COLORS['random'], 'Random'),
            ('flops_only', COLORS['flops_only'], 'FLOPs-only'),
        ]:
            if method not in sd: continue
            
            if 'history' in sd[method]:
                history = sd[method]['history']
            elif 'results' in sd[method]:
                history = sd[method]['results']
            else:
                continue
            
            n = len(history)
            accs = [h['accuracy'] for h in history]
            energies = [h['train_energy_kwh'] * 1000 for h in history]
            
            running_acc = np.maximum.accumulate(accs)
            running_energy = np.minimum.accumulate(energies)
            evals = np.arange(1, n+1)
            
            alpha = 0.25 if seed != '1' else 0.8
            lw = 0.6 if seed != '1' else 1.5
            
            axes[0].plot(evals, running_acc, color=color, alpha=alpha, linewidth=lw,
                        label=label if seed == '1' else None)
            axes[1].plot(evals, running_energy, color=color, alpha=alpha, linewidth=lw,
                        label=label if seed == '1' else None)
    
    axes[0].set_xlabel('Architecture Evaluations')
    axes[0].set_ylabel('Best Accuracy (%)')
    axes[0].set_title('(a) Accuracy Convergence', fontweight='bold', loc='left')
    axes[0].legend(frameon=False, fontsize=6)
    
    axes[1].set_xlabel('Architecture Evaluations')
    axes[1].set_ylabel('Best Energy (Wh)')
    axes[1].set_title('(b) Energy Convergence', fontweight='bold', loc='left')
    axes[1].legend(frameon=False, fontsize=6)
    
    plt.tight_layout()
    savefig(fig, 'fig6_convergence')
    print("  Fig 6: Convergence")

# ============================================================
# FIGURE 7: NVML PROXY VALIDATION (if data available)
# ============================================================
def fig7_proxy_validation():
    nvml = load_nvml()
    if nvml is None:
        print("  Fig 7: SKIPPED (no NVML data yet - Modal job still running)")
        return False
    
    measurements = nvml['measurements']
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    
    # (a) Proxy energy vs measured energy
    ax = axes[0]
    proxy_e = [m['proxy']['proxy_energy_kwh'] * 1000 for m in measurements]
    meas_e = [m['measured']['energy_kwh'] * 1000 for m in measurements]
    ax.scatter(proxy_e, meas_e, s=12, alpha=0.6, c=COLORS['ours'], edgecolors='none')
    
    # Fit line
    slope, intercept, r_val, p_val, std_err = stats.linregress(proxy_e, meas_e)
    x_fit = np.linspace(min(proxy_e), max(proxy_e), 100)
    ax.plot(x_fit, slope*x_fit + intercept, '--', color=COLORS['dark'], linewidth=1.0, alpha=0.7)
    
    # Identity line
    lims = [min(min(proxy_e), min(meas_e)), max(max(proxy_e), max(meas_e))]
    ax.plot(lims, lims, ':', color=COLORS['gray'], linewidth=0.8)
    
    rho, _ = stats.spearmanr(proxy_e, meas_e)
    tau, _ = stats.kendalltau(proxy_e, meas_e)
    ax.set_xlabel('Proxy Energy (Wh)')
    ax.set_ylabel('Measured Energy (Wh)')
    ax.set_title('(a) Energy Validation', fontweight='bold', loc='left')
    ax.text(0.05, 0.95, f'r={r_val:.3f}\nρ={rho:.3f}\nτ={tau:.3f}',
            transform=ax.transAxes, va='top', fontsize=6,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # (b) Memory traffic vs measured power (proxy power saturates at TDP)
    ax = axes[1]
    mem_gb = [m['proxy']['mem_gb'] for m in measurements]
    meas_p = [m['measured']['avg_power_w'] for m in measurements]
    ax.scatter(mem_gb, meas_p, s=12, alpha=0.6, c=COLORS['ours'], edgecolors='none')
    
    rho_p, _ = stats.spearmanr(mem_gb, meas_p)
    try:
        slope_p, intercept_p, r_p, _, _ = stats.linregress(mem_gb, meas_p)
        x_fit_p = np.linspace(min(mem_gb), max(mem_gb), 100)
        ax.plot(x_fit_p, slope_p*x_fit_p + intercept_p, '--', color=COLORS['dark'], linewidth=1.0, alpha=0.7)
    except ValueError:
        r_p = float('nan')
    ax.set_xlabel('Memory Traffic (GB)')
    ax.set_ylabel('Measured Power (W)')
    ax.set_title('(b) Memory vs Power', fontweight='bold', loc='left')
    ax.text(0.05, 0.95, f'r={r_p:.3f}\nρ={rho_p:.3f}',
            transform=ax.transAxes, va='top', fontsize=6,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # (c) Rank correlation across error bars
    ax = axes[2]
    # Sort by proxy energy and show measured energy with error bars
    sorted_idx = np.argsort(proxy_e)
    x = np.arange(len(sorted_idx))
    meas_sorted = [meas_e[i] for i in sorted_idx]
    proxy_sorted = [proxy_e[i] for i in sorted_idx]
    
    ax.plot(x, proxy_sorted, '-', color=COLORS['ours'], linewidth=1.0, label='Proxy', alpha=0.8)
    ax.plot(x, meas_sorted, '.', color=COLORS['random'], markersize=3, label='Measured', alpha=0.6)
    ax.set_xlabel('Architecture (sorted by proxy energy)')
    ax.set_ylabel('Energy (Wh)')
    ax.set_title('(c) Rank Preservation', fontweight='bold', loc='left')
    ax.legend(frameon=False, fontsize=6)
    
    plt.tight_layout()
    savefig(fig, 'fig7_proxy_validation')
    print(f"  Fig 7: Proxy validation (GPU: {nvml['gpu']}, n={len(measurements)})")
    return True

# ============================================================
# FIGURE 8: PROXY ABLATION
# ============================================================
def fig8_ablation():
    data = load_results()
    
    # Get ablation from seed 1
    sd = data['cifar10']['1']
    if 'proxy_ablation' not in sd:
        print("  Fig 8: SKIPPED (no ablation data)")
        return
    
    ablation = sd['proxy_ablation']
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    
    configs = ['flops_only', 'flops_mem', 'flops_power', 'full_composite']
    config_names = ['FLOPs\nOnly', 'FLOPs+\nMemory', 'FLOPs+\nPower', 'Full\nComposite']
    config_colors = ['#CC79A7', '#56B4E9', '#D55E00', '#0072B2']
    
    # (a) Best accuracy
    ax = axes[0]
    best_accs = [max(p['metrics']['accuracy'] for p in ablation[c]) for c in configs]
    x_pos = np.arange(len(configs))
    ax.bar(x_pos, best_accs, color=config_colors, alpha=0.85, edgecolor='#333333', linewidth=0.6, width=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_names, fontsize=6)
    ax.set_ylabel('Best Accuracy (%)')
    ax.set_title('(a) Best Accuracy by Proxy', fontweight='bold', loc='left')
    
    # (b) Best energy at high accuracy (acc >= 80%)
    ax = axes[1]
    best_energies = []
    for c in configs:
        high_acc = [p['metrics'] for p in ablation[c] if p['metrics']['accuracy'] >= 75]
        if high_acc:
            best_energies.append(min(m['train_energy_kwh'] for m in high_acc) * 1000)
        else:
            best_energies.append(0)
    
    ax.bar(x_pos, best_energies, color=config_colors, alpha=0.85, edgecolor='#333333', linewidth=0.6, width=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_names, fontsize=6)
    ax.set_ylabel('Best Energy (Wh)\nat Acc≥75%')
    ax.set_title('(b) Energy at High Accuracy', fontweight='bold', loc='left')
    
    plt.tight_layout()
    savefig(fig, 'fig8_ablation')
    print("  Fig 8: Proxy ablation")

# ============================================================
# FIGURE 9: CONSTRAINT-BASED SELECTION
# ============================================================
def fig9_constraints():
    data = load_results()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    
    # Aggregate constraint results across seeds
    energy_budgets = [0.005, 0.008, 0.010, 0.012, 0.015]
    latency_budget = 4.0  # fixed
    
    for ax_idx, method_pair in enumerate([('ours', 'flops_only'), ('ours', 'random')]):
        ax = axes[ax_idx]
        m1, m2 = method_pair
        
        accs_m1 = {e: [] for e in energy_budgets}
        accs_m2 = {e: [] for e in energy_budgets}
        
        for seed in ['1', '2', '3', '4', '5']:
            sd = data['cifar10'][seed]
            if 'constraint_selection' not in sd: continue
            
            for e in energy_budgets:
                key = f"E{e:.3f}_L{latency_budget:.1f}"
                if key in sd['constraint_selection']:
                    cr = sd['constraint_selection'][key]
                    if m1 in cr and cr[m1]['best_accuracy'] is not None:
                        accs_m1[e].append(cr[m1]['best_accuracy'])
                    if m2 in cr and cr[m2]['best_accuracy'] is not None:
                        accs_m2[e].append(cr[m2]['best_accuracy'])
        
        x = np.arange(len(energy_budgets))
        width = 0.35
        
        means1 = [np.mean(accs_m1[e]) if accs_m1[e] else 0 for e in energy_budgets]
        stds1 = [np.std(accs_m1[e]) if accs_m1[e] else 0 for e in energy_budgets]
        means2 = [np.mean(accs_m2[e]) if accs_m2[e] else 0 for e in energy_budgets]
        stds2 = [np.std(accs_m2[e]) if accs_m2[e] else 0 for e in energy_budgets]
        
        ax.bar(x - width/2, means1, width, yerr=stds1, capsize=2,
               color=COLORS['ours'], alpha=0.85, label='GreenNAS', edgecolor='#333333', linewidth=0.5)
        ax.bar(x + width/2, means2, width, yerr=stds2, capsize=2,
               color=COLORS[m2], alpha=0.85, 
               label={'flops_only': 'FLOPs-only', 'random': 'Random'}[m2],
               edgecolor='#333333', linewidth=0.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels([f'{e*1000:.0f}' for e in energy_budgets], fontsize=6)
        ax.set_xlabel('Energy Budget (Wh)')
        ax.set_ylabel('Best Accuracy (%) ↑')
        name2 = {'flops_only': 'FLOPs-only', 'random': 'Random'}[m2]
        ax.set_title(f'({["a","b"][ax_idx]}) GreenNAS vs {name2}', fontweight='bold', loc='left')
        ax.legend(frameon=False, fontsize=6)
    
    plt.tight_layout()
    savefig(fig, 'fig9_constraints')
    print("  Fig 9: Constraint-based selection")

# ============================================================
# FIGURE 10: BATCH SIZE SENSITIVITY
# ============================================================
def fig10_bs_sensitivity():
    data = load_results()
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    all_archs = {}
    sd = data['cifar10']['1']
    for method in ['ours', 'random', 'flops_only']:
        if method in sd and 'cache' in sd[method]:
            for k, v in sd[method]['cache'].items():
                if k not in all_archs:
                    all_archs[k] = v
    
    arch_list = list(all_archs.values())
    accs = [a['accuracy'] for a in arch_list]
    bs_sens = [a['bs_sensitivity'] for a in arch_list]
    energies = [a['train_energy_kwh']*1000 for a in arch_list]
    
    sc = ax.scatter(bs_sens, accs, c=energies, cmap='viridis', s=8, alpha=0.5, edgecolors='none')
    ax.set_xlabel('Batch Size Sensitivity (CV)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('BS Sensitivity vs Accuracy', fontweight='bold', fontsize=8)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label('Energy (Wh)', fontsize=7)
    
    plt.tight_layout()
    savefig(fig, 'fig10_bs_sensitivity')
    print("  Fig 10: BS sensitivity")

# ============================================================
# FIGURE 11: TRAINING vs INFERENCE ENERGY PARETO
# ============================================================
def fig11_train_vs_inf():
    data = load_results()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    
    sd = data['cifar10']['1']
    
    # (a) Training energy Pareto
    ax = axes[0]
    if 'train_energy_nas' in sd:
        for p in sd['train_energy_nas']['population']:
            ax.scatter(p['metrics']['accuracy'], p['metrics']['train_energy_kwh']*1000,
                      c=COLORS['train_energy'], alpha=0.4, s=12, marker='o', zorder=3)
    if 'inf_energy_nas' in sd:
        for p in sd['inf_energy_nas']['population']:
            ax.scatter(p['metrics']['accuracy'], p['metrics']['train_energy_kwh']*1000,
                      c=COLORS['inf_energy'], alpha=0.4, s=12, marker='^', zorder=3, edgecolors='#333333', linewidths=0.3)
    for p in sd['ours']['population']:
        ax.scatter(p['metrics']['accuracy'], p['metrics']['train_energy_kwh']*1000,
                  c=COLORS['ours'], alpha=0.4, s=12, marker='s', zorder=4)
    
    ax.set_xlabel('Accuracy (%)')
    ax.set_ylabel('Training Energy (Wh)')
    ax.set_title('(a) Training Energy', fontweight='bold', loc='left')
    
    # (b) Inference energy Pareto
    ax = axes[1]
    if 'train_energy_nas' in sd:
        for p in sd['train_energy_nas']['population']:
            ax.scatter(p['metrics']['accuracy'], p['metrics']['inf_energy_j']*1000,
                      c=COLORS['train_energy'], alpha=0.4, s=12, marker='o', zorder=3)
    if 'inf_energy_nas' in sd:
        for p in sd['inf_energy_nas']['population']:
            ax.scatter(p['metrics']['accuracy'], p['metrics']['inf_energy_j']*1000,
                      c=COLORS['inf_energy'], alpha=0.4, s=12, marker='^', zorder=3, edgecolors='#333333', linewidths=0.3)
    for p in sd['ours']['population']:
        ax.scatter(p['metrics']['accuracy'], p['metrics']['inf_energy_j']*1000,
                  c=COLORS['ours'], alpha=0.4, s=12, marker='s', zorder=4)
    
    ax.set_xlabel('Accuracy (%)')
    ax.set_ylabel('Inference Energy (mJ)')
    ax.set_title('(b) Inference Energy', fontweight='bold', loc='left')
    
    handles = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['ours'], markersize=5, label='GreenNAS (4-obj)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['train_energy'], markersize=5, label='Train-Energy NAS'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS['inf_energy'], markersize=5, label='Inf-Energy NAS'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.08), frameon=False, fontsize=7)
    
    plt.tight_layout()
    savefig(fig, 'fig11_train_vs_inf')
    print("  Fig 11: Train vs Inference energy")

# ============================================================
# TABLE DATA
# ============================================================
def print_tables():
    data = load_results()
    
    for dataset in ['cifar10', 'cifar100']:
        print(f"\n  Table: {dataset.upper()} Summary")
        print(f"  {'Method':<18} {'Acc (%)':<16} {'Energy (Wh)':<18} {'Cost (¢)':<16} {'Lat (ms)':<14} {'HV':<14}")
        print(f"  {'-'*96}")
        
        for name, key in [('GreenNAS', 'ours'), ('Random', 'random'), ('FLOPs-only', 'flops_only'),
                          ('Weighted-Sum', 'weighted_sum'), ('ε-Constraint', 'epsilon_constraint'),
                          ('Filter+Rank', 'filter_rank')]:
            accs=[]; es=[]; cs=[]; ls=[]; hvs=[]
            for seed in ['1','2','3','4','5']:
                sd = data[dataset][seed]
                if key not in sd: continue
                if 'population' in sd[key]:
                    items = sd[key]['population']
                    accs.append(max(p['metrics']['accuracy'] for p in items))
                    es.append(min(p['metrics']['train_energy_kwh'] for p in items)*1000)
                    cs.append(min(p['metrics']['dollar_cost'] for p in items)*100)
                    ls.append(min(p['metrics']['latency_ms'] for p in items))
                elif 'results' in sd[key]:
                    items = sd[key]['results']
                    accs.append(max(r['accuracy'] for r in items))
                    es.append(min(r['train_energy_kwh'] for r in items)*1000)
                    cs.append(min(r['dollar_cost'] for r in items)*100)
                    ls.append(min(r['latency_ms'] for r in items))
                if 'metrics' in sd and key in sd['metrics'].get('hypervolume', {}):
                    hvs.append(sd['metrics']['hypervolume'][key])
            
            if accs:
                hv_str = f"{np.mean(hvs):.4f}±{np.std(hvs):.4f}" if hvs else "N/A"
                print(f"  {name:<18} {np.mean(accs):.1f}±{np.std(accs):.1f}     "
                      f"{np.mean(es):.2f}±{np.std(es):.2f}       "
                      f"{np.mean(cs):.2f}±{np.std(cs):.2f}     "
                      f"{np.mean(ls):.2f}±{np.std(ls):.2f}   "
                      f"{hv_str}")

# ============================================================
# MAIN
# ============================================================
def main():
    print("Generating figures...")
    fig1_overview()
    fig2_pareto()
    fig3_hypervolume()
    fig4_correlations()
    fig5_reproducibility()
    fig6_convergence()
    has_nvml = fig7_proxy_validation()
    fig8_ablation()
    fig9_constraints()
    fig10_bs_sensitivity()
    fig11_train_vs_inf()
    print_tables()
    print(f"\nAll figures saved to {FIG_DIR}/")
    if not has_nvml:
        print("NOTE: fig7 (proxy validation) will be generated once Modal job completes.")

if __name__ == '__main__':
    main()
