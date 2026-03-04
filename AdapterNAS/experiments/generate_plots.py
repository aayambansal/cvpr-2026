"""Generate all publication-quality figures for the AdapterNAS paper."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from scipy import stats

plt.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

COLORS = {
    'uniform': '#0077BB',
    'attn': '#33BBEE',
    'mlp': '#009988',
    'partial': '#EE7733',
    'structured': '#CC3311',
    'random': '#BBBBBB',
    'baseline': '#000000',
    'proxy_top': '#CC3311',
    'proxy_bot': '#0077BB',
}

def load_results():
    with open('../results/cifar100_finetune.json') as f:
        c100 = json.load(f)
    with open('../results/flowers102_finetune.json') as f:
        f102 = json.load(f)
    with open('../results/cifar100_proxy.json') as f:
        c100_proxy = json.load(f)
    return c100, f102, c100_proxy


def categorize(label):
    if 'uniform' in label: return 'uniform'
    if 'attn' in label: return 'attn'
    if 'mlp' in label: return 'mlp'
    if label in ['last4_r16', 'first4_r16', 'first4_r8', 'even_r16', 'even_r8']: return 'partial'
    if label in ['increasing', 'decreasing', 'high_attn_low_mlp', 'low_attn_high_mlp']: return 'structured'
    if 'rand' in label: return 'random'
    return 'baseline'


# ---- Figure 1: Pareto front (Accuracy vs Params) for both datasets ----
def plot_pareto(c100, f102):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    
    for ax, data, title in [(ax1, c100, 'CIFAR-100'), (ax2, f102, 'Flowers-102')]:
        for r in data:
            cat = categorize(r['label'])
            color = COLORS.get(cat, '#888888')
            marker = 'o' if cat != 'baseline' else 's'
            size = 40 if cat != 'random' else 20
            alpha = 0.9 if cat != 'random' else 0.5
            
            ax.scatter(r['n_params'] / 1e6, r['val_acc'],
                      c=color, s=size, marker=marker, alpha=alpha,
                      edgecolors='white', linewidth=0.5, zorder=3)
            
            # Label key points
            if r['label'] in ['uniform_r8', 'uniform_r32', 'linear_probe', 'increasing',
                              'attn_only_r8', 'first4_r16']:
                offset = (5, 5) if r['val_acc'] > 55 else (5, -10)
                ax.annotate(r['label'].replace('_', ' '), 
                           (r['n_params']/1e6, r['val_acc']),
                           fontsize=6, xytext=offset, textcoords='offset points',
                           arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        
        # Compute and draw Pareto front
        points = [(r['n_params']/1e6, r['val_acc']) for r in data if r['label'] != 'linear_probe']
        points.sort(key=lambda x: x[0])
        pareto = []
        best_acc = -1
        for p, a in points:
            if a > best_acc:
                pareto.append((p, a))
                best_acc = a
        if pareto:
            px, py = zip(*pareto)
            ax.plot(px, py, 'k--', alpha=0.4, linewidth=1, zorder=2)
        
        ax.set_xlabel('Added Parameters (M)')
        ax.set_ylabel('Validation Accuracy (%)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['uniform'], markersize=6, label='Uniform'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['attn'], markersize=6, label='Attn-only'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['mlp'], markersize=6, label='MLP-only'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['partial'], markersize=6, label='Layer-subset'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['structured'], markersize=6, label='Structured'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['random'], markersize=6, label='Random'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['baseline'], markersize=6, label='Linear probe'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
               bbox_to_anchor=(0.5, -0.08), frameon=True, fancybox=True)
    
    plt.tight_layout()
    plt.savefig('../figures/pareto_front.pdf', bbox_inches='tight')
    plt.savefig('../figures/pareto_front.png', bbox_inches='tight')
    print("Saved pareto_front.pdf/png")


# ---- Figure 2: Proxy correlation with accuracy ----
def plot_proxy_correlation(c100):
    fig, axes = plt.subplots(1, 4, figsize=(7, 2.2))
    
    # Filter out linear_probe
    data = [r for r in c100 if r['label'] != 'linear_probe' and r.get('gradnorm', 0) > 0]
    
    metrics = [('gradnorm', 'GradNorm'), ('snip', 'SNIP'), 
               ('fisher', 'Fisher'), ('neg_entropy', 'Neg. Entropy')]
    
    for ax, (metric, name) in zip(axes, metrics):
        x = [r[metric] for r in data]
        y = [r['val_acc'] for r in data]
        
        for r in data:
            cat = categorize(r['label'])
            color = COLORS.get(cat, '#888888')
            ax.scatter(r[metric], r['val_acc'], c=color, s=20, alpha=0.7,
                      edgecolors='white', linewidth=0.3)
        
        # Correlation
        if len(x) > 2:
            rho, pval = stats.spearmanr(x, y)
            tau, _ = stats.kendalltau(x, y)
            ax.set_title(f'{name}\n$\\rho$={rho:.2f}, $\\tau$={tau:.2f}', fontsize=8)
        
        ax.set_xlabel(name, fontsize=7)
        if ax == axes[0]:
            ax.set_ylabel('Val Acc (%)', fontsize=8)
        ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('../figures/proxy_correlation.pdf', bbox_inches='tight')
    plt.savefig('../figures/proxy_correlation.png', bbox_inches='tight')
    print("Saved proxy_correlation.pdf/png")


# ---- Figure 3: Bar chart comparing adapter configurations ----
def plot_config_comparison(c100, f102):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))
    
    # Select key configs present in both
    key_labels = ['linear_probe', 'uniform_r4', 'attn_only_r8', 'uniform_r8',
                  'attn_only_r16', 'mlp_only_r8', 'first4_r16', 'high_attn_low_mlp',
                  'uniform_r16', 'increasing', 'uniform_r32']
    
    for ax, data, title in [(ax1, c100, 'CIFAR-100'), (ax2, f102, 'Flowers-102')]:
        # Sort by accuracy
        filtered = [r for r in data if r['label'] in key_labels]
        filtered.sort(key=lambda x: x['val_acc'])
        
        names = [r['label'].replace('_', '\n') for r in filtered]
        accs = [r['val_acc'] for r in filtered]
        params = [r['n_params'] for r in filtered]
        colors_list = [COLORS.get(categorize(r['label']), '#888') for r in filtered]
        
        bars = ax.barh(range(len(names)), accs, color=colors_list, alpha=0.85,
                       edgecolor='white', linewidth=0.5)
        
        # Add param counts
        for i, (a, p) in enumerate(zip(accs, params)):
            ax.text(a - 1, i, f'{p/1e6:.1f}M', ha='right', va='center',
                   fontsize=5.5, color='white', fontweight='bold')
        
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=6)
        ax.set_xlabel('Validation Accuracy (%)')
        ax.set_title(title)
        ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../figures/config_comparison.pdf', bbox_inches='tight')
    plt.savefig('../figures/config_comparison.png', bbox_inches='tight')
    print("Saved config_comparison.pdf/png")


# ---- Figure 4: Proxy score distribution (all 52 configs) ----
def plot_proxy_distribution(c100_proxy):
    fig, axes = plt.subplots(2, 2, figsize=(7, 4))
    
    valid = [r for r in c100_proxy if r['n_params'] > 0]
    
    metrics = [('gradnorm', 'GradNorm'), ('snip', 'SNIP'), 
               ('fisher', 'Fisher'), ('neg_entropy', 'Neg. Entropy')]
    
    for ax, (metric, name) in zip(axes.flat, metrics):
        vals = [r[metric] for r in valid]
        
        # Color by structured vs random
        struct_vals = [r[metric] for r in valid if not r['label'].startswith('rand')]
        rand_vals = [r[metric] for r in valid if r['label'].startswith('rand')]
        
        ax.hist(rand_vals, bins=15, alpha=0.5, color=COLORS['random'], label='Random', density=True)
        ax.hist(struct_vals, bins=10, alpha=0.7, color=COLORS['uniform'], label='Structured', density=True)
        
        ax.set_xlabel(name)
        ax.set_ylabel('Density')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('../figures/proxy_distribution.pdf', bbox_inches='tight')
    plt.savefig('../figures/proxy_distribution.png', bbox_inches='tight')
    print("Saved proxy_distribution.pdf/png")


# ---- Figure 5: Search space heatmap visualization ----
def plot_search_heatmap(c100):
    """Visualize the layer-wise adapter pattern of top configs."""
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))
    
    configs_to_show = [
        ('uniform_r8', 'Uniform r=8 (Best)'),
        ('increasing', 'Increasing Rank'),
        ('high_attn_low_mlp', 'High Attn, Low MLP'),
    ]
    
    module_labels = ['QKV', 'MLP fc1', 'MLP fc2']
    
    for ax, (cfg_name, title) in zip(axes, configs_to_show):
        # Reconstruct config
        if cfg_name == 'uniform_r8':
            grid = np.full((3, 12), 8)
        elif cfg_name == 'increasing':
            ranks = [4,4,4,8,8,8,16,16,16,32,32,32]
            grid = np.array([ranks, ranks, ranks])
        elif cfg_name == 'high_attn_low_mlp':
            grid = np.array([[16]*12, [4]*12, [4]*12])
        
        im = ax.imshow(grid, aspect='auto', cmap='YlOrRd', vmin=0, vmax=32)
        ax.set_xticks(range(12))
        ax.set_xticklabels([f'{i}' for i in range(12)], fontsize=6)
        ax.set_yticks(range(3))
        ax.set_yticklabels(module_labels, fontsize=7)
        ax.set_xlabel('Layer', fontsize=8)
        ax.set_title(title, fontsize=8)
        
        # Add rank values
        for i in range(3):
            for j in range(12):
                ax.text(j, i, str(int(grid[i, j])), ha='center', va='center',
                       fontsize=5, color='black' if grid[i,j] < 20 else 'white')
    
    fig.colorbar(im, ax=axes, label='LoRA Rank', shrink=0.8)
    plt.tight_layout()
    plt.savefig('../figures/search_heatmap.pdf', bbox_inches='tight')
    plt.savefig('../figures/search_heatmap.png', bbox_inches='tight')
    print("Saved search_heatmap.pdf/png")


# ---- Figure 6: Accuracy vs Latency ----
def plot_acc_vs_latency(c100):
    fig, ax = plt.subplots(figsize=(3.5, 3))
    
    for r in c100:
        cat = categorize(r['label'])
        color = COLORS.get(cat, '#888')
        marker = 'o' if cat != 'baseline' else 's'
        size = 40 if cat != 'random' else 20
        
        ax.scatter(r['latency_ms'], r['val_acc'], c=color, s=size, 
                  marker=marker, alpha=0.8, edgecolors='white', linewidth=0.5)
    
    ax.set_xlabel('Inference Latency (ms)')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('CIFAR-100: Accuracy vs. Latency')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../figures/acc_vs_latency.pdf', bbox_inches='tight')
    plt.savefig('../figures/acc_vs_latency.png', bbox_inches='tight')
    print("Saved acc_vs_latency.pdf/png")


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs('../figures', exist_ok=True)
    
    c100, f102, c100_proxy = load_results()
    
    plot_pareto(c100, f102)
    plot_proxy_correlation(c100)
    plot_config_comparison(c100, f102)
    plot_proxy_distribution(c100_proxy)
    plot_search_heatmap(c100)
    plot_acc_vs_latency(c100)
    
    print("\nAll figures generated!")
