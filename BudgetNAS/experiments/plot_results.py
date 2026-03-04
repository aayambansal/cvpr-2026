"""
Generate publication-quality figures for the BudgetNAS paper.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Okabe-Ito colorblind-safe palette
COLORS = {
    'fixed': '#0072B2',     # Blue
    'growing': '#D55E00',   # Red-orange
    'budget': '#009E73',    # Green
}
MARKERS = {
    'fixed': 'o',
    'growing': 's',
    'budget': '^',
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load results
with open(os.path.join(RESULTS_DIR, "experiment_results.json"), "r") as f:
    all_results = json.load(f)


def fig1_accuracy_timeline():
    """Figure 1: Test accuracy over streaming steps for all methods."""
    fig, ax = plt.subplots(figsize=(7, 2.8))
    
    ds_boundaries = [5, 10]  # step indices where datasets change
    ds_labels = ['CIFAR-10', 'CIFAR-100', 'SVHN']
    
    for method_key, label in [('fixed_backbone', 'Fixed Backbone'),
                               ('growing_backbone', 'Growing (Naive)'),
                               ('budget_nas', 'BudgetNAS (Ours)')]:
        timeline = all_results[method_key]['timeline']
        steps = [t['step'] for t in timeline]
        accs = [t['test_acc'] * 100 for t in timeline]
        ax.plot(steps, accs, color=COLORS[method_key.split('_')[0] if method_key != 'budget_nas' else 'budget'],
                marker=MARKERS[method_key.split('_')[0] if method_key != 'budget_nas' else 'budget'],
                markersize=4, linewidth=1.5, label=label)
    
    # Add domain shift indicators
    for boundary in ds_boundaries:
        ax.axvline(x=boundary - 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    
    # Add dataset labels
    ax.text(2, 2, 'CIFAR-10', ha='center', fontsize=8, fontstyle='italic', color='gray')
    ax.text(7, 2, 'CIFAR-100', ha='center', fontsize=8, fontstyle='italic', color='gray')
    ax.text(12, 2, 'SVHN', ha='center', fontsize=8, fontstyle='italic', color='gray')
    
    ax.set_xlabel('Streaming Step')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='none')
    ax.set_xticks(range(15))
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig1_accuracy_timeline.pdf'))
    fig.savefig(os.path.join(FIGURES_DIR, 'fig1_accuracy_timeline.png'))
    plt.close()
    print("Saved fig1_accuracy_timeline")


def fig2_params_compute():
    """Figure 2: Parameters and compute over time (dual axis)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.5))
    
    # Panel A: Parameters over time
    for method_key, label in [('fixed_backbone', 'Fixed Backbone'),
                               ('growing_backbone', 'Growing (Naive)'),
                               ('budget_nas', 'BudgetNAS (Ours)')]:
        timeline = all_results[method_key]['timeline']
        steps = [t['step'] for t in timeline]
        params = [t['num_params'] / 1e6 for t in timeline]
        color = COLORS[method_key.split('_')[0] if method_key != 'budget_nas' else 'budget']
        ax1.plot(steps, params, color=color,
                marker=MARKERS[method_key.split('_')[0] if method_key != 'budget_nas' else 'budget'],
                markersize=3, linewidth=1.5, label=label)
    
    for boundary in [5, 10]:
        ax1.axvline(x=boundary - 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    
    ax1.set_xlabel('Streaming Step')
    ax1.set_ylabel('Parameters (M)')
    ax1.text(-0.15, 1.05, 'A', transform=ax1.transAxes, fontsize=10, fontweight='bold', va='top')
    ax1.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='none', fontsize=6)
    
    # Panel B: Number of blocks over time
    for method_key, label in [('fixed_backbone', 'Fixed Backbone'),
                               ('growing_backbone', 'Growing (Naive)'),
                               ('budget_nas', 'BudgetNAS (Ours)')]:
        timeline = all_results[method_key]['timeline']
        steps = [t['step'] for t in timeline]
        blocks = [t['num_blocks'] for t in timeline]
        color = COLORS[method_key.split('_')[0] if method_key != 'budget_nas' else 'budget']
        ax2.plot(steps, blocks, color=color,
                marker=MARKERS[method_key.split('_')[0] if method_key != 'budget_nas' else 'budget'],
                markersize=3, linewidth=1.5, label=label)
    
    for boundary in [5, 10]:
        ax2.axvline(x=boundary - 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    
    ax2.set_xlabel('Streaming Step')
    ax2.set_ylabel('Number of Blocks')
    ax2.text(-0.15, 1.05, 'B', transform=ax2.transAxes, fontsize=10, fontweight='bold', va='top')
    ax2.set_yticks(range(0, 12, 2))
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig2_params_blocks.pdf'))
    fig.savefig(os.path.join(FIGURES_DIR, 'fig2_params_blocks.png'))
    plt.close()
    print("Saved fig2_params_blocks")


def fig3_final_comparison():
    """Figure 3: Bar chart comparing final accuracy on each dataset."""
    datasets = ['CIFAR-10', 'CIFAR-100', 'SVHN']
    ds_keys = ['cifar10', 'cifar100', 'svhn']
    
    methods = [
        ('fixed_backbone', 'Fixed\nBackbone', COLORS['fixed']),
        ('growing_backbone', 'Growing\n(Naive)', COLORS['growing']),
        ('budget_nas', 'BudgetNAS\n(Ours)', COLORS['budget']),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5), sharey=False)
    
    for ds_idx, (ds_name, ds_key) in enumerate(zip(datasets, ds_keys)):
        ax = axes[ds_idx]
        x = np.arange(len(methods))
        bars = []
        for m_idx, (m_key, m_label, m_color) in enumerate(methods):
            timeline = all_results[m_key]['timeline']
            ds_accs = [t['test_acc'] * 100 for t in timeline if t['dataset'] == ds_key]
            final_acc = ds_accs[-1]
            avg_acc = np.mean(ds_accs)
            bar = ax.bar(m_idx, final_acc, color=m_color, alpha=0.8, width=0.6, edgecolor='white')
            bars.append(bar)
            ax.text(m_idx, final_acc + 1, f'{final_acc:.1f}%', ha='center', va='bottom', fontsize=6.5)
        
        ax.set_title(ds_name, fontsize=9, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m[1] for m in methods], fontsize=6)
        ax.set_ylabel('Final Accuracy (%)' if ds_idx == 0 else '')
        ax.set_ylim(0, max(100, max([t['test_acc'] * 100 for t in all_results['budget_nas']['timeline'] if t['dataset'] == ds_key]) + 15))
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig3_final_comparison.pdf'))
    fig.savefig(os.path.join(FIGURES_DIR, 'fig3_final_comparison.png'))
    plt.close()
    print("Saved fig3_final_comparison")


def fig4_efficiency_frontier():
    """Figure 4: Accuracy vs Parameters efficiency plot."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    
    for method_key, label in [('fixed_backbone', 'Fixed Backbone'),
                               ('growing_backbone', 'Growing (Naive)'),
                               ('budget_nas', 'BudgetNAS (Ours)')]:
        timeline = all_results[method_key]['timeline']
        
        # Plot one point per dataset (final accuracy, final params)
        for ds_key, ds_label, marker_offset in [('cifar10', 'C10', 0), ('cifar100', 'C100', 0.1), ('svhn', 'SVHN', 0.2)]:
            ds_entries = [t for t in timeline if t['dataset'] == ds_key]
            final_entry = ds_entries[-1]
            params_m = final_entry['num_params'] / 1e6
            acc = final_entry['test_acc'] * 100
            
            color = COLORS[method_key.split('_')[0] if method_key != 'budget_nas' else 'budget']
            marker = MARKERS[method_key.split('_')[0] if method_key != 'budget_nas' else 'budget']
            
            ax.scatter(params_m, acc, color=color, marker=marker, s=40, zorder=5,
                      edgecolors='white', linewidths=0.5)
            ax.annotate(ds_label, (params_m, acc), fontsize=5.5, ha='left',
                       xytext=(3, 2), textcoords='offset points')
    
    # Custom legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['fixed'], markersize=6, label='Fixed Backbone'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['growing'], markersize=6, label='Growing (Naive)'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS['budget'], markersize=6, label='BudgetNAS (Ours)'),
    ]
    ax.legend(handles=handles, loc='center left', frameon=True, framealpha=0.9, edgecolor='none', fontsize=6)
    
    ax.set_xlabel('Parameters (M)')
    ax.set_ylabel('Final Accuracy (%)')
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig4_efficiency.pdf'))
    fig.savefig(os.path.join(FIGURES_DIR, 'fig4_efficiency.png'))
    plt.close()
    print("Saved fig4_efficiency")


def fig5_mutation_timeline():
    """Figure 5: Architecture mutations over time for BudgetNAS."""
    budget_data = all_results['budget_nas']
    timeline = budget_data['timeline']
    arch_changes = budget_data['arch_changes']
    
    fig, ax = plt.subplots(figsize=(7, 2.2))
    
    steps = [t['step'] for t in timeline]
    blocks = [t['num_blocks'] for t in timeline]
    accs = [t['test_acc'] * 100 for t in timeline]
    
    # Plot blocks
    ax.plot(steps, blocks, color=COLORS['budget'], marker='^', markersize=4, linewidth=1.5, label='Num. Blocks')
    
    # Mark mutations
    for change in arch_changes:
        step = change['step']
        mutation = change['mutation']
        marker_color = '#E69F00' if 'add' in mutation else '#CC79A7'
        symbol = '+' if 'add' in mutation else '-'
        y_val = [t['num_blocks'] for t in timeline if t['step'] == step][0]
        ax.annotate(f'{mutation.replace("_", " ").title()}',
                   (step, y_val), fontsize=5, ha='center', va='bottom',
                   xytext=(0, 8), textcoords='offset points',
                   arrowprops=dict(arrowstyle='->', color=marker_color, lw=0.8),
                   color=marker_color, fontweight='bold')
    
    # Domain shifts
    for boundary in [5, 10]:
        ax.axvline(x=boundary - 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    
    ax.text(2, 1, 'CIFAR-10', ha='center', fontsize=7, fontstyle='italic', color='gray')
    ax.text(7, 1, 'CIFAR-100', ha='center', fontsize=7, fontstyle='italic', color='gray')
    ax.text(12, 1, 'SVHN', ha='center', fontsize=7, fontstyle='italic', color='gray')
    
    ax.set_xlabel('Streaming Step')
    ax.set_ylabel('Number of Blocks')
    ax.set_ylim(0, 12)
    ax.set_xticks(range(15))
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig5_mutations.pdf'))
    fig.savefig(os.path.join(FIGURES_DIR, 'fig5_mutations.png'))
    plt.close()
    print("Saved fig5_mutations")


def table_summary():
    """Generate a summary table as a text file for the paper."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"{'Method':<22} {'Dataset':<12} {'Final Acc':>10} {'Avg Acc':>10} {'Params':>12} {'Blocks':>8}")
    lines.append("=" * 80)
    
    for method_key, method_label in [('fixed_backbone', 'Fixed Backbone'),
                                      ('growing_backbone', 'Growing (Naive)'),
                                      ('budget_nas', 'BudgetNAS (Ours)')]:
        timeline = all_results[method_key]['timeline']
        for ds_key in ['cifar10', 'cifar100', 'svhn']:
            ds_entries = [t for t in timeline if t['dataset'] == ds_key]
            final = ds_entries[-1]
            avg_acc = np.mean([t['test_acc'] for t in ds_entries])
            lines.append(f"{method_label:<22} {ds_key:<12} {final['test_acc']*100:>9.1f}% {avg_acc*100:>9.1f}% {final['num_params']:>12,} {final['num_blocks']:>8}")
        lines.append("-" * 80)
    
    table_str = "\n".join(lines)
    with open(os.path.join(RESULTS_DIR, "summary_table.txt"), "w") as f:
        f.write(table_str)
    print(table_str)
    return table_str


if __name__ == "__main__":
    fig1_accuracy_timeline()
    fig2_params_compute()
    fig3_final_comparison()
    fig4_efficiency_frontier()
    fig5_mutation_timeline()
    table_summary()
    print(f"\nAll figures saved to {FIGURES_DIR}")
