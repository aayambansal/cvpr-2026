"""
Generate publication-quality figures for BudgetNAS v2 paper.
7 methods, 3 seeds (mean±std), budget sweep, gradual shift.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    'legend.fontsize': 6.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Okabe-Ito colorblind-safe palette (7 methods)
COLORS = {
    'fixed':   '#0072B2',  # Blue
    'growing': '#D55E00',  # Red-orange
    'ewc':     '#CC79A7',  # Pink
    'er':      '#F0E442',  # Yellow
    'den':     '#56B4E9',  # Light blue
    'rnas':    '#999999',  # Gray
    'bnas':    '#009E73',  # Green (ours)
}
MARKERS = {'fixed':'o','growing':'s','ewc':'D','er':'v','den':'p','rnas':'x','bnas':'^'}
LABELS = {
    'fixed':   'Fixed Backbone',
    'growing': 'Growing (Naive)',
    'ewc':     'EWC',
    'er':      'Exp. Replay',
    'den':     'DEN-style',
    'rnas':    'RandomNAS',
    'bnas':    'BudgetNAS (Ours)',
}
MKS = ['fixed','growing','ewc','er','den','rnas','bnas']
DOM = ['cifar10','cifar100','svhn']
DOM_LABELS = {'cifar10':'CIFAR-10','cifar100':'CIFAR-100','svhn':'SVHN'}
SEEDS = ['42','123','7']

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, '..', 'results')
FIG = os.path.join(BASE, '..', 'figures')
os.makedirs(FIG, exist_ok=True)

with open(os.path.join(RES, 'v2_results.json')) as f:
    ALL = json.load(f)


def _save(fig, name):
    fig.savefig(os.path.join(FIG, f'{name}.pdf'))
    fig.savefig(os.path.join(FIG, f'{name}.png'))
    plt.close(fig)
    print(f'  Saved {name}')


# ═══════════════════════════════════════════════════════════════
# Fig 1: Accuracy timeline (mean of 3 seeds, 7 methods)
# ═══════════════════════════════════════════════════════════════
def fig1_accuracy_timeline():
    fig, ax = plt.subplots(figsize=(7, 3))
    n_steps = 12  # 3 domains × 4 chunks

    for mk in MKS:
        # Collect per-step accs across seeds
        accs_per_step = {i: [] for i in range(n_steps)}
        for s in SEEDS:
            tl = ALL['seeds'][s][mk]['timeline']
            for t in tl:
                accs_per_step[t['step']].append(t['acc'] * 100)
        steps = sorted(accs_per_step.keys())
        means = [np.mean(accs_per_step[i]) for i in steps]
        stds = [np.std(accs_per_step[i]) for i in steps]

        ax.plot(steps, means, color=COLORS[mk], marker=MARKERS[mk],
                markersize=4, linewidth=1.5, label=LABELS[mk], zorder=3 if mk == 'bnas' else 2)
        ax.fill_between(steps, np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        color=COLORS[mk], alpha=0.1)

    # Domain boundaries
    for b in [3.5, 7.5]:
        ax.axvline(x=b, color='gray', linestyle='--', alpha=0.5, lw=0.8)
    ax.text(1.5, 3, 'CIFAR-10', ha='center', fontsize=7, fontstyle='italic', color='gray')
    ax.text(5.5, 3, 'CIFAR-100', ha='center', fontsize=7, fontstyle='italic', color='gray')
    ax.text(9.5, 3, 'SVHN', ha='center', fontsize=7, fontstyle='italic', color='gray')

    ax.set_xlabel('Streaming Step')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_ylim(0, 60)
    ax.set_xticks(range(n_steps))
    ax.legend(loc='upper left', ncol=2, frameon=True, framealpha=0.9, edgecolor='none')
    fig.tight_layout()
    _save(fig, 'fig1_accuracy_timeline')


# ═══════════════════════════════════════════════════════════════
# Fig 2: Parameters and blocks over time (7 methods)
# ═══════════════════════════════════════════════════════════════
def fig2_params_blocks():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.5))
    n_steps = 12
    for mk in MKS:
        # Use seed 42 for structure (params/blocks are deterministic per architecture)
        tl = ALL['seeds']['42'][mk]['timeline']
        steps = [t['step'] for t in tl]
        params = [t['p'] / 1e6 for t in tl]
        blocks = [t['bl'] for t in tl]
        ax1.plot(steps, params, color=COLORS[mk], marker=MARKERS[mk],
                 markersize=3, lw=1.2, label=LABELS[mk])
        ax2.plot(steps, blocks, color=COLORS[mk], marker=MARKERS[mk],
                 markersize=3, lw=1.2, label=LABELS[mk])

    for ax in [ax1, ax2]:
        for b in [3.5, 7.5]:
            ax.axvline(x=b, color='gray', linestyle='--', alpha=0.5, lw=0.8)

    ax1.set_xlabel('Streaming Step'); ax1.set_ylabel('Parameters (M)')
    ax1.text(-0.12, 1.05, 'A', transform=ax1.transAxes, fontsize=10, fontweight='bold', va='top')
    ax1.legend(loc='upper left', fontsize=5.5, ncol=2, frameon=True, framealpha=0.9, edgecolor='none')

    ax2.set_xlabel('Streaming Step'); ax2.set_ylabel('Number of Blocks')
    ax2.text(-0.12, 1.05, 'B', transform=ax2.transAxes, fontsize=10, fontweight='bold', va='top')
    ax2.set_yticks(range(0, 14, 2))

    fig.tight_layout()
    _save(fig, 'fig2_params_blocks')


# ═══════════════════════════════════════════════════════════════
# Fig 3: Bar chart — final accuracy per dataset (7 methods, with error bars)
# ═══════════════════════════════════════════════════════════════
def fig3_final_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.8), sharey=False)
    agg = ALL['aggregated']

    for di, dk in enumerate(DOM):
        ax = axes[di]
        x = np.arange(len(MKS))
        means = [agg[mk][dk]['fm'] for mk in MKS]
        stds = [agg[mk][dk]['fs'] for mk in MKS]
        colors = [COLORS[mk] for mk in MKS]

        bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, width=0.65,
                      edgecolor='white', capsize=2, error_kw={'lw': 0.8})

        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 1.2, f'{m:.1f}', ha='center', va='bottom', fontsize=5.5)

        ax.set_title(DOM_LABELS[dk], fontsize=9, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[mk].replace(' (Ours)', '\n(Ours)').replace(' (Naive)', '\n(Naive)').replace('-style', '')
                            for mk in MKS], fontsize=5, rotation=45, ha='right')
        if di == 0:
            ax.set_ylabel('Final Accuracy (%)')
        ax.set_ylim(0, max(means) + max(stds) + 12)

    fig.tight_layout()
    _save(fig, 'fig3_final_comparison')


# ═══════════════════════════════════════════════════════════════
# Fig 4: Budget sweep — B vs final accuracy (3 datasets)
# ═══════════════════════════════════════════════════════════════
def fig4_budget_sweep():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))
    bs_data = ALL['budget_sweep']
    budgets = [0, 1, 2, 3, 5]
    ds_colors = {'cifar10': '#0072B2', 'cifar100': '#D55E00', 'svhn': '#009E73'}
    ds_markers = {'cifar10': 'o', 'cifar100': 's', 'svhn': '^'}

    # Panel A: Final accuracy vs B
    for dk in DOM:
        accs = []
        for B in budgets:
            sb = str(B)
            tl = bs_data[sb]['timeline']
            es = [t for t in tl if t['ds'] == dk]
            accs.append(es[-1]['acc'] * 100 if es else 0)
        ax1.plot(budgets, accs, color=ds_colors[dk], marker=ds_markers[dk],
                 markersize=5, lw=1.5, label=DOM_LABELS[dk])

    ax1.set_xlabel('Budget B (mutations per domain)')
    ax1.set_ylabel('Final Accuracy (%)')
    ax1.set_xticks(budgets)
    ax1.legend(frameon=True, framealpha=0.9, edgecolor='none')
    ax1.text(-0.12, 1.05, 'A', transform=ax1.transAxes, fontsize=10, fontweight='bold', va='top')

    # Panel B: Final params vs B
    params = []
    total_muts = []
    for B in budgets:
        sb = str(B)
        tl = bs_data[sb]['timeline']
        params.append(tl[-1]['p'] / 1e6)
        total_muts.append(bs_data[sb]['over']['mut'])

    ax2.bar(range(len(budgets)), params, color='#009E73', alpha=0.7, width=0.5)
    for i, (p, m) in enumerate(zip(params, total_muts)):
        ax2.text(i, p + 0.1, f'{p:.1f}M\n({int(m)} mut)', ha='center', va='bottom', fontsize=6)
    ax2.set_xlabel('Budget B')
    ax2.set_ylabel('Final Parameters (M)')
    ax2.set_xticks(range(len(budgets)))
    ax2.set_xticklabels(budgets)
    ax2.text(-0.12, 1.05, 'B', transform=ax2.transAxes, fontsize=10, fontweight='bold', va='top')

    fig.tight_layout()
    _save(fig, 'fig4_budget_sweep')


# ═══════════════════════════════════════════════════════════════
# Fig 5: Gradual shift stream comparison
# ═══════════════════════════════════════════════════════════════
def fig5_gradual_shift():
    fig, ax = plt.subplots(figsize=(7, 2.8))
    grad = ALL['gradual']

    for meth, color, marker, label in [
        ('fixed', '#0072B2', 'o', 'Fixed (Gradual)'),
        ('bnas', '#009E73', '^', 'BudgetNAS (Gradual)'),
    ]:
        tl = grad[meth]['timeline']
        steps = [t['step'] for t in tl]
        accs = [t['acc'] * 100 for t in tl]
        labels_s = [t['label'] for t in tl]
        ax.plot(steps, accs, color=color, marker=marker, markersize=5, lw=1.5, label=label)

    # Mark blend chunks
    tl = grad['fixed']['timeline']
    for t in tl:
        if 'blend' in t['label']:
            ax.axvspan(t['step'] - 0.4, t['step'] + 0.4, color='orange', alpha=0.15)

    # Domain regions
    ax.text(1, 2, 'CIFAR-10', ha='center', fontsize=7, fontstyle='italic', color='gray')
    ax.text(3, 2, 'Blend', ha='center', fontsize=6, fontstyle='italic', color='orange')
    ax.text(5.5, 2, 'CIFAR-100', ha='center', fontsize=7, fontstyle='italic', color='gray')
    ax.text(7, 2, 'Blend', ha='center', fontsize=6, fontstyle='italic', color='orange')
    ax.text(9, 2, 'SVHN', ha='center', fontsize=7, fontstyle='italic', color='gray')

    ax.set_xlabel('Streaming Step')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_ylim(0, 50)
    ax.legend(frameon=True, framealpha=0.9, edgecolor='none')
    fig.tight_layout()
    _save(fig, 'fig5_gradual_shift')


# ═══════════════════════════════════════════════════════════════
# Fig 6: Mutation timeline for BudgetNAS (seed 42)
# ═══════════════════════════════════════════════════════════════
def fig6_mutation_timeline():
    fig, ax = plt.subplots(figsize=(7, 2.5))
    tl = ALL['seeds']['42']['bnas']['timeline']
    arch = ALL['seeds']['42']['bnas']['arch']

    steps = [t['step'] for t in tl]
    blocks = [t['bl'] for t in tl]
    accs = [t['acc'] * 100 for t in tl]

    ax.plot(steps, blocks, color=COLORS['bnas'], marker='^', markersize=5, lw=1.5, label='Num. Blocks')

    # Mark mutations
    mut_colors = {'add_blk': '#E69F00', 'add_ds': '#56B4E9', 'rm_blk': '#CC79A7'}
    mut_labels = {'add_blk': 'Add Block', 'add_ds': 'Add Downsample', 'rm_blk': 'Remove Block'}
    plotted = set()
    for a in arch:
        step = a['step']
        mut = a['m']
        y = [t['bl'] for t in tl if t['step'] == step]
        if not y:
            continue
        y = y[0]
        lbl = mut_labels[mut] if mut not in plotted else None
        ax.scatter(step, y, color=mut_colors[mut], s=80, zorder=5,
                   edgecolors='black', linewidths=0.5, marker='*', label=lbl)
        plotted.add(mut)

    for b in [3.5, 7.5]:
        ax.axvline(x=b, color='gray', linestyle='--', alpha=0.5, lw=0.8)

    ax.text(1.5, 1, 'CIFAR-10', ha='center', fontsize=7, fontstyle='italic', color='gray')
    ax.text(5.5, 1, 'CIFAR-100', ha='center', fontsize=7, fontstyle='italic', color='gray')
    ax.text(9.5, 1, 'SVHN', ha='center', fontsize=7, fontstyle='italic', color='gray')

    ax.set_xlabel('Streaming Step')
    ax.set_ylabel('Number of Blocks')
    ax.set_ylim(0, 13)
    ax.set_xticks(range(12))
    ax.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='none')
    fig.tight_layout()
    _save(fig, 'fig6_mutations')


# ═══════════════════════════════════════════════════════════════
# Fig 7: Efficiency scatter — accuracy vs params (all 7 methods, 3 datasets)
# ═══════════════════════════════════════════════════════════════
def fig7_efficiency():
    fig, ax = plt.subplots(figsize=(4, 3))
    agg = ALL['aggregated']

    for mk in MKS:
        for dk in DOM:
            a = agg[mk][dk]
            ax.scatter(a['p'] / 1e6, a['fm'], color=COLORS[mk],
                       marker=MARKERS[mk], s=40, zorder=3 if mk == 'bnas' else 2,
                       edgecolors='white', linewidths=0.5)

    # Legend (methods only)
    handles = [Line2D([0], [0], marker=MARKERS[mk], color='w',
               markerfacecolor=COLORS[mk], markersize=6, label=LABELS[mk]) for mk in MKS]
    ax.legend(handles=handles, loc='upper left', frameon=True, framealpha=0.9, edgecolor='none', fontsize=5.5)
    ax.set_xlabel('Parameters (M)')
    ax.set_ylabel('Final Accuracy (%)')
    fig.tight_layout()
    _save(fig, 'fig7_efficiency')


# ═══════════════════════════════════════════════════════════════
# Table: Full results with mean±std
# ═══════════════════════════════════════════════════════════════
def table_summary():
    agg = ALL['aggregated']
    lines = []
    lines.append('=' * 110)
    lines.append(f'{"Method":<22} {"Dataset":<12} {"Final Acc":>14} {"Avg Acc":>14} {"Params":>10} {"Blocks":>6} {"Time(s)":>8} {"Muts":>5}')
    lines.append('=' * 110)

    for mk in MKS:
        first = True
        for dk in DOM:
            a = agg[mk][dk]
            name = LABELS[mk] if first else ''
            lines.append(
                f'{name:<22} {DOM_LABELS[dk]:<12} '
                f'{a["fm"]:5.1f} ± {a["fs"]:4.1f}%  '
                f'{a["am"]:5.1f} ± {a["a_s"]:4.1f}%  '
                f'{a["p"]/1e6:7.2f}M  {a["bl"]:4d}  '
                + (f'{agg[mk]["wt"]:7.0f}  {agg[mk]["mut"]:4.0f}' if first else '')
            )
            first = False
        lines.append('-' * 110)

    # Budget sweep
    lines.append('')
    lines.append('BUDGET SWEEP (seed 42)')
    lines.append(f'{"B":>3} {"CIFAR-10":>10} {"CIFAR-100":>10} {"SVHN":>10} {"Params(M)":>10} {"Muts":>5}')
    bs = ALL['budget_sweep']
    for B in ['0','1','2','3','5']:
        tl = bs[B]['timeline']
        row = f'{B:>3} '
        for dk in DOM:
            es = [t for t in tl if t['ds'] == dk]
            row += f'{es[-1]["acc"]*100:9.1f}% '
        row += f'{tl[-1]["p"]/1e6:9.2f}  {bs[B]["over"]["mut"]:4.0f}'
        lines.append(row)

    table_str = '\n'.join(lines)
    with open(os.path.join(RES, 'v2_summary_table.txt'), 'w') as f:
        f.write(table_str)
    print(table_str)


if __name__ == '__main__':
    print('Generating v2 figures...')
    fig1_accuracy_timeline()
    fig2_params_blocks()
    fig3_final_comparison()
    fig4_budget_sweep()
    fig5_gradual_shift()
    fig6_mutation_timeline()
    fig7_efficiency()
    table_summary()
    print(f'\nAll figures saved to {FIG}')
