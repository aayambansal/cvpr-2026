"""
Generate publication-quality figures for BudgetNAS v3 paper.
10 methods, 3 seeds (mean±std), budget sweep, trigger/stabilization ablations,
gradual shift, long stream, efficiency analysis.
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
    'legend.fontsize': 6,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Okabe-Ito colorblind-safe palette (10 methods)
COLORS = {
    'fixed':       '#0072B2',  # Blue
    'growing':     '#D55E00',  # Red-orange
    'ewc':         '#CC79A7',  # Pink
    'er':          '#F0E442',  # Yellow
    'den':         '#56B4E9',  # Light blue
    'smart_grow':  '#E69F00',  # Orange
    'rnas':        '#999999',  # Gray
    'bnas_heur':   '#009E73',  # Green (ours, primary)
    'bnas_ws':     '#882255',  # Wine
    'bnas_bandit': '#332288',  # Indigo
}
MARKERS = {
    'fixed':'o', 'growing':'s', 'ewc':'D', 'er':'v', 'den':'p',
    'smart_grow':'h', 'rnas':'x', 'bnas_heur':'^', 'bnas_ws':'<', 'bnas_bandit':'>'
}
LABELS = {
    'fixed':       'Fixed',
    'growing':     'Growing',
    'ewc':         'EWC',
    'er':          'Exp. Replay',
    'den':         'DEN-style',
    'smart_grow':  'Smart Growing',
    'rnas':        'RandomNAS',
    'bnas_heur':   'BudgetNAS-Heur (Ours)',
    'bnas_ws':     'BudgetNAS+WS+FZ',
    'bnas_bandit': 'BudgetNAS-Bandit',
}

# Key methods for main figures (avoid clutter)
KEY_MKS = ['fixed', 'growing', 'ewc', 'den', 'rnas', 'bnas_heur', 'bnas_bandit']
ALL_MKS = ['fixed', 'growing', 'ewc', 'er', 'den', 'smart_grow', 'rnas', 'bnas_heur', 'bnas_ws', 'bnas_bandit']

DOM = ['cifar10', 'cifar100', 'svhn']
DOM_LABELS = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100', 'svhn': 'SVHN'}
SEEDS = ['42', '123', '7']

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, '..', 'results')
FIG = os.path.join(BASE, '..', 'figures')
os.makedirs(FIG, exist_ok=True)

with open(os.path.join(RES, 'v3_results.json')) as f:
    ALL = json.load(f)

AGG = ALL['aggregated_v3']


def _save(fig, name):
    fig.savefig(os.path.join(FIG, f'{name}.pdf'))
    fig.savefig(os.path.join(FIG, f'{name}.png'))
    plt.close(fig)
    print(f'  Saved {name}')


# ═══════════════════════════════════════════════════════════════
# Fig 2: Accuracy timeline (key methods, mean+std, 3 seeds)
# ═══════════════════════════════════════════════════════════════
def fig2_accuracy_timeline():
    fig, ax = plt.subplots(figsize=(7, 3))
    n_steps = 12

    for mk in KEY_MKS:
        accs_per_step = {i: [] for i in range(n_steps)}
        for s in SEEDS:
            tl = ALL['seeds'][s][mk]['timeline']
            for t in tl:
                accs_per_step[t['step']].append(t['acc'] * 100)
        steps = sorted(accs_per_step.keys())
        means = [np.mean(accs_per_step[i]) for i in steps]
        stds = [np.std(accs_per_step[i]) for i in steps]

        lw = 2.0 if mk == 'bnas_heur' else 1.2
        zo = 5 if mk == 'bnas_heur' else 2
        ax.plot(steps, means, color=COLORS[mk], marker=MARKERS[mk],
                markersize=4, linewidth=lw, label=LABELS[mk], zorder=zo)
        ax.fill_between(steps, np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        color=COLORS[mk], alpha=0.12)

    for b in [3.5, 7.5]:
        ax.axvline(x=b, color='gray', linestyle='--', alpha=0.5, lw=0.8)
    ax.text(1.5, 3, 'CIFAR-10', ha='center', fontsize=7, fontstyle='italic', color='gray')
    ax.text(5.5, 3, 'CIFAR-100', ha='center', fontsize=7, fontstyle='italic', color='gray')
    ax.text(9.5, 3, 'SVHN', ha='center', fontsize=7, fontstyle='italic', color='gray')

    ax.set_xlabel('Streaming Step')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_ylim(0, 55)
    ax.set_xticks(range(n_steps))
    ax.legend(loc='upper left', ncol=2, frameon=True, framealpha=0.9, edgecolor='none')
    fig.tight_layout()
    _save(fig, 'fig2_accuracy_timeline')


# ═══════════════════════════════════════════════════════════════
# Fig 3: Bar chart — final accuracy per dataset (all 10 methods, error bars)
# ═══════════════════════════════════════════════════════════════
def fig3_final_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.2), sharey=False)

    for di, dk in enumerate(DOM):
        ax = axes[di]
        x = np.arange(len(ALL_MKS))
        means = [AGG[mk][dk]['fm'] for mk in ALL_MKS]
        stds = [AGG[mk][dk]['fs'] for mk in ALL_MKS]
        colors = [COLORS[mk] for mk in ALL_MKS]

        bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, width=0.7,
                      edgecolor='white', capsize=2, error_kw={'lw': 0.8})

        # Highlight the best
        best_idx = np.argmax(means)
        bars[best_idx].set_edgecolor('black')
        bars[best_idx].set_linewidth(1.5)

        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 1.0, f'{m:.1f}', ha='center', va='bottom', fontsize=4.5, rotation=45)

        ax.set_title(DOM_LABELS[dk], fontsize=9, fontweight='bold')
        ax.set_xticks(x)
        short_labels = ['Fix', 'Grow', 'EWC', 'ER', 'DEN', 'SmGr', 'Rnd', 'Heur', 'WS', 'Band']
        ax.set_xticklabels(short_labels, fontsize=5.5, rotation=45, ha='right')
        if di == 0:
            ax.set_ylabel('Final Accuracy (%)')
        ax.set_ylim(0, max(means) + max(stds) + 12)

    fig.tight_layout()
    _save(fig, 'fig3_final_comparison')


# ═══════════════════════════════════════════════════════════════
# Fig 4: Budget sweep (B=0..5, bandit controller)
# ═══════════════════════════════════════════════════════════════
def fig4_budget_sweep():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))
    bs_data = ALL['budget_sweep_v3']
    budgets = [0, 1, 2, 3, 5]
    ds_colors = {'cifar10': '#0072B2', 'cifar100': '#D55E00', 'svhn': '#009E73'}
    ds_markers = {'cifar10': 'o', 'cifar100': 's', 'svhn': '^'}

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
# Fig 5: Trigger ablation (drift-only, acc-only, both, neither)
# ═══════════════════════════════════════════════════════════════
def fig5_trigger_ablation():
    fig, ax = plt.subplots(figsize=(5, 3))
    ta = ALL['trigger_ablation']
    conditions = ['drift-only', 'acc-only', 'both', 'neither']
    cond_labels = ['Drift Only', 'Acc. Only', 'Both', 'Neither']
    cond_colors = ['#56B4E9', '#009E73', '#D55E00', '#999999']

    x = np.arange(len(DOM))
    width = 0.18

    for ci, cond in enumerate(conditions):
        tl = ta[cond]['timeline']
        accs = []
        for dk in DOM:
            es = [t for t in tl if t['ds'] == dk]
            accs.append(es[-1]['acc'] * 100 if es else 0)
        offset = (ci - 1.5) * width
        bars = ax.bar(x + offset, accs, width, label=cond_labels[ci],
                      color=cond_colors[ci], alpha=0.85, edgecolor='white')
        for j, v in enumerate(accs):
            ax.text(x[j] + offset, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=5.5)

    ax.set_ylabel('Final Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([DOM_LABELS[dk] for dk in DOM])
    ax.legend(frameon=True, framealpha=0.9, edgecolor='none')
    ax.set_title('Trigger Ablation (B=1, seed 42)', fontsize=9)
    fig.tight_layout()
    _save(fig, 'fig5_trigger_ablation')


# ═══════════════════════════════════════════════════════════════
# Fig 6: Stabilization ablation (none, ws_only, fz_only, ws+fz)
# ═══════════════════════════════════════════════════════════════
def fig6_stabilization_ablation():
    fig, ax = plt.subplots(figsize=(5, 3))
    sa = ALL['stabilization_ablation']
    conditions = ['none', 'ws_only', 'fz_only', 'ws+fz']
    cond_labels = ['None', 'Warm-Start Only', 'Freeze Only', 'WS + Freeze']
    cond_colors = ['#009E73', '#0072B2', '#D55E00', '#CC79A7']

    x = np.arange(len(DOM))
    width = 0.18

    for ci, cond in enumerate(conditions):
        tl = sa[cond]['timeline']
        accs = []
        for dk in DOM:
            es = [t for t in tl if t['ds'] == dk]
            accs.append(es[-1]['acc'] * 100 if es else 0)
        offset = (ci - 1.5) * width
        bars = ax.bar(x + offset, accs, width, label=cond_labels[ci],
                      color=cond_colors[ci], alpha=0.85, edgecolor='white')
        for j, v in enumerate(accs):
            ax.text(x[j] + offset, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=5.5)

    ax.set_ylabel('Final Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([DOM_LABELS[dk] for dk in DOM])
    ax.legend(frameon=True, framealpha=0.9, edgecolor='none')
    ax.set_title('Stabilization Ablation (B=1, Heuristic, seed 42)', fontsize=9)
    fig.tight_layout()
    _save(fig, 'fig6_stabilization_ablation')


# ═══════════════════════════════════════════════════════════════
# Fig 7: Gradual shift comparison
# ═══════════════════════════════════════════════════════════════
def fig7_gradual_shift():
    fig, ax = plt.subplots(figsize=(7, 2.8))
    grad = ALL['gradual_v3']

    method_info = [
        ('fixed',     '#0072B2', 'o', 'Fixed'),
        ('heuristic', '#009E73', '^', 'BudgetNAS-Heur'),
        ('bandit',    '#332288', '>', 'BudgetNAS-Bandit'),
    ]

    for meth, color, marker, label in method_info:
        tl = grad[meth]['timeline']
        steps = [t['step'] for t in tl]
        accs = [t['acc'] * 100 for t in tl]
        ax.plot(steps, accs, color=color, marker=marker, markersize=5, lw=1.5, label=label)

    # Mark blend chunks
    tl = grad['fixed']['timeline']
    for t in tl:
        lbl = t.get('label', '')
        if 'blend' in str(lbl).lower():
            ax.axvspan(t['step'] - 0.4, t['step'] + 0.4, color='orange', alpha=0.15)

    n = len(grad['fixed']['timeline'])
    # Domain labels
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
    _save(fig, 'fig7_gradual_shift')


# ═══════════════════════════════════════════════════════════════
# Fig 8: Long stream (4 domains, seed 42)
# ═══════════════════════════════════════════════════════════════
def fig8_long_stream():
    fig, ax = plt.subplots(figsize=(7, 3))
    ls = ALL['long_stream']
    ls_methods = ['fixed', 'growing', 'bnas_bandit', 'rnas']
    ls_colors = ['#0072B2', '#D55E00', '#332288', '#999999']
    ls_markers = ['o', 's', '>', 'x']
    ls_labels = ['Fixed', 'Growing', 'BudgetNAS-Bandit', 'RandomNAS']

    for mi, mk in enumerate(ls_methods):
        tl = ls[mk]['timeline']
        steps = [t['step'] for t in tl]
        accs = [t['acc'] * 100 for t in tl]
        ax.plot(steps, accs, color=ls_colors[mi], marker=ls_markers[mi],
                markersize=4, lw=1.3, label=ls_labels[mi])

    # Domain boundaries: 4 chunks each => 3.5, 7.5, 11.5
    for b in [3.5, 7.5, 11.5]:
        ax.axvline(x=b, color='gray', linestyle='--', alpha=0.5, lw=0.8)
    ax.text(1.5, 3, 'CIFAR-10', ha='center', fontsize=6.5, fontstyle='italic', color='gray')
    ax.text(5.5, 3, 'CIFAR-100', ha='center', fontsize=6.5, fontstyle='italic', color='gray')
    ax.text(9.5, 3, 'SVHN', ha='center', fontsize=6.5, fontstyle='italic', color='gray')
    ax.text(13.5, 3, 'FashionMNIST', ha='center', fontsize=6.5, fontstyle='italic', color='gray')

    ax.set_xlabel('Streaming Step')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_ylim(0, 85)
    ax.set_xticks(range(16))
    ax.legend(frameon=True, framealpha=0.9, edgecolor='none')
    fig.tight_layout()
    _save(fig, 'fig8_long_stream')


# ═══════════════════════════════════════════════════════════════
# Fig 9: Efficiency scatter — accuracy vs params (all methods)
# ═══════════════════════════════════════════════════════════════
def fig9_efficiency():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    # Panel A: mean final acc vs final params (from aggregated, last domain = svhn)
    for mk in ALL_MKS:
        # Mean across datasets
        mean_acc = np.mean([AGG[mk][dk]['fm'] for dk in DOM])
        # Use max params (final state)
        max_params = max(AGG[mk][dk]['p'] for dk in DOM) / 1e6
        zo = 5 if mk == 'bnas_heur' else 2
        sz = 70 if mk == 'bnas_heur' else 40
        ax1.scatter(max_params, mean_acc, color=COLORS[mk],
                   marker=MARKERS[mk], s=sz, zorder=zo,
                   edgecolors='black' if mk == 'bnas_heur' else 'white', linewidths=0.8)

    handles = [Line2D([0], [0], marker=MARKERS[mk], color='w',
               markerfacecolor=COLORS[mk], markersize=5, label=LABELS[mk]) for mk in ALL_MKS]
    ax1.legend(handles=handles, loc='upper right', frameon=True, framealpha=0.9,
               edgecolor='none', fontsize=5)
    ax1.set_xlabel('Max Parameters (M)')
    ax1.set_ylabel('Mean Final Accuracy (%)')
    ax1.text(-0.12, 1.05, 'A', transform=ax1.transAxes, fontsize=10, fontweight='bold', va='top')

    # Panel B: Gain per M params bar chart
    gain_data = []
    for mk in ALL_MKS:
        mean_acc = np.mean([AGG[mk][dk]['fm'] for dk in DOM])
        max_params = max(AGG[mk][dk]['p'] for dk in DOM) / 1e6
        gain_data.append((mk, mean_acc / max_params if max_params > 0 else 0))

    gain_data.sort(key=lambda x: x[1], reverse=True)
    x = np.arange(len(gain_data))
    bar_colors = [COLORS[g[0]] for g in gain_data]
    bar_labels = [LABELS[g[0]].replace(' (Ours)', '\n(Ours)') for g in gain_data]
    vals = [g[1] for g in gain_data]

    bars = ax2.barh(x, vals, color=bar_colors, alpha=0.85, edgecolor='white')
    # Highlight BudgetNAS-Heur
    for i, (mk, _) in enumerate(gain_data):
        if mk == 'bnas_heur':
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(1.5)

    ax2.set_yticks(x)
    ax2.set_yticklabels(bar_labels, fontsize=5.5)
    ax2.set_xlabel('Gain per M Params\n(Mean Acc / Max Params)')
    ax2.invert_yaxis()
    ax2.text(-0.12, 1.05, 'B', transform=ax2.transAxes, fontsize=10, fontweight='bold', va='top')

    fig.tight_layout()
    _save(fig, 'fig9_efficiency')


# ═══════════════════════════════════════════════════════════════
# Summary Table
# ═══════════════════════════════════════════════════════════════
def table_summary():
    lines = []
    lines.append('=' * 130)
    lines.append(f'{"Method":<28} {"CIFAR-10":>16} {"CIFAR-100":>16} {"SVHN":>16} {"MaxParams":>10} {"Blocks":>6} {"Time(s)":>8} {"Muts":>5} {"Gain/M":>8}')
    lines.append('=' * 130)

    for mk in ALL_MKS:
        a = AGG[mk]
        c10 = f'{a["cifar10"]["fm"]:.1f}±{a["cifar10"]["fs"]:.1f}'
        c100 = f'{a["cifar100"]["fm"]:.1f}±{a["cifar100"]["fs"]:.1f}'
        svhn = f'{a["svhn"]["fm"]:.1f}±{a["svhn"]["fs"]:.1f}'
        max_p = max(a[dk]['p'] for dk in DOM)
        max_bl = max(a[dk]['bl'] for dk in DOM)
        mean_acc = np.mean([a[dk]['fm'] for dk in DOM])
        gain = mean_acc / (max_p / 1e6) if max_p > 0 else 0
        lines.append(
            f'{LABELS[mk]:<28} {c10:>16} {c100:>16} {svhn:>16} '
            f'{max_p/1e6:8.2f}M  {max_bl:4d}  {a["wt"]:7.0f}  {a["mut"]:4.0f}  {gain:7.1f}'
        )

    lines.append('=' * 130)

    # Budget sweep
    lines.append('')
    lines.append('BUDGET SWEEP (bandit controller, seed 42)')
    lines.append(f'{"B":>3} {"CIFAR-10":>10} {"CIFAR-100":>10} {"SVHN":>10} {"Params(M)":>10} {"Muts":>5}')
    bs = ALL['budget_sweep_v3']
    for B in ['0', '1', '2', '3', '5']:
        tl = bs[B]['timeline']
        row = f'{B:>3} '
        for dk in DOM:
            es = [t for t in tl if t['ds'] == dk]
            row += f'{es[-1]["acc"]*100:9.1f}% '
        row += f'{tl[-1]["p"]/1e6:9.2f}  {bs[B]["over"]["mut"]:4.0f}'
        lines.append(row)

    # Trigger ablation
    lines.append('')
    lines.append('TRIGGER ABLATION (B=1, seed 42)')
    lines.append(f'{"Condition":<15} {"CIFAR-10":>10} {"CIFAR-100":>10} {"SVHN":>10} {"Muts":>5}')
    ta = ALL['trigger_ablation']
    for cond in ['drift-only', 'acc-only', 'both', 'neither']:
        tl = ta[cond]['timeline']
        row = f'{cond:<15} '
        for dk in DOM:
            es = [t for t in tl if t['ds'] == dk]
            row += f'{es[-1]["acc"]*100:9.1f}% '
        row += f'{ta[cond]["over"]["mut"]:4.0f}'
        lines.append(row)

    # Stabilization ablation
    lines.append('')
    lines.append('STABILIZATION ABLATION (B=1, Heuristic, seed 42)')
    lines.append(f'{"Condition":<15} {"CIFAR-10":>10} {"CIFAR-100":>10} {"SVHN":>10} {"Muts":>5}')
    sa = ALL['stabilization_ablation']
    for cond in ['none', 'ws_only', 'fz_only', 'ws+fz']:
        tl = sa[cond]['timeline']
        row = f'{cond:<15} '
        for dk in DOM:
            es = [t for t in tl if t['ds'] == dk]
            row += f'{es[-1]["acc"]*100:9.1f}% '
        row += f'{sa[cond]["over"]["mut"]:4.0f}'
        lines.append(row)

    # Long stream
    lines.append('')
    lines.append('LONG STREAM (4 domains, seed 42)')
    ls = ALL['long_stream']
    long_doms = ['cifar10', 'cifar100', 'svhn', 'fashion']
    long_dom_labels = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100', 'svhn': 'SVHN', 'fashion': 'FashionMNIST'}
    lines.append(f'{"Method":<20} {"CIFAR-10":>10} {"CIFAR-100":>10} {"SVHN":>10} {"Fashion":>10} {"Params(M)":>10}')
    for mk in ['fixed', 'growing', 'bnas_bandit', 'rnas']:
        tl = ls[mk]['timeline']
        row = f'{mk:<20} '
        for dk in long_doms:
            es = [t for t in tl if t['ds'] == dk]
            if es:
                row += f'{es[-1]["acc"]*100:9.1f}% '
            else:
                row += f'{"N/A":>9}  '
        row += f'{tl[-1]["p"]/1e6:9.2f}'
        lines.append(row)

    table_str = '\n'.join(lines)
    with open(os.path.join(RES, 'v3_summary_table.txt'), 'w') as f:
        f.write(table_str)
    print(table_str)


if __name__ == '__main__':
    print('Generating v3 figures...')
    fig2_accuracy_timeline()
    fig3_final_comparison()
    fig4_budget_sweep()
    fig5_trigger_ablation()
    fig6_stabilization_ablation()
    fig7_gradual_shift()
    fig8_long_stream()
    fig9_efficiency()
    table_summary()
    print(f'\nAll figures saved to {FIG}')
