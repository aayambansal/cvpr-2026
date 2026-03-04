"""
Generate V2 publication figures for CVPR-NAS'26 paper.
Uses CIFAR-100 (main), ViT-S/16, and Data Regime results.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
import os

# Publication style
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'text.usetex': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

FIGDIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
RESDIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'v2')
os.makedirs(FIGDIR, exist_ok=True)

# Color palette (colorblind-safe)
C_BLUE = '#0072B2'
C_ORANGE = '#E69F00'
C_GREEN = '#009E73'
C_RED = '#D55E00'
C_PURPLE = '#CC79A7'
C_CYAN = '#56B4E9'
C_YELLOW = '#F0E442'
C_GRAY = '#999999'


def load_results():
    with open(os.path.join(RESDIR, 'cifar100_results.json')) as f:
        cifar = json.load(f)
    with open(os.path.join(RESDIR, 'vit_small_results.json')) as f:
        vit_s = json.load(f)
    with open(os.path.join(RESDIR, 'data_regime_results.json')) as f:
        data_regime = json.load(f)
    with open(os.path.join(RESDIR, 'adalora_results_fixed.json')) as f:
        adalora = json.load(f)
    return cifar, vit_s, data_regime, adalora


# ============================================================
# Figure 1: Proxy Correlation Scatter (per-proxy and combined)
# ============================================================
def fig_proxy_correlation(cifar):
    ft = cifar['finetune_results']
    proxy_all = cifar['proxy_scores']
    sq = cifar['selection_quality']

    # Build lookup
    proxy_map = {r['label']: r for r in proxy_all}

    # Collect matched data
    labels, accs, gn, sn, fi, ne, comb = [], [], [], [], [], [], []
    for r in ft:
        lab = r['label']
        if lab == 'linear_probe' or r['val_acc'] == 0:
            continue
        if lab not in proxy_map:
            continue
        p = proxy_map[lab]
        labels.append(lab)
        accs.append(r['val_acc'])
        gn.append(p.get('gradnorm', 0))
        sn.append(p.get('snip', 0))
        fi.append(p.get('fisher', 0))
        ne.append(p.get('neg_entropy', 0))
        comb.append(p.get('combined', 0))

    accs = np.array(accs)
    metrics = {
        'GradNorm': (np.array(gn), sq.get('gradnorm_spearman', 0)),
        'SNIP': (np.array(sn), sq.get('snip_spearman', 0)),
        'Fisher': (np.array(fi), sq.get('fisher_spearman', 0)),
        'Combined': (np.array(comb), sq.get('spearman_rho', 0)),
    }
    colors = [C_BLUE, C_ORANGE, C_GREEN, C_RED]

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.8))
    for ax, (name, (vals, rho)), color in zip(axes, metrics.items(), colors):
        ax.scatter(vals, accs, s=12, alpha=0.6, color=color, edgecolors='white', linewidth=0.3)
        # Add trend line
        if len(vals) > 2 and np.std(vals) > 0:
            z = np.polyfit(vals, accs, 1)
            p = np.poly1d(z)
            x_line = np.linspace(vals.min(), vals.max(), 50)
            ax.plot(x_line, p(x_line), '--', color='black', alpha=0.4, linewidth=0.8)
        ax.set_xlabel(name)
        if ax == axes[0]:
            ax.set_ylabel('Val. Accuracy (%)')
        rho_str = f'ρ = {rho:.3f}'
        ax.text(0.05, 0.95, rho_str, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax.set_title(name, fontsize=9)

    plt.tight_layout(w_pad=0.8)
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGDIR, f'proxy_correlation_v2.{ext}'))
    plt.close(fig)
    print("  [1] proxy_correlation_v2")


# ============================================================
# Figure 2: Selection Quality Bar Chart
# ============================================================
def fig_selection_quality(cifar):
    sq = cifar['selection_quality']

    ks = [1, 3, 5, 10]
    regret_oracle = [sq.get(f'top{k}_regret_vs_oracle', 0) for k in ks]
    regret_random = [sq.get(f'top{k}_regret_vs_random', 0) for k in ks]
    hit_rates = [sq.get(f'top{k}_hit_rate', 0) for k in ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.2))

    x = np.arange(len(ks))
    w = 0.35
    ax1.bar(x - w/2, regret_oracle, w, label='vs. Oracle', color=C_BLUE, alpha=0.85)
    ax1.bar(x + w/2, regret_random, w, label='vs. Random', color=C_ORANGE, alpha=0.85)
    ax1.set_xlabel('Top-k')
    ax1.set_ylabel('Accuracy Gap (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'k={k}' for k in ks])
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.set_title('Regret Analysis')
    ax1.axhline(y=0, color='black', linewidth=0.5)

    ax2.bar(x, hit_rates, color=C_GREEN, alpha=0.85)
    ax2.set_xlabel('Top-k')
    ax2.set_ylabel('Hit Rate')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'k={k}' for k in ks])
    ax2.set_ylim(0, 1.0)
    ax2.set_title('Hit Rate (proxy top-k ∩ oracle top-k)')
    for i, v in enumerate(hit_rates):
        ax2.text(i, v + 0.02, f'{v:.1%}', ha='center', fontsize=7)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGDIR, f'selection_quality_v2.{ext}'))
    plt.close(fig)
    print("  [2] selection_quality_v2")


# ============================================================
# Figure 3: Data Regime Line Plot
# ============================================================
def fig_data_regime(data_regime):
    fracs = sorted(set(d['data_frac'] for d in data_regime))
    configs = sorted(set(d['config'] for d in data_regime if d['config'] != 'linear_probe'))
    
    # Build data
    data = {}
    for d in data_regime:
        data[(d['config'], d['data_frac'])] = d['val_acc']

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    
    # Config display settings
    style_map = {
        'uniform_r4': (C_CYAN, '-', 'o'),
        'uniform_r8': (C_BLUE, '-', 's'),
        'uniform_r16': (C_GREEN, '-', '^'),
        'uniform_r32': (C_ORANGE, '-', 'D'),
        'attn_only_r8': (C_PURPLE, '--', 'v'),
        'attn_only_r16': (C_RED, '--', '<'),
        'mlp_only_r8': (C_GRAY, '--', '>'),
        'increasing': ('#333333', '-.', 'p'),
    }

    for cfg in configs:
        accs = [data.get((cfg, f), None) for f in fracs]
        if any(a is None for a in accs):
            # Skip configs with missing fractions
            valid_fracs = [f for f, a in zip(fracs, accs) if a is not None]
            valid_accs = [a for a in accs if a is not None]
        else:
            valid_fracs = fracs
            valid_accs = accs
        
        color, ls, marker = style_map.get(cfg, (C_GRAY, '-', '.'))
        pcts = [f * 100 for f in valid_fracs]
        ax.plot(pcts, valid_accs, ls, marker=marker, color=color,
                markersize=4, linewidth=1.2, label=cfg.replace('_', ' '))

    # Linear probe
    lp_accs = [data.get(('linear_probe', f), None) for f in fracs]
    valid_lp_fracs = [f for f, a in zip(fracs, lp_accs) if a is not None]
    valid_lp_accs = [a for a in lp_accs if a is not None]
    pcts_lp = [f * 100 for f in valid_lp_fracs]
    ax.plot(pcts_lp, valid_lp_accs, ':', marker='x', color='black',
            markersize=5, linewidth=1.5, label='linear probe')

    ax.set_xlabel('Training Data (%)')
    ax.set_ylabel('Val. Accuracy (%)')
    ax.set_title('CIFAR-100: Accuracy vs. Data Budget')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=6.5, ncol=1)
    ax.set_xticks([f * 100 for f in fracs])
    ax.set_xticklabels([f'{f*100:.0f}%' for f in fracs])

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGDIR, f'data_regime_v2.{ext}'))
    plt.close(fig)
    print("  [3] data_regime_v2")


# ============================================================
# Figure 4: ViT-B vs ViT-S Comparison
# ============================================================
def fig_backbone_comparison(cifar, vit_s):
    # ViT-B configs from cifar finetune results
    vit_b_map = {}
    for r in cifar['finetune_results']:
        vit_b_map[r['label']] = r['val_acc']

    # ViT-S configs
    vit_s_map = {}
    for r in vit_s:
        vit_s_map[r['label']] = r['val_acc']

    # Common configs
    common = sorted(set(vit_b_map.keys()) & set(vit_s_map.keys()) - {'linear_probe', 'no_adapter'})
    if not common:
        print("  [4] SKIP - no common configs between ViT-B and ViT-S")
        return

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    
    x = np.arange(len(common))
    w = 0.35
    b_accs = [vit_b_map[c] for c in common]
    s_accs = [vit_s_map[c] for c in common]
    
    ax.bar(x - w/2, b_accs, w, label='ViT-B/16', color=C_BLUE, alpha=0.85)
    ax.bar(x + w/2, s_accs, w, label='ViT-S/16', color=C_ORANGE, alpha=0.85)
    
    # Add linear probe
    lp_b = vit_b_map.get('linear_probe', 0)
    lp_s = vit_s_map.get('linear_probe', 0)
    ax.axhline(y=lp_b, color=C_BLUE, linestyle=':', linewidth=0.8, alpha=0.5)
    ax.axhline(y=lp_s, color=C_ORANGE, linestyle=':', linewidth=0.8, alpha=0.5)
    ax.text(len(common) - 0.5, lp_b + 0.3, f'LP-B={lp_b:.1f}', fontsize=6, color=C_BLUE)
    ax.text(len(common) - 0.5, lp_s + 0.3, f'LP-S={lp_s:.1f}', fontsize=6, color=C_ORANGE)
    
    ax.set_ylabel('Val. Accuracy (%)')
    ax.set_title('CIFAR-100: Backbone Comparison (2% data)')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in common], fontsize=6, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(50, 85)
    
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGDIR, f'backbone_comparison_v2.{ext}'))
    plt.close(fig)
    print("  [4] backbone_comparison_v2")


# ============================================================
# Figure 5: Pareto Front (accuracy vs params)
# ============================================================
def fig_pareto_front(cifar):
    ft = cifar['finetune_results']
    sq = cifar['selection_quality']
    proxy_all = cifar['proxy_scores']
    proxy_map = {r['label']: r.get('combined', 0) for r in proxy_all}

    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    # Categorize configs
    labels_ft = []
    params_ft = []
    accs_ft = []
    types_ft = []
    
    for r in ft:
        if r['label'] == 'linear_probe' or r['val_acc'] == 0:
            continue
        labels_ft.append(r['label'])
        params_ft.append(r['n_params'] / 1e6)  # millions
        accs_ft.append(r['val_acc'])
        if r['label'].startswith('evo_'):
            types_ft.append('evo')
        elif r['label'].startswith('rand_'):
            types_ft.append('random')
        else:
            types_ft.append('structured')

    params_ft = np.array(params_ft)
    accs_ft = np.array(accs_ft)

    # Plot by type
    for typ, color, marker, label in [
        ('structured', C_BLUE, 'o', 'Structured'),
        ('random', C_GRAY, 's', 'Random'),
        ('evo', C_GREEN, '^', 'Evolutionary'),
    ]:
        mask = np.array([t == typ for t in types_ft])
        if mask.sum() > 0:
            ax.scatter(params_ft[mask], accs_ft[mask], s=20, c=color,
                      marker=marker, alpha=0.7, label=label, edgecolors='white', linewidth=0.3)

    # Highlight top-5 by proxy
    valid_proxy = [(lab, proxy_map.get(lab, 0)) for lab in labels_ft]
    valid_proxy.sort(key=lambda x: x[1], reverse=True)
    top5_labels = set(l for l, _ in valid_proxy[:5])
    for i, lab in enumerate(labels_ft):
        if lab in top5_labels:
            ax.scatter(params_ft[i], accs_ft[i], s=60, facecolors='none',
                      edgecolors=C_RED, linewidth=1.5, zorder=5)

    # Oracle best
    best_idx = np.argmax(accs_ft)
    ax.scatter(params_ft[best_idx], accs_ft[best_idx], s=80, marker='*',
              color=C_RED, zorder=6, label=f'Oracle ({labels_ft[best_idx]})')

    # Linear probe
    lp = next((r for r in ft if r['label'] == 'linear_probe'), None)
    if lp:
        ax.scatter(lp['n_params'] / 1e6, lp['val_acc'], s=40, marker='x',
                  color='black', linewidth=1.5, zorder=5, label='Linear Probe')

    # Pareto frontier
    sorted_idx = np.argsort(params_ft)
    pareto_params = []
    pareto_accs = []
    best_so_far = -1
    for idx in sorted_idx:
        if accs_ft[idx] > best_so_far:
            pareto_params.append(params_ft[idx])
            pareto_accs.append(accs_ft[idx])
            best_so_far = accs_ft[idx]
    ax.plot(pareto_params, pareto_accs, '--', color=C_RED, alpha=0.4, linewidth=1.0)

    ax.set_xlabel('Trainable Parameters (M)')
    ax.set_ylabel('Val. Accuracy (%)')
    ax.set_title('CIFAR-100: Accuracy–Parameter Pareto Front')
    ax.legend(loc='lower right', fontsize=6.5)
    
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGDIR, f'pareto_front_v2.{ext}'))
    plt.close(fig)
    print("  [5] pareto_front_v2")


# ============================================================
# Figure 6: Evolutionary Search Convergence
# ============================================================
def fig_evo_convergence(cifar):
    proxy_all = cifar['proxy_scores']
    
    # Extract evo configs and their proxy scores
    evo_scores = []
    for r in proxy_all:
        if r['label'].startswith('evo_'):
            idx = int(r['label'].split('_')[1])
            evo_scores.append((idx, r.get('combined', 0)))
    
    if not evo_scores:
        print("  [6] SKIP - no evo configs")
        return
    
    evo_scores.sort(key=lambda x: x[0])
    idxs = [x[0] for x in evo_scores]
    scores = [x[1] for x in evo_scores]
    
    # Group into generations (pop_size=20, 6 gens = 120 configs)
    pop_size = 20
    n_gens = len(evo_scores) // pop_size + (1 if len(evo_scores) % pop_size else 0)
    
    gen_best = []
    gen_mean = []
    gen_labels = []
    for g in range(n_gens):
        start = g * pop_size
        end = min((g + 1) * pop_size, len(scores))
        gen_scores = scores[start:end]
        if gen_scores:
            gen_best.append(max(gen_scores))
            gen_mean.append(np.mean(gen_scores))
            gen_labels.append(f'Gen {g}')
    
    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    
    x = np.arange(len(gen_best))
    ax.plot(x, gen_best, '-o', color=C_RED, markersize=5, linewidth=1.5, label='Best in Gen')
    ax.plot(x, gen_mean, '-s', color=C_BLUE, markersize=4, linewidth=1.0, label='Mean in Gen')
    ax.fill_between(x, gen_mean, gen_best, alpha=0.15, color=C_BLUE)
    
    # Also show the initial structured + random baseline
    struct_scores = [r.get('combined', 0) for r in proxy_all
                    if not r['label'].startswith('evo_') and r.get('combined', 0) > 0]
    if struct_scores:
        ax.axhline(y=max(struct_scores), color=C_GRAY, linestyle='--', linewidth=0.8,
                  label=f'Best initial (struct+rand)')
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Combined Proxy Score')
    ax.set_title('Evolutionary Search Convergence')
    ax.set_xticks(x)
    ax.set_xticklabels(gen_labels, fontsize=7)
    ax.legend(fontsize=7)
    
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGDIR, f'evo_convergence_v2.{ext}'))
    plt.close(fig)
    print("  [6] evo_convergence_v2")


# ============================================================
# Figure 7: Per-proxy Spearman comparison bar
# ============================================================
def fig_proxy_comparison_bar(cifar):
    sq = cifar['selection_quality']
    
    proxies = ['GradNorm', 'SNIP', 'Fisher', 'NegEntropy', 'Combined']
    rhos = [
        sq.get('gradnorm_spearman', 0),
        sq.get('snip_spearman', 0),
        sq.get('fisher_spearman', 0),
        sq.get('neg_entropy_spearman', 0),
        sq.get('spearman_rho', 0),
    ]
    colors = [C_BLUE, C_ORANGE, C_GREEN, C_PURPLE, C_RED]
    
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    x = np.arange(len(proxies))
    bars = ax.bar(x, rhos, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    
    for i, (bar, rho) in enumerate(zip(bars, rhos)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{rho:.3f}', ha='center', fontsize=7, fontweight='bold')
    
    ax.set_ylabel('Spearman ρ')
    ax.set_title('Proxy–Accuracy Correlation (CIFAR-100)')
    ax.set_xticks(x)
    ax.set_xticklabels(proxies, fontsize=7, rotation=15)
    ax.set_ylim(0, max(rhos) * 1.2)
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGDIR, f'proxy_comparison_bar_v2.{ext}'))
    plt.close(fig)
    print("  [7] proxy_comparison_bar_v2")


# ============================================================
# Figure 8: Config heatmap (per-layer rank allocation)
# ============================================================
def fig_config_heatmap(cifar):
    """Show rank allocation heatmap for top-5 configs by accuracy."""
    ft = cifar['finetune_results']
    ft_sorted = sorted([r for r in ft if r['val_acc'] > 0 and r['label'] != 'linear_probe'],
                       key=lambda x: x['val_acc'], reverse=True)
    
    # We'd need the actual per-layer configs to make this properly.
    # Instead, show a summary table of structured config properties.
    # For now, skip this if we don't have per-layer data.
    # The paper will use a LaTeX table instead.
    print("  [8] SKIP - config heatmap needs per-layer data (use LaTeX table)")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("Loading results...")
    cifar, vit_s, data_regime, adalora = load_results()
    
    print(f"CIFAR-100: {len(cifar['proxy_scores'])} proxy, {len(cifar['finetune_results'])} finetune")
    print(f"ViT-S: {len(vit_s)} configs")
    print(f"Data regime: {len(data_regime)} entries")
    print()
    
    print("Generating figures...")
    fig_proxy_correlation(cifar)
    fig_selection_quality(cifar)
    fig_data_regime(data_regime)
    fig_backbone_comparison(cifar, vit_s)
    fig_pareto_front(cifar)
    fig_evo_convergence(cifar)
    fig_proxy_comparison_bar(cifar)
    fig_config_heatmap(cifar)
    
    print("\nDone! Figures saved to:", FIGDIR)
