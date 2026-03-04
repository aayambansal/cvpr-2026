"""
Generate all publication figures for the RepNAS paper.
Uses the real experimental data from repnas_results.json.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats
import seaborn as sns

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
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Okabe-Ito colorblind-safe palette
COLORS = {
    'CNN': '#E69F00',
    'EffNet': '#56B4E9', 
    'Mobile': '#009E73',
    'ConvNeXt': '#F0E442',
    'ViT': '#0072B2',
    'Swin': '#D55E00',
    'DeiT': '#CC79A7',
    'RegNet': '#000000',
}

FAMILY_MARKERS = {
    'CNN': 'o',
    'EffNet': 's',
    'Mobile': '^',
    'ConvNeXt': 'D',
    'ViT': 'v',
    'Swin': 'P',
    'DeiT': 'X',
    'RegNet': 'p',
}

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Load results
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiments', 'repnas_results.json')) as f:
    data = json.load(f)

results = data['results']
names = list(results.keys())


def get_arrays(*keys):
    arrs = []
    for k in keys:
        arrs.append(np.array([results[n][k] for n in names]))
    return arrs if len(arrs) > 1 else arrs[0]


# ============================================================
# Figure 2: Correlation scatter plots — Our metrics vs GT Accuracy
# ============================================================
def fig2_correlation_scatter():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    
    metrics = [
        ('cka_pretrained', 'Linear CKA', 'a'),
        ('cosine_pretrained', 'Centered Cosine', 'b'),
        ('knn_pretrained', 'Mutual kNN', 'c'),
    ]
    
    gt = get_arrays('gt_acc')
    
    for ax, (key, label, panel) in zip(axes, metrics):
        vals = get_arrays(key)
        
        for name in names:
            r = results[name]
            fam = r['family']
            ax.scatter(r[key], r['gt_acc'], 
                      c=COLORS[fam], marker=FAMILY_MARKERS[fam],
                      s=35, alpha=0.85, edgecolors='white', linewidths=0.3,
                      zorder=3)
        
        # Trend line
        mask = np.isfinite(vals)
        if mask.sum() > 2:
            z = np.polyfit(vals[mask], gt[mask], 1)
            p = np.poly1d(z)
            x_line = np.linspace(vals[mask].min(), vals[mask].max(), 100)
            ax.plot(x_line, p(x_line), 'k--', alpha=0.4, linewidth=0.8)
            
            sp, p_val = stats.spearmanr(vals[mask], gt[mask])
            ax.text(0.05, 0.95, f'ρ = {sp:.3f}', transform=ax.transAxes,
                   fontsize=7, va='top', ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(f'{label} Score')
        ax.set_ylabel('ImageNet Top-1 Acc (%)' if panel == 'a' else '')
        ax.text(-0.12, 1.05, f'({panel})', transform=ax.transAxes,
               fontsize=10, fontweight='bold', va='top')
    
    # Legend
    handles = [mpatches.Patch(color=COLORS[f], label=f) for f in COLORS if any(results[n]['family'] == f for n in names)]
    axes[2].legend(handles=handles, loc='lower right', fontsize=5, ncol=2,
                  framealpha=0.8, handlelength=1.0, handletextpad=0.3, columnspacing=0.5)
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig2_correlation_scatter.pdf'))
    fig.savefig(os.path.join(FIG_DIR, 'fig2_correlation_scatter.png'))
    plt.close()
    print("  Figure 2: Correlation scatter plots saved")


# ============================================================
# Figure 3: Bar chart comparing Spearman correlations of all methods
# ============================================================
def fig3_correlation_comparison():
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    
    gt = get_arrays('gt_acc')
    
    method_data = []
    for label, key, color, is_ours in [
        ('CKA\n(pretrained)', 'cka_pretrained', '#0072B2', True),
        ('Cosine\n(pretrained)', 'cosine_pretrained', '#56B4E9', True),
        ('kNN\n(pretrained)', 'knn_pretrained', '#009E73', True),
        ('CKA\n(random)', 'cka_random', '#0072B2', True),
        ('GradNorm', 'gradnorm', '#999999', False),
        ('NASWOT', 'naswot', '#999999', False),
        ('SynFlow', 'synflow', '#999999', False),
    ]:
        vals = get_arrays(key)
        mask = np.isfinite(vals)
        if mask.sum() > 5:
            rho, _ = stats.spearmanr(vals[mask], gt[mask])
            method_data.append((label, abs(rho), is_ours, color))
    
    # Sort by correlation
    method_data.sort(key=lambda x: x[1], reverse=True)
    
    labels = [d[0] for d in method_data]
    rhos = [d[1] for d in method_data]
    colors_bar = ['#0072B2' if d[2] else '#BBBBBB' for d in method_data]
    hatches = ['' if d[2] else '///' for d in method_data]
    
    bars = ax.barh(range(len(labels)), rhos, color=colors_bar, edgecolor='white', height=0.6)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('|Spearman ρ| with ImageNet Accuracy')
    ax.set_xlim(0, 0.75)
    ax.invert_yaxis()
    
    # Add value labels
    for i, v in enumerate(rhos):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=6)
    
    # Legend
    ours_patch = mpatches.Patch(facecolor='#0072B2', label='Ours (RepNAS)')
    base_patch = mpatches.Patch(facecolor='#BBBBBB', hatch='///', label='ZS-NAS Baselines')
    ax.legend(handles=[ours_patch, base_patch], loc='lower right', fontsize=6, framealpha=0.8)
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig3_correlation_comparison.pdf'))
    fig.savefig(os.path.join(FIG_DIR, 'fig3_correlation_comparison.png'))
    plt.close()
    print("  Figure 3: Correlation comparison bar chart saved")


# ============================================================
# Figure 4: Architecture family analysis — per-family accuracy vs RepNAS score
# ============================================================
def fig4_family_analysis():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    
    # Panel A: Per-family mean accuracy vs mean GradNorm (strongest proxy)
    ax = axes[0]
    families = {}
    for n in names:
        r = results[n]
        fam = r['family']
        if fam not in families:
            families[fam] = {'acc': [], 'gradnorm': [], 'cka_rand': []}
        families[fam]['acc'].append(r['gt_acc'])
        families[fam]['gradnorm'].append(r['gradnorm'])
        families[fam]['cka_rand'].append(r['cka_random'])
    
    for fam in families:
        mean_acc = np.mean(families[fam]['acc'])
        std_acc = np.std(families[fam]['acc'])
        mean_gn = np.mean(families[fam]['gradnorm'])
        ax.errorbar(mean_gn, mean_acc, yerr=std_acc,
                   fmt=FAMILY_MARKERS.get(fam, 'o'), color=COLORS[fam],
                   markersize=8, capsize=3, label=fam, linewidth=1,
                   markeredgecolor='white', markeredgewidth=0.5)
    
    ax.set_xlabel('Mean GradNorm Score')
    ax.set_ylabel('Mean ImageNet Top-1 Acc (%)')
    ax.legend(fontsize=6, ncol=2, loc='lower right', framealpha=0.8)
    ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')
    
    # Panel B: Rank comparison — top-5 by each method
    ax = axes[1]
    
    # Get rankings by different methods
    gt = {n: results[n]['gt_acc'] for n in names}
    gn = {n: results[n]['gradnorm'] for n in names}
    cka = {n: results[n]['cka_random'] for n in names}
    
    gt_rank = sorted(names, key=lambda n: gt[n], reverse=True)[:10]
    gn_rank = sorted(names, key=lambda n: gn[n], reverse=True)[:10]
    cka_rank = sorted(names, key=lambda n: cka[n], reverse=True)[:10]
    
    # Compute overlap
    gt_top5 = set(gt_rank[:5])
    gt_top10 = set(gt_rank[:10])
    gn_top5_overlap = len(gt_top5 & set(gn_rank[:5]))
    gn_top10_overlap = len(gt_top10 & set(gn_rank[:10]))
    cka_top5_overlap = len(gt_top5 & set(cka_rank[:5]))
    cka_top10_overlap = len(gt_top10 & set(cka_rank[:10]))
    
    x = np.arange(2)
    width = 0.3
    ax.bar(x - width/2, [gn_top5_overlap, gn_top10_overlap], width, label='GradNorm', color='#BBBBBB', hatch='///')
    ax.bar(x + width/2, [cka_top5_overlap, cka_top10_overlap], width, label='CKA (random)', color='#0072B2')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Top-5', 'Top-10'])
    ax.set_ylabel('Overlap with GT Ranking')
    ax.set_ylim(0, 10.5)
    ax.legend(fontsize=6, framealpha=0.8)
    ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig4_family_analysis.pdf'))
    fig.savefig(os.path.join(FIG_DIR, 'fig4_family_analysis.png'))
    plt.close()
    print("  Figure 4: Family analysis saved")


# ============================================================
# Figure 5: Heatmap of all scores
# ============================================================
def fig5_heatmap():
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    
    # Sort by GT accuracy
    sorted_names = sorted(names, key=lambda n: results[n]['gt_acc'], reverse=True)
    
    metrics = ['gt_acc', 'gradnorm', 'cka_random', 'cka_pretrained', 'knn_pretrained', 'naswot']
    metric_labels = ['GT Accuracy', 'GradNorm', 'CKA (rand)', 'CKA (pre)', 'kNN (pre)', 'NASWOT']
    
    # Build matrix (normalized per column for visualization)
    mat = np.zeros((len(sorted_names), len(metrics)))
    for i, n in enumerate(sorted_names):
        for j, m in enumerate(metrics):
            mat[i, j] = results[n][m]
    
    # Normalize columns to [0, 1]
    mat_norm = np.zeros_like(mat)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        valid = np.isfinite(col)
        if valid.sum() > 0:
            mn, mx = col[valid].min(), col[valid].max()
            if mx > mn:
                mat_norm[:, j] = np.where(valid, (col - mn) / (mx - mn), 0.5)
            else:
                mat_norm[:, j] = 0.5
    
    # Short names for y-axis
    short_names = [n.replace('_patch16', '').replace('_patch4_window7', '') for n in sorted_names]
    
    im = ax.imshow(mat_norm, aspect='auto', cmap='viridis', interpolation='nearest')
    
    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, rotation=30, ha='right', fontsize=7)
    ax.set_yticks(range(len(short_names)))
    ax.set_yticklabels(short_names, fontsize=5.5)
    
    # Add text annotations
    for i in range(mat_norm.shape[0]):
        for j in range(mat_norm.shape[1]):
            if j == 0:
                txt = f'{mat[i,j]:.1f}'
            else:
                txt = f'{mat[i,j]:.3f}'
            color = 'white' if mat_norm[i, j] < 0.5 else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=4, color=color)
    
    plt.colorbar(im, ax=ax, shrink=0.6, label='Normalized Score')
    ax.set_title('Architecture Ranking by Different Metrics', fontsize=9)
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig5_heatmap.pdf'))
    fig.savefig(os.path.join(FIG_DIR, 'fig5_heatmap.png'))
    plt.close()
    print("  Figure 5: Heatmap saved")


# ============================================================
# Table data for the paper
# ============================================================
def generate_table_data():
    """Print LaTeX-ready table data."""
    gt = get_arrays('gt_acc')
    
    print("\n  === TABLE 1: Full Results ===")
    sorted_names = sorted(names, key=lambda n: results[n]['gt_acc'], reverse=True)
    
    print("  Architecture & Family & Params & GT Acc & CKA(pre) & Cos(pre) & kNN(pre) & CKA(rand) & GradNorm & NASWOT \\\\")
    for n in sorted_names:
        r = results[n]
        print(f"  {n} & {r['family']} & {r['params']}M & {r['gt_acc']:.1f} & "
              f"{r['cka_pretrained']:.4f} & {r['cosine_pretrained']:.4f} & {r['knn_pretrained']:.4f} & "
              f"{r['cka_random']:.4f} & {r['gradnorm']:.2f} & {r['naswot']:.1f} \\\\")
    
    print("\n  === TABLE 2: Correlation Summary ===")
    corr = data.get('correlations', {})
    for method, vals in corr.items():
        print(f"  {method}: ρ={vals['spearman_rho']:.4f}, τ={vals['kendall_tau']:.4f}")


if __name__ == '__main__':
    print("Generating publication figures...")
    fig2_correlation_scatter()
    fig3_correlation_comparison()
    fig4_family_analysis()
    fig5_heatmap()
    generate_table_data()
    print("\nAll figures saved to:", FIG_DIR)
