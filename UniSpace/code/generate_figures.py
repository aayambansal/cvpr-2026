"""
Generate all publication figures from v2 experimental results (N=500).
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Okabe-Ito colorblind-safe palette
OI = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']

fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(fig_dir, exist_ok=True)

res_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
with open(os.path.join(res_dir, 'results_v2.json')) as f:
    results = json.load(f)
with open(os.path.join(res_dir, 'analysis_v2.json')) as f:
    analysis = json.load(f)

N = len(results)
print(f"Loaded {N} architectures from results_v2.json")


# ===================================================================
# Figure 2: Primitive Dominance Bar Chart (Token Mixers, Norms, DS)
# ===================================================================
def fig_primitive_dominance():
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.2))
    
    categories = [
        ('mixer_naswot', 'Token Mixer', ['conv', 'attention', 'gated_mlp', 'ssm_lite'],
         ['DWConv', 'Attention', 'GatedMLP', 'SSM-Lite']),
        ('norm_naswot', 'Normalization', ['batch', 'group', 'layer', 'rms'],
         ['BatchNorm', 'GroupNorm', 'LayerNorm', 'RMSNorm']),
        ('downsample_naswot', 'Downsampling', ['strided_conv', 'patch_merging', 'avgpool', 'maxpool'],
         ['StridedConv', 'PatchMerge', 'AvgPool', 'MaxPool']),
    ]
    
    for idx, (key, title, order, labels) in enumerate(categories):
        ax = axes[idx]
        data = analysis[key]
        means = [data[k]['mean'] for k in order]
        stds = [data[k]['std'] for k in order]
        ns = [data[k]['n'] for k in order]
        # Compute SEM for error bars (more appropriate than std for comparison)
        sems = [data[k]['std'] / np.sqrt(data[k]['n']) for k in order]
        
        bars = ax.bar(range(len(order)), means, yerr=sems, capsize=3,
                      color=OI[:len(order)], edgecolor='black', linewidth=0.5,
                      alpha=0.85, error_kw={'linewidth': 0.8})
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=6)
        ax.set_ylabel('NASWOT Score' if idx == 0 else '')
        ax.set_title(title, fontsize=9, fontweight='bold')
        
        # Highlight best
        best_idx = np.argmax(means)
        bars[best_idx].set_edgecolor('#D55E00')
        bars[best_idx].set_linewidth(2)
        
        # Add n= annotation
        for i, n in enumerate(ns):
            ax.text(i, means[i] - stds[i]*0.3, f'n={n}', ha='center', va='top', fontsize=5, color='gray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig_primitive_dominance.pdf'))
    plt.savefig(os.path.join(fig_dir, 'fig_primitive_dominance.png'), dpi=300)
    plt.close()
    print("  Saved fig_primitive_dominance")


# ===================================================================
# Figure 3: Score Correlation Heatmap
# ===================================================================
def fig_score_correlation():
    if 'score_corr' not in analysis:
        print("  Skipping score correlation (no data)")
        return
    
    corr = np.array(analysis['score_corr']['matrix'])
    names = analysis['score_corr']['names']
    display_names = ['NASWOT', 'SynFlow', 'GradNorm', 'SNIP', 'log(Params)']
    
    fig, ax = plt.subplots(figsize=(3.5, 3))
    
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    for i in range(len(names)):
        for j in range(len(names)):
            if not mask[i, j]:
                color = 'white' if abs(corr[i, j]) > 0.6 else 'black'
                ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center',
                       fontsize=6, color=color)
    
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(display_names, fontsize=7)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label='Pearson r')
    cbar.ax.tick_params(labelsize=6)
    
    n_corr = analysis['score_corr']['n']
    ax.set_title(f'Zero-Cost Proxy Correlations (N={n_corr})', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig_score_correlation.pdf'))
    plt.savefig(os.path.join(fig_dir, 'fig_score_correlation.png'), dpi=300)
    plt.close()
    print("  Saved fig_score_correlation")


# ===================================================================
# Figure 4: PCA Scree Plot (Effective Dimensionality)
# ===================================================================
def fig_pca_scree():
    pca = analysis['pca']
    eigenvalues = np.array(pca['eigenvalues'])
    cumulative = np.array(pca['cumulative'])
    n = min(len(eigenvalues), 15)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5))
    
    # Scree plot
    ax1.bar(range(1, n+1), eigenvalues[:n]/eigenvalues.sum()*100, 
            color=OI[1], alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Variance Explained (%)')
    ax1.set_title('Scree Plot', fontsize=9, fontweight='bold')
    ax1.set_xticks(range(1, n+1, 2))
    
    # Cumulative
    ax2.plot(range(1, n+1), cumulative[:n]*100, 'o-', color=OI[4], markersize=4, linewidth=1.5)
    ax2.axhline(90, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax2.axhline(95, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
    ax2.text(n-1, 91, '90%', fontsize=6, color='gray')
    ax2.text(n-1, 96, '95%', fontsize=6, color='gray')
    
    eff90 = pca['eff_dim_90']
    eff95 = pca['eff_dim_95']
    ax2.axvline(eff90, color=OI[5], linestyle='--', linewidth=0.8, alpha=0.7)
    ax2.axvline(eff95, color=OI[6], linestyle='--', linewidth=0.8, alpha=0.7)
    
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Variance (%)')
    ax2.set_title(f'Effective Dim. (90%={eff90}, 95%={eff95})', fontsize=9, fontweight='bold')
    ax2.set_xticks(range(1, n+1, 2))
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig_pca_scree.pdf'))
    plt.savefig(os.path.join(fig_dir, 'fig_pca_scree.png'), dpi=300)
    plt.close()
    print("  Saved fig_pca_scree")


# ===================================================================
# Figure 5: Parameter Efficiency Scatter
# ===================================================================
def fig_param_efficiency():
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    params = [r['params']/1e3 for r in results]
    naswot = [r['scores']['naswot'] for r in results]
    
    mixer_colors = {'conv': OI[0], 'attention': OI[1], 'gated_mlp': OI[2], 'ssm_lite': OI[4]}
    
    for mixer, color in mixer_colors.items():
        idxs = [i for i, r in enumerate(results) 
                if r['config']['stages'][0]['mixer'] == mixer]
        if idxs:
            px = [params[i] for i in idxs]
            py = [naswot[i] for i in idxs]
            label = {'conv': 'DWConv', 'attention': 'Attention', 
                     'gated_mlp': 'GatedMLP', 'ssm_lite': 'SSM-Lite'}[mixer]
            ax.scatter(px, py, c=color, s=12, alpha=0.5, label=label,
                      edgecolors='black', linewidth=0.2)
    
    ax.set_xlabel('Parameters (K)')
    ax.set_ylabel('NASWOT Score')
    ax.set_title(f'Parameter Efficiency (N={N})', fontsize=9, fontweight='bold')
    ax.legend(loc='best', frameon=False, fontsize=6, ncol=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig_param_efficiency.pdf'))
    plt.savefig(os.path.join(fig_dir, 'fig_param_efficiency.png'), dpi=300)
    plt.close()
    print("  Saved fig_param_efficiency")


# ===================================================================
# Figure 6: NASWOT Score Distribution
# ===================================================================
def fig_score_distribution():
    if 'naswot_dist' not in analysis:
        print("  Skipping score distribution (no data)")
        return
    
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    
    hist = analysis['naswot_dist']['hist']
    bins = analysis['naswot_dist']['bins']
    bin_centers = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
    widths = [bins[i+1] - bins[i] for i in range(len(bins)-1)]
    
    ax.bar(bin_centers, hist, width=widths, color=OI[2], alpha=0.8,
           edgecolor='black', linewidth=0.5)
    
    mean_nw = analysis['naswot_dist']['mean']
    std_nw = analysis['naswot_dist']['std']
    ax.axvline(mean_nw, color=OI[5], linestyle='--', linewidth=1.5, 
               label=f'Mean={mean_nw:.1f}, SD={std_nw:.1f}')
    
    ax.set_xlabel('NASWOT Score')
    ax.set_ylabel('Count')
    ax.set_title(f'Score Distribution (N={N})', fontsize=9, fontweight='bold')
    ax.legend(frameon=False, fontsize=7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig_score_distribution.pdf'))
    plt.savefig(os.path.join(fig_dir, 'fig_score_distribution.png'), dpi=300)
    plt.close()
    print("  Saved fig_score_distribution")


# ===================================================================
# NEW Figure 7: Top-k Regret Analysis Bar Chart
# ===================================================================
def fig_regret_analysis():
    if 'regret_analysis' not in analysis:
        print("  Skipping regret analysis (no data)")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5))
    
    # Panel A: Group-wise accuracy comparison
    ra = analysis['regret_analysis']
    groups = ['top', 'random', 'bottom']
    labels = ['Top-10\n(NASWOT)', 'Random-10', 'Bottom-10\n(NASWOT)']
    colors = [OI[2], OI[1], OI[5]]
    
    means = [ra[g]['mean'] for g in groups]
    stds = [ra[g]['std'] for g in groups]
    
    bars = ax1.bar(range(3), means, yerr=stds, capsize=4, color=colors,
                   edgecolor='black', linewidth=0.5, alpha=0.85,
                   error_kw={'linewidth': 1.0})
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_ylabel('3-Epoch Accuracy')
    ax1.set_title('(a) Top-k Selection Analysis', fontsize=9, fontweight='bold')
    
    # Add value labels
    for i, (m, s) in enumerate(zip(means, stds)):
        ax1.text(i, m + s + 0.003, f'{m:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    # Panel B: Proxy-accuracy correlation
    if 'proxy_vs_acc' in analysis:
        proxies = ['naswot', 'synflow', 'gradnorm', 'snip']
        proxy_labels = ['NASWOT', 'SynFlow', 'GradNorm', 'SNIP']
        rhos = [analysis['proxy_vs_acc'].get(p, {}).get('spearman', 0) for p in proxies]
        pvals = [analysis['proxy_vs_acc'].get(p, {}).get('sp_p', 1) for p in proxies]
        
        bar_colors = [OI[2] if pv < 0.05 else OI[1] for pv in pvals]
        bars2 = ax2.bar(range(4), rhos, color=bar_colors, edgecolor='black', 
                        linewidth=0.5, alpha=0.85)
        ax2.set_xticks(range(4))
        ax2.set_xticklabels(proxy_labels, fontsize=7, rotation=20, ha='right')
        ax2.set_ylabel("Spearman's rho")
        ax2.set_title('(b) Proxy vs 3-Epoch Accuracy', fontsize=9, fontweight='bold')
        ax2.axhline(0, color='gray', linewidth=0.5)
        
        # Mark significance
        for i, (rho, pv) in enumerate(zip(rhos, pvals)):
            sig = '*' if pv < 0.05 else 'ns'
            y = rho + 0.02 if rho >= 0 else rho - 0.05
            ax2.text(i, y, f'{rho:.2f}\n({sig})', ha='center', va='bottom' if rho >= 0 else 'top', fontsize=6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig_regret_analysis.pdf'))
    plt.savefig(os.path.join(fig_dir, 'fig_regret_analysis.png'), dpi=300)
    plt.close()
    print("  Saved fig_regret_analysis")


# ===================================================================
# NEW Figure 8: Cross-Resolution Consistency Scatter
# ===================================================================
def fig_cross_resolution():
    # Get architectures with cross-resolution scores
    cross = [r for r in results if 'scores_16x16_c100' in r and r['scores_16x16_c100'] is not None]
    if len(cross) < 10:
        print("  Skipping cross-resolution (insufficient data)")
        return
    
    fig, ax = plt.subplots(figsize=(3.5, 3))
    
    nw32 = [r['scores']['naswot'] for r in cross]
    nw16 = [r['scores_16x16_c100']['naswot'] for r in cross]
    
    ax.scatter(nw32, nw16, s=12, c=OI[4], alpha=0.5, edgecolors='black', linewidth=0.2)
    
    # Fit line
    valid = [(a, b) for a, b in zip(nw32, nw16) if a != 0 and b != 0 and np.isfinite(a) and np.isfinite(b)]
    if len(valid) > 5:
        x, y = zip(*valid)
        x, y = np.array(x), np.array(y)
        z = np.polyfit(x, y, 1)
        xline = np.linspace(min(x), max(x), 100)
        ax.plot(xline, np.polyval(z, xline), '--', color=OI[5], linewidth=1.5)
    
    # Annotation
    cr = analysis.get('cross_resolution', {})
    rho = cr.get('spearman', 0)
    n = cr.get('n', 0)
    ax.text(0.05, 0.95, f"Spearman rho={rho:.3f}\nN={n}", transform=ax.transAxes,
            fontsize=7, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('NASWOT (32x32, nc=10)')
    ax.set_ylabel('NASWOT (16x16, nc=100)')
    ax.set_title('Cross-Resolution Consistency', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig_cross_resolution.pdf'))
    plt.savefig(os.path.join(fig_dir, 'fig_cross_resolution.png'), dpi=300)
    plt.close()
    print("  Saved fig_cross_resolution")


# ===================================================================
# Figure 9: Multi-panel results overview (updated for N=500)
# ===================================================================
def fig_overview():
    fig = plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(3, 3, hspace=0.55, wspace=0.45)
    
    # Panel A: Mixer dominance
    ax_a = fig.add_subplot(gs[0, 0])
    data = analysis['mixer_naswot']
    order = ['conv', 'attention', 'gated_mlp', 'ssm_lite']
    labels = ['DWConv', 'Attn', 'gMLP', 'SSM']
    means = [data[k]['mean'] for k in order]
    sems = [data[k]['std']/np.sqrt(data[k]['n']) for k in order]
    ax_a.bar(range(4), means, yerr=sems, capsize=3, color=OI[:4],
             edgecolor='black', linewidth=0.5, alpha=0.85, error_kw={'linewidth': 0.8})
    ax_a.set_xticks(range(4))
    ax_a.set_xticklabels(labels, fontsize=6)
    ax_a.set_ylabel('NASWOT')
    ax_a.set_title('(a) Token Mixers', fontsize=8, fontweight='bold')
    
    # Panel B: Norm dominance
    ax_b = fig.add_subplot(gs[0, 1])
    data = analysis['norm_naswot']
    order = ['batch', 'group', 'layer', 'rms']
    labels = ['BN', 'GN', 'LN', 'RMS']
    means = [data[k]['mean'] for k in order]
    sems = [data[k]['std']/np.sqrt(data[k]['n']) for k in order]
    ax_b.bar(range(4), means, yerr=sems, capsize=3, color=OI[:4],
             edgecolor='black', linewidth=0.5, alpha=0.85, error_kw={'linewidth': 0.8})
    ax_b.set_xticks(range(4))
    ax_b.set_xticklabels(labels, fontsize=6)
    ax_b.set_title('(b) Normalizations', fontsize=8, fontweight='bold')
    
    # Panel C: Downsample dominance
    ax_c = fig.add_subplot(gs[0, 2])
    data = analysis['downsample_naswot']
    order = ['strided_conv', 'patch_merging', 'avgpool', 'maxpool']
    labels = ['SConv', 'PMerge', 'AvgP', 'MaxP']
    means = [data[k]['mean'] for k in order]
    sems = [data[k]['std']/np.sqrt(data[k]['n']) for k in order]
    ax_c.bar(range(4), means, yerr=sems, capsize=3, color=OI[:4],
             edgecolor='black', linewidth=0.5, alpha=0.85, error_kw={'linewidth': 0.8})
    ax_c.set_xticks(range(4))
    ax_c.set_xticklabels(labels, fontsize=6)
    ax_c.set_title('(c) Downsampling', fontsize=8, fontweight='bold')
    
    # Panel D: PCA
    ax_d = fig.add_subplot(gs[1, 0])
    pca = analysis['pca']
    eig = np.array(pca['eigenvalues'])
    cum = np.array(pca['cumulative'])
    n = min(len(eig), 15)
    ax_d.plot(range(1, n+1), cum[:n]*100, 'o-', color=OI[4], markersize=3, linewidth=1.2)
    ax_d.axhline(90, color='gray', linestyle='--', linewidth=0.6)
    ax_d.axhline(95, color='gray', linestyle=':', linewidth=0.6)
    ax_d.fill_between(range(1, n+1), 0, cum[:n]*100, alpha=0.1, color=OI[4])
    ax_d.set_xlabel('# Components')
    ax_d.set_ylabel('Cum. Variance (%)')
    ax_d.set_title(f'(d) Eff. Dim (90%={pca["eff_dim_90"]})', fontsize=8, fontweight='bold')
    
    # Panel E: Param scatter
    ax_e = fig.add_subplot(gs[1, 1])
    p_list = [r['params']/1e3 for r in results]
    nw_list = [r['scores']['naswot'] for r in results]
    ax_e.scatter(p_list, nw_list, s=8, c=OI[1], alpha=0.4, edgecolors='black', linewidth=0.2)
    ax_e.set_xlabel('Params (K)')
    ax_e.set_ylabel('NASWOT')
    ax_e.set_title(f'(e) Param Efficiency (N={N})', fontsize=8, fontweight='bold')
    
    # Panel F: Score distribution
    ax_f = fig.add_subplot(gs[1, 2])
    nw_scores = [r['scores']['naswot'] for r in results if r['scores']['naswot'] != 0]
    ax_f.hist(nw_scores, bins=20, color=OI[2], alpha=0.8, edgecolor='black', linewidth=0.5)
    if nw_scores:
        ax_f.axvline(np.mean(nw_scores), color=OI[5], linestyle='--', linewidth=1.2)
    ax_f.set_xlabel('NASWOT Score')
    ax_f.set_ylabel('Count')
    ax_f.set_title('(f) Score Distribution', fontsize=8, fontweight='bold')
    
    # Panel G: Regret analysis
    ax_g = fig.add_subplot(gs[2, 0])
    if 'regret_analysis' in analysis:
        ra = analysis['regret_analysis']
        groups = ['top', 'random', 'bottom']
        labels = ['Top-10', 'Random', 'Bot-10']
        colors = [OI[2], OI[1], OI[5]]
        means = [ra[g]['mean'] for g in groups]
        stds = [ra[g]['std'] for g in groups]
        ax_g.bar(range(3), means, yerr=stds, capsize=3, color=colors,
                 edgecolor='black', linewidth=0.5, alpha=0.85)
        ax_g.set_xticks(range(3))
        ax_g.set_xticklabels(labels, fontsize=6)
        ax_g.set_ylabel('3-ep Accuracy')
        ax_g.set_title('(g) Top-k Regret', fontsize=8, fontweight='bold')
    
    # Panel H: Cross-resolution
    ax_h = fig.add_subplot(gs[2, 1])
    cross = [r for r in results if 'scores_16x16_c100' in r]
    if cross:
        nw32 = [r['scores']['naswot'] for r in cross]
        nw16 = [r['scores_16x16_c100']['naswot'] for r in cross]
        ax_h.scatter(nw32, nw16, s=8, c=OI[4], alpha=0.4, edgecolors='black', linewidth=0.2)
        cr = analysis.get('cross_resolution', {})
        rho = cr.get('spearman', 0)
        ax_h.set_xlabel('NASWOT 32x32')
        ax_h.set_ylabel('NASWOT 16x16')
        ax_h.set_title(f'(h) Cross-Res (rho={rho:.2f})', fontsize=8, fontweight='bold')
    
    # Panel I: Correlation heatmap
    ax_i = fig.add_subplot(gs[2, 2])
    if 'score_corr' in analysis:
        corr = np.array(analysis['score_corr']['matrix'])
        display = ['NW', 'SF', 'GN', 'SN', 'P']
        im = ax_i.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        for i in range(5):
            for j in range(5):
                c = 'white' if abs(corr[i,j]) > 0.6 else 'black'
                ax_i.text(j, i, f'{corr[i,j]:.1f}', ha='center', va='center', fontsize=5, color=c)
        ax_i.set_xticks(range(5))
        ax_i.set_xticklabels(display, fontsize=6)
        ax_i.set_yticks(range(5))
        ax_i.set_yticklabels(display, fontsize=6)
        ax_i.set_title('(i) Proxy Correlations', fontsize=8, fontweight='bold')
    
    plt.savefig(os.path.join(fig_dir, 'fig_overview.pdf'))
    plt.savefig(os.path.join(fig_dir, 'fig_overview.png'), dpi=300)
    plt.close()
    print("  Saved fig_overview")


if __name__ == '__main__':
    print("Generating figures from v2 results...")
    fig_primitive_dominance()
    fig_score_correlation()
    fig_pca_scree()
    fig_param_efficiency()
    fig_score_distribution()
    fig_regret_analysis()
    fig_cross_resolution()
    fig_overview()
    print("Done! All figures saved to figures/")
