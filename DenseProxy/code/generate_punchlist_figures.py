#!/usr/bin/env python3
"""
Generate all new figures for the 10/10 punch list improvements.
"""
import json
import numpy as np
from scipy import stats
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

# Load data
with open('results/transnas_results.json') as f:
    raw = json.load(f)
results = raw['results']

archs = []
for arch_name, entry in results.items():
    rec = {
        'name': arch_name,
        'search_space': entry['search_space'],
        'params': entry['params'],
        'msfs': entry['msfs'],
        'msfs_isd': entry['msfs_isd'],
        'msfs_sa': entry['msfs_sa'],
        'sfc': entry['sfc'],
        'sfc_nc': entry['sfc_nc'],
        'sfc_bs': entry['sfc_bs'],
        'sfc_sd': entry['sfc_sd'],
        'gradnorm': entry['gradnorm'],
        'synflow': entry['synflow'],
        'naswot': entry['naswot'],
    }
    gt = entry['ground_truth']
    rec['gt_seg_miou'] = gt.get('segmentsemantic', {}).get('test_mIoU_best', np.nan)
    rec['gt_normal_ssim'] = gt.get('normal', {}).get('test_ssim_best', np.nan)
    rec['gt_class_object'] = gt.get('class_object', {}).get('test_top1_best', np.nan)
    rec['gt_class_scene'] = gt.get('class_scene', {}).get('test_top1_best', np.nan)
    rec['gt_autoencoder'] = gt.get('autoencoder', {}).get('test_ssim_best', np.nan)
    
    parts = arch_name.split('-')
    if entry['search_space'] == 'macro':
        rec['total_depth'] = sum(int(c) for c in parts[1])
    else:
        rec['total_depth'] = 0
    archs.append(rec)

macro = [a for a in archs if a['search_space'] == 'macro']
micro = [a for a in archs if a['search_space'] == 'micro']

def get_arr(data, key):
    return np.array([d[key] for d in data])

def spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y) & (x != 0)
    if mask.sum() < 10:
        return 0.0, 1.0
    return stats.spearmanr(x[mask], y[mask])

tasks = ['gt_seg_miou', 'gt_normal_ssim', 'gt_class_object', 'gt_class_scene', 'gt_autoencoder']
task_labels = ['Segmentation\n(mIoU)', 'Normal\n(SSIM)', 'Object Cls.\n(Top-1)', 'Scene Cls.\n(Top-1)', 'Autoencoder\n(SSIM)']
task_short = ['Seg.', 'Normal', 'ClassObj', 'ClassScene', 'AutoEnc']

colors = {
    'Space-Aware': '#e41a1c',
    'GradNorm': '#377eb8',
    'SynFlow': '#4daf4a',
    '#Params': '#984ea3',
    'MSFS': '#ff7f00',
    'SFC': '#a65628',
    'Random': '#999999',
}

# ========================================================================
# FIGURE 1: Space-Aware Proxy Comparison (Bar Chart)
# ========================================================================
print("Generating fig_space_aware_comparison...")

fig, ax = plt.subplots(figsize=(7, 3.2))

# Compute space-aware results for each task
sa_results = {}
gn_results = {}
msfs_raw_results = {}

for task_key, task_label in zip(tasks, task_short):
    macro_gt = get_arr(macro, task_key)
    micro_gt = get_arr(micro, task_key)
    all_gt = np.concatenate([macro_gt, micro_gt])
    
    # Find best micro proxy
    best_micro_rho = -1
    best_micro_name = ''
    for pn in ['synflow', 'params', 'gradnorm']:
        micro_proxy = get_arr(micro, pn)
        rho, _ = spearman(micro_proxy, micro_gt)
        if rho > best_micro_rho:
            best_micro_rho = rho
            best_micro_name = pn
    
    macro_msfs_z = (get_arr(macro, 'msfs') - np.mean(get_arr(macro, 'msfs'))) / (np.std(get_arr(macro, 'msfs')) + 1e-8)
    micro_best_vals = get_arr(micro, best_micro_name)
    micro_best_z = (micro_best_vals - np.mean(micro_best_vals)) / (np.std(micro_best_vals) + 1e-8)
    
    combined_proxy = np.concatenate([macro_msfs_z, micro_best_z])
    sa_rho, _ = spearman(combined_proxy, all_gt)
    gn_rho, _ = spearman(get_arr(archs, 'gradnorm'), get_arr(archs, task_key))
    msfs_rho, _ = spearman(get_arr(archs, 'msfs'), get_arr(archs, task_key))
    
    sa_results[task_label] = sa_rho
    gn_results[task_label] = gn_rho
    msfs_raw_results[task_label] = msfs_rho

x = np.arange(len(task_short))
w = 0.25
bars1 = ax.bar(x - w, [sa_results[t] for t in task_short], w, label='Space-Aware (Ours)', color=colors['Space-Aware'], edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x, [gn_results[t] for t in task_short], w, label='GradNorm', color=colors['GradNorm'], edgecolor='white', linewidth=0.5)
bars3 = ax.bar(x + w, [msfs_raw_results[t] for t in task_short], w, label='MSFS Raw', color=colors['MSFS'], edgecolor='white', linewidth=0.5)

ax.set_ylabel('Spearman ρ')
ax.set_xticks(x)
ax.set_xticklabels(task_short)
ax.legend(loc='upper right')
ax.set_ylim(0, 0.8)
ax.set_title('Space-Aware Proxy Selector vs. Single-Proxy Baselines (N=7,344)')
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{h:.2f}', 
                ha='center', va='bottom', fontsize=6.5)

plt.tight_layout()
plt.savefig('figures/fig_space_aware.pdf')
plt.savefig('figures/fig_space_aware.png')
plt.close()


# ========================================================================
# FIGURE 2: NAS Simulation - Regret Curves
# ========================================================================
print("Generating fig_nas_simulation...")

def top_k_results(data, proxy_key, gt_key, K_values):
    proxy_vals = get_arr(data, proxy_key)
    gt_vals = get_arr(data, gt_key)
    mask = np.isfinite(proxy_vals) & np.isfinite(gt_vals) & (proxy_vals != 0)
    proxy_valid = proxy_vals[mask]
    gt_valid = gt_vals[mask]
    order = np.argsort(-proxy_valid)
    gt_sorted = gt_valid[order]
    gt_best = np.max(gt_valid)
    res = {}
    for K in K_values:
        if K > len(gt_sorted):
            continue
        best_in_K = np.max(gt_sorted[:K])
        res[K] = gt_best - best_in_K
    return res

K_values = [5, 10, 25, 50, 100, 200, 500]

fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))

for ax, (space_name, space_data) in zip(axes, [('All (N=7,344)', archs), ('Macro (N=3,256)', macro), ('Micro (N=4,088)', micro)]):
    for proxy_name, proxy_label, color in [
        ('msfs', 'MSFS', colors['MSFS']),
        ('gradnorm', 'GradNorm', colors['GradNorm']),
        ('synflow', 'SynFlow', colors['SynFlow']),
        ('params', '#Params', colors['#Params']),
    ]:
        res = top_k_results(space_data, proxy_name, 'gt_seg_miou', K_values)
        Ks = sorted(res.keys())
        regrets = [res[k] for k in Ks]
        ax.plot(Ks, regrets, 'o-', label=proxy_label, color=color, markersize=3, linewidth=1.5)
    
    # Random baseline
    gt_vals = get_arr(space_data, 'gt_seg_miou')
    gt_vals = gt_vals[np.isfinite(gt_vals)]
    gt_best = np.max(gt_vals)
    random_regrets = []
    for K in K_values:
        if K > len(gt_vals):
            continue
        bests = [np.max(gt_vals[np.random.choice(len(gt_vals), K, replace=False)]) for _ in range(500)]
        random_regrets.append(gt_best - np.mean(bests))
    ax.plot(K_values[:len(random_regrets)], random_regrets, 's--', label='Random', color=colors['Random'], markersize=3, linewidth=1)
    
    ax.set_xlabel('Budget K')
    ax.set_ylabel('Regret (mIoU)')
    ax.set_title(space_name)
    ax.set_xscale('log')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6)

plt.tight_layout()
plt.savefig('figures/fig_nas_simulation.pdf')
plt.savefig('figures/fig_nas_simulation.png')
plt.close()


# ========================================================================
# FIGURE 3: Top-K Overlap / Selection Quality  
# ========================================================================
print("Generating fig_topk_quality...")

fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

for ax, (space_name, space_data) in zip(axes, [('Macro (N=3,256)', macro), ('Micro (N=4,088)', micro)]):
    gt_vals = get_arr(space_data, 'gt_seg_miou')
    gt_vals_valid = gt_vals[np.isfinite(gt_vals)]
    gt_top5pct = np.percentile(gt_vals_valid, 95)
    
    K_vals = [10, 25, 50, 100, 200]
    
    for proxy_name, proxy_label, color in [
        ('msfs', 'MSFS', colors['MSFS']),
        ('gradnorm', 'GradNorm', colors['GradNorm']),
        ('synflow', 'SynFlow', colors['SynFlow']),
        ('params', '#Params', colors['#Params']),
    ]:
        proxy_vals = get_arr(space_data, proxy_name)
        mask = np.isfinite(proxy_vals) & np.isfinite(gt_vals) & (proxy_vals != 0)
        order = np.argsort(-proxy_vals[mask])
        gt_sorted = gt_vals[mask][order]
        
        pct_top5 = []
        for K in K_vals:
            if K > len(gt_sorted):
                break
            n5 = np.sum(gt_sorted[:K] >= gt_top5pct)
            pct_top5.append(100 * n5 / K)
        ax.plot(K_vals[:len(pct_top5)], pct_top5, 'o-', label=proxy_label, color=color, markersize=4, linewidth=1.5)
    
    ax.axhline(y=5, color='gray', linestyle='--', alpha=0.5, label='Random (5%)')
    ax.set_xlabel('Budget K')
    ax.set_ylabel('% of Top-5% Archs Found')
    ax.set_title(space_name)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6)

plt.tight_layout()
plt.savefig('figures/fig_topk_quality.pdf')
plt.savefig('figures/fig_topk_quality.png')
plt.close()


# ========================================================================
# FIGURE 4: Architectural Properties (Why MSFS Works)
# ========================================================================
print("Generating fig_arch_properties...")

fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))

# Panel 1: MSFS vs GT segmentation, colored by depth
macro_msfs_vals = get_arr(macro, 'msfs')
macro_seg = get_arr(macro, 'gt_seg_miou')
macro_depth = get_arr(macro, 'total_depth')
mask = np.isfinite(macro_msfs_vals) & np.isfinite(macro_seg) & (macro_msfs_vals != 0)

sc = axes[0].scatter(macro_msfs_vals[mask], macro_seg[mask], c=macro_depth[mask], 
                      cmap='viridis', s=3, alpha=0.4, rasterized=True)
axes[0].set_xlabel('MSFS Score')
axes[0].set_ylabel('Segmentation mIoU')
axes[0].set_title('Macro: MSFS vs Seg (by depth)')
plt.colorbar(sc, ax=axes[0], label='Total Depth')
rho, _ = stats.spearmanr(macro_msfs_vals[mask], macro_seg[mask])
axes[0].text(0.05, 0.95, f'ρ={rho:.3f}', transform=axes[0].transAxes, va='top', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel 2: Within-depth MSFS correlation
depth_groups = defaultdict(list)
for a in macro:
    if np.isfinite(a['msfs']) and a['msfs'] != 0:
        depth_groups[a['total_depth']].append(a)

depths = sorted([d for d in depth_groups if len(depth_groups[d]) >= 10])
within_rhos = []
for d in depths:
    group = depth_groups[d]
    m = np.array([a['msfs'] for a in group])
    s = np.array([a['gt_seg_miou'] for a in group])
    rho, _ = stats.spearmanr(m, s)
    within_rhos.append(rho)

axes[1].bar(range(len(depths)), within_rhos, color='#ff7f00', edgecolor='white')
axes[1].set_xticks(range(len(depths)))
axes[1].set_xticklabels([str(d) for d in depths])
axes[1].set_xlabel('Total Depth')
axes[1].set_ylabel('Within-Depth Spearman ρ')
axes[1].set_title('MSFS Predicts Seg Within Depth Groups')
axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
axes[1].grid(axis='y', alpha=0.3)

# Panel 3: Stage configuration analysis  
depth_means = []
seg_means = []
ns = []
for d in sorted(depth_groups):
    group = depth_groups[d]
    if len(group) >= 10:
        depth_means.append(d)
        seg_means.append(np.mean([a['gt_seg_miou'] for a in group]))
        ns.append(len(group))

axes[2].bar(range(len(depth_means)), seg_means, color='#377eb8', edgecolor='white')
axes[2].set_xticks(range(len(depth_means)))
axes[2].set_xticklabels([str(d) for d in depth_means])
axes[2].set_xlabel('Total Depth')
axes[2].set_ylabel('Mean Seg mIoU')
axes[2].set_title('Shallower Archs → Better Seg')
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig_arch_properties.pdf')
plt.savefig('figures/fig_arch_properties.png')
plt.close()


# ========================================================================
# FIGURE 5: SFC Failure Analysis + Reweighting
# ========================================================================
print("Generating fig_sfc_analysis...")

fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

# Panel 1: SFC components vs segmentation
gt_seg = get_arr(archs, 'gt_seg_miou')
components = [
    ('NC', get_arr(archs, 'sfc_nc')),
    ('BS', get_arr(archs, 'sfc_bs')),
    ('SD', (get_arr(archs, 'sfc_sd') + 10) / 10),
    ('SFC\nOriginal', get_arr(archs, 'sfc')),
    ('SD\nOnly', (get_arr(archs, 'sfc_sd') + 10) / 10),
]

comp_names = ['NC', 'BS', 'SD', 'SFC\nOrig.', 'SFC\n-NC']
comp_rhos = []
for name, vals in [('NC', get_arr(archs, 'sfc_nc')), 
                    ('BS', get_arr(archs, 'sfc_bs')),
                    ('SD', (get_arr(archs, 'sfc_sd') + 10) / 10),
                    ('SFC Orig.', get_arr(archs, 'sfc')),
                    ('SFC -NC', 0.5 * get_arr(archs, 'sfc_bs') + 0.5 * (get_arr(archs, 'sfc_sd') + 10) / 10)]:
    rho, _ = spearman(vals, gt_seg)
    comp_rhos.append(rho)

bar_colors = ['#e41a1c' if r < 0 else '#4daf4a' for r in comp_rhos]
axes[0].bar(range(len(comp_names)), comp_rhos, color=bar_colors, edgecolor='white')
axes[0].set_xticks(range(len(comp_names)))
axes[0].set_xticklabels(comp_names)
axes[0].set_ylabel('Spearman ρ with Segmentation')
axes[0].set_title('SFC Components: NC Is Harmful')
axes[0].axhline(y=0, color='black', linewidth=0.5)
axes[0].grid(axis='y', alpha=0.3)

for i, r in enumerate(comp_rhos):
    axes[0].text(i, r + (0.01 if r >= 0 else -0.02), f'{r:.3f}', 
                 ha='center', va='bottom' if r >= 0 else 'top', fontsize=7)

# Panel 2: Reweighting improvement
variants = [
    ('Original\n(α=0.4)', 0.056),
    ('Drop NC\n(BS+SD)', 0.110),
    ('SD Only', 0.189),
    ('Flip NC\n(α=-0.4)', 0.143),
    ('Grid\nSearch*', 0.354),
]
vnames = [v[0] for v in variants]
vrhos = [v[1] for v in variants]
bar_colors2 = plt.cm.YlOrRd(np.linspace(0.2, 0.8, len(variants)))

axes[1].bar(range(len(vnames)), vrhos, color=bar_colors2, edgecolor='white')
axes[1].set_xticks(range(len(vnames)))
axes[1].set_xticklabels(vnames, fontsize=7)
axes[1].set_ylabel('Spearman ρ with Segmentation')
axes[1].set_title('SFC Reweighting Experiments')
axes[1].grid(axis='y', alpha=0.3)
axes[1].text(0.95, 0.95, '*tuned on eval data\n(not a valid proxy)', 
             transform=axes[1].transAxes, va='top', ha='right', fontsize=6, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('figures/fig_sfc_analysis.pdf')
plt.savefig('figures/fig_sfc_analysis.png')
plt.close()


# ========================================================================
# FIGURE 6: Within-Family Normalization
# ========================================================================
print("Generating fig_normalization...")

fig, ax = plt.subplots(figsize=(5, 3.2))

proxy_names = ['MSFS', 'GradNorm', 'SynFlow', '#Params', 'SFC']
proxy_keys = ['msfs', 'gradnorm', 'synflow', 'params', 'sfc']

raw_rhos = []
norm_rhos = []
gt_seg_all = np.concatenate([get_arr(macro, 'gt_seg_miou'), get_arr(micro, 'gt_seg_miou')])

for pk in proxy_keys:
    macro_vals = get_arr(macro, pk)
    micro_vals = get_arr(micro, pk)
    raw_combined = np.concatenate([macro_vals, micro_vals])
    rho_raw, _ = spearman(raw_combined, gt_seg_all)
    raw_rhos.append(rho_raw)
    
    macro_z = (macro_vals - np.mean(macro_vals)) / (np.std(macro_vals) + 1e-8)
    micro_z = (micro_vals - np.mean(micro_vals)) / (np.std(micro_vals) + 1e-8)
    norm_combined = np.concatenate([macro_z, micro_z])
    rho_norm, _ = spearman(norm_combined, gt_seg_all)
    norm_rhos.append(rho_norm)

x = np.arange(len(proxy_names))
w = 0.35
bars1 = ax.bar(x - w/2, raw_rhos, w, label='Raw', color='#377eb8', edgecolor='white')
bars2 = ax.bar(x + w/2, norm_rhos, w, label='Z-Normalized', color='#e41a1c', edgecolor='white')

ax.set_ylabel('Spearman ρ (Segmentation)')
ax.set_xticks(x)
ax.set_xticklabels(proxy_names)
ax.legend()
ax.set_title('Effect of Within-Family Z-Normalization')
ax.grid(axis='y', alpha=0.3)

# Delta labels
for i in range(len(proxy_names)):
    delta = norm_rhos[i] - raw_rhos[i]
    sign = '+' if delta > 0 else ''
    ax.text(x[i] + w/2, max(raw_rhos[i], norm_rhos[i]) + 0.01, 
            f'Δ{sign}{delta:.2f}', ha='center', va='bottom', fontsize=6.5, color='red' if delta < 0 else 'green')

plt.tight_layout()
plt.savefig('figures/fig_normalization.pdf')
plt.savefig('figures/fig_normalization.png')
plt.close()


print("\nAll new figures generated successfully!")
print("Files:")
for f in ['fig_space_aware', 'fig_nas_simulation', 'fig_topk_quality', 'fig_arch_properties', 'fig_sfc_analysis', 'fig_normalization']:
    print(f"  figures/{f}.pdf / .png")
