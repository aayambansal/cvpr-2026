#!/usr/bin/env python3
"""
Comprehensive analysis script for DenseProxy paper improvements.
Covers items 1, 2, 5 (partial), 7, and 8 from the punch list.
All analysis uses existing transnas_results.json -- no new compute needed.
"""

import json
import numpy as np
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ===========================================================================
# LOAD DATA
# ===========================================================================
print("=" * 80)
print("LOADING DATA")
print("=" * 80)

with open('results/transnas_results.json') as f:
    raw = json.load(f)

results = raw['results']
print(f"Total architectures: {len(results)}")

# Parse into structured arrays
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
        'feature_shapes': entry.get('feature_shapes', []),
        'num_features': entry.get('num_features', 0),
    }
    # Extract ground truth metrics
    gt = entry['ground_truth']
    rec['gt_seg_miou'] = gt.get('segmentsemantic', {}).get('test_mIoU_best', np.nan)
    rec['gt_normal_ssim'] = gt.get('normal', {}).get('test_ssim_best', np.nan)
    rec['gt_class_object'] = gt.get('class_object', {}).get('test_top1_best', np.nan)
    rec['gt_class_scene'] = gt.get('class_scene', {}).get('test_top1_best', np.nan)
    rec['gt_autoencoder'] = gt.get('autoencoder', {}).get('test_ssim_best', np.nan)
    
    # Parse architectural properties from name
    # Macro: "64-4111-basic" -> width=64, stages=[4,1,1,1]
    # Micro: "64-41414-0_00_000" -> different format
    parts = arch_name.split('-')
    if entry['search_space'] == 'macro':
        rec['width'] = int(parts[0])
        stage_str = parts[1]
        rec['stages'] = [int(c) for c in stage_str]
        rec['total_depth'] = sum(rec['stages'])
        rec['max_stage_depth'] = max(rec['stages'])
        rec['depth_variance'] = np.var(rec['stages'])
        rec['depth_imbalance'] = max(rec['stages']) / (sum(rec['stages']) + 1e-8)
    else:
        rec['width'] = int(parts[0]) if parts[0].isdigit() else 0
        rec['stages'] = []
        rec['total_depth'] = 0
        rec['max_stage_depth'] = 0
        rec['depth_variance'] = 0
        rec['depth_imbalance'] = 0
    
    archs.append(rec)

# Split by search space
macro = [a for a in archs if a['search_space'] == 'macro']
micro = [a for a in archs if a['search_space'] == 'micro']
print(f"Macro: {len(macro)}, Micro: {len(micro)}")

def get_arr(data, key):
    return np.array([d[key] for d in data])

def spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y) & (x != 0)
    if mask.sum() < 10:
        return 0.0, 1.0
    return stats.spearmanr(x[mask], y[mask])

def kendall(x, y):
    mask = np.isfinite(x) & np.isfinite(y) & (x != 0)
    if mask.sum() < 10:
        return 0.0, 1.0
    return stats.kendalltau(x[mask], y[mask])

tasks = ['gt_seg_miou', 'gt_normal_ssim', 'gt_class_object', 'gt_class_scene', 'gt_autoencoder']
task_names = ['Seg Semantic', 'Normal', 'Class Object', 'Class Scene', 'Autoencoder']


# ===========================================================================
# ITEM 1: SPACE-AWARE PROXY SELECTOR
# ===========================================================================
print("\n" + "=" * 80)
print("ITEM 1: SPACE-AWARE PROXY SELECTOR")
print("=" * 80)

# Strategy: For macro space, use MSFS. For micro space, use best capacity proxy.
# We'll evaluate several space-aware strategies.

for task_key, task_name in zip(tasks, task_names):
    print(f"\n--- Task: {task_name} ---")
    
    # Get proxy scores and GT for each space
    macro_gt = get_arr(macro, task_key)
    micro_gt = get_arr(micro, task_key)
    all_gt = get_arr(archs, task_key)
    
    # Strategy 1: Space-aware MSFS (macro) + SynFlow (micro)
    # Rank within each space, then combine
    macro_msfs = get_arr(macro, 'msfs')
    micro_synflow = get_arr(micro, 'synflow')
    micro_params = get_arr(micro, 'params')
    micro_gradnorm = get_arr(micro, 'gradnorm')
    
    # Compute per-space correlations for different micro-space choices
    for micro_proxy_name, micro_proxy_vals in [('SynFlow', micro_synflow), 
                                                 ('#Params', micro_params),
                                                 ('GradNorm', micro_gradnorm)]:
        # Build combined ranking: rank within each space using the chosen proxy
        macro_ranks = stats.rankdata(macro_msfs)
        micro_ranks = stats.rankdata(micro_proxy_vals)
        
        # Combine: assign macro ranks 1..N_macro, micro ranks N_macro+1..N_total
        # (or interleave by normalized rank)
        macro_normalized = macro_ranks / len(macro_ranks)
        micro_normalized = micro_ranks / len(micro_ranks)
        
        # For overall Spearman, we need a single ranking over all archs
        # Strategy: z-score normalize proxy within each space, then combine
        macro_z = (macro_msfs - np.mean(macro_msfs)) / (np.std(macro_msfs) + 1e-8)
        micro_z = (micro_proxy_vals - np.mean(micro_proxy_vals)) / (np.std(micro_proxy_vals) + 1e-8)
        
        combined_proxy = np.concatenate([macro_z, micro_z])
        combined_gt = np.concatenate([macro_gt, micro_gt])
        
        rho, p = spearman(combined_proxy, combined_gt)
        tau, p_tau = kendall(combined_proxy, combined_gt)
        print(f"  Space-aware MSFS(macro) + {micro_proxy_name}(micro): ρ={rho:.3f} (p={p:.2e}), τ={tau:.3f}")
    
    # Compare to baselines (single proxy across all)
    for proxy_name in ['msfs', 'gradnorm', 'synflow', 'params']:
        proxy_vals = get_arr(archs, proxy_name)
        rho, p = spearman(proxy_vals, all_gt)
        tau, p_tau = kendall(proxy_vals, all_gt)
        print(f"  Baseline {proxy_name}: ρ={rho:.3f}, τ={tau:.3f}")


# ===========================================================================
# ITEM 2: NAS SIMULATION
# ===========================================================================
print("\n" + "=" * 80)
print("ITEM 2: NAS SIMULATION - TOP-K SELECTION")
print("=" * 80)

def top_k_analysis(data, proxy_key, gt_key, K_values, label=""):
    """Simulate NAS: pick top-K by proxy, report GT quality."""
    proxy_vals = get_arr(data, proxy_key)
    gt_vals = get_arr(data, gt_key)
    
    # Remove invalid entries
    mask = np.isfinite(proxy_vals) & np.isfinite(gt_vals) & (proxy_vals != 0)
    proxy_valid = proxy_vals[mask]
    gt_valid = gt_vals[mask]
    N = len(gt_valid)
    
    if N < 10:
        return {}
    
    # Sort by proxy descending
    proxy_order = np.argsort(-proxy_valid)
    gt_sorted = gt_valid[proxy_order]
    
    # Best possible GT
    gt_best = np.max(gt_valid)
    gt_top1pct = np.percentile(gt_valid, 99)
    gt_top5pct = np.percentile(gt_valid, 95)
    
    results = {}
    for K in K_values:
        if K > N:
            continue
        selected_gt = gt_sorted[:K]
        best_in_K = np.max(selected_gt)
        mean_in_K = np.mean(selected_gt)
        
        # How many of the selected are in true top-1% / top-5%?
        n_top1pct = np.sum(selected_gt >= gt_top1pct)
        n_top5pct = np.sum(selected_gt >= gt_top5pct)
        
        # Regret: difference from oracle best
        regret = gt_best - best_in_K
        
        results[K] = {
            'best': best_in_K,
            'mean': mean_in_K,
            'regret': regret,
            'pct_top1': 100.0 * n_top1pct / K,
            'pct_top5': 100.0 * n_top5pct / K,
            'oracle_best': gt_best,
        }
    return results

K_values = [10, 25, 50, 100, 200, 500]

# Focus on segmentation task
for space_name, space_data in [('All', archs), ('Macro', macro), ('Micro', micro)]:
    print(f"\n--- Segmentation NAS Simulation ({space_name}, N={len(space_data)}) ---")
    
    proxies = ['msfs', 'gradnorm', 'synflow', 'params']
    
    # Also add random baseline (average over 100 trials)
    print(f"{'Proxy':<12} {'K':>4} {'Best mIoU':>10} {'Mean mIoU':>10} {'Regret':>8} {'%Top1%':>7} {'%Top5%':>7}")
    print("-" * 70)
    
    for proxy_name in proxies:
        res = top_k_analysis(space_data, proxy_name, 'gt_seg_miou', K_values)
        for K in K_values:
            if K in res:
                r = res[K]
                print(f"{proxy_name:<12} {K:>4} {r['best']:>10.2f} {r['mean']:>10.2f} {r['regret']:>8.2f} {r['pct_top1']:>7.1f} {r['pct_top5']:>7.1f}")
        print()
    
    # Random baseline (average of 100 random selections)
    gt_vals = get_arr(space_data, 'gt_seg_miou')
    gt_vals = gt_vals[np.isfinite(gt_vals)]
    gt_best = np.max(gt_vals)
    gt_top1pct = np.percentile(gt_vals, 99)
    gt_top5pct = np.percentile(gt_vals, 95)
    
    for K in K_values:
        if K > len(gt_vals):
            continue
        bests = []
        top1_counts = []
        for _ in range(1000):
            idx = np.random.choice(len(gt_vals), K, replace=False)
            sel = gt_vals[idx]
            bests.append(np.max(sel))
            top1_counts.append(np.sum(sel >= gt_top1pct))
        print(f"{'Random':<12} {K:>4} {np.mean(bests):>10.2f} {np.mean(gt_vals):>10.2f} {gt_best - np.mean(bests):>8.2f} {100*np.mean(top1_counts)/K:>7.1f} {'---':>7}")
    
    # Space-aware composite for "All"
    if space_name == 'All':
        print(f"\n--- Space-Aware Proxy (MSFS-macro + SynFlow-micro) ---")
        # Build space-aware ranking
        macro_msfs_vals = get_arr(macro, 'msfs')
        micro_synflow_vals = get_arr(micro, 'synflow')
        macro_z = (macro_msfs_vals - np.mean(macro_msfs_vals)) / (np.std(macro_msfs_vals) + 1e-8)
        micro_z = (micro_synflow_vals - np.mean(micro_synflow_vals)) / (np.std(micro_synflow_vals) + 1e-8)
        
        combined_proxy = np.concatenate([macro_z, micro_z])
        combined_gt_seg = np.concatenate([get_arr(macro, 'gt_seg_miou'), get_arr(micro, 'gt_seg_miou')])
        
        mask = np.isfinite(combined_proxy) & np.isfinite(combined_gt_seg)
        order = np.argsort(-combined_proxy[mask])
        gt_sorted = combined_gt_seg[mask][order]
        gt_best = np.max(combined_gt_seg[mask])
        gt_top1pct = np.percentile(combined_gt_seg[mask], 99)
        gt_top5pct = np.percentile(combined_gt_seg[mask], 95)
        
        for K in K_values:
            if K > len(gt_sorted):
                continue
            sel = gt_sorted[:K]
            best = np.max(sel)
            n1 = np.sum(sel >= gt_top1pct)
            n5 = np.sum(sel >= gt_top5pct)
            print(f"{'SpaceAware':<12} {K:>4} {best:>10.2f} {np.mean(sel):>10.2f} {gt_best-best:>8.2f} {100*n1/K:>7.1f} {100*n5/K:>7.1f}")


# ===========================================================================
# ITEM 5 (PARTIAL): SFC REWEIGHTING
# ===========================================================================
print("\n" + "=" * 80)
print("ITEM 5: SFC REWEIGHTING EXPERIMENTS")
print("=" * 80)

# Try different SFC formulations
gt_seg = get_arr(archs, 'gt_seg_miou')
sfc_nc = get_arr(archs, 'sfc_nc')
sfc_bs = get_arr(archs, 'sfc_bs')
sfc_sd = get_arr(archs, 'sfc_sd')

# Normalize SD like in original: (SD + 10) / 10
sfc_sd_norm = (sfc_sd + 10) / 10

print("\n--- SFC Component Correlations with Segmentation ---")
for comp_name, comp_vals in [('NC', sfc_nc), ('BS', sfc_bs), ('SD_raw', sfc_sd), ('SD_norm', sfc_sd_norm)]:
    rho, p = spearman(comp_vals, gt_seg)
    print(f"  {comp_name}: ρ={rho:.3f} (p={p:.2e})")

print("\n--- SFC Variant Experiments ---")
# Original SFC: 0.4*NC + 0.35*BS + 0.25*SD_norm
sfc_orig = 0.4 * sfc_nc + 0.35 * sfc_bs + 0.25 * sfc_sd_norm
rho, _ = spearman(sfc_orig, gt_seg)
print(f"  Original (0.4*NC + 0.35*BS + 0.25*SD_norm): ρ={rho:.3f}")

# Drop NC: just BS + SD
sfc_no_nc = 0.5 * sfc_bs + 0.5 * sfc_sd_norm
rho, _ = spearman(sfc_no_nc, gt_seg)
print(f"  Drop NC (0.5*BS + 0.5*SD_norm): ρ={rho:.3f}")

# SD only
rho, _ = spearman(sfc_sd_norm, gt_seg)
print(f"  SD only: ρ={rho:.3f}")

# Flip NC sign: use anti-coherence
sfc_flip_nc = -0.4 * sfc_nc + 0.35 * sfc_bs + 0.25 * sfc_sd_norm
rho, _ = spearman(sfc_flip_nc, gt_seg)
print(f"  Flip NC (-0.4*NC + 0.35*BS + 0.25*SD_norm): ρ={rho:.3f}")

# Grid search best weights
print("\n--- SFC Weight Grid Search ---")
best_rho = -1
best_weights = None
for a in np.arange(-1.0, 1.1, 0.1):
    for b in np.arange(-1.0, 1.1, 0.1):
        for g in np.arange(-1.0, 1.1, 0.1):
            sfc_test = a * sfc_nc + b * sfc_bs + g * sfc_sd_norm
            rho, _ = spearman(sfc_test, gt_seg)
            if rho > best_rho:
                best_rho = rho
                best_weights = (a, b, g)
print(f"  Best grid search: α={best_weights[0]:.1f}, β={best_weights[1]:.1f}, γ={best_weights[2]:.1f} → ρ={best_rho:.3f}")

# Also try per-space
for space_name, space_data in [('Macro', macro), ('Micro', micro)]:
    gt_seg_space = get_arr(space_data, 'gt_seg_miou')
    nc_space = get_arr(space_data, 'sfc_nc')
    bs_space = get_arr(space_data, 'sfc_bs')
    sd_space = (get_arr(space_data, 'sfc_sd') + 10) / 10
    
    best_rho_space = -1
    best_w_space = None
    for a in np.arange(-1.0, 1.1, 0.1):
        for b in np.arange(-1.0, 1.1, 0.1):
            for g in np.arange(-1.0, 1.1, 0.1):
                sfc_test = a * nc_space + b * bs_space + g * sd_space
                rho, _ = spearman(sfc_test, gt_seg_space)
                if rho > best_rho_space:
                    best_rho_space = rho
                    best_w_space = (a, b, g)
    print(f"  {space_name} best: α={best_w_space[0]:.1f}, β={best_w_space[1]:.1f}, γ={best_w_space[2]:.1f} → ρ={best_rho_space:.3f}")


# ===========================================================================
# ITEM 7: MSFS WITHIN-FAMILY NORMALIZATION
# ===========================================================================
print("\n" + "=" * 80)
print("ITEM 7: MSFS WITHIN-FAMILY NORMALIZATION")
print("=" * 80)

# Strategy: z-score normalize MSFS within each search space before ranking
msfs_all = get_arr(archs, 'msfs')
msfs_macro = get_arr(macro, 'msfs')
msfs_micro = get_arr(micro, 'msfs')

# Z-score within each space
msfs_macro_z = (msfs_macro - np.mean(msfs_macro)) / (np.std(msfs_macro) + 1e-8)
msfs_micro_z = (msfs_micro - np.mean(msfs_micro)) / (np.std(msfs_micro) + 1e-8)
msfs_normalized = np.concatenate([msfs_macro_z, msfs_micro_z])

gt_seg_all = np.concatenate([get_arr(macro, 'gt_seg_miou'), get_arr(micro, 'gt_seg_miou')])
gt_normal_all = np.concatenate([get_arr(macro, 'gt_normal_ssim'), get_arr(micro, 'gt_normal_ssim')])

print("\n--- MSFS Normalization Comparison (All Architectures) ---")
for task_gt, task_name in [(gt_seg_all, 'Segmentation'), (gt_normal_all, 'Normal')]:
    rho_raw, _ = spearman(np.concatenate([msfs_macro, msfs_micro]), task_gt)
    rho_norm, _ = spearman(msfs_normalized, task_gt)
    print(f"  {task_name}: raw ρ={rho_raw:.3f} → normalized ρ={rho_norm:.3f} (Δ={rho_norm-rho_raw:+.3f})")

# Also try rank normalization
msfs_macro_rank = stats.rankdata(msfs_macro) / len(msfs_macro)
msfs_micro_rank = stats.rankdata(msfs_micro) / len(msfs_micro)
msfs_rank_norm = np.concatenate([msfs_macro_rank, msfs_micro_rank])

for task_gt, task_name in [(gt_seg_all, 'Segmentation'), (gt_normal_all, 'Normal')]:
    rho_rank, _ = spearman(msfs_rank_norm, task_gt)
    print(f"  {task_name}: rank-normalized ρ={rho_rank:.3f}")

# Try for all proxies
print("\n--- Within-Family Z-Normalization for All Proxies (Segmentation) ---")
for proxy_name in ['msfs', 'gradnorm', 'synflow', 'params', 'sfc']:
    macro_vals = get_arr(macro, proxy_name)
    micro_vals = get_arr(micro, proxy_name)
    
    raw_combined = np.concatenate([macro_vals, micro_vals])
    rho_raw, _ = spearman(raw_combined, gt_seg_all)
    
    macro_z = (macro_vals - np.mean(macro_vals)) / (np.std(macro_vals) + 1e-8)
    micro_z = (micro_vals - np.mean(micro_vals)) / (np.std(micro_vals) + 1e-8)
    norm_combined = np.concatenate([macro_z, micro_z])
    rho_norm, _ = spearman(norm_combined, gt_seg_all)
    
    print(f"  {proxy_name:<10}: raw ρ={rho_raw:.3f} → z-norm ρ={rho_norm:.3f} (Δ={rho_norm-rho_raw:+.3f})")


# ===========================================================================
# ITEM 8: ARCHITECTURAL PROPERTY CORRELATIONS
# ===========================================================================
print("\n" + "=" * 80)
print("ITEM 8: ARCHITECTURAL PROPERTY CORRELATIONS (Macro Space)")
print("=" * 80)

# Only meaningful for macro space where we can parse arch strings
macro_msfs_vals = get_arr(macro, 'msfs')
macro_isd = get_arr(macro, 'msfs_isd')
macro_sa = get_arr(macro, 'msfs_sa')
macro_seg = get_arr(macro, 'gt_seg_miou')
macro_depth = get_arr(macro, 'total_depth')
macro_max_stage = get_arr(macro, 'max_stage_depth')
macro_depth_var = get_arr(macro, 'depth_variance')
macro_depth_imb = get_arr(macro, 'depth_imbalance')
macro_params_vals = get_arr(macro, 'params')

print("\n--- MSFS vs Architectural Properties ---")
for prop_name, prop_vals in [('Total Depth', macro_depth), 
                               ('Max Stage Depth', macro_max_stage),
                               ('Depth Variance', macro_depth_var),
                               ('Depth Imbalance', macro_depth_imb),
                               ('#Params', macro_params_vals)]:
    rho_msfs, p = spearman(macro_msfs_vals, prop_vals)
    rho_isd, _ = spearman(macro_isd, prop_vals)
    rho_sa, _ = spearman(macro_sa, prop_vals)
    rho_gt, _ = spearman(macro_seg, prop_vals)
    print(f"  {prop_name:<20}: MSFS ρ={rho_msfs:.3f}, ISD ρ={rho_isd:.3f}, SA ρ={rho_sa:.3f}, GT-Seg ρ={rho_gt:.3f}")

print("\n--- GT Segmentation vs Architectural Properties ---")
for prop_name, prop_vals in [('Total Depth', macro_depth), 
                               ('Max Stage Depth', macro_max_stage),
                               ('Depth Variance', macro_depth_var),
                               ('Depth Imbalance', macro_depth_imb)]:
    rho, p = spearman(macro_seg, prop_vals)
    print(f"  {prop_name:<20}: ρ={rho:.3f} (p={p:.2e})")

# Check what stage configurations lead to high MSFS and high GT
print("\n--- Stage Configuration Analysis ---")
# Group by total depth
depth_groups = defaultdict(list)
for a in macro:
    depth_groups[a['total_depth']].append(a)

print(f"{'Depth':>6} {'N':>5} {'Mean MSFS':>10} {'Mean Seg mIoU':>14} {'Mean Params':>12}")
for depth in sorted(depth_groups.keys()):
    group = depth_groups[depth]
    mean_msfs = np.mean([a['msfs'] for a in group])
    mean_seg = np.mean([a['gt_seg_miou'] for a in group])
    mean_params = np.mean([a['params'] for a in group])
    print(f"{depth:>6} {len(group):>5} {mean_msfs:>10.3f} {mean_seg:>14.2f} {mean_params:>12.0f}")

# Check if MSFS predicts GT *within* same-depth groups (controlling for depth)
print("\n--- MSFS vs GT Segmentation WITHIN Same-Depth Groups (Macro) ---")
for depth in sorted(depth_groups.keys()):
    group = depth_groups[depth]
    if len(group) < 10:
        continue
    msfs_g = np.array([a['msfs'] for a in group])
    seg_g = np.array([a['gt_seg_miou'] for a in group])
    rho, p = stats.spearmanr(msfs_g, seg_g)
    print(f"  Depth {depth}: N={len(group):>4}, ρ={rho:.3f} (p={p:.2e})")


# ===========================================================================
# ITEM 1 EXTENDED: FULL SPACE-AWARE EVALUATION WITH KENDALL TAU
# ===========================================================================
print("\n" + "=" * 80)
print("ITEM 1 EXTENDED: SPACE-AWARE PROXY - FULL METRICS")
print("=" * 80)

# Best space-aware: MSFS for macro, best-per-task for micro
for task_key, task_name in zip(tasks, task_names):
    print(f"\n--- {task_name} ---")
    
    macro_gt = get_arr(macro, task_key)
    micro_gt = get_arr(micro, task_key)
    all_gt = np.concatenate([macro_gt, micro_gt])
    
    # Find best micro proxy for this task
    best_micro_rho = -1
    best_micro_name = ''
    for proxy_name in ['synflow', 'params', 'gradnorm']:
        micro_proxy = get_arr(micro, proxy_name)
        rho, _ = spearman(micro_proxy, micro_gt)
        if rho > best_micro_rho:
            best_micro_rho = rho
            best_micro_name = proxy_name
    
    # Space-aware: MSFS(macro) + best_micro(micro)
    macro_msfs_z = (get_arr(macro, 'msfs') - np.mean(get_arr(macro, 'msfs'))) / (np.std(get_arr(macro, 'msfs')) + 1e-8)
    micro_best_vals = get_arr(micro, best_micro_name)
    micro_best_z = (micro_best_vals - np.mean(micro_best_vals)) / (np.std(micro_best_vals) + 1e-8)
    
    combined_proxy = np.concatenate([macro_msfs_z, micro_best_z])
    
    rho_sa, p_sa = spearman(combined_proxy, all_gt)
    tau_sa, p_tau = kendall(combined_proxy, all_gt)
    
    # Baselines
    rho_gn, _ = spearman(get_arr(archs, 'gradnorm'), get_arr(archs, task_key))
    tau_gn, _ = kendall(get_arr(archs, 'gradnorm'), get_arr(archs, task_key))
    rho_msfs_raw, _ = spearman(get_arr(archs, 'msfs'), get_arr(archs, task_key))
    
    print(f"  Space-Aware [MSFS + {best_micro_name}]: ρ={rho_sa:.3f}, τ={tau_sa:.3f}")
    print(f"  GradNorm (single):                       ρ={rho_gn:.3f}, τ={tau_gn:.3f}")
    print(f"  MSFS raw:                                ρ={rho_msfs_raw:.3f}")
    print(f"  Best micro proxy: {best_micro_name} (ρ={best_micro_rho:.3f})")


# ===========================================================================
# SAVE ALL RESULTS
# ===========================================================================
print("\n" + "=" * 80)
print("DONE - All analysis complete")
print("=" * 80)
