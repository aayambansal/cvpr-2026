"""
Analyze new baseline results (TE-NAS, ZenNAS, NASWOT-fixed) and robustness sweep.
Generates figures and prints LaTeX-ready tables.

Usage: python3 code/analyze_extended.py
"""
import json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# Ground truth task keys
TASK_KEYS = {
    'segmentsemantic': ('test_mIoU_best', 'Segmentation'),
    'normal': ('test_ssim_best', 'Normal'),
    'class_object': ('test_top1_best', 'Cls. Object'),
    'class_scene': ('test_top1_best', 'Cls. Scene'),
    'autoencoder': ('test_ssim_best', 'Autoencoder'),
}


def load_original_results():
    """Load the original 7,344-arch results for ground truth."""
    with open(RESULTS_DIR / "transnas_results.json") as f:
        data = json.load(f)
    return data['results']


def analyze_new_baselines():
    """Compute Spearman rho for TE-NAS, ZenNAS, NASWOT-fixed."""
    path = RESULTS_DIR / "transnas_new_baselines.json"
    if not path.exists():
        print("transnas_new_baselines.json not found yet")
        return None

    with open(path) as f:
        new_data = json.load(f)
    
    print(f"\n{'='*60}")
    print("NEW BASELINES ANALYSIS")
    print(f"{'='*60}")
    print(f"Total evaluated: {new_data['meta']['successful']} / {new_data['meta']['total']}")
    print(f"Errors: {new_data['meta']['errors']}")
    print(f"Elapsed: {new_data['meta']['elapsed_seconds']:.0f}s")
    
    orig = load_original_results()
    new_results = new_data['results']
    
    # Build aligned arrays
    proxies = ['tenas', 'zennas', 'naswot_fixed']
    proxy_labels = ['TE-NAS', 'ZenNAS', 'NASWOT-fixed']
    
    results_table = {}
    
    for task_key, (metric_key, task_label) in TASK_KEYS.items():
        results_table[task_label] = {}
        
        for proxy_name, proxy_label in zip(proxies, proxy_labels):
            scores = []
            gt_vals = []
            
            for arch_name in new_results:
                if arch_name not in orig:
                    continue
                gt = orig[arch_name].get('ground_truth', {}).get(task_key, {})
                gt_val = gt.get(metric_key)
                if gt_val is None:
                    continue
                
                proxy_score = new_results[arch_name].get(proxy_name)
                if proxy_score is None or proxy_score == 0.0:
                    continue
                
                scores.append(proxy_score)
                gt_vals.append(gt_val)
            
            if len(scores) > 10:
                rho, p = stats.spearmanr(scores, gt_vals)
                results_table[task_label][proxy_label] = (rho, p, len(scores))
            else:
                results_table[task_label][proxy_label] = (0.0, 1.0, len(scores))
    
    # Print results
    print(f"\n{'Proxy':<15} ", end="")
    for task_label in TASK_KEYS.values():
        print(f"{task_label[1]:<14}", end="")
    print()
    print("-" * 85)
    
    for proxy_label in proxy_labels:
        print(f"{proxy_label:<15} ", end="")
        for task_key, (_, task_label) in TASK_KEYS.items():
            rho, p, n = results_table[task_label].get(proxy_label, (0, 1, 0))
            print(f"{rho:>6.3f} (n={n:<4})", end=" ")
        print()
    
    # Also compute per-space results
    print(f"\n--- Per-space breakdown ---")
    for space_name in ['macro', 'micro']:
        print(f"\n{space_name.upper()} space:")
        print(f"{'Proxy':<15} ", end="")
        for task_label in TASK_KEYS.values():
            print(f"{task_label[1]:<14}", end="")
        print()
        
        for proxy_name, proxy_label in zip(proxies, proxy_labels):
            print(f"{proxy_label:<15} ", end="")
            for task_key, (metric_key, task_label) in TASK_KEYS.items():
                scores = []
                gt_vals = []
                
                for arch_name in new_results:
                    if arch_name not in orig:
                        continue
                    if new_results[arch_name].get('search_space') != space_name:
                        continue
                    gt = orig[arch_name].get('ground_truth', {}).get(task_key, {})
                    gt_val = gt.get(metric_key)
                    if gt_val is None:
                        continue
                    
                    proxy_score = new_results[arch_name].get(proxy_name)
                    if proxy_score is None or proxy_score == 0.0:
                        continue
                    
                    scores.append(proxy_score)
                    gt_vals.append(gt_val)
                
                if len(scores) > 10:
                    rho, _ = stats.spearmanr(scores, gt_vals)
                    print(f"{rho:>6.3f} (n={len(scores):<4})", end=" ")
                else:
                    print(f"  N/A  (n={len(scores):<4})", end=" ")
            print()
    
    # Space-aware results for new baselines
    print(f"\n--- Space-Aware Proxy Selector with new baselines ---")
    for task_key, (metric_key, task_label) in TASK_KEYS.items():
        # Collect per-space scores for each new proxy
        for proxy_name, proxy_label in zip(proxies, proxy_labels):
            all_z_scores = []
            all_gt = []
            
            for space_name in ['macro', 'micro']:
                scores = []
                gt_vals = []
                
                for arch_name in new_results:
                    if arch_name not in orig:
                        continue
                    if new_results[arch_name].get('search_space') != space_name:
                        continue
                    gt = orig[arch_name].get('ground_truth', {}).get(task_key, {})
                    gt_val = gt.get(metric_key)
                    if gt_val is None:
                        continue
                    proxy_score = new_results[arch_name].get(proxy_name)
                    if proxy_score is None or proxy_score == 0.0:
                        continue
                    scores.append(proxy_score)
                    gt_vals.append(gt_val)
                
                if len(scores) > 10:
                    mean_s = np.mean(scores)
                    std_s = np.std(scores)
                    if std_s > 0:
                        z = [(s - mean_s) / std_s for s in scores]
                    else:
                        z = [0.0] * len(scores)
                    all_z_scores.extend(z)
                    all_gt.extend(gt_vals)
            
            if len(all_z_scores) > 10:
                rho, _ = stats.spearmanr(all_z_scores, all_gt)
    
    return results_table


def analyze_robustness():
    """Analyze robustness sweep results."""
    path = RESULTS_DIR / "transnas_robustness.json"
    if not path.exists():
        print("transnas_robustness.json not found yet")
        return None

    with open(path) as f:
        rob_data = json.load(f)
    
    print(f"\n{'='*60}")
    print("ROBUSTNESS SWEEP ANALYSIS")
    print(f"{'='*60}")
    print(f"Configs: {len(rob_data['meta']['configs'])}")
    print(f"Architectures: {rob_data['meta']['n_archs']}")
    print(f"Elapsed: {rob_data['meta']['elapsed_seconds']:.0f}s")
    
    orig = load_original_results()
    results = rob_data['results']
    sample_archs = rob_data['sample_archs']
    
    # 1. SEED STABILITY: Compare proxy rankings across seeds at default config
    print(f"\n--- Seed Stability (res=64, bs=4) ---")
    seed_configs = [k for k in results if '_res64_bs4' in k]
    seed_configs.sort()
    
    proxy_names = ['msfs', 'sfc', 'gradnorm', 'synflow']
    proxy_labels = ['MSFS', 'SFC', 'GradNorm', 'SynFlow']
    
    for task_key, (metric_key, task_label) in TASK_KEYS.items():
        print(f"\n  {task_label}:")
        
        for pname, plabel in zip(proxy_names, proxy_labels):
            rhos = []
            for config_key in seed_configs:
                scores = []
                gt_vals = []
                for ss, arch_name in sample_archs:
                    if arch_name not in results[config_key]:
                        continue
                    if arch_name not in orig:
                        continue
                    gt = orig[arch_name].get('ground_truth', {}).get(task_key, {})
                    gt_val = gt.get(metric_key)
                    if gt_val is None:
                        continue
                    ps = results[config_key][arch_name].get(pname)
                    if ps is None:
                        continue
                    scores.append(ps)
                    gt_vals.append(gt_val)
                
                if len(scores) > 10:
                    rho, _ = stats.spearmanr(scores, gt_vals)
                    rhos.append(rho)
            
            if rhos:
                mean_rho = np.mean(rhos)
                std_rho = np.std(rhos)
                print(f"    {plabel:<12}: rho = {mean_rho:.3f} +/- {std_rho:.3f} (n_seeds={len(rhos)})")
    
    # 2. RESOLUTION STABILITY
    print(f"\n--- Resolution Stability (seed=42, bs=4) ---")
    res_configs = {32: 'seed42_res32_bs4', 64: 'seed42_res64_bs4', 128: 'seed42_res128_bs4'}
    
    for task_key, (metric_key, task_label) in TASK_KEYS.items():
        print(f"\n  {task_label}:")
        for pname, plabel in zip(proxy_names, proxy_labels):
            res_rhos = {}
            for res, config_key in res_configs.items():
                if config_key not in results:
                    continue
                scores = []
                gt_vals = []
                for ss, arch_name in sample_archs:
                    if arch_name not in results[config_key]:
                        continue
                    if arch_name not in orig:
                        continue
                    gt = orig[arch_name].get('ground_truth', {}).get(task_key, {})
                    gt_val = gt.get(metric_key)
                    if gt_val is None:
                        continue
                    ps = results[config_key][arch_name].get(pname)
                    if ps is None:
                        continue
                    scores.append(ps)
                    gt_vals.append(gt_val)
                
                if len(scores) > 10:
                    rho, _ = stats.spearmanr(scores, gt_vals)
                    res_rhos[res] = rho
            
            if res_rhos:
                rho_str = ", ".join(f"{r}px: {v:.3f}" for r, v in sorted(res_rhos.items()))
                print(f"    {plabel:<12}: {rho_str}")
    
    # 3. BATCH SIZE STABILITY
    print(f"\n--- Batch Size Stability (seed=42, res=64) ---")
    bs_configs = {1: 'seed42_res64_bs1', 4: 'seed42_res64_bs4', 16: 'seed42_res64_bs16'}
    
    for task_key, (metric_key, task_label) in TASK_KEYS.items():
        print(f"\n  {task_label}:")
        for pname, plabel in zip(proxy_names, proxy_labels):
            bs_rhos = {}
            for bs, config_key in bs_configs.items():
                if config_key not in results:
                    continue
                scores = []
                gt_vals = []
                for ss, arch_name in sample_archs:
                    if arch_name not in results[config_key]:
                        continue
                    if arch_name not in orig:
                        continue
                    gt = orig[arch_name].get('ground_truth', {}).get(task_key, {})
                    gt_val = gt.get(metric_key)
                    if gt_val is None:
                        continue
                    ps = results[config_key][arch_name].get(pname)
                    if ps is None:
                        continue
                    scores.append(ps)
                    gt_vals.append(gt_val)
                
                if len(scores) > 10:
                    rho, _ = stats.spearmanr(scores, gt_vals)
                    bs_rhos[bs] = rho
            
            if bs_rhos:
                rho_str = ", ".join(f"bs={b}: {v:.3f}" for b, v in sorted(bs_rhos.items()))
                print(f"    {plabel:<12}: {rho_str}")
    
    # 4. RANK CONSISTENCY across seeds (Kendall W)
    print(f"\n--- Rank Consistency (Kendall W across seeds, segmentation) ---")
    task_key = 'segmentsemantic'
    metric_key = 'test_mIoU_best'
    
    for pname, plabel in zip(proxy_names, proxy_labels):
        rankings = []
        for config_key in seed_configs:
            arch_scores = {}
            for ss, arch_name in sample_archs:
                if arch_name not in results.get(config_key, {}):
                    continue
                ps = results[config_key][arch_name].get(pname)
                if ps is not None:
                    arch_scores[arch_name] = ps
            
            if arch_scores:
                sorted_archs = sorted(arch_scores.keys())
                scores = [arch_scores[a] for a in sorted_archs]
                ranks = stats.rankdata(scores)
                rankings.append(ranks)
        
        if len(rankings) >= 2:
            # Compute rank correlation between first and each other seed
            corrs = []
            for i in range(1, len(rankings)):
                if len(rankings[0]) == len(rankings[i]):
                    r, _ = stats.spearmanr(rankings[0], rankings[i])
                    corrs.append(r)
            if corrs:
                mean_corr = np.mean(corrs)
                std_corr = np.std(corrs)
                print(f"  {plabel:<12}: mean rank corr = {mean_corr:.4f} +/- {std_corr:.4f}")
    
    return rob_data


def generate_baseline_figure(results_table):
    """Generate comparison figure including new baselines."""
    if results_table is None:
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    
    task_labels = [v[1] for v in TASK_KEYS.values()]
    
    # All proxies to plot
    all_proxies = ['TE-NAS', 'ZenNAS', 'NASWOT-fixed']
    existing_proxies = ['MSFS', 'GradNorm', 'SynFlow', '#Params']
    
    # Known existing results from the paper
    existing_rhos = {
        'MSFS': [0.132, 0.163, 0.207, 0.244, 0.262],
        'GradNorm': [0.423, 0.519, 0.506, 0.506, 0.718],
        'SynFlow': [0.259, 0.328, 0.291, 0.321, 0.036],
        '#Params': [0.245, 0.371, 0.202, 0.444, -0.111],
    }
    
    # New proxy results
    new_rhos = {}
    for proxy_label in all_proxies:
        rhos = []
        for task_label in task_labels:
            if task_label in results_table and proxy_label in results_table[task_label]:
                rho, _, _ = results_table[task_label][proxy_label]
                rhos.append(rho)
            else:
                rhos.append(0.0)
        new_rhos[proxy_label] = rhos
    
    all_proxy_names = existing_proxies + all_proxies
    all_rhos_data = {**existing_rhos, **new_rhos}
    
    x = np.arange(len(task_labels))
    width = 0.11
    n_proxies = len(all_proxy_names)
    
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', '#00BCD4', '#795548']
    
    for i, proxy_name in enumerate(all_proxy_names):
        offset = (i - n_proxies / 2 + 0.5) * width
        rhos = all_rhos_data.get(proxy_name, [0]*5)
        bars = ax.bar(x + offset, rhos, width, label=proxy_name, color=colors[i % len(colors)], alpha=0.85)
    
    ax.set_xlabel('Task')
    ax.set_ylabel('Spearman ρ')
    ax.set_title('All Proxies: Spearman ρ (N=7,344)')
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=9)
    ax.legend(fontsize=7, ncol=4, loc='upper left')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_ylim(-0.2, 0.8)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_all_baselines.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / "fig_all_baselines.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved fig_all_baselines.pdf/png")


def generate_robustness_figure(rob_data):
    """Generate robustness sweep figures."""
    if rob_data is None:
        return
    
    orig = load_original_results()
    results = rob_data['results']
    sample_archs = rob_data['sample_archs']
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    
    proxy_names = ['msfs', 'gradnorm', 'synflow']
    proxy_labels = ['MSFS', 'GradNorm', 'SynFlow']
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    task_key = 'segmentsemantic'
    metric_key = 'test_mIoU_best'
    
    # Panel 1: Seed stability
    ax = axes[0]
    seed_configs = sorted([k for k in results if '_res64_bs4' in k])
    
    for pname, plabel, color in zip(proxy_names, proxy_labels, colors):
        rhos = []
        for config_key in seed_configs:
            scores, gt_vals = [], []
            for ss, arch_name in sample_archs:
                if arch_name not in results.get(config_key, {}):
                    continue
                if arch_name not in orig:
                    continue
                gt = orig[arch_name].get('ground_truth', {}).get(task_key, {})
                gt_val = gt.get(metric_key)
                ps = results[config_key][arch_name].get(pname)
                if gt_val is not None and ps is not None:
                    scores.append(ps)
                    gt_vals.append(gt_val)
            if len(scores) > 10:
                rho, _ = stats.spearmanr(scores, gt_vals)
                rhos.append(rho)
        
        if rhos:
            ax.plot(range(len(rhos)), rhos, 'o-', color=color, label=plabel, markersize=4, alpha=0.8)
    
    ax.set_xlabel('Random Seed')
    ax.set_ylabel('Spearman ρ')
    ax.set_title('(a) Seed Stability (Seg.)')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    
    # Panel 2: Resolution stability
    ax = axes[1]
    resolutions = [32, 64, 128]
    
    for pname, plabel, color in zip(proxy_names, proxy_labels, colors):
        rhos = []
        for res in resolutions:
            config_key = f"seed42_res{res}_bs4"
            if config_key not in results:
                rhos.append(0)
                continue
            scores, gt_vals = [], []
            for ss, arch_name in sample_archs:
                if arch_name not in results.get(config_key, {}):
                    continue
                if arch_name not in orig:
                    continue
                gt = orig[arch_name].get('ground_truth', {}).get(task_key, {})
                gt_val = gt.get(metric_key)
                ps = results[config_key][arch_name].get(pname)
                if gt_val is not None and ps is not None:
                    scores.append(ps)
                    gt_vals.append(gt_val)
            if len(scores) > 10:
                rho, _ = stats.spearmanr(scores, gt_vals)
                rhos.append(rho)
            else:
                rhos.append(0)
        
        ax.plot(resolutions, rhos, 's-', color=color, label=plabel, markersize=6)
    
    ax.set_xlabel('Input Resolution')
    ax.set_ylabel('Spearman ρ')
    ax.set_title('(b) Resolution Sensitivity (Seg.)')
    ax.set_xticks(resolutions)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    
    # Panel 3: Batch size stability
    ax = axes[2]
    batch_sizes = [1, 4, 16]
    
    for pname, plabel, color in zip(proxy_names, proxy_labels, colors):
        rhos = []
        for bs in batch_sizes:
            config_key = f"seed42_res64_bs{bs}"
            if config_key not in results:
                rhos.append(0)
                continue
            scores, gt_vals = [], []
            for ss, arch_name in sample_archs:
                if arch_name not in results.get(config_key, {}):
                    continue
                if arch_name not in orig:
                    continue
                gt = orig[arch_name].get('ground_truth', {}).get(task_key, {})
                gt_val = gt.get(metric_key)
                ps = results[config_key][arch_name].get(pname)
                if gt_val is not None and ps is not None:
                    scores.append(ps)
                    gt_vals.append(gt_val)
            if len(scores) > 10:
                rho, _ = stats.spearmanr(scores, gt_vals)
                rhos.append(rho)
            else:
                rhos.append(0)
        
        ax.plot(batch_sizes, rhos, 'D-', color=color, label=plabel, markersize=6)
    
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Spearman ρ')
    ax.set_title('(c) Batch Size Sensitivity (Seg.)')
    ax.set_xticks(batch_sizes)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_robustness.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / "fig_robustness.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved fig_robustness.pdf/png")


if __name__ == "__main__":
    print("Analyzing extended evaluation results...")
    
    # Analyze baselines
    baseline_results = analyze_new_baselines()
    
    # Analyze robustness
    rob_data = analyze_robustness()
    
    # Generate figures
    if baseline_results:
        generate_baseline_figure(baseline_results)
    if rob_data:
        generate_robustness_figure(rob_data)
    
    print("\nDone!")
