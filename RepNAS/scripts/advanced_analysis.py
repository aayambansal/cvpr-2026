"""
RepNAS: Advanced Analyses for 10/10 paper
==========================================
1. Bootstrap confidence intervals on all correlations
2. Proxy-guided evolutionary search simulation
3. Cross-family vs within-family sign flip analysis
4. Family-controlled correlation (residualization)
"""

import json
import math
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict

# ============================================================
# Load data
# ============================================================
with open("/Users/aayambansal/Desktop/VStudio/#mas-2/1/zero-shot-nas-paper/experiments/repnas_v3_results.json") as f:
    data = json.load(f)

rows = []
for key, val in data["results"].items():
    row = {}
    for k, v in val.items():
        if isinstance(v, float) and math.isnan(v):
            row[k] = np.nan
        elif v == "nan":
            row[k] = np.nan
        else:
            row[k] = v
    rows.append(row)
df = pd.DataFrame(rows)

# ============================================================
# 1. BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================
print("=" * 80)
print("1. BOOTSTRAP CONFIDENCE INTERVALS (10,000 resamples)")
print("=" * 80)

def bootstrap_spearman(x, y, n_boot=10000, seed=42):
    """Bootstrap CI for Spearman correlation."""
    rng = np.random.RandomState(seed)
    n = len(x)
    rhos = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        rho, _ = stats.spearmanr(x[idx], y[idx])
        if not np.isnan(rho):
            rhos.append(rho)
    rhos = np.array(rhos)
    ci_low = np.percentile(rhos, 2.5)
    ci_high = np.percentile(rhos, 97.5)
    return np.mean(rhos), ci_low, ci_high, np.std(rhos)

teachers = ["dinov2_small", "clip_vit_b32", "convnext_base_fcmae", "mae_vit_base"]
teacher_labels = ["DINOv2", "CLIP", "ConvNeXtV2", "MAE"]

bootstrap_results = {}

for teacher, tlabel in zip(teachers, teacher_labels):
    tdf = df[df["teacher"] == teacher]
    print(f"\n--- {tlabel} ---")
    
    for metric_label, col in [
        ("CKA noise", "cka_pretrained_noise"),
        ("CKA imgval", "cka_pretrained_imagenet_val"),
        ("CKA augment", "cka_pretrained_augmented"),
        ("CKA random", "cka_random"),
        ("kNN noise", "knn_pretrained_noise"),
        ("kNN imgval", "knn_pretrained_imagenet_val"),
    ]:
        valid = tdf.dropna(subset=[col, "gt_acc"])
        if len(valid) > 10:
            x = valid[col].values
            y = valid["gt_acc"].values
            mean_rho, ci_low, ci_high, std = bootstrap_spearman(x, y)
            print(f"  {metric_label:<14s}: ρ={mean_rho:+.4f} [{ci_low:+.4f}, {ci_high:+.4f}]  n={len(valid)}")
            bootstrap_results[f"{teacher}__{col}"] = {
                "mean_rho": float(mean_rho),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "std": float(std),
                "n": int(len(valid)),
            }

    # Baselines (only for DINOv2)
    if teacher == "dinov2_small":
        for bl_name, bl_col in [("GradNorm", "gradnorm"), ("NASWOT", "naswot"), ("SynFlow", "synflow")]:
            valid = tdf.dropna(subset=[bl_col, "gt_acc"])
            if len(valid) > 10:
                x = valid[bl_col].values
                y = valid["gt_acc"].values
                mean_rho, ci_low, ci_high, std = bootstrap_spearman(x, y)
                print(f"  {bl_name:<14s}: ρ={mean_rho:+.4f} [{ci_low:+.4f}, {ci_high:+.4f}]  n={len(valid)}")
                bootstrap_results[f"{teacher}__{bl_col}"] = {
                    "mean_rho": float(mean_rho),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "std": float(std),
                    "n": int(len(valid)),
                }

# ============================================================
# 2. PROXY-GUIDED SEARCH SIMULATION
# ============================================================
print("\n" + "=" * 80)
print("2. PROXY-GUIDED SEARCH SIMULATION")
print("=" * 80)

def proxy_search_simulation(proxy_scores, gt_accs, budget=10, n_trials=1000, seed=42):
    """
    Simulate proxy-guided architecture search vs random search.
    
    Proxy-guided: rank all N architectures by proxy, evaluate the top-`budget` by GT.
    This is the realistic use case: you compute the cheap proxy for all candidates,
    then train only the top-K as ranked by the proxy.
    
    Random: randomly sample `budget` architectures and train them.
    
    Metrics:
    - best_acc: best GT accuracy found in the selected set
    - mean_acc: mean GT accuracy of the selected set
    - hit_rate_top5: fraction of true top-5 that appear in selected set
    - hit_rate_top10: fraction of true top-10 that appear in selected set
    - regret: (oracle_best - proxy_best) / oracle_best * 100
    """
    rng = np.random.RandomState(seed)
    n = len(proxy_scores)
    
    # True top-K indices
    gt_sorted = np.argsort(gt_accs)[::-1]  # descending by GT acc
    true_top5 = set(gt_sorted[:5])
    true_top10 = set(gt_sorted[:10])
    oracle_best = gt_accs[gt_sorted[0]]
    
    # Proxy-guided: select top-budget by proxy (LOWEST proxy = best for inverted CKA)
    proxy_sorted = np.argsort(proxy_scores)  # ascending = lowest CKA first
    proxy_selected = set(proxy_sorted[:budget])
    proxy_best = max(gt_accs[i] for i in proxy_selected)
    proxy_mean = np.mean([gt_accs[i] for i in proxy_selected])
    proxy_hit5 = len(proxy_selected & true_top5) / min(5, budget)
    proxy_hit10 = len(proxy_selected & true_top10) / min(10, budget)
    proxy_regret = (oracle_best - proxy_best) / oracle_best * 100
    
    # Random search: sample `budget` random, repeat n_trials times
    random_bests = []
    random_means = []
    random_hit5s = []
    random_hit10s = []
    for _ in range(n_trials):
        rand_selected = set(rng.choice(n, budget, replace=False))
        random_bests.append(max(gt_accs[i] for i in rand_selected))
        random_means.append(np.mean([gt_accs[i] for i in rand_selected]))
        random_hit5s.append(len(rand_selected & true_top5) / min(5, budget))
        random_hit10s.append(len(rand_selected & true_top10) / min(10, budget))
    
    return {
        "proxy_best": float(proxy_best),
        "proxy_mean_acc": float(proxy_mean),
        "proxy_hit5": float(proxy_hit5),
        "proxy_hit10": float(proxy_hit10),
        "proxy_regret_pct": float(proxy_regret),
        "random_best_mean": float(np.mean(random_bests)),
        "random_best_std": float(np.std(random_bests)),
        "random_mean_acc": float(np.mean(random_means)),
        "random_hit5_mean": float(np.mean(random_hit5s)),
        "random_hit10_mean": float(np.mean(random_hit10s)),
        "random_regret_pct": float((oracle_best - np.mean(random_bests)) / oracle_best * 100),
        "oracle_best": float(oracle_best),
        "n": int(n),
        "budget": int(budget),
    }

# Run search simulation for different proxies and budgets
search_results = {}
print(f"\n{'Proxy':<30s} | {'Budget':>6s} | {'P-Best':>7s} | {'R-Best':>13s} | {'P-Hit10':>7s} | {'R-Hit10':>7s} | {'P-Regret':>8s}")
print("-" * 100)

for teacher, tlabel in zip(teachers, teacher_labels):
    tdf = df[df["teacher"] == teacher].copy()
    
    for proxy_name, proxy_col in [
        (f"{tlabel} CKA-noise", "cka_pretrained_noise"),
        (f"{tlabel} CKA-imgval", "cka_pretrained_imagenet_val"),
        (f"{tlabel} kNN-imgval", "knn_pretrained_imagenet_val"),
    ]:
        valid = tdf.dropna(subset=[proxy_col, "gt_acc"]).reset_index(drop=True)
        if len(valid) < 20:
            continue
        
        proxy = valid[proxy_col].values
        accs = valid["gt_acc"].values
        
        for budget in [5, 10, 15, 20]:
            result = proxy_search_simulation(proxy, accs, budget=budget)
            key = f"{teacher}__{proxy_col}__budget{budget}"
            search_results[key] = result
            
            if budget == 10:
                print(f"  {proxy_name:<28s} | {budget:>6d} | {result['proxy_best']:>7.1f} | "
                      f"{result['random_best_mean']:>5.1f}±{result['random_best_std']:>4.1f} | "
                      f"{result['proxy_hit10']:>7.0%} | {result['random_hit10_mean']:>7.0%} | "
                      f"{result['proxy_regret_pct']:>7.2f}%")

# Baselines: GradNorm, SynFlow (for DINOv2 subset)
dino_df = df[df["teacher"] == "dinov2_small"].copy()

for bl_name, bl_col, invert in [("GradNorm", "gradnorm", True), ("SynFlow", "synflow", False)]:
    valid = dino_df.dropna(subset=[bl_col, "gt_acc"]).reset_index(drop=True)
    if len(valid) < 20:
        continue
    proxy = valid[bl_col].values
    accs = valid["gt_acc"].values
    if invert:
        proxy = -proxy  # Higher GradNorm = better → negate for ascending sort
    
    for budget in [5, 10, 15, 20]:
        result = proxy_search_simulation(proxy, accs, budget=budget)
        key = f"{bl_name}__budget{budget}"
        search_results[key] = result
        if budget == 10:
            print(f"  {bl_name:<28s} | {budget:>6d} | {result['proxy_best']:>7.1f} | "
                  f"{result['random_best_mean']:>5.1f}±{result['random_best_std']:>4.1f} | "
                  f"{result['proxy_hit10']:>7.0%} | {result['random_hit10_mean']:>7.0%} | "
                  f"{result['proxy_regret_pct']:>7.2f}%")


# ============================================================
# 3. CROSS-FAMILY vs WITHIN-FAMILY ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("3. CROSS-FAMILY VS WITHIN-FAMILY SIGN FLIP ANALYSIS")
print("=" * 80)

dino_df = df[df["teacher"] == "dinov2_small"].copy()

# 3a. Family-residualized correlation
# For each candidate, compute family mean accuracy and family mean CKA
# Then correlate the residuals
print("\n3a. Family-Residualized Correlation (DINOv2, noise)")
family_means_acc = dino_df.groupby("family")["gt_acc"].transform("mean")
family_means_cka = dino_df.dropna(subset=["cka_pretrained_noise"]).groupby("family")["cka_pretrained_noise"].transform("mean")

valid = dino_df.dropna(subset=["cka_pretrained_noise", "gt_acc"]).copy()
valid["acc_resid"] = valid["gt_acc"] - valid.groupby("family")["gt_acc"].transform("mean")
valid["cka_resid"] = valid["cka_pretrained_noise"] - valid.groupby("family")["cka_pretrained_noise"].transform("mean")

rho_resid, p_resid = stats.spearmanr(valid["cka_resid"], valid["acc_resid"])
print(f"  Family-residualized ρ = {rho_resid:+.4f} (p = {p_resid:.4f})")

# 3b. Between-family correlation (family means)
print("\n3b. Between-Family Correlation (family means)")
family_agg = dino_df.dropna(subset=["cka_pretrained_noise"]).groupby("family").agg({
    "gt_acc": "mean",
    "cka_pretrained_noise": "mean",
    "params": "mean",
}).reset_index()
family_agg = family_agg[family_agg.index.isin(
    dino_df.dropna(subset=["cka_pretrained_noise"]).groupby("family").filter(lambda x: len(x) >= 2).groupby("family").ngroups
    if False else range(len(family_agg))
)]

rho_between, p_between = stats.spearmanr(family_agg["cka_pretrained_noise"], family_agg["gt_acc"])
print(f"  Between-family ρ = {rho_between:+.4f} (p = {p_between:.4f})")
print(f"  (n = {len(family_agg)} families)")
for _, row in family_agg.sort_values("gt_acc", ascending=False).iterrows():
    print(f"    {row['family']:<12s}: acc={row['gt_acc']:.1f}  CKA={row['cka_pretrained_noise']:.4f}  params={row['params']:.1f}M")

# 3c. Within-family correlations with significance
print("\n3c. Within-Family Correlations (families with n >= 4)")
family_sizes = dino_df.dropna(subset=["cka_pretrained_noise"]).groupby("family").size()
for family in family_sizes[family_sizes >= 4].index:
    fdf = dino_df[dino_df["family"] == family].dropna(subset=["cka_pretrained_noise", "gt_acc"])
    rho, p = stats.spearmanr(fdf["cka_pretrained_noise"], fdf["gt_acc"])
    
    # Also check if within-family, SIZE explains the correlation
    if len(fdf) >= 4:
        rho_size, p_size = stats.spearmanr(fdf["params"], fdf["gt_acc"])
        rho_cka_size, _ = stats.spearmanr(fdf["cka_pretrained_noise"], fdf["params"])
        print(f"  {family:<12s} (n={len(fdf):2d}): CKA_ρ={rho:+.4f} (p={p:.3f}) | "
              f"size_ρ={rho_size:+.4f} | CKA~size_ρ={rho_cka_size:+.4f}")

# 3d. Sign flip analysis: does the sign of correlation change for 
# small vs large models?
print("\n3d. Small vs Large Model Split (DINOv2, CKA noise)")
valid = dino_df.dropna(subset=["cka_pretrained_noise", "gt_acc"]).copy()
median_params = valid["params"].median()
small = valid[valid["params"] <= median_params]
large = valid[valid["params"] > median_params]

rho_small, p_small = stats.spearmanr(small["cka_pretrained_noise"], small["gt_acc"])
rho_large, p_large = stats.spearmanr(large["cka_pretrained_noise"], large["gt_acc"])
print(f"  Small models (≤{median_params:.1f}M, n={len(small)}): ρ={rho_small:+.4f} (p={p_small:.4f})")
print(f"  Large models (>{median_params:.1f}M, n={len(large)}): ρ={rho_large:+.4f} (p={p_large:.4f})")

# 3e. CNN vs Transformer split
print("\n3e. CNN vs Transformer Architecture Split")
cnn_families = {"CNN", "EffNet", "Mobile", "ConvNeXt", "RegNet", "MetaFormer", "EdgeNeXt", "NAS"}
transformer_families = {"ViT", "Swin", "DeiT", "CaiT", "Mixer", "MaxViT", "EFormer"}

cnn_df = valid[valid["family"].isin(cnn_families)]
trans_df = valid[valid["family"].isin(transformer_families)]

if len(cnn_df) > 5:
    rho_cnn, p_cnn = stats.spearmanr(cnn_df["cka_pretrained_noise"], cnn_df["gt_acc"])
    print(f"  CNN-like (n={len(cnn_df)}): ρ={rho_cnn:+.4f} (p={p_cnn:.4f})")

if len(trans_df) > 5:
    rho_trans, p_trans = stats.spearmanr(trans_df["cka_pretrained_noise"], trans_df["gt_acc"])
    print(f"  Transformer-like (n={len(trans_df)}): ρ={rho_trans:+.4f} (p={p_trans:.4f})")

# ============================================================
# 4. COMPREHENSIVE SUMMARY TABLE FOR PAPER
# ============================================================
print("\n" + "=" * 80)
print("4. COMPREHENSIVE SUMMARY TABLE")
print("=" * 80)

print("\nTeacher | Probe | Metric | ρ | 95% CI | p-value | n")
print("-" * 80)
for teacher, tlabel in zip(teachers, teacher_labels):
    tdf = df[df["teacher"] == teacher]
    for ptype in ["noise", "imagenet_val"]:
        for metric, col in [("CKA", f"cka_pretrained_{ptype}"), ("kNN", f"knn_pretrained_{ptype}")]:
            valid = tdf.dropna(subset=[col, "gt_acc"])
            if len(valid) > 10:
                rho, p = stats.spearmanr(valid[col], valid["gt_acc"])
                bkey = f"{teacher}__{col}"
                if bkey in bootstrap_results:
                    ci = bootstrap_results[bkey]
                    print(f"{tlabel:>10s} | {ptype:<10s} | {metric:<4s} | {rho:+.3f} | [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}] | {p:.5f} | {len(valid)}")

# ============================================================
# Save all results
# ============================================================
output = {
    "bootstrap_results": bootstrap_results,
    "search_results": search_results,
    "family_residualized_rho": float(rho_resid),
    "family_residualized_p": float(p_resid),
    "between_family_rho": float(rho_between),
    "between_family_p": float(p_between),
    "small_model_rho": float(rho_small),
    "large_model_rho": float(rho_large),
    "cnn_rho": float(rho_cnn) if len(cnn_df) > 5 else None,
    "transformer_rho": float(rho_trans) if len(trans_df) > 5 else None,
}

with open("/Users/aayambansal/Desktop/VStudio/#mas-2/1/zero-shot-nas-paper/experiments/advanced_analysis.json", "w") as f:
    json.dump(output, f, indent=2)

print("\n\nAdvanced analysis complete. Results saved to experiments/advanced_analysis.json")
