"""
RepNAS v3: Comprehensive Analysis
==================================
Analyzes the v3 experiment results:
1. Cross-teacher Spearman correlations
2. Probe-type ablation
3. Partial correlations (controlling for params)
4. Within-family correlations
5. Top-k hit rate / regret
6. Layer-wise CKA analysis
7. Alignment inversion analysis
8. Composite score optimization
"""

import json
import math
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict

# Attempt pingouin for partial correlations
try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False

# ============================================================
# Load data
# ============================================================
with open("/Users/aayambansal/Desktop/VStudio/#mas-2/1/zero-shot-nas-paper/experiments/repnas_v3_results.json") as f:
    data = json.load(f)

results = data["results"]
config = data["config"]

# Build a DataFrame
rows = []
for key, val in results.items():
    # Replace NaN strings with actual NaN
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
print(f"Total rows: {len(df)}")
print(f"Unique candidates: {df['candidate'].nunique()}")
print(f"Unique teachers: {df['teacher'].nunique()}")
print(f"Columns: {list(df.columns)}")

# ============================================================
# 1. CROSS-TEACHER SPEARMAN CORRELATIONS
# ============================================================
print("\n" + "=" * 80)
print("1. CROSS-TEACHER SPEARMAN CORRELATIONS (CKA vs GT Accuracy)")
print("=" * 80)

teachers = df["teacher"].unique()
probe_types = ["noise", "imagenet_val", "augmented"]
metrics = ["cka", "cosine", "knn"]

for teacher in teachers:
    tdf = df[df["teacher"] == teacher].copy()
    print(f"\n--- {teacher} ({len(tdf)} models) ---")
    
    for metric in metrics:
        for ptype in probe_types:
            col = f"{metric}_pretrained_{ptype}"
            if col in tdf.columns:
                valid = tdf.dropna(subset=[col, "gt_acc"])
                if len(valid) > 5:
                    rho, p = stats.spearmanr(valid[col], valid["gt_acc"])
                    print(f"  {metric:>6s}_{ptype:<12s}: ρ={rho:+.4f}  p={p:.4f}  (n={len(valid)})")
    
    # Random init (noise only)
    for metric in metrics:
        col = f"{metric}_random"
        if col in tdf.columns:
            valid = tdf.dropna(subset=[col, "gt_acc"])
            if len(valid) > 5:
                rho, p = stats.spearmanr(valid[col], valid["gt_acc"])
                print(f"  {metric:>6s}_random       : ρ={rho:+.4f}  p={p:.4f}  (n={len(valid)})")

# ============================================================
# 2. ZS-NAS BASELINES (from DINOv2 teacher pass)
# ============================================================
print("\n" + "=" * 80)
print("2. ZS-NAS BASELINES")
print("=" * 80)

dino_df = df[df["teacher"] == "dinov2_small"].copy()
for baseline in ["gradnorm", "naswot", "synflow"]:
    if baseline in dino_df.columns:
        valid = dino_df.dropna(subset=[baseline, "gt_acc"])
        if len(valid) > 5:
            rho, p = stats.spearmanr(valid[baseline], valid["gt_acc"])
            print(f"  {baseline:>10s}: ρ={rho:+.4f}  p={p:.6f}  (n={len(valid)})")

# ============================================================
# 3. PROBE TYPE ABLATION (averaged across teachers)
# ============================================================
print("\n" + "=" * 80)
print("3. PROBE TYPE ABLATION (average ρ across teachers)")
print("=" * 80)

for metric in metrics:
    print(f"\n  --- {metric.upper()} ---")
    for ptype in probe_types:
        col = f"{metric}_pretrained_{ptype}"
        rhos = []
        for teacher in teachers:
            tdf = df[df["teacher"] == teacher]
            valid = tdf.dropna(subset=[col, "gt_acc"])
            if len(valid) > 5:
                rho, p = stats.spearmanr(valid[col], valid["gt_acc"])
                rhos.append(rho)
        if rhos:
            print(f"    {ptype:<14s}: mean_ρ={np.mean(rhos):+.4f}  std={np.std(rhos):.4f}  teachers={len(rhos)}")
    
    # Random init
    col = f"{metric}_random"
    rhos = []
    for teacher in teachers:
        tdf = df[df["teacher"] == teacher]
        valid = tdf.dropna(subset=[col, "gt_acc"])
        if len(valid) > 5:
            rho, p = stats.spearmanr(valid[col], valid["gt_acc"])
            rhos.append(rho)
    if rhos:
        print(f"    {'random':<14s}: mean_ρ={np.mean(rhos):+.4f}  std={np.std(rhos):.4f}  teachers={len(rhos)}")

# ============================================================
# 4. PARTIAL CORRELATIONS (controlling for params/FLOPs)
# ============================================================
print("\n" + "=" * 80)
print("4. PARTIAL CORRELATIONS (controlling for log(params))")
print("=" * 80)

for teacher in teachers:
    tdf = df[df["teacher"] == teacher].copy()
    tdf["log_params"] = np.log(tdf["params"])
    print(f"\n--- {teacher} ---")
    
    for ptype in ["noise", "imagenet_val", "augmented"]:
        col = f"cka_pretrained_{ptype}"
        valid = tdf.dropna(subset=[col, "gt_acc", "log_params"])
        if len(valid) > 10 and HAS_PINGOUIN:
            result = pg.partial_corr(data=valid, x=col, y="gt_acc", covar="log_params", method="spearman")
            r = result["r"].values[0]
            p = result["p_val"].values[0]
            print(f"  CKA_{ptype:<12s} | ρ_partial={r:+.4f}  p={p:.4f}")
        elif len(valid) > 10:
            # Manual partial correlation using residuals
            from scipy.stats import spearmanr
            # Rank-based partial correlation
            r_xz = spearmanr(valid[col], valid["log_params"])[0]
            r_yz = spearmanr(valid["gt_acc"], valid["log_params"])[0]
            r_xy = spearmanr(valid[col], valid["gt_acc"])[0]
            r_partial = (r_xy - r_xz * r_yz) / (np.sqrt(1 - r_xz**2) * np.sqrt(1 - r_yz**2) + 1e-10)
            print(f"  CKA_{ptype:<12s} | ρ_partial≈{r_partial:+.4f} (manual)")
    
    # Baselines partial
    if teacher == "dinov2_small":
        for baseline in ["gradnorm", "naswot", "synflow"]:
            if baseline in valid.columns:
                valid2 = tdf.dropna(subset=[baseline, "gt_acc", "log_params"])
                if len(valid2) > 10 and HAS_PINGOUIN:
                    result = pg.partial_corr(data=valid2, x=baseline, y="gt_acc", covar="log_params", method="spearman")
                    r = result["r"].values[0]
                    p = result["p_val"].values[0]
                    print(f"  {baseline:>10s}     | ρ_partial={r:+.4f}  p={p:.4f}")

# ============================================================
# 5. WITHIN-FAMILY CORRELATIONS
# ============================================================
print("\n" + "=" * 80)
print("5. WITHIN-FAMILY CORRELATIONS (DINOv2, CKA noise)")
print("=" * 80)

dino_df = df[df["teacher"] == "dinov2_small"].copy()
families = dino_df["family"].value_counts()
print(f"\nFamily sizes: {dict(families)}")

for family in families.index:
    fdf = dino_df[dino_df["family"] == family]
    valid = fdf.dropna(subset=["cka_pretrained_noise", "gt_acc"])
    if len(valid) >= 3:
        rho, p = stats.spearmanr(valid["cka_pretrained_noise"], valid["gt_acc"])
        rho_img, p_img = stats.spearmanr(
            valid.dropna(subset=["cka_pretrained_imagenet_val"])["cka_pretrained_imagenet_val"],
            valid.dropna(subset=["cka_pretrained_imagenet_val"])["gt_acc"]
        ) if "cka_pretrained_imagenet_val" in valid.columns and len(valid.dropna(subset=["cka_pretrained_imagenet_val"])) >= 3 else (np.nan, np.nan)
        print(f"  {family:<12s} (n={len(valid):2d}): noise_ρ={rho:+.4f}  imgval_ρ={rho_img:+.4f}")

# Also within-family for imagenet_val probes
print("\n  --- Per probe type within-family ρ (DINOv2) ---")
for ptype in ["noise", "imagenet_val", "augmented"]:
    col = f"cka_pretrained_{ptype}"
    within_rhos = []
    for family in families.index:
        fdf = dino_df[dino_df["family"] == family]
        valid = fdf.dropna(subset=[col, "gt_acc"])
        if len(valid) >= 3:
            rho, _ = stats.spearmanr(valid[col], valid["gt_acc"])
            within_rhos.append(rho)
    if within_rhos:
        print(f"  {ptype:<14s}: mean_within_ρ={np.mean(within_rhos):+.4f}  (n_families={len(within_rhos)})")

# ============================================================
# 6. TOP-K HIT RATE / REGRET
# ============================================================
print("\n" + "=" * 80)
print("6. TOP-K HIT RATE & REGRET ANALYSIS")
print("=" * 80)

for teacher in teachers:
    tdf = df[df["teacher"] == teacher].copy()
    
    for ptype in ["noise", "imagenet_val"]:
        col = f"cka_pretrained_{ptype}"
        valid = tdf.dropna(subset=[col, "gt_acc"])
        if len(valid) < 10:
            continue
        
        # Oracle top-k by accuracy
        sorted_by_acc = valid.sort_values("gt_acc", ascending=False)
        
        for k in [5, 10, 20]:
            if k > len(valid):
                continue
            oracle_top_k = set(sorted_by_acc.head(k)["candidate"].values)
            oracle_mean_acc = sorted_by_acc.head(k)["gt_acc"].mean()
            
            # Top-k by proxy (higher CKA = better? or lower?)
            # Test both directions
            sorted_high = valid.sort_values(col, ascending=False)
            sorted_low = valid.sort_values(col, ascending=True)
            
            proxy_top_k_high = set(sorted_high.head(k)["candidate"].values)
            proxy_top_k_low = set(sorted_low.head(k)["candidate"].values)
            
            hit_high = len(oracle_top_k & proxy_top_k_high) / k
            hit_low = len(oracle_top_k & proxy_top_k_low) / k
            
            mean_acc_high = sorted_high.head(k)["gt_acc"].mean()
            mean_acc_low = sorted_low.head(k)["gt_acc"].mean()
            
            regret_high = oracle_mean_acc - mean_acc_high
            regret_low = oracle_mean_acc - mean_acc_low
            
            if k == 10:  # Only print k=10 for brevity
                print(f"  {teacher:>20s} | {ptype:<12s} | k={k:2d} | "
                      f"hit↑={hit_high:.2f} reg↑={regret_high:.1f} | "
                      f"hit↓={hit_low:.2f} reg↓={regret_low:.1f}")
    
    # Also GradNorm for DINOv2
    if teacher == "dinov2_small":
        valid = tdf.dropna(subset=["gradnorm", "gt_acc"])
        sorted_by_acc = valid.sort_values("gt_acc", ascending=False)
        for k in [10]:
            oracle_top_k = set(sorted_by_acc.head(k)["candidate"].values)
            oracle_mean_acc = sorted_by_acc.head(k)["gt_acc"].mean()
            sorted_high = valid.sort_values("gradnorm", ascending=False)
            proxy_top_k = set(sorted_high.head(k)["candidate"].values)
            hit = len(oracle_top_k & proxy_top_k) / k
            mean_acc = sorted_high.head(k)["gt_acc"].mean()
            regret = oracle_mean_acc - mean_acc
            print(f"  {'GradNorm':>20s} | {'baseline':<12s} | k={k:2d} | "
                  f"hit={hit:.2f} reg={regret:.1f}")

# ============================================================
# 7. LAYER-WISE CKA ANALYSIS (DINOv2 only)
# ============================================================
print("\n" + "=" * 80)
print("7. LAYER-WISE CKA ANALYSIS (DINOv2)")
print("=" * 80)

for layer in ["early", "mid", "late"]:
    col = f"cka_{layer}_layer"
    if col in dino_df.columns:
        valid = dino_df.dropna(subset=[col, "gt_acc"])
        if len(valid) > 5:
            rho, p = stats.spearmanr(valid[col], valid["gt_acc"])
            print(f"  {layer:>5s} layer CKA: ρ={rho:+.4f}  p={p:.6f}  (n={len(valid)})")
            # Also check range
            print(f"         range: [{valid[col].min():.6f}, {valid[col].max():.6f}]  mean={valid[col].mean():.6f}")

# ============================================================
# 8. ALIGNMENT INVERSION: Does (1-CKA) beat direct CKA?
# ============================================================
print("\n" + "=" * 80)
print("8. ALIGNMENT INVERSION ANALYSIS")
print("=" * 80)

for teacher in teachers:
    tdf = df[df["teacher"] == teacher].copy()
    print(f"\n--- {teacher} ---")
    
    for ptype in ["noise", "imagenet_val", "augmented"]:
        col = f"cka_pretrained_{ptype}"
        valid = tdf.dropna(subset=[col, "gt_acc"])
        if len(valid) > 5:
            rho_direct, p_direct = stats.spearmanr(valid[col], valid["gt_acc"])
            rho_inv, p_inv = stats.spearmanr(1.0 - valid[col], valid["gt_acc"])
            print(f"  {ptype:<14s}: direct_ρ={rho_direct:+.4f} (p={p_direct:.4f}) | "
                  f"inverted_ρ={rho_inv:+.4f} (p={p_inv:.4f}) | "
                  f"{'INVERSION' if rho_direct < 0 and abs(rho_direct) > 0.1 else 'no inversion'}")

# ============================================================
# 9. CROSS-TEACHER COMPARISON SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("9. CROSS-TEACHER SUMMARY (CKA noise ρ with GT accuracy)")
print("=" * 80)

teacher_summary = []
for teacher in teachers:
    tdf = df[df["teacher"] == teacher]
    valid = tdf.dropna(subset=["cka_pretrained_noise", "gt_acc"])
    rho, p = stats.spearmanr(valid["cka_pretrained_noise"], valid["gt_acc"])
    
    valid_img = tdf.dropna(subset=["cka_pretrained_imagenet_val", "gt_acc"])
    rho_img, p_img = stats.spearmanr(valid_img["cka_pretrained_imagenet_val"], valid_img["gt_acc"])
    
    print(f"  {teacher:<22s}: noise_ρ={rho:+.4f} (p={p:.4f})  imgval_ρ={rho_img:+.4f} (p={p_img:.4f})  n={len(valid)}")
    teacher_summary.append({"teacher": teacher, "rho_noise": rho, "rho_imgval": rho_img})

# ============================================================
# 10. COMPOSITE SCORE: weighted combination
# ============================================================
print("\n" + "=" * 80)
print("10. COMPOSITE SCORE OPTIMIZATION")
print("=" * 80)

# For DINOv2, try combinations
dino_valid = dino_df.dropna(subset=["cka_pretrained_noise", "cka_pretrained_imagenet_val", 
                                     "gradnorm", "gt_acc"]).copy()
if len(dino_valid) > 10:
    # Normalize each metric to [0, 1]
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    
    metrics_to_combine = ["cka_pretrained_noise", "cka_pretrained_imagenet_val", 
                          "cka_pretrained_augmented", "gradnorm"]
    valid_metrics = [m for m in metrics_to_combine if m in dino_valid.columns and dino_valid[m].notna().sum() > 10]
    
    for m in valid_metrics:
        dino_valid[f"{m}_norm"] = scaler.fit_transform(dino_valid[[m]])
    
    # Try different weightings
    best_rho = -1
    best_config = ""
    
    # Grid search over weights for noise CKA, imgval CKA, gradnorm
    for w_noise in np.arange(0, 1.1, 0.1):
        for w_img in np.arange(0, 1.1 - w_noise, 0.1):
            w_grad = 1.0 - w_noise - w_img
            if w_grad < -0.01:
                continue
            
            score = (w_noise * dino_valid["cka_pretrained_noise_norm"] + 
                     w_img * dino_valid["cka_pretrained_imagenet_val_norm"] +
                     w_grad * dino_valid["gradnorm_norm"])
            rho, p = stats.spearmanr(score, dino_valid["gt_acc"])
            if abs(rho) > abs(best_rho):
                best_rho = rho
                best_config = f"noise={w_noise:.1f}, img={w_img:.1f}, grad={w_grad:.1f}"
    
    print(f"  Best composite: ρ={best_rho:+.4f} with {best_config}")
    
    # Also try inverted CKA + GradNorm
    dino_valid["inv_cka_noise_norm"] = 1.0 - dino_valid["cka_pretrained_noise_norm"]
    dino_valid["inv_cka_img_norm"] = 1.0 - dino_valid["cka_pretrained_imagenet_val_norm"]
    
    for w_inv in np.arange(0, 1.1, 0.1):
        w_grad = 1.0 - w_inv
        score = w_inv * dino_valid["inv_cka_noise_norm"] + w_grad * dino_valid["gradnorm_norm"]
        rho, p = stats.spearmanr(score, dino_valid["gt_acc"])
        if abs(rho) > abs(best_rho):
            best_rho = rho
            best_config = f"inv_noise={w_inv:.1f}, gradnorm={w_grad:.1f}"
    
    print(f"  Best with inversion: ρ={best_rho:+.4f} with {best_config}")

# ============================================================
# 11. DATA SUMMARY TABLE (for paper)
# ============================================================
print("\n" + "=" * 80)
print("11. CANDIDATE SUMMARY (DINOv2 teacher)")
print("=" * 80)

cols_to_show = ["candidate", "family", "params", "gt_acc", 
                "cka_pretrained_noise", "cka_pretrained_imagenet_val",
                "cka_random", "gradnorm", "naswot", "synflow"]
summary = dino_df[cols_to_show].sort_values("gt_acc", ascending=False)
print(summary.to_string(index=False, float_format="%.4f"))

# Save analysis summary
analysis = {
    "cross_teacher": [],
    "baselines": {},
    "probe_ablation": {},
    "layer_wise": {},
}

for teacher in teachers:
    tdf = df[df["teacher"] == teacher]
    valid = tdf.dropna(subset=["cka_pretrained_noise", "gt_acc"])
    rho, p = stats.spearmanr(valid["cka_pretrained_noise"], valid["gt_acc"])
    analysis["cross_teacher"].append({
        "teacher": teacher,
        "rho": float(rho),
        "p": float(p),
        "n": int(len(valid))
    })

with open("/Users/aayambansal/Desktop/VStudio/#mas-2/1/zero-shot-nas-paper/experiments/analysis_v3_summary.json", "w") as f:
    json.dump(analysis, f, indent=2)

print("\n\nAnalysis complete. Summary saved to experiments/analysis_v3_summary.json")
