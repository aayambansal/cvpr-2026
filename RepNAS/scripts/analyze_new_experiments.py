#!/usr/bin/env python3
"""
Analyze new RepNAS experiments (upgrades + transfer) and generate figures.
Run this after downloading repnas_upgrades.json and repnas_transfer.json.
Usage: python scripts/analyze_new_experiments.py
"""

import json
import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from pathlib import Path

# Paths
BASE = Path(__file__).resolve().parent.parent
EXP_DIR = BASE / "experiments"
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Matplotlib setup ──────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "font.family": "sans-serif",
})

# Family color map
FAMILY_COLORS = {
    "CNN": "#1f77b4", "EffNet": "#ff7f0e", "Mobile": "#2ca02c",
    "ConvNeXt": "#d62728", "ViT": "#9467bd", "Swin": "#8c564b",
    "DeiT": "#e377c2", "RegNet": "#7f7f7f", "MaxViT": "#bcbd22",
    "EFormer": "#17becf", "EdgeNeXt": "#aec7e8", "NAS": "#ffbb78",
    "CaiT": "#98df8a", "Mixer": "#ff9896", "MetaFormer": "#c5b0d5",
}


def load_v3():
    """Load v3 results and return per-arch dict keyed by candidate name."""
    with open(EXP_DIR / "repnas_v3_results.json") as f:
        raw = json.load(f)
    
    # Build per-candidate dict (aggregate across teachers)
    archs = {}
    for key, entry in raw["results"].items():
        cand = entry["candidate"]
        if cand not in archs:
            archs[cand] = {
                "acc": entry["gt_acc"],
                "params": entry["params"],
                "family": entry["family"],
            }
        # Store teacher-specific scores
        teacher = entry["teacher"]
        archs[cand][f"cka_noise_{teacher}"] = entry.get("cka_pretrained_noise")
        archs[cand][f"cka_natural_{teacher}"] = entry.get("cka_pretrained_imagenet_val")
        archs[cand][f"knn_noise_{teacher}"] = entry.get("knn_pretrained_noise")
        archs[cand][f"knn_natural_{teacher}"] = entry.get("knn_pretrained_imagenet_val")
        archs[cand][f"cka_random_{teacher}"] = entry.get("cka_random")
    
    return archs


def bootstrap_ci(x, y, n_boot=10000, ci=0.95):
    """Bootstrap CI for Spearman rho."""
    rhos = []
    n = len(x)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        rho, _ = stats.spearmanr(x[idx], y[idx])
        if not np.isnan(rho):
            rhos.append(rho)
    rhos = np.array(rhos)
    alpha = (1 - ci) / 2
    return np.percentile(rhos, [alpha * 100, (1 - alpha) * 100])


# ================================================================
# ANALYSIS 1: TRAJECTORY (init → pretrained sign flip)
# ================================================================
def analyze_trajectory(upgrades_data, v3_archs):
    """Analyze trajectory data showing CKA at init vs pretrained."""
    traj = upgrades_data.get("trajectory", {})
    if not traj:
        print("  [SKIP] No trajectory data")
        return None
    
    print("\n" + "=" * 60)
    print("TRAJECTORY ANALYSIS (init → pretrained CKA)")
    print("=" * 60)
    
    results = {}
    for name, entry in traj.items():
        if "error" in entry and entry.get("cka_pretrained") is None:
            continue
        results[name] = {
            "init_cka": entry.get("cka_init"),
            "pretrained_cka": entry.get("cka_pretrained"),
            "acc": entry.get("acc"),
            "family": entry.get("family", v3_archs.get(name, {}).get("family", "?")),
        }
    
    if not results:
        print("  No valid trajectory entries")
        return None
    
    names = list(results.keys())
    init_cka = [results[n]["init_cka"] for n in names]
    pre_cka = [results[n]["pretrained_cka"] for n in names]
    accs = [results[n]["acc"] for n in names]
    
    print(f"  Architectures with data: {len(names)}")
    for n in names:
        r = results[n]
        delta = "↓" if r["pretrained_cka"] < r["init_cka"] else "↑"
        print(f"    {n}: init={r['init_cka']:.4f} → pre={r['pretrained_cka']:.4f} {delta} (acc={r['acc']}%)")
    
    # Stats
    drops = sum(1 for n in names if results[n]["pretrained_cka"] < results[n]["init_cka"])
    print(f"\n  Sign flip (init > pretrained): {drops}/{len(names)}")
    
    rho_init, p_init = stats.spearmanr(init_cka, accs) if len(names) >= 5 else (np.nan, np.nan)
    rho_pre, p_pre = stats.spearmanr(pre_cka, accs) if len(names) >= 5 else (np.nan, np.nan)
    print(f"  Init CKA vs acc: ρ = {rho_init:.3f} (p = {p_init:.4f})")
    print(f"  Pretrained CKA vs acc: ρ = {rho_pre:.3f} (p = {p_pre:.4f})")
    
    # ── Figure: Trajectory sign flip ──
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    
    # Panel A: Paired bar chart
    ax = axes[0]
    x_pos = np.arange(len(names))
    w = 0.35
    short_names = [n.replace("efficientnetv2_", "effv2_").replace("efficientnet_", "eff_")
                   .replace("convnext_", "cnx_").replace("vit_base_patch16", "vit_b16")
                   .replace("swin_", "sw_").replace("mobilenetv3_", "mv3_")
                   .replace("resnet", "rn").replace("deit_", "deit") for n in names]
    bars1 = ax.bar(x_pos - w/2, init_cka, w, label="Random init", color="#4ECDC4", alpha=0.8)
    bars2 = ax.bar(x_pos + w/2, pre_cka, w, label="Pretrained", color="#FF6B6B", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("CKA with DINOv2")
    ax.set_title("A. CKA at init vs. pretrained")
    ax.legend(fontsize=7)
    
    # Panel B: Delta (init - pretrained)
    ax = axes[1]
    deltas = [results[n]["init_cka"] - results[n]["pretrained_cka"] for n in names]
    colors = ["#4ECDC4" if d > 0 else "#FF6B6B" for d in deltas]
    ax.bar(x_pos, deltas, color=colors)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("ΔCKA (init − pretrained)")
    ax.set_title(f"B. CKA drop during learning ({drops}/{len(names)} decrease)")
    
    # Panel C: Correlation scatter
    ax = axes[2]
    for n in names:
        fam = results[n]["family"]
        c = FAMILY_COLORS.get(fam, "#999999")
        ax.scatter(results[n]["init_cka"], results[n]["acc"], marker="^", 
                   color=c, s=40, alpha=0.7, edgecolors="k", linewidths=0.3)
        ax.scatter(results[n]["pretrained_cka"], results[n]["acc"], marker="o",
                   color=c, s=40, alpha=0.7, edgecolors="k", linewidths=0.3)
        ax.annotate("", xy=(results[n]["pretrained_cka"], results[n]["acc"]),
                     xytext=(results[n]["init_cka"], results[n]["acc"]),
                     arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    
    ax.scatter([], [], marker="^", color="gray", s=30, label=f"Init (ρ={rho_init:.2f})")
    ax.scatter([], [], marker="o", color="gray", s=30, label=f"Pretrained (ρ={rho_pre:.2f})")
    ax.set_xlabel("CKA with DINOv2")
    ax.set_ylabel("ImageNet Top-1 Acc (%)")
    ax.set_title("C. Learning inverts the correlation")
    ax.legend(fontsize=7, loc="lower left")
    
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig13_trajectory_signflip.{ext}")
    plt.close(fig)
    print(f"  → Saved fig13_trajectory_signflip.{{png,pdf}}")
    
    return {
        "n_archs": len(names),
        "n_drops": drops,
        "rho_init": float(rho_init),
        "p_init": float(p_init),
        "rho_pretrained": float(rho_pre),
        "p_pretrained": float(p_pre),
        "per_arch": results,
    }


# ================================================================
# ANALYSIS 2: NEW BASELINES (SNIP, GraSP)
# ================================================================
def analyze_baselines(upgrades_data, v3_archs):
    """Analyze SNIP and GraSP baseline correlations."""
    baselines = upgrades_data.get("baselines", {})
    if not baselines:
        print("  [SKIP] No baseline data")
        return None
    
    print("\n" + "=" * 60)
    print("EXPANDED BASELINES (SNIP, GraSP)")
    print("=" * 60)
    
    # Collect valid entries
    snip_data = [(v["snip"], v["acc"]) for v in baselines.values() 
                 if v.get("snip") is not None and np.isfinite(v["snip"])]
    grasp_data = [(v["grasp"], v["acc"]) for v in baselines.values()
                  if v.get("grasp") is not None and np.isfinite(v["grasp"])]
    
    results = {}
    for name, data_pairs in [("SNIP", snip_data), ("GraSP", grasp_data)]:
        if len(data_pairs) < 10:
            print(f"  {name}: only {len(data_pairs)} valid entries, skipping")
            continue
        scores, accs = zip(*data_pairs)
        scores, accs = np.array(scores), np.array(accs)
        rho, p = stats.spearmanr(scores, accs)
        ci = bootstrap_ci(scores, accs)
        print(f"  {name}: ρ = {rho:.4f} [{ci[0]:.3f}, {ci[1]:.3f}] (p = {p:.2e}, n = {len(scores)})")
        results[name] = {
            "rho": float(rho), "p": float(p), "n": len(scores),
            "ci_lower": float(ci[0]), "ci_upper": float(ci[1]),
        }
    
    # Merge with existing baselines for comparison table
    print("\n  Combined baseline summary:")
    print(f"  {'Proxy':<20} {'ρ':>8} {'95% CI':>18} {'n':>5}")
    print(f"  {'-'*55}")
    for name, r in results.items():
        print(f"  {name:<20} {r['rho']:>+8.4f} [{r['ci_lower']:>+.3f}, {r['ci_upper']:>+.3f}] {r['n']:>5}")
    
    return results


# ================================================================
# ANALYSIS 3: kNN SENSITIVITY
# ================================================================
def analyze_knn_sensitivity(upgrades_data):
    """Analyze kNN sensitivity to k, metric, and whitening."""
    knn_data = upgrades_data.get("knn_sensitivity", {})
    if not knn_data:
        print("  [SKIP] No kNN sensitivity data")
        return None
    
    print("\n" + "=" * 60)
    print("kNN SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    print(f"  {'Config':<25} {'ρ':>8} {'p':>12} {'n':>5}")
    print(f"  {'-'*55}")
    for config, entry in sorted(knn_data.items()):
        print(f"  {config:<25} {entry['rho']:>+8.4f} {entry['p']:>12.2e} {entry['n']:>5}")
    
    # ── Figure: kNN sensitivity curves ──
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    
    # Panel A: ρ vs k for cosine vs euclidean
    ax = axes[0]
    for metric, color, label in [("cosine", "#1f77b4", "Cosine"), ("euclidean", "#ff7f0e", "Euclidean")]:
        ks, rhos = [], []
        for config, entry in knn_data.items():
            if config.startswith(metric) and "whitened" not in config:
                ks.append(entry["k"])
                rhos.append(entry["rho"])
        if ks:
            order = np.argsort(ks)
            ks, rhos = np.array(ks)[order], np.array(rhos)[order]
            ax.plot(ks, rhos, "-o", color=color, label=label, markersize=4)
    
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("k (number of neighbors)")
    ax.set_ylabel("Spearman ρ (kNN vs. accuracy)")
    ax.set_title("A. kNN metric comparison")
    ax.legend()
    
    # Panel B: Whitened vs unwhitened
    ax = axes[1]
    cosine_ks, cosine_rhos = [], []
    whitened_ks, whitened_rhos = [], []
    for config, entry in knn_data.items():
        if config.startswith("cosine_k"):
            cosine_ks.append(entry["k"])
            cosine_rhos.append(entry["rho"])
        elif config.startswith("whitened"):
            whitened_ks.append(entry["k"])
            whitened_rhos.append(entry["rho"])
    
    if cosine_ks:
        order = np.argsort(cosine_ks)
        ax.plot(np.array(cosine_ks)[order], np.array(cosine_rhos)[order], 
                "-o", color="#1f77b4", label="Standard", markersize=4)
    if whitened_ks:
        order = np.argsort(whitened_ks)
        ax.plot(np.array(whitened_ks)[order], np.array(whitened_rhos)[order],
                "-s", color="#2ca02c", label="Whitened", markersize=4)
    
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("k")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("B. Effect of whitening")
    ax.legend()
    
    # Panel C: |ρ| heatmap-like bar chart for all configs
    ax = axes[2]
    configs = sorted(knn_data.keys())
    abs_rhos = [abs(knn_data[c]["rho"]) for c in configs]
    short_labels = [c.replace("cosine_", "cos ").replace("euclidean_", "euc ")
                     .replace("whitened_cosine_", "wh ") for c in configs]
    colors_bar = ["#1f77b4" if "cos" in c and "whitened" not in c 
                  else "#ff7f0e" if "euc" in c 
                  else "#2ca02c" for c in configs]
    ax.barh(range(len(configs)), abs_rhos, color=colors_bar, alpha=0.8)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(short_labels, fontsize=6)
    ax.set_xlabel("|Spearman ρ|")
    ax.set_title("C. All configurations")
    ax.invert_yaxis()
    
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig14_knn_sensitivity.{ext}")
    plt.close(fig)
    print(f"\n  → Saved fig14_knn_sensitivity.{{png,pdf}}")
    
    return knn_data


# ================================================================
# ANALYSIS 4: LAYER-WISE CKA
# ================================================================
def analyze_layerwise(upgrades_data, v3_archs):
    """Analyze layer-wise CKA at different depth quartiles."""
    layerwise = upgrades_data.get("layerwise", {})
    if not layerwise:
        print("  [SKIP] No layer-wise data")
        return None
    
    print("\n" + "=" * 60)
    print("LAYER-WISE CKA ANALYSIS")
    print("=" * 60)
    
    # Collect architectures with at least 2 layer CKA values
    valid = {}
    for name, entry in layerwise.items():
        layers = entry.get("layers", {})
        if len(layers) >= 2:
            valid[name] = {
                "layers": layers,
                "acc": entry["acc"],
                "family": entry.get("family", v3_archs.get(name, {}).get("family", "?")),
            }
    
    print(f"  Architectures with layer-wise data: {len(valid)}")
    
    if len(valid) < 10:
        print("  Too few architectures, skipping detailed analysis")
        return None
    
    # For each architecture, compute early vs late CKA
    # Assign layers to quartiles by position
    early_ckas, late_ckas, accs, families = [], [], [], []
    for name, entry in valid.items():
        layer_names = list(entry["layers"].keys())
        layer_vals = list(entry["layers"].values())
        if len(layer_vals) < 2:
            continue
        early_ckas.append(layer_vals[0])  # First layer
        late_ckas.append(layer_vals[-1])  # Last layer
        accs.append(entry["acc"])
        families.append(entry["family"])
    
    early_ckas = np.array(early_ckas)
    late_ckas = np.array(late_ckas)
    accs = np.array(accs)
    
    rho_early, p_early = stats.spearmanr(early_ckas, accs)
    rho_late, p_late = stats.spearmanr(late_ckas, accs)
    
    print(f"  Early-layer CKA vs acc: ρ = {rho_early:.4f} (p = {p_early:.2e})")
    print(f"  Late-layer CKA vs acc:  ρ = {rho_late:.4f} (p = {p_late:.2e})")
    
    # ── Figure: Layer-wise analysis ──
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    
    # Panel A: Early vs late CKA scatter
    ax = axes[0]
    for e, l, a, fam in zip(early_ckas, late_ckas, accs, families):
        c = FAMILY_COLORS.get(fam, "#999999")
        ax.scatter(e, l, c=c, s=20, alpha=0.6, edgecolors="k", linewidths=0.3)
    ax.plot([0, max(early_ckas.max(), late_ckas.max())], 
            [0, max(early_ckas.max(), late_ckas.max())], "k--", lw=0.5, alpha=0.3)
    ax.set_xlabel("Early-layer CKA")
    ax.set_ylabel("Late-layer CKA")
    ax.set_title("A. Early vs. late CKA")
    
    # Panel B: Scatter of early CKA vs acc
    ax = axes[1]
    for e, a, fam in zip(early_ckas, accs, families):
        c = FAMILY_COLORS.get(fam, "#999999")
        ax.scatter(e, a, c=c, s=20, alpha=0.6, edgecolors="k", linewidths=0.3)
    ax.set_xlabel("Early-layer CKA")
    ax.set_ylabel("ImageNet Acc (%)")
    ax.set_title(f"B. Early CKA (ρ = {rho_early:.3f})")
    
    # Panel C: Scatter of late CKA vs acc
    ax = axes[2]
    for l, a, fam in zip(late_ckas, accs, families):
        c = FAMILY_COLORS.get(fam, "#999999")
        ax.scatter(l, a, c=c, s=20, alpha=0.6, edgecolors="k", linewidths=0.3)
    ax.set_xlabel("Late-layer CKA")
    ax.set_ylabel("ImageNet Acc (%)")
    ax.set_title(f"C. Late CKA (ρ = {rho_late:.3f})")
    
    # Add family legend
    handles = []
    for fam in sorted(set(families)):
        c = FAMILY_COLORS.get(fam, "#999999")
        handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, 
                                  markersize=5, label=fam))
    axes[2].legend(handles=handles, fontsize=5, ncol=2, loc="lower left",
                   framealpha=0.8, handletextpad=0.3, columnspacing=0.5)
    
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig15_layerwise_cka.{ext}")
    plt.close(fig)
    print(f"  → Saved fig15_layerwise_cka.{{png,pdf}}")
    
    return {
        "n_valid": len(valid),
        "rho_early": float(rho_early), "p_early": float(p_early),
        "rho_late": float(rho_late), "p_late": float(p_late),
    }


# ================================================================
# ANALYSIS 5: TRANSFER BENCHMARKS
# ================================================================
def analyze_transfer(transfer_data, v3_archs):
    """Analyze transfer benchmark correlations."""
    transfer = transfer_data.get("transfer", {})
    if not transfer:
        print("  [SKIP] No transfer data")
        return None
    
    print("\n" + "=" * 60)
    print("TRANSFER BENCHMARK ANALYSIS")
    print("=" * 60)
    
    # Collect valid CIFAR-100 and Flowers entries
    cifar_valid = [(name, v) for name, v in transfer.items() 
                   if v.get("cifar100_acc") is not None]
    flowers_valid = [(name, v) for name, v in transfer.items()
                     if v.get("flowers102_acc") is not None]
    
    print(f"  CIFAR-100 valid: {len(cifar_valid)}/{len(transfer)}")
    print(f"  Flowers-102 valid: {len(flowers_valid)}/{len(transfer)}")
    
    results = {}
    
    for dataset_name, valid_entries in [("CIFAR-100", cifar_valid), ("Flowers-102", flowers_valid)]:
        if len(valid_entries) < 10:
            print(f"  {dataset_name}: too few entries")
            continue
        
        names = [n for n, _ in valid_entries]
        acc_key = "cifar100_acc" if "CIFAR" in dataset_name else "flowers102_acc"
        transfer_accs = np.array([v[acc_key] for _, v in valid_entries])
        imagenet_accs = np.array([v["imagenet_acc"] for _, v in valid_entries])
        
        # Correlate transfer acc with RepNAS proxy scores
        # Get CKA noise scores from v3 for these architectures
        cka_noise_scores = []
        knn_natural_scores = []
        valid_names_cka = []
        valid_names_knn = []
        
        for name in names:
            arch = v3_archs.get(name, {})
            # DINOv2 CKA noise
            cka = arch.get("cka_noise_dinov2_small")
            if cka is not None:
                cka_noise_scores.append(cka)
                valid_names_cka.append(name)
            # MAE kNN natural
            knn = arch.get("knn_natural_mae_base")
            if knn is not None:
                knn_natural_scores.append(knn)
                valid_names_knn.append(name)
        
        # Transfer acc ↔ ImageNet acc
        rho_in, p_in = stats.spearmanr(imagenet_accs, transfer_accs)
        print(f"\n  {dataset_name}:")
        print(f"    ImageNet ↔ {dataset_name}: ρ = {rho_in:.4f} (p = {p_in:.2e})")
        
        # RepNAS proxy ↔ Transfer acc
        if len(valid_names_cka) >= 10:
            # Match transfer accs to CKA names
            t_accs_cka = [transfer[n][acc_key] for n in valid_names_cka]
            rho_cka, p_cka = stats.spearmanr(cka_noise_scores, t_accs_cka)
            ci_cka = bootstrap_ci(np.array(cka_noise_scores), np.array(t_accs_cka))
            print(f"    DINOv2 CKA noise ↔ {dataset_name}: ρ = {rho_cka:.4f} [{ci_cka[0]:.3f}, {ci_cka[1]:.3f}]")
            results[f"{dataset_name}_cka_noise"] = {
                "rho": float(rho_cka), "p": float(p_cka),
                "ci": [float(ci_cka[0]), float(ci_cka[1])], "n": len(valid_names_cka),
            }
        
        results[f"{dataset_name}_imagenet"] = {
            "rho": float(rho_in), "p": float(p_in), "n": len(valid_entries),
        }
        results[f"{dataset_name}_accs"] = {
            "names": names,
            "transfer_accs": transfer_accs.tolist(),
            "imagenet_accs": imagenet_accs.tolist(),
        }
    
    # ── Figure: Transfer correlation scatter ──
    n_panels = sum(1 for ds in ["CIFAR-100", "Flowers-102"] 
                   if f"{ds}_accs" in results)
    if n_panels == 0:
        return results
    
    fig, axes = plt.subplots(1, min(n_panels * 2, 4), figsize=(min(n_panels * 6, 12), 3.2))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    panel_idx = 0
    for dataset_name in ["CIFAR-100", "Flowers-102"]:
        accs_key = f"{dataset_name}_accs"
        if accs_key not in results:
            continue
        
        names = results[accs_key]["names"]
        t_accs = np.array(results[accs_key]["transfer_accs"])
        in_accs = np.array(results[accs_key]["imagenet_accs"])
        
        # Panel: ImageNet acc vs Transfer acc
        if panel_idx < len(axes):
            ax = axes[panel_idx]
            for name, ta, ia in zip(names, t_accs, in_accs):
                fam = v3_archs.get(name, {}).get("family", "?")
                c = FAMILY_COLORS.get(fam, "#999999")
                ax.scatter(ia, ta, c=c, s=15, alpha=0.6, edgecolors="k", linewidths=0.3)
            rho_val = results[f"{dataset_name}_imagenet"]["rho"]
            ax.set_xlabel("ImageNet Top-1 Acc (%)")
            ax.set_ylabel(f"{dataset_name} Acc (%)")
            ax.set_title(f"ImageNet ↔ {dataset_name} (ρ={rho_val:.3f})")
            panel_idx += 1
        
        # Panel: CKA noise vs Transfer acc
        cka_key = f"{dataset_name}_cka_noise"
        if cka_key in results and panel_idx < len(axes):
            ax = axes[panel_idx]
            valid_names_cka = [n for n in names if v3_archs.get(n, {}).get("cka_noise_dinov2_small") is not None]
            for name in valid_names_cka:
                fam = v3_archs.get(name, {}).get("family", "?")
                c = FAMILY_COLORS.get(fam, "#999999")
                cka_val = v3_archs[name]["cka_noise_dinov2_small"]
                ta = transfer[name]["cifar100_acc" if "CIFAR" in dataset_name else "flowers102_acc"]
                if ta is not None:
                    ax.scatter(cka_val, ta, c=c, s=15, alpha=0.6, edgecolors="k", linewidths=0.3)
            rho_val = results[cka_key]["rho"]
            ci = results[cka_key]["ci"]
            ax.set_xlabel("DINOv2 CKA (noise probes)")
            ax.set_ylabel(f"{dataset_name} Acc (%)")
            ax.set_title(f"CKA ↔ {dataset_name} (ρ={rho_val:.3f} [{ci[0]:.2f},{ci[1]:.2f}])")
            panel_idx += 1
    
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig16_transfer_correlation.{ext}")
    plt.close(fig)
    print(f"\n  → Saved fig16_transfer_correlation.{{png,pdf}}")
    
    return results


# ================================================================
# ANALYSIS 6: PROXY ENSEMBLE
# ================================================================
def analyze_ensemble(upgrades_data, v3_archs):
    """Try linear combinations of inverted CKA + other proxies."""
    baselines = upgrades_data.get("baselines", {})
    if not baselines:
        print("  [SKIP] No baseline data for ensemble")
        return None
    
    print("\n" + "=" * 60)
    print("PROXY ENSEMBLE ANALYSIS")
    print("=" * 60)
    
    # Collect architectures with all scores
    ensemble_data = {}
    for name, arch in v3_archs.items():
        cka_noise = arch.get("cka_noise_dinov2_small")
        if cka_noise is None:
            continue
        
        bl = baselines.get(name, {})
        snip = bl.get("snip")
        grasp = bl.get("grasp")
        
        if snip is not None and grasp is not None and np.isfinite(snip) and np.isfinite(grasp):
            ensemble_data[name] = {
                "cka_noise": cka_noise,
                "snip": snip,
                "grasp": grasp,
                "acc": arch["acc"],
                "params": arch["params"],
                "log_params": np.log10(arch["params"]),
            }
    
    if len(ensemble_data) < 20:
        print(f"  Only {len(ensemble_data)} archs with complete data, too few for ensemble")
        return None
    
    names = list(ensemble_data.keys())
    accs = np.array([ensemble_data[n]["acc"] for n in names])
    cka = np.array([ensemble_data[n]["cka_noise"] for n in names])
    snip = np.array([ensemble_data[n]["snip"] for n in names])
    grasp = np.array([ensemble_data[n]["grasp"] for n in names])
    log_params = np.array([ensemble_data[n]["log_params"] for n in names])
    
    # Standardize
    def zscore(x):
        return (x - x.mean()) / (x.std() + 1e-10)
    
    cka_z = zscore(cka)
    snip_z = zscore(snip)
    grasp_z = zscore(grasp)
    size_z = zscore(log_params)
    
    # Try combinations (invert CKA = multiply by -1)
    inv_cka_z = -cka_z
    
    combos = {
        "Inverted CKA only": inv_cka_z,
        "SNIP only": snip_z,
        "GraSP only": grasp_z,
        "log(params) only": size_z,
        "InvCKA + SNIP": inv_cka_z + snip_z,
        "InvCKA + GraSP": inv_cka_z + grasp_z,
        "InvCKA + size": inv_cka_z + size_z,
        "InvCKA + SNIP + size": inv_cka_z + snip_z + size_z,
        "InvCKA + GraSP + size": inv_cka_z + grasp_z + size_z,
        "AZ-NAS style (all 4)": inv_cka_z + snip_z + grasp_z + size_z,
    }
    
    print(f"\n  Ensemble analysis (n = {len(names)}):")
    print(f"  {'Combination':<30} {'ρ':>8} {'Best@10':>10} {'Regret%':>10}")
    print(f"  {'-'*60}")
    
    ensemble_results = {}
    for combo_name, combo_score in combos.items():
        rho, p = stats.spearmanr(combo_score, accs)
        
        # Search simulation
        top10_idx = np.argsort(-combo_score)[:10]  # Higher = predicted better
        best_found = accs[top10_idx].max()
        oracle_best = accs.max()
        regret = 100 * (oracle_best - best_found) / oracle_best
        
        print(f"  {combo_name:<30} {rho:>+8.4f} {best_found:>9.1f}% {regret:>9.2f}%")
        ensemble_results[combo_name] = {
            "rho": float(rho), "p": float(p),
            "best_found": float(best_found), "regret": float(regret),
        }
    
    # ── Figure: Ensemble comparison ──
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    combo_names = list(ensemble_results.keys())
    rhos = [ensemble_results[c]["rho"] for c in combo_names]
    regrets = [ensemble_results[c]["regret"] for c in combo_names]
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(combo_names)))
    bars = ax.barh(range(len(combo_names)), rhos, color=colors, alpha=0.8)
    ax.set_yticks(range(len(combo_names)))
    ax.set_yticklabels(combo_names, fontsize=7)
    ax.set_xlabel("Spearman ρ (proxy vs. accuracy)")
    ax.set_title(f"Proxy Ensemble Comparison (n = {len(names)})")
    ax.axvline(0, color="k", lw=0.5)
    
    # Add regret annotations
    for i, (r, reg) in enumerate(zip(rhos, regrets)):
        ax.annotate(f"{reg:.1f}% regret", xy=(r, i), fontsize=6, va="center",
                    ha="left" if r > 0 else "right", 
                    xytext=(3 if r > 0 else -3, 0), textcoords="offset points")
    
    ax.invert_yaxis()
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig17_proxy_ensemble.{ext}")
    plt.close(fig)
    print(f"\n  → Saved fig17_proxy_ensemble.{{png,pdf}}")
    
    return ensemble_results


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 70)
    print("RepNAS: Analyzing New Experiments")
    print("=" * 70)
    
    # Load existing v3 data
    v3_archs = load_v3()
    print(f"Loaded v3 data: {len(v3_archs)} architectures")
    
    # Load upgrades data
    upgrades_path = EXP_DIR / "repnas_upgrades.json"
    upgrades_data = {}
    if upgrades_path.exists():
        with open(upgrades_path) as f:
            upgrades_data = json.load(f)
        print(f"Loaded upgrades data: keys = {list(upgrades_data.keys())}")
    else:
        print(f"WARNING: {upgrades_path} not found. Skipping upgrade analyses.")
    
    # Load transfer data
    transfer_path = EXP_DIR / "repnas_transfer.json"
    transfer_data = {}
    if transfer_path.exists():
        with open(transfer_path) as f:
            transfer_data = json.load(f)
        print(f"Loaded transfer data: keys = {list(transfer_data.keys())}")
    else:
        print(f"WARNING: {transfer_path} not found. Skipping transfer analysis.")
    
    # Run all analyses
    all_results = {}
    
    if upgrades_data:
        traj = analyze_trajectory(upgrades_data, v3_archs)
        if traj:
            all_results["trajectory"] = traj
        
        baselines = analyze_baselines(upgrades_data, v3_archs)
        if baselines:
            all_results["baselines"] = baselines
        
        knn = analyze_knn_sensitivity(upgrades_data)
        if knn:
            all_results["knn_sensitivity"] = knn
        
        layerwise = analyze_layerwise(upgrades_data, v3_archs)
        if layerwise:
            all_results["layerwise"] = layerwise
        
        ensemble = analyze_ensemble(upgrades_data, v3_archs)
        if ensemble:
            all_results["ensemble"] = ensemble
    
    if transfer_data:
        transfer = analyze_transfer(transfer_data, v3_archs)
        if transfer:
            all_results["transfer"] = transfer
    
    # Save combined analysis
    out_path = EXP_DIR / "new_experiments_analysis.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'=' * 70}")
    print(f"Analysis complete. Results saved to {out_path}")
    print(f"Figures saved to {FIG_DIR}/fig13-fig17*.{{png,pdf}}")
    print(f"{'=' * 70}")
    
    return all_results


if __name__ == "__main__":
    main()
