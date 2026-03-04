"""
RepNAS v4: Updated Publication Figures with Bootstrap CIs, Search Simulation, and Family Analysis
=================================================================================================
Regenerates all figures + adds new figures from advanced analysis.
"""

import json
import math
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ============================================================
# Load data
# ============================================================
with open("/Users/aayambansal/Desktop/VStudio/#mas-2/1/zero-shot-nas-paper/experiments/repnas_v3_results.json") as f:
    data = json.load(f)

with open("/Users/aayambansal/Desktop/VStudio/#mas-2/1/zero-shot-nas-paper/experiments/advanced_analysis.json") as f:
    adv = json.load(f)

results = data["results"]
rows = []
for key, val in results.items():
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

FIGDIR = "/Users/aayambansal/Desktop/VStudio/#mas-2/1/zero-shot-nas-paper/figures"

# Color map for families
FAMILY_COLORS = {
    "CNN": "#1f77b4", "EffNet": "#ff7f0e", "Mobile": "#2ca02c",
    "ConvNeXt": "#d62728", "ViT": "#9467bd", "Swin": "#8c564b",
    "DeiT": "#e377c2", "RegNet": "#7f7f7f", "MaxViT": "#bcbd22",
    "EFormer": "#17becf", "EdgeNeXt": "#ff6347", "MetaFormer": "#6b8e23",
    "NAS": "#daa520", "CaiT": "#4169e1", "Mixer": "#ff1493",
}

FAMILY_MARKERS = {
    "CNN": "o", "EffNet": "s", "Mobile": "^", "ConvNeXt": "D",
    "ViT": "P", "Swin": "*", "DeiT": "X", "RegNet": "v",
    "MaxViT": "p", "EFormer": "h", "EdgeNeXt": ">", "MetaFormer": "<",
    "NAS": "H", "CaiT": "d", "Mixer": "+",
}

teachers = ["dinov2_small", "clip_vit_b32", "convnext_base_fcmae", "mae_vit_base"]
teacher_labels = ["DINOv2 ViT-S/14", "CLIP ViT-B/32", "ConvNeXtV2-B (FCMAE)", "MAE ViT-B/16"]
teacher_short = ["DINOv2", "CLIP", "ConvNeXtV2", "MAE"]

bootstrap = adv["bootstrap_results"]
search = adv["search_results"]

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# ============================================================
# Helper: get bootstrap CI for a teacher__col key
# ============================================================
def get_ci(teacher, col):
    key = f"{teacher}__{col}"
    if key in bootstrap:
        b = bootstrap[key]
        return b["mean_rho"], b["ci_low"], b["ci_high"]
    return None, None, None


# ============================================================
# FIGURE 1: Main scatter — CKA vs Accuracy (all 4 teachers, 2x2) [UNCHANGED]
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
for idx, (teacher, label) in enumerate(zip(teachers, teacher_labels)):
    ax = axes[idx // 2][idx % 2]
    tdf = df[df["teacher"] == teacher].dropna(subset=["cka_pretrained_noise", "gt_acc"])
    
    for family in tdf["family"].unique():
        fdf = tdf[tdf["family"] == family]
        ax.scatter(fdf["cka_pretrained_noise"], fdf["gt_acc"],
                  c=FAMILY_COLORS.get(family, "#999999"),
                  marker=FAMILY_MARKERS.get(family, "o"),
                  s=35, alpha=0.75, edgecolors='white', linewidth=0.3,
                  label=family)
    
    rho, p = stats.spearmanr(tdf["cka_pretrained_noise"], tdf["gt_acc"])
    _, ci_lo, ci_hi = get_ci(teacher, "cka_pretrained_noise")
    z = np.polyfit(tdf["cka_pretrained_noise"], tdf["gt_acc"], 1)
    xline = np.linspace(tdf["cka_pretrained_noise"].min(), tdf["cka_pretrained_noise"].max(), 100)
    ax.plot(xline, np.polyval(z, xline), 'k--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel("CKA (pretrained, noise probes)")
    ax.set_ylabel("ImageNet-1k Accuracy (%)")
    ci_str = f" [{ci_lo:.2f}, {ci_hi:.2f}]" if ci_lo is not None else ""
    ax.set_title(f"{label}\n" + r"$\rho$ = " + f"{rho:.3f}{ci_str}")
    
    if idx == 0:
        handles = [Line2D([0], [0], marker=FAMILY_MARKERS.get(f, 'o'), color='w',
                         markerfacecolor=FAMILY_COLORS.get(f, '#999'), markersize=6, label=f)
                  for f in sorted(tdf["family"].unique())]
        ax.legend(handles=handles, loc='upper right', ncol=2, framealpha=0.8, fontsize=6)

plt.suptitle("Alignment Inversion: CKA vs. Accuracy Across Teachers (n=83, noise probes)", 
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig1_alignment_inversion_4teachers.png")
plt.savefig(f"{FIGDIR}/fig1_alignment_inversion_4teachers.pdf")
plt.close()
print("Figure 1: Alignment inversion (4 teachers) saved")


# ============================================================
# FIGURE 2: Probe ablation with Bootstrap CIs [UPDATED]
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# Panel A: CKA ρ by probe type and teacher WITH ERROR BARS
probe_types = ["noise", "imagenet_val", "augmented"]
probe_labels = ["Noise", "ImageNet-val\n(CIFAR-100)", "Augmented\n(gratings)"]
bar_width = 0.18
x = np.arange(len(teachers))

for pidx, (ptype, plabel) in enumerate(zip(probe_types, probe_labels)):
    rhos = []
    errors_lo = []
    errors_hi = []
    for teacher in teachers:
        col = f"cka_pretrained_{ptype}"
        mean_rho, ci_lo, ci_hi = get_ci(teacher, col)
        if mean_rho is not None:
            rhos.append(mean_rho)
            errors_lo.append(abs(mean_rho - ci_lo))
            errors_hi.append(abs(ci_hi - mean_rho))
        else:
            tdf = df[df["teacher"] == teacher]
            valid = tdf.dropna(subset=[col, "gt_acc"])
            rho, _ = stats.spearmanr(valid[col], valid["gt_acc"])
            rhos.append(rho)
            errors_lo.append(0)
            errors_hi.append(0)
    
    axes[0].bar(x + pidx * bar_width, rhos, bar_width, 
                yerr=[errors_lo, errors_hi], capsize=2, label=plabel, alpha=0.85,
                error_kw=dict(linewidth=0.8))

# Add random-init
rhos_rand = []
for teacher in teachers:
    _, ci_lo, ci_hi = get_ci(teacher, "cka_random")
    tdf = df[df["teacher"] == teacher]
    valid = tdf.dropna(subset=["cka_random", "gt_acc"])
    rho, _ = stats.spearmanr(valid["cka_random"], valid["gt_acc"])
    rhos_rand.append(rho)
axes[0].bar(x + 3 * bar_width, rhos_rand, bar_width, label="Random-init", alpha=0.85, color='#555555')

axes[0].set_xticks(x + 1.5 * bar_width)
axes[0].set_xticklabels(teacher_short, fontsize=9)
axes[0].set_ylabel(r"Spearman $\rho$ (CKA vs. Accuracy)")
axes[0].set_title("A. CKA Correlation by Probe Type (95% CI)")
axes[0].axhline(y=0, color='black', linewidth=0.5)
axes[0].legend(fontsize=8, loc='lower left')
axes[0].set_ylim(-0.75, 0.55)

# Panel B: All proxies comparison (DINOv2) WITH CIs
dino_df = df[df["teacher"] == "dinov2_small"].copy()
metric_names = ["CKA\nnoise", "CKA\nimgval", "kNN\nnoise", "kNN\nimgval",
                "CKA\nrandom", "Grad\nNorm", "NAS\nWOT", "Syn\nFlow"]
metric_keys = [
    ("dinov2_small", "cka_pretrained_noise"),
    ("dinov2_small", "cka_pretrained_imagenet_val"),
    ("dinov2_small", "knn_pretrained_noise"),
    ("dinov2_small", "knn_pretrained_imagenet_val"),
    ("dinov2_small", "cka_random"),
    ("dinov2_small", "gradnorm"),
    ("dinov2_small", "naswot"),
    ("dinov2_small", "synflow"),
]

rhos = []
errors_lo = []
errors_hi = []
for teacher, col in metric_keys:
    mean_rho, ci_lo, ci_hi = get_ci(teacher, col)
    if mean_rho is not None:
        rhos.append(mean_rho)
        errors_lo.append(abs(mean_rho - ci_lo))
        errors_hi.append(abs(ci_hi - mean_rho))
    else:
        rhos.append(0)
        errors_lo.append(0)
        errors_hi.append(0)

colors = ['#d62728' if r < -0.1 else '#2ca02c' if r > 0.1 else '#999999' for r in rhos]
bars = axes[1].bar(range(len(metric_names)), rhos, color=colors, alpha=0.85,
                   yerr=[errors_lo, errors_hi], capsize=2, error_kw=dict(linewidth=0.8))
axes[1].set_xticks(range(len(metric_names)))
axes[1].set_xticklabels(metric_names, fontsize=8, rotation=0)
axes[1].set_ylabel(r"Spearman $\rho$ with Accuracy")
axes[1].set_title("B. All Proxies (DINOv2, 95% CI)")
axes[1].axhline(y=0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig2_probe_ablation.png")
plt.savefig(f"{FIGDIR}/fig2_probe_ablation.pdf")
plt.close()
print("Figure 2: Probe ablation with CIs saved")


# ============================================================
# FIGURE 3: Partial correlation heatmap [UNCHANGED]
# ============================================================
import pingouin as pg

fig, ax = plt.subplots(1, 1, figsize=(7, 5))

partial_data = []
for teacher in teachers:
    tdf = df[df["teacher"] == teacher].copy()
    tdf["log_params"] = np.log(tdf["params"])
    row = {"Teacher": teacher.replace("_", " ").title()}
    
    for ptype in ["noise", "imagenet_val", "augmented"]:
        col = f"cka_pretrained_{ptype}"
        valid = tdf.dropna(subset=[col, "gt_acc", "log_params"])
        if len(valid) > 10:
            result = pg.partial_corr(data=valid, x=col, y="gt_acc", covar="log_params", method="spearman")
            row[f"CKA {ptype}"] = result["r"].values[0]
        else:
            row[f"CKA {ptype}"] = np.nan
    
    valid = tdf.dropna(subset=["cka_random", "gt_acc", "log_params"])
    if len(valid) > 10:
        result = pg.partial_corr(data=valid, x="cka_random", y="gt_acc", covar="log_params", method="spearman")
        row["CKA random"] = result["r"].values[0]
    
    partial_data.append(row)

dino_tdf = df[df["teacher"] == "dinov2_small"].copy()
dino_tdf["log_params"] = np.log(dino_tdf["params"])
baseline_row = {"Teacher": "Baselines (DINOv2)"}
for bl in ["gradnorm", "naswot", "synflow"]:
    valid = dino_tdf.dropna(subset=[bl, "gt_acc", "log_params"])
    if len(valid) > 10:
        result = pg.partial_corr(data=valid, x=bl, y="gt_acc", covar="log_params", method="spearman")
        baseline_row[bl.capitalize()] = result["r"].values[0]
partial_data.append(baseline_row)

pdf = pd.DataFrame(partial_data).set_index("Teacher")
heatmap_data = pdf.fillna(0).values
xlabels = pdf.columns.tolist()
ylabels = pdf.index.tolist()

im = ax.imshow(heatmap_data, cmap='RdBu_r', vmin=-0.7, vmax=0.7, aspect='auto')
ax.set_xticks(range(len(xlabels)))
ax.set_xticklabels(xlabels, rotation=35, ha='right', fontsize=9)
ax.set_yticks(range(len(ylabels)))
ax.set_yticklabels(ylabels, fontsize=9)

for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        val = heatmap_data[i, j]
        if val != 0:
            color = 'white' if abs(val) > 0.4 else 'black'
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', color=color, fontsize=9)

plt.colorbar(im, ax=ax, label=r"Partial $\rho$ (controlling for log(params))")
ax.set_title("Partial Spearman Correlations (controlling for model size)", fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig3_partial_correlation_heatmap.png")
plt.savefig(f"{FIGDIR}/fig3_partial_correlation_heatmap.pdf")
plt.close()
print("Figure 3: Partial correlation heatmap saved")


# ============================================================
# FIGURE 4: Top-k analysis [UNCHANGED]
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

for teacher, color, label in zip(teachers,
    ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"], teacher_short):
    tdf = df[df["teacher"] == teacher].dropna(subset=["cka_pretrained_noise", "gt_acc"])
    sorted_by_acc = tdf.sort_values("gt_acc", ascending=False)
    
    ks = range(3, min(40, len(tdf)))
    hit_rates, regrets = [], []
    
    for k in ks:
        oracle = set(sorted_by_acc.head(k)["candidate"].values)
        oracle_mean = sorted_by_acc.head(k)["gt_acc"].mean()
        sorted_proxy = tdf.sort_values("cka_pretrained_noise", ascending=True)
        proxy_top = set(sorted_proxy.head(k)["candidate"].values)
        hit_rates.append(len(oracle & proxy_top) / k)
        regrets.append(oracle_mean - sorted_proxy.head(k)["gt_acc"].mean())
    
    axes[0].plot(list(ks), hit_rates, color=color, label=f"Inv-CKA ({label})", linewidth=1.5)
    axes[1].plot(list(ks), regrets, color=color, label=f"Inv-CKA ({label})", linewidth=1.5)

# GradNorm baseline
dino_valid = dino_df.dropna(subset=["gradnorm", "gt_acc"])
sorted_by_acc = dino_valid.sort_values("gt_acc", ascending=False)
ks = range(3, min(40, len(dino_valid)))
hit_gn, reg_gn = [], []
for k in ks:
    oracle = set(sorted_by_acc.head(k)["candidate"].values)
    oracle_mean = sorted_by_acc.head(k)["gt_acc"].mean()
    sorted_gn = dino_valid.sort_values("gradnorm", ascending=False)
    proxy_top = set(sorted_gn.head(k)["candidate"].values)
    hit_gn.append(len(oracle & proxy_top) / k)
    reg_gn.append(oracle_mean - sorted_gn.head(k)["gt_acc"].mean())

axes[0].plot(list(ks), hit_gn, 'k--', label="GradNorm", linewidth=1.5)
axes[1].plot(list(ks), reg_gn, 'k--', label="GradNorm", linewidth=1.5)

axes[0].axhline(y=0, color='gray', linewidth=0.3)
axes[1].axhline(y=0, color='gray', linewidth=0.3)

axes[0].set_xlabel("k (top-k selected)")
axes[0].set_ylabel("Hit Rate")
axes[0].set_title("A. Top-k Hit Rate")
axes[0].legend(fontsize=7)
axes[0].set_ylim(-0.05, 1.05)

axes[1].set_xlabel("k (top-k selected)")
axes[1].set_ylabel("Regret (%)")
axes[1].set_title("B. Top-k Regret")
axes[1].legend(fontsize=7)

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig4_topk_analysis.png")
plt.savefig(f"{FIGDIR}/fig4_topk_analysis.pdf")
plt.close()
print("Figure 4: Top-k analysis saved")


# ============================================================
# FIGURE 5: Best config + cross-teacher [UNCHANGED]
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

mae_df = df[df["teacher"] == "mae_vit_base"].dropna(subset=["cka_pretrained_imagenet_val", "gt_acc"])
for family in mae_df["family"].unique():
    fdf = mae_df[mae_df["family"] == family]
    axes[0].scatter(fdf["cka_pretrained_imagenet_val"], fdf["gt_acc"],
                   c=FAMILY_COLORS.get(family, "#999999"),
                   marker=FAMILY_MARKERS.get(family, "o"),
                   s=35, alpha=0.75, edgecolors='white', linewidth=0.3, label=family)

rho, p = stats.spearmanr(mae_df["cka_pretrained_imagenet_val"], mae_df["gt_acc"])
z = np.polyfit(mae_df["cka_pretrained_imagenet_val"], mae_df["gt_acc"], 1)
xline = np.linspace(mae_df["cka_pretrained_imagenet_val"].min(), mae_df["cka_pretrained_imagenet_val"].max(), 100)
axes[0].plot(xline, np.polyval(z, xline), 'k--', alpha=0.5, linewidth=1)
axes[0].set_xlabel("CKA (pretrained, ImageNet-val probes)")
axes[0].set_ylabel("ImageNet-1k Accuracy (%)")
axes[0].set_title(f"A. Best CKA Config: MAE + ImageNet-val\n" + r"$\rho$ = " + f"{rho:.3f} (p < 0.0001)")

handles = [Line2D([0], [0], marker=FAMILY_MARKERS.get(f, 'o'), color='w',
                 markerfacecolor=FAMILY_COLORS.get(f, '#999'), markersize=6, label=f)
          for f in sorted(mae_df["family"].unique())]
axes[0].legend(handles=handles, loc='upper right', ncol=2, framealpha=0.8, fontsize=6)

noise_rhos, imgval_rhos = [], []
for teacher in teachers:
    tdf = df[df["teacher"] == teacher]
    valid_n = tdf.dropna(subset=["cka_pretrained_noise", "gt_acc"])
    valid_i = tdf.dropna(subset=["cka_pretrained_imagenet_val", "gt_acc"])
    r_n, _ = stats.spearmanr(valid_n["cka_pretrained_noise"], valid_n["gt_acc"])
    r_i, _ = stats.spearmanr(valid_i["cka_pretrained_imagenet_val"], valid_i["gt_acc"])
    noise_rhos.append(r_n)
    imgval_rhos.append(r_i)

x = np.arange(len(teacher_short))
axes[1].bar(x - 0.15, noise_rhos, 0.3, label="Noise probes", color="#1f77b4", alpha=0.85)
axes[1].bar(x + 0.15, imgval_rhos, 0.3, label="ImageNet-val probes", color="#ff7f0e", alpha=0.85)
axes[1].axhline(y=0, color='black', linewidth=0.5)
axes[1].set_xticks(x)
axes[1].set_xticklabels(teacher_short)
axes[1].set_ylabel(r"Spearman $\rho$ (CKA vs. Accuracy)")
axes[1].set_title("B. Cross-Teacher CKA Correlation")
axes[1].legend(fontsize=9)
axes[1].set_ylim(-0.75, 0.15)

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig5_best_config_and_cross_teacher.png")
plt.savefig(f"{FIGDIR}/fig5_best_config_and_cross_teacher.pdf")
plt.close()
print("Figure 5: Best config + cross-teacher saved")


# ============================================================
# FIGURE 6: kNN MAE imagenet [UNCHANGED]
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(6, 5))

mae_knn = df[df["teacher"] == "mae_vit_base"].dropna(subset=["knn_pretrained_imagenet_val", "gt_acc"])
for family in mae_knn["family"].unique():
    fdf = mae_knn[mae_knn["family"] == family]
    ax.scatter(fdf["knn_pretrained_imagenet_val"], fdf["gt_acc"],
              c=FAMILY_COLORS.get(family, "#999999"),
              marker=FAMILY_MARKERS.get(family, "o"),
              s=40, alpha=0.75, edgecolors='white', linewidth=0.3, label=family)

rho, p = stats.spearmanr(mae_knn["knn_pretrained_imagenet_val"], mae_knn["gt_acc"])
z = np.polyfit(mae_knn["knn_pretrained_imagenet_val"], mae_knn["gt_acc"], 1)
xline = np.linspace(mae_knn["knn_pretrained_imagenet_val"].min(), mae_knn["knn_pretrained_imagenet_val"].max(), 100)
ax.plot(xline, np.polyval(z, xline), 'k--', alpha=0.5, linewidth=1)
ax.set_xlabel("Mutual kNN Agreement (pretrained, ImageNet-val probes)")
ax.set_ylabel("ImageNet-1k Accuracy (%)")
ax.set_title(f"MAE Teacher: kNN vs. Accuracy\n" + r"$\rho$ = " + f"{rho:.3f} (p < 0.0001)")

handles = [Line2D([0], [0], marker=FAMILY_MARKERS.get(f, 'o'), color='w',
                 markerfacecolor=FAMILY_COLORS.get(f, '#999'), markersize=6, label=f)
          for f in sorted(mae_knn["family"].unique())]
ax.legend(handles=handles, loc='upper right', ncol=2, framealpha=0.8, fontsize=7)

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig6_knn_mae_imagenet.png")
plt.savefig(f"{FIGDIR}/fig6_knn_mae_imagenet.pdf")
plt.close()
print("Figure 6: kNN MAE imagenet saved")


# ============================================================
# FIGURE 7: Random-init positive [UNCHANGED]
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(6, 5))

dino_rand = dino_df.dropna(subset=["cka_random", "gt_acc"])
for family in dino_rand["family"].unique():
    fdf = dino_rand[dino_rand["family"] == family]
    ax.scatter(fdf["cka_random"], fdf["gt_acc"],
              c=FAMILY_COLORS.get(family, "#999999"),
              marker=FAMILY_MARKERS.get(family, "o"),
              s=40, alpha=0.75, edgecolors='white', linewidth=0.3, label=family)

rho, p = stats.spearmanr(dino_rand["cka_random"], dino_rand["gt_acc"])
z = np.polyfit(dino_rand["cka_random"], dino_rand["gt_acc"], 1)
xline = np.linspace(dino_rand["cka_random"].min(), dino_rand["cka_random"].max(), 100)
ax.plot(xline, np.polyval(z, xline), 'k--', alpha=0.5, linewidth=1)
ax.set_xlabel("CKA (random-init, noise probes)")
ax.set_ylabel("ImageNet-1k Accuracy (%)")
ax.set_title(f"Random-Init CKA vs. Accuracy (DINOv2)\n" + r"$\rho$ = " + f"{rho:+.3f} (p = {p:.4f})")

handles = [Line2D([0], [0], marker=FAMILY_MARKERS.get(f, 'o'), color='w',
                 markerfacecolor=FAMILY_COLORS.get(f, '#999'), markersize=6, label=f)
          for f in sorted(dino_rand["family"].unique())]
ax.legend(handles=handles, loc='upper left', ncol=2, framealpha=0.8, fontsize=6)

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig7_random_init_positive.png")
plt.savefig(f"{FIGDIR}/fig7_random_init_positive.pdf")
plt.close()
print("Figure 7: Random-init positive saved")


# ============================================================
# FIGURE 8 (NEW): Proxy-Guided Search Simulation
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# Panel A: Best-found accuracy at budget=10 across proxies
proxy_configs = [
    ("MAE CKA-noise", "mae_vit_base__cka_pretrained_noise__budget10"),
    ("MAE CKA-imgval", "mae_vit_base__cka_pretrained_imagenet_val__budget10"),
    ("MAE kNN-imgval", "mae_vit_base__knn_pretrained_imagenet_val__budget10"),
    ("CLIP CKA-noise", "clip_vit_b32__cka_pretrained_noise__budget10"),
    ("CLIP CKA-imgval", "clip_vit_b32__cka_pretrained_imagenet_val__budget10"),
    ("ConvNeXt CKA-noise", "convnext_base_fcmae__cka_pretrained_noise__budget10"),
    ("DINOv2 CKA-noise", "dinov2_small__cka_pretrained_noise__budget10"),
    ("GradNorm", "GradNorm__budget10"),
    ("SynFlow", "SynFlow__budget10"),
]

labels_a, bests_a, rand_means, rand_stds = [], [], [], []
for label, key in proxy_configs:
    if key in search:
        s = search[key]
        labels_a.append(label)
        bests_a.append(s["proxy_best"])
        rand_means.append(s["random_best_mean"])
        rand_stds.append(s["random_best_std"])

x = np.arange(len(labels_a))
bar_colors = ['#d62728' if 'MAE' in l else '#ff7f0e' if 'CLIP' in l else 
              '#2ca02c' if 'ConvNeXt' in l else '#1f77b4' if 'DINOv2' in l else '#555555' 
              for l in labels_a]

axes[0].bar(x, bests_a, color=bar_colors, alpha=0.85, label="Proxy-guided (top-10)")
axes[0].errorbar(x, rand_means, yerr=rand_stds, fmt='ko', markersize=4, capsize=3, 
                label="Random (mean$\\pm$std)", linewidth=1)
axes[0].axhline(y=search.get("mae_vit_base__cka_pretrained_noise__budget10", {}).get("oracle_best", 85.8),
               color='green', linestyle='--', linewidth=1, alpha=0.7, label="Oracle best")
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels_a, rotation=45, ha='right', fontsize=7)
axes[0].set_ylabel("Best Accuracy Found (%)")
axes[0].set_title("A. Best Accuracy at Budget=10")
axes[0].legend(fontsize=7, loc='lower right')
axes[0].set_ylim(82, 86.5)

# Panel B: Hit rate for top-10 at different budgets
budgets = [5, 10, 15, 20]
best_proxies = [
    ("MAE CKA-noise", "mae_vit_base__cka_pretrained_noise", "#d62728"),
    ("MAE CKA-imgval", "mae_vit_base__cka_pretrained_imagenet_val", "#d62728"),
    ("CLIP CKA-noise", "clip_vit_b32__cka_pretrained_noise", "#ff7f0e"),
    ("ConvNeXt CKA-noise", "convnext_base_fcmae__cka_pretrained_noise", "#2ca02c"),
    ("GradNorm", "GradNorm", "#555555"),
]

for label, prefix, color in best_proxies:
    hits = []
    for b in budgets:
        key = f"{prefix}__budget{b}"
        if key in search:
            hits.append(search[key]["proxy_hit10"])
        else:
            hits.append(np.nan)
    ls = '--' if 'GradNorm' in label else '-'
    axes[1].plot(budgets, hits, f'{ls}o', color=color, label=label, linewidth=1.5, markersize=5)

# Random baseline hit rate
random_hits = []
for b in budgets:
    key = f"mae_vit_base__cka_pretrained_noise__budget{b}"
    if key in search:
        random_hits.append(search[key]["random_hit10_mean"])
random_line = axes[1].plot(budgets, random_hits, 'k:', label="Random", linewidth=1.5, markersize=5)

axes[1].set_xlabel("Search Budget (architectures evaluated)")
axes[1].set_ylabel("Hit Rate (fraction of true top-10)")
axes[1].set_title("B. Search Hit Rate vs. Budget")
axes[1].legend(fontsize=6.5, loc='upper left')
axes[1].set_ylim(-0.05, 0.65)

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig8_search_simulation.png")
plt.savefig(f"{FIGDIR}/fig8_search_simulation.pdf")
plt.close()
print("Figure 8: Search simulation saved")


# ============================================================
# FIGURE 9 (NEW): Between-Family vs Within-Family Decomposition
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# Panel A: Between-family scatter (family means)
dino_all = df[df["teacher"] == "dinov2_small"].dropna(subset=["cka_pretrained_noise", "gt_acc"])
family_agg = dino_all.groupby("family").agg({
    "gt_acc": "mean",
    "cka_pretrained_noise": "mean",
    "params": ["mean", "count"],
}).reset_index()
family_agg.columns = ["family", "acc_mean", "cka_mean", "params_mean", "n"]
family_agg = family_agg[family_agg["n"] >= 1]

for _, row in family_agg.iterrows():
    fam = row["family"]
    axes[0].scatter(row["cka_mean"], row["acc_mean"],
                   c=FAMILY_COLORS.get(fam, "#999999"),
                   marker=FAMILY_MARKERS.get(fam, "o"),
                   s=max(40, row["n"] * 12), alpha=0.85, edgecolors='black', linewidth=0.5)
    axes[0].annotate(fam, (row["cka_mean"], row["acc_mean"]),
                    fontsize=6, ha='center', va='bottom',
                    xytext=(0, 4), textcoords='offset points')

rho_between = adv["between_family_rho"]
p_between = adv["between_family_p"]
z = np.polyfit(family_agg["cka_mean"], family_agg["acc_mean"], 1)
xline = np.linspace(family_agg["cka_mean"].min(), family_agg["cka_mean"].max(), 100)
axes[0].plot(xline, np.polyval(z, xline), 'k--', alpha=0.5, linewidth=1)
axes[0].set_xlabel("Mean CKA (DINOv2, noise)")
axes[0].set_ylabel("Mean ImageNet-1k Accuracy (%)")
axes[0].set_title(f"A. Between-Family\n" + r"$\rho$ = " + f"{rho_between:.3f} (p = {p_between:.3f}, n={len(family_agg)})")

# Panel B: Within-family correlations bar chart
within_families = []
within_rhos = []
within_ns = []
for family in dino_all["family"].unique():
    fdf = dino_all[dino_all["family"] == family].dropna(subset=["cka_pretrained_noise", "gt_acc"])
    if len(fdf) >= 4:
        rho, p = stats.spearmanr(fdf["cka_pretrained_noise"], fdf["gt_acc"])
        within_families.append(family)
        within_rhos.append(rho)
        within_ns.append(len(fdf))

# Sort by rho
sort_idx = np.argsort(within_rhos)
within_families = [within_families[i] for i in sort_idx]
within_rhos = [within_rhos[i] for i in sort_idx]
within_ns = [within_ns[i] for i in sort_idx]

bar_colors = [FAMILY_COLORS.get(f, '#999') for f in within_families]
bar_labels = [f"{f} (n={n})" for f, n in zip(within_families, within_ns)]
x = np.arange(len(within_families))
axes[1].barh(x, within_rhos, color=bar_colors, alpha=0.85)
axes[1].set_yticks(x)
axes[1].set_yticklabels(bar_labels, fontsize=8)
axes[1].axvline(x=0, color='black', linewidth=0.5)
axes[1].axvline(x=adv["family_residualized_rho"], color='red', linestyle='--', linewidth=1, 
               alpha=0.7, label=f"Resid. $\\rho$={adv['family_residualized_rho']:.2f}")
axes[1].set_xlabel(r"Within-Family Spearman $\rho$ (CKA vs. Acc)")
axes[1].set_title("B. Within-Family Correlations\n(families with n $\\geq$ 4)")
axes[1].legend(fontsize=8)
axes[1].set_xlim(-1.0, 1.0)

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig9_family_decomposition.png")
plt.savefig(f"{FIGDIR}/fig9_family_decomposition.pdf")
plt.close()
print("Figure 9: Family decomposition saved")


# ============================================================
# FIGURE 10 (NEW): Small vs Large and CNN vs Transformer
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# Panel A: Small vs Large models scatter
valid = dino_all.copy()
median_params = valid["params"].median()
small = valid[valid["params"] <= median_params]
large = valid[valid["params"] > median_params]

for subset, color, label in [(small, '#1f77b4', f'Small ($\\leq${median_params:.0f}M)'),
                               (large, '#d62728', f'Large (>{median_params:.0f}M)')]:
    axes[0].scatter(subset["cka_pretrained_noise"], subset["gt_acc"],
                   c=color, s=30, alpha=0.6, edgecolors='white', linewidth=0.3, label=label)
    rho, p = stats.spearmanr(subset["cka_pretrained_noise"], subset["gt_acc"])
    z = np.polyfit(subset["cka_pretrained_noise"], subset["gt_acc"], 1)
    xline = np.linspace(subset["cka_pretrained_noise"].min(), subset["cka_pretrained_noise"].max(), 100)
    axes[0].plot(xline, np.polyval(z, xline), color=color, linestyle='--', alpha=0.7, linewidth=1.5)

rho_s = adv["small_model_rho"]
rho_l = adv["large_model_rho"]
axes[0].set_xlabel("CKA (DINOv2, noise)")
axes[0].set_ylabel("ImageNet-1k Accuracy (%)")
axes[0].set_title(f"A. Small vs. Large Models\n"
                  f"Small: $\\rho$={rho_s:.2f}, Large: $\\rho$={rho_l:.2f}")
axes[0].legend(fontsize=8)

# Panel B: CNN vs Transformer
cnn_families = {"CNN", "EffNet", "Mobile", "ConvNeXt", "RegNet", "MetaFormer", "EdgeNeXt", "NAS"}
transformer_families = {"ViT", "Swin", "DeiT", "CaiT", "Mixer", "MaxViT", "EFormer"}

cnn_data = valid[valid["family"].isin(cnn_families)]
trans_data = valid[valid["family"].isin(transformer_families)]

for subset, color, label in [(cnn_data, '#1f77b4', 'CNN-like'),
                               (trans_data, '#d62728', 'Transformer-like')]:
    axes[1].scatter(subset["cka_pretrained_noise"], subset["gt_acc"],
                   c=color, s=30, alpha=0.6, edgecolors='white', linewidth=0.3, label=label)
    rho, p = stats.spearmanr(subset["cka_pretrained_noise"], subset["gt_acc"])
    z = np.polyfit(subset["cka_pretrained_noise"], subset["gt_acc"], 1)
    xline = np.linspace(subset["cka_pretrained_noise"].min(), subset["cka_pretrained_noise"].max(), 100)
    axes[1].plot(xline, np.polyval(z, xline), color=color, linestyle='--', alpha=0.7, linewidth=1.5)

rho_c = adv["cnn_rho"]
rho_t = adv["transformer_rho"]
axes[1].set_xlabel("CKA (DINOv2, noise)")
axes[1].set_ylabel("ImageNet-1k Accuracy (%)")
axes[1].set_title(f"B. CNN vs. Transformer\n"
                  f"CNN: $\\rho$={rho_c:.2f} (n={len(cnn_data)}), Trans: $\\rho$={rho_t:.2f} (n={len(trans_data)})")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig10_subpopulation_analysis.png")
plt.savefig(f"{FIGDIR}/fig10_subpopulation_analysis.pdf")
plt.close()
print("Figure 10: Subpopulation analysis saved")


# ============================================================
# FIGURE 11 (NEW): Comprehensive Summary — Bootstrap CIs for all key metrics
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Collect key metrics with CIs
summary_items = []

# CKA noise across teachers
for teacher, tlabel in zip(teachers, teacher_short):
    key = f"{teacher}__cka_pretrained_noise"
    if key in bootstrap:
        b = bootstrap[key]
        summary_items.append((f"{tlabel} CKA (noise)", b["mean_rho"], b["ci_low"], b["ci_high"], "CKA"))

# Best configs
for teacher, tlabel, col in [
    ("mae_vit_base", "MAE", "cka_pretrained_imagenet_val"),
    ("convnext_base_fcmae", "ConvNeXtV2", "cka_pretrained_imagenet_val"),
    ("mae_vit_base", "MAE", "knn_pretrained_imagenet_val"),
    ("convnext_base_fcmae", "ConvNeXtV2", "knn_pretrained_imagenet_val"),
]:
    key = f"{teacher}__{col}"
    metric_short = "CKA" if "cka" in col else "kNN"
    probe_short = "imgval" if "imagenet" in col else "noise"
    if key in bootstrap:
        b = bootstrap[key]
        summary_items.append((f"{tlabel} {metric_short} ({probe_short})", b["mean_rho"], b["ci_low"], b["ci_high"], metric_short))

# Baselines
for bl_name, bl_key in [("GradNorm", "dinov2_small__gradnorm"), 
                          ("NASWOT", "dinov2_small__naswot"),
                          ("SynFlow", "dinov2_small__synflow")]:
    if bl_key in bootstrap:
        b = bootstrap[bl_key]
        summary_items.append((bl_name, b["mean_rho"], b["ci_low"], b["ci_high"], "Baseline"))

# Sort by mean rho
summary_items.sort(key=lambda x: x[1])

labels = [x[0] for x in summary_items]
means = [x[1] for x in summary_items]
ci_los = [x[2] for x in summary_items]
ci_his = [x[3] for x in summary_items]
cats = [x[4] for x in summary_items]

cat_colors = {"CKA": "#1f77b4", "kNN": "#d62728", "Baseline": "#555555"}
y = np.arange(len(labels))

for i in range(len(labels)):
    color = cat_colors.get(cats[i], '#999')
    ax.barh(y[i], means[i], color=color, alpha=0.75, height=0.7)
    ax.plot([ci_los[i], ci_his[i]], [y[i], y[i]], color='black', linewidth=1.5)
    ax.plot([ci_los[i], ci_los[i]], [y[i]-0.15, y[i]+0.15], color='black', linewidth=1)
    ax.plot([ci_his[i], ci_his[i]], [y[i]-0.15, y[i]+0.15], color='black', linewidth=1)

ax.axvline(x=0, color='black', linewidth=0.5)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel(r"Spearman $\rho$ with ImageNet-1k Accuracy")
ax.set_title("All Proxies: Bootstrap 95% Confidence Intervals (n=83, 10K resamples)", fontweight='bold')

# Legend
legend_handles = [mpatches.Patch(color=cat_colors["CKA"], label="CKA", alpha=0.75),
                  mpatches.Patch(color=cat_colors["kNN"], label="kNN", alpha=0.75),
                  mpatches.Patch(color=cat_colors["Baseline"], label="Baseline", alpha=0.75)]
ax.legend(handles=legend_handles, fontsize=8, loc='lower left')
ax.set_xlim(-0.85, 0.55)

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig11_bootstrap_summary.png")
plt.savefig(f"{FIGDIR}/fig11_bootstrap_summary.pdf")
plt.close()
print("Figure 11: Bootstrap summary saved")


print("\n=== ALL FIGURES GENERATED SUCCESSFULLY ===")
print(f"Figures 1-7: Updated (CI annotations on Fig 1, CI error bars on Fig 2)")
print(f"Figure 8: NEW - Search simulation")
print(f"Figure 9: NEW - Family decomposition (between vs within)")
print(f"Figure 10: NEW - Subpopulation analysis (small/large, CNN/transformer)")
print(f"Figure 11: NEW - Bootstrap CI summary (all key metrics)")
