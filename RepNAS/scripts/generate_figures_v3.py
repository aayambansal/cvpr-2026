"""
RepNAS v3: Publication Figure Generation
=========================================
Generates all figures for the revised paper using v3 experiment data.
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
    "CNN": "#1f77b4",
    "EffNet": "#ff7f0e",
    "Mobile": "#2ca02c",
    "ConvNeXt": "#d62728",
    "ViT": "#9467bd",
    "Swin": "#8c564b",
    "DeiT": "#e377c2",
    "RegNet": "#7f7f7f",
    "MaxViT": "#bcbd22",
    "EFormer": "#17becf",
    "EdgeNeXt": "#ff6347",
    "MetaFormer": "#6b8e23",
    "NAS": "#daa520",
    "CaiT": "#4169e1",
    "Mixer": "#ff1493",
}

FAMILY_MARKERS = {
    "CNN": "o", "EffNet": "s", "Mobile": "^", "ConvNeXt": "D",
    "ViT": "P", "Swin": "*", "DeiT": "X", "RegNet": "v",
    "MaxViT": "p", "EFormer": "h", "EdgeNeXt": ">", "MetaFormer": "<",
    "NAS": "H", "CaiT": "d", "Mixer": "+",
}

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
# FIGURE 1: Main scatter — CKA vs Accuracy (all 4 teachers, 2x2)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
teachers = ["dinov2_small", "clip_vit_b32", "convnext_base_fcmae", "mae_vit_base"]
teacher_labels = ["DINOv2 ViT-S/14", "CLIP ViT-B/32", "ConvNeXtV2-B (FCMAE)", "MAE ViT-B/16"]

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
    
    # Fit line
    rho, p = stats.spearmanr(tdf["cka_pretrained_noise"], tdf["gt_acc"])
    z = np.polyfit(tdf["cka_pretrained_noise"], tdf["gt_acc"], 1)
    xline = np.linspace(tdf["cka_pretrained_noise"].min(), tdf["cka_pretrained_noise"].max(), 100)
    ax.plot(xline, np.polyval(z, xline), 'k--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel("CKA (pretrained, noise probes)")
    ax.set_ylabel("ImageNet-1k Accuracy (%)")
    ax.set_title(f"{label}\n" + r"$\rho$ = " + f"{rho:.3f} (p < {max(p, 1e-4):.4f})")
    
    if idx == 0:
        # Legend only on first panel
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
# FIGURE 2: Probe type ablation (bar chart, ρ across teachers)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# Panel A: CKA ρ by probe type and teacher
probe_types = ["noise", "imagenet_val", "augmented"]
probe_labels = ["Noise", "ImageNet-val\n(CIFAR-100)", "Augmented\n(gratings)"]
bar_width = 0.18
x = np.arange(len(teachers))

for pidx, (ptype, plabel) in enumerate(zip(probe_types, probe_labels)):
    rhos = []
    for teacher in teachers:
        tdf = df[df["teacher"] == teacher]
        col = f"cka_pretrained_{ptype}"
        valid = tdf.dropna(subset=[col, "gt_acc"])
        rho, _ = stats.spearmanr(valid[col], valid["gt_acc"])
        rhos.append(rho)
    axes[0].bar(x + pidx * bar_width, rhos, bar_width, label=plabel, alpha=0.85)

# Add random-init
rhos_rand = []
for teacher in teachers:
    tdf = df[df["teacher"] == teacher]
    valid = tdf.dropna(subset=["cka_random", "gt_acc"])
    rho, _ = stats.spearmanr(valid["cka_random"], valid["gt_acc"])
    rhos_rand.append(rho)
axes[0].bar(x + 3 * bar_width, rhos_rand, bar_width, label="Random-init", alpha=0.85, color='#555555')

axes[0].set_xticks(x + 1.5 * bar_width)
axes[0].set_xticklabels(["DINOv2", "CLIP", "ConvNeXtV2", "MAE"], fontsize=9)
axes[0].set_ylabel(r"Spearman $\rho$ (CKA vs. Accuracy)")
axes[0].set_title("A. CKA Correlation by Probe Type")
axes[0].axhline(y=0, color='black', linewidth=0.5)
axes[0].legend(fontsize=8, loc='lower left')
axes[0].set_ylim(-0.7, 0.45)

# Panel B: Metric comparison (CKA, kNN, GradNorm, etc.) — DINOv2 only
dino_df = df[df["teacher"] == "dinov2_small"].copy()
metric_names = ["CKA\n(noise)", "CKA\n(imgval)", "Cosine\n(noise)", "kNN\n(noise)", 
                "kNN\n(imgval)", "CKA\n(random)", "GradNorm", "NASWOT", "SynFlow"]
metric_cols = ["cka_pretrained_noise", "cka_pretrained_imagenet_val", "cosine_pretrained_noise",
               "knn_pretrained_noise", "knn_pretrained_imagenet_val", "cka_random",
               "gradnorm", "naswot", "synflow"]
metric_rhos = []
metric_ps = []
for col in metric_cols:
    valid = dino_df.dropna(subset=[col, "gt_acc"])
    if len(valid) > 5:
        rho, p = stats.spearmanr(valid[col], valid["gt_acc"])
    else:
        rho, p = np.nan, np.nan
    metric_rhos.append(rho)
    metric_ps.append(p)

colors = ['#d62728' if r < -0.1 else '#2ca02c' if r > 0.1 else '#999999' for r in metric_rhos]
bars = axes[1].bar(range(len(metric_names)), metric_rhos, color=colors, alpha=0.85)
axes[1].set_xticks(range(len(metric_names)))
axes[1].set_xticklabels(metric_names, fontsize=7, rotation=0)
axes[1].set_ylabel(r"Spearman $\rho$ with Accuracy")
axes[1].set_title("B. All Proxies (DINOv2 Teacher)")
axes[1].axhline(y=0, color='black', linewidth=0.5)

# Add significance stars
for i, (rho, p) in enumerate(zip(metric_rhos, metric_ps)):
    if not np.isnan(p):
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if stars:
            y = rho + (0.03 if rho >= 0 else -0.05)
            axes[1].text(i, y, stars, ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig2_probe_ablation.png")
plt.savefig(f"{FIGDIR}/fig2_probe_ablation.pdf")
plt.close()
print("Figure 2: Probe type ablation saved")

# ============================================================
# FIGURE 3: Partial correlation heatmap
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
    
    # Random CKA
    valid = tdf.dropna(subset=["cka_random", "gt_acc", "log_params"])
    if len(valid) > 10:
        result = pg.partial_corr(data=valid, x="cka_random", y="gt_acc", covar="log_params", method="spearman")
        row["CKA random"] = result["r"].values[0]
    
    partial_data.append(row)

# Add baselines for DINOv2
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
# Fill NaN for display
heatmap_data = pdf.fillna(0).values
xlabels = pdf.columns.tolist()
ylabels = pdf.index.tolist()

im = ax.imshow(heatmap_data, cmap='RdBu_r', vmin=-0.7, vmax=0.7, aspect='auto')
ax.set_xticks(range(len(xlabels)))
ax.set_xticklabels(xlabels, rotation=35, ha='right', fontsize=9)
ax.set_yticks(range(len(ylabels)))
ax.set_yticklabels(ylabels, fontsize=9)

# Annotate cells
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
# FIGURE 4: Top-k regret curves
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# Panel A: Hit rate at various k
for teacher, color, label in zip(
    ["dinov2_small", "clip_vit_b32", "convnext_base_fcmae", "mae_vit_base"],
    ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    ["DINOv2", "CLIP", "ConvNeXtV2", "MAE"]
):
    tdf = df[df["teacher"] == teacher].dropna(subset=["cka_pretrained_noise", "gt_acc"])
    sorted_by_acc = tdf.sort_values("gt_acc", ascending=False)
    
    ks = range(3, min(40, len(tdf)))
    hit_rates = []
    regrets = []
    
    for k in ks:
        oracle = set(sorted_by_acc.head(k)["candidate"].values)
        oracle_mean = sorted_by_acc.head(k)["gt_acc"].mean()
        # Use INVERTED CKA (lowest = best)
        sorted_proxy = tdf.sort_values("cka_pretrained_noise", ascending=True)
        proxy_top = set(sorted_proxy.head(k)["candidate"].values)
        hit_rates.append(len(oracle & proxy_top) / k)
        regrets.append(oracle_mean - sorted_proxy.head(k)["gt_acc"].mean())
    
    axes[0].plot(list(ks), hit_rates, color=color, label=f"Inv-CKA ({label})", linewidth=1.5)
    axes[1].plot(list(ks), regrets, color=color, label=f"Inv-CKA ({label})", linewidth=1.5)

# Add GradNorm baseline
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

# Random baseline
axes[0].axhline(y=0, color='gray', linewidth=0.3)
axes[1].axhline(y=0, color='gray', linewidth=0.3)

axes[0].set_xlabel("k (top-k selected)")
axes[0].set_ylabel("Hit Rate (fraction in oracle top-k)")
axes[0].set_title("A. Top-k Hit Rate")
axes[0].legend(fontsize=7)
axes[0].set_ylim(-0.05, 1.05)

axes[1].set_xlabel("k (top-k selected)")
axes[1].set_ylabel("Regret (oracle mean acc - proxy mean acc)")
axes[1].set_title("B. Top-k Regret (%)")
axes[1].legend(fontsize=7)

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig4_topk_analysis.png")
plt.savefig(f"{FIGDIR}/fig4_topk_analysis.pdf")
plt.close()
print("Figure 4: Top-k analysis saved")

# ============================================================
# FIGURE 5: Cross-teacher agreement & ImageNet-val scatter
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# Panel A: Best teacher x probe combination scatter
# Use MAE + imagenet_val (strongest correlation)
mae_df = df[df["teacher"] == "mae_vit_base"].dropna(subset=["cka_pretrained_imagenet_val", "gt_acc"])
for family in mae_df["family"].unique():
    fdf = mae_df[mae_df["family"] == family]
    axes[0].scatter(fdf["cka_pretrained_imagenet_val"], fdf["gt_acc"],
                   c=FAMILY_COLORS.get(family, "#999999"),
                   marker=FAMILY_MARKERS.get(family, "o"),
                   s=35, alpha=0.75, edgecolors='white', linewidth=0.3,
                   label=family)

rho, p = stats.spearmanr(mae_df["cka_pretrained_imagenet_val"], mae_df["gt_acc"])
z = np.polyfit(mae_df["cka_pretrained_imagenet_val"], mae_df["gt_acc"], 1)
xline = np.linspace(mae_df["cka_pretrained_imagenet_val"].min(), mae_df["cka_pretrained_imagenet_val"].max(), 100)
axes[0].plot(xline, np.polyval(z, xline), 'k--', alpha=0.5, linewidth=1)
axes[0].set_xlabel("CKA (pretrained, ImageNet-val probes)")
axes[0].set_ylabel("ImageNet-1k Accuracy (%)")
axes[0].set_title(f"A. Best Config: MAE + ImageNet-val\n" + r"$\rho$ = " + f"{rho:.3f} (p < 0.0001)")

# Legend
handles = [Line2D([0], [0], marker=FAMILY_MARKERS.get(f, 'o'), color='w',
                 markerfacecolor=FAMILY_COLORS.get(f, '#999'), markersize=6, label=f)
          for f in sorted(mae_df["family"].unique())]
axes[0].legend(handles=handles, loc='upper right', ncol=2, framealpha=0.8, fontsize=6)

# Panel B: Cross-teacher correlation bar chart (imagenet_val)
teacher_short = ["DINOv2", "CLIP", "ConvNeXt", "MAE"]
noise_rhos = []
imgval_rhos = []
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
# FIGURE 6: kNN agreement for ConvNeXt/MAE teachers (imagenet_val)
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(6, 5))

# MAE + imagenet_val kNN shows strongest correlation (ρ=-0.70!)
mae_knn = df[df["teacher"] == "mae_vit_base"].dropna(subset=["knn_pretrained_imagenet_val", "gt_acc"])
for family in mae_knn["family"].unique():
    fdf = mae_knn[mae_knn["family"] == family]
    ax.scatter(fdf["knn_pretrained_imagenet_val"], fdf["gt_acc"],
              c=FAMILY_COLORS.get(family, "#999999"),
              marker=FAMILY_MARKERS.get(family, "o"),
              s=40, alpha=0.75, edgecolors='white', linewidth=0.3,
              label=family)

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
print("Figure 6: kNN (MAE, imagenet-val) saved")

# ============================================================
# FIGURE 7: Random-init positive correlation
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(6, 5))

dino_rand = dino_df.dropna(subset=["cka_random", "gt_acc"])
for family in dino_rand["family"].unique():
    fdf = dino_rand[dino_rand["family"] == family]
    ax.scatter(fdf["cka_random"], fdf["gt_acc"],
              c=FAMILY_COLORS.get(family, "#999999"),
              marker=FAMILY_MARKERS.get(family, "o"),
              s=40, alpha=0.75, edgecolors='white', linewidth=0.3,
              label=family)

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
print("Figure 7: Random-init positive correlation saved")

print("\nAll figures generated successfully!")
