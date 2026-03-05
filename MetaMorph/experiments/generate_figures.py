#!/usr/bin/env python3
"""Generate all publication-quality figures for the metamorphic testing paper."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from pathlib import Path

# ── Style ──────────────────────────────────────────────────────────────────
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
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colorblind-safe palette (Okabe-Ito)
COLORS = {
    'gpt4o': '#0072B2',
    'gpt4o-mini': '#E69F00',
    'gemini-flash': '#009E73',
}
MODEL_LABELS = {
    'gpt4o': 'GPT-4o',
    'gpt4o-mini': 'GPT-4o Mini',
    'gemini-flash': 'Gemini 2.0 Flash',
}
TRANSFORM_LABELS = {
    'resize': 'Resize',
    'crop': 'Crop',
    'rotation': 'Rotation',
    'jpeg': 'JPEG Compr.',
    'blur': 'Gaussian Blur',
    'border_text': 'Border Text',
}
TRANSFORM_ORDER = ['resize', 'crop', 'rotation', 'jpeg', 'blur', 'border_text']
CATEGORY_LABELS = {
    'counting': 'Counting',
    'color_recognition': 'Color Recog.',
    'text_reading': 'Text/OCR',
    'chart_reading': 'Chart Reading',
    'scene_understanding': 'Scene Underst.',
    'spatial': 'Spatial Reasoning',
}

FIGDIR = Path("figures")
FIGDIR.mkdir(exist_ok=True)

# ── Load Data ──────────────────────────────────────────────────────────────
with open("experiments/results/results_partial.json") as f:
    raw = json.load(f)

results = [r for r in raw if r['consistency'] is not None and r['model'] != 'claude-sonnet']
print(f"Loaded {len(results)} valid results")

# ── Helper ─────────────────────────────────────────────────────────────────
def group_mean_std(results, key1, key2):
    """Group by key1 -> key2 -> list of consistency values, return means and stds."""
    data = defaultdict(lambda: defaultdict(list))
    for r in results:
        data[r[key1]][r[key2]].append(r['consistency'])
    means = {k1: {k2: np.mean(v2) for k2, v2 in v1.items()} for k1, v1 in data.items()}
    stds = {k1: {k2: np.std(v2) for k2, v2 in v1.items()} for k1, v1 in data.items()}
    counts = {k1: {k2: len(v2) for k2, v2 in v1.items()} for k1, v1 in data.items()}
    return means, stds, counts

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Robustness Curves - Consistency vs Severity (per model)
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Figure 1: Robustness curves...")

fig, axes = plt.subplots(2, 3, figsize=(7, 4.2), sharey=True)
axes_flat = axes.flatten()

for idx, tname in enumerate(TRANSFORM_ORDER):
    ax = axes_flat[idx]
    for model in ['gpt4o', 'gpt4o-mini', 'gemini-flash']:
        subset = [r for r in results if r['model'] == model and r['transform'] == tname]
        if not subset:
            continue
        sev_data = defaultdict(list)
        for r in subset:
            sev_data[r['severity']].append(r['consistency'])
        
        sevs = sorted(sev_data.keys())
        means = [np.mean(sev_data[s]) for s in sevs]
        stds = [np.std(sev_data[s]) / np.sqrt(len(sev_data[s])) for s in sevs]  # SEM
        
        ax.errorbar(sevs, means, yerr=stds, marker='o', markersize=4,
                    color=COLORS[model], label=MODEL_LABELS[model],
                    linewidth=1.5, capsize=2, capthick=1)
    
    ax.set_title(TRANSFORM_LABELS[tname], fontweight='bold')
    ax.set_xlabel('Severity Level')
    ax.set_ylim(0.3, 1.05)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    if idx % 3 == 0:
        ax.set_ylabel('Consistency Score')

handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02),
           frameon=False, fontsize=8)
plt.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(FIGDIR / "fig1_robustness_curves.pdf")
fig.savefig(FIGDIR / "fig1_robustness_curves.png")
print("  Saved fig1_robustness_curves.pdf")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Heatmap - Model x Transformation mean consistency
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Figure 2: Model-Transform heatmap...")

means_mt, _, _ = group_mean_std(results, 'model', 'transform')
models_order = ['gpt4o', 'gpt4o-mini', 'gemini-flash']
transforms_in_data = [t for t in TRANSFORM_ORDER if any(t in means_mt.get(m, {}) for m in models_order)]

matrix = np.zeros((len(models_order), len(transforms_in_data)))
for i, m in enumerate(models_order):
    for j, t in enumerate(transforms_in_data):
        matrix[i, j] = means_mt.get(m, {}).get(t, np.nan)

fig, ax = plt.subplots(figsize=(5, 2.2))
im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)
ax.set_xticks(range(len(transforms_in_data)))
ax.set_xticklabels([TRANSFORM_LABELS.get(t, t) for t in transforms_in_data], rotation=30, ha='right')
ax.set_yticks(range(len(models_order)))
ax.set_yticklabels([MODEL_LABELS[m] for m in models_order])

for i in range(len(models_order)):
    for j in range(len(transforms_in_data)):
        val = matrix[i, j]
        if not np.isnan(val):
            color = 'white' if val < 0.7 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)

cbar = fig.colorbar(im, ax=ax, shrink=0.8, label='Mean Consistency')
ax.set_title('Consistency by Model and Transformation Type', fontweight='bold', fontsize=9)
plt.tight_layout()
fig.savefig(FIGDIR / "fig2_heatmap.pdf")
fig.savefig(FIGDIR / "fig2_heatmap.png")
print("  Saved fig2_heatmap.pdf")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Category breakdown (grouped bar chart)
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Figure 3: Task category breakdown...")

means_mc, stds_mc, counts_mc = group_mean_std(results, 'model', 'category')
categories = sorted(set(r['category'] for r in results))

fig, ax = plt.subplots(figsize=(6, 3))
x = np.arange(len(categories))
width = 0.25

for i, model in enumerate(models_order):
    vals = [means_mc.get(model, {}).get(c, 0) for c in categories]
    errs = [stds_mc.get(model, {}).get(c, 0) / np.sqrt(max(counts_mc.get(model, {}).get(c, 1), 1)) for c in categories]
    ax.bar(x + i*width, vals, width, yerr=errs, label=MODEL_LABELS[model],
           color=COLORS[model], edgecolor='white', linewidth=0.5, capsize=2)

ax.set_ylabel('Mean Consistency Score')
ax.set_xlabel('Task Category')
ax.set_xticks(x + width)
ax.set_xticklabels([CATEGORY_LABELS.get(c, c) for c in categories], rotation=25, ha='right')
ax.set_ylim(0.4, 1.05)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
ax.legend(frameon=False, fontsize=7)
ax.set_title('Consistency Score by Task Category', fontweight='bold', fontsize=9)
plt.tight_layout()
fig.savefig(FIGDIR / "fig3_category_breakdown.pdf")
fig.savefig(FIGDIR / "fig3_category_breakdown.png")
print("  Saved fig3_category_breakdown.pdf")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Overall model comparison (box/violin plot)
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Figure 4: Overall model comparison...")

fig, ax = plt.subplots(figsize=(3.5, 3))
model_data = []
model_names = []
for m in models_order:
    vals = [r['consistency'] for r in results if r['model'] == m]
    model_data.append(vals)
    model_names.append(MODEL_LABELS[m])

bp = ax.boxplot(model_data, labels=model_names, patch_artist=True, widths=0.5,
                medianprops=dict(color='black', linewidth=1.5),
                whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1))

for patch, model in zip(bp['boxes'], models_order):
    patch.set_facecolor(COLORS[model])
    patch.set_alpha(0.7)

ax.set_ylabel('Consistency Score')
ax.set_title('Overall Robustness by Model', fontweight='bold', fontsize=9)
ax.set_ylim(-0.05, 1.1)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)

# Add mean markers
for i, vals in enumerate(model_data):
    mean_val = np.mean(vals)
    ax.plot(i + 1, mean_val, 'D', color='red', markersize=5, zorder=5)
    ax.annotate(f'{mean_val:.2f}', (i + 1, mean_val), textcoords="offset points",
                xytext=(12, 0), fontsize=7, color='red')

plt.tight_layout()
fig.savefig(FIGDIR / "fig4_model_comparison.pdf")
fig.savefig(FIGDIR / "fig4_model_comparison.png")
print("  Saved fig4_model_comparison.pdf")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Failure case analysis - worst performing transform x test combos
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Figure 5: Failure rate by severity...")

fig, ax = plt.subplots(figsize=(4.5, 3))

for model in models_order:
    subset = [r for r in results if r['model'] == model]
    sev_data = defaultdict(list)
    for r in subset:
        sev_data[r['severity']].append(1.0 if r['consistency'] < 0.5 else 0.0)
    
    sevs = sorted(sev_data.keys())
    failure_rates = [np.mean(sev_data[s]) * 100 for s in sevs]
    ax.plot(sevs, failure_rates, marker='s', markersize=5, color=COLORS[model],
            label=MODEL_LABELS[model], linewidth=1.5)

ax.set_xlabel('Perturbation Severity Level')
ax.set_ylabel('Failure Rate (%)')
ax.set_title('Metamorphic Violation Rate vs. Severity', fontweight='bold', fontsize=9)
ax.set_xticks([1, 2, 3, 4, 5])
ax.legend(frameon=False)
ax.set_ylim(-2, 60)
plt.tight_layout()
fig.savefig(FIGDIR / "fig5_failure_rate.pdf")
fig.savefig(FIGDIR / "fig5_failure_rate.png")
print("  Saved fig5_failure_rate.pdf")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6: Transformation examples strip (for qualitative figure)
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Figure 6: Transformation examples strip...")
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import io

img = Image.open("experiments/images/document.png")

def jpeg_compress(img, q):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).copy()

examples = [
    ("Original", img),
    ("Resize (50%)", img.resize((int(img.width*0.5), int(img.height*0.5)), Image.LANCZOS).resize(img.size, Image.LANCZOS)),
    ("Crop (15%)", (lambda: (lambda w,h,m: img.crop((m,m,w-m,h-m)).resize(img.size, Image.LANCZOS))(img.width, img.height, int(min(img.size)*0.15)))()),
    ("Rotate (5°)", img.rotate(5.0, resample=Image.BICUBIC, expand=False, fillcolor=(255,255,255))),
    ("JPEG (Q=10)", jpeg_compress(img, 10)),
    ("Blur (σ=3.5)", img.filter(ImageFilter.GaussianBlur(radius=3.5))),
]

fig, axes = plt.subplots(1, 6, figsize=(7, 1.5))
for ax, (title, im) in zip(axes, examples):
    ax.imshow(np.array(im))
    ax.set_title(title, fontsize=6, fontweight='bold')
    ax.axis('off')

plt.tight_layout(pad=0.5)
fig.savefig(FIGDIR / "fig6_transform_examples.pdf")
fig.savefig(FIGDIR / "fig6_transform_examples.png")
print("  Saved fig6_transform_examples.pdf")

# ═══════════════════════════════════════════════════════════════════════════
# Print summary statistics for the paper
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SUMMARY STATISTICS FOR PAPER")
print("="*60)

for model in models_order:
    vals = [r['consistency'] for r in results if r['model'] == model]
    failures = [1 for r in results if r['model'] == model and r['consistency'] < 0.5]
    print(f"\n{MODEL_LABELS[model]}:")
    print(f"  Mean consistency: {np.mean(vals):.3f} ± {np.std(vals):.3f}")
    print(f"  Median: {np.median(vals):.3f}")
    print(f"  Failure rate (<0.5): {len(failures)/len(vals)*100:.1f}%")
    print(f"  N = {len(vals)}")

print(f"\nTotal data points: {len(results)}")

# Per transform
print("\nBy transformation (all models):")
for t in TRANSFORM_ORDER:
    subset = [r['consistency'] for r in results if r['transform'] == t]
    if subset:
        print(f"  {TRANSFORM_LABELS.get(t,t)}: {np.mean(subset):.3f} ± {np.std(subset):.3f} (n={len(subset)})")

# Per category
print("\nBy category (all models):")
for c in sorted(set(r['category'] for r in results)):
    subset = [r['consistency'] for r in results if r['category'] == c]
    print(f"  {CATEGORY_LABELS.get(c,c)}: {np.mean(subset):.3f} ± {np.std(subset):.3f} (n={len(subset)})")

# Most vulnerable: worst test_id x transform x severity
print("\nTop 10 most inconsistent test-transform combinations:")
combo_data = defaultdict(list)
for r in results:
    combo_data[(r['test_id'], r['transform'], r['severity'])].append(r['consistency'])

combo_means = {k: np.mean(v) for k, v in combo_data.items()}
worst = sorted(combo_means.items(), key=lambda x: x[1])[:10]
for (tid, t, s), mean_c in worst:
    print(f"  {tid} + {t} (sev={s}): consistency={mean_c:.3f}")

# Qualitative failures
print("\nNotable failure examples:")
failures = [r for r in results if r['consistency'] is not None and r['consistency'] < 0.3]
seen = set()
for r in sorted(failures, key=lambda x: x['consistency']):
    key = (r['test_id'], r['transform'])
    if key not in seen and len(seen) < 8:
        seen.add(key)
        print(f"  [{r['model']}] {r['test_id']} + {r['transform']}(s={r['severity']})")
        print(f"    Original: {r['original_answer'][:60]}")
        print(f"    Transformed: {r['transformed_answer'][:60]}")
        print(f"    Consistency: {r['consistency']:.3f}")

print("\nAll figures saved to figures/")
