#!/usr/bin/env python3
"""
Analyze VLM experiment results and generate publication-quality figures.
Metrics:
  - Answer Exact Match (AEM)
  - Answer Containment (AC) - ground truth contained in VLM answer
  - Grounding IoU - bbox overlap between predicted and GT evidence
  - Drift Robustness Score (DRS) - composite
  - Per-drift-type and per-severity breakdowns
"""

import json
import os
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = DATA_DIR / "results"
FIG_DIR = Path(__file__).parent.parent / "latex" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["gpt4o-mini", "gpt4o", "gemini-flash"]
MODEL_LABELS = {"gpt4o-mini": "GPT-4o-mini", "gpt4o": "GPT-4o", "gemini-flash": "Gemini-2.0-Flash"}
DRIFT_NAMES = ["theme_change", "sidebar_toggle", "responsive_scale", "viewport_crop", "composite"]
DRIFT_LABELS = {"theme_change": "Theme\nChange", "sidebar_toggle": "Sidebar\nToggle", 
                "responsive_scale": "Responsive\nScale", "viewport_crop": "Viewport\nCrop",
                "composite": "Composite"}
DRIFT_SEVERITY = {"theme_change": 1, "sidebar_toggle": 2, "responsive_scale": 3,
                  "viewport_crop": 4, "composite": 5}

# ── Publication style ──────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Colorblind-safe palette (Okabe-Ito)
COLORS = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]


# ── Metrics ────────────────────────────────────────────────────
def normalize(s):
    """Normalize string for comparison."""
    if not isinstance(s, str):
        s = str(s)
    return s.strip().lower().replace(",", "").replace("$", "").replace("%", "").replace(" ", "")


def exact_match(pred, gt):
    return 1.0 if normalize(pred) == normalize(gt) else 0.0


def containment(pred, gt):
    """Check if GT is contained in prediction."""
    return 1.0 if normalize(gt) in normalize(pred) else 0.0


def bbox_iou(box1, box2):
    """Compute IoU between two bounding boxes [x1,y1,x2,y2]."""
    if not box1 or not box2 or len(box1) < 4 or len(box2) < 4:
        return 0.0
    if all(v == 0 for v in box1) or all(v == 0 for v in box2):
        return 0.0
    
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = max(0, box1[2]-box1[0]) * max(0, box1[3]-box1[1])
    area2 = max(0, box2[2]-box2[0]) * max(0, box2[3]-box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0


def answer_consistency(base_answer, drift_answer):
    """Check if drift answer matches base answer."""
    return 1.0 if normalize(base_answer) == normalize(drift_answer) else 0.0


# ── Load and analyze results ───────────────────────────────────
def load_results(model_name):
    path = RESULTS_DIR / f"results_{model_name}.json"
    with open(path) as f:
        return json.load(f)


def compute_all_metrics():
    """Compute comprehensive metrics across all models."""
    all_metrics = {}
    
    for model_name in MODELS:
        results = load_results(model_name)
        
        # Base metrics
        base_em = []
        base_ac = []
        base_iou = []
        
        # Per-drift metrics
        drift_em = defaultdict(list)
        drift_ac = defaultdict(list)
        drift_iou = defaultdict(list)
        drift_consistency = defaultdict(list)
        
        # Per-page-type metrics
        type_base_em = defaultdict(list)
        type_drift_em = defaultdict(lambda: defaultdict(list))
        
        # Per-QA-type metrics  
        qa_type_base_em = defaultdict(list)
        qa_type_drift_em = defaultdict(lambda: defaultdict(list))
        
        for page_key, page in results.items():
            page_type = page["page_type"]
            
            # Base results
            base_answers = {}
            for r in page["base_results"]:
                em = exact_match(r["vlm_answer"], r["ground_truth"])
                ac = containment(r["vlm_answer"], r["ground_truth"])
                iou = bbox_iou(r.get("vlm_bbox", []), r.get("gt_bbox", []))
                
                base_em.append(em)
                base_ac.append(ac)
                base_iou.append(iou)
                type_base_em[page_type].append(em)
                qa_type_base_em[r["qa_type"]].append(em)
                base_answers[r["question"]] = r["vlm_answer"]
            
            # Variant results
            for vr in page["variant_results"]:
                dtype = vr["drift_type"]
                for r in vr["results"]:
                    em = exact_match(r["vlm_answer"], r["ground_truth"])
                    ac = containment(r["vlm_answer"], r["ground_truth"])
                    iou = bbox_iou(r.get("vlm_bbox", []), r.get("gt_bbox", []))
                    
                    drift_em[dtype].append(em)
                    drift_ac[dtype].append(ac)
                    drift_iou[dtype].append(iou)
                    type_drift_em[page_type][dtype].append(em)
                    qa_type_drift_em[r["qa_type"]][dtype].append(em)
                    
                    # Consistency with base
                    base_ans = base_answers.get(r["question"], "")
                    cons = answer_consistency(base_ans, r["vlm_answer"])
                    drift_consistency[dtype].append(cons)
        
        all_metrics[model_name] = {
            "base_em": np.mean(base_em), "base_em_std": np.std(base_em),
            "base_ac": np.mean(base_ac), "base_ac_std": np.std(base_ac),
            "base_iou": np.mean(base_iou), "base_iou_std": np.std(base_iou),
            "base_n": len(base_em),
            "drift_em": {k: np.mean(v) for k, v in drift_em.items()},
            "drift_em_std": {k: np.std(v) for k, v in drift_em.items()},
            "drift_ac": {k: np.mean(v) for k, v in drift_ac.items()},
            "drift_iou": {k: np.mean(v) for k, v in drift_iou.items()},
            "drift_consistency": {k: np.mean(v) for k, v in drift_consistency.items()},
            "drift_consistency_std": {k: np.std(v) for k, v in drift_consistency.items()},
            "drift_n": {k: len(v) for k, v in drift_em.items()},
            "type_base_em": {k: np.mean(v) for k, v in type_base_em.items()},
            "type_drift_em": {k: {kk: np.mean(vv) for kk, vv in v.items()} 
                             for k, v in type_drift_em.items()},
            "qa_type_base_em": {k: np.mean(v) for k, v in qa_type_base_em.items()},
            "qa_type_drift_em": {k: {kk: np.mean(vv) for kk, vv in v.items()}
                                for k, v in qa_type_drift_em.items()},
            # Raw arrays for detailed analysis
            "_base_em_arr": base_em,
            "_drift_em_arrs": {k: v for k, v in drift_em.items()},
            "_drift_consistency_arrs": {k: v for k, v in drift_consistency.items()},
        }
    
    return all_metrics


# ── Figure generators ──────────────────────────────────────────

def fig1_main_performance(metrics):
    """MAIN FIGURE: Performance vs drift severity for all models.
    This is the hero figure of the paper."""
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.4), sharey=True)
    
    severities = [0, 1, 2, 3, 4, 5]  # 0 = base
    sev_labels = ["Base", "Theme\nChange", "Sidebar\nToggle", "Resp.\nScale", "Viewport\nCrop", "Composite"]
    
    for ax_idx, metric_name in enumerate(["em", "ac", "consistency"]):
        ax = axes[ax_idx]
        
        for mi, model in enumerate(MODELS):
            m = metrics[model]
            
            if metric_name == "em":
                vals = [m["base_em"]] + [m["drift_em"].get(d, 0) for d in DRIFT_NAMES]
                title = "Answer Exact Match"
            elif metric_name == "ac":
                vals = [m["base_ac"]] + [m["drift_ac"].get(d, 0) for d in DRIFT_NAMES]
                title = "Answer Containment"
            else:
                vals = [1.0] + [m["drift_consistency"].get(d, 0) for d in DRIFT_NAMES]
                title = "Answer Consistency"
            
            ax.plot(severities, vals, marker="o", color=COLORS[mi], 
                   label=MODEL_LABELS[model], linewidth=1.5, markersize=4)
        
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Drift Type (increasing severity)")
        ax.set_xticks(severities)
        ax.set_xticklabels(sev_labels, fontsize=6)
        
        if ax_idx == 0:
            ax.set_ylabel("Score")
        
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
        ax.grid(axis="y", alpha=0.2)
    
    axes[0].legend(loc="lower left", frameon=True, fancybox=True, framealpha=0.9, fontsize=6)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_performance_vs_drift.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1_performance_vs_drift.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Generated: fig1_performance_vs_drift.pdf")


def fig2_heatmap(metrics):
    """Heatmap of drift-type vs page-type performance degradation."""
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.8))
    
    page_types = ["dashboard", "settings", "table", "article", "analytics"]
    page_labels = ["Dashboard", "Settings", "Table", "Article", "Analytics"]
    drift_labels_short = ["Theme", "Sidebar", "Scale", "Crop", "Composite"]
    
    for mi, model in enumerate(MODELS):
        m = metrics[model]
        ax = axes[mi]
        
        # Build matrix: (page_type x drift_type) showing EM accuracy
        matrix = np.zeros((len(page_types), len(DRIFT_NAMES)))
        for pi, pt in enumerate(page_types):
            base_em = m["type_base_em"].get(pt, 0)
            for di, dt in enumerate(DRIFT_NAMES):
                drift_val = m["type_drift_em"].get(pt, {}).get(dt, 0)
                # Show relative degradation from base
                matrix[pi, di] = drift_val - base_em if base_em > 0 else 0
        
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-0.6, vmax=0.2)
        ax.set_xticks(range(len(DRIFT_NAMES)))
        ax.set_xticklabels(drift_labels_short, fontsize=6, rotation=45, ha="right")
        ax.set_yticks(range(len(page_types)))
        if mi == 0:
            ax.set_yticklabels(page_labels, fontsize=7)
        else:
            ax.set_yticklabels([])
        ax.set_title(MODEL_LABELS[model], fontweight="bold", fontsize=8)
        
        # Add values
        for pi in range(len(page_types)):
            for di in range(len(DRIFT_NAMES)):
                val = matrix[pi, di]
                color = "white" if abs(val) > 0.3 else "black"
                ax.text(di, pi, f"{val:+.2f}", ha="center", va="center", 
                       fontsize=5.5, color=color)
    
    fig.colorbar(im, ax=axes, label="EM Change from Base", shrink=0.8, pad=0.02)
    fig.suptitle("Performance Degradation by Page Type and Drift Type", 
                fontweight="bold", fontsize=9, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_heatmap.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig2_heatmap.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Generated: fig2_heatmap.pdf")


def fig3_grounding_iou(metrics):
    """Grounding IoU comparison: base vs drifted."""
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    
    x = np.arange(len(DRIFT_NAMES) + 1)
    width = 0.22
    
    for mi, model in enumerate(MODELS):
        m = metrics[model]
        vals = [m["base_iou"]] + [m["drift_iou"].get(d, 0) for d in DRIFT_NAMES]
        ax.bar(x + mi*width - width, vals, width, label=MODEL_LABELS[model],
              color=COLORS[mi], alpha=0.85)
    
    ax.set_ylabel("Grounding IoU")
    ax.set_xlabel("Condition")
    ax.set_xticks(x)
    ax.set_xticklabels(["Base"] + [DRIFT_LABELS[d].replace("\n", " ") for d in DRIFT_NAMES],
                       fontsize=6, rotation=30, ha="right")
    ax.legend(fontsize=6, frameon=True)
    ax.set_ylim(0, max(0.5, ax.get_ylim()[1]*1.1))
    ax.grid(axis="y", alpha=0.2)
    ax.set_title("Evidence Grounding IoU", fontweight="bold")
    
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_grounding_iou.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig3_grounding_iou.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Generated: fig3_grounding_iou.pdf")


def fig4_qa_type_breakdown(metrics):
    """Per QA type breakdown across drift conditions."""
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.4))
    
    qa_types = ["value_extraction", "trend_detection", "table_lookup", "text_extraction", 
                "ui_state", "chart_reading"]
    qa_labels = ["Value\nExtract", "Trend\nDetect", "Table\nLookup", "Text\nExtract",
                "UI State", "Chart\nRead"]
    
    for mi, model in enumerate(MODELS):
        m = metrics[model]
        ax = axes[mi]
        
        base_vals = [m["qa_type_base_em"].get(qt, 0) for qt in qa_types]
        
        # Average across all drift types
        drift_avg = []
        for qt in qa_types:
            qt_drifts = m["qa_type_drift_em"].get(qt, {})
            if qt_drifts:
                drift_avg.append(np.mean(list(qt_drifts.values())))
            else:
                drift_avg.append(0)
        
        x = np.arange(len(qa_types))
        width = 0.35
        ax.bar(x - width/2, base_vals, width, label="Base", color=COLORS[0], alpha=0.85)
        ax.bar(x + width/2, drift_avg, width, label="Avg. Drifted", color=COLORS[1], alpha=0.85)
        
        ax.set_title(MODEL_LABELS[model], fontweight="bold", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(qa_labels, fontsize=5.5)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.2)
        
        if mi == 0:
            ax.set_ylabel("Exact Match")
            ax.legend(fontsize=6)
    
    fig.suptitle("Performance by Question Type: Base vs. Drifted", fontweight="bold", fontsize=9, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig4_qa_type.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig4_qa_type.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Generated: fig4_qa_type.pdf")


def fig5_consistency_distribution(metrics):
    """Distribution of answer consistency scores."""
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    
    for mi, model in enumerate(MODELS):
        m = metrics[model]
        # Collect all consistency scores
        all_cons = []
        for dtype in DRIFT_NAMES:
            all_cons.extend(m["_drift_consistency_arrs"].get(dtype, []))
        
        if all_cons:
            # Plot as kernel density
            from scipy import stats
            kde = stats.gaussian_kde(all_cons, bw_method=0.15)
            x_range = np.linspace(-0.1, 1.1, 200)
            ax.plot(x_range, kde(x_range), color=COLORS[mi], label=MODEL_LABELS[model], linewidth=1.5)
            ax.fill_between(x_range, kde(x_range), alpha=0.15, color=COLORS[mi])
    
    ax.set_xlabel("Answer Consistency Score")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Answer Consistency Under Drift", fontweight="bold")
    ax.legend(fontsize=6)
    ax.set_xlim(-0.1, 1.1)
    ax.grid(alpha=0.2)
    
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_consistency_dist.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig5_consistency_dist.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Generated: fig5_consistency_dist.pdf")


def fig6_drs_radar(metrics):
    """Drift Robustness Score (DRS) radar chart - composite metric."""
    fig, ax = plt.subplots(figsize=(3.5, 3.0), subplot_kw=dict(polar=True))
    
    categories = ["Theme\nRobust.", "Sidebar\nRobust.", "Scale\nRobust.", 
                  "Crop\nRobust.", "Composite\nRobust.", "Grounding\nStability"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    for mi, model in enumerate(MODELS):
        m = metrics[model]
        # DRS per drift: average of (EM, AC, consistency) normalized
        vals = []
        for dtype in DRIFT_NAMES:
            em = m["drift_em"].get(dtype, 0)
            ac = m["drift_ac"].get(dtype, 0)
            cons = m["drift_consistency"].get(dtype, 0)
            drs = (em + ac + cons) / 3.0
            vals.append(drs)
        
        # Add grounding stability (avg IoU across drifts)
        avg_iou = np.mean([m["drift_iou"].get(d, 0) for d in DRIFT_NAMES])
        vals.append(min(avg_iou * 5, 1.0))  # Scale IoU to 0-1 range
        
        vals += vals[:1]
        ax.plot(angles, vals, color=COLORS[mi], linewidth=1.5, label=MODEL_LABELS[model])
        ax.fill(angles, vals, alpha=0.1, color=COLORS[mi])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=6)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=6)
    ax.set_title("Drift Robustness Score (DRS)", fontweight="bold", fontsize=9, pad=20)
    
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig6_drs_radar.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig6_drs_radar.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Generated: fig6_drs_radar.pdf")


def print_summary_table(metrics):
    """Print LaTeX-ready summary table."""
    print("\n" + "="*80)
    print("SUMMARY TABLE (for LaTeX)")
    print("="*80)
    
    print("\n% Base performance")
    for model in MODELS:
        m = metrics[model]
        print(f"  {MODEL_LABELS[model]:20s} & {m['base_em']:.3f} & {m['base_ac']:.3f} & {m['base_iou']:.3f} \\\\")
    
    print("\n% Per-drift EM")
    for model in MODELS:
        m = metrics[model]
        row = f"  {MODEL_LABELS[model]:20s}"
        for dtype in DRIFT_NAMES:
            row += f" & {m['drift_em'].get(dtype, 0):.3f}"
        row += " \\\\"
        print(row)
    
    print("\n% Per-drift Consistency")
    for model in MODELS:
        m = metrics[model]
        row = f"  {MODEL_LABELS[model]:20s}"
        for dtype in DRIFT_NAMES:
            row += f" & {m['drift_consistency'].get(dtype, 0):.3f}"
        row += " \\\\"
        print(row)
    
    # Compute overall DRS
    print("\n% Overall DRS (composite)")
    for model in MODELS:
        m = metrics[model]
        drs_vals = []
        for dtype in DRIFT_NAMES:
            em = m["drift_em"].get(dtype, 0)
            ac = m["drift_ac"].get(dtype, 0)
            cons = m["drift_consistency"].get(dtype, 0)
            drs_vals.append((em + ac + cons) / 3.0)
        overall_drs = np.mean(drs_vals)
        print(f"  {MODEL_LABELS[model]:20s}: DRS = {overall_drs:.3f}")
    
    # Save numeric results
    summary = {}
    for model in MODELS:
        m = metrics[model]
        drs_vals = []
        for dtype in DRIFT_NAMES:
            em = m["drift_em"].get(dtype, 0)
            ac = m["drift_ac"].get(dtype, 0)
            cons = m["drift_consistency"].get(dtype, 0)
            drs_vals.append((em + ac + cons) / 3.0)
        
        summary[model] = {
            "base_em": float(m["base_em"]),
            "base_ac": float(m["base_ac"]),
            "base_iou": float(m["base_iou"]),
            "drift_em": {k: float(v) for k, v in m["drift_em"].items()},
            "drift_ac": {k: float(v) for k, v in m["drift_ac"].items()},
            "drift_iou": {k: float(v) for k, v in m["drift_iou"].items()},
            "drift_consistency": {k: float(v) for k, v in m["drift_consistency"].items()},
            "overall_drs": float(np.mean(drs_vals)),
            "type_base_em": {k: float(v) for k, v in m["type_base_em"].items()},
            "qa_type_base_em": {k: float(v) for k, v in m["qa_type_base_em"].items()},
        }
    
    with open(DATA_DIR / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {DATA_DIR / 'summary_metrics.json'}")


def main():
    print("Computing metrics...")
    metrics = compute_all_metrics()
    
    print("\nGenerating figures...")
    fig1_main_performance(metrics)
    fig2_heatmap(metrics)
    fig3_grounding_iou(metrics)
    fig4_qa_type_breakdown(metrics)
    fig5_consistency_distribution(metrics)
    fig6_drs_radar(metrics)
    
    print_summary_table(metrics)
    
    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
