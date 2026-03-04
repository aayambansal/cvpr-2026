# RepNAS: Architecture Selection in Pretrained Model Zoos via Representation Agreement with Foundation Models

**Authors:** Aayam Bansal, Ishaan Gangwani

**Venue:** 6th Workshop on Neural Architecture Search (CVPR-NAS'26), CVPR 2026, Denver CO

[Paper PDF](latex/main.pdf)

---

## Summary

RepNAS is a zero-shot architecture selection method that scores pretrained architectures by their representation agreement with foundation models using Centered Kernel Alignment (CKA). Instead of fine-tuning every candidate in a model zoo, RepNAS requires only a single forward pass through each architecture with a small probe dataset.

### Key Finding: Alignment Inversion

Architectures whose representations are **most similar** to a foundation-model teacher consistently rank **worst** on ImageNet. This counterintuitive inversion is robust across four teachers, three probe types, and five random seeds.

| Teacher | Spearman rho (CKA noise vs. ImageNet acc) |
|---------|------------------------------------------|
| DINOv2 | -0.42 |
| CLIP | -0.52 |
| MAE | -0.45 |
| ConvNeXtV2-FCMAE | -0.40 |

All bootstrap 95% CIs exclude zero.

### Key Results

- **Architecture search:** Selecting the least-similar architecture yields **85.1% ImageNet top-1** with only **0.82% regret** vs. oracle (85.8%). Random baseline: 84.3 +/- 1.0%.
- **Not a confound:** Partial correlations controlling for parameter count remain strong (CLIP partial rho = -0.579). Between-family rho = -0.557; within-family rho = -0.068.
- **Transfer asymmetry:** The same CKA scores **positively** predict transfer performance on CIFAR-100 (rho = +0.826) and Flowers-102 (rho = +0.518), while ImageNet accuracy alone does not (rho = -0.034 for Flowers).
- **Layer-wise sign flip:** Early layers show positive correlation (rho = +0.370), late layers show negative (rho = -0.377), suggesting the inversion originates in task-specific head representations.
- **Seed stability:** Mean rho = -0.439, sigma = 0.037 across 5 seeds; pairwise Kendall tau = 0.627.
- **Ensemble proxy:** InvCKA + model size achieves rho = +0.625; all-four ensemble rho = +0.685 (n=56).

---

## Repository Structure

```
RepNAS/
├── latex/                    # Paper source
│   ├── main.tex              # Paper v7 (8 pages, official CVPR2026 format)
│   ├── main.pdf              # Compiled paper
│   ├── cvpr.sty              # Official CVPR2026 author kit style
│   ├── ieeenat_fullname.bst  # Official CVPR2026 bibliography style
│   ├── preamble.tex          # Custom spacing tweaks
│   └── references.bib        # Bibliography (21 entries)
├── figures/                  # All figures (.pdf + .png)
│   ├── fig1_alignment_inversion_4teachers.*   # Main result (Fig. 1)
│   ├── fig2_probe_ablation.*                  # Probe type comparison (Fig. 2)
│   ├── fig8_search_simulation.*               # Search simulation (Fig. 3)
│   ├── fig9_family_decomposition.*            # Family decomposition (Fig. 4)
│   ├── fig10_subpopulation_analysis.*         # Subpopulation analysis (Fig. 5)
│   ├── fig11_bootstrap_summary.*              # Bootstrap CIs (Fig. 6)
│   ├── fig12_seed_stability.*                 # Seed stability (Fig. 7)
│   ├── fig13_trajectory_signflip.*            # Training trajectory (Fig. 8)
│   ├── fig15_layerwise_cka.*                  # Layer-wise CKA (Fig. 9)
│   ├── fig16_transfer_correlation.*           # Transfer correlation (Fig. 10)
│   └── ...                                    # Additional supplementary figures
├── experiments/              # Raw experiment results (JSON)
│   ├── repnas_v3_results.json       # Main v3 experiment (83 archs, 4 teachers)
│   ├── repnas_upgrades.json         # Trajectory, SNIP/GraSP, kNN, layer-wise
│   ├── repnas_transfer.json         # Transfer experiments (CIFAR-100, Flowers)
│   ├── repnas_seed_stability.json   # 5-seed stability experiment
│   ├── seed_stability_analysis.json # Seed stability summary statistics
│   ├── new_experiments_analysis.json # Analysis of upgrade experiments
│   ├── advanced_analysis.json       # Partial correlations, subgroups, bootstrap
│   └── ...                          # Additional analysis outputs
└── scripts/                  # Experiment and analysis scripts
    ├── repnas_modal_v3.py           # Main experiment (Modal A100)
    ├── repnas_seed_stability.py     # Seed stability experiment
    ├── repnas_upgrades.py           # Upgrade experiments
    ├── repnas_transfer.py           # Transfer experiments
    ├── analyze_new_experiments.py   # Analysis + figure generation
    ├── generate_figures_v4.py       # Figure generation (all 17 figures)
    └── ...                          # Earlier experiment/analysis versions
```

---

## Compiling the Paper

Requires TeX Live 2025 (or compatible). From the `latex/` directory:

```bash
cd latex
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

The compiled `main.pdf` is already included in the repository.

---

## Reproducing Experiments

All experiments were run on Modal (NVIDIA A100 GPU). The main experiment takes approximately 1 hour.

### Main experiment (83 architectures, 4 teachers)
```bash
# Requires Modal account and credentials
modal run scripts/repnas_modal_v3.py
```

### Seed stability (5 seeds x 83 architectures)
```bash
modal run scripts/repnas_seed_stability.py
```

### Upgrade experiments (trajectory, SNIP/GraSP, layer-wise CKA)
```bash
modal run scripts/repnas_upgrades.py
```

### Transfer experiments (CIFAR-100, Flowers-102)
```bash
modal run scripts/repnas_transfer.py
```

### Analysis and figures
```bash
python scripts/analyze_new_experiments.py
python scripts/generate_figures_v4.py
```

---

## Method Overview

1. **Probe generation:** Create a small probe dataset (Gaussian noise, natural images from CIFAR-100, or augmented versions)
2. **Feature extraction:** Pass the probe through both a pretrained candidate architecture and a foundation-model teacher
3. **CKA scoring:** Compute Centered Kernel Alignment between the two feature matrices
4. **Ranking:** Rank architectures by CKA score. Due to the alignment inversion, **lower CKA = better ImageNet performance**

The entire scoring pipeline requires only forward passes (no training, no gradients), making it orders of magnitude cheaper than fine-tuning-based selection.

---

## Citation

```bibtex
@inproceedings{bansal2026repnas,
  title={RepNAS: Architecture Selection in Pretrained Model Zoos via Representation Agreement with Foundation Models},
  author={Bansal, Aayam and Gangwani, Ishaan},
  booktitle={Proceedings of the 6th Workshop on Neural Architecture Search (CVPR-NAS'26)},
  year={2026}
}
```
