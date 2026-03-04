# MemoryNAS: Multi-Objective Neural Architecture Search with Peak Memory as a First-Class Constraint

**Authors:** Aayam Bansal, Ishaan Gangwani

**Venue:** 6th Workshop on Neural Architecture Search (CVPR-NAS'26)

## Overview

MemoryNAS is a multi-objective NAS framework that treats peak GPU memory as a first-class optimization objective alongside accuracy and FLOPs. We show that FLOPs-only NAS produces architectures that can consume 2.7x more memory than memory-aware alternatives at comparable trained accuracy.

## Key Results

| Architecture | CIFAR-10 | CIFAR-100 | Params | FLOPs | Peak Mem (bs=32) |
|---|---|---|---|---|---|
| FP-Large | 94.26% | 75.49% | 8.76M | 693M | 127.8 MB |
| FP-Med | 93.37% | 74.26% | 2.25M | 199M | **158.3 MB** |
| FP-HighCap | 93.87% | 75.24% | 3.52M | 381M | 92.4 MB |
| FP-Wide | 93.62% | 73.86% | 2.62M | 287M | 141.6 MB |
| MN-B (ours) | 92.46% | 71.94% | 1.34M | 108M | **58.7 MB** |
| MN-E (ours) | 91.96% | 70.27% | 0.45M | 65M | **40.2 MB** |
| MBv2-1.0 | 93.46% | 74.04% | 2.25M | 199M | 88.2 MB |

- FP-Med (2.25M params, 199M FLOPs) uses **more memory** than FP-Large (8.76M params, 693M FLOPs)
- MN-B achieves 92.5% CIFAR-10 at **63% less memory** than FP-Med (59 MB vs 158 MB)
- Memory estimator achieves **Spearman rho = 0.995** ranking correlation against real GPU measurements

## Structure

```
MemoryNAS/
├── main.tex                    # Paper source (LaTeX, CVPR two-column)
├── main.pdf                    # Compiled paper
├── main.bbl                    # Compiled bibliography
├── references.bib              # BibTeX references (30 entries)
├── figures/
│   ├── fig_memory_validation.pdf   # Memory estimator validation (rho=0.995)
│   ├── fig_training_results.pdf    # Trained accuracy bar chart
│   ├── fig_pareto_trained.pdf      # Accuracy vs memory Pareto front
│   ├── fig_correlation.pdf         # FLOPs/params/memory correlations
│   ├── fig_memory_breakdown.pdf    # Weight vs activation memory breakdown
│   ├── fig_ablation_budget.pdf     # Search budget sensitivity
│   ├── fig_ablation_mem_objective.pdf  # Memory as objective vs constraint
│   └── method_pipeline.png         # Method overview schematic
├── experiments/
│   ├── modal_train.py              # Modal GPU training script (A10G)
│   ├── generate_figures_v2.py      # V2 figure generation
│   ├── training_results.json       # 14 trained architectures (CIFAR-10/100)
│   ├── memory_validation.json      # 50-arch GPU memory validation
│   └── baselines_ablations.json    # NSGA-II/III baselines + ablations
└── tables/
```

## Method

1. **Search Space:** MobileNetV2-style, 7 stages, ~2.2x10^14 architectures (width x depth x expansion x kernel)
2. **Memory Estimator:** Analytical peak activation memory tracking (Eq. 2), validated at Spearman rho = 0.995
3. **Optimizer:** NSGA-III with 3 objectives (accuracy, FLOPs, peak memory) + optional hard memory constraints
4. **Training:** 14 architectures trained on CIFAR-10 and CIFAR-100 (50 epochs, SGD+cosine, AMP, batch 128) on NVIDIA A10G

## Reproducing

All GPU experiments run on [Modal](https://modal.com) serverless infrastructure:

```bash
# Deploy and run experiments
pip install modal
cd experiments/
modal deploy modal_train.py
python3 -c "import modal; modal.Function.from_name('memorynas-experiments', 'orchestrate').spawn()"

# Generate figures from results
python3 generate_figures_v2.py
```

## Citation

```bibtex
@inproceedings{bansal2026memorynasworkshop,
  title     = {{MemoryNAS}: Multi-Objective Neural Architecture Search with Peak Memory as a First-Class Constraint},
  author    = {Bansal, Aayam and Gangwani, Ishaan},
  booktitle = {6th Workshop on Neural Architecture Search (CVPR-NAS)},
  year      = {2026}
}
```
