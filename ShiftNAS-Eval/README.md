# ShiftNAS-Eval: Fair NAS Evaluation Under Distribution Shift

**Aayam Bansal, Ishaan Gangwani**

6th Workshop on Neural Architecture Search at CVPR 2026 (CVPR-NAS'26)

---

## Key Results

- Top-100 architectures on CIFAR-10 overlap only **34%** with CIFAR-100 and **10%** with ImageNet-16-120
- At K=10, ImageNet-16 overlap drops to **0%** -- the best in-distribution architecture is not the best cross-domain
- Top-weighted Kendall-tau is **negative** (tau = -0.41), meaning top clean and top cross-domain rankings are inversely related
- SASC-Pool improves ImageNet-16 generalization by **+0.4--2.8%** across 4 NAS algorithms at zero extra cost
- Real CIFAR-10-C validation on 46 architectures confirms Corruption Gaps up to 51 (rho = 0.58)

## Repository Structure

```
ShiftNAS-Eval/
├── latex/                          # Paper source
│   ├── main.tex                    # Top-level LaTeX file
│   ├── main.pdf                    # Compiled paper
│   ├── main.bib                    # Bibliography (20 references)
│   ├── main.bbl                    # Compiled bibliography
│   ├── preamble.tex                # Macros and packages
│   ├── cvpr.sty                    # CVPR style file
│   ├── ieeenat_fullname.bst        # Bibliography style
│   └── sec/                        # Paper sections
│       ├── abstract.tex
│       ├── introduction.tex
│       ├── related.tex
│       ├── method.tex
│       ├── experiments.tex
│       └── conclusion.tex
├── experiments/                    # Experiment code
│   ├── run_experiments.py          # Full-space simulation (15,625 architectures)
│   ├── generate_figures.py         # Generate all paper figures
│   ├── modal_train_eval.py         # Real CIFAR-10-C training on Modal GPUs
│   └── analyze_real_results.py     # Analysis of real CIFAR-10-C results
├── figures/                        # All paper figures (PDF + PNG)
│   ├── protocol_schematic.png      # Protocol overview diagram
│   ├── protocol_overview.pdf
│   ├── rank_correlation_heatmap.pdf
│   ├── overlap_multi_k.pdf         # Multi-K overlap curves
│   ├── ranking_overlap.pdf
│   ├── strategy_comparison.pdf
│   ├── nas_algorithm_eval.pdf      # End-to-end NAS algorithm evaluation
│   ├── scatter_correlations.pdf
│   ├── corruption_radar.pdf
│   ├── arch_properties.pdf
│   ├── sasc_sensitivity.pdf
│   ├── sasc_stability.pdf
│   ├── real_cifar10c_validation.pdf # Real CIFAR-10-C scatter + per-category
│   └── real_severity_degradation.pdf
├── data/                           # All experimental data
│   ├── architecture_results.csv    # 15,625 architectures with metrics
│   ├── rank_correlations.csv       # Spearman rho between metrics
│   ├── ranking_overlap.json        # Top-K overlap data
│   ├── overlap_multi_k.json        # Multi-K overlap (K=10..500)
│   ├── kendall_tau.json            # Top-weighted Kendall-tau
│   ├── strategy_comparison.csv     # 6 selection strategies compared
│   ├── nas_algorithm_eval.json     # 4 NAS algorithms, clean vs SASC
│   ├── sasc_sensitivity.csv        # SASC weight sensitivity analysis
│   ├── sasc_stability.csv          # SASC-Pool stability vs pool size
│   ├── real_results.json           # 46 real-trained architectures
│   └── real_analysis.json          # Statistics from real CIFAR-10-C
└── ShiftNAS_Eval_CVPR_NAS26.pdf    # Standalone paper PDF
```

## Reproducing Results

### Full-space simulation (no GPU required)

```bash
pip install numpy pandas matplotlib seaborn scipy
cd experiments
python run_experiments.py       # Generates data/ files
python generate_figures.py      # Generates figures/
```

### Real CIFAR-10-C validation (requires GPU)

```bash
pip install modal torch torchvision numpy
cd experiments
modal run modal_train_eval.py   # Trains 46 architectures on CIFAR-10 + CIFAR-10-C
python analyze_real_results.py  # Analyzes results, generates validation figures
```

### Compile paper

```bash
cd latex
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Citation

```bibtex
@inproceedings{bansal2026shiftnas,
  title={ShiftNAS-Eval: Fair NAS Evaluation Under Distribution Shift},
  author={Bansal, Aayam and Gangwani, Ishaan},
  booktitle={Proceedings of the 6th Workshop on Neural Architecture Search at CVPR},
  year={2026}
}
```
