# BudgetNAS: Continual Online Neural Architecture Search with Budgeted Architecture Mutation

**Aayam Bansal, Ishaan Gangwani**

6th Workshop on Neural Architecture Search at CVPR 2026 (CVPR-NAS'26)

---

## Key Results

- BudgetNAS-Heuristic (B=1) achieves **42.7%** CIFAR-10, **11.6%** CIFAR-100, **35.6%** SVHN with only **1.47M** params
- Best efficiency: **Gain/M = 20.4**, highest among all architecture-changing methods
- **5.8x variance reduction** vs RandomNAS on SVHN (1.3% vs 7.5% std across 3 seeds)
- Stabilization (warm-start, freezing) **hurts** -- none > ws > fz > ws+fz (negative result)
- Drift detection fires **0 times** -- all mutations triggered by accuracy monitoring
- Budget sweet spot at **B=1**; B>=3 degrades performance due to over-mutation
- Gradual shift: **+4.0%** over Fixed architecture on final domain with blended transitions

## Repository Structure

```
BudgetNAS/
├── latex/                              # Paper source
│   ├── main.tex                        # Top-level LaTeX file
│   ├── main.pdf                        # Compiled paper
│   ├── references.bib                  # Bibliography (25 references)
│   ├── main.bbl                        # Compiled bibliography
│   └── cvpr.sty                        # CVPR style file
├── experiments/                        # Experiment code
│   ├── run_v3.py                       # Main v3 experiment script (all experiments)
│   ├── plot_results_v3.py              # Generate all v3 paper figures
│   ├── experiment.py                   # v1 experiment script (historical)
│   ├── experiment_v2.py                # v2 experiment script (historical)
│   ├── run_all_v2.py                   # v2 batch runner (historical)
│   ├── run_continue.py                 # v2 continuation helper (historical)
│   ├── plot_results.py                 # v1 plotting (historical)
│   └── plot_results_v2.py              # v2 plotting (historical)
├── figures/                            # All paper figures (PDF + PNG)
│   ├── fig_method_overview.png         # Method overview schematic
│   ├── fig2_accuracy_timeline.pdf      # Accuracy over streaming chunks (7 methods)
│   ├── fig3_final_comparison.pdf       # Bar chart: 10 methods x 3 datasets
│   ├── fig4_budget_sweep.pdf           # Budget B=0..5 accuracy + params
│   ├── fig5_trigger_ablation.pdf       # Drift-only / acc-only / both / neither
│   ├── fig6_stabilization_ablation.pdf # none / ws / fz / ws+fz
│   ├── fig7_gradual_shift.pdf          # Fixed vs Heuristic vs Bandit (blended)
│   ├── fig8_long_stream.pdf            # 4-domain stream (+ FashionMNIST)
│   └── fig9_efficiency.pdf             # Accuracy vs params + Gain/M ranking
├── data/                               # All experimental data
│   ├── v3_results.json                 # All v3 experiment results
│   ├── v3_summary_table.txt            # Human-readable summary table
│   ├── v2_results.json                 # v2 results (historical)
│   ├── v2_summary_table.txt            # v2 summary (historical)
│   └── experiment_results.json         # v1 results (historical)
└── BudgetNAS_CVPR_NAS26.pdf            # Standalone paper PDF
```

## Reproducing Results

### Run all experiments (CPU, ~40 min)

```bash
pip install torch torchvision numpy matplotlib
cd experiments
python run_v3.py              # Runs all v3 experiments (10 methods x 3 seeds + ablations)
python plot_results_v3.py     # Generates all figures
```

### Compile paper

```bash
cd latex
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Citation

```bibtex
@inproceedings{bansal2026budgetnas,
  title={BudgetNAS: Continual Online Neural Architecture Search with Budgeted Architecture Mutation},
  author={Bansal, Aayam and Gangwani, Ishaan},
  booktitle={Proceedings of the 6th Workshop on Neural Architecture Search at CVPR},
  year={2026}
}
```
