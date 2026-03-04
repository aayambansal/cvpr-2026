# UniSpace: A Unified Modern Vision Backbone Search Space with Factorized Token Mixers

**Aayam Bansal, Ishaan Gangwani**

6th Workshop on Neural Architecture Search at CVPR 2026 (CVPR-NAS'26)

---

## Key Results

- **6.5x10^10** architectures spanning 4 token mixer families (DWConv, Attention, Gated MLP, SSM-Lite), 4 normalizations, 4 downsamplers
- **500 architectures** scored with 4 zero-cost proxies (NASWOT, SynFlow, GradNorm, SNIP)
- Zero-cost proxies **fail** to rank architectures across heterogeneous mixer families: SNIP is the only significant predictor (rho = -0.377, p = 0.040) and its sign is **negative**
- **Cross-resolution consistency**: NASWOT rankings are highly stable (Spearman rho = 0.949 between 32x32 and 16x16)
- **Top-k regret analysis**: Proxy-based selection provides only marginal benefit over random sampling
- **PCA analysis**: 18 of 21 design dimensions needed to capture 90% of variance -- well-covered search space
- **Primitive dominance**: DWConv and GroupNorm show modest but consistent advantage in zero-cost scores
- **Top architectures**: Hybrid mixers with conv in early stages and SSM-Lite in later stages

## Repository Structure

```
UniSpace/
├── latex/                              # Paper source
│   ├── main.tex                        # Top-level LaTeX file
│   ├── main.pdf                        # Compiled paper
│   ├── references.bib                  # Bibliography
│   ├── main.bbl                        # Compiled bibliography
│   ├── cvpr.sty                        # CVPR style file
│   └── ieeenat_fullname.bst            # IEEE bibliography style
├── code/                               # Experiment code
│   ├── search_space.py                 # UniSpace search space (PyTorch)
│   ├── training_free_scores.py         # Zero-cost proxy implementations
│   ├── run_modal.py                    # Scaled experiment on Modal T4 GPU (N=500)
│   ├── run_scaled.py                   # CPU-based scaled experiment (alternative)
│   ├── run_minimal.py                  # Quick validation experiment (N=22)
│   └── generate_figures.py             # Generate all paper figures
├── figures/                            # All paper figures (PDF + PNG)
│   ├── fig_search_space.png            # Architecture schematic
│   ├── fig_primitive_dominance.pdf     # Primitive dominance bar charts
│   ├── fig_score_correlation.pdf       # 5x5 proxy correlation heatmap
│   ├── fig_pca_scree.pdf              # PCA scree + cumulative variance
│   ├── fig_param_efficiency.pdf        # Params vs NASWOT scatter
│   ├── fig_score_distribution.pdf      # NASWOT score histogram
│   ├── fig_regret_analysis.pdf         # Top-k regret + proxy-accuracy bars
│   ├── fig_cross_resolution.pdf        # Cross-resolution scatter (rho=0.949)
│   └── fig_overview.pdf                # 9-panel combined overview
├── results/                            # Experimental data
│   ├── results_v2.json                 # All 500 architecture scores + 30 trained
│   └── analysis_v2.json                # Full statistical analysis
└── UniSpace_CVPR_NAS26.pdf             # Standalone paper PDF
```

## Reproducing Results

### Run on Modal GPU (~5 min, ~$0.05)

```bash
pip install modal torch torchvision numpy
cd code
modal run run_modal.py
```

### Run on CPU (~40 min)

```bash
pip install torch torchvision numpy matplotlib scipy
cd code
python run_scaled.py
```

### Generate figures

```bash
pip install matplotlib numpy scipy
cd code
python generate_figures.py
```

### Compile paper

```bash
cd latex
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Citation

```bibtex
@inproceedings{bansal2026unispace,
  title={UniSpace: A Unified Modern Vision Backbone Search Space with Factorized Token Mixers},
  author={Bansal, Aayam and Gangwani, Ishaan},
  booktitle={Proceedings of the 6th Workshop on Neural Architecture Search at CVPR},
  year={2026}
}
```
