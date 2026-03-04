# CVPR 2026 Papers

A collection of papers submitted to CVPR 2026 workshops and the main conference.

---

## Paper 1: ShiftNAS-Eval

**Title:** ShiftNAS-Eval: Fair NAS Evaluation Under Distribution Shift

**Authors:** Aayam Bansal, Ishaan Gangwani

**Venue:** 6th Workshop on Neural Architecture Search (CVPR-NAS'26)

**Abstract:** NAS methods are evaluated almost exclusively on clean, in-distribution test sets, masking a critical weakness: top-ranked architectures on the search distribution do not remain top-ranked under distribution shift. We propose ShiftNAS-Eval, a multi-shift evaluation protocol across cross-dataset, cross-domain, and corruption robustness axes using 15,625 NAS-Bench-201 architectures. Top-100 overlap between CIFAR-10 and ImageNet-16-120 is just 10% (0% at K=10). We introduce SASC (Shift-Aware Selection Criterion), a pool-based composite metric that improves cross-domain generalization by +0.4--2.8% at zero additional evaluation cost, validated with real CIFAR-10-C experiments.

[Paper PDF](ShiftNAS-Eval/latex/main.pdf) | [Code & Data](ShiftNAS-Eval/)

---

## Paper 2: BudgetNAS

**Title:** BudgetNAS: Continual Online Neural Architecture Search with Budgeted Architecture Mutation

**Authors:** Aayam Bansal, Ishaan Gangwani

**Venue:** 6th Workshop on Neural Architecture Search (CVPR-NAS'26)

**Abstract:** Neural Architecture Search typically assumes a fixed target domain, but real deployments face non-stationary data streams with domain shifts. We propose BudgetNAS, a continual online NAS framework where a deployed model mutates its architecture under a strict per-shift budget: at most B blocks may be added or removed when a domain change is detected. BudgetNAS combines a lightweight accuracy-based trigger with three mutation controllers (fixed, heuristic, and Thompson-sampling bandit) and optional stabilization via warm-starting and selective freezing. On a 3-domain stream (CIFAR-10, CIFAR-100, SVHN), BudgetNAS-Heuristic with B=1 achieves the best efficiency (Gain/M = 20.4) while reducing cross-seed variance by 5.8x compared to unconstrained RandomNAS. We find that (i) B=1 is the budget sweet spot, (ii) stabilization hurts rather than helps, and (iii) the bandit controller excels on gradual domain transitions.

[Paper PDF](BudgetNAS/latex/main.pdf) | [Code & Data](BudgetNAS/)

---

## Paper 3: UniSpace

**Title:** UniSpace: A Unified Modern Vision Backbone Search Space with Factorized Token Mixers

**Authors:** Aayam Bansal, Ishaan Gangwani

**Venue:** 6th Workshop on Neural Architecture Search (CVPR-NAS'26)

**Abstract:** We introduce UniSpace, a factorized neural architecture search space that unifies four families of modern vision token mixers (depthwise convolution, multi-head attention, gated MLP, and a lightweight state-space variant) together with orthogonal choices for normalization layers and spatial downsampling strategies. The resulting combinatorial space contains approximately 6.5×10^10 architectures, yet its factorized structure enables systematic analysis of which primitives matter most. Through a scaled sample-and-evaluate protocol that scores 500 architectures with four zero-cost proxies and trains 30 with short schedules, we show that (i) convolution-based mixers and GroupNorm enjoy a statistically modest but consistent advantage in training-free scores, (ii) zero-cost proxies exhibit weak Spearman rank correlation with short-training accuracy in this heterogeneous space, with SNIP as the only statistically significant predictor (ρ = -0.377, p = 0.040) and notably with a negative sign, (iii) a top-k regret analysis shows that proxy-based selection provides only marginal benefit over random sampling, and (iv) NASWOT rankings are highly stable across input resolutions (Spearman ρ = 0.949), validating resolution-agnostic proxy evaluation.

[Paper PDF](UniSpace/latex/main.pdf) | [Code & Data](UniSpace/)

---

## Paper 4: DenseProxy

**Title:** Space-Aware Proxy Selection for Training-Free Dense Prediction NAS

**Authors:** Aayam Bansal, Ishaan Gangwani

**Venue:** 6th Workshop on Neural Architecture Search (CVPR-NAS'26)

**Abstract:** Zero-cost proxies score untrained networks in seconds, enabling efficient neural architecture search (NAS), yet no single proxy works well across all search space types. We show that space-aware proxy selection, a simple strategy that assigns different proxies to macro-level and micro-level search spaces, substantially outperforms any individual proxy for dense prediction tasks. Evaluating 7,344 architectures across five tasks on TransNAS-Bench-101 with real trained ground truth, our Space-Aware selector achieves Spearman ρ=0.631 for semantic segmentation, a +49% improvement over the best single proxy (GradNorm, ρ=0.423). Similar gains appear for surface normal prediction (+24%), object classification (+16%), and scene classification (+22%). The selector is enabled by our proposed Multi-Scale Feature Separability (MSFS) proxy, which achieves ρ=0.706 for segmentation within macro search spaces. Z-score normalization within each space removes inter-space scale artifacts. We also report Spatial Feature Consistency (SFC) as a negative result with detailed failure analysis. All experiments run on a single A10G GPU in under one hour.

[Paper PDF](DenseProxy/latex/main.pdf) | [Code & Data](DenseProxy/)

---

## Paper 5: MemoryNAS

**Title:** MemoryNAS: Multi-Objective Neural Architecture Search with Peak Memory as a First-Class Constraint

**Authors:** Aayam Bansal, Ishaan Gangwani

**Venue:** 6th Workshop on Neural Architecture Search (CVPR-NAS'26)

**Abstract:** We introduce MemoryNAS, a multi-objective neural architecture search framework that elevates peak GPU memory from a neglected byproduct to a first-class optimization objective alongside accuracy and FLOPs. While hardware-aware NAS methods predominantly optimize for latency or FLOPs, we demonstrate that these proxies correlate poorly with actual peak memory footprint---the binding constraint on memory-limited deployment targets such as edge GPUs, mobile NPUs, and microcontrollers. Our approach integrates an analytical memory estimator into NSGA-III-based search over a MobileNetV2-style space with >10^15 candidate architectures, treating peak memory as both an optimization objective and a hard constraint. We validate the estimator against real GPU measurements on 50 random architectures, achieving a ranking Spearman rho = 0.995---near-perfect ordinal fidelity for guiding search. We train 14 representative architectures on CIFAR-10/100 using an NVIDIA A10G GPU, revealing that architectures selected by FLOPs-only NAS can consume 2.7x more memory than MemoryNAS-selected alternatives at comparable accuracy (e.g., 158 MB vs. 59 MB at ~93% CIFAR-10 accuracy). We show that FLOPs, parameters, and peak memory are fundamentally different axes: an architecture with 4x fewer parameters and FLOPs can use more peak memory than a larger model.

[Paper PDF](MemoryNAS/main.pdf) | [Code & Data](MemoryNAS/)

---

## Paper 6: GreenNAS

**Title:** GreenNAS: Carbon- and Cost-Aware Neural Architecture Search with Validated Energy Proxies

**Authors:** Aayam Bansal, Ishaan Gangwani

**Venue:** 6th Workshop on Neural Architecture Search (CVPR-NAS'26)

**Abstract:** We introduce GreenNAS, a multi-objective neural architecture search framework that replaces FLOPs as the sole efficiency proxy with a composite energy model incorporating GPU power draw, wall-clock time, per-operation memory traffic, and batch-size sensitivity. Using NSGA-II with four simultaneous objectives---accuracy, training energy (Wh), cloud cost ($), and inference latency---we search a NAS-Bench-201-style cell space on CIFAR-10 and CIFAR-100 across five seeds. We validate our energy proxy against real NVML power measurements on NVIDIA T4 and L4 GPUs, achieving Spearman rank correlation ρ = 0.92 (T4) and ρ = 0.74--0.79 (L4) between predicted and measured energy. GreenNAS achieves a 10.5% higher hypervolume than FLOPs-only NAS (0.225 ± 0.020 vs. 0.204 ± 0.018) while consuming 25.6% less energy (1.28 ± 0.01 Wh vs. 1.71 ± 0.06 Wh) at comparable accuracy (86.3 ± 0.5% vs. 86.1 ± 1.3%). Full 200-epoch training of selected architectures on A100 confirms that GreenNAS selections draw 7% lower power than FLOPs-optimized alternatives. All code and measurements are released to facilitate standardized green NAS benchmarking.

[Paper PDF](GreenNAS/GreenNAS_CVPR_NAS26.pdf) | [Code & Data](GreenNAS/)

---
