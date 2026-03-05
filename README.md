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

## Paper 7: AdapterNAS

**Title:** AdapterNAS: Training-Free Neural Architecture Search for Foundation Model Adapter Topologies

**Authors:** Aayam Bansal, Ishaan Gangwani

**Venue:** 6th Workshop on Neural Architecture Search (CVPR-NAS'26)

**Abstract:** Parameter-efficient fine-tuning (PEFT) methods like LoRA adapt foundation models by inserting low-rank modules, yet the adapter topology---which layers, modules, and ranks to use---is typically chosen by hand. We propose AdapterNAS, a training-free NAS framework that searches this adapter topology using zero-cost proxies. Given a pre-trained Vision Transformer, AdapterNAS scores candidate LoRA configurations via an ensemble of GradNorm, SNIP, Fisher, and entropy proxies on a single calibration batch, then refines the search space with proxy-guided evolutionary optimization. On CIFAR-100 with only 2% training data and ViT-B/16, AdapterNAS searches 175 configurations in under 15 minutes on a single GPU, with the proxy-selected top-5 always containing the oracle-best configuration (79.3% accuracy, vs. 60.0% linear probe). GradNorm alone achieves Spearman rho = 0.62 with downstream accuracy (p < 0.001). We validate across backbones (ViT-S/16), data regimes (1%--10%, up to 89.0% at 10% data), and datasets (Flowers-102, where all configs achieve >99%), showing consistent proxy-guided selection quality. Our analysis reveals that adapter topology significantly impacts performance---a 19 percentage point spread across configurations---and that MLP adaptation, typically ignored, contributes as much as attention adaptation.

[Paper PDF](AdapterNAS/main.pdf) | [Code & Data](AdapterNAS/)

---

## Paper 8: SRAS

**Title:** SRAS: Seed-Robust Architecture Selection for Reproducible One-Shot Neural Architecture Search

**Authors:** Aayam Bansal, Ishaan Gangwani

**Venue:** 6th Workshop on Neural Architecture Search (CVPR-NAS'26)

**Abstract:** One-shot neural architecture search (NAS) methods train a shared supernet to amortize evaluation cost, but we show that the resulting architecture rankings are alarmingly sensitive to the random seed. Across 20 independent supernet runs on a NAS-Bench-201-calibrated search space, the pairwise Kendall's tau between rankings averages only 0.71, top-5 overlap is 29%, and 10 distinct "best" architectures are selected---meaning the chosen architecture is largely a seed artifact. We propose SRAS (Seed-Robust Architecture Selection), which replaces one long supernet run with K short warmup runs under diverse seeds, followed by z-score normalized rank aggregation and batch normalization (BN) recalibration. At equal compute budget, SRAS raises pairwise tau to 0.85 (+0.15), top-5 overlap to 69% (+40pp), and cuts top-1 regret from 1.53% to 0.27%. We provide extensive ablations: aggregation strategies (including naive baselines that reveal the contribution of ensembling vs. smart normalization), BN recalibration sensitivity, the independence assumption behind 1/sqrt(K) noise reduction, search-space difficulty sweeps, failure modes, and a two-stage prescreening variant that reduces evaluation cost by 45% while preserving 99% of the accuracy gain.

[Paper PDF](SRAS/main.pdf) | [Code & Data](SRAS/)

---

## Paper 9: RepNAS

**Title:** RepNAS: Architecture Selection in Pretrained Model Zoos via Representation Agreement with Foundation Models

**Authors:** Aayam Bansal, Ishaan Gangwani

**Venue:** 6th Workshop on Neural Architecture Search (CVPR-NAS'26)

**Abstract:** Choosing which pretrained architecture to deploy from a model zoo currently requires expensive per-task fine-tuning of every candidate. We propose RepNAS, a zero-shot selection method that scores pretrained architectures by their representation agreement with foundation models using Centered Kernel Alignment (CKA). Evaluating 83 torchvision architectures against four foundation-model teachers (DINOv2, CLIP, MAE, ConvNeXtV2-FCMAE), we discover an *alignment inversion*: architectures whose representations are most similar to the teacher consistently rank worst on ImageNet (Spearman rho = -0.42 to -0.52 across teachers, all bootstrap CIs excluding zero). Partial-correlation and subgroup analyses confirm this is not a size or family artifact (CLIP partial rho = -0.579 controlling for parameters; between-family rho = -0.557). We exploit the inversion for architecture search: selecting the least-similar architecture yields 85.1% ImageNet top-1 with 0.82% regret vs. the oracle. Remarkably, the same CKA scores positively predict transfer to CIFAR-100 (rho = +0.826) and Flowers-102 (rho = +0.518), while ImageNet accuracy alone does not (rho = -0.034). Layer-wise analysis reveals a sign flip from positive correlation in early layers (rho = +0.370) to negative in late layers (rho = -0.377), suggesting the inversion originates in task-specific head representations.

[Paper PDF](RepNAS/latex/main.pdf) | [Code & Data](RepNAS/)

---

## Paper 10: ActiveNAS

**Title:** ActiveNAS: Data-Efficient Neural Architecture Search via Active Subset Selection

**Authors:** Aayam Bansal, Ishaan Gangwani

**Venue:** 6th Workshop on Neural Architecture Search (CVPR-NAS'26)

**Abstract:** Neural architecture search (NAS) requires training each candidate on the full training set, making the search computationally expensive. We investigate whether principled data subset selection can preserve architecture rankings at reduced cost. We propose ActiveNAS, which combines submodular facility location with gradient-space diversity and class-balance weighting, and evaluate it alongside random sampling, stratified sampling, CRAIG, and component ablations on 48 CNN architectures with three random seeds. Our multi-seed evaluation reveals a surprising finding: simple baselines are remarkably hard to beat. Random and stratified sampling achieve Spearman ρ ≥ 0.86 at all tested fractions (1–3%), while feature-aware methods (facility location, gradient diversity, ActiveNAS) show substantially higher variance across seeds and generally lower correlation. To exploit these regime-dependent dynamics, we introduce ActiveNAS+, a meta-policy that dispatches to the best-performing strategy based on per-class sample budget, achieving robust performance across all fractions. Our systematic study provides actionable guidance for practitioners and demonstrates that the gap between sophisticated and simple subset selection is smaller than previously assumed, with 9–56× speedups achievable using simple stratified sampling at 1–2% data budgets while maintaining ρ > 0.90.

[Paper PDF](ActiveNAS/latex/main.pdf) | [Code & Data](ActiveNAS/)

---

## Paper 11: DualProc

**Title:** DualProc: Dual-Process Prompting Reduces Confident Errors in Vision-Language Models for Grounded Retrieval and Agentic Pipelines

**Authors:** Aayam Bansal

**Venue:** CogVL: Cognitive Foundations for Multimodal Models Workshop (CVPR 2026)

**Abstract:** Vision-language models (VLMs) frequently produce confident errors—incorrect answers accompanied by high self-reported confidence—undermining their reliability in grounded retrieval and agentic pipelines. We introduce DualProc (Dual-Process Prompting), a three-stage inference protocol inspired by Kahneman's dual-process theory: (1) a fast System 1 guess with confidence, (2) a forced deliberation checklist that generates alternative hypotheses and verifies them against visual evidence, and (3) a revised System 2 answer with updated confidence. Across 500 visual reasoning items spanning five categories and five VLMs (GPT-4o-mini, Gemini-2.0-Flash, Claude-3.5-Sonnet, LLaVA-1.6-34B, InternVL2-26B), DualProc reduces the confident error rate by 83–100% while maintaining or improving accuracy. In downstream applications, DualProc improves evidence precision by 8% while reducing confident errors by 86–93% in retrieval, and reduces tool misuse by 75–95% while increasing task completion by 3–20 pp in agentic loops. We further introduce adaptive DualProc, which conditionally triggers deliberation only when System 1 confidence exceeds a threshold, achieving 98% of the calibration benefit at 97% of the token cost.

[Paper PDF](DualProc/latex/main.pdf) | [Code & Data](DualProc/)

---

## Paper 12: MetamorphicVLM

**Title:** MetamorphicVLM: Probing Vision-Language Model Robustness Through Metamorphic Testing

**Authors:** Aayam Bansal

**Venue:** CogVL: Cognitive Foundations for Multimodal Models Workshop (CVPR 2026)

**Abstract:** We introduce MetamorphicVLM, a systematic metamorphic testing framework for evaluating the robustness of vision-language models (VLMs) to semantics-preserving image transformations. Our framework applies six families of image transformations (resize, crop, rotation, JPEG compression, Gaussian blur, and border text overlay) at four severity levels to a controlled test suite of 60 images spanning six visual reasoning categories, yielding 3,000 total evaluations across two open-source VLMs. We propose the Metamorphic Consistency Index (MCI), a novel metric that quantifies the fraction of semantics-preserving transformations under which a model's answer remains stable, and find that LLaVA-v1.6-Mistral-7B achieves an MCI of 96.5% while Qwen2-VL-2B-Instruct achieves 94.4%, revealing that even state-of-the-art VLMs change their answers on 3.5–5.6% of trivially transformed inputs. Most remarkably, we discover that certain transformations such as resizing and blur can improve Qwen2-VL accuracy from 75.0% to 83.3%, suggesting that VLMs have not learned human-like perceptual invariances but instead exploit fragile, resolution-dependent features.

[Paper PDF](MetamorphicVLM/latex/main.pdf) | [Code & Data](MetamorphicVLM/)

---

## Paper 13: BeliefRevision

**Title:** Do Vision-Language Models Revise Beliefs or Just Rationalize? Evidence Update Prompting for Non-Monotonic Visual Reasoning

**Authors:** Aayam Bansal

**Venue:** CogVL: Cognitive Foundations for Multimodal Models Workshop (CVPR 2026)

**Abstract:** When new visual evidence contradicts an initial interpretation, do vision-language models (VLMs) genuinely revise their beliefs, or do they merely rationalize their first guess? We introduce Evidence Update Prompting (EUP), a two-phase evaluation protocol inspired by defeasible and non-monotonic reasoning from cognitive science. In Phase A, a model receives limited pre-event evidence and forms an initial hypothesis; in Phase B, additional post-event evidence arrives that often requires the model to revise. We compare three prompting strategies—Baseline, Belief-State (explicit hypothesis tracking with confidence), and Counterfactual Update ("would your answer differ without the new evidence?")—across three frontier VLMs (GPT-4o, Gemini 2.0 Flash, Claude 3.5 Sonnet) on 52 BlackSwan-style scenarios requiring abductive reasoning about surprising events. Our findings reveal that (i) all models exhibit substantial stubbornness: 37–62% of initially incorrect answers are never revised despite conflicting evidence; (ii) Belief-State prompting reduces stubbornness by 13–18 percentage points and increases accuracy by 4–8 pp over baseline; (iii) Counterfactual prompting helps models recognize when evidence matters but produces only modest behavioral change; and (iv) models display striking confidence inflation in Phase B, with high-confidence predictions rising 2–3x regardless of whether the answer actually changed.

[Paper PDF](BeliefRevision/latex/main.pdf) | [Code & Data](BeliefRevision/)

---

## Paper 14: CounterBench

**Title:** CounterBench: A Cheap, Controllable Counterfactual Testbed Reveals Systematic Failures in Vision-Language Models

**Authors:** Aayam Bansal, Ishaan Gangwani

**Venue:** Workshop on Grounded Retrieval and Agentic Intelligence for Vision-Language (CVPR 2026)

**Abstract:** Standard vision-language model (VLM) benchmarks evaluate whether a model can answer a question correctly given a single image, but they cannot distinguish genuine visual understanding from spurious pattern matching. We introduce CounterBench, a counterfactual consistency benchmark that pairs each synthetic scene with a minimal intervention---removing an object, swapping positions, or changing an attribute---and checks whether the model's answer changes if and only if it should. CounterBench comprises 325 programmatically generated image pairs spanning five reasoning categories (spatial, causal, compositional, counting, and occlusion) and nine intervention types, all produced with a fully deterministic PIL pipeline at zero annotation cost. We evaluate six VLMs---three proprietary (GPT-4o, Claude 3.5 Sonnet, Gemini 2.0 Flash) and three open-weight (Qwen2.5-VL 72B, Pixtral Large, Llama 3.2 11B)---using the Counterfactual Consistency Score (CCS), which measures the fraction of pairs where the model's answer changes only when the ground-truth answer changes. Results reveal a striking 19.5-point gap in CCS across models (99.7% for Gemini 2.0 Flash vs. 80.2% for Llama 3.2 11B), even though all models exceed 82% accuracy on individual images. We decompose CCS into sensitivity (correct change detection) and specificity (correct invariance), uncovering distinct failure profiles: Claude 3.5 Sonnet suffers from low specificity (spurious answer changes), while Llama 3.2 11B suffers from low sensitivity (sticky answers). Notably, Qwen2.5-VL 72B achieves 97.5% CCS with perfect specificity, rivaling proprietary models.

[Paper PDF](CounterBench/latex/main.pdf) | [Code & Data](CounterBench/)

---
