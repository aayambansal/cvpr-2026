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

**Authors:** Aayam Bansal

**Venue:** 6th Workshop on Neural Architecture Search (CVPR-NAS'26)

**Abstract:** Neural Architecture Search typically assumes a fixed target domain, but real deployments face non-stationary data streams with domain shifts. We propose BudgetNAS, a continual online NAS framework where a deployed model mutates its architecture under a strict per-shift budget: at most B blocks may be added or removed when a domain change is detected. BudgetNAS combines a lightweight accuracy-based trigger with three mutation controllers (fixed, heuristic, and Thompson-sampling bandit) and optional stabilization via warm-starting and selective freezing. On a 3-domain stream (CIFAR-10, CIFAR-100, SVHN), BudgetNAS-Heuristic with B=1 achieves the best efficiency (Gain/M = 20.4) while reducing cross-seed variance by 5.8x compared to unconstrained RandomNAS. We find that (i) B=1 is the budget sweet spot, (ii) stabilization hurts rather than helps, and (iii) the bandit controller excels on gradual domain transitions.

[Paper PDF](BudgetNAS/latex/main.pdf) | [Code & Data](BudgetNAS/)

---
