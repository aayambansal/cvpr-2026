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
