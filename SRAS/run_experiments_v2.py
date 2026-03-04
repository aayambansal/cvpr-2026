#!/usr/bin/env python3
"""
SRAS Experiments v2: Extended experiments for 10/10 workshop submission
======================================================================
Adds:
  - Dumb-ensemble baselines (avg scores, median rank, pick-best-seed, majority vote)
  - BN recalibration ablation (with/without, sensitivity to B)
  - Independence assumption check (rank noise scaling vs K)
  - Search-space difficulty sweep (varying top-gap)
  - Failure mode analysis (very short warmups, correlated seeds, supernet collapse)
  - Two-stage SRAS with prescreening
  - tau calibration sanity check (distribution of GT correlations)
"""

import os
import json
import numpy as np
from scipy import stats
from collections import Counter

np.set_printoptions(precision=4)

# ============================================================================
# Configuration
# ============================================================================
NUM_ARCHS = 500
NUM_SEEDS = 20
NUM_WARMUPS_K = 5
NOISE_SCALE = 3.5
ARCH_DEPENDENT_NOISE = 1.2
DATA_ORDER_NOISE = 0.8
BN_NOISE = 0.5
WARMUP_NOISE_SCALE = 4.5
GT_MEAN = 73.5
GT_STD = 5.2
GT_SKEW = -0.8
RANDOM_STATE = 2026

# ============================================================================
# Noise Model
# ============================================================================

def generate_ground_truth(n_archs, rng):
    n_good = int(n_archs * 0.85)
    n_bad = n_archs - n_good
    good_accs = rng.normal(GT_MEAN, GT_STD * 0.7, n_good)
    bad_accs = rng.normal(GT_MEAN - 20, GT_STD * 2, n_bad)
    accs = np.concatenate([good_accs, bad_accs])
    rng.shuffle(accs)
    accs = np.clip(accs, 10.0, 94.5)
    return accs


def simulate_supernet_ranking(gt_accs, seed, noise_scale, rng_base, bn_noise_scale=None):
    """Simulate a supernet ranking. bn_noise_scale allows BN ablation."""
    n = len(gt_accs)
    rng = np.random.RandomState(seed * 7919 + 13)
    
    global_bias = rng.normal(0, noise_scale * 0.3)
    arch_sensitivity = rng_base.gamma(2, 0.5, n)
    arch_noise = rng.normal(0, ARCH_DEPENDENT_NOISE, n) * arch_sensitivity
    
    n_groups = 20
    group_effects = rng.normal(0, DATA_ORDER_NOISE, n_groups)
    group_assignments = rng_base.randint(0, n_groups, n)
    data_noise = group_effects[group_assignments]
    
    bn_scale = bn_noise_scale if bn_noise_scale is not None else BN_NOISE
    bn_noise = rng.normal(0, bn_scale, n)
    
    predicted = gt_accs + global_bias + arch_noise + data_noise + bn_noise
    rank_noise = rng.normal(0, noise_scale * 0.15, n)
    predicted += rank_noise
    
    return predicted


def simulate_warmup_ranking(gt_accs, seed, rng_base, bn_noise_scale=None):
    return simulate_supernet_ranking(gt_accs, seed, WARMUP_NOISE_SCALE, rng_base, bn_noise_scale)


def simulate_correlated_warmup(gt_accs, seed, rng_base, correlation=0.0):
    """Warmup with controllable seed correlation (for independence check)."""
    n = len(gt_accs)
    rng_independent = np.random.RandomState(seed * 7919 + 13)
    rng_shared = np.random.RandomState(42)  # shared component
    
    independent_noise = rng_independent.normal(0, WARMUP_NOISE_SCALE, n)
    shared_noise = rng_shared.normal(0, WARMUP_NOISE_SCALE, n)
    
    combined_noise = correlation * shared_noise + (1 - correlation) * independent_noise
    global_bias = rng_independent.normal(0, WARMUP_NOISE_SCALE * 0.3)
    
    return gt_accs + global_bias + combined_noise


# ============================================================================
# Aggregation Methods
# ============================================================================

def sras_zscore_aggregation(score_lists):
    n = len(score_lists[0])
    agg = np.zeros(n)
    for scores in score_lists:
        s = np.array(scores)
        if s.std() > 0:
            s = (s - s.mean()) / s.std()
        agg += s
    return agg

def sras_borda_aggregation(score_lists):
    n = len(score_lists[0])
    borda = np.zeros(n)
    for scores in score_lists:
        borda += np.argsort(np.argsort(scores))
    return borda

def sras_trimmed_aggregation(score_lists, trim=1):
    mat = np.array(score_lists)
    if mat.shape[0] <= 2 * trim:
        return mat.mean(axis=0)
    sorted_mat = np.sort(mat, axis=0)
    trimmed = sorted_mat[trim:-trim]
    return trimmed.mean(axis=0)

# --- NEW: Dumb ensemble baselines ---

def avg_raw_scores(score_lists):
    """Simple average of raw scores (no normalization)."""
    return np.mean(score_lists, axis=0)

def median_raw_scores(score_lists):
    """Median of raw scores."""
    return np.median(score_lists, axis=0)

def median_rank(score_lists):
    """Median rank across runs."""
    ranks = [np.argsort(np.argsort(-np.array(s))) for s in score_lists]
    return -np.median(ranks, axis=0)  # negate so higher = better

def pick_best_seed(score_lists, gt_accs):
    """Oracle pick-best-seed: pick the run whose top-1 has highest GT accuracy.
    This is the upper bound of 'just pick the lucky seed'."""
    best_score = -np.inf
    best_run = None
    for scores in score_lists:
        top1_idx = np.argmax(scores)
        gt_val = gt_accs[top1_idx]
        if gt_val > best_score:
            best_score = gt_val
            best_run = scores
    return np.array(best_run)

def majority_vote_topk(score_lists, k=5):
    """Majority vote over top-k: count how often each arch appears in top-k."""
    n = len(score_lists[0])
    votes = np.zeros(n)
    for scores in score_lists:
        topk = set(np.argsort(scores)[-k:])
        for idx in topk:
            votes[idx] += 1
    return votes


# ============================================================================
# Metrics
# ============================================================================

def kendall_tau(s1, s2):
    tau, p = stats.kendalltau(s1, s2)
    return tau, p

def spearman_rho(s1, s2):
    rho, p = stats.spearmanr(s1, s2)
    return rho, p

def top_k_overlap(s1, s2, k):
    t1 = set(np.argsort(s1)[-k:])
    t2 = set(np.argsort(s2)[-k:])
    return len(t1 & t2) / k

def regret_at_k(gt_accs, predicted_scores, k=1):
    oracle_top = np.sort(gt_accs)[-k:]
    selected_idx = np.argsort(predicted_scores)[-k:]
    selected_accs = gt_accs[selected_idx]
    return oracle_top.mean() - selected_accs.mean()

def pairwise_correlation_matrix(all_scores, metric='tau'):
    n = len(all_scores)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if metric == 'tau':
                mat[i, j], _ = kendall_tau(all_scores[i], all_scores[j])
            else:
                mat[i, j], _ = spearman_rho(all_scores[i], all_scores[j])
    return mat


# ============================================================================
# Experiment 1: Baseline (same as before)
# ============================================================================

def run_experiment_1_baseline(gt_accs, rng_base):
    print("=" * 70)
    print("EXPERIMENT 1: BASELINE - Single-Seed Supernet Rankings")
    print("=" * 70)
    
    all_scores = []
    gt_correlations = {'tau': [], 'rho': []}
    
    for seed in range(NUM_SEEDS):
        scores = simulate_supernet_ranking(gt_accs, seed, NOISE_SCALE, rng_base)
        all_scores.append(scores)
        tau, _ = kendall_tau(gt_accs, scores)
        rho, _ = spearman_rho(gt_accs, scores)
        gt_correlations['tau'].append(tau)
        gt_correlations['rho'].append(rho)
    
    tau_matrix = pairwise_correlation_matrix(all_scores, 'tau')
    rho_matrix = pairwise_correlation_matrix(all_scores, 'rho')
    mask = ~np.eye(NUM_SEEDS, dtype=bool)
    
    topk_matrices = {}
    for k in [1, 3, 5, 10, 20]:
        tk_mat = np.zeros((NUM_SEEDS, NUM_SEEDS))
        for i in range(NUM_SEEDS):
            for j in range(NUM_SEEDS):
                tk_mat[i, j] = top_k_overlap(all_scores[i], all_scores[j], k)
        topk_matrices[k] = tk_mat
    
    regrets = {k: [] for k in [1, 3, 5]}
    for seed in range(NUM_SEEDS):
        for k in [1, 3, 5]:
            regrets[k].append(regret_at_k(gt_accs, all_scores[seed], k))
    
    top1_per_seed = [int(np.argmax(s)) for s in all_scores]
    unique_top1 = len(set(top1_per_seed))
    
    print(f"  GT tau:  {np.mean(gt_correlations['tau']):.4f} +/- {np.std(gt_correlations['tau']):.4f}")
    print(f"  Pairwise tau: {tau_matrix[mask].mean():.4f} +/- {tau_matrix[mask].std():.4f}")
    print(f"  Top-5 overlap: {topk_matrices[5][mask].mean():.4f}")
    print(f"  Unique top-1: {unique_top1}/{NUM_SEEDS}")
    
    return {
        'scores': [s.tolist() for s in all_scores],
        'gt_tau_mean': float(np.mean(gt_correlations['tau'])),
        'gt_tau_std': float(np.std(gt_correlations['tau'])),
        'gt_rho_mean': float(np.mean(gt_correlations['rho'])),
        'gt_rho_std': float(np.std(gt_correlations['rho'])),
        'gt_tau_per_seed': [float(t) for t in gt_correlations['tau']],
        'gt_rho_per_seed': [float(r) for r in gt_correlations['rho']],
        'tau_matrix': tau_matrix.tolist(),
        'rho_matrix': rho_matrix.tolist(),
        'tau_mean': float(tau_matrix[mask].mean()),
        'tau_std': float(tau_matrix[mask].std()),
        'rho_mean': float(rho_matrix[mask].mean()),
        'rho_std': float(rho_matrix[mask].std()),
        'topk_means': {str(k): float(topk_matrices[k][mask].mean()) for k in [1, 3, 5, 10, 20]},
        'topk_stds': {str(k): float(topk_matrices[k][mask].std()) for k in [1, 3, 5, 10, 20]},
        'topk_matrices': {str(k): topk_matrices[k].tolist() for k in [5, 10, 20]},
        'regrets': {str(k): {'mean': float(np.mean(regrets[k])), 'std': float(np.std(regrets[k]))} for k in [1, 3, 5]},
        'unique_top1': unique_top1,
        'top1_per_seed': top1_per_seed,
    }


# ============================================================================
# Experiment 2: SRAS (same as before)
# ============================================================================

def run_experiment_2_sras(gt_accs, rng_base):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: SRAS - K=5 z-score aggregation")
    print("=" * 70)
    
    K = NUM_WARMUPS_K
    all_agg_scores = {'zscore': [], 'borda': [], 'trimmed': []}
    gt_correlations = {'zscore': {'tau': [], 'rho': []},
                       'borda': {'tau': [], 'rho': []},
                       'trimmed': {'tau': [], 'rho': []}}
    
    warmup_data = {}
    
    for meta_seed in range(NUM_SEEDS):
        warmup_scores = []
        for k in range(K):
            ws = meta_seed * 1000 + k
            scores = simulate_warmup_ranking(gt_accs, ws, rng_base)
            warmup_scores.append(scores)
        warmup_data[meta_seed] = [s.tolist() for s in warmup_scores]
        
        zscore_agg = sras_zscore_aggregation(warmup_scores)
        borda_agg = sras_borda_aggregation(warmup_scores)
        trimmed_agg = sras_trimmed_aggregation(warmup_scores)
        
        all_agg_scores['zscore'].append(zscore_agg)
        all_agg_scores['borda'].append(borda_agg)
        all_agg_scores['trimmed'].append(trimmed_agg)
        
        for method, agg in [('zscore', zscore_agg), ('borda', borda_agg), ('trimmed', trimmed_agg)]:
            tau, _ = kendall_tau(gt_accs, agg)
            rho, _ = spearman_rho(gt_accs, agg)
            gt_correlations[method]['tau'].append(tau)
            gt_correlations[method]['rho'].append(rho)
    
    results = {}
    mask = ~np.eye(NUM_SEEDS, dtype=bool)
    
    for method in ['zscore', 'borda', 'trimmed']:
        tau_mat = pairwise_correlation_matrix(all_agg_scores[method], 'tau')
        rho_mat = pairwise_correlation_matrix(all_agg_scores[method], 'rho')
        
        topk_mats = {}
        for k_val in [1, 3, 5, 10, 20]:
            tk = np.zeros((NUM_SEEDS, NUM_SEEDS))
            for i in range(NUM_SEEDS):
                for j in range(NUM_SEEDS):
                    tk[i, j] = top_k_overlap(all_agg_scores[method][i], all_agg_scores[method][j], k_val)
            topk_mats[k_val] = tk
        
        regrets = {k_val: [] for k_val in [1, 3, 5]}
        for ms in range(NUM_SEEDS):
            for k_val in [1, 3, 5]:
                regrets[k_val].append(regret_at_k(gt_accs, all_agg_scores[method][ms], k_val))
        
        top1_list = [int(np.argmax(s)) for s in all_agg_scores[method]]
        unique_top1 = len(set(top1_list))
        
        print(f"  {method}: GT tau={np.mean(gt_correlations[method]['tau']):.4f}  "
              f"Pairwise tau={tau_mat[mask].mean():.4f}  "
              f"Top-5={topk_mats[5][mask].mean():.4f}  "
              f"Unique top-1={unique_top1}")
        
        results[method] = {
            'gt_tau_mean': float(np.mean(gt_correlations[method]['tau'])),
            'gt_tau_std': float(np.std(gt_correlations[method]['tau'])),
            'gt_rho_mean': float(np.mean(gt_correlations[method]['rho'])),
            'gt_rho_std': float(np.std(gt_correlations[method]['rho'])),
            'tau_matrix': tau_mat.tolist(),
            'rho_matrix': rho_mat.tolist(),
            'tau_mean': float(tau_mat[mask].mean()),
            'tau_std': float(tau_mat[mask].std()),
            'rho_mean': float(rho_mat[mask].mean()),
            'rho_std': float(rho_mat[mask].std()),
            'topk_means': {str(k_val): float(topk_mats[k_val][mask].mean()) for k_val in [1, 3, 5, 10, 20]},
            'topk_stds': {str(k_val): float(topk_mats[k_val][mask].std()) for k_val in [1, 3, 5, 10, 20]},
            'topk_matrices': {str(k_val): topk_mats[k_val].tolist() for k_val in [5, 10, 20]},
            'regrets': {str(k_val): {'mean': float(np.mean(regrets[k_val])), 'std': float(np.std(regrets[k_val]))} for k_val in [1, 3, 5]},
            'unique_top1': unique_top1,
            'top1_list': top1_list,
            'scores': [s.tolist() for s in all_agg_scores[method]],
        }
    
    results['warmup_data'] = warmup_data
    return results


# ============================================================================
# Experiment 3: Ablation over K (same as before)
# ============================================================================

def run_experiment_3_ablation_K(gt_accs, rng_base):
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: ABLATION - Number of Warmup Runs K")
    print("=" * 70)
    
    results = {}
    for K in [1, 2, 3, 5, 7, 10]:
        pairwise_taus = []
        gt_taus = []
        gt_rhos = []
        regrets_1 = []
        top1_list = []
        
        for meta_seed in range(NUM_SEEDS):
            warmup_scores = []
            for k in range(K):
                ws = meta_seed * 1000 + k
                scores = simulate_warmup_ranking(gt_accs, ws, rng_base)
                warmup_scores.append(scores)
            
            agg = sras_zscore_aggregation(warmup_scores)
            tau_gt, _ = kendall_tau(gt_accs, agg)
            rho_gt, _ = spearman_rho(gt_accs, agg)
            gt_taus.append(tau_gt)
            gt_rhos.append(rho_gt)
            regrets_1.append(regret_at_k(gt_accs, agg, 1))
            top1_list.append(int(np.argmax(agg)))
        
        for i in range(NUM_SEEDS):
            for j in range(i+1, NUM_SEEDS):
                ws_i = [simulate_warmup_ranking(gt_accs, i*1000+k, rng_base) for k in range(K)]
                ws_j = [simulate_warmup_ranking(gt_accs, j*1000+k, rng_base) for k in range(K)]
                agg_i = sras_zscore_aggregation(ws_i)
                agg_j = sras_zscore_aggregation(ws_j)
                tau_ij, _ = kendall_tau(agg_i, agg_j)
                pairwise_taus.append(tau_ij)
        
        unique_top1 = len(set(top1_list))
        
        results[K] = {
            'gt_tau_mean': float(np.mean(gt_taus)),
            'gt_tau_std': float(np.std(gt_taus)),
            'gt_rho_mean': float(np.mean(gt_rhos)),
            'gt_rho_std': float(np.std(gt_rhos)),
            'pairwise_tau_mean': float(np.mean(pairwise_taus)),
            'pairwise_tau_std': float(np.std(pairwise_taus)),
            'regret_1_mean': float(np.mean(regrets_1)),
            'regret_1_std': float(np.std(regrets_1)),
            'unique_top1': unique_top1,
        }
        print(f"  K={K:2d}: GT tau={np.mean(gt_taus):.4f}  Regret@1={np.mean(regrets_1):.3f}%  Unique={unique_top1}")
    
    return results


# ============================================================================
# Experiment 4: Per-Architecture Variance (same as before)
# ============================================================================

def run_experiment_4_variance(gt_accs, baseline_scores, sras_scores):
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: PER-ARCHITECTURE SCORE VARIANCE")
    print("=" * 70)
    
    bl_mat = np.array(baseline_scores)
    sr_mat = np.array(sras_scores)
    
    bl_norm = np.zeros_like(bl_mat)
    sr_norm = np.zeros_like(sr_mat)
    for i in range(bl_mat.shape[0]):
        bl_norm[i] = (bl_mat[i] - bl_mat[i].mean()) / bl_mat[i].std()
        sr_norm[i] = (sr_mat[i] - sr_mat[i].mean()) / sr_mat[i].std()
    
    bl_var = bl_norm.var(axis=0)
    sr_var = sr_norm.var(axis=0)
    
    bl_ranks = np.array([np.argsort(np.argsort(-s)) for s in baseline_scores])
    sr_ranks = np.array([np.argsort(np.argsort(-s)) for s in sras_scores])
    
    bl_rank_var = bl_ranks.var(axis=0)
    sr_rank_var = sr_ranks.var(axis=0)
    
    gt_ranks = np.argsort(np.argsort(-gt_accs))
    top_mask = gt_ranks < 50
    mid_mask = (gt_ranks >= 150) & (gt_ranks < 350)
    bot_mask = gt_ranks >= 450
    
    print(f"  Baseline rank std: {np.sqrt(bl_rank_var).mean():.2f}")
    print(f"  SRAS rank std:     {np.sqrt(sr_rank_var).mean():.2f}")
    
    return {
        'baseline_score_std_mean': float(np.sqrt(bl_var).mean()),
        'sras_score_std_mean': float(np.sqrt(sr_var).mean()),
        'baseline_rank_std_mean': float(np.sqrt(bl_rank_var).mean()),
        'sras_rank_std_mean': float(np.sqrt(sr_rank_var).mean()),
        'baseline_rank_var_per_arch': bl_rank_var.tolist(),
        'sras_rank_var_per_arch': sr_rank_var.tolist(),
        'strata': {
            'top50': {
                'baseline_rank_std': float(np.sqrt(bl_rank_var[top_mask]).mean()),
                'sras_rank_std': float(np.sqrt(sr_rank_var[top_mask]).mean()),
            },
            'mid200': {
                'baseline_rank_std': float(np.sqrt(bl_rank_var[mid_mask]).mean()),
                'sras_rank_std': float(np.sqrt(sr_rank_var[mid_mask]).mean()),
            },
            'bot50': {
                'baseline_rank_std': float(np.sqrt(bl_rank_var[bot_mask]).mean()),
                'sras_rank_std': float(np.sqrt(sr_rank_var[bot_mask]).mean()),
            },
        },
    }


# ============================================================================
# Experiment 5: Budget-Matched Comparison (same as before)
# ============================================================================

def run_experiment_5_budget_comparison(gt_accs, rng_base):
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: BUDGET-MATCHED COMPARISON")
    print("=" * 70)
    
    results = {}
    
    # Baseline (budget=1.0)
    bl_taus, bl_regrets = [], []
    for s in range(NUM_SEEDS):
        scores = simulate_supernet_ranking(gt_accs, s, NOISE_SCALE, rng_base)
        tau, _ = kendall_tau(gt_accs, scores)
        bl_taus.append(tau); bl_regrets.append(regret_at_k(gt_accs, scores, 1))
    results['baseline_1x'] = {
        'tau_mean': float(np.mean(bl_taus)), 'tau_std': float(np.std(bl_taus)),
        'regret_mean': float(np.mean(bl_regrets)), 'regret_std': float(np.std(bl_regrets)),
        'budget': 1.0,
    }
    
    # Single long training (budget=1.67)
    long_taus, long_regrets = [], []
    long_noise = NOISE_SCALE * 0.8
    for s in range(NUM_SEEDS):
        scores = simulate_supernet_ranking(gt_accs, s, long_noise, rng_base)
        tau, _ = kendall_tau(gt_accs, scores)
        long_taus.append(tau); long_regrets.append(regret_at_k(gt_accs, scores, 1))
    results['long_1.67x'] = {
        'tau_mean': float(np.mean(long_taus)), 'tau_std': float(np.std(long_taus)),
        'regret_mean': float(np.mean(long_regrets)), 'regret_std': float(np.std(long_regrets)),
        'budget': 1.67,
    }
    
    # SRAS K=3 (budget~1.0)
    sras3_taus, sras3_regrets = [], []
    for ms in range(NUM_SEEDS):
        ws = [simulate_warmup_ranking(gt_accs, ms*1000+k, rng_base) for k in range(3)]
        agg = sras_zscore_aggregation(ws)
        tau, _ = kendall_tau(gt_accs, agg)
        sras3_taus.append(tau); sras3_regrets.append(regret_at_k(gt_accs, agg, 1))
    results['sras_k3_1x'] = {
        'tau_mean': float(np.mean(sras3_taus)), 'tau_std': float(np.std(sras3_taus)),
        'regret_mean': float(np.mean(sras3_regrets)), 'regret_std': float(np.std(sras3_regrets)),
        'budget': 1.0,
    }
    
    # SRAS K=5 (budget~1.67)
    sras5_taus, sras5_regrets = [], []
    for ms in range(NUM_SEEDS):
        ws = [simulate_warmup_ranking(gt_accs, ms*1000+k, rng_base) for k in range(5)]
        agg = sras_zscore_aggregation(ws)
        tau, _ = kendall_tau(gt_accs, agg)
        sras5_taus.append(tau); sras5_regrets.append(regret_at_k(gt_accs, agg, 1))
    results['sras_k5_1.67x'] = {
        'tau_mean': float(np.mean(sras5_taus)), 'tau_std': float(np.std(sras5_taus)),
        'regret_mean': float(np.mean(sras5_regrets)), 'regret_std': float(np.std(sras5_regrets)),
        'budget': 1.67,
    }
    
    for name, r in results.items():
        print(f"  {name:<25} budget={r['budget']:.2f}  tau={r['tau_mean']:.4f}  regret={r['regret_mean']:.3f}%")
    
    return results


# ============================================================================
# NEW Experiment 6: Dumb-Ensemble Baselines
# ============================================================================

def run_experiment_6_dumb_ensembles(gt_accs, rng_base):
    """Compare SRAS z-score to naive multi-seed aggregation methods."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: DUMB-ENSEMBLE BASELINES (K=5 warmups each)")
    print("=" * 70)
    
    K = NUM_WARMUPS_K
    methods = {
        'zscore': [],
        'avg_raw': [],
        'median_raw': [],
        'median_rank': [],
        'majority_vote_5': [],
        'majority_vote_10': [],
    }
    gt_corr = {m: {'tau': [], 'rho': []} for m in methods}
    
    for meta_seed in range(NUM_SEEDS):
        warmup_scores = []
        for k in range(K):
            ws = meta_seed * 1000 + k
            scores = simulate_warmup_ranking(gt_accs, ws, rng_base)
            warmup_scores.append(scores)
        
        agg_zscore = sras_zscore_aggregation(warmup_scores)
        agg_avg = avg_raw_scores(warmup_scores)
        agg_median = median_raw_scores(warmup_scores)
        agg_median_rank = median_rank(warmup_scores)
        agg_mv5 = majority_vote_topk(warmup_scores, k=5)
        agg_mv10 = majority_vote_topk(warmup_scores, k=10)
        
        for name, agg in [('zscore', agg_zscore), ('avg_raw', agg_avg),
                          ('median_raw', agg_median), ('median_rank', agg_median_rank),
                          ('majority_vote_5', agg_mv5), ('majority_vote_10', agg_mv10)]:
            methods[name].append(agg)
            tau, _ = kendall_tau(gt_accs, agg)
            rho, _ = spearman_rho(gt_accs, agg)
            gt_corr[name]['tau'].append(tau)
            gt_corr[name]['rho'].append(rho)
    
    results = {}
    for name in methods:
        pairwise_taus = []
        for i in range(NUM_SEEDS):
            for j in range(i+1, NUM_SEEDS):
                tau, _ = kendall_tau(methods[name][i], methods[name][j])
                pairwise_taus.append(tau)
        
        regrets = [regret_at_k(gt_accs, s, 1) for s in methods[name]]
        top1_list = [int(np.argmax(s)) for s in methods[name]]
        unique_top1 = len(set(top1_list))
        
        # Top-5 overlap
        top5_pairs = []
        for i in range(NUM_SEEDS):
            for j in range(i+1, NUM_SEEDS):
                top5_pairs.append(top_k_overlap(methods[name][i], methods[name][j], 5))
        
        results[name] = {
            'gt_tau_mean': float(np.mean(gt_corr[name]['tau'])),
            'gt_tau_std': float(np.std(gt_corr[name]['tau'])),
            'gt_rho_mean': float(np.mean(gt_corr[name]['rho'])),
            'gt_rho_std': float(np.std(gt_corr[name]['rho'])),
            'pairwise_tau_mean': float(np.mean(pairwise_taus)),
            'pairwise_tau_std': float(np.std(pairwise_taus)),
            'top5_overlap_mean': float(np.mean(top5_pairs)),
            'top5_overlap_std': float(np.std(top5_pairs)),
            'regret_1_mean': float(np.mean(regrets)),
            'regret_1_std': float(np.std(regrets)),
            'unique_top1': unique_top1,
        }
        print(f"  {name:<20}: GT tau={results[name]['gt_tau_mean']:.4f}  "
              f"Pairwise tau={results[name]['pairwise_tau_mean']:.4f}  "
              f"Regret@1={results[name]['regret_1_mean']:.3f}%  "
              f"Top-5={results[name]['top5_overlap_mean']:.3f}  "
              f"Unique={unique_top1}")
    
    # Also compute pick-best-seed (needs GT, so it's an oracle baseline)
    pbs_regrets = []
    for meta_seed in range(NUM_SEEDS):
        warmup_scores = []
        for k in range(K):
            ws = meta_seed * 1000 + k
            scores = simulate_warmup_ranking(gt_accs, ws, rng_base)
            warmup_scores.append(scores)
        pbs = pick_best_seed(warmup_scores, gt_accs)
        pbs_regrets.append(regret_at_k(gt_accs, pbs, 1))
    results['pick_best_seed_oracle'] = {
        'regret_1_mean': float(np.mean(pbs_regrets)),
        'regret_1_std': float(np.std(pbs_regrets)),
        'note': 'Oracle: picks the seed whose top-1 has highest GT accuracy. Not achievable in practice.',
    }
    print(f"  {'pick_best_seed':<20}: Regret@1={np.mean(pbs_regrets):.3f}% (oracle, uses GT)")
    
    return results


# ============================================================================
# NEW Experiment 7: BN Recalibration Ablation
# ============================================================================

def run_experiment_7_bn_ablation(gt_accs, rng_base):
    """Ablate BN recalibration: vary BN noise scale (proxy for number of BN passes)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: BN RECALIBRATION ABLATION")
    print("=" * 70)
    
    K = NUM_WARMUPS_K
    # BN noise levels: 0 = perfect recalibration, 0.5 = our default, 1.0/2.0 = no/bad recalib
    bn_levels = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    
    results = {}
    for bn_scale in bn_levels:
        gt_taus = []
        regrets = []
        pairwise_taus = []
        
        for meta_seed in range(NUM_SEEDS):
            warmup_scores = []
            for k in range(K):
                ws = meta_seed * 1000 + k
                scores = simulate_warmup_ranking(gt_accs, ws, rng_base, bn_noise_scale=bn_scale)
                warmup_scores.append(scores)
            
            agg = sras_zscore_aggregation(warmup_scores)
            tau, _ = kendall_tau(gt_accs, agg)
            gt_taus.append(tau)
            regrets.append(regret_at_k(gt_accs, agg, 1))
        
        # Pairwise (subsample for speed)
        for i in range(0, NUM_SEEDS, 2):
            for j in range(i+2, NUM_SEEDS, 2):
                ws_i = [simulate_warmup_ranking(gt_accs, i*1000+k, rng_base, bn_noise_scale=bn_scale) for k in range(K)]
                ws_j = [simulate_warmup_ranking(gt_accs, j*1000+k, rng_base, bn_noise_scale=bn_scale) for k in range(K)]
                agg_i = sras_zscore_aggregation(ws_i)
                agg_j = sras_zscore_aggregation(ws_j)
                tau_ij, _ = kendall_tau(agg_i, agg_j)
                pairwise_taus.append(tau_ij)
        
        results[str(bn_scale)] = {
            'bn_noise_scale': bn_scale,
            'gt_tau_mean': float(np.mean(gt_taus)),
            'gt_tau_std': float(np.std(gt_taus)),
            'pairwise_tau_mean': float(np.mean(pairwise_taus)),
            'pairwise_tau_std': float(np.std(pairwise_taus)),
            'regret_1_mean': float(np.mean(regrets)),
            'regret_1_std': float(np.std(regrets)),
        }
        print(f"  BN_noise={bn_scale:.2f}: GT tau={np.mean(gt_taus):.4f}  "
              f"Pairwise tau={np.mean(pairwise_taus):.4f}  Regret@1={np.mean(regrets):.3f}%")
    
    # Also: baseline with/without BN recalibration (single seed)
    for bn_scale in [0.0, 0.5, 2.0]:
        bl_taus = []
        for s in range(NUM_SEEDS):
            scores = simulate_supernet_ranking(gt_accs, s, NOISE_SCALE, rng_base, bn_noise_scale=bn_scale)
            tau, _ = kendall_tau(gt_accs, scores)
            bl_taus.append(tau)
        results[f'baseline_bn_{bn_scale}'] = {
            'gt_tau_mean': float(np.mean(bl_taus)),
            'gt_tau_std': float(np.std(bl_taus)),
        }
        print(f"  Baseline BN_noise={bn_scale:.1f}: GT tau={np.mean(bl_taus):.4f}")
    
    return results


# ============================================================================
# NEW Experiment 8: Independence Assumption Check
# ============================================================================

def run_experiment_8_independence(gt_accs, rng_base):
    """Test whether rank noise scales as 1/sqrt(K) and when it breaks (correlated seeds)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 8: INDEPENDENCE ASSUMPTION CHECK")
    print("=" * 70)
    
    results = {}
    
    # A) Measure actual noise reduction vs theoretical 1/sqrt(K)
    K_values = [1, 2, 3, 5, 7, 10, 15, 20]
    empirical_noise = []
    
    for K in K_values:
        rank_stds = []
        for meta_seed in range(NUM_SEEDS):
            warmup_scores = []
            for k in range(K):
                ws = meta_seed * 1000 + k
                scores = simulate_warmup_ranking(gt_accs, ws, rng_base)
                warmup_scores.append(scores)
            agg = sras_zscore_aggregation(warmup_scores)
            # Measure: rank error vs GT
            gt_ranks = np.argsort(np.argsort(-gt_accs))
            agg_ranks = np.argsort(np.argsort(-agg))
            rank_error = np.sqrt(np.mean((gt_ranks - agg_ranks) ** 2))
            rank_stds.append(rank_error)
        
        empirical_noise.append({
            'K': K,
            'mean_rmse': float(np.mean(rank_stds)),
            'std_rmse': float(np.std(rank_stds)),
        })
        print(f"  K={K:2d}: Rank RMSE = {np.mean(rank_stds):.2f} +/- {np.std(rank_stds):.2f}")
    
    # Fit: RMSE ~ A / K^alpha. Under independence, alpha = 0.5
    Ks = np.array([e['K'] for e in empirical_noise])
    rmses = np.array([e['mean_rmse'] for e in empirical_noise])
    # Log-log fit: log(RMSE) = log(A) - alpha * log(K)
    log_fit = np.polyfit(np.log(Ks), np.log(rmses), 1)
    alpha = -log_fit[0]
    A = np.exp(log_fit[1])
    print(f"\n  Fit: RMSE ~ {A:.1f} / K^{alpha:.3f} (theoretical: alpha=0.500)")
    
    results['scaling_independent'] = {
        'data': empirical_noise,
        'fit_alpha': float(alpha),
        'fit_A': float(A),
        'theoretical_alpha': 0.5,
    }
    
    # B) Correlated seeds: what happens when seeds share randomness?
    correlations = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    corr_results = []
    K = 5
    
    for corr in correlations:
        gt_taus = []
        for meta_seed in range(NUM_SEEDS):
            warmup_scores = []
            for k in range(K):
                ws = meta_seed * 1000 + k
                scores = simulate_correlated_warmup(gt_accs, ws, rng_base, correlation=corr)
                warmup_scores.append(scores)
            agg = sras_zscore_aggregation(warmup_scores)
            tau, _ = kendall_tau(gt_accs, agg)
            gt_taus.append(tau)
        
        corr_results.append({
            'seed_correlation': corr,
            'gt_tau_mean': float(np.mean(gt_taus)),
            'gt_tau_std': float(np.std(gt_taus)),
        })
        print(f"  Seed correlation={corr:.1f}: GT tau={np.mean(gt_taus):.4f} +/- {np.std(gt_taus):.4f}")
    
    results['seed_correlation'] = corr_results
    
    return results


# ============================================================================
# NEW Experiment 9: Search-Space Difficulty Sweep
# ============================================================================

def run_experiment_9_difficulty(rng_base):
    """Vary search space difficulty (top-gap) and measure SRAS gains."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 9: SEARCH-SPACE DIFFICULTY SWEEP")
    print("=" * 70)
    
    rng = np.random.RandomState(RANDOM_STATE)
    K = 5
    
    # Vary GT_STD to change how distinguishable top architectures are
    # Lower STD = harder to distinguish (small gaps)
    # Higher STD = easier (large gaps)
    gt_stds = [1.0, 2.0, 3.0, 5.2, 8.0, 12.0]
    
    results = {}
    for gt_std_val in gt_stds:
        rng_local = np.random.RandomState(RANDOM_STATE)
        n_good = int(NUM_ARCHS * 0.85)
        n_bad = NUM_ARCHS - n_good
        good_accs = rng_local.normal(GT_MEAN, gt_std_val * 0.7, n_good)
        bad_accs = rng_local.normal(GT_MEAN - 20, gt_std_val * 2, n_bad)
        gt = np.concatenate([good_accs, bad_accs])
        rng_local.shuffle(gt)
        gt = np.clip(gt, 10.0, 94.5)
        
        # Top-gap: difference between #1 and #10 architecture
        sorted_gt = np.sort(gt)
        top_gap = sorted_gt[-1] - sorted_gt[-10]
        
        # Baseline
        bl_taus = []
        for s in range(NUM_SEEDS):
            scores = simulate_supernet_ranking(gt, s, NOISE_SCALE, rng_base)
            tau, _ = kendall_tau(gt, scores)
            bl_taus.append(tau)
        
        # SRAS
        sras_taus = []
        for ms in range(NUM_SEEDS):
            ws = [simulate_warmup_ranking(gt, ms*1000+k, rng_base) for k in range(K)]
            agg = sras_zscore_aggregation(ws)
            tau, _ = kendall_tau(gt, agg)
            sras_taus.append(tau)
        
        # Regrets
        bl_regrets = []
        sras_regrets = []
        for ms in range(NUM_SEEDS):
            bl_scores = simulate_supernet_ranking(gt, ms, NOISE_SCALE, rng_base)
            bl_regrets.append(regret_at_k(gt, bl_scores, 1))
            ws = [simulate_warmup_ranking(gt, ms*1000+k, rng_base) for k in range(K)]
            agg = sras_zscore_aggregation(ws)
            sras_regrets.append(regret_at_k(gt, agg, 1))
        
        gain = np.mean(sras_taus) - np.mean(bl_taus)
        
        results[str(gt_std_val)] = {
            'gt_std': gt_std_val,
            'top_gap': float(top_gap),
            'baseline_tau_mean': float(np.mean(bl_taus)),
            'baseline_tau_std': float(np.std(bl_taus)),
            'sras_tau_mean': float(np.mean(sras_taus)),
            'sras_tau_std': float(np.std(sras_taus)),
            'tau_gain': float(gain),
            'baseline_regret_mean': float(np.mean(bl_regrets)),
            'sras_regret_mean': float(np.mean(sras_regrets)),
            'regret_reduction': float(np.mean(bl_regrets) - np.mean(sras_regrets)),
        }
        print(f"  GT_STD={gt_std_val:<5.1f} top_gap={top_gap:.2f}: "
              f"BL tau={np.mean(bl_taus):.4f}  SRAS tau={np.mean(sras_taus):.4f}  "
              f"gain={gain:+.4f}  BL regret={np.mean(bl_regrets):.2f}%  "
              f"SRAS regret={np.mean(sras_regrets):.2f}%")
    
    return results


# ============================================================================
# NEW Experiment 10: Failure Mode Analysis
# ============================================================================

def run_experiment_10_failure_modes(gt_accs, rng_base):
    """Identify regimes where SRAS fails or degrades."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 10: FAILURE MODE ANALYSIS")
    print("=" * 70)
    
    results = {}
    K = 5
    
    # A) Very short warmups (extreme noise)
    print("\n  A) Very short warmups (increasing noise scale):")
    warmup_noise_scales = [4.5, 6.0, 8.0, 10.0, 15.0, 20.0]
    extreme_results = []
    for wns in warmup_noise_scales:
        gt_taus = []
        for ms in range(NUM_SEEDS):
            warmup_scores = []
            for k in range(K):
                ws = ms * 1000 + k
                scores = simulate_supernet_ranking(gt_accs, ws, wns, rng_base)
                warmup_scores.append(scores)
            agg = sras_zscore_aggregation(warmup_scores)
            tau, _ = kendall_tau(gt_accs, agg)
            gt_taus.append(tau)
        
        extreme_results.append({
            'warmup_noise_scale': wns,
            'gt_tau_mean': float(np.mean(gt_taus)),
            'gt_tau_std': float(np.std(gt_taus)),
        })
        print(f"    noise={wns:5.1f}: GT tau={np.mean(gt_taus):.4f}")
    results['extreme_warmup_noise'] = extreme_results
    
    # B) All seeds use same data order (correlated batch ordering)
    print("\n  B) Correlated data ordering (shared batch effects):")
    # Simulate by adding a large shared noise component
    shared_results = []
    shared_fractions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for sf in shared_fractions:
        gt_taus = []
        rng_shared = np.random.RandomState(999)
        shared_component = rng_shared.normal(0, WARMUP_NOISE_SCALE * 0.5, NUM_ARCHS)
        
        for ms in range(NUM_SEEDS):
            warmup_scores = []
            for k in range(K):
                ws = ms * 1000 + k
                rng_ind = np.random.RandomState(ws * 7919 + 13)
                independent_noise = rng_ind.normal(0, WARMUP_NOISE_SCALE, NUM_ARCHS)
                combined = gt_accs + sf * shared_component + (1 - sf) * independent_noise
                warmup_scores.append(combined)
            agg = sras_zscore_aggregation(warmup_scores)
            tau, _ = kendall_tau(gt_accs, agg)
            gt_taus.append(tau)
        
        shared_results.append({
            'shared_fraction': sf,
            'gt_tau_mean': float(np.mean(gt_taus)),
            'gt_tau_std': float(np.std(gt_taus)),
        })
        print(f"    shared_frac={sf:.1f}: GT tau={np.mean(gt_taus):.4f}")
    results['shared_data_order'] = shared_results
    
    # C) Supernet collapse (all architectures get similar scores)
    print("\n  C) Supernet collapse (score compression):")
    collapse_scales = [1.0, 0.5, 0.2, 0.1, 0.05]  # multiply GT range
    collapse_results = []
    for cs in collapse_scales:
        collapsed_gt = GT_MEAN + (gt_accs - GT_MEAN) * cs
        gt_taus = []
        for ms in range(NUM_SEEDS):
            warmup_scores = []
            for k in range(K):
                ws = ms * 1000 + k
                scores = simulate_warmup_ranking(collapsed_gt, ws, rng_base)
                warmup_scores.append(scores)
            agg = sras_zscore_aggregation(warmup_scores)
            tau, _ = kendall_tau(collapsed_gt, agg)
            gt_taus.append(tau)
        
        collapse_results.append({
            'collapse_scale': cs,
            'effective_std': float(collapsed_gt.std()),
            'gt_tau_mean': float(np.mean(gt_taus)),
            'gt_tau_std': float(np.std(gt_taus)),
        })
        print(f"    collapse={cs:.2f} (eff_std={collapsed_gt.std():.2f}): GT tau={np.mean(gt_taus):.4f}")
    results['supernet_collapse'] = collapse_results
    
    return results


# ============================================================================
# NEW Experiment 11: Two-Stage SRAS (Prescreening)
# ============================================================================

def run_experiment_11_two_stage(gt_accs, rng_base):
    """Two-stage SRAS: cheap prescreen all N, expensive BN recalib only for top-M."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 11: TWO-STAGE SRAS (PRESCREENING)")
    print("=" * 70)
    
    K = NUM_WARMUPS_K
    N = NUM_ARCHS
    
    # M = number of architectures that get full BN recalibration
    M_values = [10, 20, 50, 100, 200, 500]  # 500 = no prescreening (full SRAS)
    
    results = {}
    for M in M_values:
        gt_taus_full = []  # tau over all N (for comparability)
        regrets = []
        top1_list = []
        
        for meta_seed in range(NUM_SEEDS):
            # Stage 1: Cheap evaluation (high BN noise = no recalibration)
            warmup_scores_nobn = []
            for k in range(K):
                ws = meta_seed * 1000 + k
                scores = simulate_warmup_ranking(gt_accs, ws, rng_base, bn_noise_scale=2.0)
                warmup_scores_nobn.append(scores)
            
            agg_stage1 = sras_zscore_aggregation(warmup_scores_nobn)
            
            # Select top-M from stage 1
            top_M_idx = np.argsort(agg_stage1)[-M:]
            
            # Stage 2: BN-recalibrated evaluation only for top-M
            warmup_scores_bn = []
            for k in range(K):
                ws = meta_seed * 1000 + k
                scores = simulate_warmup_ranking(gt_accs, ws, rng_base, bn_noise_scale=0.5)
                warmup_scores_bn.append(scores)
            
            agg_stage2 = sras_zscore_aggregation(warmup_scores_bn)
            
            # Final ranking: for top-M use stage 2, others use stage 1
            final_scores = agg_stage1.copy()
            final_scores[top_M_idx] = agg_stage2[top_M_idx]
            
            tau, _ = kendall_tau(gt_accs, final_scores)
            gt_taus_full.append(tau)
            regrets.append(regret_at_k(gt_accs, final_scores, 1))
            top1_list.append(int(np.argmax(final_scores)))
        
        unique_top1 = len(set(top1_list))
        
        # Compute cost: Stage1 cost + Stage2 cost
        # Stage 1: K warmups * N archs (no BN recalib) = K * N * t_eval
        # Stage 2: K warmups * M archs * (t_eval + B * t_bn_pass)
        # Simplify: relative cost ~ K*N + K*M*(1+B/N_BN_passes)
        # For now, report as fraction of full SRAS cost
        cost_fraction = (N + M) / (2 * N)  # rough approximation
        
        results[str(M)] = {
            'M': M,
            'gt_tau_mean': float(np.mean(gt_taus_full)),
            'gt_tau_std': float(np.std(gt_taus_full)),
            'regret_1_mean': float(np.mean(regrets)),
            'regret_1_std': float(np.std(regrets)),
            'unique_top1': unique_top1,
            'cost_fraction': float(cost_fraction),
        }
        print(f"  M={M:3d}: GT tau={np.mean(gt_taus_full):.4f}  "
              f"Regret@1={np.mean(regrets):.3f}%  Unique={unique_top1}  "
              f"Cost~{cost_fraction:.2f}x full SRAS")
    
    return results


# ============================================================================
# NEW Experiment 12: Tau Calibration Sanity Check
# ============================================================================

def run_experiment_12_tau_calibration(gt_accs, rng_base):
    """Sanity check: show distribution of GT correlations matches literature."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 12: TAU CALIBRATION SANITY CHECK")
    print("=" * 70)
    
    # Run 100 seeds with full training noise to get distribution
    n_calibration_seeds = 100
    tau_gt_values = []
    rho_gt_values = []
    
    for seed in range(n_calibration_seeds):
        scores = simulate_supernet_ranking(gt_accs, seed, NOISE_SCALE, rng_base)
        tau, _ = kendall_tau(gt_accs, scores)
        rho, _ = spearman_rho(gt_accs, scores)
        tau_gt_values.append(tau)
        rho_gt_values.append(rho)
    
    # Also: pairwise between seeds
    pairwise_taus = []
    for i in range(0, 40, 2):
        s_i = simulate_supernet_ranking(gt_accs, i, NOISE_SCALE, rng_base)
        for j in range(i+2, 40, 2):
            s_j = simulate_supernet_ranking(gt_accs, j, NOISE_SCALE, rng_base)
            tau, _ = kendall_tau(s_i, s_j)
            pairwise_taus.append(tau)
    
    # Warmup noise calibration
    warmup_tau_values = []
    for seed in range(n_calibration_seeds):
        scores = simulate_warmup_ranking(gt_accs, seed, rng_base)
        tau, _ = kendall_tau(gt_accs, scores)
        warmup_tau_values.append(tau)
    
    results = {
        'full_training_tau_gt': {
            'values': [float(t) for t in tau_gt_values],
            'mean': float(np.mean(tau_gt_values)),
            'std': float(np.std(tau_gt_values)),
            'min': float(np.min(tau_gt_values)),
            'max': float(np.max(tau_gt_values)),
            'median': float(np.median(tau_gt_values)),
            'q25': float(np.percentile(tau_gt_values, 25)),
            'q75': float(np.percentile(tau_gt_values, 75)),
        },
        'full_training_rho_gt': {
            'values': [float(r) for r in rho_gt_values],
            'mean': float(np.mean(rho_gt_values)),
            'std': float(np.std(rho_gt_values)),
        },
        'full_training_pairwise_tau': {
            'values': [float(t) for t in pairwise_taus],
            'mean': float(np.mean(pairwise_taus)),
            'std': float(np.std(pairwise_taus)),
        },
        'warmup_tau_gt': {
            'values': [float(t) for t in warmup_tau_values],
            'mean': float(np.mean(warmup_tau_values)),
            'std': float(np.std(warmup_tau_values)),
            'min': float(np.min(warmup_tau_values)),
            'max': float(np.max(warmup_tau_values)),
        },
    }
    
    print(f"  Full training tau_GT: {np.mean(tau_gt_values):.4f} +/- {np.std(tau_gt_values):.4f} "
          f"[{np.min(tau_gt_values):.4f}, {np.max(tau_gt_values):.4f}]")
    print(f"  Full training rho_GT: {np.mean(rho_gt_values):.4f} +/- {np.std(rho_gt_values):.4f}")
    print(f"  Full training pairwise tau: {np.mean(pairwise_taus):.4f} +/- {np.std(pairwise_taus):.4f}")
    print(f"  Warmup tau_GT: {np.mean(warmup_tau_values):.4f} +/- {np.std(warmup_tau_values):.4f} "
          f"[{np.min(warmup_tau_values):.4f}, {np.max(warmup_tau_values):.4f}]")
    print(f"\n  Published reference values (Yu et al., 2020):")
    print(f"    - SPOS on NAS-Bench-201: tau_GT ~ 0.3-0.5 (our full training: ~{np.mean(tau_gt_values):.2f})")
    print(f"    - Cross-seed pairwise tau: ~ 0.35-0.55 (our pairwise: ~{np.mean(pairwise_taus):.2f})")
    
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("SRAS EXPERIMENTS v2: EXTENDED FOR 10/10 SUBMISSION")
    print("=" * 70)
    print(f"Architectures: {NUM_ARCHS}")
    print(f"Seeds: {NUM_SEEDS}")
    print(f"SRAS warmups (K): {NUM_WARMUPS_K}")
    print()
    
    rng = np.random.RandomState(RANDOM_STATE)
    gt_accs = generate_ground_truth(NUM_ARCHS, rng)
    
    print(f"Ground truth stats: mean={gt_accs.mean():.2f}, std={gt_accs.std():.2f}, "
          f"min={gt_accs.min():.2f}, max={gt_accs.max():.2f}\n")
    
    rng_base = np.random.RandomState(42)
    
    # Original experiments
    exp1 = run_experiment_1_baseline(gt_accs, rng_base)
    exp2 = run_experiment_2_sras(gt_accs, rng_base)
    exp3 = run_experiment_3_ablation_K(gt_accs, rng_base)
    exp4 = run_experiment_4_variance(
        gt_accs,
        [np.array(s) for s in exp1['scores']],
        [np.array(s) for s in exp2['zscore']['scores']]
    )
    exp5 = run_experiment_5_budget_comparison(gt_accs, rng_base)
    
    # NEW experiments
    exp6 = run_experiment_6_dumb_ensembles(gt_accs, rng_base)
    exp7 = run_experiment_7_bn_ablation(gt_accs, rng_base)
    exp8 = run_experiment_8_independence(gt_accs, rng_base)
    exp9 = run_experiment_9_difficulty(rng_base)
    exp10 = run_experiment_10_failure_modes(gt_accs, rng_base)
    exp11 = run_experiment_11_two_stage(gt_accs, rng_base)
    exp12 = run_experiment_12_tau_calibration(gt_accs, rng_base)
    
    # Save
    all_results = {
        'config': {
            'num_archs': NUM_ARCHS, 'num_seeds': NUM_SEEDS, 'num_warmups': NUM_WARMUPS_K,
            'noise_scale': NOISE_SCALE, 'warmup_noise_scale': WARMUP_NOISE_SCALE,
            'gt_mean': GT_MEAN, 'gt_std': GT_STD,
        },
        'ground_truth': gt_accs.tolist(),
        'experiment_1_baseline': exp1,
        'experiment_2_sras': exp2,
        'experiment_3_ablation_K': {str(k): v for k, v in exp3.items()},
        'experiment_4_variance': exp4,
        'experiment_5_budget': exp5,
        'experiment_6_dumb_ensembles': exp6,
        'experiment_7_bn_ablation': exp7,
        'experiment_8_independence': exp8,
        'experiment_9_difficulty': exp9,
        'experiment_10_failure_modes': exp10,
        'experiment_11_two_stage': exp11,
        'experiment_12_tau_calibration': exp12,
    }
    
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_results_v2.json')
    with open(outpath, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {outpath}")
    print(f"  Total experiments: 12")


if __name__ == '__main__':
    main()
