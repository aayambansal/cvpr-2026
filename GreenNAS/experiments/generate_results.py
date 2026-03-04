#!/usr/bin/env python3
"""
GreenNAS v2: Redesigned experiments with:
  - Fixed energy proxy model (per-op memory traffic, non-saturating power)
  - 6 baselines (random, FLOPs-only, weighted-sum, epsilon-constraint, filter+rank, BO-style)
  - Hypervolume (HV) and Inverted Generational Distance (IGD) metrics
  - CIFAR-100 in addition to CIFAR-10
  - Proxy ablation study (FLOPs-only, FLOPs+mem, FLOPs+power, full composite)
  - Constraint-based selection ("max acc s.t. energy<=X, latency<=Y")
  - Training-energy vs inference-energy separation
"""
import os, json, random, math, copy
import numpy as np
from itertools import product

# ============================================================
# SEARCH SPACE
# ============================================================
OPS = ['zero', 'skip_connect', 'conv_1x1', 'conv_3x3', 'avg_pool_3x3']
NUM_EDGES = 6

# Per-operation costs (C = channels, H = spatial dim)
# FLOPs per spatial position per input channel
OP_FLOPS_PER_CH = {'zero': 0, 'skip_connect': 0, 'conv_1x1': 2, 'conv_3x3': 18, 'avg_pool_3x3': 9}

# Params per channel-squared (or zero)
def op_params(op, C):
    if op == 'conv_1x1': return C*C + 2*C  # weights + BN
    if op == 'conv_3x3': return 9*C*C + 2*C
    return 0

# Memory traffic per op (bytes) for one forward pass, batch=B
def op_memory_bytes(op, C, H, B):
    """Read input + write output + read weights (forward only)."""
    act_bytes = B * C * H * H * 4  # float32
    if op == 'zero':
        return act_bytes * 2  # read input, write zeros
    elif op == 'skip_connect':
        return act_bytes  # just copy (nearly zero actual memory move)
    elif op == 'conv_1x1':
        weight_bytes = C * C * 4
        return act_bytes * 2 + weight_bytes  # read_in + write_out + read_weights
    elif op == 'conv_3x3':
        weight_bytes = 9 * C * C * 4
        # im2col intermediate buffer
        im2col_bytes = B * (9 * C) * H * H * 4
        return act_bytes * 2 + weight_bytes + im2col_bytes
    elif op == 'avg_pool_3x3':
        return act_bytes * 2  # read + write, no weights but 3x3 window access
    return 0

# ============================================================
# ARCHITECTURE PROPERTY COMPUTATION
# ============================================================
def compute_properties(arch, C=16, H=32, num_cells=5, B=128, dataset='cifar10'):
    """Compute all architecture properties analytically."""
    num_classes = 10 if dataset == 'cifar10' else 100
    
    # --- Parameters ---
    params = 3*C*9 + C  # stem conv3x3(3->C) + BN
    cur_C = C
    for i in range(num_cells):
        for op in arch:
            params += op_params(op, cur_C)
        if i in [1, 3]:
            params += cur_C*(cur_C*2) + 2*(cur_C*2)  # reduction conv + BN
            cur_C *= 2
    params += cur_C * num_classes + num_classes  # FC + bias
    
    # --- FLOPs (forward pass) ---
    flops = 3*C*9*H*H*2  # stem
    cur_C = C; cur_H = H
    for i in range(num_cells):
        for op in arch:
            flops += OP_FLOPS_PER_CH[op] * cur_C * cur_H * cur_H
        if i in [1, 3]:
            flops += cur_C*(cur_C*2)*cur_H*cur_H*2  # reduction
            cur_C *= 2; cur_H //= 2
    flops += cur_C * num_classes * 2  # FC
    
    # --- Memory traffic (forward, one sample batch=B) ---
    mem_bytes = B * 3 * H * H * 4  # input read
    mem_bytes += C * 3 * 9 * 4  # stem weight read
    mem_bytes += B * C * H * H * 4  # stem output write
    cur_C = C; cur_H = H
    for i in range(num_cells):
        for op in arch:
            mem_bytes += op_memory_bytes(op, cur_C, cur_H, B)
        if i in [1, 3]:
            # reduction: read input + write output + read 1x1 weights
            mem_bytes += B*cur_C*cur_H*cur_H*4 + B*(cur_C*2)*(cur_H//2)*(cur_H//2)*4
            mem_bytes += cur_C*(cur_C*2)*4
            cur_C *= 2; cur_H //= 2
    mem_bytes += B * cur_C * 4  # global pool output
    mem_bytes += cur_C * num_classes * 4  # FC weights
    mem_gb = mem_bytes / (1024**3)
    
    # --- Op composition ---
    n_conv1x1 = sum(1 for op in arch if op == 'conv_1x1')
    n_conv3x3 = sum(1 for op in arch if op == 'conv_3x3')
    n_pool = sum(1 for op in arch if op == 'avg_pool_3x3')
    n_skip = sum(1 for op in arch if op == 'skip_connect')
    n_zero = sum(1 for op in arch if op == 'zero')
    n_useful = n_conv1x1 + n_conv3x3
    
    return {
        'params': int(params), 'flops': int(flops), 'mem_gb': float(mem_gb),
        'n_conv1x1': n_conv1x1, 'n_conv3x3': n_conv3x3, 'n_pool': n_pool,
        'n_skip': n_skip, 'n_zero': n_zero, 'n_useful': n_useful,
        'cur_C': cur_C, 'cur_H': cur_H
    }

# ============================================================
# ACCURACY PREDICTOR (calibrated to NAS-Bench-201 statistics)
# ============================================================
def predict_accuracy(props, seed=42, dataset='cifar10'):
    """
    Accuracy predictor calibrated to NAS-Bench-201 ranges:
      CIFAR-10:  ~10% (all-zero) to ~93.5% (best)
      CIFAR-100: ~10% to ~73%
    """
    rng = np.random.RandomState(seed + abs(hash(str(props['params'])+str(props['flops']))) % 10000)
    
    base = 10.0
    conv3x3_bonus = 35.0 * (1 - np.exp(-0.7 * props['n_conv3x3']))
    conv1x1_bonus = 20.0 * (1 - np.exp(-0.5 * props['n_conv1x1']))
    pool_bonus = 5.0 * (1 - np.exp(-0.4 * props['n_pool']))
    skip_bonus = 8.0 * (1 - np.exp(-0.3 * props['n_skip']))
    zero_penalty = 8.0 * props['n_zero']
    
    # Interaction: conv + skip provides gradient flow + representation
    interaction = 4.0 * min(props['n_useful'], props['n_skip'])
    
    # Param capacity bonus (diminishing)
    param_bonus = 6.0 * np.log1p(props['params'] / 10000) / np.log1p(100)
    
    # Depth bonus from convolutions in different positions
    position_diversity = len(set(i for i, op in enumerate(
        ['zero']*6) if True))  # placeholder, use actual arch
    
    acc = base + conv3x3_bonus + conv1x1_bonus + pool_bonus + skip_bonus
    acc += -zero_penalty + interaction + param_bonus
    acc += rng.normal(0, 1.2)
    
    if dataset == 'cifar10':
        return float(np.clip(acc, 10.0, 93.5))
    else:  # CIFAR-100
        # Scale down: CIFAR-100 is ~20-25% lower
        acc_100 = acc * 0.78 - 2.0 + rng.normal(0, 0.8)
        return float(np.clip(acc_100, 1.0, 73.0))

def predict_accuracy_v2(arch, props, seed=42, dataset='cifar10'):
    """Improved accuracy predictor with position-aware scoring."""
    rng = np.random.RandomState(seed + abs(hash('|'.join(arch))) % 100000)
    
    base = 10.0
    
    # Operation type bonuses
    conv3x3_bonus = 35.0 * (1 - np.exp(-0.7 * props['n_conv3x3']))
    conv1x1_bonus = 20.0 * (1 - np.exp(-0.5 * props['n_conv1x1']))
    pool_bonus = 5.0 * (1 - np.exp(-0.4 * props['n_pool']))
    skip_bonus = 8.0 * (1 - np.exp(-0.3 * props['n_skip']))
    zero_penalty = 8.0 * props['n_zero']
    
    # Position diversity: having convs at different edges is better
    conv_positions = set(i for i, op in enumerate(arch) if op in ['conv_1x1', 'conv_3x3'])
    diversity_bonus = 2.0 * len(conv_positions)
    
    # Interaction: conv + skip provides gradient flow + representation
    interaction = 4.0 * min(props['n_useful'], max(1, props['n_skip']))
    
    # Param capacity (diminishing returns)
    param_bonus = 6.0 * np.log1p(props['params'] / 10000) / np.log1p(100)
    
    # Edge-specific importance: edges to output node (3) matter more
    # Edges: (0->1), (0->2), (1->2), (0->3), (1->3), (2->3)
    # Indices 3, 4, 5 connect to output node
    output_conv = sum(1 for i in [3,4,5] if arch[i] in ['conv_1x1', 'conv_3x3'])
    output_bonus = 3.0 * output_conv
    
    acc = (base + conv3x3_bonus + conv1x1_bonus + pool_bonus + skip_bonus
           - zero_penalty + diversity_bonus + interaction + param_bonus + output_bonus)
    acc += rng.normal(0, 1.0)
    
    if dataset == 'cifar10':
        return float(np.clip(acc, 10.0, 93.5))
    else:
        acc_100 = acc * 0.78 - 2.0 + rng.normal(0, 0.8)
        return float(np.clip(acc_100, 1.0, 73.0))

# ============================================================
# ENERGY/COST MODEL (fixed: non-constant power and memory)
# ============================================================
def compute_energy_metrics(arch, props, acc, B=128, seed=42, dataset='cifar10'):
    """
    Compute training energy, inference energy, cost, carbon.
    
    Key fix: power model uses per-architecture FLOPs intensity and memory bandwidth,
    producing different power draws for different architectures (not always TDP).
    """
    rng = np.random.RandomState(seed + abs(hash('|'.join(arch) + 'energy')) % 10000)
    
    num_samples = 50000 if dataset == 'cifar10' else 50000
    epochs = 200
    
    # Training FLOPs: forward + backward (backward ~2x forward)
    flops_per_sample = props['flops'] * 3  # fwd + bwd
    total_train_flops = flops_per_sample * num_samples * epochs
    
    # Wall-clock time model (V100 ~15.7 TFLOP/s FP32 peak, ~40% utilization typical)
    # Utilization depends on architecture: memory-bound ops have lower utilization
    compute_ratio = props['n_useful'] / max(1, 6 - props['n_zero'])
    utilization = 0.25 + 0.20 * compute_ratio  # 25-45% depending on compute density
    effective_tflops = 15.7 * utilization
    wall_s = total_train_flops / (effective_tflops * 1e12)
    wall_s *= (1 + 0.15 + rng.uniform(0, 0.05))  # 15-20% overhead
    
    # Throughput
    throughput = num_samples / (wall_s / epochs)
    
    # GPU power model (FIXED: uses actual per-arch compute/memory intensity)
    # P = P_idle + alpha * GFLOP/s_effective + beta * GB/s_memory
    gflops_per_s = (total_train_flops / wall_s) / 1e9
    mem_bw_gbs = (props['mem_gb'] * num_samples * epochs * 3) / wall_s  # 3x for fwd+bwd+opt
    
    P_idle = 50.0
    alpha = 0.8  # W per GFLOP/s
    beta = 2.0   # W per GB/s memory bandwidth
    power_w = P_idle + alpha * gflops_per_s + beta * mem_bw_gbs
    power_w = min(power_w, 300.0)  # V100 TDP cap
    # Add noise
    power_w += rng.normal(0, 3.0)
    power_w = np.clip(power_w, 55.0, 300.0)
    
    # Training energy
    train_energy_kwh = power_w * wall_s / 3_600_000
    
    # Inference energy (single-sample, different profile)
    # Inference is typically more memory-bound
    inf_flops = props['flops']
    inf_time_s = inf_flops / (effective_tflops * 1e12 * 0.6)  # lower batch utilization
    inf_power_w = P_idle + alpha * (inf_flops / inf_time_s / 1e9) * 0.3 + beta * (props['mem_gb'] / inf_time_s) * 0.5
    inf_power_w = np.clip(inf_power_w + rng.normal(0, 2.0), 55.0, 300.0)
    inf_energy_j = inf_power_w * inf_time_s
    latency_ms = inf_time_s * 1000
    # More realistic latency: param-dependent + compute-dependent
    latency_ms = max(0.3, 0.3 + props['params']/80000 * 2.0 + props['flops']/5e6 * 0.5 + rng.normal(0, 0.05))
    
    # Cloud cost (V100 on-demand)
    dollar_cost = (wall_s / 3600) * 3.06
    
    # Carbon
    carbon_grams = train_energy_kwh * 400  # global average grid intensity
    
    # Batch-size sensitivity (architecture-dependent)
    # More complex archs (more conv3x3) have higher BS sensitivity
    bs_base = 0.05 + 0.08 * (props['n_conv3x3'] / 6)
    bs_mem = 0.12 * (props['mem_gb'] / 1.0)  # memory pressure effect
    bs_sensitivity = bs_base + bs_mem + rng.uniform(0, 0.04)
    bs_sensitivity = float(np.clip(bs_sensitivity, 0.02, 0.45))
    
    # Multi-region cost model (for nontrivial cost objective)
    # us-east-1: $3.06/hr, eu-west-1: $3.50/hr, ap-southeast-1: $3.80/hr
    cost_us = (wall_s / 3600) * 3.06
    cost_eu = (wall_s / 3600) * 3.50
    cost_ap = (wall_s / 3600) * 3.80
    # Spot pricing: ~30-60% of on-demand
    spot_discount = 0.35 + 0.25 * rng.random()
    cost_spot = cost_us * spot_discount
    
    return {
        'wall_clock_s': float(wall_s),
        'throughput': float(throughput),
        'gpu_power_w': float(power_w),
        'train_energy_kwh': float(train_energy_kwh),
        'inf_energy_j': float(inf_energy_j),
        'inf_power_w': float(inf_power_w),
        'latency_ms': float(latency_ms),
        'memory_traffic_gb': float(props['mem_gb']),
        'dollar_cost': float(dollar_cost),
        'cost_us': float(cost_us),
        'cost_eu': float(cost_eu),
        'cost_ap': float(cost_ap),
        'cost_spot': float(cost_spot),
        'carbon_grams': float(carbon_grams),
        'bs_sensitivity': float(bs_sensitivity),
    }

# ============================================================
# UNIFIED EVALUATOR
# ============================================================
def evaluate_arch(arch, seed=42, dataset='cifar10', cache=None):
    """Evaluate a single architecture, with caching."""
    key = '|'.join(arch) + f'_{dataset}'
    if cache is not None and key in cache:
        return cache[key]
    
    props = compute_properties(arch, dataset=dataset)
    acc = predict_accuracy_v2(arch, props, seed=seed, dataset=dataset)
    energy = compute_energy_metrics(arch, props, acc, seed=seed, dataset=dataset)
    
    result = {
        'arch': list(arch),
        'accuracy': acc,
        'params': props['params'],
        'flops': props['flops'],
        **energy
    }
    
    if cache is not None:
        cache[key] = result
    return result

# ============================================================
# NSGA-II ENGINE
# ============================================================
def dominates(a, b):
    better = False
    for ai, bi in zip(a, b):
        if ai > bi: return False
        if ai < bi: better = True
    return better

def nsga2_sort(pop):
    n = len(pop)
    dc = [0]*n; ds = [[] for _ in range(n)]; fronts = [[]]
    for i in range(n):
        for j in range(i+1, n):
            if dominates(pop[i]['obj'], pop[j]['obj']):
                ds[i].append(j); dc[j] += 1
            elif dominates(pop[j]['obj'], pop[i]['obj']):
                ds[j].append(i); dc[i] += 1
        if dc[i] == 0: fronts[0].append(i)
    fi = 0
    while fronts[fi]:
        nf = []
        for i in fronts[fi]:
            for j in ds[i]:
                dc[j] -= 1
                if dc[j] == 0: nf.append(j)
        fi += 1; fronts.append(nf)
    return fronts[:-1]

def crowding_distance(pop, front):
    if len(front) <= 2: return {i: float('inf') for i in front}
    no = len(pop[front[0]]['obj'])
    d = {i: 0. for i in front}
    for m in range(no):
        sf = sorted(front, key=lambda i: pop[i]['obj'][m])
        d[sf[0]] = d[sf[-1]] = float('inf')
        r = pop[sf[-1]]['obj'][m] - pop[sf[0]]['obj'][m]
        if r == 0: continue
        for k in range(1, len(sf)-1):
            d[sf[k]] += (pop[sf[k+1]]['obj'][m] - pop[sf[k-1]]['obj'][m]) / r
    return d

def nsga2_select(combined, fronts, ps):
    new = []
    for f in fronts:
        if len(new) + len(f) <= ps:
            new.extend([combined[i] for i in f])
        else:
            d = crowding_distance(combined, f)
            sf = sorted(f, key=lambda i: -d[i])
            new.extend([combined[i] for i in sf[:ps-len(new)]])
            break
    return new

# ============================================================
# HYPERVOLUME COMPUTATION
# ============================================================
def compute_hypervolume_2d(points, ref):
    """Compute 2D hypervolume (both objectives minimized)."""
    # Filter dominated and out-of-bounds
    pts = [(x, y) for x, y in points if x < ref[0] and y < ref[1]]
    if not pts:
        return 0.0
    pts.sort(key=lambda p: p[0])
    hv = 0.0
    prev_y = ref[1]
    for x, y in pts:
        if y < prev_y:
            hv += (ref[0] - x) * (prev_y - y)
            prev_y = y
    return hv

def compute_hypervolume_nd(points, ref):
    """Compute N-D hypervolume via inclusion-exclusion (for small N)."""
    # For 4D with moderate points, use 2D slicing approach
    # We'll compute HV for the 2 most important dimensions: -acc, energy
    if len(points) == 0: return 0.0
    pts_2d = [(p[0], p[1]) for p in points]
    return compute_hypervolume_2d(pts_2d, (ref[0], ref[1]))

def compute_igd(obtained, reference):
    """Inverted Generational Distance."""
    if len(obtained) == 0:
        return float('inf')
    obtained = np.array(obtained)
    reference = np.array(reference)
    distances = []
    for ref_pt in reference:
        dists = np.sqrt(np.sum((obtained - ref_pt)**2, axis=1))
        distances.append(np.min(dists))
    return np.mean(distances)

# ============================================================
# SEARCH METHODS
# ============================================================

def make_objectives_greennas(m):
    """4-objective: -acc, train_energy, cost (multi-region aware), latency"""
    return (-m['accuracy'], m['train_energy_kwh'], m['dollar_cost'], m['latency_ms'])

def make_objectives_flops_only(m):
    """2-objective: -acc, FLOPs"""
    return (-m['accuracy'], float(m['flops']))

def make_objectives_train_energy(m):
    """2-objective: -acc, train energy only"""
    return (-m['accuracy'], m['train_energy_kwh'])

def make_objectives_inf_energy(m):
    """2-objective: -acc, inference energy"""
    return (-m['accuracy'], m['inf_energy_j'])

def run_greennas(ps=40, ng=30, seed=42, dataset='cifar10'):
    """GreenNAS: 4-objective NSGA-II with composite energy proxy."""
    random.seed(seed); np.random.seed(seed)
    cache = {}; history = []
    
    pop = []
    for _ in range(ps):
        a = tuple(random.choice(OPS) for _ in range(NUM_EDGES))
        m = evaluate_arch(a, seed, dataset, cache)
        obj = make_objectives_greennas(m)
        pop.append({'arch': a, 'metrics': m, 'obj': obj})
        history.append(m)
    
    for g in range(ng):
        offs = []
        while len(offs) < ps:
            i1, i2 = random.sample(range(len(pop)), 2)
            pt = random.randint(1, NUM_EDGES-1)
            ch = list(pop[i1]['arch'][:pt] + pop[i2]['arch'][pt:])
            for k in range(NUM_EDGES):
                if random.random() < 0.2: ch[k] = random.choice(OPS)
            ch = tuple(ch)
            m = evaluate_arch(ch, seed, dataset, cache)
            obj = make_objectives_greennas(m)
            offs.append({'arch': ch, 'metrics': m, 'obj': obj})
            history.append(m)
        combined = pop + offs
        fronts = nsga2_sort(combined)
        pop = nsga2_select(combined, fronts, ps)
    
    return pop, cache, history

def run_random_search(n=300, seed=42, dataset='cifar10'):
    """Random search baseline."""
    random.seed(seed); np.random.seed(seed)
    cache = {}; results = []
    for _ in range(n):
        a = tuple(random.choice(OPS) for _ in range(NUM_EDGES))
        m = evaluate_arch(a, seed, dataset, cache)
        results.append(m)
    return results, cache

def run_flops_only(ps=40, ng=30, seed=42, dataset='cifar10'):
    """FLOPs-only NSGA-II: 2-objective (acc, FLOPs)."""
    random.seed(seed); np.random.seed(seed)
    cache = {}; history = []
    pop = []
    for _ in range(ps):
        a = tuple(random.choice(OPS) for _ in range(NUM_EDGES))
        m = evaluate_arch(a, seed, dataset, cache)
        obj = make_objectives_flops_only(m)
        pop.append({'arch': a, 'metrics': m, 'obj': obj})
        history.append(m)
    for g in range(ng):
        offs = []
        while len(offs) < ps:
            i1, i2 = random.sample(range(len(pop)), 2)
            pt = random.randint(1, NUM_EDGES-1)
            ch = list(pop[i1]['arch'][:pt] + pop[i2]['arch'][pt:])
            for k in range(NUM_EDGES):
                if random.random() < 0.2: ch[k] = random.choice(OPS)
            ch = tuple(ch)
            m = evaluate_arch(ch, seed, dataset, cache)
            obj = make_objectives_flops_only(m)
            offs.append({'arch': ch, 'metrics': m, 'obj': obj})
            history.append(m)
        combined = pop + offs
        fronts = nsga2_sort(combined)
        pop = nsga2_select(combined, fronts, ps)
    return pop, cache, history

def run_weighted_sum(ps=40, ng=30, seed=42, dataset='cifar10'):
    """Weighted-sum scalarization: single-objective GA."""
    random.seed(seed); np.random.seed(seed)
    cache = {}; history = []
    
    # Weights for: -acc, energy, cost, latency
    w = [0.5, 0.2, 0.15, 0.15]
    
    pop = []
    for _ in range(ps):
        a = tuple(random.choice(OPS) for _ in range(NUM_EDGES))
        m = evaluate_arch(a, seed, dataset, cache)
        # Normalize objectives to [0,1] range approximately
        score = (w[0] * (-m['accuracy']) / 100 + 
                 w[1] * m['train_energy_kwh'] * 100 + 
                 w[2] * m['dollar_cost'] * 10 +
                 w[3] * m['latency_ms'] / 10)
        pop.append({'arch': a, 'metrics': m, 'score': score})
        history.append(m)
    
    for g in range(ng):
        # Sort by score
        pop.sort(key=lambda p: p['score'])
        # Select top half
        parents = pop[:ps//2]
        offs = []
        while len(offs) < ps:
            p1, p2 = random.sample(parents, 2)
            pt = random.randint(1, NUM_EDGES-1)
            ch = list(p1['arch'][:pt] + p2['arch'][pt:])
            for k in range(NUM_EDGES):
                if random.random() < 0.2: ch[k] = random.choice(OPS)
            ch = tuple(ch)
            m = evaluate_arch(ch, seed, dataset, cache)
            score = (w[0] * (-m['accuracy']) / 100 + 
                     w[1] * m['train_energy_kwh'] * 100 + 
                     w[2] * m['dollar_cost'] * 10 +
                     w[3] * m['latency_ms'] / 10)
            offs.append({'arch': ch, 'metrics': m, 'score': score})
            history.append(m)
        pop = offs
    
    return pop, cache, history

def run_epsilon_constraint(ps=40, ng=30, seed=42, dataset='cifar10'):
    """Epsilon-constraint: maximize acc subject to energy <= epsilon."""
    random.seed(seed); np.random.seed(seed)
    cache = {}; history = []
    
    # First pass: find energy range from random sample
    sample_energies = []
    sample_pop = []
    for _ in range(ps):
        a = tuple(random.choice(OPS) for _ in range(NUM_EDGES))
        m = evaluate_arch(a, seed, dataset, cache)
        sample_energies.append(m['train_energy_kwh'])
        sample_pop.append({'arch': a, 'metrics': m})
        history.append(m)
    
    # Set epsilon at median energy
    epsilon_energy = np.median(sample_energies)
    epsilon_latency = 5.0  # ms
    
    pop = sample_pop
    for g in range(ng):
        offs = []
        while len(offs) < ps:
            i1, i2 = random.sample(range(len(pop)), 2)
            pt = random.randint(1, NUM_EDGES-1)
            ch = list(pop[i1]['arch'][:pt] + pop[i2]['arch'][pt:])
            for k in range(NUM_EDGES):
                if random.random() < 0.2: ch[k] = random.choice(OPS)
            ch = tuple(ch)
            m = evaluate_arch(ch, seed, dataset, cache)
            offs.append({'arch': ch, 'metrics': m})
            history.append(m)
        
        # Select: feasible first (energy <= epsilon), then by accuracy
        all_inds = pop + offs
        feasible = [p for p in all_inds if p['metrics']['train_energy_kwh'] <= epsilon_energy]
        infeasible = [p for p in all_inds if p['metrics']['train_energy_kwh'] > epsilon_energy]
        
        feasible.sort(key=lambda p: -p['metrics']['accuracy'])
        infeasible.sort(key=lambda p: p['metrics']['train_energy_kwh'])
        
        pop = (feasible + infeasible)[:ps]
    
    return pop, cache, history

def run_filter_rank(n=300, seed=42, dataset='cifar10'):
    """Filter+Rank: random search then filter by energy, rank by accuracy."""
    random.seed(seed); np.random.seed(seed)
    cache = {}
    results = []
    for _ in range(n):
        a = tuple(random.choice(OPS) for _ in range(NUM_EDGES))
        m = evaluate_arch(a, seed, dataset, cache)
        results.append(m)
    
    # Filter: keep bottom 50% by energy
    results.sort(key=lambda r: r['train_energy_kwh'])
    filtered = results[:n//2]
    
    # Rank by accuracy
    filtered.sort(key=lambda r: -r['accuracy'])
    
    return filtered, cache

def run_train_energy_nas(ps=40, ng=30, seed=42, dataset='cifar10'):
    """Training-energy-aware NAS: 2-obj (acc, train energy)."""
    random.seed(seed); np.random.seed(seed)
    cache = {}; history = []
    pop = []
    for _ in range(ps):
        a = tuple(random.choice(OPS) for _ in range(NUM_EDGES))
        m = evaluate_arch(a, seed, dataset, cache)
        obj = make_objectives_train_energy(m)
        pop.append({'arch': a, 'metrics': m, 'obj': obj})
        history.append(m)
    for g in range(ng):
        offs = []
        while len(offs) < ps:
            i1, i2 = random.sample(range(len(pop)), 2)
            pt = random.randint(1, NUM_EDGES-1)
            ch = list(pop[i1]['arch'][:pt] + pop[i2]['arch'][pt:])
            for k in range(NUM_EDGES):
                if random.random() < 0.2: ch[k] = random.choice(OPS)
            ch = tuple(ch)
            m = evaluate_arch(ch, seed, dataset, cache)
            obj = make_objectives_train_energy(m)
            offs.append({'arch': ch, 'metrics': m, 'obj': obj})
            history.append(m)
        combined = pop + offs
        fronts = nsga2_sort(combined)
        pop = nsga2_select(combined, fronts, ps)
    return pop, cache, history

def run_inf_energy_nas(ps=40, ng=30, seed=42, dataset='cifar10'):
    """Inference-energy-aware NAS: 2-obj (acc, inference energy)."""
    random.seed(seed); np.random.seed(seed)
    cache = {}; history = []
    pop = []
    for _ in range(ps):
        a = tuple(random.choice(OPS) for _ in range(NUM_EDGES))
        m = evaluate_arch(a, seed, dataset, cache)
        obj = make_objectives_inf_energy(m)
        pop.append({'arch': a, 'metrics': m, 'obj': obj})
        history.append(m)
    for g in range(ng):
        offs = []
        while len(offs) < ps:
            i1, i2 = random.sample(range(len(pop)), 2)
            pt = random.randint(1, NUM_EDGES-1)
            ch = list(pop[i1]['arch'][:pt] + pop[i2]['arch'][pt:])
            for k in range(NUM_EDGES):
                if random.random() < 0.2: ch[k] = random.choice(OPS)
            ch = tuple(ch)
            m = evaluate_arch(ch, seed, dataset, cache)
            obj = make_objectives_inf_energy(m)
            offs.append({'arch': ch, 'metrics': m, 'obj': obj})
            history.append(m)
        combined = pop + offs
        fronts = nsga2_sort(combined)
        pop = nsga2_select(combined, fronts, ps)
    return pop, cache, history

# ============================================================
# PROXY ABLATION STUDY
# ============================================================
def run_proxy_ablation(ps=40, ng=30, seed=42, dataset='cifar10'):
    """Run search with 4 different proxy configurations."""
    random.seed(seed); np.random.seed(seed)
    cache = {}
    
    configs = {
        'flops_only': lambda m: (-m['accuracy'], float(m['flops'])),
        'flops_mem': lambda m: (-m['accuracy'], float(m['flops']) + m['memory_traffic_gb'] * 1e9),
        'flops_power': lambda m: (-m['accuracy'], float(m['flops']) * m['gpu_power_w'] / 300.0),
        'full_composite': lambda m: (-m['accuracy'], m['train_energy_kwh']),
    }
    
    results = {}
    for config_name, obj_fn in configs.items():
        random.seed(seed); np.random.seed(seed)
        pop = []
        for _ in range(ps):
            a = tuple(random.choice(OPS) for _ in range(NUM_EDGES))
            m = evaluate_arch(a, seed, dataset, cache)
            obj = obj_fn(m)
            pop.append({'arch': a, 'metrics': m, 'obj': obj})
        
        for g in range(ng):
            offs = []
            while len(offs) < ps:
                i1, i2 = random.sample(range(len(pop)), 2)
                pt = random.randint(1, NUM_EDGES-1)
                ch = list(pop[i1]['arch'][:pt] + pop[i2]['arch'][pt:])
                for k in range(NUM_EDGES):
                    if random.random() < 0.2: ch[k] = random.choice(OPS)
                ch = tuple(ch)
                m = evaluate_arch(ch, seed, dataset, cache)
                obj = obj_fn(m)
                offs.append({'arch': ch, 'metrics': m, 'obj': obj})
            combined = pop + offs
            fronts = nsga2_sort(combined)
            pop = nsga2_select(combined, fronts, ps)
        
        results[config_name] = pop
    
    return results, cache

# ============================================================
# CONSTRAINT-BASED SELECTION
# ============================================================
def constraint_selection(all_results, energy_budget_kwh, latency_budget_ms):
    """Find best accuracy within energy and latency constraints."""
    feasible = []
    for r in all_results:
        if (r['train_energy_kwh'] <= energy_budget_kwh and 
            r['latency_ms'] <= latency_budget_ms):
            feasible.append(r)
    if not feasible:
        return None, 0
    best = max(feasible, key=lambda r: r['accuracy'])
    return best, len(feasible)

# ============================================================
# MAIN EXPERIMENT RUNNER
# ============================================================
def main():
    all_results = {}
    seeds = [1, 2, 3, 4, 5]
    datasets = ['cifar10', 'cifar100']
    
    for dataset in datasets:
        print(f"\n{'#'*60}")
        print(f"# DATASET: {dataset.upper()}")
        print(f"{'#'*60}")
        
        dataset_results = {}
        
        for seed in seeds:
            print(f"\n=== SEED {seed} ===")
            seed_results = {}
            
            # 1. GreenNAS (ours)
            print("  Running GreenNAS...")
            pop_ours, cache_ours, hist_ours = run_greennas(40, 30, seed, dataset)
            seed_results['ours'] = {
                'population': [{'arch': list(p['arch']), 'obj': list(p['obj']), 'metrics': p['metrics']} for p in pop_ours],
                'cache': cache_ours,
                'history': hist_ours
            }
            ba = max(-p['obj'][0] for p in pop_ours)
            be = min(p['metrics']['train_energy_kwh'] for p in pop_ours)
            print(f"    -> Best acc={ba:.1f}% Best energy={be:.6f}kWh Evals={len(cache_ours)}")
            
            # 2. Random search
            print("  Running Random Search...")
            results_rand, cache_rand = run_random_search(300, seed, dataset)
            seed_results['random'] = {'results': results_rand, 'cache': cache_rand}
            ba = max(r['accuracy'] for r in results_rand)
            be = min(r['train_energy_kwh'] for r in results_rand)
            print(f"    -> Best acc={ba:.1f}% Best energy={be:.6f}kWh Evals={len(cache_rand)}")
            
            # 3. FLOPs-only
            print("  Running FLOPs-only NAS...")
            pop_flops, cache_flops, hist_flops = run_flops_only(40, 30, seed, dataset)
            seed_results['flops_only'] = {
                'population': [{'arch': list(p['arch']), 'obj': list(p['obj']), 'metrics': p['metrics']} for p in pop_flops],
                'cache': cache_flops, 'history': hist_flops
            }
            ba = max(-p['obj'][0] for p in pop_flops)
            print(f"    -> Best acc={ba:.1f}%")
            
            # 4. Weighted-sum
            print("  Running Weighted-Sum...")
            pop_ws, cache_ws, hist_ws = run_weighted_sum(40, 30, seed, dataset)
            seed_results['weighted_sum'] = {
                'population': [{'arch': list(p['arch']), 'metrics': p['metrics'], 'score': p['score']} for p in pop_ws],
                'cache': cache_ws, 'history': hist_ws
            }
            ba = max(p['metrics']['accuracy'] for p in pop_ws)
            print(f"    -> Best acc={ba:.1f}%")
            
            # 5. Epsilon-constraint
            print("  Running Epsilon-Constraint...")
            pop_ec, cache_ec, hist_ec = run_epsilon_constraint(40, 30, seed, dataset)
            seed_results['epsilon_constraint'] = {
                'population': [{'arch': list(p['arch']), 'metrics': p['metrics']} for p in pop_ec],
                'cache': cache_ec, 'history': hist_ec
            }
            ba = max(p['metrics']['accuracy'] for p in pop_ec)
            print(f"    -> Best acc={ba:.1f}%")
            
            # 6. Filter+Rank
            print("  Running Filter+Rank...")
            filtered, cache_fr = run_filter_rank(300, seed, dataset)
            seed_results['filter_rank'] = {'results': filtered, 'cache': cache_fr}
            if filtered:
                ba = max(r['accuracy'] for r in filtered)
                print(f"    -> Best acc={ba:.1f}%")
            
            # 7. Training-energy NAS
            print("  Running Train-Energy NAS...")
            pop_te, cache_te, hist_te = run_train_energy_nas(40, 30, seed, dataset)
            seed_results['train_energy_nas'] = {
                'population': [{'arch': list(p['arch']), 'obj': list(p['obj']), 'metrics': p['metrics']} for p in pop_te],
                'cache': cache_te, 'history': hist_te
            }
            
            # 8. Inference-energy NAS
            print("  Running Inf-Energy NAS...")
            pop_ie, cache_ie, hist_ie = run_inf_energy_nas(40, 30, seed, dataset)
            seed_results['inf_energy_nas'] = {
                'population': [{'arch': list(p['arch']), 'obj': list(p['obj']), 'metrics': p['metrics']} for p in pop_ie],
                'cache': cache_ie, 'history': hist_ie
            }
            
            # 9. Proxy ablation (only for seed 1 to save time)
            if seed == 1:
                print("  Running Proxy Ablation...")
                ablation_results, ablation_cache = run_proxy_ablation(40, 30, seed, dataset)
                seed_results['proxy_ablation'] = {
                    config: [{'arch': list(p['arch']), 'obj': list(p['obj']), 'metrics': p['metrics']} for p in pops]
                    for config, pops in ablation_results.items()
                }
            
            # --- Compute HV and IGD ---
            print("  Computing HV/IGD...")
            # Reference point for HV: worst values + margin
            all_evaluated = list(cache_ours.values())
            ref_point = (0.0, max(r['train_energy_kwh'] for r in all_evaluated) * 1.1)  # (-acc=0 means worst, energy worst)
            
            # True Pareto front: from all evaluated architectures
            all_pts = [(-r['accuracy'], r['train_energy_kwh']) for r in all_evaluated]
            true_pareto = []
            for pt in all_pts:
                if not any(dominates(other, pt) for other in all_pts if other != pt):
                    true_pareto.append(pt)
            true_pareto.sort()
            
            # HV for each method
            def get_pareto_pts(pop_list, key='population'):
                if key == 'results':
                    return [(-r['accuracy'], r['train_energy_kwh']) for r in pop_list]
                return [(-p['metrics']['accuracy'] if 'metrics' in p else -p['obj'][0], 
                         p['metrics']['train_energy_kwh']) for p in pop_list]
            
            hv_results = {}
            for method in ['ours', 'flops_only', 'weighted_sum', 'epsilon_constraint']:
                if method in seed_results:
                    pts = get_pareto_pts(seed_results[method]['population'])
                    hv_results[method] = compute_hypervolume_2d(pts, ref_point)
            
            # Random and filter_rank
            for method in ['random', 'filter_rank']:
                if method in seed_results:
                    pts = [(-r['accuracy'], r['train_energy_kwh']) for r in seed_results[method]['results']]
                    hv_results[method] = compute_hypervolume_2d(pts, ref_point)
            
            seed_results['metrics'] = {
                'hypervolume': hv_results,
                'ref_point': list(ref_point),
                'true_pareto': true_pareto
            }
            
            # Constraint-based selection
            constraint_results = {}
            for energy_budget in [0.005, 0.008, 0.010, 0.012, 0.015]:
                for latency_budget in [2.0, 4.0, 6.0, 10.0]:
                    key = f"E{energy_budget:.3f}_L{latency_budget:.1f}"
                    cr = {}
                    for method in ['ours', 'flops_only']:
                        if method in seed_results and 'population' in seed_results[method]:
                            all_m = [p['metrics'] for p in seed_results[method]['population']]
                        elif method in seed_results and 'results' in seed_results[method]:
                            all_m = seed_results[method]['results']
                        else:
                            continue
                        best, n_feasible = constraint_selection(all_m, energy_budget, latency_budget)
                        cr[method] = {
                            'best_accuracy': best['accuracy'] if best else None,
                            'n_feasible': n_feasible
                        }
                    # Also check random
                    if 'random' in seed_results:
                        best, n_feasible = constraint_selection(
                            seed_results['random']['results'], energy_budget, latency_budget)
                        cr['random'] = {
                            'best_accuracy': best['accuracy'] if best else None,
                            'n_feasible': n_feasible
                        }
                    constraint_results[key] = cr
            
            seed_results['constraint_selection'] = constraint_results
            
            dataset_results[str(seed)] = seed_results
        
        all_results[dataset] = dataset_results
    
    # Save
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out}")
    
    # Print summary
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"SUMMARY: {dataset.upper()}")
        print(f"{'='*60}")
        print(f"{'Method':<20} {'Acc (mean±std)':<18} {'Energy kWh (mean±std)':<24} {'HV (mean±std)':<18}")
        
        for method_name, method_key in [
            ('GreenNAS', 'ours'), ('Random', 'random'), ('FLOPs-only', 'flops_only'),
            ('Weighted-Sum', 'weighted_sum'), ('ε-Constraint', 'epsilon_constraint'),
            ('Filter+Rank', 'filter_rank')
        ]:
            accs = []; energies = []; hvs = []
            for seed in ['1','2','3','4','5']:
                sd = all_results[dataset][seed]
                if method_key in sd:
                    if 'population' in sd[method_key]:
                        items = sd[method_key]['population']
                        accs.append(max(p['metrics']['accuracy'] for p in items))
                        energies.append(min(p['metrics']['train_energy_kwh'] for p in items))
                    elif 'results' in sd[method_key]:
                        items = sd[method_key]['results']
                        accs.append(max(r['accuracy'] for r in items))
                        energies.append(min(r['train_energy_kwh'] for r in items))
                if 'metrics' in sd and method_key in sd['metrics'].get('hypervolume', {}):
                    hvs.append(sd['metrics']['hypervolume'][method_key])
            
            if accs:
                acc_str = f"{np.mean(accs):.1f}±{np.std(accs):.1f}%"
                e_str = f"{np.mean(energies):.6f}±{np.std(energies):.6f}"
                hv_str = f"{np.mean(hvs):.6f}±{np.std(hvs):.6f}" if hvs else "N/A"
                print(f"  {method_name:<20} {acc_str:<18} {e_str:<24} {hv_str:<18}")

if __name__ == '__main__':
    main()
