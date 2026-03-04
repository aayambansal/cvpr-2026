"""
Scaled experiment: 500 architectures, multi-resolution, top-k regret analysis.
Optimized for CPU throughput: minimal batch sizes, shared random data tensors.
"""
import sys, os, json, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from search_space import (
    UnifiedBackbone, sample_random_config, config_to_vector,
    count_parameters, compute_search_space_size
)

torch.set_num_threads(os.cpu_count() or 4)
torch.manual_seed(42)

# Pre-allocate shared data tensors (avoid repeated allocation)
DATA_32 = torch.randn(8, 3, 32, 32)
DATA_16 = torch.randn(8, 3, 16, 16)
ONES_32 = torch.ones(1, 3, 32, 32)
ONES_16 = torch.ones(1, 3, 16, 16)
LABELS_10 = torch.randint(0, 10, (8,))
LABELS_100 = torch.randint(0, 100, (8,))

# ============================================================
# Ultra-fast scoring functions
# ============================================================

def naswot(model, data):
    model.eval()
    bs = data.size(0)
    acts = []
    hooks = []
    def hk(m, i, o):
        if isinstance(o, torch.Tensor) and o.dim() >= 2:
            a = (o > 0).float().view(o.size(0), -1)
            if a.size(1) > 64: a = a[:, ::max(1, a.size(1)//64)]
            acts.append(a)
    for m in model.modules():
        if isinstance(m, (nn.GELU, nn.SiLU, nn.ReLU)):
            hooks.append(m.register_forward_hook(hk))
    with torch.no_grad(): model(data)
    for h in hooks: h.remove()
    if not acts: return 0.0
    A = torch.cat(acts, 1)
    K = A @ A.T + 1e-5 * torch.eye(bs)
    try: return float(torch.slogdet(K)[1].item())
    except: return 0.0

def synflow(model, ones):
    model.eval()
    for p in model.parameters(): p.data.abs_()
    model.zero_grad()
    model(ones).sum().backward()
    s = 0.0
    for p in model.parameters():
        if p.grad is not None: s += (p.data * p.grad.data).sum().item()
    model.zero_grad()
    return float(np.log(abs(s) + 1e-10))

def gradnorm(model, data, labels):
    model.train()
    model.zero_grad()
    F.cross_entropy(model(data), labels).backward()
    g = sum(p.grad.norm(2).item()**2 for p in model.parameters() if p.grad is not None)
    model.zero_grad()
    return float(np.log(np.sqrt(g) + 1e-10))

def snip(model, data, labels):
    model.train()
    model.zero_grad()
    F.cross_entropy(model(data), labels).backward()
    s = sum((p.data * p.grad.data).abs().sum().item() for p in model.parameters() if p.grad is not None)
    model.zero_grad()
    return float(np.log(s + 1e-10))

def score_arch(cfg, nc=10, sz=32):
    """Score one architecture at given resolution and num_classes."""
    data = DATA_32 if sz == 32 else DATA_16
    ones = ONES_32 if sz == 32 else ONES_16
    labels = LABELS_10[:data.size(0)] if nc == 10 else LABELS_100[:data.size(0)]
    
    sc = {}
    m = UnifiedBackbone(cfg, num_classes=nc, input_size=sz)
    sc['params'] = count_parameters(m)
    sc['naswot'] = naswot(m, data)
    del m
    
    m = UnifiedBackbone(cfg, num_classes=nc, input_size=sz)
    sc['synflow'] = synflow(m, ones)
    del m
    
    m = UnifiedBackbone(cfg, num_classes=nc, input_size=sz)
    sc['gradnorm'] = gradnorm(m, data, labels)
    del m
    
    m = UnifiedBackbone(cfg, num_classes=nc, input_size=sz)
    sc['snip'] = snip(m, data, labels)
    del m
    
    sc['log_params'] = float(np.log(sc['params'] + 1))
    return sc

# Pre-allocate shared training data (avoid per-arch allocation)
TRAIN_X_32 = torch.randn(256, 3, 32, 32)
TRAIN_Y_10 = torch.randint(0, 10, (256,))
VAL_X_32 = torch.randn(64, 3, 32, 32)
VAL_Y_10 = torch.randint(0, 10, (64,))

def train_eval(cfg, nc=10, sz=32, epochs=3, bs=64):
    """Short training evaluation with shared data tensors."""
    model = UnifiedBackbone(cfg, num_classes=nc, input_size=sz)
    model.train()
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    
    tx, ty = TRAIN_X_32, TRAIN_Y_10
    vx, vy = VAL_X_32, VAL_Y_10
    train_n = tx.size(0)
    
    for _ in range(epochs):
        model.train()
        idx = torch.randperm(train_n)
        for i in range(0, train_n, bs):
            ii = idx[i:i+bs]
            opt.zero_grad()
            F.cross_entropy(model(tx[ii]), ty[ii]).backward()
            opt.step()
    
    model.eval()
    with torch.no_grad():
        acc = (model(vx).argmax(1) == vy).float().mean().item()
    
    del model
    return float(acc)


def save_results(results, out_dir, suffix=''):
    """Save results incrementally."""
    ser = []
    for r in results:
        d = {'id': r['id'], 'config': r['config'], 'params': r['params'],
             'scores': {k: float(v) for k, v in r['scores'].items()}}
        if 'scores_16x16_c100' in r and r['scores_16x16_c100'] is not None:
            d['scores_16x16_c100'] = {k: float(v) for k, v in r['scores_16x16_c100'].items()}
        if 'train_acc_3ep' in r:
            d['train_acc_3ep'] = r['train_acc_3ep']
            d['selection_group'] = r.get('selection_group', '')
        ser.append(d)
    fname = os.path.join(out_dir, f'results_v2{suffix}.json')
    with open(fname, 'w') as f:
        json.dump(ser, f, indent=2)
    print(f"  -> Saved {len(ser)} results to {fname}")

def main():
    N = 500
    N_TRAIN_PER_GROUP = 10  # top-10, bottom-10, random-10 = 30 trained
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Search space: {compute_search_space_size():,}")
    print(f"Target: {N} scored, {N_TRAIN_PER_GROUP*3} trained")
    print()
    
    # ============================================================
    # Phase 1: Score N architectures at 32x32 (CIFAR-10 proxy)
    # ============================================================
    results = []
    t0 = time.time()
    failures = 0
    
    for i in range(N * 2):  # oversample to get N valid
        if len(results) >= N: break
        cfg = sample_random_config(seed=i * 3 + 7)
        try:
            sc = score_arch(cfg, nc=10, sz=32)
            if sc['params'] > 2_000_000: continue
            results.append({'id': len(results), 'seed': i*3+7, 'config': cfg, 'params': sc['params'], 'scores': sc})
            
            if len(results) % 100 == 0:
                rate = len(results) / (time.time() - t0)
                print(f"  Phase 1: {len(results)}/{N} scored ({rate:.1f}/s, {time.time()-t0:.0f}s)")
        except Exception as e:
            failures += 1
            continue
        
        if len(results) % 200 == 0:
            gc.collect()
    
    print(f"Phase 1 done: {len(results)} scored in {time.time()-t0:.0f}s (failures={failures})")
    save_results(results, out_dir, '_phase1')
    
    # ============================================================
    # Phase 2: Multi-resolution scoring (16x16) for subset
    # ============================================================
    print(f"\nPhase 2: Multi-resolution (16x16, nc=100) for {min(200, len(results))} archs...")
    t1 = time.time()
    
    for j, r in enumerate(results[:200]):
        try:
            sc16 = score_arch(r['config'], nc=100, sz=16)
            r['scores_16x16_c100'] = sc16
        except:
            r['scores_16x16_c100'] = None
        if (j+1) % 50 == 0:
            print(f"  {j+1}/200 ({time.time()-t1:.0f}s)")
    
    print(f"Phase 2 done in {time.time()-t1:.0f}s")
    save_results(results, out_dir, '_phase2')
    
    # ============================================================
    # Phase 3: Top-k / Bottom-k / Random short training (reduced)
    # ============================================================
    print(f"\nPhase 3: Top-k regret analysis ({N_TRAIN_PER_GROUP} per group, 3 epochs)...")
    t2 = time.time()
    
    # Sort by NASWOT score
    scored = [(i, r['scores']['naswot']) for i, r in enumerate(results) if r['scores']['naswot'] != 0]
    scored.sort(key=lambda x: -x[1])
    
    top_indices = [s[0] for s in scored[:N_TRAIN_PER_GROUP]]
    bottom_indices = [s[0] for s in scored[-N_TRAIN_PER_GROUP:]]
    
    # Random selection (excluding top/bottom)
    middle_indices = [s[0] for s in scored[N_TRAIN_PER_GROUP:-N_TRAIN_PER_GROUP]]
    np.random.seed(42)
    random_indices = list(np.random.choice(middle_indices, size=min(N_TRAIN_PER_GROUP, len(middle_indices)), replace=False))
    
    for label, indices in [('top', top_indices), ('bottom', bottom_indices), ('random', random_indices)]:
        for j, idx in enumerate(indices):
            r = results[idx]
            try:
                acc = train_eval(r['config'], nc=10, sz=32, epochs=3)
                r['train_acc_3ep'] = acc
                r['selection_group'] = label
                print(f"  {label}[{j+1}/{len(indices)}]: acc={acc:.3f} (params={r['params']:,})")
            except Exception as e:
                r['train_acc_3ep'] = 0.0
                r['selection_group'] = label
                print(f"  {label}[{j+1}/{len(indices)}]: FAILED ({e})")
    
    print(f"Phase 3 done in {time.time()-t2:.0f}s")
    save_results(results, out_dir)
    
    # ============================================================
    # Phase 4: Analysis
    # ============================================================
    print(f"\nPhase 4: Analysis...")
    analysis = {}
    
    # --- Primitive dominance (all proxies) ---
    for key in ['mixer', 'norm', 'downsample', 'depth', 'expansion']:
        for proxy in ['naswot', 'synflow', 'gradnorm', 'snip']:
            dd = defaultdict(list)
            for r in results:
                for st in r['config']['stages']:
                    v = r['scores'].get(proxy, 0)
                    if v != 0 and np.isfinite(v):
                        dd[str(st[key])].append(v)
            analysis[f'{key}_{proxy}'] = {
                k: {'mean': round(float(np.mean(v)),2), 'std': round(float(np.std(v)),2), 'n': len(v)}
                for k, v in dd.items() if len(v) > 0
            }
    
    # --- Score correlation matrix ---
    snames = ['naswot', 'synflow', 'gradnorm', 'snip', 'log_params']
    M = np.array([[r['scores'].get(s, 0) for s in snames] for r in results])
    ok = np.all(np.isfinite(M), axis=1) & np.all(M != 0, axis=1)
    M_valid = M[ok]
    if len(M_valid) > 10:
        analysis['score_corr'] = {'names': snames, 'matrix': np.corrcoef(M_valid.T).tolist(), 'n': len(M_valid)}
    
    # --- PCA effective dimensionality ---
    V = np.array([config_to_vector(r['config']) for r in results])
    Vn = (V - V.mean(0)) / (V.std(0) + 1e-6)
    C = np.cov(Vn.T)
    eig = np.linalg.eigvalsh(C)[::-1]
    eig = eig[eig > 0]
    cum = np.cumsum(eig) / eig.sum()
    analysis['pca'] = {
        'eigenvalues': eig.tolist(), 'cumulative': cum.tolist(),
        'eff_dim_90': int(np.searchsorted(cum, 0.90)) + 1,
        'eff_dim_95': int(np.searchsorted(cum, 0.95)) + 1,
        'total': len(eig),
    }
    
    # --- Top-k regret analysis ---
    trained = [r for r in results if 'train_acc_3ep' in r]
    if len(trained) > 5:
        from scipy.stats import spearmanr, kendalltau
        
        # Proxy vs accuracy (all trained)
        accs = np.array([r['train_acc_3ep'] for r in trained])
        pc = {}
        for s in ['naswot', 'synflow', 'gradnorm', 'snip']:
            sv = np.array([r['scores'].get(s, 0) for r in trained])
            ok2 = np.isfinite(sv) & (sv != 0) & np.isfinite(accs)
            if ok2.sum() > 5:
                rho, p = spearmanr(sv[ok2], accs[ok2])
                tau, tp = kendalltau(sv[ok2], accs[ok2])
                pc[s] = {'spearman': round(float(rho), 3), 'sp_p': round(float(p), 4),
                         'kendall': round(float(tau), 3), 'kt_p': round(float(tp), 4),
                         'n': int(ok2.sum())}
        analysis['proxy_vs_acc'] = pc
        
        # Group-wise accuracy comparison
        group_accs = defaultdict(list)
        for r in trained:
            group_accs[r.get('selection_group', 'unknown')].append(r['train_acc_3ep'])
        
        analysis['regret_analysis'] = {
            k: {'mean': round(float(np.mean(v)), 4), 'std': round(float(np.std(v)), 4),
                'min': round(float(np.min(v)), 4), 'max': round(float(np.max(v)), 4), 'n': len(v)}
            for k, v in group_accs.items()
        }
    
    # --- Stage-wise mixer effect ---
    if len(trained) > 5:
        se = {}
        for si in range(4):
            ma = defaultdict(list)
            for r in trained:
                ma[r['config']['stages'][si]['mixer']].append(r['train_acc_3ep'])
            se[f'stage_{si}'] = {
                k: {'mean': round(float(np.mean(v)), 4), 'std': round(float(np.std(v)), 4), 'n': len(v)}
                for k, v in ma.items() if len(v) > 0
            }
        analysis['stage_mixer_effect'] = se
    
    # --- Cross-resolution correlation ---
    cross_res = [r for r in results if r.get('scores_16x16_c100') is not None]
    if len(cross_res) > 10:
        nw32 = np.array([r['scores']['naswot'] for r in cross_res])
        nw16 = np.array([r['scores_16x16_c100']['naswot'] for r in cross_res])
        ok3 = np.isfinite(nw32) & np.isfinite(nw16) & (nw32 != 0) & (nw16 != 0)
        if ok3.sum() > 5:
            rho, p = spearmanr(nw32[ok3], nw16[ok3])
            analysis['cross_resolution'] = {
                'spearman': round(float(rho), 3), 'p': round(float(p), 4), 'n': int(ok3.sum())
            }
    
    # --- Param stats ---
    ps = [r['params'] for r in results]
    analysis['param_stats'] = {
        'min': int(min(ps)), 'max': int(max(ps)),
        'mean': round(float(np.mean(ps))), 'median': round(float(np.median(ps)))
    }
    
    # --- NASWOT distribution ---
    nw = [r['scores']['naswot'] for r in results if r['scores']['naswot'] != 0]
    if nw:
        h, b = np.histogram(nw, bins=25)
        analysis['naswot_dist'] = {'hist': h.tolist(), 'bins': b.tolist(),
                                    'mean': round(float(np.mean(nw)), 2), 'std': round(float(np.std(nw)), 2)}
    
    # --- Best architectures ---
    scored_all = sorted(results, key=lambda r: -r['scores']['naswot'])
    top5 = []
    for r in scored_all[:5]:
        top5.append({
            'config': r['config'], 'params': r['params'],
            'naswot': r['scores']['naswot'], 'synflow': r['scores']['synflow'],
            'train_acc': r.get('train_acc_3ep', None)
        })
    analysis['top5_architectures'] = top5
    
    # ============================================================
    # Save analysis
    # ============================================================
    with open(os.path.join(out_dir, 'analysis_v2.json'), 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"  -> Saved analysis to {out_dir}/analysis_v2.json")
    
    # ============================================================
    # Print summary
    # ============================================================
    total = time.time() - t0
    print("\n" + "="*70)
    print(f"SCALED RESULTS (N={len(results)}, trained={len(trained)}, time={total:.0f}s)")
    print("="*70)
    
    print(f"\n--- Token Mixer NASWOT (N={len(results)}) ---")
    for k, v in sorted(analysis.get('mixer_naswot', {}).items(), key=lambda x: -x[1]['mean']):
        print(f"  {k:15s}: {v['mean']:.2f} ± {v['std']:.2f} (n={v['n']})")
    
    print(f"\n--- Norm NASWOT ---")
    for k, v in sorted(analysis.get('norm_naswot', {}).items(), key=lambda x: -x[1]['mean']):
        print(f"  {k:15s}: {v['mean']:.2f} ± {v['std']:.2f} (n={v['n']})")
    
    print(f"\n--- Downsample NASWOT ---")
    for k, v in sorted(analysis.get('downsample_naswot', {}).items(), key=lambda x: -x[1]['mean']):
        print(f"  {k:15s}: {v['mean']:.2f} ± {v['std']:.2f} (n={v['n']})")
    
    p = analysis['pca']
    print(f"\n--- PCA: total={p['total']}, eff90={p['eff_dim_90']}, eff95={p['eff_dim_95']} ---")
    
    if 'proxy_vs_acc' in analysis:
        print(f"\n--- Proxy vs Trained Accuracy (3-epoch) ---")
        for k, v in analysis['proxy_vs_acc'].items():
            print(f"  {k:12s}: ρ={v['spearman']:.3f} (p={v['sp_p']:.4f}), τ={v['kendall']:.3f}, n={v['n']}")
    
    if 'regret_analysis' in analysis:
        print(f"\n--- Top-k Regret Analysis ---")
        for k, v in sorted(analysis['regret_analysis'].items()):
            print(f"  {k:10s}: mean={v['mean']:.4f} ± {v['std']:.4f} [min={v['min']:.4f}, max={v['max']:.4f}] n={v['n']}")
    
    if 'cross_resolution' in analysis:
        cr = analysis['cross_resolution']
        print(f"\n--- Cross-Resolution: 32x32 vs 16x16 NASWOT ρ={cr['spearman']:.3f} (n={cr['n']}) ---")
    
    print(f"\n--- Top-5 Architectures ---")
    for i, a in enumerate(analysis.get('top5_architectures', [])):
        mixers = [s['mixer'] for s in a['config']['stages']]
        acc_str = f"acc={a['train_acc']:.3f}" if a['train_acc'] else "not trained"
        print(f"  #{i+1}: NASWOT={a['naswot']:.1f}, params={a['params']:,}, mixers={mixers}, {acc_str}")
    
    print(f"\nParams: [{analysis['param_stats']['min']:,} - {analysis['param_stats']['max']:,}]")
    print(f"\nAll results saved to {out_dir}/")
    print("DONE.")

if __name__ == '__main__':
    main()
