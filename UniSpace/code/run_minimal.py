"""
Minimal experiment - only NASWOT and SynFlow (fastest proxies).
50 architectures, 20 trained. Saves incrementally.
"""
import sys, os, json, time
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

torch.set_num_threads(4)

def naswot(model, bs=16):
    model.eval()
    x = torch.randn(bs, 3, 32, 32)
    acts = []
    hooks = []
    def hk(m, i, o):
        if isinstance(o, torch.Tensor) and o.dim() >= 2:
            a = (o > 0).float().view(o.size(0), -1)
            if a.size(1) > 64: a = a[:, :64]
            acts.append(a)
    for m in model.modules():
        if isinstance(m, (nn.GELU, nn.SiLU, nn.ReLU)):
            hooks.append(m.register_forward_hook(hk))
    with torch.no_grad(): model(x)
    for h in hooks: h.remove()
    if not acts: return 0.0
    A = torch.cat(acts, 1)
    K = A @ A.T + 1e-5 * torch.eye(bs)
    try: return float(torch.slogdet(K)[1].item())
    except: return 0.0

def synflow(model):
    model.eval()
    for p in model.parameters(): p.data.abs_()
    x = torch.ones(1, 3, 32, 32)
    model.zero_grad()
    model(x).sum().backward()
    s = sum((p.data * p.grad.data).sum().item() for p in model.parameters() if p.grad is not None)
    model.zero_grad()
    return float(np.log(abs(s) + 1e-10))

def gradnorm(model, bs=16):
    model.train()
    x = torch.randn(bs, 3, 32, 32); y = torch.randint(0, 10, (bs,))
    model.zero_grad()
    F.cross_entropy(model(x), y).backward()
    g = sum(p.grad.norm(2).item()**2 for p in model.parameters() if p.grad is not None)
    model.zero_grad()
    return float(np.log(np.sqrt(g) + 1e-10))

def snip(model, bs=16):
    model.train()
    x = torch.randn(bs, 3, 32, 32); y = torch.randint(0, 10, (bs,))
    model.zero_grad()
    F.cross_entropy(model(x), y).backward()
    s = sum((p.data * p.grad.data).abs().sum().item() for p in model.parameters() if p.grad is not None)
    model.zero_grad()
    return float(np.log(s + 1e-10))

def train_short(model, epochs=3):
    model.train()
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    torch.manual_seed(0)
    tx = torch.randn(256, 3, 32, 32); ty = torch.randint(0, 10, (256,))
    vx = torch.randn(64, 3, 32, 32); vy = torch.randint(0, 10, (64,))
    for _ in range(epochs):
        model.train()
        for i in range(0, 256, 64):
            opt.zero_grad(); F.cross_entropy(model(tx[i:i+64]), ty[i:i+64]).backward(); opt.step()
    model.eval()
    with torch.no_grad(): return float((model(vx).argmax(1) == vy).float().mean().item())

def main():
    N = 50
    N_TRAIN = 20
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Search space: {compute_search_space_size():,}")
    
    results = []
    t0 = time.time()
    
    for i in range(N):
        cfg = sample_random_config(seed=i*7)  # spread seeds
        try:
            model = UnifiedBackbone(cfg, num_classes=10, input_size=32)
            par = count_parameters(model)
            if par > 1_500_000: continue
            
            sc = {'params': par, 'log_params': float(np.log(par+1))}
            sc['naswot'] = naswot(UnifiedBackbone(cfg, 10, 32))
            sc['synflow'] = synflow(UnifiedBackbone(cfg, 10, 32))
            sc['gradnorm'] = gradnorm(UnifiedBackbone(cfg, 10, 32))
            sc['snip'] = snip(UnifiedBackbone(cfg, 10, 32))
            
            results.append({'id': i, 'config': cfg, 'params': par, 'scores': sc})
            print(f"  [{i+1}/{N}] params={par:,} naswot={sc['naswot']:.1f} synflow={sc['synflow']:.1f} ({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"  [{i+1}/{N}] ERROR: {e}")
            continue
    
    print(f"\nScored {len(results)} in {time.time()-t0:.0f}s")
    
    # Short train
    print(f"\nTraining {N_TRAIN}...")
    by_p = sorted(results, key=lambda r: r['params'])
    step = max(1, len(by_p)//N_TRAIN)
    subset = by_p[::step][:N_TRAIN]
    
    for j, r in enumerate(subset):
        try:
            r['train_acc'] = train_short(UnifiedBackbone(r['config'], 10, 32), epochs=3)
            print(f"  [{j+1}/{len(subset)}] acc={r['train_acc']:.3f} params={r['params']:,}")
        except Exception as e:
            r['train_acc'] = 0.0
            print(f"  [{j+1}/{len(subset)}] ERROR: {e}")
    
    # === ANALYSIS ===
    analysis = {}
    
    for key in ['mixer', 'norm', 'downsample', 'depth', 'expansion']:
        for score in ['naswot', 'synflow']:
            dd = defaultdict(list)
            for r in results:
                for st in r['config']['stages']:
                    dd[str(st[key])].append(r['scores'][score])
            analysis[f'{key}_{score}'] = {
                k: {'mean': round(float(np.mean(v)),2), 'std': round(float(np.std(v)),2), 'n': len(v)}
                for k, v in dd.items()
            }
    
    # Score correlations
    sn = ['naswot', 'synflow', 'gradnorm', 'snip', 'log_params']
    M = np.array([[r['scores'].get(s,0) for s in sn] for r in results])
    ok = np.all(np.isfinite(M), axis=1)
    M = M[ok]
    if len(M) > 3:
        analysis['score_corr'] = {'names': sn, 'matrix': np.corrcoef(M.T).tolist()}
    
    # PCA
    V = np.array([config_to_vector(r['config']) for r in results])
    Vn = (V - V.mean(0)) / (V.std(0) + 1e-6)
    C = np.cov(Vn.T)
    eig = np.linalg.eigvalsh(C)[::-1]
    eig = eig[eig > 0]
    cum = np.cumsum(eig)/eig.sum()
    analysis['pca'] = {
        'eigenvalues': eig.tolist(), 'cumulative': cum.tolist(),
        'eff_dim_90': int(np.searchsorted(cum, 0.90))+1,
        'eff_dim_95': int(np.searchsorted(cum, 0.95))+1,
        'total': len(eig)
    }
    
    # Proxy vs accuracy
    trained = [r for r in results if 'train_acc' in r]
    if len(trained) > 3:
        from scipy.stats import spearmanr, kendalltau
        accs = np.array([r['train_acc'] for r in trained])
        pc = {}
        for s in ['naswot', 'synflow', 'gradnorm', 'snip']:
            sv = np.array([r['scores'].get(s,0) for r in trained])
            ok2 = np.isfinite(sv) & (sv != 0)
            if ok2.sum() > 3:
                rho, p = spearmanr(sv[ok2], accs[ok2])
                tau, tp = kendalltau(sv[ok2], accs[ok2])
                pc[s] = {'spearman': round(float(rho),3), 'sp_p': round(float(p),4),
                          'kendall': round(float(tau),3), 'kt_p': round(float(tp),4)}
        analysis['proxy_vs_acc'] = pc
    
    # Stage-wise
    if len(trained) > 3:
        se = {}
        for si in range(4):
            ma = defaultdict(list)
            for r in trained:
                ma[r['config']['stages'][si]['mixer']].append(r['train_acc'])
            se[f'stage_{si}'] = {
                k: {'mean': round(float(np.mean(v)),4), 'std': round(float(np.std(v)),4), 'n': len(v)}
                for k, v in ma.items()
            }
        analysis['stage_mixer_effect'] = se
    
    # Param stats
    ps = [r['params'] for r in results]
    analysis['param_stats'] = {'min': min(ps), 'max': max(ps), 'mean': round(float(np.mean(ps))), 'median': round(float(np.median(ps)))}
    
    # NASWOT distribution
    nw = [r['scores']['naswot'] for r in results if r['scores']['naswot'] != 0]
    if nw:
        h, b = np.histogram(nw, bins=15)
        analysis['naswot_dist'] = {'hist': h.tolist(), 'bins': b.tolist()}
    
    # Save
    ser = [{'id': r['id'], 'config': r['config'], 'params': r['params'],
            'scores': {k: float(v) for k,v in r['scores'].items()},
            **({'train_acc': r['train_acc']} if 'train_acc' in r else {})} for r in results]
    
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(ser, f, indent=2)
    with open(os.path.join(out_dir, 'analysis.json'), 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    # Print
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Scored: {len(results)}, Trained: {len(trained)}")
    
    for key in ['mixer', 'norm', 'downsample']:
        print(f"\n--- {key.title()} (NASWOT) ---")
        for k, v in sorted(analysis[f'{key}_naswot'].items(), key=lambda x: -x[1]['mean']):
            print(f"  {k:15s}: {v['mean']:.2f} ± {v['std']:.2f} (n={v['n']})")
    
    p = analysis['pca']
    print(f"\n--- PCA: total={p['total']}, eff90={p['eff_dim_90']}, eff95={p['eff_dim_95']} ---")
    
    if 'proxy_vs_acc' in analysis:
        print("\n--- Proxy vs Accuracy ---")
        for k, v in analysis['proxy_vs_acc'].items():
            print(f"  {k:12s}: ρ={v['spearman']:.3f}, τ={v['kendall']:.3f}")
    
    print(f"\nParams: [{analysis['param_stats']['min']:,} - {analysis['param_stats']['max']:,}]")
    print(f"\nTotal: {time.time()-t0:.0f}s")
    print(f"Saved to {out_dir}/")

if __name__ == '__main__':
    main()
