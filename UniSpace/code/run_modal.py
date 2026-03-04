"""
Run the scaled NAS experiment on Modal with a GPU.
Usage:
  modal run run_modal.py
Results are written to a Modal Volume and downloaded locally.
"""
import modal
import json

app = modal.App("nas-experiment")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch", "numpy", "scipy")
)

volume = modal.Volume.from_name("nas-results", create_if_missing=True)


@app.function(
    gpu="T4",
    image=image,
    volumes={"/results": volume},
    timeout=1800,
    memory=8192,
)
def run_experiment():
    import sys, os, json, time, gc, math
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from collections import defaultdict

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(42)

    # ============================================================
    # Search Space (inlined to avoid file dependency issues)
    # ============================================================

    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps
        def forward(self, x):
            if x.dim() == 4:
                rms = torch.sqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
                return x / rms * self.weight[None, :, None, None]
            else:
                rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
                return x / rms * self.weight

    def get_norm(norm_type, channels):
        if norm_type == 'batch': return nn.BatchNorm2d(channels)
        elif norm_type == 'layer': return nn.GroupNorm(1, channels)
        elif norm_type == 'group':
            ng = min(32, channels)
            while channels % ng != 0: ng -= 1
            return nn.GroupNorm(ng, channels)
        elif norm_type == 'rms': return RMSNorm(channels)
        else: raise ValueError(f"Unknown norm: {norm_type}")

    class DepthwiseConvMixer(nn.Module):
        def __init__(self, dim, kernel_size=7):
            super().__init__()
            self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)
            self.pwconv = nn.Conv2d(dim, dim, 1)
        def forward(self, x): return self.pwconv(self.dwconv(x))

    class AttentionMixer(nn.Module):
        def __init__(self, dim, num_heads=4):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = max(dim // num_heads, 1)
            self.scale = self.head_dim ** -0.5
            self.qkv = nn.Conv2d(dim, dim * 3, 1)
            self.proj = nn.Conv2d(dim, dim, 1)
        def forward(self, x):
            B, C, H, W = x.shape
            qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W)
            q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
            attn = (q.transpose(-2, -1) @ k) * self.scale
            attn = attn.softmax(dim=-1)
            out = (v @ attn.transpose(-2, -1)).reshape(B, C, H, W)
            return self.proj(out)

    class GatedMLPMixer(nn.Module):
        def __init__(self, dim, spatial_size=None):
            super().__init__()
            self.proj_in = nn.Conv2d(dim, dim * 2, 1)
            self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
            self.proj_out = nn.Conv2d(dim, dim, 1)
        def forward(self, x):
            xz = self.proj_in(x)
            u, v = xz.chunk(2, dim=1)
            v = self.dwconv(v)
            return self.proj_out(u * F.silu(v))

    class SSMLiteMixer(nn.Module):
        def __init__(self, dim, state_dim=16):
            super().__init__()
            self.dim = dim; self.state_dim = state_dim
            self.proj_in = nn.Conv2d(dim, dim * 2, 1)
            self.conv1d = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
            self.dt_proj = nn.Linear(dim, dim)
            self.A = nn.Parameter(torch.randn(dim, state_dim))
            self.D = nn.Parameter(torch.ones(dim))
            self.proj_out = nn.Conv2d(dim, dim, 1)
        def forward(self, x):
            B, C, H, W = x.shape
            xz = self.proj_in(x); x_in, z = xz.chunk(2, dim=1)
            x_flat = x_in.reshape(B, C, H*W)
            x_conv = F.silu(self.conv1d(x_flat))
            dt = F.softplus(self.dt_proj(x_conv.transpose(1, 2))).transpose(1, 2)
            y = self.D[None, :, None] * x_conv + dt * x_conv
            z_act = F.silu(z.reshape(B, C, H*W))
            return self.proj_out((y * z_act).reshape(B, C, H, W))

    def get_token_mixer(mixer_type, dim):
        if mixer_type == 'conv': return DepthwiseConvMixer(dim)
        elif mixer_type == 'attention': return AttentionMixer(dim)
        elif mixer_type == 'gated_mlp': return GatedMLPMixer(dim)
        elif mixer_type == 'ssm_lite': return SSMLiteMixer(dim)
        else: raise ValueError(f"Unknown mixer: {mixer_type}")

    class PatchMergingDown(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Conv2d(in_channels * 4, out_channels, 1)
        def forward(self, x):
            B, C, H, W = x.shape
            if H % 2 == 1: x = F.pad(x, (0, 0, 0, 1))
            if W % 2 == 1: x = F.pad(x, (0, 1, 0, 0))
            B, C, H, W = x.shape
            x0 = x[:, :, 0::2, 0::2]; x1 = x[:, :, 1::2, 0::2]
            x2 = x[:, :, 0::2, 1::2]; x3 = x[:, :, 1::2, 1::2]
            return self.conv(torch.cat([x0, x1, x2, x3], dim=1))

    def get_downsample_module(ds_type, in_ch, out_ch):
        if ds_type == 'maxpool':
            layers = [nn.MaxPool2d(2, 2)]
            if in_ch != out_ch: layers.append(nn.Conv2d(in_ch, out_ch, 1))
            return nn.Sequential(*layers)
        elif ds_type == 'avgpool':
            layers = [nn.AvgPool2d(2, 2)]
            if in_ch != out_ch: layers.append(nn.Conv2d(in_ch, out_ch, 1))
            return nn.Sequential(*layers)
        elif ds_type == 'strided_conv': return nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        elif ds_type == 'patch_merging': return PatchMergingDown(in_ch, out_ch)
        else: raise ValueError(f"Unknown downsample: {ds_type}")

    class MetaBlock(nn.Module):
        def __init__(self, dim, mixer_type, norm_type, ffn_ratio=4):
            super().__init__()
            self.norm1 = get_norm(norm_type, dim)
            self.mixer = get_token_mixer(mixer_type, dim)
            self.norm2 = get_norm(norm_type, dim)
            ffn_hidden = int(dim * ffn_ratio)
            self.ffn = nn.Sequential(nn.Conv2d(dim, ffn_hidden, 1), nn.GELU(), nn.Conv2d(ffn_hidden, dim, 1))
        def forward(self, x):
            x = x + self.mixer(self.norm1(x))
            x = x + self.ffn(self.norm2(x))
            return x

    class UnifiedBackbone(nn.Module):
        def __init__(self, config, num_classes=10, input_size=32):
            super().__init__()
            stem_ch = config['stem_channels']
            self.stem = nn.Sequential(nn.Conv2d(3, stem_ch, 3, stride=1, padding=1), nn.BatchNorm2d(stem_ch), nn.GELU())
            stages = []; in_ch = stem_ch
            for i, sc in enumerate(config['stages']):
                out_ch = max(in_ch * sc['expansion'], in_ch)
                if i > 0: stages.append(get_downsample_module(sc['downsample'], in_ch, out_ch))
                elif in_ch != out_ch: stages.append(nn.Conv2d(in_ch, out_ch, 1))
                for _ in range(sc['depth']): stages.append(MetaBlock(out_ch, sc['mixer'], sc['norm']))
                in_ch = out_ch
            self.stages = nn.Sequential(*stages)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Linear(in_ch, num_classes)
        def forward(self, x):
            x = self.stem(x); x = self.stages(x); x = self.pool(x).flatten(1)
            return self.head(x)

    SEARCH_SPACE = {
        'stem_channels': [32, 48, 64],
        'mixer_types': ['conv', 'attention', 'gated_mlp', 'ssm_lite'],
        'norm_types': ['batch', 'layer', 'group', 'rms'],
        'downsample_types': ['maxpool', 'avgpool', 'strided_conv', 'patch_merging'],
        'expansion_ratios': [1, 2],
        'depths': [1, 2, 3],
    }

    def compute_search_space_size():
        n = len(SEARCH_SPACE['stem_channels'])
        per_stage = len(SEARCH_SPACE['mixer_types']) * len(SEARCH_SPACE['norm_types']) * len(SEARCH_SPACE['downsample_types']) * len(SEARCH_SPACE['expansion_ratios']) * len(SEARCH_SPACE['depths'])
        return n * (per_stage ** 4)

    def sample_random_config(seed=None):
        rng = np.random.RandomState(seed) if seed is not None else np.random
        config = {'stem_channels': int(rng.choice(SEARCH_SPACE['stem_channels'])), 'stages': []}
        for _ in range(4):
            config['stages'].append({
                'depth': int(rng.choice(SEARCH_SPACE['depths'])),
                'mixer': str(rng.choice(SEARCH_SPACE['mixer_types'])),
                'norm': str(rng.choice(SEARCH_SPACE['norm_types'])),
                'downsample': str(rng.choice(SEARCH_SPACE['downsample_types'])),
                'expansion': int(rng.choice(SEARCH_SPACE['expansion_ratios'])),
            })
        return config

    def config_to_vector(config):
        mixer_map = {'conv': 0, 'attention': 1, 'gated_mlp': 2, 'ssm_lite': 3}
        norm_map = {'batch': 0, 'layer': 1, 'group': 2, 'rms': 3}
        ds_map = {'maxpool': 0, 'avgpool': 1, 'strided_conv': 2, 'patch_merging': 3}
        stem_map = {32: 0, 48: 1, 64: 2}
        vec = [stem_map.get(config['stem_channels'], 0)]
        for s in config['stages']:
            vec.extend([mixer_map[s['mixer']], norm_map[s['norm']], ds_map[s['downsample']], s['expansion'], s['depth']])
        return np.array(vec, dtype=np.float32)

    def count_parameters(model): return sum(p.numel() for p in model.parameters())

    # ============================================================
    # Scoring functions (GPU-accelerated)
    # ============================================================
    DATA_32 = torch.randn(8, 3, 32, 32, device=device)
    DATA_16 = torch.randn(8, 3, 16, 16, device=device)
    ONES_32 = torch.ones(1, 3, 32, 32, device=device)
    ONES_16 = torch.ones(1, 3, 16, 16, device=device)
    LABELS_10 = torch.randint(0, 10, (8,), device=device)
    LABELS_100 = torch.randint(0, 100, (8,), device=device)

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
        K = A @ A.T + 1e-5 * torch.eye(bs, device=data.device)
        try: return float(torch.slogdet(K)[1].item())
        except: return 0.0

    def synflow(model, ones):
        model.eval()
        for p in model.parameters(): p.data.abs_()
        model.zero_grad()
        model(ones).sum().backward()
        s = sum((p.data * p.grad.data).sum().item() for p in model.parameters() if p.grad is not None)
        model.zero_grad()
        return float(np.log(abs(s) + 1e-10))

    def gradnorm(model, data, labels):
        model.train(); model.zero_grad()
        F.cross_entropy(model(data), labels).backward()
        g = sum(p.grad.norm(2).item()**2 for p in model.parameters() if p.grad is not None)
        model.zero_grad()
        return float(np.log(np.sqrt(g) + 1e-10))

    def snip(model, data, labels):
        model.train(); model.zero_grad()
        F.cross_entropy(model(data), labels).backward()
        s = sum((p.data * p.grad.data).abs().sum().item() for p in model.parameters() if p.grad is not None)
        model.zero_grad()
        return float(np.log(s + 1e-10))

    def score_arch(cfg, nc=10, sz=32):
        data = DATA_32 if sz == 32 else DATA_16
        ones = ONES_32 if sz == 32 else ONES_16
        labels = LABELS_10[:data.size(0)] if nc == 10 else LABELS_100[:data.size(0)]
        sc = {}
        m = UnifiedBackbone(cfg, num_classes=nc, input_size=sz).to(device)
        sc['params'] = count_parameters(m)
        sc['naswot'] = naswot(m, data); del m
        m = UnifiedBackbone(cfg, num_classes=nc, input_size=sz).to(device)
        sc['synflow'] = synflow(m, ones); del m
        m = UnifiedBackbone(cfg, num_classes=nc, input_size=sz).to(device)
        sc['gradnorm'] = gradnorm(m, data, labels); del m
        m = UnifiedBackbone(cfg, num_classes=nc, input_size=sz).to(device)
        sc['snip'] = snip(m, data, labels); del m
        sc['log_params'] = float(np.log(sc['params'] + 1))
        return sc

    # Shared training data on GPU
    TRAIN_X = torch.randn(256, 3, 32, 32, device=device)
    TRAIN_Y = torch.randint(0, 10, (256,), device=device)
    VAL_X = torch.randn(64, 3, 32, 32, device=device)
    VAL_Y = torch.randint(0, 10, (64,), device=device)

    def train_eval(cfg, epochs=3, bs=64):
        model = UnifiedBackbone(cfg, num_classes=10, input_size=32).to(device)
        model.train()
        opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        for _ in range(epochs):
            model.train()
            idx = torch.randperm(256, device=device)
            for i in range(0, 256, bs):
                ii = idx[i:i+bs]
                opt.zero_grad()
                F.cross_entropy(model(TRAIN_X[ii]), TRAIN_Y[ii]).backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(VAL_X).argmax(1) == VAL_Y).float().mean().item()
        del model
        return float(acc)

    # ============================================================
    # Main experiment
    # ============================================================
    N = 500
    N_TRAIN_PER_GROUP = 10
    out_dir = "/results"

    print(f"Search space: {compute_search_space_size():,}")
    print(f"Target: {N} scored, {N_TRAIN_PER_GROUP*3} trained\n")

    # Phase 1: Score 500 architectures
    results = []
    t0 = time.time()
    failures = 0
    for i in range(N * 2):
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
    print(f"Phase 1 done: {len(results)} scored in {time.time()-t0:.0f}s (failures={failures})")

    # Phase 2: Multi-resolution
    print(f"\nPhase 2: Multi-resolution (16x16, nc=100) for 200 archs...")
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

    # Phase 3: Top-k regret
    print(f"\nPhase 3: Top-k regret ({N_TRAIN_PER_GROUP}/group, 3 epochs)...")
    t2 = time.time()
    scored = [(i, r['scores']['naswot']) for i, r in enumerate(results) if r['scores']['naswot'] != 0]
    scored.sort(key=lambda x: -x[1])
    top_idx = [s[0] for s in scored[:N_TRAIN_PER_GROUP]]
    bot_idx = [s[0] for s in scored[-N_TRAIN_PER_GROUP:]]
    mid_idx = [s[0] for s in scored[N_TRAIN_PER_GROUP:-N_TRAIN_PER_GROUP]]
    np.random.seed(42)
    rnd_idx = list(np.random.choice(mid_idx, size=min(N_TRAIN_PER_GROUP, len(mid_idx)), replace=False))

    for label, indices in [('top', top_idx), ('bottom', bot_idx), ('random', rnd_idx)]:
        for j, idx in enumerate(indices):
            r = results[idx]
            try:
                acc = train_eval(r['config'], epochs=3)
                r['train_acc_3ep'] = acc
                r['selection_group'] = label
                print(f"  {label}[{j+1}/{len(indices)}]: acc={acc:.3f}")
            except Exception as e:
                r['train_acc_3ep'] = 0.0
                r['selection_group'] = label
                print(f"  {label}[{j+1}/{len(indices)}]: FAILED ({e})")
    print(f"Phase 3 done in {time.time()-t2:.0f}s")

    # Phase 4: Analysis
    print("\nPhase 4: Analysis...")
    analysis = {}

    # Primitive dominance
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

    # Score correlation
    snames = ['naswot', 'synflow', 'gradnorm', 'snip', 'log_params']
    M = np.array([[r['scores'].get(s, 0) for s in snames] for r in results])
    ok = np.all(np.isfinite(M), axis=1) & np.all(M != 0, axis=1)
    M_valid = M[ok]
    if len(M_valid) > 10:
        analysis['score_corr'] = {'names': snames, 'matrix': np.corrcoef(M_valid.T).tolist(), 'n': len(M_valid)}

    # PCA
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

    # Proxy vs accuracy
    trained = [r for r in results if 'train_acc_3ep' in r]
    if len(trained) > 5:
        from scipy.stats import spearmanr, kendalltau
        accs = np.array([r['train_acc_3ep'] for r in trained])
        pc = {}
        for s in ['naswot', 'synflow', 'gradnorm', 'snip']:
            sv = np.array([r['scores'].get(s, 0) for r in trained])
            ok2 = np.isfinite(sv) & (sv != 0) & np.isfinite(accs)
            if ok2.sum() > 5:
                rho, p = spearmanr(sv[ok2], accs[ok2])
                tau, tp = kendalltau(sv[ok2], accs[ok2])
                pc[s] = {'spearman': round(float(rho), 3), 'sp_p': round(float(p), 4),
                         'kendall': round(float(tau), 3), 'kt_p': round(float(tp), 4), 'n': int(ok2.sum())}
        analysis['proxy_vs_acc'] = pc

        group_accs = defaultdict(list)
        for r in trained:
            group_accs[r.get('selection_group', 'unknown')].append(r['train_acc_3ep'])
        analysis['regret_analysis'] = {
            k: {'mean': round(float(np.mean(v)), 4), 'std': round(float(np.std(v)), 4),
                'min': round(float(np.min(v)), 4), 'max': round(float(np.max(v)), 4), 'n': len(v)}
            for k, v in group_accs.items()
        }

    # Stage-wise mixer effect
    if len(trained) > 5:
        se = {}
        for si in range(4):
            ma = defaultdict(list)
            for r in trained: ma[r['config']['stages'][si]['mixer']].append(r['train_acc_3ep'])
            se[f'stage_{si}'] = {
                k: {'mean': round(float(np.mean(v)), 4), 'std': round(float(np.std(v)), 4), 'n': len(v)}
                for k, v in ma.items() if len(v) > 0
            }
        analysis['stage_mixer_effect'] = se

    # Cross-resolution correlation
    cross_res = [r for r in results if r.get('scores_16x16_c100') is not None]
    if len(cross_res) > 10:
        from scipy.stats import spearmanr
        nw32 = np.array([r['scores']['naswot'] for r in cross_res])
        nw16 = np.array([r['scores_16x16_c100']['naswot'] for r in cross_res])
        ok3 = np.isfinite(nw32) & np.isfinite(nw16) & (nw32 != 0) & (nw16 != 0)
        if ok3.sum() > 5:
            rho, p = spearmanr(nw32[ok3], nw16[ok3])
            analysis['cross_resolution'] = {'spearman': round(float(rho), 3), 'p': round(float(p), 4), 'n': int(ok3.sum())}

    # Param stats
    ps = [r['params'] for r in results]
    analysis['param_stats'] = {
        'min': int(min(ps)), 'max': int(max(ps)),
        'mean': round(float(np.mean(ps))), 'median': round(float(np.median(ps)))
    }

    # NASWOT distribution
    nw = [r['scores']['naswot'] for r in results if r['scores']['naswot'] != 0]
    if nw:
        h, b = np.histogram(nw, bins=25)
        analysis['naswot_dist'] = {'hist': h.tolist(), 'bins': b.tolist(),
                                    'mean': round(float(np.mean(nw)), 2), 'std': round(float(np.std(nw)), 2)}

    # Top-5 architectures
    scored_all = sorted(results, key=lambda r: -r['scores']['naswot'])
    analysis['top5_architectures'] = [
        {'config': r['config'], 'params': r['params'],
         'naswot': r['scores']['naswot'], 'synflow': r['scores']['synflow'],
         'train_acc': r.get('train_acc_3ep', None)}
        for r in scored_all[:5]
    ]

    # ============================================================
    # Save results
    # ============================================================
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

    with open(os.path.join(out_dir, 'results_v2.json'), 'w') as f:
        json.dump(ser, f, indent=2)
    with open(os.path.join(out_dir, 'analysis_v2.json'), 'w') as f:
        json.dump(analysis, f, indent=2, default=str)

    volume.commit()

    total = time.time() - t0
    print(f"\n{'='*70}")
    print(f"DONE: N={len(results)}, trained={len(trained)}, time={total:.0f}s")
    print(f"{'='*70}")

    # Print summary
    print(f"\n--- Token Mixer NASWOT (N={len(results)}) ---")
    for k, v in sorted(analysis.get('mixer_naswot', {}).items(), key=lambda x: -x[1]['mean']):
        print(f"  {k:15s}: {v['mean']:.2f} +/- {v['std']:.2f} (n={v['n']})")

    if 'regret_analysis' in analysis:
        print(f"\n--- Top-k Regret Analysis ---")
        for k, v in sorted(analysis['regret_analysis'].items()):
            print(f"  {k:10s}: mean={v['mean']:.4f} +/- {v['std']:.4f} n={v['n']}")

    if 'cross_resolution' in analysis:
        cr = analysis['cross_resolution']
        print(f"\n--- Cross-Resolution NASWOT rho={cr['spearman']:.3f} (n={cr['n']}) ---")

    print(f"\n--- Top-5 Architectures ---")
    for i, a in enumerate(analysis.get('top5_architectures', [])):
        mixers = [s['mixer'] for s in a['config']['stages']]
        acc_s = f"acc={a['train_acc']:.3f}" if a['train_acc'] else "not trained"
        print(f"  #{i+1}: NASWOT={a['naswot']:.1f}, params={a['params']:,}, mixers={mixers}, {acc_s}")

    return {"results_count": len(results), "trained_count": len(trained), "time_s": total}


@app.local_entrypoint()
def main():
    result = run_experiment.remote()
    print(f"\nExperiment returned: {result}")

    # Download results from volume
    import subprocess
    print("\nDownloading results from Modal volume...")
    subprocess.run(["modal", "volume", "get", "nas-results", "results_v2.json", "--force"], check=False)
    subprocess.run(["modal", "volume", "get", "nas-results", "analysis_v2.json", "--force"], check=False)
    print("Done!")
