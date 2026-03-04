"""
AdapterNAS v2 — Split into independent Modal functions with incremental saves.
Each experiment saves to Volume immediately so partial results survive timeouts.
"""

import modal
import json

app = modal.App("adapter-nas-v2b")
vol = modal.Volume.from_name("adapter-nas-results-v2", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch", "torchvision", "timm", "peft", "transformers",
        "numpy", "scipy", "accelerate",
    )
)

# ============================================================
# Shared code as a string — injected into each function
# ============================================================
SHARED_CODE = """
import os, time, random, copy, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
import timm
from scipy import stats

device = 'cuda'

class LoRALinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.rank = rank
        for p in self.base.parameters(): p.requires_grad = False
        self.A = nn.Parameter(torch.randn(base.in_features, rank, device='cpu') * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, base.out_features, device='cpu'))
        self.s = float(rank) / rank
    def forward(self, x):
        return self.base(x) + (x @ self.A @ self.B) * self.s

MODULE_TYPES = ['qkv', 'mlp_fc1', 'mlp_fc2']
NUM_LAYERS = 12
RANKS = [0, 4, 8, 16, 32]

def count_params(cfg, h=768, m=3072):
    t = 0
    for (l, mod), r in cfg.items():
        if r == 0: continue
        if mod == 'qkv': t += h*r + r*3*h
        elif mod == 'mlp_fc1': t += h*r + r*m
        elif mod == 'mlp_fc2': t += m*r + r*h
    return t

def count_params_small(cfg, h=384, m=1536):
    t = 0
    for (l, mod), r in cfg.items():
        if r == 0: continue
        if mod == 'qkv': t += h*r + r*3*h
        elif mod == 'mlp_fc1': t += h*r + r*m
        elif mod == 'mlp_fc2': t += m*r + r*h
    return t

def apply_lora(model, cfg, num_layers=12):
    for p in model.parameters(): p.requires_grad = False
    for l in range(num_layers):
        b = model.blocks[l]
        r = cfg.get((l,'qkv'),0)
        if r > 0: b.attn.qkv = LoRALinear(b.attn.qkv, r)
        r = cfg.get((l,'mlp_fc1'),0)
        if r > 0: b.mlp.fc1 = LoRALinear(b.mlp.fc1, r)
        r = cfg.get((l,'mlp_fc2'),0)
        if r > 0: b.mlp.fc2 = LoRALinear(b.mlp.fc2, r)
    for p in model.head.parameters(): p.requires_grad = True
    return model

def get_loaders(dataset_name, batch_size=64, train_frac=0.02, cal_size=128):
    tr = T.Compose([T.Resize(224), T.RandomCrop(224, padding=4),
                    T.RandomHorizontalFlip(), T.ToTensor(),
                    T.Normalize([.485,.456,.406],[.229,.224,.225])])
    te = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize([.485,.456,.406],[.229,.224,.225])])
    if dataset_name == 'cifar100':
        trainset = torchvision.datasets.CIFAR100('/tmp/data', train=True, download=True, transform=tr)
        testset = torchvision.datasets.CIFAR100('/tmp/data', train=False, download=True, transform=te)
        nc = 100
    else:
        trainset = torchvision.datasets.Flowers102('/tmp/data', split='train', download=True, transform=tr)
        testset = torchvision.datasets.Flowers102('/tmp/data', split='test', download=True, transform=te)
        nc = 102
    cal_loader = DataLoader(Subset(trainset, range(min(cal_size, len(trainset)))),
                            batch_size=cal_size, shuffle=False, num_workers=4, pin_memory=True)
    n_train = max(int(train_frac * len(trainset)), 50)
    train_loader = DataLoader(Subset(trainset, range(n_train)),
                              batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(Subset(testset, range(min(2000, len(testset)))),
                            batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return cal_loader, train_loader, val_loader, nc

def compute_proxies(model, cal_loader, device):
    model.to(device); model.train(); model.zero_grad()
    imgs, labs = next(iter(cal_loader))
    imgs, labs = imgs.to(device), labs.to(device)
    out = model(imgs)
    F.cross_entropy(out, labs).backward()
    gn, sn, fi = 0.0, 0.0, 0.0
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            g = p.grad
            gn += g.norm(2).item()
            sn += (g * p.data).abs().sum().item()
            fi += (g ** 2).sum().item()
    model.eval()
    with torch.no_grad():
        o2 = model(imgs)
        pr = F.softmax(o2, dim=-1)
        ent = -(pr * (pr+1e-10).log()).sum(-1).mean().item()
    model.zero_grad()
    return {'gradnorm': gn, 'snip': sn, 'fisher': fi, 'neg_entropy': -ent}

def benchmark_latency(model, device, batch_size=1, n_warmup=20, n_runs=50):
    model.to(device); model.eval()
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return {'mean_ms': float(np.mean(times)), 'std_ms': float(np.std(times)),
            'p50_ms': float(np.median(times)), 'p95_ms': float(np.percentile(times, 95))}

def finetune(model, train_loader, val_loader, device, epochs=5, lr=1e-3):
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params: return evaluate(model, val_loader, device)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(train_loader))
    best_acc = 0
    for ep in range(epochs):
        model.train()
        for imgs, labs in train_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(imgs), labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
        acc = evaluate(model, val_loader, device)
        best_acc = max(best_acc, acc)
    return best_acc

def evaluate(model, loader, device):
    model.to(device); model.eval()
    c, t = 0, 0
    with torch.no_grad():
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            _, pred = model(imgs).max(1)
            t += labs.size(0); c += pred.eq(labs).sum().item()
    return 100.*c/t

def sample_structured_configs():
    configs, labels = [], []
    for r in [4, 8, 16, 32]:
        c = {(l, m): r for l in range(12) for m in MODULE_TYPES}
        configs.append(c); labels.append(f'uniform_r{r}')
    for r in [4, 8, 16, 32]:
        c = {(l, m): (r if m == 'qkv' else 0) for l in range(12) for m in MODULE_TYPES}
        configs.append(c); labels.append(f'attn_only_r{r}')
    for r in [4, 8, 16]:
        c = {(l, m): (r if m != 'qkv' else 0) for l in range(12) for m in MODULE_TYPES}
        configs.append(c); labels.append(f'mlp_only_r{r}')
    for tag, cond in [('last4', lambda l: l >= 8), ('first4', lambda l: l < 4),
                      ('even', lambda l: l % 2 == 0), ('middle4', lambda l: 4 <= l < 8)]:
        for r in [8, 16]:
            c = {(l, m): (r if cond(l) else 0) for l in range(12) for m in MODULE_TYPES}
            configs.append(c); labels.append(f'{tag}_r{r}')
    for name, ranks in [('increasing', [4,4,4,8,8,8,16,16,16,32,32,32]),
                        ('decreasing', [32,32,32,16,16,16,8,8,8,4,4,4])]:
        c = {(l, m): ranks[l] for l in range(12) for m in MODULE_TYPES}
        configs.append(c); labels.append(name)
    for name, qkv_r, mlp_r in [('high_attn_low_mlp', 16, 4), ('low_attn_high_mlp', 4, 16),
                                 ('balanced_8', 8, 8)]:
        c = {}
        for l in range(12):
            c[(l, 'qkv')] = qkv_r
            c[(l, 'mlp_fc1')] = mlp_r
            c[(l, 'mlp_fc2')] = mlp_r
        configs.append(c); labels.append(name)
    c = {(l, m): 0 for l in range(12) for m in MODULE_TYPES}
    configs.append(c); labels.append('no_adapter')
    return configs, labels

def sample_random(n, seed=42):
    rng = random.Random(seed)
    configs = []
    for _ in range(n):
        c = {(l, m): rng.choice(RANKS) for l in range(12) for m in MODULE_TYPES}
        configs.append(c)
    return configs

def serialize_config(cfg):
    return {f'{k[0]}_{k[1]}': v for k, v in cfg.items()}

def deserialize_config(d):
    cfg = {}
    for k, v in d.items():
        parts = k.split('_', 1)
        cfg[(int(parts[0]), parts[1])] = v
    return cfg
"""


# ============================================================
# EXPERIMENT 1: CIFAR-100 Search + Validate
# ============================================================
@app.function(gpu="A10G", image=image, volumes={"/results": vol}, timeout=5400, memory=32768)
def exp1_cifar100():
    exec(SHARED_CODE, globals())
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 70)
    print("  EXPERIMENT 1: CIFAR-100 Search + Validate")
    print("=" * 70)

    cal_loader, train_loader, val_loader, nc = get_loaders('cifar100', train_frac=0.02)

    # Phase 1a: Structured configs — proxy score only
    print("\n  Phase 1a: Scoring structured configs...")
    struct_configs, struct_labels = sample_structured_configs()
    proxy_results = []
    config_bank = {}  # label -> config

    for i, (cfg, lab) in enumerate(zip(struct_configs, struct_labels)):
        np_p = count_params(cfg)
        config_bank[lab] = cfg
        if np_p == 0:
            proxy_results.append({'label': lab, 'n_params': 0, 'gradnorm': 0, 'snip': 0, 'fisher': 0, 'neg_entropy': 0})
            continue
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=nc)
        model = apply_lora(model, cfg)
        proxies = compute_proxies(model, cal_loader, device)
        proxy_results.append({'label': lab, 'n_params': np_p, **proxies})
        del model; torch.cuda.empty_cache()
        print(f"    [{i+1}/{len(struct_configs)}] {lab}: params={np_p:,}")

    # Phase 1b: 30 random configs (reduced from 50)
    print("\n  Phase 1b: Scoring 30 random configs...")
    rand_configs = sample_random(30, seed=42)
    for i, cfg in enumerate(rand_configs):
        lab = f'rand_{i}'
        np_p = count_params(cfg)
        config_bank[lab] = cfg
        if np_p == 0:
            proxy_results.append({'label': lab, 'n_params': 0, 'gradnorm': 0, 'snip': 0, 'fisher': 0, 'neg_entropy': 0})
            continue
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=nc)
        model = apply_lora(model, cfg)
        proxies = compute_proxies(model, cal_loader, device)
        proxy_results.append({'label': lab, 'n_params': np_p, **proxies})
        del model; torch.cuda.empty_cache()
        if (i+1) % 10 == 0:
            print(f"    Random [{i+1}/30]")

    # Phase 1c: Evolutionary search — 6 generations x 20 pop = ~120 configs
    print("\n  Phase 1c: Evolutionary search...")
    rng = random.Random(42)

    def random_config():
        return {(l, m): rng.choice(RANKS) for l in range(12) for m in MODULE_TYPES}

    def mutate(cfg, n_mutations=3):
        new_cfg = dict(cfg)
        keys = list(new_cfg.keys())
        for _ in range(n_mutations):
            k = rng.choice(keys)
            new_cfg[k] = rng.choice(RANKS)
        return new_cfg

    def crossover(cfg1, cfg2):
        new_cfg = {}
        for k in cfg1:
            new_cfg[k] = cfg1[k] if rng.random() < 0.5 else cfg2[k]
        return new_cfg

    pop_size, n_gens = 20, 6
    population = [random_config() for _ in range(pop_size)]
    evo_all = []

    for gen in range(n_gens):
        scores = []
        for cfg in population:
            np_p = count_params(cfg)
            if np_p == 0:
                proxies = {'gradnorm': 0, 'snip': 0, 'fisher': 0, 'neg_entropy': 0}
            else:
                model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=nc)
                model = apply_lora(model, cfg)
                proxies = compute_proxies(model, cal_loader, device)
                del model; torch.cuda.empty_cache()
            scores.append(proxies)
            evo_all.append({'config': cfg, 'proxies': proxies, 'n_params': np_p})

        # Normalize and rank
        for metric in ['gradnorm', 'snip', 'fisher', 'neg_entropy']:
            vals = [s[metric] for s in scores]
            lo, hi = min(vals), max(vals)
            rng_val = hi - lo if hi != lo else 1.0
            for s in scores:
                s[f'{metric}_n'] = (s[metric] - lo) / rng_val

        combined = [(s['gradnorm_n'] + s['snip_n'] + s['fisher_n'] + s['neg_entropy_n']) / 4.0 for s in scores]
        ranked = sorted(zip(combined, population), key=lambda x: -x[0])
        print(f"    Gen {gen+1}/{n_gens}: best={ranked[0][0]:.4f}, mean={np.mean(combined):.4f}, total_seen={len(evo_all)}")

        survivors = [cfg for _, cfg in ranked[:pop_size // 2]]
        next_gen = list(survivors)
        for _ in range(12):
            next_gen.append(mutate(rng.choice(survivors), n_mutations=rng.randint(1, 5)))
        while len(next_gen) < pop_size:
            p1, p2 = rng.sample(survivors, 2)
            next_gen.append(crossover(p1, p2))
        population = next_gen[:pop_size]

    # Add evo configs to proxy results
    evo_counter = 0
    for e in evo_all:
        if e['n_params'] > 0:
            lab = f'evo_{evo_counter}'
            proxy_results.append({'label': lab, 'n_params': e['n_params'], **e['proxies']})
            config_bank[lab] = e['config']
            evo_counter += 1

    print(f"\n  Total configs scored: {len(proxy_results)}")

    # Normalize all and compute combined score
    valid = [r for r in proxy_results if r['n_params'] > 0]
    for metric in ['gradnorm', 'snip', 'fisher', 'neg_entropy']:
        vals = [r[metric] for r in valid]
        lo, hi = min(vals), max(vals)
        rng_val = hi - lo if hi != lo else 1.0
        for r in valid:
            r[f'{metric}_n'] = (r[metric] - lo) / rng_val
    for r in valid:
        r['combined'] = (r['gradnorm_n'] + r['snip_n'] + r['fisher_n'] + r['neg_entropy_n']) / 4.0
    valid.sort(key=lambda x: x['combined'], reverse=True)

    # Phase 2: Fine-tune selected configs
    print(f"\n  Phase 2: Fine-tuning for selection quality...")
    ft_labels = set()
    ft_queue = []

    # Top-8 by proxy
    for r in valid[:8]:
        if r['label'] not in ft_labels:
            ft_queue.append(r); ft_labels.add(r['label'])
    # Bottom-8
    for r in valid[-8:]:
        if r['label'] not in ft_labels:
            ft_queue.append(r); ft_labels.add(r['label'])
    # 8 from middle
    mid = valid[len(valid)//3 : 2*len(valid)//3]
    rng2 = random.Random(123)
    rng2.shuffle(mid)
    for r in mid[:8]:
        if r['label'] not in ft_labels:
            ft_queue.append(r); ft_labels.add(r['label'])
    # Key baselines
    for bl in ['uniform_r4', 'uniform_r8', 'uniform_r16', 'uniform_r32',
               'attn_only_r8', 'attn_only_r16', 'mlp_only_r8', 'mlp_only_r16',
               'last4_r8', 'last4_r16', 'first4_r8', 'first4_r16',
               'even_r8', 'even_r16', 'middle4_r8', 'middle4_r16',
               'increasing', 'decreasing',
               'high_attn_low_mlp', 'low_attn_high_mlp', 'balanced_8']:
        for r in valid:
            if r['label'] == bl and r['label'] not in ft_labels:
                ft_queue.append(r); ft_labels.add(r['label'])

    print(f"  Fine-tuning {len(ft_queue)} configs...")
    ft_results = []

    for fi, r in enumerate(ft_queue):
        lab = r['label']
        cfg = config_bank.get(lab)
        if cfg is None:
            continue
        print(f"  [{fi+1}/{len(ft_queue)}] {lab} (params={r['n_params']:,}, proxy={r.get('combined',0):.3f})")
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=nc)
        model = apply_lora(model, cfg)
        val_acc = finetune(model, train_loader, val_loader, device, epochs=5, lr=1e-3)
        lat = benchmark_latency(model, device)
        ft_results.append({
            'label': lab, 'n_params': r['n_params'],
            'combined_proxy': r.get('combined', 0),
            'gradnorm': r['gradnorm'], 'snip': r['snip'],
            'fisher': r['fisher'], 'neg_entropy': r['neg_entropy'],
            'val_acc': val_acc, **lat,
        })
        print(f"    -> Acc={val_acc:.2f}%, Lat={lat['mean_ms']:.1f}+-{lat['std_ms']:.1f}ms")
        del model; torch.cuda.empty_cache()

    # Linear probe baseline
    print(f"  Linear probe baseline...")
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=nc)
    for p in model.parameters(): p.requires_grad = False
    for p in model.head.parameters(): p.requires_grad = True
    lp_acc = finetune(model, train_loader, val_loader, device, epochs=5, lr=1e-3)
    lp_lat = benchmark_latency(model, device)
    ft_results.append({
        'label': 'linear_probe', 'n_params': sum(p.numel() for p in model.head.parameters()),
        'combined_proxy': 0, 'gradnorm': 0, 'snip': 0, 'fisher': 0, 'neg_entropy': 0,
        'val_acc': lp_acc, **lp_lat,
    })
    print(f"    -> Linear probe: {lp_acc:.2f}%")
    del model; torch.cuda.empty_cache()

    # Selection quality
    acc_map = {r['label']: r['val_acc'] for r in ft_results}
    proxy_map = {r['label']: r.get('combined', 0) for r in valid}
    common_labels = [l for l in acc_map if l in proxy_map and l != 'linear_probe']

    sq = {}
    if len(common_labels) >= 5:
        by_proxy = sorted(common_labels, key=lambda l: proxy_map[l], reverse=True)
        by_oracle = sorted(common_labels, key=lambda l: acc_map[l], reverse=True)
        oracle_top1 = acc_map[by_oracle[0]]
        random_mean = np.mean([acc_map[l] for l in common_labels])

        for k in [1, 3, 5, 10]:
            if k > len(by_proxy): continue
            proxy_topk = by_proxy[:k]
            oracle_topk = set(by_oracle[:k])
            best_in_topk = max(acc_map[l] for l in proxy_topk)
            hits = sum(1 for l in proxy_topk if l in oracle_topk)
            sq[f'top{k}_best_acc'] = best_in_topk
            sq[f'top{k}_hit_rate'] = hits / k
            sq[f'top{k}_regret_vs_oracle'] = oracle_top1 - best_in_topk
            sq[f'top{k}_regret_vs_random'] = best_in_topk - random_mean

        proxy_vals = [proxy_map[l] for l in common_labels]
        acc_vals = [acc_map[l] for l in common_labels]
        rho, rho_p = stats.spearmanr(proxy_vals, acc_vals)
        tau, tau_p = stats.kendalltau(proxy_vals, acc_vals)
        sq['spearman_rho'] = float(rho)
        sq['spearman_p'] = float(rho_p)
        sq['kendall_tau'] = float(tau)
        sq['kendall_p'] = float(tau_p)

        for metric in ['gradnorm', 'snip', 'fisher', 'neg_entropy']:
            metric_map = {r['label']: r.get(metric, 0) for r in valid}
            metric_vals = [metric_map.get(l, 0) for l in common_labels]
            r_val, _ = stats.spearmanr(metric_vals, acc_vals)
            sq[f'{metric}_spearman'] = float(r_val)

    result = {
        'proxy_scores': proxy_results,
        'finetune_results': ft_results,
        'selection_quality': sq,
        'config_bank': {k: serialize_config(v) for k, v in config_bank.items()},
    }

    with open('/results/cifar100_results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    vol.commit()
    print("\n  CIFAR-100 SAVED!")

    # Print summary
    ft_results.sort(key=lambda x: x['val_acc'], reverse=True)
    print(f"\n  Top-10 CIFAR-100:")
    for r in ft_results[:10]:
        print(f"    {r['label']:<25} acc={r['val_acc']:.2f}%  params={r['n_params']:,}  proxy={r.get('combined_proxy',0):.3f}")
    print(f"\n  Selection Quality: {sq}")

    return result


# ============================================================
# EXPERIMENT 2: Flowers-102 Search + Validate
# ============================================================
@app.function(gpu="A10G", image=image, volumes={"/results": vol}, timeout=5400, memory=32768)
def exp2_flowers102():
    exec(SHARED_CODE, globals())
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 70)
    print("  EXPERIMENT 2: Flowers-102 Search + Validate")
    print("=" * 70)

    cal_loader, train_loader, val_loader, nc = get_loaders('flowers102', train_frac=0.10)

    # Score structured + random configs
    struct_configs, struct_labels = sample_structured_configs()
    proxy_results = []
    config_bank = {}

    print("\n  Scoring structured configs...")
    for i, (cfg, lab) in enumerate(zip(struct_configs, struct_labels)):
        np_p = count_params(cfg)
        config_bank[lab] = cfg
        if np_p == 0:
            proxy_results.append({'label': lab, 'n_params': 0, 'gradnorm': 0, 'snip': 0, 'fisher': 0, 'neg_entropy': 0})
            continue
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=nc)
        model = apply_lora(model, cfg)
        proxies = compute_proxies(model, cal_loader, device)
        proxy_results.append({'label': lab, 'n_params': np_p, **proxies})
        del model; torch.cuda.empty_cache()
        print(f"    [{i+1}/{len(struct_configs)}] {lab}: params={np_p:,}")

    print("\n  Scoring 20 random configs...")
    rand_configs = sample_random(20, seed=99)
    for i, cfg in enumerate(rand_configs):
        lab = f'rand_{i}'
        np_p = count_params(cfg)
        config_bank[lab] = cfg
        if np_p == 0:
            proxy_results.append({'label': lab, 'n_params': 0, 'gradnorm': 0, 'snip': 0, 'fisher': 0, 'neg_entropy': 0})
            continue
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=nc)
        model = apply_lora(model, cfg)
        proxies = compute_proxies(model, cal_loader, device)
        proxy_results.append({'label': lab, 'n_params': np_p, **proxies})
        del model; torch.cuda.empty_cache()

    # Normalize
    valid = [r for r in proxy_results if r['n_params'] > 0]
    for metric in ['gradnorm', 'snip', 'fisher', 'neg_entropy']:
        vals = [r[metric] for r in valid]
        lo, hi = min(vals), max(vals)
        rng_val = hi - lo if hi != lo else 1.0
        for r in valid:
            r[f'{metric}_n'] = (r[metric] - lo) / rng_val
    for r in valid:
        r['combined'] = (r['gradnorm_n'] + r['snip_n'] + r['fisher_n'] + r['neg_entropy_n']) / 4.0
    valid.sort(key=lambda x: x['combined'], reverse=True)

    # Fine-tune key configs
    ft_labels = set()
    ft_queue = []
    for r in valid[:5]:
        if r['label'] not in ft_labels: ft_queue.append(r); ft_labels.add(r['label'])
    for r in valid[-5:]:
        if r['label'] not in ft_labels: ft_queue.append(r); ft_labels.add(r['label'])
    for bl in ['uniform_r4', 'uniform_r8', 'uniform_r16', 'uniform_r32',
               'attn_only_r8', 'attn_only_r16', 'mlp_only_r8',
               'last4_r16', 'first4_r16', 'increasing', 'decreasing']:
        for r in valid:
            if r['label'] == bl and r['label'] not in ft_labels:
                ft_queue.append(r); ft_labels.add(r['label'])

    print(f"\n  Fine-tuning {len(ft_queue)} configs...")
    ft_results = []
    for fi, r in enumerate(ft_queue):
        lab = r['label']
        cfg = config_bank.get(lab)
        if cfg is None: continue
        print(f"  [{fi+1}/{len(ft_queue)}] {lab}")
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=nc)
        model = apply_lora(model, cfg)
        val_acc = finetune(model, train_loader, val_loader, device, epochs=5, lr=1e-3)
        lat = benchmark_latency(model, device)
        ft_results.append({
            'label': lab, 'n_params': r['n_params'],
            'combined_proxy': r.get('combined', 0),
            'gradnorm': r['gradnorm'], 'snip': r['snip'],
            'fisher': r['fisher'], 'neg_entropy': r['neg_entropy'],
            'val_acc': val_acc, **lat,
        })
        print(f"    -> Acc={val_acc:.2f}%")
        del model; torch.cuda.empty_cache()

    # Linear probe
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=nc)
    for p in model.parameters(): p.requires_grad = False
    for p in model.head.parameters(): p.requires_grad = True
    lp_acc = finetune(model, train_loader, val_loader, device, epochs=5, lr=1e-3)
    lp_lat = benchmark_latency(model, device)
    ft_results.append({
        'label': 'linear_probe', 'n_params': sum(p.numel() for p in model.head.parameters()),
        'combined_proxy': 0, 'gradnorm': 0, 'snip': 0, 'fisher': 0, 'neg_entropy': 0,
        'val_acc': lp_acc, **lp_lat,
    })
    del model; torch.cuda.empty_cache()

    # Selection quality
    acc_map = {r['label']: r['val_acc'] for r in ft_results}
    proxy_map = {r['label']: r.get('combined', 0) for r in valid}
    common_labels = [l for l in acc_map if l in proxy_map and l != 'linear_probe']
    sq = {}
    if len(common_labels) >= 5:
        by_proxy = sorted(common_labels, key=lambda l: proxy_map[l], reverse=True)
        by_oracle = sorted(common_labels, key=lambda l: acc_map[l], reverse=True)
        oracle_top1 = acc_map[by_oracle[0]]
        for k in [1, 3, 5]:
            if k > len(by_proxy): continue
            proxy_topk = by_proxy[:k]
            best_in_topk = max(acc_map[l] for l in proxy_topk)
            sq[f'top{k}_best_acc'] = best_in_topk
            sq[f'top{k}_regret_vs_oracle'] = oracle_top1 - best_in_topk
        rho, _ = stats.spearmanr([proxy_map[l] for l in common_labels], [acc_map[l] for l in common_labels])
        sq['spearman_rho'] = float(rho)

    result = {'proxy_scores': proxy_results, 'finetune_results': ft_results, 'selection_quality': sq}
    with open('/results/flowers102_results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    vol.commit()
    print("\n  FLOWERS-102 SAVED!")
    return result


# ============================================================
# EXPERIMENT 3: Data Regime Sweep (CIFAR-100)
# ============================================================
@app.function(gpu="A10G", image=image, volumes={"/results": vol}, timeout=3600, memory=32768)
def exp3_data_regime():
    exec(SHARED_CODE, globals())
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 70)
    print("  EXPERIMENT 3: Data Regime Sweep (CIFAR-100)")
    print("=" * 70)

    fracs = [0.01, 0.02, 0.05, 0.10]
    configs_to_test = {
        'uniform_r4': {(l, m): 4 for l in range(12) for m in MODULE_TYPES},
        'uniform_r8': {(l, m): 8 for l in range(12) for m in MODULE_TYPES},
        'uniform_r16': {(l, m): 16 for l in range(12) for m in MODULE_TYPES},
        'uniform_r32': {(l, m): 32 for l in range(12) for m in MODULE_TYPES},
        'attn_only_r8': {(l, m): (8 if m == 'qkv' else 0) for l in range(12) for m in MODULE_TYPES},
        'attn_only_r16': {(l, m): (16 if m == 'qkv' else 0) for l in range(12) for m in MODULE_TYPES},
        'mlp_only_r8': {(l, m): (8 if m != 'qkv' else 0) for l in range(12) for m in MODULE_TYPES},
        'increasing': {(l, m): [4,4,4,8,8,8,16,16,16,32,32,32][l] for l in range(12) for m in MODULE_TYPES},
    }

    results = []
    for frac in fracs:
        print(f"\n  Data fraction: {frac*100:.0f}%")
        _, train_loader, val_loader, nc = get_loaders('cifar100', train_frac=frac)
        for name, cfg in configs_to_test.items():
            model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=nc)
            model = apply_lora(model, cfg)
            acc = finetune(model, train_loader, val_loader, device, epochs=5, lr=1e-3)
            results.append({'config': name, 'data_frac': frac, 'val_acc': acc,
                           'n_params': count_params(cfg), 'n_train_samples': int(frac * 50000)})
            print(f"    {name}: {acc:.2f}%")
            del model; torch.cuda.empty_cache()

        # Linear probe
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=nc)
        for p in model.parameters(): p.requires_grad = False
        for p in model.head.parameters(): p.requires_grad = True
        acc = finetune(model, train_loader, val_loader, device, epochs=5, lr=1e-3)
        results.append({'config': 'linear_probe', 'data_frac': frac, 'val_acc': acc,
                       'n_params': 76900, 'n_train_samples': int(frac * 50000)})
        print(f"    linear_probe: {acc:.2f}%")
        del model; torch.cuda.empty_cache()

    with open('/results/data_regime_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    vol.commit()
    print("\n  DATA REGIME SAVED!")
    return results


# ============================================================
# EXPERIMENT 4: ViT-S/16
# ============================================================
@app.function(gpu="A10G", image=image, volumes={"/results": vol}, timeout=3600, memory=32768)
def exp4_vit_small():
    exec(SHARED_CODE, globals())
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 70)
    print("  EXPERIMENT 4: ViT-S/16 Backbone")
    print("=" * 70)

    cal_loader, train_loader, val_loader, nc = get_loaders('cifar100', train_frac=0.02)
    configs = {
        'uniform_r4': {(l, m): 4 for l in range(12) for m in MODULE_TYPES},
        'uniform_r8': {(l, m): 8 for l in range(12) for m in MODULE_TYPES},
        'uniform_r16': {(l, m): 16 for l in range(12) for m in MODULE_TYPES},
        'uniform_r32': {(l, m): 32 for l in range(12) for m in MODULE_TYPES},
        'attn_only_r8': {(l, m): (8 if m == 'qkv' else 0) for l in range(12) for m in MODULE_TYPES},
        'attn_only_r16': {(l, m): (16 if m == 'qkv' else 0) for l in range(12) for m in MODULE_TYPES},
        'mlp_only_r8': {(l, m): (8 if m != 'qkv' else 0) for l in range(12) for m in MODULE_TYPES},
        'last4_r16': {(l, m): (16 if l >= 8 else 0) for l in range(12) for m in MODULE_TYPES},
        'first4_r16': {(l, m): (16 if l < 4 else 0) for l in range(12) for m in MODULE_TYPES},
        'increasing': {(l, m): [4,4,4,8,8,8,16,16,16,32,32,32][l] for l in range(12) for m in MODULE_TYPES},
        'decreasing': {(l, m): [32,32,32,16,16,16,8,8,8,4,4,4][l] for l in range(12) for m in MODULE_TYPES},
    }

    results = []
    for name, cfg in configs.items():
        model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=nc)
        model = apply_lora(model, cfg)
        proxies = compute_proxies(model, cal_loader, device)
        acc = finetune(model, train_loader, val_loader, device, epochs=5, lr=1e-3)
        lat = benchmark_latency(model, device)
        results.append({'label': name, 'n_params': count_params_small(cfg),
                       'val_acc': acc, **proxies, **lat})
        print(f"  {name}: acc={acc:.2f}%, params={count_params_small(cfg):,}")
        del model; torch.cuda.empty_cache()

    # Linear probe
    model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=nc)
    for p in model.parameters(): p.requires_grad = False
    for p in model.head.parameters(): p.requires_grad = True
    acc = finetune(model, train_loader, val_loader, device, epochs=5, lr=1e-3)
    lat = benchmark_latency(model, device)
    results.append({'label': 'linear_probe', 'n_params': sum(p.numel() for p in model.head.parameters()),
                   'val_acc': acc, 'gradnorm': 0, 'snip': 0, 'fisher': 0, 'neg_entropy': 0, **lat})
    print(f"  linear_probe: acc={acc:.2f}%")
    del model; torch.cuda.empty_cache()

    with open('/results/vit_small_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    vol.commit()
    print("\n  VIT-S SAVED!")
    return results


# ============================================================
# EXPERIMENT 5: AdaLoRA Baseline
# ============================================================
@app.function(gpu="A10G", image=image, volumes={"/results": vol}, timeout=2400, memory=32768)
def exp5_adalora():
    exec(SHARED_CODE, globals())
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 70)
    print("  EXPERIMENT 5: AdaLoRA Baseline")
    print("=" * 70)

    from peft import get_peft_model, AdaLoraConfig
    from transformers import ViTForImageClassification

    _, train_loader, val_loader, nc = get_loaders('cifar100', train_frac=0.02)
    results = []

    # Our LoRA baselines for fair comparison
    for r, name in [(8, 'our_lora_r8'), (16, 'our_lora_r16')]:
        cfg = {(l, m): r for l in range(12) for m in MODULE_TYPES}
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=nc)
        model = apply_lora(model, cfg)
        acc = finetune(model, train_loader, val_loader, device, epochs=5, lr=1e-3)
        lat = benchmark_latency(model, device)
        results.append({'label': name, 'n_params': count_params(cfg), 'val_acc': acc, **lat})
        print(f"  {name}: acc={acc:.2f}%")
        del model; torch.cuda.empty_cache()

    # AdaLoRA via peft + HuggingFace ViT
    for init_r, name in [(12, 'adalora_r12'), (24, 'adalora_r24')]:
        try:
            hf_model = ViTForImageClassification.from_pretrained(
                'google/vit-base-patch16-224-in21k', num_labels=nc, ignore_mismatched_sizes=True)
            adalora_config = AdaLoraConfig(
                r=init_r, lora_alpha=init_r,
                target_modules=["query", "value", "key", "dense"],
                lora_dropout=0.05, init_r=init_r, target_r=init_r // 2,
                deltaT=10, beta1=0.85, beta2=0.85,
            )
            hf_model = get_peft_model(hf_model, adalora_config)
            hf_model.to(device)

            opt = torch.optim.AdamW([p for p in hf_model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.01)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5 * len(train_loader))

            for ep in range(5):
                hf_model.train()
                for imgs, labs in train_loader:
                    imgs, labs = imgs.to(device), labs.to(device)
                    opt.zero_grad()
                    loss = F.cross_entropy(hf_model(imgs).logits, labs)
                    loss.backward()
                    opt.step()
                    sched.step()

            hf_model.eval()
            c, t = 0, 0
            with torch.no_grad():
                for imgs, labs in val_loader:
                    imgs, labs = imgs.to(device), labs.to(device)
                    _, pred = hf_model(imgs).logits.max(1)
                    t += labs.size(0); c += pred.eq(labs).sum().item()
            acc = 100.*c/t
            n_trainable = sum(p.numel() for p in hf_model.parameters() if p.requires_grad)
            results.append({'label': name, 'n_params': n_trainable, 'val_acc': acc,
                           'mean_ms': 0, 'std_ms': 0, 'p50_ms': 0, 'p95_ms': 0})
            print(f"  {name}: acc={acc:.2f}%, params={n_trainable:,}")
            del hf_model; torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            results.append({'label': name, 'n_params': 0, 'val_acc': 0,
                           'mean_ms': 0, 'std_ms': 0, 'p50_ms': 0, 'p95_ms': 0})

    with open('/results/adalora_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    vol.commit()
    print("\n  ADALORA SAVED!")
    return results


# ============================================================
# Orchestrator: run all 5 in parallel
# ============================================================
@app.local_entrypoint()
def main():
    # Spawn all 5 experiments in parallel
    handles = []
    handles.append(exp1_cifar100.spawn())
    handles.append(exp2_flowers102.spawn())
    handles.append(exp3_data_regime.spawn())
    handles.append(exp4_vit_small.spawn())
    handles.append(exp5_adalora.spawn())

    print(f"Spawned {len(handles)} experiments in parallel!")
    for i, h in enumerate(handles):
        print(f"  Exp {i+1}: {h.object_id}")

    print("\nAll experiments running on separate A10G GPUs.")
    print("Results save incrementally to Volume 'adapter-nas-results-v2'.")
    print("Monitor at: https://modal.com/apps")
