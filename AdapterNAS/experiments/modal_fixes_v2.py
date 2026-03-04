"""
Fix experiments V2: Flowers-102 (proper training) and Data Regime 10% fraction.
Split into two separate functions for parallel execution.
"""

import modal
import json

app = modal.App("adapter-nas-fixes-v2")
vol = modal.Volume.from_name("adapter-nas-results-v2", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch", "torchvision", "timm",
        "numpy", "scipy",
    )
)

SHARED_CODE = """
import os, time, random, copy, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
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

def count_params(cfg, h=768, m=3072):
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
        for _ in range(n_warmup): model(x)
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
        print(f"      Epoch {ep+1}/{epochs}: acc={acc:.2f}% (best={best_acc:.2f}%)")
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
"""


# ============================================================
# Flowers-102 with proper train+val+test splits
# Focus: only fine-tune 12 representative configs (not all 25)
# ============================================================
@app.function(gpu="A10G", image=image, volumes={"/results": vol}, timeout=7200, memory=32768)
def fix_flowers102():
    exec(SHARED_CODE, globals())
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 70)
    print("  Flowers-102 with full train+val data")
    print("=" * 70)

    # Flowers-102: train=1020, val=1020, test=6149
    # Combine train+val for training (2040 images), use test for eval
    tr = T.Compose([T.Resize(256), T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.2),
                    T.ToTensor(),
                    T.Normalize([.485,.456,.406],[.229,.224,.225])])
    te = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize([.485,.456,.406],[.229,.224,.225])])

    train1 = torchvision.datasets.Flowers102('/tmp/data', split='train', download=True, transform=tr)
    train2 = torchvision.datasets.Flowers102('/tmp/data', split='val', download=True, transform=tr)
    testset = torchvision.datasets.Flowers102('/tmp/data', split='test', download=True, transform=te)
    trainset = ConcatDataset([train1, train2])

    print(f"  Train size: {len(trainset)} (train+val combined)")
    print(f"  Test size: {len(testset)}")

    # Verify labels range
    sample_labels = set()
    for i in range(min(50, len(train1))):
        _, lab = train1[i]
        sample_labels.add(lab)
    print(f"  Sample label range: {min(sample_labels)}-{max(sample_labels)} (expect 0-101)")
    nc = 102

    # Build loaders
    cal_loader = DataLoader(Subset(trainset, range(min(128, len(trainset)))),
                            batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    train_loader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # Score ALL structured configs with proxies
    struct_configs, struct_labels = sample_structured_configs()
    proxy_results = []

    print("\n  Scoring structured configs...")
    for i, (cfg, lab) in enumerate(zip(struct_configs, struct_labels)):
        np_p = count_params(cfg)
        if np_p == 0:
            proxy_results.append({'label': lab, 'n_params': 0, 'gradnorm': 0, 'snip': 0, 'fisher': 0, 'neg_entropy': 0})
            continue
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=nc)
        model = apply_lora(model, cfg)
        proxies = compute_proxies(model, cal_loader, device)
        proxy_results.append({'label': lab, 'n_params': np_p, **proxies})
        del model; torch.cuda.empty_cache()
        if (i+1) % 5 == 0:
            print(f"    [{i+1}/{len(struct_configs)}] scored")

    # Add random configs proxy scores
    random.seed(42)
    rand_configs, rand_labels = [], []
    for j in range(20):
        cfg = {}
        for l in range(12):
            for m in MODULE_TYPES:
                cfg[(l, m)] = random.choice([0, 4, 8, 16, 32])
        rand_configs.append(cfg)
        rand_labels.append(f'rand_{j}')

    print(f"  Scoring {len(rand_configs)} random configs...")
    for i, (cfg, lab) in enumerate(zip(rand_configs, rand_labels)):
        np_p = count_params(cfg)
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

    # Select configs to finetune: top-5 by proxy + bottom-5 + 5 diverse structured
    valid_sorted = sorted(valid, key=lambda x: x['combined'], reverse=True)
    ft_labels_set = set()
    # Top-5 by combined proxy
    for r in valid_sorted[:5]:
        ft_labels_set.add(r['label'])
    # Bottom-5 by combined proxy
    for r in valid_sorted[-5:]:
        ft_labels_set.add(r['label'])
    # Key structured configs
    for lab in ['uniform_r8', 'uniform_r16', 'uniform_r32',
                'attn_only_r8', 'mlp_only_r8', 'high_attn_low_mlp',
                'increasing', 'decreasing']:
        ft_labels_set.add(lab)
    ft_labels_set.discard('no_adapter')

    # Build config map
    all_configs = dict(zip(struct_labels, struct_configs))
    all_configs.update(dict(zip(rand_labels, rand_configs)))

    print(f"\n  Fine-tuning {len(ft_labels_set)} configs (15 epochs each)...")
    ft_results = []
    for i, lab in enumerate(sorted(ft_labels_set)):
        cfg = all_configs[lab]
        np_p = count_params(cfg)
        print(f"  [{i+1}/{len(ft_labels_set)}] {lab} (params={np_p:,})")
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=nc)
        model = apply_lora(model, cfg)
        # Use lower lr and more epochs for this small dataset
        acc = finetune(model, train_loader, val_loader, device, epochs=15, lr=5e-4)
        lat = benchmark_latency(model, device)
        proxy_info = next((r for r in proxy_results if r['label'] == lab), {})
        ft_results.append({
            'label': lab, 'n_params': np_p, 'val_acc': acc,
            'combined_proxy': proxy_info.get('combined', 0),
            'gradnorm': proxy_info.get('gradnorm', 0), 'snip': proxy_info.get('snip', 0),
            'fisher': proxy_info.get('fisher', 0), 'neg_entropy': proxy_info.get('neg_entropy', 0),
            **lat,
        })
        print(f"    -> Acc={acc:.2f}%")
        del model; torch.cuda.empty_cache()

        # Save incrementally every 5 configs
        if (i+1) % 5 == 0:
            _save_flowers(proxy_results, ft_results, valid, vol)

    # Linear probe
    print("  Training linear probe...")
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=nc)
    for p in model.parameters(): p.requires_grad = False
    for p in model.head.parameters(): p.requires_grad = True
    lp_acc = finetune(model, train_loader, val_loader, device, epochs=15, lr=5e-4)
    lp_lat = benchmark_latency(model, device)
    ft_results.append({'label': 'linear_probe', 'n_params': sum(p.numel() for p in model.head.parameters()),
                      'val_acc': lp_acc, 'combined_proxy': 0, 'gradnorm': 0, 'snip': 0, 'fisher': 0, 'neg_entropy': 0, **lp_lat})
    print(f"  Linear probe: {lp_acc:.2f}%")
    del model; torch.cuda.empty_cache()

    result = _save_flowers(proxy_results, ft_results, valid, vol)

    ft_results.sort(key=lambda x: x['val_acc'], reverse=True)
    print(f"\n  Top results:")
    for r in ft_results[:10]:
        print(f"    {r['label']:<25} {r['val_acc']:.2f}%")
    print(f"\n  FLOWERS-102 DONE!")
    return result


def _save_flowers(proxy_results, ft_results, valid, vol):
    """Helper to compute selection quality and save."""
    from scipy import stats
    acc_map = {r['label']: r['val_acc'] for r in ft_results if r['val_acc'] > 0}
    proxy_map = {r['label']: r.get('combined', 0) for r in valid}
    common_labels = [l for l in acc_map if l in proxy_map and l != 'linear_probe']
    sq = {}
    if len(common_labels) >= 5:
        by_proxy = sorted(common_labels, key=lambda l: proxy_map[l], reverse=True)
        by_oracle = sorted(common_labels, key=lambda l: acc_map[l], reverse=True)
        oracle_top1 = acc_map[by_oracle[0]]
        for k in [1, 3, 5, 10]:
            if k > len(by_proxy): continue
            proxy_topk = by_proxy[:k]
            oracle_topk = set(by_oracle[:k])
            best_in_topk = max(acc_map[l] for l in proxy_topk)
            hits = sum(1 for l in proxy_topk if l in oracle_topk)
            sq[f'top{k}_best_acc'] = best_in_topk
            sq[f'top{k}_hit_rate'] = hits / k
            sq[f'top{k}_regret_vs_oracle'] = oracle_top1 - best_in_topk
        rho, p = stats.spearmanr([proxy_map[l] for l in common_labels], [acc_map[l] for l in common_labels])
        sq['spearman_rho'] = float(rho)
        sq['spearman_p'] = float(p)
        for metric in ['gradnorm', 'snip', 'fisher', 'neg_entropy']:
            m_map = {r['label']: r.get(metric, 0) for r in valid}
            m_vals = [m_map.get(l, 0) for l in common_labels]
            r_val, _ = stats.spearmanr(m_vals, [acc_map[l] for l in common_labels])
            sq[f'{metric}_spearman'] = float(r_val)

    result = {'proxy_scores': proxy_results, 'finetune_results': ft_results, 'selection_quality': sq}
    with open('/results/flowers102_results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    vol.commit()
    print("  [Saved flowers102_results.json]")
    return result


# ============================================================
# Data Regime: ONLY the 10% fraction (0.01, 0.02, 0.05 already done)
# ============================================================
@app.function(gpu="A10G", image=image, volumes={"/results": vol}, timeout=3600, memory=32768)
def fix_data_regime_10pct():
    exec(SHARED_CODE, globals())
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 70)
    print("  Data Regime: 10% fraction only")
    print("=" * 70)

    tr = T.Compose([T.Resize(224), T.RandomCrop(224, padding=4),
                    T.RandomHorizontalFlip(), T.ToTensor(),
                    T.Normalize([.485,.456,.406],[.229,.224,.225])])
    te = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize([.485,.456,.406],[.229,.224,.225])])

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

    trainset_full = torchvision.datasets.CIFAR100('/tmp/data', train=True, download=True, transform=tr)
    testset = torchvision.datasets.CIFAR100('/tmp/data', train=False, download=True, transform=te)
    nc = 100
    val_loader = DataLoader(Subset(testset, range(2000)), batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    frac = 0.10
    n_train = int(frac * len(trainset_full))
    print(f"\n  Data fraction: 10% ({n_train} samples)")
    train_loader = DataLoader(Subset(trainset_full, range(n_train)),
                              batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    new_results = []
    for name, cfg in configs_to_test.items():
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=nc)
        model = apply_lora(model, cfg)
        acc = finetune(model, train_loader, val_loader, device, epochs=5, lr=1e-3)
        new_results.append({'config': name, 'data_frac': frac, 'val_acc': acc,
                           'n_params': count_params(cfg), 'n_train_samples': n_train})
        print(f"    {name}: {acc:.2f}%")
        del model; torch.cuda.empty_cache()

    # Linear probe
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=nc)
    for p in model.parameters(): p.requires_grad = False
    for p in model.head.parameters(): p.requires_grad = True
    acc = finetune(model, train_loader, val_loader, device, epochs=5, lr=1e-3)
    new_results.append({'config': 'linear_probe', 'data_frac': frac, 'val_acc': acc,
                       'n_params': 76900, 'n_train_samples': n_train})
    print(f"    linear_probe: {acc:.2f}%")
    del model; torch.cuda.empty_cache()

    # Load existing results and append
    try:
        existing_data = b''
        for chunk in vol.read_file('data_regime_results.json'):
            existing_data += chunk
        existing = json.loads(existing_data)
        # Remove any existing 0.10 entries
        existing = [r for r in existing if r['data_frac'] != 0.10]
        print(f"  Loaded {len(existing)} existing entries (removed old 10%)")
    except:
        existing = []
        print("  No existing data_regime_results.json found")

    combined = existing + new_results

    with open('/results/data_regime_results.json', 'w') as f:
        json.dump(combined, f, indent=2)
    vol.commit()

    print(f"\n  Saved {len(combined)} total entries ({len(new_results)} new for 10%)")
    print("  DATA REGIME 10% DONE!")
    return new_results


# ============================================================
# Quick Flowers-102 linear probe only
# ============================================================
@app.function(gpu="A10G", image=image, volumes={"/results": vol}, timeout=1800, memory=32768)
def fix_flowers_linear_probe():
    exec(SHARED_CODE, globals())
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("Flowers-102 linear probe only")

    tr = T.Compose([T.Resize(256), T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.2),
                    T.ToTensor(),
                    T.Normalize([.485,.456,.406],[.229,.224,.225])])
    te = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize([.485,.456,.406],[.229,.224,.225])])

    train1 = torchvision.datasets.Flowers102('/tmp/data', split='train', download=True, transform=tr)
    train2 = torchvision.datasets.Flowers102('/tmp/data', split='val', download=True, transform=tr)
    testset = torchvision.datasets.Flowers102('/tmp/data', split='test', download=True, transform=te)
    trainset = ConcatDataset([train1, train2])

    train_loader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    nc = 102

    print("  Training linear probe...")
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=nc)
    for p in model.parameters(): p.requires_grad = False
    for p in model.head.parameters(): p.requires_grad = True
    lp_acc = finetune(model, train_loader, val_loader, device, epochs=15, lr=5e-4)
    lp_lat = benchmark_latency(model, device)
    print(f"  Linear probe: {lp_acc:.2f}%")

    # Load existing flowers results and add LP
    import json as json2
    existing_data = b''
    for chunk in vol.read_file('flowers102_results.json'):
        existing_data += chunk
    flowers = json2.loads(existing_data)

    # Remove any old LP entries
    flowers['finetune_results'] = [r for r in flowers['finetune_results'] if r.get('label') != 'linear_probe']
    flowers['finetune_results'].append({
        'label': 'linear_probe',
        'n_params': sum(p.numel() for p in model.head.parameters()),
        'val_acc': lp_acc, 'combined_proxy': 0, 'gradnorm': 0, 'snip': 0,
        'fisher': 0, 'neg_entropy': 0, **lp_lat
    })

    with open('/results/flowers102_results.json', 'w') as f:
        json2.dump(flowers, f, indent=2, default=str)
    vol.commit()
    print(f"  Saved with LP. Total configs: {len(flowers['finetune_results'])}")
    return {'linear_probe_acc': lp_acc}
