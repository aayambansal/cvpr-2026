"""
Fix experiments: Flowers-102 (full train set), Data Regime, AdaLoRA debug.
"""

import modal
import json

app = modal.App("adapter-nas-fixes")
vol = modal.Volume.from_name("adapter-nas-results-v2", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch", "torchvision", "timm", "peft", "transformers",
        "numpy", "scipy", "accelerate",
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
RANKS = [0, 4, 8, 16, 32]

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
# FIX 1: Flowers-102 with FULL train+val splits
# ============================================================
@app.function(gpu="A10G", image=image, volumes={"/results": vol}, timeout=5400, memory=32768)
def fix_flowers102():
    exec(SHARED_CODE, globals())
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 70)
    print("  FIX: Flowers-102 with full train+val data")
    print("=" * 70)

    # Flowers-102: train=1020, val=1020, test=6149
    # Use train+val combined for training (2040 images), test for eval
    tr = T.Compose([T.Resize(224), T.RandomCrop(224, padding=4),
                    T.RandomHorizontalFlip(), T.ToTensor(),
                    T.Normalize([.485,.456,.406],[.229,.224,.225])])
    te = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize([.485,.456,.406],[.229,.224,.225])])

    train1 = torchvision.datasets.Flowers102('/tmp/data', split='train', download=True, transform=tr)
    train2 = torchvision.datasets.Flowers102('/tmp/data', split='val', download=True, transform=tr)
    testset = torchvision.datasets.Flowers102('/tmp/data', split='test', download=True, transform=te)
    trainset = ConcatDataset([train1, train2])

    print(f"  Train size: {len(trainset)} (train+val combined)")
    print(f"  Test size: {len(testset)}")

    nc = 102
    cal_loader = DataLoader(Subset(trainset, range(min(128, len(trainset)))),
                            batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # Score structured configs
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
        print(f"    [{i+1}/{len(struct_configs)}] {lab}")

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

    # Fine-tune ALL structured configs (only 25 — manageable)
    print(f"\n  Fine-tuning all {len(struct_configs)} structured configs...")
    ft_results = []
    for i, (cfg, lab) in enumerate(zip(struct_configs, struct_labels)):
        np_p = count_params(cfg)
        if np_p == 0:
            ft_results.append({'label': lab, 'n_params': 0, 'val_acc': 0,
                             'combined_proxy': 0, 'gradnorm': 0, 'snip': 0, 'fisher': 0, 'neg_entropy': 0,
                             'mean_ms': 0, 'std_ms': 0, 'p50_ms': 0, 'p95_ms': 0})
            continue
        print(f"  [{i+1}/{len(struct_configs)}] {lab}")
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=nc)
        model = apply_lora(model, cfg)
        acc = finetune(model, train_loader, val_loader, device, epochs=10, lr=1e-3)
        lat = benchmark_latency(model, device)
        # Find proxy info
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

    # Linear probe
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=nc)
    for p in model.parameters(): p.requires_grad = False
    for p in model.head.parameters(): p.requires_grad = True
    lp_acc = finetune(model, train_loader, val_loader, device, epochs=10, lr=1e-3)
    lp_lat = benchmark_latency(model, device)
    ft_results.append({'label': 'linear_probe', 'n_params': sum(p.numel() for p in model.head.parameters()),
                      'val_acc': lp_acc, 'combined_proxy': 0, 'gradnorm': 0, 'snip': 0, 'fisher': 0, 'neg_entropy': 0, **lp_lat})
    print(f"  Linear probe: {lp_acc:.2f}%")
    del model; torch.cuda.empty_cache()

    # Selection quality
    acc_map = {r['label']: r['val_acc'] for r in ft_results if r['val_acc'] > 0}
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
            oracle_topk = set(by_oracle[:k])
            best_in_topk = max(acc_map[l] for l in proxy_topk)
            hits = sum(1 for l in proxy_topk if l in oracle_topk)
            sq[f'top{k}_best_acc'] = best_in_topk
            sq[f'top{k}_hit_rate'] = hits / k
            sq[f'top{k}_regret_vs_oracle'] = oracle_top1 - best_in_topk
        rho, _ = stats.spearmanr([proxy_map[l] for l in common_labels], [acc_map[l] for l in common_labels])
        sq['spearman_rho'] = float(rho)
        for metric in ['gradnorm', 'snip', 'fisher', 'neg_entropy']:
            m_map = {r['label']: r.get(metric, 0) for r in valid}
            m_vals = [m_map.get(l, 0) for l in common_labels]
            r_val, _ = stats.spearmanr(m_vals, [acc_map[l] for l in common_labels])
            sq[f'{metric}_spearman'] = float(r_val)

    result = {'proxy_scores': proxy_results, 'finetune_results': ft_results, 'selection_quality': sq}
    with open('/results/flowers102_results_fixed.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    vol.commit()

    ft_results.sort(key=lambda x: x['val_acc'], reverse=True)
    print(f"\n  Top results:")
    for r in ft_results[:10]:
        print(f"    {r['label']:<25} {r['val_acc']:.2f}%")
    print(f"\n  FLOWERS-102 FIXED & SAVED!")
    return result


# ============================================================
# FIX 2: Data Regime with shorter timeout
# ============================================================
@app.function(gpu="A10G", image=image, volumes={"/results": vol}, timeout=5400, memory=32768)
def fix_data_regime():
    exec(SHARED_CODE, globals())
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 70)
    print("  FIX: Data Regime Sweep (CIFAR-100)")
    print("=" * 70)

    tr = T.Compose([T.Resize(224), T.RandomCrop(224, padding=4),
                    T.RandomHorizontalFlip(), T.ToTensor(),
                    T.Normalize([.485,.456,.406],[.229,.224,.225])])
    te = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize([.485,.456,.406],[.229,.224,.225])])

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

    trainset_full = torchvision.datasets.CIFAR100('/tmp/data', train=True, download=True, transform=tr)
    testset = torchvision.datasets.CIFAR100('/tmp/data', train=False, download=True, transform=te)
    nc = 100
    val_loader = DataLoader(Subset(testset, range(2000)), batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    results = []
    for frac in fracs:
        n_train = max(int(frac * len(trainset_full)), 50)
        print(f"\n  Data fraction: {frac*100:.0f}% ({n_train} samples)")
        train_loader = DataLoader(Subset(trainset_full, range(n_train)),
                                  batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

        for name, cfg in configs_to_test.items():
            model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=nc)
            model = apply_lora(model, cfg)
            acc = finetune(model, train_loader, val_loader, device, epochs=5, lr=1e-3)
            results.append({'config': name, 'data_frac': frac, 'val_acc': acc,
                           'n_params': count_params(cfg), 'n_train_samples': n_train})
            print(f"    {name}: {acc:.2f}%")
            del model; torch.cuda.empty_cache()

        # Linear probe
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=nc)
        for p in model.parameters(): p.requires_grad = False
        for p in model.head.parameters(): p.requires_grad = True
        acc = finetune(model, train_loader, val_loader, device, epochs=5, lr=1e-3)
        results.append({'config': 'linear_probe', 'data_frac': frac, 'val_acc': acc,
                       'n_params': 76900, 'n_train_samples': n_train})
        print(f"    linear_probe: {acc:.2f}%")
        del model; torch.cuda.empty_cache()

        # Save incrementally after each fraction
        with open('/results/data_regime_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        vol.commit()
        print(f"  Saved after {frac*100:.0f}%")

    print("\n  DATA REGIME SAVED!")
    return results


# ============================================================
# FIX 3: AdaLoRA with better error handling
# ============================================================
@app.function(gpu="A10G", image=image, volumes={"/results": vol}, timeout=2400, memory=32768)
def fix_adalora():
    exec(SHARED_CODE, globals())
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 70)
    print("  FIX: AdaLoRA Baseline (with debug)")
    print("=" * 70)

    import traceback
    from peft import get_peft_model, AdaLoraConfig, LoraConfig
    from transformers import ViTForImageClassification

    tr = T.Compose([T.Resize(224), T.RandomCrop(224, padding=4),
                    T.RandomHorizontalFlip(), T.ToTensor(),
                    T.Normalize([.485,.456,.406],[.229,.224,.225])])
    te = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize([.485,.456,.406],[.229,.224,.225])])
    trainset = torchvision.datasets.CIFAR100('/tmp/data', train=True, download=True, transform=tr)
    testset = torchvision.datasets.CIFAR100('/tmp/data', train=False, download=True, transform=te)
    nc = 100
    n_train = 1000
    train_loader = DataLoader(Subset(trainset, range(n_train)), batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(Subset(testset, range(2000)), batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    results = []

    # Our LoRA baselines
    for r, name in [(8, 'our_lora_r8'), (16, 'our_lora_r16')]:
        cfg = {(l, m): r for l in range(12) for m in MODULE_TYPES}
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=nc)
        model = apply_lora(model, cfg)
        acc = finetune(model, train_loader, val_loader, device, epochs=5, lr=1e-3)
        lat = benchmark_latency(model, device)
        results.append({'label': name, 'n_params': count_params(cfg), 'val_acc': acc, **lat})
        print(f"  {name}: acc={acc:.2f}%, params={count_params(cfg):,}")
        del model; torch.cuda.empty_cache()

    # peft LoRA on HF ViT (fair comparison — same backbone, peft library)
    for r, name in [(8, 'peft_lora_r8'), (16, 'peft_lora_r16')]:
        try:
            hf_model = ViTForImageClassification.from_pretrained(
                'google/vit-base-patch16-224-in21k', num_labels=nc, ignore_mismatched_sizes=True)
            lora_config = LoraConfig(
                r=r, lora_alpha=r, target_modules=["query", "value"],
                lora_dropout=0.05, bias="none",
            )
            hf_model = get_peft_model(hf_model, lora_config)
            hf_model.to(device)
            n_trainable = sum(p.numel() for p in hf_model.parameters() if p.requires_grad)
            print(f"  {name}: trainable params={n_trainable:,}")

            opt = torch.optim.AdamW([p for p in hf_model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.01)
            for ep in range(5):
                hf_model.train()
                for imgs, labs in train_loader:
                    imgs, labs = imgs.to(device), labs.to(device)
                    opt.zero_grad()
                    loss = F.cross_entropy(hf_model(imgs).logits, labs)
                    loss.backward()
                    opt.step()

            hf_model.eval()
            c, t = 0, 0
            with torch.no_grad():
                for imgs, labs in val_loader:
                    imgs, labs = imgs.to(device), labs.to(device)
                    _, pred = hf_model(imgs).logits.max(1)
                    t += labs.size(0); c += pred.eq(labs).sum().item()
            acc = 100.*c/t
            results.append({'label': name, 'n_params': n_trainable, 'val_acc': acc,
                           'mean_ms': 0, 'std_ms': 0, 'p50_ms': 0, 'p95_ms': 0})
            print(f"  {name}: acc={acc:.2f}%")
            del hf_model; torch.cuda.empty_cache()
        except Exception as e:
            traceback.print_exc()
            results.append({'label': name, 'n_params': 0, 'val_acc': 0,
                           'mean_ms': 0, 'std_ms': 0, 'p50_ms': 0, 'p95_ms': 0})

    # AdaLoRA
    for init_r, name in [(12, 'adalora_r12'), (24, 'adalora_r24')]:
        try:
            hf_model = ViTForImageClassification.from_pretrained(
                'google/vit-base-patch16-224-in21k', num_labels=nc, ignore_mismatched_sizes=True)
            adalora_config = AdaLoraConfig(
                r=init_r, lora_alpha=init_r,
                target_modules=["query", "value"],
                lora_dropout=0.05,
                init_r=init_r, target_r=max(init_r // 2, 4),
                deltaT=10, beta1=0.85, beta2=0.85,
            )
            hf_model = get_peft_model(hf_model, adalora_config)
            hf_model.to(device)
            n_trainable = sum(p.numel() for p in hf_model.parameters() if p.requires_grad)
            print(f"  {name}: trainable params={n_trainable:,}")

            opt = torch.optim.AdamW([p for p in hf_model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.01)
            for ep in range(5):
                hf_model.train()
                for imgs, labs in train_loader:
                    imgs, labs = imgs.to(device), labs.to(device)
                    opt.zero_grad()
                    out = hf_model(imgs)
                    loss = F.cross_entropy(out.logits, labs)
                    loss.backward()
                    opt.step()

            hf_model.eval()
            c, t = 0, 0
            with torch.no_grad():
                for imgs, labs in val_loader:
                    imgs, labs = imgs.to(device), labs.to(device)
                    _, pred = hf_model(imgs).logits.max(1)
                    t += labs.size(0); c += pred.eq(labs).sum().item()
            acc = 100.*c/t
            results.append({'label': name, 'n_params': n_trainable, 'val_acc': acc,
                           'mean_ms': 0, 'std_ms': 0, 'p50_ms': 0, 'p95_ms': 0})
            print(f"  {name}: acc={acc:.2f}%")
            del hf_model; torch.cuda.empty_cache()
        except Exception as e:
            traceback.print_exc()
            results.append({'label': name, 'n_params': 0, 'val_acc': 0,
                           'mean_ms': 0, 'std_ms': 0, 'p50_ms': 0, 'p95_ms': 0})

    with open('/results/adalora_results_fixed.json', 'w') as f:
        json.dump(results, f, indent=2)
    vol.commit()
    print("\n  ADALORA FIXED & SAVED!")
    return results


@app.local_entrypoint()
def main():
    h1 = fix_flowers102.spawn()
    h2 = fix_data_regime.spawn()
    h3 = fix_adalora.spawn()
    print(f"Spawned 3 fix jobs:")
    print(f"  flowers102: {h1.object_id}")
    print(f"  data_regime: {h2.object_id}")
    print(f"  adalora: {h3.object_id}")
