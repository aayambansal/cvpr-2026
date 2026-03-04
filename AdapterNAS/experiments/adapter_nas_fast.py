"""
AdapterNAS: Training-Free NAS for Foundation Model Adapter Topologies (Fast Version)

Optimized for speed: fewer configs, smaller calibration, efficient proxy computation.
"""

import os
import sys
import json
import time
import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import timm

# ---- LoRA Module ----

class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float = None):
        super().__init__()
        self.base_linear = base_linear
        self.rank = rank
        self.alpha = alpha if alpha else float(rank)
        in_f = base_linear.in_features
        out_f = base_linear.out_features
        for p in self.base_linear.parameters():
            p.requires_grad = False
        self.lora_A = nn.Parameter(torch.randn(in_f, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_f))
        self.scaling = self.alpha / self.rank

    def forward(self, x):
        return self.base_linear(x) + (x @ self.lora_A @ self.lora_B) * self.scaling

# ---- Constants ----
RANKS = [0, 4, 8, 16, 32]
NUM_LAYERS = 12
MODULE_TYPES = ['qkv', 'mlp_fc1', 'mlp_fc2']

def count_params(config, hidden=768, mlp=3072):
    total = 0
    for (l, m), r in config.items():
        if r == 0: continue
        if m == 'qkv':
            total += hidden * r + r * (3 * hidden)  # qkv is 768->2304
        elif m == 'mlp_fc1':
            total += hidden * r + r * mlp
        elif m == 'mlp_fc2':
            total += mlp * r + r * hidden
    return total

def sample_structured_configs():
    configs, labels = [], []
    
    # Uniform ranks
    for r in [4, 8, 16, 32]:
        c = {(l, m): r for l in range(12) for m in MODULE_TYPES}
        configs.append(c); labels.append(f'uniform_r{r}')
    
    # Attention-only (qkv)
    for r in [4, 8, 16, 32]:
        c = {(l, m): (r if m == 'qkv' else 0) for l in range(12) for m in MODULE_TYPES}
        configs.append(c); labels.append(f'attn_only_r{r}')
    
    # MLP-only
    for r in [4, 8, 16]:
        c = {(l, m): (r if m != 'qkv' else 0) for l in range(12) for m in MODULE_TYPES}
        configs.append(c); labels.append(f'mlp_only_r{r}')
    
    # Last-4 layers only
    for r in [8, 16]:
        c = {(l, m): (r if l >= 8 else 0) for l in range(12) for m in MODULE_TYPES}
        configs.append(c); labels.append(f'last4_r{r}')
    
    # First-4 layers only
    for r in [8, 16]:
        c = {(l, m): (r if l < 4 else 0) for l in range(12) for m in MODULE_TYPES}
        configs.append(c); labels.append(f'first4_r{r}')
    
    # Every-other layer
    for r in [8, 16]:
        c = {(l, m): (r if l % 2 == 0 else 0) for l in range(12) for m in MODULE_TYPES}
        configs.append(c); labels.append(f'even_r{r}')
    
    # Increasing rank (shallow=low, deep=high)
    c = {}
    for l in range(12):
        r = [4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32][l]
        for m in MODULE_TYPES:
            c[(l, m)] = r
    configs.append(c); labels.append('increasing')
    
    # Decreasing rank
    c = {}
    for l in range(12):
        r = [32, 32, 32, 16, 16, 16, 8, 8, 8, 4, 4, 4][l]
        for m in MODULE_TYPES:
            c[(l, m)] = r
    configs.append(c); labels.append('decreasing')
    
    # Mixed: high attn, low mlp
    c = {}
    for l in range(12):
        c[(l, 'qkv')] = 16
        c[(l, 'mlp_fc1')] = 4
        c[(l, 'mlp_fc2')] = 4
    configs.append(c); labels.append('high_attn_low_mlp')
    
    # Mixed: low attn, high mlp
    c = {}
    for l in range(12):
        c[(l, 'qkv')] = 4
        c[(l, 'mlp_fc1')] = 16
        c[(l, 'mlp_fc2')] = 16
    configs.append(c); labels.append('low_attn_high_mlp')
    
    # No adapter (baseline)
    c = {(l, m): 0 for l in range(12) for m in MODULE_TYPES}
    configs.append(c); labels.append('no_adapter')
    
    return configs, labels

def sample_random_configs(n, seed=42):
    rng = random.Random(seed)
    configs = []
    for _ in range(n):
        c = {}
        for l in range(12):
            for m in MODULE_TYPES:
                c[(l, m)] = rng.choice(RANKS)
        configs.append(c)
    return configs

# ---- Model ----

def apply_lora(model, config):
    """Apply LoRA to ViT-B/16 timm model."""
    for p in model.parameters():
        p.requires_grad = False
    
    lora_mods = {}
    for l in range(12):
        block = model.blocks[l]
        
        qkv_r = config.get((l, 'qkv'), 0)
        if qkv_r > 0:
            lora = LoRALinear(block.attn.qkv, qkv_r)
            block.attn.qkv = lora
            lora_mods[f'b{l}_qkv'] = lora
        
        fc1_r = config.get((l, 'mlp_fc1'), 0)
        if fc1_r > 0:
            lora = LoRALinear(block.mlp.fc1, fc1_r)
            block.mlp.fc1 = lora
            lora_mods[f'b{l}_fc1'] = lora
        
        fc2_r = config.get((l, 'mlp_fc2'), 0)
        if fc2_r > 0:
            lora = LoRALinear(block.mlp.fc2, fc2_r)
            block.mlp.fc2 = lora
            lora_mods[f'b{l}_fc2'] = lora
    
    # Unfreeze head
    for p in model.head.parameters():
        p.requires_grad = True
    
    return model, lora_mods

# ---- Data ----

def get_loaders(dataset_name, batch_size=32, cal_size=64):
    transform_train = transforms.Compose([
        transforms.Resize(224), transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    if dataset_name == 'cifar100':
        trainset = torchvision.datasets.CIFAR100('./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100('./data', train=False, download=True, transform=transform_test)
        num_classes = 100
    else:
        trainset = torchvision.datasets.Flowers102('./data', split='train', download=True, transform=transform_train)
        testset = torchvision.datasets.Flowers102('./data', split='test', download=True, transform=transform_test)
        num_classes = 102
    
    cal_loader = DataLoader(Subset(trainset, range(cal_size)), batch_size=cal_size, shuffle=False, num_workers=0)
    
    # 2% for fine-tuning (small)
    n_train = min(int(0.02 * len(trainset)), 500)
    train_loader = DataLoader(Subset(trainset, range(n_train)), batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Validation: first 1000 of test
    n_val = min(1000, len(testset))
    val_loader = DataLoader(Subset(testset, range(n_val)), batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Full test
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return cal_loader, train_loader, val_loader, test_loader, num_classes

# ---- Zero-Cost Proxies (efficient single-batch) ----

def compute_all_proxies(model, cal_loader, device='cpu'):
    """Compute all proxies in a single pass for efficiency."""
    model.to(device)
    model.train()
    model.zero_grad()
    
    images, labels = next(iter(cal_loader))
    images, labels = images.to(device), labels.to(device)
    
    output = model(images)
    loss = F.cross_entropy(output, labels)
    loss.backward()
    
    gradnorm = 0.0
    snip = 0.0
    fisher = 0.0
    
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            g = p.grad
            gradnorm += g.norm(2).item()
            snip += (g * p.data).abs().sum().item()
            fisher += (g ** 2).sum().item()
    
    # Entropy (label-free)
    model.eval()
    with torch.no_grad():
        output2 = model(images)
        probs = F.softmax(output2, dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean().item()
    
    model.zero_grad()
    
    return {
        'gradnorm': gradnorm,
        'snip': snip,
        'fisher': fisher,
        'neg_entropy': -entropy,  # higher = more confident
    }

def estimate_latency(model, device='cpu', n_runs=5):
    model.to(device)
    model.eval()
    x = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(2): model(x)
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.time()
            model(x)
            times.append(time.time() - t0)
    return np.mean(times) * 1000

# ---- Fine-Tuning ----

def finetune(model, train_loader, val_loader, device='cpu', epochs=5, lr=1e-3):
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return evaluate(model, val_loader, device)
    
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(train_loader))
    
    for ep in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labs in train_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = F.cross_entropy(out, labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            total_loss += loss.item()
            _, pred = out.max(1)
            total += labs.size(0)
            correct += pred.eq(labs).sum().item()
        
        acc = 100. * correct / total
        print(f'    Ep {ep+1}/{epochs}: loss={total_loss/len(train_loader):.4f} trainAcc={acc:.1f}%')
    
    return evaluate(model, val_loader, device)

def evaluate(model, loader, device='cpu'):
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            _, pred = model(imgs).max(1)
            total += labs.size(0)
            correct += pred.eq(labs).sum().item()
    return 100. * correct / total

# ---- Main ----

def run_experiment(dataset_name, device='cpu', results_dir='../results'):
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  AdapterNAS: {dataset_name}")
    print(f"{'='*60}")
    
    cal_loader, train_loader, val_loader, test_loader, nc = get_loaders(dataset_name, cal_size=64)
    
    # Generate configs
    struct_configs, struct_labels = sample_structured_configs()
    rand_configs = sample_random_configs(30, seed=42)
    rand_labels = [f'rand_{i}' for i in range(30)]
    
    all_configs = struct_configs + rand_configs
    all_labels = struct_labels + rand_labels
    print(f"  Total configs: {len(all_configs)} (struct={len(struct_configs)}, rand={len(rand_configs)})")
    
    # Phase 1: Proxy scoring
    print("\n  Phase 1: Zero-cost proxy scoring...")
    proxy_results = []
    
    for i, (cfg, lab) in enumerate(zip(all_configs, all_labels)):
        np_params = count_params(cfg)
        
        if np_params == 0:
            proxy_results.append({
                'label': lab, 'n_params': 0,
                'gradnorm': 0, 'snip': 0, 'fisher': 0, 'neg_entropy': 0,
                'latency_ms': 0, 'combined': 0
            })
            print(f"    [{i+1}/{len(all_configs)}] {lab}: no adapter")
            continue
        
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=nc)
        model, _ = apply_lora(model, cfg)
        
        proxies = compute_all_proxies(model, cal_loader, device)
        lat = estimate_latency(model, device)
        
        r = {'label': lab, 'n_params': np_params, 'latency_ms': lat, **proxies, 'combined': 0}
        proxy_results.append(r)
        
        print(f"    [{i+1}/{len(all_configs)}] {lab}: params={np_params:,} gn={proxies['gradnorm']:.1f} snip={proxies['snip']:.1f} fisher={proxies['fisher']:.2f} lat={lat:.0f}ms")
        del model
    
    # Normalize and combine
    valid = [r for r in proxy_results if r['n_params'] > 0]
    for metric in ['gradnorm', 'snip', 'fisher', 'neg_entropy']:
        vals = [r[metric] for r in valid]
        lo, hi = min(vals), max(vals)
        rng = hi - lo if hi != lo else 1.0
        for r in valid:
            r[f'{metric}_n'] = (r[metric] - lo) / rng
    
    for r in valid:
        r['combined'] = (r['gradnorm_n'] + r['snip_n'] + r['fisher_n'] + r['neg_entropy_n']) / 4.0
    
    valid.sort(key=lambda x: x['combined'], reverse=True)
    
    # Save proxy scores
    with open(os.path.join(results_dir, f'{dataset_name}_proxy.json'), 'w') as f:
        json.dump(proxy_results, f, indent=2)
    
    # Phase 2: Select and fine-tune
    print(f"\n  Phase 2: Fine-tuning top/bottom/baselines...")
    
    # Select configs to fine-tune
    ft_labels = set()
    ft_list = []
    
    # Top-3 by proxy
    for r in valid[:3]:
        ft_list.append(r); ft_labels.add(r['label'])
    # Bottom-3
    for r in valid[-3:]:
        if r['label'] not in ft_labels:
            ft_list.append(r); ft_labels.add(r['label'])
    # Key baselines
    for bl in ['uniform_r4', 'uniform_r8', 'uniform_r16', 'uniform_r32',
               'attn_only_r8', 'attn_only_r16', 'mlp_only_r8',
               'last4_r16', 'first4_r16', 'even_r16',
               'increasing', 'decreasing', 'high_attn_low_mlp', 'low_attn_high_mlp']:
        for r in valid:
            if r['label'] == bl and r['label'] not in ft_labels:
                ft_list.append(r); ft_labels.add(r['label'])
    
    print(f"  Fine-tuning {len(ft_list)} configs...")
    ft_results = []
    
    for fi, r in enumerate(ft_list):
        lab = r['label']
        # Find config
        idx = all_labels.index(lab)
        cfg = all_configs[idx]
        
        print(f"\n  [{fi+1}/{len(ft_list)}] {lab} (params={r['n_params']:,}, proxy={r['combined']:.3f})")
        
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=nc)
        model, _ = apply_lora(model, cfg)
        
        val_acc = finetune(model, train_loader, val_loader, device, epochs=5, lr=1e-3)
        lat = estimate_latency(model, device)
        
        ft_results.append({
            'label': lab, 'n_params': r['n_params'],
            'combined_proxy': r['combined'],
            'gradnorm': r['gradnorm'], 'snip': r['snip'],
            'fisher': r['fisher'], 'neg_entropy': r['neg_entropy'],
            'val_acc': val_acc, 'latency_ms': lat,
        })
        print(f"    -> ValAcc={val_acc:.2f}%, Latency={lat:.0f}ms")
        del model
    
    # Linear probe baseline
    print(f"\n  Fine-tuning linear probe baseline...")
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=nc)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True
    
    lp_acc = finetune(model, train_loader, val_loader, device, epochs=5, lr=1e-3)
    lp_lat = estimate_latency(model, device)
    ft_results.append({
        'label': 'linear_probe', 'n_params': sum(p.numel() for p in model.head.parameters()),
        'combined_proxy': 0, 'gradnorm': 0, 'snip': 0, 'fisher': 0, 'neg_entropy': 0,
        'val_acc': lp_acc, 'latency_ms': lp_lat,
    })
    print(f"    -> Linear probe: {lp_acc:.2f}%")
    del model
    
    # Save
    with open(os.path.join(results_dir, f'{dataset_name}_finetune.json'), 'w') as f:
        json.dump(ft_results, f, indent=2)
    
    # Print summary
    ft_results.sort(key=lambda x: x['val_acc'], reverse=True)
    print(f"\n{'='*70}")
    print(f"  RESULTS: {dataset_name}")
    print(f"{'='*70}")
    print(f"  {'Label':<25} {'Params':>10} {'ValAcc':>8} {'Lat(ms)':>8} {'Proxy':>7}")
    print(f"  {'-'*62}")
    for r in ft_results:
        print(f"  {r['label']:<25} {r['n_params']:>10,} {r['val_acc']:>7.2f}% {r['latency_ms']:>7.0f} {r['combined_proxy']:>6.3f}")
    
    return proxy_results, ft_results

if __name__ == '__main__':
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    p1, f1 = run_experiment('cifar100', device, '../results')
    p2, f2 = run_experiment('flowers102', device, '../results')
    
    print("\n\nDone!")
