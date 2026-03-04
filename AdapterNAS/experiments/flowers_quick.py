"""Quick Flowers-102 experiment with key configs only."""
import os, sys, json, time, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision, torchvision.transforms as T
import timm

class LoRALinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.rank = rank
        for p in self.base.parameters(): p.requires_grad = False
        self.A = nn.Parameter(torch.randn(base.in_features, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, base.out_features))
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

def apply_lora(model, cfg):
    for p in model.parameters(): p.requires_grad = False
    for l in range(12):
        b = model.blocks[l]
        r = cfg.get((l,'qkv'),0)
        if r > 0: b.attn.qkv = LoRALinear(b.attn.qkv, r)
        r = cfg.get((l,'mlp_fc1'),0)
        if r > 0: b.mlp.fc1 = LoRALinear(b.mlp.fc1, r)
        r = cfg.get((l,'mlp_fc2'),0)
        if r > 0: b.mlp.fc2 = LoRALinear(b.mlp.fc2, r)
    for p in model.head.parameters(): p.requires_grad = True
    return model

def get_loaders():
    tr = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(), T.ToTensor(),
                     T.Normalize([.485,.456,.406],[.229,.224,.225])])
    te = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize([.485,.456,.406],[.229,.224,.225])])
    train = torchvision.datasets.Flowers102('./data', split='train', download=True, transform=tr)
    test = torchvision.datasets.Flowers102('./data', split='test', download=True, transform=te)
    cal = DataLoader(Subset(train, range(64)), batch_size=64, num_workers=0)
    trl = DataLoader(train, batch_size=32, shuffle=True, num_workers=0)
    tsl = DataLoader(Subset(test, range(1000)), batch_size=32, num_workers=0)
    return cal, trl, tsl

def compute_proxies(model, cal, device):
    model.to(device); model.train(); model.zero_grad()
    imgs, labs = next(iter(cal))
    imgs, labs = imgs.to(device), labs.to(device)
    out = model(imgs)
    F.cross_entropy(out, labs).backward()
    gn, sn, fi = 0, 0, 0
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            gn += p.grad.norm(2).item()
            sn += (p.grad * p.data).abs().sum().item()
            fi += (p.grad ** 2).sum().item()
    model.eval()
    with torch.no_grad():
        o2 = model(imgs)
        pr = F.softmax(o2, dim=-1)
        ent = -(pr * (pr+1e-10).log()).sum(-1).mean().item()
    model.zero_grad()
    return {'gradnorm': gn, 'snip': sn, 'fisher': fi, 'neg_entropy': -ent}

def finetune(model, trl, tsl, device, epochs=5):
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params: return evaluate(model, tsl, device)
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    for ep in range(epochs):
        model.train()
        for imgs, labs in trl:
            imgs, labs = imgs.to(device), labs.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(imgs), labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        acc = evaluate(model, tsl, device)
        print(f'    Ep{ep+1}: val={acc:.1f}%')
    return evaluate(model, tsl, device)

def evaluate(model, loader, device):
    model.to(device); model.eval()
    c, t = 0, 0
    with torch.no_grad():
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            _, p = model(imgs).max(1)
            t += labs.size(0); c += p.eq(labs).sum().item()
    return 100.*c/t

if __name__ == '__main__':
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    cal, trl, tsl = get_loaders()
    
    configs = {
        'uniform_r4':  {(l,m): 4 for l in range(12) for m in MODULE_TYPES},
        'uniform_r8':  {(l,m): 8 for l in range(12) for m in MODULE_TYPES},
        'uniform_r16': {(l,m): 16 for l in range(12) for m in MODULE_TYPES},
        'uniform_r32': {(l,m): 32 for l in range(12) for m in MODULE_TYPES},
        'attn_only_r8': {(l,m): (8 if m=='qkv' else 0) for l in range(12) for m in MODULE_TYPES},
        'attn_only_r16': {(l,m): (16 if m=='qkv' else 0) for l in range(12) for m in MODULE_TYPES},
        'mlp_only_r8': {(l,m): (8 if m!='qkv' else 0) for l in range(12) for m in MODULE_TYPES},
        'last4_r16': {(l,m): (16 if l>=8 else 0) for l in range(12) for m in MODULE_TYPES},
        'first4_r16': {(l,m): (16 if l<4 else 0) for l in range(12) for m in MODULE_TYPES},
        'increasing': {(l,m): [4,4,4,8,8,8,16,16,16,32,32,32][l] for l in range(12) for m in MODULE_TYPES},
        'decreasing': {(l,m): [32,32,32,16,16,16,8,8,8,4,4,4][l] for l in range(12) for m in MODULE_TYPES},
        'high_attn_low_mlp': {},
        'low_attn_high_mlp': {},
    }
    for l in range(12):
        configs['high_attn_low_mlp'][(l,'qkv')] = 16
        configs['high_attn_low_mlp'][(l,'mlp_fc1')] = 4
        configs['high_attn_low_mlp'][(l,'mlp_fc2')] = 4
        configs['low_attn_high_mlp'][(l,'qkv')] = 4
        configs['low_attn_high_mlp'][(l,'mlp_fc1')] = 16
        configs['low_attn_high_mlp'][(l,'mlp_fc2')] = 16
    
    results = []
    for name, cfg in configs.items():
        np_params = count_params(cfg)
        print(f'\n{name} (params={np_params:,})')
        
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=102)
        model = apply_lora(model, cfg)
        
        proxies = compute_proxies(model, cal, device)
        acc = finetune(model, trl, tsl, device, epochs=5)
        lat = 0
        model.to(device); model.eval()
        x = torch.randn(1,3,224,224).to(device)
        with torch.no_grad():
            for _ in range(3): model(x)
            t0 = time.time()
            for _ in range(5): model(x)
            lat = (time.time()-t0)/5*1000
        
        results.append({
            'label': name, 'n_params': np_params, 'val_acc': acc, 'latency_ms': lat,
            **proxies
        })
        print(f'  -> {acc:.1f}% lat={lat:.0f}ms')
        del model
    
    # Linear probe
    print('\nlinear_probe')
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=102)
    for p in model.parameters(): p.requires_grad = False
    for p in model.head.parameters(): p.requires_grad = True
    acc = finetune(model, trl, tsl, device, epochs=5)
    results.append({'label':'linear_probe','n_params':76902,'val_acc':acc,'latency_ms':3,
                    'gradnorm':0,'snip':0,'fisher':0,'neg_entropy':0})
    print(f'  -> {acc:.1f}%')
    
    os.makedirs('../results', exist_ok=True)
    with open('../results/flowers102_finetune.json','w') as f:
        json.dump(results, f, indent=2)
    
    results.sort(key=lambda x: x['val_acc'], reverse=True)
    print(f"\n{'Label':<25} {'Params':>10} {'ValAcc':>8}")
    for r in results:
        print(f"{r['label']:<25} {r['n_params']:>10,} {r['val_acc']:>7.1f}%")
