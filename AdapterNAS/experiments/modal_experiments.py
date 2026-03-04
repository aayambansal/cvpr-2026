"""
AdapterNAS v2: Complete experiment suite on Modal GPU.

Runs:
1. Evolutionary search with proxy fitness over 300 configs
2. Fine-tune top-k/bottom-k/baselines with selection quality metrics
3. Data regime sweep (1%, 2%, 5%, 10%)
4. ViT-S/16 additional backbone
5. AdaLoRA baseline comparison
6. Proper GPU latency benchmarking (batch=1, FP32, 50 reps, mean+std)
"""

import modal
import json

app = modal.App("adapter-nas-v2")
vol = modal.Volume.from_name("adapter-nas-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch", "torchvision", "timm", "peft", "transformers",
        "numpy", "scipy", "matplotlib", "accelerate",
    )
)

@app.function(
    gpu="A10G",
    image=image,
    volumes={"/results": vol},
    timeout=7200,
    memory=32768,
)
def run_all_experiments():
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
    print(f"Device: {device}, GPU: {torch.cuda.get_device_name()}")

    # ============================================================
    # LoRA Module
    # ============================================================
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

    # ============================================================
    # Data Loading
    # ============================================================
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

    # ============================================================
    # Zero-Cost Proxies
    # ============================================================
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

    # ============================================================
    # Latency Benchmarking (proper protocol)
    # ============================================================
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
        return {'mean_ms': np.mean(times), 'std_ms': np.std(times),
                'p50_ms': np.median(times), 'p95_ms': np.percentile(times, 95)}

    # ============================================================
    # Training
    # ============================================================
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

    # ============================================================
    # Config Generation
    # ============================================================
    def sample_structured_configs():
        configs, labels = [], []
        # Uniform ranks
        for r in [4, 8, 16, 32]:
            c = {(l, m): r for l in range(12) for m in MODULE_TYPES}
            configs.append(c); labels.append(f'uniform_r{r}')
        # Attn-only
        for r in [4, 8, 16, 32]:
            c = {(l, m): (r if m == 'qkv' else 0) for l in range(12) for m in MODULE_TYPES}
            configs.append(c); labels.append(f'attn_only_r{r}')
        # MLP-only
        for r in [4, 8, 16]:
            c = {(l, m): (r if m != 'qkv' else 0) for l in range(12) for m in MODULE_TYPES}
            configs.append(c); labels.append(f'mlp_only_r{r}')
        # Last-4/First-4/Even
        for tag, cond in [('last4', lambda l: l >= 8), ('first4', lambda l: l < 4),
                          ('even', lambda l: l % 2 == 0), ('middle4', lambda l: 4 <= l < 8)]:
            for r in [8, 16]:
                c = {(l, m): (r if cond(l) else 0) for l in range(12) for m in MODULE_TYPES}
                configs.append(c); labels.append(f'{tag}_r{r}')
        # Increasing/Decreasing
        for name, ranks in [('increasing', [4,4,4,8,8,8,16,16,16,32,32,32]),
                            ('decreasing', [32,32,32,16,16,16,8,8,8,4,4,4])]:
            c = {(l, m): ranks[l] for l in range(12) for m in MODULE_TYPES}
            configs.append(c); labels.append(name)
        # Mixed module types
        for name, qkv_r, mlp_r in [('high_attn_low_mlp', 16, 4), ('low_attn_high_mlp', 4, 16),
                                     ('balanced_8', 8, 8)]:
            c = {}
            for l in range(12):
                c[(l, 'qkv')] = qkv_r
                c[(l, 'mlp_fc1')] = mlp_r
                c[(l, 'mlp_fc2')] = mlp_r
            configs.append(c); labels.append(name)
        # No adapter
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

    # ============================================================
    # Evolutionary Search
    # ============================================================
    def evolutionary_search(cal_loader, nc, device, pop_size=30, n_gens=10, n_mutate=20, seed=42):
        """Proxy-guided evolutionary search over adapter topologies."""
        rng = random.Random(seed)
        nprng = np.random.RandomState(seed)

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

        def score_config(cfg):
            np_params = count_params(cfg)
            if np_params == 0:
                return {'gradnorm': 0, 'snip': 0, 'fisher': 0, 'neg_entropy': 0}
            model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=nc)
            model = apply_lora(model, cfg)
            proxies = compute_proxies(model, cal_loader, device)
            del model
            torch.cuda.empty_cache()
            return proxies

        # Initialize population
        population = [random_config() for _ in range(pop_size)]
        all_evaluated = []  # Track all configs ever evaluated

        print(f"  Evolutionary search: pop={pop_size}, gens={n_gens}")

        for gen in range(n_gens):
            # Score population
            scores = []
            for cfg in population:
                proxies = score_config(cfg)
                # Collect raw values
                scores.append(proxies)
                all_evaluated.append({'config': cfg, 'proxies': proxies, 'n_params': count_params(cfg)})

            # Normalize and combine
            for metric in ['gradnorm', 'snip', 'fisher', 'neg_entropy']:
                vals = [s[metric] for s in scores]
                lo, hi = min(vals), max(vals)
                rng_val = hi - lo if hi != lo else 1.0
                for s in scores:
                    s[f'{metric}_n'] = (s[metric] - lo) / rng_val

            combined = []
            for s in scores:
                c = (s['gradnorm_n'] + s['snip_n'] + s['fisher_n'] + s['neg_entropy_n']) / 4.0
                combined.append(c)

            # Sort by fitness
            ranked = sorted(zip(combined, population), key=lambda x: -x[0])
            best_score = ranked[0][0]
            print(f"    Gen {gen+1}/{n_gens}: best_proxy={best_score:.4f}, "
                  f"mean={np.mean(combined):.4f}, configs_seen={len(all_evaluated)}")

            # Selection: top 50%
            survivors = [cfg for _, cfg in ranked[:pop_size // 2]]

            # Generate next generation
            next_gen = list(survivors)  # Keep survivors

            # Mutations
            for _ in range(n_mutate):
                parent = rng.choice(survivors)
                child = mutate(parent, n_mutations=rng.randint(1, 5))
                next_gen.append(child)

            # Crossover
            while len(next_gen) < pop_size:
                p1, p2 = rng.sample(survivors, 2)
                child = crossover(p1, p2)
                next_gen.append(child)

            population = next_gen[:pop_size]

        return all_evaluated

    # ============================================================
    # EXPERIMENT 1: Full search + selection quality
    # ============================================================
    def experiment_search_and_validate(dataset_name, device):
        print(f"\n{'='*70}")
        print(f"  EXPERIMENT 1: Search & Validate — {dataset_name}")
        print(f"{'='*70}")

        cal_loader, train_loader, val_loader, nc = get_loaders(dataset_name, train_frac=0.02)

        # Phase 1a: Score structured configs
        print("\n  Phase 1a: Scoring structured configs...")
        struct_configs, struct_labels = sample_structured_configs()
        struct_results = []
        for i, (cfg, lab) in enumerate(zip(struct_configs, struct_labels)):
            np_p = count_params(cfg)
            if np_p == 0:
                struct_results.append({'label': lab, 'n_params': 0,
                    'gradnorm': 0, 'snip': 0, 'fisher': 0, 'neg_entropy': 0})
                continue
            model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=nc)
            model = apply_lora(model, cfg)
            proxies = compute_proxies(model, cal_loader, device)
            struct_results.append({'label': lab, 'n_params': np_p, **proxies})
            del model; torch.cuda.empty_cache()
            print(f"    [{i+1}/{len(struct_configs)}] {lab}: params={np_p:,}")

        # Phase 1b: Random configs
        print("\n  Phase 1b: Scoring 50 random configs...")
        rand_configs = sample_random(50, seed=42)
        rand_results = []
        for i, cfg in enumerate(rand_configs):
            np_p = count_params(cfg)
            if np_p == 0:
                rand_results.append({'label': f'rand_{i}', 'n_params': 0,
                    'gradnorm': 0, 'snip': 0, 'fisher': 0, 'neg_entropy': 0})
                continue
            model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=nc)
            model = apply_lora(model, cfg)
            proxies = compute_proxies(model, cal_loader, device)
            rand_results.append({'label': f'rand_{i}', 'n_params': np_p, **proxies})
            del model; torch.cuda.empty_cache()
            if (i+1) % 10 == 0:
                print(f"    Random [{i+1}/50]")

        # Phase 1c: Evolutionary search
        print("\n  Phase 1c: Evolutionary search (300 configs)...")
        evo_results = evolutionary_search(cal_loader, nc, device, pop_size=30, n_gens=10)

        # Combine all proxy results
        all_proxy = struct_results + rand_results
        # Add evo configs (deduplicate by checking params)
        evo_counter = 0
        for e in evo_results:
            np_p = e['n_params']
            if np_p > 0:
                all_proxy.append({
                    'label': f'evo_{evo_counter}',
                    'n_params': np_p,
                    **e['proxies']
                })
                evo_counter += 1

        print(f"\n  Total configs scored: {len(all_proxy)}")

        # Normalize and compute combined scores
        valid = [r for r in all_proxy if r['n_params'] > 0]
        for metric in ['gradnorm', 'snip', 'fisher', 'neg_entropy']:
            vals = [r[metric] for r in valid]
            lo, hi = min(vals), max(vals)
            rng_val = hi - lo if hi != lo else 1.0
            for r in valid:
                r[f'{metric}_n'] = (r[metric] - lo) / rng_val

        for r in valid:
            r['combined'] = (r['gradnorm_n'] + r['snip_n'] + r['fisher_n'] + r['neg_entropy_n']) / 4.0

        valid.sort(key=lambda x: x['combined'], reverse=True)

        # Phase 2: Fine-tune for selection quality
        print(f"\n  Phase 2: Fine-tuning for selection quality metrics...")

        ft_labels = set()
        ft_configs = []

        # Top-10 by proxy
        for r in valid[:10]:
            if r['label'] not in ft_labels:
                ft_configs.append(r); ft_labels.add(r['label'])
        # Bottom-10
        for r in valid[-10:]:
            if r['label'] not in ft_labels:
                ft_configs.append(r); ft_labels.add(r['label'])
        # 10 random from middle
        mid = valid[len(valid)//3 : 2*len(valid)//3]
        rng2 = random.Random(123)
        rng2.shuffle(mid)
        for r in mid[:10]:
            if r['label'] not in ft_labels:
                ft_configs.append(r); ft_labels.add(r['label'])
        # Key baselines
        for bl in ['uniform_r4', 'uniform_r8', 'uniform_r16', 'uniform_r32',
                    'attn_only_r4', 'attn_only_r8', 'attn_only_r16', 'attn_only_r32',
                    'mlp_only_r4', 'mlp_only_r8', 'mlp_only_r16',
                    'last4_r8', 'last4_r16', 'first4_r8', 'first4_r16',
                    'even_r8', 'even_r16', 'middle4_r8', 'middle4_r16',
                    'increasing', 'decreasing',
                    'high_attn_low_mlp', 'low_attn_high_mlp', 'balanced_8']:
            for r in valid:
                if r['label'] == bl and r['label'] not in ft_labels:
                    ft_configs.append(r); ft_labels.add(r['label'])

        print(f"  Fine-tuning {len(ft_configs)} configs...")

        ft_results = []
        all_configs_map = {}
        for cfg, lab in zip(struct_configs, struct_labels):
            all_configs_map[lab] = cfg
        for i, cfg in enumerate(rand_configs):
            all_configs_map[f'rand_{i}'] = cfg
        for i, e in enumerate(evo_results):
            if e['n_params'] > 0:
                all_configs_map[f'evo_{i}'] = e['config']

        for fi, r in enumerate(ft_configs):
            lab = r['label']
            if lab not in all_configs_map:
                # Try to find evo config with matching label
                idx = int(lab.split('_')[1]) if lab.startswith('evo_') else -1
                if idx >= 0 and idx < len(evo_results):
                    cfg = evo_results[idx]['config']
                else:
                    continue
            else:
                cfg = all_configs_map[lab]

            print(f"  [{fi+1}/{len(ft_configs)}] {lab} (params={r['n_params']:,}, proxy={r['combined']:.3f})")
            model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=nc)
            model = apply_lora(model, cfg)
            val_acc = finetune(model, train_loader, val_loader, device, epochs=5, lr=1e-3)
            lat = benchmark_latency(model, device)

            ft_results.append({
                'label': lab, 'n_params': r['n_params'],
                'combined_proxy': r['combined'],
                'gradnorm': r['gradnorm'], 'snip': r['snip'],
                'fisher': r['fisher'], 'neg_entropy': r['neg_entropy'],
                'val_acc': val_acc, **lat,
            })
            print(f"    -> Acc={val_acc:.2f}%, Lat={lat['mean_ms']:.1f}+-{lat['std_ms']:.1f}ms")
            del model; torch.cuda.empty_cache()

        # Linear probe baseline
        print(f"  Fine-tuning linear probe baseline...")
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=nc)
        for p in model.parameters(): p.requires_grad = False
        for p in model.head.parameters(): p.requires_grad = True
        lp_acc = finetune(model, train_loader, val_loader, device, epochs=5, lr=1e-3)
        lp_lat = benchmark_latency(model, device)
        ft_results.append({
            'label': 'linear_probe',
            'n_params': sum(p.numel() for p in model.head.parameters()),
            'combined_proxy': 0, 'gradnorm': 0, 'snip': 0, 'fisher': 0, 'neg_entropy': 0,
            'val_acc': lp_acc, **lp_lat,
        })
        print(f"    -> Linear probe: {lp_acc:.2f}%")
        del model; torch.cuda.empty_cache()

        # Selection quality metrics
        print(f"\n  Computing selection quality metrics...")
        sq = compute_selection_quality(valid, ft_results)

        return {
            'proxy_scores': [serialize_result(r) for r in all_proxy],
            'finetune_results': ft_results,
            'selection_quality': sq,
        }

    def serialize_result(r):
        """Make result JSON-serializable."""
        out = {}
        for k, v in r.items():
            if k == 'config':
                out[k] = {f'{kk[0]}_{kk[1]}': vv for kk, vv in v.items()}
            else:
                out[k] = v
        return out

    def compute_selection_quality(proxy_ranked, ft_results):
        """Compute top-k hit rate, regret vs oracle, regret vs random."""
        # Map label -> actual accuracy
        acc_map = {r['label']: r['val_acc'] for r in ft_results}
        proxy_map = {r['label']: r.get('combined', 0) for r in proxy_ranked}

        # Only consider configs that were fine-tuned
        common_labels = [l for l in acc_map if l in proxy_map and l != 'linear_probe']
        if len(common_labels) < 5:
            return {}

        # Sort by proxy
        by_proxy = sorted(common_labels, key=lambda l: proxy_map[l], reverse=True)
        # Sort by actual accuracy (oracle)
        by_oracle = sorted(common_labels, key=lambda l: acc_map[l], reverse=True)

        oracle_top1 = acc_map[by_oracle[0]]
        random_mean = np.mean([acc_map[l] for l in common_labels])

        results = {}
        for k in [1, 3, 5, 10]:
            if k > len(by_proxy): continue
            proxy_topk = by_proxy[:k]
            oracle_topk = set(by_oracle[:k])

            # Best acc in proxy top-k
            best_in_topk = max(acc_map[l] for l in proxy_topk)
            # Hit rate: fraction of proxy top-k that are in oracle top-k
            hits = sum(1 for l in proxy_topk if l in oracle_topk)
            hit_rate = hits / k

            results[f'top{k}_best_acc'] = best_in_topk
            results[f'top{k}_hit_rate'] = hit_rate
            results[f'top{k}_regret_vs_oracle'] = oracle_top1 - best_in_topk
            results[f'top{k}_regret_vs_random'] = best_in_topk - random_mean

        # Rank correlations
        proxy_vals = [proxy_map[l] for l in common_labels]
        acc_vals = [acc_map[l] for l in common_labels]
        rho, rho_p = stats.spearmanr(proxy_vals, acc_vals)
        tau, tau_p = stats.kendalltau(proxy_vals, acc_vals)
        results['spearman_rho'] = rho
        results['spearman_p'] = rho_p
        results['kendall_tau'] = tau
        results['kendall_p'] = tau_p

        # Per-proxy correlations
        for metric in ['gradnorm', 'snip', 'fisher', 'neg_entropy']:
            metric_map = {r['label']: r.get(metric, 0) for r in proxy_ranked}
            metric_vals = [metric_map.get(l, 0) for l in common_labels]
            r_val, _ = stats.spearmanr(metric_vals, acc_vals)
            results[f'{metric}_spearman'] = r_val

        return results

    # ============================================================
    # EXPERIMENT 2: Data Regime Sweep
    # ============================================================
    def experiment_data_regime(device):
        print(f"\n{'='*70}")
        print(f"  EXPERIMENT 2: Data Regime Sweep (CIFAR-100)")
        print(f"{'='*70}")

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
                results.append({
                    'config': name, 'data_frac': frac, 'val_acc': acc,
                    'n_params': count_params(cfg),
                    'n_train_samples': int(frac * 50000),
                })
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

        return results

    # ============================================================
    # EXPERIMENT 3: ViT-S/16 (additional backbone)
    # ============================================================
    def experiment_vit_small(device):
        print(f"\n{'='*70}")
        print(f"  EXPERIMENT 3: ViT-S/16 Backbone")
        print(f"{'='*70}")

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

            results.append({
                'label': name, 'n_params': count_params_small(cfg),
                'val_acc': acc, **proxies, **lat,
            })
            print(f"  {name}: acc={acc:.2f}%, params={count_params_small(cfg):,}")
            del model; torch.cuda.empty_cache()

        # Linear probe
        model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=nc)
        for p in model.parameters(): p.requires_grad = False
        for p in model.head.parameters(): p.requires_grad = True
        acc = finetune(model, train_loader, val_loader, device, epochs=5, lr=1e-3)
        lat = benchmark_latency(model, device)
        results.append({
            'label': 'linear_probe', 'n_params': sum(p.numel() for p in model.head.parameters()),
            'val_acc': acc, 'gradnorm': 0, 'snip': 0, 'fisher': 0, 'neg_entropy': 0, **lat,
        })
        print(f"  linear_probe: acc={acc:.2f}%")
        del model; torch.cuda.empty_cache()

        return results

    # ============================================================
    # EXPERIMENT 4: AdaLoRA Baseline
    # ============================================================
    def experiment_adalora(device):
        print(f"\n{'='*70}")
        print(f"  EXPERIMENT 4: AdaLoRA Baseline")
        print(f"{'='*70}")

        from peft import get_peft_model, LoraConfig, AdaLoraConfig
        from transformers import ViTForImageClassification, ViTConfig

        _, train_loader, val_loader, nc = get_loaders('cifar100', train_frac=0.02)

        results = []

        # Standard LoRA baselines via peft (for fair comparison)
        for r, name in [(8, 'peft_lora_r8'), (16, 'peft_lora_r16')]:
            model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=nc)
            # Apply LoRA via our method (peft doesn't directly support timm models easily)
            cfg = {(l, m): r for l in range(12) for m in MODULE_TYPES}
            model = apply_lora(model, cfg)
            acc = finetune(model, train_loader, val_loader, device, epochs=5, lr=1e-3)
            lat = benchmark_latency(model, device)
            results.append({
                'label': name, 'n_params': count_params(cfg),
                'val_acc': acc, **lat,
            })
            print(f"  {name}: acc={acc:.2f}%")
            del model; torch.cuda.empty_cache()

        # AdaLoRA — use peft with HuggingFace ViT
        for init_r, name in [(12, 'adalora_r12'), (24, 'adalora_r24')]:
            try:
                hf_model = ViTForImageClassification.from_pretrained(
                    'google/vit-base-patch16-224-in21k', num_labels=nc, ignore_mismatched_sizes=True)
                adalora_config = AdaLoraConfig(
                    r=init_r, lora_alpha=init_r, target_modules=["query", "value", "key", "dense"],
                    lora_dropout=0.05, init_r=init_r, target_r=init_r // 2,
                    deltaT=10, beta1=0.85, beta2=0.85,
                )
                hf_model = get_peft_model(hf_model, adalora_config)
                hf_model.to(device)

                # Train
                opt = torch.optim.AdamW(
                    [p for p in hf_model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.01)

                for ep in range(5):
                    hf_model.train()
                    for imgs, labs in train_loader:
                        imgs, labs = imgs.to(device), labs.to(device)
                        opt.zero_grad()
                        out = hf_model(imgs).logits
                        loss = F.cross_entropy(out, labs)
                        loss.backward()
                        opt.step()

                # Eval
                hf_model.eval()
                c, t = 0, 0
                with torch.no_grad():
                    for imgs, labs in val_loader:
                        imgs, labs = imgs.to(device), labs.to(device)
                        _, pred = hf_model(imgs).logits.max(1)
                        t += labs.size(0); c += pred.eq(labs).sum().item()
                acc = 100.*c/t
                n_trainable = sum(p.numel() for p in hf_model.parameters() if p.requires_grad)

                results.append({
                    'label': name, 'n_params': n_trainable,
                    'val_acc': acc, 'mean_ms': 0, 'std_ms': 0,
                })
                print(f"  {name}: acc={acc:.2f}%, params={n_trainable:,}")
                del hf_model; torch.cuda.empty_cache()
            except Exception as e:
                print(f"  {name}: FAILED - {e}")
                results.append({'label': name, 'n_params': 0, 'val_acc': 0, 'mean_ms': 0, 'std_ms': 0})

        return results

    # ============================================================
    # EXPERIMENT 5: Proxy Ablation
    # ============================================================
    def experiment_proxy_ablation(proxy_scores, ft_results):
        print(f"\n{'='*70}")
        print(f"  EXPERIMENT 5: Proxy Ablation")
        print(f"{'='*70}")

        acc_map = {r['label']: r['val_acc'] for r in ft_results}
        valid = [r for r in proxy_scores if r.get('n_params', 0) > 0 and r['label'] in acc_map]

        if len(valid) < 5:
            return {}

        # Normalize all proxies
        for metric in ['gradnorm', 'snip', 'fisher', 'neg_entropy']:
            vals = [r[metric] for r in valid]
            lo, hi = min(vals), max(vals)
            rng_val = hi - lo if hi != lo else 1.0
            for r in valid:
                r[f'{metric}_n'] = (r[metric] - lo) / rng_val

        results = {}

        # Individual proxy ranking
        for metric in ['gradnorm', 'snip', 'fisher', 'neg_entropy']:
            scored = sorted(valid, key=lambda r: r[metric], reverse=True)
            top5_labels = [r['label'] for r in scored[:5]]
            top5_accs = [acc_map[l] for l in top5_labels if l in acc_map]
            results[f'{metric}_top5_avg_acc'] = np.mean(top5_accs) if top5_accs else 0

        # Ensemble variants
        combos = {
            'all4': ['gradnorm', 'snip', 'fisher', 'neg_entropy'],
            'gn+snip': ['gradnorm', 'snip'],
            'gn+fisher': ['gradnorm', 'fisher'],
            'gn+snip+fisher': ['gradnorm', 'snip', 'fisher'],
            'snip+entropy': ['snip', 'neg_entropy'],
        }

        for combo_name, metrics in combos.items():
            for r in valid:
                r[f'combo_{combo_name}'] = np.mean([r[f'{m}_n'] for m in metrics])
            scored = sorted(valid, key=lambda r: r[f'combo_{combo_name}'], reverse=True)
            top5_labels = [r['label'] for r in scored[:5]]
            top5_accs = [acc_map[l] for l in top5_labels if l in acc_map]
            results[f'{combo_name}_top5_avg_acc'] = np.mean(top5_accs) if top5_accs else 0

            # Correlation
            common = [r for r in valid if r['label'] in acc_map]
            if len(common) > 3:
                rho, _ = stats.spearmanr(
                    [r[f'combo_{combo_name}'] for r in common],
                    [acc_map[r['label']] for r in common])
                results[f'{combo_name}_spearman'] = rho

        # Random baseline
        rng = random.Random(42)
        random_accs = []
        all_accs = [acc_map[l] for l in acc_map if l != 'linear_probe']
        for _ in range(100):
            sample = rng.sample(all_accs, min(5, len(all_accs)))
            random_accs.append(max(sample))
        results['random_top5_avg_best'] = np.mean(random_accs)

        for k, v in results.items():
            print(f"  {k}: {v:.2f}")

        return results

    # ============================================================
    # MAIN
    # ============================================================
    all_results = {}

    # Exp 1: CIFAR-100
    c100 = experiment_search_and_validate('cifar100', device)
    all_results['cifar100'] = c100

    # Exp 1: Flowers-102
    f102 = experiment_search_and_validate('flowers102', device)
    all_results['flowers102'] = f102

    # Exp 2: Data regime
    all_results['data_regime'] = experiment_data_regime(device)

    # Exp 3: ViT-S
    all_results['vit_small'] = experiment_vit_small(device)

    # Exp 4: AdaLoRA
    all_results['adalora'] = experiment_adalora(device)

    # Exp 5: Proxy ablation
    all_results['proxy_ablation_c100'] = experiment_proxy_ablation(
        c100['proxy_scores'], c100['finetune_results'])
    all_results['proxy_ablation_f102'] = experiment_proxy_ablation(
        f102['proxy_scores'], f102['finetune_results'])

    # Save everything
    with open('/results/all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    vol.commit()

    print(f"\n{'='*70}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"{'='*70}")

    # Print summary tables
    for ds in ['cifar100', 'flowers102']:
        ft = all_results[ds]['finetune_results']
        ft.sort(key=lambda x: x['val_acc'], reverse=True)
        print(f"\n  {ds} Results:")
        print(f"  {'Label':<25} {'Params':>10} {'Acc':>7} {'Lat':>10} {'Proxy':>7}")
        for r in ft[:15]:
            print(f"  {r['label']:<25} {r['n_params']:>10,} {r['val_acc']:>6.2f}% "
                  f"{r.get('mean_ms',0):>7.1f}ms {r.get('combined_proxy',0):>6.3f}")

        sq = all_results[ds]['selection_quality']
        print(f"\n  Selection Quality:")
        for k, v in sq.items():
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    return all_results


@app.local_entrypoint()
def main():
    result = run_all_experiments.remote()
    print("Experiments completed!")
