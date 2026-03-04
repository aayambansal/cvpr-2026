"""
TransNAS-Bench-101: Optimized New Baselines + Robustness Sweeps
===============================================================
Faster version: TE-NAS uses 2 samples, parallel execution, reduced robustness sweep.

Deploy: modal deploy code/transnas_fast_eval.py
Launch: python3 -c "
import modal
modal.Function.from_name('transnas-fast', 'evaluate_new_baselines').spawn()
modal.Function.from_name('transnas-fast', 'robustness_sweep').spawn()
print('Both jobs spawned!')
"
Monitor: modal app logs transnas-fast
"""
import modal
import json
import os

app = modal.App("transnas-fast")
volume = modal.Volume.from_name("transnas-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .uv_pip_install(
        "torch==2.2.2",
        "numpy<2",
        "scipy",
        "gdown",
        "torchsummary",
    )
    .run_commands(
        "git clone https://github.com/yawen-d/TransNASBench.git /opt/transnas",
        "find /opt/transnas -name '*.py' -exec sed -i 's/from torchsummary import summary/# from torchsummary import summary/g' {} +",
        "sed -i 's/from procedures import task_demo/# from procedures import task_demo/g' /opt/transnas/lib/models/utils.py",
    )
)


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="A10G",
    timeout=7200,  # 2 hours
    memory=32768,
)
def evaluate_new_baselines():
    """
    Evaluate TE-NAS, ZenNAS, and NASWOT-fixed on all 7,344 architectures.
    Optimized: TE-NAS uses 2 samples (not batch_size=4), batch checkpoints.
    """
    import sys
    sys.path.insert(0, "/opt/transnas")
    sys.path.insert(0, "/opt/transnas/api")
    sys.path.insert(0, "/opt/transnas/lib")

    import torch
    import torch.nn as nn
    import numpy as np
    import math
    import time
    import traceback

    from api import TransNASBenchAPI as API
    from models.net_infer.net_macro import MacroNet

    volume.reload()
    bench_path = "/data/transnas-bench_v10141024.pth"
    api = API(bench_path, verbose=False)

    device = torch.device("cuda")
    SEED = 42

    # ================================================================
    # PROXY IMPLEMENTATIONS (optimized)
    # ================================================================

    def tenas_proxy(model, device, input_dim):
        """
        TE-NAS: NTK condition number approximation.
        Use 2 samples with batch_size=1 each for speed.
        """
        model.train()
        grads_list = []
        
        for i in range(2):  # 2 samples only
            torch.manual_seed(SEED + i)
            x = torch.randn(1, 3, input_dim, input_dim, device=device)
            model.zero_grad()
            out = model(x)
            if isinstance(out, list):
                loss = sum(o.sum() for o in out)
            else:
                loss = out.sum()
            loss.backward()
            
            grad_vec = []
            for p in model.parameters():
                if p.grad is not None:
                    grad_vec.append(p.grad.data.view(-1).clone())
            if grad_vec:
                grads_list.append(torch.cat(grad_vec))
        
        if len(grads_list) < 2:
            return 0.0
        
        G = torch.stack(grads_list)
        K = G @ G.T
        
        try:
            eigs = torch.linalg.eigvalsh(K)
            eigs = eigs[eigs > 1e-10]
            if len(eigs) < 2:
                return 0.0
            cond = (eigs.max() / eigs.min()).item()
            return -math.log10(cond + 1)
        except:
            return 0.0

    def zennas_proxy(model, device, input_dim):
        """ZenNAS: Gaussian complexity - log(E[||f(x)||])."""
        model.eval()
        scores = []
        for i in range(3):  # 3 samples
            torch.manual_seed(SEED + i + 100)
            x = torch.randn(2, 3, input_dim, input_dim, device=device)
            with torch.no_grad():
                out = model(x)
                if isinstance(out, list):
                    out_norm = sum(o.norm().item() for o in out)
                else:
                    out_norm = out.norm().item()
                scores.append(out_norm)
        
        mean_score = np.mean(scores)
        if mean_score <= 0:
            return 0.0
        return math.log(mean_score + 1e-10)

    def naswot_fixed_proxy(model, device, input_dim):
        """NASWOT-fixed: ReLU activation diversity on intermediate layers."""
        model.eval()
        acts = []
        hooks = []
        
        for name, mod in model.named_modules():
            if isinstance(mod, (nn.ReLU, nn.ReLU6, nn.LeakyReLU)):
                hooks.append(mod.register_forward_hook(
                    lambda _m, _i, o, _n=name: acts.append(
                        (o > 0).float().view(o.size(0), -1)
                    )
                ))
            elif isinstance(mod, nn.BatchNorm2d):
                hooks.append(mod.register_forward_hook(
                    lambda _m, _i, o, _n=name: acts.append(
                        (o > 0).float().view(o.size(0), -1)
                    )
                ))
        
        torch.manual_seed(SEED)
        x = torch.randn(4, 3, input_dim, input_dim, device=device)
        with torch.no_grad():
            model(x)
        
        for h in hooks:
            h.remove()
        
        if not acts:
            return 0.0
        
        if len(acts) > 20:
            indices = np.linspace(0, len(acts)-1, 20, dtype=int)
            acts = [acts[i] for i in indices]
        
        c = torch.cat(acts, 1)
        if c.size(1) > 50000:
            idx = torch.randperm(c.size(1))[:50000]
            c = c[:, idx]
        
        K = (c @ c.t()) / c.size(1) + 1e-5 * torch.eye(c.size(0), device=c.device)
        try:
            return float(np.log(np.abs(np.linalg.det(K.cpu().numpy())) + 1e-10))
        except:
            return 0.0

    # ================================================================
    # MAIN EVALUATION
    # ================================================================
    results = {}
    errors = []

    all_archs = []
    for ss in api.search_spaces:
        for arch_str in api.all_arch_dict[ss]:
            all_archs.append((ss, arch_str))

    print(f"Evaluating {len(all_archs)} architectures with TE-NAS, ZenNAS, NASWOT-fixed...")
    t_start = time.time()

    for idx, (ss, arch_str) in enumerate(all_archs):
        if idx % 200 == 0:
            elapsed = time.time() - t_start
            rate = idx / max(elapsed, 1)
            eta = (len(all_archs) - idx) / max(rate, 0.01)
            print(f"[{idx}/{len(all_archs)}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, {len(errors)} errors")

        try:
            input_dim = 64
            
            # TE-NAS
            torch.manual_seed(SEED)
            model = MacroNet(arch_str, structure='backbone', input_dim=(input_dim, input_dim)).to(device)
            te_score = tenas_proxy(model, device, input_dim)
            del model

            # ZenNAS
            torch.manual_seed(SEED)
            model = MacroNet(arch_str, structure='backbone', input_dim=(input_dim, input_dim)).to(device)
            zen_score = zennas_proxy(model, device, input_dim)
            del model

            # NASWOT-fixed
            torch.manual_seed(SEED)
            model = MacroNet(arch_str, structure='backbone', input_dim=(input_dim, input_dim)).to(device)
            nw_score = naswot_fixed_proxy(model, device, input_dim)
            del model

            results[arch_str] = {
                'search_space': ss,
                'tenas': te_score,
                'zennas': zen_score,
                'naswot_fixed': nw_score,
            }

            if idx % 100 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            errors.append({
                'arch': arch_str,
                'search_space': ss,
                'error': str(e),
            })
            if len(errors) <= 10:
                print(f"  ERROR on {arch_str}: {e}")

        # Checkpoint every 1000
        if (idx + 1) % 1000 == 0:
            checkpoint = {
                'meta': {'evaluated': len(results), 'total': len(all_archs), 'errors': len(errors)},
                'results': results,
                'errors': errors[:20],
            }
            with open("/data/transnas_new_baselines_checkpoint.json", 'w') as f:
                json.dump(checkpoint, f)
            volume.commit()
            print(f"  Checkpoint saved: {len(results)} results")

    elapsed = time.time() - t_start
    print(f"\nDone! Evaluated {len(results)} architectures in {elapsed:.0f}s ({len(errors)} errors)")

    output = {
        'meta': {
            'total': len(all_archs),
            'successful': len(results),
            'errors': len(errors),
            'elapsed_seconds': elapsed,
            'input_dim': 64,
            'seed': SEED,
            'proxies': ['tenas', 'zennas', 'naswot_fixed'],
        },
        'results': results,
        'errors': errors[:50],
    }

    with open("/data/transnas_new_baselines.json", 'w') as f:
        json.dump(output, f)
    print(f"Saved to /data/transnas_new_baselines.json")
    volume.commit()
    return output['meta']


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="A10G",
    timeout=7200,  # 2 hours
    memory=32768,
)
def robustness_sweep():
    """
    Robustness sweep: MSFS, SFC, GradNorm, SynFlow across
    10 seeds x 3 resolutions x 3 batch sizes on 300 representative architectures.
    """
    import sys
    sys.path.insert(0, "/opt/transnas")
    sys.path.insert(0, "/opt/transnas/api")
    sys.path.insert(0, "/opt/transnas/lib")

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import math
    import time

    from api import TransNASBenchAPI as API
    from models.net_infer.net_macro import MacroNet

    volume.reload()
    bench_path = "/data/transnas-bench_v10141024.pth"
    api = API(bench_path, verbose=False)

    device = torch.device("cuda")

    # Proxy implementations
    def extract_multiscale_features(model, x):
        feats = []
        x = model.stem(x)
        feats.append(x)
        for layer_name in model.layers:
            layer = getattr(model, layer_name)
            x = layer(x)
            feats.append(x)
        return feats

    def msfs_component_isd(feats):
        vecs = [f.mean(dim=(2, 3)) for f in feats]
        div_val, cnt = 0.0, 0
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                d = min(vecs[i].size(1), vecs[j].size(1))
                a, b = vecs[i][:, :d], vecs[j][:, :d]
                md = (a.mean(0) - b.mean(0)).norm().item()
                vr = abs(math.log((a.var(0).mean().item() + 1e-8) /
                                  (b.var(0).mean().item() + 1e-8)))
                ca = (a.t() @ a) / a.size(0)
                cb = (b.t() @ b) / b.size(0)
                cd = (ca - cb).norm().item() / d
                div_val += md + vr + cd
                cnt += 1
        return div_val / max(cnt, 1)

    def msfs_component_sa(feats):
        sa_val = sum(
            f.var(dim=(2, 3)).mean().item() * f.var(dim=1).mean().item()
            for f in feats
        )
        return math.log(sa_val + 1e-10)

    def msfs_proxy(feats, delta=0.6, eta=0.4):
        if len(feats) < 2:
            return 0.0
        isd = msfs_component_isd(feats)
        sa = msfs_component_sa(feats)
        sa_norm = max(sa + 10, 0) / 10
        return delta * isd + eta * sa_norm

    def sfc_proxy(feats, alpha=0.4, beta=0.35, gamma=0.25):
        nc_val, nc_cnt = 0.0, 0
        for f in feats:
            B, C, H, W = f.shape
            ps = 4
            if H < ps * 2 or W < ps * 2:
                continue
            ph, pw = H // ps, W // ps
            fc = f[:, :, :ph*ps, :pw*ps].reshape(B, C, ph, ps, pw, ps).mean(dim=(3, 5))
            if pw > 1:
                nc_val += F.cosine_similarity(
                    fc[:, :, :, :-1].reshape(B, C, -1),
                    fc[:, :, :, 1:].reshape(B, C, -1), dim=1
                ).mean().item()
                nc_cnt += 1
            if ph > 1:
                nc_val += F.cosine_similarity(
                    fc[:, :, :-1, :].reshape(B, C, -1),
                    fc[:, :, 1:, :].reshape(B, C, -1), dim=1
                ).mean().item()
                nc_cnt += 1
        nc = nc_val / max(nc_cnt, 1)
        
        bs_val, bs_cnt = 0.0, 0
        for f in feats:
            if f.size(2) < 3 or f.size(3) < 3:
                continue
            gm = ((f[:, :, :, 1:] - f[:, :, :, :-1]).abs().mean() +
                  (f[:, :, 1:, :] - f[:, :, :-1, :]).abs().mean()) / 2
            bs_val += (gm / (f.abs().mean() + 1e-8)).item()
            bs_cnt += 1
        bs = bs_val / max(bs_cnt, 1)
        
        sd_vals = [f.var(dim=(2, 3)).mean().item() for f in feats]
        sd = np.mean([math.log(v + 1e-10) for v in sd_vals])
        sd_norm = (sd + 10) / 10
        
        return alpha * nc + beta * bs + gamma * sd_norm

    def gradnorm_proxy(model, x):
        model.train()
        model.zero_grad()
        out = model(x)
        if isinstance(out, list):
            loss = sum(o.sum() for o in out)
        else:
            loss = out.sum()
        loss.backward()
        gn = math.sqrt(sum(
            p.grad.norm(2).item() ** 2
            for p in model.parameters() if p.grad is not None
        ))
        return math.log10(gn + 1)

    def synflow_proxy(model, x):
        model.eval()
        signs = {n: torch.sign(p.data).clone() for n, p in model.named_parameters()}
        for p in model.parameters():
            p.data.abs_()
        model.zero_grad()
        out = model(torch.ones_like(x))
        if isinstance(out, list):
            loss = sum(o.sum() for o in out)
        else:
            loss = out.sum()
        loss.backward()
        log_terms = []
        for p in model.parameters():
            if p.grad is not None:
                prod = (p.data * p.grad.data).sum().item()
                if prod > 0:
                    log_terms.append(math.log10(prod))
        for n, p in model.named_parameters():
            p.data *= signs[n]
        if not log_terms:
            return 0.0
        max_lt = max(log_terms)
        s = max_lt + math.log10(sum(10 ** (lt - max_lt) for lt in log_terms))
        if math.isnan(s) or math.isinf(s):
            return 30.0
        return s

    # ================================================================
    # SELECT 300 REPRESENTATIVE ARCHITECTURES
    # ================================================================
    all_archs = []
    for ss in api.search_spaces:
        for arch_str in api.all_arch_dict[ss]:
            all_archs.append((ss, arch_str))

    macro_archs = [(ss, a) for ss, a in all_archs if ss == 'macro']
    micro_archs = [(ss, a) for ss, a in all_archs if ss == 'micro']

    np.random.seed(42)
    n_macro = min(135, len(macro_archs))
    n_micro = min(165, len(micro_archs))
    sample_macro = [macro_archs[i] for i in np.random.choice(len(macro_archs), n_macro, replace=False)]
    sample_micro = [micro_archs[i] for i in np.random.choice(len(micro_archs), n_micro, replace=False)]
    sample_archs = sample_macro + sample_micro
    print(f"Sampled {len(sample_archs)} architectures ({n_macro} macro, {n_micro} micro)")

    # ================================================================
    # ROBUSTNESS SWEEP: 10 seeds, 3 resolutions, 3 batch sizes
    # ================================================================
    seeds = list(range(10))
    resolutions = [32, 64, 128]
    batch_sizes = [1, 4, 16]

    configs = []
    # Seed sweep: 10 seeds at default res=64, batch=4
    for seed in seeds:
        configs.append(('seed', seed, 64, 4))
    # Resolution sweep: 3 resolutions at default seed=42, batch=4
    for res in resolutions:
        configs.append(('resolution', 42, res, 4))
    # Batch size sweep: 3 batch sizes at default seed=42, res=64
    for bs in batch_sizes:
        configs.append(('batch_size', 42, 64, bs))
    
    # Remove duplicates
    seen = set()
    unique_configs = []
    for config_type, seed, res, bs in configs:
        key = (seed, res, bs)
        if key not in seen:
            seen.add(key)
            unique_configs.append((config_type, seed, res, bs))
    
    print(f"Running {len(unique_configs)} unique configs x {len(sample_archs)} archs = {len(unique_configs) * len(sample_archs)} evaluations")

    all_results = {}
    t_start = time.time()

    for cfg_idx, (config_type, seed, res, bs) in enumerate(unique_configs):
        config_key = f"seed{seed}_res{res}_bs{bs}"
        elapsed = time.time() - t_start
        rate = cfg_idx / max(elapsed, 1)
        eta = (len(unique_configs) - cfg_idx) / max(rate, 0.01) if rate > 0 else 0
        print(f"\n[Config {cfg_idx+1}/{len(unique_configs)}] {config_key} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
        
        config_results = {}
        errors_this = 0
        
        for arch_idx, (ss, arch_str) in enumerate(sample_archs):
            try:
                input_size = (bs, 3, res, res)
                
                # MSFS + SFC
                torch.manual_seed(seed)
                model = MacroNet(arch_str, structure='backbone', input_dim=(res, res)).to(device)
                model.eval()
                x = torch.randn(*input_size, device=device)
                
                with torch.no_grad():
                    feats = extract_multiscale_features(model, x)
                
                msfs_val = msfs_proxy(feats)
                sfc_val = sfc_proxy(feats)
                del model, feats, x

                # GradNorm
                torch.manual_seed(seed)
                model_gn = MacroNet(arch_str, structure='backbone', input_dim=(res, res)).to(device)
                x_gn = torch.randn(*input_size, device=device)
                gn_val = gradnorm_proxy(model_gn, x_gn)
                del model_gn, x_gn

                # SynFlow
                torch.manual_seed(seed)
                model_sf = MacroNet(arch_str, structure='backbone', input_dim=(res, res)).to(device)
                x_sf = torch.randn(*input_size, device=device)
                sf_val = synflow_proxy(model_sf, x_sf)
                del model_sf, x_sf

                config_results[arch_str] = {
                    'msfs': msfs_val,
                    'sfc': sfc_val,
                    'gradnorm': gn_val,
                    'synflow': sf_val,
                    'search_space': ss,
                }

            except Exception as e:
                errors_this += 1
                if errors_this <= 3:
                    print(f"    ERROR on {arch_str}: {e}")

            if arch_idx % 100 == 0:
                torch.cuda.empty_cache()

        all_results[config_key] = config_results
        print(f"  Completed: {len(config_results)} archs, {errors_this} errors")
        
        # Checkpoint every 3 configs
        if (cfg_idx + 1) % 3 == 0:
            checkpoint = {
                'meta': {
                    'configs_completed': cfg_idx + 1,
                    'total_configs': len(unique_configs),
                    'n_archs': len(sample_archs),
                    'elapsed_seconds': time.time() - t_start,
                },
                'results': all_results,
                'sample_archs': [(ss, a) for ss, a in sample_archs],
            }
            with open("/data/transnas_robustness_checkpoint.json", 'w') as f:
                json.dump(checkpoint, f)
            volume.commit()

    elapsed = time.time() - t_start
    print(f"\nAll configs done in {elapsed:.0f}s")

    output = {
        'meta': {
            'configs': [(ct, s, r, b) for ct, s, r, b in unique_configs],
            'n_archs': len(sample_archs),
            'n_macro': n_macro,
            'n_micro': n_micro,
            'elapsed_seconds': elapsed,
        },
        'results': all_results,
        'sample_archs': [(ss, a) for ss, a in sample_archs],
    }

    with open("/data/transnas_robustness.json", 'w') as f:
        json.dump(output, f)
    print(f"Saved to /data/transnas_robustness.json")
    volume.commit()
    return output['meta']


@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={"/data": volume},
    timeout=300,
)
def get_results(filename: str):
    """Download results from volume."""
    volume.reload()
    path = f"/data/{filename}"
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


@app.local_entrypoint()
def main():
    """Download results."""
    for filename in ['transnas_new_baselines.json', 'transnas_robustness.json']:
        print(f"\nDownloading {filename}...")
        data = get_results.remote(filename)
        if data:
            local_path = f"results/{filename}"
            with open(local_path, 'w') as f:
                json.dump(data, f)
            print(f"Saved to {local_path}")
        else:
            print(f"Not found: {filename}")
