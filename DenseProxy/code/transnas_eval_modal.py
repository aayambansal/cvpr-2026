"""
TransNAS-Bench-101 Proxy Evaluation via Modal
==============================================
Evaluates MSFS, SFC, and baseline proxies on all 7,352 TransNAS-Bench-101
architectures and correlates with REAL trained ground truth performance.

The benchmark file must be pre-uploaded to a Modal Volume.
"""
import modal
import json
import os

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
app = modal.App("transnas-proxy-eval")

volume = modal.Volume.from_name("transnas-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .uv_pip_install(
        "torch==2.2.2",
        "numpy",
        "scipy",
        "gdown",
        "torchsummary",
    )
    .run_commands(
        # Clone TransNAS-Bench repo
        "git clone https://github.com/yawen-d/TransNASBench.git /opt/transnas",
        # Patch out ALL torchsummary imports (they're only used in __main__ blocks)
        "find /opt/transnas -name '*.py' -exec sed -i 's/from torchsummary import summary/# from torchsummary import summary/g' {} +",
        # Patch out the skimage dependency chain: utils.py imports procedures.task_demo
        # which imports skimage. We only need merge_list from utils.py.
        "sed -i 's/from procedures import task_demo/# from procedures import task_demo/g' /opt/transnas/lib/models/utils.py",
    )
)


# ---------------------------------------------------------------------------
# Step 1: Download the TransNAS-Bench-101 benchmark file to Modal Volume
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=3600,
    memory=16384,
)
def download_benchmark():
    """Download the TransNAS-Bench-101 .pth file from Google Drive."""
    import gdown
    import os

    out_path = "/data/transnas-bench_v10141024.pth"
    if os.path.exists(out_path) and os.path.getsize(out_path) > 100_000_000:
        print(f"Benchmark file already exists ({os.path.getsize(out_path) / 1e6:.1f} MB), skipping download.")
        volume.commit()
        return out_path

    print("Downloading TransNAS-Bench-101 from Google Drive...")
    # The file is at: https://drive.google.com/drive/folders/1HlLr2ihZX_ZuV3lJX_4i7q4w-ZBdhJ6o
    # Try multiple approaches
    import subprocess

    # Approach 1: gdown with fuzzy mode
    try:
        url = "https://drive.google.com/uc?id=1ilc5SEYHT-GVpfC1PoGu9pfVDJGOFBfR"
        gdown.download(url, out_path, quiet=False, fuzzy=True)
    except Exception as e:
        print(f"Approach 1 failed: {e}")

    if not os.path.exists(out_path) or os.path.getsize(out_path) < 100_000_000:
        # Approach 2: Try folder download
        try:
            print("Trying folder download...")
            folder_url = "https://drive.google.com/drive/folders/1HlLr2ihZX_ZuV3lJX_4i7q4w-ZBdhJ6o"
            gdown.download_folder(folder_url, output="/data/transnas_dl/", quiet=False)
            for root, dirs, files in os.walk("/data/transnas_dl"):
                for f in files:
                    if f.endswith(".pth"):
                        os.rename(os.path.join(root, f), out_path)
                        print(f"Found and moved: {f}")
                        break
        except Exception as e:
            print(f"Approach 2 failed: {e}")

    if not os.path.exists(out_path) or os.path.getsize(out_path) < 100_000_000:
        # Approach 3: gdown with cookies/different method
        try:
            print("Trying gdown with confirm=True...")
            subprocess.run([
                "gdown", "--id", "1ilc5SEYHT-GVpfC1PoGu9pfVDJGOFBfR",
                "-O", out_path, "--fuzzy"
            ], check=True)
        except Exception as e:
            print(f"Approach 3 failed: {e}")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 100_000_000:
        print(f"Downloaded successfully: {os.path.getsize(out_path) / 1e6:.1f} MB")
    else:
        raise RuntimeError(
            "Failed to download TransNAS-Bench-101. The Google Drive file may have "
            "access restrictions. Please download manually from "
            "https://drive.google.com/drive/folders/1HlLr2ihZX_ZuV3lJX_4i7q4w-ZBdhJ6o "
            "and upload to the Modal volume 'transnas-data' at /transnas-bench_v10141024.pth"
        )

    volume.commit()
    return out_path


# ---------------------------------------------------------------------------
# Step 2: Evaluate all architectures
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="A10G",  # Fast GPU for quicker eval of 7352 architectures
    timeout=7200,  # 2 hours should be plenty for 7352 small networks
    memory=32768,
)
def evaluate_all_architectures():
    """
    For each TransNAS-Bench-101 architecture:
      1. Build the network from its encoding string
      2. Compute our proxies (MSFS, SFC + components) and baselines
      3. Retrieve trained ground truth from the API
      4. Save everything to JSON
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
    import traceback
    from collections import OrderedDict

    # Import TransNAS-Bench API
    from api import TransNASBenchAPI as API

    # Import network construction
    from models.net_infer.net_macro import MacroNet

    volume.reload()
    bench_path = "/data/transnas-bench_v10141024.pth"
    if not os.path.exists(bench_path):
        raise FileNotFoundError(f"Benchmark file not found at {bench_path}. Run download_benchmark() first.")

    print(f"Loading TransNAS-Bench-101 API from {bench_path}...")
    api = API(bench_path, verbose=False)
    print(f"Loaded: {len(api)} architectures, {len(api.task_list)} tasks")
    print(f"Tasks: {api.task_list}")
    print(f"Search spaces: {api.search_spaces}")
    for ss in api.search_spaces:
        print(f"  {ss}: {len(api.all_arch_dict[ss])} architectures")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    SEED = 42
    INPUT_SIZE = (4, 3, 64, 64)  # Smaller input for speed (7352 archs)

    # ================================================================
    # Proxy implementations (adapted for TransNAS-Bench architectures)
    # ================================================================

    def count_params(m):
        return sum(p.numel() for p in m.parameters())

    # --- Multi-scale feature extraction ---
    def extract_multiscale_features(model, x):
        """
        Extract intermediate features from MacroNet layers.
        Returns list of feature tensors at different scales.
        """
        feats = []
        x = model.stem(x)
        feats.append(x)
        for layer_name in model.layers:
            layer = getattr(model, layer_name)
            x = layer(x)
            feats.append(x)
        return feats

    # --- SFC components ---
    def sfc_component_nc(feats):
        val, cnt = 0.0, 0
        for f in feats:
            B, C, H, W = f.shape
            ps = 4
            if H < ps * 2 or W < ps * 2:
                continue
            ph, pw = H // ps, W // ps
            fc = f[:, :, :ph*ps, :pw*ps].reshape(B, C, ph, ps, pw, ps).mean(dim=(3, 5))
            if pw > 1:
                val += F.cosine_similarity(
                    fc[:, :, :, :-1].reshape(B, C, -1),
                    fc[:, :, :, 1:].reshape(B, C, -1), dim=1
                ).mean().item()
                cnt += 1
            if ph > 1:
                val += F.cosine_similarity(
                    fc[:, :, :-1, :].reshape(B, C, -1),
                    fc[:, :, 1:, :].reshape(B, C, -1), dim=1
                ).mean().item()
                cnt += 1
        return val / max(cnt, 1)

    def sfc_component_bs(feats):
        val, cnt = 0.0, 0
        for f in feats:
            if f.size(2) < 3 or f.size(3) < 3:
                continue
            gm = ((f[:, :, :, 1:] - f[:, :, :, :-1]).abs().mean() +
                  (f[:, :, 1:, :] - f[:, :, :-1, :]).abs().mean()) / 2
            val += (gm / (f.abs().mean() + 1e-8)).item()
            cnt += 1
        return val / max(cnt, 1)

    def sfc_component_sd(feats):
        sd_vals = [f.var(dim=(2, 3)).mean().item() for f in feats]
        return np.mean([math.log(v + 1e-10) for v in sd_vals])

    def sfc_proxy(feats, alpha=0.4, beta=0.35, gamma=0.25):
        nc = sfc_component_nc(feats)
        bs = sfc_component_bs(feats)
        sd = sfc_component_sd(feats)
        sd_norm = (sd + 10) / 10
        return alpha * nc + beta * bs + gamma * sd_norm

    # --- MSFS components ---
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

    # --- Baseline proxies ---
    def naswot_proxy(model, x):
        model.eval()
        acts = []
        hooks = []
        for mod in model.modules():
            if isinstance(mod, nn.ReLU):
                hooks.append(mod.register_forward_hook(
                    lambda _m, _i, o: acts.append((o > 0).float().view(o.size(0), -1))
                ))
        with torch.no_grad():
            model(x)
        for h in hooks:
            h.remove()
        if not acts:
            return 0.0
        c = torch.cat(acts, 1)
        K = (c @ c.t()) / c.size(1) + 1e-5 * torch.eye(c.size(0), device=c.device)
        try:
            return float(np.log(np.abs(np.linalg.det(K.cpu().numpy())) + 1e-10))
        except:
            return 0.0

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
    # Main evaluation loop
    # ================================================================
    results = {}
    errors = []
    total = len(api)

    # We'll process MACRO space architectures (3,256 archs with 'basic' cells)
    # These are the ones we can reliably construct with MacroNet
    # Micro space uses custom cell structures that also work with MacroNet

    all_archs = []
    for ss in api.search_spaces:
        for arch_str in api.all_arch_dict[ss]:
            all_archs.append((ss, arch_str))

    print(f"\nEvaluating {len(all_archs)} architectures...")
    t_start = time.time()

    for idx, (ss, arch_str) in enumerate(all_archs):
        if idx % 500 == 0:
            elapsed = time.time() - t_start
            rate = idx / max(elapsed, 1)
            eta = (len(all_archs) - idx) / max(rate, 0.01)
            print(f"[{idx}/{len(all_archs)}] {elapsed:.0f}s elapsed, {eta:.0f}s remaining, {len(errors)} errors")

        try:
            # Build network
            torch.manual_seed(SEED)
            # Use backbone mode (no classification head)
            model = MacroNet(arch_str, structure='backbone', input_dim=(64, 64))
            model = model.to(device)
            model.eval()

            # Count params
            params = count_params(model)

            # Generate random input
            torch.manual_seed(SEED)
            x = torch.randn(*INPUT_SIZE, device=device)

            # Extract multi-scale features
            with torch.no_grad():
                feats = extract_multiscale_features(model, x)

            # Compute our proxies
            sfc_val = sfc_proxy(feats)
            sfc_nc = sfc_component_nc(feats)
            sfc_bs = sfc_component_bs(feats)
            sfc_sd = sfc_component_sd(feats)

            msfs_val = msfs_proxy(feats)
            msfs_isd = msfs_component_isd(feats)
            msfs_sa = msfs_component_sa(feats)

            # Compute baseline proxies (need fresh model instances for some)
            torch.manual_seed(SEED)
            x_nw = torch.randn(*INPUT_SIZE, device=device)
            nw = naswot_proxy(model, x_nw)

            # GradNorm needs fresh model
            torch.manual_seed(SEED)
            model_gn = MacroNet(arch_str, structure='backbone', input_dim=(64, 64)).to(device)
            x_gn = torch.randn(*INPUT_SIZE, device=device)
            gn = gradnorm_proxy(model_gn, x_gn)
            del model_gn

            # SynFlow needs fresh model
            torch.manual_seed(SEED)
            model_sf = MacroNet(arch_str, structure='backbone', input_dim=(64, 64)).to(device)
            x_sf = torch.randn(*INPUT_SIZE, device=device)
            sf = synflow_proxy(model_sf, x_sf)
            del model_sf

            # Get ground truth from API
            gt = {}
            for task in api.task_list:
                task_gt = {}
                # Get model info
                try:
                    task_gt['encoder_params'] = api.get_model_info(arch_str, task, 'encoder_params')
                    task_gt['encoder_flops'] = api.get_model_info(arch_str, task, 'encoder_FLOPs')
                except:
                    pass

                # Get performance metrics
                for metric in api.metrics_dict[task]:
                    try:
                        task_gt[f'{metric}_best'] = api.get_single_metric(
                            arch_str, task, metric, mode='best'
                        )
                    except:
                        pass
                    try:
                        task_gt[f'{metric}_final'] = api.get_single_metric(
                            arch_str, task, metric, mode='final'
                        )
                    except:
                        pass
                gt[task] = task_gt

            results[arch_str] = {
                'search_space': ss,
                'params': params,
                'naswot': nw,
                'synflow': sf,
                'gradnorm': gn,
                'sfc': sfc_val,
                'sfc_nc': sfc_nc,
                'sfc_bs': sfc_bs,
                'sfc_sd': sfc_sd,
                'msfs': msfs_val,
                'msfs_isd': msfs_isd,
                'msfs_sa': msfs_sa,
                'num_features': len(feats),
                'feature_shapes': [list(f.shape) for f in feats],
                'ground_truth': gt,
            }

            # Clean up GPU memory
            del model, feats, x
            if idx % 100 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            errors.append({
                'arch': arch_str,
                'search_space': ss,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            if len(errors) <= 5:
                print(f"  ERROR on {arch_str}: {e}")

    elapsed = time.time() - t_start
    print(f"\nDone! Evaluated {len(results)} architectures in {elapsed:.0f}s ({len(errors)} errors)")

    # Save results
    output = {
        'meta': {
            'total_architectures': len(all_archs),
            'successful': len(results),
            'errors': len(errors),
            'elapsed_seconds': elapsed,
            'input_size': list(INPUT_SIZE),
            'seed': SEED,
            'tasks': api.task_list,
            'search_spaces': api.search_spaces,
        },
        'results': results,
        'errors': errors[:50],  # Save first 50 errors for debugging
    }

    out_path = "/data/transnas_proxy_results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f)
    print(f"Results saved to {out_path}")

    volume.commit()
    return f"Evaluated {len(results)}/{len(all_archs)} architectures in {elapsed:.0f}s"


# ---------------------------------------------------------------------------
# Step 3: Download results
# ---------------------------------------------------------------------------
@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={"/data": volume},
    timeout=300,
)
def get_results():
    """Download the results JSON."""
    volume.reload()
    path = "/data/transnas_proxy_results.json"
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Local entrypoint for quick testing
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    print("Step 1: Downloading TransNAS-Bench-101 benchmark file...")
    result = download_benchmark.remote()
    print(f"Download result: {result}")

    print("\nStep 2: Evaluating all architectures...")
    result = evaluate_all_architectures.remote()
    print(f"Evaluation result: {result}")

    print("\nStep 3: Downloading results...")
    data = get_results.remote()
    if data:
        print(f"Got results for {data['meta']['successful']} architectures")
        # Save locally
        with open("results/transnas_results.json", 'w') as f:
            json.dump(data, f)
        print("Saved to results/transnas_results.json")
    else:
        print("No results found!")
