"""
RepNAS Seed Stability Experiment
=================================
Re-runs CKA computation with 5 different seeds (different noise probes)
for DINOv2 teacher to measure ranking stability across probe sets.
Also tests different probe set SIZES (128, 256, 512, 1024) for the primary seed.
"""

import modal
import json

app = modal.App("repnas-seeds")
volume = modal.Volume.from_name("repnas-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch", "torchvision", "timm", "scipy", "numpy", "scikit-learn")
)

# Same candidate pool as v3
CANDIDATES = {
    "resnet18":           {"timm": "resnet18.a1_in1k",              "acc": 71.5, "family": "CNN", "params": 11.7},
    "resnet34":           {"timm": "resnet34.a1_in1k",              "acc": 75.1, "family": "CNN", "params": 21.8},
    "resnet50":           {"timm": "resnet50.a1_in1k",              "acc": 79.8, "family": "CNN", "params": 25.6},
    "resnet101":          {"timm": "resnet101.a1_in1k",             "acc": 81.5, "family": "CNN", "params": 44.5},
    "resnet152":          {"timm": "resnet152.a1_in1k",             "acc": 82.0, "family": "CNN", "params": 60.2},
    "resnext50_32x4d":    {"timm": "resnext50_32x4d.a1_in1k",      "acc": 80.1, "family": "CNN", "params": 25.0},
    "resnext101_32x8d":   {"timm": "resnext101_32x8d.fb_wsl_ig1b_ft_in1k", "acc": 82.2, "family": "CNN", "params": 88.8},
    "wide_resnet50_2":    {"timm": "wide_resnet50_2.tv2_in1k",     "acc": 81.6, "family": "CNN", "params": 68.9},
    "wide_resnet101_2":   {"timm": "wide_resnet101_2.tv2_in1k",    "acc": 82.5, "family": "CNN", "params": 126.9},
    "densenet121":        {"timm": "densenet121.ra_in1k",           "acc": 75.6, "family": "CNN", "params": 8.0},
    "densenet169":        {"timm": "densenet169.tv_in1k",           "acc": 76.2, "family": "CNN", "params": 14.1},
    "densenet201":        {"timm": "densenet201.tv_in1k",           "acc": 77.4, "family": "CNN", "params": 20.0},
    "efficientnet_b0":    {"timm": "efficientnet_b0.ra_in1k",      "acc": 77.7, "family": "EffNet", "params": 5.3},
    "efficientnet_b1":    {"timm": "efficientnet_b1.ft_in1k",      "acc": 79.2, "family": "EffNet", "params": 7.8},
    "efficientnet_b2":    {"timm": "efficientnet_b2.ra_in1k",      "acc": 80.6, "family": "EffNet", "params": 9.1},
    "efficientnet_b3":    {"timm": "efficientnet_b3.ra2_in1k",     "acc": 82.0, "family": "EffNet", "params": 12.2},
    "efficientnet_b4":    {"timm": "efficientnet_b4.ra2_in1k",     "acc": 83.4, "family": "EffNet", "params": 19.3},
    "efficientnetv2_s":   {"timm": "tf_efficientnetv2_s.in1k",     "acc": 83.9, "family": "EffNet", "params": 21.5},
    "efficientnetv2_m":   {"timm": "tf_efficientnetv2_m.in1k",     "acc": 85.1, "family": "EffNet", "params": 54.1},
    "mobilenetv2_100":    {"timm": "mobilenetv2_100.ra_in1k",      "acc": 72.9, "family": "Mobile", "params": 3.5},
    "mobilenetv2_140":    {"timm": "mobilenetv2_140.ra_in1k",      "acc": 76.5, "family": "Mobile", "params": 6.1},
    "mobilenetv3_large":  {"timm": "mobilenetv3_large_100.ra_in1k","acc": 75.8, "family": "Mobile", "params": 5.5},
    "mobilenetv3_small":  {"timm": "mobilenetv3_small_100.lamb_in1k","acc": 67.7, "family": "Mobile", "params": 2.5},
    "mnasnet_100":        {"timm": "mnasnet_100.rmsp_in1k",        "acc": 74.7, "family": "Mobile", "params": 4.4},
    "convnext_tiny":      {"timm": "convnext_tiny.fb_in1k",        "acc": 82.1, "family": "ConvNeXt", "params": 28.6},
    "convnext_small":     {"timm": "convnext_small.fb_in1k",       "acc": 83.1, "family": "ConvNeXt", "params": 50.2},
    "convnext_base":      {"timm": "convnext_base.fb_in1k",        "acc": 83.8, "family": "ConvNeXt", "params": 88.6},
    "convnext_large":     {"timm": "convnext_large.fb_in1k",       "acc": 84.3, "family": "ConvNeXt", "params": 197.8},
    "convnextv2_tiny":    {"timm": "convnextv2_tiny.fcmae_ft_in1k","acc": 82.9, "family": "ConvNeXt", "params": 28.6},
    "convnextv2_base":    {"timm": "convnextv2_base.fcmae_ft_in1k","acc": 84.9, "family": "ConvNeXt", "params": 88.7},
    "vit_tiny_patch16":   {"timm": "vit_tiny_patch16_224.augreg_in21k_ft_in1k", "acc": 75.5, "family": "ViT", "params": 5.7},
    "vit_small_patch16":  {"timm": "vit_small_patch16_224.augreg_in21k_ft_in1k","acc": 81.4, "family": "ViT", "params": 22.1},
    "vit_base_patch16":   {"timm": "vit_base_patch16_224.augreg_in21k_ft_in1k", "acc": 84.0, "family": "ViT", "params": 86.6},
    "vit_base_patch32":   {"timm": "vit_base_patch32_224.augreg_in21k_ft_in1k", "acc": 80.7, "family": "ViT", "params": 88.2},
    "vit_large_patch16":  {"timm": "vit_large_patch16_224.augreg_in21k_ft_in1k","acc": 85.8, "family": "ViT", "params": 304.3},
    "vit_small_patch16_dino": {"timm": "vit_small_patch16_224.dino", "acc": 78.0, "family": "ViT", "params": 22.1},
    "swin_tiny":          {"timm": "swin_tiny_patch4_window7_224.ms_in1k",  "acc": 81.2, "family": "Swin", "params": 28.3},
    "swin_small":         {"timm": "swin_small_patch4_window7_224.ms_in1k", "acc": 83.2, "family": "Swin", "params": 49.6},
    "swin_base":          {"timm": "swin_base_patch4_window7_224.ms_in1k",  "acc": 83.5, "family": "Swin", "params": 87.8},
    "deit_tiny":          {"timm": "deit_tiny_patch16_224.fb_in1k", "acc": 72.2, "family": "DeiT", "params": 5.7},
    "deit_small":         {"timm": "deit_small_patch16_224.fb_in1k","acc": 79.9, "family": "DeiT", "params": 22.1},
    "deit_base":          {"timm": "deit_base_patch16_224.fb_in1k", "acc": 81.8, "family": "DeiT", "params": 86.6},
    "deit3_small":        {"timm": "deit3_small_patch16_224.fb_in1k","acc": 81.4, "family": "DeiT", "params": 22.1},
    "deit3_base":         {"timm": "deit3_base_patch16_224.fb_in1k","acc": 83.8, "family": "DeiT", "params": 86.6},
    "deit3_large":        {"timm": "deit3_large_patch16_224.fb_in1k","acc": 84.9, "family": "DeiT", "params": 304.4},
    "regnetx_016":        {"timm": "regnetx_016.tv2_in1k",         "acc": 73.0, "family": "RegNet", "params": 9.2},
    "regnetx_032":        {"timm": "regnetx_032.tv2_in1k",         "acc": 75.2, "family": "RegNet", "params": 15.3},
    "regnetx_064":        {"timm": "regnetx_064.pycls_in1k",       "acc": 76.4, "family": "RegNet", "params": 26.2},
    "regnetx_160":        {"timm": "regnetx_160.tv2_in1k",         "acc": 79.7, "family": "RegNet", "params": 54.3},
    "regnety_016":        {"timm": "regnety_016.tv2_in1k",         "acc": 74.0, "family": "RegNet", "params": 11.2},
    "regnety_032":        {"timm": "regnety_032.tv2_in1k",         "acc": 76.6, "family": "RegNet", "params": 19.4},
    "regnety_064":        {"timm": "regnety_064.pycls_in1k",       "acc": 77.2, "family": "RegNet", "params": 30.6},
    "regnety_160":        {"timm": "regnety_160.tv2_in1k",         "acc": 80.4, "family": "RegNet", "params": 83.6},
    "maxvit_tiny":        {"timm": "maxvit_tiny_tf_224.in1k",      "acc": 83.4, "family": "MaxViT", "params": 30.9},
    "maxvit_small":       {"timm": "maxvit_small_tf_224.in1k",     "acc": 84.5, "family": "MaxViT", "params": 68.9},
    "coatnet_0":          {"timm": "coatnet_0_rw_224.sw_in1k",     "acc": 82.4, "family": "MaxViT", "params": 27.4},
    "efficientformer_l1": {"timm": "efficientformer_l1.snap_dist_in1k", "acc": 80.2, "family": "EFormer", "params": 12.3},
    "efficientformer_l3": {"timm": "efficientformer_l3.snap_dist_in1k", "acc": 82.4, "family": "EFormer", "params": 31.4},
    "efficientformer_l7": {"timm": "efficientformer_l7.snap_dist_in1k", "acc": 83.3, "family": "EFormer", "params": 82.2},
    "edgenext_small":     {"timm": "edgenext_small.usi_in1k",      "acc": 81.1, "family": "EdgeNeXt", "params": 5.6},
    "edgenext_base":      {"timm": "edgenext_base.usi_in1k",       "acc": 83.3, "family": "EdgeNeXt", "params": 18.5},
    "inception_v3":       {"timm": "inception_v3.tv_in1k",         "acc": 77.3, "family": "CNN", "params": 23.8},
    "inception_v4":       {"timm": "inception_v4.tf_in1k",         "acc": 80.2, "family": "CNN", "params": 42.7},
    "dpn68":              {"timm": "dpn68.mx_in1k",                "acc": 76.3, "family": "CNN", "params": 12.6},
    "dpn92":              {"timm": "dpn92.mx_in1k",                "acc": 79.5, "family": "CNN", "params": 37.7},
    "dpn131":             {"timm": "dpn131.mx_in1k",               "acc": 79.8, "family": "CNN", "params": 79.3},
    "nasnetalarge":       {"timm": "nasnetalarge.tf_in1k",         "acc": 82.6, "family": "NAS", "params": 88.8},
    "senet154":           {"timm": "senet154.gluon_in1k",          "acc": 81.3, "family": "CNN", "params": 115.1},
    "seresnext50_32x4d":  {"timm": "seresnext50_32x4d.gluon_in1k", "acc": 79.9, "family": "CNN", "params": 27.6},
    "res2net50_26w_4s":   {"timm": "res2net50_26w_4s.in1k",        "acc": 78.0, "family": "CNN", "params": 25.7},
    "res2net101_26w_4s":  {"timm": "res2net101_26w_4s.in1k",       "acc": 79.2, "family": "CNN", "params": 45.2},
    "hrnet_w18":          {"timm": "hrnet_w18.ms_aug_in1k",        "acc": 78.0, "family": "CNN", "params": 21.3},
    "hrnet_w32":          {"timm": "hrnet_w32.ms_in1k",            "acc": 78.4, "family": "CNN", "params": 41.2},
    "hrnet_w48":          {"timm": "hrnet_w48.ms_in1k",            "acc": 79.3, "family": "CNN", "params": 77.5},
    "ghostnet_100":       {"timm": "ghostnet_100.in1k",            "acc": 74.0, "family": "Mobile", "params": 5.2},
    "poolformer_s12":     {"timm": "poolformer_s12.sail_in1k",     "acc": 77.2, "family": "MetaFormer", "params": 11.9},
    "poolformer_s24":     {"timm": "poolformer_s24.sail_in1k",     "acc": 80.3, "family": "MetaFormer", "params": 21.4},
    "poolformer_s36":     {"timm": "poolformer_s36.sail_in1k",     "acc": 81.4, "family": "MetaFormer", "params": 30.9},
    "poolformer_m36":     {"timm": "poolformer_m36.sail_in1k",     "acc": 82.1, "family": "MetaFormer", "params": 56.2},
    "cait_s24_224":       {"timm": "cait_s24_224.fb_dist_in1k",    "acc": 83.5, "family": "CaiT", "params": 47.0},
    "mixer_b16":          {"timm": "mixer_b16_224.goog_in21k_ft_in1k","acc": 78.5, "family": "Mixer", "params": 59.9},
    "vgg16_bn":           {"timm": "vgg16_bn.tv_in1k",             "acc": 73.4, "family": "CNN", "params": 138.4},
    "vgg19_bn":           {"timm": "vgg19_bn.tv_in1k",             "acc": 74.2, "family": "CNN", "params": 143.7},
}


@app.function(
    gpu="A100",
    image=image,
    volumes={"/results": volume},
    timeout=7200,
    memory=32768,
)
def run_seed_stability():
    """Run CKA with multiple seeds and probe sizes for ranking stability."""
    import time
    import random
    import warnings
    warnings.filterwarnings('ignore')

    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy import stats
    import timm

    DEVICE = "cuda"
    CAND_IMG_SIZE = 224
    TEACHER_IMG_SIZE = 518  # DINOv2 ViT-S/14 needs 518px
    BATCH_SIZE = 32
    SEEDS = [42, 123, 456, 789, 2024]
    PROBE_SIZES = [64, 128, 256, 512, 1024]

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def get_features(model, images):
        model.eval()
        model = model.to(DEVICE)
        feats = []
        with torch.no_grad():
            for i in range(0, len(images), BATCH_SIZE):
                batch = images[i:i+BATCH_SIZE].to(DEVICE)
                f = model.forward_features(batch)
                if isinstance(f, (list, tuple)):
                    f = f[-1]
                if f.dim() == 3:
                    f = f[:, 0]
                elif f.dim() == 4:
                    f = F.adaptive_avg_pool2d(f, 1).flatten(1)
                feats.append(f.cpu().float())
        model.cpu()
        torch.cuda.empty_cache()
        return torch.cat(feats, 0).numpy()

    def linear_cka(X, Y):
        X = X - X.mean(0, keepdims=True)
        Y = Y - Y.mean(0, keepdims=True)
        XtX = X @ X.T
        YtY = Y @ Y.T
        hsic_xy = np.sum(XtX * YtY)
        hsic_xx = np.sum(XtX * XtX)
        hsic_yy = np.sum(YtY * YtY)
        return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10))

    def generate_noise_probes(n, seed, img_size):
        set_seed(seed)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        images = torch.randn(n, 3, img_size, img_size)
        images = (images * std + mean).clamp(0, 1)
        return images

    t0 = time.time()
    print("=" * 80)
    print("RepNAS Seed Stability Experiment")
    print(f"Device: {DEVICE}, GPU: {torch.cuda.get_device_name(0)}")
    print(f"Seeds: {SEEDS}, Probe sizes: {PROBE_SIZES}")
    print(f"Candidates: {len(CANDIDATES)}")
    print("=" * 80)

    # Load DINOv2 teacher once
    print("\nLoading DINOv2 teacher...")
    teacher_model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)

    all_results = {}

    # ---- Part 1: Different seeds, fixed size (512) ----
    print("\n--- PART 1: Seed Stability (5 seeds, N=512) ---")
    for seed in SEEDS:
        print(f"\n  [Seed {seed}]")
        teacher_probes = generate_noise_probes(512, seed, TEACHER_IMG_SIZE)
        cand_probes = generate_noise_probes(512, seed, CAND_IMG_SIZE)
        teacher_feats = get_features(teacher_model, teacher_probes)
        print(f"    Teacher features: {teacher_feats.shape}")
        del teacher_probes
        torch.cuda.empty_cache()

        seed_results = {}
        for idx, (name, info) in enumerate(CANDIDATES.items()):
            try:
                m = timm.create_model(info["timm"], pretrained=True, num_classes=0)
                cand_feats = get_features(m, cand_probes)
                del m
                torch.cuda.empty_cache()
                cka = linear_cka(teacher_feats, cand_feats)
                seed_results[name] = {"cka": cka, "acc": info["acc"]}
            except Exception as e:
                seed_results[name] = {"cka": float('nan'), "acc": info["acc"]}

            if (idx + 1) % 20 == 0:
                print(f"    {idx+1}/{len(CANDIDATES)} done")

        del cand_probes
        torch.cuda.empty_cache()
        all_results[f"seed_{seed}_n512"] = seed_results

        # Compute rho for this seed
        valid = [(v["cka"], v["acc"]) for v in seed_results.values()
                 if not np.isnan(v["cka"])]
        if valid:
            ckas, accs = zip(*valid)
            rho, p = stats.spearmanr(ckas, accs)
            print(f"    ρ = {rho:.4f} (p = {p:.6f}, n = {len(valid)})")

    # ---- Part 2: Different probe sizes, fixed seed (42) ----
    print("\n--- PART 2: Probe Size Sensitivity (seed=42) ---")
    for n_probes in PROBE_SIZES:
        print(f"\n  [N={n_probes}]")
        teacher_probes = generate_noise_probes(n_probes, 42, TEACHER_IMG_SIZE)
        cand_probes = generate_noise_probes(n_probes, 42, CAND_IMG_SIZE)
        teacher_feats = get_features(teacher_model, teacher_probes)
        del teacher_probes
        torch.cuda.empty_cache()

        size_results = {}
        for idx, (name, info) in enumerate(CANDIDATES.items()):
            try:
                m = timm.create_model(info["timm"], pretrained=True, num_classes=0)
                cand_feats = get_features(m, cand_probes)
                del m
                torch.cuda.empty_cache()
                cka = linear_cka(teacher_feats, cand_feats)
                size_results[name] = {"cka": cka, "acc": info["acc"]}
            except:
                size_results[name] = {"cka": float('nan'), "acc": info["acc"]}

            if (idx + 1) % 20 == 0:
                print(f"    {idx+1}/{len(CANDIDATES)} done")

        del cand_probes
        torch.cuda.empty_cache()
        all_results[f"seed_42_n{n_probes}"] = size_results

        valid = [(v["cka"], v["acc"]) for v in size_results.values()
                 if not np.isnan(v["cka"])]
        if valid:
            ckas, accs = zip(*valid)
            rho, p = stats.spearmanr(ckas, accs)
            print(f"    ρ = {rho:.4f} (p = {p:.6f}, n = {len(valid)})")

    # ---- Part 3: Compute pairwise rank correlations between seeds ----
    print("\n--- PART 3: Pairwise Kendall τ between seed rankings ---")
    seed_rankings = {}
    for seed in SEEDS:
        key = f"seed_{seed}_n512"
        ranks = {}
        for name, val in all_results[key].items():
            if not np.isnan(val["cka"]):
                ranks[name] = val["cka"]
        seed_rankings[seed] = ranks

    pairwise_tau = {}
    for i, s1 in enumerate(SEEDS):
        for s2 in SEEDS[i+1:]:
            common = set(seed_rankings[s1].keys()) & set(seed_rankings[s2].keys())
            v1 = [seed_rankings[s1][k] for k in common]
            v2 = [seed_rankings[s2][k] for k in common]
            tau, p = stats.kendalltau(v1, v2)
            pairwise_tau[f"{s1}_vs_{s2}"] = {"tau": float(str(tau)), "p": float(str(p)), "n": len(common)}
            print(f"  Seed {s1} vs {s2}: τ = {tau:.4f} (p = {p:.6f})")

    del teacher_model
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"COMPLETE in {elapsed/60:.1f} minutes")

    output = {
        "results": {},
        "pairwise_tau": pairwise_tau,
        "config": {
            "seeds": SEEDS,
            "probe_sizes": PROBE_SIZES,
            "num_candidates": len(CANDIDATES),
            "teacher": "dinov2_small",
            "elapsed_seconds": elapsed,
        },
    }
    # Convert results for JSON serialization
    for key, val in all_results.items():
        output["results"][key] = val

    with open("/results/repnas_seed_stability.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    volume.commit()
    print("Results saved to /results/repnas_seed_stability.json")
    return {"status": "complete", "elapsed_minutes": elapsed/60}
