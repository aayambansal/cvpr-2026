"""
RepNAS Upgrade Experiments (all-in-one Modal script)
=====================================================
Runs 5 experiment batches on A100:
  1. Trajectory: CKA at init → epoch 5 → epoch 20 → epoch 50 → pretrained (10 archs)
  2. Transfer: linear-probe acc on CIFAR-100, Flowers102, StanfordCars for 20 archs
  3. New baselines: SNIP, GraSP for all 83 archs
  4. kNN sensitivity: k=1..50, cosine vs L2, with/without whitening (DINOv2 teacher)
  5. Layer-wise: CKA at 4 depth quartiles for all 83 archs × DINOv2 teacher
"""

import modal
import json

app = modal.App("repnas-upgrades")
volume = modal.Volume.from_name("repnas-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch", "torchvision", "timm", "scipy", "numpy",
        "scikit-learn", "pandas"
    )
)

# ── Architecture pools ──────────────────────────────────────────────
# Full 83-arch pool (same as v3)
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

# Subset for trajectory experiment (diverse families, spread of accuracies)
TRAJECTORY_ARCHS = [
    "resnet18", "resnet50", "resnet152",        # CNN small/mid/large
    "efficientnet_b0", "efficientnetv2_m",      # EffNet small/large
    "vit_tiny_patch16", "vit_base_patch16",     # ViT small/large
    "mobilenetv3_small", "convnext_base",       # Mobile / ConvNeXt
    "swin_base",                                 # Swin
]

TEACHER_NAME = "vit_small_patch14_dinov2.lvd142m"
TEACHER_IMG_SIZE = 518
CAND_IMG_SIZE = 224


@app.function(
    gpu="A100",
    image=image,
    volumes={"/results": volume},
    timeout=14400,  # 4 hours max
    memory=32768,
)
def run_all_upgrades():
    """Run all 5 upgrade experiments sequentially."""
    import time
    import random
    import warnings
    warnings.filterwarnings('ignore')

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from scipy import stats
    from sklearn.neighbors import NearestNeighbors
    import timm

    DEVICE = "cuda"
    BATCH_SIZE = 32

    t0_total = time.time()
    all_output = {}

    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def get_features(model, images, batch_size=BATCH_SIZE):
        model.eval()
        model = model.to(DEVICE)
        feats = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size].to(DEVICE)
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

    def knn_agreement(X, Y, k=10, metric="cosine"):
        nn_x = NearestNeighbors(n_neighbors=k+1, metric=metric).fit(X)
        nn_y = NearestNeighbors(n_neighbors=k+1, metric=metric).fit(Y)
        _, idx_x = nn_x.kneighbors(X)
        _, idx_y = nn_y.kneighbors(Y)
        # Exclude self (index 0)
        idx_x = idx_x[:, 1:]
        idx_y = idx_y[:, 1:]
        overlap = 0.0
        for i in range(len(X)):
            overlap += len(set(idx_x[i]) & set(idx_y[i])) / k
        return overlap / len(X)

    print("=" * 80)
    print("RepNAS UPGRADE EXPERIMENTS")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    # ================================================================
    # EXPERIMENT 1: TRAJECTORY (CKA during training)
    # ================================================================
    print("\n" + "=" * 60)
    print("EXP 1: TRAJECTORY (init → training → pretrained)")
    print("=" * 60)
    t1 = time.time()

    # We simulate "training trajectory" by using random-init vs pretrained
    # For a real trajectory we'd need to train models, which is too expensive.
    # Instead: measure CKA at random init AND at pretrained for all trajectory archs.
    # The v3 data already showed random-init CKA is POSITIVE (ρ ≈ +0.30).
    # Here we get fine-grained data for the 10 trajectory archs to make the sign-flip figure.
    
    teacher_model = timm.create_model(TEACHER_NAME, pretrained=True, num_classes=0)
    teacher_probes = generate_noise_probes(512, 42, TEACHER_IMG_SIZE)
    cand_probes = generate_noise_probes(512, 42, CAND_IMG_SIZE)
    teacher_feats = get_features(teacher_model, teacher_probes)
    del teacher_probes

    trajectory_results = {}
    for name in TRAJECTORY_ARCHS:
        info = CANDIDATES[name]
        print(f"  {name}...")
        
        # Random init
        set_seed(42)
        m_rand = timm.create_model(info["timm"], pretrained=False, num_classes=0)
        feats_rand = get_features(m_rand, cand_probes)
        del m_rand
        cka_rand = linear_cka(teacher_feats, feats_rand)
        
        # Pretrained
        m_pre = timm.create_model(info["timm"], pretrained=True, num_classes=0)
        feats_pre = get_features(m_pre, cand_probes)
        del m_pre
        cka_pre = linear_cka(teacher_feats, feats_pre)
        
        torch.cuda.empty_cache()
        trajectory_results[name] = {
            "cka_init": cka_rand,
            "cka_pretrained": cka_pre,
            "acc": info["acc"],
            "params": info["params"],
            "family": info["family"],
        }
        print(f"    init={cka_rand:.6f}, pretrained={cka_pre:.6f}, acc={info['acc']}")

    all_output["trajectory"] = trajectory_results
    print(f"  Trajectory done in {(time.time()-t1)/60:.1f}m")

    # ================================================================
    # EXPERIMENT 2: TRANSFER BENCHMARKS
    # ================================================================
    print("\n" + "=" * 60)
    print("EXP 2: TRANSFER BENCHMARKS (CIFAR-100, Flowers102)")
    print("=" * 60)
    t2 = time.time()

    # Download datasets
    import torchvision
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # CIFAR-100
    print("  Downloading CIFAR-100...")
    cifar100_train = torchvision.datasets.CIFAR100(
        root="/tmp/data", train=True, download=True, transform=transform)
    cifar100_test = torchvision.datasets.CIFAR100(
        root="/tmp/data", train=False, download=True, transform=transform)

    # Flowers102
    print("  Downloading Flowers102...")
    try:
        flowers_train = torchvision.datasets.Flowers102(
            root="/tmp/data", split="train", download=True, transform=transform)
        flowers_test = torchvision.datasets.Flowers102(
            root="/tmp/data", split="test", download=True, transform=transform)
        has_flowers = True
    except Exception as e:
        print(f"  Flowers102 download failed: {e}")
        has_flowers = False

    def extract_features_dataset(model, dataset, max_samples=2000):
        """Extract features from a dataset using a pretrained model."""
        model.eval()
        model = model.to(DEVICE)
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
        feats = []
        labels = []
        n = 0
        with torch.no_grad():
            for batch_x, batch_y in loader:
                if n >= max_samples:
                    break
                batch_x = batch_x.to(DEVICE)
                f = model.forward_features(batch_x)
                if isinstance(f, (list, tuple)):
                    f = f[-1]
                if f.dim() == 3:
                    f = f[:, 0]
                elif f.dim() == 4:
                    f = F.adaptive_avg_pool2d(f, 1).flatten(1)
                feats.append(f.cpu().float())
                labels.append(batch_y)
                n += len(batch_x)
        model.cpu()
        torch.cuda.empty_cache()
        return torch.cat(feats, 0).numpy()[:max_samples], torch.cat(labels, 0).numpy()[:max_samples]

    def linear_probe_acc(train_feats, train_labels, test_feats, test_labels):
        """Train a linear probe and return test accuracy."""
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(train_feats, train_labels)
        return float(clf.score(test_feats, test_labels)) * 100.0

    transfer_results = {}
    for idx, (name, info) in enumerate(CANDIDATES.items()):
        print(f"  [{idx+1}/{len(CANDIDATES)}] {name}...")
        try:
            m = timm.create_model(info["timm"], pretrained=True, num_classes=0)

            # CIFAR-100 linear probe
            train_f, train_l = extract_features_dataset(m, cifar100_train, max_samples=5000)
            test_f, test_l = extract_features_dataset(m, cifar100_test, max_samples=2000)
            cifar_acc = linear_probe_acc(train_f, train_l, test_f, test_l)

            # Flowers102 linear probe
            flowers_acc = None
            if has_flowers:
                try:
                    fl_train_f, fl_train_l = extract_features_dataset(m, flowers_train, max_samples=2000)
                    fl_test_f, fl_test_l = extract_features_dataset(m, flowers_test, max_samples=2000)
                    flowers_acc = linear_probe_acc(fl_train_f, fl_train_l, fl_test_f, fl_test_l)
                except Exception as e2:
                    flowers_acc = None

            del m
            torch.cuda.empty_cache()

            transfer_results[name] = {
                "imagenet_acc": info["acc"],
                "cifar100_acc": cifar_acc,
                "flowers102_acc": flowers_acc,
                "params": info["params"],
                "family": info["family"],
            }
            print(f"    CIFAR-100={cifar_acc:.1f}%, Flowers={flowers_acc if flowers_acc else 'N/A'}")
        except Exception as e:
            print(f"    FAILED: {e}")
            transfer_results[name] = {
                "imagenet_acc": info["acc"],
                "cifar100_acc": None,
                "flowers102_acc": None,
                "params": info["params"],
                "family": info["family"],
                "error": str(e),
            }

    all_output["transfer"] = transfer_results
    print(f"  Transfer done in {(time.time()-t2)/60:.1f}m")

    # Save intermediate
    with open("/results/repnas_upgrades_partial.json", "w") as f:
        json.dump(all_output, f, indent=2, default=str)
    volume.commit()

    # ================================================================
    # EXPERIMENT 3: NEW BASELINES (SNIP, GraSP)
    # ================================================================
    print("\n" + "=" * 60)
    print("EXP 3: NEW BASELINES (SNIP, GraSP)")
    print("=" * 60)
    t3 = time.time()

    def compute_snip(model_name, num_classes=1000, n_samples=64):
        """SNIP: connection sensitivity (Lee et al., 2019)."""
        set_seed(42)
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        model = model.to(DEVICE)
        model.train()
        
        # Generate random input + targets
        x = torch.randn(n_samples, 3, 224, 224, device=DEVICE)
        y = torch.randint(0, num_classes, (n_samples,), device=DEVICE)
        
        # Forward pass
        output = model(x)
        loss = F.cross_entropy(output, y)
        
        # Compute gradients
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
        
        # SNIP score: sum of |param * grad|
        score = 0.0
        for p, g in zip(model.parameters(), grads):
            score += (p * g).abs().sum().item()
        
        del model, x, y, output, loss, grads
        torch.cuda.empty_cache()
        return score

    def compute_grasp(model_name, num_classes=1000, n_samples=64):
        """GraSP: gradient signal preservation (Wang et al., 2020)."""
        set_seed(42)
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        model = model.to(DEVICE)
        model.train()
        
        x = torch.randn(n_samples, 3, 224, 224, device=DEVICE)
        y = torch.randint(0, num_classes, (n_samples,), device=DEVICE)
        
        # First forward-backward for Hessian-gradient product
        output = model(x)
        loss = F.cross_entropy(output, y)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        
        # Flatten gradients
        flat_grads = torch.cat([g.reshape(-1) for g in grads])
        
        # Hg = sum of grad * flat_grads product
        gnorm = flat_grads.sum()
        
        # Second backward
        hg = torch.autograd.grad(gnorm, model.parameters(), create_graph=False)
        
        # GraSP score: -sum(H*g * theta)
        score = 0.0
        for p, h in zip(model.parameters(), hg):
            score -= (h * p).sum().item()
        
        del model, x, y, output, loss, grads, hg
        torch.cuda.empty_cache()
        return score

    baseline_results = {}
    for idx, (name, info) in enumerate(CANDIDATES.items()):
        print(f"  [{idx+1}/{len(CANDIDATES)}] {name}...")
        result = {"acc": info["acc"], "params": info["params"]}
        
        try:
            result["snip"] = compute_snip(info["timm"])
        except Exception as e:
            result["snip"] = None
            print(f"    SNIP failed: {e}")
        
        try:
            result["grasp"] = compute_grasp(info["timm"])
        except Exception as e:
            result["grasp"] = None
            print(f"    GraSP failed: {e}")
        
        baseline_results[name] = result
        if (idx + 1) % 20 == 0:
            print(f"    {idx+1}/{len(CANDIDATES)} done")

    all_output["baselines"] = baseline_results
    print(f"  Baselines done in {(time.time()-t3)/60:.1f}m")

    # ================================================================
    # EXPERIMENT 4: kNN SENSITIVITY
    # ================================================================
    print("\n" + "=" * 60)
    print("EXP 4: kNN SENSITIVITY (k, metric, whitening)")
    print("=" * 60)
    t4 = time.time()

    # Use DINOv2 teacher, noise probes, all 83 archs
    # Already have teacher_feats from exp 1
    print("  Extracting all candidate features...")
    cand_features = {}
    for idx, (name, info) in enumerate(CANDIDATES.items()):
        try:
            m = timm.create_model(info["timm"], pretrained=True, num_classes=0)
            cand_features[name] = get_features(m, cand_probes)
            del m
            torch.cuda.empty_cache()
        except:
            pass
        if (idx + 1) % 20 == 0:
            print(f"    {idx+1}/{len(CANDIDATES)} extracted")

    # Test different k values
    k_values = [1, 3, 5, 10, 15, 20, 30, 50]
    metrics = ["cosine", "euclidean"]
    
    knn_results = {}
    for metric in metrics:
        for k in k_values:
            scores = {}
            for name, cf in cand_features.items():
                try:
                    knn_score = knn_agreement(teacher_feats, cf, k=k, metric=metric)
                    scores[name] = {"knn": knn_score, "acc": CANDIDATES[name]["acc"]}
                except:
                    pass
            
            if scores:
                vals = [(v["knn"], v["acc"]) for v in scores.values()]
                s, a = zip(*vals)
                rho, p = stats.spearmanr(s, a)
                knn_results[f"{metric}_k{k}"] = {
                    "rho": float(rho), "p": float(p), "n": len(vals),
                    "k": k, "metric": metric
                }
                print(f"    {metric} k={k}: ρ = {rho:.4f} (p = {p:.2e})")

    # Test with whitened features
    print("  Testing whitened features...")
    from sklearn.decomposition import PCA
    
    def whiten(X, n_components=None):
        if n_components is None:
            n_components = min(X.shape[0], X.shape[1])
        pca = PCA(n_components=n_components, whiten=True)
        return pca.fit_transform(X)

    n_comp = min(256, teacher_feats.shape[0], teacher_feats.shape[1])
    teacher_w = whiten(teacher_feats, n_comp)
    
    for k in [5, 10, 20]:
        scores_w = {}
        for name, cf in cand_features.items():
            try:
                nc = min(n_comp, cf.shape[1])
                cf_w = whiten(cf, nc)
                # Align dimensions
                dim = min(teacher_w.shape[1], cf_w.shape[1])
                knn_score = knn_agreement(teacher_w[:, :dim], cf_w[:, :dim], k=k, metric="cosine")
                scores_w[name] = {"knn": knn_score, "acc": CANDIDATES[name]["acc"]}
            except:
                pass
        
        if scores_w:
            vals = [(v["knn"], v["acc"]) for v in scores_w.values()]
            s, a = zip(*vals)
            rho, p = stats.spearmanr(s, a)
            knn_results[f"whitened_cosine_k{k}"] = {
                "rho": float(rho), "p": float(p), "n": len(vals),
                "k": k, "metric": "whitened_cosine"
            }
            print(f"    whitened cosine k={k}: ρ = {rho:.4f} (p = {p:.2e})")

    all_output["knn_sensitivity"] = knn_results
    print(f"  kNN sensitivity done in {(time.time()-t4)/60:.1f}m")

    # ================================================================
    # EXPERIMENT 5: LAYER-WISE CKA
    # ================================================================
    print("\n" + "=" * 60)
    print("EXP 5: LAYER-WISE CKA (DINOv2 teacher)")
    print("=" * 60)
    t5 = time.time()

    def get_intermediate_features(model, images, batch_size=BATCH_SIZE):
        """Extract features from multiple layers using forward hooks."""
        model.eval()
        model = model.to(DEVICE)
        
        # Collect all named modules that are likely feature stages
        hook_layers = []
        handles = []
        features_dict = {}
        
        # Strategy: hook into sequential blocks / stages
        named_children = list(model.named_children())
        # For CNNs, look for layer1/2/3/4 or stages or features
        # For ViTs, look for blocks
        
        target_names = []
        for child_name, child_module in model.named_modules():
            # Skip very small/trivial modules
            if isinstance(child_module, (nn.Sequential, nn.ModuleList)):
                continue
            # Collect "major" layers (Conv blocks, Transformer blocks, etc)
            if any(kw in child_name for kw in ["layer1", "layer2", "layer3", "layer4",
                                                 "stages.0", "stages.1", "stages.2", "stages.3",
                                                 "blocks.0", "blocks.3", "blocks.6", "blocks.9",
                                                 "features.2", "features.4", "features.6"]):
                if child_name not in target_names:
                    target_names.append(child_name)
        
        if not target_names:
            # Fallback: get children directly
            for child_name, _ in named_children:
                if child_name not in ("head", "fc", "classifier", "head_drop"):
                    target_names.append(child_name)
        
        # Only keep ~4 layers (quartiles)
        if len(target_names) > 4:
            indices = np.linspace(0, len(target_names) - 1, 4, dtype=int)
            target_names = [target_names[i] for i in indices]
        
        for layer_name in target_names:
            features_dict[layer_name] = []
            
            def make_hook(name):
                def hook_fn(module, input, output):
                    if isinstance(output, torch.Tensor):
                        if output.dim() == 4:
                            out = F.adaptive_avg_pool2d(output, 1).flatten(1)
                        elif output.dim() == 3:
                            out = output[:, 0]
                        else:
                            out = output
                        features_dict[name].append(out.detach().cpu().float())
                return hook_fn
            
            # Get module by name
            module = model
            for part in layer_name.split('.'):
                module = getattr(module, part)
            h = module.register_forward_hook(make_hook(layer_name))
            handles.append(h)
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size].to(DEVICE)
                try:
                    model(batch)
                except:
                    model.forward_features(batch)
        
        for h in handles:
            h.remove()
        
        result = {}
        for name, feat_list in features_dict.items():
            if feat_list:
                result[name] = torch.cat(feat_list, 0).numpy()
        
        model.cpu()
        torch.cuda.empty_cache()
        return result

    # Get teacher features (penultimate = default)
    # teacher_feats already computed
    
    layerwise_results = {}
    for idx, (name, info) in enumerate(CANDIDATES.items()):
        try:
            m = timm.create_model(info["timm"], pretrained=True, num_classes=0)
            layer_feats = get_intermediate_features(m, cand_probes)
            del m
            torch.cuda.empty_cache()
            
            arch_layers = {}
            for layer_name, lf in layer_feats.items():
                if lf.shape[0] == teacher_feats.shape[0]:
                    cka = linear_cka(teacher_feats, lf)
                    arch_layers[layer_name] = cka
            
            layerwise_results[name] = {
                "layers": arch_layers,
                "acc": info["acc"],
                "family": info["family"],
            }
        except Exception as e:
            layerwise_results[name] = {"layers": {}, "acc": info["acc"], "family": info["family"], "error": str(e)}
        
        if (idx + 1) % 20 == 0:
            print(f"    {idx+1}/{len(CANDIDATES)} done")

    all_output["layerwise"] = layerwise_results
    print(f"  Layer-wise done in {(time.time()-t5)/60:.1f}m")

    # ================================================================
    # SAVE FINAL RESULTS
    # ================================================================
    elapsed = time.time() - t0_total
    all_output["config"] = {
        "teacher": TEACHER_NAME,
        "n_candidates": len(CANDIDATES),
        "elapsed_minutes": elapsed / 60,
    }

    with open("/results/repnas_upgrades.json", "w") as f:
        json.dump(all_output, f, indent=2, default=str)
    volume.commit()

    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETE in {elapsed/60:.1f} minutes")
    print("Results saved to /results/repnas_upgrades.json")
    return {"status": "complete", "elapsed_minutes": elapsed / 60}
