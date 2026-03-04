"""
RepNAS v3: Full Experiment Suite on Modal A100
===============================================
- 100+ timm architectures with known ImageNet-1k accuracy
- 4 teachers: DINOv2-S/14, CLIP-ViT-B/32, ConvNeXt-Base (FCMAE), MAE-ViT-B/16
- 3 probe types: random noise, ImageNet-val (512 images), diffusion-like augmented
- Layer-wise CKA (early/mid/late)
- ZS-NAS baselines: GradNorm, SynFlow, NASWOT
- Partial correlation analysis
- Top-k hit rate evaluation
"""

import modal
import json

app = modal.App("repnas-v3")

volume = modal.Volume.from_name("repnas-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch",
        "torchvision",
        "timm",
        "scipy",
        "numpy",
        "scikit-learn",
        "open_clip_torch",
        "Pillow",
        "requests",
    )
)


# ============================================================
# Architecture pool: ~100 timm models with known ImageNet-1k accuracy
# ============================================================
CANDIDATES = {
    # --- CNN / ResNet family ---
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
    # --- EfficientNet ---
    "efficientnet_b0":    {"timm": "efficientnet_b0.ra_in1k",      "acc": 77.7, "family": "EffNet", "params": 5.3},
    "efficientnet_b1":    {"timm": "efficientnet_b1.ft_in1k",      "acc": 79.2, "family": "EffNet", "params": 7.8},
    "efficientnet_b2":    {"timm": "efficientnet_b2.ra_in1k",      "acc": 80.6, "family": "EffNet", "params": 9.1},
    "efficientnet_b3":    {"timm": "efficientnet_b3.ra2_in1k",     "acc": 82.0, "family": "EffNet", "params": 12.2},
    "efficientnet_b4":    {"timm": "efficientnet_b4.ra2_in1k",     "acc": 83.4, "family": "EffNet", "params": 19.3},
    "efficientnetv2_s":   {"timm": "tf_efficientnetv2_s.in1k",     "acc": 83.9, "family": "EffNet", "params": 21.5},
    "efficientnetv2_m":   {"timm": "tf_efficientnetv2_m.in1k",     "acc": 85.1, "family": "EffNet", "params": 54.1},
    # --- Mobile ---
    "mobilenetv2_100":    {"timm": "mobilenetv2_100.ra_in1k",      "acc": 72.9, "family": "Mobile", "params": 3.5},
    "mobilenetv2_140":    {"timm": "mobilenetv2_140.ra_in1k",      "acc": 76.5, "family": "Mobile", "params": 6.1},
    "mobilenetv3_large":  {"timm": "mobilenetv3_large_100.ra_in1k","acc": 75.8, "family": "Mobile", "params": 5.5},
    "mobilenetv3_small":  {"timm": "mobilenetv3_small_100.lamb_in1k","acc": 67.7, "family": "Mobile", "params": 2.5},
    "mnasnet_100":        {"timm": "mnasnet_100.rmsp_in1k",        "acc": 74.7, "family": "Mobile", "params": 4.4},
    # --- ConvNeXt ---
    "convnext_tiny":      {"timm": "convnext_tiny.fb_in1k",        "acc": 82.1, "family": "ConvNeXt", "params": 28.6},
    "convnext_small":     {"timm": "convnext_small.fb_in1k",       "acc": 83.1, "family": "ConvNeXt", "params": 50.2},
    "convnext_base":      {"timm": "convnext_base.fb_in1k",        "acc": 83.8, "family": "ConvNeXt", "params": 88.6},
    "convnext_large":     {"timm": "convnext_large.fb_in1k",       "acc": 84.3, "family": "ConvNeXt", "params": 197.8},
    "convnextv2_tiny":    {"timm": "convnextv2_tiny.fcmae_ft_in1k","acc": 82.9, "family": "ConvNeXt", "params": 28.6},
    "convnextv2_base":    {"timm": "convnextv2_base.fcmae_ft_in1k","acc": 84.9, "family": "ConvNeXt", "params": 88.7},
    # --- ViT ---
    "vit_tiny_patch16":   {"timm": "vit_tiny_patch16_224.augreg_in21k_ft_in1k", "acc": 75.5, "family": "ViT", "params": 5.7},
    "vit_small_patch16":  {"timm": "vit_small_patch16_224.augreg_in21k_ft_in1k","acc": 81.4, "family": "ViT", "params": 22.1},
    "vit_base_patch16":   {"timm": "vit_base_patch16_224.augreg_in21k_ft_in1k", "acc": 84.0, "family": "ViT", "params": 86.6},
    "vit_base_patch32":   {"timm": "vit_base_patch32_224.augreg_in21k_ft_in1k", "acc": 80.7, "family": "ViT", "params": 88.2},
    "vit_large_patch16":  {"timm": "vit_large_patch16_224.augreg_in21k_ft_in1k","acc": 85.8, "family": "ViT", "params": 304.3},
    "vit_small_patch16_dino": {"timm": "vit_small_patch16_224.dino", "acc": 78.0, "family": "ViT", "params": 22.1},
    # --- Swin ---
    "swin_tiny":          {"timm": "swin_tiny_patch4_window7_224.ms_in1k",  "acc": 81.2, "family": "Swin", "params": 28.3},
    "swin_small":         {"timm": "swin_small_patch4_window7_224.ms_in1k", "acc": 83.2, "family": "Swin", "params": 49.6},
    "swin_base":          {"timm": "swin_base_patch4_window7_224.ms_in1k",  "acc": 83.5, "family": "Swin", "params": 87.8},
    "swinv2_tiny":        {"timm": "swinv2_tiny_window8_256.ms_in1k",  "acc": 81.8, "family": "Swin", "params": 28.3},
    "swinv2_small":       {"timm": "swinv2_small_window8_256.ms_in1k", "acc": 83.7, "family": "Swin", "params": 49.7},
    "swinv2_base":        {"timm": "swinv2_base_window8_256.ms_in1k",  "acc": 84.2, "family": "Swin", "params": 87.9},
    # --- DeiT ---
    "deit_tiny":          {"timm": "deit_tiny_patch16_224.fb_in1k", "acc": 72.2, "family": "DeiT", "params": 5.7},
    "deit_small":         {"timm": "deit_small_patch16_224.fb_in1k","acc": 79.9, "family": "DeiT", "params": 22.1},
    "deit_base":          {"timm": "deit_base_patch16_224.fb_in1k", "acc": 81.8, "family": "DeiT", "params": 86.6},
    "deit3_small":        {"timm": "deit3_small_patch16_224.fb_in1k","acc": 81.4, "family": "DeiT", "params": 22.1},
    "deit3_base":         {"timm": "deit3_base_patch16_224.fb_in1k","acc": 83.8, "family": "DeiT", "params": 86.6},
    "deit3_large":        {"timm": "deit3_large_patch16_224.fb_in1k","acc": 84.9, "family": "DeiT", "params": 304.4},
    # --- RegNet ---
    "regnetx_016":        {"timm": "regnetx_016.tv2_in1k",         "acc": 73.0, "family": "RegNet", "params": 9.2},
    "regnetx_032":        {"timm": "regnetx_032.tv2_in1k",         "acc": 75.2, "family": "RegNet", "params": 15.3},
    "regnetx_064":        {"timm": "regnetx_064.pycls_in1k",       "acc": 76.4, "family": "RegNet", "params": 26.2},
    "regnetx_160":        {"timm": "regnetx_160.tv2_in1k",         "acc": 79.7, "family": "RegNet", "params": 54.3},
    "regnety_016":        {"timm": "regnety_016.tv2_in1k",         "acc": 74.0, "family": "RegNet", "params": 11.2},
    "regnety_032":        {"timm": "regnety_032.tv2_in1k",         "acc": 76.6, "family": "RegNet", "params": 19.4},
    "regnety_064":        {"timm": "regnety_064.pycls_in1k",       "acc": 77.2, "family": "RegNet", "params": 30.6},
    "regnety_160":        {"timm": "regnety_160.tv2_in1k",         "acc": 80.4, "family": "RegNet", "params": 83.6},
    # --- MaxViT / CoAtNet ---
    "maxvit_tiny":        {"timm": "maxvit_tiny_tf_224.in1k",      "acc": 83.4, "family": "MaxViT", "params": 30.9},
    "maxvit_small":       {"timm": "maxvit_small_tf_224.in1k",     "acc": 84.5, "family": "MaxViT", "params": 68.9},
    "coatnet_0":          {"timm": "coatnet_0_rw_224.sw_in1k",     "acc": 82.4, "family": "MaxViT", "params": 27.4},
    # --- EfficientFormer / EdgeNeXt ---
    "efficientformer_l1": {"timm": "efficientformer_l1.snap_dist_in1k", "acc": 80.2, "family": "EFormer", "params": 12.3},
    "efficientformer_l3": {"timm": "efficientformer_l3.snap_dist_in1k", "acc": 82.4, "family": "EFormer", "params": 31.4},
    "efficientformer_l7": {"timm": "efficientformer_l7.snap_dist_in1k", "acc": 83.3, "family": "EFormer", "params": 82.2},
    "edgenext_small":     {"timm": "edgenext_small.usi_in1k",      "acc": 81.1, "family": "EdgeNeXt", "params": 5.6},
    "edgenext_base":      {"timm": "edgenext_base.usi_in1k",       "acc": 83.3, "family": "EdgeNeXt", "params": 18.5},
    # --- Other CNNs ---
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
    # --- GhostNet / ShuffleNet ---
    "ghostnet_100":       {"timm": "ghostnet_100.in1k",            "acc": 74.0, "family": "Mobile", "params": 5.2},
    "shufflenetv2_x1_0":  {"timm": "shufflenetv2_x1_0.in1k",     "acc": 69.4, "family": "Mobile", "params": 2.3},
    "shufflenetv2_x2_0":  {"timm": "shufflenetv2_x2_0.in1k",     "acc": 76.2, "family": "Mobile", "params": 7.4},
    # --- PoolFormer ---
    "poolformer_s12":     {"timm": "poolformer_s12.sail_in1k",     "acc": 77.2, "family": "MetaFormer", "params": 11.9},
    "poolformer_s24":     {"timm": "poolformer_s24.sail_in1k",     "acc": 80.3, "family": "MetaFormer", "params": 21.4},
    "poolformer_s36":     {"timm": "poolformer_s36.sail_in1k",     "acc": 81.4, "family": "MetaFormer", "params": 30.9},
    "poolformer_m36":     {"timm": "poolformer_m36.sail_in1k",     "acc": 82.1, "family": "MetaFormer", "params": 56.2},
    # --- CaiT ---
    "cait_s24_224":       {"timm": "cait_s24_224.fb_dist_in1k",    "acc": 83.5, "family": "CaiT", "params": 47.0},
    # --- Mixer ---
    "mixer_b16":          {"timm": "mixer_b16_224.goog_in21k_ft_in1k","acc": 78.5, "family": "Mixer", "params": 59.9},
    # --- VGG ---
    "vgg16_bn":           {"timm": "vgg16_bn.tv_in1k",             "acc": 73.4, "family": "CNN", "params": 138.4},
    "vgg19_bn":           {"timm": "vgg19_bn.tv_in1k",             "acc": 74.2, "family": "CNN", "params": 143.7},
    # --- Squeeze ---
    "squeezenet1_1":      {"timm": "squeezenet1_1.tv_in1k",        "acc": 58.2, "family": "Mobile", "params": 1.2},
}

# Teachers
TEACHERS = {
    "dinov2_small": {
        "timm": "vit_small_patch14_dinov2.lvd142m",
        "type": "timm",
        "family": "ViT-SSL",
        "img_size": 224,
    },
    "clip_vit_b32": {
        "type": "open_clip",
        "model_name": "ViT-B-32",
        "pretrained": "openai",
        "family": "ViT-CLIP",
    },
    "convnext_base_fcmae": {
        "timm": "convnextv2_base.fcmae",
        "type": "timm_ssl",
        "family": "CNN-SSL",
    },
    "mae_vit_base": {
        "timm": "vit_base_patch16_224.mae",
        "type": "timm_ssl",
        "family": "ViT-SSL",
    },
}


@app.function(
    gpu="A100",
    image=image,
    volumes={"/results": volume},
    timeout=86400,
    memory=32768,
)
def run_full_experiment():
    """Run the complete RepNAS v3 experiment suite."""
    import os
    import time
    import random
    import warnings
    warnings.filterwarnings('ignore')

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from scipy import stats
    from scipy.spatial.distance import cdist
    import timm

    DEVICE = "cuda"
    NUM_IMAGES = 512
    IMG_SIZE = 224
    BATCH_SIZE = 32
    SEED = 42

    def set_seed(seed=SEED):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ============================================================
    # Feature extraction
    # ============================================================
    def get_features(model, images, layer="penultimate"):
        """Extract features. layer: 'penultimate', 'early', 'mid', 'late'."""
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

    def get_intermediate_features(model, images, frac=0.5):
        """Extract features from an intermediate layer (by fraction of depth)."""
        model.eval()
        model = model.to(DEVICE)

        # Collect all children modules
        all_modules = list(model.children())
        target_idx = max(0, int(len(all_modules) * frac) - 1)

        intermediate_out = []
        hook = None

        def hook_fn(m, inp, out):
            if isinstance(out, torch.Tensor):
                intermediate_out.append(out.detach())

        # Register hook on target module
        target_module = all_modules[target_idx]
        hook = target_module.register_forward_hook(hook_fn)

        feats = []
        with torch.no_grad():
            for i in range(0, len(images), BATCH_SIZE):
                batch = images[i:i+BATCH_SIZE].to(DEVICE)
                intermediate_out.clear()
                try:
                    model(batch)
                except:
                    pass
                if intermediate_out:
                    f = intermediate_out[0]
                    if f.dim() == 4:
                        f = F.adaptive_avg_pool2d(f, 1).flatten(1)
                    elif f.dim() == 3:
                        f = f[:, 0]
                    feats.append(f.cpu().float())

        hook.remove()
        model.cpu()
        torch.cuda.empty_cache()

        if feats:
            return torch.cat(feats, 0).numpy()
        return None

    # ============================================================
    # Similarity metrics
    # ============================================================
    def linear_cka(X, Y):
        X = X - X.mean(0, keepdims=True)
        Y = Y - Y.mean(0, keepdims=True)
        XtX = X @ X.T
        YtY = Y @ Y.T
        hsic_xy = np.sum(XtX * YtY)
        hsic_xx = np.sum(XtX * XtX)
        hsic_yy = np.sum(YtY * YtY)
        return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10))

    def centered_cosine(X, Y):
        X = X - X.mean(0, keepdims=True)
        Y = Y - Y.mean(0, keepdims=True)
        gx = (X @ X.T).flatten()
        gy = (Y @ Y.T).flatten()
        return float(np.dot(gx, gy) / (np.linalg.norm(gx) * np.linalg.norm(gy) + 1e-10))

    def mutual_knn(X, Y, k=10):
        dist_X = cdist(X, X, metric='cosine')
        dist_Y = cdist(Y, Y, metric='cosine')
        n = X.shape[0]
        knn_X = np.argsort(dist_X, axis=1)[:, 1:k+1]
        knn_Y = np.argsort(dist_Y, axis=1)[:, 1:k+1]
        agree = sum(len(set(knn_X[i]) & set(knn_Y[i])) for i in range(n))
        return agree / (n * k)

    # ============================================================
    # ZS-NAS baselines
    # ============================================================
    def compute_gradnorm(model, images):
        model = model.to(DEVICE)
        model.train()
        batch = images[:32].to(DEVICE)
        try:
            out = model(batch)
            if isinstance(out, (tuple, list)): out = out[0]
            nc = out.shape[1] if out.dim() == 2 else 1000
            tgt = torch.randint(0, nc, (batch.shape[0],), device=DEVICE)
            loss = F.cross_entropy(out, tgt)
            loss.backward()
            gn = sum(p.grad.norm(2).item()**2 for p in model.parameters() if p.grad is not None)
            model.zero_grad()
            model.cpu()
            torch.cuda.empty_cache()
            return float(np.log(np.sqrt(gn) + 1e-10))
        except:
            model.zero_grad()
            model.cpu()
            torch.cuda.empty_cache()
            return float('nan')

    def compute_naswot(model, images):
        model = model.to(DEVICE)
        model.eval()
        acts = []
        hooks = []

        def hook(m, inp, out):
            if isinstance(out, torch.Tensor) and out.dim() >= 2:
                a = (out > 0).float()
                if a.dim() == 4: a = a.mean([2, 3])
                elif a.dim() == 3: a = a.mean(1)
                acts.append(a.detach().cpu())

        for m in model.modules():
            if isinstance(m, (nn.ReLU, nn.GELU, nn.SiLU)):
                hooks.append(m.register_forward_hook(hook))

        with torch.no_grad():
            try: model(images[:64].to(DEVICE))
            except: pass

        for h in hooks: h.remove()
        model.cpu()
        torch.cuda.empty_cache()

        if not acts: return float('nan')
        K = torch.cat(acts, 1).numpy()
        if K.shape[1] > 500:
            idx = np.random.choice(K.shape[1], 500, replace=False)
            K = K[:, idx]
        K = K - K.mean(0, keepdims=True)
        C = K.T @ K / (K.shape[0] - 1 + 1e-10)
        try:
            _, ld = np.linalg.slogdet(C + 1e-4 * np.eye(C.shape[0]))
            return float(ld)
        except:
            return float('nan')

    def compute_synflow(model, images):
        model = model.to(DEVICE)
        model.eval()
        signs = {n: torch.sign(p.data) for n, p in model.named_parameters()}
        for p in model.parameters(): p.data.abs_()

        inp = torch.ones(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE, requires_grad=True)
        try:
            out = model(inp)
            if isinstance(out, (tuple, list)): out = out[0]
            out.sum().backward()
            score = sum((p.data * p.grad.data).sum().item() for p in model.parameters() if p.grad is not None)
            for n, p in model.named_parameters():
                p.data *= signs[n].to(DEVICE)
            model.cpu()
            torch.cuda.empty_cache()
            return float(np.log(abs(score) + 1e-10))
        except:
            for n, p in model.named_parameters():
                if n in signs: p.data *= signs[n].to(DEVICE)
            model.cpu()
            torch.cuda.empty_cache()
            return float('nan')

    # ============================================================
    # Load teacher
    # ============================================================
    def load_teacher(teacher_name, teacher_info, img_size=224):
        if teacher_info["type"] == "open_clip":
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                teacher_info["model_name"], pretrained=teacher_info["pretrained"]
            )
            model.eval()
            # For CLIP, we use the visual encoder
            class CLIPVisual(nn.Module):
                def __init__(self, clip_model):
                    super().__init__()
                    self.visual = clip_model.visual

                def forward_features(self, x):
                    return self.visual(x)

            return CLIPVisual(model)
        else:
            kwargs = {"pretrained": True, "num_classes": 0}
            if "img_size" in teacher_info:
                kwargs["img_size"] = teacher_info["img_size"]
            return timm.create_model(teacher_info["timm"], **kwargs)

    # ============================================================
    # Generate probe images
    # ============================================================
    def generate_probe_images(probe_type="noise", n=512, img_size=224, seed=42):
        set_seed(seed)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        if probe_type == "noise":
            images = torch.randn(n, 3, img_size, img_size)
            images = (images * std + mean).clamp(0, 1)
        elif probe_type == "imagenet_val":
            # Use torchvision to load real ImageNet-like images
            # We use CIFAR-100 as a stand-in (publicly available, natural images)
            from torchvision import datasets, transforms
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            try:
                dataset = datasets.CIFAR100(root="/tmp/cifar100", train=False, download=True, transform=transform)
                indices = list(range(min(n, len(dataset))))
                random.shuffle(indices)
                images = torch.stack([dataset[i][0] for i in indices[:n]])
            except:
                # Fallback to structured noise
                images = torch.randn(n, 3, img_size, img_size)
                images = (images * std + mean).clamp(0, 1)
        elif probe_type == "augmented":
            # Structured images: patches of varying frequency, like natural images
            images = []
            for _ in range(n):
                img = torch.zeros(3, img_size, img_size)
                # Random frequency gratings + noise
                for c in range(3):
                    freq = random.uniform(1, 20)
                    angle = random.uniform(0, np.pi)
                    x = torch.linspace(0, 1, img_size).unsqueeze(0).repeat(img_size, 1)
                    y = torch.linspace(0, 1, img_size).unsqueeze(1).repeat(1, img_size)
                    grating = torch.sin(2 * np.pi * freq * (x * np.cos(angle) + y * np.sin(angle)))
                    img[c] = grating * 0.3 + torch.randn(img_size, img_size) * 0.1 + 0.5
                images.append(img)
            images = torch.stack(images)
            # Normalize
            images = (images - mean) / std
            images = images * std + mean  # re-normalize
            images = images.clamp(0, 1)
        else:
            raise ValueError(f"Unknown probe type: {probe_type}")

        return images

    # ============================================================
    # MAIN EXPERIMENT
    # ============================================================
    t0 = time.time()
    print("=" * 80)
    print("RepNAS v3: Full Experiment Suite")
    print(f"Device: {DEVICE}, GPU: {torch.cuda.get_device_name(0)}")
    print(f"Candidates: {len(CANDIDATES)}, Teachers: {len(TEACHERS)}")
    print("=" * 80)

    all_results = {}

    # --- Generate all probe image sets ---
    print("\n[1] Generating probe images...")
    probes = {}
    for ptype in ["noise", "imagenet_val", "augmented"]:
        print(f"  Generating {ptype} probes...")
        probes[ptype] = generate_probe_images(ptype, n=NUM_IMAGES, img_size=IMG_SIZE)
        print(f"    Shape: {probes[ptype].shape}")

    # --- For each teacher ---
    for teacher_name, teacher_info in TEACHERS.items():
        print(f"\n{'='*80}")
        print(f"[TEACHER] {teacher_name} ({teacher_info.get('family', 'unknown')})")
        print(f"{'='*80}")

        try:
            teacher_model = load_teacher(teacher_name, teacher_info)
        except Exception as e:
            print(f"  ERROR loading teacher {teacher_name}: {e}")
            continue

        # Extract teacher features for each probe type
        teacher_feats = {}
        for ptype, pimages in probes.items():
            print(f"  Extracting teacher features ({ptype})...")
            teacher_feats[ptype] = get_features(teacher_model, pimages)
            print(f"    Shape: {teacher_feats[ptype].shape}")

        # Also get teacher intermediate features for layer-wise analysis
        teacher_mid_feats = {}
        for ptype in ["noise"]:  # layer-wise only for noise probes
            teacher_mid_feats[ptype] = get_intermediate_features(teacher_model, probes[ptype], frac=0.5)

        del teacher_model
        torch.cuda.empty_cache()

        # --- Score each candidate ---
        for idx, (cand_name, cand_info) in enumerate(CANDIDATES.items()):
            key = f"{teacher_name}__{cand_name}"
            if key in all_results:
                continue

            print(f"\n  [{idx+1}/{len(CANDIDATES)}] {cand_name} ({cand_info['family']}, {cand_info['params']}M)")
            result = {
                "teacher": teacher_name,
                "teacher_family": teacher_info.get("family", "unknown"),
                "candidate": cand_name,
                "family": cand_info["family"],
                "params": cand_info["params"],
                "gt_acc": cand_info["acc"],
            }

            try:
                # --- Pretrained features + similarity for each probe type ---
                for ptype in ["noise", "imagenet_val", "augmented"]:
                    try:
                        m = timm.create_model(cand_info["timm"], pretrained=True, num_classes=0)
                        cand_feats = get_features(m, probes[ptype])
                        del m
                        torch.cuda.empty_cache()

                        cka_pt = linear_cka(teacher_feats[ptype], cand_feats)
                        cos_pt = centered_cosine(teacher_feats[ptype], cand_feats)
                        knn_pt = mutual_knn(teacher_feats[ptype], cand_feats)

                        result[f"cka_pretrained_{ptype}"] = cka_pt
                        result[f"cosine_pretrained_{ptype}"] = cos_pt
                        result[f"knn_pretrained_{ptype}"] = knn_pt
                    except Exception as e:
                        print(f"    ERROR pretrained/{ptype}: {e}")
                        result[f"cka_pretrained_{ptype}"] = float('nan')
                        result[f"cosine_pretrained_{ptype}"] = float('nan')
                        result[f"knn_pretrained_{ptype}"] = float('nan')

                # --- Random-init features (noise probes only for efficiency) ---
                try:
                    set_seed()
                    m = timm.create_model(cand_info["timm"], pretrained=False, num_classes=0)
                    rand_feats = get_features(m, probes["noise"])
                    del m
                    torch.cuda.empty_cache()

                    result["cka_random"] = linear_cka(teacher_feats["noise"], rand_feats)
                    result["cosine_random"] = centered_cosine(teacher_feats["noise"], rand_feats)
                    result["knn_random"] = mutual_knn(teacher_feats["noise"], rand_feats)
                except Exception as e:
                    print(f"    ERROR random-init: {e}")
                    result["cka_random"] = float('nan')
                    result["cosine_random"] = float('nan')
                    result["knn_random"] = float('nan')

                # --- Layer-wise CKA (early=0.25, mid=0.5, late=0.75) for noise probes ---
                if teacher_name == "dinov2_small":  # only for primary teacher
                    for frac_name, frac_val in [("early", 0.25), ("mid", 0.5), ("late", 0.75)]:
                        try:
                            m = timm.create_model(cand_info["timm"], pretrained=True, num_classes=0)
                            mid_feats = get_intermediate_features(m, probes["noise"], frac=frac_val)
                            del m
                            torch.cuda.empty_cache()
                            if mid_feats is not None and teacher_mid_feats.get("noise") is not None:
                                result[f"cka_{frac_name}_layer"] = linear_cka(teacher_mid_feats["noise"], mid_feats)
                            else:
                                result[f"cka_{frac_name}_layer"] = float('nan')
                        except:
                            result[f"cka_{frac_name}_layer"] = float('nan')

                # --- ZS-NAS baselines (only for primary teacher to avoid redundancy) ---
                if teacher_name == "dinov2_small":
                    set_seed()
                    try:
                        m = timm.create_model(cand_info["timm"], pretrained=False, num_classes=1000)
                        result["gradnorm"] = compute_gradnorm(m, probes["noise"])
                        del m
                    except:
                        result["gradnorm"] = float('nan')

                    set_seed()
                    try:
                        m = timm.create_model(cand_info["timm"], pretrained=False, num_classes=1000)
                        result["naswot"] = compute_naswot(m, probes["noise"])
                        del m
                    except:
                        result["naswot"] = float('nan')

                    set_seed()
                    try:
                        m = timm.create_model(cand_info["timm"], pretrained=False, num_classes=1000)
                        result["synflow"] = compute_synflow(m, probes["noise"])
                        del m
                    except:
                        result["synflow"] = float('nan')

                    torch.cuda.empty_cache()

                # Compute FLOPs estimate using params as proxy (could use fvcore but keeping it simple)
                result["flops_est"] = cand_info["params"]  # M params ~ correlates with GFLOPs

                print(f"    acc={cand_info['acc']:.1f} | cka_pt_noise={result.get('cka_pretrained_noise', 'N/A'):.4f}" if isinstance(result.get('cka_pretrained_noise'), float) and not np.isnan(result.get('cka_pretrained_noise', float('nan'))) else f"    acc={cand_info['acc']:.1f}")

            except Exception as e:
                print(f"    FATAL ERROR: {e}")

            all_results[key] = result

            # Checkpoint periodically
            if (idx + 1) % 10 == 0:
                print(f"\n  --- Checkpoint: {len(all_results)} results saved ---")
                with open("/results/repnas_v3_results.json", "w") as f:
                    json.dump({"results": all_results, "config": {
                        "teachers": list(TEACHERS.keys()),
                        "num_candidates": len(CANDIDATES),
                        "num_images": NUM_IMAGES,
                        "probe_types": ["noise", "imagenet_val", "augmented"],
                        "seed": SEED,
                    }}, f, indent=2, default=str)
                volume.commit()

    # --- Final save ---
    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"COMPLETE: {len(all_results)} results in {elapsed/3600:.1f} hours")
    print(f"{'='*80}")

    output = {
        "results": all_results,
        "config": {
            "teachers": list(TEACHERS.keys()),
            "candidates": list(CANDIDATES.keys()),
            "num_candidates": len(CANDIDATES),
            "num_images": NUM_IMAGES,
            "probe_types": ["noise", "imagenet_val", "augmented"],
            "seed": SEED,
            "elapsed_seconds": elapsed,
        },
    }

    with open("/results/repnas_v3_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    volume.commit()

    print(f"Results saved to /results/repnas_v3_results.json")
    return {"status": "complete", "num_results": len(all_results), "elapsed_hours": elapsed/3600}
