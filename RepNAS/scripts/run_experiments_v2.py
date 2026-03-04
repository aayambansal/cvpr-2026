"""
Zero-Shot NAS via Representation Agreement (RepNAS) — v2 Experiment
====================================================================
KEY INSIGHT: We compute representation agreement between a FROZEN PRETRAINED 
teacher (DINOv2) and FROZEN PRETRAINED candidate architectures on UNLABELED data.
This is NOT training — it's evaluation of existing models.

The zero-shot score predicts: "Which architecture family best captures 
foundation-model-level visual representations?"

For true zero-cost scoring, we ALSO evaluate random-init candidates against 
the teacher to measure *architectural inductive bias* for representational 
alignment.

Approach:
  1. Extract features from DINOv2 teacher (frozen, pretrained)
  2. For each candidate architecture:
     a. Pretrained features → CKA/cosine/kNN vs teacher (upper bound)
     b. Random-init features → CKA/cosine/kNN vs teacher (zero-shot score)
  3. Compare with classical ZS-NAS baselines (SynFlow, NASWOT, GradNorm)
  4. Ground truth = ImageNet-1k top-1 accuracy
"""

import os
import sys
import json
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
from collections import OrderedDict

# ============================================================
DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
NUM_IMAGES = 512
IMG_SIZE = 224
BATCH_SIZE = 16
SEED = 42

TEACHER_NAME = 'vit_small_patch14_dinov2.lvd142m'

# 30 diverse architectures with known ImageNet-1k accuracies
CANDIDATES = OrderedDict({
    'resnet18':           {'timm': 'resnet18.a1_in1k',              'acc': 71.5, 'family': 'CNN', 'params': 11.7},
    'resnet34':           {'timm': 'resnet34.a1_in1k',              'acc': 75.1, 'family': 'CNN', 'params': 21.8},
    'resnet50':           {'timm': 'resnet50.a1_in1k',              'acc': 79.8, 'family': 'CNN', 'params': 25.6},
    'resnet101':          {'timm': 'resnet101.a1_in1k',             'acc': 81.5, 'family': 'CNN', 'params': 44.5},
    'resnet152':          {'timm': 'resnet152.a1_in1k',             'acc': 82.0, 'family': 'CNN', 'params': 60.2},
    'resnext50_32x4d':    {'timm': 'resnext50_32x4d.a1_in1k',      'acc': 80.1, 'family': 'CNN', 'params': 25.0},
    'wide_resnet50_2':    {'timm': 'wide_resnet50_2.tv2_in1k',     'acc': 81.6, 'family': 'CNN', 'params': 68.9},
    'densenet121':        {'timm': 'densenet121.ra_in1k',           'acc': 75.6, 'family': 'CNN', 'params': 8.0},
    'efficientnet_b0':    {'timm': 'efficientnet_b0.ra_in1k',      'acc': 77.7, 'family': 'EffNet', 'params': 5.3},
    'efficientnet_b1':    {'timm': 'efficientnet_b1.ft_in1k',      'acc': 79.2, 'family': 'EffNet', 'params': 7.8},
    'efficientnet_b2':    {'timm': 'efficientnet_b2.ra_in1k',      'acc': 80.6, 'family': 'EffNet', 'params': 9.1},
    'efficientnet_b3':    {'timm': 'efficientnet_b3.ra2_in1k',     'acc': 82.0, 'family': 'EffNet', 'params': 12.2},
    'mobilenetv2_100':    {'timm': 'mobilenetv2_100.ra_in1k',      'acc': 72.9, 'family': 'Mobile', 'params': 3.5},
    'mobilenetv3_large':  {'timm': 'mobilenetv3_large_100.ra_in1k','acc': 75.8, 'family': 'Mobile', 'params': 5.5},
    'mobilenetv3_small':  {'timm': 'mobilenetv3_small_100.lamb_in1k','acc': 67.7, 'family': 'Mobile', 'params': 2.5},
    'convnext_tiny':      {'timm': 'convnext_tiny.fb_in1k',        'acc': 82.1, 'family': 'ConvNeXt', 'params': 28.6},
    'convnext_small':     {'timm': 'convnext_small.fb_in1k',       'acc': 83.1, 'family': 'ConvNeXt', 'params': 50.2},
    'convnext_base':      {'timm': 'convnext_base.fb_in1k',        'acc': 83.8, 'family': 'ConvNeXt', 'params': 88.6},
    'vit_tiny_patch16':   {'timm': 'vit_tiny_patch16_224.augreg_in21k_ft_in1k', 'acc': 75.5, 'family': 'ViT', 'params': 5.7},
    'vit_small_patch16':  {'timm': 'vit_small_patch16_224.augreg_in21k_ft_in1k','acc': 81.4, 'family': 'ViT', 'params': 22.1},
    'vit_base_patch16':   {'timm': 'vit_base_patch16_224.augreg_in21k_ft_in1k', 'acc': 84.0, 'family': 'ViT', 'params': 86.6},
    'swin_tiny':          {'timm': 'swin_tiny_patch4_window7_224.ms_in1k',  'acc': 81.2, 'family': 'Swin', 'params': 28.3},
    'swin_small':         {'timm': 'swin_small_patch4_window7_224.ms_in1k', 'acc': 83.2, 'family': 'Swin', 'params': 49.6},
    'swin_base':          {'timm': 'swin_base_patch4_window7_224.ms_in1k',  'acc': 83.5, 'family': 'Swin', 'params': 87.8},
    'regnetx_032':        {'timm': 'regnetx_032.tv2_in1k',         'acc': 75.2, 'family': 'RegNet', 'params': 15.3},
    'regnety_032':        {'timm': 'regnety_032.tv2_in1k',         'acc': 76.6, 'family': 'RegNet', 'params': 19.4},
    'deit_tiny_patch16':  {'timm': 'deit_tiny_patch16_224.fb_in1k', 'acc': 72.2, 'family': 'DeiT', 'params': 5.7},
    'deit_small_patch16': {'timm': 'deit_small_patch16_224.fb_in1k','acc': 79.9, 'family': 'DeiT', 'params': 22.1},
    'deit_base_patch16':  {'timm': 'deit_base_patch16_224.fb_in1k', 'acc': 81.8, 'family': 'DeiT', 'params': 86.6},
    'regnetx_016':        {'timm': 'regnetx_016.tv2_in1k',         'acc': 73.0, 'family': 'RegNet', 'params': 9.2},
})


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_features(model, images):
    """Extract penultimate features."""
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
    return torch.cat(feats, 0).numpy()


def linear_cka(X, Y):
    """Linear CKA."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    XtX = X @ X.T
    YtY = Y @ Y.T
    hsic_xy = np.sum(XtX * YtY)
    hsic_xx = np.sum(XtX * XtX)
    hsic_yy = np.sum(YtY * YtY)
    return hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)


def centered_cosine(X, Y):
    """Centered cosine similarity via Gram matrices."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    gx = (X @ X.T).flatten()
    gy = (Y @ Y.T).flatten()
    return float(np.dot(gx, gy) / (np.linalg.norm(gx) * np.linalg.norm(gy) + 1e-10))


def mutual_knn(X, Y, k=10):
    """Mutual kNN agreement."""
    dist_X = cdist(X, X, metric='cosine')
    dist_Y = cdist(Y, Y, metric='cosine')
    n = X.shape[0]
    knn_X = np.argsort(dist_X, axis=1)[:, 1:k+1]
    knn_Y = np.argsort(dist_Y, axis=1)[:, 1:k+1]
    agree = sum(len(set(knn_X[i]) & set(knn_Y[i])) for i in range(n))
    return agree / (n * k)


def compute_naswot(model, images):
    """NASWOT score."""
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
    
    if not acts: return 0.0
    K = torch.cat(acts, 1).numpy()
    if K.shape[1] > 500:
        idx = np.random.choice(K.shape[1], 500, replace=False)
        K = K[:, idx]
    K = K - K.mean(0, keepdims=True)
    C = K.T @ K / (K.shape[0] - 1 + 1e-10)
    _, ld = np.linalg.slogdet(C + 1e-4 * np.eye(C.shape[0]))
    return float(ld)


def compute_gradnorm(model, images):
    """GradNorm score."""
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
        return float(np.log(np.sqrt(gn) + 1e-10))
    except:
        model.zero_grad()
        model.cpu()
        return 0.0


def compute_synflow(model, images):
    """SynFlow score."""
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
        return float(np.log(abs(score) + 1e-10))
    except:
        for n, p in model.named_parameters():
            if n in signs: p.data *= signs[n].to(DEVICE)
        model.cpu()
        return 0.0


def main():
    set_seed()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiments')
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 70)
    print("RepNAS: Zero-Shot NAS via Representation Agreement")
    print("=" * 70)
    
    # Generate unlabeled data (random ImageNet-like images)
    print("\n[1] Generating unlabeled images...")
    set_seed()
    images = torch.randn(NUM_IMAGES, 3, IMG_SIZE, IMG_SIZE)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images = (images * std + mean).clamp(0, 1)
    
    # Teacher features
    print(f"\n[2] Loading DINOv2 teacher...")
    teacher = timm.create_model(TEACHER_NAME, pretrained=True, num_classes=0, img_size=IMG_SIZE)
    teacher_feats = get_features(teacher, images)
    print(f"  Teacher features: {teacher_feats.shape}")
    del teacher
    torch.mps.empty_cache() if DEVICE == 'mps' else None
    
    # Score candidates
    print(f"\n[3] Scoring {len(CANDIDATES)} architectures...\n")
    results = {}
    
    for i, (name, info) in enumerate(CANDIDATES.items()):
        print(f"  [{i+1}/{len(CANDIDATES)}] {name} ({info['family']}, {info['params']}M params)")
        
        try:
            # === Pretrained features → representation agreement ===
            m = timm.create_model(info['timm'], pretrained=True, num_classes=0)
            feats = get_features(m, images)
            del m
            
            cka_pt = linear_cka(teacher_feats, feats)
            cos_pt = centered_cosine(teacher_feats, feats)
            knn_pt = mutual_knn(teacher_feats, feats)
            
            # === Random-init features → zero-shot representation agreement ===
            set_seed()  # same init seed for all
            m = timm.create_model(info['timm'], pretrained=False, num_classes=0)
            feats_rand = get_features(m, images)
            del m
            
            cka_rand = linear_cka(teacher_feats, feats_rand)
            cos_rand = centered_cosine(teacher_feats, feats_rand)
            knn_rand = mutual_knn(teacher_feats, feats_rand)
            
            # === Classical ZS-NAS baselines (random init) ===
            set_seed()
            m = timm.create_model(info['timm'], pretrained=False, num_classes=1000)
            naswot = compute_naswot(m, images)
            del m
            
            set_seed()
            m = timm.create_model(info['timm'], pretrained=False, num_classes=1000)
            gradnorm = compute_gradnorm(m, images)
            del m
            
            set_seed()
            m = timm.create_model(info['timm'], pretrained=False, num_classes=1000)
            synflow = compute_synflow(m, images)
            del m
            
            results[name] = {
                'family': info['family'],
                'params': info['params'],
                'gt_acc': info['acc'],
                # Our method (pretrained)
                'cka_pretrained': float(cka_pt),
                'cosine_pretrained': float(cos_pt),
                'knn_pretrained': float(knn_pt),
                # Our method (random init — true zero-shot)
                'cka_random': float(cka_rand),
                'cosine_random': float(cos_rand),
                'knn_random': float(knn_rand),
                # Combined score: weighted sum of pretrained metrics
                'repnas_score': float(0.4 * cka_pt + 0.3 * cos_pt + 0.3 * knn_pt),
                # Baselines
                'naswot': float(naswot),
                'gradnorm': float(gradnorm),
                'synflow': float(synflow),
            }
            
            print(f"    Acc={info['acc']:.1f} | CKA_pt={cka_pt:.4f} cos_pt={cos_pt:.4f} kNN_pt={knn_pt:.4f}")
            print(f"    CKA_rand={cka_rand:.4f} | NASWOT={naswot:.1f} GradNorm={gradnorm:.2f} SynFlow={synflow:.2f}")
            
        except Exception as e:
            print(f"    ERROR: {e}")
        
        if DEVICE == 'mps': torch.mps.empty_cache()
    
    # Compute correlations
    print("\n\n[4] Rank Correlations with Ground-Truth Accuracy")
    print("=" * 80)
    
    names = list(results.keys())
    gt = np.array([results[n]['gt_acc'] for n in names])
    
    metric_keys = [
        ('CKA (pretrained)', 'cka_pretrained'),
        ('Cosine (pretrained)', 'cosine_pretrained'),
        ('kNN (pretrained)', 'knn_pretrained'),
        ('RepNAS Combined', 'repnas_score'),
        ('CKA (random init)', 'cka_random'),
        ('Cosine (random init)', 'cosine_random'),
        ('kNN (random init)', 'knn_random'),
        ('NASWOT', 'naswot'),
        ('GradNorm', 'gradnorm'),
        ('SynFlow', 'synflow'),
    ]
    
    corr_results = {}
    print(f"\n{'Metric':<25} {'Spearman ρ':>12} {'p-val':>10} {'Kendall τ':>12} {'p-val':>10}")
    print("-" * 72)
    
    for label, key in metric_keys:
        vals = np.array([results[n][key] for n in names])
        # Filter out NaN/Inf
        mask = np.isfinite(vals)
        if mask.sum() < 5:
            print(f"{label:<25} {'N/A':>12} {'N/A':>10} {'N/A':>12} {'N/A':>10}")
            continue
        
        sp_rho, sp_p = stats.spearmanr(vals[mask], gt[mask])
        kt_tau, kt_p = stats.kendalltau(vals[mask], gt[mask])
        
        corr_results[label] = {
            'spearman_rho': round(float(sp_rho), 4),
            'spearman_p': float(sp_p),
            'kendall_tau': round(float(kt_tau), 4),
            'kendall_p': float(kt_p),
            'n_valid': int(mask.sum()),
        }
        
        marker = " ***" if sp_p < 0.001 else (" **" if sp_p < 0.01 else (" *" if sp_p < 0.05 else ""))
        print(f"{label:<25} {sp_rho:>12.4f} {sp_p:>10.1e} {kt_tau:>12.4f} {kt_p:>10.1e}{marker}")
    
    # Save
    output = {
        'config': {
            'teacher': TEACHER_NAME,
            'num_images': NUM_IMAGES,
            'seed': SEED,
            'device': DEVICE,
        },
        'results': results,
        'correlations': corr_results,
    }
    
    path = os.path.join(out_dir, 'repnas_results.json')
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {path}")
    print("\nDONE!")
    return output


if __name__ == '__main__':
    main()
