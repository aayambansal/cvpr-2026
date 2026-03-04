"""
RepNAS Transfer Benchmark (focused re-run)
==========================================
Linear probe accuracy on CIFAR-100 and Flowers-102 for all 83 architectures.
Fixes: removed deprecated multi_class param from LogisticRegression.
"""

import modal
import json

app = modal.App("repnas-transfer")
volume = modal.Volume.from_name("repnas-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch", "torchvision", "timm", "scipy", "numpy", "scikit-learn")
)

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
def run_transfer():
    """Linear probe transfer benchmarks for all 83 architectures."""
    import time
    import warnings
    warnings.filterwarnings('ignore')

    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    import torchvision
    import torchvision.transforms as T
    import timm

    DEVICE = "cuda"
    t0 = time.time()

    print("=" * 60)
    print("TRANSFER BENCHMARKS")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print("Downloading CIFAR-100...")
    cifar100_train = torchvision.datasets.CIFAR100(
        root="/tmp/data", train=True, download=True, transform=transform)
    cifar100_test = torchvision.datasets.CIFAR100(
        root="/tmp/data", train=False, download=True, transform=transform)

    print("Downloading Flowers102...")
    try:
        flowers_train = torchvision.datasets.Flowers102(
            root="/tmp/data", split="train", download=True, transform=transform)
        flowers_test = torchvision.datasets.Flowers102(
            root="/tmp/data", split="test", download=True, transform=transform)
        has_flowers = True
    except Exception as e:
        print(f"Flowers102 failed: {e}")
        has_flowers = False

    def extract_features_dataset(model, dataset, max_samples=5000):
        model.eval()
        model = model.to(DEVICE)
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
        feats, labels = [], []
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
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(train_feats, train_labels)
        return float(clf.score(test_feats, test_labels)) * 100.0

    results = {}
    for idx, (name, info) in enumerate(CANDIDATES.items()):
        print(f"  [{idx+1}/{len(CANDIDATES)}] {name}...")
        try:
            m = timm.create_model(info["timm"], pretrained=True, num_classes=0)

            train_f, train_l = extract_features_dataset(m, cifar100_train, max_samples=5000)
            test_f, test_l = extract_features_dataset(m, cifar100_test, max_samples=2000)
            cifar_acc = linear_probe_acc(train_f, train_l, test_f, test_l)

            flowers_acc = None
            if has_flowers:
                try:
                    fl_train_f, fl_train_l = extract_features_dataset(m, flowers_train, max_samples=2000)
                    fl_test_f, fl_test_l = extract_features_dataset(m, flowers_test, max_samples=5000)
                    flowers_acc = linear_probe_acc(fl_train_f, fl_train_l, fl_test_f, fl_test_l)
                except Exception as e2:
                    print(f"    Flowers failed: {e2}")

            del m
            torch.cuda.empty_cache()
            results[name] = {
                "imagenet_acc": info["acc"], "cifar100_acc": cifar_acc,
                "flowers102_acc": flowers_acc, "params": info["params"], "family": info["family"],
            }
            print(f"    CIFAR-100={cifar_acc:.1f}%, Flowers={flowers_acc:.1f}%" if flowers_acc else f"    CIFAR-100={cifar_acc:.1f}%")
        except Exception as e:
            print(f"    FAILED: {e}")
            results[name] = {
                "imagenet_acc": info["acc"], "cifar100_acc": None,
                "flowers102_acc": None, "params": info["params"], "family": info["family"],
                "error": str(e),
            }

    elapsed = time.time() - t0
    output = {"transfer": results, "elapsed_minutes": elapsed / 60}

    with open("/results/repnas_transfer.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    volume.commit()
    print(f"\nCOMPLETE in {elapsed/60:.1f}m. Saved to /results/repnas_transfer.json")
    return {"status": "complete", "elapsed_minutes": elapsed / 60}
