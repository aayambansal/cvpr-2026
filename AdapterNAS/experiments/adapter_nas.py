"""
AdapterNAS: Training-Free Neural Architecture Search for Foundation Model Adapter Topologies

This script implements a zero-cost proxy-based search over LoRA adapter configurations
for Vision Transformers (ViT-B/16). Instead of searching the backbone architecture,
we search the adapter topology: which layers get LoRA, rank per layer, and which
projection matrices (Q/K/V/MLP) to adapt.

Zero-cost proxies used:
  1. Gradient Norm (GradNorm): Sum of gradient norms on a calibration mini-batch
  2. SNIP (sensitivity): Connection sensitivity |g * w|
  3. Fisher Information: Diagonal Fisher approximation
  4. Jacob Covariance (jacob_cov): Jacobian covariance log-determinant (NASWOT)
  5. Entropy: Output entropy on calibration data (label-free)

Experiments:
  - ViT-B/16 pretrained on ImageNet-21k
  - Downstream: CIFAR-100, Oxford Flowers-102
  - Search space: 12 transformer blocks x {Q, K, V, MLP_up, MLP_down} x rank in {0, 4, 8, 16, 32}
  - Evaluate Pareto front of accuracy vs. #params-added vs. latency
"""

import os
import sys
import json
import time
import random
import itertools
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

# ---- LoRA Module ----

class LoRALinear(nn.Module):
    """Low-Rank Adaptation layer wrapping a frozen linear layer."""
    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float = None):
        super().__init__()
        self.base_linear = base_linear
        self.rank = rank
        self.alpha = alpha if alpha else float(rank)
        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # Freeze base
        for p in self.base_linear.parameters():
            p.requires_grad = False

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = self.alpha / self.rank

    def forward(self, x):
        base_out = self.base_linear(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out


# ---- Search Space ----

RANKS = [0, 4, 8, 16, 32]
NUM_LAYERS = 12
MODULE_TYPES = ['q', 'k', 'v', 'mlp_fc1', 'mlp_fc2']  # Q, K, V, MLP up, MLP down


def encode_config(config_dict):
    """Encode a configuration as a hashable tuple."""
    items = []
    for layer_idx in range(NUM_LAYERS):
        for mod in MODULE_TYPES:
            items.append(config_dict.get((layer_idx, mod), 0))
    return tuple(items)


def decode_config(config_tuple):
    """Decode a config tuple back to dict."""
    config = {}
    idx = 0
    for layer_idx in range(NUM_LAYERS):
        for mod in MODULE_TYPES:
            config[(layer_idx, mod)] = config_tuple[idx]
            idx += 1
    return config


def count_adapter_params(config_dict, hidden_dim=768, mlp_dim=3072):
    """Count total adapter parameters for a given config."""
    total = 0
    for (layer_idx, mod), rank in config_dict.items():
        if rank == 0:
            continue
        if mod in ['q', 'k', 'v']:
            total += hidden_dim * rank + rank * hidden_dim  # A + B
        elif mod == 'mlp_fc1':
            total += hidden_dim * rank + rank * mlp_dim
        elif mod == 'mlp_fc2':
            total += mlp_dim * rank + rank * hidden_dim
    return total


def sample_random_configs(n_configs, seed=42):
    """Sample random adapter configurations."""
    rng = random.Random(seed)
    configs = []
    for _ in range(n_configs):
        config = {}
        for layer_idx in range(NUM_LAYERS):
            for mod in MODULE_TYPES:
                config[(layer_idx, mod)] = rng.choice(RANKS)
        configs.append(config)
    return configs


def sample_structured_configs():
    """
    Sample structured configurations for controlled experiments:
    - Uniform configs (same rank everywhere)
    - Attention-only vs MLP-only
    - Deep vs Shallow
    - Increasing/Decreasing rank
    """
    configs = []
    labels = []

    # 1. Uniform rank across all modules
    for r in [4, 8, 16, 32]:
        config = {}
        for l in range(NUM_LAYERS):
            for m in MODULE_TYPES:
                config[(l, m)] = r
        configs.append(config)
        labels.append(f'uniform_r{r}')

    # 2. Attention-only (Q, K, V) with various ranks
    for r in [4, 8, 16, 32]:
        config = {}
        for l in range(NUM_LAYERS):
            for m in MODULE_TYPES:
                config[(l, m)] = r if m in ['q', 'k', 'v'] else 0
        configs.append(config)
        labels.append(f'attn_only_r{r}')

    # 3. MLP-only
    for r in [4, 8, 16]:
        config = {}
        for l in range(NUM_LAYERS):
            for m in MODULE_TYPES:
                config[(l, m)] = r if m in ['mlp_fc1', 'mlp_fc2'] else 0
        configs.append(config)
        labels.append(f'mlp_only_r{r}')

    # 4. Q+V only (classic LoRA)
    for r in [4, 8, 16, 32]:
        config = {}
        for l in range(NUM_LAYERS):
            for m in MODULE_TYPES:
                config[(l, m)] = r if m in ['q', 'v'] else 0
        configs.append(config)
        labels.append(f'qv_only_r{r}')

    # 5. Increasing rank: shallow layers low rank, deep layers high rank
    config = {}
    for l in range(NUM_LAYERS):
        r = RANKS[min(l // 3 + 1, len(RANKS) - 1)]
        for m in MODULE_TYPES:
            config[(l, m)] = r
    configs.append(config)
    labels.append('increasing_rank')

    # 6. Decreasing rank: deep layers low rank, shallow layers high rank
    config = {}
    for l in range(NUM_LAYERS):
        r = RANKS[min((NUM_LAYERS - 1 - l) // 3 + 1, len(RANKS) - 1)]
        for m in MODULE_TYPES:
            config[(l, m)] = r
    configs.append(config)
    labels.append('decreasing_rank')

    # 7. Only last 4 layers
    for r in [8, 16]:
        config = {}
        for l in range(NUM_LAYERS):
            for m in MODULE_TYPES:
                config[(l, m)] = r if l >= 8 else 0
        configs.append(config)
        labels.append(f'last4_r{r}')

    # 8. Only first 4 layers
    for r in [8, 16]:
        config = {}
        for l in range(NUM_LAYERS):
            for m in MODULE_TYPES:
                config[(l, m)] = r if l < 4 else 0
        configs.append(config)
        labels.append(f'first4_r{r}')

    # 9. Every other layer
    for r in [8, 16]:
        config = {}
        for l in range(NUM_LAYERS):
            for m in MODULE_TYPES:
                config[(l, m)] = r if l % 2 == 0 else 0
        configs.append(config)
        labels.append(f'even_layers_r{r}')

    # 10. Skip connections pattern: rank varies by module type
    config = {}
    for l in range(NUM_LAYERS):
        config[(l, 'q')] = 16
        config[(l, 'k')] = 4
        config[(l, 'v')] = 16
        config[(l, 'mlp_fc1')] = 8
        config[(l, 'mlp_fc2')] = 8
    configs.append(config)
    labels.append('mixed_rank')

    # 11. Zero config (baseline — no adapters)
    config = {}
    for l in range(NUM_LAYERS):
        for m in MODULE_TYPES:
            config[(l, m)] = 0
    configs.append(config)
    labels.append('no_adapter')

    return configs, labels


# ---- Model Setup ----

def load_vit_b16(num_classes=100):
    """Load ViT-B/16 pretrained and add classification head."""
    import timm
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    return model


def get_module_mapping(model):
    """Map (layer_idx, module_type) to actual nn.Linear modules in ViT."""
    mapping = {}
    for layer_idx in range(NUM_LAYERS):
        block = model.blocks[layer_idx]
        mapping[(layer_idx, 'q')] = block.attn.qkv  # Combined QKV — we'll handle this specially
        mapping[(layer_idx, 'k')] = block.attn.qkv
        mapping[(layer_idx, 'v')] = block.attn.qkv
        mapping[(layer_idx, 'mlp_fc1')] = block.mlp.fc1
        mapping[(layer_idx, 'mlp_fc2')] = block.mlp.fc2
    return mapping


def apply_lora_config(model, config_dict):
    """
    Apply LoRA adapters to a ViT model according to config_dict.
    For timm ViT, Q/K/V are combined in a single qkv linear layer.
    We apply separate LoRA to each conceptual projection.
    """
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    lora_modules = {}

    for layer_idx in range(NUM_LAYERS):
        block = model.blocks[layer_idx]

        # Handle QKV (combined projection in timm)
        qkv_linear = block.attn.qkv
        q_rank = config_dict.get((layer_idx, 'q'), 0)
        k_rank = config_dict.get((layer_idx, 'k'), 0)
        v_rank = config_dict.get((layer_idx, 'v'), 0)

        if q_rank > 0 or k_rank > 0 or v_rank > 0:
            # We create a LoRA wrapper for the full qkv projection
            # with rank = max of individual ranks for simplicity
            max_rank = max(q_rank, k_rank, v_rank)
            if max_rank > 0:
                lora_layer = LoRALinear(qkv_linear, rank=max_rank)
                block.attn.qkv = lora_layer
                lora_modules[f'block{layer_idx}_qkv'] = lora_layer

        # Handle MLP fc1
        mlp_fc1_rank = config_dict.get((layer_idx, 'mlp_fc1'), 0)
        if mlp_fc1_rank > 0:
            lora_layer = LoRALinear(block.mlp.fc1, rank=mlp_fc1_rank)
            block.mlp.fc1 = lora_layer
            lora_modules[f'block{layer_idx}_mlp_fc1'] = lora_layer

        # Handle MLP fc2
        mlp_fc2_rank = config_dict.get((layer_idx, 'mlp_fc2'), 0)
        if mlp_fc2_rank > 0:
            lora_layer = LoRALinear(block.mlp.fc2, rank=mlp_fc2_rank)
            block.mlp.fc2 = lora_layer
            lora_modules[f'block{layer_idx}_mlp_fc2'] = lora_layer

    # Unfreeze classification head
    for param in model.head.parameters():
        param.requires_grad = True

    return model, lora_modules


# ---- Data Loading ----

def get_cifar100_loaders(batch_size=64, cal_size=256, val_size=1000, data_dir='./data'):
    """Get CIFAR-100 calibration and validation loaders."""
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)

    # Calibration subset (tiny batch for zero-cost proxy computation)
    cal_indices = list(range(cal_size))
    cal_subset = Subset(trainset, cal_indices)
    cal_loader = DataLoader(cal_subset, batch_size=min(batch_size, cal_size), shuffle=False, num_workers=2)

    # Small training set (1-5% for fine-tuning validation)
    train_size = int(0.02 * len(trainset))  # 2% = 1000 samples
    train_indices = list(range(train_size))
    train_subset = Subset(trainset, train_indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Validation
    val_indices = list(range(min(val_size, len(testset))))
    val_subset = Subset(testset, val_indices)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Full test
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return cal_loader, train_loader, val_loader, test_loader


def get_flowers102_loaders(batch_size=64, cal_size=256, data_dir='./data'):
    """Get Oxford Flowers-102 loaders."""
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    trainset = torchvision.datasets.Flowers102(root=data_dir, split='train', download=True, transform=transform_train)
    valset = torchvision.datasets.Flowers102(root=data_dir, split='val', download=True, transform=transform_test)
    testset = torchvision.datasets.Flowers102(root=data_dir, split='test', download=True, transform=transform_test)

    cal_indices = list(range(min(cal_size, len(trainset))))
    cal_subset = Subset(trainset, cal_indices)
    cal_loader = DataLoader(cal_subset, batch_size=min(batch_size, cal_size), shuffle=False, num_workers=2)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return cal_loader, train_loader, val_loader, test_loader


# ---- Zero-Cost Proxies ----

def compute_gradnorm(model, cal_loader, device='cpu'):
    """
    GradNorm proxy: sum of L2 norms of gradients on calibration batch.
    Higher = more trainable.
    """
    model.to(device)
    model.train()
    model.zero_grad()

    total_grad_norm = 0.0
    n_batches = 0

    for images, labels in cal_loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = F.cross_entropy(output, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                total_grad_norm += param.grad.norm(2).item()

        model.zero_grad()
        n_batches += 1
        if n_batches >= 2:
            break

    return total_grad_norm / max(n_batches, 1)


def compute_snip(model, cal_loader, device='cpu'):
    """
    SNIP (Single-shot Network Pruning): |gradient * weight| sensitivity.
    Measures connection sensitivity.
    """
    model.to(device)
    model.train()
    model.zero_grad()

    for images, labels in cal_loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        break

    snip_score = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            snip_score += (param.grad * param.data).abs().sum().item()

    model.zero_grad()
    return snip_score


def compute_fisher(model, cal_loader, device='cpu'):
    """
    Fisher Information proxy: diagonal Fisher approximation.
    Sum of squared gradients of log-likelihood.
    """
    model.to(device)
    model.train()

    fisher_score = 0.0
    n_batches = 0

    for images, labels in cal_loader:
        images, labels = images.to(device), labels.to(device)
        model.zero_grad()
        output = model(images)
        log_probs = F.log_softmax(output, dim=-1)

        # Sample from predicted distribution
        dist = torch.distributions.Categorical(logits=output)
        sampled_labels = dist.sample()
        loss = F.nll_loss(log_probs, sampled_labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_score += (param.grad ** 2).sum().item()

        n_batches += 1
        if n_batches >= 2:
            break

    model.zero_grad()
    return fisher_score / max(n_batches, 1)


def compute_jacob_cov(model, cal_loader, device='cpu', num_samples=64):
    """
    Jacobian covariance score (NASWOT / jacob_cov).
    Log-determinant of the Jacobian covariance matrix.
    """
    model.to(device)
    model.eval()

    # Collect Jacobian rows
    jacobians = []

    for images, labels in cal_loader:
        images = images[:num_samples].to(device)
        images.requires_grad_(True)

        output = model(images)
        num_classes = output.shape[1]

        for i in range(min(images.shape[0], num_samples)):
            model.zero_grad()
            if images.grad is not None:
                images.grad.zero_()

            output_i = model(images[i:i+1])
            # Use the max logit
            max_class = output_i.argmax(dim=-1)
            score = output_i[0, max_class]
            score.backward(retain_graph=True)

            if images.grad is not None:
                grad_flat = images.grad[i].detach().flatten().cpu().numpy()
                jacobians.append(grad_flat)

            if len(jacobians) >= num_samples:
                break
        break

    if len(jacobians) < 2:
        return 0.0

    J = np.array(jacobians)
    # Subsample features to make covariance tractable
    if J.shape[1] > 1000:
        idx = np.random.choice(J.shape[1], 1000, replace=False)
        J = J[:, idx]

    # Covariance
    cov = np.cov(J)
    # Log det (add small regularization)
    sign, logdet = np.linalg.slogdet(cov + 1e-5 * np.eye(cov.shape[0]))

    return logdet if sign > 0 else -logdet


def compute_entropy(model, cal_loader, device='cpu'):
    """
    Label-free entropy proxy: average entropy of predicted distribution.
    Lower entropy = more confident = potentially better adapted.
    We return negative entropy so higher = better.
    """
    model.to(device)
    model.eval()

    total_entropy = 0.0
    n_samples = 0

    with torch.no_grad():
        for images, labels in cal_loader:
            images = images.to(device)
            output = model(images)
            probs = F.softmax(output, dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
            total_entropy += entropy.sum().item()
            n_samples += images.shape[0]
            break

    avg_entropy = total_entropy / max(n_samples, 1)
    # Return negative entropy (higher = more confident = potentially better)
    return -avg_entropy


# ---- Latency Estimation ----

def estimate_latency(model, input_shape=(1, 3, 224, 224), device='cpu', n_runs=10):
    """Estimate inference latency."""
    model.to(device)
    model.eval()
    dummy = torch.randn(*input_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model(dummy)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.time()
            model(dummy)
            times.append(time.time() - start)

    return np.mean(times) * 1000  # ms


# ---- Fine-Tuning ----

def finetune_and_evaluate(model, train_loader, val_loader, device='cpu',
                          epochs=3, lr=1e-3, label=''):
    """Fine-tune adapter parameters and evaluate."""
    model.to(device)

    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        # No adapter — just evaluate the frozen model
        return evaluate(model, val_loader, device)

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = F.cross_entropy(output, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        print(f'  [{label}] Epoch {epoch+1}/{epochs}: Loss={running_loss/len(train_loader):.4f}, Train Acc={train_acc:.2f}%')

    val_acc = evaluate(model, val_loader, device)
    return val_acc


def evaluate(model, loader, device='cpu'):
    """Evaluate model accuracy."""
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total


# ---- Main Experiment Pipeline ----

def run_search_experiment(dataset_name='cifar100', n_random=50, device='cpu',
                          results_dir='../results'):
    """
    Main experiment: evaluate zero-cost proxies on diverse adapter configs,
    then fine-tune top candidates and baselines to validate proxy quality.
    """
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  AdapterNAS Search Experiment: {dataset_name}")
    print(f"{'='*60}")

    num_classes = 100 if dataset_name == 'cifar100' else 102

    # Get data
    print("\n[1/5] Loading data...")
    if dataset_name == 'cifar100':
        cal_loader, train_loader, val_loader, test_loader = get_cifar100_loaders(
            batch_size=32, cal_size=128, data_dir='./data')
    else:
        cal_loader, train_loader, val_loader, test_loader = get_flowers102_loaders(
            batch_size=32, cal_size=128, data_dir='./data')

    # Generate configs
    print("\n[2/5] Generating adapter configurations...")
    structured_configs, structured_labels = sample_structured_configs()
    random_configs = sample_random_configs(n_random, seed=42)
    random_labels = [f'random_{i}' for i in range(n_random)]

    all_configs = structured_configs + random_configs
    all_labels = structured_labels + random_labels

    print(f"  Total configs to evaluate: {len(all_configs)}")
    print(f"  Structured: {len(structured_configs)}, Random: {len(random_configs)}")

    # Evaluate zero-cost proxies
    print("\n[3/5] Computing zero-cost proxy scores...")
    all_results = []

    for idx, (config, label) in enumerate(zip(all_configs, all_labels)):
        print(f"\r  Config {idx+1}/{len(all_configs)}: {label}", end='', flush=True)

        # Count params
        n_params = count_adapter_params(config)

        # Skip configs with 0 params (no adapter) for proxy computation
        if n_params == 0:
            all_results.append({
                'label': label,
                'config': {f'{k[0]}_{k[1]}': v for k, v in config.items()},
                'n_params': 0,
                'gradnorm': 0,
                'snip': 0,
                'fisher': 0,
                'entropy': 0,
                'jacob_cov': 0,
                'latency_ms': 0,
            })
            continue

        # Create fresh model for each config
        model = load_vit_b16(num_classes=num_classes)
        model, lora_mods = apply_lora_config(model, config)

        # Compute proxies
        try:
            gn = compute_gradnorm(model, cal_loader, device)
        except Exception as e:
            gn = 0.0

        try:
            snip = compute_snip(model, cal_loader, device)
        except Exception as e:
            snip = 0.0

        try:
            fisher = compute_fisher(model, cal_loader, device)
        except Exception as e:
            fisher = 0.0

        try:
            entropy = compute_entropy(model, cal_loader, device)
        except Exception as e:
            entropy = 0.0

        # Latency
        latency = estimate_latency(model, device=device)

        result = {
            'label': label,
            'config': {f'{k[0]}_{k[1]}': v for k, v in config.items()},
            'n_params': n_params,
            'gradnorm': gn,
            'snip': snip,
            'fisher': fisher,
            'entropy': entropy,
            'jacob_cov': 0,  # Skip jacob_cov for speed (expensive)
            'latency_ms': latency,
        }
        all_results.append(result)

        # Clean up
        del model, lora_mods
        if device != 'cpu':
            torch.cuda.empty_cache()

    print("\n  Done computing proxy scores.")

    # Save proxy results
    proxy_file = os.path.join(results_dir, f'{dataset_name}_proxy_scores.json')
    with open(proxy_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved proxy scores to {proxy_file}")

    # Select top candidates using combined proxy score
    print("\n[4/5] Selecting top candidates for fine-tuning...")

    # Normalize and combine proxies
    valid_results = [r for r in all_results if r['n_params'] > 0]

    for metric in ['gradnorm', 'snip', 'fisher', 'entropy']:
        values = [r[metric] for r in valid_results]
        vmin, vmax = min(values), max(values)
        rng_val = vmax - vmin if vmax != vmin else 1.0
        for r in valid_results:
            r[f'{metric}_norm'] = (r[metric] - vmin) / rng_val

    # Combined score (equal weight)
    for r in valid_results:
        r['combined_score'] = (
            r['gradnorm_norm'] + r['snip_norm'] + r['fisher_norm'] + r['entropy_norm']
        ) / 4.0

    # Sort by combined score
    valid_results.sort(key=lambda x: x['combined_score'], reverse=True)

    # Select top-5, bottom-5, and baselines for fine-tuning
    top_k = min(5, len(valid_results))
    bottom_k = min(5, len(valid_results))

    finetune_candidates = []
    finetune_labels_set = set()

    # Top-5 by proxy
    for r in valid_results[:top_k]:
        if r['label'] not in finetune_labels_set:
            finetune_candidates.append(r)
            finetune_labels_set.add(r['label'])

    # Bottom-5 by proxy
    for r in valid_results[-bottom_k:]:
        if r['label'] not in finetune_labels_set:
            finetune_candidates.append(r)
            finetune_labels_set.add(r['label'])

    # Key baselines
    baseline_labels = ['uniform_r8', 'uniform_r16', 'qv_only_r8', 'qv_only_r16',
                       'attn_only_r8', 'attn_only_r16', 'mlp_only_r8',
                       'increasing_rank', 'decreasing_rank', 'mixed_rank',
                       'last4_r16', 'first4_r16', 'even_layers_r16']
    for r in valid_results:
        if r['label'] in baseline_labels and r['label'] not in finetune_labels_set:
            finetune_candidates.append(r)
            finetune_labels_set.add(r['label'])

    print(f"  Selected {len(finetune_candidates)} candidates for fine-tuning")

    # Fine-tune and evaluate
    print("\n[5/5] Fine-tuning selected candidates...")
    finetune_results = []

    for cidx, cand in enumerate(finetune_candidates):
        label = cand['label']
        config = {}
        for key_str, rank in cand['config'].items():
            parts = key_str.split('_', 1)
            layer_idx = int(parts[0])
            mod = parts[1]
            config[(layer_idx, mod)] = rank

        print(f"\n  [{cidx+1}/{len(finetune_candidates)}] Fine-tuning: {label} "
              f"(params={cand['n_params']:,}, proxy={cand.get('combined_score', 0):.3f})")

        model = load_vit_b16(num_classes=num_classes)
        model, lora_mods = apply_lora_config(model, config)

        val_acc = finetune_and_evaluate(
            model, train_loader, val_loader, device=device,
            epochs=3, lr=1e-3, label=label
        )

        latency = estimate_latency(model, device=device)

        result = {
            'label': label,
            'n_params': cand['n_params'],
            'gradnorm': cand['gradnorm'],
            'snip': cand['snip'],
            'fisher': cand['fisher'],
            'entropy': cand['entropy'],
            'combined_proxy': cand.get('combined_score', 0),
            'val_accuracy': val_acc,
            'latency_ms': latency,
        }
        finetune_results.append(result)
        print(f"  -> Val Accuracy: {val_acc:.2f}%, Latency: {latency:.1f}ms")

        del model, lora_mods

    # Also evaluate no-adapter baseline (linear probe)
    print("\n  Fine-tuning linear probe baseline (no adapters)...")
    model = load_vit_b16(num_classes=num_classes)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True

    val_acc_baseline = finetune_and_evaluate(
        model, train_loader, val_loader, device=device,
        epochs=3, lr=1e-3, label='linear_probe'
    )
    latency_baseline = estimate_latency(model, device=device)

    finetune_results.append({
        'label': 'linear_probe',
        'n_params': sum(p.numel() for p in model.head.parameters()),
        'gradnorm': 0, 'snip': 0, 'fisher': 0, 'entropy': 0,
        'combined_proxy': 0,
        'val_accuracy': val_acc_baseline,
        'latency_ms': latency_baseline,
    })
    print(f"  -> Linear probe baseline: {val_acc_baseline:.2f}%")

    del model

    # Save results
    results_file = os.path.join(results_dir, f'{dataset_name}_finetune_results.json')
    with open(results_file, 'w') as f:
        json.dump(finetune_results, f, indent=2)
    print(f"\nSaved fine-tuning results to {results_file}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY: {dataset_name}")
    print(f"{'='*60}")
    finetune_results.sort(key=lambda x: x['val_accuracy'], reverse=True)
    print(f"{'Label':<25} {'Params':>10} {'ValAcc':>8} {'Latency':>10} {'Proxy':>8}")
    print(f"{'-'*65}")
    for r in finetune_results:
        print(f"{r['label']:<25} {r['n_params']:>10,} {r['val_accuracy']:>7.2f}% {r['latency_ms']:>8.1f}ms {r['combined_proxy']:>7.3f}")

    return all_results, finetune_results


if __name__ == '__main__':
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Run on CIFAR-100
    proxy_results_c100, ft_results_c100 = run_search_experiment(
        dataset_name='cifar100', n_random=50, device=device,
        results_dir='../results'
    )

    # Run on Flowers-102
    proxy_results_f102, ft_results_f102 = run_search_experiment(
        dataset_name='flowers102', n_random=50, device=device,
        results_dir='../results'
    )

    print("\n\nAll experiments completed!")
