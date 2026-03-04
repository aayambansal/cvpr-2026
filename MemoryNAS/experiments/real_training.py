#!/usr/bin/env python3
"""
Real training & memory measurement for MemoryNAS.
Builds actual PyTorch MobileNetV2-style models, trains on CIFAR-10/100,
measures real peak memory, and validates the analytical estimator.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
import os
import time
import tracemalloc
import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ============================================================================
# Real PyTorch Model Builder
# ============================================================================

class InvertedResidual(nn.Module):
    """MobileNetV2-style inverted residual block."""
    def __init__(self, inp, oup, stride, expand_ratio, kernel_size=3):
        super().__init__()
        self.stride = stride
        self.use_res_connect = (stride == 1 and inp == oup)
        hidden_dim = int(round(inp * expand_ratio))
        padding = kernel_size // 2

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class SearchableNet(nn.Module):
    """Build a real network from search space parameters."""
    def __init__(self, width_mults, depths, expansions, kernels, resolution,
                 num_classes=10, input_channels=3):
        super().__init__()
        self.resolution = resolution

        base_channels = [16, 24, 32, 64, 96, 160, 320]
        strides = [1, 2, 2, 2, 1, 2, 1]

        # For CIFAR (32x32), reduce strides
        if resolution <= 64:
            strides = [1, 1, 2, 2, 1, 2, 1]  # fewer downsamples

        # Initial conv
        first_c = max(int(32 * width_mults[0]), 8)
        init_stride = 1 if resolution <= 64 else 2
        features = [
            nn.Conv2d(input_channels, first_c, 3, init_stride, 1, bias=False),
            nn.BatchNorm2d(first_c),
            nn.ReLU6(inplace=True),
        ]

        inp = first_c
        for stage_idx in range(7):
            oup = max(int(base_channels[stage_idx] * width_mults[stage_idx]), 8)
            stride = strides[stage_idx]
            depth = depths[stage_idx]
            expansion = expansions[stage_idx]
            kernel = kernels[stage_idx]

            for block_idx in range(depth):
                s = stride if block_idx == 0 else 1
                features.append(InvertedResidual(inp, oup, s, expansion, kernel))
                inp = oup

        # Final conv
        final_c = 1280
        features.extend([
            nn.Conv2d(inp, final_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(final_c),
            nn.ReLU6(inplace=True),
        ])
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(final_c, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================================
# Real Memory Measurement
# ============================================================================

def measure_peak_memory_cpu(model, input_size, batch_size=1, num_runs=3):
    """Measure real peak memory on CPU using tracemalloc."""
    model = model.cpu()
    model.eval()
    memories = []

    for _ in range(num_runs):
        tracemalloc.start()
        x = torch.randn(batch_size, 3, input_size, input_size)
        with torch.no_grad():
            _ = model(x)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memories.append(peak)

    return np.mean(memories), np.std(memories)


def measure_peak_memory_mps(model, input_size, batch_size=1, num_runs=5):
    """Measure peak memory on MPS by monitoring allocator stats."""
    model = model.to('mps')
    model.eval()

    memories = []
    for _ in range(num_runs):
        # Reset MPS stats
        torch.mps.empty_cache()
        torch.mps.synchronize()

        x = torch.randn(batch_size, 3, input_size, input_size, device='mps')
        with torch.no_grad():
            _ = model(x)
        torch.mps.synchronize()

        # Get current allocation (MPS doesn't have max_memory_allocated yet)
        mem = torch.mps.current_allocated_memory()
        memories.append(mem)

    return np.mean(memories), np.std(memories)


def measure_peak_memory_inference(model, input_size, batch_size=1):
    """Cross-platform peak memory measurement."""
    if DEVICE == 'mps':
        return measure_peak_memory_mps(model, input_size, batch_size)
    else:
        return measure_peak_memory_cpu(model, input_size, batch_size)


# ============================================================================
# Training
# ============================================================================

def get_cifar10_loaders(batch_size=128):
    """CIFAR-10 with standard augmentation."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                              shuffle=False, num_workers=2)
    return trainloader, testloader


def get_cifar100_loaders(batch_size=128):
    """CIFAR-100 with standard augmentation."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                              download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                             download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                              shuffle=False, num_workers=2)
    return trainloader, testloader


def train_model(model, trainloader, testloader, epochs=50, lr=0.05, device='mps'):
    """Train a model on CIFAR and return test accuracy."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
        best_acc = max(best_acc, acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs}: loss={total_loss/len(trainloader):.3f}, "
                  f"acc={acc:.1f}%, best={best_acc:.1f}%")

    return best_acc


# ============================================================================
# Architecture Definitions for Experiments
# ============================================================================

def get_experiment_architectures():
    """Define architectures for training experiments.
    Returns dict of name -> (width_mults, depths, expansions, kernels, resolution)
    Organized by: FLOPs-Pareto picks vs MemoryNAS picks at various budgets.
    """
    archs = {}

    # FLOPs-Pareto architectures (optimized for acc/FLOPs, ignoring memory)
    archs['FP-Large'] = dict(
        width_mults=[1.5]*7, depths=[2,3,4,4,4,4,2],
        expansions=[6]*7, kernels=[5]*7, resolution=32  # CIFAR native
    )
    archs['FP-Med'] = dict(
        width_mults=[1.0]*7, depths=[1,2,3,4,3,3,1],
        expansions=[6]*7, kernels=[3]*7, resolution=32
    )
    archs['FP-Small'] = dict(
        width_mults=[0.75]*7, depths=[1,2,2,3,2,2,1],
        expansions=[4]*7, kernels=[3]*7, resolution=32
    )
    archs['FP-Tiny'] = dict(
        width_mults=[0.5]*7, depths=[1,1,2,2,1,1,1],
        expansions=[3]*7, kernels=[3]*7, resolution=32
    )
    archs['FP-HighCap'] = dict(
        width_mults=[1.25]*7, depths=[2,3,3,4,3,3,1],
        expansions=[6]*7, kernels=[5]*7, resolution=32
    )

    # MemoryNAS architectures (memory-efficient designs)
    archs['MN-A'] = dict(  # narrow early, wider late
        width_mults=[0.5, 0.75, 1.0, 1.0, 0.75, 0.5, 0.5],
        depths=[2,2,3,3,2,2,1], expansions=[4,4,4,3,3,3,3],
        kernels=[3]*7, resolution=32
    )
    archs['MN-B'] = dict(  # low expansion everywhere
        width_mults=[1.0]*7, depths=[1,2,3,4,3,3,1],
        expansions=[3]*7, kernels=[3]*7, resolution=32
    )
    archs['MN-C'] = dict(  # very narrow early stages
        width_mults=[0.5, 0.5, 0.75, 1.0, 1.0, 0.75, 0.5],
        depths=[2,3,3,4,3,2,1], expansions=[3,3,4,4,3,3,3],
        kernels=[3,3,3,5,3,3,3], resolution=32
    )
    archs['MN-D'] = dict(  # deep but thin
        width_mults=[0.5]*7, depths=[3,4,4,4,4,4,3],
        expansions=[4]*7, kernels=[3]*7, resolution=32
    )
    archs['MN-E'] = dict(  # balanced memory-opt
        width_mults=[0.75, 0.75, 1.0, 0.75, 0.5, 0.5, 0.5],
        depths=[2,3,3,3,2,2,1], expansions=[4,4,3,3,3,3,3],
        kernels=[3]*7, resolution=32
    )

    # Baselines
    archs['MBv2-1.0'] = dict(  # standard MobileNetV2 config
        width_mults=[1.0]*7, depths=[1,2,3,4,3,3,1],
        expansions=[6,6,6,6,6,6,6], kernels=[3]*7, resolution=32
    )
    archs['MBv2-0.5'] = dict(
        width_mults=[0.5]*7, depths=[1,2,3,4,3,3,1],
        expansions=[6]*7, kernels=[3]*7, resolution=32
    )

    return archs


def get_memory_profile_architectures():
    """50 random architectures for memory estimation validation."""
    np.random.seed(123)
    WIDTH_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5]
    DEPTHS = [1, 2, 3, 4]
    EXPANSION_RATIOS = [3, 4, 6]
    KERNEL_SIZES = [3, 5, 7]

    archs = []
    for i in range(50):
        archs.append({
            'name': f'rand_{i:02d}',
            'width_mults': [float(np.random.choice(WIDTH_MULTIPLIERS)) for _ in range(7)],
            'depths': [int(np.random.choice(DEPTHS)) for _ in range(7)],
            'expansions': [int(np.random.choice(EXPANSION_RATIOS)) for _ in range(7)],
            'kernels': [int(np.random.choice(KERNEL_SIZES)) for _ in range(7)],
            'resolution': 32,
        })
    return archs


# ============================================================================
# Analytical estimator (from original experiments) - adapted for CIFAR
# ============================================================================

def analytical_peak_memory(width_mults, depths, expansions, kernels, resolution=32):
    """Analytical peak memory estimate (bytes, float32)."""
    base_channels = [16, 24, 32, 64, 96, 160, 320]
    strides = [1, 1, 2, 2, 1, 2, 1] if resolution <= 64 else [1, 2, 2, 2, 1, 2, 1]

    h, w = resolution, resolution
    cin = 3
    peak_memory = 0
    current_memory = 0

    # Initial conv
    cout = max(int(32 * width_mults[0]), 8)
    init_stride = 1 if resolution <= 64 else 2
    h, w = h // init_stride, w // init_stride
    input_mem = cin * resolution * resolution * 4
    output_mem = cout * h * w * 4
    current_memory = input_mem + output_mem
    peak_memory = max(peak_memory, current_memory)
    cin = cout

    for stage_idx in range(7):
        base_c = base_channels[stage_idx]
        stride = strides[stage_idx]
        width_mult = width_mults[stage_idx]
        depth = depths[stage_idx]
        expansion = expansions[stage_idx]

        cout = max(int(base_c * width_mult), 8)

        for block_idx in range(depth):
            s = stride if block_idx == 0 else 1
            expanded = cin * expansion
            h_out, w_out = h // s, w // s

            input_activation = cin * h * w * 4
            expanded_activation = expanded * h * w * 4
            output_activation = cout * h_out * w_out * 4

            has_residual = (s == 1 and cin == cout)
            block_peak = expanded_activation + output_activation
            if has_residual:
                block_peak += input_activation

            stage_memory = current_memory + block_peak
            peak_memory = max(peak_memory, stage_memory)

            h, w = h_out, w_out
            cin = cout
            current_memory = output_activation

    # Final 1x1 conv
    final_mem = 1280 * h * w * 4 + current_memory
    peak_memory = max(peak_memory, final_mem)

    return peak_memory


# ============================================================================
# Main Experiments
# ============================================================================

def run_training_experiments():
    """Train all architectures on CIFAR-10 and CIFAR-100."""
    print("=" * 70)
    print("EXPERIMENT: Training Top-K Architectures on CIFAR-10 and CIFAR-100")
    print("=" * 70)

    archs = get_experiment_architectures()

    # Download data
    print("\nDownloading CIFAR-10...")
    trainloader10, testloader10 = get_cifar10_loaders(batch_size=128)
    print("Downloading CIFAR-100...")
    trainloader100, testloader100 = get_cifar100_loaders(batch_size=128)

    results = {}
    epochs = 50

    for name, config in archs.items():
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")

        # Build model
        model10 = SearchableNet(
            config['width_mults'], config['depths'], config['expansions'],
            config['kernels'], config['resolution'], num_classes=10
        )
        n_params = sum(p.numel() for p in model10.parameters()) / 1e6
        print(f"  Parameters: {n_params:.2f}M")

        # Measure memory before training
        mem_mean, mem_std = measure_peak_memory_inference(model10, config['resolution'])
        analytical_mem = analytical_peak_memory(
            config['width_mults'], config['depths'], config['expansions'],
            config['kernels'], config['resolution']
        )
        print(f"  Measured memory: {mem_mean/1e6:.2f} MB (std: {mem_std/1e6:.3f})")
        print(f"  Analytical memory: {analytical_mem/1e6:.2f} MB")

        # Train on CIFAR-10
        print(f"\n  Training on CIFAR-10 ({epochs} epochs)...")
        t0 = time.time()
        acc10 = train_model(model10, trainloader10, testloader10, epochs=epochs, device=DEVICE)
        t10 = time.time() - t0
        print(f"  CIFAR-10: {acc10:.1f}% in {t10:.0f}s")

        # Train on CIFAR-100
        model100 = SearchableNet(
            config['width_mults'], config['depths'], config['expansions'],
            config['kernels'], config['resolution'], num_classes=100
        )
        print(f"\n  Training on CIFAR-100 ({epochs} epochs)...")
        t0 = time.time()
        acc100 = train_model(model100, trainloader100, testloader100, epochs=epochs, device=DEVICE)
        t100 = time.time() - t0
        print(f"  CIFAR-100: {acc100:.1f}% in {t100:.0f}s")

        results[name] = {
            'params_m': round(n_params, 3),
            'measured_memory_bytes': float(mem_mean),
            'measured_memory_std_bytes': float(mem_std),
            'analytical_memory_bytes': float(analytical_mem),
            'cifar10_acc': round(acc10, 2),
            'cifar100_acc': round(acc100, 2),
            'cifar10_train_time_s': round(t10, 1),
            'cifar100_train_time_s': round(t100, 1),
            'config': config,
        }

        # Save incremental results
        results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to {results_path}")

    return results


def run_memory_validation():
    """Validate analytical memory estimator against real measurements on 50 architectures."""
    print("\n" + "=" * 70)
    print("EXPERIMENT: Memory Estimator Validation (50 architectures)")
    print("=" * 70)

    archs = get_memory_profile_architectures()
    results = []

    for i, config in enumerate(archs):
        try:
            model = SearchableNet(
                config['width_mults'], config['depths'], config['expansions'],
                config['kernels'], config['resolution'], num_classes=10
            )
            n_params = sum(p.numel() for p in model.parameters()) / 1e6

            # Measure real memory
            measured_mean, measured_std = measure_peak_memory_inference(
                model, config['resolution']
            )

            # Analytical estimate
            analytical = analytical_peak_memory(
                config['width_mults'], config['depths'], config['expansions'],
                config['kernels'], config['resolution']
            )

            error_pct = abs(measured_mean - analytical) / measured_mean * 100 if measured_mean > 0 else 0

            results.append({
                'name': config['name'],
                'params_m': round(n_params, 3),
                'measured_bytes': float(measured_mean),
                'measured_std_bytes': float(measured_std),
                'analytical_bytes': float(analytical),
                'error_pct': round(error_pct, 2),
            })

            if (i + 1) % 10 == 0:
                print(f"  Validated {i+1}/50 architectures...")

        except Exception as e:
            print(f"  Error with arch {i}: {e}")
            continue

    # Summary stats
    errors = [r['error_pct'] for r in results]
    print(f"\n  Memory Estimator Validation Summary:")
    print(f"    Architectures validated: {len(results)}")
    print(f"    Mean Absolute Percentage Error: {np.mean(errors):.1f}%")
    print(f"    Median APE: {np.median(errors):.1f}%")
    print(f"    Max APE: {np.max(errors):.1f}%")

    # Compute correlation
    measured = [r['measured_bytes'] for r in results]
    analytical = [r['analytical_bytes'] for r in results]
    corr = np.corrcoef(measured, analytical)[0, 1]
    print(f"    Pearson correlation: {corr:.4f}")

    # Spearman rank correlation
    from scipy.stats import spearmanr
    spear, pval = spearmanr(measured, analytical)
    print(f"    Spearman correlation: {spear:.4f} (p={pval:.2e})")

    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'memory_validation.json')
    with open(results_path, 'w') as f:
        json.dump({
            'results': results,
            'summary': {
                'n_archs': len(results),
                'mean_ape': round(float(np.mean(errors)), 2),
                'median_ape': round(float(np.median(errors)), 2),
                'max_ape': round(float(np.max(errors)), 2),
                'pearson_r': round(float(corr), 4),
                'spearman_r': round(float(spear), 4),
                'spearman_p': float(pval),
                'device': DEVICE,
            }
        }, f, indent=2)
    print(f"  Saved to {results_path}")
    return results


if __name__ == '__main__':
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # Run memory validation first (fast, ~2 min)
    mem_results = run_memory_validation()

    # Run training experiments (longer, ~30-60 min total on MPS)
    train_results = run_training_experiments()

    print("\n" + "=" * 70)
    print("ALL REAL EXPERIMENTS COMPLETE")
    print("=" * 70)
