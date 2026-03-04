"""
MemoryNAS: Complete GPU experiment suite on Modal.
Trains real architectures, measures real GPU memory, runs all baselines & ablations.
"""

import modal
import json

app = modal.App("memorynas-experiments")

volume = modal.Volume.from_name("memorynas-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch", "torchvision", "numpy", "scipy", "pymoo"
    )
)

# ============================================================================
# Architecture builder + profiling code (runs inside Modal container)
# ============================================================================

ARCH_CODE = """
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import json
import os

torch.manual_seed(42)
np.random.seed(42)

# ---------- Model Definition ----------

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, kernel_size=3):
        super().__init__()
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
    def __init__(self, width_mults, depths, expansions, kernels, num_classes=10):
        super().__init__()
        base_channels = [16, 24, 32, 64, 96, 160, 320]
        strides = [1, 1, 2, 2, 1, 2, 1]  # CIFAR-adapted

        first_c = max(int(32 * width_mults[0]), 8)
        features = [
            nn.Conv2d(3, first_c, 3, 1, 1, bias=False),
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
        features.extend([
            nn.Conv2d(inp, 1280, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
        ])
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------- Memory Measurement ----------

def measure_gpu_peak_memory(model, input_size=32, batch_sizes=[1, 8, 32]):
    device = torch.device('cuda')
    model = model.to(device)
    model.eval()
    
    # Measure baseline weight memory
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    weight_mem = torch.cuda.memory_allocated()
    
    results = {}
    results_activation_only = {}
    for bs in batch_sizes:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        x = torch.randn(bs, 3, input_size, input_size, device=device)
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        results[bs] = peak
        # Activation-only = peak - weight baseline (approximate)
        results_activation_only[bs] = max(0, peak - weight_mem)
        del x
    return results, results_activation_only, weight_mem


# ---------- Analytical Estimator ----------

def analytical_peak_memory(width_mults, depths, expansions, kernels, resolution=32):
    base_channels = [16, 24, 32, 64, 96, 160, 320]
    strides = [1, 1, 2, 2, 1, 2, 1]
    h, w = resolution, resolution
    cin = 3
    peak_memory = 0
    current_memory = 0
    cout = max(int(32 * width_mults[0]), 8)
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
    final_mem = 1280 * h * w * 4 + current_memory
    peak_memory = max(peak_memory, final_mem)
    return peak_memory


# ---------- FLOPs Counter ----------

def count_flops(width_mults, depths, expansions, kernels, resolution=32):
    base_channels = [16, 24, 32, 64, 96, 160, 320]
    strides = [1, 1, 2, 2, 1, 2, 1]
    h, w = resolution, resolution
    cin = 3
    total_flops = 0
    cout = max(int(32 * width_mults[0]), 8)
    total_flops += 2 * 3 * 3 * cin * cout * h * w
    cin = cout
    for stage_idx in range(7):
        base_c = base_channels[stage_idx]
        stride = strides[stage_idx]
        cout = max(int(base_c * width_mults[stage_idx]), 8)
        for block_idx in range(depths[stage_idx]):
            s = stride if block_idx == 0 else 1
            expanded = cin * expansions[stage_idx]
            h_out, w_out = h // s, w // s
            k = kernels[stage_idx]
            total_flops += 2 * cin * expanded * h * w  # pw1
            total_flops += 2 * k * k * expanded * h_out * w_out  # dw
            total_flops += 2 * expanded * cout * h_out * w_out  # pw2
            h, w = h_out, w_out
            cin = cout
    total_flops += 2 * cin * 1280 * h * w
    total_flops += 2 * 1280 * 1000
    return total_flops


# ---------- Training ----------

def get_loaders(dataset='cifar10', batch_size=128):
    if dataset == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        DS = torchvision.datasets.CIFAR10
    else:
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        DS = torchvision.datasets.CIFAR100
    tr = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    trainset = DS(root='/data/cifar', train=True, download=True, transform=tr)
    testset = DS(root='/data/cifar', train=False, download=True, transform=te)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                              shuffle=False, num_workers=4, pin_memory=True)
    return trainloader, testloader


def train_and_eval(model, trainloader, testloader, epochs=50, lr=0.05):
    device = torch.device('cuda')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda')
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
        best_acc = max(best_acc, acc)
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: acc={acc:.1f}%, best={best_acc:.1f}%")
    return best_acc
"""


# ============================================================================
# Architecture definitions
# ============================================================================

ARCHITECTURES = {
    # FLOPs-Pareto picks (high FLOPs, ignoring memory)
    "FP-Large": dict(width_mults=[1.5]*7, depths=[2,3,4,4,4,4,2], expansions=[6]*7, kernels=[5]*7),
    "FP-Med": dict(width_mults=[1.0]*7, depths=[1,2,3,4,3,3,1], expansions=[6]*7, kernels=[3]*7),
    "FP-Small": dict(width_mults=[0.75]*7, depths=[1,2,2,3,2,2,1], expansions=[4]*7, kernels=[3]*7),
    "FP-Tiny": dict(width_mults=[0.5]*7, depths=[1,1,2,2,1,1,1], expansions=[3]*7, kernels=[3]*7),
    "FP-HighCap": dict(width_mults=[1.25]*7, depths=[2,3,3,4,3,3,1], expansions=[6]*7, kernels=[5]*7),
    "FP-Wide": dict(width_mults=[1.5]*7, depths=[1,1,2,2,1,1,1], expansions=[6]*7, kernels=[7]*7),
    # MemoryNAS picks (memory-optimized)
    "MN-A": dict(width_mults=[0.5,0.75,1.0,1.0,0.75,0.5,0.5], depths=[2,2,3,3,2,2,1], expansions=[4,4,4,3,3,3,3], kernels=[3]*7),
    "MN-B": dict(width_mults=[1.0]*7, depths=[1,2,3,4,3,3,1], expansions=[3]*7, kernels=[3]*7),
    "MN-C": dict(width_mults=[0.5,0.5,0.75,1.0,1.0,0.75,0.5], depths=[2,3,3,4,3,2,1], expansions=[3,3,4,4,3,3,3], kernels=[3,3,3,5,3,3,3]),
    "MN-D": dict(width_mults=[0.5]*7, depths=[3,4,4,4,4,4,3], expansions=[4]*7, kernels=[3]*7),
    "MN-E": dict(width_mults=[0.75,0.75,1.0,0.75,0.5,0.5,0.5], depths=[2,3,3,3,2,2,1], expansions=[4,4,3,3,3,3,3], kernels=[3]*7),
    "MN-F": dict(width_mults=[0.5,0.75,0.75,0.75,0.5,0.5,0.5], depths=[2,2,3,3,3,2,1], expansions=[3,3,3,4,3,3,3], kernels=[3]*7),
    # Baselines
    "MBv2-1.0": dict(width_mults=[1.0]*7, depths=[1,2,3,4,3,3,1], expansions=[6]*7, kernels=[3]*7),
    "MBv2-0.5": dict(width_mults=[0.5]*7, depths=[1,2,3,4,3,3,1], expansions=[6]*7, kernels=[3]*7),
}


# ============================================================================
# Main training function - runs on GPU
# ============================================================================

@app.function(
    gpu="A10G",
    image=image,
    volumes={"/results": volume},
    timeout=14400,  # 4 hours max
    memory=16384,
)
def train_all():
    """Train all architectures on CIFAR-10 and CIFAR-100, measure GPU memory."""
    exec(ARCH_CODE, globals())

    # Resume: load existing results if any
    all_results = {}
    try:
        volume.reload()
        with open('/results/training_results.json') as f:
            all_results = json.load(f)
        print(f"Resuming: {len(all_results)} architectures already done")
    except:
        print("Starting fresh")

    print("=" * 70)
    print("PHASE 1: Training all architectures")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 70)

    for name, config in ARCHITECTURES.items():
        print(f"\n{'='*60}")
        print(f"Architecture: {name}")
        print(f"{'='*60}")

        # Skip if already trained
        if name in all_results:
            print(f"  Already trained, skipping.")
            continue

        # Build model
        model10 = SearchableNet(
            config['width_mults'], config['depths'], config['expansions'],
            config['kernels'], num_classes=10
        )
        n_params = sum(p.numel() for p in model10.parameters()) / 1e6
        print(f"  Params: {n_params:.2f}M")

        # Measure real GPU peak memory
        mem_total, mem_act, weight_mem = measure_gpu_peak_memory(model10, input_size=32, batch_sizes=[1, 8, 32])
        analytical_mem = analytical_peak_memory(
            config['width_mults'], config['depths'], config['expansions'],
            config['kernels'], resolution=32
        )
        flops = count_flops(
            config['width_mults'], config['depths'], config['expansions'],
            config['kernels'], resolution=32
        )

        print(f"  FLOPs: {flops/1e6:.1f}M")
        print(f"  Weight Memory: {weight_mem/1e6:.2f} MB")
        print(f"  GPU Peak Total (bs=1): {mem_total[1]/1e6:.2f} MB")
        print(f"  GPU Peak Activations (bs=1): {mem_act[1]/1e6:.2f} MB")
        print(f"  GPU Peak Total (bs=32): {mem_total[32]/1e6:.2f} MB")
        print(f"  Analytical Estimate (activations): {analytical_mem/1e6:.2f} MB")

        # Train CIFAR-10
        print(f"\n  Training CIFAR-10 (50 epochs)...")
        trainloader10, testloader10 = get_loaders('cifar10', batch_size=128)
        t0 = time.time()
        acc10 = train_and_eval(model10, trainloader10, testloader10, epochs=50)
        t10 = time.time() - t0
        print(f"  CIFAR-10 Best Acc: {acc10:.2f}% ({t10:.0f}s)")

        # Train CIFAR-100
        model100 = SearchableNet(
            config['width_mults'], config['depths'], config['expansions'],
            config['kernels'], num_classes=100
        )
        print(f"\n  Training CIFAR-100 (50 epochs)...")
        trainloader100, testloader100 = get_loaders('cifar100', batch_size=128)
        t0 = time.time()
        acc100 = train_and_eval(model100, trainloader100, testloader100, epochs=50)
        t100 = time.time() - t0
        print(f"  CIFAR-100 Best Acc: {acc100:.2f}% ({t100:.0f}s)")

        all_results[name] = {
            'params_m': round(n_params, 3),
            'flops_m': round(flops / 1e6, 1),
            'gpu_peak_total_bs1': mem_total[1],
            'gpu_peak_total_bs8': mem_total[8],
            'gpu_peak_total_bs32': mem_total[32],
            'gpu_peak_act_bs1': mem_act[1],
            'gpu_peak_act_bs8': mem_act[8],
            'gpu_peak_act_bs32': mem_act[32],
            'weight_memory': weight_mem,
            'analytical_memory': analytical_mem,
            'cifar10_acc': round(acc10, 2),
            'cifar100_acc': round(acc100, 2),
            'train_time_c10': round(t10, 1),
            'train_time_c100': round(t100, 1),
            'config': config,
        }

        # Save incrementally
        with open('/results/training_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        volume.commit()
        print(f"  Saved checkpoint.")

    return all_results


# ============================================================================
# Memory validation on 50 random architectures
# ============================================================================

@app.function(
    gpu="A10G",
    image=image,
    volumes={"/results": volume},
    timeout=3600,
    memory=8192,
)
def validate_memory_estimator():
    """Validate analytical memory estimator against real GPU measurements."""
    exec(ARCH_CODE, globals())

    print("=" * 70)
    print("PHASE 2: Memory Estimator Validation (50 random architectures)")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    WIDTH_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5]
    DEPTHS = [1, 2, 3, 4]
    EXPANSION_RATIOS = [3, 4, 6]
    KERNEL_SIZES = [3, 5, 7]

    rng = np.random.RandomState(123)
    results = []

    for i in range(50):
        wm = [float(rng.choice(WIDTH_MULTIPLIERS)) for _ in range(7)]
        dp = [int(rng.choice(DEPTHS)) for _ in range(7)]
        ex = [int(rng.choice(EXPANSION_RATIOS)) for _ in range(7)]
        ks = [int(rng.choice(KERNEL_SIZES)) for _ in range(7)]

        try:
            model = SearchableNet(wm, dp, ex, ks, num_classes=10)
            n_params = sum(p.numel() for p in model.parameters()) / 1e6

            # Measure real GPU memory at batch_size=1
            mem_total, mem_act, weight_mem = measure_gpu_peak_memory(model, input_size=32, batch_sizes=[1])
            measured_total = mem_total[1]
            measured_act = mem_act[1]

            # Analytical estimate (activation-only)
            analytical = analytical_peak_memory(wm, dp, ex, ks, resolution=32)

            # Weight memory (for total estimator = analytical + weight_mem_estimate)
            weight_mem_estimate = sum(p.numel() * 4 for p in model.parameters())

            # FLOPs
            flops = count_flops(wm, dp, ex, ks, resolution=32)

            results.append({
                'idx': i,
                'params_m': round(n_params, 3),
                'flops_m': round(flops / 1e6, 1),
                'measured_total_bytes': int(measured_total),
                'measured_act_bytes': int(measured_act),
                'weight_mem_bytes': int(weight_mem),
                'weight_mem_estimate_bytes': int(weight_mem_estimate),
                'analytical_act_bytes': int(analytical),
                'analytical_total_bytes': int(analytical + weight_mem_estimate),
                'measured_total_mb': round(measured_total / 1e6, 3),
                'measured_act_mb': round(measured_act / 1e6, 3),
                'analytical_act_mb': round(analytical / 1e6, 3),
                'analytical_total_mb': round((analytical + weight_mem_estimate) / 1e6, 3),
            })

            if (i + 1) % 10 == 0:
                print(f"  Validated {i+1}/50")

        except Exception as e:
            print(f"  Error with arch {i}: {e}")
            continue

    # Compute stats for BOTH total memory and activation-only memory
    from scipy.stats import spearmanr, pearsonr
    
    def compute_stats(measured_arr, analytical_arr, label):
        errors_pct = np.abs(measured_arr - analytical_arr) / np.maximum(measured_arr, 1) * 100
        pr, _ = pearsonr(measured_arr, analytical_arr)
        sr, sp = spearmanr(measured_arr, analytical_arr)
        ss_res = np.sum((measured_arr - analytical_arr) ** 2)
        ss_tot = np.sum((measured_arr - np.mean(measured_arr)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        stats = {
            'mean_ape': round(float(np.mean(errors_pct)), 2),
            'median_ape': round(float(np.median(errors_pct)), 2),
            'max_ape': round(float(np.max(errors_pct)), 2),
            'pearson_r': round(float(pr), 4),
            'spearman_r': round(float(sr), 4),
            'spearman_p': float(sp),
            'r_squared': round(float(r2), 4),
        }
        print(f"\n  === {label} ===")
        print(f"  Mean APE: {stats['mean_ape']:.1f}%")
        print(f"  Pearson r: {stats['pearson_r']:.4f}")
        print(f"  Spearman r: {stats['spearman_r']:.4f}")
        print(f"  R²: {stats['r_squared']:.4f}")
        return stats
    
    # Total memory: measured_total vs analytical_total (activations + weights)
    meas_total = np.array([r['measured_total_bytes'] for r in results])
    anal_total = np.array([r['analytical_total_bytes'] for r in results])
    stats_total = compute_stats(meas_total, anal_total, "Total Memory (activations + weights)")
    
    # Activation-only memory: measured_act vs analytical_act
    meas_act = np.array([r['measured_act_bytes'] for r in results])
    anal_act = np.array([r['analytical_act_bytes'] for r in results])
    stats_act = compute_stats(meas_act, anal_act, "Activation-Only Memory")
    
    # Ranking correlation: does our estimator rank architectures correctly?
    # This is the most important metric for NAS
    rank_sr, rank_sp = spearmanr(meas_total, anal_total)
    print(f"\n  === Ranking Correlation (most important for NAS) ===")
    print(f"  Spearman rank correlation (total): {rank_sr:.4f} (p={rank_sp:.2e})")

    summary = {
        'n_archs': len(results),
        'total_memory_stats': stats_total,
        'activation_memory_stats': stats_act,
        'ranking_spearman': round(float(rank_sr), 4),
        'ranking_p_value': float(rank_sp),
        'device': torch.cuda.get_device_name(0),
    }

    output = {'results': results, 'summary': summary}
    with open('/results/memory_validation.json', 'w') as f:
        json.dump(output, f, indent=2)
    volume.commit()

    return output


# ============================================================================
# Baselines & Ablations (runs on CPU, no GPU needed)
# ============================================================================

@app.function(
    image=image,
    volumes={"/results": volume},
    timeout=3600,
    memory=8192,
    cpu=4,
)
def run_baselines_and_ablations():
    """Run all baseline search methods and ablation experiments."""
    import numpy as np
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.indicators.hv import HV

    np.random.seed(42)

    exec(ARCH_CODE, globals())

    WIDTH_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5]
    DEPTHS_LIST = [1, 2, 3, 4]
    EXPANSION_RATIOS = [3, 4, 6]
    KERNEL_SIZES = [3, 5, 7]

    # ---------- Proxy accuracy model ----------
    def proxy_accuracy(wm, dp, ex, ks, resolution=32):
        flops = count_flops(wm, dp, ex, ks, resolution)
        base_acc = 55 + 8.5 * np.log2(flops / 1e8 + 1)
        res_bonus = 3.0 * np.log2(resolution / 128 + 1)
        depth_bonus = 1.5 * np.log2(sum(dp) / 7 + 1)
        width_bonus = 2.0 * (np.mean(wm) - 0.5)
        kernel_bonus = 0.5 * (np.mean(ks) - 3) / 4
        expansion_bonus = 0.3 * (np.mean(ex) - 3) / 3
        accuracy = base_acc + res_bonus + depth_bonus + width_bonus + kernel_bonus + expansion_bonus
        accuracy = np.clip(accuracy, 50, 83) + np.random.normal(0, 0.3)
        return np.clip(accuracy, 45, 84)

    # ---------- Problem definitions ----------
    class ThreeObjProblem(ElementwiseProblem):
        def __init__(self, memory_constraint=None):
            n_constr = 1 if memory_constraint else 0
            self.memory_constraint = memory_constraint
            super().__init__(n_var=29, n_obj=3, n_ieq_constr=n_constr,
                           xl=np.zeros(29), xu=np.ones(29))

        def _decode(self, x):
            wm = [WIDTH_MULTIPLIERS[int(x[i]*(len(WIDTH_MULTIPLIERS)-0.01))] for i in range(7)]
            dp = [DEPTHS_LIST[int(x[7+i]*(len(DEPTHS_LIST)-0.01))] for i in range(7)]
            ex = [EXPANSION_RATIOS[int(x[14+i]*(len(EXPANSION_RATIOS)-0.01))] for i in range(7)]
            ks = [KERNEL_SIZES[int(x[21+i]*(len(KERNEL_SIZES)-0.01))] for i in range(7)]
            return wm, dp, ex, ks

        def _evaluate(self, x, out, *args, **kwargs):
            wm, dp, ex, ks = self._decode(x)
            acc = proxy_accuracy(wm, dp, ex, ks)
            flops = count_flops(wm, dp, ex, ks) / 1e9
            mem = analytical_peak_memory(wm, dp, ex, ks) / 1e6
            out["F"] = [-acc, flops, mem]
            if self.memory_constraint:
                out["G"] = [mem - self.memory_constraint]

    class TwoObjProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=29, n_obj=2, xl=np.zeros(29), xu=np.ones(29))
        def _decode(self, x):
            wm = [WIDTH_MULTIPLIERS[int(x[i]*(len(WIDTH_MULTIPLIERS)-0.01))] for i in range(7)]
            dp = [DEPTHS_LIST[int(x[7+i]*(len(DEPTHS_LIST)-0.01))] for i in range(7)]
            ex = [EXPANSION_RATIOS[int(x[14+i]*(len(EXPANSION_RATIOS)-0.01))] for i in range(7)]
            ks = [KERNEL_SIZES[int(x[21+i]*(len(KERNEL_SIZES)-0.01))] for i in range(7)]
            return wm, dp, ex, ks
        def _evaluate(self, x, out, *args, **kwargs):
            wm, dp, ex, ks = self._decode(x)
            acc = proxy_accuracy(wm, dp, ex, ks)
            flops = count_flops(wm, dp, ex, ks) / 1e9
            out["F"] = [-acc, flops]

    class WeightedSumProblem(ElementwiseProblem):
        """Single-objective weighted sum of normalized objectives."""
        def __init__(self, w_acc=0.4, w_flops=0.3, w_mem=0.3, memory_constraint=None):
            n_constr = 1 if memory_constraint else 0
            self.w_acc, self.w_flops, self.w_mem = w_acc, w_flops, w_mem
            self.memory_constraint = memory_constraint
            super().__init__(n_var=29, n_obj=1, n_ieq_constr=n_constr,
                           xl=np.zeros(29), xu=np.ones(29))
        def _decode(self, x):
            wm = [WIDTH_MULTIPLIERS[int(x[i]*(len(WIDTH_MULTIPLIERS)-0.01))] for i in range(7)]
            dp = [DEPTHS_LIST[int(x[7+i]*(len(DEPTHS_LIST)-0.01))] for i in range(7)]
            ex = [EXPANSION_RATIOS[int(x[14+i]*(len(EXPANSION_RATIOS)-0.01))] for i in range(7)]
            ks = [KERNEL_SIZES[int(x[21+i]*(len(KERNEL_SIZES)-0.01))] for i in range(7)]
            return wm, dp, ex, ks
        def _evaluate(self, x, out, *args, **kwargs):
            wm, dp, ex, ks = self._decode(x)
            acc = proxy_accuracy(wm, dp, ex, ks)
            flops = count_flops(wm, dp, ex, ks) / 1e9
            mem = analytical_peak_memory(wm, dp, ex, ks) / 1e6
            # Normalize (approximate ranges)
            norm_acc = (84 - acc) / 40   # ~0 at high acc, ~1 at low
            norm_flops = flops / 5.0
            norm_mem = mem / 30.0
            out["F"] = [self.w_acc * norm_acc + self.w_flops * norm_flops + self.w_mem * norm_mem]
            if self.memory_constraint:
                out["G"] = [mem - self.memory_constraint]

    class PenaltyProblem(ElementwiseProblem):
        """Single-objective: maximize accuracy with penalty for memory violation."""
        def __init__(self, memory_budget, penalty_coeff=10.0):
            self.budget = memory_budget
            self.penalty = penalty_coeff
            super().__init__(n_var=29, n_obj=1, xl=np.zeros(29), xu=np.ones(29))
        def _decode(self, x):
            wm = [WIDTH_MULTIPLIERS[int(x[i]*(len(WIDTH_MULTIPLIERS)-0.01))] for i in range(7)]
            dp = [DEPTHS_LIST[int(x[7+i]*(len(DEPTHS_LIST)-0.01))] for i in range(7)]
            ex = [EXPANSION_RATIOS[int(x[14+i]*(len(EXPANSION_RATIOS)-0.01))] for i in range(7)]
            ks = [KERNEL_SIZES[int(x[21+i]*(len(KERNEL_SIZES)-0.01))] for i in range(7)]
            return wm, dp, ex, ks
        def _evaluate(self, x, out, *args, **kwargs):
            wm, dp, ex, ks = self._decode(x)
            acc = proxy_accuracy(wm, dp, ex, ks)
            mem = analytical_peak_memory(wm, dp, ex, ks) / 1e6
            violation = max(0, mem - self.budget)
            out["F"] = [-acc + self.penalty * violation]

    # ---------- Helper to extract results ----------
    def extract_pareto(result, problem, n_obj):
        solutions = []
        if result.X is None:
            return solutions
        Xs = result.X if result.X.ndim > 1 else result.X.reshape(1, -1)
        for x in Xs:
            wm, dp, ex, ks = problem._decode(x)
            acc = proxy_accuracy(wm, dp, ex, ks)
            flops = count_flops(wm, dp, ex, ks) / 1e9
            mem = analytical_peak_memory(wm, dp, ex, ks) / 1e6
            solutions.append({
                'accuracy': round(float(acc), 2),
                'flops_g': round(float(flops), 4),
                'peak_memory_mb': round(float(mem), 3),
            })
        return solutions

    # ========== Run all baseline methods ==========
    POP_SIZE = 200
    N_GEN = 100
    ref_dirs = get_reference_directions("das-dennis", n_dim=3, n_partitions=12)
    memory_budgets = [2.0, 3.0, 5.0, 8.0, 10.0, 15.0]

    baseline_results = {}

    # 1. FLOPs-only NSGA-II
    print("\n=== Baseline 1: FLOPs-only NSGA-II ===")
    prob_flops = TwoObjProblem()
    res_flops = minimize(prob_flops, NSGA2(pop_size=POP_SIZE), ('n_gen', N_GEN), seed=42, verbose=False)
    baseline_results['flops_only'] = extract_pareto(res_flops, prob_flops, 2)
    print(f"  Found {len(baseline_results['flops_only'])} Pareto solutions")

    # 2. MemoryNAS (3-obj NSGA-III) - unconstrained
    print("\n=== Method: MemoryNAS NSGA-III (3-obj) ===")
    prob_mem3 = ThreeObjProblem()
    res_mem3 = minimize(prob_mem3, NSGA3(ref_dirs=ref_dirs, pop_size=POP_SIZE),
                       ('n_gen', N_GEN), seed=42, verbose=False)
    baseline_results['memorynasNSGA3'] = extract_pareto(res_mem3, prob_mem3, 3)
    print(f"  Found {len(baseline_results['memorynasNSGA3'])} Pareto solutions")

    # 3. MemoryNAS with hard constraints per budget
    for budget in memory_budgets:
        print(f"\n=== MemoryNAS constrained (budget={budget} MB) ===")
        prob_c = ThreeObjProblem(memory_constraint=budget)
        res_c = minimize(prob_c, NSGA3(ref_dirs=ref_dirs, pop_size=POP_SIZE),
                        ('n_gen', N_GEN), seed=42, verbose=False)
        key = f'memorynasConstrained_{budget}'
        baseline_results[key] = extract_pareto(res_c, prob_c, 3)
        print(f"  Found {len(baseline_results[key])} feasible solutions")

    # 4. Random search + memory filter
    print("\n=== Baseline 2: Random Search + Memory Filter ===")
    N_RANDOM = 20000  # Same total evaluations as NSGA-III (200 * 100)
    random_results = {}
    rng = np.random.RandomState(42)
    all_random = []
    for _ in range(N_RANDOM):
        wm = [float(rng.choice(WIDTH_MULTIPLIERS)) for _ in range(7)]
        dp = [int(rng.choice(DEPTHS_LIST)) for _ in range(7)]
        ex = [int(rng.choice(EXPANSION_RATIOS)) for _ in range(7)]
        ks = [int(rng.choice(KERNEL_SIZES)) for _ in range(7)]
        acc = proxy_accuracy(wm, dp, ex, ks)
        flops = count_flops(wm, dp, ex, ks) / 1e9
        mem = analytical_peak_memory(wm, dp, ex, ks) / 1e6
        all_random.append({'accuracy': round(float(acc),2), 'flops_g': round(float(flops),4),
                          'peak_memory_mb': round(float(mem),3)})
    for budget in memory_budgets:
        feasible = [r for r in all_random if r['peak_memory_mb'] <= budget]
        # Get Pareto front from feasible
        if feasible:
            feasible.sort(key=lambda x: (-x['accuracy'], x['flops_g']))
        random_results[f'random_filter_{budget}'] = feasible[:50]  # top 50
    baseline_results['random_all'] = all_random[:500]  # save sample
    baseline_results.update(random_results)
    print(f"  Sampled {N_RANDOM} architectures")

    # 5. FLOPs-only + filter (search on FLOPs, then discard infeasible)
    print("\n=== Baseline 3: FLOPs-only + Memory Filter ===")
    flops_pareto = baseline_results['flops_only']
    for budget in memory_budgets:
        feasible = [r for r in flops_pareto if r['peak_memory_mb'] <= budget]
        key = f'flops_filter_{budget}'
        baseline_results[key] = feasible
        n_fail = len(flops_pareto) - len(feasible)
        pct = n_fail / max(len(flops_pareto), 1) * 100
        print(f"  Budget {budget} MB: {len(feasible)}/{len(flops_pareto)} feasible ({pct:.1f}% fail)")

    # 6. Weighted-sum with memory
    print("\n=== Baseline 4: Weighted-Sum Optimization ===")
    from pymoo.algorithms.soo.nonconvex.ga import GA
    for budget in memory_budgets:
        prob_ws = WeightedSumProblem(w_acc=0.4, w_flops=0.3, w_mem=0.3, memory_constraint=budget)
        res_ws = minimize(prob_ws, GA(pop_size=POP_SIZE), ('n_gen', N_GEN), seed=42, verbose=False)
        sols = extract_pareto(res_ws, prob_ws, 1)
        baseline_results[f'weighted_sum_{budget}'] = sols
        print(f"  Budget {budget} MB: found {len(sols)} solutions")

    # 7. Penalty method
    print("\n=== Baseline 5: Penalty Method ===")
    for budget in memory_budgets:
        prob_pen = PenaltyProblem(memory_budget=budget, penalty_coeff=10.0)
        res_pen = minimize(prob_pen, GA(pop_size=POP_SIZE), ('n_gen', N_GEN), seed=42, verbose=False)
        sols = extract_pareto(res_pen, prob_pen, 1)
        baseline_results[f'penalty_{budget}'] = sols
        print(f"  Budget {budget} MB: found {len(sols)} solutions")

    # ========== Ablations ==========
    print("\n" + "=" * 70)
    print("ABLATION EXPERIMENTS")
    print("=" * 70)

    ablation_results = {}

    # Ablation 1: Memory as objective vs constraint only
    print("\n=== Ablation: Memory-as-objective vs Memory-as-constraint ===")
    for budget in [3.0, 5.0, 10.0]:
        # 2-obj + constraint (NSGA-II with memory constraint but not objective)
        class TwoObjConstrained(ElementwiseProblem):
            def __init__(self, mem_budget):
                self.mem_budget = mem_budget
                super().__init__(n_var=29, n_obj=2, n_ieq_constr=1,
                               xl=np.zeros(29), xu=np.ones(29))
            def _decode(self, x):
                wm = [WIDTH_MULTIPLIERS[int(x[i]*(len(WIDTH_MULTIPLIERS)-0.01))] for i in range(7)]
                dp = [DEPTHS_LIST[int(x[7+i]*(len(DEPTHS_LIST)-0.01))] for i in range(7)]
                ex = [EXPANSION_RATIOS[int(x[14+i]*(len(EXPANSION_RATIOS)-0.01))] for i in range(7)]
                ks = [KERNEL_SIZES[int(x[21+i]*(len(KERNEL_SIZES)-0.01))] for i in range(7)]
                return wm, dp, ex, ks
            def _evaluate(self, x, out, *args, **kwargs):
                wm, dp, ex, ks = self._decode(x)
                acc = proxy_accuracy(wm, dp, ex, ks)
                flops = count_flops(wm, dp, ex, ks) / 1e9
                mem = analytical_peak_memory(wm, dp, ex, ks) / 1e6
                out["F"] = [-acc, flops]
                out["G"] = [mem - self.mem_budget]

        prob_2c = TwoObjConstrained(budget)
        res_2c = minimize(prob_2c, NSGA2(pop_size=POP_SIZE), ('n_gen', N_GEN), seed=42, verbose=False)
        sols = extract_pareto(res_2c, prob_2c, 2)
        ablation_results[f'2obj_constrained_{budget}'] = sols
        print(f"  Budget {budget}: 2-obj+constraint found {len(sols)} solutions")

    # Ablation 2: Search budget sensitivity
    print("\n=== Ablation: Search Budget Sensitivity ===")
    for n_eval_gen in [25, 50, 100, 200]:
        prob_s = ThreeObjProblem(memory_constraint=5.0)
        res_s = minimize(prob_s, NSGA3(ref_dirs=ref_dirs, pop_size=POP_SIZE),
                        ('n_gen', n_eval_gen), seed=42, verbose=False)
        sols = extract_pareto(res_s, prob_s, 3)
        best_acc = max((s['accuracy'] for s in sols), default=0)
        ablation_results[f'budget_sensitivity_{n_eval_gen}'] = {
            'n_generations': n_eval_gen,
            'total_evals': POP_SIZE * n_eval_gen,
            'n_solutions': len(sols),
            'best_accuracy': round(best_acc, 2),
            'solutions': sols,
        }
        print(f"  {n_eval_gen} gens ({POP_SIZE*n_eval_gen} evals): {len(sols)} solutions, best acc={best_acc:.1f}%")

    # ========== Feasibility Analysis (for theorem-like claim) ==========
    print("\n=== Feasibility Analysis ===")
    feasibility = {}
    all_flops_pareto = baseline_results['flops_only']
    for budget in [2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0]:
        total = len(all_flops_pareto)
        feasible = sum(1 for a in all_flops_pareto if a['peak_memory_mb'] <= budget)
        infeasible = total - feasible
        pct_fail = infeasible / total * 100 if total > 0 else 0
        feasibility[str(budget)] = {
            'budget_mb': budget,
            'total_pareto': total,
            'feasible': feasible,
            'infeasible': infeasible,
            'failure_pct': round(pct_fail, 1),
        }
        print(f"  Budget {budget:5.1f} MB: {infeasible}/{total} infeasible ({pct_fail:.1f}%)")

    # ========== Save everything ==========
    output = {
        'baselines': baseline_results,
        'ablations': ablation_results,
        'feasibility': feasibility,
    }
    with open('/results/baselines_ablations.json', 'w') as f:
        json.dump(output, f, indent=2)
    volume.commit()
    print("\nAll baselines and ablations saved.")

    return output


# ============================================================================
# Orchestrator - runs entirely server-side (disconnect-safe)
# ============================================================================

@app.function(
    image=image,
    volumes={"/results": volume},
    timeout=10800,  # 3 hours max
    memory=4096,
)
def orchestrate():
    """Server-side orchestrator: spawns all 3 jobs and waits for them."""
    import json

    print("=" * 70)
    print("MemoryNAS: Launching all experiments (disconnect-safe)")
    print("=" * 70)

    # Spawn all 3 jobs in parallel
    print("\n>>> Spawning baselines & ablations (CPU)...")
    baselines_future = run_baselines_and_ablations.spawn()

    print(">>> Spawning memory validation (A10G GPU)...")
    memory_future = validate_memory_estimator.spawn()

    print(">>> Spawning training (A10G GPU)...")
    training_future = train_all.spawn()

    # Wait for all to complete
    print("\nWaiting for baselines...")
    baselines = baselines_future.get()
    print("Baselines complete!")

    print("\nWaiting for memory validation...")
    memory = memory_future.get()
    print("Memory validation complete!")

    print("\nWaiting for training...")
    training = training_future.get()
    print("Training complete!")

    # Write a completion marker
    summary = {
        "status": "complete",
        "n_trained": len(training) if training else 0,
        "n_validated": memory.get("summary", {}).get("n_archs", 0) if memory else 0,
        "baselines_keys": list(baselines.get("baselines", {}).keys()) if baselines else [],
    }
    with open("/results/completion_marker.json", "w") as f:
        json.dump(summary, f, indent=2)
    volume.commit()

    print("\n" + "=" * 70)
    print("ALL MODAL EXPERIMENTS COMPLETE")
    print("Results saved to memorynas-results volume")
    print("=" * 70)
    return summary
