"""
Modal script: Measure real GPU energy consumption via NVML for proxy validation.
Trains 100 diverse architectures from the GreenNAS search space on CIFAR-10,
recording GPU power draw every 100ms via pynvml.

Outputs: JSON with per-architecture measured energy, power trace, and proxy estimates.
"""
import modal
import json

app = modal.App("greennas-energy-validation")
volume = modal.Volume.from_name("greennas-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch", "torchvision", "numpy", "pynvml", "scipy"
    )
)

@app.function(
    gpu="T4",
    image=image,
    volumes={"/results": volume},
    timeout=7200,
    memory=16384,
)
def measure_energy():
    import os, time, random, threading
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Subset
    import pynvml
    
    # Initialize NVML
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    tdp = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0  # mW -> W
    print(f"GPU: {gpu_name}, TDP: {tdp}W")
    
    device = torch.device('cuda')
    
    # ============================================================
    # SEARCH SPACE (same as generate_results.py)
    # ============================================================
    OPS = ['zero', 'skip_connect', 'conv_1x1', 'conv_3x3', 'avg_pool_3x3']
    NUM_EDGES = 6
    
    OP_FLOPS_PER_CH = {'zero': 0, 'skip_connect': 0, 'conv_1x1': 2, 'conv_3x3': 18, 'avg_pool_3x3': 9}
    
    def op_params(op, C):
        if op == 'conv_1x1': return C*C + 2*C
        if op == 'conv_3x3': return 9*C*C + 2*C
        return 0
    
    def op_memory_bytes(op, C, H, B):
        act_bytes = B * C * H * H * 4
        if op == 'zero':
            return act_bytes * 2
        elif op == 'skip_connect':
            return act_bytes
        elif op == 'conv_1x1':
            return act_bytes * 2 + C * C * 4
        elif op == 'conv_3x3':
            return act_bytes * 2 + 9 * C * C * 4 + B * (9 * C) * H * H * 4
        elif op == 'avg_pool_3x3':
            return act_bytes * 2
        return 0
    
    # ============================================================
    # NETWORK MODULES
    # ============================================================
    class ZeroOp(nn.Module):
        def forward(self, x): return torch.zeros_like(x)
    
    class SkipConnect(nn.Module):
        def forward(self, x): return x
    
    class Conv1x1(nn.Module):
        def __init__(self, C):
            super().__init__()
            self.op = nn.Sequential(nn.Conv2d(C, C, 1, bias=False), nn.BatchNorm2d(C), nn.ReLU(True))
        def forward(self, x): return self.op(x)
    
    class Conv3x3(nn.Module):
        def __init__(self, C):
            super().__init__()
            self.op = nn.Sequential(nn.Conv2d(C, C, 3, padding=1, bias=False), nn.BatchNorm2d(C), nn.ReLU(True))
        def forward(self, x): return self.op(x)
    
    class AvgPool3x3(nn.Module):
        def __init__(self, C):
            super().__init__()
            self.pool = nn.AvgPool2d(3, stride=1, padding=1)
        def forward(self, x): return self.pool(x)
    
    def get_op(name, C):
        if name == 'zero': return ZeroOp()
        if name == 'skip_connect': return SkipConnect()
        if name == 'conv_1x1': return Conv1x1(C)
        if name == 'conv_3x3': return Conv3x3(C)
        if name == 'avg_pool_3x3': return AvgPool3x3(C)
    
    class Cell(nn.Module):
        def __init__(self, arch, C):
            super().__init__()
            self.ops = nn.ModuleList([get_op(op, C) for op in arch])
            self.edges = [(0,1),(0,2),(1,2),(0,3),(1,3),(2,3)]
        def forward(self, x):
            nodes = [x, None, None, None]
            for idx, (s, d) in enumerate(self.edges):
                out = self.ops[idx](nodes[s])
                nodes[d] = out if nodes[d] is None else nodes[d] + out
            return nodes[3] if nodes[3] is not None else torch.zeros_like(x)
    
    class Network(nn.Module):
        def __init__(self, arch, C=16, num_cells=5, num_classes=10):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(3, C, 3, padding=1, bias=False), nn.BatchNorm2d(C), nn.ReLU(True))
            layers = []
            for i in range(num_cells):
                layers.append(Cell(arch, C))
                if i in [1, 3]:
                    layers.append(nn.Sequential(nn.Conv2d(C, C*2, 1, stride=2, bias=False), nn.BatchNorm2d(C*2), nn.ReLU(True)))
                    C *= 2
            self.cells = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(C, num_classes)
        
        def forward(self, x):
            x = self.stem(x)
            x = self.cells(x)
            x = self.pool(x).flatten(1)
            return self.fc(x)
    
    # ============================================================
    # DATA
    # ============================================================
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
    
    trainset = torchvision.datasets.CIFAR10(root='/tmp/data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='/tmp/data', train=False, download=True, transform=transform_test)
    
    # Use 10k subset for each arch to keep total time manageable
    random.seed(42)
    indices = list(range(len(trainset)))
    random.shuffle(indices)
    train_subset = Subset(trainset, indices[:10000])
    
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    
    # ============================================================
    # POWER MONITORING THREAD
    # ============================================================
    class PowerMonitor:
        def __init__(self, handle, interval_ms=100):
            self.handle = handle
            self.interval = interval_ms / 1000.0
            self.readings = []
            self.running = False
            self.thread = None
        
        def start(self):
            self.readings = []
            self.running = True
            self.thread = threading.Thread(target=self._monitor, daemon=True)
            self.thread.start()
        
        def stop(self):
            self.running = False
            if self.thread:
                self.thread.join(timeout=2.0)
            return self.readings
        
        def _monitor(self):
            while self.running:
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                    self.readings.append({
                        'time': time.time(),
                        'power_w': power_mw / 1000.0
                    })
                except:
                    pass
                time.sleep(self.interval)
    
    monitor = PowerMonitor(handle, interval_ms=100)
    
    # ============================================================
    # PROXY ESTIMATION (matching generate_results.py)
    # ============================================================
    def compute_proxy_estimates(arch, B=128):
        C, H = 16, 32; num_cells = 5
        
        # Params
        params = 3*C*9 + C
        cur_C = C
        for i in range(num_cells):
            for op in arch: params += op_params(op, cur_C)
            if i in [1, 3]:
                params += cur_C*(cur_C*2) + 2*(cur_C*2)
                cur_C *= 2
        params += cur_C * 10 + 10
        
        # FLOPs
        flops = 3*C*9*H*H*2; cur_C = C; cur_H = H
        for i in range(num_cells):
            for op in arch: flops += OP_FLOPS_PER_CH[op] * cur_C * cur_H * cur_H
            if i in [1, 3]:
                flops += cur_C*(cur_C*2)*cur_H*cur_H*2
                cur_C *= 2; cur_H //= 2
        flops += cur_C * 10 * 2
        
        # Memory traffic
        mem_bytes = B * 3 * H * H * 4 + C * 3 * 9 * 4 + B * C * H * H * 4
        cur_C = C; cur_H = H
        for i in range(num_cells):
            for op in arch: mem_bytes += op_memory_bytes(op, cur_C, cur_H, B)
            if i in [1, 3]:
                mem_bytes += B*cur_C*cur_H*cur_H*4 + B*(cur_C*2)*(cur_H//2)*(cur_H//2)*4 + cur_C*(cur_C*2)*4
                cur_C *= 2; cur_H //= 2
        mem_bytes += B * cur_C * 4 + cur_C * 10 * 4
        mem_gb = mem_bytes / (1024**3)
        
        # Op counts
        n_useful = sum(1 for op in arch if op in ['conv_1x1', 'conv_3x3'])
        n_zero = sum(1 for op in arch if op == 'zero')
        
        # Proxy power model
        num_samples = 50000; epochs = 200
        flops_per_sample = flops * 3
        total_train_flops = flops_per_sample * num_samples * epochs
        compute_ratio = n_useful / max(1, 6 - n_zero)
        utilization = 0.25 + 0.20 * compute_ratio
        effective_tflops = 15.7 * utilization
        wall_s = total_train_flops / (effective_tflops * 1e12) * 1.175
        gflops_per_s = (total_train_flops / wall_s) / 1e9
        mem_bw_gbs = (mem_gb * num_samples * epochs * 3) / wall_s
        P_idle = 50.0; alpha = 0.8; beta = 2.0
        proxy_power_w = min(P_idle + alpha * gflops_per_s + beta * mem_bw_gbs, 300.0)
        proxy_energy_kwh = proxy_power_w * wall_s / 3_600_000
        
        return {
            'params': int(params), 'flops': int(flops), 'mem_gb': float(mem_gb),
            'proxy_power_w': float(proxy_power_w), 'proxy_energy_kwh': float(proxy_energy_kwh),
            'proxy_wall_s': float(wall_s), 'n_useful': n_useful, 'n_zero': n_zero
        }
    
    # ============================================================
    # GENERATE DIVERSE ARCHITECTURES
    # ============================================================
    def generate_diverse_archs(n=100):
        """Generate architectures that span the full search space diversity."""
        archs = set()
        
        # Extreme cases
        archs.add(tuple(['zero'] * 6))
        archs.add(tuple(['skip_connect'] * 6))
        archs.add(tuple(['conv_1x1'] * 6))
        archs.add(tuple(['conv_3x3'] * 6))
        archs.add(tuple(['avg_pool_3x3'] * 6))
        
        # Systematic: 0-6 conv3x3 ops, rest filled with other ops
        for n_conv3x3 in range(7):
            for n_conv1x1 in range(7 - n_conv3x3):
                remaining = 6 - n_conv3x3 - n_conv1x1
                for fill_op in ['skip_connect', 'avg_pool_3x3', 'zero']:
                    if len(archs) >= n: break
                    arch = ['conv_3x3'] * n_conv3x3 + ['conv_1x1'] * n_conv1x1 + [fill_op] * remaining
                    random.shuffle(arch)
                    archs.add(tuple(arch))
        
        # Random fill to reach n
        random.seed(42)
        while len(archs) < n:
            arch = tuple(random.choice(OPS) for _ in range(NUM_EDGES))
            archs.add(arch)
        
        return list(archs)[:n]
    
    architectures = generate_diverse_archs(100)
    print(f"Will measure {len(architectures)} architectures")
    
    # ============================================================
    # MAIN MEASUREMENT LOOP
    # ============================================================
    all_measurements = []
    
    # Warmup GPU
    print("Warming up GPU...")
    dummy = torch.randn(128, 3, 32, 32, device=device)
    dummy_model = Network(['conv_3x3']*6, C=16, num_cells=5).to(device)
    for _ in range(50):
        _ = dummy_model(dummy)
    del dummy_model, dummy
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    
    # Measure idle power
    monitor.start()
    time.sleep(3)
    idle_readings = monitor.stop()
    idle_power_w = np.mean([r['power_w'] for r in idle_readings]) if idle_readings else 30.0
    print(f"Idle power: {idle_power_w:.1f}W")
    
    for arch_idx, arch in enumerate(architectures):
        arch_str = '|'.join(arch)
        print(f"\n[{arch_idx+1}/{len(architectures)}] {arch_str}")
        
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        model = Network(arch, C=16, num_cells=5).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=3e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
        criterion = nn.CrossEntropyLoss()
        
        param_count = sum(p.numel() for p in model.parameters())
        
        # Training with power monitoring
        torch.cuda.synchronize()
        monitor.start()
        train_start = time.time()
        total_samples = 0
        
        model.train()
        for epoch in range(8):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_samples += inputs.size(0)
            scheduler.step()
        
        torch.cuda.synchronize()
        train_wall_s = time.time() - train_start
        train_power_readings = monitor.stop()
        
        # Compute measured energy
        train_powers = [r['power_w'] for r in train_power_readings]
        avg_train_power_w = np.mean(train_powers) if train_powers else idle_power_w
        std_train_power_w = np.std(train_powers) if train_powers else 0.0
        measured_energy_j = avg_train_power_w * train_wall_s
        measured_energy_kwh = measured_energy_j / 3_600_000
        throughput = total_samples / train_wall_s
        
        # Evaluation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                pred = model(inputs).argmax(1)
                correct += (pred == targets).sum().item()
                total += targets.size(0)
        accuracy = 100. * correct / total
        
        # Inference latency + power
        dummy = torch.randn(1, 3, 32, 32, device=device)
        torch.cuda.synchronize()
        monitor.start()
        latencies = []
        with torch.no_grad():
            for _ in range(100):
                torch.cuda.synchronize()
                t0 = time.time()
                _ = model(dummy)
                torch.cuda.synchronize()
                latencies.append((time.time() - t0) * 1000)
        inf_readings = monitor.stop()
        latency_ms = np.median(latencies)
        avg_inf_power_w = np.mean([r['power_w'] for r in inf_readings]) if inf_readings else idle_power_w
        inf_energy_per_sample_j = avg_inf_power_w * (latency_ms / 1000)
        
        # Batch-size sensitivity
        bs_throughputs = []
        for bs in [32, 64, 128, 256]:
            sub_loader = DataLoader(Subset(trainset, indices[:1000]), batch_size=bs, shuffle=False, num_workers=0)
            torch.cuda.synchronize()
            t0 = time.time()
            ns = 0
            with torch.no_grad():
                for xx, _ in sub_loader:
                    _ = model(xx.to(device))
                    ns += xx.size(0)
            torch.cuda.synchronize()
            bs_throughputs.append(ns / (time.time() - t0 + 1e-8))
        bs_sensitivity = np.std(bs_throughputs) / (np.mean(bs_throughputs) + 1e-8)
        
        # Proxy estimates
        proxy = compute_proxy_estimates(arch, B=128)
        
        measurement = {
            'arch': list(arch),
            'arch_str': arch_str,
            'accuracy': float(accuracy),
            'param_count': int(param_count),
            
            # Measured values
            'measured': {
                'train_wall_s': float(train_wall_s),
                'avg_power_w': float(avg_train_power_w),
                'std_power_w': float(std_train_power_w),
                'energy_kwh': float(measured_energy_kwh),
                'energy_j': float(measured_energy_j),
                'throughput': float(throughput),
                'total_samples': int(total_samples),
                'n_power_readings': len(train_powers),
                'power_percentiles': {
                    'p10': float(np.percentile(train_powers, 10)) if train_powers else 0,
                    'p50': float(np.percentile(train_powers, 50)) if train_powers else 0,
                    'p90': float(np.percentile(train_powers, 90)) if train_powers else 0,
                },
                'latency_ms': float(latency_ms),
                'inf_power_w': float(avg_inf_power_w),
                'inf_energy_j': float(inf_energy_per_sample_j),
                'bs_sensitivity': float(bs_sensitivity),
            },
            
            # Proxy estimates
            'proxy': proxy,
        }
        
        all_measurements.append(measurement)
        print(f"  Acc={accuracy:.1f}% | Measured: P={avg_train_power_w:.1f}W E={measured_energy_kwh:.6f}kWh | "
              f"Proxy: P={proxy['proxy_power_w']:.1f}W E={proxy['proxy_energy_kwh']:.6f}kWh | "
              f"Params={param_count} Lat={latency_ms:.2f}ms")
        
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
    
    # Save results
    output = {
        'gpu': gpu_name,
        'tdp_w': tdp,
        'idle_power_w': idle_power_w,
        'n_architectures': len(all_measurements),
        'measurements': all_measurements,
    }
    
    output_path = '/results/nvml_measurements.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    volume.commit()
    
    print(f"\n{'='*60}")
    print(f"DONE: {len(all_measurements)} architectures measured")
    print(f"Results saved to {output_path}")
    
    # Quick correlation summary
    measured_energies = [m['measured']['energy_kwh'] for m in all_measurements]
    proxy_energies = [m['proxy']['proxy_energy_kwh'] for m in all_measurements]
    measured_powers = [m['measured']['avg_power_w'] for m in all_measurements]
    proxy_powers = [m['proxy']['proxy_power_w'] for m in all_measurements]
    
    from scipy import stats
    
    # Energy correlation
    r_energy, p_energy = stats.pearsonr(measured_energies, proxy_energies)
    rho_energy, _ = stats.spearmanr(measured_energies, proxy_energies)
    print(f"\nEnergy: Pearson r={r_energy:.4f} (p={p_energy:.2e}), Spearman ρ={rho_energy:.4f}")
    
    # Power correlation
    r_power, p_power = stats.pearsonr(measured_powers, proxy_powers)
    rho_power, _ = stats.spearmanr(measured_powers, proxy_powers)
    print(f"Power:  Pearson r={r_power:.4f} (p={p_power:.2e}), Spearman ρ={rho_power:.4f}")
    
    return output
