"""
Modal script: Train top-K architectures from GreenNAS vs FLOPs-only vs Random
with NVML energy monitoring. Full 200-epoch CIFAR-10 training.
Also runs 50-arch L4 validation for cross-GPU proxy validation.
"""
import modal
import json

app = modal.App("greennas-real-training")
volume = modal.Volume.from_name("greennas-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch", "torchvision", "numpy", "pynvml", "scipy")
)

# ============================================================
# TOP-K REAL TRAINING (A100 GPU, ~15-20 min per arch × 5 archs)
# ============================================================
@app.function(
    gpu="A100-40GB",
    image=image,
    volumes={"/results": volume},
    timeout=21600,  # 6 hours max
    memory=16384,
)
def train_topk():
    import os, time, random, threading
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import pynvml
    
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(gpu_name, bytes): gpu_name = gpu_name.decode()
    print(f"GPU: {gpu_name}")
    device = torch.device('cuda')
    
    # Power monitor
    class PowerMonitor:
        def __init__(self, handle, interval_ms=200):
            self.handle = handle
            self.interval = interval_ms / 1000.0
            self.readings = []
            self.running = False
        def start(self):
            self.readings = []
            self.running = True
            self.thread = threading.Thread(target=self._monitor, daemon=True)
            self.thread.start()
        def stop(self):
            self.running = False
            if self.thread: self.thread.join(timeout=2.0)
            return self.readings
        def _monitor(self):
            while self.running:
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                    self.readings.append({'time': time.time(), 'power_w': power_mw / 1000.0})
                except: pass
                time.sleep(self.interval)
    
    monitor = PowerMonitor(handle)
    
    # Network definition (same as before)
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
        ops = {'zero': ZeroOp, 'skip_connect': SkipConnect, 'conv_1x1': Conv1x1,
               'conv_3x3': Conv3x3, 'avg_pool_3x3': AvgPool3x3}
        return ops[name]() if name in ('zero', 'skip_connect') else ops[name](C)
    
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
    
    # Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='/tmp/data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='/tmp/data', train=False, download=True, transform=transform_test)
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    
    # Top architectures to train (deduplicated)
    archs_to_train = [
        # GreenNAS top-2 (unique)
        {'method': 'GreenNAS', 'arch': ['conv_1x1','conv_3x3','conv_3x3','conv_1x1','conv_3x3','conv_1x1']},
        {'method': 'GreenNAS', 'arch': ['conv_1x1','conv_1x1','conv_1x1','conv_3x3','conv_3x3','conv_3x3']},
        # FLOPs-only top-2 (unique)
        {'method': 'FLOPs-only', 'arch': ['conv_3x3','conv_3x3','conv_3x3','conv_3x3','conv_1x1','conv_1x1']},
        {'method': 'FLOPs-only', 'arch': ['conv_1x1','conv_3x3','conv_1x1','conv_3x3','conv_1x1','conv_3x3']},
        # Random top-1
        {'method': 'Random', 'arch': ['conv_3x3','conv_3x3','conv_3x3','conv_1x1','conv_1x1','conv_1x1']},
    ]
    
    EPOCHS = 200
    all_results = []
    
    # Warmup
    print("Warming up...")
    dummy = torch.randn(128, 3, 32, 32, device=device)
    dummy_model = Network(['conv_3x3']*6, C=16).to(device)
    for _ in range(50): _ = dummy_model(dummy)
    del dummy_model, dummy; torch.cuda.empty_cache(); time.sleep(2)
    
    for idx, entry in enumerate(archs_to_train):
        arch = entry['arch']
        method = entry['method']
        arch_str = '|'.join(arch)
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(archs_to_train)}] {method}: {arch_str}")
        print(f"Training {EPOCHS} epochs on full CIFAR-10 (50k)")
        print(f"{'='*60}")
        
        torch.manual_seed(42); np.random.seed(42); random.seed(42)
        torch.cuda.empty_cache()
        
        model = Network(arch, C=16, num_cells=5).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=3e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.CrossEntropyLoss()
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Params: {param_count:,}")
        
        # Train with power monitoring
        torch.cuda.synchronize()
        monitor.start()
        train_start = time.time()
        
        best_acc = 0
        epoch_log = []
        model.train()
        for epoch in range(EPOCHS):
            running_loss = 0; total = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                total += inputs.size(0)
            scheduler.step()
            
            # Eval every 20 epochs
            if (epoch + 1) % 20 == 0 or epoch == EPOCHS - 1:
                model.eval()
                correct = tot = 0
                with torch.no_grad():
                    for inp, tgt in test_loader:
                        inp, tgt = inp.to(device), tgt.to(device)
                        pred = model(inp).argmax(1)
                        correct += (pred == tgt).sum().item()
                        tot += tgt.size(0)
                acc = 100. * correct / tot
                if acc > best_acc: best_acc = acc
                avg_loss = running_loss / total
                epoch_log.append({'epoch': epoch+1, 'acc': acc, 'loss': avg_loss})
                print(f"  Epoch {epoch+1:3d}/{EPOCHS}: Loss={avg_loss:.4f} Acc={acc:.1f}% (best={best_acc:.1f}%)")
                model.train()
        
        torch.cuda.synchronize()
        train_wall_s = time.time() - train_start
        train_readings = monitor.stop()
        
        train_powers = [r['power_w'] for r in train_readings]
        avg_power = np.mean(train_powers) if train_powers else 0
        energy_j = avg_power * train_wall_s
        energy_kwh = energy_j / 3_600_000
        
        # Final eval
        model.eval()
        correct = tot = 0
        with torch.no_grad():
            for inp, tgt in test_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                pred = model(inp).argmax(1)
                correct += (pred == tgt).sum().item()
                tot += tgt.size(0)
        final_acc = 100. * correct / tot
        
        result = {
            'method': method,
            'arch': arch,
            'arch_str': arch_str,
            'params': param_count,
            'final_accuracy': final_acc,
            'best_accuracy': best_acc,
            'train_wall_s': train_wall_s,
            'avg_power_w': avg_power,
            'energy_kwh': energy_kwh,
            'energy_j': energy_j,
            'n_epochs': EPOCHS,
            'epoch_log': epoch_log,
            'gpu': gpu_name,
        }
        all_results.append(result)
        print(f"\n  FINAL: Acc={final_acc:.1f}% Best={best_acc:.1f}% Power={avg_power:.1f}W Energy={energy_kwh:.6f}kWh Wall={train_wall_s:.0f}s")
        
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
    
    # Save
    output_path = '/results/topk_training_results.json'
    with open(output_path, 'w') as f:
        json.dump({'gpu': gpu_name, 'results': all_results}, f, indent=2)
    volume.commit()
    print(f"\nSaved {len(all_results)} results to {output_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  {r['method']:12s} {r['arch_str']:50s} Acc={r['final_accuracy']:.1f}% E={r['energy_kwh']*1000:.1f}Wh P={r['avg_power_w']:.1f}W")
    
    return all_results


# ============================================================
# L4 GPU VALIDATION (cross-GPU proxy validation, 50 archs)
# ============================================================
@app.function(
    gpu="L4",
    image=image,
    volumes={"/results": volume},
    timeout=7200,
    memory=16384,
)
def validate_l4():
    import os, time, random, threading
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Subset
    import pynvml
    
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(gpu_name, bytes): gpu_name = gpu_name.decode()
    tdp = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
    print(f"GPU: {gpu_name}, TDP: {tdp}W")
    device = torch.device('cuda')
    
    class PowerMonitor:
        def __init__(self, handle, interval_ms=100):
            self.handle = handle; self.interval = interval_ms / 1000.0
            self.readings = []; self.running = False
        def start(self):
            self.readings = []; self.running = True
            self.thread = threading.Thread(target=self._monitor, daemon=True); self.thread.start()
        def stop(self):
            self.running = False
            if self.thread: self.thread.join(timeout=2.0)
            return self.readings
        def _monitor(self):
            while self.running:
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                    self.readings.append({'time': time.time(), 'power_w': power_mw / 1000.0})
                except: pass
                time.sleep(self.interval)
    
    monitor = PowerMonitor(handle)
    
    # Same network definition
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
        ops = {'zero': ZeroOp, 'skip_connect': SkipConnect, 'conv_1x1': Conv1x1,
               'conv_3x3': Conv3x3, 'avg_pool_3x3': AvgPool3x3}
        return ops[name]() if name in ('zero', 'skip_connect') else ops[name](C)
    
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
            layers = []; cur_C = C
            for i in range(num_cells):
                layers.append(Cell(arch, cur_C))
                if i in [1, 3]:
                    layers.append(nn.Sequential(nn.Conv2d(cur_C, cur_C*2, 1, stride=2, bias=False), nn.BatchNorm2d(cur_C*2), nn.ReLU(True)))
                    cur_C *= 2
            self.cells = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(cur_C, num_classes)
        def forward(self, x):
            x = self.stem(x); x = self.cells(x); x = self.pool(x).flatten(1); return self.fc(x)
    
    # Data (10k subset, 8 epochs — same as T4 validation)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='/tmp/data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='/tmp/data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]))
    
    random.seed(42)
    indices = list(range(len(trainset))); random.shuffle(indices)
    train_loader = DataLoader(Subset(trainset, indices[:10000]), batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    
    # Same 50 diverse architectures (subset of T4's 100)
    OPS = ['zero', 'skip_connect', 'conv_1x1', 'conv_3x3', 'avg_pool_3x3']
    archs = set()
    for op in OPS: archs.add(tuple([op]*6))
    random.seed(42)
    while len(archs) < 50:
        archs.add(tuple(random.choice(OPS) for _ in range(6)))
    architectures = list(archs)[:50]
    
    # Proxy estimation
    OP_FLOPS = {'zero': 0, 'skip_connect': 0, 'conv_1x1': 2, 'conv_3x3': 18, 'avg_pool_3x3': 9}
    def compute_proxy_flops_mem(arch):
        C, H = 16, 32
        flops = 3*C*9*H*H*2; mem = 0
        cur_C = C; cur_H = H
        for i in range(5):
            for op in arch:
                flops += OP_FLOPS[op] * cur_C * cur_H * cur_H
                act = 128 * cur_C * cur_H * cur_H * 4
                if op == 'conv_3x3': mem += act*2 + 9*cur_C*cur_C*4 + 128*(9*cur_C)*cur_H*cur_H*4
                elif op == 'conv_1x1': mem += act*2 + cur_C*cur_C*4
                elif op == 'skip_connect': mem += act
                else: mem += act*2
            if i in [1,3]: cur_C *= 2; cur_H //= 2
        return flops, mem / (1024**3)
    
    # Warmup
    print("Warming up L4 GPU...")
    dummy = torch.randn(128, 3, 32, 32, device=device)
    dummy_model = Network(['conv_3x3']*6).to(device)
    for _ in range(50): _ = dummy_model(dummy)
    del dummy_model, dummy; torch.cuda.empty_cache(); time.sleep(2)
    
    # Idle power
    monitor.start(); time.sleep(3)
    idle_readings = monitor.stop()
    idle_power = np.mean([r['power_w'] for r in idle_readings]) if idle_readings else 20.0
    print(f"L4 Idle power: {idle_power:.1f}W")
    
    measurements = []
    for arch_idx, arch in enumerate(architectures):
        arch_str = '|'.join(arch)
        print(f"\n[{arch_idx+1}/50] {arch_str}")
        
        torch.manual_seed(42); random.seed(42); torch.cuda.empty_cache()
        model = Network(arch).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=3e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
        criterion = nn.CrossEntropyLoss()
        
        torch.cuda.synchronize(); monitor.start()
        train_start = time.time()
        model.train()
        for epoch in range(8):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                optimizer.zero_grad(); loss = criterion(model(inputs), targets); loss.backward(); optimizer.step()
            scheduler.step()
        torch.cuda.synchronize()
        wall_s = time.time() - train_start
        readings = monitor.stop()
        
        powers = [r['power_w'] for r in readings]
        avg_power = np.mean(powers) if powers else idle_power
        energy_kwh = avg_power * wall_s / 3_600_000
        
        # Eval
        model.eval(); correct = tot = 0
        with torch.no_grad():
            for inp, tgt in test_loader:
                pred = model(inp.to(device)).argmax(1); correct += (pred == tgt.to(device)).sum().item(); tot += tgt.size(0)
        accuracy = 100. * correct / tot
        
        flops, mem_gb = compute_proxy_flops_mem(arch)
        measurements.append({
            'arch': list(arch), 'accuracy': accuracy,
            'measured': {'energy_kwh': energy_kwh, 'avg_power_w': avg_power, 'train_wall_s': wall_s},
            'proxy': {'flops': flops, 'mem_gb': mem_gb, 'composite': flops * mem_gb}
        })
        print(f"  Acc={accuracy:.1f}% P={avg_power:.1f}W E={energy_kwh:.6f}kWh")
        
        del model, optimizer, scheduler; torch.cuda.empty_cache()
    
    # Save
    from scipy import stats
    measured_e = np.array([m['measured']['energy_kwh'] for m in measurements])
    composite = np.array([m['proxy']['composite'] for m in measurements])
    flops_arr = np.array([m['proxy']['flops'] for m in measurements])
    
    rho_c, _ = stats.spearmanr(composite, measured_e)
    rho_f, _ = stats.spearmanr(flops_arr, measured_e)
    r_c, _ = stats.pearsonr(composite, measured_e)
    
    output = {
        'gpu': gpu_name, 'tdp_w': tdp, 'idle_power_w': idle_power,
        'n_architectures': len(measurements), 'measurements': measurements,
        'correlations': {
            'composite_spearman': rho_c, 'composite_pearson': r_c,
            'flops_spearman': rho_f
        }
    }
    
    output_path = '/results/l4_validation.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    volume.commit()
    
    print(f"\n{'='*60}")
    print(f"L4 VALIDATION DONE: {len(measurements)} architectures")
    print(f"Composite proxy: Spearman ρ = {rho_c:.4f}, Pearson r = {r_c:.4f}")
    print(f"FLOPs only:      Spearman ρ = {rho_f:.4f}")
    
    return output
