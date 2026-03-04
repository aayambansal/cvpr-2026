"""
Continual / Online NAS with Budgeted Architecture Mutation
==========================================================
Streaming scenario: CIFAR-10 -> CIFAR-100 -> SVHN
Compare: Fixed Backbone, Growing Backbone (naive), BudgetNAS (ours)

Run on MPS (Apple Silicon) or CPU.
"""

import os
import sys
import json
import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

# ===================== SEED =====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ===================== DEVICE =====================
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# ===================== HYPERPARAMETERS =====================
BATCH_SIZE = 128
STREAM_EPOCHS = 5          # epochs per streaming chunk
MUTATION_BUDGET = 3         # max mutations allowed per domain shift
CHUNKS_PER_DATASET = 5      # split each dataset into streaming chunks
LR = 0.01
WEIGHT_DECAY = 1e-4
MAX_TRAIN_SAMPLES = 15000   # subsample for speed
MAX_TEST_SAMPLES = 5000

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===================== DATA LOADING =====================
def get_transforms(dataset_name):
    """Dataset-specific transforms."""
    if dataset_name in ["cifar10", "cifar100"]:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:  # SVHN
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
    return train_tf, test_tf


def load_datasets():
    """Load all three datasets."""
    data_root = os.path.join(os.path.dirname(__file__), "..", "data")
    
    # CIFAR-10
    c10_tr_tf, c10_te_tf = get_transforms("cifar10")
    cifar10_train = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=c10_tr_tf)
    cifar10_test = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=c10_te_tf)
    
    # CIFAR-100
    c100_tr_tf, c100_te_tf = get_transforms("cifar100")
    cifar100_train = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True, transform=c100_tr_tf)
    cifar100_test = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True, transform=c100_te_tf)
    
    # SVHN
    svhn_tr_tf, svhn_te_tf = get_transforms("svhn")
    svhn_train = torchvision.datasets.SVHN(root=data_root, split="train", download=True, transform=svhn_tr_tf)
    svhn_test = torchvision.datasets.SVHN(root=data_root, split="test", download=True, transform=svhn_te_tf)
    
    # Subsample for tractable runtime
    def subsample(ds, max_n):
        if len(ds) > max_n:
            indices = random.sample(range(len(ds)), max_n)
            return Subset(ds, indices)
        return ds
    
    cifar10_train = subsample(cifar10_train, MAX_TRAIN_SAMPLES)
    cifar10_test = subsample(cifar10_test, MAX_TEST_SAMPLES)
    cifar100_train = subsample(cifar100_train, MAX_TRAIN_SAMPLES)
    cifar100_test = subsample(cifar100_test, MAX_TEST_SAMPLES)
    svhn_train = subsample(svhn_train, MAX_TRAIN_SAMPLES)
    svhn_test = subsample(svhn_test, MAX_TEST_SAMPLES)
    
    return {
        "cifar10": (cifar10_train, cifar10_test, 10),
        "cifar100": (cifar100_train, cifar100_test, 100),
        "svhn": (svhn_train, svhn_test, 10),
    }


def make_chunks(dataset, n_chunks):
    """Split dataset into n streaming chunks."""
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    chunk_size = len(indices) // n_chunks
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < n_chunks - 1 else len(indices)
        chunks.append(Subset(dataset, indices[start:end]))
    return chunks


# ===================== MODULAR CNN BLOCK =====================
class ConvBlock(nn.Module):
    """A single convolutional block: Conv -> BN -> ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """Residual block with optional downsample."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


# ===================== MODULAR NETWORK =====================
class ModularNetwork(nn.Module):
    """
    A modular CNN with blocks that can be added/removed.
    Structure: stem -> [blocks] -> global_pool -> classifier
    """
    def __init__(self, num_classes=10, initial_blocks=None):
        super().__init__()
        self.stem = ConvBlock(3, 32, kernel_size=3, stride=1)
        
        if initial_blocks is None:
            # Default: 3 blocks with increasing channels
            initial_blocks = [
                ("res", 32, 64, 1),    # (type, in_ch, out_ch, stride)
                ("res", 64, 128, 2),
                ("res", 128, 128, 1),
            ]
        
        self.blocks = nn.ModuleList()
        self.block_configs = []
        for btype, in_ch, out_ch, stride in initial_blocks:
            if btype == "res":
                self.blocks.append(ResBlock(in_ch, out_ch, stride))
            else:
                self.blocks.append(ConvBlock(in_ch, out_ch, stride=stride))
            self.block_configs.append((btype, in_ch, out_ch, stride))
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        last_ch = initial_blocks[-1][2] if initial_blocks else 32
        self.classifier = nn.Linear(last_ch, num_classes)
        self._last_ch = last_ch
    
    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
    
    def get_arch_description(self):
        """Return architecture as list of (type, channels) tuples."""
        return [(cfg[0], cfg[2]) for cfg in self.block_configs]
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def count_flops_approx(self):
        """Approximate FLOPs for 32x32 input."""
        # Rough estimate
        total = 0
        h, w = 32, 32
        in_ch = 3
        # stem
        total += h * w * 3 * 32 * 3 * 3
        in_ch = 32
        for cfg in self.block_configs:
            btype, _, out_ch, stride = cfg
            if stride == 2:
                h, w = h // 2, w // 2
            total += h * w * in_ch * out_ch * 3 * 3 * 2  # 2 convs for res
            in_ch = out_ch
        return total


# ===================== ARCHITECTURE MUTATIONS =====================
def add_block(model, position="end", block_type="res"):
    """Add a new block to the model. Returns new model."""
    configs = list(model.block_configs)
    if position == "end":
        last_ch = configs[-1][2]
        new_config = (block_type, last_ch, last_ch, 1)
        configs.append(new_config)
    elif position == "widen":
        # Widen last block
        idx = len(configs) - 1
        old = configs[idx]
        new_ch = int(old[2] * 1.5)
        configs[idx] = (old[0], old[1], new_ch, old[3])
        # Need to fix subsequent connections
    
    new_model = ModularNetwork.__new__(ModularNetwork)
    nn.Module.__init__(new_model)
    new_model.stem = copy.deepcopy(model.stem)
    new_model.blocks = nn.ModuleList()
    new_model.block_configs = []
    
    for i, (btype, in_ch, out_ch, stride) in enumerate(configs):
        if i < len(model.blocks) and i < len(model.block_configs) and configs[i] == model.block_configs[i]:
            new_model.blocks.append(copy.deepcopy(model.blocks[i]))
        else:
            if btype == "res":
                new_model.blocks.append(ResBlock(in_ch, out_ch, stride))
            else:
                new_model.blocks.append(ConvBlock(in_ch, out_ch, stride=stride))
        new_model.block_configs.append((btype, in_ch, out_ch, stride))
    
    new_model.pool = nn.AdaptiveAvgPool2d(1)
    last_ch = configs[-1][2]
    new_model._last_ch = last_ch
    new_model.classifier = nn.Linear(last_ch, model.classifier.out_features)
    # Try to copy classifier weights if dimensions match
    if last_ch == model._last_ch:
        new_model.classifier.load_state_dict(model.classifier.state_dict())
    
    return new_model


def remove_block(model, position=-2):
    """Remove a block from the model. Returns new model."""
    if len(model.block_configs) <= 2:
        return model  # Don't remove if too few blocks
    
    configs = list(model.block_configs)
    old_blocks = list(model.blocks)
    
    # Remove the specified block
    idx = position if position >= 0 else len(configs) + position
    idx = max(0, min(idx, len(configs) - 1))
    
    # Fix channel connections after removal
    new_configs = []
    new_block_list = []
    prev_ch = 32  # stem output
    for i in range(len(configs)):
        if i == idx:
            continue
        btype, in_ch, out_ch, stride = configs[i]
        if in_ch != prev_ch:
            in_ch = prev_ch
        new_configs.append((btype, in_ch, out_ch, stride))
        if configs[i] == model.block_configs[i] and in_ch == model.block_configs[i][1]:
            new_block_list.append(copy.deepcopy(old_blocks[i]))
        else:
            if btype == "res":
                new_block_list.append(ResBlock(in_ch, out_ch, stride))
            else:
                new_block_list.append(ConvBlock(in_ch, out_ch, stride=stride))
        prev_ch = out_ch
    
    new_model = ModularNetwork.__new__(ModularNetwork)
    nn.Module.__init__(new_model)
    new_model.stem = copy.deepcopy(model.stem)
    new_model.blocks = nn.ModuleList(new_block_list)
    new_model.block_configs = new_configs
    new_model.pool = nn.AdaptiveAvgPool2d(1)
    last_ch = new_configs[-1][2]
    new_model._last_ch = last_ch
    new_model.classifier = nn.Linear(last_ch, model.classifier.out_features)
    if last_ch == model._last_ch:
        new_model.classifier.load_state_dict(model.classifier.state_dict())
    return new_model


def add_downsample_block(model):
    """Add a block with stride=2 to increase receptive field."""
    configs = list(model.block_configs)
    last_ch = configs[-1][2]
    new_ch = min(last_ch * 2, 256)
    new_config = ("res", last_ch, new_ch, 2)
    configs.append(new_config)
    
    new_model = ModularNetwork.__new__(ModularNetwork)
    nn.Module.__init__(new_model)
    new_model.stem = copy.deepcopy(model.stem)
    new_model.blocks = nn.ModuleList()
    new_model.block_configs = []
    
    for i, (btype, in_ch, out_ch, stride) in enumerate(configs):
        if i < len(model.blocks):
            new_model.blocks.append(copy.deepcopy(model.blocks[i]))
        else:
            new_model.blocks.append(ResBlock(in_ch, out_ch, stride))
        new_model.block_configs.append((btype, in_ch, out_ch, stride))
    
    new_model.pool = nn.AdaptiveAvgPool2d(1)
    last_ch = configs[-1][2]
    new_model._last_ch = last_ch
    new_model.classifier = nn.Linear(last_ch, model.classifier.out_features)
    return new_model


def replace_classifier(model, new_num_classes):
    """Replace the classifier head for a new number of classes."""
    model.classifier = nn.Linear(model._last_ch, new_num_classes)
    return model


# ===================== DRIFT DETECTOR =====================
class DriftDetector:
    """Simple drift detection based on loss increase."""
    def __init__(self, window_size=3, threshold=0.3):
        self.window_size = window_size
        self.threshold = threshold
        self.loss_history = []
    
    def update(self, loss):
        self.loss_history.append(loss)
    
    def detect_drift(self):
        if len(self.loss_history) < self.window_size + 1:
            return False
        recent = np.mean(self.loss_history[-self.window_size:])
        older = np.mean(self.loss_history[-2*self.window_size:-self.window_size]) if len(self.loss_history) >= 2*self.window_size else self.loss_history[0]
        return (recent - older) / (older + 1e-8) > self.threshold


# ===================== BUDGETED NAS CONTROLLER =====================
class BudgetNASController:
    """
    Online NAS controller with mutation budget.
    Uses a simple scoring mechanism to decide which mutation to apply.
    """
    def __init__(self, mutation_budget=3):
        self.budget = mutation_budget
        self.mutations_used = 0
        self.mutation_log = []
    
    def propose_mutations(self, model, val_acc, prev_val_acc, drift_detected):
        """Propose a mutation based on current performance and drift."""
        if self.mutations_used >= self.budget:
            return None, "budget_exhausted"
        
        if not drift_detected and val_acc > 0.5:
            return None, "no_action_needed"
        
        # Score candidate mutations
        candidates = []
        
        # 1. Add a block (increase capacity)
        if len(model.block_configs) < 8:
            score = 0.5
            if drift_detected:
                score += 0.3
            if val_acc < 0.4:
                score += 0.2
            candidates.append(("add_block", score))
        
        # 2. Add downsample block (for more complex data)
        n_downsample = sum(1 for c in model.block_configs if c[3] == 2)
        if n_downsample < 3:
            score = 0.4
            if drift_detected:
                score += 0.4
            candidates.append(("add_downsample", score))
        
        # 3. Remove block (reduce if overfitting)
        if len(model.block_configs) > 3:
            score = 0.2
            if val_acc > prev_val_acc and not drift_detected:
                score += 0.1
            candidates.append(("remove_block", score))
        
        if not candidates:
            return None, "no_valid_mutations"
        
        # Select best mutation
        candidates.sort(key=lambda x: x[1], reverse=True)
        mutation = candidates[0][0]
        self.mutations_used += 1
        self.mutation_log.append({
            "mutation": mutation,
            "budget_remaining": self.budget - self.mutations_used,
            "val_acc": val_acc,
            "drift_detected": drift_detected,
        })
        return mutation, "applied"
    
    def reset_budget(self):
        """Reset budget for new domain."""
        self.mutations_used = 0


# ===================== TRAINING UTILITIES =====================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return total_loss / len(loader), correct / total


# ===================== METHOD 1: FIXED BACKBONE =====================
def run_fixed_backbone(datasets_dict, device):
    """Baseline: Fixed architecture, just retrain classifier for new domains."""
    print("\n" + "="*60)
    print("METHOD: FIXED BACKBONE")
    print("="*60)
    
    results = {"method": "fixed_backbone", "timeline": [], "arch_changes": [], "compute_log": []}
    total_flops = 0
    step = 0
    
    for ds_name in ["cifar10", "cifar100", "svhn"]:
        train_ds, test_ds, num_classes = datasets_dict[ds_name]
        
        # Create fixed model
        model = ModularNetwork(num_classes=num_classes)
        model = model.to(device)
        
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        chunks = make_chunks(train_ds, CHUNKS_PER_DATASET)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CHUNKS_PER_DATASET * STREAM_EPOCHS)
        
        train_acc = 0.0
        for chunk_idx, chunk in enumerate(chunks):
            chunk_loader = DataLoader(chunk, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
            
            for epoch in range(STREAM_EPOCHS):
                t0 = time.time()
                train_loss, train_acc = train_one_epoch(model, chunk_loader, optimizer, criterion, device)
                elapsed = time.time() - t0
                scheduler.step()
                total_flops += model.count_flops_approx() * len(chunk) * STREAM_EPOCHS
            
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            
            results["timeline"].append({
                "step": step,
                "dataset": ds_name,
                "chunk": chunk_idx,
                "test_acc": test_acc,
                "test_loss": test_loss,
                "train_acc": train_acc,
                "num_params": model.count_params(),
                "num_blocks": len(model.block_configs),
                "cumulative_flops": total_flops,
            })
            print(f"  [{ds_name}] Chunk {chunk_idx}: test_acc={test_acc:.4f}, params={model.count_params()}, blocks={len(model.block_configs)}")
            step += 1
    
    return results


# ===================== METHOD 2: GROWING BACKBONE (NAIVE) =====================
def run_growing_backbone(datasets_dict, device):
    """Baseline: Grows network by adding blocks at every domain shift (no budget)."""
    print("\n" + "="*60)
    print("METHOD: GROWING BACKBONE (NAIVE)")
    print("="*60)
    
    results = {"method": "growing_backbone", "timeline": [], "arch_changes": [], "compute_log": []}
    total_flops = 0
    step = 0
    model = None
    
    for ds_idx, ds_name in enumerate(["cifar10", "cifar100", "svhn"]):
        train_ds, test_ds, num_classes = datasets_dict[ds_name]
        
        if model is None:
            model = ModularNetwork(num_classes=num_classes)
        else:
            # Naively add blocks at each domain shift
            model = add_block(model, position="end")
            model = add_block(model, position="end")
            model = replace_classifier(model, num_classes)
            results["arch_changes"].append({
                "step": step, "dataset": ds_name,
                "action": "add_2_blocks + new_classifier",
                "new_blocks": len(model.block_configs),
            })
        
        model = model.to(device)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        chunks = make_chunks(train_ds, CHUNKS_PER_DATASET)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CHUNKS_PER_DATASET * STREAM_EPOCHS)
        
        train_acc = 0.0
        for chunk_idx, chunk in enumerate(chunks):
            chunk_loader = DataLoader(chunk, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
            
            for epoch in range(STREAM_EPOCHS):
                train_loss, train_acc = train_one_epoch(model, chunk_loader, optimizer, criterion, device)
                scheduler.step()
                total_flops += model.count_flops_approx() * len(chunk) * STREAM_EPOCHS
            
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            
            results["timeline"].append({
                "step": step,
                "dataset": ds_name,
                "chunk": chunk_idx,
                "test_acc": test_acc,
                "test_loss": test_loss,
                "train_acc": train_acc,
                "num_params": model.count_params(),
                "num_blocks": len(model.block_configs),
                "cumulative_flops": total_flops,
            })
            print(f"  [{ds_name}] Chunk {chunk_idx}: test_acc={test_acc:.4f}, params={model.count_params()}, blocks={len(model.block_configs)}")
            step += 1
    
    return results


# ===================== METHOD 3: BUDGETED ONLINE NAS (OURS) =====================
def run_budget_nas(datasets_dict, device):
    """Our method: Online NAS with budgeted mutations and drift detection."""
    print("\n" + "="*60)
    print("METHOD: BUDGETED ONLINE NAS (OURS)")
    print("="*60)
    
    results = {"method": "budget_nas", "timeline": [], "arch_changes": [], "compute_log": []}
    total_flops = 0
    step = 0
    model = None
    prev_val_acc = 0.0
    controller = BudgetNASController(mutation_budget=MUTATION_BUDGET)
    drift_detector = DriftDetector(window_size=2, threshold=0.25)
    
    for ds_idx, ds_name in enumerate(["cifar10", "cifar100", "svhn"]):
        train_ds, test_ds, num_classes = datasets_dict[ds_name]
        
        if model is None:
            model = ModularNetwork(num_classes=num_classes)
        else:
            # Reset budget for new domain
            controller.reset_budget()
            drift_detector = DriftDetector(window_size=2, threshold=0.25)
            model = replace_classifier(model, num_classes)
        
        model = model.to(device)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        chunks = make_chunks(train_ds, CHUNKS_PER_DATASET)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CHUNKS_PER_DATASET * STREAM_EPOCHS)
        
        train_acc = 0.0
        for chunk_idx, chunk in enumerate(chunks):
            chunk_loader = DataLoader(chunk, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
            
            # Train on chunk
            for epoch in range(STREAM_EPOCHS):
                train_loss, train_acc = train_one_epoch(model, chunk_loader, optimizer, criterion, device)
                scheduler.step()
                total_flops += model.count_flops_approx() * len(chunk)
            
            # Evaluate
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            drift_detector.update(test_loss)
            
            # Check for drift and propose mutations
            drift_detected = drift_detector.detect_drift()
            mutation, status = controller.propose_mutations(
                model, test_acc, prev_val_acc, drift_detected
            )
            
            if mutation is not None:
                print(f"    >>> MUTATION: {mutation} (budget remaining: {controller.budget - controller.mutations_used})")
                old_params = model.count_params()
                
                if mutation == "add_block":
                    model = add_block(model, position="end")
                elif mutation == "add_downsample":
                    model = add_downsample_block(model)
                elif mutation == "remove_block":
                    model = remove_block(model)
                
                model = model.to(device)
                # Re-create optimizer for new parameters
                optimizer = optim.SGD(model.parameters(), lr=LR * 0.5, momentum=0.9, weight_decay=WEIGHT_DECAY)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, (CHUNKS_PER_DATASET - chunk_idx) * STREAM_EPOCHS)
                )
                
                results["arch_changes"].append({
                    "step": step,
                    "dataset": ds_name,
                    "chunk": chunk_idx,
                    "mutation": mutation,
                    "old_params": old_params,
                    "new_params": model.count_params(),
                    "budget_remaining": controller.budget - controller.mutations_used,
                    "drift_detected": drift_detected,
                })
            
            prev_val_acc = test_acc
            results["timeline"].append({
                "step": step,
                "dataset": ds_name,
                "chunk": chunk_idx,
                "test_acc": test_acc,
                "test_loss": test_loss,
                "train_acc": train_acc,
                "num_params": model.count_params(),
                "num_blocks": len(model.block_configs),
                "cumulative_flops": total_flops,
                "drift_detected": drift_detected,
                "mutation_applied": mutation,
            })
            print(f"  [{ds_name}] Chunk {chunk_idx}: test_acc={test_acc:.4f}, params={model.count_params()}, blocks={len(model.block_configs)}, drift={drift_detected}")
            step += 1
    
    results["mutation_log"] = controller.mutation_log
    return results


# ===================== MAIN =====================
def main():
    print("Loading datasets...")
    datasets_dict = load_datasets()
    print("Datasets loaded.")
    
    # Run all methods
    fixed_results = run_fixed_backbone(datasets_dict, DEVICE)
    growing_results = run_growing_backbone(datasets_dict, DEVICE)
    budget_results = run_budget_nas(datasets_dict, DEVICE)
    
    # Save results
    all_results = {
        "fixed_backbone": fixed_results,
        "growing_backbone": growing_results,
        "budget_nas": budget_results,
    }
    
    results_path = os.path.join(RESULTS_DIR, "experiment_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {results_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for method_name, method_results in all_results.items():
        timeline = method_results["timeline"]
        # Average accuracy per dataset
        for ds in ["cifar10", "cifar100", "svhn"]:
            ds_accs = [t["test_acc"] for t in timeline if t["dataset"] == ds]
            if ds_accs:
                print(f"  {method_name:25s} | {ds:10s} | avg_acc={np.mean(ds_accs):.4f} | final_acc={ds_accs[-1]:.4f}")
        # Final params
        final_params = timeline[-1]["num_params"]
        final_blocks = timeline[-1]["num_blocks"]
        print(f"  {method_name:25s} | final_params={final_params:,} | final_blocks={final_blocks}")
        print()


if __name__ == "__main__":
    main()
