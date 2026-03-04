"""
Training-Free NAS Proxy Scores
================================
Implements multiple zero-cost proxies:
1. NASWOT (Neural Architecture Search Without Training)
2. SynFlow (Synaptic Flow)
3. GradNorm
4. Jacob Covariance (jacob_cov)
5. SNIP
6. Parameter count (baseline)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict
import copy


def get_batch(batch_size=64, input_size=32, num_classes=10):
    """Generate a random batch for scoring."""
    x = torch.randn(batch_size, 3, input_size, input_size)
    y = torch.randint(0, num_classes, (batch_size,))
    return x, y


# ============================================================
# NASWOT Score (Mellor et al., 2021)
# ============================================================

def compute_naswot(model, batch_size=64, input_size=32):
    """
    NASWOT: score based on overlap of activation patterns.
    Higher is better (more diverse activations).
    """
    model.eval()
    x, _ = get_batch(batch_size, input_size)
    
    # Collect activation patterns (binary: >0 or not)
    activations = []
    hooks = []
    
    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor) and output.dim() >= 2:
            # Binary activation pattern
            act = (output > 0).float()
            act = act.view(act.size(0), -1)  # flatten spatial
            activations.append(act)
    
    for module in model.modules():
        if isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU)):
            hooks.append(module.register_forward_hook(hook_fn))
    
    with torch.no_grad():
        model(x)
    
    for h in hooks:
        h.remove()
    
    if len(activations) == 0:
        return 0.0
    
    # Concatenate all activation patterns
    all_acts = torch.cat(activations, dim=1)  # (B, total_features)
    
    # Compute kernel matrix K = all_acts @ all_acts.T
    K = all_acts @ all_acts.T
    
    # Score = log|K| (higher = more diverse)
    try:
        # Add small diagonal for numerical stability
        K = K + 1e-5 * torch.eye(K.size(0))
        score = torch.slogdet(K)[1].item()
    except:
        score = 0.0
    
    return score


# ============================================================
# SynFlow (Tanaka et al., 2020)
# ============================================================

def compute_synflow(model, input_size=32):
    """
    SynFlow: sum of (parameter * gradient) products.
    Higher is better (more synaptic flow).
    """
    model.eval()
    
    # Make all parameters positive
    signs = {}
    for name, p in model.named_parameters():
        signs[name] = torch.sign(p.data)
        p.data.abs_()
    
    # Forward pass with ones
    x = torch.ones(1, 3, input_size, input_size)
    
    model.zero_grad()
    output = model(x)
    loss = output.sum()
    loss.backward()
    
    # SynFlow score = sum of (param * grad)
    score = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None:
            score += (p.data * p.grad.data).sum().item()
    
    # Restore signs
    for name, p in model.named_parameters():
        if name in signs:
            p.data *= signs[name]
    
    model.zero_grad()
    return np.log(abs(score) + 1e-10)


# ============================================================
# GradNorm
# ============================================================

def compute_gradnorm(model, batch_size=64, input_size=32, num_classes=10):
    """
    GradNorm: L2 norm of gradients at initialization.
    """
    model.train()
    x, y = get_batch(batch_size, input_size, num_classes)
    
    model.zero_grad()
    output = model(x)
    loss = F.cross_entropy(output, y)
    loss.backward()
    
    grad_norm = 0.0
    count = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).item() ** 2
            count += 1
    
    model.zero_grad()
    return np.log(np.sqrt(grad_norm) + 1e-10) if count > 0 else 0.0


# ============================================================
# Jacob Covariance (jacob_cov)
# ============================================================

def compute_jacob_cov(model, batch_size=32, input_size=32, num_classes=10):
    """
    Jacob Covariance: score based on Jacobian of network output w.r.t. input.
    Higher log-determinant of covariance = more expressive.
    """
    model.eval()
    x = torch.randn(batch_size, 3, input_size, input_size, requires_grad=True)
    
    output = model(x)
    
    # Compute Jacobian rows
    jacobians = []
    for i in range(min(num_classes, output.size(1))):
        model.zero_grad()
        if x.grad is not None:
            x.grad.zero_()
        output[:, i].sum().backward(retain_graph=True)
        if x.grad is not None:
            jacobians.append(x.grad.data.view(batch_size, -1).clone())
    
    if len(jacobians) == 0:
        return 0.0
    
    # Stack: (num_classes, batch_size, input_dim)
    J = torch.stack(jacobians, dim=0)  # (C, B, D)
    
    # Covariance of Jacobian across batch
    J_flat = J.permute(1, 0, 2).reshape(batch_size, -1)  # (B, C*D)
    
    # Use correlation matrix for stability
    J_centered = J_flat - J_flat.mean(dim=0, keepdim=True)
    cov = (J_centered.T @ J_centered) / (batch_size - 1)
    
    try:
        # Use eigenvalues for score
        eigenvalues = torch.linalg.eigvalsh(cov + 1e-5 * torch.eye(cov.size(0)))
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        score = eigenvalues.log().sum().item()
    except:
        score = 0.0
    
    return score


# ============================================================
# SNIP (Single-shot Network Pruning)
# ============================================================

def compute_snip(model, batch_size=64, input_size=32, num_classes=10):
    """
    SNIP: connection sensitivity at initialization.
    """
    model.train()
    x, y = get_batch(batch_size, input_size, num_classes)
    
    model.zero_grad()
    output = model(x)
    loss = F.cross_entropy(output, y)
    loss.backward()
    
    score = 0.0
    for p in model.parameters():
        if p.grad is not None:
            score += (p.data * p.grad.data).abs().sum().item()
    
    model.zero_grad()
    return np.log(score + 1e-10)


# ============================================================
# Combined Scoring
# ============================================================

def compute_all_scores(model, input_size=32, num_classes=10) -> Dict[str, float]:
    """Compute all training-free scores for a model."""
    from search_space import count_parameters
    
    scores = {}
    
    # Parameter count
    scores['params'] = count_parameters(model)
    scores['log_params'] = np.log(scores['params'] + 1)
    
    # NASWOT
    try:
        model_copy = copy.deepcopy(model)
        scores['naswot'] = compute_naswot(model_copy, batch_size=32, input_size=input_size)
    except Exception as e:
        scores['naswot'] = 0.0
    
    # SynFlow
    try:
        model_copy = copy.deepcopy(model)
        scores['synflow'] = compute_synflow(model_copy, input_size=input_size)
    except Exception as e:
        scores['synflow'] = 0.0
    
    # GradNorm
    try:
        model_copy = copy.deepcopy(model)
        scores['gradnorm'] = compute_gradnorm(model_copy, batch_size=32, 
                                                input_size=input_size, num_classes=num_classes)
    except Exception as e:
        scores['gradnorm'] = 0.0
    
    # SNIP
    try:
        model_copy = copy.deepcopy(model)
        scores['snip'] = compute_snip(model_copy, batch_size=32,
                                       input_size=input_size, num_classes=num_classes)
    except Exception as e:
        scores['snip'] = 0.0
    
    # Jacob Cov (expensive, use smaller batch)
    try:
        model_copy = copy.deepcopy(model)
        scores['jacob_cov'] = compute_jacob_cov(model_copy, batch_size=8,
                                                  input_size=input_size, num_classes=num_classes)
    except Exception as e:
        scores['jacob_cov'] = 0.0
    
    return scores


if __name__ == '__main__':
    from search_space import UnifiedBackbone, sample_random_config
    
    config = sample_random_config(seed=42)
    model = UnifiedBackbone(config, num_classes=10)
    
    scores = compute_all_scores(model)
    print("Training-free scores:")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}")
