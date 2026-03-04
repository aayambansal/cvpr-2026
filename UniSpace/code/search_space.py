"""
Unified Modern Vision Backbone Search Space
============================================
A compact, factorized search space mixing:
- Token Mixers: Attention, Depthwise Conv, Gated MLP, State-Space-Lite (SSM-Lite)
- Normalization: BatchNorm, LayerNorm, GroupNorm, RMSNorm
- Downsampling: MaxPool, AvgPool, StridedConv, PatchMerging
- Channel expansion: {1, 2, 4}
- Depth per stage: {1, 2, 3, 4}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools
import numpy as np
from typing import List, Dict, Tuple, Optional
import json

# ============================================================
# Normalization Choices
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W) or (B, N, C)
        if x.dim() == 4:
            rms = torch.sqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
            return x / rms * self.weight[None, :, None, None]
        else:
            rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return x / rms * self.weight

def get_norm(norm_type: str, channels: int):
    if norm_type == 'batch':
        return nn.BatchNorm2d(channels)
    elif norm_type == 'layer':
        return nn.GroupNorm(1, channels)  # LayerNorm equivalent for 4D
    elif norm_type == 'group':
        num_groups = min(32, channels)
        while channels % num_groups != 0:
            num_groups -= 1
        return nn.GroupNorm(num_groups, channels)
    elif norm_type == 'rms':
        return RMSNorm(channels)
    else:
        raise ValueError(f"Unknown norm: {norm_type}")

# ============================================================
# Token Mixer Choices
# ============================================================

class DepthwiseConvMixer(nn.Module):
    """Depthwise separable convolution as token mixer (CNN-style)"""
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)
        self.pwconv = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        return self.pwconv(self.dwconv(x))

class AttentionMixer(nn.Module):
    """Multi-head self-attention as token mixer (ViT-style)"""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = max(dim // num_heads, 1)
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        out = (v @ attn.transpose(-2, -1)).reshape(B, C, H, W)
        return self.proj(out)

class GatedMLPMixer(nn.Module):
    """Gated MLP as token mixer (gMLP-style) - channel-only gating for spatial invariance"""
    def __init__(self, dim, spatial_size=None):
        super().__init__()
        self.proj_in = nn.Conv2d(dim, dim * 2, 1)
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.proj_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        xz = self.proj_in(x)
        u, v = xz.chunk(2, dim=1)
        v = self.dwconv(v)
        out = u * F.silu(v)
        return self.proj_out(out)

class SSMLiteMixer(nn.Module):
    """Simplified State-Space-style mixer (Mamba-lite, 1D scan over flattened spatial)"""
    def __init__(self, dim, state_dim=16):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.proj_in = nn.Conv2d(dim, dim * 2, 1)
        self.conv1d = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.dt_proj = nn.Linear(dim, dim)
        self.A = nn.Parameter(torch.randn(dim, state_dim))
        self.D = nn.Parameter(torch.ones(dim))
        self.proj_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        xz = self.proj_in(x)
        x_in, z = xz.chunk(2, dim=1)
        
        # 1D scan along flattened spatial
        x_flat = x_in.reshape(B, C, H*W)
        x_conv = self.conv1d(x_flat)
        x_conv = F.silu(x_conv)
        
        # Simplified selective scan (no recurrence for efficiency)
        dt = F.softplus(self.dt_proj(x_conv.transpose(1, 2))).transpose(1, 2)
        A_exp = torch.exp(-F.softplus(self.A))  # (C, state_dim)
        
        # Approximate: y = D * x + cumulative gated sum
        y = self.D[None, :, None] * x_conv + dt * x_conv
        
        # Gate with z
        z_act = F.silu(z.reshape(B, C, H*W))
        out = y * z_act
        out = out.reshape(B, C, H, W)
        return self.proj_out(out)

def get_token_mixer(mixer_type: str, dim: int):
    if mixer_type == 'conv':
        return DepthwiseConvMixer(dim)
    elif mixer_type == 'attention':
        return AttentionMixer(dim)
    elif mixer_type == 'gated_mlp':
        return GatedMLPMixer(dim)
    elif mixer_type == 'ssm_lite':
        return SSMLiteMixer(dim)
    else:
        raise ValueError(f"Unknown mixer: {mixer_type}")

# ============================================================
# Downsampling Choices
# ============================================================

def get_downsample(ds_type: str, in_channels: int, out_channels: int):
    if ds_type == 'maxpool':
        return nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )
    elif ds_type == 'avgpool':
        return nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )
    elif ds_type == 'strided_conv':
        return nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
    elif ds_type == 'patch_merging':
        return nn.Sequential(
            nn.Unfold(kernel_size=2, stride=2),
            nn.Linear(in_channels * 4, out_channels),
            # Reshape handled in forward
        )
    else:
        raise ValueError(f"Unknown downsample: {ds_type}")

class PatchMergingDown(nn.Module):
    """Swin-style patch merging"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 4, out_channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        # Pad if odd
        if H % 2 == 1: x = F.pad(x, (0, 0, 0, 1))
        if W % 2 == 1: x = F.pad(x, (0, 1, 0, 0))
        B, C, H, W = x.shape
        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x_cat = torch.cat([x0, x1, x2, x3], dim=1)  # B, 4C, H/2, W/2
        return self.conv(x_cat)

def get_downsample_module(ds_type: str, in_channels: int, out_channels: int):
    if ds_type == 'maxpool':
        layers = [nn.MaxPool2d(2, 2)]
        if in_channels != out_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, 1))
        return nn.Sequential(*layers)
    elif ds_type == 'avgpool':
        layers = [nn.AvgPool2d(2, 2)]
        if in_channels != out_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, 1))
        return nn.Sequential(*layers)
    elif ds_type == 'strided_conv':
        return nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
    elif ds_type == 'patch_merging':
        return PatchMergingDown(in_channels, out_channels)
    else:
        raise ValueError(f"Unknown downsample: {ds_type}")

# ============================================================
# Building Block
# ============================================================

class MetaBlock(nn.Module):
    """Universal building block: Norm -> TokenMixer -> Residual -> FFN"""
    def __init__(self, dim, mixer_type, norm_type, ffn_ratio=4):
        super().__init__()
        self.norm1 = get_norm(norm_type, dim)
        self.mixer = get_token_mixer(mixer_type, dim)
        self.norm2 = get_norm(norm_type, dim)
        ffn_hidden = int(dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, ffn_hidden, 1),
            nn.GELU(),
            nn.Conv2d(ffn_hidden, dim, 1),
        )

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# ============================================================
# Full Architecture from Config
# ============================================================

class UnifiedBackbone(nn.Module):
    """
    A backbone assembled from the factorized search space.
    
    Config: {
        'stem_channels': int,
        'stages': [
            {
                'depth': int (1-4),
                'mixer': str ('conv', 'attention', 'gated_mlp', 'ssm_lite'),
                'norm': str ('batch', 'layer', 'group', 'rms'),
                'downsample': str ('maxpool', 'avgpool', 'strided_conv', 'patch_merging'),
                'expansion': int (1, 2, 4),
            },
            ...  # 4 stages
        ],
    }
    """
    def __init__(self, config, num_classes=10, input_size=32):
        super().__init__()
        stem_ch = config['stem_channels']
        
        # Stem: simple conv
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(stem_ch),
            nn.GELU(),
        )
        
        stages = []
        in_ch = stem_ch
        for i, stage_cfg in enumerate(config['stages']):
            out_ch = in_ch * stage_cfg['expansion']
            out_ch = max(out_ch, in_ch)  # never shrink
            
            # Downsample (except first stage)
            if i > 0:
                stages.append(get_downsample_module(
                    stage_cfg['downsample'], in_ch, out_ch
                ))
            elif in_ch != out_ch:
                stages.append(nn.Conv2d(in_ch, out_ch, 1))
            
            # Blocks
            for _ in range(stage_cfg['depth']):
                stages.append(MetaBlock(
                    out_ch, stage_cfg['mixer'], stage_cfg['norm']
                ))
            
            in_ch = out_ch
        
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

# ============================================================
# Search Space Definition
# ============================================================

SEARCH_SPACE = {
    'stem_channels': [32, 48, 64],
    'mixer_types': ['conv', 'attention', 'gated_mlp', 'ssm_lite'],
    'norm_types': ['batch', 'layer', 'group', 'rms'],
    'downsample_types': ['maxpool', 'avgpool', 'strided_conv', 'patch_merging'],
    'expansion_ratios': [1, 2],  # per stage
    'depths': [1, 2, 3],
}

def compute_search_space_size():
    """Compute total number of unique architectures."""
    n_stem = len(SEARCH_SPACE['stem_channels'])
    n_mixer = len(SEARCH_SPACE['mixer_types'])
    n_norm = len(SEARCH_SPACE['norm_types'])
    n_ds = len(SEARCH_SPACE['downsample_types'])
    n_exp = len(SEARCH_SPACE['expansion_ratios'])
    n_depth = len(SEARCH_SPACE['depths'])
    
    # 4 stages, each with independent choices
    stage_choices = (n_mixer * n_norm * n_ds * n_exp * n_depth) ** 4
    total = n_stem * stage_choices
    return total

def sample_random_config(seed=None):
    """Sample a random architecture configuration."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    config = {
        'stem_channels': int(rng.choice(SEARCH_SPACE['stem_channels'])),
        'stages': []
    }
    
    for i in range(4):
        stage = {
            'depth': int(rng.choice(SEARCH_SPACE['depths'])),
            'mixer': str(rng.choice(SEARCH_SPACE['mixer_types'])),
            'norm': str(rng.choice(SEARCH_SPACE['norm_types'])),
            'downsample': str(rng.choice(SEARCH_SPACE['downsample_types'])),
            'expansion': int(rng.choice(SEARCH_SPACE['expansion_ratios'])),
        }
        config['stages'].append(stage)
    
    return config

def config_to_vector(config):
    """Convert config to a numerical vector for analysis."""
    mixer_map = {'conv': 0, 'attention': 1, 'gated_mlp': 2, 'ssm_lite': 3}
    norm_map = {'batch': 0, 'layer': 1, 'group': 2, 'rms': 3}
    ds_map = {'maxpool': 0, 'avgpool': 1, 'strided_conv': 2, 'patch_merging': 3}
    stem_map = {32: 0, 48: 1, 64: 2}
    
    vec = [stem_map.get(config['stem_channels'], 0)]
    for stage in config['stages']:
        vec.extend([
            mixer_map[stage['mixer']],
            norm_map[stage['norm']],
            ds_map[stage['downsample']],
            stage['expansion'],
            stage['depth'],
        ])
    return np.array(vec, dtype=np.float32)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_flops_approx(model, input_size=(1, 3, 32, 32)):
    """Approximate FLOPs using parameter count heuristic."""
    # Simple approximation: ~2 * params * spatial_elements
    params = count_parameters(model)
    return params * 2  # rough estimate

if __name__ == '__main__':
    total = compute_search_space_size()
    print(f"Total search space size: {total:,} architectures")
    print(f"  = {total:.2e}")
    
    # Sample and build
    config = sample_random_config(seed=42)
    print(f"\nSample config: {json.dumps(config, indent=2)}")
    
    model = UnifiedBackbone(config, num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {count_parameters(model):,}")
