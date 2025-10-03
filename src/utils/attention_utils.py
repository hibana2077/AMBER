#!/usr/bin/env python
"""
Attention extraction utilities for video transformer models.
"""

import numpy as np
import torch


def minmax_norm(arr: np.ndarray, eps: float = 1e-8):
    """
    Min-max normalize array to [0, 1] range.
    
    Args:
        arr: Input array
        eps: Small epsilon to prevent division by zero
        
    Returns:
        Normalized array
    """
    mn, mx = float(arr.min()), float(arr.max())
    return (arr - mn) / (mx - mn + eps)


def pick_layer_indices(num_layers: int, k: int):
    """
    Pick evenly spaced layer indices.
    
    Args:
        num_layers: Total number of layers
        k: Number of layers to select
        
    Returns:
        List of selected layer indices
    """
    if k >= num_layers:
        return list(range(num_layers))
    ticks = np.linspace(0, num_layers - 1, k)
    return sorted(set(int(round(t)) for t in ticks))


def pick_frame_indices(num_frames: int, k: int):
    """
    Pick evenly spaced frame indices.
    
    Args:
        num_frames: Total number of frames
        k: Number of frames to select
        
    Returns:
        List of selected frame indices
    """
    if k >= num_frames:
        return list(range(num_frames))
    ticks = np.linspace(0, num_frames - 1, k)
    return sorted(set(int(round(t)) for t in ticks))


def extract_cls_to_patch_attn_timesformer(attentions, num_frames: int, h_p: int, w_p: int):
    """
    Extract CLS-to-patch attention maps for TimeSformer models.
    
    Args:
        attentions: Tuple of attention tensors from model output
        num_frames: Number of frames in video
        h_p: Height in patches
        w_p: Width in patches
        
    Returns:
        List of attention maps per layer (num_layers, num_frames, h_p, w_p)
    """
    P = h_p * w_p
    maps = []
    for L, attn in enumerate(attentions):
        if isinstance(attn, (list, tuple)) and len(attn) > 0:
            attn = attn[0]
        attn = attn.detach().to(torch.float32).cpu()
        B_eff, H, N, _ = attn.shape
        attn_mean = attn.mean(dim=1)
        cls_tokens = attn_mean[:, 0, 1:]
        factor = (cls_tokens.shape[1]) // P
        groups = cls_tokens.view(B_eff, factor, P)
        if factor == num_frames and B_eff == 1:
            per_frame = groups[0]
        else:
            per_seq = groups.mean(dim=1)
            if B_eff % num_frames == 0:
                B = B_eff // num_frames
                per_seq = per_seq.view(B, num_frames, P)
                per_frame = per_seq[0]
            elif B_eff == num_frames:
                per_frame = per_seq
            else:
                actual_T = min(B_eff, num_frames)
                per_frame = per_seq[:actual_T]
                if actual_T < num_frames:
                    pad = per_seq[-1:].repeat(num_frames - actual_T, 1)
                    per_frame = torch.cat([per_frame, pad], dim=0)
        grid = per_frame.view(num_frames, h_p, w_p).numpy()
        for t in range(num_frames):
            grid[t] = minmax_norm(grid[t])
        maps.append(grid)
    return maps


def extract_cls_to_patch_attn_videomae_vivit(attentions, num_frames: int, h_p: int, w_p: int, tubelet_size: int):
    """
    Extract CLS-to-patch attention maps for VideoMAE/ViViT models.
    
    Args:
        attentions: Tuple of attention tensors from model output
        num_frames: Number of frames in video
        h_p: Height in patches
        w_p: Width in patches
        tubelet_size: Temporal tubelet size
        
    Returns:
        List of attention maps per layer (num_layers, num_frames, h_p, w_p)
    """
    P = h_p * w_p
    T_patches = max(1, num_frames // max(1, tubelet_size))
    maps = []
    for L, attn in enumerate(attentions):
        if isinstance(attn, (list, tuple)) and len(attn) > 0:
            attn = attn[0]
        attn = attn.detach().to(torch.float32).cpu()
        B, H, N, _ = attn.shape
        attn_mean = attn.mean(dim=1)
        expected_with_cls = T_patches * P + 1
        expected_without_cls = T_patches * P
        if N == expected_with_cls:
            tokens = attn_mean[:, 0, 1:]
            num_patch_tokens = N - 1
        elif N == expected_without_cls:
            tokens = attn_mean.mean(dim=1)
            num_patch_tokens = N
        elif (N - 1) % P == 0:
            tokens = attn_mean[:, 0, 1:]
            num_patch_tokens = N - 1
        elif N % P == 0:
            tokens = attn_mean.mean(dim=1)
            num_patch_tokens = N
        else:
            raise ValueError(f"Unexpected sequence length at layer {L}: {N}")
        t_patches = num_patch_tokens // P
        per_patch = tokens[0].view(t_patches, P)
        grid = per_patch.view(t_patches, h_p, w_p).numpy()
        up = np.zeros((num_frames, h_p, w_p), dtype=grid.dtype)
        for t in range(num_frames):
            tp = min(t // max(1, tubelet_size), t_patches - 1)
            up[t] = grid[tp]
        for t in range(num_frames):
            up[t] = minmax_norm(up[t])
        maps.append(up)
    return maps
