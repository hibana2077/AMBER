#!/usr/bin/env python
"""
Visualization utilities for attention maps and video frames.
"""

from typing import List, Optional

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def overlay_grid(
    frames_pil: List[Image.Image],
    attn_maps,
    layer_idxs,
    frame_idxs,
    title_prefix: str = "",
    cmap="jet",
    custom_colors: Optional[List[str]] = None
):
    """
    Create a grid visualization of attention maps overlaid on video frames.
    
    Args:
        frames_pil: List of PIL Images (video frames)
        attn_maps: List of attention maps per layer
        layer_idxs: Indices of layers to visualize
        frame_idxs: Indices of frames to visualize
        title_prefix: Title for the figure
        cmap: Colormap name for attention heatmap
        custom_colors: Custom color list for colormap
        
    Returns:
        Matplotlib figure object
    """
    assert len(frames_pil) > 0
    W, H = frames_pil[0].size
    if custom_colors:
        cmap = LinearSegmentedColormap.from_list("custom_attn", custom_colors)

    rows = len(layer_idxs)
    cols = len(frame_idxs)
    fig_w = max(6, cols * 3)
    fig_h = max(6, rows * 3.2)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[None, :]
    elif cols == 1:
        axes = axes[:, None]

    for r, L in enumerate(layer_idxs):
        grid = attn_maps[L]
        for c, t in enumerate(frame_idxs):
            ax = axes[r, c]
            ax.imshow(frames_pil[t])
            heat = Image.fromarray((grid[t] * 255).astype(np.uint8)).resize((W, H), resample=Image.BILINEAR)
            ax.imshow(heat, alpha=0.5, cmap=cmap)
            if r == 0:
                ax.set_title(f"frame {t}")
            if c == 0:
                ax.set_ylabel(f"layer {L}")
            ax.set_xticks([])
            ax.set_yticks([])
    if title_prefix:
        fig.suptitle(title_prefix, fontsize=14)
    fig.tight_layout()
    return fig
