"""Lightweight video transform utilities.

These replace the previous dependency on pytorchvideo.transforms to avoid
deprecated torchvision private APIs (e.g. functional_tensor) while keeping
the minimal functionality needed for training/evaluating video models.

All transforms operate on a 4D video tensor shaped (C, T, H, W).
"""

from __future__ import annotations

from typing import Callable, List, Any
import torch
import random

__all__ = [
    "UniformTemporalSubsampleTransform",
    "RandomShortSideScaleTransform",
    "ShortSideResizeTransform",
    "RandomCropVideoTransform",
    "CenterCropVideoTransform",
    "RandomHorizontalFlipVideoTransform",
    "NormalizeVideoTransform",
    "ComposeVideo",
    "apply_to_key",
]


class UniformTemporalSubsampleTransform:
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        if video.dim() != 4:
            raise ValueError(f"Expected (C,T,H,W); got {video.shape}")
        C, T, H, W = video.shape
        if T == self.num_samples:
            return video
        indices = torch.linspace(0, T - 1, steps=self.num_samples).long()
        return video[:, indices]


class RandomShortSideScaleTransform:
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        size = random.randint(self.min_size, self.max_size)
        return short_side_resize(video, size)


class ShortSideResizeTransform:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        return short_side_resize(video, self.size)


def short_side_resize(video: torch.Tensor, target_short: int) -> torch.Tensor:
    C, T, H, W = video.shape
    short = min(H, W)
    if short == target_short:
        return video
    scale = target_short / short
    new_h = int(round(H * scale))
    new_w = int(round(W * scale))
    v = video.permute(1, 0, 2, 3)  # (T,C,H,W)
    v = torch.nn.functional.interpolate(v, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return v.permute(1, 0, 2, 3)


class RandomCropVideoTransform:
    def __init__(self, size: tuple[int, int]):
        self.th, self.tw = size

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        C, T, H, W = video.shape
        if H < self.th or W < self.tw:
            v = video.permute(1, 0, 2, 3)
            v = torch.nn.functional.interpolate(v, size=(max(self.th, H), max(self.tw, W)), mode="bilinear", align_corners=False)
            video = v.permute(1, 0, 2, 3)
            C, T, H, W = video.shape
        i = random.randint(0, H - self.th)
        j = random.randint(0, W - self.tw)
        return video[:, :, i : i + self.th, j : j + self.tw]


class CenterCropVideoTransform:
    def __init__(self, size: tuple[int, int]):
        self.th, self.tw = size

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        C, T, H, W = video.shape
        i = max(0, (H - self.th) // 2)
        j = max(0, (W - self.tw) // 2)
        i2 = min(H, i + self.th)
        j2 = min(W, j + self.tw)
        return video[:, :, i:i2, j:j2]


class RandomHorizontalFlipVideoTransform:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return video.flip(-1)
        return video


class NormalizeVideoTransform:
    def __init__(self, mean, std):
        self.registered_mean = torch.tensor(mean).view(-1, 1, 1, 1)
        self.registered_std = torch.tensor(std).view(-1, 1, 1, 1)

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        return (video - self.registered_mean.to(video.device)) / self.registered_std.to(video.device)


class ComposeVideo:
    def __init__(self, ops: List[Callable[[torch.Tensor], torch.Tensor]]):
        self.ops = ops

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        for op in self.ops:
            video = op(video)
        return video


def apply_to_key(key: str, fn: Callable[[Any], Any]):
    """Return a callable(sample_dict) that applies fn to sample[key]."""
    def wrapper(sample: dict):
        sample[key] = fn(sample[key])
        return sample
    return wrapper
