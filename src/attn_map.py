#!/usr/bin/env python
"""
AMBER attention map extractor

CLI to extract layer-wise attention heatmaps over frames for supported
Transformers video models (TimeSformer, VideoMAE, ViViT, V-JEPA2). It follows the
approach in example/*_test.py files, consolidating into one tool.

Example (PowerShell):
  python -m src.attn_map \
    --video "path/to/video.mp4" \
    --model facebook/timesformer-base-finetuned-k400 \
    --out outputs/attn_timesformer.png

  python -m src.attn_map --video path/to/video.mp4 --model MCG-NJU/videomae-base-finetuned-kinetics --out outputs/attn_videomae.png
  python -m src.attn_map --video path/to/video.mp4 --model google/vivit-b-16x2-kinetics400 --out outputs/attn_vivit.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch

try:
    import cv2
except Exception as e:  # pragma: no cover
    raise SystemExit("OpenCV is required: pip install opencv-python\n" + str(e))

try:
    import imageio.v2 as imageio
    _HAS_IMAGEIO = True
except Exception:
    imageio = None
    _HAS_IMAGEIO = False

from transformers import (
    AutoImageProcessor,
    AutoVideoProcessor,
    AutoModelForVideoClassification,
    TimesformerForVideoClassification,
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
    VivitImageProcessor,
    VivitForVideoClassification,
)

# Optional explicit V-JEPA2 imports (available in newer Transformers)
try:  # pragma: no cover
    from transformers import VJEPA2ForVideoClassification, VJEPA2VideoProcessor  # type: ignore
    _HAS_VJEPA2 = True
except Exception:  # pragma: no cover
    VJEPA2ForVideoClassification = None  # type: ignore
    VJEPA2VideoProcessor = None  # type: ignore
    _HAS_VJEPA2 = False


def sample_video_frames_cv2(path: str, num_frames: int = 16, target_size: int = 224) -> List[Image.Image]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        idxs = np.round(np.linspace(0, total - 1, num_frames)).astype(int)
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
    else:
        # sequential read
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        if not frames:
            cap.release()
            raise RuntimeError("No frames decoded from video (OpenCV)")
        idxs = np.round(np.linspace(0, len(frames) - 1, num_frames)).astype(int)
        frames = [frames[i] for i in idxs]
    cap.release()

    # Ensure exact count via padding
    while len(frames) < num_frames:
        frames.append(frames[-1])

    pil = []
    for fr in frames[:num_frames]:
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        h, w, _ = fr.shape
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        fr = fr[y0:y0 + side, x0:x0 + side]
        fr = cv2.resize(fr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        pil.append(Image.fromarray(fr))
    return pil


def sample_video_frames_imageio(path: str, num_frames: int = 16, target_size: int = 224) -> List[Image.Image]:
    if not _HAS_IMAGEIO:
        raise RuntimeError("imageio not available. pip install imageio imageio-ffmpeg")
    try:
        reader = imageio.get_reader(path)
    except Exception as e:
        raise FileNotFoundError(f"Could not open video with imageio: {path}\n{e}")

    try:
        meta = reader.get_meta_data()
        total = int(meta.get("nframes", 0))
    except Exception:
        total = 0
    frames = []
    if total and total > 0 and total != float("inf"):
        idxs = np.round(np.linspace(0, total - 1, num_frames)).astype(int)
        for i in idxs:
            try:
                fr = reader.get_data(int(i))
                frames.append(fr)
            except Exception:
                pass
    else:
        try:
            for fr in reader:
                frames.append(fr)
        except Exception:
            pass
        if not frames:
            reader.close()
            raise RuntimeError("No frames decoded from video (imageio)")
        idxs = np.round(np.linspace(0, len(frames) - 1, num_frames)).astype(int)
        frames = [frames[i] for i in idxs]
    reader.close()

    while len(frames) < num_frames and len(frames) > 0:
        frames.append(frames[-1])

    pil = []
    for fr in frames[:num_frames]:
        if fr.ndim == 2:
            fr = np.stack([fr, fr, fr], axis=-1)
        if fr.shape[-1] == 4:
            fr = fr[..., :3]
        fr = fr.astype(np.uint8)
        h, w, _ = fr.shape
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        fr = fr[y0:y0 + side, x0:x0 + side]
        fr = cv2.resize(fr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        pil.append(Image.fromarray(fr))
    return pil


def sample_video_frames(path: str, num_frames: int, target_size: int = 224) -> List[Image.Image]:
    try:
        return sample_video_frames_cv2(path, num_frames=num_frames, target_size=target_size)
    except Exception as e_cv:
        print(f"[decoder] OpenCV failed: {e_cv}. Trying imageio fallback…")
        return sample_video_frames_imageio(path, num_frames=num_frames, target_size=target_size)


def minmax_norm(arr: np.ndarray, eps: float = 1e-8):
    mn, mx = float(arr.min()), float(arr.max())
    return (arr - mn) / (mx - mn + eps)


def pick_layer_indices(num_layers: int, k: int):
    if k >= num_layers:
        return list(range(num_layers))
    ticks = np.linspace(0, num_layers - 1, k)
    return sorted(set(int(round(t)) for t in ticks))


def pick_frame_indices(num_frames: int, k: int):
    if k >= num_frames:
        return list(range(num_frames))
    ticks = np.linspace(0, num_frames - 1, k)
    return sorted(set(int(round(t)) for t in ticks))


def extract_cls_to_patch_attn_timesformer(attentions, num_frames: int, h_p: int, w_p: int):
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


def overlay_grid(frames_pil: List[Image.Image], attn_maps, layer_idxs, frame_idxs, title_prefix: str = "", cmap="jet", custom_colors=None):
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


def main():
    parser = argparse.ArgumentParser(description="Extract attention maps for video transformer models")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, help="HF model id or local path")
    parser.add_argument("--num-frames", type=int, default=None, help="Frames to sample; default uses model.config num_frames")
    parser.add_argument("--grid-frames", type=int, default=4)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--cmap", type=str, default="jet")
    parser.add_argument("--custom-colors", type=str, nargs="+", default=None)
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model_id = args.model

    # Identify model family by attempting to load processors/models
    family = None
    processor = None
    model = None

    try:
        # Try TimeSformer first
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = TimesformerForVideoClassification.from_pretrained(model_id)
        family = "timesformer"
    except Exception:
        try:
            processor = VideoMAEImageProcessor.from_pretrained(model_id)
            model = VideoMAEForVideoClassification.from_pretrained(model_id, attn_implementation="eager")
            model.config.output_attentions = True
            family = "videomae"
        except Exception:
            try:
                processor = VivitImageProcessor.from_pretrained(model_id)
                model = VivitForVideoClassification.from_pretrained(model_id, attn_implementation="eager")
                model.config.output_attentions = True
                family = "vivit"
            except Exception:
                # Try V-JEPA2 specific or generic Auto* as a final attempt
                try:
                    if _HAS_VJEPA2:
                        processor = VJEPA2VideoProcessor.from_pretrained(model_id)  # type: ignore
                        model = VJEPA2ForVideoClassification.from_pretrained(model_id, attn_implementation="eager")  # type: ignore
                    else:
                        processor = AutoVideoProcessor.from_pretrained(model_id)
                        model = AutoModelForVideoClassification.from_pretrained(model_id, attn_implementation="eager")
                    if hasattr(model, "config"):
                        model.config.output_attentions = True
                    family = "vjepa2"
                except Exception as e:
                    raise SystemExit(f"Failed to load model/processor for id '{model_id}'. Install latest transformers and verify the repo id.\n{e}")

    model.to(args.device)
    model.eval()

    image_size = getattr(model.config, "image_size", 224)
    patch_size = getattr(model.config, "patch_size", 16)
    h_p = image_size // patch_size
    w_p = image_size // patch_size

    expected_frames = getattr(model.config, "num_frames", None) or getattr(model.config, "frames_per_clip", 16)
    num_frames = args.num_frames or expected_frames

    frames_pil = sample_video_frames(str(video_path), num_frames=num_frames, target_size=image_size)
    T = len(frames_pil)

    if family == "timesformer":
        # Adjust to expected frames for TimeSformer
        if T != expected_frames:
            if T > expected_frames:
                idxs = np.linspace(0, T - 1, expected_frames).astype(int)
                frames_pil = [frames_pil[i] for i in idxs]
            else:
                frames_pil = frames_pil + [frames_pil[-1]] * (expected_frames - T)
            T = expected_frames
        inputs = processor(frames_pil, return_tensors="pt")
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            logits = outputs.logits
            attentions = outputs.attentions
        maps = extract_cls_to_patch_attn_timesformer(attentions, num_frames=T, h_p=h_p, w_p=w_p)
    else:
        # VideoMAE/ViViT/V-JEPA2 expect list of numpy arrays
        video_np = [np.array(f) for f in frames_pil]
        inputs = processor(video_np, return_tensors="pt")
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            attentions = outputs.attentions
        tubelet_raw = getattr(model.config, "tubelet_size", 1)
        tubelet = tubelet_raw[0] if isinstance(tubelet_raw, (list, tuple)) else tubelet_raw
        maps = extract_cls_to_patch_attn_videomae_vivit(attentions, num_frames=T, h_p=h_p, w_p=w_p, tubelet_size=tubelet)

    if attentions is None:
        raise RuntimeError("Model did not return attentions; ensure output_attentions=True and eager attention if needed.")

    num_layers = len(attentions)
    layer_idxs = pick_layer_indices(num_layers, args.layers)
    frame_idxs = pick_frame_indices(T, args.grid_frames)

    pred_idx = int(logits.argmax(-1).item())
    label = model.config.id2label.get(pred_idx, str(pred_idx))
    title = f"{video_path.name}  |  pred: {label}"

    custom_colors = None
    if args.custom_colors:
        custom_colors = [c if c.startswith('#') else f"#{c}" for c in args.custom_colors]

    fig = overlay_grid(frames_pil, maps, layer_idxs, frame_idxs, title_prefix=title, cmap=args.cmap, custom_colors=custom_colors)
    out_path = Path(args.out) if args.out else Path(f"attn_{family}_{video_path.stem}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved attention grid to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
