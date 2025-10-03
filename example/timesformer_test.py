#!/usr/bin/env python
"""
Timesformer Attention Map Visualizer

- Loads a video, uniformly samples frames, runs TimeSformer, and extracts
  CLS->patch attention maps across layers.
- Produces a grid image: rows = selected layers, cols = selected frames.

Requirements (install if needed):
    pip install torch torchvision transformers pillow matplotlib opencv-python numpy tqdm
    # Optional fallback decoder for broader codec support (e.g., some .avi files):
    pip install imageio imageio-ffmpeg

Example:
    python timesformer_attention_viz.py \
        --video "/content/video/P02_122.MP4" \
        --num-frames 16 \
        --layers 5 \
        --grid-frames 4 \
        --out "attn_P02_122.png"

    python timesformer_attention_viz.py --video "/content/video/P03_cereal.avi" --out attn_P03.png

Notes:
- This script averages heads to a single attention map per layer.
- It assumes output_attentions=True is supported by the model version.
- If attentions are None, please upgrade `transformers`.
- Video decoding: tries OpenCV first; if it fails to open or sample frames (common with rare AVI codecs),
  it will fall back to ImageIO+FFmpeg when available.
"""

import argparse
import os
from pathlib import Path
import math

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
from tqdm import tqdm

try:
    import cv2
except ImportError as e:
    raise SystemExit("OpenCV (opencv-python) is required. Install with `pip install opencv-python`.\n" + str(e))

from transformers import AutoImageProcessor, TimesformerForVideoClassification

# Optional: ImageIO fallback for broader container/codec support
try:
    import imageio.v2 as imageio  # v2 API compatibility
    _HAS_IMAGEIO = True
except Exception:
    imageio = None
    _HAS_IMAGEIO = False


def sample_video_frames_cv2(path: str, num_frames: int = 8, target_size: int = 224):
    """Sample `num_frames` frames uniformly from the video and resize to target_size.
    Returns: list of PIL.Image in RGB of size (target_size, target_size).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video with OpenCV: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # Some codecs might not report total; fallback to linear read
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        total = len(frames)
        if total == 0:
            cap.release()
            raise RuntimeError("Video appears to have 0 frames (OpenCV path).")
        # Pick indices and convert
        idxs = np.linspace(0, total - 1, num_frames).astype(int)
        sampled = [frames[i] for i in idxs]
    else:
        idxs = np.linspace(0, total - 1, num_frames).astype(int)
        sampled = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                # Attempt to continue; if last frames fail, we break
                continue
            sampled.append(frame)
        if len(sampled) == 0:
            cap.release()
            raise RuntimeError("Failed to sample frames from video (OpenCV path).")

    cap.release()

    # Convert to PIL RGB and resize square
    pil_frames = []
    for fr in sampled:
        # BGR -> RGB
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        # center-crop to square
        h, w, _ = fr.shape
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        fr = fr[y0:y0 + side, x0:x0 + side]
        # resize
        fr = cv2.resize(fr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        pil_frames.append(Image.fromarray(fr))
    return pil_frames


def sample_video_frames_imageio(path: str, num_frames: int = 8, target_size: int = 224):
    """Fallback sampler using ImageIO (FFmpeg under the hood).
    Returns: list of PIL.Image in RGB of size (target_size, target_size).
    """
    if not _HAS_IMAGEIO:
        raise RuntimeError(
            "ImageIO is not available. Install with `pip install imageio imageio-ffmpeg` to enable fallback decoding."
        )
    try:
        reader = imageio.get_reader(path)
    except Exception as e:
        raise FileNotFoundError(f"Could not open video with ImageIO: {path}\n{e}")

    # Try to get total frames; if not available, iterate to count
    try:
        meta = reader.get_meta_data()
        total = int(meta.get('nframes', 0))
    except Exception:
        total = 0

    frames_np = []
    if total and total > 0 and total != float('inf'):
        idxs = np.linspace(0, total - 1, num_frames).astype(int)
        for i in idxs:
            try:
                fr = reader.get_data(int(i))
                frames_np.append(fr)
            except Exception:
                continue
    else:
        # Sequential read
        try:
            for fr in reader:
                frames_np.append(fr)
        except Exception:
            pass
        if len(frames_np) == 0:
            reader.close()
            raise RuntimeError("Video appears to have 0 frames (ImageIO path).")
        idxs = np.linspace(0, len(frames_np) - 1, num_frames).astype(int)
        frames_np = [frames_np[i] for i in idxs]

    reader.close()

    # Convert to PIL RGB and resize square
    pil_frames = []
    for fr in frames_np:
        # Ensure uint8 RGB
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
        pil_frames.append(Image.fromarray(fr))
    return pil_frames


def sample_video_frames(path: str, num_frames: int = 8, target_size: int = 224):
    """Wrapper: try OpenCV first, then ImageIO as a fallback."""
    try:
        return sample_video_frames_cv2(path, num_frames=num_frames, target_size=target_size)
    except Exception as e_cv:
        print(f"[decoder] OpenCV failed ({e_cv}). Trying ImageIO fallback…")
        try:
            return sample_video_frames_imageio(path, num_frames=num_frames, target_size=target_size)
        except Exception as e_io:
            raise RuntimeError(
                "Both OpenCV and ImageIO decoders failed. "
                "Please ensure the video exists and install codecs or run: pip install imageio imageio-ffmpeg\n"
                f"OpenCV error: {e_cv}\nImageIO error: {e_io}"
            )


def minmax_norm(arr: np.ndarray, eps: float = 1e-8):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + eps)


def pick_layer_indices(num_layers: int, k: int):
    """Pick k roughly evenly spaced layer indices from range(num_layers)."""
    if k >= num_layers:
        return list(range(num_layers))
    ticks = np.linspace(0, num_layers - 1, k)
    return sorted(set(int(round(t)) for t in ticks))


def pick_frame_indices(num_frames: int, k: int):
    if k >= num_frames:
        return list(range(num_frames))
    ticks = np.linspace(0, num_frames - 1, k)
    return sorted(set(int(round(t)) for t in ticks))


def extract_cls_to_patch_attn(attentions, num_frames: int, h_p: int, w_p: int):
    """Make this robust to TimeSformer variants.

    HF TimeSformer can emit attentions with different shapes per layer:
      A) Joint space-time:      [B, H, (1+T*P), (1+T*P)]
      B) Divided (spatial step): [B*T, H, (1+P), (1+P)]
      C) Some builds yield       [B*T, H, (1+2*P), (1+2*P)]
         where tokens are concatenated (e.g., space + something else).

    We collapse heads, take CLS->token weights, then reshape to per-frame
    patch grids [T, H_p, W_p]. If there are extra groups (factor > 1), we
    average them.
    """
    P = h_p * w_p
    maps = []
    for L, attn in enumerate(attentions):
        # Some versions return a tuple per block; prefer the first map if so
        if isinstance(attn, (list, tuple)) and len(attn) > 0:
            attn = attn[0]

        if attn.ndim != 4:
            raise ValueError(f"Unexpected attention shape at layer {L}: {tuple(attn.shape)}")
        attn = attn.detach().to(torch.float32).cpu()
        B_eff, H, N, N2 = attn.shape
        if N != N2:
            raise ValueError(f"Non-square attention at layer {L}: {N}x{N2}")

        # Avg heads -> [B_eff, N, N]
        attn_mean = attn.mean(dim=1)

        # CLS->tokens (drop CLS from keys)
        cls_to_tokens = attn_mean[:, 0, 1:]  # [B_eff, N-1]
        if (N - 1) % P != 0:
            raise ValueError(
                f"At layer {L}, tokens after CLS = {N-1} not divisible by patches={P}. "
                f"h_p={h_p}, w_p={w_p}, N={N}."
            )
        factor = (N - 1) // P  # how many P-sized groups are present

        # Reshape into groups of P tokens
        cls_groups = cls_to_tokens.view(B_eff, factor, P)  # [B_eff, factor, P]

        # Case: joint attention where factor == T and B_eff == 1 -> each group is a frame
        if factor == num_frames and B_eff == 1:
            per_frame = cls_groups[0]  # [T, P]
        else:
            # Collapse any extra groups (e.g., factor=1 or 2) by average
            per_seq = cls_groups.mean(dim=1)  # [B_eff, P]

            # Now map B_eff to (B, T) if possible; otherwise assume B_eff == T
            if B_eff % num_frames == 0:
                B = B_eff // num_frames
                per_seq = per_seq.view(B, num_frames, P)
                per_frame = per_seq[0]  # take batch 0 -> [T, P]
            elif B_eff == num_frames:
                per_frame = per_seq  # [T, P]
            else:
                # Fallback: actual frames available might differ from requested
                actual_frames = min(B_eff, num_frames)
                per_frame = per_seq[:actual_frames]  # [actual_T, P]
                # Pad or interpolate if needed
                if actual_frames < num_frames:
                    # Repeat last frame to fill
                    padding = per_seq[-1:].repeat(num_frames - actual_frames, 1)
                    per_frame = torch.cat([per_frame, padding], dim=0)

        # Determine actual frame count from per_frame shape
        actual_T = per_frame.shape[0]
        if actual_T * P != per_frame.numel():
            raise ValueError(
                f"At layer {L}, per_frame has {per_frame.numel()} elements but "
                f"actual_T={actual_T}, P={P} requires {actual_T * P}"
            )
        
        # To [actual_T, H_p, W_p]
        grid = per_frame.view(actual_T, h_p, w_p).numpy()
        for t in range(actual_T):
            grid[t] = minmax_norm(grid[t])
        maps.append(grid)
    return maps  # list of [T, H_p, W_p]


def overlay_grid(frames_pil, attn_maps, layer_idxs, frame_idxs, title_prefix: str = "", cmap='jet', custom_colors=None):
    """Create a matplotlib grid overlaying attention heatmaps.
    frames_pil: list of PIL RGB frames (all same size)
    attn_maps: list over layers of [T, H_p, W_p]
    layer_idxs: layer indices to plot
    frame_idxs: frame indices to plot
    cmap: colormap name or object to use
    custom_colors: list of hex colors to create custom colormap (overrides cmap if provided)
    """
    assert len(frames_pil) > 0
    W, H = frames_pil[0].size

    # Create custom colormap if colors are provided
    if custom_colors:
        cmap = LinearSegmentedColormap.from_list('custom_attn', custom_colors)

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
        grid = attn_maps[L]  # [T, H_p, W_p]
        for c, t in enumerate(frame_idxs):
            ax = axes[r, c]
            ax.imshow(frames_pil[t])
            # Resize heatmap to frame size
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
    parser = argparse.ArgumentParser(description="Visualize TimeSformer attentions across layers and frames.")
    parser.add_argument('--video', type=str, required=True, help='Path to a video file.')
    parser.add_argument('--num-frames', type=int, default=8, help='Number of frames to sample uniformly.')
    parser.add_argument('--grid-frames', type=int, default=4, help='How many frames to show per row in the grid.')
    parser.add_argument('--layers', type=int, default=5, help='How many layers to visualize (evenly spaced).')
    parser.add_argument('--model', type=str, default='facebook/timesformer-base-finetuned-k400', help='HF model id or local path.')
    parser.add_argument('--out', type=str, default=None, help='Output PNG path. Defaults to ./attn_<video_basename>.png')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run on.')
    parser.add_argument('--cmap', type=str, default='jet', help='Colormap to use (e.g., jet, viridis, plasma). Use "custom" to enable custom colors.')
    parser.add_argument('--custom-colors', type=str, nargs='+', default=None, help='Custom hex colors for colormap (e.g., 704693 985ca5 8a79b5). Overrides --cmap if provided.')
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video path not found: {video_path}")

    print(f"\n[1/5] Sampling frames from: {video_path}")
    frames_pil = sample_video_frames(str(video_path), num_frames=args.num_frames, target_size=224)
    T = len(frames_pil)
    print(f"Sampled {T} frames.")

    print("[2/5] Loading processor and model…")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = TimesformerForVideoClassification.from_pretrained(args.model)
    model.to(args.device)
    model.eval()

    # Determine patch grid from config
    image_size = getattr(model.config, 'image_size', 224)
    patch_size = getattr(model.config, 'patch_size', 16)
    h_p = image_size // patch_size
    w_p = image_size // patch_size
    
    # Get the expected number of frames from model config
    expected_num_frames = getattr(model.config, 'num_frames', 8)
    print(f"Model expects {expected_num_frames} frames. Adjusting if needed...")
    
    # Adjust frames to match model expectations
    if T != expected_num_frames:
        print(f"Resampling from {T} to {expected_num_frames} frames to match model config.")
        if T > expected_num_frames:
            # Subsample
            idxs = np.linspace(0, T - 1, expected_num_frames).astype(int)
            frames_pil = [frames_pil[i] for i in idxs]
        else:
            # Repeat last frame to fill
            frames_pil = frames_pil + [frames_pil[-1]] * (expected_num_frames - T)
        T = expected_num_frames
        print(f"Adjusted to {T} frames.")

    print("[3/5] Preparing inputs…")
    inputs = processor(frames_pil, return_tensors='pt')
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    print("[4/5] Running model with attentions…")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        logits = outputs.logits
        attentions = outputs.attentions

    if attentions is None:
        raise RuntimeError(
            "Model did not return attentions. Please ensure your `transformers` version supports\n"
            "TimeSformer attentions and try passing `output_attentions=True`."
        )

    num_layers = len(attentions)
    print(f"Collected attentions from {num_layers} layers.")

    # Extract CLS->patch maps per layer
    maps = extract_cls_to_patch_attn(attentions, num_frames=T, h_p=h_p, w_p=w_p)

    # Choose which layers and frames to display
    layer_idxs = pick_layer_indices(num_layers, args.layers)
    frame_idxs = pick_frame_indices(T, args.grid_frames)

    # Prediction label (optional)
    pred_idx = int(logits.argmax(-1).item())
    label = model.config.id2label.get(pred_idx, str(pred_idx))
    title = f"{video_path.name}  |  pred: {label}"

    # Prepare colormap settings
    custom_colors = None
    if args.custom_colors:
        # Add # prefix if not present
        custom_colors = ['#' + c if not c.startswith('#') else c for c in args.custom_colors]
        print(f"Using custom colors: {custom_colors}")
    elif args.cmap == 'custom':
        # Default custom colors (purple scheme)
        custom_colors = ['#704693', '#985ca5', '#8a79b5', '#b19dcb', '#c7bed8', '#afb5d5']
        print(f"Using default custom colors: {custom_colors}")
    else:
        print(f"Using colormap: {args.cmap}")

    print("[5/5] Building figure…")
    fig = overlay_grid(frames_pil, maps, layer_idxs, frame_idxs, title_prefix=title, 
                       cmap=args.cmap, custom_colors=custom_colors)

    out_dir = Path('.')
    out_path = Path(args.out) if args.out else out_dir / f"attn_{video_path.stem}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print(f"\nDone. Saved attention grid to: {out_path.resolve()}")


if __name__ == '__main__':
    main()
