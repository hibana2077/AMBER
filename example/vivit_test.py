#!/usr/bin/env python
"""
ViViT Attention Map Visualizer

Similar to VideoMAE visualizer but for ViViT (Video Vision Transformer) models.
ViViT uses output_hidden_states and output_attentions parameters.
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

try:
    import av
except ImportError as e:
    raise SystemExit("PyAV is required. Install with `pip install av`.\n" + str(e))

from transformers import VivitImageProcessor, VivitForVideoClassification

# Optional: ImageIO fallback for broader container/codec support
try:
    import imageio.v2 as imageio  # v2 API compatibility
    _HAS_IMAGEIO = True
except Exception:
    imageio = None
    _HAS_IMAGEIO = False


def read_video_pyav(container, indices):
    """Decode the video with PyAV decoder.
    Args:
        container: PyAV container.
        indices: 1D np.ndarray of frame indices to decode (can contain duplicates).
    Returns:
        result: np array of decoded frames of shape (num_frames, height, width, 3).
    Notes:
        - This implementation preserves duplicates in `indices` by repeating frames as needed.
    """
    if not isinstance(indices, np.ndarray):
        indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        raise RuntimeError("No indices provided to read_video_pyav().")

    start_index = int(indices.min())
    end_index = int(indices.max())

    # Decode range once; store in a dict
    container.seek(0)
    frame_map = {}
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i < start_index:
            continue
        frame_map[i] = frame.to_ndarray(format="rgb24")

    # Build output in the exact order of `indices` (preserve duplicates)
    out_list = []
    for idx in indices.tolist():
        if idx not in frame_map:
            # If a requested index is missing (decoder skipped), fallback to nearest available
            # Find nearest key
            if frame_map:
                nearest = min(frame_map.keys(), key=lambda k: abs(k - idx))
                out_list.append(frame_map[nearest])
            else:
                raise RuntimeError("PyAV decoder produced no frames in requested range.")
        else:
            out_list.append(frame_map[idx])

    if len(out_list) == 0:
        raise RuntimeError("PyAV decoder produced 0 frames for requested indices.")

    return np.stack(out_list)


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    """Sample a given number of frame indices from the video.
    Args:
        clip_len: Total number of frames to sample (output length).
        frame_sample_rate: Sample every n-th frame (not used in current implementation).
        seg_len: Total number of frames available in the stream.
    Returns:
        indices: np.ndarray of sampled frame indices of length `clip_len`.
    """
    if seg_len is None or seg_len <= 0:
        # Degenerate: just return zeros (will be padded later upstream)
        return np.zeros(clip_len, dtype=np.int64)

    if seg_len == 1:
        return np.zeros(clip_len, dtype=np.int64)
    
    # Simple uniform sampling from 0 to seg_len-1
    indices = np.linspace(0, seg_len - 1, num=clip_len)
    return indices.astype(np.int64)


def sample_video_frames_cv2(path: str, num_frames: int = 32, target_size: int = 224):
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


def sample_video_frames_pyav(path: str, num_frames: int = 32, target_size: int = 224, frame_sample_rate: int = 4):
    """Sample frames using PyAV decoder (like the ViViT example).
    Returns: list of PIL.Image in RGB of size (target_size, target_size).
    """
    try:
        container = av.open(path)
    except Exception as e:
        raise FileNotFoundError(f"Could not open video with PyAV: {path}\n{e}")
    
    try:
        seg_len = container.streams.video[0].frames
        if seg_len <= 0:
            # Count frames manually
            container.seek(0)
            seg_len = sum(1 for _ in container.decode(video=0))
            container.seek(0)
    except Exception:
        # Fallback: count frames
        container.seek(0)
        seg_len = sum(1 for _ in container.decode(video=0))
        container.seek(0)
    
    indices = sample_frame_indices(clip_len=num_frames, frame_sample_rate=frame_sample_rate, seg_len=seg_len)
    video = read_video_pyav(container=container, indices=indices)
    container.close()
    
    # Convert to PIL RGB and resize square
    pil_frames = []
    for fr in video:
        # Already in RGB format from PyAV
        h, w, _ = fr.shape
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        fr = fr[y0:y0 + side, x0:x0 + side]
        # resize
        fr = cv2.resize(fr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        pil_frames.append(Image.fromarray(fr))
    return pil_frames


def sample_video_frames_imageio(path: str, num_frames: int = 32, target_size: int = 224):
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


def sample_video_frames(path: str, num_frames: int = 32, target_size: int = 224, use_pyav: bool = True, frame_sample_rate: int = 4):
    """Wrapper: try PyAV first (ViViT style), then OpenCV, then ImageIO as fallback.
    Ensures the returned list has exactly `num_frames` elements by padding the last frame if needed.
    """
    def _pad_to_length(fr_list, target):
        if len(fr_list) == 0:
            raise RuntimeError("No frames decoded from any backend.")
        if len(fr_list) < target:
            last = fr_list[-1]
            # Use .copy() to avoid referencing the same image object
            pad = [last.copy() if hasattr(last, "copy") else last for _ in range(target - len(fr_list))]
            return fr_list + pad
        if len(fr_list) > target:
            return fr_list[:target]
        return fr_list

    e_pyav = None
    frames = None
    if use_pyav:
        try:
            frames = sample_video_frames_pyav(path, num_frames=num_frames, target_size=target_size, frame_sample_rate=frame_sample_rate)
        except Exception as e:
            e_pyav = e
            print(f"[decoder] PyAV failed ({e_pyav}). Trying OpenCV…")
    
    if frames is None:
        try:
            frames = sample_video_frames_cv2(path, num_frames=num_frames, target_size=target_size)
        except Exception as e_cv:
            print(f"[decoder] OpenCV failed ({e_cv}). Trying ImageIO fallback…")
            try:
                frames = sample_video_frames_imageio(path, num_frames=num_frames, target_size=target_size)
            except Exception as e_io:
                raise RuntimeError(
                    "All decoders (PyAV, OpenCV, ImageIO) failed. "
                    "Please ensure the video exists and install codecs or run: pip install av imageio imageio-ffmpeg\n"
                    f"PyAV error: {e_pyav if e_pyav else 'not tried'}\nOpenCV error: {e_cv}\nImageIO error: {e_io}"
                )

    # Guarantee exact length by padding/truncating
    return _pad_to_length(frames, num_frames)


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


def extract_cls_to_patch_attn(attentions, num_frames: int, h_p: int, w_p: int, tubelet_size: int = 2):
    """Extract CLS to patch attention maps from ViViT attentions.
    
    ViViT attentions shape: [batch_size, num_heads, seq_len, seq_len]
    
    Note: ViViT for video classification may not use a traditional CLS token.
    We'll use the mean attention across all tokens as a proxy for global attention.
    """
    P = h_p * w_p
    T_patches = num_frames // tubelet_size  # Number of temporal patches after tubelet downsampling
    maps = []
    
    for L, attn in enumerate(attentions):
        # Some versions return a tuple per block; prefer the first map if so
        if isinstance(attn, (list, tuple)) and len(attn) > 0:
            attn = attn[0]

        if attn.ndim != 4:
            raise ValueError(f"Unexpected attention shape at layer {L}: {tuple(attn.shape)}")
        
        attn = attn.detach().to(torch.float32).cpu()
        B, H, N, N2 = attn.shape
        if N != N2:
            raise ValueError(f"Non-square attention at layer {L}: {N}x{N2}")

        # Avg heads -> [B, N, N]
        attn_mean = attn.mean(dim=1)

        # Check if N matches expected patch tokens (with or without CLS)
        expected_with_cls = T_patches * P + 1
        expected_without_cls = T_patches * P
        
        if N == expected_with_cls:
            # Has CLS token at position 0
            cls_to_tokens = attn_mean[:, 0, 1:]  # [B, N-1]
            num_patch_tokens = N - 1
            has_cls = True
        elif N == expected_without_cls:
            # No CLS token - use mean attention from all tokens
            # Average attention from all source tokens to all target tokens
            cls_to_tokens = attn_mean[:, :, :].mean(dim=1)  # [B, N]
            num_patch_tokens = N
            has_cls = False
            if L == 0:
                print(f"Info: No CLS token detected. Using mean attention across all tokens.")
        else:
            # Unexpected size - try to infer
            if N % P == 0:
                # Likely no CLS
                cls_to_tokens = attn_mean[:, :, :].mean(dim=1)  # [B, N]
                num_patch_tokens = N
                has_cls = False
                if L == 0:
                    print(f"Info: Sequence length {N} suggests no CLS token. Using mean attention.")
            elif (N - 1) % P == 0:
                # Likely has CLS
                cls_to_tokens = attn_mean[:, 0, 1:]  # [B, N-1]
                num_patch_tokens = N - 1
                has_cls = True
                if L == 0:
                    print(f"Info: Sequence length {N} suggests CLS at position 0.")
            else:
                raise ValueError(
                    f"At layer {L}, sequence length {N} doesn't match expected patterns. "
                    f"Expected {expected_with_cls} (with CLS) or {expected_without_cls} (without CLS). "
                    f"P={P}, T_patches={T_patches}"
                )
        
        # Verify divisibility
        if num_patch_tokens % P != 0:
            raise ValueError(
                f"At layer {L}, patch tokens = {num_patch_tokens} not divisible by spatial patches={P}. "
                f"h_p={h_p}, w_p={w_p}, tubelet_size={tubelet_size}"
            )
        
        actual_t_patches = num_patch_tokens // P
        
        # Reshape to [B, T_patches, P]
        per_patch = cls_to_tokens[0].view(actual_t_patches, P)  # [T_patches, P]
        
        # To [T_patches, H_p, W_p]
        grid = per_patch.view(actual_t_patches, h_p, w_p).numpy()
        
        # Upsample temporally to match original frame count for visualization
        # Simple nearest-neighbor interpolation along temporal dimension
        upsampled_grid = np.zeros((num_frames, h_p, w_p), dtype=grid.dtype)
        for t in range(num_frames):
            t_patch = min(t // tubelet_size, actual_t_patches - 1)
            upsampled_grid[t] = grid[t_patch]
        
        # Normalize each frame
        for t in range(num_frames):
            upsampled_grid[t] = minmax_norm(upsampled_grid[t])
        
        maps.append(upsampled_grid)
    
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
    parser = argparse.ArgumentParser(description="Visualize ViViT attentions across layers and frames.")
    parser.add_argument('--video', type=str, required=True, help='Path to a video file.')
    parser.add_argument('--num-frames', type=int, default=32, help='Number of frames to sample.')
    parser.add_argument('--frame-sample-rate', type=int, default=4, help='Frame sample rate for PyAV decoder.')
    parser.add_argument('--grid-frames', type=int, default=4, help='How many frames to show per row in the grid.')
    parser.add_argument('--layers', type=int, default=5, help='How many layers to visualize (evenly spaced).')
    parser.add_argument('--model', type=str, default='google/vivit-b-16x2-kinetics400', help='HF model id or local path.')
    parser.add_argument('--out', type=str, default=None, help='Output PNG path. Defaults to ./attn_vivit_<video_basename>.png')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run on.')
    parser.add_argument('--use-pyav', action='store_true', default=True, help='Use PyAV decoder (ViViT style sampling).')
    parser.add_argument('--cmap', type=str, default='jet', help='Colormap to use (e.g., jet, viridis, plasma). Use "custom" to enable custom colors.')
    parser.add_argument('--custom-colors', type=str, nargs='+', default=None, help='Custom hex colors for colormap (e.g., 704693 985ca5 8a79b5). Overrides --cmap if provided.')
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video path not found: {video_path}")

    print(f"\n[1/5] Sampling frames from: {video_path}")
    frames_pil = sample_video_frames(
        str(video_path), 
        num_frames=args.num_frames, 
        target_size=224,
        use_pyav=args.use_pyav,
        frame_sample_rate=args.frame_sample_rate
    )
    T = len(frames_pil)
    print(f"Sampled {T} frames.")

    print("[2/5] Loading processor and model…")
    processor = VivitImageProcessor.from_pretrained(args.model)
    model = VivitForVideoClassification.from_pretrained(
        args.model,
        attn_implementation="eager"
    )
    model.config.output_attentions = True
    model.to(args.device)
    model.eval()
    
    print(f"Model config output_attentions: {model.config.output_attentions}")
    print(f"Model class: {model.__class__.__name__}")
    print(f"Transformers version: {__import__('transformers').__version__}")

    # Determine patch grid from config
    image_size = getattr(model.config, 'image_size', 224)
    patch_size = getattr(model.config, 'patch_size', 16)
    tubelet_size_raw = getattr(model.config, 'tubelet_size', 2)
    # tubelet_size may be an int or a list [temporal, spatial_h, spatial_w]
    if isinstance(tubelet_size_raw, (list, tuple)):
        tubelet_size = tubelet_size_raw[0]  # Extract temporal dimension
    else:
        tubelet_size = tubelet_size_raw
    h_p = image_size // patch_size
    w_p = image_size // patch_size
    
    # Calculate expected number of frames based on position embeddings
    # Position embeddings shape is typically: [1, 1 + (T/tubelet_size) * (H/ps)*(W/ps), hidden]
    # Some checkpoints may store embeddings without CLS included. Infer this robustly.
    # Note: Attention matrices ALWAYS include CLS as first token, even if pos_embed doesn't
    if hasattr(model.vivit.embeddings, 'position_embeddings'):
        pos_embed_shape = model.vivit.embeddings.position_embeddings.shape
        total_pos_tokens = int(pos_embed_shape[1])  # may or may not include CLS
        P = h_p * w_p

        if total_pos_tokens % P == 0:
            # Likely no CLS included in position embeddings
            has_cls_in_pos = False
            patch_tokens = total_pos_tokens
        else:
            # Likely includes CLS token
            has_cls_in_pos = True
            patch_tokens = total_pos_tokens - 1

        t_patches_float = patch_tokens / float(P)
        expected_temporal_patches = int(round(t_patches_float))
        expected_frames = expected_temporal_patches * tubelet_size

        print(
            f"Model expects ~{t_patches_float:.3f} temporal patches -> {expected_temporal_patches} "
            f"(tubelet_size={tubelet_size}) => frames={expected_frames}"
        )
        print(
            f"Position embedding length: {total_pos_tokens} (has_cls={has_cls_in_pos}), "
            f"patch tokens: {patch_tokens}, spatial patches/frame: {P}"
        )

        # Always resample to match the model's expectations
        if T != expected_frames and expected_temporal_patches > 0:
            print(f"Adjusting num_frames from {T} to {expected_frames} to match model")
            frames_pil = sample_video_frames(
                str(video_path), 
                num_frames=expected_frames, 
                target_size=image_size,
                use_pyav=args.use_pyav,
                frame_sample_rate=args.frame_sample_rate
            )
            T = len(frames_pil)
            print(f"Resampled {T} frames.")
            args.num_frames = T
    
    print(f"Image size: {image_size}, Patch size: {patch_size}, Tubelet size: {tubelet_size}, Grid: {h_p}x{w_p}")

    print("[3/5] Preparing inputs…")
    # ViViT expects list of numpy arrays
    video_np = [np.array(frame) for frame in frames_pil]
    inputs = processor(video_np, return_tensors='pt')
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    print("[4/5] Running model with attentions…")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        attentions = outputs.attentions
    
    if attentions is None or len(attentions) == 0:
        raise RuntimeError(
            "Model did not return attentions. This could be due to:\n"
            "1. The model was not loaded with attn_implementation='eager'\n"
            "2. output_attentions was not set to True\n"
            f"3. Transformers version: {__import__('transformers').__version__}\n\n"
            "Possible solutions:\n"
            "- Ensure attn_implementation='eager' is set when loading the model\n"
            "- Ensure model.config.output_attentions = True is set\n"
            "- Update transformers: pip install --upgrade transformers"
        )
    
    print(f"Collected attentions: {len(attentions)} layers")
    print(f"First attention shape: {attentions[0].shape}")

    num_layers = len(attentions)
    print(f"Collected attentions from {num_layers} layers.")

    # Extract CLS->patch maps per layer
    maps = extract_cls_to_patch_attn(attentions, num_frames=T, h_p=h_p, w_p=w_p, tubelet_size=tubelet_size)

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
    out_path = Path(args.out) if args.out else out_dir / f"attn_vivit_{video_path.stem}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print(f"\nDone. Saved attention grid to: {out_path.resolve()}")


if __name__ == '__main__':
    main()
