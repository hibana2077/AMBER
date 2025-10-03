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

import numpy as np
import torch

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

# Import utilities from utils module
from .utils.video_utils import sample_video_frames
from .utils.attention_utils import (
    pick_layer_indices,
    pick_frame_indices,
    extract_cls_to_patch_attn_timesformer,
    extract_cls_to_patch_attn_videomae_vivit,
)
from .utils.visualization_utils import overlay_grid


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
                        processor = VJEPA2VideoProcessor.from_pretrained(model_id)
                        model = VJEPA2ForVideoClassification.from_pretrained(model_id, attn_implementation="eager")
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
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"Saved attention grid to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
