#!/usr/bin/env python
"""
AMBER video classification runner

CLI for fine-tuning and evaluating a VideoMAE model on the UCF101 subset
dataset as described in docs/video_cls.md. Uses PyTorchVideo for datasets and
Hugging Face Transformers Trainer for training/eval.

Example usage (PowerShell):
  # Train
  python -m src.run train --data-root UCF101_subset --output-dir runs/videomae-ucf --epochs 4 --batch-size 4

  # Eval (validation)
  python -m src.run eval --data-root UCF101_subset --model runs/videomae-ucf --split val --batch-size 4

Notes:
- This focuses on VideoMAE per the docs provided. You can switch to a different
  VideoMAE checkpoint via --model-id.
- The dataset layout is assumed to follow the UCF101 subset structure from docs/video_cls.md.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import torchvision
    from torchvision.transforms import RandomCrop as TVRandomCrop, RandomHorizontalFlip as TVRandomHorizontalFlip, Resize as TVResize
    import pytorchvideo.data  # We still rely on the dataset class only.
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "This script requires pytorchvideo (data only) and torchvision. Install requirements (see requirements.txt) and try again.\n"
        + str(e)
    )

from src.utils.video_transforms import (
    UniformTemporalSubsampleTransform,
    RandomShortSideScaleTransform,
    ShortSideResizeTransform,
    RandomCropVideoTransform,
    CenterCropVideoTransform,
    RandomHorizontalFlipVideoTransform,
    NormalizeVideoTransform,
    ComposeVideo,
    apply_to_key,
)

from transformers import (
    AutoModelForVideoClassification,
    AutoImageProcessor,
    AutoVideoProcessor,
    TrainingArguments,
    Trainer,
)

try:
    import evaluate  # type: ignore
except Exception:
    evaluate = None

# -----------------------------
# Data / transforms
# -----------------------------


def build_label_maps(data_root: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Scan class folders to build label mappings.

    Expects structure:
      data_root/{train,val,test}/<CLASS_NAME>/*.avi|*.mp4
    """
    import pathlib

    root = pathlib.Path(data_root)
    class_dirs = set()
    for split in ("train", "val", "test"):
        split_dir = root / split
        print(f"Scanning {split_dir}...")
        if split_dir.exists():
            for p in split_dir.glob("*"):
                if p.is_dir():
                    class_dirs.add(p.name)
    class_labels = sorted(class_dirs)
    if not class_labels:
        raise FileNotFoundError(
            f"No class folders found under {data_root}. Expected UCF101-like structure with train/val/test."
        )
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def _get_num_frames_from_config(cfg: Any, default: int = 16) -> int:
    # Support various configs: VideoMAE/ViViT/TimeSformer use `num_frames`; V-JEPA2 uses `frames_per_clip`.
    for key in ("num_frames", "frames_per_clip"):
        if hasattr(cfg, key) and getattr(cfg, key):
            return int(getattr(cfg, key))
    return default


def build_transforms(processor: Any, model: Any):
    # Derive normalization + size from processor (if available)
    mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
    std = getattr(processor, "image_std", [0.229, 0.224, 0.225])

    size = getattr(processor, "size", {}) or {}
    if isinstance(size, int):
        height = width = size
    elif "shortest_edge" in size:  # some processors expose shortest_edge
        height = width = size["shortest_edge"]
    else:
        height = size.get("height", 224)
        width = size.get("width", 224)
    resize_to = (height, width)

    num_frames_to_sample = _get_num_frames_from_config(model.config, 16)
    sample_rate = 4
    fps = 30
    clip_duration = num_frames_to_sample * sample_rate / fps

    # Compose sequence for video tensor (C,T,H,W)
    train_video_pipeline = ComposeVideo(
        [
            UniformTemporalSubsampleTransform(num_frames_to_sample),
            lambda v: v / 255.0,
            RandomShortSideScaleTransform(256, 320),
            RandomCropVideoTransform(resize_to),
            RandomHorizontalFlipVideoTransform(0.5),
            NormalizeVideoTransform(mean, std),
        ]
    )

    eval_video_pipeline = ComposeVideo(
        [
            UniformTemporalSubsampleTransform(num_frames_to_sample),
            lambda v: v / 255.0,
            ShortSideResizeTransform(min(resize_to)),  # ensure shortest side matches target before center crop
            CenterCropVideoTransform(resize_to),
            NormalizeVideoTransform(mean, std),
        ]
    )

    # Wrap to act on sample dict under key 'video'
    train_transform = apply_to_key("video", train_video_pipeline)
    eval_transform = apply_to_key("video", eval_video_pipeline)

    return train_transform, eval_transform, clip_duration


def build_datasets(
    data_root: str,
    processor: Any,
    model: Any,
):
    train_t, eval_t, clip_duration = build_transforms(processor, model)

    def make(split: str, transform):
        split_path = os.path.join(data_root, split)
        if not os.path.isdir(split_path):
            return None
        ds = pytorchvideo.data.Ucf101(
            data_path=split_path,
            clip_sampler=pytorchvideo.data.make_clip_sampler(
                "random" if split == "train" else "uniform", clip_duration
            ),
            decode_audio=False,
            transform=None,  # we'll wrap via custom __getitem__ below if needed
        )
        # If transform is provided, monkey-patch __getitem__ to apply it
        if transform is not None:
            base_get = ds.__getitem__

            def _get(i):
                sample = base_get(i)
                # pytorchvideo's Ucf101 returns video shaped (C,T,H,W). We keep that ordering
                # throughout the transform pipeline (all custom transforms expect (C,T,H,W)).
                # Older comment assumed (T,C,H,W) which caused an unnecessary permute leading to
                # temporal dimension misalignment and variable frame counts in a batch.
                if isinstance(sample, dict) and 'video' in sample:
                    v = sample['video']
                    if torch.is_tensor(v) and v.dim() == 4:
                        # If a future version returned (T,C,H,W), detect and correct it once.
                        if v.shape[0] not in (1,3) and v.shape[1] in (1,3):  # (T,C,H,W) heuristic
                            v = v.permute(1,0,2,3)  # -> (C,T,H,W)
                        sample['video'] = v
                    sample = transform(sample)
                return sample

            ds.__getitem__ = _get  # type: ignore
        return ds

    return make("train", train_t), make("val", eval_t), make("test", eval_t)


def collate_fn(examples):
    """Collate a list of samples into a batch.

    Accepts samples whose video tensors are either (C,T,H,W) (preferred) or (T,C,H,W).
    Produces pixel_values shaped (B, T, C, H, W) as expected by HF video models.
    Ensures a consistent temporal dimension by relying on prior UniformTemporalSubsampleTransform.
    Raises a clear error if temporal dims still mismatch to aid debugging.
    """
    vids = []
    for ex in examples:
        v = ex["video"]
        if v.dim() != 4:
            raise ValueError(f"Unexpected video tensor rank: {v.shape}")
        # Detect ordering.
        if v.shape[0] in (1,3) and v.shape[1] not in (1,3):  # (C,T,H,W)
            c_thw = v
        elif v.shape[1] in (1,3) and v.shape[0] not in (1,3):  # (T,C,H,W)
            c_thw = v.permute(1,0,2,3)
        else:
            # Ambiguous; default assume first dim is channels.
            c_thw = v
        vids.append(c_thw)

    # Verify uniform temporal length; if inconsistent, perform a robust fallback
    # uniform temporal resampling to the MIN length (to avoid padding / extra memory).
    t_lens_list = [v.shape[1] for v in vids]
    unique_t = sorted(set(t_lens_list))
    if len(unique_t) != 1:
        # Fallback path: uniformly sample each video to target_len
        target_len = min(unique_t)
        # We log only once per batch (stderr) to avoid noisy output; import locally.
        import sys
        print(
            f"[collate_fn warning] Inconsistent frame counts {unique_t}; resampling all to {target_len} frames.",
            file=sys.stderr,
        )
        new_vids = []
        for v in vids:
            T = v.shape[1]
            if T == target_len:
                new_vids.append(v)
            else:
                # Uniform index selection over existing frames
                if target_len == 1:
                    idx = torch.tensor([T // 2])
                else:
                    idx = torch.linspace(0, T - 1, target_len).long()
                new_vids.append(v[:, idx])
        vids = new_vids

    batch = torch.stack(vids)  # (B,C,T,H,W)

    # ------------------------------------------------------------------
    # Spatial size enforcement fallback
    # Upstream transforms SHOULD guarantee (target_size, target_size) frames
    # (e.g. 224x224 for VideoMAE). If a mismatch slips through (as observed
    # in user error: 240x320), we correct it here to avoid trainer crash.
    # This is defensive and logs only when adjustment is actually needed.
    # ------------------------------------------------------------------
    import os, sys, math
    target_size_env = os.environ.get("AMBER_FORCE_IMAGE_SIZE")
    try:
        target_size = int(target_size_env) if target_size_env else 224
    except ValueError:
        target_size = 224

    B, C, T, H, W = batch.shape

    # ------------------------------------------------------------------
    # Temporal length enforcement (addresses mismatch e.g. 26 vs expected 8)
    # We let model.config.num_frames (propagated via AMBER_FORCE_NUM_FRAMES) drive
    # the target. If provided and different from incoming T, uniformly sample
    # or duplicate frames to reach target length.
    # ------------------------------------------------------------------
    target_frames_env = os.environ.get("AMBER_FORCE_NUM_FRAMES")
    if target_frames_env:
        try:
            target_T = int(target_frames_env)
        except ValueError:
            target_T = T
        if target_T > 0 and T != target_T:
            import sys
            # Build uniform indices (works for both shortening and lengthening)
            if target_T == 1:
                idx = torch.tensor([T // 2])
            else:
                idx = torch.linspace(0, T - 1, target_T).round().long().clamp_(0, T - 1)
            batch = batch[:, :, idx]
            _, _, T, _, _ = batch.shape
            print(
                f"[collate_fn info] Adjusted temporal length from {T} to {target_T} frames (env target={target_frames_env}).",
                file=sys.stderr,
            )
    if (H, W) != (target_size, target_size):
        # Center crop to square first (min side) then resize if needed.
        min_side = min(H, W)
        if H != W:
            top = (H - min_side) // 2
            left = (W - min_side) // 2
            batch = batch[:, :, :, top: top + min_side, left: left + min_side]
            _, _, _, Hc, Wc = batch.shape
            H, W = Hc, Wc
        if H != target_size:  # need resize (either down or up)
            # Merge (B,T) for single interpolate call for efficiency
            bt = B * T
            merged = batch.permute(0,2,1,3,4).reshape(bt, C, H, W)  # (B*T,C,H,W)
            merged = torch.nn.functional.interpolate(
                merged, size=(target_size, target_size), mode="bilinear", align_corners=False
            )
            batch = merged.reshape(B, T, C, target_size, target_size).permute(0,2,1,3,4)
        # Log once
        # print(
        #     f"[collate_fn info] Adjusted frame size from {(H,W)} to {(target_size,target_size)} (env target={target_size_env}).",
        #     file=sys.stderr,
        # )

    pixel_values = batch.permute(0,2,1,3,4)  # -> (B,T,C,H,W)
    labels = torch.tensor([int(ex["label"]) for ex in examples])
    return {"pixel_values": pixel_values, "labels": labels}


# -----------------------------
# Metrics
# -----------------------------


def build_metrics():
    """Return a metrics function for Hugging Face Trainer.

    Adds accuracy, macro precision, macro recall, macro F1.
    Falls back to a pure NumPy implementation if `evaluate` is not available.
    """

    if evaluate is None:
        # Fallback implementation (no external dependencies beyond numpy)
        def _compute_metrics(eval_pred):
            logits = eval_pred.predictions
            labels = eval_pred.label_ids
            preds = np.argmax(logits, axis=1)

            # Accuracy
            acc = float((preds == labels).mean())

            # Per-class stats for macro metrics
            unique_labels = np.unique(labels)
            precisions = []
            recalls = []
            f1s = []
            for c in unique_labels:
                tp = float(np.sum((preds == c) & (labels == c)))
                fp = float(np.sum((preds == c) & (labels != c)))
                fn = float(np.sum((preds != c) & (labels == c)))
                precision_c = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall_c = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                if precision_c + recall_c > 0:
                    f1_c = 2 * precision_c * recall_c / (precision_c + recall_c)
                else:
                    f1_c = 0.0
                precisions.append(precision_c)
                recalls.append(recall_c)
                f1s.append(f1_c)

            precision_macro = float(np.mean(precisions)) if precisions else 0.0
            recall_macro = float(np.mean(recalls)) if recalls else 0.0
            f1_macro = float(np.mean(f1s)) if f1s else 0.0

            return {
                "accuracy": acc,
                "precision_macro": precision_macro,
                "recall_macro": recall_macro,
                "f1_macro": f1_macro,
            }

        return _compute_metrics

    # Use evaluate for richer metrics
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=1)
        labels = eval_pred.label_ids
        acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
        precision = precision_metric.compute(predictions=preds, references=labels, average="macro")[
            "precision"
        ]
        recall = recall_metric.compute(predictions=preds, references=labels, average="macro")["recall"]
        f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
        return {
            "accuracy": acc,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
        }

    return compute_metrics


# -----------------------------
# Train / Eval entrypoints
# -----------------------------


def do_train(args: argparse.Namespace):
    label2id, id2label = build_label_maps(args.data_root)

    # Load a compatible processor (try AutoVideoProcessor first, then AutoImageProcessor)
    try:
        processor = AutoVideoProcessor.from_pretrained(args.model_id)
    except Exception:
        processor = AutoImageProcessor.from_pretrained(args.model_id)

    model = AutoModelForVideoClassification.from_pretrained(
        args.model_id,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    # Propagate expected num_frames to collate_fn via environment variable for enforcement.
    try:
        expected_frames = _get_num_frames_from_config(model.config, 16)
        os.environ.setdefault("AMBER_FORCE_NUM_FRAMES", str(expected_frames))
    except Exception:
        pass

    train_ds, val_ds, test_ds = build_datasets(args.data_root, processor, model)
    if train_ds is None:
        raise ValueError("Training split not found. Ensure data_root/train exists.")

    metric_fn = build_metrics()

    # Determine if dataset exposes a length; if not, we MUST provide max_steps to Trainer.
    max_steps = None
    has_len = True
    try:  # Some pytorchvideo datasets do not implement __len__
        _ = len(train_ds)  # noqa: F841
    except Exception:  # pragma: no cover - environment dependent
        has_len = False

    if not has_len:
        # Derive number of training videos, fallback to heuristic if unavailable.
        if getattr(train_ds, "num_videos", None):
            train_videos = int(train_ds.num_videos)
        else:
            train_videos = 300  # heuristic similar to earlier default
        steps_per_epoch = max(1, train_videos // max(1, args.batch_size))
        target_epochs = args.epochs if args.epochs is not None else 4
        max_steps = steps_per_epoch * target_epochs
    elif args.epochs is None:
        # User omitted epochs; still create a reasonable schedule based on heuristic
        if getattr(train_ds, "num_videos", None):
            train_videos = int(train_ds.num_videos)
        else:
            train_videos = 300
        steps_per_epoch = max(1, train_videos // max(1, args.batch_size))
        max_steps = steps_per_epoch * 4  # default to 4 epochs worth of steps

    # ------------------------------------------------------------------
    # TrainingArguments compatibility across transformers versions.
    # Older versions (<3.x) lacked evaluation_strategy/save_strategy/etc.
    # We introspect the __init__ signature to only pass supported params.
    # ------------------------------------------------------------------
    import inspect

    ta_sig = inspect.signature(TrainingArguments.__init__)
    ta_params = set(ta_sig.parameters.keys())

    eval_strategy = "epoch" if val_ds is not None else ("no" if "evaluation_strategy" in ta_params else None)
    save_strategy = "epoch" if val_ds is not None else ("no" if "save_strategy" in ta_params else None)

    ta_kwargs = {
        "output_dir": args.output_dir,
    }
    # Always safe / common params
    # Build a map of common TrainingArguments. IMPORTANT: do NOT pass max_steps=None
    # because some transformers versions perform numeric comparisons on it and
    # expect an int. We only include max_steps when we intentionally computed
    # it (i.e. when epochs was not provided by the user and we derived a steps schedule).
    common_map = {
        "remove_unused_columns": False,
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "warmup_ratio": 0.1,
        "logging_steps": 10,
        # If dataset lacks __len__, Trainer will rely purely on max_steps; keep epochs minimal (1) to avoid confusion.
        "num_train_epochs": args.epochs if (args.epochs is not None and max_steps is None) else (args.epochs if args.epochs is not None else 1),
        # max_steps intentionally excluded if None (added conditionally below)
        "fp16": args.fp16,
        "dataloader_num_workers": args.num_workers,
    }
    if max_steps is not None:
        common_map["max_steps"] = max_steps
    for k, v in common_map.items():
        if k in ta_params:
            ta_kwargs[k] = v

    # Optionally disable tqdm progress bars if user requested and transformers version supports it
    if getattr(args, "no_tqdm", False) and "disable_tqdm" in ta_params:
        ta_kwargs["disable_tqdm"] = True

    # Conditional / version-specific
    if "evaluation_strategy" in ta_params and eval_strategy is not None:
        ta_kwargs["evaluation_strategy"] = eval_strategy
    elif "eval_strategy" in ta_params and eval_strategy is not None:  # newer alias just in case
        ta_kwargs["eval_strategy"] = eval_strategy
    elif "evaluate_during_training" in ta_params and val_ds is not None:
        ta_kwargs["evaluate_during_training"] = True

    if "save_strategy" in ta_params and save_strategy is not None:
        ta_kwargs["save_strategy"] = save_strategy
    elif "save_steps" in ta_params and val_ds is not None:
        # fallback: save every epoch approximation -> set to large steps if dataset length known.
        if hasattr(train_ds, "__len__"):
            try:
                steps_per_epoch = max(1, len(train_ds) // max(1, args.batch_size))
                ta_kwargs["save_steps"] = steps_per_epoch
            except Exception:
                pass

    if "load_best_model_at_end" in ta_params:
        ta_kwargs["load_best_model_at_end"] = bool(val_ds is not None)
    if "metric_for_best_model" in ta_params:
        ta_kwargs["metric_for_best_model"] = "accuracy"
    if "report_to" in ta_params:
        ta_kwargs["report_to"] = ["none"]

    training_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=processor,
        compute_metrics=metric_fn,
        data_collator=collate_fn,
    )

    train_result = trainer.train()
    trainer.save_model()  # saves to output_dir

    metrics = {"train_runtime": float(train_result.metrics.get("train_runtime", 0))}
    if val_ds is not None:
        eval_metrics = trainer.evaluate()
        metrics.update({f"val_{k}": v for k, v in eval_metrics.items()})

    # Optional test evaluation
    if args.eval_test and test_ds is not None:
        test_metrics = trainer.evaluate(test_ds)
        metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

    # Print final summary
    for k, v in metrics.items():
        print(f"{k}: {v}")


def do_eval(args: argparse.Namespace):
    label2id, id2label = build_label_maps(args.data_root)
    # Load processor and model automatically for eval
    model_path = args.model if args.model else args.model_id
    try:
        processor = AutoVideoProcessor.from_pretrained(model_path)
    except Exception:
        processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModelForVideoClassification.from_pretrained(model_path)

    # Ensure temporal enforcement during evaluation as well.
    try:
        expected_frames = _get_num_frames_from_config(model.config, 16)
        os.environ.setdefault("AMBER_FORCE_NUM_FRAMES", str(expected_frames))
    except Exception:
        pass

    # Ensure label maps align if possible
    if set(model.config.id2label.values()) != set(id2label.values()):
        print("Warning: Model labels differ from dataset labels. Proceeding with model's label set.")

    _, val_ds, test_ds = build_datasets(args.data_root, processor, model)
    if args.split == "val" and val_ds is None:
        raise ValueError("Validation split not found.")
    if args.split == "test" and test_ds is None:
        raise ValueError("Test split not found.")

    eval_ds = val_ds if args.split == "val" else test_ds

    metric_fn = build_metrics()

    import inspect
    ta_sig = inspect.signature(TrainingArguments.__init__)
    ta_params = set(ta_sig.parameters.keys())
    ta_kwargs = {"output_dir": os.path.join(args.output_dir, "eval")}
    if "per_device_eval_batch_size" in ta_params:
        ta_kwargs["per_device_eval_batch_size"] = args.batch_size
    if "dataloader_num_workers" in ta_params:
        ta_kwargs["dataloader_num_workers"] = args.num_workers
    if "report_to" in ta_params:
        ta_kwargs["report_to"] = ["none"]
    if getattr(args, "no_tqdm", False) and "disable_tqdm" in ta_params:
        ta_kwargs["disable_tqdm"] = True
    training_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=None,
        eval_dataset=eval_ds,
        processing_class=processor,
        compute_metrics=metric_fn,
        data_collator=collate_fn,
    )

    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"{k}: {v}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AMBER: Video classification train/eval runner (VideoMAE)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Train
    p_train = sub.add_parser("train", help="Fine-tune VideoMAE on UCF101 subset")
    p_train.add_argument("--data-root", type=str, required=True, help="Root folder of dataset (train/val/test)")
    p_train.add_argument("--model-id", type=str, default="MCG-NJU/videomae-base", help="Pretrained model id/path")
    p_train.add_argument("--output-dir", type=str, default="runs/videomae-ucf")
    p_train.add_argument("--batch-size", type=int, default=4)
    p_train.add_argument("--epochs", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=5e-5)
    p_train.add_argument("--num-workers", type=int, default=4)
    p_train.add_argument("--fp16", action="store_true")
    p_train.add_argument("--eval-test", action="store_true", help="Also evaluate on test split after training")
    p_train.add_argument("--no_tqdm", action="store_true", help="Disable tqdm progress bars if supported by transformers version")

    # Eval
    p_eval = sub.add_parser("eval", help="Evaluate a fine-tuned model on val/test")
    p_eval.add_argument("--data-root", type=str, required=True)
    p_eval.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to fine-tuned model dir or HF repo id; if omitted, uses --model-id",
    )
    p_eval.add_argument("--model-id", type=str, default="MCG-NJU/videomae-base")
    p_eval.add_argument("--split", type=str, choices=["val", "test"], default="val")
    p_eval.add_argument("--batch-size", type=int, default=4)
    p_eval.add_argument("--num-workers", type=int, default=4)
    p_eval.add_argument("--output-dir", type=str, default="runs/videomae-ucf")
    p_eval.add_argument("--no_tqdm", action="store_true", help="Disable tqdm progress bars if supported by transformers version")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.cmd == "train":
        do_train(args)
    elif args.cmd == "eval":
        do_eval(args)
    else:  # pragma: no cover
        raise SystemExit(f"Unknown subcommand: {args.cmd}")


if __name__ == "__main__":
    main()
