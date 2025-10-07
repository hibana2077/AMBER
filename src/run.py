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
    import pytorchvideo.data  # We still rely on the dataset class only.
    import torchvision
    from torchvision.transforms import RandomCrop as TVRandomCrop, RandomHorizontalFlip as TVRandomHorizontalFlip, Resize as TVResize
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
        # split_dir = root / split
        # Scanning .\datasets\ucf101_subset\temp_download\extracted\UCF101_subset/train...
        # Scanning .\datasets\ucf101_subset\temp_download\extracted\UCF101_subset/val...
        # Scanning .\datasets\ucf101_subset\temp_download\extracted\UCF101_subset/test...
        split_dir = f"{root}\{split}"
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
                # pytorchvideo returns dict with 'video' (T,C,H,W); convert to (C,T,H,W) for our pipeline
                if isinstance(sample, dict) and 'video' in sample:
                    v = sample['video']  # (T,C,H,W)
                    if torch.is_tensor(v) and v.dim() == 4:
                        sample['video'] = v.permute(1,0,2,3)  # -> (C,T,H,W)
                    sample = transform(sample)
                    # convert back to (T,C,H,W) for consistency with downstream expectation
                    v2 = sample['video']
                    if torch.is_tensor(v2) and v2.shape[0] in (1,3):
                        sample['video'] = v2.permute(1,0,2,3)
                return sample

            ds.__getitem__ = _get  # type: ignore
        return ds

    return make("train", train_t), make("val", eval_t), make("test", eval_t)


def collate_fn(examples):
    # After our pipeline final shape stored as (T,C,H,W); convert to (C,T,H,W) then to HF expected (B, num_frames, C, H, W)
    processed = []
    for ex in examples:
        v = ex["video"]  # (T,C,H,W)
        if v.dim() != 4:
            raise ValueError(f"Unexpected video tensor shape {v.shape}")
        # Convert to (C,T,H,W) then rearrange for stack later
        v_cthw = v.permute(1,0,2,3)
        processed.append(v_cthw)
    pixel_values = torch.stack(processed)  # (B,C,T,H,W)
    # HF Video classification models expect pixel_values: (B, num_frames, C, H, W)
    pixel_values = pixel_values.permute(0,2,1,3,4)
    labels = torch.tensor([int(example["label"]) for example in examples])
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

    train_ds, val_ds, test_ds = build_datasets(args.data_root, processor, model)
    if train_ds is None:
        raise ValueError("Training split not found. Ensure data_root/train exists.")

    metric_fn = build_metrics()

    # Some datasets from pytorchvideo don't implement __len__, set max_steps accordingly
    max_steps = None
    if args.epochs is None:
        # Use steps if epochs not set: assume 300 videos by default like docs example
        if hasattr(train_ds, "num_videos") and train_ds.num_videos:
            train_videos = train_ds.num_videos
        else:
            train_videos = 300  # heuristic fallback
        steps_per_epoch = max(1, train_videos // max(1, args.batch_size))
        max_steps = steps_per_epoch * 4  # 4 epochs default

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        remove_unused_columns=False,
        evaluation_strategy="epoch" if val_ds is not None else "no",
        save_strategy="epoch" if val_ds is not None else "no",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=bool(val_ds is not None),
        metric_for_best_model="accuracy",
        num_train_epochs=args.epochs if args.epochs is not None else 1,
        max_steps=max_steps,
        fp16=args.fp16,
        dataloader_num_workers=args.num_workers,
        report_to=["none"],
    )

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

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "eval"),
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=args.num_workers,
        report_to=["none"],
    )

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
