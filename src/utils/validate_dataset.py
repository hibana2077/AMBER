#!/usr/bin/env python
"""
Dataset validation utility for video classification datasets.

Validates that a dataset follows the expected structure and reports statistics.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False


def check_video_readable(video_path: str) -> Tuple[bool, str]:
    """Check if a video file is readable and get basic info."""
    if not _HAS_CV2:
        return True, "OpenCV not available, skipping video validation"
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Cannot open video file"
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        if frame_count == 0:
            return False, "Video has 0 frames"
        
        if width == 0 or height == 0:
            return False, f"Invalid resolution: {width}x{height}"
        
        duration = frame_count / fps if fps > 0 else 0
        info = f"{frame_count} frames, {fps:.1f} FPS, {width}x{height}, {duration:.1f}s"
        
        if duration < 1.0:
            return False, f"Video too short: {duration:.1f}s"
        
        return True, info
    
    except Exception as e:
        return False, str(e)


def validate_dataset(data_root: str, check_videos: bool = False) -> Dict:
    """Validate dataset structure and collect statistics."""
    root = Path(data_root)
    
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")
    
    print(f"Validating dataset at: {data_root}")
    print("=" * 80)
    
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
        "splits": {},
    }
    
    # Check for required directories
    splits_found = []
    for split in ["train", "val", "test"]:
        split_dir = root / split
        if split_dir.exists() and split_dir.is_dir():
            splits_found.append(split)
    
    if not splits_found:
        results["valid"] = False
        results["errors"].append("No split directories found (train/val/test)")
        return results
    
    if "train" not in splits_found:
        results["valid"] = False
        results["errors"].append("Training split (train/) not found")
    
    if "val" not in splits_found:
        results["warnings"].append("Validation split (val/) not found - recommended for model selection")
    
    print(f"✓ Found splits: {', '.join(splits_found)}")
    
    # Collect class information
    all_classes = set()
    split_class_counts = defaultdict(dict)
    video_formats = defaultdict(int)
    problematic_videos = []
    
    for split in splits_found:
        split_dir = root / split
        class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        
        if not class_dirs:
            results["errors"].append(f"No class directories found in {split}/ split")
            results["valid"] = False
            continue
        
        print(f"\n{split.upper()} split:")
        print("-" * 80)
        
        for class_dir in sorted(class_dirs):
            class_name = class_dir.name
            all_classes.add(class_name)
            
            # Find video files
            video_extensions = ['.avi', '.mp4', '.mkv', '.mov', '.webm']
            videos = []
            for ext in video_extensions:
                videos.extend(class_dir.glob(f"*{ext}"))
            
            video_count = len(videos)
            split_class_counts[split][class_name] = video_count
            
            # Count formats
            for video in videos:
                video_formats[video.suffix] += 1
            
            # Check video files
            if check_videos and videos:
                print(f"  {class_name}: {video_count} videos - checking...")
                
                # Sample check (check first, middle, last videos)
                sample_indices = [0, len(videos) // 2, len(videos) - 1] if len(videos) > 1 else [0]
                sample_videos = [videos[i] for i in sample_indices]
                
                for video_path in sample_videos:
                    readable, info = check_video_readable(str(video_path))
                    if not readable:
                        problematic_videos.append((split, class_name, video_path.name, info))
                        print(f"    ✗ {video_path.name}: {info}")
            else:
                print(f"  {class_name}: {video_count} videos")
            
            # Warnings for small classes
            if split == "train" and video_count < 20:
                results["warnings"].append(
                    f"Class '{class_name}' in train split has only {video_count} videos (recommend ≥20)"
                )
    
    # Check class consistency across splits
    train_classes = set(split_class_counts.get("train", {}).keys())
    for split in ["val", "test"]:
        if split in split_class_counts:
            split_classes = set(split_class_counts[split].keys())
            missing = train_classes - split_classes
            extra = split_classes - train_classes
            
            if missing:
                results["warnings"].append(
                    f"{split} split missing classes present in train: {', '.join(sorted(missing))}"
                )
            if extra:
                results["warnings"].append(
                    f"{split} split has extra classes not in train: {', '.join(sorted(extra))}"
                )
    
    # Compute statistics
    results["stats"]["num_classes"] = len(all_classes)
    results["stats"]["classes"] = sorted(all_classes)
    results["stats"]["video_formats"] = dict(video_formats)
    
    for split in splits_found:
        if split in split_class_counts:
            class_counts = split_class_counts[split]
            results["splits"][split] = {
                "num_classes": len(class_counts),
                "total_videos": sum(class_counts.values()),
                "videos_per_class": class_counts,
                "min_videos": min(class_counts.values()) if class_counts else 0,
                "max_videos": max(class_counts.values()) if class_counts else 0,
                "avg_videos": sum(class_counts.values()) / len(class_counts) if class_counts else 0,
            }
    
    if problematic_videos:
        results["warnings"].append(f"Found {len(problematic_videos)} problematic video files")
        results["problematic_videos"] = problematic_videos
    
    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    print(f"\n✓ Total classes: {results['stats']['num_classes']}")
    print(f"✓ Video formats: {', '.join(f'{ext} ({count})' for ext, count in video_formats.items())}")
    
    for split in splits_found:
        if split in results["splits"]:
            split_info = results["splits"][split]
            print(f"\n{split.upper()} split:")
            print(f"  Total videos: {split_info['total_videos']}")
            print(f"  Classes: {split_info['num_classes']}")
            print(f"  Videos per class: min={split_info['min_videos']}, "
                  f"max={split_info['max_videos']}, avg={split_info['avg_videos']:.1f}")
    
    # Print warnings
    if results["warnings"]:
        print("\n⚠ WARNINGS:")
        for warning in results["warnings"]:
            print(f"  - {warning}")
    
    # Print errors
    if results["errors"]:
        print("\n✗ ERRORS:")
        for error in results["errors"]:
            print(f"  - {error}")
        results["valid"] = False
    
    if results["valid"]:
        print("\n✓ Dataset validation PASSED")
    else:
        print("\n✗ Dataset validation FAILED")
    
    print("=" * 80)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate video classification dataset structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation (structure and counts)
  python -m src.utils.validate_dataset --data-root ./datasets/ucf101_subset
  
  # Full validation (includes video file checks)
  python -m src.utils.validate_dataset --data-root ./datasets/ucf101_subset --check-videos
  
  # Save validation report to JSON
  python -m src.utils.validate_dataset --data-root ./datasets/ucf101_subset --output report.json
        """
    )
    
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory of the dataset to validate"
    )
    parser.add_argument(
        "--check-videos",
        action="store_true",
        help="Check if video files are readable (slower but more thorough)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save validation report to JSON file"
    )
    
    args = parser.parse_args()
    
    try:
        results = validate_dataset(args.data_root, args.check_videos)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nValidation report saved to: {args.output}")
        
        # Exit with appropriate code
        exit(0 if results["valid"] else 1)
    
    except Exception as e:
        print(f"\nError during validation: {e}")
        exit(1)


if __name__ == "__main__":
    main()
