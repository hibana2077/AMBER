#!/usr/bin/env python
"""
Dataset downloader for video classification datasets.

Supports downloading and preparing common datasets like UCF101, HMDB51, and Kinetics subsets.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import urlretrieve
from urllib.parse import urlparse, unquote

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# Dataset configurations
DATASETS = {
    "ucf101_subset": {
        "name": "UCF101 Subset (10 classes)",
        "description": "A subset of UCF101 with 10 action classes for quick experiments",
        "url": "https://huggingface.co/datasets/sayakpaul/ucf101-subset/resolve/main/UCF101_subset.tar.gz?download=true",
        "type": "ucf101",
        "classes": [
            "ApplyEyeMakeup", "BasketballDunk", "BenchPress", "Biking", "Billiards",
            "BlowingCandles", "Bowling", "BoxingPunchingBag", "BoxingSpeedBag", "Breaststroke"
        ],
        "train_per_class": 30,
        "val_per_class": 10,
        "test_per_class": 10,
    },
    "ucf101_full": {
        "name": "UCF101 Full (101 classes)",
        "description": "Complete UCF101 dataset with 101 action classes",
        "url": "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar",
        "splits_url": "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip",
        "type": "ucf101",
        "classes": "all",
    },
    "hmdb51": {
        "name": "HMDB51 (51 classes)",
        "description": "HMDB51 dataset with 51 action classes",
        "url": "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar",
        "splits_url": "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar",
        "type": "hmdb51",
        "classes": "all",
    },
}


class DownloadProgressBar:
    """Progress bar for downloads."""
    
    def __init__(self, desc: str = "Downloading"):
        self.pbar = None
        self.desc = desc
    
    def __call__(self, block_num: int, block_size: int, total_size: int):
        if self.pbar is None:
            if _HAS_TQDM:
                self.pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=self.desc)
            else:
                print(f"{self.desc}... (install tqdm for progress bar)")
        
        if _HAS_TQDM and self.pbar is not None:
            downloaded = block_num * block_size
            if downloaded < total_size:
                self.pbar.update(block_size)
            else:
                self.pbar.close()


def _looks_like_html(file_path: str) -> bool:
    """Heuristically determine if downloaded file is an HTML error page instead of the expected binary archive."""
    try:
        with open(file_path, 'rb') as f:
            start = f.read(512).lower()
        if not start:
            return True
        # Typical HTML markers
        if start.startswith(b'<!doctype html') or start.startswith(b'<html'):
            return True
        # Cloudflare / HuggingFace / gateway error strings
        if b'<title>' in start and b'error' in start:
            return True
        return False
    except Exception:
        return True


def download_file(url: str, output_path: str, desc: str = "Downloading"):
    """Download a file from URL with progress bar and basic validation.

    If the resulting file appears to be HTML (common when a host returns an error page
    or a login/ratelimit notice), raise a helpful error early instead of failing later
    during extraction with a confusing gzip/zip error.
    """
    print(f"Downloading from {url} -> {output_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    urlretrieve(url, output_path, DownloadProgressBar(desc))
    size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
    if _looks_like_html(output_path):
        with open(output_path, 'rb') as f:
            snippet = f.read(300).decode(errors='ignore')
        raise RuntimeError(
            "Downloaded file appears to be HTML, not an archive. This usually means the host "
            "returned an error or you hit a rate limit. URL: " + url + "\nFirst bytes:\n" + snippet
        )
    if size < 1024:  # Highly unlikely for valid dataset archive
        raise RuntimeError(f"Downloaded file too small ({size} bytes) – download likely failed: {url}")
    print(f"Downloaded {size/1_048_576:.2f} MB to {output_path}")


def extract_archive(archive_path: str, output_dir: str):
    """Extract archive (zip, rar, tar/tar.gz/tgz) to output directory with magic-byte detection.

    Adds resilience against mis-labelled files and provides clearer diagnostics when the
    downloaded content is not an archive (e.g. HTML error page).
    """
    print(f"Extracting {archive_path} -> {output_dir}")
    lower_name = archive_path.lower()

    if not os.path.isfile(archive_path):
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    # Quick HTML re-check (in case validation skipped)
    if _looks_like_html(archive_path):
        raise RuntimeError(
            "The file to extract looks like HTML, not an archive. The download likely failed. "
            f"Path: {archive_path}"
        )

    # Read magic bytes
    with open(archive_path, 'rb') as f:
        magic = f.read(8)

    def try_tar(mode: str = 'r:*'):
        with tarfile.open(archive_path, mode) as tar:
            tar.extractall(output_dir)

    try:
        # ZIP: PK\x03\x04
        if magic.startswith(b'PK') or lower_name.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
        # RAR: Rar!\x1A\x07\x00
        elif magic.startswith(b'Rar!') or lower_name.endswith('.rar'):
            try:
                subprocess.run(['unrar', 'x', '-y', archive_path, output_dir], check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("ERROR: RAR archive detected but 'unrar' command not found.")
                print("Install unrar (e.g. 'sudo apt-get install unrar' or 'brew install unrar').")
                sys.exit(1)
        # Gzip/Tar: gzip magic 1F 8B; or a tar file (ustar later in file) – tarfile can auto-detect with r:*
        elif magic.startswith(b'\x1f\x8b') or lower_name.endswith(('.tar.gz', '.tgz', '.tar')) or tarfile.is_tarfile(archive_path):
            try:
                try_tar('r:*')  # auto-detect compression
            except tarfile.ReadError as e:
                # Provide guidance
                raise RuntimeError(
                    f"Failed to read tar archive (auto mode). Original error: {e}\n"
                    "This may happen if the download returned an HTML error page or was truncated. "
                    "Try deleting the file and re-running the download, or check your network."
                ) from e
        else:
            raise ValueError(
                "Unsupported or unrecognized archive format. Magic bytes: " + magic.hex() +
                f"  File: {archive_path}"
            )
    except Exception as e:
        if '?' in archive_path:
            print("Hint: The filename previously contained a '?' which may have confused extension detection.")
        raise

    print(f"Extraction complete: {output_dir}")


def organize_ucf101(source_dir: str, output_dir: str, config: Dict, split_file: Optional[str] = None):
    """Organize UCF101 dataset into train/val/test splits."""
    print("Organizing UCF101 dataset...")
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Find UCF101 directory
    ucf_dirs = list(source_path.glob("**/UCF-101"))
    if not ucf_dirs:
        ucf_dirs = [source_path]
    
    ucf_dir = ucf_dirs[0]
    print(f"Found UCF101 directory: {ucf_dir}")
    
    # Get all class directories
    all_classes = sorted([d.name for d in ucf_dir.iterdir() if d.is_dir()])
    print(f"Found {len(all_classes)} classes")
    
    # Filter classes if subset
    if config["classes"] != "all":
        target_classes = config["classes"]
        all_classes = [c for c in all_classes if c in target_classes]
        print(f"Using subset of {len(all_classes)} classes")
    
    if not all_classes:
        raise ValueError("No classes found in the dataset")
    
    # Create output structure
    for split in ["train", "val", "test"]:
        for class_name in all_classes:
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Organize videos
    class_label_map = {}
    for idx, class_name in enumerate(all_classes):
        class_label_map[class_name] = idx
        class_dir = ucf_dir / class_name
        
        if not class_dir.exists():
            print(f"Warning: Class directory not found: {class_dir}")
            continue
        
        videos = sorted(list(class_dir.glob("*.avi")) + list(class_dir.glob("*.mp4")))
        print(f"Processing {class_name}: {len(videos)} videos")
        
        if config["classes"] == "all":
            # Use official splits if available
            # For simplicity, use 70/15/15 split
            n_train = int(len(videos) * 0.7)
            n_val = int(len(videos) * 0.15)
            
            train_videos = videos[:n_train]
            val_videos = videos[n_train:n_train + n_val]
            test_videos = videos[n_train + n_val:]
        else:
            # Use config-specified counts
            n_train = config.get("train_per_class", 30)
            n_val = config.get("val_per_class", 10)
            n_test = config.get("test_per_class", 10)
            
            train_videos = videos[:n_train]
            val_videos = videos[n_train:n_train + n_val]
            test_videos = videos[n_train + n_val:n_train + n_val + n_test]
        
        # Copy videos to splits
        for video in train_videos:
            shutil.copy2(video, output_path / "train" / class_name / video.name)
        
        for video in val_videos:
            shutil.copy2(video, output_path / "val" / class_name / video.name)
        
        for video in test_videos:
            shutil.copy2(video, output_path / "test" / class_name / video.name)
    
    # Create metadata
    metadata = {
        "dataset_name": config["name"],
        "description": config["description"],
        "num_classes": len(all_classes),
        "classes": sorted([c.lower().replace(" ", "_") for c in all_classes]),
        "train_videos": sum(len(list((output_path / "train" / c).iterdir())) for c in all_classes),
        "val_videos": sum(len(list((output_path / "val" / c).iterdir())) for c in all_classes),
        "test_videos": sum(len(list((output_path / "test" / c).iterdir())) for c in all_classes),
    }
    
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*50)
    print("Dataset organization complete!")
    print(f"Output directory: {output_path}")
    print(f"Classes: {metadata['num_classes']}")
    print(f"Train videos: {metadata['train_videos']}")
    print(f"Val videos: {metadata['val_videos']}")
    print(f"Test videos: {metadata['test_videos']}")
    print("="*50)


def organize_hmdb51(source_dir: str, output_dir: str, config: Dict, splits_dir: Optional[str] = None):
    """Organize HMDB51 dataset into train/val/test splits."""
    print("Organizing HMDB51 dataset...")
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # HMDB51 videos are in nested RAR files, need to extract first
    # This is a simplified version - actual HMDB51 needs more preprocessing
    print("Note: HMDB51 requires manual extraction of nested RAR archives")
    print("Please refer to HMDB51 official instructions for full preprocessing")
    
    # For now, assume videos are already extracted
    all_classes = sorted([d.name for d in source_path.iterdir() if d.is_dir()])
    print(f"Found {len(all_classes)} classes")
    
    # Create output structure with 70/15/15 split
    for split in ["train", "val", "test"]:
        for class_name in all_classes:
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
    
    for class_name in all_classes:
        class_dir = source_path / class_name
        videos = sorted(list(class_dir.glob("*.avi")) + list(class_dir.glob("*.mp4")))
        
        n_train = int(len(videos) * 0.7)
        n_val = int(len(videos) * 0.15)
        
        train_videos = videos[:n_train]
        val_videos = videos[n_train:n_train + n_val]
        test_videos = videos[n_train + n_val:]
        
        for video in train_videos:
            shutil.copy2(video, output_path / "train" / class_name / video.name)
        for video in val_videos:
            shutil.copy2(video, output_path / "val" / class_name / video.name)
        for video in test_videos:
            shutil.copy2(video, output_path / "test" / class_name / video.name)
    
    print(f"Organized HMDB51 dataset to {output_path}")


def download_dataset(dataset_key: str, output_dir: str, keep_archive: bool = False):
    """Download and prepare a dataset."""
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_key}. Available: {list(DATASETS.keys())}")
    
    config = DATASETS[dataset_key]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Downloading: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"{'='*60}\n")
    
    # Download main dataset
    temp_dir = output_path / "temp_download"
    temp_dir.mkdir(exist_ok=True)
    
    # Derive a clean archive file name (strip query parameters)
    parsed = urlparse(config["url"])
    archive_name = os.path.basename(unquote(parsed.path))
    archive_path = temp_dir / archive_name

    # If a previously downloaded file kept query params, rename it so extension detection works
    # e.g., 'UCF101_subset.tar.gz?download=true'
    for existing in temp_dir.iterdir():
        if existing.name.startswith(archive_name) and '?' in existing.name and not archive_path.exists():
            # Rename to clean name
            print(f"Renaming previously downloaded file {existing.name} -> {archive_name}")
            existing.rename(archive_path)
            break

    if not archive_path.exists():
        # Download to a temporary path first to avoid partial file with final name
        tmp_download_path = archive_path.with_suffix(archive_path.suffix + '.part')
        if tmp_download_path.exists():
            tmp_download_path.unlink()
        download_file(config["url"], str(tmp_download_path), f"Downloading {config['name']}")
        tmp_download_path.rename(archive_path)
    else:
        print(f"Archive already exists: {archive_path}")
    
    # Extract archive
    extract_dir = temp_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)
    extract_archive(str(archive_path), str(extract_dir))
    
    # Download splits if needed
    splits_dir = None
    if "splits_url" in config:
        parsed_splits = urlparse(config["splits_url"])
        splits_name = os.path.basename(unquote(parsed_splits.path))
        splits_path = temp_dir / splits_name

        # Handle legacy downloaded names with query params
        for existing in temp_dir.iterdir():
            if existing.name.startswith(splits_name) and '?' in existing.name and not splits_path.exists():
                print(f"Renaming previously downloaded splits file {existing.name} -> {splits_name}")
                existing.rename(splits_path)
                break

        if not splits_path.exists():
            tmp_splits_path = splits_path.with_suffix(splits_path.suffix + '.part')
            if tmp_splits_path.exists():
                tmp_splits_path.unlink()
            download_file(config["splits_url"], str(tmp_splits_path), "Downloading splits")
            tmp_splits_path.rename(splits_path)

        splits_extract_dir = temp_dir / "splits"
        splits_extract_dir.mkdir(exist_ok=True)
        extract_archive(str(splits_path), str(splits_extract_dir))
        splits_dir = str(splits_extract_dir)
    
    # Organize dataset based on type
    if config["type"] == "ucf101":
        organize_ucf101(str(extract_dir), str(output_path), config, splits_dir)
    elif config["type"] == "hmdb51":
        organize_hmdb51(str(extract_dir), str(output_path), config, splits_dir)
    else:
        raise ValueError(f"Unknown dataset type: {config['type']}")
    
    # Cleanup
    if not keep_archive:
        print("Cleaning up temporary files...")
        shutil.rmtree(temp_dir)
    
    print(f"\n✓ Dataset ready at: {output_path}")
    print(f"\nYou can now train with:")
    print(f"  python -m src.run train --data-root {output_path} --output-dir runs/experiment")


def list_datasets():
    """List all available datasets."""
    print("\nAvailable datasets:")
    print("=" * 80)
    for key, config in DATASETS.items():
        print(f"\n{key}:")
        print(f"  Name: {config['name']}")
        print(f"  Description: {config['description']}")
        if config['classes'] != 'all':
            print(f"  Classes: {len(config['classes'])}")
            print(f"  Videos per class: ~{config.get('train_per_class', 0) + config.get('val_per_class', 0) + config.get('test_per_class', 0)}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare video classification datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available datasets
  python -m src.utils.dataset_downloader --list
  
  # Download UCF101 subset (10 classes, ~500 videos)
  python -m src.utils.dataset_downloader --dataset ucf101_subset --output-dir ./datasets/ucf101_subset
  
  # Download full UCF101 (101 classes, ~13k videos)
  python -m src.utils.dataset_downloader --dataset ucf101_full --output-dir ./datasets/ucf101_full
  
  # Download HMDB51
  python -m src.utils.dataset_downloader --dataset hmdb51 --output-dir ./datasets/hmdb51
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASETS.keys()),
        help="Dataset to download"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./datasets",
        help="Output directory for the dataset (default: ./datasets)"
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep downloaded archive files after extraction"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available datasets"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_datasets()
        return
    
    if not args.dataset:
        parser.error("--dataset is required (or use --list to see available datasets)")
    
    try:
        download_dataset(args.dataset, args.output_dir, args.keep_archive)
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
