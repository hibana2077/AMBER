#!/usr/bin/env python
"""
Example: Download dataset, validate, and train a video classification model.

This script demonstrates the complete workflow:
1. Download a dataset (UCF101 subset)
2. Validate dataset structure
3. Train a VideoMAE model
4. Evaluate the model
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"\n✗ Error: {description} failed with exit code {result.returncode}")
        sys.exit(1)
    
    print(f"\n✓ {description} completed successfully")


def main():
    # Configuration
    dataset_name = "ucf101_subset"
    dataset_dir = Path("./datasets/ucf101_subset")
    output_dir = Path("./runs/videomae-ucf-example")
    model_id = "MCG-NJU/videomae-base"
    
    print("="*80)
    print("AMBER Video Classification - Complete Workflow Example")
    print("="*80)
    print(f"\nDataset: {dataset_name}")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Model: {model_id}")
    print(f"Output directory: {output_dir}")
    
    # Step 1: Download dataset (if not exists)
    if not dataset_dir.exists():
        print(f"\n→ Dataset not found, downloading {dataset_name}...")
        run_command(
            ["python", "-m", "src.utils.dataset_downloader", 
             "--dataset", dataset_name,
             "--output-dir", str(dataset_dir)],
            f"Step 1: Download {dataset_name}"
        )
    else:
        print(f"\n→ Dataset already exists at {dataset_dir}, skipping download")
    
    # Step 2: Validate dataset
    run_command(
        ["python", "-m", "src.utils.validate_dataset",
         "--data-root", str(dataset_dir)],
        "Step 2: Validate dataset structure"
    )
    
    # Step 3: Train model
    print(f"\n→ Training will start. This may take a while...")
    print(f"   Tip: Use --fp16 flag for faster training with mixed precision")
    print(f"   Tip: Adjust --batch-size based on your GPU memory")
    
    run_command(
        ["python", "-m", "src.run", "train",
         "--data-root", str(dataset_dir),
         "--model-id", model_id,
         "--output-dir", str(output_dir),
         "--epochs", "2",
         "--batch-size", "2",
         "--lr", "5e-5"],
        "Step 3: Train VideoMAE model"
    )
    
    # Step 4: Evaluate model
    run_command(
        ["python", "-m", "src.run", "eval",
         "--data-root", str(dataset_dir),
         "--model", str(output_dir),
         "--split", "val",
         "--batch-size", "4"],
        "Step 4: Evaluate model on validation set"
    )
    
    # Optional: Evaluate on test set
    if (dataset_dir / "test").exists():
        run_command(
            ["python", "-m", "src.run", "eval",
             "--data-root", str(dataset_dir),
             "--model", str(output_dir),
             "--split", "test",
             "--batch-size", "4"],
            "Step 5 (Optional): Evaluate model on test set"
        )
    
    print("\n" + "="*80)
    print("✓ Complete workflow finished successfully!")
    print("="*80)
    print(f"\nTrained model saved at: {output_dir}")
    print(f"\nNext steps:")
    print(f"  - Check training logs in {output_dir}")
    print(f"  - Visualize attention maps:")
    print(f"    python -m src.attn_map --video <path/to/video.mp4> --model {output_dir}")
    print(f"  - Fine-tune with more epochs for better results")
    print(f"  - Try different models (TimeSformer, ViViT, etc.)")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        sys.exit(1)
