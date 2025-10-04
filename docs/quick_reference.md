# Quick Reference Guide

## Dataset Management

### List Available Datasets
```pwsh
python -m src.utils.dataset_downloader --list
```

### Download UCF101 Subset (10 classes, ~500 videos)
```pwsh
python -m src.utils.dataset_downloader --dataset ucf101_subset --output-dir ./datasets/ucf101_subset
```

### Download Full UCF101 (101 classes, ~13k videos)
```pwsh
python -m src.utils.dataset_downloader --dataset ucf101_full --output-dir ./datasets/ucf101_full
```

### Validate Dataset
```pwsh
# Basic validation (structure and counts)
python -m src.utils.validate_dataset --data-root ./datasets/ucf101_subset

# Full validation (includes video file checks)
python -m src.utils.validate_dataset --data-root ./datasets/ucf101_subset --check-videos

# Save validation report
python -m src.utils.validate_dataset --data-root ./datasets/ucf101_subset --output report.json
```

## Training

### Basic Training
```pwsh
python -m src.run train --data-root ./datasets/ucf101_subset --output-dir runs/experiment
```

### Training with Custom Settings
```pwsh
python -m src.run train `
  --data-root ./datasets/ucf101_subset `
  --model-id MCG-NJU/videomae-base `
  --output-dir runs/videomae-custom `
  --epochs 10 `
  --batch-size 4 `
  --lr 5e-5 `
  --num-workers 4 `
  --fp16
```

### Training with Different Models
```pwsh
# VideoMAE
python -m src.run train --data-root ./datasets/ucf101_subset --model-id MCG-NJU/videomae-base --output-dir runs/videomae

# TimeSformer
python -m src.run train --data-root ./datasets/ucf101_subset --model-id facebook/timesformer-base-finetuned-k400 --output-dir runs/timesformer

# ViViT
python -m src.run train --data-root ./datasets/ucf101_subset --model-id google/vivit-b-16x2-kinetics400 --output-dir runs/vivit
```

## Evaluation

### Evaluate on Validation Set
```pwsh
python -m src.run eval --data-root ./datasets/ucf101_subset --model runs/videomae --split val
```

### Evaluate on Test Set
```pwsh
python -m src.run eval --data-root ./datasets/ucf101_subset --model runs/videomae --split test
```

### Evaluate Pre-trained Model
```pwsh
python -m src.run eval --data-root ./datasets/ucf101_subset --model-id MCG-NJU/videomae-base-finetuned-kinetics --split val
```

## Attention Visualization

### Extract Attention Maps
```pwsh
# VideoMAE
python -m src.attn_map --video path/to/video.mp4 --model MCG-NJU/videomae-base-finetuned-kinetics

# TimeSformer
python -m src.attn_map --video path/to/video.mp4 --model facebook/timesformer-base-finetuned-k400

# ViViT
python -m src.attn_map --video path/to/video.mp4 --model google/vivit-b-16x2-kinetics400

# Your fine-tuned model
python -m src.attn_map --video path/to/video.mp4 --model runs/videomae
```

## Complete Workflow Example

Run the complete example script:
```pwsh
python example\complete_workflow.py
```

Or manually:

```pwsh
# 1. Download dataset
python -m src.utils.dataset_downloader --dataset ucf101_subset --output-dir ./datasets/ucf101_subset

# 2. Validate dataset
python -m src.utils.validate_dataset --data-root ./datasets/ucf101_subset

# 3. Train model
python -m src.run train --data-root ./datasets/ucf101_subset --output-dir runs/videomae --epochs 4 --batch-size 4

# 4. Evaluate model
python -m src.run eval --data-root ./datasets/ucf101_subset --model runs/videomae --split val

# 5. Visualize attention (optional)
python -m src.attn_map --video path/to/test_video.mp4 --model runs/videomae
```

## Troubleshooting

### Out of Memory
```pwsh
# Reduce batch size
python -m src.run train --data-root ./datasets/ucf101_subset --batch-size 1

# Enable mixed precision training
python -m src.run train --data-root ./datasets/ucf101_subset --fp16
```

### Slow Training
```pwsh
# Increase number of workers
python -m src.run train --data-root ./datasets/ucf101_subset --num-workers 8

# Enable mixed precision
python -m src.run train --data-root ./datasets/ucf101_subset --fp16
```

### Video Decoding Issues
```pwsh
# Validate dataset to find problematic videos
python -m src.utils.validate_dataset --data-root ./datasets/ucf101_subset --check-videos
```

### RAR Archive Extraction Issues
For UCF101/HMDB51 datasets, you need `unrar`:

**Windows:**
- Download from https://www.rarlab.com/rar_add.htm
- Add to PATH

**Linux/Ubuntu:**
```bash
sudo apt-get install unrar
```

**macOS:**
```bash
brew install unrar
```

## Model IDs

### VideoMAE
- `MCG-NJU/videomae-base`
- `MCG-NJU/videomae-base-finetuned-kinetics`
- `MCG-NJU/videomae-large`

### TimeSformer
- `facebook/timesformer-base-finetuned-k400`
- `facebook/timesformer-base-finetuned-k600`

### ViViT
- `google/vivit-b-16x2-kinetics400`

### V-JEPA2 (requires latest transformers)
- `facebook/vjepa2-vitl-fpc16-256-ssv2`

## Dataset Structure

Your dataset should follow this structure:

```
<DATASET_ROOT>/
├── train/
│   ├── class_1/
│   │   ├── video1.avi
│   │   ├── video2.mp4
│   │   └── ...
│   ├── class_2/
│   └── ...
├── val/
│   ├── class_1/
│   └── ...
└── test/ (optional)
    ├── class_1/
    └── ...
```

See [docs/video_cls_dataset_spec.md](video_cls_dataset_spec.md) for detailed specifications.

## Tips and Best Practices

### Training
1. Start with a small number of epochs (2-4) to verify setup
2. Use validation set for model selection
3. Enable `--fp16` for faster training on modern GPUs
4. Adjust `--batch-size` based on your GPU memory
5. Use `--eval-test` to evaluate on test set after training

### Dataset Preparation
1. Aim for at least 20-30 videos per class for training
2. Use 70/15/15 split for train/val/test
3. Ensure class balance across splits
4. Validate dataset before training
5. Remove corrupted or duplicate videos

### Hardware Recommendations
- **Minimum:** 8GB GPU memory, batch size 1-2
- **Recommended:** 16GB+ GPU memory, batch size 4-8
- **Optimal:** 24GB+ GPU memory, batch size 8-16

### Expected Training Time
With RTX 3090 (24GB), UCF101 subset (10 classes, 500 videos):
- 4 epochs: ~20-30 minutes
- 10 epochs: ~50-75 minutes

Times vary based on video resolution, model size, and hardware.
