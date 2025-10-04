# Video Classification Dataset Specification

## Overview
This document defines the standard structure and requirements for video classification datasets used with AMBER training code.

## Directory Structure

The dataset must follow this hierarchical structure:

```
<DATASET_ROOT>/
├── train/
│   ├── <CLASS_1>/
│   │   ├── video_001.avi
│   │   ├── video_002.mp4
│   │   └── ...
│   ├── <CLASS_2>/
│   │   ├── video_001.avi
│   │   └── ...
│   └── ...
├── val/
│   ├── <CLASS_1>/
│   │   ├── video_001.avi
│   │   └── ...
│   ├── <CLASS_2>/
│   │   └── ...
│   └── ...
└── test/
    ├── <CLASS_1>/
    │   ├── video_001.avi
    │   └── ...
    ├── <CLASS_2>/
    │   └── ...
    └── ...
```

## Requirements

### 1. Split Directories
- **train/**: Training set (required)
- **val/**: Validation set (recommended, used for model selection)
- **test/**: Test set (optional, used for final evaluation)

At minimum, the `train/` directory must exist.

### 2. Class Directories
- Each split must contain subdirectories named after class labels
- Class names should be:
  - Alphanumeric with underscores (e.g., `apply_eye_makeup`, `basketball_dunk`)
  - Consistent across all splits
  - Human-readable and descriptive

### 3. Video Files
- **Supported formats**: `.avi`, `.mp4`, `.mkv`, `.mov`, `.webm`
- **Minimum requirements**:
  - Duration: At least 1 second
  - Resolution: At least 224x224 pixels (higher is better)
  - Frame rate: At least 15 FPS (30 FPS recommended)
  - Codec: H.264, VP9, or similar standard codec

### 4. File Naming
- Video filenames can be arbitrary but should be unique within each class directory
- Recommended pattern: `{class}_{id}.{ext}` or `v_{sequential_id}.{ext}`
- Avoid special characters except underscore and hyphen

## Metadata (Optional)

You can optionally provide a `metadata.json` file in the dataset root:

```json
{
  "dataset_name": "UCF101_subset",
  "num_classes": 10,
  "classes": [
    "apply_eye_makeup",
    "basketball_dunk",
    "bench_press",
    "biking",
    "billiards",
    "blowing_candles",
    "bowling",
    "boxing_punching_bag",
    "boxing_speed_bag",
    "breaststroke"
  ],
  "train_videos": 300,
  "val_videos": 100,
  "test_videos": 100,
  "fps": 30,
  "frame_width": 320,
  "frame_height": 240,
  "avg_duration_sec": 7.5,
  "description": "Subset of UCF101 with 10 action classes"
}
```

## Example Datasets

### UCF101 Subset
```
UCF101_subset/
├── train/
│   ├── apply_eye_makeup/
│   │   ├── v_ApplyEyeMakeup_g01_c01.avi
│   │   ├── v_ApplyEyeMakeup_g01_c02.avi
│   │   └── ... (30 videos)
│   ├── basketball_dunk/
│   │   └── ... (30 videos)
│   └── ... (8 more classes)
├── val/
│   ├── apply_eye_makeup/
│   │   └── ... (10 videos)
│   └── ... (9 more classes)
└── test/
    ├── apply_eye_makeup/
    │   └── ... (10 videos)
    └── ... (9 more classes)
```

### Kinetics Subset
```
Kinetics400_subset/
├── train/
│   ├── abseiling/
│   │   ├── 001.mp4
│   │   └── ...
│   ├── air_drumming/
│   │   └── ...
│   └── ... (more classes)
└── val/
    ├── abseiling/
    │   └── ...
    └── ...
```

## Data Preparation Guidelines

### 1. Class Balance
- Aim for balanced class distribution (similar number of videos per class)
- Minimum: 20 videos per class for training
- Recommended: 50-100+ videos per class for training

### 2. Split Ratios
Recommended split ratios:
- **Train: 70-80%** of total videos
- **Val: 10-15%** of total videos
- **Test: 10-15%** of total videos

### 3. Data Quality
- Remove corrupted videos (check with `cv2.VideoCapture` or `ffprobe`)
- Ensure videos contain the labeled action clearly
- Remove duplicates or near-duplicates
- Verify all videos are readable by pytorchvideo/OpenCV

### 4. Video Preprocessing (Optional)
For better training performance, consider:
- Trimming videos to relevant action segments (5-15 seconds)
- Re-encoding to consistent format (e.g., H.264 MP4)
- Resizing to consistent resolution (e.g., 320x240 or 640x480)
- Normalizing frame rates (e.g., 30 FPS)

## Validation Script

Use the provided validation script to check your dataset:

```bash
python -m src.utils.validate_dataset --data-root /path/to/dataset
```

This will:
- Check directory structure
- Count videos per class and split
- Verify video files are readable
- Report class distribution statistics
- Identify potential issues

## Integration with Training Code

Once your dataset follows this specification, you can use it directly with the training script:

```bash
# Train
python -m src.run train --data-root /path/to/dataset --output-dir runs/experiment --epochs 4

# Evaluate
python -m src.run eval --data-root /path/to/dataset --model runs/experiment --split val
```

The training code will automatically:
1. Scan class directories to build label mappings
2. Create train/val/test datasets using pytorchvideo
3. Apply appropriate data augmentation transforms
4. Handle video decoding and frame sampling

## Common Issues and Solutions

### Issue: "No class folders found"
- **Solution**: Ensure at least `train/` directory exists with class subdirectories

### Issue: "Could not open video"
- **Solution**: Check video codec compatibility; try re-encoding to H.264 MP4

### Issue: Training is slow
- **Solutions**:
  - Reduce video resolution (e.g., 320x240)
  - Use SSD storage instead of HDD
  - Increase `--num-workers` parameter
  - Enable `--fp16` for mixed precision training

### Issue: Out of memory
- **Solutions**:
  - Reduce `--batch-size`
  - Reduce number of frames sampled per video
  - Use a smaller model variant

## References

- UCF101: https://www.crcv.ucf.edu/data/UCF101.php
- Kinetics: https://deepmind.com/research/open-source/kinetics
- PyTorchVideo: https://pytorchvideo.org/
- Hugging Face Transformers: https://huggingface.co/docs/transformers/
