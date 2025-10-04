# AMBER Project Structure

## Complete File Structure

```
AMBER/
├── README.md                              # Main documentation with quick start
├── requirements.txt                       # Python dependencies
├── .gitignore                            # Git ignore rules
│
├── docs/                                 # Documentation directory
│   ├── video_cls_dataset_spec.md        # ⭐ NEW: Dataset format specification
│   ├── quick_reference.md               # ⭐ NEW: Command reference guide
│   ├── new_features.md                  # ⭐ NEW: New features summary
│   └── refactoring_summary.md           # Code refactoring history
│
├── example/                              # Example scripts
│   ├── complete_workflow.py             # ⭐ NEW: End-to-end workflow example
│   ├── timesformer_test.py              # TimeSformer model test
│   ├── videomae_test.py                 # VideoMAE model test
│   └── vivit_test.py                    # ViViT model test
│
└── src/                                  # Source code
    ├── __init__.py
    ├── run.py                            # Main training/evaluation CLI
    ├── attn_map.py                       # Attention map visualization
    │
    ├── amber/                            # AMBER package
    │   └── __init__.py
    │
    └── utils/                            # Utility modules
        ├── __init__.py
        ├── video_utils.py                # Video processing utilities
        ├── attention_utils.py            # Attention extraction utilities
        ├── visualization_utils.py        # Visualization utilities
        ├── dataset_downloader.py         # ⭐ NEW: Dataset download tool
        └── validate_dataset.py           # ⭐ NEW: Dataset validation tool
```

## New Files Added (⭐)

### Tools
1. **src/utils/dataset_downloader.py** - Automated dataset downloading and preparation
2. **src/utils/validate_dataset.py** - Dataset structure and integrity validation

### Documentation
3. **docs/video_cls_dataset_spec.md** - Complete dataset format specification
4. **docs/quick_reference.md** - Handy command reference guide
5. **docs/new_features.md** - New features summary and integration guide

### Examples
6. **example/complete_workflow.py** - Full pipeline example script

### Updated Files
- **README.md** - Added new features, installation instructions, quick start
- **requirements.txt** - Added `tqdm` for progress bars

## Core Components

### 1. Training & Evaluation (`src/run.py`)
Main CLI for training and evaluating video classification models.

**Key Features:**
- Supports VideoMAE, TimeSformer, ViViT, V-JEPA2
- Auto model/processor selection
- Built-in train/val/test split handling
- Integrated metrics computation
- Mixed precision training support

### 2. Attention Visualization (`src/attn_map.py`)
Extract and visualize attention maps from video transformers.

**Key Features:**
- Multi-model support (VideoMAE, TimeSformer, ViViT, V-JEPA2)
- Layer and frame selection
- Heatmap overlay on original frames
- Customizable visualization

### 3. Dataset Management (NEW)

#### Dataset Downloader (`src/utils/dataset_downloader.py`)
Download and organize video classification datasets.

**Supported Datasets:**
- UCF101 subset (10 classes)
- UCF101 full (101 classes)
- HMDB51 (51 classes)

**Features:**
- Automatic download with progress tracking
- Archive extraction (zip/rar)
- Train/val/test split creation
- Metadata generation

#### Dataset Validator (`src/utils/validate_dataset.py`)
Validate dataset structure and quality.

**Checks:**
- Directory structure
- Class consistency
- Video file integrity
- Distribution statistics
- Format compatibility

### 4. Utility Modules (`src/utils/`)

- **video_utils.py** - Video frame extraction and processing
- **attention_utils.py** - Attention map extraction and processing
- **visualization_utils.py** - Attention heatmap visualization

## Workflow

### Standard Training Workflow

```
1. Download Dataset
   └─> dataset_downloader.py --dataset ucf101_subset --output-dir ./datasets/ucf101

2. Validate Dataset
   └─> validate_dataset.py --data-root ./datasets/ucf101

3. Train Model
   └─> run.py train --data-root ./datasets/ucf101 --output-dir runs/exp1

4. Evaluate Model
   └─> run.py eval --data-root ./datasets/ucf101 --model runs/exp1 --split val

5. Visualize (Optional)
   └─> attn_map.py --video video.mp4 --model runs/exp1
```

### Custom Dataset Workflow

```
1. Organize Your Dataset
   └─> Follow structure in docs/video_cls_dataset_spec.md

2. Validate Structure
   └─> validate_dataset.py --data-root /path/to/dataset --check-videos

3. Train Model
   └─> run.py train --data-root /path/to/dataset --output-dir runs/custom

4. Evaluate
   └─> run.py eval --data-root /path/to/dataset --model runs/custom --split val
```

## Documentation Index

### User Guides
- **README.md** - Start here for overview and installation
- **docs/quick_reference.md** - Quick command reference
- **docs/video_cls_dataset_spec.md** - Dataset format specification

### Developer Docs
- **docs/new_features.md** - New features and integration
- **docs/refactoring_summary.md** - Code organization history

### Examples
- **example/complete_workflow.py** - Full pipeline automation
- **example/videomae_test.py** - VideoMAE usage example
- **example/timesformer_test.py** - TimeSformer usage example
- **example/vivit_test.py** - ViViT usage example

## Dependencies

### Core Dependencies
- torch - PyTorch framework
- transformers - Hugging Face Transformers
- pytorchvideo - Video data handling
- torchvision - Vision utilities
- evaluate - Metrics computation

### Visualization
- matplotlib - Plotting
- pillow - Image processing
- opencv-python - Video decoding

### Video Processing
- imageio - Alternative video decoder
- imageio-ffmpeg - FFmpeg backend

### Utilities
- numpy - Numerical operations
- tqdm - Progress bars (NEW)
- timm - Model architectures

### System Tools (Optional)
- unrar - RAR extraction for UCF101/HMDB51

## Key Features

### 1. Model Support
- ✅ VideoMAE (all variants)
- ✅ TimeSformer (all variants)
- ✅ ViViT (all variants)
- ✅ V-JEPA2 (requires latest transformers)

### 2. Dataset Support
- ✅ UCF101 (subset and full)
- ✅ HMDB51
- ✅ Custom datasets (with validation)
- ✅ Multiple video formats (.avi, .mp4, .mkv, .mov, .webm)

### 3. Training Features
- ✅ Mixed precision (FP16)
- ✅ Multi-GPU support
- ✅ Validation during training
- ✅ Best model selection
- ✅ Comprehensive logging

### 4. Dataset Features (NEW)
- ✅ Automated downloading
- ✅ Structure validation
- ✅ Quality checks
- ✅ Statistics reporting
- ✅ Format standardization

## Quick Commands

### Dataset Management
```bash
# List datasets
python -m src.utils.dataset_downloader --list

# Download UCF101 subset
python -m src.utils.dataset_downloader --dataset ucf101_subset --output-dir ./datasets/ucf101

# Validate dataset
python -m src.utils.validate_dataset --data-root ./datasets/ucf101
```

### Training
```bash
# Basic training
python -m src.run train --data-root ./datasets/ucf101 --output-dir runs/exp1

# Advanced training
python -m src.run train \
  --data-root ./datasets/ucf101 \
  --model-id MCG-NJU/videomae-base \
  --output-dir runs/exp1 \
  --epochs 10 \
  --batch-size 4 \
  --lr 5e-5 \
  --fp16
```

### Evaluation
```bash
# Evaluate on validation set
python -m src.run eval --data-root ./datasets/ucf101 --model runs/exp1 --split val

# Evaluate on test set
python -m src.run eval --data-root ./datasets/ucf101 --model runs/exp1 --split test
```

### Visualization
```bash
# Extract attention maps
python -m src.attn_map --video video.mp4 --model runs/exp1
```

## Development Status

✅ = Implemented  
🚧 = In Progress  
📋 = Planned

### Current Features
- ✅ Video classification training
- ✅ Model evaluation
- ✅ Attention visualization
- ✅ Dataset downloading (NEW)
- ✅ Dataset validation (NEW)
- ✅ Dataset specification (NEW)
- ✅ Complete documentation (NEW)

### Future Enhancements
- 📋 More datasets (Kinetics, Something-Something, AVA)
- 📋 Data augmentation configuration
- 📋 Hyperparameter tuning utilities
- 📋 Model ensemble support
- 📋 Export to ONNX/TensorRT
- 📋 Gradio/Streamlit demo app

## Support

For issues, questions, or contributions, please refer to:
- **Quick Reference**: `docs/quick_reference.md`
- **Dataset Spec**: `docs/video_cls_dataset_spec.md`
- **Examples**: `example/` directory
