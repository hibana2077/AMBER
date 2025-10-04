# AMBER

Utilities for video classification experiments using Hugging Face Transformers.

## Features

### Training & Evaluation

- **src/run.py** — fine-tune and evaluate video transformers (VideoMAE / ViViT / TimeSformer / V-JEPA2) on UCF101-like datasets. Uses AutoModel/AutoProcessor so you can switch via `--model-id`.
  - Train: `python -m src.run train --data-root UCF101_subset --model-id MCG-NJU/videomae-base --output-dir runs/videomae-ucf --epochs 4 --batch-size 4`
  - Eval: `python -m src.run eval --data-root UCF101_subset --model runs/videomae-ucf --split val`
  - Other models:
    - TimeSformer: `--model-id facebook/timesformer-base-finetuned-k400`
    - ViViT: `--model-id google/vivit-b-16x2-kinetics400`
    - V-JEPA2: `--model-id facebook/vjepa2-vitl-fpc16-256-ssv2` (requires latest Transformers)

### Attention Visualization

- **src/attn_map.py** — extract attention maps for TimeSformer / VideoMAE / ViViT / V-JEPA2.
  - TimeSformer: `python -m src.attn_map --video path/to.mp4 --model facebook/timesformer-base-finetuned-k400`
  - VideoMAE: `python -m src.attn_map --video path/to.mp4 --model MCG-NJU/videomae-base-finetuned-kinetics`
  - ViViT: `python -m src.attn_map --video path/to.mp4 --model google/vivit-b-16x2-kinetics400`
  - V-JEPA2: `python -m src.attn_map --video path/to.mp4 --model facebook/vjepa2-vitl-fpc16-256-ssv2`

### Dataset Management

- **src/utils/dataset_downloader.py** — download and prepare common video classification datasets
  - List datasets: `python -m src.utils.dataset_downloader --list`
  - Download UCF101 subset: `python -m src.utils.dataset_downloader --dataset ucf101_subset --output-dir ./datasets/ucf101_subset`
  - Download full UCF101: `python -m src.utils.dataset_downloader --dataset ucf101_full --output-dir ./datasets/ucf101_full`
  - Download HMDB51: `python -m src.utils.dataset_downloader --dataset hmdb51 --output-dir ./datasets/hmdb51`

- **src/utils/validate_dataset.py** — validate dataset structure and integrity
  - Basic validation: `python -m src.utils.validate_dataset --data-root ./datasets/ucf101_subset`
  - Full validation: `python -m src.utils.validate_dataset --data-root ./datasets/ucf101_subset --check-videos`

## Installation

Install requirements (PowerShell):

```pwsh
pip install -r requirements.txt
```

For RAR archive support (needed for UCF101/HMDB51):
- Windows: Download from https://www.rarlab.com/rar_add.htm
- Linux/Ubuntu: `sudo apt-get install unrar`
- macOS: `brew install unrar`

## Dataset Structure

See [docs/video_cls_dataset_spec.md](docs/video_cls_dataset_spec.md) for detailed dataset specifications.

Expected structure (UCF101 subset style):

```text
UCF101_subset/
  train/CLASS_A/*.avi
  val/CLASS_A/*.avi
  test/CLASS_A/*.avi
  ...
```

## Quick Start

1. Download a dataset:
```pwsh
python -m src.utils.dataset_downloader --dataset ucf101_subset --output-dir ./datasets/ucf101_subset
```

2. Validate the dataset:
```pwsh
python -m src.utils.validate_dataset --data-root ./datasets/ucf101_subset
```

3. Train a model:
```pwsh
python -m src.run train --data-root ./datasets/ucf101_subset --output-dir runs/videomae-ucf --epochs 4 --batch-size 4
```

4. Evaluate the model:
```pwsh
python -m src.run eval --data-root ./datasets/ucf101_subset --model runs/videomae-ucf --split val
```

## Documentation

- [Dataset Specification](docs/video_cls_dataset_spec.md) - Detailed dataset format requirements
- [Refactoring Summary](docs/refactoring_summary.md) - Code organization details

