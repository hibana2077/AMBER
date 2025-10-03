# AMBER

Utilities for video classification experiments using Hugging Face Transformers.

New CLIs:

- src/run.py — fine-tune and evaluate video transformers (VideoMAE / ViViT / TimeSformer / V-JEPA2) on UCF101-like datasets. Uses AutoModel/AutoProcessor so you can switch via `--model-id`.
  - Train: `python -m src.run train --data-root UCF101_subset --model-id MCG-NJU/videomae-base --output-dir runs/videomae-ucf --epochs 4 --batch-size 4`
  - Eval: `python -m src.run eval --data-root UCF101_subset --model runs/videomae-ucf --split val`
  - Other models:
    - TimeSformer: `--model-id facebook/timesformer-base-finetuned-k400`
    - ViViT: `--model-id google/vivit-b-16x2-kinetics400`
    - V-JEPA2: `--model-id facebook/vjepa2-vitl-fpc16-256-ssv2` (requires latest Transformers)

- src/attn_map.py — extract attention maps for TimeSformer / VideoMAE / ViViT / V-JEPA2.
  - TimeSformer: `python -m src.attn_map --video path/to.mp4 --model facebook/timesformer-base-finetuned-k400`
  - VideoMAE: `python -m src.attn_map --video path/to.mp4 --model MCG-NJU/videomae-base-finetuned-kinetics`
  - ViViT: `python -m src.attn_map --video path/to.mp4 --model google/vivit-b-16x2-kinetics400`
  - V-JEPA2: `python -m src.attn_map --video path/to.mp4 --model facebook/vjepa2-vitl-fpc16-256-ssv2`

Install requirements (PowerShell):

```pwsh
pip install -r requirements.txt
```

Dataset structure expected (UCF101 subset style):

```text
UCF101_subset/
  train/CLASS_A/*.avi
  val/CLASS_A/*.avi
  test/CLASS_A/*.avi
  ...
```
