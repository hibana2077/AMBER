# Code Refactoring Summary

## Overview
Successfully decomposed the monolithic `attn_map.py` file into modular utilities organized in the `src/utils/` directory.

## Changes Made

### New Files Created

#### 1. `src/utils/video_utils.py`
Contains video processing and frame extraction functions:
- `sample_video_frames_cv2()` - Sample frames using OpenCV
- `sample_video_frames_imageio()` - Sample frames using imageio (fallback)
- `sample_video_frames()` - Main interface with automatic fallback

**Features:**
- Automatic decoder fallback (OpenCV → imageio)
- Center cropping and resizing to target size
- Frame padding to ensure exact frame count
- Support for various video formats

#### 2. `src/utils/attention_utils.py`
Contains attention map extraction and processing utilities:
- `minmax_norm()` - Min-max normalization for attention maps
- `pick_layer_indices()` - Select evenly spaced layer indices
- `pick_frame_indices()` - Select evenly spaced frame indices
- `extract_cls_to_patch_attn_timesformer()` - Extract attention for TimeSformer models
- `extract_cls_to_patch_attn_videomae_vivit()` - Extract attention for VideoMAE/ViViT models

**Features:**
- Model-specific attention extraction logic
- Temporal tubelet handling for VideoMAE/ViViT
- Flexible sequence length detection
- Automatic normalization of attention maps

#### 3. `src/utils/visualization_utils.py`
Contains visualization functions:
- `overlay_grid()` - Create attention heatmap grid overlays

**Features:**
- Customizable colormap support
- Multi-layer, multi-frame grid layout
- Automatic figure sizing
- Alpha-blended heatmap overlays

#### 4. `src/utils/__init__.py`
Package initialization with clean exports:
- Exposes all utility functions through `__all__`
- Organized imports by category (video, attention, visualization)
- Enables `from src.utils import *` usage

### Modified Files

#### `src/attn_map.py`
**Before:** ~420 lines with all logic inline
**After:** ~183 lines (57% reduction)

**Changes:**
- Removed 9 function definitions (~240 lines)
- Added organized imports from utils modules
- Cleaner, more maintainable main script
- All functionality preserved

## Benefits

### 1. **Modularity**
- Functions are organized by purpose (video, attention, visualization)
- Easy to import utilities in other scripts
- Clear separation of concerns

### 2. **Reusability**
- Utilities can be used in other parts of the project
- Example scripts can import common functions
- No code duplication needed

### 3. **Maintainability**
- Easier to locate and fix bugs
- Changes to utilities affect all users consistently
- Better testing isolation

### 4. **Readability**
- Main script is now focused on CLI and orchestration
- Function names and module structure are self-documenting
- Reduced cognitive load when reading code

### 5. **Extensibility**
- Easy to add new video decoders
- Simple to support new model architectures
- Straightforward to add visualization options

## Usage Examples

### Import utilities directly:
```python
from src.utils import sample_video_frames, overlay_grid
from src.utils.attention_utils import pick_layer_indices
```

### Or import entire modules:
```python
from src.utils import video_utils, attention_utils, visualization_utils
```

### Use in new scripts:
```python
# New analysis script
from src.utils.video_utils import sample_video_frames
from src.utils.attention_utils import minmax_norm

frames = sample_video_frames("video.mp4", num_frames=16)
# ... process frames ...
```

## File Structure
```
src/
├── attn_map.py              # Main CLI script (now ~183 lines)
├── run.py
├── __init__.py
├── amber/
│   └── __init__.py
└── utils/
    ├── __init__.py          # Package exports
    ├── video_utils.py       # Video processing (187 lines)
    ├── attention_utils.py   # Attention extraction (142 lines)
    └── visualization_utils.py # Visualization (52 lines)
```

## Testing Recommendations

To ensure the refactoring didn't break functionality:

1. **Test the CLI tool:**
   ```powershell
   python -m src.attn_map --video path/to/video.mp4 --model MODEL_ID --out output.png
   ```

2. **Test individual utilities:**
   ```python
   from src.utils import sample_video_frames
   frames = sample_video_frames("test.mp4", num_frames=8)
   assert len(frames) == 8
   ```

3. **Run existing example scripts** to verify compatibility

## Future Improvements

Potential enhancements now that code is modular:

1. Add unit tests for each utility module
2. Create additional video decoders (e.g., decord, torchvision)
3. Add more visualization styles (animated GIFs, HTML reports)
4. Implement caching for processed videos
5. Add batch processing utilities
6. Create configuration classes for model-specific settings
