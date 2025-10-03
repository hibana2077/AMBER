#!/usr/bin/env python
"""
Video processing utilities for frame extraction and sampling.
"""

from typing import List

import numpy as np
from PIL import Image

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False

try:
    import imageio.v2 as imageio
    _HAS_IMAGEIO = True
except Exception:
    imageio = None
    _HAS_IMAGEIO = False


def sample_video_frames_cv2(path: str, num_frames: int = 16, target_size: int = 224) -> List[Image.Image]:
    """
    Sample frames from video using OpenCV.
    
    Args:
        path: Path to video file
        num_frames: Number of frames to sample
        target_size: Target size for resizing frames (square)
        
    Returns:
        List of PIL Images
    """
    if not _HAS_CV2:
        raise RuntimeError("OpenCV is required: pip install opencv-python")
    
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        idxs = np.round(np.linspace(0, total - 1, num_frames)).astype(int)
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
    else:
        # sequential read
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        if not frames:
            cap.release()
            raise RuntimeError("No frames decoded from video (OpenCV)")
        idxs = np.round(np.linspace(0, len(frames) - 1, num_frames)).astype(int)
        frames = [frames[i] for i in idxs]
    cap.release()

    # Ensure exact count via padding
    while len(frames) < num_frames:
        frames.append(frames[-1])

    pil = []
    for fr in frames[:num_frames]:
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        h, w, _ = fr.shape
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        fr = fr[y0:y0 + side, x0:x0 + side]
        fr = cv2.resize(fr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        pil.append(Image.fromarray(fr))
    return pil


def sample_video_frames_imageio(path: str, num_frames: int = 16, target_size: int = 224) -> List[Image.Image]:
    """
    Sample frames from video using imageio.
    
    Args:
        path: Path to video file
        num_frames: Number of frames to sample
        target_size: Target size for resizing frames (square)
        
    Returns:
        List of PIL Images
    """
    if not _HAS_IMAGEIO:
        raise RuntimeError("imageio not available. pip install imageio imageio-ffmpeg")
    try:
        reader = imageio.get_reader(path)
    except Exception as e:
        raise FileNotFoundError(f"Could not open video with imageio: {path}\n{e}")

    try:
        meta = reader.get_meta_data()
        total = int(meta.get("nframes", 0))
    except Exception:
        total = 0
    frames = []
    if total and total > 0 and total != float("inf"):
        idxs = np.round(np.linspace(0, total - 1, num_frames)).astype(int)
        for i in idxs:
            try:
                fr = reader.get_data(int(i))
                frames.append(fr)
            except Exception:
                pass
    else:
        try:
            for fr in reader:
                frames.append(fr)
        except Exception:
            pass
        if not frames:
            reader.close()
            raise RuntimeError("No frames decoded from video (imageio)")
        idxs = np.round(np.linspace(0, len(frames) - 1, num_frames)).astype(int)
        frames = [frames[i] for i in idxs]
    reader.close()

    while len(frames) < num_frames and len(frames) > 0:
        frames.append(frames[-1])

    pil = []
    for fr in frames[:num_frames]:
        if fr.ndim == 2:
            fr = np.stack([fr, fr, fr], axis=-1)
        if fr.shape[-1] == 4:
            fr = fr[..., :3]
        fr = fr.astype(np.uint8)
        h, w, _ = fr.shape
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        fr = fr[y0:y0 + side, x0:x0 + side]
        if _HAS_CV2:
            fr = cv2.resize(fr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        else:
            # Fallback to PIL if cv2 not available
            pil_img = Image.fromarray(fr)
            pil_img = pil_img.resize((target_size, target_size), resample=Image.BILINEAR)
            fr = np.array(pil_img)
        pil.append(Image.fromarray(fr))
    return pil


def sample_video_frames(path: str, num_frames: int, target_size: int = 224) -> List[Image.Image]:
    """
    Sample frames from video with automatic fallback.
    
    Tries OpenCV first, then falls back to imageio if OpenCV fails.
    
    Args:
        path: Path to video file
        num_frames: Number of frames to sample
        target_size: Target size for resizing frames (square)
        
    Returns:
        List of PIL Images
    """
    try:
        return sample_video_frames_cv2(path, num_frames=num_frames, target_size=target_size)
    except Exception as e_cv:
        print(f"[decoder] OpenCV failed: {e_cv}. Trying imageio fallback…")
        return sample_video_frames_imageio(path, num_frames=num_frames, target_size=target_size)
