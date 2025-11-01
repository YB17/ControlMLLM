"""Utilities for working with COCO Panoptic annotations and masks.

This module provides helpers for loading Panoptic JSON indices, reading RGB images
and panoptic segmentation masks, converting masks to binary arrays, and preparing
attention heatmap visualisations.

The functions here are intentionally lightweight and can operate without optional
panopticapi dependency. When panopticapi is available, its ``rgb2id`` helper is used
for robustness; otherwise a pure NumPy fallback is used.

Example:
    >>> index = load_panoptic_index("/path/to/panoptic_val2017.json")
    >>> info = index["anns"][397133]
    >>> seg_map = load_panoptic_mask("/path/to/panoptic_val2017", info["file_name"])
    >>> mask = mask_from_segment_id(seg_map, info["segments_info"][0]["id"])

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image
from matplotlib import cm

try:  # pragma: no cover - optional dependency for fast resampling
    import cv2  # type: ignore
except Exception:  # pragma: no cover - fall back when OpenCV is unavailable
    cv2 = None

try:  # pragma: no cover - optional dependency
    from panopticapi.utils import rgb2id as _rgb2id
except Exception:  # pragma: no cover - absence of panopticapi
    _rgb2id = None


def _fallback_rgb2id(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB panoptic mask to integer ids using NumPy operations."""
    rgb = rgb.astype(np.int64)
    return rgb[..., 0] + (rgb[..., 1] << 8) + (rgb[..., 2] << 16)


def load_panoptic_index(panoptic_json: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Load COCO panoptic annotations index.

    Args:
        panoptic_json: Path to ``panoptic_{split}.json``.

    Returns:
        A dictionary containing image, annotation, and category metadata keyed by
        ``image_id`` or ``category_id`` for efficient lookup.
    """

    path = Path(panoptic_json)
    with path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)

    images = {int(img["id"]): {
        "file_name": img["file_name"],
        "height": int(img["height"]),
        "width": int(img["width"]),
    } for img in raw.get("images", [])}

    anns = {}
    for ann in raw.get("annotations", []):
        image_id = int(ann["image_id"])
        anns[image_id] = {
            "file_name": ann["file_name"],
            "segments_info": [
                {
                    "id": int(seg["id"]),
                    "category_id": int(seg["category_id"]),
                    "area": int(seg["area"]),
                    "bbox": list(seg["bbox"]),
                    "iscrowd": bool(seg.get("iscrowd", 0)),
                }
                for seg in ann.get("segments_info", [])
            ],
        }

    cats = {int(cat["id"]): {
        "name": cat.get("name", ""),
        "isthing": bool(cat.get("isthing", 0)),
        "color": list(cat.get("color", [0, 0, 0])),
    } for cat in raw.get("categories", [])}

    return {"images": images, "anns": anns, "cats": cats}


def load_image(img_root: str, img_file_name: str) -> Image.Image:
    """Load an RGB image from disk.

    Args:
        img_root: Root directory containing the COCO images.
        img_file_name: Relative file name of the target image.

    Returns:
        The loaded :class:`PIL.Image.Image` in RGB mode.
    """

    path = Path(img_root) / img_file_name
    with Image.open(path) as img:
        return img.convert("RGB")


def load_panoptic_mask(mask_root: str, mask_file_name: str) -> np.ndarray:
    """Load a panoptic segmentation PNG mask and convert to a segment id map."""

    path = Path(mask_root) / mask_file_name
    with Image.open(path) as img:
        rgb = np.array(img, dtype=np.uint8)

    if rgb.ndim == 2:  # greyscale fallback
        rgb = np.repeat(rgb[..., None], 3, axis=-1)

    if _rgb2id is not None:  # pragma: no branch - fast path when available
        seg_map = _rgb2id(rgb)
    else:
        seg_map = _fallback_rgb2id(rgb)
    return seg_map.astype(np.int64)


def mask_from_segment_id(seg_id_map: np.ndarray, target_sid: int) -> np.ndarray:
    """Return a binary mask for the given segment id."""

    mask = (seg_id_map == int(target_sid)).astype(np.uint8)
    return mask


def compute_mask_bbox(binary_mask: np.ndarray, padding: float = 0.05) -> Tuple[int, int, int, int]:
    """Compute a padded bounding box for a binary mask."""

    if binary_mask.ndim != 2:
        raise ValueError("Binary mask must be 2D")
    ys, xs = np.nonzero(binary_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Mask is empty")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    pad_x = int(round(width * padding))
    pad_y = int(round(height * padding))
    h, w = binary_mask.shape
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w - 1, x1 + pad_x)
    y1 = min(h - 1, y1 + pad_y)
    return x0, y0, x1, y1


def extract_padded_crop(image: Image.Image, bbox: Tuple[int, int, int, int], crop_size: int) -> Image.Image:
    """Extract and resize a crop defined by an inclusive bounding box."""

    if crop_size <= 0:
        raise ValueError("crop_size must be positive")
    x0, y0, x1, y1 = [int(v) for v in bbox]
    width, height = image.size
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0, min(x1, width - 1))
    y1 = max(y0, min(y1, height - 1))
    crop_box = (x0, y0, x1 + 1, y1 + 1)
    crop = image.crop(crop_box)
    if crop.size != (crop_size, crop_size):
        crop = crop.resize((crop_size, crop_size), Image.BICUBIC)
    return crop


def _resize_array(arr: np.ndarray, shape: tuple[int, int], mode: str) -> np.ndarray:
    if cv2 is not None:
        if mode == "down":
            return cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        return cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    pil_image = Image.fromarray(arr)
    resample = Image.BOX if mode == "down" else Image.BILINEAR
    resized = pil_image.resize((shape[1], shape[0]), resample=resample)
    return np.array(resized, dtype=np.float32)


def downsample_mask_to_tokens(binary_mask: np.ndarray, token_h: int, token_w: int) -> np.ndarray:
    """Downsample a binary mask to match the vision token grid using average pooling."""

    if token_h <= 0 or token_w <= 0:
        raise ValueError("token grid dimensions must be positive")

    binary = binary_mask.astype(np.float32)
    pooled = _resize_array(binary, (token_h, token_w), mode="down")
    pooled = pooled.astype(np.float32)
    if pooled.max() > 1.0:
        pooled = pooled / 255.0
    return np.clip(pooled, 0.0, 1.0)


def upsample_attn_to_image(attn_hw: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Upsample a token-level attention map back to image resolution."""

    attn = attn_hw.astype(np.float32)
    upsampled = _resize_array(attn, (out_h, out_w), mode="up")
    if upsampled.max() > 1.0:
        upsampled = upsampled / 255.0
    return upsampled


def draw_overlay_heatmap(rgb: Image.Image, heatmap01: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Overlay a heatmap onto the original RGB image and return a new PIL image."""

    img_array = np.array(rgb.convert("RGB"), dtype=np.float32) / 255.0
    hm = heatmap01.astype(np.float32)
    if hm.size == 0:
        hm = np.zeros_like(img_array[..., 0])
    hm = hm - hm.min()
    max_val = hm.max()
    if max_val > 0:
        hm = hm / max_val
    cmap = cm.get_cmap("jet")
    heat_rgb = cmap(np.clip(hm, 0.0, 1.0))[..., :3]
    overlay = (1 - alpha) * img_array + alpha * heat_rgb
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


__all__ = [
    "load_panoptic_index",
    "load_image",
    "load_panoptic_mask",
    "mask_from_segment_id",
    "compute_mask_bbox",
    "extract_padded_crop",
    "downsample_mask_to_tokens",
    "upsample_attn_to_image",
    "draw_overlay_heatmap",
]
