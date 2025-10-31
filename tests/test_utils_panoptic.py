"""Unit tests for the panoptic utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ROC_DIR = ROOT / "controlmllm++" / "qwen2_5_vl" / "roc"
if str(ROC_DIR) not in sys.path:
    sys.path.insert(0, str(ROC_DIR))

import utils_panoptic as up  # type: ignore


def test_rgb2id_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(up, "_rgb2id", None, raising=False)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 0] = 1
    rgb[..., 1] = np.arange(4, dtype=np.uint8)
    rgb[..., 2] = np.arange(4, dtype=np.uint8)[::-1]
    Image.fromarray(rgb).save(tmp_path / "mask.png")

    seg_map = up.load_panoptic_mask(str(tmp_path), "mask.png")
    expected = rgb[..., 0].astype(np.int64) + (rgb[..., 1].astype(np.int64) << 8) + (rgb[..., 2].astype(np.int64) << 16)
    assert seg_map.shape == (4, 4)
    assert np.array_equal(seg_map, expected)


def test_downsample_and_upsample_shapes():
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[2:5, 1:6] = 1
    tokens = up.downsample_mask_to_tokens(mask, 3, 4)
    assert tokens.shape == (3, 4)
    assert float(tokens.min()) >= 0.0 and float(tokens.max()) <= 1.0

    restored = up.upsample_attn_to_image(tokens, 6, 8)
    assert restored.shape == (6, 8)
    assert float(restored.min()) >= 0.0
    assert float(restored.max()) <= 1.0
