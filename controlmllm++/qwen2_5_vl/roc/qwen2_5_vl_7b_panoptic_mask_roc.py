"""Panoptic mask-driven ROC pipeline for Qwen2.5-VL-7B.

This script extends the bounding-box ROC workflow to COCO Panoptic masks. It reads
Panoptic annotations, selects segments, formulates mask-aware questions, optimises
visual prompt tokens with the same activation objective used in ``qwen2_5_vl_7b_roc.py``,
and stores visualisations and metrics for further analysis.

Example usage::

    python qwen2_5_vl_7b_panoptic_mask_roc.py \
      --coco-img-root /data/coco/val2017 \
      --panoptic-json /data/coco/annotations/panoptic_val2017.json \
      --panoptic-mask-root /data/coco/annotations/panoptic_val2017 \
      --sample-image-ids 397133 \
      --layers 20,24 \
      --heads 0,1,2,3,4,5,6,7 \
      --steps 5 --lr 0.02 \
      --hires-crop-size 448 --save-all

Definition of Done
------------------
* 左上角热点消失，且 KV 切片日志输出满足 ``end - start == h_tok * w_tok``
* Query 集合非空，``summary.jsonl`` 中记录的比率 ``ratio_after`` 普遍高于 ``ratio_before``
* CLI 示例可直接运行并输出 ``kv_slice_meta.json``、前/后 overlay、token 灰度图
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info
from qwen_utils import get_grid_shape
from .utils_panoptic import (
    compute_mask_bbox,
    draw_overlay_heatmap,
    downsample_mask_to_tokens,
    extract_padded_crop,
    load_image,
    load_panoptic_index,
    load_panoptic_mask,
    mask_from_segment_id,
    upsample_attn_to_image,
)

try:  # pragma: no cover - optional import for text mask refinement
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

import qwen2_5_vl_7b_roc as roc_base  # type: ignore

DEVICE = torch.device(getattr(roc_base, "device", "cuda" if torch.cuda.is_available() else "cpu"))
compute_activation_loss = getattr(roc_base, "compute_activation_loss_qwen", None)
if compute_activation_loss is None:
    from qwen_utils import compute_activation_loss_qwen as compute_activation_loss  # type: ignore
get_grid_shape_fn = getattr(roc_base, "get_grid_shape", get_grid_shape)
EPS = 1e-3
TEXT_KEYWORDS = {"text", "word", "letter", "character", "sign", "label"}


LOGGER = logging.getLogger(__name__)
KV_SLICE_LOGGED = False

VERBALIZER_MAP: Dict[str, List[str]] = {
    "cabinet": ["cabinet", "cupboard"],
    "oven": ["oven", "stove"],
    "sink": ["sink", "basin"],
    "cup": ["cup", "mug"],
    "toilet": ["toilet", "commode"],
    "couch": ["sofa", "couch"],
    "traffic light": ["traffic", "trafficlight"],
    "tv": ["television", "tv"],

    "wall-brick": ["brick wall"],
    "wall-stone": ["stone wall"],
    "wall-tile": ["tiled wall"],
    "wall-wood": ["wooden wall"],

    "water-other": ["water"],
    "window-blind": ["window blinds"],
    "window-other": ["window"],

    "tree-merged": ["trees"],
    "fence-merged": ["fence"],
    "ceiling-merged": ["ceiling"],
    "sky-other-merged": ["sky"],
    "cabinet-merged": ["cabinet"],
    "table-merged": ["table"],
    "floor-other-merged": ["floor"],
    "pavement-merged": ["pavement"],
    "mountain-merged": ["mountains"],
    "grass-merged": ["grass"],
    "dirt-merged": ["dirt ground"],
    "paper-merged": ["paper"],
    "food-other-merged": ["food"],
    "building-other-merged": ["building"],
    "rock-merged": ["rocks"],
    "wall-other-merged": ["wall"],
    "rug-merged": ["rug"],

    "mirror-stuff": ["mirror"],
    "door-stuff": ["door"],
    "floor-wood": ["wooden floor"],

}


@dataclass
class InstanceRecord:
    image_id: int
    segment_id: int
    category_id: int
    category_name: str
    prompt: str
    answer: str
    ratio_before: float
    ratio_after: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Panoptic mask ROC evaluation")
    parser.add_argument("--model-path", default="/home/host/qwen2.5-vl")
    parser.add_argument("--coco-img-root", required=True, help="COCO image root directory")
    parser.add_argument("--panoptic-json", required=True, help="Path to panoptic JSON file")
    parser.add_argument("--panoptic-mask-root", required=True, help="Directory with panoptic PNG masks")
    parser.add_argument("--sample-image-ids", default="", help="Comma separated list of image ids to process")
    parser.add_argument("--n-samples", type=int, default=5, help="Number of images to sample when ids not provided")
    parser.add_argument("--min-area", type=int, default=4096, help="Skip segments with area smaller than this")
    parser.add_argument("--category-filter", default="", help="Comma separated category names or ids to keep")
    parser.add_argument("--outdir", default="outputs/panoptic_mask", help="Directory for saving outputs")
    parser.add_argument("--layers", default="11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27", help="Decoder layer indices to aggregate (comma separated or 'all')")
    parser.add_argument("--heads", default="0,1,2,3,4,5,6,7", help="Attention head indices to aggregate (comma separated or 'all')")
    parser.add_argument("--steps", type=int, default=5, help="Number of visual prompt optimisation steps")
    parser.add_argument("--lr", type=float, default=0.02, help="Learning rate for visual prompt updates")
    parser.add_argument("--alpha", type=float, default=400.0, help="Scaling factor for mask optimisation loss")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Maximum tokens to generate for answers")
    parser.add_argument("--save-all", action="store_true", help="Save both pre/post overlays and token grids")
    parser.add_argument("--text-mask-mode", action="store_true", help="Enable text-specific mask refinement and exports")
    parser.add_argument("--hires-crop-size", type=int, default=0, help="Enable high resolution branch with the given crop size (0 disables)")
    parser.add_argument("--small-mask-thr", type=int, default=0, help="Only enable high resolution branch when mask area is smaller than this (0 disables filtering)")
    parser.add_argument(
        "--hires-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When enabled (default), prefer the high-resolution vision span whenever available.",
    )
    parser.add_argument(
        "--span-idx",
        type=int,
        default=None,
        help="Optional explicit vision span index to use when --no-hires-only is set.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_index_list(spec: str, upper: int) -> List[int]:
    spec = (spec or "").strip()
    if not spec or spec.lower() == "all":
        return list(range(upper))
    values: List[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            idx = int(token)
        except ValueError as exc:  # pragma: no cover - validated by CLI
            raise ValueError(f"Invalid index '{token}' in specification '{spec}'") from exc
        if idx < 0 or idx >= upper:
            raise ValueError(f"Index {idx} out of range [0, {upper})")
        values.append(idx)
    return sorted(set(values))


def parse_category_filter(filter_str: str) -> set[str]:
    tokens = set()
    for token in (filter_str or "").split(","):
        tok = token.strip()
        if not tok:
            continue
        if tok.isdigit():
            tokens.add(tok)
        else:
            tokens.add(tok.lower())
    return tokens


def slugify(value: str) -> str:
    sanitized = [ch if ch.isalnum() else "_" for ch in value]
    name = "".join(sanitized).strip("_")
    return name or "unknown"


def normalise_category_label(name: str) -> str:
    """Normalise raw panoptic category names into readable labels."""

    if not name:
        return ""
    lowered = name.lower().strip()
    for suffix in ("_merged", "_stuff", "_thing"):
        if lowered.endswith(suffix):
            lowered = lowered[: -len(suffix)]
            break
    lowered = lowered.replace("_", " ").replace("-", " ")
    lowered = " ".join(lowered.split())
    return lowered


def get_verbalizer_options(cat_name: Optional[str]) -> Tuple[str, List[str]]:
    """Return normalised category label and candidate verbaliser options."""

    normalised = normalise_category_label(cat_name or "")
    if not normalised:
        return "", []
    options = VERBALIZER_MAP.get(normalised, None)
    if options is None:
        options = [normalised]
    else:
        options = list(dict.fromkeys(options + [normalised]))
    fallback_no_space = normalised.replace(" ", "")
    if fallback_no_space and fallback_no_space not in options:
        options.append(fallback_no_space)
    return normalised, options


def match_answer_to_options(answer: str, options: Sequence[str]) -> str:
    """Match a model answer against verbaliser options using simple heuristics."""

    cleaned = "".join(ch for ch in answer.lower().strip() if ch.isalnum() or ch.isspace())
    tokens = cleaned.split()
    if not tokens:
        return answer.strip()
    candidate = tokens[0]
    for option in options:
        opt_lower = option.lower()
        variants = {opt_lower, opt_lower.replace(" ", ""), opt_lower.rstrip("s")}
        if any(candidate.startswith(var) or var.startswith(candidate) for var in variants if var):
            return option
    return candidate


def build_prompt(cat_name: Optional[str]) -> Tuple[str, List[str], str]:
    """Construct a single-word prompt and return verbaliser options and label."""

    normalised, options = get_verbalizer_options(cat_name)
    if options:
        choices = ", ".join(sorted(dict.fromkeys(options)))
        prompt = (
            "Focus only on the object outlined by the provided mask. Note that the object might be either for-ground related, e.g. an apple, a person or a car, or back-ground related, e.g. trees, sky, walls, ceiling. Think carefully before you answer the question: Choose one label "
            f"from {{{choices}}}. Answer with exactly one word."
        )
    else:
        prompt = (
            "Focus only on the object outlined by the provided mask. Note that the object might be either for-ground related, e.g. an apple, a person or a car, or back-ground related, e.g. trees, sky, walls, ceiling. Think carefully before you answer the question: What is the object? "
            "Answer with a single noun."
        )
    return prompt, options, normalised


def is_text_category(name: str) -> bool:
    """Return ``True`` if the provided category name is text-like."""

    lowered = (name or "").lower()
    return any(keyword in lowered for keyword in TEXT_KEYWORDS)


def refine_text_mask(mask: np.ndarray) -> np.ndarray:
    """Apply light morphology to stabilise text masks when OpenCV is available."""

    if cv2 is None:
        return mask
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    return (dilated > 0).astype(np.uint8)


def extract_padded_crop_mask(
    binary_mask: np.ndarray, bbox: Tuple[int, int, int, int], out_size: int
) -> np.ndarray:
    """Crop and resize a binary mask using the same bbox/padding as the high-res branch."""

    if out_size <= 0:
        raise ValueError("Output size for cropped mask must be positive")
    x0, y0, x1, y1 = [int(v) for v in bbox]
    height, width = binary_mask.shape
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0, min(x1, width - 1))
    y1 = max(y0, min(y1, height - 1))
    crop = binary_mask[y0 : y1 + 1, x0 : x1 + 1]
    pil_mask = Image.fromarray((crop > 0).astype(np.uint8) * 255)
    if pil_mask.size != (out_size, out_size):
        pil_mask = pil_mask.resize((out_size, out_size), Image.NEAREST)
    resized = np.array(pil_mask, dtype=np.uint8)
    return (resized > 127).astype(np.float32)


def get_text_query_indices(input_ids: List[int], special_ids: Set[int], vision_start: int, vision_end: int) -> List[int]:
    """Select textual query indices while excluding specials and the vision span."""

    indices = [
        idx
        for idx, token_id in enumerate(input_ids)
        if (idx < vision_start or idx >= vision_end) and token_id not in special_ids
    ]
    if not indices:
        fallback = max(0, vision_end - 1)
        indices.append(fallback)
    return indices


def compute_token_attention(
    attn_list: Sequence[torch.Tensor],
    layer_indices: Sequence[int],
    head_indices: Sequence[int],
    query_indices: Sequence[int],
    vision_start: int,
    vision_end: int,
    grid_shape: Tuple[int, int],
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Aggregate attentions over layers, heads, and textual queries.

    Args:
        attn_list: Sequence of attention tensors ``[(B, H, Q, K), ...]``.
        layer_indices: Decoder layer indices to include.
        head_indices: Optional attention head indices; averages over all heads when empty.
        query_indices: Iterable of textual token indices used as query positions.
        vision_start: Index of the first vision token.
        vision_end: Index immediately after the last vision token.
        grid_shape: Tuple ``(H_tok, W_tok)`` describing the vision token grid.

    Returns:
        A tuple of ``(token_att, meta)`` where ``token_att`` has shape ``(grid_len,)``
        and ``meta`` contains the KV slicing diagnostics for logging and saving.
    """

    if not attn_list:
        raise ValueError("Attention list is empty")
    if not layer_indices:
        raise ValueError("No layer indices provided")
    if not query_indices:
        raise ValueError("No query indices provided")

    selected_layers: List[torch.Tensor] = []
    for idx in layer_indices:
        if idx < 0 or idx >= len(attn_list):
            raise ValueError(f"Layer index {idx} out of range for {len(attn_list)} layers")
        layer_attn = attn_list[idx].to(DEVICE)
        selected_layers.append(layer_attn.unsqueeze(0))
    stacked = torch.cat(selected_layers, dim=0)  # (L, B, H, Q, K)
    layer_mean = stacked.mean(dim=0)  # (B, H, Q, K)

    if head_indices:
        head_tensor = layer_mean[:, head_indices, :, :]
    else:
        head_tensor = layer_mean
    head_mean = head_tensor.mean(dim=1)  # (B, Q, K)

    query_idx_tensor = torch.tensor(query_indices, device=head_mean.device, dtype=torch.long)
    query_selected = torch.index_select(head_mean, dim=1, index=query_idx_tensor)
    query_mean = query_selected.mean(dim=1)  # (B, K)

    if query_mean.shape[0] != 1:
        raise ValueError(f"Expected batch size 1, got {query_mean.shape[0]}")

    kv_len = vision_end - vision_start
    h_tok, w_tok = grid_shape
    grid_len = int(h_tok * w_tok)
    if grid_len <= 0:
        raise RuntimeError(f"Invalid grid length {grid_len} from grid shape {grid_shape}")
    start = vision_start + max(0, kv_len - grid_len)
    end = start + grid_len
    if end - start != grid_len or end > vision_end:
        raise RuntimeError(
            f"KV slicing mismatch: grid_len={grid_len}, kv_len={kv_len}, start={start}, end={end}, vision_end={vision_end}"
        )

    slice_meta: Dict[str, object] = {
        "vision_start": int(vision_start),
        "vision_end": int(vision_end),
        "kv_start": int(start),
        "kv_end": int(end),
        "kv_len": int(kv_len),
        "grid_len": int(grid_len),
        "grid_shape": [int(h_tok), int(w_tok)],
        "layers": [int(idx) for idx in layer_indices],
        "heads": [int(idx) for idx in head_indices],
        "query_indices": [int(idx) for idx in query_indices],
    }

    global KV_SLICE_LOGGED
    if not KV_SLICE_LOGGED:
        LOGGER.info(
            "KV slice stats: kv_len=%d grid_len=%d start=%d end=%d",
            slice_meta["kv_len"],
            slice_meta["grid_len"],
            slice_meta["kv_start"],
            slice_meta["kv_end"],
        )
        KV_SLICE_LOGGED = True

    token_slice = query_mean[0, start:end].contiguous()
    return token_slice, slice_meta


def token_attention_to_map(token_att: torch.Tensor, grid_shape: Tuple[int, int]) -> torch.Tensor:
    """Reshape a flattened token attention vector back to ``(H_tok, W_tok)``."""

    h, w = grid_shape
    return token_att.reshape(h, w)


def compute_attention_ratio(
    token_att: torch.Tensor, mask_tok: torch.Tensor, grid_shape: Tuple[int, int], eps: float = 1e-6
) -> float:
    """Compute the fraction of attention mass falling inside the mask tokens."""

    attn_map = token_attention_to_map(token_att, grid_shape)
    mask = mask_tok.float()
    numerator = float(torch.sum(attn_map * mask).item())
    denominator = float(torch.sum(attn_map).item())
    ratio = (numerator + eps) / (denominator + eps)
    return float(ratio)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_image_safe(image: Image.Image, path: Path) -> None:
    try:
        ensure_dir(path.parent)
        image.save(path)
    except Exception as exc:  # pragma: no cover - IO guard
        LOGGER.warning("Failed to save image %s: %s", path, exc)


def save_token_map(attn_map: np.ndarray, path: Path) -> None:
    try:
        ensure_dir(path.parent)
        norm = attn_map - attn_map.min()
        denom = norm.max()
        if denom > 0:
            norm = norm / denom
        Image.fromarray((norm * 255).astype(np.uint8)).save(path)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Failed to save token map %s: %s", path, exc)


def append_summary(path: Path, record: InstanceRecord) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps({
            "image_id": record.image_id,
            "segment_id": record.segment_id,
            "category_id": record.category_id,
            "category_name": record.category_name,
            "prompt": record.prompt,
            "answer": record.answer,
            "ratio_before": record.ratio_before,
            "ratio_after": record.ratio_after,
        }) + "\n")


def prepare_msgs(
    image_path: str, prompt: str, highres_patch: Optional[Image.Image] = None
) -> List[Dict[str, object]]:
    """Prepare multimodal chat messages for the processor."""

    content: List[Dict[str, object]] = [
        {"type": "image", "image": image_path, "max_pixels": 512 * 28 * 28},
    ]
    if highres_patch is not None:
        content.append({"type": "image", "image": highres_patch})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def prepare_inputs(
    processor: AutoProcessor, msgs: List[Dict[str, object]]
) -> Tuple[
    Dict[str, torch.Tensor],
    List[Tuple[int, int]],
    List[Tuple[int, int]],
    List[int],
    Set[int],
]:
    """Tokenise inputs and extract metadata for all vision token spans."""

    image_inputs, _ = process_vision_info(msgs)
    grid_shapes_per_image: List[Tuple[int, int]] = []
    for idx, image in enumerate(image_inputs):
        grid_shape = get_grid_shape_fn(processor, [image])
        grid_shapes_per_image.append(grid_shape)
        LOGGER.debug("Grid shape for image %d: %s", idx, grid_shape)
    text_prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=image_inputs, padding=True, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    tokenizer = processor.tokenizer
    vision_start_token_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_token_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    input_ids = inputs["input_ids"][0].tolist()
    if vision_start_token_id not in input_ids or vision_end_token_id not in input_ids:
        raise ValueError("Vision boundary tokens not found in the input sequence")

    starts: List[int] = []
    ends: List[int] = []
    for idx, token_id in enumerate(input_ids):
        if token_id == vision_start_token_id:
            starts.append(idx + 1)
        elif token_id == vision_end_token_id:
            ends.append(idx)
    if len(starts) != len(ends):
        raise ValueError(
            f"Mismatch between vision start ({len(starts)}) and end ({len(ends)}) tokens"
        )
    if len(starts) != len(image_inputs):
        raise ValueError(
            "Number of vision spans does not match number of image inputs"
        )
    vision_spans: List[Tuple[int, int]] = []
    end_cursor = 0
    for start in starts:
        while end_cursor < len(ends) and ends[end_cursor] < start:
            end_cursor += 1
        if end_cursor >= len(ends):
            raise ValueError("Unable to match vision start token with an end token")
        end = ends[end_cursor]
        end_cursor += 1
        if end <= start:
            raise ValueError(f"Invalid vision span ({start}, {end})")
        vision_spans.append((start, end))

    special_ids = set(tokenizer.all_special_ids or [])
    LOGGER.info(
        "Input sequence prepared: len=%d span_count=%d spans=%s grid_shapes=%s",
        len(input_ids),
        len(vision_spans),
        vision_spans,
        grid_shapes_per_image,
    )
    if len(vision_spans) != len(grid_shapes_per_image):
        raise ValueError(
            "Vision span count does not match grid shape count from image inputs"
        )
    return inputs, grid_shapes_per_image, vision_spans, input_ids, special_ids


def select_images(index: Dict[str, Dict[int, Dict[str, object]]], image_ids_arg: str, n_samples: int) -> List[int]:
    available_ids = sorted(index["images"].keys())
    if image_ids_arg.strip():
        ids = []
        for tok in image_ids_arg.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                ids.append(int(tok))
            except ValueError as exc:
                raise ValueError(f"Invalid image id '{tok}'") from exc
        return ids
    if n_samples >= len(available_ids):
        return available_ids
    return random.sample(available_ids, n_samples)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    LOGGER.info("Initialising Panoptic ROC on device %s", DEVICE)
    set_seed(args.seed)

    index = load_panoptic_index(args.panoptic_json)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        attn_implementation="eager",
        device_map="auto" if DEVICE.type == "cuda" else None,
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side="left",
        use_fast=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    layer_indices = parse_index_list(args.layers, model.config.num_hidden_layers)
    head_indices = parse_index_list(args.heads, model.config.num_attention_heads)
    category_filter = parse_category_filter(args.category_filter)

    chosen_images = select_images(index, args.sample_image_ids, args.n_samples)

    total_instances = 0
    processed_instances = 0
    ratio_gains: List[float] = []

    for image_id in tqdm(chosen_images, desc="Panoptic-ROC"):
        image_meta = index["images"].get(image_id)
        ann_meta = index["anns"].get(image_id)
        if image_meta is None or ann_meta is None:
            LOGGER.warning("Missing metadata for image %s", image_id)
            continue

        img_path = os.path.join(args.coco_img_root, image_meta["file_name"])
        try:
            image = load_image(args.coco_img_root, image_meta["file_name"])
        except FileNotFoundError:
            LOGGER.warning("Image file not found: %s", img_path)
            continue

        try:
            seg_map = load_panoptic_mask(args.panoptic_mask_root, ann_meta["file_name"])
        except FileNotFoundError:
            LOGGER.warning("Mask file not found for image %s", image_id)
            continue

        out_dir = Path(args.outdir) / str(image_id)
        ensure_dir(out_dir)
        summary_path = out_dir / "summary.jsonl"

        for seg_info in ann_meta.get("segments_info", []):
            total_instances += 1
            area = int(seg_info.get("area", 0))
            if area < args.min_area:
                continue

            seg_id = int(seg_info["id"])
            cat_id = int(seg_info["category_id"])
            cat_meta = index["cats"].get(cat_id, {})
            cat_name = cat_meta.get("name", "")
            cat_slug = slugify(cat_name) if cat_name else f"cat{cat_id}"

            if category_filter:
                if str(cat_id) not in category_filter and cat_name.lower() not in category_filter:
                    continue

            binary_mask = mask_from_segment_id(seg_map, seg_id)
            if args.text_mask_mode and is_text_category(cat_name):
                binary_mask = refine_text_mask(binary_mask)

            if binary_mask.sum() == 0:
                LOGGER.debug("Skipping empty mask for image %s seg %s", image_id, seg_id)
                continue

            prompt, verbalizer_options, normalised_label = build_prompt(cat_name)
            LOGGER.info("Prompt preview: %s", prompt[:120])
            # if normalised_label in {"cabinet", "oven", "sink", "cup"}:
            #     LOGGER.info(
            #         "Verbalizer candidates for %s (%s): %s",
            #         cat_name or f"cat{cat_id}",
            #         normalised_label or "unknown",
            #         verbalizer_options,
            #     )

            highres_patch: Optional[Image.Image] = None
            enable_hires = False
            bbox: Optional[Tuple[int, int, int, int]] = None
            if args.hires_crop_size > 0 and (args.small_mask_thr <= 0 or area < args.small_mask_thr):
                try:
                    bbox = compute_mask_bbox(binary_mask, padding=0.05)
                    highres_patch = extract_padded_crop(image, bbox, args.hires_crop_size)
                    enable_hires = True
                except ValueError as exc:
                    LOGGER.warning(
                        "High-res crop failed for image %s seg %s: %s",
                        image_id,
                        seg_id,
                        exc,
                    )
            if enable_hires and highres_patch is not None:
                LOGGER.info(
                    "High-res branch enabled for image %s seg %s: area=%d crop_size=%s",
                    image_id,
                    seg_id,
                    area,
                    highres_patch.size,
                )

            msgs = prepare_msgs(img_path, prompt, highres_patch)
            try:
                inputs, grid_shapes, vision_spans, input_ids, special_ids = prepare_inputs(processor, msgs)
            except ValueError as exc:
                LOGGER.warning("Failed to prepare inputs for image %s seg %s: %s", image_id, seg_id, exc)
                continue

            if not vision_spans:
                LOGGER.warning("No vision spans detected for image %s seg %s", image_id, seg_id)
                continue

            span_count = len(vision_spans)
            if len(grid_shapes) != span_count:
                LOGGER.error(
                    "Span count %d does not match grid shape count %d for image %s seg %s",
                    span_count,
                    len(grid_shapes),
                    image_id,
                    seg_id,
                )
                continue

            vision_min_start = min(span[0] for span in vision_spans)
            vision_max_end = max(span[1] for span in vision_spans)
            query_indices = get_text_query_indices(input_ids, special_ids, vision_min_start, vision_max_end)
            head_sample = query_indices[:3]
            tail_sample = query_indices[-3:] if len(query_indices) > 3 else query_indices[:]
            LOGGER.info(
                "Query indices count=%d head=%s tail=%s",
                len(query_indices),
                head_sample,
                tail_sample,
            )

            use_highres_span = enable_hires and highres_patch is not None and span_count >= 2
            if args.span_idx is not None:
                if args.span_idx < 0 or args.span_idx >= span_count:
                    LOGGER.error(
                        "Requested span_idx %s out of bounds for %d spans", args.span_idx, span_count
                    )
                    continue
                span_idx = args.span_idx
            elif use_highres_span and args.hires_only:
                span_idx = 1
            elif use_highres_span:
                span_idx = 1
            else:
                span_idx = 0

            if span_idx >= span_count:
                LOGGER.warning(
                    "Adjusted span index %d exceeds available spans %d for image %s seg %s",
                    span_idx,
                    span_count,
                    image_id,
                    seg_id,
                )
                span_idx = span_count - 1

            vision_start, vision_end = vision_spans[span_idx]
            grid_shape = grid_shapes[span_idx]
            h_tok, w_tok = grid_shape
            if vision_end - vision_start <= 0 or h_tok * w_tok <= 0:
                LOGGER.warning(
                    "Invalid vision span or grid shape for image %s seg %s: span=%s grid=%s",
                    image_id,
                    seg_id,
                    vision_spans[span_idx],
                    grid_shape,
                )
                continue

            LOGGER.info(
                "Chosen vision span idx=%d/%d span=%s grid_shape=%s",
                span_idx,
                span_count,
                vision_spans[span_idx],
                grid_shape,
            )

            if use_highres_span and span_idx == 1 and bbox is not None and args.hires_crop_size > 0:
                try:
                    mask_patch = extract_padded_crop_mask(binary_mask, bbox, args.hires_crop_size)
                    mask_tokens = downsample_mask_to_tokens(mask_patch, h_tok, w_tok)
                except ValueError as exc:
                    LOGGER.warning(
                        "Failed to align high-res mask for image %s seg %s: %s; falling back to full mask",
                        image_id,
                        seg_id,
                        exc,
                    )
                    mask_tokens = downsample_mask_to_tokens(binary_mask, h_tok, w_tok)
            else:
                mask_tokens = downsample_mask_to_tokens(binary_mask, h_tok, w_tok)
            if mask_tokens.sum() <= 0:
                LOGGER.debug("Token mask empty after downsampling for image %s seg %s", image_id, seg_id)
                continue
            mask_tok_tensor = torch.from_numpy(mask_tokens).to(DEVICE).float()

            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)
            token_att_pre, kv_meta = compute_token_attention(
                outputs.attentions,
                layer_indices,
                head_indices,
                query_indices,
                vision_start,
                vision_end,
                grid_shape,
            )
            ratio_before = compute_attention_ratio(token_att_pre, mask_tok_tensor, grid_shape)

            # num_tokens = h_tok * w_tok
            num_tokens = vision_end - vision_start  # 使用实际的vision token范围
            # model.visual_prompt = torch.nn.Parameter(torch.zeros((num_tokens, model.config.hidden_size), dtype=model.dtype, device=DEVICE))
            # 使用inputs所在的设备（主模型设备）
            inputs_device = inputs['input_ids'].device
            model.visual_prompt = torch.nn.Parameter(
                torch.zeros((num_tokens, model.config.hidden_size), 
                            dtype=model.dtype, device=inputs_device)
            )
            beta1, beta2, eps = 0.9, 0.999, EPS
            hyperparams = {'lr': args.lr, 't': 1}
            state = {
                'm': torch.zeros_like(model.visual_prompt),
                's': torch.zeros_like(model.visual_prompt),
            }

            for _ in range(args.steps):
                outputs = model(**inputs, output_attentions=True)
                token_att, _ = compute_token_attention(
                    outputs.attentions,
                    layer_indices,
                    head_indices,
                    query_indices,
                    vision_start,
                    vision_end,
                    grid_shape,
                )
                loss = args.alpha * compute_activation_loss(token_att.unsqueeze(0), [mask_tok_tensor])
                grad = torch.autograd.grad(loss, model.visual_prompt, retain_graph=False)[0]
                state['m'] = beta1 * state['m'] + (1 - beta1) * grad
                state['s'] = beta2 * state['s'] + (1 - beta2) * grad.pow(2)
                m_hat = state['m'] / (1 - beta1 ** hyperparams['t'])
                s_hat = state['s'] / (1 - beta2 ** hyperparams['t'])
                with torch.no_grad():
                    model.visual_prompt.data = model.visual_prompt.data - hyperparams['lr'] * m_hat / (torch.sqrt(s_hat) + eps)
                hyperparams['t'] += 1
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

            with torch.no_grad():
                outputs_post = model(**inputs, output_attentions=True)
                token_att_post, _ = compute_token_attention(
                    outputs_post.attentions,
                    layer_indices,
                    head_indices,
                    query_indices,
                    vision_start,
                    vision_end,
                    grid_shape,
                )
                ratio_after = compute_attention_ratio(token_att_post, mask_tok_tensor, grid_shape)
                # 临时保存并移除visual_prompt以避免generate时的设备冲突
                saved_prompt = model.visual_prompt
                model.visual_prompt = None
                generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
                model.visual_prompt = saved_prompt  # 恢复
                trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
                decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                raw_answer = decoded[0] if decoded else ""

            LOGGER.info("Raw answer: %s", raw_answer.strip())
            parsed_answer = match_answer_to_options(raw_answer, verbalizer_options)
            LOGGER.info(
                "Attention ratio %.3f -> %.3f (Δ=%.3f) | parsed=%s",
                ratio_before,
                ratio_after,
                ratio_after - ratio_before,
                parsed_answer,
            )

            ratio_gains.append(ratio_after - ratio_before)
            processed_instances += 1

            attn_pre_np = token_attention_to_map(token_att_pre.detach(), grid_shape).cpu().numpy()
            attn_post_np = token_attention_to_map(token_att_post.detach(), grid_shape).cpu().numpy()

            full_heat_pre: np.ndarray
            full_heat_post: np.ndarray
            if use_highres_span and span_idx == 1 and bbox is not None:
                x0, y0, x1, y1 = bbox
                bbox_w = max(1, int(x1 - x0 + 1))
                bbox_h = max(1, int(y1 - y0 + 1))
                full_heat_pre = np.zeros((image.height, image.width), dtype=np.float32)
                full_heat_post = np.zeros_like(full_heat_pre)
                pre_roi = upsample_attn_to_image(attn_pre_np, bbox_h, bbox_w)
                post_roi = upsample_attn_to_image(attn_post_np, bbox_h, bbox_w)
                full_heat_pre[y0 : y1 + 1, x0 : x1 + 1] = pre_roi
                full_heat_post[y0 : y1 + 1, x0 : x1 + 1] = post_roi
            else:
                full_heat_pre = upsample_attn_to_image(attn_pre_np, image.height, image.width)
                full_heat_post = upsample_attn_to_image(attn_post_np, image.height, image.width)

            overlay_pre = draw_overlay_heatmap(image, full_heat_pre)
            overlay_post = draw_overlay_heatmap(image, full_heat_post)

            prefix = f"{seg_id}_{cat_slug}"
            save_image_safe(overlay_pre, out_dir / f"{prefix}_pre_overlay.png")
            save_image_safe(overlay_post, out_dir / f"{prefix}_post_overlay.png")
            if args.save_all:
                save_token_map(attn_pre_np, out_dir / f"{prefix}_pre_token.png")
                save_token_map(attn_post_np, out_dir / f"{prefix}_post_token.png")

            if use_highres_span and span_idx == 1 and highres_patch is not None and args.hires_crop_size > 0:
                crop_heat_pre = upsample_attn_to_image(attn_pre_np, args.hires_crop_size, args.hires_crop_size)
                crop_heat_post = upsample_attn_to_image(attn_post_np, args.hires_crop_size, args.hires_crop_size)
                overlay_crop_pre = draw_overlay_heatmap(highres_patch, crop_heat_pre)
                overlay_crop_post = draw_overlay_heatmap(highres_patch, crop_heat_post)
                save_image_safe(overlay_crop_pre, out_dir / f"{prefix}_crop_pre_overlay.png")
                save_image_safe(overlay_crop_post, out_dir / f"{prefix}_crop_post_overlay.png")

            npz_path = out_dir / f"{prefix}_attn.npz"
            try:
                ensure_dir(npz_path.parent)
                np.savez_compressed(npz_path, attn_pre=attn_pre_np, attn_post=attn_post_np, mask_tokens=mask_tokens)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Failed to save attention arrays for %s: %s", npz_path, exc)

            if args.text_mask_mode and is_text_category(cat_name):
                text_mask = refine_text_mask(binary_mask)
                text_overlay = draw_overlay_heatmap(image, upsample_attn_to_image(text_mask.astype(np.float32), image.height, image.width))
                save_image_safe(text_overlay, out_dir / f"{prefix}_text_overlay.png")

            kv_meta_path = out_dir / f"{prefix}_kv_slice_meta.json"
            kv_meta_out = dict(kv_meta)
            kv_meta_out.update({
                "ratio_before": ratio_before,
                "ratio_after": ratio_after,
                "span_idx": int(span_idx),
                "span_count": int(span_count),
                "spans": [[int(s), int(e)] for (s, e) in vision_spans],
                "grid_shapes_per_image": [[int(h), int(w)] for (h, w) in grid_shapes],
                "chosen_grid_shape": [int(h_tok), int(w_tok)],
            })
            try:
                ensure_dir(kv_meta_path.parent)
                with kv_meta_path.open("w", encoding="utf-8") as fp:
                    json.dump(kv_meta_out, fp, indent=2)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Failed to save KV metadata for %s: %s", kv_meta_path, exc)

            record = InstanceRecord(
                image_id=image_id,
                segment_id=seg_id,
                category_id=cat_id,
                category_name=cat_name,
                prompt=prompt,
                answer=parsed_answer,
                ratio_before=ratio_before,
                ratio_after=ratio_after,
            )
            append_summary(summary_path, record)

            LOGGER.info(
                "image %s seg %s (%s) ratio %.3f -> %.3f | %s",
                image_id,
                seg_id,
                cat_slug,
                ratio_before,
                ratio_after,
                parsed_answer,
            )

    if processed_instances:
        gains = np.array(ratio_gains, dtype=np.float32)
        mean_gain = float(gains.mean())
        median_gain = float(np.median(gains))
        positive_rate = float((gains > 0).mean())
        LOGGER.info("=== Summary ===")
        LOGGER.info("Total instances: %d", total_instances)
        LOGGER.info("Processed instances: %d", processed_instances)
        LOGGER.info("Average ratio improvement: %.4f", mean_gain)
        LOGGER.info("Median ratio improvement: %.4f", median_gain)
        LOGGER.info("Improvement ratio>0: %.2f%%", positive_rate * 100.0)
    else:
        LOGGER.info("No valid instances were processed.")


if __name__ == "__main__":
    main()
