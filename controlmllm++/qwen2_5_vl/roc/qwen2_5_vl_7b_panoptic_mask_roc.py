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
      --n-samples 5 \
      --min-area 4096 \
      --layers 12,28 \
      --heads 0,1,2,3,4,5,6,7 \
      --steps 2 --lr 0.02 \
      --outdir outputs/panoptic_val2017_demo \
      --save-all

The command above processes a single COCO image and saves pre/post optimisation
attention overlays, token maps, ``npz`` dumps, and JSON summary lines under the
``outputs/panoptic_val2017_demo/397133/`` directory.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info
from qwen_utils import get_grid_shape
from .utils_panoptic import (
    draw_overlay_heatmap,
    downsample_mask_to_tokens,
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
    parser.add_argument("--hires-crop-size", type=int, default=0, help="Optional crop size for high resolution text patches")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
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


def is_text_category(name: str) -> bool:
    lowered = (name or "").lower()
    return any(keyword in lowered for keyword in TEXT_KEYWORDS)


def refine_text_mask(mask: np.ndarray) -> np.ndarray:
    if cv2 is None:
        return mask
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    return (dilated > 0).astype(np.uint8)


def compute_token_attention(attn_list: Sequence[torch.Tensor], layer_indices: Sequence[int],
                            head_indices: Sequence[int], query_index: int,
                            vision_start: int, vision_end: int) -> torch.Tensor:
    if not attn_list:
        raise ValueError("Attention list is empty")
    selected_layers: List[torch.Tensor] = []
    for idx in layer_indices:
        layer_attn = attn_list[idx].to(DEVICE)
        selected_layers.append(layer_attn)
    stacked = torch.cat(selected_layers, dim=0)
    attn_mean = stacked.mean(dim=0)
    if head_indices:
        attn_mean = attn_mean[head_indices, :, :]
    token_slice = attn_mean[:, query_index, vision_start:vision_end]
    return token_slice.mean(dim=0, keepdim=True)


def token_attention_to_map(token_att: torch.Tensor, grid_shape: Tuple[int, int]) -> torch.Tensor:
    h, w = grid_shape
    return token_att.reshape(h, w)


def compute_attention_ratio(token_att: torch.Tensor, mask_tok: torch.Tensor,
                            grid_shape: Tuple[int, int], eps: float = EPS) -> float:
    attn_map = token_attention_to_map(token_att, grid_shape)
    mask = mask_tok.float()
    numerator = torch.sum(attn_map * mask)
    denominator = torch.sum(attn_map)
    ratio = (numerator + eps) / (denominator + eps)
    return float(ratio.item())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_image_safe(image: Image.Image, path: Path) -> None:
    try:
        ensure_dir(path.parent)
        image.save(path)
    except Exception as exc:  # pragma: no cover - IO guard
        print(f"[WARN] Failed to save {path}: {exc}")


def save_token_map(attn_map: np.ndarray, path: Path) -> None:
    try:
        ensure_dir(path.parent)
        norm = attn_map - attn_map.min()
        denom = norm.max()
        if denom > 0:
            norm = norm / denom
        Image.fromarray((norm * 255).astype(np.uint8)).save(path)
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Failed to save token map {path}: {exc}")


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


def build_prompt(cat_name: Optional[str]) -> str:
    if cat_name:
        return (
            f"Focus on the object inside the given mask region. "
            f"Is the object a {cat_name}? Answer with yes or no and briefly justify."
        )
    return "What object is inside the given mask region? Respond with a short phrase."


def prepare_msgs(image_path: str, prompt: str, highres_patch: Optional[Image.Image] = None) -> List[Dict[str, object]]:
    content: List[Dict[str, object]] = [
        {"type": "image", "image": image_path, "max_pixels": 512 * 28 * 28},
    ]
    if highres_patch is not None:
        content.append({"type": "image", "image": highres_patch})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def prepare_inputs(processor: AutoProcessor, msgs: List[Dict[str, object]]) -> Tuple[Dict[str, torch.Tensor], Tuple[int, int], Tuple[int, int]]:
    image_inputs, _ = process_vision_info(msgs)
    grid_shape = get_grid_shape_fn(processor, image_inputs)
    text_prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=image_inputs, padding=True, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    input_ids = inputs["input_ids"][0].tolist()
    vision_start = input_ids.index(vision_start_token_id) + 1
    vision_end = input_ids.index(vision_end_token_id)
    return inputs, grid_shape, (vision_start, vision_end)


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
            print(f"[WARN] Missing metadata for image {image_id}")
            continue

        img_path = os.path.join(args.coco_img_root, image_meta["file_name"])
        try:
            image = load_image(args.coco_img_root, image_meta["file_name"])
        except FileNotFoundError:
            print(f"[WARN] Image file not found: {img_path}")
            continue

        try:
            seg_map = load_panoptic_mask(args.panoptic_mask_root, ann_meta["file_name"])
        except FileNotFoundError:
            print(f"[WARN] Mask file not found for image {image_id}")
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
                continue

            prompt = build_prompt(cat_name)
            highres_patch: Optional[Image.Image] = None
            if args.text_mask_mode and args.hires_crop_size > 0 and is_text_category(cat_name):
                if cv2 is not None:
                    ys, xs = np.where(binary_mask > 0)
                    if len(xs) and len(ys):
                        x0, x1 = xs.min(), xs.max()
                        y0, y1 = ys.min(), ys.max()
                        pad = int(0.05 * max(x1 - x0 + 1, y1 - y0 + 1))
                        x0 = max(0, x0 - pad)
                        y0 = max(0, y0 - pad)
                        x1 = min(binary_mask.shape[1] - 1, x1 + pad)
                        y1 = min(binary_mask.shape[0] - 1, y1 + pad)
                        crop = image.crop((x0, y0, x1 + 1, y1 + 1))
                        highres_patch = crop.resize((args.hires_crop_size, args.hires_crop_size), Image.BICUBIC)

            msgs = prepare_msgs(img_path, prompt, highres_patch)
            try:
                inputs, grid_shape, (vision_start, vision_end) = prepare_inputs(processor, msgs)
            except ValueError as exc:
                print(f"[WARN] Failed to prepare inputs for image {image_id}: {exc}")
                continue

            h_tok, w_tok = grid_shape
            mask_tokens = downsample_mask_to_tokens(binary_mask, h_tok, w_tok)
            if mask_tokens.sum() <= 0:
                continue
            mask_tok_tensor = torch.from_numpy(mask_tokens).to(DEVICE).float()

            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)
            token_att_pre = compute_token_attention(outputs.attentions, layer_indices, head_indices, -1, vision_start, vision_end)
            ratio_before = compute_attention_ratio(token_att_pre, mask_tok_tensor, grid_shape)

            num_tokens = h_tok * w_tok
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
                token_att = compute_token_attention(outputs.attentions, layer_indices, head_indices, -1, vision_start, vision_end)
                loss = args.alpha * compute_activation_loss(token_att, [mask_tok_tensor])
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
                token_att_post = compute_token_attention(outputs_post.attentions, layer_indices, head_indices, -1, vision_start, vision_end)
                ratio_after = compute_attention_ratio(token_att_post, mask_tok_tensor, grid_shape)
                # 临时保存并移除visual_prompt以避免generate时的设备冲突
                saved_prompt = model.visual_prompt
                model.visual_prompt = None
                generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
                model.visual_prompt = saved_prompt  # 恢复
                trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
                decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                answer = decoded[0] if decoded else ""

            ratio_gains.append(ratio_after - ratio_before)
            processed_instances += 1

            attn_pre_np = token_attention_to_map(token_att_pre, grid_shape).detach().cpu().numpy()
            attn_post_np = token_attention_to_map(token_att_post, grid_shape).detach().cpu().numpy()

            overlay_pre = draw_overlay_heatmap(image, upsample_attn_to_image(attn_pre_np, image.height, image.width))
            overlay_post = draw_overlay_heatmap(image, upsample_attn_to_image(attn_post_np, image.height, image.width))

            prefix = f"{seg_id}_{cat_slug}"
            if args.save_all:
                save_image_safe(overlay_pre, out_dir / f"{prefix}_pre_overlay.png")
                save_image_safe(overlay_post, out_dir / f"{prefix}_post_overlay.png")
                save_token_map(attn_pre_np, out_dir / f"{prefix}_pre_token.png")
                save_token_map(attn_post_np, out_dir / f"{prefix}_post_token.png")
            else:
                save_image_safe(overlay_post, out_dir / f"{prefix}_post_overlay.png")

            npz_path = out_dir / f"{prefix}_attn.npz"
            try:
                ensure_dir(npz_path.parent)
                np.savez_compressed(npz_path, attn_pre=attn_pre_np, attn_post=attn_post_np, mask_tokens=mask_tokens)
            except Exception as exc:  # pragma: no cover
                print(f"[WARN] Failed to save attention arrays for {npz_path}: {exc}")

            if args.text_mask_mode and is_text_category(cat_name):
                text_mask = refine_text_mask(binary_mask)
                text_overlay = draw_overlay_heatmap(image, upsample_attn_to_image(text_mask.astype(np.float32), image.height, image.width))
                save_image_safe(text_overlay, out_dir / f"{prefix}_text_overlay.png")

            record = InstanceRecord(
                image_id=image_id,
                segment_id=seg_id,
                category_id=cat_id,
                category_name=cat_name,
                prompt=prompt,
                answer=answer,
                ratio_before=ratio_before,
                ratio_after=ratio_after,
            )
            append_summary(summary_path, record)

            print(f"[INFO] image {image_id} seg {seg_id} ({cat_slug}) ratio {ratio_before:.3f} -> {ratio_after:.3f} | {answer}")

    if processed_instances:
        gains = np.array(ratio_gains, dtype=np.float32)
        mean_gain = float(gains.mean())
        median_gain = float(np.median(gains))
        positive_rate = float((gains > 0).mean())
        print("=== Summary ===")
        print(f"Total instances: {total_instances}")
        print(f"Processed instances: {processed_instances}")
        print(f"Average ratio improvement: {mean_gain:.4f}")
        print(f"Median ratio improvement: {median_gain:.4f}")
        print(f"Improvement ratio>0: {positive_rate * 100:.2f}%")
    else:
        print("No valid instances were processed.")


if __name__ == "__main__":
    main()
