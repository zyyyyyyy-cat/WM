"""Strict M-SpecGene ViT-B/16 checkpoint conversion for UAD.

The official M-SpecGene release provides both the original MMPretrain
checkpoint and a ViTDet-oriented encoder conversion.  This module accepts
either representation, retains encoder tensors only, resizes the positional
embedding for UAD's rectangular patch grid, and rejects partial loads.
"""

from collections import OrderedDict
import hashlib
import json
import math
import os
import re

import torch
import torch.distributed as dist
import torch.nn.functional as F


_BLOCK_PARTS = (
    "norm1.weight", "norm1.bias",
    "attn.qkv.weight", "attn.qkv.bias",
    "attn.proj.weight", "attn.proj.bias",
    "norm2.weight", "norm2.bias",
    "mlp.fc1.weight", "mlp.fc1.bias",
    "mlp.fc2.weight", "mlp.fc2.bias",
)


def expected_encoder_keys(depth=12):
    """Return the complete UAD ViT encoder key set (150 keys for ViT-B)."""
    keys = [
        "cls_token", "pos_embed",
        "patch_embed.proj.weight", "patch_embed.proj.bias",
    ]
    for index in range(depth):
        keys.extend("blocks.{}.{}".format(index, part) for part in _BLOCK_PARTS)
    keys.extend(("norm.weight", "norm.bias"))
    return tuple(keys)


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_state_dict(checkpoint):
    """Unwrap common checkpoint containers without accepting arbitrary data."""
    current = checkpoint
    wrappers = []
    for _ in range(4):
        if not isinstance(current, dict):
            break
        tensor_values = [value for value in current.values()
                         if isinstance(value, torch.Tensor)]
        if tensor_values:
            return current, wrappers
        for key in ("state_dict", "model", "module", "student", "encoder"):
            if key in current and isinstance(current[key], dict):
                wrappers.append(key)
                current = current[key]
                break
        else:
            break
    raise ValueError(
        "Could not find a tensor state dictionary in the checkpoint; "
        "expected a top-level state dict or a model/state_dict wrapper."
    )


def _strip_prefixes(key):
    while key.startswith("module."):
        key = key[len("module."):]
    return key


def _map_encoder_key(key):
    """Map official raw/MMPretrain and det-transform names to UAD names."""
    key = _strip_prefixes(key)
    if key.startswith("backbone."):
        key = key[len("backbone."):]

    # Official det-transform checkpoints already use these UAD/timm names.
    if key in ("cls_token", "pos_embed", "norm.weight", "norm.bias"):
        return key
    if key in ("patch_embed.proj.weight", "patch_embed.proj.bias"):
        return key
    if re.fullmatch(r"blocks\.\d+\.(?:" + "|".join(
            re.escape(part) for part in _BLOCK_PARTS) + r")", key):
        return key

    # Original checkpoint (and the segmentation conversion) use MMPretrain.
    if key == "patch_embed.projection.weight":
        return "patch_embed.proj.weight"
    if key == "patch_embed.projection.bias":
        return "patch_embed.proj.bias"
    if key == "ln1.weight":
        return "norm.weight"
    if key == "ln1.bias":
        return "norm.bias"

    match = re.fullmatch(r"layers\.(\d+)\.(.+)", key)
    if match is None:
        return None
    index, suffix = match.groups()
    replacements = {
        "ln1.weight": "norm1.weight",
        "ln1.bias": "norm1.bias",
        "attn.qkv.weight": "attn.qkv.weight",
        "attn.qkv.bias": "attn.qkv.bias",
        "attn.proj.weight": "attn.proj.weight",
        "attn.proj.bias": "attn.proj.bias",
        "ln2.weight": "norm2.weight",
        "ln2.bias": "norm2.bias",
        "ffn.layers.0.0.weight": "mlp.fc1.weight",
        "ffn.layers.0.0.bias": "mlp.fc1.bias",
        "ffn.layers.1.weight": "mlp.fc2.weight",
        "ffn.layers.1.bias": "mlp.fc2.bias",
    }
    mapped_suffix = replacements.get(suffix)
    if mapped_suffix is None:
        return None
    return "blocks.{}.{}".format(index, mapped_suffix)


def map_mspecgene_encoder(state_dict):
    """Extract and rename exactly the shared M-SpecGene ViT encoder."""
    mapped = OrderedDict()
    ignored = []
    source_names = {}
    for source_key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            ignored.append(source_key)
            continue
        target_key = _map_encoder_key(source_key)
        if target_key is None:
            ignored.append(source_key)
            continue
        if target_key in mapped:
            raise ValueError(
                "Multiple checkpoint tensors map to {!r}: {!r} and {!r}".format(
                    target_key, source_names[target_key], source_key
                )
            )
        mapped[target_key] = value.detach().cpu()
        source_names[target_key] = source_key
    return mapped, ignored, source_names


def _infer_square_grid(token_count):
    side = int(math.sqrt(token_count))
    if side * side != token_count:
        raise ValueError(
            "Cannot infer the source positional grid from {} patch tokens. "
            "Pass source_grid=(height, width) explicitly.".format(token_count)
        )
    return side, side


def resize_position_embedding(pos_embed, target_grid, source_grid=None,
                              mode="bilinear"):
    """Resize patch positions while preserving the CLS position exactly."""
    if pos_embed.ndim != 3 or pos_embed.shape[0] != 1:
        raise ValueError(
            "Expected pos_embed shape [1, tokens, dim], got {}".format(
                tuple(pos_embed.shape))
        )
    source_tokens = pos_embed.shape[1] - 1
    if source_grid is None:
        source_grid = _infer_square_grid(source_tokens)
    source_grid = tuple(int(value) for value in source_grid)
    target_grid = tuple(int(value) for value in target_grid)
    if source_grid[0] * source_grid[1] != source_tokens:
        raise ValueError(
            "source_grid {} does not match {} patch tokens".format(
                source_grid, source_tokens
            )
        )
    if len(target_grid) != 2 or min(target_grid) <= 0:
        raise ValueError("target_grid must contain two positive integers")
    if mode not in ("bilinear", "bicubic"):
        raise ValueError("interpolation mode must be 'bilinear' or 'bicubic'")

    cls_position = pos_embed[:, :1]
    patch_positions = pos_embed[:, 1:].reshape(
        1, source_grid[0], source_grid[1], pos_embed.shape[-1]
    ).permute(0, 3, 1, 2)
    patch_positions = F.interpolate(
        patch_positions.float(), size=target_grid, mode=mode,
        align_corners=False,
    ).to(dtype=pos_embed.dtype)
    patch_positions = patch_positions.permute(0, 2, 3, 1).reshape(
        1, target_grid[0] * target_grid[1], pos_embed.shape[-1]
    )
    return torch.cat((cls_position, patch_positions), dim=1)


def _encoder_target_state(backbone):
    state = backbone.state_dict()
    expected = expected_encoder_keys(depth=len(backbone.blocks))
    missing_from_model = [key for key in expected if key not in state]
    if missing_from_model:
        raise ValueError(
            "The target backbone is not the expected UAD ViT: {}".format(
                missing_from_model
            )
        )
    return OrderedDict((key, state[key]) for key in expected)


def prepare_mspecgene_checkpoint(checkpoint_path, backbone, source_grid=None,
                                 interpolation="bilinear"):
    """Convert a checkpoint and strictly validate every UAD encoder tensor."""
    checkpoint = _torch_load(checkpoint_path)
    state_dict, wrappers = _extract_state_dict(checkpoint)
    mapped, ignored, source_names = map_mspecgene_encoder(state_dict)
    target_state = _encoder_target_state(backbone)
    expected = tuple(target_state.keys())

    missing = [key for key in expected if key not in mapped]
    unexpected = [key for key in mapped if key not in target_state]
    if missing or unexpected:
        raise ValueError(
            "Incomplete M-SpecGene encoder: missing={} unexpected={}".format(
                missing, unexpected
            )
        )

    source_pos_shape = tuple(mapped["pos_embed"].shape)
    target_grid = (backbone.patch_embed.num_y, backbone.patch_embed.num_x)
    if mapped["pos_embed"].shape != target_state["pos_embed"].shape:
        mapped["pos_embed"] = resize_position_embedding(
            mapped["pos_embed"], target_grid=target_grid,
            source_grid=source_grid, mode=interpolation,
        )

    shape_errors = {}
    for key in expected:
        if tuple(mapped[key].shape) != tuple(target_state[key].shape):
            shape_errors[key] = {
                "checkpoint": list(mapped[key].shape),
                "target": list(target_state[key].shape),
            }
    if shape_errors:
        raise ValueError(
            "M-SpecGene encoder tensor shape mismatch: {}".format(shape_errors)
        )

    meta = checkpoint.get("meta", {}) if isinstance(checkpoint, dict) else {}
    report = {
        "source_checkpoint": os.path.abspath(checkpoint_path),
        "checkpoint_wrappers": wrappers,
        "source_metadata": meta,
        "source_pos_embed_shape": list(source_pos_shape),
        "target_pos_embed_shape": list(mapped["pos_embed"].shape),
        "target_patch_grid": list(target_grid),
        "interpolation": interpolation,
        "loaded_tensor_count": len(mapped),
        "required_tensor_count": len(expected),
        "loaded_tensor_ratio": len(mapped) / len(expected),
        "ignored_tensor_count": len(ignored),
        "ignored_tensors": sorted(ignored),
        "mapped_tensors": [
            {
                "source": source_names[key],
                "target": key,
                "shape": list(mapped[key].shape),
            }
            for key in expected
        ],
    }
    return OrderedDict((key, mapped[key]) for key in expected), report


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def convert_mspecgene_checkpoint(checkpoint_path, output_path, backbone,
                                 report_path=None, source_grid=None,
                                 interpolation="bilinear"):
    """Create a deterministic, encoder-only UAD checkpoint and JSON report."""
    converted, report = prepare_mspecgene_checkpoint(
        checkpoint_path, backbone, source_grid=source_grid,
        interpolation=interpolation,
    )
    report["source_sha256"] = sha256_file(checkpoint_path)
    report["output_checkpoint"] = os.path.abspath(output_path)
    output = {
        "model": converted,
        "meta": {
            "format": "mspecgene_uad_vit_b16_encoder_v1",
            "source_checkpoint": report["source_checkpoint"],
            "source_sha256": report["source_sha256"],
            "target_patch_grid": report["target_patch_grid"],
            "interpolation": interpolation,
            "tensor_count": len(converted),
        },
    }
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    torch.save(output, output_path)

    if report_path is None:
        report_path = output_path + ".json"
    report_dir = os.path.dirname(os.path.abspath(report_path))
    os.makedirs(report_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return report


def _is_main_process():
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def load_mspecgene_backbone(backbone, checkpoint_path, source_grid=None,
                            interpolation="bilinear"):
    """Strictly load all and only M-SpecGene encoder tensors into UAD."""
    converted, report = prepare_mspecgene_checkpoint(
        checkpoint_path, backbone, source_grid=source_grid,
        interpolation=interpolation,
    )
    with torch.no_grad():
        target_state = backbone.state_dict()
        for key, value in converted.items():
            target_state[key].copy_(value)
    if _is_main_process():
        print(
            "Loaded M-SpecGene encoder: {}/{} tensors (100.0%), "
            "position {} -> {}, source={}".format(
                report["loaded_tensor_count"],
                report["required_tensor_count"],
                report["source_pos_embed_shape"],
                report["target_pos_embed_shape"],
                checkpoint_path,
            )
        )
    return report
