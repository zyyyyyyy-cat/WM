"""Convert an official M-SpecGene ViT-B checkpoint into UAD encoder format."""

import argparse
import json
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.backbones.mspecgene import convert_mspecgene_checkpoint  # noqa: E402
from model.backbones.vit_pytorch import vit_base_in  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract and strictly validate the 150-tensor M-SpecGene ViT-B/16 "
            "encoder for the UAD 256x128 backbone. Both the original "
            "M-SpecGene_VIT-B.pth and official det_transform checkpoint are "
            "accepted."
        )
    )
    parser.add_argument("input", help="official M-SpecGene checkpoint")
    parser.add_argument("output", help="output UAD encoder checkpoint")
    parser.add_argument(
        "--report", default=None,
        help="JSON load report (default: OUTPUT.json)",
    )
    parser.add_argument(
        "--image-size", nargs=2, type=int, default=(256, 128),
        metavar=("HEIGHT", "WIDTH"),
    )
    parser.add_argument(
        "--stride-size", nargs=2, type=int, default=(16, 16),
        metavar=("HEIGHT", "WIDTH"),
    )
    parser.add_argument(
        "--source-grid", nargs=2, type=int, default=None,
        metavar=("HEIGHT", "WIDTH"),
        help="only needed when source patch positions are not a square grid",
    )
    parser.add_argument(
        "--interpolation", choices=("bilinear", "bicubic"),
        default="bilinear",
        help="UAD uses bilinear positional interpolation by default",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    backbone = vit_base_in(
        img_size=tuple(args.image_size),
        stride_size=tuple(args.stride_size),
        drop_path_rate=0.1,
    )
    report = convert_mspecgene_checkpoint(
        checkpoint_path=args.input,
        output_path=args.output,
        backbone=backbone,
        report_path=args.report,
        source_grid=args.source_grid,
        interpolation=args.interpolation,
    )
    summary = {
        key: report[key]
        for key in (
            "source_checkpoint", "source_sha256", "output_checkpoint",
            "source_pos_embed_shape", "target_pos_embed_shape",
            "loaded_tensor_count", "required_tensor_count",
            "loaded_tensor_ratio", "ignored_tensor_count",
        )
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
