import tempfile
import unittest
from collections import OrderedDict
import json
from pathlib import Path

import torch

from model.backbones.mspecgene import (
    convert_mspecgene_checkpoint,
    expected_encoder_keys,
    load_mspecgene_backbone,
    prepare_mspecgene_checkpoint,
)
from model.backbones.vit_pytorch import vit_base_in


def to_raw_mmpretrain_key(key):
    if key == "norm.weight":
        return "backbone.ln1.weight"
    if key == "norm.bias":
        return "backbone.ln1.bias"
    if key.startswith("patch_embed.proj"):
        return "backbone." + key.replace(
            "patch_embed.proj", "patch_embed.projection"
        )
    if key.startswith("blocks."):
        raw = key.replace("blocks.", "backbone.layers.", 1)
        raw = raw.replace(".norm1.", ".ln1.")
        raw = raw.replace(".norm2.", ".ln2.")
        raw = raw.replace(".mlp.fc1.", ".ffn.layers.0.0.")
        raw = raw.replace(".mlp.fc2.", ".ffn.layers.1.")
        return raw
    return "backbone." + key


class MSpecGeneConversionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.backbone = vit_base_in(
            img_size=(256, 128), stride_size=(16, 16), drop_path_rate=0.0
        )
        state = self.backbone.state_dict()
        self.encoder = OrderedDict(
            (key, torch.randn_like(state[key]))
            for key in expected_encoder_keys()
        )
        self.encoder["pos_embed"] = torch.randn(1, 197, 768)

    def _save(self, checkpoint):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "checkpoint.pth"
        torch.save(checkpoint, path)
        return directory, path

    def test_official_det_transform_format(self):
        directory, path = self._save({"model": self.encoder})
        try:
            converted, report = prepare_mspecgene_checkpoint(
                str(path), self.backbone
            )
        finally:
            directory.cleanup()
        self.assertEqual(len(converted), 150)
        self.assertEqual(tuple(converted["pos_embed"].shape), (1, 129, 768))
        self.assertEqual(report["loaded_tensor_ratio"], 1.0)
        self.assertTrue(torch.equal(
            converted["blocks.11.attn.qkv.bias"],
            self.encoder["blocks.11.attn.qkv.bias"],
        ))

    def test_original_mmpretrain_format_and_strict_load(self):
        raw = OrderedDict(
            (to_raw_mmpretrain_key(key), value)
            for key, value in self.encoder.items()
        )
        raw["neck.decoder.weight"] = torch.randn(4, 4)
        directory, path = self._save({"state_dict": raw})
        try:
            report = load_mspecgene_backbone(self.backbone, str(path))
        finally:
            directory.cleanup()
        self.assertEqual(report["loaded_tensor_count"], 150)
        self.assertIn("neck.decoder.weight", report["ignored_tensors"])
        self.assertTrue(torch.equal(
            self.backbone.state_dict()["blocks.0.mlp.fc1.bias"],
            self.encoder["blocks.0.mlp.fc1.bias"],
        ))

    def test_missing_qkv_bias_is_fatal(self):
        incomplete = OrderedDict(self.encoder)
        del incomplete["blocks.3.attn.qkv.bias"]
        directory, path = self._save({"model": incomplete})
        try:
            with self.assertRaisesRegex(ValueError, "blocks.3.attn.qkv.bias"):
                prepare_mspecgene_checkpoint(str(path), self.backbone)
        finally:
            directory.cleanup()

    def test_converter_writes_auditable_checkpoint_and_report(self):
        directory, path = self._save({"model": self.encoder})
        output = Path(directory.name) / "converted.pth"
        report_path = Path(directory.name) / "report.json"
        try:
            report = convert_mspecgene_checkpoint(
                str(path), str(output), self.backbone,
                report_path=str(report_path),
            )
            saved = torch.load(output, map_location="cpu", weights_only=False)
            with report_path.open("r", encoding="utf-8") as stream:
                saved_report = json.load(stream)
        finally:
            directory.cleanup()
        self.assertEqual(saved["meta"]["tensor_count"], 150)
        self.assertEqual(saved["meta"]["target_patch_grid"], [16, 8])
        self.assertEqual(len(saved["meta"]["source_sha256"]), 64)
        self.assertEqual(saved_report["source_sha256"], report["source_sha256"])


if __name__ == "__main__":
    unittest.main()
