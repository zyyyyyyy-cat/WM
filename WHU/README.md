# DDP
## E1: M-SpecGene initialization

E1 changes only the ViT-B/16 initialization. The UAD architecture, losses,
sampler, optimizer, augmentations, single-GPU/DDP flow, and evaluation protocol
remain unchanged.

Use the official encoder-only detection conversion
`M-SpecGene_VIT-B_det_transform.pth` as the preferred source. It contains the
same self-supervised encoder as the original foundation checkpoint but already
uses UAD-compatible ViT key names; do not use a downstream dataset checkpoint
such as `FLIR_iter_90000.pth`. The original `M-SpecGene_VIT-B.pth` is also
accepted and is mapped with the official conversion rules.

```bash
mkdir -p pretrained
python tools/convert_mspecgene_to_uad.py \
  /path/to/M-SpecGene_VIT-B_det_transform.pth \
  pretrained/M-SpecGene_VIT-B_UAD_256x128.pth
```

The converter requires all 150 ViT encoder tensors, verifies their shapes,
resizes positional embeddings from `[1,197,768]` (14x14 patches plus CLS) to
`[1,129,768]` (16x8 patches plus CLS), preserves the CLS position, and writes a
JSON report next to the output containing the source SHA-256 and complete tensor
mapping. Missing QKV biases, blocks, norms, or FFN tensors are fatal.

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=7 python train.py \
  --config_file configs/UAD_MSpecGene.yml \
  MODEL.DIST_TRAIN False \
  SOLVER.IMS_PER_BATCH 64 \
  SOLVER.BASE_LR 0.008
```

Four-GPU DDP (same scaling used by the UAD baseline):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone \
  --nproc_per_node=4 \
  train.py \
  --config_file configs/UAD_MSpecGene.yml \
  MODEL.DIST_TRAIN True \
  SOLVER.IMS_PER_BATCH 256 \
  SOLVER.BASE_LR 0.032 \
  OUTPUT_DIR logs/UAD_MSpecGene_1000_DDP
```

## WHU-MARS-1000

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone \
  --nproc_per_node=4 \
  train.py \
  --config_file configs/UAD.yml \
  MODEL.DIST_TRAIN True \
  SOLVER.IMS_PER_BATCH 256 \
  SOLVER.BASE_LR 0.032 \
  OUTPUT_DIR logs/UAD_1000_DDP

CUDA_VISIBLE_DEVICES=7 python test.py \
  --config_file configs/UAD.yml \
  DATASETS.DATASET_DIR WHU-MARS \
  TEST.IMS_PER_BATCH 512 \
  TEST.WEIGHT logs/UAD_1000_DDP/transformer_120.pth \
  OUTPUT_DIR logs/UAD_1000_DDP

CUDA_VISIBLE_DEVICES=7 python test.py \
  --config_file configs/UAD.yml \
  MODEL.DIST_TRAIN False \
  TEST.IMS_PER_BATCH 512 \
  TEST.WEIGHT logs/UAD_1000_DDP/transformer_120.pth \
  TEST.DETAILED_DOMAIN_EVAL True \
  OUTPUT_DIR logs/UAD_1000_DDP_domain_eval

## WHU-MARS-1000-GD
CUDA_VISIBLE_DEVICES=6 python test.py \
  --config_file configs/UAD.yml \
  DATASETS.FILTER_GD True \
  TEST.IMS_PER_BATCH 512 \
  TEST.WEIGHT logs/UAD_1000_DDP/transformer_120.pth \
  OUTPUT_DIR logs/UAD_1000_GD

## WHU-MARS-2337
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone \
  --nproc_per_node=4 \
  train.py \
  --config_file configs/UAD.yml \
  MODEL.DIST_TRAIN True \
  DATASETS.DATASET_DIR WHU-MARS-2337 \
  SOLVER.IMS_PER_BATCH 256 \
  SOLVER.BASE_LR 0.032 \
  OUTPUT_DIR logs/UAD_2337_DDP

# Single
## WHU-MARS-1000

cd /project/code/WHU_base/WHU_DDP

CUDA_VISIBLE_DEVICES=7 python train.py \
  --config_file configs/UAD.yml \
  MODEL.DIST_TRAIN False \
  DATASETS.DATASET_DIR WHU-MARS \
  SOLVER.IMS_PER_BATCH 64 \
  SOLVER.BASE_LR 0.008 \
  SOLVER.SEED 1234 \
  TEST.IMS_PER_BATCH 1024 \
  OUTPUT_DIR logs/UAD_1000_single2

CUDA_VISIBLE_DEVICES=0 python test.py \
  --config_file configs/UAD.yml \
  MODEL.DIST_TRAIN False \
  DATASETS.DATASET_DIR WHU-MARS \
  TEST.IMS_PER_BATCH 1024 \
  TEST.WEIGHT logs/UAD_1000_single/transformer_120.pth \
  OUTPUT_DIR logs/UAD_1000_single

## WHU-MARS-1000-GD

CUDA_VISIBLE_DEVICES=0 python test.py \
  --config_file configs/UAD.yml \
  MODEL.DIST_TRAIN False \
  DATASETS.DATASET_DIR WHU-MARS \
  DATASETS.FILTER_GD True \
  TEST.IMS_PER_BATCH 512 \
  TEST.WEIGHT logs/UAD_1000_single/transformer_120.pth \
  OUTPUT_DIR logs/UAD_1000_single_GD

## WHU-MARS-2337

CUDA_VISIBLE_DEVICES=6 python train.py \
  --config_file configs/UAD.yml \
  MODEL.DIST_TRAIN False \
  DATASETS.DATASET_DIR WHU-MARS-2337 \
  SOLVER.IMS_PER_BATCH 64 \
  SOLVER.BASE_LR 0.008 \
  SOLVER.SEED 1234 \
  TEST.IMS_PER_BATCH 1024 \
  OUTPUT_DIR logs/UAD_2337_single

CUDA_VISIBLE_DEVICES=0 python test.py \
  --config_file configs/UAD.yml \
  MODEL.DIST_TRAIN False \
  DATASETS.DATASET_DIR WHU-MARS-2337 \
  TEST.IMS_PER_BATCH 1024 \
  TEST.WEIGHT logs/UAD_2337_single/transformer_120.pth \
  OUTPUT_DIR logs/UAD_2337_single
