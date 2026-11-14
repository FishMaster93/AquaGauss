#!/bin/bash
# Reproduce the experiments of the paper (after data/manifest_video.csv and pretrained/S3D400.pt are in place).
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}

# 1. Phase 0: adapt S3D on the base species, then extract frozen features of all clips
#    (base0 = red tilapia, main setting; base6 = O. punctatus and base5 = sunfish for the base-species study)
for K in 0 6 5; do
  [ -f workspace/S3D_base$K/phase0_best.pt ] || \
    $PY scripts/train_phase0.py --exp_name S3D_base$K --phase $K --backbone s3d --epochs 20 --batch_size 24 --lr 1e-4
  $PY scripts/extract_features.py --name s3d_base$K --ckpt workspace/S3D_base$K/phase0_best.pt
done

# 2. AquaGauss: select hyperparameters on the validation set, then run all orders and base species
$PY scripts/select_hparams.py --feat s3d_base0
$PY scripts/run_incremental.py --feat s3d_base0 --orders default reversed random
$PY scripts/run_incremental.py --feat s3d_base6 --orders base6
$PY scripts/run_incremental.py --feat s3d_base5 --orders base5

# 3. Baselines on the same features
$PY scripts/select_baseline_hp.py --feat s3d_base0
$PY scripts/run_baselines.py --feat s3d_base0 --orders default reversed random
$PY scripts/run_baselines.py --feat s3d_base6 --orders base6 --methods dsal der
$PY scripts/run_baselines.py --feat s3d_base5 --orders base5 --methods dsal der

# 4. Analyses and deployment cost
$PY scripts/analyze.py --feat s3d_base0
$PY scripts/measure_cost.py --feat s3d_base0 --ckpt workspace/S3D_base0/phase0_best.pt
