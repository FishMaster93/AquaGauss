# AquaGauss

Code for **AquaGauss**, an exemplar-free method for species-incremental fish feeding intensity recognition (FFIR)
from video. A single model recognises four feeding intensity levels (None, Weak, Medium, Strong) and is extended one
fish species at a time. No video, clip or feature of an earlier species is kept, and no network is trained after the
first species.

Each species–intensity **cell** is stored as a Gaussian (mean, covariance, number of clips) of frozen video features.
AquaGauss has three components:

1. **Distribution memory**: four new cells per species; earlier cells are never changed, so the final model does not
   depend on the order in which species arrive.
2. **Distribution calibration**: a power transform of the features (λ = 0.5), shrinkage of every cell covariance
   towards a covariance pooled over all learned cells (α = 0.75), and a scaled ridge (ε = 0.03).
3. **Spread-aware scoring**: Mahalanobis distance plus β·log|Σ″ₖ| (β = 0.5), which removes the bias of the
   Mahalanobis distance towards the broad cells of high feeding intensity.

The encoder is S3D (Kinetics-400 pretrained), adapted once on the first species with a temporary ordinal head and
then frozen.

Main result (seven species, default order, test set):

| Method | Data kept | Memory | Final acc. (%) | Avg. inc. acc. (%) | Forgetting (%) |
|---|---|---|---|---|---|
| NCM | cell means | 0.11 MB | 60.63 | 67.15 | 0.54 |
| FeCAM | cell means and covariances | 56.16 MB | 85.68 | 87.76 | 0.00 |
| **AquaGauss** | cell means and covariances, pooled scatter | 58.17 MB | **89.15** | **90.57** | 0.18 |

## Repository structure

```
aquagauss/
  gauss.py              GaussMemory: AquaGauss (and the NCM / FeCAM baselines) - statistics, calibration, scoring
  common.py             species orders, incremental loop, metrics
vcil/
  paths.py              paths (all relative to the repository root)
  data.py               species/phase definition, manifest reader, clip dataset, GPU augmentation
  features.py           frozen feature cache
  models/s3d.py         S3D backbone
  models/video.py       backbone + ordinal head (phase 0)
  cil.py, methods/      continual-learning baselines on frozen features (fine-tuning, LwF, EWC, SI, MAS,
                        ACIL, DS-AL, ER, A-GEM, DER++, iCaRL, GDumb, joint training)
scripts/
  preprocess_video.py   mp4 -> (16, 224, 224, 3) uint8 .npy arrays
  build_manifest.py     build data/manifest_video.csv from the released split and your local arrays
  train_phase0.py       phase 0: adapt the encoder on the base species
  extract_features.py   extract frozen features of all clips
  select_hparams.py     select lambda, alpha, eps, beta of AquaGauss and its ablations on the validation set
  run_incremental.py    species-incremental runs of AquaGauss, its ablations, FeCAM and NCM
  select_baseline_hp.py, run_baselines.py   hyperparameter selection and runs of the baselines
  analyze.py            significance tests, sensitivity, cell spread, cell granularity
  measure_cost.py       memory, time to add a species, inference time
  run_all.sh            the whole pipeline
data/splits/split_clip.csv   clip-level 80/10/10 split (99,960 clips), stratified by species and intensity
```

## Installation

```bash
conda create -n aquagauss python=3.11 -y
conda activate aquagauss
pip install -r requirements.txt
```

A CUDA GPU is required for phase-0 training and feature extraction. All later steps run on frozen features.

## Data

Seven species are learned in seven phases:

| Phase | Species | Site |
|---|---|---|
| 0 | Red tilapia (`Red_tilapia_2` + `Red_tilapia_3`) | Guangzhou |
| 1 | Tilapia | Guangzhou |
| 2 | Jade perch | Guangzhou |
| 3 | Largemouth bass | Guangzhou |
| 4 | Lotus carp | Guangzhou |
| 5 | Sunfish | Guangzhou |
| 6 | *Oplegnathus punctatus* | Yantai |

1. **Download** the Guangzhou clips (six species) from Zenodo: https://zenodo.org/records/23623291
   (`video_dataset.tar`, 28 GB), and the *O. punctatus* clips from https://doi.org/10.5281/zenodo.12759731.

   ```bash
   mkdir -p <data_root>/gz && tar -xf video_dataset.tar -C <data_root>/gz/
   # -> <data_root>/gz/video_dataset/<session>/<Species>/<Intensity>/<n>.mp4
   ```

2. **Preprocess** every clip into 16 uniformly sampled frames at 224 × 224:

   ```bash
   python scripts/preprocess_video.py --dataset gz    --src <data_root>/gz/video_dataset    --dst <data_root>/gz/video_npy
   python scripts/preprocess_video.py --dataset opleg --src <data_root>/opleg/video_dataset --dst <data_root>/opleg/video_npy
   ```

3. **Build the manifest** with the released split:

   ```bash
   python scripts/build_manifest.py --gz_root <data_root>/gz/video_npy --opleg_root <data_root>/opleg/video_npy
   # -> data/manifest_video.csv (99,960 clips)
   ```

The Zenodo clips are transcoded to 854 × 480. The paper used the original-resolution recordings, so numbers
reproduced from the released clips may differ slightly.

## Pretrained backbone

Download the Kinetics-400 S3D weights `S3D_kinetics400.pt` from https://github.com/kylemin/S3D and save them as
`pretrained/S3D400.pt`. The X3D-M and MC3-18 weights are downloaded automatically when those backbones are used.

## Running the code

The whole pipeline is in `scripts/run_all.sh`. Step by step:

**1. Phase 0 – adapt and freeze the encoder.** S3D is fine-tuned on the base species (phase 0, red tilapia) with a
temporary ordinal head (Adam, lr 1e-4, batch 24, 20 epochs, cosine schedule); the checkpoint with the best validation
accuracy is kept. Features of all clips are then extracted once.

```bash
python scripts/train_phase0.py --exp_name S3D_base0 --phase 0 --backbone s3d --epochs 20 --batch_size 24 --lr 1e-4
python scripts/extract_features.py --name s3d_base0 --ckpt workspace/S3D_base0/phase0_best.pt
```

**2. AquaGauss.** Hyperparameters are selected on the validation set (final accuracy); the selected configuration
is then used, unchanged, for every species order and base species.

```bash
python scripts/select_hparams.py --feat s3d_base0                                     # -> results/hparams.json
python scripts/run_incremental.py --feat s3d_base0 --orders default reversed random   # -> results/incremental/
```

`run_incremental.py` reports, for every method and order, the final accuracy, the average incremental accuracy, the
forgetting, the time to add a species and the memory. It also runs the ablations (`abl_t<λ>_p<pool>_<score>`: power
transform 1 or 0.5, pooled shrinkage off or on, score Mahalanobis / spread-aware / full likelihood).

**3. Other base species.** To use another species as the base species, adapt the encoder on it and use the
matching order (`base6`: *O. punctatus* first, `base5`: sunfish first):

```bash
python scripts/train_phase0.py --exp_name S3D_base6 --phase 6 --backbone s3d --epochs 20 --batch_size 24 --lr 1e-4
python scripts/extract_features.py --name s3d_base6 --ckpt workspace/S3D_base6/phase0_best.pt
python scripts/run_incremental.py --feat s3d_base6 --orders base6
```

**4. Baselines.** All baselines use the same frozen features. Their hyperparameters are selected on the validation
set (average incremental accuracy) and reported as mean ± sample standard deviation over seeds 42, 43 and 44.

```bash
python scripts/select_baseline_hp.py --feat s3d_base0
python scripts/run_baselines.py --feat s3d_base0 --orders default reversed random
```

**5. Analyses and cost.**

```bash
python scripts/analyze.py --feat s3d_base0         # significance, sensitivity, cell spread, 4 vs 28 cells
python scripts/measure_cost.py --feat s3d_base0 --ckpt workspace/S3D_base0/phase0_best.pt
```

### Using AquaGauss on your own features

```python
from aquagauss import GaussMemory

mem = GaussMemory(lam=0.5, alpha=0.75, eps=0.03, beta=0.5)
for species, (x, y) in enumerate(new_species_stream):    # x: (N, d) frozen features, y: (N,) levels 0..3
    mem.add_species(x, y, species=species)              # stores 4 cells, recalibrates all cells
    pred = mem.predict(x_test)                          # intensity level of the best-matching cell
print(mem.memory_bytes() / 2**20, 'MB')
```

## Species orders

Species are numbered 0 red tilapia, 1 tilapia, 2 jade perch, 3 largemouth bass, 4 lotus carp, 5 sunfish,
6 *O. punctatus*. Orders: default 0-1-2-3-4-5-6; reversed 0-6-5-4-3-2-1; random 0-3-6-1-5-2-4; *O. punctatus* as base
species 6-0-1-2-3-4-5; sunfish as base species 5-0-1-2-3-4-6.

## Metrics

After phase t the model is tested on the test clips of all species learned so far.
A_t is the accuracy over all these clips (each species weighted by its number of test clips); the final accuracy is
A_6 and the average incremental accuracy is the mean of A_0 … A_6. Forgetting is the mean over the six earlier species
of (highest accuracy from the phase it was learned to the last phase − accuracy after the last phase).
Memory is reported in MB with 1 MB = 2^20 bytes, fp32, symmetric matrices stored as upper triangles.

## Notes

- Features are extracted with bfloat16 autocast, so re-extracted features can differ slightly (about 0.3 % relative)
  from run to run, and reproduced accuracies may differ in the second decimal.
- The released clips are transcoded to 854 × 480; the paper used the original recordings.
