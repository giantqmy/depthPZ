<div align="center">
  <h1 style="border-bottom: none; margin-bottom: 0px">Gated RCCB Stereo</h1>
  <h2 style="border-top: none; margin-top: 3px;">Cross-spectral Gated-RGB Stereo Depth Estimation</h2>
  <h3 style="border-top: none; margin-top: 3px;">CVPR 2024</h3>


  [**Samuel Brucker**](https://scholar.google.de/citations?user=TUmqR0kAAAAJ&hl=de) ·
  [**Stefanie Walz**](https://scholar.google.com/citations?user=ZeHkzXwAAAAJ) ·
  [**Mario Bijelic**](https://scholar.google.com/citations?user=aJV5jRYAAAAJ) ·
  [**Felix Heide**](https://www.cs.princeton.edu/~fheide/)
  <br>

  <a href="https://arxiv.org/abs/2405.12759">
    <img src="https://img.shields.io/badge/arXiv-2405.12759-red" alt="arXiv">
  </a>
  <a href="https://light.princeton.edu/publication/gatedrccbstereo/">
    <img src="https://img.shields.io/badge/Project_Page-Gated%20RCCB%20Stereo-green" alt="Project Page">
  </a>
  <a href="https://light.princeton.edu/datasets/automated_driving_dataset/#long-range-stereo-extension">
    <img src="https://img.shields.io/badge/Dataset-Download-blue" alt="Download Dataset">
  </a>

</div>

We propose Gated RCCB Stereo, a novel stereo-depth estimation method that is capable of exploiting multi-modal multi-view depth cues from RCCB and gated stereo images. We combine complementary cross-spectral features within an iterative stereo matching network with our proposed cross-spectral matching layer, utilizing intermediate depth estimations for accurate feature-registration.

![Demo](assets/demo.gif)

## Environment Setup

Create and activate the conda environment:
```bash
conda create --name gatedrccbstereo python=3.8.8
conda activate gatedrccbstereo
```

Install dependencies:
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install timm
pip install smt==1.3.0
pip install openmim mmengine
mim install mmcv-full
pip install mmsegmentation==0.30.0
pip install einops
pip install ittr-pytorch
pip install cupy-cuda117
pip install scikit-image
pip install hydra-core
pip install tensorboardX
```
## Data Preparation

### 1) Download the dataset
[Download](https://light.princeton.edu/datasets/automated_driving_dataset/#long-range-stereo-extension) Gated RCCB Stereo Dataset **(Long-Range Stereo Extension)** 

### 2) Prepare dataset PKL files
Generate the dataset `.pkl` files using the provided split lists:
```bash
python dataset/create_pkl_gatedstereo.py --split_file splits/train_gatedstereo.txt --data_path /path/to/dataset
```

Repeat for the test splits:
```bash
python dataset/create_pkl_gatedstereo.py --split_file splits/test_day_gatedstereo.txt --data_path /path/to/dataset
python dataset/create_pkl_gatedstereo.py --split_file splits/test_night_gatedstereo.txt --data_path /path/to/dataset
```

### 3) Training
start training with:
```bash
python trainer.py
```
For training with reduced GPU memory requirements, train without the high-resolution prediction stage by lowering the resolution of the RCCB images with:
```bash
python trainer.py dataset.pipeline.cfg_list.4.scales.rccb=[512,1024]
```

### 4) Evaluation
Download pre-trained weights [here](https://drive.google.com/file/d/1x2vMwa_uTBxjirB73VciZXSsrm_cn2my/view?usp=sharing). \
Place them into 'checkpoints' folder.

Then start evaluation of pre-trained weights with:
```bash
python evaluator.py 
```
Or evaluate your own model with:
```bash
python evaluator.py resume_path=path/to/checkpoint.pt
```

## Citations
If you find Gated RCCB Stereo useful in your research or projects, please cite our work:
```bash
@inproceedings{bruckergatedrccbstereo,
title={Cross-Spectral Depth Estimation from RGB and NIR-Gated Stereo Images}, 
author={Samuel Brucker and Stefanie Walz and Mario Bijelic and Felix Heide},
journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2024}
}
```



## RGB-left / NIR-right pretrained fine-tuning

### RGB/NIR + DoLP comparison

The separate `train_rgb_nir_lidar_dolp_finetune.py` reuses the baseline dataset,
calibration, loss and metrics. Run from `GatedRCCBStereo` with the same experiment
settings as the baseline:

```powershell
& D:\anaconda\envs\pytorch\python.exe -u train_rgb_nir_lidar_dolp_finetune.py --epochs 20 --limit 50 --seed 17 --output rgb_nir_lidar_dolp_finetune_depth_new
```

The four MPViT encoders have independent parameters: left RGB and RGB DoLP
(3 channels each), right NIR (checkpoint channels, normally 5 repeated grayscale
channels) and NIR DoLP (1 channel). Each side has an independent iAFF fusion
block before AGCL. AoLP is unused. Default DoLP directories are
`I:\KEDADATA\data\fog\RGB_color_polar_processed\RGBDOLP_color` and
`I:\KEDADATA\data\fog\NIRuseI4\output_dolp`; override with `--rgb-dolp-dir`
and `--nir-dolp-dir`. These must be uint8 PNGs in the raw camera coordinates,
paired as `<timestamp>_dolp_color_rgb.png` and `<timestamp>_dolp_nir.png`.
They receive the corresponding left/right rectification maps.

Both image encoders and the stereo trunk load from `--checkpoint` (default
`gatedrccbstereo.pt`). DoLP encoders copy their respective image encoder values;
the NIR DoLP stem sums input-channel weights to adapt to single-channel input.
Both iAFF blocks copy the original `csm.fusion.*` weights from
`--fusion-checkpoint` (default `gatedrccbstereo.pt`), without sharing parameters.
The original iAFF has 128 channels; trainable 256-to-128 and 128-to-256
projections adapt it to the 256-channel stereo features. This is a feature-level
adaptation, not a restoration of the original CSM geometry pipeline.

Add `--from-scratch` to train without loading either checkpoint:

```powershell
& D:\anaconda\envs\pytorch\python.exe -u train_rgb_nir_lidar_dolp_finetune.py --from-scratch --epochs 20 --seed 17 --checkpoint-encoders
```

All four encoders and both iAFF blocks retain their own initialization, with
single-channel primary NIR input as in `train_rgb_nir_lidar.py`. Fusion keeps
its context-selection and zero-residual projection initialization.
`--checkpoint` and `--fusion-checkpoint` are ignored in this mode;
`--freeze-encoders` cannot be combined with it. The intensity control also
accepts `--from-scratch`. Scratch runs default to the separate output directory
`rgb_nir_lidar_<auxiliary>_scratch_depth_new`; `--output` overrides it.
Saved metadata records the initialization mode and null checkpoint paths.

Fusion uses an additive residual with zero-initialized output projections,
so initial evaluation features match the baseline. The first optimizer step
updates those projections; subsequent steps propagate gradients to DoLP/iAFF.
Use `--checkpoint-encoders` on GPUs with limited memory to recompute encoder
activations during backward; RNG state is preserved and BatchNorm statistics
stay frozen. This changes memory/runtime rather than the intended objective.
All four encoders train by default; `--freeze-encoders` freezes all four while
leaving fusion trainable. BatchNorm statistics stay frozen as in the baseline.

Seed defaults to **17**. The baseline sorted RGB/NIR/depth sample list and split
algorithm are unchanged; missing DoLP files cause an error instead of changing
the sample set. With the same data, `--limit 50 --val-fraction 0.1` gives the
same 45 training / 5 validation samples. Extra branch initialization preserves
the baseline Torch RNG state, but extra stochastic encoder operations mean
training trajectories are not bitwise identical. Use the same source checkpoint,
epochs, learning rate and split settings for comparisons.

Outputs use a separate directory and include `train.txt`, `val.txt`,
`run_config.json`, losses/MAE, history and `latest.pt` with all four encoders and
both fusers. `--check-data-only` writes manifests and checks one sample without
training. Fusion checkpoint reconstruction requires the new model branches
(`enable_dolp_fusion`) and `dolp_fusion=True`; the existing two-image inference
entry point does not evaluate DoLP fusion checkpoints.

### Four-encoder wavelet PIM fusion

`train_rgb_nir_lidar_dolp_pim_finetune.py` uses the same data pairing,
rectification, split, losses and training loop as the DoLP iAFF trainer, with
independent wavelet PIM blocks for left RGB/DoLP and right NIR/DoLP.

The PIM core is ported from `PZDualPIMStage` in
`ultralytics-hmoe-joint-scratch/ultralytics/nn/modules/pz.py`: branch-specific
PoolFormer-style pooling/MLP residuals, Haar low/high-frequency softmax-gated
fusion, and a shared fusion residual added to both branches. Its original
layer scales (`1e-5`) and fusion scale (`0.5`) are retained. The existing
MPViT encoders provide 256-channel features, so no YOLO ConvNeXt stage is
added. `networks/pim_fusion.py` concatenates the two enhanced outputs and
projects 512 channels back to 256 for stereo matching; that trainable
projection initially averages the branches. This adapter changes the initial
features and does not preserve the baseline's initial predictions. SAFFConcat
and HMoE are not part of this PIM experiment.

Run from `GatedRCCBStereo` for pretrained encoder/stereo initialization:

```powershell
& D:\anaconda\envs\pytorch\python.exe -u train_rgb_nir_lidar_dolp_pim_finetune.py --epochs 20 --seed 17 --checkpoint-encoders
```

Initialize the complete model from scratch:

```powershell
& D:\anaconda\envs\pytorch\python.exe -u train_rgb_nir_lidar_dolp_pim_finetune.py --from-scratch --epochs 30 --seed 17 --checkpoint-encoders
```

Both PIMs always start with independent new weights. In fine-tuning mode,
`--checkpoint` initializes the baseline encoders and stereo modules; DoLP
encoders copy the corresponding image encoder weights, with the NIR stem
adapted to one channel. PIM does not load iAFF weights and has no
`--fusion-checkpoint` option. In scratch mode no checkpoint is read or encoder
copied, and primary NIR also uses one channel. Encoder/PSP BatchNorm statistics
remain frozen. The new PIM blocks update their own spatial BatchNorm statistics
during training, matching the source PIM; freezing their initial 0/1 statistics
can leave scratch predictions saturated at the disparity clamp with zero
gradients. PIM is outside encoder activation checkpointing, so its statistics
update only once per forward. Validation uses the learned running statistics.
`--freeze-encoders` is available only for
fine-tuning; `--limit` and `--check-data-only` are also supported.

Default output directories are `rgb_nir_lidar_dolp_pim_finetune_depth_new`
and `rgb_nir_lidar_dolp_pim_scratch_depth_new`. Metadata identifies
`fusion_type="pim"` and architecture `rgb_nir_dolp_4encoders_wavelet_pim256_v1`;
`latest.pt` includes all four encoders, both PIMs, optimizer and history.
Reconstruct these checkpoints with `enable_dolp_fusion(fusion_type="pim")`
and load the entire model state strictly; `--checkpoint` is baseline
initialization, not full training resume. The two-image inference entry point
does not evaluate these four-input checkpoints. The shared loop rejects
non-finite gradients before updating parameters and reports epoch/step/sample.

### Four-encoder intensity control

To test the effect of polarization information while keeping the same model
capacity, run the intensity control with matching experiment settings:

```powershell
& D:\anaconda\envs\pytorch\python.exe -u train_rgb_nir_lidar_intensity_finetune.py --epochs 20 --limit 50 --seed 17 --checkpoint-encoders --output rgb_nir_lidar_intensity_finetune_depth_new
```

This replaces left RGB DoLP with the same three-channel RGB S0 image and right
NIR DoLP with the same single-channel NIR S0 image. Both auxiliary images reuse
the primary views' rectification. The main NIR encoder still receives the
checkpoint's channel count (normally 5 repeated channels). No DoLP/AoLP files
are required. All four encoders and both iAFF fusers remain independent and use
the identical initialization and shared training loop as the DoLP experiment.
Seed 17 and the same data/limit preserve train/validation membership.

Outputs use the separate `rgb_nir_lidar_intensity_finetune_depth_new` directory.
Manifests record RGB/NIR S0 paths again as the auxiliary sources;
`run_config.json`, `history.json` and `latest.pt` record
`auxiliary_input="intensity"` and `uses_dolp_data=false`. Architecture identifiers
and model state keys retain their original DoLP names for exact compatibility;
these names describe the branch slots, not the control experiment's input data.
The original two-encoder training entry point is unchanged. Use
`--check-data-only --limit 50 --seed 17 --output <check-directory>` to check the
control data and manifests without training.

### Baseline training

Run from this repository with the existing `pytorch` environment:

```powershell
& D:\anaconda\envs\pytorch\python.exe -u train_rgb_nir_lidar_finetune.py --epochs 5 --limit 100 --log-interval 1
```

The default checkpoint is `gatedrccbstereo.pt`. Both image encoders (including
their feature necks), stereo attention/update modules, disparity decoder and
learned upsampling weights are loaded strictly and fine-tuned. No encoder is
reset or frozen by default. The original checkpoint uses RGB with 3 channels
and gated input with 5 channels, so the right NIR image is repeated into 5
channels. A checkpoint with a single-channel NIR encoder is also supported.
Same-side modality fusion remains disabled.

Default data directories are `I:\KEDADATA\data\fog\RGBS0`, `NIRS0` and
`depth_new`; LiDAR PNG values are converted from millimetres to metres
with `--depth-scale 0.001`. The seeded split uses 90% for training and 10% for
validation (`--limit 100`: 90 training, 10 validation). Remove `--limit` to use
all matched triplets. Batch size is one; learning rate defaults to `1e-5`.

Outputs go to `rgb_nir_lidar_finetune_run`: per-epoch `loss.txt`, `history.json`,
`calibration.json`, and `latest.pt`. The loss retains LiDAR-derived depth and
disparity BerHu supervision. The existing `train_rgb_nir_lidar.py` remains the
scratch-training entry point.

The fine-tuning script also writes `train.txt` and `val.txt` in the output
folder. Each file records the exact timestamp prefixes and absolute RGB, NIR
and depth paths assigned to that split, together with the seed and split
settings. The files capture train/validation membership; training sample order
is reshuffled each epoch. Use a separate `--output` directory to preserve split
manifests from different runs. The default depth folder is the regenerated
`I:\KEDADATA\data\fog\depth_new`.

Fine-tuning progress also prints `depth_mae_m` (metres) and `disp_mae_px`
(pixels), computed from the final output at valid LiDAR supervision pixels.
These are plain absolute errors, without BerHu or intermediate-iteration
weights, and do not change the training objective. Per-epoch train/validation
MAE columns are saved to `loss.txt`, `history.json` and checkpoint history.
Epoch MAE is the total absolute error divided by the total valid pixel count;
`NA` means no valid pixels. Training metrics use predictions from each training
step, while validation metrics evaluate the fixed model after that epoch.
Disparity targets are derived from LiDAR depth and calibration, not independent
disparity annotations. Metrics reuse the model's supervision targets, including
its existing depth clamp to 1--220 metres. Existing runs need a restart to enable these metrics;
old loss logs cannot be used to reconstruct MAE.

Test the fine-tuned checkpoint with raw left RGB and right NIR images. The
inference script applies the same camera undistortion and stereo rectification
as training, and saves the rectified pair alongside the disparity outputs:

```powershell
& D:\anaconda\envs\pytorch\python.exe run_gatedrccbstereo_stereo.py --checkpoint .\rgb_nir_lidar_finetune_run\latest.pt --left path/to/left_rgb.png --right path/to/right_nir.png --output .\rgb_nir_lidar_finetune_test
```

The default inputs are the raw 640x512 pair in `depth-视差图代码/data/10`.
Pass `--skip-rectification` only if both supplied inputs have already been
rectified with this same calibration.

# qmy

python run_gatedrccbstereo_stereo.py --checkpoint "D:\Desktop\txt\txt2\rewrite\DEPTH\GatedRCCBStereo\rgb_nir_lidar_dolp_finetune_depth_new\latest.pt" --left "D:\Desktop\txt\txt2\rewrite\DEPTH\depth-视差图代码\data\5\left.jpg" --right "D:\Desktop\txt\txt2\rewrite\DEPTH\depth-视差图代码\data\5\right.jpg" --output "D:\Desktop\txt\txt2\rewrite\DEPTH\depth-视差图代码\data\5\dolp_finetune"

python run_gatedrccbstereo_stereo.py --checkpoint ".\rgb_nir_lidar_finetune_run\latest.pt" --left "D:\Desktop\txt\txt2\rewrite\DEPTH\depth-视差图代码\data\val\2000-01-01-12-22-49_747_S0_rgb.png" --right "D:\Desktop\txt\txt2\rewrite\DEPTH\depth-视差图代码\data\val\2000-01-01-12-22-49_747_S0_nir.png" --output "..\depth-视差图代码\data\10\gatedrccbstereo_latest1_50_result_depthold_校正"



python run_gatedrccbstereo_stereo.py --checkpoint "D:\Desktop\txt\txt2\rewrite\DEPTH\GatedRCCBStereo\rgb_nir_lidar_dolp_finetune_depth_new\latest.pt" --left "D:\Desktop\txt\txt2\rewrite\DEPTH\depth-视差图代码\data\val\2000-01-01-12-22-49_747_S0_rgb.png" --right "D:\Desktop\txt\txt2\rewrite\DEPTH\depth-视差图代码\data\val\2000-01-01-12-22-49_747_S0_nir.png" --output "..\depth-视差图代码\data\10\dolpall_finetune"

python -u train_rgb_nir_lidar_finetune.py 
  --depth-dir "I:\KEDADATA\data\fog\depth_new" 
  --epochs 20 
  --limit 50 
  --seed 17 
  --output rgb_nir_lidar_finetune_depth_new
