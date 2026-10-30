# Liver and Tumour Segmentation on CT (NIfTI)

Segment the liver and liver tumours in abdominal CT scans. Every pixel is
labelled background, liver or tumour. Everything is written from scratch in
PyTorch; no segmentation framework is used.

The data is **MSD Task03_Liver**: the 131 LiTS patients as NIfTI files, in
Hounsfield units. The framework, the models and the 3D viewer come from the
internship, which worked on a PNG export of the same patients (see
[Internship](#internship-png-export) at the end).

## Project layout

```
src/                     the framework: data loading, models, training, evaluation, viewer
data/                    datasets (not in git)
  Task03_Liver/            the NIfTI dataset
  nifti_cache_512/         the same scans cut into .npy slices, built by src.nifti
                           (nifti_cache_256/ for half resolution)
  internship_png/          the internship's PNG export
runs/                    trained models, one folder per run
  nifti/                   new models
  internship_png/          ablation_2070/, ablation_laptop/
results/                 reports and figures
  nifti/                   dataset report, old models on NIfTI, viewer figures
  internship_png/          internship results, 3D Slicer files in slicer/
report/internship/       internship notebook and technical report (LaTeX + PDF)
architecture/            diagrams of the two models
```

## The data

One `.nii.gz` file is one whole CT scan: all slices stacked in 3D, plus a header
with the real size of a voxel in mm. Each voxel is a **Hounsfield unit** (HU):
air -1000, water 0, liver about 100, bone 1000 and up.

```
data/Task03_Liver/
  imagesTr/liver_0.nii.gz ... liver_130.nii.gz   131 CT scans, 512x512, 74-987 slices
  labelsTr/liver_0.nii.gz ... liver_130.nii.gz   131 label maps: 0 background, 1 liver, 2 tumour
  imagesTs/                                      70 scans WITHOUT labels - not used
  dataset.json                                   short description
```

Download (28.9 GB, CC BY-SA 4.0):

```bash
curl -L -o data/Task03_Liver.tar https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar
tar -xf data/Task03_Liver.tar -C data         # MD5 a90ec6c4aa7f6a3d087205e23d4e6397
```

The full study (spacing, HU per class, lesion sizes, split) is in
`results/nifti/dataset_report.txt`. In short: slices are 0.7-5 mm thick, liver
is 2.2% of voxels and tumour 0.12%, and two thirds of the 1,037 lesions are
smaller than 1 mL.

## How the data reaches the model

1. **Once:** `python -m src.nifti` cuts every scan into its 2D slices and saves
   them as `.npy` in `data/nifti_cache_512/` (~45 min, resumable). Values stay
   in HU. This exists because a `.nii.gz` must be unpacked whole, up to 1 GB,
   to read a single slice. `--resolution half` makes a 256 px cache instead.
2. **Every training step:** a slice is read from the cache, its HU are clipped
   to the window **[-17, 201]** (the range of liver and tumour voxels) and
   scaled to 0..1, then augmented and normalised.

Slices with no liver are skipped, as before (`skip_empty`). The split is by
patient with seed 42: 91 train, 20 validation, 20 test.

## Usage

Run everything from the project root with `python -m`, in the `seg` conda env.

```bash
python -m src.utils.viewer --scan 16                  # look at a scan + its labels, no model

python -m src.nifti                                   # build the 512 slice cache, once
python -m src.utils.nifti_report                      # dataset report -> results/nifti/
python -m src.utils.selftest unet                     # 9 checks before training

python -m src.train --arch unet                       # 512 px -> runs/nifti/baseline/unet_<time>/
python -m src.train --arch attention_unet
python -m src.utils.compare --group nifti/baseline    # side-by-side table and figure

python -m src.utils.reeval --group nifti/baseline     # re-score, incl. Dice per patient
python -m src.utils.viewer --volume 16                # a trained model on one patient, 2D + 3D
python -m src.utils.view3d --volume 16 --nifti        # figures + .nii.gz for 3D Slicer
```

The data format defaults to NIfTI. `--group` picks the folder under `runs/`;
`train` saves into `runs/nifti/<group>/` (default group `baseline`).

**Resolution.** Training defaults to the full 512x512. `--resolution
full|half|quarter|eighth` (512/256/128/64 px) on `train` and `src.nifti`; on a
small GPU use e.g. `--resolution half` for both. A smaller cache is made from a
larger one in about a minute; otherwise the NIfTI files are read (~45 min).
Measured on an RTX 3050 Laptop (4 GB), one training step with mixed precision:

| | disk (all slices) | U-Net | Attention U-Net |
|---|---:|---|---|
| full, 512 | ~46 GB | batch ≤ 8 (3.1 GB) | batch ≤ 4 (1.9 GB) |
| half, 256 | ~11.5 GB | batch 16 fits (1.6 GB) | batch 16 fits (1.9 GB) |
| quarter, 128 | ~2.9 GB | trivial | trivial |

At 512 a step costs about 4x the time per slice of 256. Past 4 GB Windows moves
GPU memory into system RAM and training slows ~10x instead of failing, so watch
the time per epoch.

**The viewer** has two uses:
- `viewer --scan 16` shows the data only: `liver_16.nii.gz` and its labels, all
  slices, at 512, with no model. A slider scrolls through the scan, the planes
  show it from the front and side, and the right panel spins the liver and
  tumours in 3D. `--window -1000 1000` changes which HU are shown.
- `viewer --volume 16` with a trained model runs it on every slice of that
  patient and shows CT, truth and prediction, and where they disagree. Only
  slices containing liver are loaded; `--all-slices` loads all.

For NIfTI the slice thickness is read from the scan header, and `view3d --nifti`
exports the CT in HU at true voxel size. `viewer --from-export 16` reopens such
an export without the model.

**Models.** `--arch` picks one of four:

| `--arch` | Encoder | Parameters |
| --- | --- | --- |
| `unet`, `attention_unet` | trained from scratch, 32 base filters, depth 4 | 7.8 M, 7.9 M |
| `unet_r50`, `attention_unet_r50` | ResNet-50 pretrained on ImageNet (`--encoder-weights none` = random start) | 32.5 M, 34.4 M |

`--in-slices 3` gives any model **2.5D input**: the slice and the one above and
below as three channels, the label staying the middle slice's (NIfTI only). It
also fills the three colour channels an ImageNet encoder expects. The ImageNet
weights (~100 MB) are downloaded on the first run. At 512 the ResNet-50 models
train about as fast as the from-scratch ones and need ~3.7 GB at batch 8.

```bash
python -m src.train --arch unet_r50 --in-slices 3 --group r50
python -m src.train --arch attention_unet_r50 --in-slices 3 --group r50
```

**Adding a model.** Add the class to `ARCHES` in `src/model.py` and its name to
`ARCH_NAMES` in `src/config.py`. The self-test's parameter check compares
against `src/utils/diagram.py`, which only knows the two from-scratch U-Nets.

## Scores

A model predicting background everywhere scores about 93% pixel accuracy,
because tumour is 0.12% of pixels and liver about 2%, so accuracy is never used.

- **Main score: tumour Dice per patient**, LiTS rules. Dice is computed for
  each test patient, then averaged (median reported too). A patient with no
  tumour scores 1 if none is predicted and 0 if some is. Pooled Dice, over all
  pixels together, is reported as well, but big tumours dominate it.
- **Best epoch:** training keeps the epoch with the best validation tumour
  Dice per patient (`--select-metric tumor_case`, the default;
  `--select-metric dice_fg` for the old rule).
- **3D clean-up** (`src/postprocess.py`, no retraining): the slices are stacked
  back into 3D, only the largest connected liver piece is kept, and tumour
  outside it is removed. `reeval` and `train` report every score raw and
  cleaned; `reeval --min-tumor-voxels N` also drops tumour pieces under N voxels.

## Future work

- **Tumour Dice, the main goal.** On the internship PNGs the tumour was the weak
  class: Dice 0.73 pooled over all pixels but only about 0.4 averaged per
  patient, recall 0.66. Most lesions are under 1 mL, and those are the ones
  missed.
- **More models,** which ones to be decided with the supervisor. A new
  architecture goes into `ARCHES` in `src/model.py`.
- **The 3D viewer:** improve the interactive viewer.

## Internship (PNG export)

The internship trained both models on a 2D PNG export of LiTS (8-bit,
256x256, each slice stretched by its own min and max). Data in
`data/internship_png/`, models in `runs/internship_png/`, results in
`results/internship_png/`, report in `report/internship/`.

| Model | Parameters | fg Dice | Liver Dice | Tumour Dice | Train time |
|---|---:|---:|---:|---:|---:|
| U-Net | 7,762,531 | 0.8222 | 0.9101 | 0.7343 | 93 min |
| Attention U-Net | 7,894,783 | 0.8247 | 0.9172 | 0.7322 | 112 min |

40 epochs each on an RTX 2070, same split and seed. The difference is smaller
than the variation between two runs of the same model (0.003-0.005 fg Dice).
Tumour recall was the weak point: 0.664, and 154 of 582 test slices with a
lesion scored Dice 0.

These models learnt PNG-style images. On the NIfTI test patients converted the
same way they score as on the PNGs (fg Dice 0.824 / 0.824), on HU input about 0,
so NIfTI models are trained from scratch:

```bash
python -m src.utils.old_models_on_nifti     # -> results/nifti/old_models_on_nifti.txt
```

To use the internship tools on them, pass the group and `--png` where needed.
They need the PNG export in `data/internship_png/` (and `old_models_on_nifti` a
256 px NIfTI cache, `python -m src.nifti --resolution half`):

```bash
kaggle datasets download -d andrewmvd/lits-png -p data/internship_png --unzip   # 3.4 GB
python -m src.utils.selftest unet --png
python -m src.train --arch unet --data-format png --group ablation   # -> runs/internship_png/ablation/
python -m src.utils.viewer --group internship_png/ablation_2070 --volume 16
python -m src.utils.dataset_report                                  # -> results/internship_png/
```

## Notes

Model weights and data are gitignored. Each run's `history.json` and
`metrics_test.json` keep the exact config and results.

Do **not** mark `src/` as a Sources Root in an IDE: it puts `src` on
`sys.path`, which exposes `src/utils` as top-level `utils` and collides with
`torch.utils`.
