# Liver and Tumour Segmentation from CT Slices

Semantic segmentation of liver and liver tumours in abdominal CT slices, on the
LiTS dataset. Every pixel is labelled background, liver or tumour.

Two architectures are implemented and compared under identical conditions: a
plain **U-Net** and an **Attention U-Net**. No segmentation framework is used;
everything is written from scratch in PyTorch.

## Results

| Model | Parameters | fg Dice | Liver Dice | Tumour Dice | Train time |
|---|---:|---:|---:|---:|---:|
| U-Net (baseline) | 7,762,531 | 0.8222 | 0.9101 | **0.7343** | 93 min |
| Attention U-Net | 7,894,783 | **0.8247** | **0.9172** | 0.7322 | 112 min |

Both trained 40 epochs on the same patient-level split with the same seed, on an
RTX 2070. The 132,252-parameter difference is exactly the four attention gates,
so nothing else varies between the two.

**The difference is not meaningful.** It is smaller than the variation between
two runs of the *same* architecture (measured at 0.0031-0.0053 fg Dice across
machines), and it changes sign depending on the metric.

The harder finding is the tumour class: precision 0.816 but recall **0.664**, and
**154 of 582** test slices containing a lesion scored a Dice of exactly zero.

## Why not accuracy

A model predicting background everywhere scores **93.4%** pixel accuracy on this
test set, because tumour is 0.12% of pixels and liver 2.15%. The trained model
scores 98.9%. That 5-point difference corresponds to foreground Dice going from
0.000 to 0.825. Model selection is therefore done on **mean foreground Dice**,
never on accuracy or loss.

## Dataset

LiTS, 2D PNG export: 131 patients, 58,638 slices at 256x256.

- 67.4% of slices contain no liver at all, so they are dropped (`skip_empty`),
  leaving 19,143 usable slices
- the split is **by patient**, not by slice: adjacent slices of one patient are
  near-duplicates, and a slice-level split leaks between train and test
- labels arrive as two overlapping binary masks per slice; they are merged with
  the lesion written **last**, because the liver mask also covers the lesion

## Layout

```
src/
  config.py            every setting in one dataclass
  data.py              loading, mask merging, patient-level split, augmentation
  model.py             DoubleConv / Down / Up / AttentionGate, both networks
  losses.py            Dice + cross-entropy
  train.py             training loop
  evaluate.py          metrics and figures
  utils/
    dataset_report.py  describes the dataset before any training
    diagram.py         architecture figures, generated from the config
    summary.py         layer table + parameter count, cross-checked
    compare.py         side-by-side comparison of runs in a group
    attention_viz.py   reads out the alpha maps from a trained gate
    reeval.py          re-run evaluation on saved checkpoints
    selftest.py        nine checks on the whole pipeline
    download.py        dataset fetch and verification
```

## Getting started

```bash
conda create -n seg python=3.12 -y && conda activate seg
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install numpy pillow matplotlib kaggle

kaggle datasets download -d andrewmvd/lits-png -p data --unzip

python -m src.utils.selftest unet             # 9 checks before you commit hours
python -m src.utils.selftest attention_unet
```

Then train and compare:

```bash
python -m src.train --arch unet --group ablation
python -m src.train --arch attention_unet --group ablation
python -m src.utils.compare --group ablation
```

Run everything from the project root with `python -m`. Do **not** mark `src/` as
a Sources Root in an IDE: it puts `src` on `sys.path`, which exposes `src/utils`
as top-level `utils` and collides with `torch.utils`.

## Notes

Model weights are gitignored. `runs/*/metrics_test.json` and `history.json` keep
the full results and the exact config for every run, so the numbers above are
reproducible from what is committed.

Known limitations are listed in `report/internship_notebook.tex`: one run per
architecture, Dice pooled over pixels rather than averaged per patient, the
8-bit export destroying the Hounsfield scale, and per-patient brightness varying
from 95 to 182 against a single fixed normalisation.
