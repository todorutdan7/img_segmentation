"""Do the internship's PNG-trained models work on the NIfTI scans?

    python -m src.utils.old_models_on_nifti
    python -m src.utils.old_models_on_nifti --group internship_png/ablation_laptop

Each model is scored on its own 20 test patients, read three ways:

    png             the internship result, from the run's metrics_test.json
    nifti as png    the NIfTI slices converted the way the png export was made:
                    each 256 px slice stretched from its own darkest to its own
                    brightest pixel. Same patients and pixels as the pngs, so a
                    match shows the NIfTI is read correctly
    nifti as HU     HU clipped to the window, the input a model trained on NIfTI
                    gets. The png models never saw input like this

WRITES results/nifti/old_models_on_nifti.txt
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from src.config import get_device, results_dir, RUNS_DIR
from src.data import split_pairs
from src.evaluate import accumulate_confusion, metrics_from_confusion
from src.model import load_checkpoint
from src.nifti import NiftiSliceDataset, nifti_slices


class PngLikeSlices(NiftiSliceDataset):
    """NIfTI slices scaled the way the png export scaled them"""

    def scale(self, hu):
        lo, hi = hu.min(), hu.max()
        return np.clip((hu - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def nifti_test_loader(cfg, dataset_cls):
    """the model's own test patients, from the nifti cache at its resolution"""
    items, _, volumes, cache_dir = nifti_slices(cfg, verbose=False)
    (_, _, test), _, _ = split_pairs(items, cfg.val_fraction, cfg.test_fraction, cfg.seed,
                                     True, key=lambda p: p[0])
    ds = dataset_cls(test, cache_dir, volumes, hu_window=cfg.hu_window)
    return DataLoader(ds, batch_size=16, shuffle=False)


def dice_row(m):
    return [m["mean_dice_fg"], m["per_class"]["liver"]["dice"], m["per_class"]["tumor"]["dice"]]


def main():
    p = argparse.ArgumentParser(description="the internship's png models on the nifti scans")
    p.add_argument("--group", default="internship_png/ablation_2070",
                   help="folder under runs/ holding the png-trained models")
    a = p.parse_args()

    ckpts = sorted((Path(RUNS_DIR) / a.group).glob("*/best_model.pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints under runs/{a.group}")
    device = get_device()

    L = ["The internship's PNG-trained models on the NIfTI test patients",
         f"runs/{a.group}, each model on its own 20 test patients", "",
         f"  {'':16}{'input':<16}{'fg Dice':>9}{'liver':>8}{'tumour':>8}"]
    for ckpt in ckpts:
        model, cfg, _ = load_checkpoint(ckpt, device)
        png = json.loads((ckpt.parent / "metrics_test.json").read_text())
        rows = [("png", dice_row(png))]
        ncfg = replace(cfg, data_format="nifti")
        for label, cls in (("nifti as png", PngLikeSlices), ("nifti as HU", NiftiSliceDataset)):
            print(f"[{cfg.arch}] {label} ...", flush=True)
            cm = accumulate_confusion(model, nifti_test_loader(ncfg, cls), device, cfg.num_classes)
            rows.append((label, dice_row(metrics_from_confusion(cm, cfg.class_names))))
        for i, (label, r) in enumerate(rows):
            L.append(f"  {cfg.arch if i == 0 else '':<16}{label:<16}"
                     + "".join(f"{v:>9.4f}" if j == 0 else f"{v:>8.4f}" for j, v in enumerate(r)))
        L.append("")

    L += ["nifti as png ~ png: the NIfTI files hold the same patients and are read correctly.",
          "nifti as HU ~ 0: these models learnt per-slice stretched 8-bit images and do not",
          "understand Hounsfield units. Models for the NIfTI data have to be trained on it."]
    text = "\n".join(L)
    print("\n" + text)
    out = results_dir("nifti") / "old_models_on_nifti.txt"
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
