"""Two stages: find the liver, then look for tumour in a zoomed box around it.

Stage 1 is any trained run (e.g. runs/nifti/r50/unet_r50_2.5d_lr1e-4). Its
predicted liver gives each test patient a box. Stage 2 is trained with

    python -m src.train --arch unet_r50 --in-slices 3 --lr 1e-4 --group r50 \\
        --crop-liver runs/nifti/r50/unet_r50_2.5d_lr1e-4

and sees every slice cropped to that box and enlarged to the training size.
The liver's box is usually ~288 of 512 px wide, so lesions appear ~1.8x larger
per side, and nothing far outside the liver can become false tumour.

    the box        one square per patient over all its slices (so every slice
                   of a patient gets the same zoom and the 2.5D neighbours
                   line up), the liver's in-plane extent plus --crop-margin
                   (10%) on each side
    training       boxes around the TRUE liver, shifted and resized a little at
                   random so stage 2 copes with an imperfect stage-1 box;
                   validation uses the true boxes unchanged
    test           boxes around stage 1's PREDICTED liver (after its 3D
                   clean-up), from <stage-1 run>/liver_boxes_test.json, which
                   train.py writes the first time (or: python -m src.cascade
                   --stage1 <run>). Stage 1 must have the same test patients,
                   i.e. the same split and fold.
    scoring        the stage-2 output is shrunk back to the box and pasted into
                   an empty full slice, then scored exactly like every other
                   run: same slices, same full-resolution truth, so a box that
                   cuts off liver or tumour costs what it should.

Validation (which picks the best epoch) is scored on the crops, which is
cheaper and close enough for choosing an epoch.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.nifti import NiftiSliceDataset

BOX_FILE = "liver_boxes_test.json"


# ------------------------------------------------------------------ boxes

def mask_box(mask):
    """in-plane box of a (z, y, x) or (y, x) mask as fractions of the slice,
    [y0, y1, x0, x1], or None if the mask is empty"""
    m = mask.any(0) if mask.ndim == 3 else mask
    ys, xs = np.where(m.any(1))[0], np.where(m.any(0))[0]
    if not len(ys):
        return None
    h, w = m.shape
    return [float(ys[0] / h), float((ys[-1] + 1) / h), float(xs[0] / w), float((xs[-1] + 1) / w)]


def square_box(b, margin, jitter=False, rng=random):
    """a box -> the square crop actually used, still as fractions.
    jitter (training): side x0.9..1.15 and centre moved up to 5% of the side"""
    y0, y1, x0, x1 = b
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    side = max(y1 - y0, x1 - x0) * (1 + 2 * margin)
    if jitter:
        side *= rng.uniform(0.9, 1.15)
        cy += rng.uniform(-0.05, 0.05) * side
        cx += rng.uniform(-0.05, 0.05) * side
    side = min(side, 1.0)
    # moved, not shrunk, where it would stick out of the slice
    y0 = min(max(cy - side / 2, 0.0), 1.0 - side)
    x0 = min(max(cx - side / 2, 0.0), 1.0 - side)
    return [y0, y0 + side, x0, x0 + side]


def px_box(fr, size):
    """fractions -> pixel rows/cols (y0, y1, x0, x1) of a size x size slice"""
    s = min(size, max(1, round((fr[1] - fr[0]) * size)))
    y0 = min(max(round(fr[0] * size), 0), size - s)
    x0 = min(max(round(fr[2] * size), 0), size - s)
    return y0, y0 + s, x0, x0 + s


def true_boxes(cache_dir, volumes, verbose=True):
    """every labelled scan's true liver box (liver + tumour), computed once from
    the label cache and kept in <cache>/liver_boxes.json"""
    path = Path(cache_dir) / "liver_boxes.json"
    boxes = json.loads(path.read_text()) if path.exists() else {}
    todo = [k for k, v in volumes.items() if v.get("split") == "train" and k not in boxes]
    if todo:
        from src.nifti import open_volume
        if verbose:
            print(f"[crop] true liver boxes for {len(todo)} scans, once -> {path}")
        for k in todo:
            v = volumes[k]
            zs = [z for z in range(v["n_slices"]) if v["liver_px"][z] + v["tumor_px"][z]]
            _, lab = open_volume(cache_dir, v)
            boxes[k] = mask_box(np.asarray(lab[zs[0]:zs[-1] + 1]) > 0)
        path.write_text(json.dumps(boxes))
    return boxes


# ------------------------------------------------------------------ stage 1

def stage1_boxes(run_dir, device=None, margin=0.1, verbose=True):
    """run stage 1 on its test patients and save the box of its predicted liver
    (after the 3D clean-up) to <run_dir>/liver_boxes_test.json. Also reports
    how much true liver and tumour falls inside the box stage 2 will crop."""
    from src.data import get_dataloaders, slice_keys
    from src.evaluate import predicted_volumes
    from src.model import load_checkpoint
    from src.postprocess import clean_volume
    from dataclasses import replace

    run_dir = Path(run_dir)
    model, cfg, _ = load_checkpoint(run_dir / "best_model.pt", device)
    if cfg.crop_liver:
        raise SystemExit(f"{run_dir} is itself a stage-2 run; give a full-slice run as stage 1")
    cfg = replace(cfg, num_workers=0)
    _, _, test_loader, _ = get_dataloaders(cfg, verbose=False)
    if verbose:
        print(f"[crop] stage 1 = {run_dir.name}: predicting the liver of its "
              f"{len(set(slice_keys(test_loader.dataset)))} test patients")
    out, cover = {}, {}
    for k, gt, pr in predicted_volumes(model, test_loader, model_device(model),
                                       slice_keys(test_loader.dataset)):
        b = mask_box(clean_volume(pr) > 0)
        if b is None:
            print(f"[crop] {k}: stage 1 found no liver - the whole slice is used")
            b = [0.0, 1.0, 0.0, 1.0]
        out[k] = b
        y0, y1, x0, x1 = px_box(square_box(b, margin), gt.shape[1])
        inside = np.zeros(gt.shape[1:], bool)
        inside[y0:y1, x0:x1] = True
        cover[k] = {c: (float((gt[:, inside] == v).sum() / max((gt == v).sum(), 1))
                        if (gt == v).any() else None)
                    for c, v in (("liver", 1), ("tumour", 2))}
        cover[k]["zoom"] = round(gt.shape[1] / (y1 - y0), 2)
    test_loader.dataset.close()
    del model
    if device is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    res = {"stage1": str(run_dir), "image_size": cfg.image_size, "boxes": out,
           "check_with_margin": margin, "inside_box": cover}
    (run_dir / BOX_FILE).write_text(json.dumps(res, indent=1))
    if verbose:
        print_cover(cover)
        print(f"[crop] saved {run_dir / BOX_FILE}")
    return res


def model_device(model):
    return next(model.parameters()).device


def print_cover(cover):
    zoom = [c["zoom"] for c in cover.values()]
    tum = {k: c["tumour"] for k, c in cover.items() if c["tumour"] is not None}
    liv = [c["liver"] for c in cover.values() if c["liver"] is not None]
    print(f"[crop] zoom per side: median {np.median(zoom):.2f} (range {min(zoom):.2f}-{max(zoom):.2f})")
    print(f"[crop] true liver inside the box: worst patient {min(liv):.1%}")
    if tum:
        worst = min(tum, key=tum.get)
        cut = [k for k, v in tum.items() if v < 0.999]
        print(f"[crop] true tumour inside the box: worst {worst} {tum[worst]:.1%}; "
              f"{len(cut)} of {len(tum)} patients lose some" + (f" ({', '.join(cut)})" if cut else ""))


def load_stage1_boxes(run_dir, keys):
    path = Path(run_dir) / BOX_FILE
    if not path.exists():
        raise SystemExit(f"no {path}. Make it with  python -m src.cascade --stage1 {run_dir}  "
                         f"(train.py does this by itself)")
    boxes = json.loads(path.read_text())["boxes"]
    missing = sorted(set(keys) - set(boxes))
    if missing:
        raise SystemExit(f"stage 1 ({run_dir}) has no box for {len(missing)} test patient(s), "
                         f"e.g. {missing[:3]}: it was tested on other patients. Use a stage-1 "
                         f"run with the same split and fold.")
    return boxes


# ------------------------------------------------------------------ stage 2 data

class LiverCropDataset(NiftiSliceDataset):
    """NiftiSliceDataset whose slices are cropped to a box around the liver and
    enlarged back to image_size. boxes: {key: [y0, y1, x0, x1] fractions}.
    With augment on, every read moves and resizes the box a little."""

    def __init__(self, items, cache_dir, volumes, boxes, margin=0.1, **kw):
        super().__init__(items, cache_dir, volumes, **kw)
        self.boxes, self.margin = boxes, margin

    def _box(self, key, size):
        return px_box(square_box(self.boxes[key], self.margin, jitter=self.augment), size)

    def _load(self, i):
        key, z = self.pairs[i]
        if key not in self._maps:
            from src.nifti import open_volume
            self._maps[key] = open_volume(self.cache_dir, self.volumes[key])
        img, lab = self._maps[key]
        y0, y1, x0, x1 = self._box(key, img.shape[1])
        k = self.in_slices // 2
        zs = np.clip(np.arange(z - k, z + k + 1), 0, img.shape[0] - 1)
        x = torch.from_numpy(self.scale(img[zs, y0:y1, x0:x1].astype(np.float32)))
        y = torch.from_numpy(lab[z, y0:y1, x0:x1].astype(np.int64))
        S = self.image_size
        if x.shape[-1] != S:
            x = F.interpolate(x[None], size=(S, S), mode="bilinear", align_corners=False)[0]
            y = F.interpolate(y[None, None].float(), size=(S, S), mode="nearest-exact")[0, 0].long()
        if self.liver_only:
            y = (y > 0).long()
        return x, y

    @torch.no_grad()
    def full_batch(self, logits, i0):
        """model output for items i0.. -> (prediction, truth) on the full slice:
        the logits shrunk back to the box, pasted into an empty slice, argmax"""
        preds, truths = [], []
        for b in range(logits.shape[0]):
            key, z = self.pairs[i0 + b]
            if key not in self._maps:
                from src.nifti import open_volume
                self._maps[key] = open_volume(self.cache_dir, self.volumes[key])
            _, lab = self._maps[key]
            size = lab.shape[1]
            y0, y1, x0, x1 = px_box(square_box(self.boxes[key], self.margin), size)
            lg = logits[b:b + 1].float()
            if y1 - y0 != lg.shape[-1]:
                lg = F.interpolate(lg, size=(y1 - y0, x1 - x0), mode="bilinear", align_corners=False)
            full = torch.zeros((size, size), dtype=torch.long)
            full[y0:y1, x0:x1] = lg.argmax(1)[0].cpu()
            t = torch.from_numpy(lab[z].astype(np.int64))
            if self.liver_only:
                t = (t > 0).long()
            preds.append(full)
            truths.append(t)
        return torch.stack(preds), torch.stack(truths)


def make_datasets(cfg, tr, va, te, cache_dir, index, verbose=True):
    """train / val / test LiverCropDatasets for get_dataloaders"""
    tb = true_boxes(cache_dir, index, verbose)
    sb = load_stage1_boxes(cfg.crop_liver, {k for k, _ in te})
    kw = dict(liver_only=cfg.liver_only, hu_window=cfg.hu_window, in_slices=cfg.in_slices)
    mk = lambda p, boxes, aug: LiverCropDataset(p, cache_dir, index, boxes, cfg.crop_margin,
                                                augment=aug, **kw)
    if verbose:
        print(f"[crop] stage 2: slices cropped to the liver (+{cfg.crop_margin:.0%} margin); "
              f"train/val on true boxes, test on stage 1's ({Path(cfg.crop_liver).name})")
    return mk(tr, tb, cfg.augment), mk(va, tb, False), mk(te, sb, False)


def main():
    from src.config import get_device
    p = argparse.ArgumentParser(description="stage 1 of the liver-crop cascade: save the "
                                            "predicted liver box of each test patient")
    p.add_argument("--stage1", required=True, help="a trained full-slice run folder")
    p.add_argument("--margin", type=float, default=0.1, help="margin used for the coverage check")
    a = p.parse_args()
    stage1_boxes(a.stage1, get_device(), a.margin)


if __name__ == "__main__":
    main()
