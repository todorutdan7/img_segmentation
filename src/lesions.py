"""Lesion detection: how many of the true tumours did the model find, by size?

Tumour Dice per patient gives one number per patient, and a missed 0.3 mL
lesion next to a found 100 mL one barely moves it. Here every lesion counts
on its own, so small-tumour changes show up: the 20 test patients of the
fixed split hold 168 lesions (114 under 1 mL), against 20 patient scores.

A lesion is one 3D connected piece of true tumour, counted the way the dataset
report counts them (6-connectivity; 1037 lesions in the 131 scans at 512 px).

    found       at least one voxel of it is predicted tumour
    found_50    at least half of its voxels are
    false       a predicted tumour piece that touches no true lesion

The scores are written by train.py and reeval into metrics_test.json, under
scores -> raw / cleaned -> lesions. To compare runs:

    python -m src.lesions --group nifti/r50
    python -m src.lesions runs/nifti/r50/unet_r50_2.5d_lr1e-4 runs/nifti/ablation/unet_2d

Runs of one k-fold model are put together: all 131 patients, every lesion once.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

BINS = [("< 0.1 mL", 0.0, 0.1), ("0.1-1 mL", 0.1, 1.0), ("1-10 mL", 1.0, 10.0),
        (">= 10 mL", 10.0, float("inf"))]


def voxel_ml(entry, image_size):
    """mL of one voxel of a cached slice: the scan's spacing, with the in-plane
    size scaled from the native grid to the training resolution"""
    sx, sy, sz = entry["spacing"]
    nx, ny = entry["shape"][:2]
    return sx * (nx / image_size) * sy * (ny / image_size) * sz / 1000


def lesion_records(gt, pr, vox_ml, tumour=2):
    """one patient. gt, pr: (z, y, x) labels.
    -> {"lesions": [[mL, share of it predicted tumour], ...] largest first,
        "false": predicted tumour pieces touching no true lesion}"""
    t, p = gt == tumour, pr == tumour
    if not (t.any() or p.any()):
        return {"lesions": [], "false": 0}
    # only the box holding every true and predicted tumour voxel is labelled,
    # not the whole scan
    box = ndimage.find_objects((t | p).astype(np.uint8))[0]
    t, p = t[box], p[box]
    lesions = []
    cc, n = ndimage.label(t)
    if n:
        size = np.bincount(cc[t], minlength=n + 1)
        hit = np.bincount(cc[t & p], minlength=n + 1)
        lesions = [[round(float(size[i] * vox_ml), 4), round(float(hit[i] / size[i]), 4)]
                   for i in range(1, n + 1)]
    false = 0
    pc, m = ndimage.label(p)
    if m:
        touch = np.bincount(pc[p & t], minlength=m + 1)
        false = int((touch[1:] == 0).sum())
    return {"lesions": sorted(lesions, reverse=True), "false": false}


def summarise(per_patient):
    """per_patient: {key: lesion_records(...)} -> the counts by size"""
    recs = [l for v in per_patient.values() for l in v["lesions"]]

    def count(cov):
        return {"lesions": len(cov), "found": int(sum(c > 0 for c in cov)),
                "found_50": int(sum(c >= 0.5 for c in cov))}

    bins = {name: count([c for ml, c in recs if lo <= ml < hi]) for name, lo, hi in BINS}
    bins["all"] = count([c for _, c in recs])
    false = [v["false"] for v in per_patient.values()]
    none = [k for k, v in per_patient.items() if not v["lesions"]]
    return {"bins": bins,
            "false_pieces": int(sum(false)),
            "patients_with_false": int(sum(f > 0 for f in false)),
            "patients": len(per_patient),
            "no_tumour_patients": len(none),
            "no_tumour_with_false": int(sum(per_patient[k]["false"] > 0 for k in none)),
            "per_patient": per_patient}


def pct(a, b):
    return f"{a:>4} ({100 * a / b:3.0f}%)" if b else f"{a:>4}  (  -)"


def print_lesions(s, title="lesions"):
    print(f"\n{title}\n  {'size':<12}{'lesions':>9}{'found':>13}{'>= 50% found':>15}")
    for name, b in s["bins"].items():
        print(f"  {name:<12}{b['lesions']:>9}{pct(b['found'], b['lesions']):>13}"
              f"{pct(b['found_50'], b['lesions']):>15}")
    print(f"  false tumour pieces: {s['false_pieces']} in {s['patients_with_false']} of "
          f"{s['patients']} patients; patients without tumour that got some: "
          f"{s['no_tumour_with_false']} of {s['no_tumour_patients']}")


# ------------------------------------------------------------ comparing runs

def _runs(paths, group, split="test"):
    from src.config import RUNS_DIR
    dirs = [Path(p) for p in paths]
    if group:
        dirs += sorted(h.parent for h in (Path(RUNS_DIR) / group).glob(f"*/metrics_{split}.json"))
    return dirs


def main():
    from src.utils import plotstyle as ps
    p = argparse.ArgumentParser(description="lesion detection by size, run by run")
    p.add_argument("runs", nargs="*", help="run folders")
    p.add_argument("--group", default=None, help="every run under runs/<group>, e.g. nifti/r50")
    p.add_argument("--raw", action="store_true", help="before the 3D clean-up (default: after)")
    p.add_argument("--split", default="test",
                   help="test (default), or test_tta for the scores with test-time flips (reeval --tta)")
    a = p.parse_args()

    which = "raw" if a.raw else "cleaned"
    models, missing = {}, []
    for d in _runs(a.runs, a.group, a.split):
        m, h = d / f"metrics_{a.split}.json", d / "history.json"
        if not (m.exists() and h.exists()):
            continue
        s = json.loads(m.read_text()).get("scores", {}).get(which, {}).get("lesions")
        if not s:
            missing.append(d.name)
            continue
        H = json.loads(h.read_text())
        cfg = H.get("config", {})
        label = ps.run_label(cfg, H.get("select_metric"))
        if cfg.get("folds"):
            label += f", {cfg['folds']}-fold"
        models.setdefault(label, {}).update(s["per_patient"])
    if not models:
        raise SystemExit("no run with lesion scores yet - rescore with "
                         "python -m src.utils.reeval --ckpt <run>/best_model.pt")
    for label, pp in models.items():
        print_lesions(summarise(pp), title=label)
    if missing:
        print(f"\nno lesion scores yet (rescore with reeval): {', '.join(missing)}")


if __name__ == "__main__":
    main()
