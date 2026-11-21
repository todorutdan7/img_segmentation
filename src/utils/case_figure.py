"""One patient looked at closely: what the model sees, the truth, and how bright
the tumour is against the liver.

For each patient a row of four panels:
    the slice with the most tumour, in the model's HU window
    the same with the true liver (green) and tumour (red) outlined
    the same in a wider window, which shows contrast the model's window cuts off
    the HU of every liver and tumour voxel of the scan, against the model window

A second patient (--compare) adds a row to set it against, e.g. one the model
gets right. The tumour Dice in the titles is per patient, after the clean-up,
from the run's metrics_test.json.

    python -m src.utils.case_figure 77 --compare 16 --out results/nifti/cases/patient77_faint_tumour.png
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.config import Config

WIDE = (-160, 240)
LIVER, TUMOUR = "#3aa655", "#d64545"
WATER = (0, 20)          # water, bile and cysts


def tumour_dice(run, n):
    """per-patient tumour Dice after the clean-up, or None"""
    m = Path(run) / "metrics_test.json"
    if not m.exists():
        return None
    pp = json.loads(m.read_text()).get("scores", {}).get("cleaned", {}).get("per_patient", {})
    v = pp.get(f"volume-{n}")
    return v["tumor"] if v else None


def body_box(img):
    """rows and columns holding the body, so the empty air is cropped off"""
    body = img > -500
    rows, cols = np.where(body.any(1))[0], np.where(body.any(0))[0]
    if not len(rows):
        return slice(None), slice(None)
    return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)


def draw_row(axes, n, hu, lab, spacing, dice, window, z=None):
    """hu, lab: (z, y, x) at native size; spacing (z, y, x) mm"""
    if z is None:
        z = int(np.argmax((lab == 2).sum((1, 2))))
    # turned 180 degrees from how the network sees it (and the viewer shows it),
    # to the usual radiology view: front of the body up, patient's right on the
    # left - as in patient104_dark_tumour.png
    img, l = hu[z][::-1, ::-1], lab[z][::-1, ::-1]
    r, c = body_box(img)
    img, l = img[r, c], l[r, c]
    aspect = spacing[1] / spacing[2]
    head = f"patient {n}" + (f" - tumour Dice {dice:.2f}" if dice is not None else "")

    def show(ax, win):
        ax.imshow(img, cmap="gray", vmin=win[0], vmax=win[1], aspect=aspect,
                  interpolation="nearest")
        ax.set_axis_off()

    show(axes[0], window)
    axes[0].set_title(f"{head}\nwhat the model sees: window [{window[0]}, {window[1]}] HU, "
                      f"slice {z}", fontsize=10)
    show(axes[1], window)
    axes[1].contour(l > 0, levels=[0.5], colors=LIVER, linewidths=1.2)
    axes[1].contour(l == 2, levels=[0.5], colors=TUMOUR, linewidths=1.2)
    axes[1].set_title("truth: liver green, tumour red", fontsize=10)
    show(axes[2], WIDE)
    axes[2].set_title(f"wider window [{WIDE[0]}, {WIDE[1]}] HU", fontsize=10)

    ax = axes[3]
    bins = np.arange(-100, 256, 5)
    ax.axvspan(*window, color="#e3e3e3", zorder=0, label="model window")
    ax.axvspan(*WATER, color="#d6e6f5", zorder=0, label="water / bile / cysts ~0-20")
    for k, name, col in ((1, "liver", LIVER), (2, "tumour", TUMOUR)):
        v = hu[lab == k]
        if v.size:
            ax.hist(v, bins=bins, density=True, color=col, alpha=0.65,
                    label=f"{name} (median {np.median(v):.0f} HU)")
    ax.set_yticks([])
    ax.set_xlabel("HU")
    ax.set_title(f"patient {n}: HU of all liver and tumour voxels", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def figure(rows, out, run=None, window=Config.hu_window):
    """rows: [(n, hu, lab, spacing, z or None), ...]"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    window = tuple(int(w) for w in window)
    fig, axes = plt.subplots(len(rows), 4, figsize=(18, 4.8 * len(rows)),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.25]}, squeeze=False)
    for ax_row, (n, hu, lab, spacing, z) in zip(axes, rows):
        draw_row(ax_row, n, hu, lab, spacing, tumour_dice(run, n) if run else None, window, z)
    if run:
        fig.suptitle(f"tumour Dice per patient from {Path(run).name} (fixed split, after the clean-up)",
                     fontsize=10, color="#52514e", y=0.995)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[saved] {out}")


def main():
    from src.nifti import find_scan, load_scan
    p = argparse.ArgumentParser(description="one patient: slice, truth and tumour vs liver HU")
    p.add_argument("patient", type=int)
    p.add_argument("--compare", type=int, default=None, help="a second patient, as a second row")
    p.add_argument("--slice", type=int, default=None, help="default: the slice with the most tumour")
    p.add_argument("--run", default="runs/nifti/r50/unet_r50_2.5d_lr1e-4",
                   help="run whose metrics_test.json gives the tumour Dice in the titles")
    p.add_argument("--nifti-dir", default=None, help="default data/Task03_Liver")
    p.add_argument("--out", default=None, help="default results/nifti/cases/patient<N>.png")
    a = p.parse_args()

    root = a.nifti_dir or Config().root_nifti_dir
    rows = []
    for n, z in ((a.patient, a.slice), (a.compare, None)):
        if n is None:
            continue
        img, lab_path = find_scan(n, root)
        if lab_path is None:
            raise SystemExit(f"patient {n} has no label file")
        hu, lab, spacing = load_scan(img, lab_path)
        rows.append((n, hu, lab, spacing, z))
    figure(rows, a.out or f"results/nifti/cases/patient{a.patient}.png", a.run)


if __name__ == "__main__":
    main()
