"""What MSD Task03_Liver actually contains: geometry, Hounsfield units, labels,
lesion sizes in millilitres, and the train/val/test split.

Reads only the slice cache (src/nifti.py), so it runs in seconds once the cache
exists.

    python -m src.utils.nifti_report

WRITES into results/nifti/

    dataset_report.txt     the numbers, with what they imply
    dataset_stats.json     per volume, for anything else to read
    dataset_overview.png   the same as figures
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import Config, RESOLUTIONS, results_dir
from src.data import split_pairs
from src.nifti import HU_MIN, load_index, open_volume, scale_hu

OVERLAY = [(0, 0, 0, 0), (0.20, 0.55, 0.95, 0.60), (0.95, 0.25, 0.20, 0.85)]
SHOW_WINDOW = (-160, 240)      # display only: liver window, level 40 width 400


def hist_percentiles(h, qs):
    """HU percentiles from a one-HU-per-bin histogram that starts at HU_MIN"""
    c = np.cumsum(h)
    return {q: HU_MIN + int(np.searchsorted(c, q / 100 * c[-1])) for q in qs}


def hist_mean(h):
    return float((np.arange(len(h)) + HU_MIN) @ h / max(h.sum(), 1))


def collect(cache_dir, cfg=None):
    cfg = cfg or Config()
    index = load_index(cache_dir)
    if not index:
        raise SystemExit(f"no cache at {cache_dir} - build it first:  python -m src.nifti")
    size = index["image_size"]
    vols = sorted((v for v in index["volumes"].values() if v["split"] == "train"),
                  key=lambda v: v["id"])
    tests = [v for v in index["volumes"].values() if v["split"] == "test"]

    hist = sum(np.load(Path(cache_dir) / f"hist_liver_{v['id']}.npy") for v in vols)
    fg_hist = hist[1] + hist[2]

    # the same seeded volume split training uses, so the report can say who is in test
    (_, _, _), _, sets = split_pairs([(v["key"], 0) for v in vols], cfg.val_fraction,
                                     cfg.test_fraction, cfg.seed, True, key=lambda p: p[0])

    lesions = [l for v in vols for l in v["lesions_ml"]]
    # components of at most 5 voxels: annotation specks rather than lesions
    vox_ml = {v["key"]: np.prod(v["spacing"]) / 1000 for v in vols}
    specks = sum(1 for v in vols for l in v["lesions_ml"] if l / vox_ml[v["key"]] <= 5.5)
    s = {
        "image_size": size,
        "n_volumes": len(vols), "n_test_unlabelled_cached": len(tests),
        "slices": [v["n_slices"] for v in vols],
        "pixel_spacing": [v["spacing"][0] for v in vols],
        "slice_thickness": [v["spacing"][2] for v in vols],
        "extent_mm": [v["n_slices"] * v["spacing"][2] for v in vols],
        "voxels": {k: sum(v["voxels"][k] for v in vols) for k in ("background", "liver", "tumor")},
        "pixels_at_size": {"liver": sum(sum(v["liver_px"]) for v in vols),
                           "tumor": sum(sum(v["tumor_px"]) for v in vols)},
        "fg_slices_native": sum(len(v["fg_slices_native"]) for v in vols),
        "fg_slices_at_size": sum(sum(1 for a, b in zip(v["liver_px"], v["tumor_px"]) if a + b)
                                 for v in vols),
        "tumor_slices_at_size": sum(sum(1 for b in v["tumor_px"] if b) for v in vols),
        "liver_ml": [v["liver_ml"] for v in vols],
        "tumor_ml": [v["tumor_ml"] for v in vols],
        "lesions_per_volume": [len(v["lesions_ml"]) for v in vols],
        "lesions_ml": lesions,
        "lesion_specks": specks,
        "hu_all": hist_percentiles(hist[0], (0.5, 50, 99.5)),
        "hu_liver": hist_percentiles(hist[1], (0.5, 5, 50, 95, 99.5)),
        "hu_tumor": hist_percentiles(hist[2], (0.5, 5, 50, 95, 99.5)),
        "hu_fg": hist_percentiles(fg_hist, (0.5, 99.5)),
        "hu_liver_mean": hist_mean(hist[1]), "hu_tumor_mean": hist_mean(hist[2]),
        "hu_min": min(v["hu_min"] for v in vols), "hu_max": max(v["hu_max"] for v in vols),
        "split": {"train": sets[0], "val": sets[1], "test": sets[2]},
        "per_volume": {v["key"]: {"slices": v["n_slices"], "spacing": v["spacing"],
                                  "hu_min": v["hu_min"], "hu_max": v["hu_max"],
                                  "liver_ml": v["liver_ml"], "tumor_ml": v["tumor_ml"],
                                  "lesions": len(v["lesions_ml"])} for v in vols},
    }
    return s, hist, vols


def write_text(s, cache_dir, out_path):
    L = []
    a = L.append
    sl, th, ps = np.array(s["slices"]), np.array(s["slice_thickness"]), np.array(s["pixel_spacing"])
    ext, lml, tml = np.array(s["extent_mm"]), np.array(s["liver_ml"]), np.array(s["tumor_ml"])
    les, lpv = np.array(s["lesions_ml"]), np.array(s["lesions_per_volume"])
    vox = s["voxels"]
    tot = sum(vox.values())

    a("MSD Task03_Liver (LiTS in NIfTI) — study")
    a("=" * 74)
    a("")
    a("SOURCE")
    a(f"  cache             {cache_dir}  ({s['image_size']}x{s['image_size']} per slice)")
    a(f"  labelled scans    {s['n_volumes']}   (imagesTr + labelsTr)")
    a(f"  unlabelled scans  70 in imagesTs - no labels, so not usable for scoring")
    a(f"  format            float32 CT in Hounsfield units, uint8 labels")
    a(f"                    0 background, 1 liver, 2 tumour - one label map per scan,")
    a(f"                    lossless, so no compression noise to filter out")
    a("")
    a("GEOMETRY")
    a(f"  in-plane          512x512 in every labelled scan")
    a(f"  pixel spacing     {ps.min():.3f}-{ps.max():.3f} mm  (median {np.median(ps):.3f})")
    a(f"                    so one pixel is not one size: the same liver covers")
    a(f"                    {ps.max()/ps.min():.1f}x more pixels in the finest scan than the coarsest")
    a(f"  slice thickness   {th.min():.2f}-{th.max():.2f} mm  (median {np.median(th):.2f})")
    for t, n in sorted(Counter(np.round(th, 2)).items()):
        a(f"                      {t:5.2f} mm  {n:3d} scans  {'#' * n}")
    a(f"  slices per scan   {sl.min()}-{sl.max()}  (median {np.median(sl):.0f}, total {sl.sum():,})")
    a(f"  scan length       {ext.min():.0f}-{ext.max():.0f} mm  (median {np.median(ext):.0f})")
    a("")
    a("  Thin-slice scans contribute far more 2D training slices than thick ones")
    a("  for the same anatomy: a 0.7 mm scan gives ~7x the slices of a 5 mm scan.")
    a("  A 2D model therefore sees the thin-slice hospitals much more often.")
    a("")
    a("INTENSITY (Hounsfield units, from every voxel)")
    a(f"  all voxels        {s['hu_min']:.0f} to {s['hu_max']:.0f}, "
      f"0.5-99.5 pct {s['hu_all'][0.5]} to {s['hu_all'][99.5]}")
    lo_scans = sorted(int(k.split("-")[1]) for k, v in s["per_volume"].items() if v["hu_min"] < -1024)
    hi_scans = sorted(int(k.split("-")[1]) for k, v in s["per_volume"].items() if v["hu_max"] > 4000)
    a(f"  below -1024 HU    {len(lo_scans)} scans {lo_scans}: the corners outside the round")
    a(f"                    field of view are padded with -2048 instead of -1024")
    a(f"  above 4000 HU     {len(hi_scans)} scans, up to {s['hu_max']:.0f}: metal or out-of-range")
    a(f"                    values. Real tissue stops near 3000; the HU window clips")
    a(f"                    them away, so they never reach the model.")
    a(f"  {'':18}{'0.5%':>7}{'5%':>7}{'median':>8}{'95%':>7}{'99.5%':>7}{'mean':>8}")
    for nm, k, mk in (("liver", "hu_liver", "hu_liver_mean"), ("tumour", "hu_tumor", "hu_tumor_mean")):
        p = s[k]
        a(f"  {nm:<18}{p[0.5]:>7}{p[5]:>7}{p[50]:>8}{p[95]:>7}{p[99.5]:>7}{s[mk]:>8.0f}")
    lo, hi = s["hu_fg"][0.5], s["hu_fg"][99.5]
    a("")
    a(f"  Liver and tumour sit in a narrow band, and they overlap: the tumour is only")
    a(f"  ~{s['hu_liver'][50] - s['hu_tumor'][50]} HU darker at the median. That is why the model gets a")
    a(f"  narrow window around them, not the full -1024..3000 range.")
    default = tuple(Config().hu_window)
    a(f"  Foreground 0.5-99.5 percentile window: [{lo}, {hi}] HU - the clipping range")
    a(f"  nnU-Net would pick for this data. "
      + ("It is the default hu_window." if default == (lo, hi)
         else f"The default hu_window is {list(default)}; use --hu-window {lo} {hi}"))
    a("")
    a("CLASS BALANCE (by voxel, full resolution)")
    for k, v in vox.items():
        a(f"  {k:<12}{v:>15,}   {100*v/tot:8.4f}%")
    a(f"  background is {vox['background']/max(vox['tumor'],1):,.0f}x the tumour class")
    a("")
    n = sl.sum()
    a(f"SLICE CONTENT   (all {n:,} slices)")
    a(f"  any foreground (full res)      {s['fg_slices_native']:>7,}   {100*s['fg_slices_native']/n:5.1f}%")
    a(f"  any foreground at {s['image_size']}px          {s['fg_slices_at_size']:>7,}   "
      f"{100*s['fg_slices_at_size']/n:5.1f}%   <- what skip_empty keeps")
    a(f"  any tumour at {s['image_size']}px              {s['tumor_slices_at_size']:>7,}   "
      f"{100*s['tumor_slices_at_size']/n:5.1f}%")
    a("")
    a("ORGAN AND LESIONS (3D, in millilitres)")
    a(f"  liver volume      {lml.min():.0f}-{lml.max():.0f} mL  (median {np.median(lml):.0f})")
    a(f"  tumour burden     {tml.min():.1f}-{tml.max():.0f} mL  (median {np.median(tml):.1f})")
    a(f"  scans with no tumour at all: {int((tml == 0).sum())}")
    a(f"  lesions per scan  {lpv.min()}-{lpv.max()}  (median {np.median(lpv):.0f}, total {lpv.sum()})")
    if len(les):
        a(f"  lesion size       median {np.median(les):.2f} mL, 90th pct {np.percentile(les, 90):.1f} mL,"
          f" largest {les.max():.0f} mL")
        for cut in (0.1, 1, 10):
            a(f"    smaller than {cut:>4} mL: {(les < cut).sum():5d}  ({100*(les < cut).mean():4.1f}%)")
        a("")
        a(f"  {s['lesion_specks']} of the {len(les)} are 5 voxels or fewer: specks left in the annotation")
        a("  rather than lesions a radiologist would count, but they are scored as tumour.")
        a("")
        a("  Most lesions are tiny. A 1 mL sphere is ~12 mm across - about 16 pixels")
        a("  at full resolution, 8 at 256px, 4 at 128px. This is where the tumour")
        a("  recall goes, and why lowering the resolution costs tumour Dice first.")
        a("  (Lesions are 3D connected components; on thick-slice scans one lesion")
        a("  can break into pieces, so the count is an upper bound.)")
    a("")
    sp = s["split"]
    a(f"SPLIT  (seed 42, by patient - no patient is in two splits)")
    for k in ("train", "val", "test"):
        ids = sorted(int(x.split("-")[1]) for x in sp[k])
        a(f"  {k:<6}{len(ids):>4} scans" + (f"   {ids}" if k != "train" else ""))
    burden = [s["per_volume"][k]["tumor_ml"] for k in sp["test"]]
    a(f"  test tumour burden: {sum(burden):.0f} mL in total, "
      f"{sum(1 for b in burden if b == 0)} test scans with no tumour")
    a("")

    text = "\n".join(L)
    print(text)
    Path(out_path).write_text(text, encoding="utf-8")
    print(f"\n[saved] {out_path}")


def overlay_rgb(grey01, lab):
    out = np.repeat(grey01[:, :, None], 3, axis=2)
    for cls in (1, 2):
        r, g, b, alpha = OVERLAY[cls]
        m = lab == cls
        out[m] = out[m] * (1 - alpha) + np.array([r, g, b], np.float32) * alpha
    return out


def draw(s, hist, vols, cache_dir, out_path, dpi=110):
    fig = plt.figure(figsize=(15, 15.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.3, 1.0, 1.0], hspace=0.46, wspace=0.28,
                          left=0.06, right=0.97, top=0.925, bottom=0.045)
    fig.suptitle("MSD Task03_Liver (LiTS in NIfTI) — what the data looks like",
                 fontsize=15.5, fontweight="bold", y=0.982)

    # three patients with a clearly visible lesion, from different slice thicknesses
    picks, seen = [], set()
    for v in sorted(vols, key=lambda v: -max(v["tumor_px"])):
        t = round(v["spacing"][2])
        if t in seen:
            continue
        seen.add(t)
        picks.append((v, int(np.argmax(v["tumor_px"]))))
        if len(picks) == 3:
            break
    for j, (v, z) in enumerate(picks):
        ax = fig.add_subplot(gs[0, j])
        img, lab = open_volume(cache_dir, v)
        grey = scale_hu(img[z].astype(np.float32), SHOW_WINDOW)
        ax.imshow(overlay_rgb(grey, np.asarray(lab[z])), interpolation="nearest")
        ax.set_title(f"liver_{v['id']}  slice {z}   {v['spacing'][2]:.1f} mm slices\n"
                     f"liver {v['liver_ml']:.0f} mL   tumour {v['tumor_ml']:.0f} mL",
                     fontsize=9)
        ax.axis("off")

    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(hist.shape[1]) + HU_MIN
    sel = (x >= -200) & (x <= 300)
    for row, nm, c in ((1, "liver", "#4A90D9"), (2, "tumour", "#C0392B")):
        ax.plot(x[sel], hist[row][sel] / hist[row].sum(), color=c, label=nm, lw=1.3)
    lo, hi = s["hu_fg"][0.5], s["hu_fg"][99.5]
    for b in (lo, hi):
        ax.axvline(b, color="#555555", ls="--", lw=1)
    ax.set_xlabel("Hounsfield units"); ax.set_ylabel("share of class voxels")
    ax.set_title(f"Liver vs tumour intensity\nforeground 0.5-99.5% window [{lo}, {hi}] HU",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5); ax.grid(alpha=.3)

    ax = fig.add_subplot(gs[1, 1])
    c = sorted(Counter(np.round(s["slice_thickness"], 2)).items())
    ax.bar([f"{t:g}" for t, _ in c], [n for _, n in c], color="#7FB069")
    ax.set_xlabel("slice thickness (mm)"); ax.set_ylabel("scans")
    ax.set_title("Slice thickness", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=.3)

    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(s["pixel_spacing"], s["slice_thickness"], s=14, color="#4A90D9", alpha=.7)
    ax.set_xlabel("pixel spacing (mm)"); ax.set_ylabel("slice thickness (mm)")
    ax.set_title("Voxel size per scan", fontsize=11, fontweight="bold")
    ax.grid(alpha=.3)

    ax = fig.add_subplot(gs[2, 0])
    sl = np.array(s["slices"])
    ax.hist(sl, bins=30, color="#7FB069", edgecolor="white", linewidth=0.4)
    ax.axvline(sl.mean(), color="#C0392B", ls="--", lw=1.4, label=f"mean {sl.mean():.0f}")
    ax.set_xlabel("slices per scan"); ax.set_ylabel("scans")
    ax.set_title(f"Scan length ({sl.min()}-{sl.max()} slices)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5); ax.grid(axis="y", alpha=.3)

    ax = fig.add_subplot(gs[2, 1])
    tml = sorted(s["tumor_ml"], reverse=True)
    ax.bar(range(len(tml)), tml, color="#C0392B", width=1.0)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("scan (sorted by tumour burden)"); ax.set_ylabel("tumour volume (mL, symlog)")
    ax.set_title(f"Tumour burden per scan\n{sum(1 for t in tml if t == 0)} scans have none",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=.3)

    ax = fig.add_subplot(gs[2, 2])
    les = np.array(s["lesions_ml"])
    if len(les):
        ax.hist(np.log10(np.clip(les, 1e-3, None)), bins=40, color="#C0392B",
                edgecolor="white", linewidth=0.4)
        ax.axvline(0, color="#555555", ls="--", lw=1, label="1 mL (~12 mm sphere)")
        ax.legend(fontsize=8.5)
    ax.set_xlabel("lesion volume, log10 mL"); ax.set_ylabel("lesions")
    ax.set_title(f"Lesion sizes ({len(les)} lesions)\n{s['lesion_specks']} are ≤5-voxel specks "
                 f"(the bar at the far left)", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=.3)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=OVERLAY[1][:3]),
               plt.Rectangle((0, 0), 1, 1, fc=OVERLAY[2][:3])]
    fig.legend(handles, ["liver", "tumour"], loc="upper right",
               bbox_to_anchor=(0.985, 0.992), ncol=2, frameon=False, fontsize=10)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"[saved] {out_path}")


def main():
    p = argparse.ArgumentParser(description="study the nifti dataset from its slice cache")
    p.add_argument("--resolution", choices=list(RESOLUTIONS), default=None,
                   help="which cache to read, default the smallest one built; "
                        "the numbers in mL and HU do not depend on it")
    a = p.parse_args()

    from src.nifti import cache_sizes
    sizes = cache_sizes()
    if not a.resolution and not sizes:
        raise SystemExit("no slice cache yet - build it first:  python -m src.nifti")
    cfg = Config(image_size=RESOLUTIONS[a.resolution] if a.resolution else sizes[0])
    s, hist, vols = collect(cfg.nifti_cache_dir, cfg)
    out = results_dir("nifti")
    write_text(s, cfg.nifti_cache_dir, out / "dataset_report.txt")
    (out / "dataset_stats.json").write_text(json.dumps(s, indent=1))
    print(f"[saved] {out / 'dataset_stats.json'}")
    draw(s, hist, vols, cfg.nifti_cache_dir, out / "dataset_overview.png")


if __name__ == "__main__":
    main()
