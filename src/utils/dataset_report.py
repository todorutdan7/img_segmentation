
import argparse
import gc
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.config import DATA_DIR, ROOT, CLASS_NAMES, MASK_VALUES as MASK_VALUES_REF
from src.data import find_pairs, load_mask, mask_raw_values, volume_id

OVERLAY = [(0, 0, 0, 0), (0.20, 0.55, 0.95, 0.60), (0.95, 0.25, 0.20, 0.85)]
RESULTS_DIR = ROOT / "results"

# Minimum tumour pixels before a slice counts as tumour-bearing.
#
# The masks are lossy JPEG and contain stray 3s, so compression demonstrably
# altered the label values. Counting a slice as tumour-bearing on a single
# value-2 pixel flags ~91% of liver slices, which is not plausible. 50 px is a
# pragmatic floor, NOT a measured noise level: it also discards genuine lesion
# poles, where a real 3D tumour is clipped by the first or last slice through
# it. Separating the two would need the lossless NIfTI labels.
TUMOR_MIN_PX = 50


def scan(pairs, sample=0, tumor_min_px=TUMOR_MIN_PX, verbose=True):
    stride, idx = 1, range(len(pairs))
    if sample and sample < len(pairs):
        stride = len(pairs) // sample
        idx = range(0, len(pairs), stride)

    pixel_counts = Counter()
    per_volume = defaultdict(lambda: {"fg": 0, "liver": 0, "tumor": 0, "tumor_slices": 0})
    fg_frac, blob_sizes = [], []
    tumor_slices = tumor_any = stray_only = liver_slices = empty = 0
    raw_values = Counter()
    checked = 0

    for i in idx:
        ip, mp = pairs[i]
        for v in mask_raw_values(mp):
            raw_values[int(v)] += 1
        ids = load_mask(mp)

        vals, cts = np.unique(ids, return_counts=True)
        d = dict(zip(vals.tolist(), cts.tolist()))
        for k, c in d.items():
            pixel_counts[k] += c

        n_tum = d.get(2, 0)
        v = per_volume[volume_id(ip)]
        v["liver"] += d.get(1, 0)
        v["tumor"] += n_tum
        if ids.max() > 0:
            v["fg"] += 1
            fg_frac.append((d.get(1, 0) + n_tum) / ids.size)
        else:
            empty += 1
        if d.get(1, 0):
            liver_slices += 1
        if n_tum:
            tumor_any += 1
            blob_sizes.append(n_tum)
        if n_tum >= tumor_min_px:
            tumor_slices += 1
            v["tumor_slices"] += 1
        elif n_tum:
            stray_only += 1
        checked += 1

        if verbose and checked % 5000 == 0:
            print(f"  scanned {checked:,}")

    # exact, from every filename - costs nothing and must NOT be the sampled count
    true_spv = Counter(volume_id(ip) for ip, _ in pairs)

    im = Image.open(pairs[0][0])
    # the mask entry is a single path, or a (liver, lesion) tuple in the split layout
    mask0 = pairs[0][1]
    split = isinstance(mask0, (tuple, list))
    mask0 = mask0[0] if split else mask0
    return {
        "n_pairs": len(pairs),
        "n_checked": checked,
        "stride": stride,
        "tumor_min_px": tumor_min_px,
        "image_size": list(im.size),
        "image_mode": im.mode,
        "mask_mode": Image.open(mask0).mode,
        "file_suffix": pairs[0][0].suffix,
        "layout": "split binary masks (liver + lesion)" if split
                  else "single mask, pixel value = class id",
        "raw_mask_values": sorted(raw_values),
        "pixel_counts": {CLASS_NAMES[k]: int(v) for k, v in sorted(pixel_counts.items())
                         if k < len(CLASS_NAMES)},
        "n_volumes": len(true_spv),
        "slices_per_volume": dict(true_spv),                      # exact
        "fg_slices_per_volume": {k: v["fg"] * stride              # estimated
                                 for k, v in per_volume.items()},
        "tumor_pixels_per_volume": {k: v["tumor"] for k, v in per_volume.items()},
        "tumor_slices_per_volume": {k: v["tumor_slices"] for k, v in per_volume.items()},
        "empty_slices": empty,
        # any foreground at all - the exact complement of empty_slices.
        # liver_slices counts CLASS-1 pixels, which is not the same thing: a
        # slice whose liver is entirely lesion has no class-1 pixel.
        "fg_slices": checked - empty,
        "liver_slices": liver_slices,
        "tumor_slices": tumor_slices,
        "tumor_slices_any_pixel": tumor_any,
        "stray_tumor_slices": stray_only,
        "tumor_blob_sizes": blob_sizes,
        "fg_fraction_when_present": fg_frac,
    }


def pick_samples(pairs, n=4, min_tumor=300):
    """Slices with a real lesion, for the sample strip.

    The threshold is deliberately well above TUMOR_MIN_PX: a 50-pixel region is
    enough to call a slice tumour-bearing, but it is invisible in a thumbnail
    and makes the overlay look like an annotation error.
    """
    picked, seen_vols = [], set()
    step = max(1, len(pairs) // 600)
    for thresh in (min_tumor, TUMOR_MIN_PX, 1):
        for i in range(0, len(pairs), step):
            ip, mp = pairs[i]
            if volume_id(ip) in seen_vols:
                continue
            ids = load_mask(mp)
            if (ids == 2).sum() >= thresh:
                picked.append((ip, mp, ids))
                seen_vols.add(volume_id(ip))
                if len(picked) >= n:
                    return picked
    return picked[:n]


def overlay_rgb(img, ids):
    """Composite mask onto the slice in numpy, as one uint8 RGB array.

    Two stacked imshow calls make matplotlib resample twice, each time into a
    float64 buffer the size of the axes in pixels. Compositing first halves
    that, which matters on a machine that is already short on memory.
    """
    out = np.repeat(img[:, :, None].astype(np.float32), 3, axis=2) / 255.0
    for cls in range(1, len(OVERLAY)):
        r, g, b, alpha = OVERLAY[cls]
        m = ids == cls
        if m.any():
            out[m] = out[m] * (1 - alpha) + np.array([r, g, b], np.float32) * alpha
    return (out * 255).astype(np.uint8)


def draw(stats, samples, out_path, dpi=110):
    fig = plt.figure(figsize=(15, 15.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.35, 1.0, 1.0],
                          hspace=0.46, wspace=0.26,
                          left=0.05, right=0.97, top=0.925, bottom=0.045)
    fig.suptitle("LiTS 2D dataset — what the data actually looks like",
                 fontsize=15.5, fontweight="bold", y=0.982)

    total = sum(stats["pixel_counts"].values()) or 1

    for j, (ip, mp, ids) in enumerate(samples[:3]):
        ax = fig.add_subplot(gs[0, j])
        img = np.asarray(Image.open(ip).convert("L"))
        ax.imshow(overlay_rgb(img, ids), interpolation="nearest")
        liver = 100 * (ids == 1).mean()
        tum = 100 * (ids == 2).mean()
        ax.set_title(f"{ip.name}\nliver {liver:.2f}%   tumour {tum:.3f}%", fontsize=9)
        ax.axis("off")
        del img

    ax = fig.add_subplot(gs[1, 0])
    names = list(stats["pixel_counts"])
    vals = [100 * stats["pixel_counts"][n] / total for n in names]
    colors = ["#9B9B9B", "#4A90D9", "#C0392B"][:len(names)]
    ax.bar(names, vals, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("share of all pixels (%, log scale)")
    ax.set_title("Class imbalance", fontsize=11, fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(i, v * 1.25, f"{v:.3f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(min(vals) * 0.15, 900)
    ax.grid(axis="y", alpha=.3)
    if len(vals) > 2 and vals[2] > 0:
        ax.set_xlabel(f"background is {vals[0]/vals[2]:,.0f}× the tumour class",
                      fontsize=9, color="#C0392B", style="italic", labelpad=8)

    ax = fig.add_subplot(gs[1, 1])
    n = stats["n_checked"]
    tmin = stats["tumor_min_px"]
    counts = [stats["empty_slices"], stats.get("fg_slices", stats["liver_slices"]),
              stats["tumor_slices"]]
    ax.bar(["no\nforeground", "any\nforeground", f"contains tumour\n(≥{tmin} px)"],
           counts, color=["#9B9B9B", "#4A90D9", "#C0392B"])
    for i, v in enumerate(counts):
        ax.text(i, v + max(counts) * 0.02, f"{v:,}\n({100*v/n:.1f}%)",
                ha="center", fontsize=8.5)
    ax.set_ylabel("slices")
    ax.set_ylim(0, max(counts) * 1.25)
    ax.set_title(f"Slice content  (n={n:,} scanned)", fontsize=11, fontweight="bold")
    stray = stats["stray_tumor_slices"]
    ax.set_xlabel(f"a further {stray:,} slices carry only 1-{tmin-1} tumour pixels\n"
                  f"(too small to be scored reliably — excluded above)",
                  fontsize=8.5, color="#777777", style="italic", labelpad=6)
    ax.grid(axis="y", alpha=.3)

    ax = fig.add_subplot(gs[1, 2])
    fr = np.array(stats["fg_fraction_when_present"]) * 100
    if len(fr):
        ax.hist(fr, bins=40, color="#4A90D9", edgecolor="white", linewidth=0.4)
        ax.axvline(fr.mean(), color="#C0392B", ls="--", lw=1.4,
                   label=f"mean {fr.mean():.1f}%")
        ax.legend(fontsize=8.5)
    ax.set_xlabel("% of the slice that is liver or tumour")
    ax.set_ylabel("slices")
    ax.set_title("Foreground area, when present", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=.3)

    stride = stats.get("stride", 1)

    ax = fig.add_subplot(gs[2, 0])
    spv = np.array(list(stats["slices_per_volume"].values()))
    ax.hist(spv, bins=30, color="#7FB069", edgecolor="white", linewidth=0.4)
    ax.axvline(spv.mean(), color="#C0392B", ls="--", lw=1.4,
               label=f"mean {spv.mean():.0f}")
    ax.set_xlabel("slices per volume  (exact, all files)"); ax.set_ylabel("volumes")
    ax.set_title(f"Volume sizes  ({stats['n_volumes']} volumes, "
                 f"{spv.min()}-{spv.max()} slices)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5); ax.grid(axis="y", alpha=.3)

    ax = fig.add_subplot(gs[2, 1])
    fpv = np.array(list(stats["fg_slices_per_volume"].values()))
    ax.hist(fpv, bins=30, color="#4A90D9", edgecolor="white", linewidth=0.4)
    ax.axvline(fpv.mean(), color="#C0392B", ls="--", lw=1.4,
               label=f"mean {fpv.mean():.0f}")
    ax.set_xlabel("slices containing foreground"
                  + (f"  (estimated, ×{stride} from the sample)" if stride > 1 else ""))
    ax.set_ylabel("volumes")
    ax.set_title("Useful slices per volume", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5); ax.grid(axis="y", alpha=.3)

    ax = fig.add_subplot(gs[2, 2])
    tpv = sorted(stats["tumor_pixels_per_volume"].values(), reverse=True)
    ax.bar(range(len(tpv)), tpv, color="#C0392B", width=1.0)
    ax.set_yscale("symlog")
    ax.set_xlabel("volume (sorted by tumour burden)")
    ax.set_ylabel("tumour pixels in the scanned slices (symlog)")
    n_none = sum(1 for v in stats["tumor_slices_per_volume"].values() if v == 0)
    scope = "volumes" if len(tpv) == stats["n_volumes"] else "sampled volumes"
    ax.set_title(f"Tumour burden per volume\n{n_none}/{len(tpv)} {scope} have no "
                 f"slice above {stats['tumor_min_px']} px", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=.3)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=OVERLAY[1][:3]),
               plt.Rectangle((0, 0), 1, 1, fc=OVERLAY[2][:3])]
    fig.legend(handles, ["liver", "tumour"], loc="upper right",
               bbox_to_anchor=(0.985, 0.992), ncol=2, frameon=False, fontsize=10)

    # no bbox_inches="tight": the margins above are already explicit, and tight
    # bbox forces a second, larger renderer allocation that can fail outright
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"[saved] {out_path}  ({15*dpi:.0f}x{15.5*dpi:.0f} px)")


def write_text(stats, img_dir, mask_dir, out_path, pairs_example="volume-37_112.png"):
    total = sum(stats["pixel_counts"].values()) or 1
    n = stats["n_checked"]
    tmin = stats["tumor_min_px"]
    stride = stats.get("stride", 1)
    spv = np.array(list(stats["slices_per_volume"].values()))
    fpv = np.array(list(stats["fg_slices_per_volume"].values()))
    fr = np.array(stats["fg_fraction_when_present"]) * 100
    bl = np.array(stats["tumor_blob_sizes"])

    L = []
    a = L.append
    a("LiTS 2D dataset — study")
    a("=" * 74)
    a("")
    a("SOURCE")
    a(f"  images            {img_dir}")
    a(f"  masks             {mask_dir}")
    a(f"  matched pairs     {stats['n_pairs']:,}")
    a(f"  scanned for this  {n:,}")
    a(f"  image size        {stats['image_size'][0]}x{stats['image_size'][1]}, "
      f"mode {stats['image_mode']} (converted to greyscale on load)")
    a(f"  file format       {stats['file_suffix']}")
    a(f"  mask layout       {stats.get('layout', 'single mask')}")
    a("")

    # Everything in this section is decided by what was actually measured, not
    # by which dataset the author happened to have when writing it.
    raw = set(stats["raw_mask_values"])
    split = "split" in str(stats.get("layout", ""))
    lossless = stats["file_suffix"].lower() in {".png", ".bmp", ".tif", ".tiff"}
    expected = {0, 1} if split else {0, 1, 2}
    damaged = not raw <= expected and not raw <= set(MASK_VALUES_REF)

    a("MASK ENCODING")
    a(f"  raw values found  {sorted(raw)}")
    if split:
        a("  Two binary masks per slice, merged on load: liver -> 1, then lesion")
        a("  overwrites it with 2. Lesion is nested inside liver, so the merge order")
        a("  matters and 2 must be written last.")
    else:
        a("  The pixel VALUE is the class id, not a brightness - which is why the")
        a("  mask files look uniformly black in an image viewer.")
    a("")

    if damaged:
        a(f"  DAMAGED. Values outside {sorted(expected)} are present, so the labels")
        a(f"  were altered after they were written - {stats['file_suffix']} is lossy and")
        a("  is not a valid container for a label map. If compression can push 2 to 3")
        a("  it can also push 1 to 2, and that needs a smaller excursion, so spurious")
        a("  tumour pixels should be more common than the out-of-range values we can")
        a("  see directly. remap_mask clamps anything above the last class id.")
    elif lossless:
        a(f"  CLEAN. Every value is within {sorted(expected)} and the format is")
        a("  lossless, so the labels are exactly as annotated. No clamping needed and")
        a("  no compression noise to work around.")
    else:
        a(f"  All values are within {sorted(expected)}, but {stats['file_suffix']} is a lossy")
        a("  format - the labels happen to have survived, which is luck, not a")
        a("  guarantee. Prefer PNG for label maps.")
    a("")

    if len(bl):
        a("  Lesion region sizes, where a lesion is present at all:")
        if tmin > 1:
            a(f"    slices with >= 1 tumour pixel      {stats['tumor_slices_any_pixel']:>8,}"
              f"   ({100*stats['tumor_slices_any_pixel']/n:5.1f}%)")
            a(f"    slices with >= {tmin:<3} tumour pixels  {stats['tumor_slices']:>8,}"
              f"   ({100*stats['tumor_slices']/n:5.1f}%)")
        else:
            a(f"    slices with >= 1 tumour pixel      {stats['tumor_slices']:>8,}"
              f"   ({100*stats['tumor_slices']/n:5.1f}%)   (no floor applied)")
        px = stats["image_size"][0] * stats["image_size"][1]
        a(f"    median region: {np.median(bl):>6.0f} px  ({100*np.median(bl)/px:.3f}% of a slice)")
        a(f"    10th pct     : {np.percentile(bl,10):>6.0f} px")
        a(f"    90th pct     : {np.percentile(bl,90):>6.0f} px")
    a("")
    if tmin > 1:
        share = 100 * stats["tumor_slices_any_pixel"] / max(stats["liver_slices"], 1)
        a(f"  A {tmin}-pixel floor is applied before calling a slice tumour-bearing.")
        a(f"  Without it, {share:.0f}% of liver-bearing slices count as containing tumour.")
        a("  Note what the floor removes: on clean labels it discards genuine lesion")
        a("  poles, where a real 3D tumour is clipped by the first or last slice")
        a("  through it. Those labels are correct, just too small to score reliably.")
        a("  Pass --tumor-min-px 1 for the unfiltered count.")
    else:
        a("  No floor applied, so these are the annotations exactly as given.")
    a("")
    a("CLASS BALANCE (by pixel)")
    for k, v in stats["pixel_counts"].items():
        a(f"  {k:<12}{v:>15,}   {100*v/total:8.4f}%")
    if len(stats["pixel_counts"]) > 2:
        vals = list(stats["pixel_counts"].values())
        a(f"  background is {vals[0]/max(vals[2],1):,.0f}x the tumour class")
    a("")
    bg_share = 100 * list(stats["pixel_counts"].values())[0] / total
    a("  Consequences, and why the project is built the way it is:")
    a(f"    - pixel accuracy is meaningless here. A model predicting background")
    a(f"      everywhere scores {bg_share:.1f}% and segments nothing (foreground Dice 0.000).")
    a("    - the loss is Dice-based, which measures overlap per class, so tumour")
    a("      counts as much as background.")
    a("    - the checkpoint is selected on mean FOREGROUND Dice, never on loss")
    a("      or accuracy.")
    a("")
    fg_sl = stats.get("fg_slices", n - stats["empty_slices"])
    a(f"SLICE CONTENT   (of the {n:,} slices scanned)")
    a(f"  no foreground     {stats['empty_slices']:>8,}   {100*stats['empty_slices']/n:5.1f}%")
    a(f"  any foreground    {fg_sl:>8,}   {100*fg_sl/n:5.1f}%"
      f"   <- exact complement of the line above")
    a(f"    class-1 pixels  {stats['liver_slices']:>8,}   {100*stats['liver_slices']/n:5.1f}%"
      f"   (liver tissue that is not lesion)")
    a(f"    contains tumour {stats['tumor_slices']:>8,}   {100*stats['tumor_slices']/n:5.1f}%"
      f"   (>= {tmin} px)")
    if fg_sl != stats["liver_slices"]:
        a(f"  the {fg_sl - stats['liver_slices']} slice(s) of difference have foreground but no")
        a("  class-1 pixel at all - their liver region is entirely lesion.")
    a("")
    a("  Most slices are pure background - the liver only occupies part of each")
    a("  scan. Training on them mostly teaches the model to output background,")
    a("  so skip_empty=True drops them by default.")
    if len(fr):
        a("")
        a(f"  When foreground is present it covers {fr.mean():.1f}% of the slice on")
        a(f"  average (median {np.median(fr):.1f}%, max {fr.max():.1f}%).")
    a("")
    a("VOLUME STRUCTURE")
    a(f"  volumes           {stats['n_volumes']}")
    a(f"  slices per volume min {spv.min()}, mean {spv.mean():.0f}, "
      f"median {np.median(spv):.0f}, max {spv.max()}   (exact)")
    a(f"  foreground slices min {fpv.min()}, mean {fpv.mean():.0f}, max {fpv.max()}"
      + (f"   (estimated: sampled count x{stride})" if stride > 1 else "   (exact)"))
    n_none = sum(1 for v in stats["tumor_slices_per_volume"].values() if v == 0)
    a(f"  volumes with no slice above {tmin} tumour px: {n_none}/{stats['n_volumes']}"
      + ("  (in the sampled slices)" if stride > 1 else ""))
    a("")
    a("  Volumes are very unequal in length, so a 70/15/15 split by VOLUME does")
    a("  not give 70/15/15 by slice. That is expected. Tumour burden is uneven")
    a("  too, so which volumes land in test moves the tumour Dice noticeably -")
    a("  a single test score without that caveat is overclaiming.")
    a("")
    a(f"  Filenames carry the volume id ({pairs_example}), so the split is by")
    a("  VOLUME, not by slice. Adjacent slices of one liver are near-duplicates;")
    a("  a slice-level split would place them in both train and test and inflate")
    a("  every score.")
    a("")
    a("PREPROCESSING THIS IMPLIES")
    a("  - convert to single-channel greyscale (CT is not colour)")
    a("  - scale to 0..1, then standardise")
    a("  - resize masks with NEAREST only; interpolation would invent class ids")
    a("  - augment image and mask together with identical parameters, and only")
    a("    with flips and 90-degree rotations so mask values stay exact integers")

    text = "\n".join(L)
    print(text)
    Path(out_path).write_text(text, encoding="utf-8")
    print(f"\n[saved] {out_path}")


def main():
    p = argparse.ArgumentParser(description="study the dataset, no training needed")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--sample", type=int, default=6000,
                   help="masks to scan; 0 = all of them")
    p.add_argument("--full", action="store_true", help="scan every mask")
    p.add_argument("--tumor-min-px", type=int, default=TUMOR_MIN_PX,
                   help="pixels needed before a slice counts as tumour-bearing. "
                        "the JPEG masks contain compression noise; 1 pixel is "
                        "not enough evidence of a lesion")
    p.add_argument("--dpi", type=int, default=110,
                   help="figure resolution; falls back to 90 then 70 on MemoryError")
    a = p.parse_args()

    pairs, img_dir, mask_dir = find_pairs(Path(a.data_dir or DATA_DIR))
    print(f"[data] {len(pairs):,} pairs found")

    sample = 0 if a.full else a.sample
    print(f"[scan] reading {'every mask' if not sample else f'{sample:,} masks'} ...")
    stats = scan(pairs, sample, a.tumor_min_px)

    out = RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    # numbers first: the scan is the expensive part, so never let a plotting
    # problem throw away a completed scan
    write_text(stats, img_dir, mask_dir, out / "dataset_report.txt",
               pairs_example=pairs[0][0].name)
    slim = {k: v for k, v in stats.items()
            if k not in ("fg_fraction_when_present", "tumor_blob_sizes")}
    (out / "dataset_stats.json").write_text(json.dumps(slim, indent=2))
    print(f"[saved] {out / 'dataset_stats.json'}")

    samples = pick_samples(pairs)
    del pairs                     # ~117k Path objects, no longer needed
    gc.collect()

    print()
    for dpi in (a.dpi, 90, 70):   # step down rather than lose the figure
        try:
            draw(stats, samples, out / "dataset_overview.png", dpi)
            break
        except MemoryError:
            print(f"[figure] out of memory at dpi={dpi}, retrying smaller")
            plt.close("all")
            gc.collect()
        except Exception as e:
            print(f"[figure] FAILED: {type(e).__name__}: {e}")
            print("[figure] the numbers above are still valid and saved.")
            break
    else:
        print("[figure] could not render even at dpi=70. The numbers above are "
              "saved; close other applications and rerun, or raise the Windows "
              "paging file size.")


if __name__ == "__main__":
    main()
