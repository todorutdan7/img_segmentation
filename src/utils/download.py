import argparse
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import DATA_DIR, CLASS_NAMES, MASK_VALUES

KAGGLE_SLUG = "harshwardhanbhangale/lits-dataset"
ALTERNATIVES = [
    "harshwardhanbhangale/lits-dataset",
    "andrewmvd/liver-tumor-segmentation",
]

INSTRUCTIONS = f"""
LiTS 2D (256x256 PNG) - manual steps
------------------------------------
1. Make a Kaggle account, then Account -> API -> "Create New API Token".
   That downloads kaggle.json.

2. Put it where the CLI looks for it:
       Windows:  C:\\Users\\<you>\\.kaggle\\kaggle.json

3. Install the CLI:
       pip install kaggle

4. Download and unzip into this project's data folder:
       kaggle datasets download -d {KAGGLE_SLUG} -p "{DATA_DIR}" --unzip

   Or download the zip in a browser and extract it into:
       {DATA_DIR}

5. Check it worked:
       python -m src.utils.download --verify

Expected layout (the loader also accepts one extra nesting level):
       data/images/*.png
       data/masks/*.png
"""


def try_kaggle():
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("[download] kaggle CLI not found or not configured.")
        return False

    print(f"[download] kaggle CLI found, fetching {KAGGLE_SLUG} into {DATA_DIR} ...")
    r = subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_SLUG,
                        "-p", str(DATA_DIR), "--unzip"])
    if r.returncode != 0:
        print("[download] the CLI failed - most likely authentication. "
              "check ~/.kaggle/kaggle.json exists.")
        return False
    print("[download] done.")
    return True


def verify(root=None, sample=60):
    """Report what is actually in data/: counts, sizes, and mask encoding."""
    root = Path(root or DATA_DIR)
    if not root.exists():
        print(f"[verify] {root} does not exist")
        return False

    from src.data import find_pairs, load_mask, mask_raw_values
    try:
        pairs, img_dir, mask_dir = find_pairs(root)
    except FileNotFoundError as e:
        print(f"[verify] {e}")
        return False

    print(f"[verify] images : {img_dir}")
    print(f"[verify] masks  : {mask_dir}")
    print(f"[verify] matched pairs: {len(pairs):,}")

    sizes, raw_vals, counts, empty = Counter(), Counter(), Counter(), 0
    step = max(1, len(pairs) // sample)
    checked = 0
    for ip, mp in pairs[::step][:sample]:
        im = Image.open(ip)
        sizes[im.size] += 1
        for v in mask_raw_values(mp):
            raw_vals[int(v)] += 1
        ids = load_mask(mp)
        vals, cts = np.unique(ids, return_counts=True)
        for v, c in zip(vals.tolist(), cts.tolist()):
            counts[v] += c
        if ids.max() == 0:
            empty += 1
        checked += 1

    print(f"\n[verify] checked {checked} pairs")
    print(f"[verify] image sizes: {dict(sizes)}")
    print(f"[verify] raw mask values seen: {sorted(raw_vals)[:12]}")
    print(f"[verify] expected encoding {MASK_VALUES} (0/127/255) or 0/1/2")

    total = sum(counts.values()) or 1
    print("\n[verify] pixel share after remapping to class ids:")
    for i, nm in enumerate(CLASS_NAMES):
        print(f"           {i} {nm:<12}{100*counts.get(i,0)/total:8.4f}%")

    print(f"\n[verify] slices with no foreground at all: {empty}/{checked} "
          f"({100*empty/max(checked,1):.0f}%)")
    if empty > checked * 0.5:
        print("[verify] note: most slices are pure background. that is normal for "
              "LiTS - the liver only appears in part of the scan.")

    missing = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES)) if counts.get(i, 0) == 0]
    if missing:
        print(f"[verify] WARNING: no pixels found for {missing}. either the export "
              f"is liver-only, or the mask encoding differs - check the raw values above.")
    else:
        print("[verify] all three classes present. ready to train.")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true", help="only inspect what is already there")
    p.add_argument("--data-dir", default=None)
    a = p.parse_args()

    if a.verify:
        verify(a.data_dir)
        return

    print(INSTRUCTIONS)
    if try_kaggle():
        verify(a.data_dir)
    else:
        print("\nFollow the steps above, then run:  python -m src.utils.download --verify")


if __name__ == "__main__":
    main()
