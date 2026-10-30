"""Stack one patient's 2D predictions into a 3D volume, and write it out.

The network segments one 2D slice at a time and never sees the slice above or
below. A CT scan is really a stack of those slices, so if you put the
predictions back in scan order you get a 3D volume of the patient. That is what
a radiologist actually wants to look at, and it also shows problems a per slice
Dice score cannot: for example whether the predicted liver is one connected
organ or several disconnected lumps.

This script is the BATCH version it writes files and exits. Its sibling
`viewer.py` is the interactive window. Use this one when you want:

  * PNG figures for the report
  * `.nii.gz` files to open in 3D Slicer

HOW TO RUN IT

    python -m src.utils.view3d                          # first test patient, runs/nifti/baseline
    python -m src.utils.view3d --volume 16              # a specific patient
    python -m src.utils.view3d --list                   # which patients exist
    python -m src.utils.view3d --nifti                  # also export for 3D Slicer
    python -m src.utils.view3d --group internship_png/ablation_2070 --volume 16

The data comes from whatever the model was trained on. A nifti model reads the
patient from the nifti cache, with the real slice thickness from the scan
header; a png model reads the internship's png slices.

WRITES into results/nifti/ or results/internship_png/

    view3d_<patient>_slices.png     axial / coronal / sagittal cuts, ground truth on top, prediction below
    view3d_<patient>_surfaces.png   liver and tumor as 3D surfaces, 3 angles
    view3d_<patient>_{ct,truth,pred}.nii.gz     only with --nifti

PIPELINE
    main()
      -> load_checkpoint()          get the trained model back
      -> find_pairs()               list every (image, mask) file in the dataset
      -> patient_pairs()            group those files by patient, in slice order
      -> has_foreground()           drop slices with no liver in the ground truth
      -> build_volume()             run the model on each slice, stack into 3D
      -> volume_stats()             Dice for the whole patient
      -> slices_figure()            write the 2D figure
      -> surfaces_figure()          write the 3D figure
      -> nibabel.save()             write the .nii.gz files
"""

import argparse
import re
from pathlib import Path

import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.config import get_device, CLASS_NAMES, results_dir
from src.data import find_pairs, load_mask, volume_id
from src.model import load_checkpoint

# colors as matplotlib RGB floats, 0.0 to 1.0 (not 0-255).
LIVER_COLOR = (0.35, 0.60, 0.90)    # blue
TUMOR_COLOR = (0.90, 0.25, 0.20)    # red


def slice_index(path: Path) -> int:
    """pull the slice number out of a filename
    IN   path  - e.g. .../volume-16_402.png
    OUT  int   - 402, or 0 if the name has no trailing number
    USED BY: patient_pairs(), as the sort key.
    """
    m = re.search(r"_(\d+)$", path.stem)
    return int(m.group(1)) if m else 0


def patient_pairs(pairs, want=None):
    """group the flat list of dataset files by patient, sorted by slice number

    IN   pairs - list of (image_path, mask_path) from find_pairs(). This is the
                 WHOLE dataset, all 131 patients mixed together in one list.
         want  - optional. A patient to pick out: 16, "16" or "volume-16".

    OUT  if want is None:  (None, dict)   dict maps "volume-16" -> [(img, mask), ...]
         if want is given: ("volume-16", [(img, mask), ...])  just that patient

    Returning a 2 tuple in both cases is so the caller can always unpack two
    values; the first is the patient key, which is only meaningful when you
    asked for one specific patient.

    USE CASE
    Two different jobs: `--list` uses the dict form to show every patient and
    how many slices it has; everything else uses the `want` form to fetch one
    patient's slices in order, ready to feed to build_volume()
    """
    by = {}
    for ip, mp in pairs:
        # volume_id() turns ".../volume-16_402.png" into "volume-16", which is
        # the patient. setdefault creates the list the first time we see them
        by.setdefault(volume_id(ip), []).append((ip, mp))

    # sort each patient's slices head-to-foot. See slice_index() for why this
    # cannot be left to the default filename ordering
    for v in by:
        by[v].sort(key=lambda t: slice_index(t[0]))

    if want is not None:
        # Accept 16, "16" or "volume-16" so the command line can be lazy.
        key = want if str(want).startswith("volume") else f"volume-{want}"
        if key not in by:
            raise SystemExit(f"no such patient: {key}")
        return key, by[key]
    return None, by


@torch.no_grad()        # no gradients: we are only predicting, not training
def build_volume(model, slices, device, image_size, stride=1, verbose=True):
    """Run the model on every slice of one patient and stack the results in 3D

    This is the core of the file. Everything else either prepares its input or
    draws its output.

    IN   model      - the trained network, already in eval mode
         slices     - [(image_path, mask_path), ...] for ONE patient, in order
         device     - "cuda" or "cpu"
         image_size - 256; the side of one slice
         stride     - use every Nth slice. 2 halves the work and the memory
                      and barely changes how the 3D looks
         verbose    - print progress every 100 slices

    OUT  three numpy arrays, all shape (n_slices, 256, 256), dtype uint8:
         ct  - the CT images,      values 0-255
         gt  - the ground truth,   values 0 / 1 / 2  (background/liver/tumor)
         pr  - the prediction,     values 0 / 1 / 2

    The axis order is (z, y, x): the first axis walks through the scan, so
    ct[10] is the eleventh slice.
    """

    sel = slices[::stride]      # every Nth slice, if the caller asked for that
    n = len(sel)

    # np.empty, not np.zeros: we overwrite every element below, so there is no
    # point paying to fill them with zeros first.
    ct = np.empty((n, image_size, image_size), dtype=np.uint8)
    gt = np.empty((n, image_size, image_size), dtype=np.uint8)
    pr = np.empty((n, image_size, image_size), dtype=np.uint8)

    for i, (ip, mp) in enumerate(sel):
        # --- the CT image -------------------------------------------------
        img = Image.open(ip).convert("L")           # "L" = 8-bit greyscale
        if img.size != (image_size, image_size):
            # Bilinear is fine for the IMAGE. It would be wrong for a mask,
            # where averaging class 1 and class 2 would invent a class 1.5.
            img = img.resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8)
        ct[i] = arr

        # --- the prediction -----------------------------------------------
        # Normalise exactly the way training did: 0-255 -> 0-1 -> mean 0.5,
        # std 0.5. If this does not match training, the model sees inputs it
        # was never trained on and the output quietly gets worse.
        x = torch.from_numpy((arr.astype(np.float32) / 255.0 - 0.5) / 0.5)
        # x[None, None] adds the batch and channel axes: (256,256) -> (1,1,256,256).
        # The model returns one score per class, so argmax(1) picks the winning
        # class per pixel. [0] drops the batch axis, .cpu() brings it back from
        # the GPU so numpy can read it.
        pr[i] = model(x[None, None].to(device)).argmax(1)[0].cpu().numpy()

        # --- the ground truth ---------------------------------------------
        # load_mask handles the two-separate-binary-files layout and merges
        # them into one 0/1/2 mask, writing the lesion last.
        gt[i] = load_mask(mp, image_size)

        if verbose and (i + 1) % 100 == 0:
            print(f"    {i+1}/{n} slices")

    mb = 3 * ct.nbytes / 1024 ** 2      # three arrays of the same size
    if verbose:
        print(f"[run ] volumes held in memory: {mb:.0f} MB")
    return ct, gt, pr          # axis order (z, y, x)


# How the CT is SHOWN, not what the model gets: a standard liver window
# (level 40, width 400 HU). The png export's per-slice stretch makes every
# patient a different brightness; a fixed window makes them comparable.
DISPLAY_WINDOW = (-160, 240)


def test_patients(keys, cfg):
    """the patients training held out: same sort, same seeded shuffle, same cut"""
    import random
    shuffled = sorted(keys)
    random.Random(cfg.seed).shuffle(shuffled)
    return shuffled[:max(1, round(len(shuffled) * cfg.test_fraction))]


def nifti_patient(cfg, want=None):
    """one labelled patient from the nifti slice cache

    IN   cfg  - the model's config, a nifti-trained one
         want - 16, "16" or "volume-16"; None = the first test patient
    OUT  (key, index entry, cache_dir, held_out, every labelled entry by key)
    """
    from src.nifti import nifti_slices
    _, _, volumes, cache_dir = nifti_slices(cfg, verbose=False)   # builds the cache if missing
    train = {k: e for k, e in volumes.items() if e["split"] == "train"}
    test = test_patients(train, cfg)
    if want is None:
        key = test[0]
    else:
        key = want if str(want).startswith("volume") else f"volume-{want}"
    if key not in train:
        raise SystemExit(f"no such labelled patient: {key}")
    return key, train[key], cache_dir, key in test, train


def nifti_z_scale(entry, side):
    """slice thickness in units of one in-plane pixel at this resolution, from
    the nifti header - the real value the png path has to guess"""
    return entry["spacing"][2] / (entry["spacing"][0] * entry["shape"][0] / side)


@torch.no_grad()
def build_volume_nifti(model, cfg, entry, cache_dir, device, all_slices=False,
                       stride=1, batch=16, verbose=True):
    """build_volume for the nifti cache: segment one patient, stack it in 3D

    IN   cfg   - the model's config; its hu_window turns HU into the model's
                 input, exactly as NiftiSliceDataset does in training
         entry - the patient's index entry, from nifti_patient()
         all_slices - keep slices whose truth has no liver
    OUT  ct (display, uint8 through DISPLAY_WINDOW), gt, pr  - (z, y, x)
         zs - which cache slices they are, for the HU export
    """
    from src.nifti import open_volume, scale_hu
    img, lab = open_volume(cache_dir, entry)
    zs = [z for z in range(entry["n_slices"])
          if all_slices or entry["liver_px"][z] + entry["tumor_px"][z]][::stride]
    n, side = len(zs), img.shape[1]
    ct = np.empty((n, side, side), np.uint8)
    gt = np.empty((n, side, side), np.uint8)
    pr = np.empty((n, side, side), np.uint8)
    # 2.5D models get each slice with its neighbours, as in training
    k = getattr(cfg, "in_slices", 1) // 2
    # a batch at a time, not one slice: the cache is fast enough that the
    # GPU would otherwise sit idle between slices
    for s in range(0, n, batch):
        chunk = zs[s:s + batch]
        hu = img[chunk].astype(np.float32)
        near = np.clip(np.asarray(chunk)[:, None] + np.arange(-k, k + 1), 0, img.shape[0] - 1)
        inp = img[near.ravel()].astype(np.float32).reshape(len(chunk), 2 * k + 1, side, side)
        x = torch.from_numpy((scale_hu(inp, cfg.hu_window) - 0.5) / 0.5)    # mean 0.5, std 0.5
        pr[s:s + len(chunk)] = model(x.to(device)).argmax(1).cpu().numpy()
        ct[s:s + len(chunk)] = np.rint(scale_hu(hu, DISPLAY_WINDOW) * 255)
        gt[s:s + len(chunk)] = lab[chunk]
        if verbose and (s // batch) % 8 == 7:
            print(f"    {s + len(chunk)}/{n} slices")
    return ct, gt, pr, zs


def add_surface(ax, vol, cls, color, alpha, spacing, step):
    """Turn one class of a label volume into a 3D surface and draw it.

    IN   ax      - a matplotlib 3D axes to draw into
         vol     - a label volume (z, y, x) with values 0/1/2
         cls     - which class to draw: 1 = liver, 2 = tumour
         color   - RGB tuple, 0.0-1.0
         alpha   - 0 = invisible, 1 = solid. The liver is drawn semi-transparent
                   so the tumour inside it can be seen through it.
         spacing - (z, y, x) real size of one voxel. The z entry is what stops
                   the body looking squashed or stretched.
         step    - marching-cubes step size. Bigger = fewer triangles = coarser
                   but much faster.

    OUT  True if something was drawn, False if there was nothing to draw.
         The caller uses that to print "nothing segmented" instead of an
         empty box.

    HOW IT WORKS
    marching_cubes takes a 3D array and finds the surface where the values
    cross a threshold - the boundary between "inside this class" and "outside".
    It gives back `verts` (3D points) and `faces` (which triples of points form
    each triangle). plot_trisurf then draws that triangle mesh.
    """
    from skimage import measure     # imported here so the file still runs
                                    # for people who have not installed it
    mask = (vol == cls).astype(np.float32)

    # Fewer than 50 voxels is noise, not an organ, and marching cubes on an
    # almost-empty array tends to throw rather than return an empty mesh.
    if mask.sum() < 50:
        return False
    try:
        # level=0.5 is the halfway point between 0 and 1, i.e. the boundary of
        # the mask. The `_` outputs are per-vertex normals and values, unused.
        verts, faces, _, _ = measure.marching_cubes(
            mask, level=0.5, spacing=spacing, step_size=step)
    except (RuntimeError, ValueError):
        # Happens when the mask touches the array edge in an awkward way.
        # A missing surface is better than a crashed script.
        return False

    # Note the column order: verts columns are (z, y, x), and plot_trisurf
    # wants (x, y, triangles, z). So column 2 is x, column 1 is y, column 0
    # is the height. Getting this wrong silently rotates the patient.
    ax.plot_trisurf(verts[:, 2], verts[:, 1], faces, verts[:, 0],
                    color=color, alpha=alpha, lw=0, shade=True)
    return True


def surfaces_figure(gt, pr, out_path, spacing, step, title, dpi=110):
    """Write the 3D figure: truth on the top row, prediction below, 3 angles each.

    IN   gt, pr   - label volumes (z, y, x)
         out_path - where to write the PNG
         spacing  - (z, y, x) voxel size, passed straight to add_surface
         step     - marching-cubes coarseness
         title    - printed across the top
         dpi      - resolution of the saved PNG

    OUT  nothing; writes a PNG and prints the path.

    USE CASE
    Comparing the SHAPE of the prediction against the truth. The numbers can
    look fine while the 3D shows the prediction broken into pieces, and this
    figure is where you see that.
    """
    # (elevation, azimuth, label) - three camera positions around the patient.
    views = [(18, -70, "front"), (18, 20, "side"), (72, -70, "top")]
    fig = plt.figure(figsize=(4.6 * len(views), 9.4))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Two rows: ground truth, then prediction. Same columns, same camera
    # angles, so the two rows are directly comparable.
    for row, (vol, label) in enumerate([(gt, "ground truth"), (pr, "prediction")]):
        for col, (elev, azim, name) in enumerate(views):
            # add_subplot indexes from 1, hence the +1.
            ax = fig.add_subplot(2, len(views), row * len(views) + col + 1,
                                 projection="3d")
            # Liver at alpha 0.28 so the tumour shows through it; tumour at
            # 0.95 so it reads as solid.
            got_liver = add_surface(ax, vol, 1, LIVER_COLOR, 0.28, spacing, step)
            got_tumor = add_surface(ax, vol, 2, TUMOR_COLOR, 0.95, spacing, step)

            ax.view_init(elev=elev, azim=azim)
            ax.set_box_aspect((1, 1, 1))    # do not let matplotlib stretch axes
            # Tick numbers on a 3D plot are voxel indices; they mean nothing to
            # a reader, so strip them and the axis labels.
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")

            if row == 0:                    # column heading, top row only
                ax.set_title(name, fontsize=11)
            if col == 0:                    # row heading, first column only
                # text2D places text in axes coordinates (0-1), not data
                # coordinates, which is what you want for a label.
                ax.text2D(-0.06, 0.5, label, transform=ax.transAxes, rotation=90,
                          va="center", ha="center", fontsize=12, fontweight="bold")
            if not (got_liver or got_tumor):
                ax.text2D(0.5, 0.5, "nothing segmented", transform=ax.transAxes,
                          ha="center", fontsize=9, color="grey")

    # A shared legend. Rectangle((0,0),1,1) is the usual trick for a plain
    # colour swatch - the size does not matter, only the face colour.
    handles = [plt.Rectangle((0, 0), 1, 1, fc=LIVER_COLOR, alpha=.6),
               plt.Rectangle((0, 0), 1, 1, fc=TUMOR_COLOR)]
    fig.legend(handles, ["liver", "tumour"], loc="lower center", ncol=2,
               frameon=False, fontsize=11)
    # rect leaves room for the suptitle at the top and the legend at the bottom.
    fig.tight_layout(rect=[0.02, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)      # free the figure, or memory grows over repeated runs
    print(f"[saved] {out_path}")


def slices_figure(ct, gt, pr, out_path, title, dpi=110):
    """Write the 2D figure: axial, coronal and sagittal cuts through the tumour.

    IN   ct, gt, pr - the three volumes from build_volume()
         out_path   - where to write the PNG
         title      - printed across the top
         dpi        - resolution

    OUT  nothing; writes a PNG and prints the path.

    USE CASE
    The axial cut is a slice the network actually saw. The coronal and
    sagittal ones are rebuilt from the stack, so they cut ACROSS many
    predictions at once - which is exactly where slice-to-slice inconsistency
    shows up as a ragged edge.
    """
    from matplotlib.colors import ListedColormap
    # A 3-entry colormap indexed by the label value:
    #   0 -> fully transparent (alpha 0), so the CT shows through
    #   1 -> liver, 55% opaque
    #   2 -> tumour, 85% opaque
    cmap = ListedColormap([(0, 0, 0, 0), (*LIVER_COLOR, .55), (*TUMOR_COLOR, .85)])

    # Centre the cuts on the tumour if there is one, otherwise on the liver.
    # A cut through an arbitrary index usually misses the interesting part.
    target = 2 if (gt == 2).any() else 1
    zs, ys, xs = np.where(gt == target) if (gt == target).any() else np.where(gt >= 0)
    # median, not mean: the median is an actual coordinate inside the object
    # even when the object is a strange shape.
    cz, cy, cx = int(np.median(zs)), int(np.median(ys)), int(np.median(xs))

    # Each entry: (name, how to take that cut, label for the title).
    # The lambdas are just numpy slicing on a (z, y, x) array:
    #   v[cz]        one whole slice          -> axial
    #   v[:, cy, :]  fix y, keep z and x      -> coronal
    #   v[:, :, cx]  fix x, keep z and y      -> sagittal
    planes = [
        ("axial",    lambda v: v[cz],            f"z = {cz}"),
        ("coronal",  lambda v: v[:, cy, :],      f"y = {cy}"),
        ("sagittal", lambda v: v[:, :, cx],      f"x = {cx}"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for col, (name, cut, where) in enumerate(planes):
        base = cut(ct)          # the CT underneath, same cut for both rows
        for row, (vol, label) in enumerate([(gt, "ground truth"), (pr, "prediction")]):
            ax = axes[row][col]
            # Draw twice: greyscale CT first, then the labels on top. The
            # transparent 0 entry in the colormap is what lets the CT show.
            ax.imshow(base, cmap="gray", aspect="auto", origin="lower")
            # vmin/vmax pin 0->first colour and 2->last, so the mapping does
            # not shift on a slice that happens to contain no tumour.
            # interpolation="nearest" keeps label edges hard instead of blurring
            # class 1 into class 2.
            ax.imshow(cut(vol), cmap=cmap, vmin=0, vmax=2,
                      interpolation="nearest", aspect="auto", origin="lower")
            if row == 0:
                ax.set_title(f"{name}   ({where})", fontsize=11)
            if col == 0:
                ax.set_ylabel(label, fontsize=12, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"[saved] {out_path}")


def volume_stats(gt, pr):
    """Dice for this one patient, per class, counted over the whole volume.

    IN   gt, pr - label volumes (z, y, x)

    OUT  list of (class_name, true_voxels, predicted_voxels, dice), one entry
         per foreground class. Background is skipped - it is 98% of the image
         and its Dice is always near 1, which tells you nothing.

    Dice = 2 * overlap / (size of truth + size of prediction). It is 1.0 for a
    perfect match and 0.0 when they do not touch at all.

    NOTE ON SCOPE
    This is a PER-PATIENT number, over one volume. The Dice reported in
    evaluate.py is pooled over every test pixel of every patient, so the two
    will not be equal and are not meant to be compared directly.
    """
    out = []
    for cls, name in enumerate(CLASS_NAMES):    # 0=background, 1=liver, 2=tumour
        if cls == 0:
            continue
        g, p = (gt == cls), (pr == cls)         # boolean masks
        inter = int((g & p).sum())              # voxels where both agree
        # max(..., 1) guards against 0/0 when a class is in neither volume.
        d = 2 * inter / max(int(g.sum()) + int(p.sum()), 1)
        out.append((name, int(g.sum()), int(p.sum()), d))
    return out


def main():
    """Parse the command line, do the work, write the files.

    Reads nothing and returns nothing - all its input is command-line
    arguments plus files on disk, and all its output is files in results/
    plus what it prints.
    """
    from src.config import RUNS_DIR

    p = argparse.ArgumentParser(description="3D view of one patient's segmentation")
    p.add_argument("--group", default="nifti/baseline",
                   help="folder under runs/, e.g. nifti/baseline or internship_png/ablation_2070")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--volume", default=None,
                   help="patient number, e.g. 16. default: the first test patient")
    p.add_argument("--list", action="store_true", help="list patients and exit")
    p.add_argument("--stride", type=int, default=1,
                   help="use every Nth slice; 2 halves the work with little visual loss")
    p.add_argument("--all-slices", action="store_true",
                   help="keep the slices whose ground truth has no liver. Off by "
                        "default: they carry no true positives, so all they add "
                        "to the volume is free-floating false positives")
    p.add_argument("--step", type=int, default=2,
                   help="marching-cubes step size; higher = coarser but faster")
    p.add_argument("--z-scale", type=float, default=None,
                   help="through-plane spacing in units of the in-plane pixel. "
                        "nifti: read from the scan header. png: estimated as "
                        "320/n_slices, since the export is 256x256 at roughly "
                        "1.4 mm and an abdominal scan covers about 450 mm")
    p.add_argument("--nifti", action="store_true", help="also write a .nii.gz for 3D Slicer")
    p.add_argument("--dpi", type=int, default=110)
    a = p.parse_args()

    # ---- 1. get the trained model back -----------------------------------
    device = get_device()
    # If no --ckpt was given, take the first best_model.pt in the group.
    # next(iter(...), None) returns None instead of raising when it is empty.
    ckpt = Path(a.ckpt) if a.ckpt else next(
        iter(sorted((Path(RUNS_DIR) / a.group).glob("*/best_model.pt"))), None)
    if ckpt is None:
        raise SystemExit(f"no checkpoint under {Path(RUNS_DIR) / a.group}")

    # load_checkpoint gives back the model AND the config it was trained with,
    # so the split below is the one this model actually saw.
    model, cfg, ck = load_checkpoint(ckpt, device)
    print(f"[load] {ckpt}")
    print(f"[load] arch={cfg.arch}  epoch={ck.get('epoch')}  "
          f"params={model.count_parameters():,}")

    if cfg.data_format == "nifti":
        return main_nifti(a, model, cfg, device)

    # ---- 2. find the data ------------------------------------------------
    pairs, _, _ = find_pairs(cfg.root_data_dir)
    _, by = patient_pairs(pairs)        # dict form: every patient

    if a.list:
        print(f"\n{len(by)} patients:")
        # Sort by the NUMBER, not the string, so 2 comes before 10.
        for v in sorted(by, key=lambda s: int(s.split('-')[1])):
            print(f"   {v:<14}{len(by[v]):>5} slices")
        return

    # ---- 3. work out which patients were held out ------------------------
    # Repeat the same shuffle training used, with the same seed, so we can tell
    # whether the patient being viewed was in the test set. A patient the model
    # trained on will look great and prove nothing, so this is worth knowing.
    import random
    vols = sorted(by)                   # sorted first, or the shuffle is not reproducible
    rng = random.Random(cfg.seed)       # a private generator, so we do not
    shuffled = list(vols)               # disturb global random state
    rng.shuffle(shuffled)
    test_vols = shuffled[:max(1, round(len(vols) * cfg.test_fraction))]

    # ---- 4. pick the patient --------------------------------------------
    want = a.volume or test_vols[0].split("-")[1]
    key, slices = patient_pairs(pairs, want)
    held_out = key in test_vols
    total = len(slices)                 # before any filtering; needed for z_scale
    print(f"[data] {key}: {total} slices"
          f"   ({'held-out test patient' if held_out else 'WARNING: this patient was in training'})")

    # ---- 5. drop the empty slices ---------------------------------------
    # Same rule as skip_empty in training. Slices with no liver in the ground
    # truth contain no true positives, so the only thing they can add to a 3D
    # volume is false-positive blobs floating where the organ is not.
    if not a.all_slices:
        from src.data import has_foreground
        slices = [(ip, mp) for ip, mp in slices if has_foreground(mp)]
        print(f"[data] keeping {len(slices)} of {total} slices "
              f"(ground truth contains liver)")

    if a.stride > 1:
        print(f"[data] using every {a.stride}th slice")

    # ---- 6. the actual work ---------------------------------------------
    print("[run ] segmenting the whole volume ...")
    ct, gt, pr = build_volume(model, slices, device, cfg.image_size, a.stride)
    print(f"[run ] volume shape {pr.shape}  (z, y, x)")

    print("\nper-volume overlap:")
    for name, g, pp, d in volume_stats(gt, pr):
        print(f"   {name:<10} true {g:>9,} vox   predicted {pp:>9,} vox   Dice {d:.4f}")

    # ---- 7. work out the voxel spacing ----------------------------------
    out = results_dir("png")
    tag = key.replace("-", "")          # "volume-16" -> "volume16", for filenames

    # The 2D PNG export threw away the original spacing, so nothing in the
    # files says how thick a slice is. Estimate it from the slice count:
    # 256x256 at ~1.4 mm per pixel, an abdomen is ~450 mm tall, so
    # 450/n_slices mm per slice / 1.4 mm per pixel ~= 320/n_slices.
    # Note it uses `total`, the ORIGINAL slice count - the thickness of a slice
    # does not change just because we dropped some of them.
    z_scale = a.z_scale if a.z_scale else 320.0 / max(total, 1)
    print(f"[data] through-plane spacing {z_scale:.2f} in-plane pixels")
    # Multiplied by stride: skipping every other slice makes the gap twice as big.
    spacing = (z_scale * a.stride, 1.0, 1.0)
    title = (f"{key} - {cfg.arch}   "
             f"{'held-out test patient' if held_out else 'seen in training'}")

    # ---- 8. write the figures -------------------------------------------
    slices_figure(ct, gt, pr, out / f"view3d_{tag}_slices.png", title, a.dpi)

    # The surface figure is optional in two ways: scikit-image may not be
    # installed, and marching cubes on a big volume can run out of memory.
    # Neither should lose the 2D figure that was already written above.
    try:
        surfaces_figure(gt, pr, out / f"view3d_{tag}_surfaces.png",
                        spacing, a.step, title, a.dpi)
    except ImportError:
        print("[3d  ] scikit-image not installed, skipping the surface figure")
        print("       pip install scikit-image")
    except MemoryError:
        print(f"[3d  ] out of memory building the surface. Try --stride "
              f"{max(2, a.stride * 2)} or --step {a.step + 2}")









    if a.nifti:
        try:
            import nibabel as nib
        except ImportError:
            print("[nii ] nibabel not installed:  pip install nibabel")
        else:
            aff = np.diag([1.0, 1.0, z_scale * a.stride, 1.0])
            for name, vol in [("ct", ct), ("truth", gt), ("pred", pr)]:
                f = out / f"view3d_{tag}_{name}.nii.gz"
                # our arrays are (z, y, x); NIfTI expects (x, y, z), so the
                # axes are reversed
                nib.save(nib.Nifti1Image(np.transpose(vol, (2, 1, 0)), aff), f)
                print(f"[saved] {f}")
            print("[nii ] open the ct file in 3D Slicer, then load pred as a segmentation")


def main_nifti(a, model, cfg, device):
    """main() for a patient read from the nifti slice cache.

    Same figures as the png path, two things better: the slice thickness comes
    from the header instead of the 320/n_slices guess, and --nifti writes the
    CT in Hounsfield units with the true voxel size in mm, so 3D Slicer shows
    the patient at real scale and its window/level presets work.
    """
    from src.nifti import open_volume

    key, entry, cache_dir, held_out, train = nifti_patient(cfg, a.volume)
    if a.list:
        print(f"\n{len(train)} labelled patients (nifti):")
        for k in sorted(train, key=lambda s: int(s.split("-")[1])):
            e = train[k]
            print(f"   {k:<14}{e['n_slices']:>5} slices  {e['spacing'][2]:5.2f} mm"
                  f"  tumour {e['tumor_ml']:7.1f} mL")
        return
    sp = entry["spacing"]
    print(f"[data] {key}: {entry['n_slices']} slices, {sp[2]:.2f} mm thick, {sp[0]:.3f} mm pixels"
          f"   ({'held-out test patient' if held_out else 'WARNING: this patient was in training'})")
    print(f"[data] model input: HU window {list(cfg.hu_window)}")

    print("[run ] segmenting the whole volume ...")
    ct, gt, pr, zs = build_volume_nifti(model, cfg, entry, cache_dir, device,
                                        a.all_slices, a.stride)
    print(f"[run ] volume shape {pr.shape}  (z, y, x)   "
          f"{'every' if a.all_slices else 'liver-containing'} slice"
          + (f", every {a.stride}th" if a.stride > 1 else ""))

    print("\nper-volume overlap:")
    for name, g, pp, d in volume_stats(gt, pr):
        print(f"   {name:<10} true {g:>9,} vox   predicted {pp:>9,} vox   Dice {d:.4f}")

    out = results_dir("nifti")
    tag = key.replace("-", "")
    side = ct.shape[1]
    z_scale = a.z_scale or nifti_z_scale(entry, side)
    print(f"[data] through-plane spacing {z_scale:.2f} in-plane pixels (from the header)")
    spacing = (z_scale * a.stride, 1.0, 1.0)
    title = (f"{key} - {cfg.arch} - nifti   "
             f"{'held-out test patient' if held_out else 'seen in training'}")

    slices_figure(ct, gt, pr, out / f"view3d_{tag}_slices.png", title, a.dpi)
    try:
        surfaces_figure(gt, pr, out / f"view3d_{tag}_surfaces.png",
                        spacing, a.step, title, a.dpi)
    except ImportError:
        print("[3d  ] scikit-image not installed, skipping the surface figure")
    except MemoryError:
        print(f"[3d  ] out of memory building the surface. Try --stride "
              f"{max(2, a.stride * 2)} or --step {a.step + 2}")

    if a.nifti:
        import nibabel as nib
        img, _ = open_volume(cache_dir, entry)
        # mm per cached pixel: the scan's pixel size times how much it was shrunk
        aff = np.diag([sp[0] * entry["shape"][0] / side, sp[1] * entry["shape"][1] / side,
                       sp[2] * a.stride, 1.0])
        # the cache is (z, y, x) of a RAS volume, so reversing the axes gives
        # back (x, y, z) RAS and a positive diagonal affine is the right one
        for name, vol in [("ct", np.asarray(img[zs])), ("truth", gt), ("pred", pr)]:
            f = out / f"view3d_{tag}_{name}.nii.gz"
            nib.save(nib.Nifti1Image(np.transpose(vol, (2, 1, 0)), aff), f)
            print(f"[saved] {f}")
        print("[nii ] ct is in HU at true scale: open it in 3D Slicer, then load pred "
              "as a segmentation")










if __name__ == "__main__":
    main()
