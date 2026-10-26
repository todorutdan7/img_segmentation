"""Read MSD Task03_Liver: one .nii.gz per CT scan, in Hounsfield units (HU).

WHY A CACHE
A .nii.gz is one gzip stream. Reading slice 402 means decompressing the whole
volume first, up to 987 slices and 1 GB as float32, on a machine with 7 GB of
RAM. So every volume is converted ONCE into plain .npy arrays at the training
resolution - still in HU, only smaller. numpy memory-maps those, and one slice
is one 128 KB read:

    data/nifti_cache_256/
        index.json            per volume: spacing, pixel counts per slice, volumes in mL
        img_liver_16.npy      (slices, 256, 256) int16   Hounsfield units
        lab_liver_16.npy      (slices, 256, 256) uint8   0 background, 1 liver, 2 tumour
        hist_liver_16.npy     (3, 4096) int64            HU histogram: all, liver, tumour

    python -m src.nifti                         # 512px, the default; resumable if stopped
    python -m src.nifti --resolution half       # 256px, for a smaller GPU
    python -m src.nifti --include-test          # also the 70 unlabelled test scans

RESOLUTION
One cache per training size, nifti_cache_512 / _256 / _128 / _64. A smaller
one is downsampled from a larger cache in about a minute; otherwise the
.nii.gz files are read, ~45 min. Disk for all 131 scans: 512px ~46 GB, 256px ~11.5 GB,
128px ~2.9 GB, 64px ~0.7 GB.

WHAT THE MODEL GETS
A slice is read from the cache in HU, clipped to cfg.hu_window and mapped to
0..1 (scale_hu), then normalised and augmented exactly like the png slices
were. liver_N here is LiTS volume-N, so the seeded split puts the same patients
in test as the internship runs. Slices are stored (z, y, x) with each one the
RAS x-y plane transposed - the orientation of the internship's png export.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.config import Config, RESOLUTIONS
from src.data import LiTSDataset

HU_MIN, HU_MAX = -1024, 3071          # histogram range, one bin per HU
ID_RE = re.compile(r"^liver_(\d+)\.nii(\.gz)?$")     # also skips macOS "._" files


def volume_files(root, split="Tr"):
    """(id, image path, label path or None) for every scan in one split, by id"""
    root = Path(root)
    img_dir = root / f"images{split}"
    if not img_dir.is_dir():
        raise FileNotFoundError(
            f"no {img_dir}\nexpected the MSD layout: <root>/imagesTr, labelsTr, imagesTs")
    out = []
    for p in img_dir.iterdir():
        m = ID_RE.match(p.name)
        if m:
            lab = root / "labelsTr" / p.name if split == "Tr" else None
            out.append((int(m.group(1)), p, lab))
    return sorted(out)


def key_of(vid):
    # the id data.volume_id() reads from a png filename, so a seeded split by
    # volume puts the same patients in test whichever format is loaded
    return f"volume-{vid}"


def _resize(vol, size, label):
    """(z, y, x) -> (z, size, size), 64 slices at a time to keep memory flat.

    area for CT: when halving 512 it is the 2x2 mean. nearest for labels,
    interpolating would invent class ids.
    """
    out = np.empty((vol.shape[0], size, size), np.uint8 if label else np.int16)
    for s in range(0, vol.shape[0], 64):
        t = torch.from_numpy(np.ascontiguousarray(vol[s:s + 64], dtype=np.float32))[:, None]
        r = F.interpolate(t, size=(size, size), mode="nearest" if label else "area")[:, 0].numpy()
        out[s:s + 64] = r if label else np.rint(r)
    return out


def _save(path, arr):
    # write under a temporary name first, so a build stopped mid-write never
    # leaves a truncated file that looks finished
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def prepare_volume(vid, img_path, lab_path, cache_dir, size):
    """convert one scan into the cache, return its index entry"""
    import nibabel as nib
    from scipy import ndimage

    t0 = time.time()
    im = nib.as_closest_canonical(nib.load(str(img_path)))      # RAS, a no-op for MSD
    sx, sy, sz = (float(z) for z in im.header.get_zooms()[:3])
    img = np.asanyarray(im.dataobj).astype(np.float32, copy=False)     # (x, y, z)
    lab = None
    if lab_path is not None:
        lab = np.asanyarray(nib.as_closest_canonical(nib.load(str(lab_path))).dataobj).astype(np.uint8)
        if lab.shape != img.shape:
            raise ValueError(f"liver_{vid}: image {img.shape} but label {lab.shape}")

    hist = np.zeros((3, HU_MAX - HU_MIN + 1), np.int64)
    for s in range(0, img.shape[2], 32):
        h = (np.rint(np.clip(img[:, :, s:s + 32], HU_MIN, HU_MAX)) - HU_MIN).astype(np.int16)
        hist[0] += np.bincount(h.ravel(), minlength=hist.shape[1])
        if lab is not None:
            l = lab[:, :, s:s + 32]
            hist[1] += np.bincount(h[l == 1], minlength=hist.shape[1])
            hist[2] += np.bincount(h[l == 2], minlength=hist.shape[1])

    name = f"liver_{vid}"
    _save(cache_dir / f"img_{name}.npy", _resize(np.transpose(img, (2, 1, 0)), size, False))
    _save(cache_dir / f"hist_{name}.npy", hist)

    entry = {
        "id": vid, "key": key_of(vid), "file": img_path.name,
        "split": "train" if lab is not None else "test",
        "shape": [int(s) for s in img.shape], "spacing": [sx, sy, sz],
        "n_slices": int(img.shape[2]),
        "hu_min": float(img.min()), "hu_max": float(img.max()),
    }
    del img

    if lab is not None:
        lab_c = _resize(np.transpose(lab, (2, 1, 0)), size, True)
        _save(cache_dir / f"lab_{name}.npy", lab_c)

        vox_ml = sx * sy * sz / 1000
        # not np.bincount(lab.ravel()): that silently copies the volume to int64,
        # 2 GB for the longest scan
        counts = [int(np.count_nonzero(lab == k)) for k in range(3)]
        fg = lab > 0
        fg_z = np.where(fg.any((0, 1)))[0]

        # tumours as 3D connected lesions, counted inside the liver's bounding
        # box only - labelling the full volume would need another 1 GB
        lesions = []
        if counts[2]:
            xs, ys = np.where(fg.any(2).any(1))[0], np.where(fg.any(2).any(0))[0]
            sub = lab[xs[0]:xs[-1] + 1, ys[0]:ys[-1] + 1, fg_z[0]:fg_z[-1] + 1] == 2
            cc, n = ndimage.label(sub)
            sizes = np.bincount(cc[cc > 0])[1:]
            lesions = sorted((sizes * vox_ml).round(4).tolist(), reverse=True)

        entry.update({
            "voxels": {"background": int(counts[0]), "liver": int(counts[1]), "tumor": int(counts[2])},
            "liver_ml": round(float((counts[1] + counts[2]) * vox_ml), 2),     # whole organ, lesions included
            "tumor_ml": round(float(counts[2] * vox_ml), 3),
            "lesions_ml": lesions,
            "fg_slices_native": fg_z.tolist(),
            # pixel counts per slice at the cache resolution, what training sees
            "liver_px": (lab_c == 1).sum((1, 2)).tolist(),
            "tumor_px": (lab_c == 2).sum((1, 2)).tolist(),
        })
    entry["seconds"] = round(time.time() - t0, 1)
    return entry


def find_scan(scan, root):
    """(image path, label path or None) for a scan given as 16, "16" or a path

    A number is looked up in imagesTr (labelled) and then imagesTs (not).
    For a path, the label is the file of the same name in ../labelsTr.
    """
    root = Path(root)
    if str(scan).isdigit():
        for split in ("Tr", "Ts"):
            p = root / f"images{split}" / f"liver_{int(scan)}.nii.gz"
            if p.exists():
                break
        else:
            raise SystemExit(f"no liver_{scan}.nii.gz in {root}/imagesTr or imagesTs")
    else:
        p = Path(scan)
        if not p.exists():
            raise SystemExit(f"no such file: {p}")
    lab = p.parent.parent / "labelsTr" / p.name
    return p, (lab if lab.exists() else None)


def load_scan(img_path, lab_path=None, size=None):
    """one whole scan straight from its .nii.gz - no cache, no model

    IN   size - None keeps the native 512x512; a number resizes every slice
    OUT  hu      (z, y, x) float32 Hounsfield units, each slice the RAS x-y
                 plane transposed, the same orientation as the cache
         lab     (z, y, x) uint8 0/1/2, or None without a label file
         spacing (z, y, x) voxel size in mm, at the returned size
    """
    import nibabel as nib
    im = nib.as_closest_canonical(nib.load(str(img_path)))
    sx, sy, sz = (float(z) for z in im.header.get_zooms()[:3])
    hu = np.transpose(np.asanyarray(im.dataobj).astype(np.float32, copy=False), (2, 1, 0))
    lab = None
    if lab_path is not None:
        lab = np.transpose(np.asanyarray(
            nib.as_closest_canonical(nib.load(str(lab_path))).dataobj).astype(np.uint8), (2, 1, 0))
    ny, nx = hu.shape[1:]
    if size and (ny, nx) != (size, size):
        hu = _resize(hu, size, False).astype(np.float32)
        lab = _resize(lab, size, True) if lab is not None else None
        sy, sx = sy * ny / size, sx * nx / size
    return hu, lab, (sz, sy, sx)


def load_index(cache_dir):
    p = Path(cache_dir) / "index.json"
    return json.loads(p.read_text()) if p.exists() else None


def cache_sizes():
    """resolutions with a complete slice cache in data/, e.g. [256, 512]"""
    from src.config import DATA_DIR
    found = []
    for d in DATA_DIR.glob("nifti_cache_*"):
        idx = load_index(d)
        if idx and idx.get("complete_train"):
            found.append(idx["image_size"])
    return sorted(found)


def _larger_cache(cache_dir, size):
    """a complete cache at a multiple of `size`, to downsample instead of
    decompressing every .nii.gz again. Smallest such multiple, or None."""
    found = []
    for d in Path(cache_dir).parent.glob("nifti_cache_*"):
        idx = load_index(d)
        if (idx and idx.get("complete_train") and idx["image_size"] > size
                and idx["image_size"] % size == 0):
            found.append((idx["image_size"], d, idx))
    return min(found, key=lambda t: t[0])[1:] if found else None


def _derive_volume(src_dir, e, cache_dir, size):
    """one volume from a larger cache. Same arrays as converting the .nii.gz
    directly, up to rounding HU twice: a mean of 2x2 means is the 4x4 mean,
    and nearest picks the same voxel either way. Seconds, no gzip."""
    t0 = time.time()
    img, lab = open_volume(src_dir, e)
    name = f"liver_{e['id']}"
    _save(cache_dir / f"img_{name}.npy", _resize(img, size, False))
    _save(cache_dir / f"hist_{name}.npy", np.load(Path(src_dir) / f"hist_{name}.npy"))
    e = dict(e)
    if lab is not None:
        lab_c = _resize(lab, size, True)
        _save(cache_dir / f"lab_{name}.npy", lab_c)
        e["liver_px"] = (lab_c == 1).sum((1, 2)).tolist()
        e["tumor_px"] = (lab_c == 2).sum((1, 2)).tolist()
    e["seconds"] = round(time.time() - t0, 1)
    return e


def _check_disk(cache_dir, n_slices, size):
    import shutil
    need = n_slices * size * size * 3              # int16 image + uint8 label
    free = shutil.disk_usage(Path(cache_dir).anchor or ".").free
    if need > free - 5e9:
        raise SystemExit(
            f"[nifti] a {size}px cache needs ~{need/1e9:.1f} GB, only {free/1e9:.1f} GB "
            f"free (keeping 5 GB spare). Free some space or pick a lower --resolution.")
    return need


def build_cache(root, cache_dir, size, include_test=False, verbose=True):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    index = load_index(cache_dir) or {"image_size": size, "volumes": {}}
    if index["image_size"] != size:
        raise ValueError(f"{cache_dir} holds {index['image_size']}px slices, not {size}")

    done = lambda key, vid: (key in index["volumes"]
                             and (cache_dir / f"img_liver_{vid}.npy").exists())
    larger = _larger_cache(cache_dir, size)
    if larger:
        src_dir, src = larger
        jobs = [(e["id"], e) for k, e in sorted(src["volumes"].items(), key=lambda t: t[1]["id"])
                if (include_test or e["split"] == "train") and not done(k, e["id"])]
        n_slices = sum(e["n_slices"] for _, e in jobs)
        run = lambda job: _derive_volume(src_dir, job[1], cache_dir, size)
        where = f"downsampling the {src['image_size']}px cache {src_dir}"
    else:
        import nibabel as nib
        files = volume_files(root, "Tr") + (volume_files(root, "Ts") if include_test else [])
        jobs = [(vid, (ip, lp)) for vid, ip, lp in files if not done(key_of(vid), vid)]
        n_slices = sum(nib.load(str(ip)).shape[2] for _, (ip, _) in jobs)     # headers only
        run = lambda job: prepare_volume(job[0], *job[1], cache_dir, size)
        where = f"converting {root}"
    need = _check_disk(cache_dir, n_slices, size) if jobs else 0
    if verbose:
        print(f"[nifti] {size}px cache {cache_dir}: "
              + (f"{len(jobs)} volume(s), {n_slices:,} slices, ~{need/1e9:.1f} GB, {where}"
                 if jobs else "already complete"))

    t0 = time.time()
    for n, job in enumerate(jobs, 1):
        e = run(job)
        index["volumes"][e["key"]] = e
        # rewritten after every volume, so a stopped build resumes where it was
        tmp = cache_dir / "index.tmp"
        tmp.write_text(json.dumps(index))
        os.replace(tmp, cache_dir / "index.json")
        if verbose:
            left = (time.time() - t0) / n * (len(jobs) - n)
            print(f"[nifti] {n:3d}/{len(jobs)}  liver_{e['id']:<3d} {e['n_slices']:4d} slices "
                  f"{e['spacing'][2]:.2f} mm  {e['seconds']:5.1f}s   ~{left/60:.0f} min left",
                  flush=True)

    n_train = (sum(1 for e in larger[1]["volumes"].values() if e["split"] == "train")
               if larger else len(volume_files(root, "Tr")))
    trained = sum(1 for v in index["volumes"].values() if v["split"] == "train")
    index["complete_train"] = trained == n_train
    index["orientation"] = "(z, y, x): every slice is the RAS x-y plane transposed, as in lits-png"
    (cache_dir / "index.json").write_text(json.dumps(index))
    return index


def nifti_slices(cfg, verbose=True):
    """every usable (volume key, slice) of the labelled scans, building the cache if needed

    OUT  items     [("volume-0", 41), ...]   skip_empty drops slices whose
                   label, at the training resolution, is pure background
         n_all     slices before that filter
         volumes   index entries by key, the dataset needs them for scaling
         cache_dir
    """
    cache_dir = cfg.nifti_cache_dir
    index = load_index(cache_dir)
    if not (index and index.get("complete_train")):
        if verbose:
            print("[nifti] slice cache missing or incomplete - building it once, "
                  "this takes a while. Same as: python -m src.nifti")
        index = build_cache(cfg.root_nifti_dir, cache_dir, cfg.image_size, verbose=verbose)

    vols = sorted((v for v in index["volumes"].values() if v["split"] == "train"),
                  key=lambda v: v["id"])
    items, n_all = [], 0
    for v in vols:
        n_all += v["n_slices"]
        for z in range(v["n_slices"]):
            if cfg.skip_empty and v["liver_px"][z] + v["tumor_px"][z] == 0:
                continue
            items.append((v["key"], z))
    return items, n_all, index["volumes"], cache_dir


def open_volume(cache_dir, entry):
    """(img, lab) of one cached scan as read-only memory maps, (z, y, x).
    lab is None for the unlabelled test scans."""
    name = f"liver_{entry['id']}"
    img = np.load(Path(cache_dir) / f"img_{name}.npy", mmap_mode="r")
    lp = Path(cache_dir) / f"lab_{name}.npy"
    return img, (np.load(lp, mmap_mode="r") if lp.exists() else None)


def scale_hu(hu, hu_window):
    """Hounsfield units -> 0..1: clip to the window, then stretch it.
    Everything darker than the window is 0, everything brighter is 1."""
    lo, hi = hu_window
    return np.clip((hu - lo) / (hi - lo), 0, 1).astype(np.float32)


class NiftiSliceDataset(LiTSDataset):
    """LiTSDataset reading cached nifti slices instead of png files.

    Everything after loading - augmentation, normalisation to mean 0.5 std 0.5,
    liver_only - is inherited, so both formats reach the model the same way.
    """

    def __init__(self, items, cache_dir, volumes, augment=False, liver_only=False,
                 hu_window=Config.hu_window, mean=0.5, std=0.5):
        size = load_index(cache_dir)["image_size"]
        super().__init__(items, size, augment, liver_only, mean, std)
        self.cache_dir = Path(cache_dir)
        self.volumes = volumes
        self.hu_window = tuple(hu_window)
        self._maps = {}

    def scale(self, hu):
        """HU of one slice -> the model's 0..1 input"""
        return scale_hu(hu, self.hu_window)

    def __getstate__(self):
        # dataloader workers get a pickled copy; an open memmap would be
        # pickled as a full in-memory array, so each worker reopens its own
        state = self.__dict__.copy()
        state["_maps"] = {}
        return state

    def _load(self, i):
        key, z = self.pairs[i]
        if key not in self._maps:
            self._maps[key] = open_volume(self.cache_dir, self.volumes[key])
        img, lab = self._maps[key]
        x = torch.from_numpy(self.scale(img[z].astype(np.float32))).unsqueeze(0)
        y = torch.from_numpy(lab[z].astype(np.int64))
        if self.liver_only:
            y = (y > 0).long()
        return x, y


def main():
    p = argparse.ArgumentParser(
        description="convert MSD Task03_Liver .nii.gz into the slice cache training reads")
    p.add_argument("--nifti-dir", default="", help="default data/Task03_Liver")
    p.add_argument("--image-size", type=int, default=Config().image_size)
    p.add_argument("--resolution", choices=list(RESOLUTIONS), default=None,
                   help="full=512 half=256 quarter=128 eighth=64, overrides --image-size")
    p.add_argument("--include-test", action="store_true",
                   help="also convert the 70 unlabelled test scans, to look at predictions")
    a = p.parse_args()
    size = RESOLUTIONS[a.resolution] if a.resolution else a.image_size
    cfg = Config(nifti_dir=a.nifti_dir, image_size=size)
    build_cache(cfg.root_nifti_dir, cfg.nifti_cache_dir, cfg.image_size, include_test=a.include_test)


if __name__ == "__main__":
    main()
