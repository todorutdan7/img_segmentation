
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.config import Config, MASK_VALUES

IMAGE_DIR_NAMES = ["images", "image", "img", "ct", "train_images"]
MASK_DIR_NAMES = ["masks", "mask", "labels", "label", "segmentations", "train_masks"]
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SPLIT_IMG_RE = re.compile(r"^volume[-_](\d+)[-_](\d+)$", re.I)


# helper used to bypass redundant, nested dirs to find
# the actual folder where the image or mask files are stored

def _descend(d: Path) -> Path:
    for _ in range(3):
        files = [p for p in d.iterdir() if p.suffix.lower() in EXTS]
        subs = [p for p in d.iterdir() if p.is_dir()]
        if files or len(subs) != 1:
            return d
        d = subs[0]
    return d


def _split_layout_dir(root: Path, max_depth=4):
    stack = [(Path(root), 0)]
    while stack:
        d, depth = stack.pop()
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        names = {p.name for p in entries if p.suffix.lower() in EXTS}
        for n in names:
            m = SPLIT_IMG_RE.match(Path(n).stem)
            if m and f"segmentation-{m.group(1)}_livermask_{m.group(2)}{Path(n).suffix}" in names:
                return d
        if depth < max_depth:
            stack += [(p, depth + 1) for p in entries if p.is_dir()]
    return None


def _split_pairs(d: Path):
    files = {p.name for p in d.iterdir() if p.suffix.lower() in EXTS}
    out = []
    for name in files:
        m = SPLIT_IMG_RE.match(Path(name).stem)
        if not m:
            continue
        v, s = m.group(1), m.group(2)
        ext = Path(name).suffix
        liver = f"segmentation-{v}_livermask_{s}{ext}"
        lesion = f"segmentation-{v}_lesionmask_{s}{ext}"
        if liver in files and lesion in files:
            out.append(((int(v), int(s)), (d / name, (d / liver, d / lesion))))
    out.sort(key=lambda t: t[0])          # numeric order, not "volume-0_10" < "volume-0_2"

    n_img = sum(1 for n in files if SPLIT_IMG_RE.match(Path(n).stem))
    if len(out) != n_img:
        raise RuntimeError(
            f"incomplete dataset in {d}\n"
            f"  {n_img:,} images but only {len(out):,} have BOTH masks "
            f"({n_img - len(out):,} missing)\n"
            f"  Re-download before using this - a partial set silently changes "
            f"the split and makes every score incomparable:\n"
            f"      kaggle datasets download -d andrewmvd/lits-png -p data --unzip --force\n"
            f"  and delete data/foreground_index.json, which caches the old count."
        )
    return [p for _, p in out]


def _layout_help(root: Path) -> str:
    L = [f"no usable dataset under {root}", ""]
    if not root.exists():
        L.append("that folder does not exist.")
        return "\n".join(L)

    L.append("what is actually there:")
    n_shown, n_img = 0, 0
    for cur, dirs, files in os.walk(root):
        depth = Path(cur).relative_to(root).parts
        if len(depth) > 3:
            dirs[:] = []
            continue
        imgs = [f for f in files if Path(f).suffix.lower() in EXTS]
        n_img += len(imgs)
        if n_shown < 12:
            pad = "  " * (len(depth) + 1)
            name = Path(cur).name if depth else "."
            L.append(f"{pad}{name}/   {len(dirs)} folders, {len(imgs):,} images"
                     + (f"   e.g. {imgs[0]}" if imgs else ""))
            n_shown += 1
    if n_img == 0:
        L += ["", "no image files at all - the folder is empty or the download/move "
                  "did not finish."]

    L += ["", "supported layouts:",
          "  A)  <root>/images/*.png  +  <root>/masks/*.png       (one mask, pixel value = class id)",
          "  B)  one flat folder containing, per slice:",
          "        volume-0_30.png",
          "        segmentation-0_livermask_30.png",
          "        segmentation-0_lesionmask_30.png",
          "", "run  python -m src.utils.download  for download instructions."]
    return "\n".join(L)




def find_pairs(root: Path):
    root = Path(root)

    d = _split_layout_dir(root)
    if d is not None:
        pairs = _split_pairs(d)
        if pairs:
            return pairs, d, d

    img_dir = mask_dir = None

    for base in [root] + [p for p in sorted(root.iterdir()) if p.is_dir()]:
        if not base.is_dir():
            continue
        subs = {p.name.lower(): p for p in base.iterdir() if p.is_dir()}
        i = next((subs[n] for n in IMAGE_DIR_NAMES if n in subs), None)
        m = next((subs[n] for n in MASK_DIR_NAMES if n in subs), None)
        if i and m:
            img_dir, mask_dir = i, m
            break

    if img_dir is None:
        raise FileNotFoundError(_layout_help(root))

    img_dir, mask_dir = _descend(img_dir), _descend(mask_dir)

    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in EXTS)
    masks = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in EXTS}

    pairs = []
    for ip in imgs:
        mp = masks.get(ip.stem) or masks.get(ip.stem + "_mask") \
            or masks.get(ip.stem.replace("image", "mask"))
        if mp is not None:
            pairs.append((ip, mp))

    if not pairs:
        raise FileNotFoundError(f"found {len(imgs)} images but no matching masks in {mask_dir}")
    return pairs, img_dir, mask_dir

# translate the pixel values of the mask image into class labels
# so the nn can understand them
# input is a mask image, cleans up any compressed pixels and outputs 0,1,2 integer map

def remap_mask(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.int16)
    n_classes = len(MASK_VALUES)
    if a.max() <= 5:
        # map 3 to 2 and return
        return np.clip(a, 0, n_classes - 1).astype(np.int64)
    out = np.zeros_like(a, dtype=np.int64)
    for cls, v in enumerate(MASK_VALUES):
        out[np.abs(a - v) <= 20] = cls
    return out


def _open_mask(p, size=None):
    im = Image.open(p).convert("L")
    if size and im.size != (size, size):
        # NEAREST only - interpolating a label map would invent class ids
        im = im.resize((size, size), Image.NEAREST)
    return np.asarray(im)


def load_mask(mp, size=None) -> np.ndarray:
    if isinstance(mp, (tuple, list)):
        liver = _open_mask(mp[0], size)
        lesion = _open_mask(mp[1], size)
        y = (liver > 0).astype(np.int64)
        y[lesion > 0] = 2
        return y
    return remap_mask(_open_mask(mp, size))


def mask_raw_values(mp):
    paths = mp if isinstance(mp, (tuple, list)) else (mp,)
    vals = set()
    for p in paths:
        vals.update(np.unique(_open_mask(p)).tolist())
    return vals


def has_foreground(mp) -> bool:
    paths = mp if isinstance(mp, (tuple, list)) else (mp,)
    return any(_open_mask(p).max() > 0 for p in paths)

# bridge between raw files and nn

class LiTSDataset(Dataset):
    def __init__(self, pairs, image_size=256, augment=False, liver_only=False, mean=0.5, std=0.5):
        self.pairs = pairs           # list of (image_path, mask path) tuples found by find_pairs
        self.image_size = image_size # 256x256
        self.augment = augment
        self.liver_only = liver_only
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.pairs)

    # when pytorch asks for an image index (ex image 0)
    def _load(self, i):
        ip, mp = self.pairs[i]
        # convert the image to grayscale (the export stores it as RGB)
        img = Image.open(ip).convert("L")

        # resize if needed
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        # convert img to tensors and shrink from 0 255 to 0 1
        x = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)
        # load_mask handles both layouts and resizes with NEAREST internally
        y = torch.from_numpy(load_mask(mp, self.image_size))

        # if we want liver only, convert all 1 labels to 1 and all 2 labels to 1
        # see only background vs target
        if self.liver_only:
            y = (y > 0).long()
        return x, y

    def _augment(self, x, y):
        # 50% chance to flip the image horizontally
        # 50% chance to flip the image vertically
        # rnd chance to rotate the image by 90 180 or 270 degrees
        if random.random() < 0.5:
            x, y = torch.flip(x, [-1]), torch.flip(y, [-1])
        if random.random() < 0.5:
            x, y = torch.flip(x, [-2]), torch.flip(y, [-2])
        k = random.randint(0, 3)
        if k:
            x, y = torch.rot90(x, k, [-2, -1]), torch.rot90(y, k, [-2, -1])
        return x, y

    # this is called to get a piece of data

    def __getitem__(self, i):
        x, y = self._load(i) # load and resize
        if self.augment:
            x, y = self._augment(x, y) # random flip / rotate
        x = (x - self.mean) / self.std # normalize
        return x, y     # give to nn


def volume_id(path: Path) -> str:
    # regex used to extract a volume iD from a slice filename
    m = re.match(r"(volume[-_]?\d+)", path.stem, flags=re.I)
    return m.group(1).lower() if m else path.stem


def slice_keys(dataset):
    """the patient of every item of a dataset, in order: "volume-16" for both
    formats (a nifti item is ("volume-16", z), a png item a pair of paths)"""
    return [p[0] if isinstance(p[0], str) else volume_id(p[0]) for p in dataset.pairs]


# scan the dataset to find which slices actually contain liver or tumors and cache them in a file
# empty background slices are suppressed

def foreground_index(pairs, cache_path: Path, verbose=True):
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached.get("n") == len(pairs):
            return set(cached["foreground"])
        if verbose:
            print("[data] cached index is stale, rescanning")

    if verbose:
        print(f"[data] scanning {len(pairs):,} masks for foreground (one time, cached)...")
    fg = []
    for i, (_, mp) in enumerate(pairs):
        if has_foreground(mp):
            # if every mask pixel is 0 the slice is 100% background
            fg.append(i)
        if verbose and (i + 1) % 10000 == 0:
            print(f"         {i+1:,}/{len(pairs):,}  foreground so far: {len(fg):,}")
    cache_path.write_text(json.dumps({"n": len(pairs), "foreground": fg}))
    if verbose:
        print(f"[data] {len(fg):,}/{len(pairs):,} slices contain foreground "
              f"({100*len(fg)/max(len(pairs),1):.1f}%), cached to {cache_path.name}")
    return set(fg)



# divide dataset into train, validation, test

def split_pairs(pairs, val_fraction, test_fraction, seed, by_volume=True, key=None):

    # pairs - list of (image, mask) tuples
    # val_fraction 0.15
    # test_fraction 0.15
    # by volume is the correct way so that no leakage occur
    # key - how to get the volume id of one entry. default reads it from the
    # png filename; the nifti loader passes its own, with the same "volume-16"
    # ids, so both formats put the same patients in test


    rng = random.Random(seed)


    # not ideal
    if not by_volume:
        idx = list(range(len(pairs)))
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_fraction))
        n_val = int(round(len(idx) * val_fraction))
        groups = [idx[n_test + n_val:], idx[n_test:n_test + n_val], idx[:n_test]]
        return [[pairs[i] for i in g] for g in groups]



    # create a dict grouping all slice indices with their respective volume ID
    key = key or (lambda p: volume_id(p[0]))
    by_vol = {}
    for i, p in enumerate(pairs):
        by_vol.setdefault(key(p), []).append(i)
    # vols holds [{volume0 : volume0_slices}, {volume1 : volume1_slices, ....]
    vols = sorted(by_vol)
    # shuffle randomly
    rng.shuffle(vols)

    # give test 0.15 eg volume12, volume45, ....
    n_test = max(1, int(round(len(vols) * test_fraction)))
    # give validation 0.15
    n_val = max(1, int(round(len(vols) * val_fraction)))
    # train gets rest of 0.7
    # sets holds [train_pairs, val_pairs, test_pairs]
    sets = [vols[n_test + n_val:], vols[n_test:n_test + n_val], vols[:n_test]]
    return [[pairs[i] for v in s for i in by_vol[v]] for s in sets], vols, sets


# calculates distribution of classes across the dataset by counting pixels
# tells you what % is background, what % is liver, what % is tumor
# count on 200 slices, gives around 99% accurate statistical representation

def class_pixel_counts(dataset, max_samples=200):
    counts = Counter()
    n = min(len(dataset), max_samples)
    step = max(1, len(dataset) // n)
    for i in range(0, len(dataset), step):
        _, y = dataset[i]
        vals, cts = torch.unique(y, return_counts=True)
        for v, c in zip(vals.tolist(), cts.tolist()):
            counts[v] += c
    return counts




# create DataLoader objects
def get_dataloaders(cfg: Config, verbose: bool = True):

    if cfg.data_format == "nifti":
        # one entry per slice, ("volume-16", 402), read from the npy cache that
        # src/nifti.py builds out of the .nii.gz files. skip_empty is applied
        # in there, from the cache index
        from src.nifti import nifti_slices, NiftiSliceDataset
        pairs, n_all, index, cache_dir = nifti_slices(cfg, verbose)
        key = lambda p: p[0]
        sources = [f"[data] nifti from {cfg.root_nifti_dir}",
                   f"[data] slices from {cache_dir}  HU window {list(cfg.hu_window)}"
                   f"  input {cfg.in_slices} slice(s)" + (" (2.5D)" if cfg.in_slices > 1 else "")]
        mk = lambda p, aug: NiftiSliceDataset(p, cache_dir, index, aug, cfg.liver_only,
                                              cfg.hu_window, cfg.in_slices)
    else:
        if cfg.in_slices != 1:
            raise ValueError("2.5D input (in_slices > 1) needs the nifti data; "
                             "the png export has no neighbouring slices to read")
        # data directory
        data_dir = Path(cfg.root_data_dir)
        # image mask tuple, image directory, mask directory
        pairs, img_dir, mask_dir = find_pairs(data_dir)
        # size of pairs list
        n_all = len(pairs)

        # skip empty true from config, filter empty slices
        if cfg.skip_empty:
            fg = foreground_index(pairs, data_dir / "foreground_index.json", verbose)
            pairs = [p for i, p in enumerate(pairs) if i in fg]
        key = None
        sources = [f"[data] images from {img_dir}", f"[data] masks  from {mask_dir}"]
        # wrap each set of pairs into a LiTSDataset instance
        mk = lambda p, aug: LiTSDataset(p, cfg.image_size, aug, cfg.liver_only)

    # if max samples = 500 this will select random 500 pairs so we can run a quick test
    if cfg.max_samples and len(pairs) > cfg.max_samples:
        rng = random.Random(cfg.seed)
        pairs = sorted(rng.sample(pairs, cfg.max_samples))

    # train, validation, test
    split = split_pairs(pairs, cfg.val_fraction, cfg.test_fraction, cfg.seed, cfg.split_by_volume, key)

    # split by volume
    if cfg.split_by_volume:
        (tr, va, te), vols, vol_sets = split
    else:
        tr, va, te = split
        vols, vol_sets = None, None

    # augmentation on only for train set , validation and test are kept original
    train_set, val_set, test_set = mk(tr, cfg.augment), mk(va, False), mk(te, False)

    common = dict(num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=cfg.num_workers > 0)

    # dataloader objects
    # only training loader is shuffled across every epoch
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, **common)

    # pixel counts
    counts = class_pixel_counts(train_set)
    names = cfg.class_names
    total = sum(counts.values()) or 1
    pixel_counts = [counts.get(i, 0) for i in range(cfg.num_classes)]

    # prints
    if verbose:
        for line in sources:
            print(line)
        kept = f"{len(pairs):,}/{n_all:,}" if len(pairs) != n_all else f"{n_all:,}"
        print(f"[data] {kept} pairs used -> train={len(tr):,} val={len(va):,} test={len(te):,}")
        if vols is not None:
            print(f"[data] split by VOLUME: {len(vols)} volumes -> "
                  f"train={len(vol_sets[0])} val={len(vol_sets[1])} test={len(vol_sets[2])}"
                  f"  (no patient appears in two splits)")
        else:
            print("[data] split by slice - adjacent slices leak between splits, "
                  "scores will be optimistic")
        print(f"[data] size={cfg.image_size}  augmentation={'on' if cfg.augment else 'off'}"
              f"  classes={names}")
        print("[data] pixel share per class (sampled from train):")
        for i, nm in enumerate(names):
            pct = 100 * counts.get(i, 0) / total
            print(f"         {nm:<12}{pct:7.3f}%")
        rare = min((100 * counts.get(i, 0) / total) for i in range(cfg.num_classes))
        if rare < 1:
            print(f"[data] note: rarest class is {rare:.3f}% of pixels - plain pixel "
                  f"accuracy will look great and mean nothing. watch per-class Dice.")


    # return the DataLoader objects
    return train_loader, val_loader, test_loader, pixel_counts

# denormalize by using inverse operations * and +
def denormalize(x, mean=0.5, std=0.5):
    return (x.detach().cpu() * std + mean).clamp(0, 1)
