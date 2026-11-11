from dataclasses import dataclass, asdict
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"
ARCH_DIR = ROOT / "architecture"

for _d in (DATA_DIR, RUNS_DIR, RESULTS_DIR, ARCH_DIR):
    _d.mkdir(exist_ok=True)

# data/, runs/ and results/ each hold one folder per data format: the current
# nifti work, and the finished internship work on the png export
FORMAT_DIRS = {"nifti": "nifti", "png": "internship_png"}


def results_dir(data_format, sub=None) -> Path:
    """results/<format>/, or a topic folder inside it: summary, comparisons,
    kfold, gradcam, dataset, viewer, cases, baseline (see results/nifti/README.md)"""
    d = RESULTS_DIR / FORMAT_DIRS[data_format]
    if sub:
        d = d / sub
    d.mkdir(parents=True, exist_ok=True)
    return d

def readable_run_name(c: dict) -> str:
    """a run folder's name, built from what sets the run apart:
    unet_2d, unet_r50_2.5d_lr1e-4_fold0, unetpp_r50_2.5d_lr1e-4_batch4x2 ...
    Only settings that differ from the defaults are spelled out. `c` is a
    Config as a dict, or the "config" saved in a run's history.json."""
    nifti = c.get("data_format", "png") == "nifti"
    n = c.get("in_slices", 1)
    parts = [c["arch"], {1: "2d", 3: "2.5d"}.get(n, f"{n}slices")]
    if c["arch"].endswith("_r50") and c.get("encoder_weights", "imagenet") == "none":
        parts.append("random-init")
    if not c["arch"].endswith("_r50"):
        if c.get("base_filters", 32) != 32:
            parts.append(f"f{c['base_filters']}")
        if c.get("depth", 4) != 4:
            parts.append(f"depth{c['depth']}")
    if c.get("image_size", 512 if nifti else 256) != (512 if nifti else 256):
        parts.append(f"{c['image_size']}px")
    if c.get("liver_only"):
        parts.append("liver-only")
    if c.get("lr", 1e-3) != 1e-3:
        parts.append("lr" + f"{c['lr']:.0e}".replace("e-0", "e-"))
    b, k = c.get("batch_size", 8), c.get("accum_steps", 1)
    if (b, k) != (8, 1):
        parts.append(f"batch{b}x{k}" if k > 1 else f"batch{b}")
    if c.get("loss", "dice_ce") != "dice_ce":
        parts.append(f"loss-{c['loss']}")
    if c.get("class_weights", "none") != "none":
        parts.append(f"weights-{c['class_weights']}")
    if not c.get("augment", True):
        parts.append("no-aug")
    if c.get("epochs", 40) != 40:
        parts.append(f"{c['epochs']}ep")
    # the first NIfTI runs picked their best epoch by foreground Dice
    if nifti and c.get("select_metric", "dice_fg") != "tumor_case":
        parts.append("old-epoch-rule")
    if c.get("seed", 42) != 42:
        parts.append(f"seed{c['seed']}")
    if c.get("folds"):
        parts.append(f"fold{c['fold']}")
    return "_".join(parts)


CLASS_NAMES = ["background", "liver", "tumor"]
MASK_VALUES = [0, 127, 255]
IN_CHANNELS = 1


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# training resolution by name. The nifti scans are 512x512, so "full" is native
RESOLUTIONS = {"full": 512, "half": 256, "quarter": 128, "eighth": 64}

ARCH_NAME = "attention_unet"
# trained from scratch, the internship's two networks
SCRATCH_ARCHS = ["attention_unet", "unet"]
# the same two, and U-Net++, with an ImageNet-pretrained ResNet-50 as the encoder
ARCH_NAMES = SCRATCH_ARCHS + ["attention_unet_r50", "unet_r50", "unetpp_r50"]

@dataclass
class Config:

    data_format: str = "nifti"        # "nifti" | "png"
                                      # nifti = MSD Task03_Liver, CT in Hounsfield units
                                      # png   = the internship's lits-png export
    nifti_dir: str = ""               # holds imagesTr/ and labelsTr/,
                                      # default data/Task03_Liver
    hu_window: tuple = (-17, 201)     # nifti: HU clipped to this and mapped to
                                      # 0..1. the 0.5-99.5 percentile of liver
                                      # and tumour voxels, measured by
                                      # nifti_report.py, as nnU-Net picks it
    data_dir: str = ""                # png only, default data/internship_png
    image_size: int = 512             # slice size the model sees. 512 = the
                                      # native nifti resolution; --resolution
                                      # half (256) or quarter (128) on a small GPU
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    augment: bool = True
    batch_size: int = 8
    accum_steps: int = 1              # optimiser step every N batches, so the
                                      # effective batch is batch_size x N
    num_workers: int = 0

    liver_only: bool = False          # merge tumor into liver -> 2 classes
                                      # much easier, useful as a first milestone
    skip_empty: bool = True           # drop slices whose mask is pure background
                                      # ~69% of LiTS slices contain no liver at all
    max_samples: int = 0              # 0 = use everything, used for quick runs
    split_by_volume: bool = True      # split whole patients, not individual
                                      # slices, slice split leaks between train and test
    folds: int = 0                    # 0 = the fixed split above (15% test)
                                      # k = k-fold by patient: fold i tests 1/k
                                      # of the patients, so k runs test every
                                      # patient once. val stays val_fraction
    fold: int = 0                     # which fold this run tests, 0..folds-1

    arch: str = ARCH_NAME             # "attention_unet" | "unet" | "attention_unet_r50" | "unet_r50" | "unetpp_r50"

    in_slices: int = 1                # slices given to the model as channels
                                      # 1 = 2D, the slice alone
                                      # 3 = 2.5D, the slice and the one above
                                      #     and below: some 3D context, and the 3
                                      #     channels an ImageNet encoder expects
    encoder_weights: str = "imagenet" # *_r50 only: "imagenet" starts the ResNet-50
                                      # from its ImageNet weights, "none" from random

    base_filters: int = 32            # filters at full resolution; doubles per
                                      # level
    depth: int = 4                    # downsampling steps 4 -> 512 shrinks to
                                      # 32x32 at the bottleneck (256 -> 16x16)

    batch_norm: bool = True

    dropout: float = 0.0              # in the bottleneck only

    bilinear: bool = False            # upsample by interpolation instead of
                                      # transposed conv

    loss: str = "dice_ce"                    # "dice_ce" | "dice" | "ce"
    # dice ce -> calculate  both dice and cross entropy and merge them
    # together to get best of both

    dice_weight: float = 0.5                 # blend when loss="dice_ce"
    # 0.5 dice weights means 50% dice loss + 50% cross entropy loss

    # how we punish the network for missing rare classes like tumor
    class_weights: str = "none"              # "none" | "inverse" | "manual"
    # if class weights = manual it uses these numbers
    # penalty 1 if misclassify background, 2 if liver, 4 if tumor
    manual_weights: tuple = (1.0, 2.0, 4.0)  # background, liver, tumor

    epochs: int = 40
    optimizer: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    scheduler: str = "cosine"


    select_metric: str = "tumor_case"       # which validation score picks best_model.pt
    # "tumor_case" tumour Dice per patient (LiTS rules), the main goal
    # "dice_fg"    mean of liver and tumour Dice over all pixels (the
    #              internship runs and the first NIfTI baseline used this)

    early_stop_patience: int = 0
    amp: bool = True


    seed: int = 42
    run_name: str = "run"
    group: str = "baseline"


    @property
    def out_dir(self) -> Path:
        # runs/nifti/<group>/<run> or runs/internship_png/<group>/<run>
        return RUNS_DIR / FORMAT_DIRS[self.data_format] / self.group / self.run_name

    @property
    def root_data_dir(self) -> Path:
        return Path(self.data_dir) if self.data_dir else DATA_DIR / "internship_png"

    @property
    def root_nifti_dir(self) -> Path:
        return Path(self.nifti_dir) if self.nifti_dir else DATA_DIR / "Task03_Liver"

    @property
    def nifti_cache_dir(self) -> Path:
        # one cache per training resolution, built by src/nifti.py
        return DATA_DIR / f"nifti_cache_{self.image_size}"

    @property
    def num_classes(self) -> int:
        return 2 if self.liver_only else len(CLASS_NAMES)

    @property
    def class_names(self):
        return ["background", "liver"] if self.liver_only else CLASS_NAMES

    def to_dict(self):
        return asdict(self)
