from dataclasses import dataclass, asdict
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
ARCH_DIR = ROOT / "architecture"

for _d in (DATA_DIR, RUNS_DIR, ARCH_DIR):
    _d.mkdir(exist_ok=True)

CLASS_NAMES = ["background", "liver", "tumor"]
MASK_VALUES = [0, 127, 255]
IN_CHANNELS = 1


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


ARCH_NAME = "attention_unet"
ARCH_NAMES = ["attention_unet", "unet"]

@dataclass
class Config:

    data_dir: str = ""
    image_size: int = 256
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    augment: bool = True
    batch_size: int = 8
    num_workers: int = 0

    liver_only: bool = False          # merge tumor into liver -> 2 classes
                                      # much easier, useful as a first milestone
    skip_empty: bool = True           # drop slices whose mask is pure background
                                      # ~69% of LiTS slices contain no liver at all
    max_samples: int = 0              # 0 = use everything, used for quick runs
    split_by_volume: bool = True      # split whole patients, not individual
                                      # slices, slice split leaks between train and test

    arch: str = ARCH_NAME             # "attention_unet" | "unet"

    base_filters: int = 32            # filters at full resolution; doubles per
                                      # level
    depth: int = 4                    # downsampling steps 4 -> 256 shrinks to
                                      # 16x16 at the bottleneck

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


    early_stop_patience: int = 0
    amp: bool = True


    seed: int = 42
    run_name: str = "run"
    group: str = "baseline"


    @property
    def out_dir(self) -> Path:
        return RUNS_DIR / self.group / self.run_name if self.group \
            else RUNS_DIR / self.run_name

    @property
    def root_data_dir(self) -> Path:
        return Path(self.data_dir) if self.data_dir else DATA_DIR

    @property
    def num_classes(self) -> int:
        return 2 if self.liver_only else len(CLASS_NAMES)

    @property
    def class_names(self):
        return ["background", "liver"] if self.liver_only else CLASS_NAMES

    def to_dict(self):
        return asdict(self)
