"""One look for every results figure: colours, fonts and readable run names.

Colour always means the same thing in every figure:
    blue    ResNet-50 encoder pretrained on ImageNet
    orange  trained from scratch
filled marker = test, hollow = validation. The architecture and the input
(2D = one slice, 2.5D = three) are in the label, not in the colour.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# categorical slots of a colour-blind-checked palette, used in this order
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
CONTEXT = "#d9d8d4"                                      # other runs, drawn behind
SIZE_STEPS = ["#bdbcb6", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # none -> largest tumour

ARCH = {  # arch -> (name, pretrained ResNet-50 encoder)
    "unet": ("U-Net", False), "attention_unet": ("Attention U-Net", False),
    "unet_r50": ("U-Net", True), "attention_unet_r50": ("Attention U-Net", True),
    "unetpp_r50": ("U-Net++", True),
}


def pretrained(cfg):
    return ARCH.get(cfg.get("arch"), ("", False))[1]


def colour(cfg):
    return BLUE if pretrained(cfg) else ORANGE


def run_label(cfg, select_metric="tumor_case"):
    """'U-Net R50 2.5D, lr 1e-4' - only what differs from the defaults
    (lr 1e-3, batch 8) is spelled out"""
    name, r50 = ARCH.get(cfg.get("arch"), (cfg.get("arch", "?"), False))
    s = f"{name}{' R50' if r50 else ''} {'2.5D' if cfg.get('in_slices', 1) > 1 else '2D'}"
    extra = []
    if cfg.get("lr", 1e-3) != 1e-3:
        extra.append("lr " + f"{cfg['lr']:.0e}".replace("e-0", "e-"))      # 1e-4, not 0.0001
    b, k = cfg.get("batch_size", 8), cfg.get("accum_steps", 1)
    if (b, k) != (8, 1):
        extra.append(f"batch {b}x{k}" if k > 1 else f"batch {b}")
    if not select_metric:
        extra.append("old epoch rule")
    return s + (", " + ", ".join(extra) if extra else "")


def style():
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
        "axes.titlecolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "legend.frameon": False, "legend.fontsize": 9,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "lines.linewidth": 2, "lines.markersize": 8,
    })


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, path, dpi=150):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")
