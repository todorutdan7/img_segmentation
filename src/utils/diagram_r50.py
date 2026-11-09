"""Architecture diagrams of the models with a ResNet-50 encoder, at the NIfTI size:
U-Net R50, Attention U-Net R50 and U-Net++ R50.

    python -m src.utils.diagram_r50                     # all three -> architecture/nifti/
    python -m src.utils.diagram_r50 --arch unetpp_r50

The parameter counts are read from the real model (built without downloading
the ImageNet weights), so the figure cannot drift from src/model.py. The
from-scratch U-Nets are drawn by src.utils.diagram, in the same style.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.diagram import (ENC, BOT, GATE, DEC, HEAD, GREY, SKIP, LINE,
                               box, arrow, op_node, panel_title, blank)

MID = "#48A9A6"          # U-Net++ in-between nodes
NOTE = "#566573"
LAYER_BLOCKS = (3, 4, 6, 3)                 # bottleneck blocks in ResNet-50 layer1..4


def measure(arch, in_ch=3, size=512, num_classes=3):
    """parameter counts and shapes of every part, from the model itself"""
    from src.model import ARCHES, ResNet50Encoder
    m = ARCHES[arch](in_channels=in_ch, num_classes=num_classes, pretrained=False)
    n = lambda mod: sum(p.numel() for p in mod.parameters() if p.requires_grad) if mod is not None else 0
    P = dict(arch=arch, in_ch=in_ch, size=size, total=m.count_parameters(),
             enc_ch=ResNet50Encoder.channels, dec_ch=m.decoder_channels,
             enc=[n(m.encoder.stem)] + [n(l) for l in m.encoder.layers],
             enc_total=n(m.encoder), head=n(m.head), num_classes=num_classes)
    P["hw"] = [size // 2 ** (i + 1) for i in range(5)]          # 256, 128, 64, 32, 16 at 512
    if arch == "unetpp_r50":
        P["nodes"] = {k: n(v) for k, v in m.nodes.items()}
        P["last"] = n(m.last)
    else:
        P["blocks"] = [n(b) for b in m.blocks]
        P["gates"] = [n(b.gate) for b in m.blocks]
    return P


def fmt(p):
    return f"{p / 1e6:.1f} M" if p >= 1e6 else f"{p:,}"


def ladder_y(i):
    # rows of the full data path: 512 on top, then 1/2 ... 1/32
    return {-1: 93, 0: 80, 1: 66, 2: 52, 3: 38, 4: 20}[i]


def res_labels(ax, P, x=3):
    ax.text(x, 109, "resolution", fontsize=8.5, color="#7F8C8D", style="italic")
    for i, y in [(-1, 98.5)] + [(i, ladder_y(i)) for i in range(5)]:
        s = P["size"] if i < 0 else P["hw"][i]
        ax.text(x, y, f"{s}²", fontsize=10.5, fontweight="bold", color="#1B2631", va="center")


# --------------------------------------------------------------------------
# U-Net R50 and Attention U-Net R50
# --------------------------------------------------------------------------

def unet_flow(ax, P):
    att = any(P["gates"])
    E, D, S = P["enc_ch"], P["dec_ch"], P["size"]
    hw = P["hw"]
    blank(ax, ylim=(4, 119))
    name = "Attention U-Net" if att else "U-Net"
    panel_title(ax, f"{name} with a ResNet-50 encoder - full data path",
                f"input {P['in_ch']} x {S} x {S} (the slice and its two neighbours, 2.5D)   •   "
                f"{P['total']:,} trainable parameters   •   encoder pretrained on ImageNet, "
                f"decoder{' and gates' if att else ''} new")
    res_labels(ax, P)
    xe, xd, w, h = 25, 79, 27, 8

    # encoder column
    box(ax, xe, 104, w, 6.5, "INPUT   CT, 3 slices", f"{P['in_ch']} x {S} x {S}   HU window, -1..1",
        color="#34495E")
    box(ax, xe, 93, w, 6.5, "normalise", "to ImageNet mean / std, inside the model",
        color="#5D6D7E", fs=9)
    arrow(ax, (xe, 100.5), (xe, 96.5), color="#34495E")
    titles = ["stem   conv 7x7 s2 + BN + ReLU"] + \
             [f"layer{i + 1}   {b} bottleneck blocks" for i, b in enumerate(LAYER_BLOCKS[:3])]
    for i in range(4):
        box(ax, xe, ladder_y(i), w, h, titles[i], f"{E[i]} x {hw[i]} x {hw[i]}     {fmt(P['enc'][i])} par",
            color=ENC)
    arrow(ax, (xe, 89.5), (xe, 84.2), color=ENC)
    downs = ["max-pool 3x3 s2  ↓2", "↓2 (stride 2)", "↓2 (stride 2)"]
    for i in range(3):
        arrow(ax, (xe, ladder_y(i) - 4.2), (xe, ladder_y(i + 1) + 4.2), color=ENC,
              label=downs[i], lpos=0.5, loff=(0, 0), fs=8)
    # bottleneck
    xb = 52
    box(ax, xb, ladder_y(4), 32, h, f"BOTTLENECK   layer4   {LAYER_BLOCKS[3]} bottleneck blocks",
        f"{E[4]} x {hw[4]} x {hw[4]}     {fmt(P['enc'][4])} par", color=BOT)
    arrow(ax, (xe, ladder_y(3) - 4.2), (xb - 9, ladder_y(4) + 4.2), color=ENC,
          label="↓2 (stride 2)", lpos=0.45, loff=(-4, 0), fs=8)

    # decoder column: blocks.0 at 1/16 ... blocks.3 at 1/2, blocks.4 at full size
    for k in range(5):
        lvl = 3 - k                       # level it ends on; -1 = full size
        y = ladder_y(lvl)
        size = S if lvl < 0 else hw[lvl]
        skip = f"+ skip {E[lvl]}" if lvl >= 0 else "no skip"
        box(ax, xd, y, w, h, f"blocks.{k}   ↑2, {skip}, DoubleConv",
            f"{D[k]} x {size} x {size}     {fmt(P['blocks'][k] - P['gates'][k])} par", color=DEC)
    box(ax, xd, 104, w, 6.5, "head   conv 1x1", f"{P['num_classes']} x {S} x {S}   background / liver / tumour",
        color=HEAD)
    arrow(ax, (xd, 96.5), (xd, 100.5), color=HEAD)
    for k in range(4):
        arrow(ax, (xd, ladder_y(3 - k) + 4.2), (xd, ladder_y(2 - k) - 4.2), color=DEC)
    arrow(ax, (xb + 16, ladder_y(4)), (xd, ladder_y(3) - 4.2), color=DEC, rad=0.25)

    # skips, through a gate in the attention variant
    for lvl in range(4):
        y, k = ladder_y(lvl), 3 - lvl
        if att:
            xg = 52
            box(ax, xg, y, 15, 5.6, f"gate  blocks.{k}", f"α: 1 x {hw[lvl]}²   {fmt(P['gates'][k])} par",
                color=GATE, fs=8.6)
            arrow(ax, (xe + w / 2, y), (xg - 7.5, y), color=SKIP, label=f"skip {E[lvl]} ch", lpos=0.5,
                  loff=(0, 2.2), fs=7.8)
            arrow(ax, (xg + 7.5, y), (xd - w / 2, y), color=SKIP, label="x ⊙ α", lpos=0.5,
                  loff=(0, 2.2), fs=7.8)
            # the gating signal is the block's input from one level below
            src = (xb + 6, ladder_y(4) + 4.2) if lvl == 3 else (xd - w / 2 + 2.5, ladder_y(lvl + 1) + 4.1)
            arrow(ax, src, (xg + 4, y - 2.9), color="#B9770E", ls="--", lw=1.3)
        else:
            arrow(ax, (xe + w / 2, y), (xd - w / 2, y), color=SKIP, label=f"skip   {E[lvl]} ch",
                  lpos=0.5, loff=(0, 2.2), fs=8)
    note = ("dashed: the gating signal, the block's input from the level below, before it is upsampled. "
            "The gate scales each skip pixel by α in (0, 1) before the concatenation."
            if att else "Each skip joins the decoder unfiltered: the encoder's features at that "
                        "resolution are concatenated with the upsampled ones.")
    ax.text(52, 9.5, note, ha="center", fontsize=8.6, color=NOTE, style="italic")


def unet_inside(ax, P):
    blank(ax, ylim=(0, 100))
    att = any(P["gates"])
    E, D = P["enc_ch"], P["dec_ch"]
    panel_title(ax, "Inside the blocks",
                "left: one ResNet-50 bottleneck block (shown for layer3)   •   right: one decoder "
                f"step (shown for blocks.0: {E[4]} ch from below + {E[3]} ch skip)")
    # bottleneck block, layer3: 1024 -> 256 -> 256 -> 1024
    y, xs = 52, [6, 17, 28, 39]
    labels = [("1x1 conv", f"{E[3]}→{E[3] // 4}"), ("3x3 conv", f"{E[3] // 4}→{E[3] // 4}"),
              ("1x1 conv", f"{E[3] // 4}→{E[3]}")]
    box(ax, xs[0], y, 7, 12, "x", f"{E[3]} ch", color=ENC, fs=9)
    for x, (t, s) in zip(xs[1:], labels):
        box(ax, x, y, 9, 12, t, s + " +BN", color=ENC, fs=8.8)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + 4.6, y), (b - 4.6, y), color=ENC, lw=1.3)
    op_node(ax, 46, y, "+", color=ENC, r=1.7)
    arrow(ax, (xs[3] + 4.6, y), (44.3, y), color=ENC, lw=1.3)
    ax.plot([xs[0], xs[0], 46], [y - 6.3, y - 13, y - 13], color=ENC, lw=1.2)
    arrow(ax, (46, y - 13), (46, y - 2.2), color=ENC, lw=1.2)
    ax.text(26, y - 17, "shortcut (identity)", ha="center", fontsize=7.8, color=ENC)
    ax.text(26, 20, "ReLU after each conv and after the sum. The 1x1 convs squeeze the channels\n"
                    "4x and expand them back, so the 3x3 conv runs on a quarter of the width.",
            ha="center", fontsize=8.4, color=NOTE, style="italic")

    # decoder step
    x0 = 50
    box(ax, x0 + 3, 70, 10, 11, "from below", f"{E[4]} ch @ {P['hw'][4]}²", color=BOT, fs=8.6)
    box(ax, x0 + 3, 34, 10, 11, "skip", f"{E[3]} ch @ {P['hw'][3]}²", color=ENC, fs=8.6)
    xu = x0 + 17
    box(ax, xu, 70, 10, 11, "nearest ↑2", f"{E[4]} @ {P['hw'][3]}²", color=DEC, fs=8.6)
    arrow(ax, (x0 + 8.3, 70), (xu - 5.3, 70), color=GREY, lw=1.3)
    if att:
        box(ax, xu, 34, 10, 11, "gate", "skip x α", color=GATE, fs=8.6)
        arrow(ax, (x0 + 8.3, 34), (xu - 5.3, 34), color=GREY, lw=1.3)
        arrow(ax, (x0 + 3, 64.2), (xu - 2, 39.8), color="#B9770E", ls="--", lw=1.2,
              label="gating signal", lpos=0.45, loff=(-2.5, 0), fs=7.6)
        sk = (xu + 5.3, 34)
    else:
        sk = (x0 + 8.3, 34)
    op_node(ax, xu + 10, 52, "||", color=GREY, r=1.8, fs=10)
    arrow(ax, (xu + 5.3, 70), (xu + 9.3, 53.6), color=GREY, lw=1.3)
    arrow(ax, sk, (xu + 9.3, 50.4), color=GREY, lw=1.3)
    ax.text(xu + 12.5, 45, f"concat {E[4] + E[3]} ch", ha="center", fontsize=7.8, color=GREY)
    box(ax, xu + 23, 52, 15, 12, "DoubleConv", "2 x (3x3 conv, BN, ReLU)", color=DEC, fs=8.6)
    arrow(ax, (xu + 11.8, 52), (xu + 15.3, 52), color=GREY, lw=1.3)
    ax.text(xu + 23, 38, f"→ {D[0]} ch @ {P['hw'][3]}²", ha="center", fontsize=8.2, color=DEC,
            fontweight="bold")
    ax.text(76, 13, "Upsampling is plain nearest-neighbour: no learned weights. All the decoder's "
                    "parameters are in the DoubleConvs" + (" and the gates." if att else "."),
            ha="center", fontsize=8.4, color=NOTE, style="italic")


def table(ax, title, sub, head, rows, total, foot, widths):
    blank(ax, ylim=(0, 100))
    panel_title(ax, title, sub)
    xs = [4]
    for w in widths[:-1]:
        xs.append(xs[-1] + w)
    top, dy = 80, min(8.5, 68 / (len(rows) + 2))
    for x, hcell, w in zip(xs, head, widths):
        right = hcell.endswith("par")
        ax.text(x + (w - 2 if right else 0), top, hcell, fontsize=9.2, fontweight="bold",
                color="#1B2631", ha="right" if right else "left", va="center")
    ax.plot([2, 98], [top - dy * 0.55] * 2, color="#1B2631", lw=1)
    for r, row in enumerate(rows + [total]):
        y = top - dy * (r + 1)
        if r % 2 == 1 and r < len(rows):
            ax.add_patch(plt.Rectangle((2, y - dy / 2), 96, dy, color="#F4F6F7", zorder=0))
        if r == len(rows):
            ax.plot([2, 98], [y + dy / 2] * 2, color="#1B2631", lw=1)
        for x, cell, w, hcell in zip(xs, row, widths, head):
            right = hcell.endswith("par")
            ax.text(x + (w - 2 if right else 0), y, cell, fontsize=9, va="center",
                    ha="right" if right else "left", color="#1B2631",
                    fontweight="bold" if r == len(rows) else "normal")
    ax.text(50, top - dy * (len(rows) + 2), foot, ha="center", fontsize=8.8, color=NOTE, va="top")


def unet_table(ax, P):
    att = any(P["gates"])
    E, D, hw, S = P["enc_ch"], P["dec_ch"], P["hw"], P["size"]
    rows = []
    for lvl in [-1, 0, 1, 2, 3, 4]:
        k = 3 - lvl
        res = f"{S} x {S}" if lvl < 0 else f"{hw[lvl]} x {hw[lvl]}"
        enc = f"input {P['in_ch']} ch" if lvl < 0 else (["stem", "layer1", "layer2", "layer3", "layer4"][lvl]
                                                        + f"   {E[lvl]} ch")
        enc_p = "-" if lvl < 0 else f"{P['enc'][lvl]:,}"
        if lvl == 4:
            dec, dec_p, gate = "-", "-", "-"
        else:
            dec = f"blocks.{k}   {D[k]} ch" + ("   + head" if lvl < 0 else "")
            dec_p = f"{P['blocks'][k] - P['gates'][k] + (P['head'] if lvl < 0 else 0):,}"
            gate = f"{P['gates'][k]:,}" if att and P["gates"][k] else "-"
        row = [("4  (bottleneck)" if lvl == 4 else ("full" if lvl < 0 else str(lvl))), res, enc,
               enc_p] + ([gate] if att else []) + [dec, dec_p]
        rows.append(row)
    g = sum(P["gates"])
    dec_total = sum(P["blocks"]) - g + P["head"]
    total = ["TOTAL", "", "", f"{P['enc_total']:,}"] + ([f"{g:,}"] if att else []) + ["", f"{dec_total:,}"]
    head = ["level", "resolution", "encoder (pretrained)", "encoder par"] + \
           (["gate par"] if att else []) + ["decoder (new)", "decoder par"]
    widths = [11, 12, 19, 13] + ([11] if att else []) + [17, 13]
    foot = (f"encoder {fmt(P['enc_total'])} ({100 * P['enc_total'] / P['total']:.0f}%), all of it fine-tuned"
            f"   •   decoder + head {fmt(dec_total)}" + (f"   •   attention gates {fmt(g)}" if att else "")
            + f"   •   whole model {P['total']:,} parameters")
    table(ax, "Resolution and channel ladder", "every level of the U, top to bottom", head, rows, total,
          foot, widths)


def draw_unet(P, out, dpi):
    fig = plt.figure(figsize=(17.5, 16.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.5, 0.85, 1.05], hspace=0.12, left=0.03, right=0.975,
                          top=0.955, bottom=0.02)
    att = any(P["gates"])
    fig.suptitle(f"{'Attention U-Net' if att else 'U-Net'} R50 - architecture "
                 f"(NIfTI, {P['size']} x {P['size']}, 2.5D)", fontsize=19, fontweight="bold",
                 y=0.985, color="#1B2631")
    unet_flow(fig.add_subplot(gs[0]), P)
    unet_inside(fig.add_subplot(gs[1]), P)
    unet_table(fig.add_subplot(gs[2]), P)
    fig.savefig(out, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"[saved] {out}")


# --------------------------------------------------------------------------
# U-Net++ R50
# --------------------------------------------------------------------------

def width(P, i, j):
    """output channels of node X(i, j), as in ResNetUNetPlusPlus (column 0 = encoder)"""
    n = 4
    if j == 0:
        return P["enc_ch"][i]
    return P["dec_ch"][n - 1 - i] if i + j == n else P["enc_ch"][i]


def unetpp_flow(ax, P):
    E, hw, S = P["enc_ch"], P["hw"], P["size"]
    blank(ax, ylim=(4, 119))
    panel_title(ax, "U-Net++ with a ResNet-50 encoder - the nested grid of nodes",
                f"input {P['in_ch']} x {S} x {S} (2.5D)   •   {P['total']:,} trainable parameters   •   "
                f"node X(i, j): level i (resolution), column j   •   column 0 is the pretrained encoder")
    res_labels(ax, P)
    xs = [17, 36, 55, 74, 92]
    w, h = 15, 8.2
    # encoder column
    box(ax, xs[0], 104, w, 6.5, "INPUT   CT, 3 slices", f"{P['in_ch']} x {S} x {S}", color="#34495E", fs=9)
    box(ax, xs[0], 93, w, 6, "normalise", "ImageNet mean / std", color="#5D6D7E", fs=8.6)
    arrow(ax, (xs[0], 100.5), (xs[0], 96.2), color="#34495E")
    arrow(ax, (xs[0], 89.8), (xs[0], ladder_y(0) + h / 2 + 0.2), color=ENC)
    names = ["stem", "layer1", "layer2", "layer3", "layer4"]
    for i in range(5):
        box(ax, xs[0], ladder_y(i), w, h, f"X {i},0   {names[i]}",
            f"{E[i]} ch   {fmt(P['enc'][i])}", color=BOT if i == 4 else ENC, fs=8.8)
        if i < 4:
            arrow(ax, (xs[0], ladder_y(i) - h / 2 - 0.2), (xs[0], ladder_y(i + 1) + h / 2 + 0.2),
                  color=ENC, label="↓2", lpos=0.5, loff=(0, 0), fs=8)
    # nodes
    for j in range(1, 5):
        for i in range(5 - j):
            last = i + j == 4
            box(ax, xs[j], ladder_y(i), w, h, f"X {i},{j}",
                f"{width(P, i, j)} ch   {fmt(P['nodes'][f'{i}_{j}'])}",
                color=DEC if last else MID, fs=8.8)
            # from below, upsampled
            arrow(ax, (xs[j - 1] + w / 2, ladder_y(i + 1) + h / 2 - 1),
                  (xs[j] - w / 2 + 1, ladder_y(i) - h / 2), color=DEC, lw=1.4)
            # from the left: every node already on this row
            arrow(ax, (xs[j - 1] + w / 2, ladder_y(i)), (xs[j] - w / 2, ladder_y(i)), color=SKIP, lw=1.5)
            for k in range(j - 1):
                arrow(ax, (xs[k] + 3, ladder_y(i) + h / 2), (xs[j] - 3, ladder_y(i) + h / 2),
                      color=SKIP, lw=0.9, ls="--", rad=-0.075 + 0.008 * (j - k), alpha=0.8)
    # last step and head
    box(ax, xs[4], 93, w, 6, "last   ↑2, DoubleConv", f"{P['dec_ch'][-1]} ch @ {S}²   {fmt(P['last'])}",
        color=DEC, fs=8.6)
    box(ax, xs[4], 104, w, 6.5, "head   conv 1x1", f"{P['num_classes']} x {S} x {S}", color=HEAD, fs=9)
    arrow(ax, (xs[4], ladder_y(0) + h / 2), (xs[4], 90), color=DEC)
    arrow(ax, (xs[4], 96), (xs[4], 100.7), color=HEAD)
    # legend
    lx, ly = 30, 12
    for k, (c, t, ls) in enumerate([(DEC, "from the level below, upsampled ↑2", "-"),
                                    (SKIP, "from the node to the left", "-"),
                                    (SKIP, "from every node further left (concatenated too)", "--")]):
        x = lx + 22 * k
        ax.plot([x, x + 4], [ly, ly], color=c, ls=ls, lw=1.6)
        ax.text(x + 5, ly, t, fontsize=8.4, va="center", color=NOTE)
    ax.text(52, 6.5, "green-teal nodes keep the width of their encoder level; the dark green ones "
                     "(i + j = 4) are the decoder path and have the U-Net's decoder widths "
                     "(layout of segmentation_models_pytorch)", ha="center", fontsize=8.4,
            color=NOTE, style="italic")


def unetpp_inside(ax, P):
    blank(ax, ylim=(0, 100))
    E, hw = P["enc_ch"], P["hw"]
    i, j = 1, 2
    below, left = width(P, i + 1, j - 1), E[i] * j
    panel_title(ax, "Inside a node",
                f"shown for X({i},{j}): X({i + 1},{j - 1}) from below + X({i},0) and X({i},1) from the "
                f"left, at {hw[i]} x {hw[i]}")
    box(ax, 8, 70, 13, 12, f"X {i + 1},{j - 1}", f"{below} ch @ {hw[i + 1]}²", color=MID, fs=8.8)
    box(ax, 8, 34, 13, 12, f"X {i},0 ... X {i},{j - 1}", f"{j} x {E[i]} ch @ {hw[i]}²", color=ENC, fs=8.8)
    box(ax, 27, 70, 11, 12, "nearest ↑2", f"{below} @ {hw[i]}²", color=DEC, fs=8.8)
    arrow(ax, (14.7, 70), (21.3, 70), color=GREY, lw=1.3)
    op_node(ax, 40, 52, "||", color=GREY, r=1.9, fs=10)
    arrow(ax, (32.7, 70), (39.2, 53.8), color=GREY, lw=1.3)
    arrow(ax, (14.7, 34), (39.2, 50.2), color=GREY, lw=1.3)
    ax.text(40, 43.5, f"concat {below + left} ch", ha="center", fontsize=8, color=GREY)
    box(ax, 56, 52, 15, 12, "DoubleConv", "2 x (3x3 conv, BN, ReLU)", color=MID, fs=8.8)
    arrow(ax, (42, 52), (48.3, 52), color=GREY, lw=1.3)
    ax.text(56, 38, f"→ X {i},{j}: {width(P, i, j)} ch, {P['nodes'][f'{i}_{j}']:,} par",
            ha="center", fontsize=8.4, color="#1B2631", fontweight="bold")
    big = max(P["nodes"], key=P["nodes"].get)
    bi, bj = map(int, big.split("_"))
    ax.text(82, 60, "Why U-Net++ is big", ha="center", fontsize=10, fontweight="bold", color="#1B2631")
    ax.text(82, 44, f"The in-between nodes keep the encoder's widths and\n"
                    f"take every node to their left. The largest, X({bi},{bj}),\n"
                    f"turns {width(P, bi + 1, bj - 1) + E[bi] * bj} channels into {width(P, bi, bj)} at "
                    f"{hw[bi]}²: {fmt(P['nodes'][big])} parameters\non its own.",
            ha="center", va="center", fontsize=8.8, color=NOTE)


def unetpp_table(ax, P):
    E, hw = P["enc_ch"], P["hw"]
    rows = []
    for j in range(1, 5):
        for i in range(5 - j):
            below, left = width(P, i + 1, j - 1), E[i] * j
            rows.append([f"X {i},{j}", f"{hw[i]} x {hw[i]}", f"{below}", f"{j} x {E[i]} = {left}",
                         f"{width(P, i, j)}" + ("   (decoder path)" if i + j == 4 else ""),
                         f"{P['nodes'][f'{i}_{j}']:,}"])
    rows.append(["last + head", f"{P['size']} x {P['size']}", f"{P['dec_ch'][-2]}", "-",
                 f"{P['dec_ch'][-1]} -> {P['num_classes']}", f"{P['last'] + P['head']:,}"])
    rows.append(["encoder", "", "", "", "ResNet-50, pretrained", f"{P['enc_total']:,}"])
    total = ["TOTAL", "", "", "", "", f"{P['total']:,}"]
    new = P["total"] - P["enc_total"]
    U = measure("unet_r50", P["in_ch"], P["size"])
    u_dec = U["total"] - U["enc_total"]
    foot = (f"encoder {fmt(P['enc_total'])} ({100 * P['enc_total'] / P['total']:.0f}%)   •   "
            f"nodes, last step and head {fmt(new)}, {new / u_dec:.1f}x the U-Net R50 decoder "
            f"({fmt(u_dec)})")
    table(ax, "Every node", "inputs and output in channels",
          ["node", "resolution", "from below", "from the left", "out channels", "par"], rows, total,
          foot, [11, 13, 12, 18, 26, 14])


def draw_unetpp(P, out, dpi):
    fig = plt.figure(figsize=(17.5, 19))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.45, 0.7, 1.45], hspace=0.1, left=0.03, right=0.975,
                          top=0.958, bottom=0.015)
    fig.suptitle(f"U-Net++ R50 - architecture (NIfTI, {P['size']} x {P['size']}, 2.5D)", fontsize=19,
                 fontweight="bold", y=0.985, color="#1B2631")
    unetpp_flow(fig.add_subplot(gs[0]), P)
    unetpp_inside(fig.add_subplot(gs[1]), P)
    unetpp_table(fig.add_subplot(gs[2]), P)
    fig.savefig(out, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"[saved] {out}")


def main():
    from src.config import ARCH_DIR
    p = argparse.ArgumentParser(description="diagrams of the ResNet-50 models")
    p.add_argument("--arch", choices=["unet_r50", "attention_unet_r50", "unetpp_r50"], default=None)
    p.add_argument("--in-slices", type=int, default=3, help="input slices (3 = 2.5D, as trained)")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--dpi", type=int, default=100)
    a = p.parse_args()
    out = Path(ARCH_DIR) / "nifti"
    out.mkdir(parents=True, exist_ok=True)
    for arch in [a.arch] if a.arch else ["unet_r50", "attention_unet_r50", "unetpp_r50"]:
        P = measure(arch, a.in_slices, a.image_size)
        (draw_unetpp if arch == "unetpp_r50" else draw_unet)(P, out / f"{arch}.png", a.dpi)


if __name__ == "__main__":
    main()
