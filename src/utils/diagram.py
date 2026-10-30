"""Architecture diagram for the Attention U-Net.

The figure is built from a `plan()` computed analytically from the config, not
from a torch forward pass, so it renders anywhere. summary.py cross-checks the
plan's parameter total against the real model, so the two cannot drift apart.

    python -m src.utils.diagram
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ENC = "#2E6DA4"      # encoder
BOT = "#7D3C98"      # bottleneck
GATE = "#D9822B"     # attention gate
DEC = "#2E8B57"      # decoder
HEAD = "#B03A2E"     # output head
GREY = "#5D6D7E"
LINE = "#7F8C8D"
SKIP = "#16A085"


# --------------------------------------------------------------------------
# the plan: every tensor shape and parameter count, derived from the config
# --------------------------------------------------------------------------

def plan(base_filters=32, depth=4, num_classes=3, image_size=256,
         in_channels=1, batch_norm=True, bilinear=False, arch="attention_unet"):
    f = [base_filters * 2 ** i for i in range(depth + 1)]
    hw = [image_size // 2 ** i for i in range(depth + 1)]

    # nn.Conv2d(a, b, 3, padding=1, bias=not batch_norm) inside DoubleConv
    def conv(i, o, k, bias):
        return i * o * k * k + (o if bias else 0)

    def bn(c):
        return 2 * c

    def double_conv(i, o):
        b = not batch_norm
        p = conv(i, o, 3, b) + conv(o, o, 3, b)
        if batch_norm:
            p += bn(o) + bn(o)
        return p

    enc = []
    enc.append(dict(name="inc", op="DoubleConv", in_ch=in_channels, out_ch=f[0],
                    hw=hw[0], params=double_conv(in_channels, f[0]), level=0))
    for i in range(depth):
        enc.append(dict(name=f"downs.{i}", op="MaxPool2d(2) + DoubleConv",
                        in_ch=f[i], out_ch=f[i + 1], hw=hw[i + 1],
                        params=double_conv(f[i], f[i + 1]), level=i + 1))

    attention = arch == "attention_unet"
    gates, ups = [], []
    for i in reversed(range(depth)):
        g_ch, x_ch = f[i + 1], f[i]
        inter = max(f[i] // 2, 1)
        if attention:
            # AttentionGate always uses BatchNorm and 1x1 convs with bias
            p = (conv(g_ch, inter, 1, True) + bn(inter)
                 + conv(x_ch, inter, 1, True) + bn(inter)
                 + conv(inter, 1, 1, True) + bn(1))
            gates.append(dict(name=f"gates.{depth-1-i}", level=i, g_ch=g_ch,
                              x_ch=x_ch, inter=inter, g_hw=hw[i + 1], x_hw=hw[i],
                              params=p))

        if bilinear:
            up_p, up_out, up_op = 0, f[i + 1], "Upsample(bilinear, x2)"
        else:
            up_out = f[i + 1] // 2
            up_p = f[i + 1] * up_out * 4 + up_out       # ConvTranspose2d k=2 s=2
            up_op = f"ConvTranspose2d({f[i+1]}->{up_out}, k2, s2)"
        cat_ch = up_out + f[i]
        ups.append(dict(name=f"ups.{depth-1-i}", level=i, in_ch=f[i + 1],
                        up_out=up_out, up_op=up_op, skip_ch=f[i],
                        cat_ch=cat_ch, out_ch=f[i], hw=hw[i],
                        params=up_p + double_conv(cat_ch, f[i]),
                        up_params=up_p, conv_params=double_conv(cat_ch, f[i])))

    head = dict(name="head", op=f"Conv2d({f[0]}->{num_classes}, k1)",
                in_ch=f[0], out_ch=num_classes, hw=hw[0],
                params=conv(f[0], num_classes, 1, True))

    total = sum(e["params"] for e in enc) + sum(g["params"] for g in gates) \
        + sum(u["params"] for u in ups) + head["params"]

    return dict(f=f, hw=hw, depth=depth, image_size=image_size, arch=arch,
                title="Attention U-Net" if attention else "U-Net (baseline)",
                in_channels=in_channels, num_classes=num_classes,
                batch_norm=batch_norm, bilinear=bilinear,
                encoder=enc, gates=gates, ups=ups, head=head, total=total)


# --------------------------------------------------------------------------
# drawing helpers
# --------------------------------------------------------------------------

def box(ax, x, y, w, h, title, sub=None, color=ENC, fc=None, fs=9.5,
        tc="white", lw=0, ls="-"):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.25,rounding_size=0.8",
                       facecolor=fc or color, edgecolor=color,
                       linewidth=lw or 1.2, linestyle=ls, zorder=3)
    ax.add_patch(p)
    if sub:
        ax.text(x, y + h * 0.17, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=4)
        ax.text(x, y - h * 0.24, sub, ha="center", va="center",
                fontsize=fs - 1.7, color=tc, zorder=4)
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=fs,
                fontweight="bold", color=tc, zorder=4)
    return p


def arrow(ax, p0, p1, color=LINE, lw=1.6, rad=0.0, ls="-", label=None,
          lpos=0.5, loff=(0, 1.6), fs=8, lcolor=None, zorder=2, alpha=1.0,
          cs=None):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13,
                        color=color, lw=lw, linestyle=ls, alpha=alpha,
                        connectionstyle=cs or f"arc3,rad={rad}", zorder=zorder,
                        shrinkA=2, shrinkB=3)
    ax.add_patch(a)
    if label:
        mx = p0[0] + (p1[0] - p0[0]) * lpos + loff[0]
        my = p0[1] + (p1[1] - p0[1]) * lpos + loff[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=fs,
                color=lcolor or color, zorder=5,
                bbox=dict(boxstyle="round,pad=0.22", fc="white",
                          ec="none", alpha=0.88))
    return a


def op_node(ax, x, y, sym, color=GREY, r=1.5, fs=12):
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=color,
                        linewidth=1.5, zorder=4))
    ax.text(x, y, sym, ha="center", va="center", fontsize=fs,
            color=color, fontweight="bold", zorder=5)


def panel_title(ax, text, sub=None):
    """Offsets in POINTS, not data units.

    Panels have different heights, so a fixed offset in data coordinates
    collapses the gap between title and subtitle on the short ones.
    """
    ax.annotate(text, xy=(0, 1), xycoords="axes fraction",
                xytext=(0, -2), textcoords="offset points",
                fontsize=13.5, fontweight="bold", color="#1B2631",
                ha="left", va="top")
    if sub:
        ax.annotate(sub, xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, -19), textcoords="offset points",
                    fontsize=9.5, color="#566573", ha="left", va="top")


def blank(ax, xlim=(0, 100), ylim=(0, 100)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


# --------------------------------------------------------------------------
# panel 1: the full data path
# --------------------------------------------------------------------------

def flow_panel(ax, P):
    blank(ax, xlim=(-12, 104), ylim=(-13, 118))
    depth, f, hw = P["depth"], P["f"], P["hw"]

    x_enc, x_gate, x_dec = 16, 53, 88
    ys = [84 - i * 17.0 for i in range(depth + 1)]
    bw, bh = 24, 8.6
    gw, gh = 20, 7.8
    y_io = ys[0] + 15

    panel_title(ax, f"{P.get('title', 'Attention U-Net')} — full data path",
                f"input {P['in_channels']}x{P['image_size']}x{P['image_size']}   •   "
                f"base_filters={f[0]}   •   depth={depth}   •   "
                f"{P['total']:,} trainable parameters")

    # ---- resolution ruler, in its own left margin
    ax.text(-11, ys[0] + 7, "resolution", fontsize=8.5, color="#85929E",
            ha="left", style="italic")
    for i, y in enumerate(ys):
        ax.text(-11, y, f"{hw[i]}²", fontsize=10, color="#212F3D",
                ha="left", va="center", fontweight="bold")
        ax.plot([-11.5, 1.5], [y - 4.6, y - 4.6], color="#E5E8E8", lw=0.8, zorder=0)

    # ---- input / output
    box(ax, x_enc, y_io, bw, bh, "INPUT   CT slice",
        f"{P['in_channels']} x {P['image_size']} x {P['image_size']}", color="#34495E")
    arrow(ax, (x_enc, y_io - bh / 2), (x_enc, ys[0] + bh / 2), color="#34495E", lw=2)

    # ---- encoder + bottleneck
    for i in range(depth + 1):
        e = P["encoder"][i]
        last = i == depth
        box(ax, x_gate if last else x_enc, ys[i], 40 if last else bw, bh,
            f"BOTTLENECK   downs.{depth-1}" if last else
            ("inc   DoubleConv" if i == 0 else f"downs.{i-1}   Down"),
            f"{e['out_ch']} x {hw[i]} x {hw[i]}     {e['params']:,} par",
            color=BOT if last else ENC)
        if i < depth:
            dst = (x_enc, ys[i + 1] + bh / 2) if i + 1 < depth \
                else (x_gate - 14, ys[i + 1] + bh / 2)
            arrow(ax, (x_enc, ys[i] - bh / 2), dst, color=ENC, lw=2.2,
                  rad=0 if i + 1 < depth else -0.12)
            lx = x_enc if i + 1 < depth else x_enc + 6
            ax.text(lx, (ys[i] + ys[i + 1]) / 2, "MaxPool2d(2)\n↓ half the size",
                    fontsize=8.3, color="#1A5276", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))

    deep = (f"deepest features: {f[depth]} channels on a {hw[depth]}x{hw[depth]} grid — "
            f"one cell summarises a {2**depth}x{2**depth}-pixel patch.")
    if P["gates"]:
        deep += ("\nthis tensor is used TWICE: upsampled by ups.0, and sent sideways "
                 "as the gating signal for gates.0.")
    ax.text(x_gate, ys[depth] - 7.5, deep,
            ha="center", va="top", fontsize=8.4, color="#5B2C6F", style="italic")

    # ---- decoder column
    for u in P["ups"]:
        box(ax, x_dec, ys[u["level"]], bw, bh, f"{u['name']}   Up",
            f"{u['out_ch']} x {u['hw']} x {u['hw']}     {u['params']:,} par", color=DEC)

    for u in P["ups"]:
        i = u["level"]
        if i + 1 == depth:
            arrow(ax, (x_gate + 20, ys[i + 1]), (x_dec, ys[i] + bh / 2),
                  color=DEC, lw=2.2, rad=0.22)
        else:
            arrow(ax, (x_dec, ys[i + 1] - bh / 2), (x_dec, ys[i] + bh / 2),
                  color=DEC, lw=2.2)
        ax.text(x_dec, (ys[i] + ys[i + 1]) / 2,
                f"ConvTranspose2d\n{u['in_ch']}→{u['up_out']}, k2 s2\n↑ double the size",
                fontsize=8.3, color="#186A3B", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))

    # ---- the skips. gated one per level, or straight through for plain U-Net
    if P["gates"]:
        skips = P["gates"]
    else:
        skips = [dict(name=None, level=u["level"], x_ch=u["skip_ch"],
                      x_hw=u["hw"]) for u in P["ups"]]

    for g in skips:
        i, y = g["level"], ys[g["level"]]

        if g["name"]:
            box(ax, x_gate, y, gw, gh, f"{g['name']}   AttentionGate",
                f"α: 1 x {g['x_hw']}²    {g['params']:,} par", color=GATE)

            arrow(ax, (x_enc + bw / 2, y), (x_gate - gw / 2, y), color=SKIP, lw=2.3)
            ax.text((x_enc + bw / 2 + x_gate - gw / 2) / 2, y + 3.4,
                    f"skip   x: {g['x_ch']}ch", fontsize=8.2, color="#0E6251",
                    ha="center", va="center")

            arrow(ax, (x_gate + gw / 2, y), (x_dec - bw / 2, y), color=SKIP, lw=2.3)
            ax.text((x_gate + gw / 2 + x_dec - bw / 2) / 2, y + 3.4, "x ⊙ α",
                    fontsize=8.6, color="#0E6251", ha="center", va="center")
            ax.text((x_gate + gw / 2 + x_dec - bw / 2) / 2, y - 3.6,
                    f"concat → {g['x_ch'] * 2}ch", fontsize=8.0, color="#0E6251",
                    ha="center", va="center")

            # gating signal: from the level below, BEFORE it is upsampled
            gx = x_gate + 20 if i + 1 == depth else x_dec - bw / 2
            arrow(ax, (gx, ys[i + 1]), (x_gate + 4, y - gh / 2), color="#B9770E",
                  lw=1.8, rad=-0.3, ls=(0, (5, 2)))
            ax.text(x_gate + 15.5, (ys[i + 1] + y) / 2 - 1.2,
                    f"g: {g['g_ch']}ch @ {g['g_hw']}²", fontsize=8.0, color="#B9770E",
                    ha="center", va="center", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=.92))
        else:
            # plain U-Net: encoder straight to the concatenation, nothing filters it
            arrow(ax, (x_enc + bw / 2, y), (x_dec - bw / 2, y), color=SKIP, lw=2.3)
            mid = (x_enc + bw / 2 + x_dec - bw / 2) / 2
            ax.text(mid, y + 3.6, f"skip   {g['x_ch']}ch @ {g['x_hw']}²  —  unfiltered",
                    fontsize=8.4, color="#0E6251", ha="center", va="center")
            ax.text(mid, y - 3.8, f"concat → {g['x_ch'] * 2}ch", fontsize=8.0,
                    color="#0E6251", ha="center", va="center")

    # ---- head
    h = P["head"]
    box(ax, x_dec, y_io, bw, bh, "head   Conv2d 1x1",
        f"{h['out_ch']} x {h['hw']}²  logits    {h['params']} par", color=HEAD)
    arrow(ax, (x_dec, ys[0] + bh / 2), (x_dec, y_io - bh / 2), color=HEAD, lw=2)
    ax.text(x_dec, y_io + 9.5,
            "softmax over the class axis → argmax\n"
            f"{h['out_ch']} classes: background / liver / tumour, one label per pixel",
            ha="center", va="center", fontsize=8.5, color="#943126")

    if P["gates"]:
        note = ("the gate's g comes from the level BELOW, before upsampling — "
                "the very tensor the ConvTranspose is about to consume.\n"
                "So every skip is filtered by what the coarse decoder already believes is "
                "relevant, and only then concatenated. That is the whole difference from a plain U-Net.")
    else:
        note = ("BASELINE — every skip is concatenated exactly as the encoder produced it. "
                "The whole abdomen crosses over: kidneys, spine, bowel, table.\n"
                "The decoder has to sort out what matters on its own. This is the model the "
                "attention variant has to beat to justify its extra parameters.")
    ax.text(x_gate, -8, note, ha="center", va="center", fontsize=9,
            color="#7E5109", style="italic")


# --------------------------------------------------------------------------
# panel 2: what is inside an attention gate
# --------------------------------------------------------------------------

def gate_panel(ax, P):
    blank(ax, xlim=(0, 104), ylim=(-16, 112))
    g = P["gates"][0]
    panel_title(ax, "Inside AttentionGate — how a skip connection gets filtered",
                f"shown for {g['name']}:   g = {g['g_ch']}ch @ {g['g_hw']}²  "
                f"(bottleneck, one level coarser)     "
                f"x = {g['x_ch']}ch @ {g['x_hw']}²  (encoder skip)")

    ax.text(52, 90,
            r"$\alpha \;=\; \sigma\!\left(\psi\!\left(\mathrm{ReLU}\!\left("
            r"W_g\,g \;+\; W_x\,x\right)\right)\right)"
            r"\qquad\qquad \mathrm{out} \;=\; x \odot \alpha$",
            ha="center", va="center", fontsize=13.5, color="#1B2631")

    yg, yx, ym, yb = 72, 44, 58, 16
    bh = 9

    # top branch: the gating signal
    box(ax, 8, yg, 13, bh, "g", f"{g['g_ch']} x {g['g_hw']}²", color=BOT)
    box(ax, 30, yg, 17, bh, "W_g", f"Conv 1x1 → {g['inter']}ch", color=GATE)
    box(ax, 55, yg, 19, bh, "interpolate", f"{g['g_hw']}² → {g['x_hw']}²", color="#AF7AC5")
    arrow(ax, (14.5, yg), (21.5, yg), color=LINE)
    arrow(ax, (38.5, yg), (45.5, yg), color=LINE)
    ax.text(42, yg + 4.2, f"{g['inter']} x {g['g_hw']}²", fontsize=7.8,
            color=GREY, ha="center")
    ax.text(55, yg + 6.5, "g is coarser than x, so it is resized first —\n"
                          "the two must share a grid before they can be added",
            ha="center", va="bottom", fontsize=8.2, color="#6C3483", style="italic")

    # bottom branch: the skip itself
    box(ax, 8, yx, 13, bh, "x", f"{g['x_ch']} x {g['x_hw']}²", color=ENC)
    box(ax, 30, yx, 17, bh, "W_x", f"Conv 1x1 → {g['inter']}ch", color=GATE)
    arrow(ax, (14.5, yx), (21.5, yx), color=LINE)

    # merge
    op_node(ax, 74, ym, "+", color="#2C3E50", r=2.6, fs=16)
    arrow(ax, (64.5, yg), (74, ym + 2.6), color=LINE, rad=-0.25)
    arrow(ax, (38.5, yx), (74, ym - 2.6), color=LINE, rad=0.16)
    ax.text(56, yx - 3.2, f"{g['inter']} x {g['x_hw']}²", fontsize=7.8,
            color=GREY, ha="center")

    box(ax, 88, ym, 13, bh, "ReLU", color="#5D6D7E")
    arrow(ax, (76.6, ym), (81.5, ym), color=LINE)

    box(ax, 88, 32, 15, bh, "ψ + Sigmoid", "Conv 1x1 → 1", color="#CA6F1E")
    arrow(ax, (88, ym - bh / 2), (88, 32 + bh / 2), color=LINE)

    # the attention map and the product
    box(ax, 58, yb, 24, bh, "α   attention map", f"1 x {g['x_hw']}²  values in (0,1)",
        color="#E59866", tc="#4A2500")
    arrow(ax, (88, 32 - bh / 2), (70, yb), color=LINE, rad=0.28)

    op_node(ax, 32, yb, "⊙", color="#0E6251", r=2.8, fs=15)
    arrow(ax, (46, yb), (34.8, yb), color=LINE)
    arrow(ax, (8, yx - bh / 2), (29.4, yb + 1.2), color=SKIP, lw=2, rad=0.32)
    ax.text(24, yb + 13, "the raw skip x,\nnot W_x x", fontsize=8, color="#0E6251",
            ha="center", va="center", style="italic")

    box(ax, 8, yb, 13, bh, "out", f"{g['x_ch']} x {g['x_hw']}²", color=SKIP)
    arrow(ax, (29.2, yb), (14.5, yb), color=SKIP, lw=2)

    ax.text(52, -3,
            f"α holds ONE weight per pixel, broadcast across all {g['x_ch']} channels of the skip. "
            "Positions the decoder finds irrelevant are multiplied toward zero,\n"
            "so what reaches the concatenation is organ context rather than the whole abdomen. "
            f"Cost: {g['params']:,} parameters, {100*g['params']/P['total']:.1f}% of the network — "
            "cheap because every conv here is 1x1.",
            ha="center", va="center", fontsize=9, color="#7E5109")


# --------------------------------------------------------------------------
# panel 3 / 4: DoubleConv and Up internals
# --------------------------------------------------------------------------

def doubleconv_panel(ax, P):
    blank(ax, xlim=(0, 100), ylim=(-8, 112))
    e = P["encoder"][1]
    ci, co = e["in_ch"], e["out_ch"]
    panel_title(ax, "Inside Down  =  MaxPool2d(2)  +  DoubleConv",
                f"shown for downs.0:   {ci}ch @ {P['hw'][0]}²   →   {co}ch @ {P['hw'][1]}²"
                f"   ({e['params']:,} parameters)")

    xs = [7.5, 23, 37.5, 50.5, 65, 79.5, 92.5]
    y = 62
    steps = [("MaxPool2d", "2x2, stride 2", "#1A5276"),
             ("Conv2d", f"3x3 pad 1   {ci}→{co}", ENC),
             ("BatchNorm2d", f"{co}", "#5499C7"),
             ("ReLU", "", "#5D6D7E"),
             ("Conv2d", f"3x3 pad 1   {co}→{co}", ENC),
             ("BatchNorm2d", f"{co}", "#5499C7"),
             ("ReLU", "", "#5D6D7E")]
    for i, (t, s, c) in enumerate(steps):
        box(ax, xs[i], y, 13, 20, t, s or None, color=c, fs=8.4)
        if i:
            arrow(ax, (xs[i - 1] + 6.6, y), (xs[i] - 6.6, y), color=LINE, lw=1.4)

    ax.annotate("", xy=(xs[0] + 6.5, 40), xytext=(xs[0] - 6.5, 40),
                arrowprops=dict(arrowstyle="<->", color="#1A5276", lw=1.4))
    ax.text(xs[0], 33, f"{P['hw'][0]}² → {P['hw'][1]}²\nthe ONLY place\nresolution drops",
            ha="center", va="top", fontsize=8.2, color="#1A5276", fontweight="bold")

    ax.annotate("", xy=(xs[6] + 6.5, 40), xytext=(xs[1] - 6.5, 40),
                arrowprops=dict(arrowstyle="<->", color=ENC, lw=1.4))
    ax.text((xs[1] + xs[6]) / 2, 33,
            f"padding=1 holds {P['hw'][1]}² fixed through both convolutions — "
            "channels change, the grid does not",
            ha="center", va="top", fontsize=8.2, color="#1A5276")

    ax.text(50, 6,
            "Two stacked 3x3 convolutions see a 5x5 neighbourhood for 2x9 weights per channel pair instead of 25, "
            "with a ReLU in between so the pair\nis not equivalent to one linear filter.  "
            "bias=False in the convs because the BatchNorm immediately after re-centres the activations anyway.",
            ha="center", va="center", fontsize=8.6, color="#566573", style="italic")


def up_panel(ax, P):
    blank(ax, xlim=(0, 100), ylim=(-10, 112))
    u = P["ups"][0]
    gated = bool(P["gates"])
    skip_word = "gated skip" if gated else "raw skip"
    panel_title(ax, "Inside Up  —  upsample, concatenate, refine",
                f"shown for {u['name']}:   {u['in_ch']}ch @ {P['hw'][u['level']+1]}²  "
                f"+  {skip_word} {u['skip_ch']}ch @ {u['hw']}²   →   {u['out_ch']}ch @ {u['hw']}²"
                f"   ({u['params']:,} parameters)")

    y1, y2, y = 72, 28, 50
    box(ax, 9, y1, 16, 12, "from below", f"{u['in_ch']} x {P['hw'][u['level']+1]}²", color=BOT)
    box(ax, 34, y1, 24, 12, "ConvTranspose2d",
        f"k2 s2  →  {u['up_out']} x {u['hw']}²", color=DEC)
    arrow(ax, (17, y1), (22, y1), color=LINE)
    ax.text(34, y1 + 9.5, f"↑ 2x   •   {u['up_params']:,} par   •   learned upsampling",
            ha="center", fontsize=8.2, color="#186A3B")

    box(ax, 9, y2, 16, 12, skip_word, f"{u['skip_ch']} x {u['hw']}²", color=SKIP)
    ax.text(9, y2 - 9.5,
            "already multiplied by α" if gated else "straight from the encoder,\nnothing removed",
            ha="center", va="top", fontsize=8, color="#0E6251", style="italic")

    op_node(ax, 58, y, "⧺", color="#2C3E50", r=3.2, fs=14)
    ax.text(58, y + 8.5, "concat on the channel axis", ha="center", fontsize=8.4,
            color="#2C3E50")
    ax.text(58, y - 8.5, f"{u['up_out']} + {u['skip_ch']} = {u['cat_ch']} ch",
            ha="center", fontsize=8.8, color="#2C3E50", fontweight="bold")
    arrow(ax, (46, y1), (58, y + 3.4), color=LINE, rad=-0.2)
    arrow(ax, (17, y2), (58, y - 3.4), color=LINE, rad=0.14)

    box(ax, 84, y, 26, 14, "DoubleConv",
        f"{u['cat_ch']} → {u['out_ch']}   ({u['conv_params']:,} par)", color=DEC)
    arrow(ax, (61.5, y), (71, y), color=LINE)

    tail = ("That is exactly why filtering the skip matters — an unfiltered one drags "
            "the whole abdomen back into the decoder." if gated else
            "Nothing filters the skip here, so whatever the encoder saw at this "
            "resolution arrives in full — that is the baseline's weakness.")
    ax.text(50, 4,
            "The transposed convolution restores resolution but invents the detail; the skip supplies real edges from the encoder.\n"
            + tail + "  F.pad before concat handles odd sizes; at 256² it is a no-op.",
            ha="center", va="center", fontsize=8.6, color="#566573", style="italic")


# --------------------------------------------------------------------------
# panel 5: the ladder
# --------------------------------------------------------------------------

def ladder_panel(ax, P):
    blank(ax, xlim=(0, 100), ylim=(-14, 104))
    panel_title(ax, "Resolution and channel ladder",
                "every tensor the forward pass produces, top of the U to the bottom and back")

    rows = [("level", "resolution", "encoder", "gate α", "decoder", "params")]
    depth, f, hw = P["depth"], P["f"], P["hw"]
    gate_by_level = {g["level"]: g for g in P["gates"]}
    up_by_level = {u["level"]: u for u in P["ups"]}
    for i in range(depth + 1):
        e = P["encoder"][i]
        g = gate_by_level.get(i)
        u = up_by_level.get(i)
        p = e["params"] + (g["params"] if g else 0) + (u["params"] if u else 0)
        rows.append((
            f"{i}" + ("  (bottleneck)" if i == depth else ""),
            f"{hw[i]} x {hw[i]}",
            f"{e['out_ch']} ch",
            f"1 x {hw[i]}²" if g else "—",
            f"{u['out_ch']} ch" if u else "—",
            f"{p:,}",
        ))
    rows.append(("head", f"{hw[0]} x {hw[0]}", f"{f[0]} ch", "—",
                 f"{P['num_classes']} ch", f"{P['head']['params']:,}"))

    xs = [4, 22, 42, 58, 74, 92]
    y0, dy = 84, 9.6
    for r, row in enumerate(rows):
        y = y0 - r * dy
        if r == 0:
            ax.plot([2, 98], [y - 4.4, y - 4.4], color="#34495E", lw=1.4)
        elif r % 2 == 0:
            ax.add_patch(FancyBboxPatch((2, y - 4.3), 96, 8.6,
                                        boxstyle="square,pad=0",
                                        facecolor="#F4F6F7", edgecolor="none",
                                        zorder=0))
        for c, (x, v) in enumerate(zip(xs, row)):
            ax.text(x, y, v, fontsize=9.2 if r else 9.4,
                    fontweight="bold" if r == 0 or "bottleneck" in row[0] else "normal",
                    color="#1B2631" if r == 0 else "#212F3D",
                    ha="right" if c == 5 else "left", va="center")

    y = y0 - len(rows) * dy
    ax.plot([2, 98], [y + 4.0, y + 4.0], color="#34495E", lw=1.4)
    ax.text(4, y, "TOTAL", fontsize=10, fontweight="bold", color="#1B2631", va="center")
    ax.text(92, y, f"{P['total']:,}", fontsize=10, fontweight="bold",
            color="#1B2631", ha="right", va="center")

    enc_p = sum(e["params"] for e in P["encoder"])
    gat_p = sum(g["params"] for g in P["gates"])
    up_p = sum(u["params"] for u in P["ups"])
    ax.text(50, y - 11,
            f"encoder {enc_p:,} ({100*enc_p/P['total']:.1f}%)   •   "
            f"decoder {up_p:,} ({100*up_p/P['total']:.1f}%)   •   "
            f"attention gates {gat_p:,} ({100*gat_p/P['total']:.1f}%)   •   "
            f"head {P['head']['params']}",
            ha="center", va="center", fontsize=9, color="#566573")
    if gat_p:
        closing = ("attention costs about one percent of the weights. It is cheap because "
                   "every convolution in the gate is 1x1 —\nit reweights positions, it does "
                   "not look at neighbourhoods.")
    else:
        closing = ("no attention gates: every skip is concatenated as-is. Adding them costs "
                   f"132,252 parameters at this width\n(+1.70%), which is what the ablation "
                   "has to justify on foreground Dice.")
    ax.text(50, y - 19, closing,
            ha="center", va="center", fontsize=8.6, color="#7E5109", style="italic")


# --------------------------------------------------------------------------

def draw(P, out_path, dpi=100):
    has_gates = bool(P["gates"])
    # the gate panel only exists when there are gates; drop the row for the baseline
    ratios = [2.42, 1.28, 0.76, 0.80, 1.12] if has_gates else [2.42, 0.76, 0.80, 1.12]
    height = 27 if has_gates else 21.5

    fig = plt.figure(figsize=(17.5, height))
    gs = fig.add_gridspec(len(ratios), 1, height_ratios=ratios,
                          hspace=0.22, left=0.035, right=0.975,
                          top=0.962, bottom=0.018)
    fig.patch.set_facecolor("white")
    fig.suptitle(f"{P.get('title', 'Attention U-Net')} — architecture", fontsize=19,
                 fontweight="bold", y=0.988, color="#1B2631")

    r = 0
    flow_panel(fig.add_subplot(gs[r]), P); r += 1
    if has_gates:
        gate_panel(fig.add_subplot(gs[r]), P); r += 1
    doubleconv_panel(fig.add_subplot(gs[r]), P); r += 1
    up_panel(fig.add_subplot(gs[r]), P); r += 1
    ladder_panel(fig.add_subplot(gs[r]), P)

    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"[saved] {out_path}  ({17.5*dpi:.0f}x{height*dpi:.0f} px)")


def main():
    from src.config import Config, ARCH_DIR, SCRATCH_ARCHS as ARCH_NAMES, IN_CHANNELS

    p = argparse.ArgumentParser(description="draw the architecture diagram")
    # the from-scratch U-Nets only; the ResNet-50 ones are not drawn here
    p.add_argument("--arch", choices=ARCH_NAMES + ["both"], default=None)
    p.add_argument("--base-filters", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    cfg = Config()
    arches = ARCH_NAMES if a.arch == "both" else [a.arch or cfg.arch]

    for arch in arches:
        P = plan(base_filters=a.base_filters or cfg.base_filters,
                 depth=a.depth or cfg.depth,
                 num_classes=cfg.num_classes,
                 image_size=a.image_size or cfg.image_size,
                 in_channels=IN_CHANNELS,
                 batch_norm=cfg.batch_norm,
                 bilinear=cfg.bilinear,
                 arch=arch)

        out = Path(a.out) if a.out and len(arches) == 1 \
            else Path(ARCH_DIR) / f"{arch}_diagram.png"
        out.parent.mkdir(parents=True, exist_ok=True)

        for dpi in (a.dpi, 80, 65):
            try:
                draw(P, out, dpi)
                break
            except MemoryError:
                print(f"[figure] out of memory at dpi={dpi}, retrying smaller")
                plt.close("all")
        else:
            print("[figure] could not render. close other applications and retry.")

    if len(arches) > 1:
        pa = plan(arch="attention_unet", base_filters=cfg.base_filters, depth=cfg.depth,
                  num_classes=cfg.num_classes, image_size=cfg.image_size)
        pu = plan(arch="unet", base_filters=cfg.base_filters, depth=cfg.depth,
                  num_classes=cfg.num_classes, image_size=cfg.image_size)
        d = pa["total"] - pu["total"]
        print(f"\n[ablation] attention_unet {pa['total']:,}  -  unet {pu['total']:,}"
              f"  =  {d:,} parameters ({100*d/pu['total']:.2f}% more)")
        print("[ablation] that difference is exactly the attention gates; everything "
              "else is identical.")


if __name__ == "__main__":
    main()
