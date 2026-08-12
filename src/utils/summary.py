
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import torch

from src.config import Config, ARCH_DIR, ARCH_NAMES, IN_CHANNELS
from src.model import build_model

COLORS = {
    "Conv2d": "#4A90D9", "BatchNorm2d": "#7FB069", "ReLU": "#F5A623",
    "MaxPool2d": "#D0743C", "ConvTranspose2d": "#C0392B", "Upsample": "#C0392B",
    "Dropout2d": "#B565A7", "Sigmoid": "#8E44AD", "Identity": "#9B9B9B",
}
STAGE_COLOR = {"encoder": "#4A90D9", "bottleneck": "#8E44AD", "gate": "#E67E22",
               "decoder": "#C0392B", "head": "#2C3E50", "input": "#555555"}


def collect_layers(model, input_size):
    """Record every leaf layer, tagging which named module owns it.

    The owner path is what lets the table say "this BatchNorm belongs to
    attention gate 2" rather than just "batchnorm2d_17".
    """
    owner = {m: name for name, m in model.named_modules()}
    records, counters = [], {}

    def hook(module, inp, out):
        kind = type(module).__name__
        counters[kind] = counters.get(kind, 0) + 1
        n = counters[kind] - 1
        records.append({
            "name": kind.lower() if n == 0 else f"{kind.lower()}_{n}",
            "type": kind,
            "path": owner.get(module, ""),
            "out_shape": tuple(out.shape[1:]),
            "params": sum(p.numel() for p in module.parameters(recurse=False)),
            "connected_to": records[-1]["name"] if records else "input_layer",
        })

    handles = [m.register_forward_hook(hook)
               for m in model.modules() if not list(m.children())]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(*input_size))
    for h in handles:
        h.remove()
    return records


def tag_stages(records, depth):
    """Label each layer encoder / bottleneck / gate / decoder / head.

    Read straight from the module path rather than inferred from tensor sizes -
    the paths come from named_modules() so they are exact:

        inc, downs.0 .. downs.{depth-2}  encoder
        downs.{depth-1}, drop            bottleneck (deepest level)
        gates.*                          attention gates
        ups.*                            decoder
        head                             output 1x1 conv

    Also records the resolution level (0 = full size, depth = bottleneck),
    which is what the diagram lays out on.
    """
    for r in records:
        p = r["path"]
        if p.startswith("gates."):
            r["stage"] = "gate"
            # gates.0 filters the deepest skip, gates.{depth-1} the shallowest
            r["level"] = depth - 1 - int(p.split(".")[1])
        elif p.startswith("ups."):
            r["stage"] = "decoder"
            r["level"] = depth - 1 - int(p.split(".")[1])
        elif p == "head":
            r["stage"] = "head"
            r["level"] = 0
        elif p == "drop" or p.startswith(f"downs.{depth-1}"):
            r["stage"] = "bottleneck"
            r["level"] = depth
        elif p.startswith("downs."):
            r["stage"] = "encoder"
            r["level"] = int(p.split(".")[1]) + 1
        else:                                   # inc.*
            r["stage"] = "encoder"
            r["level"] = 0

    # the layer that actually consumes a gated skip
    for r in records:
        r["skip"] = r["stage"] == "decoder" and r["type"] in ("ConvTranspose2d", "Upsample")
    return records


def print_table(records, cfg, input_size, out_path):
    lines, add = [], lambda s: lines.append(s)

    labels = ["input_layer (InputLayer)"] + [f"{r['name']} ({r['type']})" for r in records]
    shapes = [f"(None, {', '.join(map(str, input_size[1:]))})"] + \
             [f"(None, {', '.join(map(str, r['out_shape']))})" for r in records]
    conns = ["-"] + [r["connected_to"] for r in records]
    paths = ["-"] + [r["path"] or "-" for r in records]
    w1, w2 = max(map(len, labels)) + 2, max(map(len, shapes)) + 2
    w3, w4, w5 = 12, max(map(len, conns)) + 2, max(max(map(len, paths)), 22) + 2
    width = w1 + w2 + w3 + w4 + w5 + 12

    add(f'Model: "Attention U-Net"  ({cfg.num_classes} classes: '
        f'{", ".join(cfg.class_names)})')
    add(f'Input: {input_size[-1]}x{input_size[-1]}, {IN_CHANNELS} channel  |  '
        f'base_filters={cfg.base_filters}, depth={cfg.depth}')
    add("=" * width)
    add(f"{'Layer (type)':<{w1}}{'Output Shape':<{w2}}{'Param #':>{w3}}  "
        f"{'Connected to':<{w4}}{'Module path':<{w5}}{'Stage':<12}")
    add("=" * width)
    add(f"{labels[0]:<{w1}}{shapes[0]:<{w2}}{0:>{w3},}  {'-':<{w4}}{'-':<{w5}}{'input':<12}")
    add("-" * width)

    for i, r in enumerate(records, start=1):
        stage = r["stage"] + (" +skip" if r["skip"] else "")
        add(f"{labels[i]:<{w1}}{shapes[i]:<{w2}}{r['params']:>{w3},}  "
            f"{r['connected_to']:<{w4}}{(r['path'] or '-'):<{w5}}{stage:<12}")
        add("-" * width)

    add(f"{'softmax (applied in loss)':<{w1}}"
        f"{f'(None, {cfg.num_classes}, {input_size[-1]}, {input_size[-1]})':<{w2}}"
        f"{0:>{w3},}  {records[-1]['name']:<{w4}}{'-':<{w5}}{'output':<12}")
    add("=" * width)

    total = sum(r["params"] for r in records)
    by_stage = {}
    for r in records:
        by_stage[r["stage"]] = by_stage.get(r["stage"], 0) + r["params"]

    add(f"Total params:     {total:>13,}")
    add(f"Trainable params: {total:>13,}")
    add("")
    for st in ["encoder", "bottleneck", "gate", "decoder", "head"]:
        if st in by_stage:
            label = "attention gates" if st == "gate" else st
            add(f"  {label:<18}{by_stage[st]:>13,}  ({100*by_stage[st]/total:5.2f}%)")
    add("")
    n_gates = sum(1 for r in records if r["stage"] == "gate" and r["type"] == "Sigmoid")
    add(f"Attention gates: {n_gates} (one per skip connection)")
    add(f"Config: base_filters={cfg.base_filters}  depth={cfg.depth}  "
        f"batch_norm={cfg.batch_norm}  dropout={cfg.dropout}  bilinear={cfg.bilinear}")

    text = "\n".join(lines)
    print(text)
    Path(out_path).write_text(text, encoding="utf-8")
    print(f"\n[saved] {out_path}")
    return total


def write_html(records, cfg, input_size, out_path):
    total = sum(r["params"] for r in records)
    by_stage = {}
    for r in records:
        by_stage[r["stage"]] = by_stage.get(r["stage"], 0) + r["params"]
    n_gates = sum(1 for r in records if r["stage"] == "gate" and r["type"] == "Sigmoid")

    rows = [f'<tr class="io"><td><b>input_layer</b> '
            f'<span class="t" style="background:#555">InputLayer</span></td>'
            f'<td class="m">(None, {", ".join(map(str, input_size[1:]))})</td>'
            f'<td class="n zero">0</td><td class="c">&mdash;</td><td class="p">&mdash;</td>'
            f'<td class="s" style="color:#555">input</td></tr>']

    prev = None
    for r in records:
        shp = f'(None, {", ".join(map(str, r["out_shape"]))})'
        sep = " sep" if r["stage"] != prev else ""
        prev = r["stage"]
        pcls = "n" if r["params"] else "n zero"
        skip = ' <span class="skip">&#8618;</span>' if r["skip"] else ""
        label = "attention gate" if r["stage"] == "gate" else r["stage"]
        rows.append(
            f'<tr class="{sep.strip()}"><td><b>{r["name"]}</b> '
            f'<span class="t" style="background:{COLORS.get(r["type"], "#888")}">{r["type"]}</span></td>'
            f'<td class="m">{shp}</td><td class="{pcls}">{r["params"]:,}</td>'
            f'<td class="c">{r["connected_to"]}</td>'
            f'<td class="p">{r["path"] or "&mdash;"}</td>'
            f'<td class="s" style="color:{STAGE_COLOR[r["stage"]]}">{label}{skip}</td></tr>')

    rows.append(f'<tr class="io sep"><td><b>softmax</b> '
                f'<span class="t" style="background:#2C3E50">applied in loss</span></td>'
                f'<td class="m">(None, {cfg.num_classes}, {input_size[-1]}, {input_size[-1]})</td>'
                f'<td class="n zero">0</td><td class="c">{records[-1]["name"]}</td>'
                f'<td class="p">&mdash;</td>'
                f'<td class="s" style="color:#2C3E50">output</td></tr>')

    order = ["encoder", "bottleneck", "gate", "decoder", "head"]
    bars = "".join(
        f'<div style="width:{100*by_stage[s]/total:.2f}%;background:{STAGE_COLOR[s]}" '
        f'title="{s}: {by_stage[s]:,}">'
        f'{"gates" if s=="gate" else s} {100*by_stage[s]/total:.0f}%</div>'
        for s in order if s in by_stage)

    stat = "".join(
        f'<tr><td>{"attention gates" if s=="gate" else s}</td>'
        f'<td class="n">{by_stage[s]:,}</td>'
        f'<td class="n">{100*by_stage[s]/total:.2f}%</td></tr>'
        for s in order if s in by_stage)

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Attention U-Net</title><style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#f5f6f8;
      color:#1a1a1a;margin:0;padding:32px}}
 .wrap{{max-width:1180px;margin:0 auto;background:#fff;border-radius:10px;
        box-shadow:0 1px 3px rgba(0,0,0,.12);overflow:hidden}}
 h1{{margin:0;padding:20px 24px 4px;font-size:20px}}
 .sub{{padding:0 24px 6px;color:#666;font-size:13px;font-family:monospace}}
 .lead{{padding:0 24px 18px;color:#444;font-size:13.5px;line-height:1.6;max-width:900px}}
 table{{width:100%;border-collapse:collapse;font-size:12.5px}}
 th{{background:#2c3e50;color:#fff;text-align:left;padding:10px 12px;
     font-weight:600;font-size:11.5px;letter-spacing:.4px;text-transform:uppercase}}
 td{{padding:6px 12px;border-bottom:1px solid #eceef0}}
 tr:hover td{{background:#f0f7ff}}
 tr.sep td{{border-top:2px solid #cfd8e3}}
 tr.io td{{background:#fafbfc}}
 .t{{display:inline-block;padding:1px 7px;border-radius:9px;color:#fff;
     font-size:10px;font-weight:600;margin-left:6px;vertical-align:1px}}
 .m{{font-family:monospace;color:#444}}
 .p{{font-family:monospace;color:#999;font-size:11px}}
 .n{{text-align:right;font-family:monospace;font-weight:600}}
 .n.zero{{color:#bbb;font-weight:400}}
 .c{{font-family:monospace;color:#777;font-size:11.5px}}
 .s{{font-size:11px;text-align:right;font-weight:600}}
 .skip{{color:#E67E22;font-weight:700;font-size:13px}}
 .foot{{padding:18px 24px 26px;background:#fafbfc;border-top:2px solid #cfd8e3}}
 .big{{font-size:23px;font-weight:700}}
 .bar{{display:flex;height:24px;border-radius:4px;overflow:hidden;margin:12px 0 14px}}
 .bar div{{display:flex;align-items:center;justify-content:center;color:#fff;
          font-size:10.5px;font-weight:600;white-space:nowrap;overflow:hidden}}
 .stats{{width:auto;font-size:12.5px;margin-bottom:14px}}
 .stats td{{padding:3px 18px 3px 0;border:none}}
 .note{{font-size:12.5px;color:#555;line-height:1.75}}
 code{{background:#eceef0;padding:1px 5px;border-radius:3px;font-size:11.5px}}
</style></head><body><div class="wrap">
<h1>Attention U-Net</h1>
<div class="sub">{cfg.num_classes} classes ({", ".join(cfg.class_names)}) &nbsp;&bull;&nbsp;
 input {input_size[-1]}&times;{input_size[-1]}&times;{IN_CHANNELS} &nbsp;&bull;&nbsp;
 base_filters={cfg.base_filters} &nbsp;&bull;&nbsp; depth={cfg.depth} &nbsp;&bull;&nbsp;
 {n_gates} attention gates</div>
<div class="lead">
 Encoder&ndash;decoder with an attention gate on every skip connection. The encoder
 answers <i>what</i> is in the image and discards position; the decoder walks the
 resolution back up to answer <i>where</i>. Skip connections carry back the spatial
 detail the encoder threw away &mdash; and each gate filters its skip first, using the
 coarser decoder tensor as the query, so only the relevant regions pass through.
</div>
<table><thead><tr><th>Layer (type)</th><th>Output Shape</th>
<th style="text-align:right">Param #</th><th>Connected to</th><th>Module path</th>
<th style="text-align:right">Stage</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<div class="foot">
 <div class="big">{total:,} <span style="font-size:13px;font-weight:400;color:#666">trainable parameters</span></div>
 <div class="bar">{bars}</div>
 <table class="stats">{stat}</table>
 <div class="note">
  Output is one score per class per pixel, at full input resolution.<br>
  Rows marked <span class="skip">&#8618;</span> are where a gated skip connection is
  concatenated back in.<br>
  Each gate computes <code>alpha = sigmoid(psi(relu(Wg(g) + Wx(x))))</code> &mdash; a
  single-channel map in 0..1 that scales the skip. One channel, so it broadcasts:
  the gate chooses <i>where</i> to look, not which feature to keep.<br>
  Conv layers carry no bias when <code>batch_norm=True</code>, since BatchNorm
  subtracts the mean and would cancel it.
 </div>
</div></div></body></html>"""
    Path(out_path).write_text(html, encoding="utf-8")
    print(f"[saved] {out_path}")


# --------------------------------------------------------------------------
# drawing helpers, shared by every panel
# --------------------------------------------------------------------------

def _box(ax, x, y, w, h, label, sub="", color="#4A90D9", fs=9, text_color="white"):
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                facecolor=color, edgecolor="black",
                                linewidth=0.9, alpha=0.94))
    ax.text(x + w / 2, y + (h * 0.16 if sub else 0), label, ha="center", va="center",
            fontsize=fs, color=text_color, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y - h * 0.24, sub, ha="center", va="center",
                fontsize=fs - 1.4, color=text_color, family="monospace")


def _arrow(ax, p0, p1, color="#666", ls="-", lw=1.4, rad=None, label=None, dy=0):
    kw = dict(arrowstyle="-|>", mutation_scale=13, color=color, linewidth=lw,
              linestyle=ls)
    if rad is not None:
        kw["connectionstyle"] = f"arc3,rad={rad}"
    ax.add_patch(FancyArrowPatch(p0, p1, **kw))
    if label:
        ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + dy, label, ha="center",
                fontsize=7.6, color=color, style="italic")


def _op(ax, x, y, sym, color="#2C3E50", r=1.9, fs=13):
    ax.add_patch(plt.Circle((x, y), r, facecolor="white", edgecolor=color,
                            linewidth=1.8, zorder=3))
    ax.text(x, y, sym, ha="center", va="center", fontsize=fs, color=color,
            fontweight="bold", zorder=4)


def _header(ax, title, body):
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(0, 99, title, fontsize=12.5, fontweight="bold", va="top")
    ax.text(0, 88, body, fontsize=9.3, va="top", color="#555", linespacing=1.5)


# --------------------------------------------------------------------------
# panel A - the U-shaped overview
# --------------------------------------------------------------------------

def _overview_panel(ax, records, cfg, input_size):
    """Rows are resolution levels: encoder left, gate centre, decoder right.

    Every skip is therefore a straight horizontal line through its gate.
    """
    depth = cfg.depth
    agg = {}
    for r in records:
        key = (r["stage"], r["level"])
        e = agg.setdefault(key, {"params": 0, "ch": r["out_shape"][0],
                                 "size": r["out_shape"][-1]})
        e["params"] += r["params"]
        if r["stage"] != "gate":
            e["ch"], e["size"] = r["out_shape"][0], r["out_shape"][-1]

    total = sum(r["params"] for r in records)
    gate_params = sum(v["params"] for k, v in agg.items() if k[0] == "gate")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    n_rows = depth + 1
    row_h = 76.0 / n_rows
    bw, bh = 25.5, min(row_h * 0.50, 8.2)
    xl, xg, xr = 5.0, 42.5, 68.0
    y_of = lambda lv: 74.0 - (depth - lv) * row_h + 0.0
    # level 0 at the top -> invert
    y_of = lambda lv: 12.0 + (depth - lv) * row_h

    def lvlbox(x, y, ch, size, params, color, w=bw, show_params=True):
        _box(ax, x, y, w, bh, f"{ch} ch", "", color, fs=10)
        ax.text(x + w - 1.0, y, f"{size}×{size}", va="center", ha="right",
                fontsize=8.6, color="white", family="monospace")
        if show_params and params:
            ax.text(x + w / 2, y - bh / 2 - 2.0, f"{params:,}", ha="center",
                    fontsize=7.4, color="#777")

    # encoder
    for lv in range(depth):
        e = agg[("encoder", lv)]
        y = y_of(lv)
        lvlbox(xl, y, e["ch"], e["size"], e["params"], STAGE_COLOR["encoder"])
        ax.text(xl - 1.2, y, f"L{lv}", va="center", ha="right", fontsize=8, color="#999")
        if lv:
            _arrow(ax, (xl + bw / 2, y_of(lv - 1) - bh / 2), (xl + bw / 2, y + bh / 2))

    # bottleneck
    b = agg[("bottleneck", depth)]
    y_b = y_of(depth)
    xb = (xl + xr) / 2
    lvlbox(xb, y_b, b["ch"], b["size"], b["params"], STAGE_COLOR["bottleneck"])
    _arrow(ax, (xl + bw / 2, y_of(depth - 1) - bh / 2), (xb + bw / 2, y_b + bh / 2))
    ax.text(xb + bw / 2, y_b - bh / 2 - 5.0, "BOTTLENECK", ha="center", fontsize=8.6,
            color=STAGE_COLOR["bottleneck"], fontweight="bold")

    # decoder
    for lv in reversed(range(depth)):
        d = agg[("decoder", lv)]
        y = y_of(lv)
        lvlbox(xr, y, d["ch"], d["size"], d["params"], STAGE_COLOR["decoder"])
        src = (xb + bw / 2, y_b + bh / 2) if lv == depth - 1 \
            else (xr + bw / 2, y_of(lv + 1) + bh / 2)
        _arrow(ax, src, (xr + bw / 2, y - bh / 2))

    # head
    h = agg[("head", 0)]
    y_h = y_of(0) + row_h * 0.74
    lvlbox(xr, y_h, cfg.num_classes, h["size"], 0, STAGE_COLOR["head"])
    _arrow(ax, (xr + bw / 2, y_of(0) + bh / 2), (xr + bw / 2, y_h - bh / 2))
    ax.text(xr + bw + 1.4, y_h, f"1×1 conv → softmax   ({h['params']:,} params)",
            va="center", ha="left", fontsize=8.4, color=STAGE_COLOR["head"])

    # gates on the skips
    for lv in range(depth):
        y = y_of(lv)
        g = agg[("gate", lv)]
        _box(ax, xg - 3.6, y, 7.2, bh * 0.80, "AG", "", STAGE_COLOR["gate"], fs=9)
        ax.text(xg, y - bh * 0.40 - 2.0, f"{g['params']:,}", ha="center",
                fontsize=7.2, color="#999")
        _arrow(ax, (xl + bw, y), (xg - 3.8, y), color=STAGE_COLOR["gate"], ls="--")
        _arrow(ax, (xg + 3.8, y), (xr, y), color=STAGE_COLOR["gate"], ls="--")
        src = (xb + bw / 2, y_b + bh / 2 + 0.2) if lv == depth - 1 \
            else (xr, y_of(lv + 1))
        _arrow(ax, src, (xg, y - bh * 0.42), color="#9AA5B1", ls=":", lw=1.0, rad=0.22)

    y_top = y_h + bh / 2 + 4.5
    ax.text(xl + bw / 2, y_top, "ENCODER   (what)", ha="center", fontsize=11.5,
            color=STAGE_COLOR["encoder"], fontweight="bold")
    ax.text(xg, y_top, "GATES", ha="center", fontsize=10.5,
            color=STAGE_COLOR["gate"], fontweight="bold")
    ax.text(xr + bw / 2, y_top, "DECODER   (where)", ha="center", fontsize=11.5,
            color=STAGE_COLOR["decoder"], fontweight="bold")

    ax.plot([], [], color=STAGE_COLOR["gate"], ls="--", lw=1.3,
            label="skip connection, filtered by its gate")
    ax.plot([], [], color="#9AA5B1", ls=":", lw=1.3,
            label="gating signal g, from one level deeper")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2,
              frameon=False, fontsize=9)

    ax.text(50, 99, f"A.  Overview  —  {total:,} parameters, "
                    f"{depth} attention gates ({gate_params:,}, "
                    f"{100*gate_params/total:.1f}%)",
            ha="center", va="top", fontsize=12.5, fontweight="bold")


# --------------------------------------------------------------------------
# panel B - DoubleConv
# --------------------------------------------------------------------------

def _doubleconv_panel(ax, cfg, input_size):
    f = [cfg.base_filters * (2 ** i) for i in range(cfg.depth + 1)]
    lv = cfg.depth - 1
    ch, sz = f[lv], input_size[-1] // (2 ** lv)

    _header(ax, "B.  DoubleConv  —  the unit every block is built from",
            "Two 3×3 convolutions cover the same receptive field as one 5×5, with fewer weights and an extra "
            "ReLU between them.\nPadding=1 preserves the spatial size; only the channel count changes. Conv "
            "carries no bias, because BatchNorm removes the mean anyway.")

    y, w, h, gap = 40, 12.6, 22, 2.4
    ax.text(0.5, y + 20, f"in\n{ch}×{sz}×{sz}", ha="left", fontsize=8.6,
            color="#333", family="monospace", fontweight="bold")
    seq = [("Conv 3×3", f"{ch}→{ch}", "#4A90D9"), ("BatchNorm", f"{ch} ch", "#7FB069"),
           ("ReLU", "max(0,x)", "#F5A623"), ("Conv 3×3", f"{ch}→{ch}", "#4A90D9"),
           ("BatchNorm", f"{ch} ch", "#7FB069"), ("ReLU", "max(0,x)", "#F5A623")]
    x0 = 6
    for i, (lab, sub, col) in enumerate(seq):
        x = x0 + i * (w + gap)
        _box(ax, x, y, w, h, lab, sub, col, fs=8.6)
        _arrow(ax, (x - gap, y), (x, y))
    xe = x0 + (len(seq) - 1) * (w + gap) + w
    _arrow(ax, (xe, y), (xe + 4, y))
    ax.text(xe + 5, y, f"out\n{ch}×{sz}×{sz}", ha="left", va="center", fontsize=8.6,
            color="#333", family="monospace", fontweight="bold")


# --------------------------------------------------------------------------
# panel C - the attention gate
# --------------------------------------------------------------------------

def _gate_panel(ax, cfg, input_size):
    f = [cfg.base_filters * (2 ** i) for i in range(cfg.depth + 1)]
    lv = cfg.depth - 1
    g_ch, x_ch = f[lv + 1], f[lv]
    inter = max(x_ch // 2, 1)
    size = input_size[-1]
    g_sz, x_sz = size // (2 ** (lv + 1)), size // (2 ** lv)

    _header(ax, "C.  Attention Gate  —  what makes this an *Attention* U-Net",
            "α = sigmoid( ψ( ReLU( W_g(g) + W_x(x) ) ) )        output = x ⊙ α\n"
            "g is the coarser decoder signal — the query, “what am I looking for”. x is the skip being filtered, "
            "“what is available here”.\nψ squeezes to ONE channel, so α is a spatial map in 0..1 that broadcasts "
            "over every channel: the gate chooses *where* to look, not which feature to keep.")

    yg, yx, ym = 62, 30, 46
    w, h = 11.5, 13

    ax.text(0.5, yg, f"g\n{g_ch}×{g_sz}×{g_sz}", ha="left", va="center", fontsize=8.6,
            color=STAGE_COLOR["bottleneck"], family="monospace", fontweight="bold")
    _box(ax, 8, yg, w, h, "upsample", f"→{x_sz}×{x_sz}", "#9B9B9B", fs=8.4)
    _box(ax, 23, yg, w, h, "W_g 1×1", f"{g_ch}→{inter}", "#4A90D9", fs=8.4)
    _box(ax, 38, yg, w, h, "BatchNorm", f"{inter} ch", "#7FB069", fs=8.4)
    for a, b in [(6.6, 8), (8 + w, 23), (23 + w, 38)]:
        _arrow(ax, (a, yg), (b, yg))

    ax.text(0.5, yx, f"x\n{x_ch}×{x_sz}×{x_sz}", ha="left", va="center", fontsize=8.6,
            color=STAGE_COLOR["encoder"], family="monospace", fontweight="bold")
    _box(ax, 23, yx, w, h, "W_x 1×1", f"{x_ch}→{inter}", "#4A90D9", fs=8.4)
    _box(ax, 38, yx, w, h, "BatchNorm", f"{inter} ch", "#7FB069", fs=8.4)
    for a, b in [(6.6, 23), (23 + w, 38)]:
        _arrow(ax, (a, yx), (b, yx))

    _op(ax, 55, ym, "+")
    _arrow(ax, (38 + w, yg), (52.9, ym + 1.4), rad=-0.20)
    _arrow(ax, (38 + w, yx), (52.9, ym - 1.4), rad=0.20)

    _box(ax, 61, ym, w, h, "ReLU", "", "#F5A623", fs=8.4)
    _box(ax, 76, ym, w, h, "ψ  1×1", f"{inter}→1", "#8E44AD", fs=8.4)
    _arrow(ax, (57.1, ym), (61, ym))
    _arrow(ax, (61 + w, ym), (76, ym))

    _box(ax, 76, ym - 19, w, h, "BN + Sigmoid", "→ 0..1", "#8E44AD", fs=8.4)
    _arrow(ax, (76 + w / 2, ym - h / 2), (76 + w / 2, ym - 19 + h / 2))
    ax.text(89.5, ym - 19, f"α\n1×{x_sz}×{x_sz}", ha="left", va="center", fontsize=8.6,
            color="#8E44AD", family="monospace", fontweight="bold")

    _op(ax, 55, 8, "×", color="#C0392B")
    _arrow(ax, (76, ym - 19), (57.2, 8), color="#8E44AD", ls="--", rad=0.28,
           label="broadcast over all channels", dy=3.0)
    _arrow(ax, (6.6, yx - h / 2 - 1), (52.8, 8), color=STAGE_COLOR["encoder"],
           rad=-0.13, label="x, unchanged", dy=2.6)
    _arrow(ax, (57.2, 8), (67, 8), color="#C0392B")
    ax.text(68, 8, f"gated skip\n{x_ch}×{x_sz}×{x_sz}", ha="left", va="center",
            fontsize=8.6, color="#C0392B", family="monospace", fontweight="bold")


# --------------------------------------------------------------------------
# panel D - the decoder step
# --------------------------------------------------------------------------

def _up_panel(ax, cfg, input_size):
    f = [cfg.base_filters * (2 ** i) for i in range(cfg.depth + 1)]
    lv = cfg.depth - 1
    size = input_size[-1]
    g_sz, x_sz = size // (2 ** (lv + 1)), size // (2 ** lv)
    up_out = f[lv + 1] // 2
    cat = up_out + f[lv]

    _header(ax, "D.  Decoder step  —  upsample, concatenate the gated skip, convolve",
            "The transposed convolution doubles H and W while halving the channels, then the gated skip is "
            "concatenated along the channel axis.\nThat concatenation is what restores the spatial detail the "
            "encoder discarded — the reason a U-Net gives sharp boundaries and a bare encoder cannot.")

    y = 38
    ax.text(0.5, y + 17, f"from below\n{f[lv+1]}×{g_sz}×{g_sz}", ha="left", va="center",
            fontsize=8.4, color=STAGE_COLOR["decoder"], family="monospace",
            fontweight="bold")
    _box(ax, 14, y + 17, 17, 17, "ConvTranspose 2×2",
         f"stride 2 · {f[lv+1]}→{up_out} · ×2 size", STAGE_COLOR["decoder"], fs=8.4)
    _arrow(ax, (12.6, y + 17), (14, y + 17))

    ax.text(0.5, y - 19, f"gated skip\n{f[lv]}×{x_sz}×{x_sz}", ha="left", va="center",
            fontsize=8.4, color=STAGE_COLOR["gate"], family="monospace",
            fontweight="bold")
    _arrow(ax, (12.6, y - 19), (40, y - 4), color=STAGE_COLOR["gate"], ls="--", rad=0.10)

    _op(ax, 43, y, "‖")
    _arrow(ax, (31, y + 17), (41.2, y + 2), rad=-0.14)
    ax.text(43, y + 12, "concatenate on channels", ha="center", fontsize=8.2,
            color="#2C3E50", style="italic")
    ax.text(43, y - 12, f"{up_out} + {f[lv]} = {cat} ch", ha="center", fontsize=8.4,
            color="#2C3E50", family="monospace")

    _box(ax, 54, y, 22, 18, "DoubleConv", f"{cat}→{f[lv]}", "#4A90D9", fs=9.5)
    _arrow(ax, (45.1, y), (54, y))
    _arrow(ax, (76, y), (81, y))
    ax.text(82, y, f"out\n{f[lv]}×{x_sz}×{x_sz}", ha="left", va="center", fontsize=8.6,
            color="#333", family="monospace", fontweight="bold")


# --------------------------------------------------------------------------

def draw_diagram(records, cfg, input_size, out_path):
    """One stacked figure: overview on top, then the three block internals."""
    fig = plt.figure(figsize=(15.5, 27.0))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.55, 0.62, 1.32, 0.72], hspace=0.16,
                          left=0.03, right=0.985, top=0.955, bottom=0.015)

    total = sum(r["params"] for r in records)
    fig.suptitle(f"Attention U-Net  —  {cfg.num_classes} classes "
                 f"({', '.join(cfg.class_names)}),  "
                 f"{input_size[-1]}×{input_size[-1]}×{IN_CHANNELS} input,  "
                 f"{total:,} parameters\n"
                 f"base_filters={cfg.base_filters},  depth={cfg.depth},  "
                 f"batch_norm={cfg.batch_norm},  bilinear={cfg.bilinear}",
                 fontsize=14.5, fontweight="bold", y=0.988)

    _overview_panel(fig.add_subplot(gs[0]), records, cfg, input_size)
    _doubleconv_panel(fig.add_subplot(gs[1]), cfg, input_size)
    _gate_panel(fig.add_subplot(gs[2]), cfg, input_size)
    _up_panel(fig.add_subplot(gs[3]), cfg, input_size)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {out_path}")


def main():
    d = Config()
    p = argparse.ArgumentParser(description="Attention U-Net architecture docs")
    p.add_argument("--base-filters", type=int, default=d.base_filters)
    p.add_argument("--depth", type=int, default=d.depth)
    p.add_argument("--size", type=int, default=d.image_size)
    p.add_argument("--liver-only", action="store_true")
    p.add_argument("--bilinear", action="store_true")
    p.add_argument("--arch", choices=ARCH_NAMES, default=d.arch,
                   help="attention_unet (default) or unet, the ablation baseline")
    a = p.parse_args()

    cfg = Config(arch=a.arch, base_filters=a.base_filters, depth=a.depth,
                 liver_only=a.liver_only, bilinear=a.bilinear, image_size=a.size)
    input_size = (1, IN_CHANNELS, a.size, a.size)

    model = build_model(cfg)
    records = tag_stages(collect_layers(model, input_size), cfg.depth)

    out = Path(ARCH_DIR)
    out.mkdir(parents=True, exist_ok=True)
    total = print_table(records, cfg, input_size, out / f"architecture_{cfg.arch}.txt")
    write_html(records, cfg, input_size, out / f"architecture_{cfg.arch}.html")

    # the diagram is built from an analytic plan, not from these hooks, so the
    # two are independent derivations of the same network - check they agree
    from src.utils.diagram import plan, draw
    P = plan(base_filters=cfg.base_filters, depth=cfg.depth,
             num_classes=cfg.num_classes, image_size=cfg.image_size,
             in_channels=IN_CHANNELS, batch_norm=cfg.batch_norm,
             bilinear=cfg.bilinear, arch=cfg.arch)
    assert P["total"] == model.count_parameters(), \
        f"diagram plan {P['total']:,} != model {model.count_parameters():,}"
    draw(P, out / f"{cfg.arch}_diagram.png")

    assert total == model.count_parameters(), \
        f"hook total {total} != model total {model.count_parameters()}"
    print(f"[check] {cfg.arch}: hooks, diagram plan and model all agree: "
          f"{total:,} parameters")


if __name__ == "__main__":
    main()
