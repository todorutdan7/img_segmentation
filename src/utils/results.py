"""Everything trained so far, in one place: the tables and figures for the thesis.

    python -m src.utils.results

Reads every run under runs/nifti/ (history.json + metrics_test.json), the
Grad-CAM tables in results/nifti/gradcam/ and each patient's tumour volume from
the slice cache index, and writes:

  results/nifti/README.md         the overview: every table and figure, and what each folder holds
  results/nifti/summary/
      runs.csv                    one row per run, every metric
      kfold_patients.csv          k-fold: each patient's tumour Dice under each model
      fixed_split.png             every model on the same 20 test patients
      training_curves.png         validation tumour Dice per epoch, one panel per model
      loss_curves.png             train and validation loss, one panel per model
      kfold_folds.png             k-fold: the score of each fold
      kfold_by_size.png           k-fold: the score by total tumour volume
      kfold_patients.png          k-fold: patient by patient, one model against the other
      gradcam.png                 how much more Grad-CAM heat falls on the tumour than its area

Run it again after every new run: nothing here is typed by hand except the
short notes in SUPERSEDED below. All tumour and liver scores are after the 3D
clean-up; "per patient" is the LiTS-style mean of each patient's Dice.
"""

import csv
import json
import re
from pathlib import Path

import numpy as np

from src.utils import plotstyle as ps

# runs kept out of the main table, with the reason (they stay in runs.csv)
SUPERSEDED = {
    "unet_2d_old-epoch-rule": "best epoch picked by foreground Dice; retrained with the tumour rule",
    "attention_unet_2d_old-epoch-rule": "best epoch picked by foreground Dice; retrained with the tumour rule",
    "unetpp_r50_2.5d_batch4": "lr 1e-3 too high: tumour Dice collapsed to 0 in several epochs",
    "unetpp_r50_2.5d_batch4x2": "lr 1e-3 with batch 4x2: still collapsed; lr 1e-4 fixed it",
}
SIZE_BINS = [("no tumour", 0, 0), ("< 10 mL", 0, 10), ("10-100 mL", 10, 100), (">= 100 mL", 100, 1e9)]
NA = "-"


# ---------------------------------------------------------------- loading

class Run:
    def __init__(self, d: Path):
        self.dir, self.name, self.group = d, d.name, d.parent.name
        H = json.loads((d / "history.json").read_text())
        mp = d / "metrics_test.json"
        self.M = json.loads(mp.read_text()) if mp.exists() else {}
        self.cfg, self.hist = H["config"], H["history"]
        self.select = H.get("select_metric")
        self.best_epoch = H.get("best_epoch")
        self.hours = (H.get("train_minutes") or self.M.get("train_minutes") or 0) / 60
        self.params = self.M.get("params") or H.get("params")
        self.folds, self.fold = self.cfg.get("folds") or 0, self.cfg.get("fold", 0)
        self.label = ps.run_label(self.cfg, self.select)
        self.colour = ps.colour(self.cfg)
        self.note = SUPERSEDED.get(self.name, "")
        s = self.M.get("scores", {}).get("cleaned")
        self.per_patient = {k: v["tumor"] for k, v in s["per_patient"].items()} if s else {}
        pc = self.M.get("per_class", {})
        # every score after the 3D clean-up; runs never rescored only have the raw pooled Dice
        self.m = {
            "val_tumour_pt": self.val_curve()[self.best_epoch - 1] if self.val_curve() else None,
            "tumour_pt": s["per_patient_mean"]["tumor"] if s else None,
            "tumour_median": s["per_patient_median"]["tumor"] if s else None,
            "tumour_pooled": s["pooled"]["tumor"] if s else pc.get("tumor", {}).get("dice"),
            "tumour_precision": s["pooled"]["tumor_precision"] if s else pc.get("tumor", {}).get("precision"),
            "tumour_recall": s["pooled"]["tumor_recall"] if s else pc.get("tumor", {}).get("recall"),
            "liver_pt": s["per_patient_mean"]["liver"] if s else None,
            "liver_pooled": s["pooled"]["liver"] if s else pc.get("liver", {}).get("dice"),
        }

    def val_curve(self):
        v = self.hist.get("val_tumor_case")
        return v if v and all(x is not None for x in v) else None

    @property
    def key(self):
        # one model = one configuration; its k-fold runs share this
        c = self.cfg
        return (c["arch"], c.get("in_slices", 1), c.get("lr", 1e-3),
                c.get("batch_size", 8), c.get("accum_steps", 1))


def load_runs(runs_dir: Path):
    return [Run(h.parent) for h in sorted(runs_dir.glob("*/*/history.json"))]


def tumour_volumes():
    """patient -> total tumour volume in mL, from any slice cache index"""
    from src.config import DATA_DIR
    for p in sorted(Path(DATA_DIR).glob("nifti_cache_*/index.json")):
        vols = json.loads(p.read_text())["volumes"]
        return {k: e["tumor_ml"] for k, e in vols.items() if "tumor_ml" in e}
    return {}


def gradcam_tables(gdir: Path, run_name: str):
    """per-patient heat share on the tumour per layer, and the mean liver line,
    parsed from the text table gradcam.py writes (fixed-width columns)"""
    f = gdir / f"{run_name}.txt"
    if not f.exists():
        return None
    lines = f.read_text(encoding="utf-8").splitlines()
    head = next(l for l in lines if l.startswith("patient"))
    n_layers = (len(head) - 31) // 24
    rows, liver = [], None
    for l in lines:
        if l.startswith("volume-"):
            cells = [l[31 + 24 * i: 31 + 24 * (i + 1)].strip().rstrip("%") for i in range(n_layers)]
            rows.append(dict(area=float(l[18:30]), shares=[float(c) for c in cells],
                             missed=l.rstrip().endswith("missed")))
        elif l.startswith("mean, liver"):
            vals = re.findall(r"(nan|[\d.]+)%", l)
            liver = dict(area=float(vals[0]), shares=[float(v) for v in vals[1:]])
    return dict(rows=rows, liver=liver, n_layers=n_layers)


# ---------------------------------------------------------------- numbers

def paired(a: dict, b: dict):
    """mean of a - b over the patients both scored, its std error, and the counts"""
    common = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in common])
    if len(d) < 2:
        return None
    return dict(n=len(d), mean=d.mean(), se=d.std(ddof=1) / np.sqrt(len(d)),
                better=int((d > 0.01).sum()), worse=int((d < -0.01).sum()),
                same=int((abs(d) <= 0.01).sum()))


def kfold_models(runs):
    models = {}
    for r in runs:
        if r.folds and r.per_patient:
            models.setdefault(r.key, []).append(r)
    out = []
    for key, rs in models.items():
        rs.sort(key=lambda r: r.fold)
        pp = {k: v for r in rs for k, v in r.per_patient.items()}
        liver = {k: v["liver"] for r in rs
                 for k, v in r.M["scores"]["cleaned"]["per_patient"].items()}
        fold_means = [r.m["tumour_pt"] for r in rs]
        out.append(dict(label=rs[0].label, colour=rs[0].colour, runs=rs, per_patient=pp,
                        folds=rs[0].folds, fold_means=fold_means,
                        mean=np.mean(list(pp.values())), median=np.median(list(pp.values())),
                        liver=np.mean(list(liver.values())), hours=sum(r.hours for r in rs)))
    return sorted(out, key=lambda m: -m["mean"])


# ---------------------------------------------------------------- tables

def f3(x):
    return NA if x is None else f"{x:.3f}"


def md_table(header, rows, align=None):
    align = align or ["l"] + ["r"] * (len(header) - 1)
    sep = {"l": ":---", "r": "---:", "c": ":---:"}
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep[a] for a in align) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fixed_split_table(main):
    best = max(main, key=lambda r: r.m["tumour_pt"] or -1)
    rows = []
    for r in main:
        p = paired(r.per_patient, best.per_patient) if r is not best else None
        delta = "best" if r is best else (f"{p['mean']:+.3f} ± {p['se']:.3f}" if p else NA)
        tpp = f3(r.m["tumour_pt"])
        rows.append([r.label, "ResNet-50 (ImageNet)" if ps.pretrained(r.cfg) else "scratch",
                     f"{r.params / 1e6:.1f} M", f"{r.hours:.1f}", r.best_epoch,
                     f3(r.m["val_tumour_pt"]), f"**{tpp}**" if r is best else tpp,
                     f3(r.m["tumour_median"]), f3(r.m["tumour_pooled"]),
                     f3(r.m["tumour_precision"]), f3(r.m["tumour_recall"]),
                     f3(r.m["liver_pt"]), delta])
    head = ["Model", "Encoder", "Params", "Hours", "Best epoch", "Val tumour/pt",
            "**Test tumour/pt**", "Median", "Pooled", "Precision", "Recall", "Liver/pt",
            "vs best (paired)"]
    return md_table(head, rows, ["l", "l"] + ["r"] * 11), best


def other_runs_table(runs):
    rows = [[r.label, f"`{r.group}/{r.name}`", f3(r.m["tumour_pt"]), f3(r.m["tumour_pooled"]),
             r.note] for r in runs]
    return md_table(["Model", "Run", "Test tumour/pt", "Pooled", "Why it is not in the main table"],
                    rows, ["l", "l", "r", "r", "l"])


def kfold_tables(models, vols):
    L = []
    rows = [[m["label"], m["folds"], len(m["per_patient"]), f"**{m['mean']:.3f}**",
             f"{m['median']:.3f}", f"{m['liver']:.3f}",
             f"{np.mean(m['fold_means']):.3f} ± {np.std(m['fold_means']):.3f}",
             f"{min(m['fold_means']):.3f} - {max(m['fold_means']):.3f}", f"{m['hours']:.0f}"]
            for m in models]
    L.append(md_table(["Model", "Folds", "Patients", "**Tumour/pt**", "Median", "Liver/pt",
                       "Folds: mean ± std", "Worst - best fold", "GPU hours"], rows))
    if len(models) < 2:
        return "\n".join(L)

    A = models[0]
    L += ["", "**Patient by patient.** The same patients, each tested once by the fold that held "
              "it out, scored by both models:", ""]
    for B in models[1:]:
        p = paired(A["per_patient"], B["per_patient"])
        verdict = "a real difference" if abs(p["mean"]) > 2 * p["se"] else "within noise"
        L.append(f"- {A['label']} minus {B['label']}: **{p['mean']:+.3f} ± {p['se']:.3f}** "
                 f"(std error, {p['n']} patients) - {verdict}. Better on {p['better']} patients, "
                 f"worse on {p['worse']}, the same on {p['same']}.")

    # per fold
    L += ["", "**Per fold:**", ""]
    head = ["Fold", "Patients"] + [m["label"] for m in models] + [f"difference (first - {m['label']})"
                                                                     for m in models[1:]]
    rows = []
    by_fold = [{r.fold: r for r in m["runs"]} for m in models]
    for ra in A["runs"]:
        f = ra.fold
        row = [f, len(ra.per_patient)] + [f3(bf[f].m["tumour_pt"]) if f in bf else NA for bf in by_fold]
        for bf in by_fold[1:]:
            p = paired(ra.per_patient, bf[f].per_patient) if f in bf else None
            row.append(f"{p['mean']:+.3f} ± {p['se']:.3f}" if p else NA)
        rows.append(row)
    L.append(md_table(head, rows))

    # by tumour size
    if vols:
        L += ["", "**By total tumour volume** (tumour Dice per patient; in brackets, patients "
                  "scored 0):", ""]
        head = ["Tumour volume", "Patients"] + [m["label"] for m in models] + \
               [f"difference (first - {m['label']})" for m in models[1:]]
        rows = []
        for name, lo, hi in SIZE_BINS:
            keys = [k for k in A["per_patient"] if k in vols and
                    (vols[k] == 0 if hi == 0 else (lo <= vols[k] < hi and vols[k] > 0))]
            row = [name, len(keys)]
            for m in models:
                v = np.array([m["per_patient"][k] for k in keys if k in m["per_patient"]])
                row.append(f"{v.mean():.3f} ({(v == 0).sum()})" if len(v) else NA)
            for B in models[1:]:
                p = paired({k: A["per_patient"][k] for k in keys}, B["per_patient"])
                row.append(f"{p['mean']:+.3f} ± {p['se']:.3f}" if p else NA)
            rows.append(row)
        L.append(md_table(head, rows))
    return "\n".join(L)


def gradcam_table(models, gdir):
    rows, notes = [], []
    for m in models:
        tabs = [t for t in (gradcam_tables(gdir, r.name) for r in m["runs"]) if t]
        if not tabs:
            continue
        pr = [p for t in tabs for p in t["rows"]]
        area = np.mean([p["area"] for p in pr])
        n_layers = tabs[0]["n_layers"]
        cells = []
        for i, layer in zip(range(n_layers), ("deep encoder", "bottleneck", "decoder")):
            share = np.nanmean([p["shares"][i] for p in pr])
            cells.append(f"{share:.1f}% ({share / area:.1f}x)")
            empty = sum(np.isnan(p["shares"][i]) for p in pr)
            if empty:
                notes.append(f"{m['label']}, {layer}: {empty} of {len(pr)} maps came out empty "
                             f"(the weighted sum was negative everywhere, so the ReLU left nothing) "
                             f"and are left out of its mean.")
        w = np.array([len(t["rows"]) for t in tabs])
        l_area = np.average([t["liver"]["area"] for t in tabs], weights=w)
        l_cells = [f"{np.average([t['liver']['shares'][i] for t in tabs], weights=w):.0f}%"
                   for i in range(n_layers)]
        missed = sum(p["missed"] for p in pr)
        rows.append([m["label"], len(pr), f"{area:.1f}%"] + cells +
                    [f"{l_area:.0f}% -> " + " / ".join(l_cells), f"{missed} of {len(pr)}"])
    if not rows:
        return None
    t = md_table(["Model", "Slices", "Tumour area", "Heat on tumour: deep encoder",
                  "bottleneck", "decoder", "Liver: area -> heat per layer", "Tumour missed"], rows)
    return t + "".join(f"\n\n_Note: {n}_" for n in notes)


# ---------------------------------------------------------------- figures

def fig_fixed_split(main, path):
    plt = ps.plt
    rs = sorted(main, key=lambda r: r.m["tumour_pt"])
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(rs) + 1.6))
    for y, r in enumerate(rs):
        t, v = r.m["tumour_pt"], r.m["val_tumour_pt"]
        if v is not None:
            ax.plot([min(t, v), max(t, v)], [y, y], color=ps.CONTEXT, lw=2, zorder=1)
            ax.scatter(v, y, s=70, facecolor="white", edgecolor=r.colour, lw=2, zorder=3)
        ax.scatter(t, y, s=80, color=r.colour, edgecolor="white", lw=1.5, zorder=4)
        # the test value sits next to its own (filled) dot, on the side away from validation
        left = v is not None and v > t
        ax.text(t + (-0.012 if left else 0.012), y, f"{t:.3f}", va="center",
                ha="right" if left else "left", fontsize=9, color=ps.INK2)
    ax.set_yticks(range(len(rs)), [r.label for r in rs])
    ax.set_xlabel("tumour Dice per patient (mean over the 20 patients)")
    lo = min(min(r.m["tumour_pt"], r.m["val_tumour_pt"] or 1) for r in rs)
    hi = max(max(r.m["tumour_pt"], r.m["val_tumour_pt"] or 0) for r in rs)
    ax.set_xlim(lo - 0.06, hi + 0.04)
    ax.set_ylim(-0.7, len(rs) - 0.3)
    ps.grid(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Fixed split: every model on the same 20 test patients", loc="left")
    H = [plt.Line2D([], [], marker="o", ls="", color=ps.BLUE, ms=8, label="ResNet-50 encoder (ImageNet)"),
         plt.Line2D([], [], marker="o", ls="", color=ps.ORANGE, ms=8, label="trained from scratch"),
         plt.Line2D([], [], marker="o", ls="", color=ps.MUTED, ms=8, label="test (number shown)"),
         plt.Line2D([], [], marker="o", ls="", mfc="white", mec=ps.MUTED, mew=2, ms=8,
                    label="validation, at the best epoch")]
    # legend and note at a fixed distance under the axis, whatever the number of rows
    ax_h = fig.get_size_inches()[1] * ax.get_position().height
    ax.legend(handles=H, loc="upper center", bbox_to_anchor=(0.4, -0.62 / ax_h), ncol=2)
    ax.text(0.4, -1.32 / ax_h, "Gaps under about 0.05 are within noise on 20 patients; "
            "the k-fold results settle the ranking.", transform=ax.transAxes, ha="center",
            fontsize=8.5, color=ps.INK2)
    ps.save(fig, path)


def curve(r, metric):
    """a run's validation curve for metric, or None if it was not recorded"""
    v = r.hist.get(metric)
    return v if v and all(x is not None for x in v) else None


def small_multiples(runs, path, kind, metric="val_tumor_case", title=None):
    """one panel per run, all on the same axes: kind "dice" draws the validation
    metric (the other runs in grey behind it), kind "loss" train and validation loss"""
    plt = ps.plt
    n = len(runs)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.7 * rows + 0.6),
                             sharex=True, sharey=True, squeeze=False)
    for ax, r in zip(axes.ravel(), runs):
        ep = np.arange(1, len(r.hist["train_loss"]) + 1)
        if kind == "dice":
            for o in runs:
                if o is not r and curve(o, metric):
                    ax.plot(np.arange(1, len(curve(o, metric)) + 1), curve(o, metric),
                            color=ps.CONTEXT, lw=1, zorder=1)
            c = curve(r, metric)
            if c:
                ax.plot(np.arange(1, len(c) + 1), c, color=r.colour, lw=2, zorder=3)
                b = r.best_epoch
                ax.scatter(b, c[b - 1], s=55, color=r.colour, edgecolor="white", lw=1.5, zorder=4)
                ax.annotate(f"best: epoch {b}", (b, c[b - 1]), xytext=(0, 8),
                            textcoords="offset points", ha="center", fontsize=8, color=ps.INK2)
            ax.set_ylim(0, 0.85 if metric == "val_tumor_case" else 1.0)
        else:
            ax.plot(ep, r.hist["train_loss"], color=r.colour, lw=2)
            ax.plot(ep, r.hist["val_loss"], color=r.colour, lw=1.6, ls=(0, (3, 2)))
        ax.set_title(r.label, fontsize=9.5, loc="left")
        ps.grid(ax)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    # the epoch axis under the lowest panel of each column
    for c in range(cols):
        low = axes[(n - 1 - c) // cols if c > (n - 1) % cols else rows - 1, c]
        low.tick_params(labelbottom=True)
        low.set_xlabel("epoch")
    what = {"val_tumor_case": "validation tumour Dice per patient",
            "val_dice_fg": "validation foreground Dice"}.get(metric, metric)
    if kind == "dice":
        fig.supylabel(what, fontsize=10, color=ps.INK2)
        default = f"{what[0].upper() + what[1:]} during training (grey: the other models)"
    else:
        fig.supylabel("loss (Dice + cross-entropy)", fontsize=10, color=ps.INK2)
        default = "Loss during training: solid = train, dashed = validation"
    fig.suptitle(title or default, x=0.01, ha="left", fontweight="bold", fontsize=11.5)
    fig.tight_layout()
    ps.save(fig, path)


def fig_test_scores(runs, path, title):
    """the test scores of a few runs side by side: one panel per score, one row per run"""
    plt = ps.plt
    scores = [("tumour_pt", "tumour Dice per patient"), ("tumour_pooled", "tumour Dice, pooled"),
              ("liver_pooled", "liver Dice, pooled")]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 0.45 * len(runs) + 1.5), sharey=True)
    for ax, (key, name) in zip(axes, scores):
        vals = [r.m[key] for r in runs if r.m[key] is not None]
        for y, r in enumerate(runs):
            v = r.m[key]
            if v is None:
                ax.text(min(vals) if vals else 0.5, y, "not scored", va="center", fontsize=8,
                        color=ps.MUTED)
                continue
            ax.scatter(v, y, s=80, color=r.colour, edgecolor="white", lw=1.5, zorder=3)
            ax.text(v, y + 0.28, f"{v:.3f}", ha="center", fontsize=8.5, color=ps.INK2)
        if vals:
            span = max(max(vals) - min(vals), 0.05)
            ax.set_xlim(min(vals) - 0.3 * span, max(vals) + 0.3 * span)
        ax.set_title(name, fontsize=10, loc="left")
        ps.grid(ax, "x")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(runs)), [r.label for r in runs])
    axes[0].set_ylim(-0.6, len(runs) - 0.3)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=11.5)
    H = [plt.Line2D([], [], marker="o", ls="", color=c, ms=8, label=t)
         for c, t in ((ps.BLUE, "ResNet-50 encoder (ImageNet)"), (ps.ORANGE, "trained from scratch"))
         if any(r.colour == c for r in runs)]
    fig.legend(handles=H, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    ps.save(fig, path)


def fig_kfold_folds(models, path):
    plt = ps.plt
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    k = models[0]["folds"]
    for m in models:
        f = [r.fold for r in m["runs"]]
        ax.plot(f, m["fold_means"], color=m["colour"], lw=1.4, alpha=0.5, zorder=2)
        ax.scatter(f, m["fold_means"], s=70, color=m["colour"], edgecolor="white", lw=1.5, zorder=3,
                   label=m["label"])
        ax.plot([-0.4, k - 0.6], [m["mean"]] * 2, color=m["colour"], lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.text(k - 0.5, m["mean"], f"all {len(m['per_patient'])} patients\n{m['mean']:.3f}",
                va="center", fontsize=8.5, color=ps.INK2)
    ax.set_xticks(range(k), [f"fold {i}" for i in range(k)])
    ax.set_xlim(-0.4, k + 0.6)
    ax.set_ylabel("tumour Dice per patient")
    ps.grid(ax)
    ax.set_title(f"{k}-fold cross-validation: each fold tests a different ~26 patients", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.12), ncol=len(models))
    ps.save(fig, path)


def fig_kfold_by_size(models, vols, path):
    plt = ps.plt
    A = models[0]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    w = 0.8 / len(models)
    labels = []
    for b, (name, lo, hi) in enumerate(SIZE_BINS):
        keys = [k for k in A["per_patient"] if k in vols and
                (vols[k] == 0 if hi == 0 else (lo <= vols[k] < hi and vols[k] > 0))]
        labels.append(f"{name}\n{len(keys)} patients")
        tops = []
        for i, m in enumerate(models):
            v = np.mean([m["per_patient"][k] for k in keys])
            x = b + (i - (len(models) - 1) / 2) * w
            ax.bar(x, v, w, color=m["colour"], edgecolor="white", lw=1.5,
                   label=m["label"] if b == 0 else None)
            tops.append(v)
        if len(models) > 1:
            p = paired({k: A["per_patient"][k] for k in keys}, models[1]["per_patient"])
            ax.text(b, max(tops) + 0.03, f"{p['mean']:+.3f}", ha="center", fontsize=9, color=ps.INK2)
    ax.set_xticks(range(len(SIZE_BINS)), labels)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("tumour Dice per patient")
    ps.grid(ax)
    ax.set_title("k-fold, by total tumour volume (number: first model minus second)", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=len(models))
    ps.save(fig, path)


def fig_kfold_patients(models, vols, path):
    plt = ps.plt
    A, B = models[0], models[1]
    keys = sorted(set(A["per_patient"]) & set(B["per_patient"]))
    fig, ax = plt.subplots(figsize=(5.6, 5.9))
    ax.plot([0, 1], [0, 1], color=ps.MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    for (name, lo, hi), c in zip(SIZE_BINS, [ps.SIZE_STEPS[0]] + ps.SIZE_STEPS[2:]):
        ks = [k for k in keys if k in vols and
              (vols[k] == 0 if hi == 0 else (lo <= vols[k] < hi and vols[k] > 0))]
        ax.scatter([B["per_patient"][k] for k in ks], [A["per_patient"][k] for k in ks], s=42,
                   color=c, edgecolor="white", lw=1, zorder=3, label=f"{name} ({len(ks)})")
    p = paired(A["per_patient"], B["per_patient"])
    ax.text(0.02, 0.97, f"above the line: {A['label']} better\n({p['better']} patients)",
            va="top", fontsize=8.5, color=ps.INK2)
    ax.text(0.98, 0.03, f"below: {B['label']} better\n({p['worse']} patients)",
            ha="right", fontsize=8.5, color=ps.INK2)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{B['label']}: tumour Dice")
    ax.set_ylabel(f"{A['label']}: tumour Dice")
    ps.grid(ax, "both")
    ax.set_title("k-fold, patient by patient", loc="left")
    ax.legend(title="total tumour volume", loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2,
              title_fontsize=9)
    ps.save(fig, path)


def fig_gradcam(models, gdir, path):
    plt = ps.plt
    data = []
    for m in models:
        pr = [p for r in m["runs"] for p in (gradcam_tables(gdir, r.name) or {"rows": []})["rows"]]
        if pr:
            area = np.mean([p["area"] for p in pr])
            data.append((m, [np.nanmean([p["shares"][i] for p in pr]) / area for i in range(3)]))
    if not data:
        return False
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    w = 0.8 / len(data)
    for i, (m, ratios) in enumerate(data):
        ax.bar(np.arange(3) + (i - (len(data) - 1) / 2) * w, ratios, w, color=m["colour"],
               edgecolor="white", lw=1.5, label=m["label"])
    ax.plot([-0.5, 2.5], [1, 1], color=ps.MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(2.55, 1, "1x: no more heat\nthan the tumour's\narea would get", fontsize=8,
            color=ps.INK2, ha="left", va="center")
    ax.set_xticks(range(3), ["deep encoder", "bottleneck", "decoder"])
    ax.set_xlim(-0.5, 3.35)
    ax.set_ylabel("heat on tumour / tumour area")
    ps.grid(ax)
    ax.set_title("Grad-CAM, tumour class: is the heat on the tumour?", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=len(data))
    ps.save(fig, path)
    return True


# ---------------------------------------------------------------- csv

def write_csv(runs, path):
    cols = ["group", "run", "model", "arch", "in_slices", "encoder", "lr", "batch", "params",
            "hours", "epochs", "best_epoch", "select_metric", "folds", "fold", "val_tumour_pt",
            "tumour_pt", "tumour_median", "tumour_pooled", "tumour_precision", "tumour_recall",
            "liver_pt", "liver_pooled", "note"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in runs:
            c = r.cfg
            b, k = c.get("batch_size", 8), c.get("accum_steps", 1)
            w.writerow([r.group, r.name, r.label, c["arch"], c.get("in_slices", 1),
                        "resnet50_imagenet" if ps.pretrained(c) else "scratch", c.get("lr"),
                        f"{b}x{k}", r.params, round(r.hours, 2), len(r.hist["train_loss"]),
                        r.best_epoch, r.select or "fg_dice", r.folds, r.fold if r.folds else ""]
                       + [("" if r.m[x] is None else round(r.m[x], 4)) for x in
                          ("val_tumour_pt", "tumour_pt", "tumour_median", "tumour_pooled",
                           "tumour_precision", "tumour_recall", "liver_pt", "liver_pooled")]
                       + [r.note])
    print(f"[saved] {path}")


def write_patients_csv(models, vols, path):
    keys = sorted({k for m in models for k in m["per_patient"]}, key=lambda k: int(k.split("-")[1]))
    fold = {k: r.fold for r in models[0]["runs"] for k in r.per_patient}
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["patient", "fold", "tumour_ml"] + [m["label"] for m in models])
        for k in keys:
            w.writerow([k, fold.get(k, ""), vols.get(k, "")] +
                       [round(m["per_patient"][k], 4) if k in m["per_patient"] else "" for m in models])
    print(f"[saved] {path}")


# ---------------------------------------------------------------- page

FOLDERS = [
    ("summary/", "this overview's tables (csv) and figures - `python -m src.utils.results`"),
    ("comparisons/", "one experiment at a time: training curves and test scores of its runs - "
                     "`python -m src.utils.compare --group nifti/<group>`"),
    ("kfold/", "k-fold summaries as text - `python -m src.utils.kfold --group nifti/kfold`"),
    ("gradcam/", "Grad-CAM heat maps and tables, one per run - "
                 "`python -m src.utils.gradcam --run <run>`"),
    ("dataset/", "the MSD Task03_Liver dataset: sizes, spacing, HU, tumour volumes - "
                 "`python -m src.utils.nifti_report`"),
    ("viewer/", "screenshots and exports of the 3D viewer - `python -m src.utils.viewer`, `view3d`"),
    ("attention/", "attention-gate maps of an Attention U-Net - `python -m src.utils.attention_viz`"),
    ("cases/", "single patients looked at closely (e.g. the dark tumour of patient 104)"),
    ("baseline/", "the first NIfTI baseline (Oct 2026) and the internship models run on the NIfTI "
                  "data - superseded, kept for the record"),
]


def page(runs, main, others, models, vols, gdir, figs, out):
    L, a = [], lambda *s: L.extend(s)
    table1, best = fixed_split_table(main)
    a("# Results: liver and tumour segmentation on MSD Task03_Liver (NIfTI, 512 x 512)", "",
      "_Generated by `python -m src.utils.results` from the run folders in `runs/nifti/`. "
      "Do not edit by hand - run it again after a new run._", "")

    a("## At a glance", "")
    if models:
        A = models[0]
        a(f"- **Best model: {A['label']}** - tumour Dice per patient **{A['mean']:.3f}** "
          f"(median {A['median']:.3f}) over all {len(A['per_patient'])} patients, "
          f"{A['folds']}-fold cross-validation; liver {A['liver']:.3f}.")
        for B in models[1:]:
            p = paired(A["per_patient"], B["per_patient"])
            a(f"- Against {B['label']} ({B['mean']:.3f}): {p['mean']:+.3f} ± {p['se']:.3f} "
              f"over the same patients, better on {p['better']} of them and worse on {p['worse']}.")
    gaps = [paired(r.per_patient, best.per_patient) for r in main if r is not best]
    noise = all(abs(p["mean"]) < 2 * p["se"] for p in gaps if p)
    a(f"- On the fixed 20-patient split the best was {best.label} ({best.m['tumour_pt']:.3f}); "
      + ("every gap to it is under 2 std errors, so the split alone cannot rank the models."
         if noise else "some gaps to it are over 2 std errors (see the table)."),
      f"- {len(runs)} training runs in total, {sum(r.hours for r in runs):.0f} GPU hours on the "
      f"RTX 2070.", "")
    a("**How to read the scores.** _Tumour/pt_ = tumour Dice of each patient, then the mean "
      "(LiTS rule: a patient with no tumour scores 1 if none is predicted, else 0). _Pooled_ = one "
      "Dice over all test pixels together, so big tumours dominate it. _Precision / recall_ are "
      "pooled. Every score is after the 3D clean-up (largest liver piece kept). Only slices that "
      "contain liver are evaluated.", "")

    a("## 1. Fixed split: every model on the same 20 test patients", "",
      "70 / 15 / 15 % of the 131 patients for train / validation / test, split by patient. "
      "The best epoch is the one with the highest validation tumour Dice per patient. "
      "_vs best_ is the paired difference to the best model on the same patients, ± its std error.",
      "", table1, "", f"![fixed split]({figs['fixed']})", "")
    a("<details><summary>Training curves and loss of each model</summary>", "",
      f"![training curves]({figs['curves']})", "", f"![loss curves]({figs['loss']})", "",
      "</details>", "")
    if others:
        a("**Runs left out of the table** (still in `summary/runs.csv`):", "", other_runs_table(others), "")

    if models:
        a(f"## 2. {models[0]['folds']}-fold cross-validation: every patient tested once", "",
          "The 131 patients are shuffled once and cut into 5 folds. Each run tests one fold, "
          "validates on 20 patients from the other four and trains on the rest (~85). Each "
          "patient is so tested exactly once, by a model that never saw it.", "",
          kfold_tables(models, vols), "")
        for key in ("folds", "size", "patients"):
            if figs.get(key):
                a(f"![k-fold {key}]({figs[key]})", "")
        gt = gradcam_table(models, gdir)
        if gt:
            a("## 3. Grad-CAM: does the model look at the tumour?", "",
              "Seg-Grad-CAM for the tumour class on each test patient's slice with the most "
              "tumour, at three depths of the network. The share of the heat that falls on the "
              "true tumour is compared with the share of the slice the tumour covers: 4x means "
              "four times more heat than its area alone would give.", "", gt, "")
            if figs.get("gradcam"):
                a(f"![grad-cam]({figs['gradcam']})", "")
            a("Heat maps for single patients: `gradcam/<run>.png`.", "")

    a("## Folders", "", md_table(["Folder", "What is in it"],
                                  [[f"`{f}`", d] for f, d in FOLDERS if (out / f).exists()],
                                  ["l", "l"]), "",
      "")

    # the models, with a link to each one's drawing
    adir = Path(__file__).resolve().parents[2] / "architecture" / "nifti"
    par, inputs = {}, {}
    for r in runs:
        k = (r.cfg.get("arch"), r.cfg.get("in_slices", 1))
        par.setdefault(k, r.params)
        inputs.setdefault(k[0], set()).add("2.5D" if k[1] > 1 else "2D")
    rows = []
    for arch, (name, r50) in ps.ARCH.items():
        png = adir / f"{arch}.png"
        if not png.exists():
            continue
        drawn = 3 if r50 else 1                 # the input each drawing shows
        p = par.get((arch, drawn))
        rows.append([f"{name}{' R50' if r50 else ''}",
                     "ResNet-50, ImageNet" if r50 else "from scratch",
                     ", ".join(sorted(inputs.get(arch, []), key=len)) or "-",
                     f"{p:,}" if p else "-",
                     f"[{png.name}](../../architecture/nifti/{png.name})"])
    if rows:
        a("## Models", "", "The drawings are in `architecture/nifti/` (the internship's 256 px ones "
          "in `architecture/internship_png/`). The from-scratch models are drawn with one slice "
          "in (2D), the ResNet-50 ones with three (2.5D); the parameter count is for that input "
          "(the first convolution grows by a few hundred weights per extra slice).", "",
          md_table(["Model", "Encoder", "Trained on NIfTI with", "Parameters (as drawn)", "Drawing"],
                   rows, ["l", "l", "l", "r", "l"]), "")
    return "\n".join(L)


def main():
    import argparse
    from src.config import RUNS_DIR, results_dir

    p = argparse.ArgumentParser(description="tables and figures of every NIfTI run")
    p.parse_args()
    ps.style()

    runs = load_runs(Path(RUNS_DIR) / "nifti")
    if not runs:
        raise SystemExit("no runs in runs/nifti")
    single = [r for r in runs if not r.folds]
    main_runs = [r for r in single if not r.note and r.m["tumour_pt"] is not None]
    others = [r for r in single if r not in main_runs]
    models = kfold_models(runs)
    vols = tumour_volumes()
    if not vols:
        print("[results] no slice cache index found - the tables by tumour volume are skipped")

    out = results_dir("nifti")
    sdir, gdir = results_dir("nifti", "summary"), out / "gradcam"
    write_csv(runs, sdir / "runs.csv")
    figs = {"fixed": "summary/fixed_split.png", "curves": "summary/training_curves.png",
            "loss": "summary/loss_curves.png"}
    fig_fixed_split(main_runs, out / figs["fixed"])
    order = sorted(main_runs, key=lambda r: -r.m["tumour_pt"])
    small_multiples(order, out / figs["curves"], "dice")
    small_multiples(order, out / figs["loss"], "loss")
    if models:
        write_patients_csv(models, vols, sdir / "kfold_patients.csv")
        figs["folds"] = "summary/kfold_folds.png"
        fig_kfold_folds(models, out / figs["folds"])
        if len(models) > 1 and vols:
            figs["size"], figs["patients"] = "summary/kfold_by_size.png", "summary/kfold_patients.png"
            fig_kfold_by_size(models, vols, out / figs["size"])
            fig_kfold_patients(models, vols, out / figs["patients"])
        if fig_gradcam(models, gdir, out / "summary/gradcam.png"):
            figs["gradcam"] = "summary/gradcam.png"

    (out / "README.md").write_text(page(runs, order, others, models, vols, gdir, figs, out),
                                   encoding="utf-8")
    print(f"[saved] {out / 'README.md'}")


if __name__ == "__main__":
    main()
