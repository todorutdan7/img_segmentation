"""Compare the runs in a group side by side.

    python -m src.utils.compare --group nifti/baseline
    python -m src.utils.compare --group internship_png/ablation_2070

Reads every history.json / metrics_test.json under runs/<group>/ and writes a
comparison figure and table. Flags unequal training budgets loudly, because a
run that stopped early has not been given the same chance as one that did not,
and comparing their best scores is not an architecture comparison.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#2E6DA4", "#C0392B", "#2E8B57", "#8E44AD", "#D9822B"]


def load_runs(group_dir: Path):
    runs = []
    for d in sorted(group_dir.iterdir()):
        h, m = d / "history.json", d / "metrics_test.json"
        if not h.exists():
            continue
        H = json.loads(h.read_text())
        M = json.loads(m.read_text()) if m.exists() else {}
        runs.append(dict(name=d.name, dir=d, cfg=H.get("config", {}),
                         hist=H["history"], best=H.get("best_val_dice_fg"),
                         best_epoch=H.get("best_epoch"),
                         minutes=H.get("train_minutes"), params=H.get("params"),
                         test=M))
    return runs


def table(runs):
    L, a = [], lambda s: L.append(s)
    a("RUN COMPARISON")
    a("=" * 96)
    n_ep = [len(r["hist"]["val_dice_fg"]) for r in runs]
    matched = len(set(n_ep)) == 1

    a(f"{'run':<26}{'params':>11}{'epochs':>8}{'best@':>7}{'val fgD':>9}"
      f"{'test fgD':>10}{'liver':>8}{'tumor':>8}{'min':>7}")
    a("-" * 96)
    for r in runs:
        t = r["test"]
        pc = t.get("per_class", {})
        a(f"{r['cfg'].get('arch', r['name'])[:24]:<26}{r['params']:>11,}"
          f"{len(r['hist']['val_dice_fg']):>8}{r['best_epoch']:>7}{r['best']:>9.4f}"
          f"{t.get('mean_dice_fg', float('nan')):>10.4f}"
          f"{pc.get('liver', {}).get('dice', float('nan')):>8.4f}"
          f"{pc.get('tumor', {}).get('dice', float('nan')):>8.4f}"
          f"{r['minutes']:>7.0f}")
    a("-" * 96)

    if not matched:
        n = min(n_ep)
        a("")
        a("!! UNEQUAL TRAINING BUDGETS - the headline numbers are NOT comparable.")
        a(f"   epochs run: {dict((r['cfg'].get('arch', r['name']), len(r['hist']['val_dice_fg'])) for r in runs)}")
        a(f"   A run stopped by early stopping was not given the same chance as one")
        a(f"   that ran to the epoch cap. Comparing their best scores compares budgets,")
        a(f"   not architectures. Matched to the shortest run ({n} epochs):")
        a("")
        a(f"   {'run':<26}{'mean val fgD':>14}{'best val fgD':>14}")
        for r in runs:
            v = r["hist"]["val_dice_fg"][:n]
            a(f"   {r['cfg'].get('arch', r['name'])[:24]:<26}"
              f"{np.mean(v):>14.4f}{max(v):>14.4f}")
        a("")
        a(f"   To settle it, rerun the short one with --early-stop-patience 0 so both")
        a(f"   get the full epoch budget, then compare again.")
    else:
        a("")
        a(f"Both runs completed {n_ep[0]} epochs - budgets are matched, the comparison is fair.")

    a("")
    a("Caveats that apply either way:")
    a("  - one seed per architecture. Differences under ~0.02 fg Dice are not")
    a("    distinguishable from run-to-run noise without repeats.")
    a("  - 20 test volumes, and tumour burden varies a lot between patients, so the")
    a("    tumour column carries the most variance.")
    a("  - Dice here is pooled over all test pixels, not averaged per volume. The")
    a("    LiTS challenge reports per-case; say which you used.")
    return "\n".join(L)


def figure(runs, out_path, dpi=110):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Ablation: plain U-Net vs Attention U-Net", fontsize=15,
                 fontweight="bold")

    for i, r in enumerate(runs):
        c = COLORS[i % len(COLORS)]
        h = r["hist"]
        ep = range(1, len(h["val_dice_fg"]) + 1)
        lbl = f"{r['cfg'].get('arch', r['name'])}  ({r['params']:,} par)"

        axes[0, 0].plot(ep, h["val_dice_fg"], color=c, lw=1.8, label=lbl)
        axes[0, 0].scatter([r["best_epoch"]], [r["best"]], color=c, zorder=5, s=45)

        axes[0, 1].plot(ep, h["train_loss"], color=c, lw=1.6, label=f"{lbl} train")
        axes[0, 1].plot(ep, h["val_loss"], color=c, lw=1.4, ls="--", alpha=.75,
                        label=f"{lbl} val")

        axes[1, 0].plot(ep, h["val_dice"], color=c, lw=1.8, label=lbl)

        t = r["test"].get("per_class", {})
        names = [k for k in ("background", "liver", "tumor") if k in t]
        x = np.arange(len(names)) + (i - (len(runs) - 1) / 2) * 0.35
        axes[1, 1].bar(x, [t[k]["dice"] for k in names], width=0.33, color=c, label=lbl)
        for xi, k in zip(x, names):
            axes[1, 1].text(xi, t[k]["dice"] + .015, f"{t[k]['dice']:.3f}",
                            ha="center", fontsize=8.5)

    n_ep = [len(r["hist"]["val_dice_fg"]) for r in runs]
    if len(set(n_ep)) > 1:
        axes[0, 0].axvline(min(n_ep), color="#7E5109", ls=":", lw=1.6)
        axes[0, 0].text(min(n_ep) - 0.5, axes[0, 0].get_ylim()[0] + 0.02,
                        f"shortest run ends ({min(n_ep)})  →  everything right of\n"
                        f"this line only one model was allowed to use",
                        fontsize=8, color="#7E5109", ha="right", style="italic")

    axes[0, 0].set_title("Validation foreground Dice  (the selection metric)",
                         fontweight="bold", fontsize=11)
    axes[0, 0].set_xlabel("epoch"); axes[0, 0].set_ylabel("mean fg Dice")
    axes[0, 1].set_title("Loss  (solid = train, dashed = val)", fontweight="bold", fontsize=11)
    axes[0, 1].set_xlabel("epoch"); axes[0, 1].set_ylabel("dice_ce loss")
    axes[1, 0].set_title("Validation mean Dice, all classes", fontweight="bold", fontsize=11)
    axes[1, 0].set_xlabel("epoch"); axes[1, 0].set_ylabel("mean Dice")
    axes[1, 1].set_title("Test Dice per class", fontweight="bold", fontsize=11)
    axes[1, 1].set_xticks(np.arange(len(names))); axes[1, 1].set_xticklabels(names)
    axes[1, 1].set_ylim(0, 1.08); axes[1, 1].set_ylabel("Dice")

    for ax in axes.ravel():
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"[saved] {out_path}")


def main():
    from src.config import RUNS_DIR, results_dir

    p = argparse.ArgumentParser(description="compare the runs in a group")
    p.add_argument("--group", default="nifti/baseline",
                   help="folder under runs/, e.g. nifti/baseline or internship_png/ablation_2070")
    p.add_argument("--dpi", type=int, default=110)
    a = p.parse_args()

    gdir = Path(RUNS_DIR) / a.group
    if not gdir.exists():
        raise SystemExit(f"no such group: {gdir}")

    runs = load_runs(gdir)
    if len(runs) < 2:
        raise SystemExit(f"need at least 2 runs in {gdir}, found {len(runs)}")

    txt = table(runs)
    print(txt)

    # results/nifti/comparison_baseline.* or results/internship_png/comparison_ablation_2070.*
    out = results_dir("png" if a.group.split("/")[0] == "internship_png" else "nifti")
    name = a.group.rstrip("/").split("/")[-1]
    (out / f"comparison_{name}.txt").write_text(txt, encoding="utf-8")
    print(f"\n[saved] {out / f'comparison_{name}.txt'}")
    figure(runs, out / f"comparison_{name}.png", a.dpi)


if __name__ == "__main__":
    main()
