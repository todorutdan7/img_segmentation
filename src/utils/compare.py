"""Compare the runs in a group side by side.

    python -m src.utils.compare --group nifti/baseline
    python -m src.utils.compare --group internship_png/ablation_2070

Reads every history.json / metrics_test.json under runs/<group>/ and writes,
for the nifti runs into results/nifti/comparisons/:

  <group>.txt          the table below
  <group>.png          the test scores of every run, side by side
  <group>_curves.png   the validation score per epoch, one panel per run
  <group>_loss.png     train and validation loss, one panel per run

(the finished internship runs keep results/internship_png/comparison_<group>.*).
The overview of every run together is  python -m src.utils.results.

Flags unequal training budgets loudly, because a run that stopped early has not
been given the same chance as one that did not, and comparing their best scores
is not an architecture comparison.
"""

import argparse
import json
from pathlib import Path

import numpy as np


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
                         best_epoch=H.get("best_epoch"), best_select=H.get("best_val"),
                         minutes=H.get("train_minutes"), params=H.get("params"),
                         test=M))
    return runs


def batch(r):
    # batch x accumulation steps, e.g. 4x2 trains like 8
    b, k = r["cfg"].get("batch_size", 8), r["cfg"].get("accum_steps", 1)
    return f"{b}x{k}" if k > 1 else str(b)


def table(runs):
    L, a = [], lambda s: L.append(s)
    a("RUN COMPARISON")
    a("=" * 126)
    n_ep = [len(r["hist"]["val_dice_fg"]) for r in runs]
    matched = len(set(n_ep)) == 1

    a(f"{'run':<26}{'params':>11}{'slices':>7}{'batch':>7}{'lr':>8}{'epochs':>8}{'best@':>7}"
      f"{'val fgD':>9}{'test fgD':>10}{'liver':>8}{'tumor':>8}{'tum/pt':>8}{'min':>7}")
    a("-" * 126)
    for r in runs:
        t = r["test"]
        pc = t.get("per_class", {})
        # tumour Dice per patient (LiTS rules), written by reeval or train
        tpp = t.get("scores", {}).get("cleaned", {}).get("per_patient_mean", {}).get("tumor", float("nan"))
        a(f"{r['cfg'].get('arch', r['name'])[:24]:<26}{r['params']:>11,}"
          f"{r['cfg'].get('in_slices', 1):>7}{batch(r):>7}{r['cfg'].get('lr', 1e-3):>8g}"
          f"{len(r['hist']['val_dice_fg']):>8}{r['best_epoch']:>7}{r['best']:>9.4f}"
          f"{t.get('mean_dice_fg', float('nan')):>10.4f}"
          f"{pc.get('liver', {}).get('dice', float('nan')):>8.4f}"
          f"{pc.get('tumor', {}).get('dice', float('nan')):>8.4f}"
          f"{tpp:>8.4f}{r['minutes']:>7.0f}")
    a("-" * 126)

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
        a(f"All runs completed {n_ep[0]} epochs - budgets are matched.")
    sizes = {r["cfg"].get("batch_size", 8) * r["cfg"].get("accum_steps", 1) for r in runs}
    if len(sizes) > 1:
        a("!! The effective batch differs between runs (batch column); a smaller one")
        a("   at the same learning rate trains noisier, so that part is not like for like.")
    if len({r["cfg"].get("lr", 1e-3) for r in runs}) > 1:
        a("!! The learning rate differs between runs (lr column).")

    a("")
    a("Caveats that apply either way:")
    a("  - one seed per architecture. Differences under ~0.02 fg Dice are not")
    a("    distinguishable from run-to-run noise without repeats.")
    a("  - 20 test volumes, and tumour burden varies a lot between patients, so the")
    a("    tumour column carries the most variance.")
    a("  - liver / tumor are pooled over all test pixels; tum/pt is tumour Dice per")
    a("    patient by LiTS rules after the 3D clean-up (run reeval to fill it for")
    a("    older runs). Say which one you quote.")
    return "\n".join(L)


def figures(group_dir, out_base, title):
    """three plain figures instead of one crowded one (see the module docstring)"""
    from src.utils import plotstyle as ps
    from src.utils.results import Run, small_multiples, fig_test_scores, curve
    ps.style()
    runs = [Run(h.parent) for h in sorted(group_dir.glob("*/history.json"))]
    fig_test_scores(runs, Path(f"{out_base}.png"), f"Test scores: {title}")
    # the curve that picked the best epoch: tumour Dice per patient, or the
    # foreground Dice for runs from before that rule
    metric = "val_tumor_case" if all(curve(r, "val_tumor_case") for r in runs) else "val_dice_fg"
    small_multiples(runs, Path(f"{out_base}_curves.png"), "dice", metric)
    small_multiples(runs, Path(f"{out_base}_loss.png"), "loss")


def main():
    from src.config import RUNS_DIR, results_dir

    p = argparse.ArgumentParser(description="compare the runs in a group")
    p.add_argument("--group", default="nifti/baseline",
                   help="folder under runs/, e.g. nifti/baseline or internship_png/ablation_2070")
    a = p.parse_args()

    gdir = Path(RUNS_DIR) / a.group
    if not gdir.exists():
        raise SystemExit(f"no such group: {gdir}")

    runs = load_runs(gdir)
    if len(runs) < 2:
        raise SystemExit(f"need at least 2 runs in {gdir}, found {len(runs)}")

    txt = table(runs)
    print(txt)

    name = a.group.rstrip("/").split("/")[-1]
    if a.group.split("/")[0] == "internship_png":
        base = results_dir("png") / f"comparison_{name}"
    else:
        base = results_dir("nifti", "comparisons") / name
    Path(f"{base}.txt").write_text(txt, encoding="utf-8")
    print(f"\n[saved] {base}.txt")
    figures(gdir, base, a.group)


if __name__ == "__main__":
    main()
