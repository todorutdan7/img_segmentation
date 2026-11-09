"""Sum up a k-fold experiment: every patient tested once, by the fold that held it out.

    python -m src.utils.kfold --group nifti/kfold

Reads every run under runs/<group>/ that was trained with --folds, groups them
by model (arch, input slices, lr, batch), and for each model prints one line
per fold and the result over all patients. With two or more models it also
compares them patient by patient, which is what k-fold is for: with ~20
patients in one test set, a gap of 0.05 can come from one or two patients.

Scores are tumour (and liver) Dice per patient by LiTS rules, after the 3D
clean-up, as written by train or reeval into metrics_test.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_folds(group_dir: Path):
    models = {}
    for d in sorted(group_dir.iterdir()):
        h, m = d / "history.json", d / "metrics_test.json"
        if not (h.exists() and m.exists()):
            continue
        H, M = json.loads(h.read_text()), json.loads(m.read_text())
        c = H["config"]
        if not c.get("folds") or "scores" not in M:
            continue
        b, k = c.get("batch_size", 8), c.get("accum_steps", 1)
        batch = f"{b}x{k}" if k > 1 else str(b)
        label = (f"{c['arch']}  {c.get('in_slices', 1)} slice(s)  lr {c.get('lr', 1e-3):g}"
                 f"  batch {batch}")
        models.setdefault(label, {"folds": c["folds"], "runs": {}})
        models[label]["runs"][c["fold"]] = dict(name=d.name, best_epoch=H.get("best_epoch"),
                                                scores=M["scores"])
    return models


def summary(label, model):
    L, a = [], lambda s: L.append(s)
    k, runs = model["folds"], model["runs"]
    a(label)
    a(f"{'fold':>6}{'patients':>10}{'best@':>7}{'tum/pt':>9}{'median':>9}{'pooled':>9}"
      f"{'liver/pt':>10}   run")
    patients = {}
    for f in sorted(runs):
        s = runs[f]["scores"]["cleaned"]
        pp = s["per_patient"]
        patients.update({v: d for v, d in pp.items()})
        a(f"{f:>6}{len(pp):>10}{runs[f]['best_epoch']:>7}"
          f"{s['per_patient_mean']['tumor']:>9.3f}{s['per_patient_median']['tumor']:>9.3f}"
          f"{s['pooled']['tumor']:>9.3f}{s['per_patient_mean']['liver']:>10.3f}"
          f"   {runs[f]['name']}")

    missing = sorted(set(range(k)) - set(runs))
    fold_means = [runs[f]["scores"]["cleaned"]["per_patient_mean"]["tumor"] for f in sorted(runs)]
    tum = np.array([d["tumor"] for d in patients.values()])
    liv = np.array([d["liver"] for d in patients.values()])
    a(f"  all {len(patients)} patients: tumour/pt mean {tum.mean():.3f}  median "
      f"{np.median(tum):.3f}   liver/pt mean {liv.mean():.3f}")
    a(f"  between folds: tumour/pt {np.mean(fold_means):.3f} +- {np.std(fold_means):.3f} "
      f"(min {min(fold_means):.3f}, max {max(fold_means):.3f})")
    if missing:
        a(f"  !! folds {missing} of {k} are missing - not every patient has been tested yet")
    return L, {v: d["tumor"] for v, d in patients.items()}


def paired(labels, tumour):
    # the same patients scored by two models: per-patient differences
    L, a = [], lambda s: L.append(s)
    a("PATIENT BY PATIENT (tumour Dice per patient, only patients both models tested)")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            A, B = tumour[labels[i]], tumour[labels[j]]
            common = sorted(set(A) & set(B))
            if not common:
                continue
            d = np.array([A[v] - B[v] for v in common])
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            a(f"  {labels[i]}")
            a(f"  minus {labels[j]}")
            a(f"    {len(common)} patients: mean difference {d.mean():+.3f} (+- {se:.3f} std error)"
              f"   better on {(d > 0.01).sum()}, worse on {(d < -0.01).sum()}, "
              f"same on {(abs(d) <= 0.01).sum()}")
            a(f"    a difference smaller than about 2 std errors ({2 * se:.3f}) is not a real one")
    return L


def main():
    from src.config import RUNS_DIR, results_dir

    p = argparse.ArgumentParser(description="sum up a k-fold experiment")
    p.add_argument("--group", default="nifti/kfold", help="folder under runs/, e.g. nifti/kfold")
    a = p.parse_args()

    gdir = Path(RUNS_DIR) / a.group
    if not gdir.exists():
        raise SystemExit(f"no such group: {gdir}")
    models = load_folds(gdir)
    if not models:
        raise SystemExit(f"no finished k-fold runs (trained with --folds) in {gdir}")

    lines, tumour = ["K-FOLD SUMMARY  (Dice per patient after the 3D clean-up)", "=" * 100], {}
    for label, model in models.items():
        L, tumour[label] = summary(label, model)
        lines += L + [""]
    if len(models) > 1:
        lines += paired(list(models), tumour)
    txt = "\n".join(lines)
    print(txt)

    out = results_dir("nifti", "kfold") / f"{a.group.rstrip('/').split('/')[-1]}.txt"
    out.write_text(txt, encoding="utf-8")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
