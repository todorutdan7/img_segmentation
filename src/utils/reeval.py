"""Re-run the test evaluation on already-trained checkpoints.

    python -m src.utils.reeval --group nifti/baseline     # every run in runs/nifti/baseline
    python -m src.utils.reeval --ckpt runs/.../best_model.pt
    python -m src.utils.reeval --group internship_png/ablation_2070

Regenerates confusion matrices, the per-class Dice/IoU bars, predictions_test.png
and metrics_test.json inside each run folder - without retraining. Useful after
changing anything in evaluate.py, since those figures are otherwise only written
once, at the end of a training run. It also records Dice per patient, next to
the pooled number.

The config is read back out of each checkpoint, so the data format and the test
split are exactly the ones the model was trained with.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.config import get_device
from src.data import get_dataloaders, has_foreground as _has_fg, volume_id
from src.model import load_checkpoint
from src.evaluate import evaluate, per_volume_confusion, case_dice


def volume_keys(dataset, cfg):
    """the patient of every dataset item, in order"""
    if cfg.data_format == "nifti":
        return [k for k, _ in dataset.pairs]
    return [volume_id(ip) for ip, _ in dataset.pairs]


def per_case(model, loader, cfg, device):
    """per-patient Dice, as LiTS reports it next to the pooled number.
    Tumour is averaged over the patients whose truth or prediction has some."""
    cms = per_volume_confusion(model, loader, device, cfg.num_classes,
                               volume_keys(loader.dataset, cfg))
    per = {k: {c: (None if np.isnan(d) else round(float(d), 4))
               for c, d in zip(cfg.class_names, case_dice(cm))}
           for k, cm in sorted(cms.items(), key=lambda t: int(t[0].split("-")[1]))}
    mean = {c: float(np.nanmean([np.nan if v[c] is None else v[c] for v in per.values()]))
            for c in cfg.class_names[1:]}
    return per, mean


def run_one(ckpt: Path, device, num_workers=0, batch_size=None,
            keep_empty=False, verbose=True):
    model, cfg, ck = load_checkpoint(ckpt, device)

    # The checkpoint carries the config it was TRAINED with, including
    # machine-specific settings. A model trained on Linux with num_workers=4
    # must not drag that onto Windows, where worker shared-memory hits the
    # pagefile commit limit. These do not affect the numbers, only how the
    # data is fed.
    cfg = replace(cfg, num_workers=num_workers,
                  batch_size=batch_size or cfg.batch_size)

    # Scoring on EVERY slice, not only the ones containing liver. Training drops
    # empty slices, but at inference the model is shown the whole scan, so this
    # is the realistic number. The volume-level split is unaffected: all 131
    # patients contain foreground either way, so the same 20 land in test and
    # the two results stay directly comparable.
    split_name = "test"
    if keep_empty:
        cfg = replace(cfg, skip_empty=False)
        split_name = "test_allslices"

    out_dir = ckpt.parent
    print(f"\n=== {out_dir.name} ===")
    print(f"[load] arch={cfg.arch}  data={cfg.data_format}  epoch={ck.get('epoch')}  "
          f"val_dice_fg={ck.get('val_dice_fg', float('nan')):.4f}  "
          f"params={model.count_parameters():,}")

    _, _, test_loader, _ = get_dataloaders(cfg, verbose=False)
    ds = test_loader.dataset
    n = len(ds)
    if not keep_empty:
        empty = 0
    elif cfg.data_format == "nifti":
        empty = sum(1 for k, z in ds.pairs
                    if ds.volumes[k]["liver_px"][z] + ds.volumes[k]["tumor_px"][z] == 0)
    else:
        empty = sum(1 for _, mp in ds.pairs if not _has_fg(mp))
    print(f"[data] test slices: {n:,}"
          + (f"   ({empty:,} of them contain no liver at all)" if keep_empty else
             "   (foreground slices only)"))

    metrics = evaluate(model, test_loader, device, cfg.class_names, out_dir,
                       split=split_name, verbose=verbose)

    metrics["params"] = model.count_parameters()
    metrics["n_test_slices"] = n
    metrics["skip_empty"] = cfg.skip_empty
    metrics["data_format"] = cfg.data_format
    metrics["per_volume_dice"], metrics["mean_case_dice"] = per_case(model, test_loader, cfg, device)
    print("mean Dice per patient: " + "  ".join(
        f"{c} {d:.4f}" for c, d in metrics["mean_case_dice"].items()))
    # keep the fields compare.py expects, if a previous metrics file had them
    old = out_dir / f"metrics_{split_name}.json"
    if old.exists():
        try:
            prev = json.loads(old.read_text())
            for k in ("best_val_dice_fg", "train_minutes"):
                if k in prev:
                    metrics[k] = prev[k]
        except json.JSONDecodeError:
            pass
    old.write_text(json.dumps(metrics, indent=2))
    print(f"[saved] {old}")
    return metrics


def main():
    from src.config import RUNS_DIR

    p = argparse.ArgumentParser(description="re-evaluate trained checkpoints")
    p.add_argument("--group", default="nifti/baseline",
                   help="folder under runs/, e.g. nifti/baseline or internship_png/ablation_2070")
    p.add_argument("--ckpt", default=None, help="a single best_model.pt")
    p.add_argument("--num-workers", type=int, default=0,
                   help="0 on Windows; the checkpoint's own value is ignored "
                        "because it belongs to the machine that trained it")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--keep-empty", action="store_true",
                   help="score EVERY slice of the test patients, including the "
                        "~67%% with no liver. Writes metrics_test_allslices.json "
                        "so the foreground-only results are not overwritten.")
    a = p.parse_args()

    device = get_device()

    if a.ckpt:
        ckpts = [Path(a.ckpt)]
    else:
        ckpts = sorted((Path(RUNS_DIR) / a.group).glob("*/best_model.pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints found under runs/{a.group}")

    mode = "ALL slices" if a.keep_empty else "foreground slices only"
    print(f"re-evaluating {len(ckpts)} run(s) on device={device} "
          f"(num_workers={a.num_workers}, {mode})")
    for c in ckpts:
        run_one(c, device, num_workers=a.num_workers, batch_size=a.batch_size,
                keep_empty=a.keep_empty)

    print(f"\ndone. Re-run  python -m src.utils.compare --group {a.group}  "
          f"to refresh the comparison.")


if __name__ == "__main__":
    main()
