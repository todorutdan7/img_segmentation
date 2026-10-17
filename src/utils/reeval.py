"""Re-run the test evaluation on already-trained checkpoints.

    python -m src.utils.reeval --group ablation          # every run in the group
    python -m src.utils.reeval --ckpt runs/.../best_model.pt

Regenerates confusion matrices, the per-class Dice/IoU bars, predictions_test.png
and metrics_test.json inside each run folder - without retraining. Useful after
changing anything in evaluate.py, since those figures are otherwise only written
once, at the end of a training run.

The config is read back out of each checkpoint, so the test split is exactly the
one the model was trained against.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from src.config import get_device
from src.data import get_dataloaders, has_foreground as _has_fg
from src.model import load_checkpoint
from src.evaluate import evaluate


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
    print(f"[load] arch={cfg.arch}  epoch={ck.get('epoch')}  "
          f"val_dice_fg={ck.get('val_dice_fg', float('nan')):.4f}  "
          f"params={model.count_parameters():,}")

    _, _, test_loader, _ = get_dataloaders(cfg, verbose=False)
    n = len(test_loader.dataset)
    empty = sum(1 for _, mp in test_loader.dataset.pairs
                if not _has_fg(mp)) if keep_empty else 0
    print(f"[data] test slices: {n:,}"
          + (f"   ({empty:,} of them contain no liver at all)" if keep_empty else
             "   (foreground slices only)"))

    metrics = evaluate(model, test_loader, device, cfg.class_names, out_dir,
                       split=split_name, verbose=verbose)

    metrics["params"] = model.count_parameters()
    metrics["n_test_slices"] = n
    metrics["skip_empty"] = cfg.skip_empty
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
    p.add_argument("--group", default="ablation")
    p.add_argument("--ckpt", default=None, help="a single best_model.pt")
    p.add_argument("--num-workers", type=int, default=0,
                   help="0 on Windows; the checkpoint's own value is ignored "
                        "because it belongs to the machine that trained it")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--keep-empty", action="store_true",
                   help="score EVERY slice of the test patients, including the "
                        "~71%% with no liver. Writes metrics_test_allslices.json "
                        "so the foreground-only results are not overwritten.")
    a = p.parse_args()

    device = get_device()

    if a.ckpt:
        ckpts = [Path(a.ckpt)]
    else:
        ckpts = sorted((Path(RUNS_DIR) / a.group).glob("*/best_model.pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints found for group={a.group}")

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
