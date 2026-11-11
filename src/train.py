
import argparse
import gc
import json
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import Config, ARCH_NAMES, RESOLUTIONS, get_device, readable_run_name
from src.data import get_dataloaders, slice_keys
from src.model import build_model
from src.losses import build_loss
from src.evaluate import (evaluate, per_volume_confusion, metrics_from_confusion,
                          case_dice, volume_scores, print_scores)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model, cfg):
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay, nesterov=True)
    raise ValueError(f"unknown optimizer: {cfg.optimizer}")


def build_scheduler(optimizer, cfg):
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg.epochs // 3), gamma=0.1)
    return None


def train_epoch(model, loader, criterion, optimizer, device, scaler=None, accum_steps=1):
    # accum_steps > 1 adds up the gradients of that many batches before one
    # optimiser step: batch 4 x 2 then trains like batch 8 on a GPU too small
    # for 8 (batch norm still sees 4 slices at a time)
    model.train()
    total, seen = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    for i, (x, y) in enumerate(loader, 1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        # batches in this step; the last step of an epoch can have fewer
        n = min(accum_steps, len(loader) - (i - 1) // accum_steps * accum_steps)
        step = i % accum_steps == 0 or i == len(loader)

        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                loss = criterion(model(x), y)
            scaler.scale(loss / n).backward()
            if step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss = criterion(model(x), y)
            (loss / n).backward()
            if step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        total += loss.item() * x.size(0)
        seen += x.size(0)

    return total / max(seen, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes, class_names):
    """validation loss, pooled metrics, and m["tumor_case"]: the mean tumour
    Dice per patient by LiTS rules (None when there is no tumour class)"""
    model.eval()
    total, seen = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        total += criterion(model(x), y).item() * x.size(0)
        seen += x.size(0)
    # one confusion matrix per patient; their sum is the pooled one
    cms = per_volume_confusion(model, loader, device, num_classes, slice_keys(loader.dataset))
    m = metrics_from_confusion(sum(cms.values()), class_names)
    m["tumor_case"] = (float(np.mean([case_dice(cm, empty=1.0)[2] for cm in cms.values()]))
                       if num_classes > 2 else None)
    return total / max(seen, 1), m


def plot_history(history, out_path):
    ep = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(ep, history["train_loss"], label="train")
    axes[0].plot(ep, history["val_loss"], label="validation")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=.3)

    axes[1].plot(ep, history["val_dice_fg"], label="val Dice (foreground)", color="#C0392B")
    axes[1].plot(ep, history["val_dice"], label="val Dice (all classes)",
                 color="#4A90D9", alpha=.7)
    if any(v is not None for v in history.get("val_tumor_case", [])):
        axes[1].plot(ep, history["val_tumor_case"], label="val tumour Dice per patient",
                     color="#8E44AD")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("Dice")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Validation Dice"); axes[1].legend(); axes[1].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train(cfg: Config, verbose=True):
    set_seed(cfg.seed)
    device = get_device()
    out_dir = cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, pixel_counts = get_dataloaders(cfg, verbose=verbose)

    model = build_model(cfg).to(device)
    if verbose:
        enc = f"  encoder={cfg.encoder_weights}" if cfg.arch.endswith("_r50") else ""
        batch = (f"{cfg.batch_size}x{cfg.accum_steps} (= {cfg.batch_size * cfg.accum_steps})"
                 if cfg.accum_steps > 1 else str(cfg.batch_size))
        print(f"[model] {cfg.arch}  device={device}  "
              f"trainable params={model.count_parameters():,}  in_slices={cfg.in_slices}{enc}  "
              f"batch={batch}  amp={'on' if (cfg.amp and device.type=='cuda') else 'off'}")

    criterion = build_loss(cfg, pixel_counts, device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler(device.type) if (cfg.amp and device.type == "cuda") else None

    history = {k: [] for k in ["train_loss", "val_loss", "val_dice", "val_dice_fg", "val_iou_fg",
                               "val_tumor_case", "lr"]}
    # the score that picks the best epoch; a liver-only run has no tumour to judge by
    select = cfg.select_metric if cfg.num_classes > 2 else "dice_fg"
    best_val, best_dice, best_epoch, patience = -1.0, -1.0, -1, 0
    ckpt_path = out_dir / "best_model.pt"
    t0 = time.time()

    def save_history(done_epochs):
        with open(out_dir / "history.json", "w") as fh:
            json.dump({"config": cfg.to_dict(), "history": history,
                       "select_metric": select, "best_val": best_val,
                       # foreground Dice at the best epoch, whatever picked it
                       "best_val_dice_fg": best_dice, "best_epoch": best_epoch,
                       "epochs_completed": done_epochs,
                       "train_minutes": (time.time() - t0) / 60,
                       "params": model.count_parameters()}, fh, indent=2)

    interrupted = False
    epoch = 0
    try:
        for epoch in range(1, cfg.epochs + 1):
            tr_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler,
                                  cfg.accum_steps)
            va_loss, m = validate(model, val_loader, criterion, device, cfg.num_classes, cfg.class_names)

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            history["val_dice"].append(m["mean_dice"])
            history["val_dice_fg"].append(m["mean_dice_fg"])
            history["val_iou_fg"].append(m["mean_iou_fg"])
            history["val_tumor_case"].append(m["tumor_case"])
            history["lr"].append(optimizer.param_groups[0]["lr"])
            if scheduler:
                scheduler.step()

            score = m["tumor_case"] if select == "tumor_case" else m["mean_dice_fg"]
            flag = ""
            if score > best_val:
                best_val, best_dice, best_epoch, patience = score, m["mean_dice_fg"], epoch, 0
                torch.save({"model_state": model.state_dict(), "config": cfg.to_dict(),
                            "epoch": epoch, "val_dice_fg": best_dice,
                            "select_metric": select, "val_select": best_val}, ckpt_path)
                flag = "  <- best"
            else:
                patience += 1

            save_history(epoch)

            if verbose:
                per = "  ".join(f"{c[:4]} {m['per_class'][c]['dice']:.3f}"
                                for c in cfg.class_names[1:])
                tpp = f"  tumour/patient {m['tumor_case']:.3f}" if m["tumor_case"] is not None else ""
                print(f"epoch {epoch:3d}/{cfg.epochs}  train {tr_loss:.4f}  "
                      f"val {va_loss:.4f}  fgDice {m['mean_dice_fg']:.4f}  [{per}]{tpp}{flag}")

            if cfg.early_stop_patience and patience >= cfg.early_stop_patience:
                if verbose:
                    print(f"[train] early stop at epoch {epoch} "
                          f"(no Dice improvement for {patience} epochs)")
                break
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n[train] interrupted at epoch {epoch}. The best checkpoint and the "
              f"history so far are kept; the test evaluation still runs below.")

    elapsed = time.time() - t0
    if verbose:
        print(f"[train] {'stopped' if interrupted else 'done'} in {elapsed/60:.1f} min. "
              f"best epoch {best_epoch} by {select}: {best_val:.4f} "
              f"(val fg Dice there {best_dice:.4f})")

    if not history["train_loss"]:
        print("[train] no epoch completed - nothing to evaluate.")
        return model, {}

    plot_history(history, out_dir / "training_curves.png")
    save_history(epoch)

    if not ckpt_path.exists():
        print(f"[train] no checkpoint at {ckpt_path} - skipping test evaluation.")
        return model, {}

    # Let go of the training data before the test. Every scan read in training
    # stays memory-mapped, so its pages count as this process's memory; with the
    # test scans on top, ws11 (31 GB) ran short and systemd-oomd killed the run
    # during the test scoring (8 Nov). Unmapped, the kernel can drop them freely.
    for dl in (train_loader, val_loader):
        if hasattr(dl.dataset, "close"):
            dl.dataset.close()
    del train_loader, val_loader          # also stops their workers, if any
    gc.collect()

    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
    metrics = evaluate(model, test_loader, device, cfg.class_names, out_dir,
                       split="test", verbose=verbose)
    # per patient and after the 3D clean-up, as reeval does
    metrics["scores"] = volume_scores(model, test_loader, device, cfg.class_names,
                                      slice_keys(test_loader.dataset))
    if verbose:
        print_scores(metrics["scores"], cfg.class_names)
    metrics["best_val_dice_fg"] = best_dice
    metrics["params"] = model.count_parameters()
    metrics["train_minutes"] = elapsed / 60
    with open(out_dir / "metrics_test.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[out] everything saved in {out_dir}")
    return model, metrics


def parse_args(argv=None):
    d = Config()
    p = argparse.ArgumentParser(description="liver and tumour segmentation on CT")

    p.add_argument("--run-name", default=None)
    p.add_argument("--group", default=d.group,
                   help="runs are saved in runs/<nifti|internship_png>/<group>/<run-name>")
    p.add_argument("--arch", choices=ARCH_NAMES, default=d.arch,
                   help="attention_unet (default) or unet, from scratch; *_r50 = the same, "
                        "or U-Net++ (unetpp_r50), with an ImageNet-pretrained ResNet-50 encoder")
    p.add_argument("--in-slices", type=int, default=d.in_slices,
                   help="slices per input: 1 = 2D (default), 3 = 2.5D, the slice and its "
                        "two neighbours as channels (nifti only)")
    p.add_argument("--encoder-weights", choices=["imagenet", "none"], default=d.encoder_weights,
                   help="*_r50 only: start the ResNet-50 from ImageNet (default) or random")

    p.add_argument("--data-format", choices=["nifti", "png"], default=d.data_format,
                   help="nifti = MSD Task03_Liver (default), png = the internship's export")
    p.add_argument("--nifti-dir", default=d.nifti_dir, help="default data/Task03_Liver")
    p.add_argument("--hu-window", type=float, nargs=2, default=d.hu_window,
                   metavar=("LO", "HI"), help="nifti: HU clipped to this and mapped to 0..1")
    p.add_argument("--data-dir", default=d.data_dir, help="png only, default data/internship_png")
    p.add_argument("--resolution", choices=list(RESOLUTIONS), default=None,
                   help="full=512 (default for nifti) half=256 quarter=128 eighth=64. "
                        "Lower it, or --batch-size, when the GPU is too small")
    p.add_argument("--image-size", type=int, default=None,
                   help="any other slice size in pixels; --resolution is simpler")
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--accum-steps", type=int, default=d.accum_steps,
                   help="one optimiser step every N batches: --batch-size 4 --accum-steps 2 "
                        "trains like batch 8 when 8 does not fit the GPU")
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--val-fraction", type=float, default=d.val_fraction)
    p.add_argument("--test-fraction", type=float, default=d.test_fraction)
    p.add_argument("--folds", type=int, default=d.folds,
                   help="k-fold by patient: 5 = five runs, each testing a different fifth "
                        "of the patients (--test-fraction is then unused). 0 = the fixed split")
    p.add_argument("--fold", type=int, default=d.fold,
                   help="which fold this run tests, 0..folds-1")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--liver-only", action="store_true",
                   help="merge tumour into liver -> 2 classes, much easier")
    p.add_argument("--keep-empty", action="store_true",
                   help="keep pure-background slices (~69%% of LiTS). off by default")
    p.add_argument("--max-samples", type=int, default=d.max_samples,
                   help="cap the number of slices used; 0 = all")
    p.add_argument("--split-by-slice", action="store_true",
                   help="split individual slices instead of whole volumes. "
                        "leaks near-duplicates between train and test - "
                        "only for comparison")

    p.add_argument("--base-filters", type=int, default=d.base_filters)
    p.add_argument("--depth", type=int, default=d.depth)
    p.add_argument("--dropout", type=float, default=d.dropout)
    p.add_argument("--no-batch-norm", action="store_true")
    p.add_argument("--bilinear", action="store_true",
                   help="upsample by interpolation instead of transposed conv")

    p.add_argument("--loss", choices=["dice_ce", "dice", "ce"], default=d.loss)
    p.add_argument("--dice-weight", type=float, default=d.dice_weight)
    p.add_argument("--class-weights", choices=["none", "inverse", "manual"],
                   default=d.class_weights)

    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--optimizer", choices=["adam", "sgd"], default=d.optimizer)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--scheduler", choices=["cosine", "step", "none"], default=d.scheduler)
    p.add_argument("--select-metric", choices=["tumor_case", "dice_fg"], default=d.select_metric,
                   help="validation score that picks best_model.pt: tumour Dice per patient "
                        "(default) or foreground Dice over all pixels")
    p.add_argument("--early-stop-patience", type=int, default=d.early_stop_patience)
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--seed", type=int, default=d.seed)

    a = p.parse_args(argv)
    if a.accum_steps < 1:
        p.error("--accum-steps must be 1 or more")
    if a.folds == 1 or a.folds < 0:
        p.error("--folds must be 0 (fixed split) or 2 or more")
    if a.folds and not 0 <= a.fold < a.folds:
        p.error(f"--fold must be 0..{a.folds - 1}")
    if a.folds and a.split_by_slice:
        p.error("--folds splits patients; it cannot be used with --split-by-slice")
    cfg = Config(
        data_dir=a.data_dir, data_format=a.data_format, nifti_dir=a.nifti_dir,
        hu_window=tuple(a.hu_window),
        # the png export is 256 px, so above that it would only be upsampled
        image_size=(RESOLUTIONS[a.resolution] if a.resolution else a.image_size
                    or (d.image_size if a.data_format == "nifti" else 256)),
        batch_size=a.batch_size, accum_steps=a.accum_steps, num_workers=a.num_workers,
        val_fraction=a.val_fraction, test_fraction=a.test_fraction,
        augment=not a.no_augment, liver_only=a.liver_only,
        skip_empty=not a.keep_empty, max_samples=a.max_samples,
        split_by_volume=not a.split_by_slice, folds=a.folds, fold=a.fold,
        arch=a.arch, in_slices=a.in_slices, encoder_weights=a.encoder_weights,
        base_filters=a.base_filters, depth=a.depth, dropout=a.dropout,
        batch_norm=not a.no_batch_norm, bilinear=a.bilinear,
        loss=a.loss, dice_weight=a.dice_weight, class_weights=a.class_weights,
        epochs=a.epochs, optimizer=a.optimizer, lr=a.lr,
        weight_decay=a.weight_decay, scheduler=a.scheduler,
        select_metric=a.select_metric,
        early_stop_patience=a.early_stop_patience, amp=not a.no_amp, seed=a.seed,
        run_name=a.run_name or "run", group=a.group,
    )
    if not a.run_name:
        # named after what sets the run apart, e.g. unet_r50_2.5d_lr1e-4_fold0;
        # a repeat with the same settings gets _2, _3 ...
        name = readable_run_name(cfg.to_dict())
        cfg.run_name, i = name, 2
        while cfg.out_dir.exists():
            cfg.run_name, i = f"{name}_{i}", i + 1
    return cfg


if __name__ == "__main__":
    train(parse_args())
