
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

OVERLAY_COLORS = [(0, 0, 0, 0), (0.20, 0.55, 0.95, 0.55), (0.95, 0.25, 0.20, 0.75)]


@torch.no_grad()
def accumulate_confusion(model, loader, device, num_classes):

    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for x, y in loader:
        x = x.to(device, non_blocking=True)

        pred = model(x).argmax(1).cpu()
        y = y.long()

        k = (y.view(-1) * num_classes + pred.view(-1))
        cm += torch.bincount(k, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return cm.numpy()


@torch.no_grad()
def per_volume_confusion(model, loader, device, num_classes, keys):
    """one confusion matrix per patient instead of one for the whole test set

    keys[i] is the patient of dataset item i. The loader must not shuffle, so
    the i-th prediction is the i-th item. Summing the matrices gives exactly
    accumulate_confusion's pooled one.
    """
    model.eval()
    cms, i = {}, 0
    for x, y in loader:
        pred = model(x.to(device, non_blocking=True)).argmax(1).cpu()
        for b in range(x.size(0)):
            k = y[b].view(-1).long() * num_classes + pred[b].view(-1)
            cm = torch.bincount(k, minlength=num_classes ** 2).reshape(num_classes, num_classes)
            cms[keys[i]] = cms.get(keys[i], 0) + cm.numpy()
            i += 1
    return cms


def case_dice(cm):
    """per-class Dice of one patient. NaN where a class is in neither the truth
    nor the prediction - no tumour there is not a score of 0 or 1"""
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    den = 2 * tp + (cm.sum(axis=0) - tp) + (cm.sum(axis=1) - tp)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, 2 * tp / den, np.nan)



def metrics_from_confusion(cm, class_names):
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    eps = 1e-9
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    support = cm.sum(axis=1)

    present = support > 0

    out = {
        "pixel_accuracy": float(tp.sum() / (cm.sum() + eps)),
        "mean_dice": float(dice[present].mean()) if present.any() else 0.0,
        "mean_iou": float(iou[present].mean()) if present.any() else 0.0,
        "mean_dice_fg": float(dice[1:][present[1:]].mean()) if present[1:].any() else 0.0,
        "mean_iou_fg": float(iou[1:][present[1:]].mean()) if present[1:].any() else 0.0,
        "per_class": {},
        "confusion_matrix": cm.astype(np.int64).tolist(),
    }
    for i, nm in enumerate(class_names):
        out["per_class"][nm] = {
            "dice": float(dice[i]), "iou": float(iou[i]),
            "precision": float(precision[i]), "recall": float(recall[i]),
            "pixels": int(support[i]),
            "pixel_share": float(support[i] / (cm.sum() + eps)),
        }
    return out


def plot_confusion(cm, class_names, out_path, normalize=True):
    data = cm.astype(float)
    if normalize:
        data = data / np.clip(data.sum(axis=1, keepdims=True), 1, None)
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * n, 1.4 + 1.0 * n))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=data.max())
    ax.set_xticks(range(n), class_names, rotation=30, ha="right")
    ax.set_yticks(range(n), class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Pixel confusion matrix ({'normalized' if normalize else 'counts'})")
    thr = data.max() / 2
    for i in range(n):
        for j in range(n):
            txt = f"{data[i, j]:.3f}" if normalize else f"{int(cm[i, j]):,}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if data[i, j] > thr else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dice_bars(metrics, class_names, out_path):
    d = [metrics["per_class"][c]["dice"] for c in class_names]
    i = [metrics["per_class"][c]["iou"] for c in class_names]
    x = np.arange(len(class_names))
    fig, ax = plt.subplots(figsize=(1.6 + 1.5 * len(class_names), 4))
    ax.bar(x - 0.2, d, 0.4, label="Dice", color="#4A90D9")
    ax.bar(x + 0.2, i, 0.4, label="IoU", color="#7FB069")
    for xi, (dv, iv) in enumerate(zip(d, i)):
        ax.text(xi - 0.2, dv + 0.015, f"{dv:.3f}", ha="center", fontsize=8)
        ax.text(xi + 0.2, iv + 0.015, f"{iv:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, class_names)
    ax.set_ylim(0, 1.08); ax.set_ylabel("score")
    ax.set_title("Per-class Dice and IoU (test set)")
    ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)



@torch.no_grad()  # disables gradient tracking to save memory and speed up evaluation
def plot_predictions(model, loader, device, class_names, out_path, n_images=6, prefer_foreground=True):
    from src.data import denormalize

    # Put the model in evaluation mode (turns off layers like dropout)
    model.eval()

    # create a custom color map for the masks using the colors defined
    cmap = ListedColormap(OVERLAY_COLORS[:len(class_names)])

    # identify which class ID belongs to the tumor (2 in this case)
    tum_cls = len(class_names) - 1 if len(class_names) > 2 else None

    # create empty lists ("pools") to hold candidate images for plotting
    fg_pool, tum_pool = [], []

    # helper function to calculate the Dice score for a single 2D image slice
    def dice_of(gt_mask, pr_mask):
        a, b_ = int(gt_mask.sum()), int(pr_mask.sum())
        if a + b_ == 0:
            return None
        return 2 * int((gt_mask & pr_mask).sum()) / (a + b_)

    # STEP 1: SCAN THE DATASET & FILTER IMAGES

    for x, y in loader:
        # generate predictions for this batch
        pred = model(x.to(device, non_blocking=True)).argmax(1).cpu()

        # look at each individual image in the batch
        for b in range(x.size(0)):
            gt, pr = y[b], pred[b]
            item = (denormalize(x[b, 0]), gt, pr)

            # FILTER 1: FOREGROUND
            # only keep slices that have at least 200 pixels of actual anatomy
            # this prevents us from plotting empty, all black slices of air
            if not prefer_foreground or int((gt > 0).sum()) >= 200:
                d = dice_of(gt > 0, pr > 0)
                if d is not None:
                    fg_pool.append((d, item))

            # FILTER 2: TUMOR
            # if the slice has at least 50 pixels of tumor, add it to a special tumor pool
            if tum_cls is not None and int((gt == tum_cls).sum()) >= 50:
                d = dice_of(gt == tum_cls, pr == tum_cls)
                if d is not None:
                    tum_pool.append((d, item))

    # FALLBACK: if the dataset is totally empty or weird, just grab the first batch of images
    if not fg_pool:
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            for b in range(min(x.size(0), n_images)):
                fg_pool.append((0.0, (denormalize(x[b, 0]), y[b], pred[b])))
            break
    if not fg_pool:
        print("[eval] no samples to plot")
        return None

    # STEP 2: IMAGE SELECTION

    # this helper function picks the "Best", "Typical" (Average), and "Worst" predictions

    def take(pool, want, label, metric):
        # sort the pool of images from lowest Dice score to highest Dice score
        pool = sorted(pool, key=lambda t: t[0])
        n = len(pool)
        k = max(1, want // 3)

        # figure out the exact indexes for worst (start of list), typical (middle), best (end)
        order = ([n - 1 - i for i in range(k)]  # Best
                 + [n // 2 + i for i in range(want - 2 * k)]  # typical/median
                 + [i for i in range(k)])  # Worst

        seen, out = set(), []
        names = ["best"] * k + ["typical"] * (want - 2 * k) + ["worst"] * k

        # grab those specific images and format them with tags
        for i, nm in zip(order, names):
            i = min(max(i, 0), n - 1)
            if i in seen: continue
            seen.add(i)
            d, (img, gt, pr) = pool[i]
            # pack it up: Image, Ground Truth, Prediction, Dice Score, Label, Metric Name
            out.append((img, gt, pr, d, f"{label} · {nm}", metric))
        return out

    # decide how many images to show, if we want 6 total, try to do 3 tumor and 3 general foreground
    n_tum = min(len(tum_pool), n_images // 2) if tum_pool else 0
    n_fg = n_images - n_tum

    # actually pick the images using the 'take' function above
    picked = take(fg_pool, n_fg, "overall", "fg Dice")
    if n_tum:
        picked += take(tum_pool, n_tum, "tumour", "tumour Dice")

    # print a summary to the console so the user knows what they are looking at
    print(f"[eval] {len(fg_pool)} slices with foreground "
          f"(median fg Dice {sorted(d for d, _ in fg_pool)[len(fg_pool) // 2]:.3f})")
    if tum_pool:
        td = sorted(d for d, _ in tum_pool)
        print(f"[eval] {len(tum_pool)} slices with >=50 tumour px "
              f"(median tumour Dice {td[len(td) // 2]:.3f}); "
              f"{sum(1 for d in td if d == 0)} of them scored 0.000")
    else:
        print("[eval] no slice had enough tumour to rank separately")

    # STEP 3: DRAW THE GRID OF IMAGES

    rows = len(picked)
    # create a Matplotlib grid, 3 columns per row (Raw Image, Ground Truth, Prediction)
    fig, axes = plt.subplots(rows, 3, figsize=(9, 3 * rows))
    axes = np.atleast_2d(axes)

    for r, (img, gt, pr, dice, tag, metric) in enumerate(picked):
        for c, (title, mask) in enumerate(
                [("CT slice", None), ("ground truth", gt), ("prediction", pr)]):
            ax = axes[r, c]

            # draw the black and white CT scan first
            ax.imshow(img, cmap="gray")

            if mask is not None:
                # np.ma.masked_where makes Class 0 (Background) transparent
                # this way, we only paint the colorful masks over the actual anatomy/tumor
                ax.imshow(np.ma.masked_where(mask == 0, mask), cmap=cmap,
                          vmin=0, vmax=len(class_names) - 1, interpolation="nearest")

            # add column titles on the very top row
            if r == 0:
                ax.set_title(title, fontsize=10)
            ax.axis("off")  # hide graph axes/numbers

        # write the text (e.g. "tumor worst   Dice: 0.120") on the left side of each row
        colour = "#FF6B6B" if tag.startswith("tumour") else "yellow"
        axes[r, 0].text(2, 14, f"{tag}   {metric} {dice:.3f}",
                        color=colour, fontsize=9, fontweight="bold")

        # if there is tumor, print exactly how many tumor pixels were found vs expected
        if tum_cls is not None:
            gt_t, pr_t = int((gt == tum_cls).sum()), int((pr == tum_cls).sum())
            if gt_t or pr_t:
                axes[r, 0].text(2, 30, f"tumour px: true {gt_t}, predicted {pr_t}",
                                color=colour, fontsize=7.5)

    # STEP 4: FINALIZE AND SAVE IMAGE

    # create a custom legend at the bottom mapping colors to class names
    handles = [plt.Rectangle((0, 0), 1, 1, fc=OVERLAY_COLORS[i][:3])
               for i in range(1, len(class_names))]
    fig.legend(handles, class_names[1:], loc="lower center",
               ncol=len(class_names) - 1, frameon=False)

    fig.suptitle("Segmentation results - test set", fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])  # space out the images so they don't overlap

    # save the picture to the disk
    fig.savefig(out_path, dpi=150)
    plt.close(fig)  # free up memory
    print(f"[eval] saved {out_path}")

    return out_path


def evaluate(model, loader, device, class_names, out_dir, split="test", verbose=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(class_names)

    cm = accumulate_confusion(model, loader, device, n)
    m = metrics_from_confusion(cm, class_names)
    m["split"] = split

    if verbose:
        print(f"\n=== {split} metrics ===")
        print(f"pixel accuracy    : {m['pixel_accuracy']:.4f}   "
              f"(inflated by background - do not quote this alone)")
        print(f"mean Dice (all)   : {m['mean_dice']:.4f}")
        print(f"mean Dice (fg)    : {m['mean_dice_fg']:.4f}   <- the headline number")
        print(f"mean IoU  (fg)    : {m['mean_iou_fg']:.4f}")
        print(f"\n{'class':<14}{'Dice':>8}{'IoU':>8}{'prec':>8}{'recall':>8}{'pixel share':>13}")
        print("-" * 59)
        for c in class_names:
            s = m["per_class"][c]
            print(f"{c:<14}{s['dice']:>8.4f}{s['iou']:>8.4f}"
                  f"{s['precision']:>8.4f}{s['recall']:>8.4f}{100*s['pixel_share']:>12.3f}%")

    plot_confusion(cm, class_names, out_dir / f"confusion_{split}.png", normalize=True)
    plot_confusion(cm, class_names, out_dir / f"confusion_{split}_counts.png", normalize=False)
    plot_dice_bars(m, class_names, out_dir / f"dice_iou_{split}.png")
    plot_predictions(model, loader, device, class_names, out_dir / f"predictions_{split}.png")

    with open(out_dir / f"metrics_{split}.json", "w") as f:
        json.dump(m, f, indent=2)
    return m
