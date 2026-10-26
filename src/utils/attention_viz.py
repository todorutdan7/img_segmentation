"""What did the attention gates actually learn?

    python -m src.utils.attention_viz --group nifti/baseline

Loads a trained Attention U-Net, runs test slices through it, and reads out the
alpha map each gate produced. Two outputs:

  results/attention_maps.png    the maps overlaid on real slices
  results/attention_stats.txt   how SELECTIVE the gates are, numerically

The second is the point. A gate that helps should give high alpha over the organ
and low alpha elsewhere. If mean alpha inside the liver is about the same as
outside it, the gate is passing the skip through nearly unchanged and the
attention mechanism is doing no work - which would explain an ablation that
shows no improvement.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.config import get_device
from src.data import get_dataloaders
from src.model import attention_maps, load_checkpoint


def load_run(ckpt_path, device):
    """Rebuild the exact model AND the exact data split from the checkpoint."""
    model, cfg, ck = load_checkpoint(ckpt_path, device)
    if cfg.arch != "attention_unet":
        raise SystemExit(f"{ckpt_path} is arch={cfg.arch}; there are no gates to read.")
    print(f"[load] {ckpt_path}")
    print(f"[load] arch={cfg.arch}  epoch={ck.get('epoch')}  "
          f"val_dice_fg={ck.get('val_dice_fg'):.4f}  params={model.count_parameters():,}")
    return model, cfg


def find_ckpt(runs_dir: Path, group: str):
    cands = sorted((runs_dir / group).glob("attention_unet*/best_model.pt"))
    if not cands:
        raise SystemExit(f"no attention_unet run with a checkpoint under {runs_dir/group}")
    return cands[-1]


@torch.no_grad()
def measure(model, loader, device, n_slices=200, min_tumor=50):
    """Mean alpha inside vs outside the foreground, per gate, over many slices.

    Also records the alpha histogram so we can see whether the gate is making a
    decision at all or just outputting a near-constant value.
    """
    depth = len(model.gates)
    inside = [[] for _ in range(depth)]
    outside = [[] for _ in range(depth)]
    tum_in = [[] for _ in range(depth)]
    # running sums give mean and std exactly, without keeping any alpha values.
    # Storing them all costs 200 x 256^2 floats for the finest gate alone.
    n = [0] * depth
    s1 = [0.0] * depth
    s2 = [0.0] * depth
    grid = [0] * depth
    sample = [[] for _ in range(depth)]      # small thinned subset, percentiles only
    seen = 0

    for x, y in loader:
        for b in range(x.shape[0]):
            xb, yb = x[b:b+1].to(device), y[b:b+1]
            if (yb == 2).sum() < min_tumor:
                continue
            maps = attention_maps(model, xb)
            for g, a in enumerate(maps):
                a = a.float()                                   # (1,1,h,w) on cpu
                # bring the mask down to the gate's grid instead of upsampling
                # alpha - avoids inventing values the gate never produced
                m = F.interpolate(yb[None].float(), size=a.shape[-2:],
                                  mode="nearest")[0, 0]
                av = a[0, 0]
                inside[g].append(av[m > 0].mean().item() if (m > 0).any() else np.nan)
                outside[g].append(av[m == 0].mean().item() if (m == 0).any() else np.nan)
                tum_in[g].append(av[m == 2].mean().item() if (m == 2).any() else np.nan)

                flat = av.reshape(-1).double()
                n[g] += flat.numel()
                s1[g] += flat.sum().item()
                s2[g] += (flat * flat).sum().item()
                grid[g] = av.shape[-1]
                if len(sample[g]) < 300:                      # ~60k values, 240 KB
                    step = max(1, flat.numel() // 200)
                    sample[g].append(flat[::step].float().numpy())
            seen += 1
            if seen >= n_slices:
                break
        if seen >= n_slices:
            break

    out = []
    for g in range(depth):
        cnt = max(n[g], 1)
        mean = s1[g] / cnt
        var = max(s2[g] / cnt - mean * mean, 0.0)          # E[x^2] - E[x]^2
        v = np.concatenate(sample[g]) if sample[g] else np.array([0.0], np.float32)
        out.append(dict(
            gate=g, hw=grid[g],
            inside=float(np.nanmean(inside[g])),
            outside=float(np.nanmean(outside[g])),
            tumor=float(np.nanmean(tum_in[g])),
            mean=float(mean), std=float(np.sqrt(var)),
            p05=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)),
        ))
    return out, seen


@torch.no_grad()
def pick(loader, device, n=3, min_tumor=200):
    got = []
    for x, y in loader:
        for b in range(x.shape[0]):
            if (y[b] == 2).sum() >= min_tumor:
                got.append((x[b:b+1].to(device), y[b:b+1]))
                if len(got) >= n:
                    return got
    return got


def figure(model, samples, stats, out_path, dpi=110):
    depth = len(model.gates)
    rows, cols = len(samples), 3 + depth
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.2 * rows),
                             squeeze=False)
    fig.suptitle("Attention U-Net — what each gate actually attends to",
                 fontsize=15, fontweight="bold")

    for r, (x, y) in enumerate(samples):
        with torch.no_grad():
            pred = model(x).argmax(1)[0].cpu().numpy()
            maps = attention_maps(model, x)
        img = x[0, 0].cpu().numpy()
        gt = y[0].numpy()

        axes[r][0].imshow(img, cmap="gray")
        axes[r][0].imshow(np.ma.masked_where(gt == 0, gt), cmap="cool",
                          vmin=0, vmax=2, alpha=.65, interpolation="nearest")
        axes[r][0].set_title("slice + ground truth" if r == 0 else "", fontsize=10)

        axes[r][1].imshow(img, cmap="gray")
        axes[r][1].imshow(np.ma.masked_where(pred == 0, pred), cmap="cool",
                          vmin=0, vmax=2, alpha=.65, interpolation="nearest")
        axes[r][1].set_title("prediction" if r == 0 else "", fontsize=10)

        # difference map: where the model got it wrong
        err = (pred != gt).astype(float)
        axes[r][2].imshow(img, cmap="gray")
        axes[r][2].imshow(np.ma.masked_where(err == 0, err), cmap="autumn",
                          alpha=.8, interpolation="nearest")
        axes[r][2].set_title("errors" if r == 0 else "", fontsize=10)

        for g, a in enumerate(maps):
            av = a[0, 0].numpy()
            ax = axes[r][3 + g]
            im = ax.imshow(av, cmap="inferno", vmin=0, vmax=1)
            if r == 0:
                ax.set_title(f"gates.{g}  α  {av.shape[0]}²", fontsize=10)
            ax.text(0.02, 0.02, f"[{av.min():.2f}, {av.max():.2f}]  σ={av.std():.3f}",
                    transform=ax.transAxes, fontsize=7.5, color="white", va="bottom")
            if r == rows - 1:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"[saved] {out_path}")


def report(stats, n_seen):
    L, a = [], lambda s: L.append(s)
    a("ATTENTION GATE SELECTIVITY")
    a("=" * 78)
    a(f"measured over {n_seen} test slices that contain tumour")
    a("")
    a("alpha is the gate's per-pixel weight on the skip connection, in (0,1).")
    a("A gate that is doing work should weight the organ higher than the rest of")
    a("the abdomen. 'ratio' below is mean alpha inside the liver divided by mean")
    a("alpha outside it. 1.00 means the gate is not selecting anything.")
    a("")
    a(f"{'gate':<8}{'grid':>7}{'in fg':>9}{'in bg':>9}{'ratio':>8}"
      f"{'on tumour':>11}{'mean':>8}{'std':>8}{'p05':>7}{'p95':>7}")
    a("-" * 78)
    for s in stats:
        ratio = s["inside"] / s["outside"] if s["outside"] else float("nan")
        a(f"gates.{s['gate']:<2}{s['hw']:>6}²{s['inside']:>9.3f}{s['outside']:>9.3f}"
          f"{ratio:>8.2f}{s['tumor']:>11.3f}{s['mean']:>8.3f}{s['std']:>8.3f}"
          f"{s['p05']:>7.3f}{s['p95']:>7.3f}")
    a("-" * 78)
    a("")

    ratios = [s["inside"] / s["outside"] for s in stats if s["outside"]]
    stds = [s["std"] for s in stats]
    strongest = max(ratios) if ratios else 1.0

    a("READING THIS")
    if strongest < 1.10:
        a(f"  The strongest gate weights foreground only {strongest:.2f}x higher than")
        a("  background. That is not selection - the gates are passing the skips")
        a("  through close to unchanged, so the attention mechanism is contributing")
        a("  almost nothing. This is consistent with an ablation showing no gain:")
        a("  the gates cost parameters and compute without altering the signal.")
    elif strongest < 1.5:
        a(f"  Mild selectivity ({strongest:.2f}x at best). The gates do prefer the")
        a("  organ, but weakly. Any benefit would be small, which matches an")
        a("  ablation where the difference sits inside the noise.")
    else:
        a(f"  Clear selectivity ({strongest:.2f}x). The gates ARE suppressing")
        a("  background. If the ablation still shows no gain, the filtering is")
        a("  real but not the bottleneck for this task.")
    a("")
    if max(stds) < 0.05:
        a(f"  Also note the spread: the widest gate has std={max(stds):.3f}, i.e. alpha")
        a("  is nearly constant across the whole image. A constant alpha is just a")
        a("  scalar rescale of the skip, which the following convolution can undo -")
        a("  functionally equivalent to no gate at all.")

    # A gate can be selective for the ORGAN while actively suppressing the
    # LESION. That is worth calling out separately: it hurts exactly the class
    # the model is worst at, and a single foreground ratio hides it.
    bad = [s for s in stats if s["inside"] > 0 and s["tumor"] < 0.6 * s["inside"]]
    if bad:
        a("")
        a("  WARNING - gate(s) suppressing tumour while keeping liver:")
        for s in bad:
            a(f"    gates.{s['gate']} ({s['hw']}²): alpha on foreground {s['inside']:.3f}, "
              f"but on TUMOUR only {s['tumor']:.3f}")
        a("    These gates down-weight the skip precisely where the lesions are, so")
        a("    fine detail about the rarest class is attenuated before it reaches the")
        a("    decoder. If the ablation shows attention doing worse on tumour Dice")
        a("    than the plain baseline, this is a plausible mechanism for it.")
    a("")
    a("CAVEAT: this describes ONE trained model. A different seed could learn")
    a("different gates. It explains this run; it is not a claim about the")
    a("architecture in general.")
    return "\n".join(L)


def main():
    from src.config import RUNS_DIR, results_dir

    p = argparse.ArgumentParser(description="visualise what the attention gates learned")
    p.add_argument("--group", default="nifti/baseline",
                   help="folder under runs/, e.g. nifti/baseline or internship_png/ablation_2070")
    p.add_argument("--ckpt", default=None, help="path to a best_model.pt")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--n-slices", type=int, default=200,
                   help="test slices used for the selectivity statistics")
    p.add_argument("--dpi", type=int, default=110)
    a = p.parse_args()

    device = get_device()
    ckpt = Path(a.ckpt) if a.ckpt else find_ckpt(Path(RUNS_DIR), a.group)
    model, cfg = load_run(ckpt, device)

    print("[data] rebuilding the same split the model was trained with ...")
    _, _, test_loader, _ = get_dataloaders(cfg, verbose=False)

    samples = pick(test_loader, device, a.samples)
    if not samples:
        raise SystemExit("no test slice with enough tumour found")
    print(f"[data] {len(samples)} sample slices, "
          f"{a.n_slices} slices for the statistics")

    stats, seen = measure(model, test_loader, device, a.n_slices)

    out = results_dir(cfg.data_format)
    figure(model, samples, stats, out / "attention_maps.png", a.dpi)

    txt = report(stats, seen)
    print("\n" + txt)
    (out / "attention_stats.txt").write_text(txt, encoding="utf-8")
    print(f"\n[saved] {out / 'attention_stats.txt'}")


if __name__ == "__main__":
    main()
