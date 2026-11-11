"""Grad-CAM for segmentation: which parts of the slice made the model say "tumour"?

    python -m src.utils.gradcam --run runs/nifti/r50/unet_r50_2.5d_lr1e-4
    python -m src.utils.gradcam --run runs/nifti/kfold/unet_r50_2.5d_lr1e-4_fold0 --cpu

Seg-Grad-CAM (Vinogradova et al., 2020). For one slice, the score explained is
the sum of the tumour logits over the pixels the model predicted as tumour, or
over the true tumour when it predicted none (then the map shows where the model
looked while missing it). The gradient of that score with respect to one layer's
feature maps, averaged over the slice, weights each map; their weighted sum
through a ReLU is the heat map, scaled to 0..1.

Uses the run's own test patients (the fold it held out, for a k-fold run), the
slice with the most tumour of each. Writes to results/nifti/gradcam/:

  <run>.png   heat maps at three depths of the network, for a few patients
  <run>.txt   share of the heat inside the tumour and inside the liver,
                      against the share of the slice they cover, every test patient

The second is the number to quote: a model that looks in the right place puts
far more of its heat on the tumour than the tumour's area alone would give.
--cpu runs it without the GPU, e.g. while the workstation is training.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

TUMOUR = 2

# layers explained by default: deep in the encoder, the bottleneck, and the
# decoder at 1/8 of the input size
LAYERS = {
    "unet_r50": ["encoder.layers.2", "encoder.layers.3", "blocks.1"],
    "attention_unet_r50": ["encoder.layers.2", "encoder.layers.3", "blocks.1"],
    "unetpp_r50": ["encoder.layers.2", "encoder.layers.3", "nodes.2_2"],
    "unet": ["downs.2", "downs.3", "ups.0"],
    "attention_unet": ["downs.2", "downs.3", "ups.0"],
}
NAMES = {"encoder.layers.2": "ResNet layer 3", "encoder.layers.3": "bottleneck (layer 4)",
         "blocks.1": "decoder", "nodes.2_2": "decoder node 2_2",
         "downs.2": "encoder level 3", "downs.3": "bottleneck", "ups.0": "decoder"}


def seg_gradcam(model, x, region, layers, cls=TUMOUR):
    """IN   x      - (1, C, H, W) model input
            region - (H, W) bool, the pixels whose class-cls score is explained
            layers - module names, e.g. "encoder.layers.3"
       OUT  {layer: heat map (H, W), 0..1}, predicted labels (H, W)
    """
    acts = {}
    hooks = [model.get_submodule(n).register_forward_hook(
        lambda m, i, o, n=n: acts.__setitem__(n, o)) for n in layers]
    try:
        logits = model(x)
    finally:
        for h in hooks:
            h.remove()
    score = logits[0, cls][torch.from_numpy(region).to(x.device)].sum()
    grads = torch.autograd.grad(score, [acts[n] for n in layers])
    heat = {}
    for n, g in zip(layers, grads):
        w = g[0].mean((1, 2), keepdim=True)                 # one weight per channel
        cam = F.relu((w * acts[n][0]).sum(0))
        cam = F.interpolate(cam[None, None], size=x.shape[-2:], mode="bilinear",
                            align_corners=False)[0, 0]
        heat[n] = (cam / cam.max() if cam.max() > 0 else cam).detach().cpu().numpy()
    return heat, logits[0].argmax(0).cpu().numpy()


def slice_input(img, z, cfg):
    """the model's input for slice z, with its neighbours for a 2.5D model, as in training"""
    from src.nifti import scale_hu
    k = cfg.in_slices // 2
    zs = np.clip(np.arange(z - k, z + k + 1), 0, img.shape[0] - 1)
    x = (scale_hu(img[zs].astype(np.float32), cfg.hu_window) - 0.5) / 0.5    # mean 0.5, std 0.5
    return torch.from_numpy(x)[None]


def heat_share(h, mask):
    # fraction of the heat that falls on mask; NaN for an all-zero map
    s = h.sum()
    return float(h[mask].sum() / s) if s > 0 else float("nan")


def explain_patients(model, cfg, device, layers):
    """one slice per test patient with tumour - its largest - explained at every layer"""
    from src.nifti import nifti_slices, open_volume, scale_hu
    from src.utils.view3d import DISPLAY_WINDOW, test_patients
    _, _, volumes, cache_dir = nifti_slices(cfg, verbose=False)
    labelled = {k: e for k, e in volumes.items() if e["split"] == "train"}
    out, no_tumour = [], []
    for key in test_patients(labelled, cfg):
        e = labelled[key]
        z = int(np.argmax(e["tumor_px"]))
        if e["tumor_px"][z] == 0:
            no_tumour.append(key)
            continue
        img, lab = open_volume(cache_dir, e)
        gt = np.asarray(lab[z])
        x = slice_input(img, z, cfg).to(device)
        # explain the predicted tumour; if the model found none, the true one
        with torch.no_grad():
            pred = model(x)[0].argmax(0).cpu().numpy()
        missed = not (pred == TUMOUR).any()
        region = gt == TUMOUR if missed else pred == TUMOUR
        heat, pred = seg_gradcam(model, x, region, layers)
        out.append(dict(key=key, z=z, gt=gt, pred=pred, heat=heat, missed=missed,
                        ct=scale_hu(np.asarray(img[z], np.float32), DISPLAY_WINDOW)))
    return out, no_tumour


def report(rows, no_tumour, layers, run_name):
    L, a = [], lambda s: L.append(s)
    a(f"GRAD-CAM  {run_name}  (tumour class, Seg-Grad-CAM)")
    a("Share of the heat on the true tumour / liver, against the share of the slice they cover.")
    a("A model looking in the right place gives far more heat than area. 'missed' = no tumour")
    a("predicted on that slice, so the map explains the true tumour region instead.")
    a("")
    a("heat on the tumour, per layer:")
    a(f"{'patient':<12}{'slice':>6}{'tumour area':>13}" + "".join(f"{NAMES.get(n, n)[:22]:>24}" for n in layers))
    for r in rows:
        t = r["gt"] == TUMOUR
        a(f"{r['key']:<12}{r['z']:>6}{100 * t.mean():>12.1f}%"
          + "".join(f"{100 * heat_share(r['heat'][n], t):>23.0f}%" for n in layers)
          + ("   missed" if r["missed"] else ""))
    a("-" * (31 + 24 * len(layers)))
    for what, sel in (("tumour", lambda g: g == TUMOUR), ("liver", lambda g: g > 0)):
        area = np.mean([sel(r["gt"]).mean() for r in rows])
        shares = [np.nanmean([heat_share(r["heat"][n], sel(r["gt"])) for r in rows]) for n in layers]
        a(f"{'mean, ' + what:<18}{100 * area:>12.1f}%" + "".join(f"{100 * s:>23.0f}%" for s in shares))
    a("")
    a(f"{len(rows)} test patients with tumour, {sum(r['missed'] for r in rows)} missed; "
      f"{len(no_tumour)} without tumour skipped: {', '.join(no_tumour) or '-'}")
    return "\n".join(L)


def figure(rows, layers, out_path, title, n=4, dpi=110):
    # n patients spread over the tumour sizes, largest to smallest
    rows = sorted(rows, key=lambda r: -(r["gt"] == TUMOUR).sum())
    rows = [rows[i] for i in sorted(set(np.linspace(0, len(rows) - 1, min(n, len(rows))).round().astype(int)))]
    fig, axes = plt.subplots(len(rows), len(layers) + 1, figsize=(3.4 * (len(layers) + 1), 3.5 * len(rows)),
                             squeeze=False)
    fig.suptitle(f"{title}\nyellow = true tumour, red = predicted tumour, cyan = true liver",
                 fontsize=12, fontweight="bold")
    for row, r in zip(axes, rows):
        t = r["gt"] == TUMOUR
        row[0].imshow(r["ct"], cmap="gray")
        row[0].contour(r["gt"] > 0, levels=[0.5], colors="cyan", linewidths=0.8)
        row[0].contour(t, levels=[0.5], colors="yellow", linewidths=1.2)
        if (r["pred"] == TUMOUR).any():
            row[0].contour(r["pred"] == TUMOUR, levels=[0.5], colors="red", linewidths=1.0)
        row[0].set_title(f"{r['key']}  slice {r['z']}" + ("  (missed)" if r["missed"] else ""), fontsize=9)
        for ax, n in zip(row[1:], layers):
            ax.imshow(r["ct"], cmap="gray")
            ax.imshow(r["heat"][n], cmap="jet", alpha=0.45, vmin=0, vmax=1)
            ax.contour(t, levels=[0.5], colors="white", linewidths=0.8)
            ax.set_title(f"{NAMES.get(n, n)}\n{100 * heat_share(r['heat'][n], t):.0f}% of heat on tumour "
                         f"({100 * t.mean():.1f}% of slice)", fontsize=8.5)
        for ax in row:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"[saved] {out_path}")


def main():
    from src.config import get_device, results_dir
    from src.model import load_checkpoint

    p = argparse.ArgumentParser(description="Grad-CAM heat maps for the tumour class")
    p.add_argument("--run", required=True, help="a run folder holding best_model.pt")
    p.add_argument("--layers", nargs="+", default=None,
                   help="module names to explain, e.g. encoder.layers.3 blocks.1; "
                        "default: three depths chosen per architecture")
    p.add_argument("--patients", type=int, default=4, help="patients shown in the figure")
    p.add_argument("--cpu", action="store_true", help="run on the CPU, leaving the GPU to training")
    p.add_argument("--dpi", type=int, default=110)
    a = p.parse_args()

    run = Path(a.run)
    device = torch.device("cpu") if a.cpu else get_device()
    model, cfg, ck = load_checkpoint(run / "best_model.pt", device)
    if cfg.data_format != "nifti":
        raise SystemExit("gradcam reads the nifti slice cache; this run was trained on the pngs")
    layers = a.layers or LAYERS[cfg.arch]
    print(f"[load] {run.name}  arch={cfg.arch}  epoch={ck.get('epoch')}  device={device}")
    print(f"[gradcam] layers: {', '.join(layers)}")

    rows, no_tumour = explain_patients(model, cfg, device, layers)
    if not rows:
        raise SystemExit("no test patient with tumour")
    txt = report(rows, no_tumour, layers, run.name)
    print(txt)
    out = results_dir("nifti", "gradcam")
    (out / f"{run.name}.txt").write_text(txt, encoding="utf-8")
    print(f"[saved] {out / f'{run.name}.txt'}")
    figure(rows, layers, out / f"{run.name}.png",
           f"Grad-CAM, tumour class: {run.name}", a.patients, a.dpi)


if __name__ == "__main__":
    main()
