"""Does the whole thing actually work? One command, no training.

    python -m src.utils.selftest

Checks the environment, the import path, the model, a real batch of data, a
forward pass, the loss, the metrics and the attention read-out. Every step
prints PASS or FAIL with the reason. Nothing is written to disk.
"""

import sys
import time
import traceback
from pathlib import Path

RESULTS = []
STATE = {}          # artifacts produced by a check, consumed by later ones


def check(name, requires=()):
    """Decorator: run a step, catch anything, record PASS / FAIL / SKIP.

    `requires` names entries of STATE this step needs. If an earlier check
    failed and never produced them, this one is reported as skipped rather
    than raising NameError - one real failure should not print four.
    """
    def wrap(fn):
        print(f"\n[{len(RESULTS)+1}] {name}")
        missing = [r for r in requires if r not in STATE]
        if missing:
            RESULTS.append((name, None, f"needs {', '.join(missing)}"))
            print(f"    SKIP   depends on {', '.join(missing)}, which did not complete")
            return fn
        t0 = time.time()
        try:
            detail = fn()
            RESULTS.append((name, True, detail or ""))
            print(f"    PASS   ({time.time()-t0:.1f}s)  {detail or ''}")
        except Exception as e:
            RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
            print(f"    FAIL   {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
        return fn
    return wrap


def main():
    print("=" * 78)
    print("SELF-TEST  —  segmentation project")
    print("=" * 78)

    # ---- 1. the import path, the thing that just bit us -------------------
    @check("sys.path is clean (src must NOT be on it)")
    def _():
        here = Path(__file__).resolve().parent.parent      # .../src
        bad = [p for p in sys.path if p and Path(p).resolve() == here]
        if bad:
            raise RuntimeError(
                f"'{here}' is on sys.path. That exposes src/utils as top-level "
                f"'utils' and collides with torch.utils. Unmark src as Sources Root.")
        return f"project root on path, src is not"

    # ---- 2. torch ---------------------------------------------------------
    @check("torch imports and sees the GPU")
    def _():
        import os
        env = os.environ.get("CONDA_DEFAULT_ENV", "(not a conda env)")
        try:
            import torch
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                f"no torch in this interpreter.\n"
                f"           python : {sys.executable}\n"
                f"           env    : {env}\n"
                f"           -> run 'conda activate seg' first; the prompt should show (seg)")
        return (f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
                f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}\n"
                f"           env={env}  python={sys.executable}")

    import torch
    from src.config import Config, get_device, IN_CHANNELS
    arch = sys.argv[1] if len(sys.argv) > 1 else Config().arch
    cfg = Config(arch=arch, max_samples=40, batch_size=2, num_workers=0)
    device = get_device()
    print(f"\n    architecture under test: {cfg.arch}")

    # ---- 3. model ---------------------------------------------------------
    @check("model builds and matches the diagram's parameter count")
    def _():
        from src.model import build_model
        from src.utils.diagram import plan
        m = build_model(cfg)
        P = plan(base_filters=cfg.base_filters, depth=cfg.depth,
                 num_classes=cfg.num_classes, image_size=cfg.image_size,
                 in_channels=IN_CHANNELS, batch_norm=cfg.batch_norm,
                 bilinear=cfg.bilinear, arch=cfg.arch)
        if P["total"] != m.count_parameters():
            raise AssertionError(
                f"arch={cfg.arch}: diagram {P['total']:,} != model {m.count_parameters():,}")
        STATE["model"] = m.to(device)
        return (f"{m.count_parameters():,} parameters ({type(m).__name__}), "
                f"agrees with the diagram")

    # ---- 4. forward on a synthetic batch ----------------------------------
    @check("forward pass produces the right output shape", requires=("model",))
    def _():
        x = torch.randn(2, IN_CHANNELS, cfg.image_size, cfg.image_size, device=device)
        STATE["model"].eval()
        with torch.no_grad():
            out = STATE["model"](x)
        want = (2, cfg.num_classes, cfg.image_size, cfg.image_size)
        if tuple(out.shape) != want:
            raise AssertionError(f"got {tuple(out.shape)}, expected {want}")
        return f"{tuple(x.shape)} -> {tuple(out.shape)}"

    # ---- 5. attention read-out -------------------------------------------
    @check("attention_maps returns one alpha per gate, coarse to fine", requires=("model",))
    def _():
        from src.model import attention_maps
        if cfg.arch != "attention_unet":
            return f"not applicable to arch={cfg.arch} (no gates by design)"
        x = torch.randn(1, IN_CHANNELS, cfg.image_size, cfg.image_size, device=device)
        maps = attention_maps(STATE["model"], x)
        if len(maps) != cfg.depth:
            raise AssertionError(f"{len(maps)} maps, expected {cfg.depth}")
        for i, a in enumerate(maps):
            # gates run deepest-first, and alpha sits on the SKIP's grid (not the
            # gating signal's): gates.0 has g at 16" but x at 32", so alpha is 32".
            exp = cfg.image_size // 2 ** (cfg.depth - 1 - i)
            if tuple(a.shape) != (1, 1, exp, exp):
                raise AssertionError(f"map {i}: {tuple(a.shape)}, expected (1,1,{exp},{exp})")
            if a.min() < 0 or a.max() > 1:
                raise AssertionError(f"map {i} outside (0,1): [{a.min():.3f}, {a.max():.3f}]")
        return "  ".join(f"{tuple(a.shape)[-2:]}" for a in maps) + "  all in (0,1)"

    # ---- 6. real data -----------------------------------------------------
    @check("dataloaders read real LiTS pairs and split by volume")
    def _():
        from src.data import get_dataloaders
        tr, va, te, counts = get_dataloaders(cfg, verbose=False)
        STATE["loaders"] = (tr, va, te, counts)
        n = len(tr.dataset) + len(va.dataset) + len(te.dataset)
        if n == 0:
            raise RuntimeError("no samples - is data/ populated?")
        return (f"train={len(tr.dataset)} val={len(va.dataset)} test={len(te.dataset)}"
                f"   pixel counts per class: {counts}")

    # ---- 7. a real batch --------------------------------------------------
    @check("a real batch has the right dtypes and label range", requires=("loaders",))
    def _():
        tr = STATE["loaders"][0]
        x, y = next(iter(tr))
        if x.dtype != torch.float32:
            raise AssertionError(f"images are {x.dtype}, expected float32")
        if y.dtype != torch.int64:
            raise AssertionError(f"masks are {y.dtype}, expected int64")
        if int(y.max()) >= cfg.num_classes:
            raise AssertionError(f"label {int(y.max())} >= num_classes {cfg.num_classes}")
        STATE["batch"] = (x, y)
        return (f"images {tuple(x.shape)} {x.dtype} range [{x.min():.2f}, {x.max():.2f}]   "
                f"masks {tuple(y.shape)} {y.dtype} labels {sorted(y.unique().tolist())}")

    # ---- 8. loss + backward ----------------------------------------------
    @check("loss computes and gradients flow", requires=("model", "loaders", "batch"))
    def _():
        from src.losses import build_loss
        x, y = STATE["batch"]
        x, y = x.to(device), y.to(device)
        crit = build_loss(cfg, STATE["loaders"][3], device)
        STATE["model"].train()
        STATE["model"].zero_grad()
        loss = crit(STATE["model"](x), y)
        if not torch.isfinite(loss):
            raise AssertionError(f"loss is {loss.item()}")
        loss.backward()
        n_grad = sum(1 for p in STATE["model"].parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        n_all = sum(1 for _ in STATE["model"].parameters())
        if n_grad == 0:
            raise AssertionError("no parameter received a gradient")
        return f"loss={loss.item():.4f} ({cfg.loss})   {n_grad}/{n_all} tensors got gradient"

    # ---- 9. metrics -------------------------------------------------------
    @check("confusion matrix and Dice/IoU compute on the test split", requires=("model", "loaders"))
    def _():
        from src.evaluate import accumulate_confusion, metrics_from_confusion
        STATE["model"].eval()
        cm = accumulate_confusion(STATE["model"], STATE["loaders"][2], device, cfg.num_classes)
        m = metrics_from_confusion(cm, cfg.class_names)
        if cm.sum() == 0:
            raise AssertionError("confusion matrix is empty")
        per = "  ".join(f"{n}={m['per_class'][n]['dice']:.3f}" for n in cfg.class_names)
        return (f"pixel_acc={m['pixel_accuracy']:.4f}  mean_dice={m['mean_dice']:.4f}  "
                f"mean_dice_fg={m['mean_dice_fg']:.4f}\n           per-class dice: {per}\n"
                f"           (an untrained net scores ~{100*m['per_class'][cfg.class_names[0]]['pixel_share']:.1f}% "
                f"accuracy by guessing background - this is why we watch foreground Dice)")

    # ---- summary ----------------------------------------------------------
    print("\n" + "=" * 78)
    ok = sum(1 for _, p, _ in RESULTS if p is True)
    bad = [r for r in RESULTS if r[1] is False]
    skipped = [r for r in RESULTS if r[1] is None]
    for name, passed, detail in RESULTS:
        tag = "PASS" if passed is True else ("SKIP" if passed is None else "FAIL")
        print(f"  {tag}  {name}")
        if passed is not True:
            print(f"        {detail}")
    print("=" * 78)
    print(f"{ok}/{len(RESULTS)} passed"
          + (f", {len(bad)} failed" if bad else "")
          + (f", {len(skipped)} skipped" if skipped else ""))
    if bad:
        print(f"\nFix this first: {bad[0][0]}")
        if skipped:
            print("The skipped checks depend on it and will run once it passes.")
    elif not skipped:
        print("\nEverything works. The only thing not exercised here is a full "
              "training run —\nstart one with:  python -m src.train --arch attention_unet")
    return 0 if ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
