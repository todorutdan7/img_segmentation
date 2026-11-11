"""Give the run folders readable names, built from their settings
(src.config.readable_run_name): runs/nifti/r50/unet_r50_1793825576 becomes
runs/nifti/r50/unet_r50_2.5d_lr1e-4.

    python -m src.utils.rename_runs            # show what would change
    python -m src.utils.rename_runs --apply    # do it

The old -> new names are kept in runs/renamed.json. On the training machine,
after `git pull` has moved the tracked files, run it with --apply once more:
it moves what git does not track (best_model.pt) into the new folders and
removes the empty old ones. Running it again changes nothing.
"""

import argparse
import json
from pathlib import Path

from src.config import RUNS_DIR, RESULTS_DIR, readable_run_name

MAP = Path(RUNS_DIR) / "renamed.json"


def plan(mapping):
    """add every run whose folder name is not its readable name yet"""
    taken = {Path(RUNS_DIR, v) for v in mapping.values()}
    for h in sorted(Path(RUNS_DIR).glob("*/*/*/history.json")):
        d = h.parent
        rel = d.relative_to(RUNS_DIR).as_posix()
        if rel in mapping:
            continue
        cfg = json.loads(h.read_text()).get("config", {})
        if "arch" not in cfg:
            continue
        name = readable_run_name(cfg)
        if name == d.name:
            continue
        new, i = d.parent / name, 2
        while new.exists() or new in taken:          # same settings twice: _2, _3 ...
            if new == d:
                break
            new, i = d.parent / f"{name}_{i}", i + 1
        if new == d:
            continue
        taken.add(new)
        mapping[rel] = new.relative_to(RUNS_DIR).as_posix()
    return mapping


def move(old: Path, new: Path):
    if not old.exists():
        return "done before"
    if not new.exists():
        old.rename(new)
        return "renamed"
    # git already made the new folder: bring over what it does not track
    left = []
    for f in old.iterdir():
        if (new / f.name).exists():
            left.append(f.name)
        else:
            f.rename(new / f.name)
    if left:
        return f"moved the rest; {old} still holds {left}, also in the new folder - check by hand"
    old.rmdir()
    return "moved the untracked files"


def fix_history(new: Path, old_name: str):
    """the run name saved in history.json; only that one line changes"""
    h = new / "history.json"
    if not h.exists():
        return
    b = h.read_bytes()
    b2 = b.replace(f'"run_name": "{old_name}"'.encode(), f'"run_name": "{new.name}"'.encode())
    if b2 != b:
        h.write_bytes(b2)


def rename_results(old_name, new_name):
    """Grad-CAM files are named after the run"""
    for f in Path(RESULTS_DIR).glob(f"*/gradcam/{old_name}.*"):
        g = f.with_name(new_name + f.suffix)
        if g.exists():
            continue
        if f.suffix == ".txt":
            g.write_bytes(f.read_bytes().replace(old_name.encode(), new_name.encode()))
            f.unlink()
        else:
            f.rename(g)
        print(f"   {f.relative_to(RESULTS_DIR)} -> {g.name}")


def main():
    p = argparse.ArgumentParser(description="readable names for the run folders")
    p.add_argument("--apply", action="store_true", help="rename (without it: only show)")
    a = p.parse_args()

    mapping = json.loads(MAP.read_text()) if MAP.exists() else {}
    mapping = plan(dict(mapping))
    if not mapping:
        print("nothing to rename")
        return
    w = max(len(k) for k in mapping)
    for old, new in mapping.items():
        state = "" if Path(RUNS_DIR, old).exists() else "   (done)"
        print(f"{old:<{w}}  ->  {new.split('/')[-1]}{state}")
    if not a.apply:
        print("\nnothing changed. Add --apply to rename.")
        return

    print()
    for old, new in mapping.items():
        o, n = Path(RUNS_DIR, old), Path(RUNS_DIR, new)
        r = move(o, n)
        if r != "done before":
            print(f"{old}: {r}")
        fix_history(n, o.name)
        rename_results(o.name, n.name)
    MAP.write_text(json.dumps(mapping, indent=2) + "\n")
    print(f"\n[saved] {MAP}")


if __name__ == "__main__":
    main()
