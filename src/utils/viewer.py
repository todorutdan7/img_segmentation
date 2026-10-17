
"""Interactive viewer for one patient: slices on the left, 3D on the right.

This is the INTERACTIVE version. Its sibling `view3d.py` is the batch version
that writes files and exits, and is the only one that can export `.nii.gz` for
3D Slicer.

HOW TO RUN IT

    python -m src.utils.viewer --group ablation_2070 --volume 16
    python -m src.utils.viewer --from-nifti 16      # reuse an earlier export
    python -m src.utils.viewer --volume 16 --save   # write PNGs, no window

WHAT YOU SEE
    left    radio buttons for the plane and the overlay, a colour key, and
            Dice for the slice you are looking at
    middle  the slice, with a slider to step through them
    right   the same patient in 3D, spun by dragging across it

WHAT IT LOADS
Only slices whose GROUND TRUTH contains liver, the same `has_foreground` rule
training and evaluation use. Most slices of a CT hold no liver at all; they
contribute no true positives, so all they can add is false positives floating
where the organ is not.

PIPELINE
    main()
      -> load_from_model()  or  from_nifti()      get ct, gt, pr as 3D arrays
      -> App(...)                                 build the window
           -> Turntable(...)                      precompute the 3D frames
           -> draw() / draw3()                    on every user action
      or
      -> save_figures(...)                        with --save, PNGs instead
"""

import argparse
from pathlib import Path

import numpy as np

# Colours as 0-255 RGB, because PIL and Tk both want bytes
# NOTE they are NOT 0.0-1.0 like matplotlib wants; save_figures divides by 255.
LIVER = (90, 153, 230)      # blue
TUMOR = (230, 64, 51)       # red
CORRECT = (120, 170, 235)   # truth and prediction agree
MISSED = (245, 160, 40)     # orange: real tissue the model did not find
FALSE = (215, 60, 200)      # magenta: the model invented it

MODES = ("ground truth", "prediction", "difference")
N_ANGLES = 36               # viewing angles, so one step is 10 degrees
DISP = 470                  # displayed width of the slice image, in pixels

# What each colour means, per mode. Drives both the on-screen key and the
# legend in the saved figures, so the two can never disagree.
KEY = {
    "ground truth": [(LIVER, "liver, drawn by the radiologist"),
                     (TUMOR, "tumour, drawn by the radiologist")],
    "prediction": [(LIVER, "liver, found by the model"),
                   (TUMOR, "tumour, found by the model")],
    "difference": [(CORRECT, "correct: the two agree"),
                   (MISSED, "missed: the model did not find it"),
                   (FALSE, "false alarm: nothing really there")],
}

# --------------------------------------------------------------------------
# 2D planes
# --------------------------------------------------------------------------

def plane_cut(vol, plane, i):
    """Take one 2D cut out of a 3D volume.

    IN   vol   - (z, y, x) array: ct, gt or pr
         plane - "axial", "coronal" or "sagittal"
         i     - which index along the fixed axis

    OUT  a 2D array

    axial is a slice exactly as the network saw it. coronal and sagittal are
    rebuilt from the STACK, cutting across many independent predictions at
    once, which is where slice-to-slice inconsistency shows up as a ragged edge.

    The [::-1] reverses z so the head ends up at the top of the picture instead
    of the bottom.
    """
    if plane == "axial":
        return vol[i]           # one whole slice
    if plane == "coronal":
        return vol[::-1, i, :]  # fix y, keep z and x
    return vol[::-1, :, i]      # fix x, keep z and y


def overlay_rgb(gray, truth, pred, mode, alpha):
    """Paint labels on top of a CT cut.

    IN   gray  - 2D CT cut, values 0-255
         truth - 2D label cut from the ground truth, values 0/1/2
         pred  - 2D label cut from the prediction, values 0/1/2
         mode  - one of MODES; decides what gets painted
         alpha - 0.0 = CT only, 1.0 = solid colour over it

    OUT  a (H, W, 3) uint8 RGB image

    In "difference" mode the two volumes are compared per pixel instead of
    being drawn separately, which is the view that actually shows where the
    model is wrong.
    """
    img = np.dstack([gray] * 3).astype(np.float32)   # grey -> RGB
    paint = np.zeros(img.shape, np.float32)          # colour to lay on top
    hit = np.zeros(gray.shape, bool)                 # where anything is painted

    if mode == "difference":
        g, p = truth > 0, pred > 0      # any foreground class, liver or tumor
        parts = ((g & p, CORRECT),      # both say yes
                 (g & ~p, MISSED),      # truth yes, model no
                 (~g & p, FALSE))       # model yes, truth no
    else:
        lab = truth if mode == "ground truth" else pred
        parts = ((lab == 1, LIVER), (lab == 2, TUMOR))

    for m, col in parts:
        paint[m] = col
        hit |= m
    # Blend only where something was painted, so untouched CT stays untouched.
    img[hit] = img[hit] * (1 - alpha) + paint[hit] * alpha
    return img.astype(np.uint8)


def display_size(plane, nz, side, z_scale):
    """How big to draw a plane so the body is not stretched.

    IN   plane   - which plane
         nz      - number of slices in the volume
         side    - 256, the in-plane size
         z_scale - thickness of one slice, in units of an in-plane pixel

    OUT  (width, height) in screen pixels

    A coronal cut is nz rows by `side` columns, but a row is one whole SLICE,
    which is z_scale in-plane pixels thick, not one. So its true height is
    nz * z_scale, not nz.

    Getting this upside down - using 1/z_scale - stretches the body by
    z_scale squared. At the 0.5 spacing this dataset works out to, that is a
    factor of four, and the liver comes out as a tall thin sliver.
    """
    if plane == "axial":
        return DISP, DISP               # square: 256 x 256
    return DISP, max(1, int(round(DISP * nz * z_scale / side)))


# --------------------------------------------------------------------------
# 3D, precomputed
# --------------------------------------------------------------------------

class Turntable:
    """The label volume projected into N viewing angles, once.

    HOW IT WORKS
    Every voxel of a class becomes a 3D point. For each viewing angle the
    points are rotated about the body axis and projected to a pixel, and the
    NEAREST one wins - that is the z-buffer. Depth is then turned into
    brightness, so the surface looks lit.

    USE CASE
    Built once in App.__init__, then frames() is called on every drag. The
    render cost is paid up front so that dragging costs nothing.
    """

    def __init__(self, gt, pr, z_scale=1.0, n_angles=N_ANGLES, pad=6):
        """IN  gt, pr - label volumes (z, y, x), values 0/1/2
              z_scale - slice thickness in in-plane pixels
              n_angles - how many angles to render; 36 = one step per 10 degrees
              pad     - blank border in pixels, so the shape never touches the edge
        """
        self.n_angles, self.pad = n_angles, pad
        g, p = gt > 0, pr > 0
        # The seven point clouds every mode is built from. Four for the plain
        # views, three for the difference view.
        masks = {"gl": gt == 1, "gt": gt == 2,      # ground truth liver / tumor
                 "pl": pr == 1, "pt": pr == 2,      # predicted liver / tumor
                 "ok": g & p, "miss": g & ~p, "bad": ~g & p}

        pts = {}
        for k, m in masks.items():
            z, y, x = np.nonzero(m)                 # coordinates of set voxels
            w = np.empty((len(z), 3), np.float32)
            # z is scaled here, once, so the rest of the class can treat all
            # three axes as being in the same unit.
            w[:, 0], w[:, 1], w[:, 2] = z * z_scale, y, x
            pts[k] = w

        # Centre everything on the middle of the whole patient, so rotation
        # spins about the organ instead of swinging it around the origin.
        allc = np.concatenate([v for v in pts.values() if len(v)], 0)
        centre = allc.mean(0)
        self.w = {k: (v - centre) if len(v) else v for k, v in pts.items()}

        # One bounding box shared by every mode. If each mode sized itself,
        # switching between them would silently rescale the organ and make the
        # comparison meaningless.
        live = [v for v in self.w.values() if len(v)]
        # radius = furthest point in the rotation plane, so nothing clips at
        # any angle. hypot(y, x) is sqrt(y*y + x*x).
        self.radius = max(float(np.hypot(v[:, 1], v[:, 2]).max()) for v in live)
        self.zmax = max(float(np.abs(v[:, 0]).max()) for v in live)
        self.cache = {}         # mode -> rendered frames, filled on demand

    def passes(self, mode):
        """What to draw for one mode, as [(layers, alpha), ...] back to front.

        Each layer is (points, colour). Layers inside one pass share a depth
        buffer; passes composite on top of each other in order.

        The plain views use two passes so the liver can be translucent (0.42)
        over a solid tumor. The difference view uses ONE pass, so the three
        colours compete on depth rather than on draw order.
        """
        w = self.w
        if mode == "ground truth":
            return [([(w["gt"], TUMOR)], 1.0), ([(w["gl"], LIVER)], 0.42)]
        if mode == "prediction":
            return [([(w["pt"], TUMOR)], 1.0), ([(w["pl"], LIVER)], 0.42)]
        return [([(w["ok"], CORRECT), (w["miss"], MISSED), (w["bad"], FALSE)], 1.0)]

    def frames(self, mode):
        """All rotation frames for one mode, rendered on first use and kept.

        IN   mode - one of MODES
        OUT  (n_angles, H, W, 3) uint8 array

        Roughly 2 seconds and 3 MB per mode. Cached, so switching back to a
        mode you have already looked at is instant.
        """
        if mode not in self.cache:
            self.cache[mode] = self._render(self.passes(mode))
        return self.cache[mode]

    def _render(self, passes):
        """Do the actual projection. Private; call frames() instead.

        IN   passes - from passes()
        OUT  (n_angles, H, W, 3) uint8
        """
        pad, R, Z = self.pad, self.radius, self.zmax
        # Frame big enough for the shape at any rotation, plus the border.
        W, H = int(2 * R) + 2 * pad, int(2 * Z) + 2 * pad
        out = np.empty((self.n_angles, H, W, 3), np.uint8)

        # Angles evenly around a full turn. endpoint=False so the last frame is
        # not a duplicate of the first, which would stutter when looping.
        for k, th in enumerate(np.linspace(0, 2 * np.pi, self.n_angles, False)):
            ct, st = np.cos(th), np.sin(th)
            img = np.full((H, W, 3), 255.0, np.float32)   # white; colours are 0-255

            for layers, alpha in passes:
                layers = [(w, c) for w, c in layers if len(w)]   # skip empties
                if not layers:
                    continue
                buf = np.full(H * W, np.inf, np.float32)   # depth per pixel
                who = np.zeros(H * W, np.int8)             # which layer won

                for tag, (w, _) in enumerate(layers, 1):
                    # Rotate about the body axis. Only y and x move; z is the
                    # height on screen and is untouched.
                    depth = w[:, 2] * st + w[:, 1] * ct     # towards the camera
                    sx = np.clip((w[:, 2] * ct - w[:, 1] * st + R + pad).astype(np.int32), 0, W - 1)
                    sy = np.clip((Z - w[:, 0] + pad).astype(np.int32), 0, H - 1)
                    flat = sy * W + sx      # 2D pixel -> index into the flat buffer

                    # Sort far to near, then write. The nearest write lands
                    # last and wins. np.minimum.at is the obvious way to build
                    # a z-buffer and is roughly fifty times slower.
                    o = np.argsort(-depth)
                    f, d = flat[o], depth[o]
                    near = d < buf[f]       # only where actually closer
                    buf[f[near]], who[f[near]] = d[near], tag

                hit = who > 0               # pixels this pass covered
                if not hit.any():
                    continue
                # Depth -> brightness: nearest is brightest. The 0.35 floor
                # stops the far side going fully black.
                lo, hi = float(buf[hit].min()), float(buf[hit].max())
                shade = np.zeros(H * W, np.float32)
                shade[hit] = 0.35 + 0.65 * np.clip(1 - (buf[hit] - lo) / max(hi - lo, 1e-6), 0, 1)

                # Palette lookup: row 0 is unused (who == 0 means "nothing"),
                # so pal[who] gives every pixel its layer's colour in one go.
                pal = np.zeros((len(layers) + 1, 3), np.float32)
                for tag, (_, col) in enumerate(layers, 1):
                    pal[tag] = col
                layer = (shade[:, None] * pal[who]).reshape(H, W, 3)
                m = (hit * alpha).reshape(H, W, 1)      # 0 outside, alpha inside
                img = img * (1 - m) + layer * m         # composite over

            out[k] = np.clip(img, 0, 255).astype(np.uint8)
        return out


# --------------------------------------------------------------------------
# numbers
# --------------------------------------------------------------------------

def slice_dice(truth, pred, cls):
    """Dice for one class on one 2D cut.

    IN   truth, pred - 2D label cuts
         cls         - 1 for liver, 2 for tumor
    OUT  float, or NaN when the class is in neither

    NaN rather than 0.0 on purpose: "this slice has no tumor" is not the same
    as "the model missed the tumor", and the readout prints "--" for it.
    """
    g, p = truth == cls, pred == cls
    tot = int(g.sum()) + int(p.sum())
    return (2 * int((g & p).sum()) / tot) if tot else float("nan")


def volume_stats(gt, pr):
    """Voxel counts over the whole volume, ignoring which class.

    IN   gt, pr - label volumes
    OUT  dict with correct / missed / false

    Computed ONCE in App.__init__, never per frame - it walks 12 million
    voxels and the answer does not change, so recomputing it on every redraw
    cost about 25 ms a frame for nothing.
    """
    g, p = gt > 0, pr > 0
    return dict(correct=int((g & p).sum()), missed=int((g & ~p).sum()),
                false=int((~g & p).sum()))


# --------------------------------------------------------------------------
# the window
# --------------------------------------------------------------------------

class App:
    """The tkinter window. Only the two image labels are ever redrawn.

    USE CASE
        App(ct, gt, pr, title, z_scale).run()

    To embed this in a larger application later, the only thing tying it to a
    standalone window is `root = tk.Tk()`; pointing it at a tk.Frame you own
    instead is a one-line change.
    """

    def __init__(self, ct, gt, pr, title, z_scale=1.0, n_angles=N_ANGLES):
        """IN  ct, gt, pr - (z, y, x) uint8 volumes from load_from_model/from_nifti
              title       - window title and heading
              z_scale     - slice thickness, for the coronal/sagittal aspect
              n_angles    - rotation frames to precompute
        """
        # Imported here, not at module level, so --save never needs a GUI
        # toolkit installed at all.
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image, ImageTk

        # Kept as attributes so the methods can use them without re-importing.
        self.tk, self.Image, self.ImageTk = tk, Image, ImageTk
        self.ct, self.gt, self.pr = ct, gt, pr
        self.nz, self.side = ct.shape[0], ct.shape[1]
        self.z_scale = z_scale
        self.turn = Turntable(gt, pr, z_scale, n_angles)   # the slow bit, once
        self.angle = 0

        root = tk.Tk()
        self.root = root
        root.title(title)
        root.configure(bg="#f4f4f4")

        # StringVars are how tkinter radio buttons report their selection.
        self.plane = tk.StringVar(value="axial")
        self.mode = tk.StringVar(value="difference")

        # Three columns: controls, slices, 3D.
        left = tk.Frame(root, bg="#f4f4f4", padx=12, pady=12)
        left.grid(row=0, column=0, sticky="n")
        mid = tk.Frame(root, bg="#f4f4f4", padx=6, pady=12)
        mid.grid(row=0, column=1)
        right = tk.Frame(root, bg="#f4f4f4", padx=12, pady=12)
        right.grid(row=0, column=2, sticky="n")

        # ---- left: controls ----------------------------------------------
        tk.Label(left, text=title, bg="#f4f4f4", font=("Segoe UI", 10, "bold"),
                 wraplength=250, justify="left").pack(anchor="w", pady=(0, 10))

        # Both radio groups are built by the same loop; they only differ in
        # which StringVar they write to and which callback they fire.
        for name, var, opts, cb in (("plane", self.plane,
                                     ("axial", "coronal", "sagittal"), self.on_plane),
                                    ("overlay", self.mode, MODES, self.on_mode)):
            tk.Label(left, text=name, bg="#f4f4f4",
                     font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(6, 0))
            for o in opts:
                tk.Radiobutton(left, text=o, value=o, variable=var, command=cb,
                               bg="#f4f4f4", anchor="w",
                               font=("Segoe UI", 9)).pack(anchor="w", fill="x")

        tk.Label(left, text="what the colours mean", bg="#f4f4f4",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(12, 2))
        # Empty frame; set_key() fills it and refills it when the mode changes.
        self.key = tk.Frame(left, bg="#f4f4f4")
        self.key.pack(anchor="w", fill="x")

        # Per-slice numbers, rewritten on every redraw.
        self.stat = tk.Label(left, bg="#f4f4f4", font=("Consolas", 9),
                             justify="left", anchor="w")
        self.stat.pack(anchor="w", pady=(12, 0))

        # Whole-volume numbers. Built once here because they never change.
        s = volume_stats(gt, pr)
        tk.Label(left, bg="#f4f4f4", font=("Consolas", 9), justify="left",
                 anchor="w",
                 text=("whole volume\n"
                       f"  correct {s['correct']:>9,}\n"
                       f"  missed  {s['missed']:>9,}\n"
                       f"  false   {s['false']:>9,}")
                 ).pack(anchor="w", pady=(10, 0))

        # ---- middle: slices ----------------------------------------------
        self.head = tk.Label(mid, bg="#f4f4f4", font=("Segoe UI", 10))
        self.head.pack()
        self.img = tk.Label(mid, bg="black", bd=0)   # the slice image lives here
        self.img.pack()

        # Both sliders call on_idx: one changes the slice, the other the
        # overlay strength, and either way the picture is rebuilt the same way.
        self.s_idx = tk.Scale(mid, from_=0, to=self.nz - 1, orient="horizontal",
                              label="slice", length=DISP, bg="#f4f4f4",
                              highlightthickness=0, command=self.on_idx)
        self.s_idx.pack(pady=(10, 0))
        self.s_alpha = tk.Scale(mid, from_=0, to=100, orient="horizontal",
                                label="overlay %", length=DISP, bg="#f4f4f4",
                                highlightthickness=0, command=self.on_idx)
        self.s_alpha.set(55)
        self.s_alpha.pack()

        # ---- right: 3D ---------------------------------------------------
        tk.Label(right, text="drag across to spin", bg="#f4f4f4",
                 font=("Segoe UI", 10)).pack()
        self.img3 = tk.Label(right, bg="white", bd=0)
        self.img3.pack()
        # Mouse-down remembers where the drag started; motion turns horizontal
        # movement into a change of angle.
        self.img3.bind("<Button-1>", self.on_press)
        self.img3.bind("<B1-Motion>", self.on_drag)
        self.s_ang = tk.Scale(right, from_=0, to=n_angles - 1, orient="horizontal",
                              label="angle", length=420, bg="#f4f4f4",
                              highlightthickness=0, command=self.on_angle)
        self.s_ang.pack(pady=(10, 0))

        # ---- initial state -----------------------------------------------
        self.drag_x = None
        self.idx = self.default_slice()
        self.s_idx.set(self.idx)
        self.set_key()
        self.draw3()
        self.draw()

    # -- helpers -------------------------------------------------------
    def default_slice(self):
        """Which slice to open on: the middle of the tumor, else of the liver.

        OUT  an index into the z axis

        Opening on slice 0 usually shows the edge of the organ, where there is
        nothing interesting and Dice is often 0.
        """
        target = 2 if (self.gt == 2).any() else 1
        z = np.where((self.gt == target).any(axis=(1, 2)))[0]
        # median, not mean: guaranteed to be an index that actually contains
        # the class, even for an awkward shape.
        return int(np.median(z)) if len(z) else self.nz // 2

    def max_idx(self):
        """Highest valid slider value, which depends on the plane.

        Axial runs over z (n_slices), the other two over an in-plane axis (256).
        """
        return (self.nz if self.plane.get() == "axial" else self.side) - 1

    def show(self, label, arr, size, smooth=False):
        """Put a numpy image into a tkinter Label.

        IN   label  - the tk.Label to update
             arr    - (H, W, 3) uint8
             size   - (width, height) to scale to
             smooth - BILINEAR when upscaling the 3D frames; NEAREST for
                      slices, so a label pixel stays one crisp colour

        `label.image = photo` is not a typo and not dead code. Tk does not keep
        a reference to the image itself, so without this line Python garbage
        collects the photo and the label goes blank.
        """
        im = self.Image.fromarray(arr).resize(
            size, self.Image.BILINEAR if smooth else self.Image.NEAREST)
        photo = self.ImageTk.PhotoImage(im)
        label.configure(image=photo)
        label.image = photo         # keep a reference; see docstring

    def set_key(self):
        """Rebuild the colour key for the current mode.

        Destroys the old rows and makes new ones, because the difference mode
        has three entries and the others have two.
        """
        for w in self.key.winfo_children():
            w.destroy()
        for col, text in KEY[self.mode.get()]:
            row = self.tk.Frame(self.key, bg="#f4f4f4")
            row.pack(anchor="w", fill="x", pady=1)
            # A tiny Canvas is the simplest way to get a plain colour swatch.
            # "#%02x%02x%02x" % col turns (90, 153, 230) into "#5a99e6".
            self.tk.Canvas(row, width=13, height=13, bg="#%02x%02x%02x" % col,
                           highlightthickness=0).pack(side="left", padx=(0, 6))
            self.tk.Label(row, text=text, bg="#f4f4f4", font=("Segoe UI", 8),
                          anchor="w").pack(side="left")

    # -- callbacks -----------------------------------------------------
    def on_plane(self):
        """Plane radio button. Resets the slider range, which differs per plane."""
        self.idx = self.side // 2 if self.plane.get() != "axial" else self.default_slice()
        self.s_idx.configure(to=self.max_idx())
        self.s_idx.set(self.idx)
        self.draw()

    def on_mode(self):
        """Overlay radio button. Redraws both panels and the colour key.

        The first time a mode is chosen its 3D frames have to be rendered, so
        there is a short pause; after that it is cached.
        """
        self.set_key()
        self.draw3()
        self.draw()

    def on_idx(self, _=None):
        """Slice or overlay slider. tk passes the value as a string, ignored."""
        self.idx = int(self.s_idx.get())
        self.draw()

    def on_angle(self, _=None):
        """Angle slider. Only the 3D panel changes."""
        self.angle = int(self.s_ang.get()) % self.turn.n_angles
        self.draw3()

    def on_press(self, ev):
        """Mouse down on the 3D image: remember where the drag started."""
        self.drag_x = ev.x

    def on_drag(self, ev):
        """Mouse drag: convert horizontal movement into rotation.

        Six pixels per step, so a normal drag sweeps the whole turn. Setting
        the slider fires on_angle, which does the drawing - so the slider and
        the image can never disagree.
        """
        if self.drag_x is None:
            return
        step = int((ev.x - self.drag_x) / 6)
        if step:
            self.drag_x = ev.x      # reset, so movement accumulates smoothly
            self.angle = (self.angle + step) % self.turn.n_angles
            self.s_ang.set(self.angle)

    # -- drawing -------------------------------------------------------
    def draw(self):
        """Rebuild the slice panel: image, heading and per-slice numbers."""
        plane, mode = self.plane.get(), self.mode.get()
        # Clamp, because the slider range may not have caught up after a
        # plane change.
        i = min(self.idx, self.max_idx())
        rgb = overlay_rgb(plane_cut(self.ct, plane, i),
                          plane_cut(self.gt, plane, i),
                          plane_cut(self.pr, plane, i),
                          mode, self.s_alpha.get() / 100)
        self.show(self.img, rgb, display_size(plane, self.nz, self.side, self.z_scale))

        where = {"axial": "z", "coronal": "y", "sagittal": "x"}[plane]
        self.head.configure(text=f"{plane}   {where} = {i}")

        # Per-slice table. Only the numbers for THIS cut; the whole-volume
        # counts were built once in __init__.
        t, p = plane_cut(self.gt, plane, i), plane_cut(self.pr, plane, i)
        lines = [f"this {plane} cut", f"{'':7}{'true':>8}{'pred':>8}{'Dice':>7}"]
        for cls, name in ((1, "liver"), (2, "tumour")):
            d = slice_dice(t, p, cls)
            lines.append(f"{name:<7}{int((t == cls).sum()):>8,}"
                         f"{int((p == cls).sum()):>8,}"
                         f"{('   -- ' if np.isnan(d) else f'{d:6.3f}'):>7}")
        self.stat.configure(text="\n".join(lines))

    def draw3(self):
        """Show one precomputed rotation frame. No rendering happens here."""
        fr = self.turn.frames(self.mode.get())[self.angle]
        h, w = fr.shape[:2]
        # Fixed width, height follows the frame's own proportions.
        self.show(self.img3, fr, (420, max(1, int(round(420 * h / w)))), smooth=True)

    def run(self):
        """Hand control to tkinter. Returns when the window is closed."""
        self.root.mainloop()


# --------------------------------------------------------------------------
# static figures, for the report
# --------------------------------------------------------------------------

def save_figures(ct, gt, pr, out_dir, tag, title, z_scale=1.0, dpi=130):

    """Write the same two views as PNGs, with no window.

    IN   ct, gt, pr - the volumes
         out_dir    - where to write
         tag        - e.g. "volume16", used in the filenames
         title      - heading on the slices figure
         z_scale    - slice thickness, for the aspect
         dpi        - resolution

    OUT  (path_to_slices_png, path_to_3d_png); also prints them

    USE CASE
    Called by --save. This is what produces the figures in the report, and it
    deliberately uses the same overlay_rgb and Turntable the window uses, so a
    figure can never show something different from what you saw on screen.
    """

    import matplotlib
    matplotlib.use("Agg", force=True)   # no GUI needed; force in case one is set
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    side = ct.shape[1]
    # Centre the axial cut on the tumor if there is one, else on the liver.
    zc = int(np.median(np.where((gt == (2 if (gt == 2).any() else 1)).any(axis=(1, 2)))[0]))
    # matplotlib's aspect = z_scale for the rebuilt planes. See display_size()
    # for why it is z_scale and not 1/z_scale.
    aspect = {"axial": 1.0, "coronal": z_scale, "sagittal": z_scale}

    # Axial is square; with the aspect right the other two are about three
    # times wider than tall. So stack those beside the axial rather than
    # putting all three in a row and letterboxing two of them.
    fig = plt.figure(figsize=(11, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.55], wspace=0.12, hspace=0.35)
    for plane, i, cell in (("axial", zc, gs[:, 0]), ("coronal", side // 2, gs[0, 1]),
                           ("sagittal", side // 2 + 12, gs[1, 1])):
        ax = fig.add_subplot(cell)
        ax.imshow(overlay_rgb(plane_cut(ct, plane, i), plane_cut(gt, plane, i),
                              plane_cut(pr, plane, i), "difference", 0.55),
                  aspect=aspect[plane], interpolation="nearest")
        where = {"axial": "z", "coronal": "y", "sagittal": "x"}[plane]
        ax.set_title(f"{plane}   ({where} = {i})", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    # KEY holds 0-255 colours; matplotlib wants 0.0-1.0, hence the /255.
    fig.legend(handles=[Patch(fc=np.array(c) / 255, label=l)
                        for c, l in KEY["difference"]],
               loc="lower center", ncol=3, frameon=False, fontsize=9.5)
    fig.suptitle(title, fontsize=11.5)
    f1 = out_dir / f"viewer_{tag}_slices.png"
    fig.savefig(f1, dpi=dpi, facecolor="white"); plt.close(fig)
    print(f"[saved] {f1}")

    # Second figure: three modes across, two rotation angles down.
    turn = Turntable(gt, pr, z_scale)
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.4))
    for r, k in enumerate((0, 9)):          # frame 0 and frame 9 = 0 and 90 degrees
        for c, mode in enumerate(MODES):
            ax = axes[r][c]
            ax.imshow(turn.frames(mode)[k], interpolation="bilinear")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)       # no box around a floating shape
            if r == 0:
                ax.set_title(mode, fontsize=11, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"rotated {k * 360 // N_ANGLES}°", fontsize=10)

    # One legend for all five colours. Modes share colours (liver appears in
    # both plain modes), so dedupe on the colour and shorten the labels:
    # "liver, drawn by the radiologist" -> "liver", "correct: ..." -> "correct".
    seen, handles = set(), []
    for mode in MODES:
        for col, lab in KEY[mode]:
            if col not in seen:
                seen.add(col)
                handles.append(Patch(fc=np.array(col) / 255, label=lab.split(",")[0].split(":")[0]))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9.5)
    fig.suptitle("The same patient stacked into 3D, two viewing angles", fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    f2 = out_dir / f"viewer_{tag}_3d.png"
    fig.savefig(f2, dpi=dpi, facecolor="white"); plt.close(fig)
    print(f"[saved] {f2}")
    return f1, f2

# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def from_nifti(tag, results, fg_only=True):
    """Rebuild the volumes from an earlier `view3d --nifti` export.

    IN   tag     - e.g. "volume16"
         results - the results/ directory
         fg_only - drop slices with no liver in the ground truth

    OUT  (ct, gt, pr, z_scale)

    USE CASE
    --from-nifti. Skips the model and the dataset entirely, so it starts in a
    second and works on a machine with no GPU, no torch and no LiTS download.
    Good for demoing, or for looking at the exact volumes you loaded in Slicer.
    """
    import nibabel as nib
    vols = []
    for name in ("ct", "truth", "pred"):
        f = results / f"view3d_{tag}_{name}.nii.gz"
        if not f.exists():
            raise SystemExit(f"missing {f}\nrun view3d with --nifti first")
        # NIfTI stores (x, y, z); everything here works in (z, y, x).
        vols.append(np.transpose(np.asanyarray(nib.load(f).dataobj), (2, 1, 0))
                    .astype(np.uint8))
    # The slice thickness view3d baked into the affine when it wrote the file.
    z = float(nib.load(results / f"view3d_{tag}_ct.nii.gz").affine[2, 2])
    ct, gt, pr = vols
    if fg_only:
        keep = (gt > 0).any(axis=(1, 2))    # one bool per slice
        print(f"[data] keeping {int(keep.sum())} of {len(keep)} slices")
        ct, gt, pr = ct[keep], gt[keep], pr[keep]
    return ct, gt, pr, z


def load_from_model(a):
    """Run the trained model over one patient and return the volumes.

    IN   a - the parsed argparse namespace (uses .ckpt, .group, .volume,
             .all_slices, .z_scale)

    OUT  (ct, gt, pr, z_scale, tag, title)

    This is the normal path. It reuses patient_pairs and build_volume from
    view3d.py rather than duplicating them, so both tools always segment in
    exactly the same way.
    """
    from src.config import get_device, RUNS_DIR, DATA_DIR
    from src.data import find_pairs, has_foreground
    from src.model import load_checkpoint
    from src.utils.view3d import patient_pairs, build_volume
    import random

    device = get_device()
    # Default to the first checkpoint in the group. next(iter(...), None)
    # returns None instead of raising when the folder is empty.
    ckpt = Path(a.ckpt) if a.ckpt else next(
        iter(sorted((Path(RUNS_DIR) / a.group).glob("*/best_model.pt"))), None)
    if ckpt is None:
        raise SystemExit(f"no checkpoint under {Path(RUNS_DIR) / a.group}")
    # The checkpoint carries its own config, so the split below is the one this
    # model was actually trained with.
    model, cfg, ck = load_checkpoint(ckpt, device)
    print(f"[load] {ckpt}  arch={cfg.arch}  epoch={ck.get('epoch')}")

    pairs, _, _ = find_pairs(Path(cfg.root_data_dir or DATA_DIR))
    _, by = patient_pairs(pairs)
    # Reproduce training's patient shuffle from the same seed, so we can warn
    # if this patient was in the training set - it would look great and prove
    # nothing. sorted() first, or the shuffle is not reproducible.
    vols = sorted(by)
    shuffled = list(vols)
    random.Random(cfg.seed).shuffle(shuffled)
    test_vols = shuffled[:max(1, round(len(vols) * cfg.test_fraction))]

    key, slices = patient_pairs(pairs, a.volume or test_vols[0].split("-")[1])
    held, total = key in test_vols, len(slices)
    if not a.all_slices:
        slices = [(ip, mp) for ip, mp in slices if has_foreground(mp)]
    print(f"[data] {key}: {len(slices)} of {total} slices   "
          f"{'held-out test patient' if held else 'WARNING: seen in training'}")

    print("[run ] segmenting ...")
    ct, gt, pr = build_volume(model, slices, device, cfg.image_size)
    # Estimate slice thickness from the ORIGINAL slice count - dropping empty
    # slices does not make the remaining ones thicker. See view3d.py for the
    # 320 / n_slices reasoning.
    z = a.z_scale or 320.0 / max(total, 1)
    return ct, gt, pr, z, key.replace("-", ""), f"{key}  -  {cfg.arch}  -  {len(slices)} slices"


def main():
    """Parse the command line, load a patient, then open the window or save PNGs."""
    from src.config import ROOT

    p = argparse.ArgumentParser(description="viewer for one patient")
    p.add_argument("--group", default="ablation_2070")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--volume", default=None)
    p.add_argument("--from-nifti", metavar="VOL", default=None,
                   help="skip the model, reuse results/view3d_volume<VOL>_*.nii.gz")
    p.add_argument("--all-slices", action="store_true",
                   help="also load slices with no liver in the ground truth")
    p.add_argument("--angles", type=int, default=N_ANGLES,
                   help="viewing angles to precompute, about 2 s per 36")
    p.add_argument("--z-scale", type=float, default=None)
    p.add_argument("--save", action="store_true", help="write PNGs, no window")
    a = p.parse_args()

    results = Path(ROOT) / "results"
    results.mkdir(parents=True, exist_ok=True)

    # Two ways in: straight from the NIfTI export, or by running the model.
    if a.from_nifti:
        tag = a.from_nifti if str(a.from_nifti).startswith("volume") else f"volume{a.from_nifti}"
        ct, gt, pr, z = from_nifti(tag, results, not a.all_slices)
        z = a.z_scale or z          # let --z-scale override what the file said
        title = f"{tag}  -  liver and tumour segmentation"
    else:
        ct, gt, pr, z, tag, title = load_from_model(a)

    s = volume_stats(gt, pr)
    print(f"[vol ] correct {s['correct']:,}  missed {s['missed']:,}  "
          f"false {s['false']:,}")

    if a.save:
        save_figures(ct, gt, pr, results, tag, title, z)
        return

    # The pause is Turntable rendering the first mode, inside App.__init__.
    print(f"[3d  ] precomputing {a.angles} views ...")
    App(ct, gt, pr, title, z, a.angles).run()


if __name__ == "__main__":
    main()
