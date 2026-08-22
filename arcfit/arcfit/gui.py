"""The click-based digitiser.

Structured so that everything that can change the data lives in
``DigitizerState``, a plain object with no matplotlib in it, and the window is a
thin layer of event handlers over that. The interactive window cannot be
exercised on a headless machine, but the state machine -- which is where the
data-losing bugs would be -- is fully testable.

Design decisions that come from the material rather than from taste:

* Images are ~15 MB. They are decoded at reduced resolution for display via
  JPEG draft mode, while snapping re-reads full-resolution crops. Coordinates
  are always stored in full-resolution pixels, so the display shortcut never
  reaches the data.
* Grey stone on grey sand is genuinely hard to see. Contrast stretch, gamma and
  an edge overlay are provided, and they affect display only.
* Work on 167 photographs will be interrupted. Every object saves to its own
  file the moment it is finished, and the session resumes where it left off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .calibrate import Calibration, CalibrationError
from .fitting import FitError, EllipseParams, fit_circle_algebraic, fit_ellipse
from .records import ObjectRecord, records_dir, scan_images
from .scalecard import CardSpec, detect_card
from .snap import gradient_magnitude, snap_points

__all__ = ["DigitizerState", "Digitizer", "run_digitizer", "KEY_HELP"]


KEY_HELP = [
    ("left click", "add an outline point (or place a calibration corner)"),
    ("right click", "undo the last point"),
    ("n / p", "next / previous object"),
    ("c", "clear all points on this object"),
    ("d", "re-run automatic scale-card detection"),
    ("m", "manual calibration: click the 4 card corners"),
    ("t", "two-point calibration fallback (NOT rectified)"),
    ("k", "copy the previous object's calibration"),
    ("s", "toggle edge snapping"),
    ("e", "toggle the edge overlay"),
    ("g", "cycle display contrast"),
    ("o", "toggle: is the scale card resting ON the object (not the ground)?"),
    ("x", "exclude / re-include this object"),
    ("h", "show this help"),
    ("q", "save and quit"),
]


@dataclass
class DigitizerState:
    """All mutable digitising state, free of any UI framework."""

    image_paths: List[Path]
    workdir: Path
    card_spec: CardSpec = field(default_factory=CardSpec)
    index: int = 0
    snap_enabled: bool = True
    snap_search_px: float = 6.0
    operator: str = ""
    mode: str = "digitise"                 # digitise | card_corners | two_point
    pending_clicks: List[List[float]] = field(default_factory=list)
    two_point_cm: float = 10.0
    record: Optional[ObjectRecord] = None
    message: str = ""
    _gradient: Optional[np.ndarray] = field(default=None, repr=False)
    _gradient_for: Optional[str] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    def __post_init__(self):
        self.image_paths = [Path(p) for p in self.image_paths]
        self.workdir = Path(self.workdir)
        if self.image_paths:
            self.load_current()

    @property
    def image_path(self) -> Path:
        return self.image_paths[self.index]

    @property
    def n_images(self) -> int:
        return len(self.image_paths)

    def progress(self) -> str:
        done = len(list(records_dir(self.workdir).glob("*.json"))) \
            if records_dir(self.workdir).is_dir() else 0
        return f"{self.index + 1}/{self.n_images} (records saved: {done})"

    # ------------------------------------------------------------------
    def load_current(self) -> ObjectRecord:
        self.record = ObjectRecord.for_image(self.image_path, self.workdir)
        if self.operator and not self.record.operator:
            self.record.operator = self.operator
        self.mode = "digitise"
        self.pending_clicks = []
        return self.record

    def save(self) -> Optional[Path]:
        if self.record is None:
            return None
        self.record.snap_used = self.snap_enabled
        if self.operator:
            self.record.operator = self.operator
        return self.record.save(self.workdir)

    def go_to(self, index: int, save_first: bool = True) -> None:
        if save_first and self.record is not None and (
                self.record.points_px or self.record.calibration.is_calibrated):
            self.save()
        self.index = int(np.clip(index, 0, self.n_images - 1))
        self.load_current()

    def next_image(self) -> None:
        if self.index < self.n_images - 1:
            self.go_to(self.index + 1)
        else:
            self.message = "already at the last object"

    def prev_image(self) -> None:
        if self.index > 0:
            self.go_to(self.index - 1)
        else:
            self.message = "already at the first object"

    def next_undigitised(self) -> bool:
        """Jump to the next object that is not finished. Returns False if none."""
        for i in range(self.index + 1, self.n_images):
            rec = ObjectRecord.for_image(self.image_paths[i], self.workdir,
                                         hash_image=False)
            if rec.status() != "ready":
                self.go_to(i)
                return True
        return False

    # ------------------------------------------------------------------
    def _normal_at(self, x: float, y: float) -> Optional[np.ndarray]:
        """Outward direction to snap along at a proposed point.

        Uses the radial direction from the provisional centre of the points so
        far. For an outline that is roughly round -- which is the premise of the
        whole analysis -- that is close to the true surface normal, and it
        degrades gracefully as points accumulate rather than needing a fit.
        """
        pts = np.asarray(self.record.points_px, float) if self.record.points_px else None
        if pts is None or len(pts) < 2:
            return None
        if len(pts) >= 3:
            try:
                c = fit_circle_algebraic(pts[:, 0], pts[:, 1])
                centre = np.array([c.cx, c.cy])
            except (FitError, np.linalg.LinAlgError):
                centre = pts.mean(axis=0)
        else:
            centre = pts.mean(axis=0)
        v = np.array([x, y], float) - centre
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else None

    def _gradient_image(self) -> Optional[np.ndarray]:
        """Cached gradient magnitude of the current image, at full resolution."""
        key = str(self.image_path)
        if self._gradient_for == key and self._gradient is not None:
            return self._gradient
        try:
            from PIL import Image
            with Image.open(self.image_path) as im:
                grey = np.asarray(im.convert("L"), float)
        except Exception:
            return None
        self._gradient = gradient_magnitude(grey)
        self._gradient_for = key
        return self._gradient

    def add_point(self, x: float, y: float) -> Tuple[float, float]:
        """Add an outline point, snapping it to the nearest edge if enabled."""
        raw = [float(x), float(y)]
        snapped = list(raw)
        if self.snap_enabled:
            normal = self._normal_at(x, y)
            mag = self._gradient_image() if normal is not None else None
            if normal is not None and mag is not None:
                moved, ok = snap_points(None, np.array([raw]), normal[None, :],
                                        search_px=self.snap_search_px, mag=mag)
                if ok[0]:
                    snapped = [float(moved[0, 0]), float(moved[0, 1])]
        self.record.points_px.append(snapped)
        self.record.points_px_raw.append(raw)
        return tuple(snapped)

    def undo(self) -> bool:
        if self.mode in ("card_corners", "two_point") and self.pending_clicks:
            self.pending_clicks.pop()
            return True
        if self.record.points_px:
            self.record.points_px.pop()
            if self.record.points_px_raw:
                self.record.points_px_raw.pop()
            return True
        return False

    def clear_points(self) -> None:
        self.record.points_px = []
        self.record.points_px_raw = []

    def toggle_card_on_object(self) -> bool:
        """Record whether the card sits on the object rather than the ground.

        This flips the sign of the parallax term. With the card on the ground it
        defines a plane *below* the traced outline and sizes read slightly
        large; with the card resting on the object it is at or above that
        outline and they read small. Left unrecorded, a mixed set turns a
        correctable constant bias into irreducible scatter.
        """
        self.record.card_on_object = not self.record.card_on_object
        return self.record.card_on_object

    def toggle_excluded(self, reason: str = "") -> bool:
        self.record.excluded = not self.record.excluded
        self.record.exclude_reason = reason if self.record.excluded else ""
        return self.record.excluded

    # ------------------------------------------------------------------
    def auto_calibrate(self) -> bool:
        """Locate the scale card and build a rectified calibration."""
        try:
            det = detect_card(self.image_path, self.card_spec)
        except (RuntimeError, FileNotFoundError) as exc:
            self.message = f"detection unavailable: {exc}"
            return False
        if det is None:
            self.message = "no scale card found - press m to calibrate manually"
            return False
        try:
            cal = Calibration.from_rect(
                det.corners_px, self.card_spec.width_cm, self.card_spec.height_cm,
                source="auto", image_path=self.image_path,
                image_size_px=self.record.image_size_px,
                detection_score=det.score,
                extra_px=det.interior_px, extra_cm=det.interior_cm)
        except CalibrationError as exc:
            self.message = f"calibration failed: {exc}"
            return False
        self.record.calibration = cal
        conf = "confident" if det.is_confident else "LOW CONFIDENCE - check it"
        self.message = f"card found ({det.method}, score {det.score:.2f}) - {conf}"
        return True

    def start_manual_calibration(self) -> None:
        self.mode = "card_corners"
        self.pending_clicks = []
        self.message = "click the 4 corners of the scale card, starting at one end of the long side"

    def start_two_point(self, known_cm: Optional[float] = None) -> None:
        self.mode = "two_point"
        self.pending_clicks = []
        if known_cm:
            self.two_point_cm = float(known_cm)
        self.message = (f"click 2 points {self.two_point_cm:g} cm apart "
                        "- WARNING: this does not correct perspective")

    def add_calibration_click(self, x: float, y: float) -> bool:
        """Feed a click into whichever calibration mode is active.

        Returns True once the calibration has been completed.
        """
        self.pending_clicks.append([float(x), float(y)])
        need = 4 if self.mode == "card_corners" else 2
        if len(self.pending_clicks) < need:
            return False
        try:
            if self.mode == "card_corners":
                self.record.calibration = Calibration.from_rect(
                    self.pending_clicks, self.card_spec.width_cm,
                    self.card_spec.height_cm, source="manual",
                    image_path=self.image_path,
                    image_size_px=self.record.image_size_px)
                self.message = "manual rectified calibration set"
            else:
                self.record.calibration = Calibration.from_two_points(
                    self.pending_clicks[0], self.pending_clicks[1],
                    self.two_point_cm, source="manual")
                self.message = "two-point scale set (NOT rectified)"
        except CalibrationError as exc:
            self.message = f"calibration failed: {exc}"
            self.pending_clicks = []
            self.mode = "digitise"
            return False
        self.pending_clicks = []
        self.mode = "digitise"
        return True

    def copy_previous_calibration(self) -> bool:
        """Reuse the previous object's calibration.

        Only sound when the camera and card did not move between shots. These
        are field photographs taken one object at a time, so it usually is not
        -- hence the explicit warning recorded on the calibration itself.
        """
        if self.index == 0:
            self.message = "no previous object to copy from"
            return False
        prev = ObjectRecord.for_image(self.image_paths[self.index - 1], self.workdir)
        if not prev.calibration.is_calibrated:
            self.message = "the previous object has no calibration"
            return False
        cal = Calibration.from_dict(prev.calibration.to_dict())
        cal.source = "copied"
        cal.notes = ((cal.notes + "; ") if cal.notes else "") + \
            f"copied from {prev.object_id}: only valid if the camera did not move"
        self.record.calibration = cal
        self.message = f"copied calibration from {prev.object_id} - verify it fits this photo"
        return True

    # ------------------------------------------------------------------
    def preview_ellipse(self) -> Optional[EllipseParams]:
        """Live fit in pixel space, for the on-screen preview only."""
        if not self.record or len(self.record.points_px) < 5:
            return None
        pts = np.asarray(self.record.points_px, float)
        try:
            return fit_ellipse(pts[:, 0], pts[:, 1])
        except (FitError, np.linalg.LinAlgError):
            return None

    def preview_metrics(self) -> Optional[dict]:
        """Approximate measurements for the live readout.

        Fitted in pixels then converted, which is exact for a two-point scale and
        a good approximation for a homography over the object's extent. The
        reported results are always fitted properly in rectified coordinates --
        this is only to give the operator feedback while clicking.
        """
        if not self.record or len(self.record.points_px) < 6:
            return None
        if not self.record.calibration.is_calibrated:
            return None
        try:
            pts = self.record.points_cm()
            ell = fit_ellipse(pts[:, 0], pts[:, 1])
        except (FitError, CalibrationError, np.linalg.LinAlgError):
            return None
        from .fitting import arc_coverage
        cov = arc_coverage(ell, pts[:, 0], pts[:, 1])
        return {"major_axis_cm": ell.major_axis, "minor_axis_cm": ell.minor_axis,
                "eccentricity": ell.eccentricity, "coverage_deg": cov["coverage_deg"]}


# --------------------------------------------------------------------------
# the window
# --------------------------------------------------------------------------

class Digitizer:
    """Matplotlib window wrapping a DigitizerState."""

    def __init__(self, state: DigitizerState, display_max_px: int = 1600,
                 auto_detect: bool = True):
        import matplotlib.pyplot as plt

        self.state = state
        self.display_max_px = int(display_max_px)
        self.auto_detect = auto_detect
        self.show_edges = False
        self.contrast_mode = 0
        self._display = None
        self._scale = 1.0

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.load()

    # -- image handling ------------------------------------------------
    def _load_display_image(self):
        """Decode at reduced size; 15 MB per image makes full decode wasteful."""
        from PIL import Image

        path = self.state.image_path
        with Image.open(path) as im:
            full_w, full_h = im.size
            im.draft("RGB", (self.display_max_px, self.display_max_px))
            im = im.convert("RGB")
            arr = np.asarray(im, float)
        self._scale = full_w / arr.shape[1]
        return arr

    def _styled(self, arr: np.ndarray) -> np.ndarray:
        a = arr.copy()
        if self.contrast_mode == 1:
            lo, hi = np.percentile(a, [2, 98])
            a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255
        elif self.contrast_mode == 2:
            a = 255.0 * np.clip(a / 255.0, 0, 1) ** 0.65
        elif self.contrast_mode == 3:
            grey = a.mean(axis=2)
            lo, hi = np.percentile(grey, [1, 99])
            g = np.clip((grey - lo) / max(hi - lo, 1e-6), 0, 1) * 255
            a = np.dstack([g, g, g])
        if self.show_edges:
            grey = a.mean(axis=2)
            mag = gradient_magnitude(grey, smooth=1.2)
            m = np.clip(mag / max(np.percentile(mag, 99), 1e-6), 0, 1)
            a = np.clip(a * (1 - 0.55 * m[..., None])
                        + np.array([255.0, 90.0, 0.0]) * (0.55 * m[..., None]), 0, 255)
        return np.clip(a, 0, 255).astype(np.uint8)

    def load(self) -> None:
        self._display = self._load_display_image()
        if self.auto_detect and not self.state.record.calibration.is_calibrated:
            self.state.auto_calibrate()
        self.draw()

    # -- drawing --------------------------------------------------------
    def draw(self) -> None:
        st = self.state
        rec = st.record
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        had_view = self.ax.has_data() and xlim != (0.0, 1.0)
        self.ax.clear()
        self.ax.imshow(self._styled(self._display))

        s = 1.0 / self._scale     # full-res pixels -> display pixels

        cal = rec.calibration
        if cal.corners_px:
            c = np.asarray(cal.corners_px, float) * s
            self.ax.plot(np.append(c[:, 0], c[0, 0]), np.append(c[:, 1], c[0, 1]),
                         "-", lw=1.6, color="#0072B2")
        if cal.two_point_px:
            c = np.asarray(cal.two_point_px, float) * s
            self.ax.plot(c[:, 0], c[:, 1], "-o", lw=1.6, ms=5, color="#0072B2")
        if st.pending_clicks:
            c = np.asarray(st.pending_clicks, float) * s
            self.ax.plot(c[:, 0], c[:, 1], "x", ms=9, mew=2, color="#762A83")

        if rec.points_px:
            p = np.asarray(rec.points_px, float) * s
            self.ax.plot(p[:, 0], p[:, 1], "o", ms=4, color="#D55E00",
                         markeredgecolor="white", markeredgewidth=0.5)

        ell = st.preview_ellipse()
        if ell is not None:
            from .fitting import ellipse_points
            pts = ellipse_points(ell, 300) * s
            self.ax.plot(pts[:, 0], pts[:, 1], "--", lw=1.5, color="#0072B2", alpha=0.9)

        cal_txt = (f"{cal.mode}" if cal.is_calibrated else "NOT CALIBRATED")
        if cal.is_calibrated and not cal.rectified:
            cal_txt += "  (perspective NOT corrected)"
        bits = [f"{rec.object_id}   {st.progress()}",
                f"points: {rec.n_points}   snap: {'on' if st.snap_enabled else 'off'}"
                f"   calibration: {cal_txt}"]
        m = st.preview_metrics()
        if m:
            bits.append(f"2a={m['major_axis_cm']:.2f} cm  2b={m['minor_axis_cm']:.2f} cm  "
                        f"e={m['eccentricity']:.3f}  arc={m['coverage_deg']:.0f}°")
        if rec.card_on_object:
            bits.append("card ON OBJECT (not ground)")
        if rec.excluded:
            bits.append("EXCLUDED")
        if st.message:
            bits.append(st.message)
        self.ax.set_title("\n".join(bits), fontsize=9, loc="left")
        self.ax.set_xticks([]); self.ax.set_yticks([])
        if had_view:
            self.ax.set_xlim(xlim); self.ax.set_ylim(ylim)
        self.fig.canvas.draw_idle()

    # -- events ---------------------------------------------------------
    def on_click(self, event) -> None:
        if event.inaxes is not self.ax or event.xdata is None:
            return
        # Ignore clicks while a pan/zoom tool is active, or they land as points.
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return

        x, y = event.xdata * self._scale, event.ydata * self._scale
        if event.button == 3:
            self.state.undo()
        elif event.button == 1:
            if self.state.mode in ("card_corners", "two_point"):
                self.state.add_calibration_click(x, y)
            else:
                self.state.add_point(x, y)
        self.draw()

    def on_key(self, event) -> None:
        st = self.state
        k = (event.key or "").lower()
        st.message = ""
        if k == "n":
            st.next_image(); self.load(); return
        if k == "p":
            st.prev_image(); self.load(); return
        if k == "q":
            st.save()
            import matplotlib.pyplot as plt
            plt.close(self.fig)
            return
        if k == "c":
            st.clear_points()
        elif k == "d":
            st.auto_calibrate()
        elif k == "m":
            st.start_manual_calibration()
        elif k == "t":
            st.start_two_point()
        elif k == "k":
            st.copy_previous_calibration()
        elif k == "s":
            st.snap_enabled = not st.snap_enabled
            st.message = f"snapping {'on' if st.snap_enabled else 'off'}"
        elif k == "e":
            self.show_edges = not self.show_edges
        elif k == "g":
            self.contrast_mode = (self.contrast_mode + 1) % 4
        elif k == "o":
            on = st.toggle_card_on_object()
            st.message = ("card recorded as resting ON the object"
                          if on else "card recorded as on the ground")
        elif k == "x":
            st.message = "excluded" if st.toggle_excluded() else "re-included"
        elif k in ("h", "?"):
            st.message = " | ".join(f"{key}: {what}" for key, what in KEY_HELP[:6])
        elif k == "u":
            st.undo()
        self.draw()


def run_digitizer(image_dir, workdir, card_spec: Optional[CardSpec] = None,
                  operator: str = "", only_missing: bool = False,
                  display_max_px: int = 1600) -> None:
    """Open the digitiser on a folder of photographs."""
    import matplotlib
    import matplotlib.pyplot as plt

    if matplotlib.get_backend().lower() == "agg":
        raise RuntimeError(
            "no interactive matplotlib backend available (current backend is Agg).\n"
            "On Linux install Tk support:  sudo apt install python3-tk\n"
            "or install a Qt backend:      pip install PySide6"
        )

    images = scan_images(image_dir)
    if not images:
        raise FileNotFoundError(f"no images found in {image_dir}")

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if only_missing:
        keep = []
        for p in images:
            rec = ObjectRecord.for_image(p, workdir, hash_image=False)
            if rec.status() != "ready":
                keep.append(p)
        if not keep:
            print("Every object is already digitised.")
            return
        print(f"{len(keep)} of {len(images)} objects still to do.")
        images = keep

    state = DigitizerState(images, workdir, card_spec or CardSpec(), operator=operator)
    print("Keys: " + ", ".join(f"{k} = {v}" for k, v in KEY_HELP))
    Digitizer(state, display_max_px=display_max_px)
    plt.show()
    state.save()
    print(f"Saved records to {records_dir(workdir)}")
