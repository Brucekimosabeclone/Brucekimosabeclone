"""Publication figures.

Two rules shape everything here.

*Show the measurement, not just the result.* The per-object figure puts the
original photograph beside the rectified plane the measurement was actually made
in, with the surviving arc drawn solid and the reconstructed remainder dashed.
A reader can then see how much of each object is evidence and how much is
inference -- which, when a quarter to a half of the outline survives, is the
first thing they will want to know.

*Colour is never the only channel.* The categorical palette is validated for
colour-vision deficiency, but the figures also separate series by line style,
marker shape and direct labels, so they survive greyscale printing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon

from .fitting import EllipseParams, ellipse_points, foot_frame

__all__ = [
    "PALETTE", "set_style", "object_figure", "fig_size_vs_shape",
    "fig_major_axis_density", "fig_eccentricity_vs_coverage",
    "fig_eccentricity_forest", "fig_bland_altman", "fig_simulation_recovery",
    "save_figure",
]

# Validated with the dataviz palette checker (light surface): every adjacent
# pair clears the colour-vision-deficiency separation floor, the normal-vision
# floor, and 3:1 contrast against white. Do not reorder casually -- the checks
# are on adjacent pairs.
PALETTE = {
    "primary":   "#0072B2",   # the fitted ellipse
    "accent":    "#D55E00",   # digitised points / observed arc
    "support":   "#009E73",   # major axis, agreement lines
    "contrast":  "#762A83",   # minor axis, secondary series
    "ink":       "#1a1a1a",
    "muted":     "#6b6b6b",
    "faint":     "#c9c9c9",
    "surface":   "#ffffff",
}

SHAPE_COLORS = {
    "circular": PALETTE["primary"],
    "elliptical": PALETTE["accent"],
    "indeterminate": PALETTE["muted"],
}
SHAPE_MARKERS = {"circular": "o", "elliptical": "s", "indeterminate": "^"}

# Tiers are ordinal, so they get one hue getting darker, not separate hues.
TIER_COLORS = {"A": "#08519c", "B": "#6baed6", "C": "#c6dbef"}


def set_style() -> None:
    """Matplotlib defaults tuned for a single journal column."""
    plt.rcParams.update({
        "figure.facecolor": PALETTE["surface"],
        "axes.facecolor": PALETTE["surface"],
        "axes.edgecolor": PALETTE["muted"],
        "axes.linewidth": 0.8,
        "axes.labelcolor": PALETTE["ink"],
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": PALETTE["muted"],
        "ytick.color": PALETTE["muted"],
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.color": PALETTE["ink"],
        "legend.fontsize": 8,
        "legend.frameon": False,
        "grid.color": PALETTE["faint"],
        "grid.linewidth": 0.5,
        "font.size": 9,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.facecolor": PALETTE["surface"],
    })


def save_figure(fig, out_path, dpi: int = 600, also_pdf: bool = True,
                data: Optional["object"] = None) -> List[Path]:
    """Write a figure at publication resolution, plus its source data.

    Journals increasingly ask for the numbers behind each figure, and producing
    them at render time guarantees they match the figure rather than drifting
    from it.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = [out_path.with_suffix(".png")]
    fig.savefig(written[0], dpi=dpi)
    if also_pdf:
        pdf = out_path.with_suffix(".pdf")
        fig.savefig(pdf)          # vector: no dpi needed
        written.append(pdf)
    if data is not None and hasattr(data, "to_csv"):
        csv = out_path.with_suffix(".csv")
        data.to_csv(csv, index=False)
        written.append(csv)
    plt.close(fig)
    return written


def _emptiest_corner(ax, xs, ys) -> str:
    """Which axes corner sits furthest from the plotted data.

    An annotation box dropped in a fixed corner will sooner or later cover the
    thing it describes; across 167 objects at every orientation that is a
    certainty rather than a risk.
    """
    x0, x1 = sorted(ax.get_xlim())
    y0, y1 = sorted(ax.get_ylim())
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    if not ok.any() or x1 <= x0 or y1 <= y0:
        return "upper left"
    u = (xs[ok] - x0) / (x1 - x0)
    v = (ys[ok] - y0) / (y1 - y0)
    best, best_d = "upper left", -1.0
    for name, (cu, cv) in {"upper left": (0.0, 1.0), "upper right": (1.0, 1.0),
                           "lower left": (0.0, 0.0), "lower right": (1.0, 0.0)}.items():
        d = float(np.min(np.hypot(u - cu, v - cv)))
        if d > best_d:
            best, best_d = name, d
    return best


def _annot(ax, lines: Sequence[str], loc: str = "upper left") -> None:
    ax.text(0.02 if "left" in loc else 0.98,
            0.98 if "upper" in loc else 0.02,
            "\n".join(lines),
            transform=ax.transAxes, fontsize=7.5, va="top" if "upper" in loc else "bottom",
            ha="left" if "left" in loc else "right", color=PALETTE["ink"], zorder=20,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=PALETTE["faint"], alpha=0.94, linewidth=0.6))


def _fmt_ci(value: float, lo: float, hi: float, unit: str = "", dp: int = 2) -> str:
    if not np.isfinite(value):
        return "n/a"
    if np.isfinite(lo) and np.isfinite(hi):
        return f"{value:.{dp}f} [{lo:.{dp}f}, {hi:.{dp}f}]{unit}"
    return f"{value:.{dp}f}{unit}"


# --------------------------------------------------------------------------
# per-object figure
# --------------------------------------------------------------------------

def _rectified_crop(image_bgr, H_px_to_cm, bounds_cm, px_per_cm: float = 40.0):
    """Warp the region around the object into centimetre coordinates."""
    import cv2

    x0, x1, y0, y1 = bounds_cm
    w = max(int(round((x1 - x0) * px_per_cm)), 16)
    h = max(int(round((y1 - y0) * px_per_cm)), 16)
    # cm -> output grid pixels
    S = np.array([[px_per_cm, 0, -x0 * px_per_cm],
                  [0, px_per_cm, -y0 * px_per_cm],
                  [0, 0, 1.0]])
    M = S @ np.asarray(H_px_to_cm, float)
    warped = cv2.warpPerspective(image_bgr, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return warped, (x0, x1, y0, y1)


def object_figure(fit, points_cm: np.ndarray, out_path,
                  image_path=None, calibration=None,
                  card_corners_px=None, points_px=None,
                  show_residuals: bool = True, dpi: int = 600) -> List[Path]:
    """Figure for a single object: the photograph, the reconstruction, the fit quality.

    The surviving arc is drawn solid and the reconstructed remainder dashed --
    the single most important distinction in the whole analysis, because on
    these fragments most of the outline is inference.
    """
    import pandas as pd

    set_style()
    pts = np.asarray(points_cm, float)
    ell: EllipseParams = fit.ellipse

    n_panels = 2 + (1 if show_residuals else 0) + (1 if image_path else 0) - 1
    ncols = 2 if not show_residuals else 3
    if image_path is None:
        ncols -= 1
    fig, axes = plt.subplots(1, ncols, figsize=(3.4 * ncols, 3.6))
    axes = np.atleast_1d(axes)
    ai = 0

    # --- panel: original photograph -------------------------------------
    if image_path is not None:
        ax = axes[ai]; ai += 1
        try:
            import cv2
            img = cv2.imread(str(image_path))
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                grey = rgb.mean(axis=2)
                # Desaturate so the overlay reads clearly over field clutter.
                shown = np.dstack([grey] * 3) * 0.55 + rgb * 0.45
                ax.imshow(np.clip(shown, 0, 255).astype(np.uint8))

                # Crop to the object and card. Shown whole, the object occupies a
                # few percent of a field photograph and the overlay is invisible.
                focus = []
                if points_px is not None and len(points_px):
                    focus.append(np.asarray(points_px, float))
                if card_corners_px is not None:
                    focus.append(np.asarray(card_corners_px, float))
                if focus:
                    f = np.vstack(focus)
                    cx0, cy0 = f.min(axis=0)
                    cx1, cy1 = f.max(axis=0)
                    pad = 0.30 * max(cx1 - cx0, cy1 - cy0, 1.0)
                    h_img, w_img = shown.shape[:2]
                    ax.set_xlim(max(cx0 - pad, 0), min(cx1 + pad, w_img))
                    ax.set_ylim(min(cy1 + pad, h_img), max(cy0 - pad, 0))
                if card_corners_px is not None:
                    c = np.asarray(card_corners_px, float)
                    ax.add_patch(MplPolygon(c, closed=True, fill=False,
                                            edgecolor=PALETTE["primary"], linewidth=1.4))
                    ax.text(c[:, 0].mean(), c[:, 1].min() - 12, "scale card",
                            color=PALETTE["primary"], fontsize=7, ha="center")
                if points_px is not None and len(points_px):
                    p = np.asarray(points_px, float)
                    ax.plot(p[:, 0], p[:, 1], "o", ms=2.0, color=PALETTE["accent"],
                            markeredgewidth=0)
        except Exception:
            ax.text(0.5, 0.5, "image unavailable", ha="center", va="center",
                    transform=ax.transAxes, color=PALETTE["muted"])
        ax.set_title("a  Photograph", loc="left", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    # --- panel: rectified reconstruction --------------------------------
    ax = axes[ai]; ai += 1
    margin = 0.25 * ell.a + 1.0
    bounds = (ell.cx - ell.a - margin, ell.cx + ell.a + margin,
              ell.cy - ell.a - margin, ell.cy + ell.a + margin)

    if image_path is not None and calibration is not None and calibration.rectified:
        try:
            import cv2
            img = cv2.imread(str(image_path))
            if img is not None:
                warped, (x0, x1, y0, y1) = _rectified_crop(
                    img, np.asarray(calibration.H, float), bounds)
                rgbw = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
                grey = rgbw.mean(axis=2)
                shown = np.dstack([grey] * 3) * 0.6 + rgbw * 0.4
                ax.imshow(np.clip(shown, 0, 255).astype(np.uint8),
                          extent=(x0, x1, y1, y0), zorder=0)
        except Exception:
            pass

    # reconstructed outline, split into surviving and inferred
    frame = foot_frame(ell, pts[:, 0], pts[:, 1])
    t_pts = np.mod(frame["t"], 2 * np.pi)
    order = np.sort(t_pts)
    gaps = np.diff(np.concatenate([order, order[:1] + 2 * np.pi]))
    k = int(np.argmax(gaps))
    t_start, t_end = order[(k + 1) % order.size], order[k]
    span = np.mod(t_end - t_start, 2 * np.pi)

    solid = ellipse_points(ell, 300, t_start, t_start + span)
    dashed = ellipse_points(ell, 300, t_start + span, t_start + 2 * np.pi)
    ax.plot(dashed[:, 0], dashed[:, 1], "--", lw=1.6, color=PALETTE["primary"],
            zorder=3, label="reconstructed")
    ax.plot(solid[:, 0], solid[:, 1], "-", lw=2.2, color=PALETTE["primary"],
            zorder=4, label="surviving arc")
    ax.plot(pts[:, 0], pts[:, 1], "o", ms=3.2, color=PALETTE["accent"],
            markeredgecolor="white", markeredgewidth=0.4, zorder=5,
            label="digitised points")

    ct, st = np.cos(ell.theta), np.sin(ell.theta)
    ax.plot([ell.cx - ell.a * ct, ell.cx + ell.a * ct],
            [ell.cy - ell.a * st, ell.cy + ell.a * st],
            "-", lw=1.3, color=PALETTE["support"], zorder=6, label="major axis")
    ax.plot([ell.cx + ell.b * st, ell.cx - ell.b * st],
            [ell.cy - ell.b * ct, ell.cy + ell.b * ct],
            "-", lw=1.3, color=PALETTE["contrast"], zorder=6, label="minor axis")
    ax.plot([ell.cx], [ell.cy], "+", ms=7, color=PALETTE["ink"], zorder=7)

    # A scale bar is meaningful here because this panel is metric.
    bar = 5.0 if ell.a < 14 else 10.0
    bx = bounds[0] + 0.06 * (bounds[1] - bounds[0])
    by = bounds[2] + 0.93 * (bounds[3] - bounds[2])
    ax.plot([bx, bx + bar], [by, by], "-", lw=3, color=PALETTE["ink"], zorder=8,
            solid_capstyle="butt")
    ax.text(bx + bar / 2, by - 0.03 * (bounds[3] - bounds[2]), f"{bar:g} cm",
            ha="center", va="bottom", fontsize=7.5, color=PALETTE["ink"], zorder=8)

    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[3], bounds[2])
    ax.set_aspect("equal")
    ax.set_xlabel("cm"); ax.set_ylabel("cm")
    ax.set_title(f"{'b' if image_path else 'a'}  Reconstruction", loc="left", fontsize=9)
    # Legend below the axes rather than inside it. The annotation block is placed
    # automatically in whichever corner is emptiest, so an in-axes legend would
    # eventually collide with it; putting the legend outside removes the whole
    # class of collision instead of trading one overlap for another.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3,
              fontsize=6.5, handlelength=1.6, columnspacing=1.2,
              borderaxespad=0.0)

    _drawn = np.vstack([solid, dashed, pts])
    _annot(ax, [
        f"{fit.object_id}",
        f"2a {_fmt_ci(fit.major_axis_cm, fit.major_axis_lo, fit.major_axis_hi, ' cm', 1)}",
        f"2b {_fmt_ci(fit.minor_axis_cm, fit.minor_axis_lo, fit.minor_axis_hi, ' cm', 1)}",
        f"e  {_fmt_ci(fit.eccentricity, fit.eccentricity_lo, fit.eccentricity_hi, '', 2)}",
        f"arc {fit.coverage_deg:.0f}° ({100 * fit.coverage_perimeter_frac:.0f}% perim.)",
        f"RMS {fit.rms_residual_mm:.2f} mm, n = {fit.n_points}",
        f"{fit.shape_class}  (p = {fit.p_bootstrap:.3f}), tier {fit.tier}",
    ], loc=_emptiest_corner(ax, _drawn[:, 0], _drawn[:, 1]))

    # --- panel: residuals ------------------------------------------------
    resid_df = None
    if show_residuals:
        ax = axes[ai]; ai += 1
        resid_mm = frame["residual"] * 10.0
        along = np.degrees(np.mod(t_pts - t_start, 2 * np.pi))
        o = np.argsort(along)
        ax.axhline(0, color=PALETTE["muted"], lw=0.8)
        ax.plot(along[o], resid_mm[o], "-o", ms=3, lw=0.9, color=PALETTE["accent"],
                markeredgecolor="white", markeredgewidth=0.3)
        band = fit.rms_residual_mm
        ax.axhspan(-band, band, color=PALETTE["primary"], alpha=0.10,
                   label=f"±1 RMS ({band:.2f} mm)")
        ax.set_xlabel("position along surviving arc (°)")
        ax.set_ylabel("orthogonal residual (mm)")
        ax.set_title("c  Fit residuals" if image_path else "b  Fit residuals",
                     loc="left", fontsize=9)
        ax.grid(axis="y", alpha=0.5)
        ax.legend(loc="upper right", fontsize=6.5)
        resid_df = pd.DataFrame({"object_id": fit.object_id,
                                 "arc_position_deg": along[o],
                                 "residual_mm": resid_mm[o]})

    fig.tight_layout()
    return save_figure(fig, out_path, dpi=dpi, data=resid_df)


# --------------------------------------------------------------------------
# summary figures
# --------------------------------------------------------------------------

def fig_size_vs_shape(results, out_path, antimode: Optional[float] = None,
                      antimode_ci: Optional[Tuple[float, float]] = None,
                      dpi: int = 600) -> List[Path]:
    """Headline figure: reconstructed size against reconstructed shape.

    Puts the size-based typological question and the shape-based grinding-motion
    question on one pair of axes, so any relationship between them is visible
    rather than needing to be inferred across two separate figures.
    """
    import pandas as pd

    set_style()
    df = results[results["fit_ok"]].copy() if "fit_ok" in results else results.copy()
    fig, ax = plt.subplots(figsize=(5.0, 4.0))

    for cls, grp in df.groupby("shape_class"):
        ax.errorbar(
            grp["major_axis_cm"], grp["eccentricity"],
            xerr=[grp["major_axis_cm"] - grp["major_axis_lo"],
                  grp["major_axis_hi"] - grp["major_axis_cm"]],
            yerr=[grp["eccentricity"] - grp["eccentricity_lo"],
                  grp["eccentricity_hi"] - grp["eccentricity"]],
            fmt=SHAPE_MARKERS.get(cls, "o"), ms=4.5, lw=0, elinewidth=0.6,
            ecolor=PALETTE["faint"], color=SHAPE_COLORS.get(cls, PALETTE["muted"]),
            markeredgecolor="white", markeredgewidth=0.4, label=str(cls), zorder=3)

    if antimode is not None and np.isfinite(antimode):
        ax.axvline(antimode, color=PALETTE["ink"], lw=1.0, ls=":", zorder=2)
        ax.text(antimode, ax.get_ylim()[1], " size-group\n boundary", fontsize=7,
                va="top", color=PALETTE["ink"])
        if antimode_ci and all(np.isfinite(antimode_ci)):
            ax.axvspan(antimode_ci[0], antimode_ci[1], color=PALETTE["faint"],
                       alpha=0.5, zorder=1)

    ax.set_xlabel("reconstructed major axis, 2a (cm)")
    ax.set_ylabel("eccentricity")
    ax.set_title("Size against shape, with 95% intervals", loc="left")
    ax.grid(alpha=0.4)
    ax.legend(title="shape class", loc="best")
    fig.tight_layout()
    keep = [c for c in ("object_id", "major_axis_cm", "major_axis_lo", "major_axis_hi",
                        "eccentricity", "eccentricity_lo", "eccentricity_hi",
                        "shape_class", "tier") if c in df]
    return save_figure(fig, out_path, dpi=dpi, data=df[keep])


def fig_major_axis_density(results, out_path, typology=None, dpi: int = 600) -> List[Path]:
    """Distribution of reconstructed size, with the fitted mixture components."""
    import pandas as pd
    from scipy import stats as sps

    set_style()
    df = results[results["fit_ok"]].copy() if "fit_ok" in results else results.copy()
    x = df["major_axis_cm"].to_numpy(float)
    x = x[np.isfinite(x)]
    fig, ax = plt.subplots(figsize=(5.0, 3.6))

    ax.hist(x, bins="auto", density=True, color=PALETTE["faint"],
            edgecolor="white", linewidth=0.6, zorder=1, label="observed")
    if x.size > 2:
        grid = np.linspace(x.min() - 2, x.max() + 2, 512)
        kde = sps.gaussian_kde(x)
        ax.plot(grid, kde(grid), "-", lw=1.8, color=PALETTE["primary"],
                zorder=3, label="kernel density")

        if typology is not None and typology.mixture_k2 is not None:
            m = typology.mixture_k2.ordered()
            for i, (w, mu, sd) in enumerate(zip(m.weights, m.means, m.sds)):
                ax.plot(grid, w * sps.norm.pdf(grid, mu, sd), "--", lw=1.2,
                        color=[PALETTE["support"], PALETTE["contrast"]][i], zorder=2,
                        label=f"component {i + 1} ({'one' if i == 0 else 'two'}-hand)")
            am = typology.antimode or {}
            if np.isfinite(am.get("antimode", np.nan)):
                ax.axvline(am["antimode"], color=PALETTE["ink"], lw=1.0, ls=":", zorder=4)
                if np.isfinite(am.get("lo", np.nan)):
                    ax.axvspan(am["lo"], am["hi"], color=PALETTE["ink"], alpha=0.10, zorder=1)

    ax.set_xlabel("reconstructed major axis, 2a (cm)")
    ax.set_ylabel("density")
    title = "Size distribution"
    if typology is not None and typology.verdict:
        title += f"\n{typology.verdict}"
    ax.set_title(title, loc="left", fontsize=8.5)
    ax.grid(axis="y", alpha=0.4)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    return save_figure(fig, out_path, dpi=dpi,
                       data=df[[c for c in ("object_id", "major_axis_cm") if c in df]])


def fig_eccentricity_vs_coverage(results, out_path, dpi: int = 600) -> List[Path]:
    """Eccentricity against how much of the outline survived.

    The limitations figure, and it belongs in the paper rather than a supplement.
    Fitted eccentricity is inflated by noise, and the less arc there is the worse
    the inflation gets, so any trend here is a warning about how much weight the
    shape estimates can bear.
    """
    set_style()
    df = results[results["fit_ok"]].copy() if "fit_ok" in results else results.copy()
    fig, ax = plt.subplots(figsize=(5.0, 3.8))

    for tier in ("C", "B", "A"):
        grp = df[df["tier"] == tier] if "tier" in df else df.iloc[0:0]
        if not len(grp):
            continue
        ax.errorbar(grp["coverage_deg"], grp["eccentricity"],
                    yerr=[grp["eccentricity"] - grp["eccentricity_lo"],
                          grp["eccentricity_hi"] - grp["eccentricity"]],
                    fmt="o", ms=4.5, lw=0, elinewidth=0.6, ecolor=PALETTE["faint"],
                    color=TIER_COLORS.get(tier, PALETTE["muted"]),
                    markeredgecolor="white", markeredgewidth=0.4,
                    label=f"tier {tier}", zorder=3)

    ax.set_xlabel("arc coverage (°)")
    ax.set_ylabel("eccentricity")
    ax.set_ylim(bottom=0)
    ax.set_title("Shape estimates against surviving arc", loc="left")
    ax.grid(alpha=0.4)
    ax.legend(loc="upper right")
    fig.tight_layout()
    keep = [c for c in ("object_id", "coverage_deg", "eccentricity",
                        "eccentricity_lo", "eccentricity_hi", "tier") if c in df]
    return save_figure(fig, out_path, dpi=dpi, data=df[keep])


def fig_eccentricity_forest(results, out_path, max_objects: int = 60,
                            dpi: int = 600) -> List[Path]:
    """Per-object eccentricity with intervals, sorted -- who is actually oval."""
    set_style()
    df = results[results["fit_ok"]].copy() if "fit_ok" in results else results.copy()
    df = df.sort_values("eccentricity")
    truncated = len(df) > max_objects
    if truncated:
        step = max(1, len(df) // max_objects)
        df = df.iloc[::step]

    fig, ax = plt.subplots(figsize=(4.6, max(3.2, 0.135 * len(df) + 1.0)))
    ypos = np.arange(len(df))
    for cls, grp in df.groupby("shape_class"):
        idx = [i for i, c in enumerate(df["shape_class"]) if c == cls]
        ax.errorbar(grp["eccentricity"], np.asarray(idx),
                    xerr=[grp["eccentricity"] - grp["eccentricity_lo"],
                          grp["eccentricity_hi"] - grp["eccentricity"]],
                    fmt=SHAPE_MARKERS.get(cls, "o"), ms=3.6, lw=0, elinewidth=0.7,
                    ecolor=PALETTE["faint"], color=SHAPE_COLORS.get(cls, PALETTE["muted"]),
                    label=str(cls), zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels(df["object_id"], fontsize=5.5)
    ax.set_xlabel("eccentricity (95% interval)")
    ax.set_xlim(left=0)
    ax.grid(axis="x", alpha=0.4)
    title = "Eccentricity by object"
    if truncated:
        title += f" (every {step}ᵗʰ of {len(results)} shown)"
    ax.set_title(title, loc="left", fontsize=9)
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    keep = [c for c in ("object_id", "eccentricity", "eccentricity_lo",
                        "eccentricity_hi", "shape_class") if c in df]
    return save_figure(fig, out_path, dpi=dpi, data=df[keep])


def fig_bland_altman(merged, agreement, out_path, dpi: int = 600) -> List[Path]:
    """Photogrammetric minor axis against caliper width."""
    import pandas as pd

    set_style()
    df = merged.dropna(subset=["minor_axis_cm", "width_across_cm"]).copy()
    m = df["minor_axis_cm"].to_numpy(float)
    r = df["width_across_cm"].to_numpy(float)
    avg, diff = 0.5 * (m + r), m - r

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))

    ax = axes[0]
    lim = [min(m.min(), r.min()) * 0.95, max(m.max(), r.max()) * 1.05]
    ax.plot(lim, lim, "-", lw=1.0, color=PALETTE["muted"], label="1:1", zorder=1)
    ax.plot(r, m, "o", ms=4.5, color=PALETTE["primary"], markeredgecolor="white",
            markeredgewidth=0.4, zorder=3)
    ax.set_xlabel("caliper width (cm)")
    ax.set_ylabel("photogrammetric minor axis (cm)")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_title("a  Method against reference", loc="left", fontsize=9)
    ax.grid(alpha=0.4); ax.legend(loc="upper left", fontsize=7)

    ax = axes[1]
    ax.axhline(0, color=PALETTE["muted"], lw=0.8, zorder=1)
    ax.plot(avg, diff, "o", ms=4.5, color=PALETTE["primary"],
            markeredgecolor="white", markeredgewidth=0.4, zorder=3)
    ax.axhline(agreement.bias, color=PALETTE["accent"], lw=1.4, zorder=2)
    ax.axhspan(agreement.bias_lo, agreement.bias_hi, color=PALETTE["accent"],
               alpha=0.15, zorder=1)
    for y in (agreement.loa_lower, agreement.loa_upper):
        ax.axhline(y, color=PALETTE["support"], lw=1.0, ls="--", zorder=2)
    ax.set_xlabel("mean of the two measurements (cm)")
    ax.set_ylabel("difference, method − reference (cm)")
    ax.set_title("b  Bland–Altman agreement", loc="left", fontsize=9)
    ax.grid(alpha=0.4)
    _annot(ax, [
        f"n = {agreement.n}",
        f"bias = {agreement.bias:+.3f} cm [{agreement.bias_lo:+.3f}, {agreement.bias_hi:+.3f}]",
        f"      = {agreement.bias_pct:+.1f}% of reference",
        f"95% limits = {agreement.loa_lower:+.3f} to {agreement.loa_upper:+.3f} cm",
        f"proportional bias p = {agreement.proportional_p:.3f}",
    ], loc="lower left")

    fig.tight_layout()
    out = pd.DataFrame({"object_id": df["object_id"], "caliper_width_cm": r,
                        "photogrammetric_minor_cm": m, "mean_cm": avg,
                        "difference_cm": diff})
    return save_figure(fig, out_path, dpi=dpi, data=out)


def fig_simulation_recovery(study, out_path, dpi: int = 600) -> List[Path]:
    """What the simulation says is recoverable, as a function of arc coverage.

    This is the figure that licenses -- or withholds -- confidence in the
    eccentricity column of the results table.
    """
    set_style()
    df = study.copy()
    noises = sorted(df["noise_cm"].unique())
    fig, axes = plt.subplots(1, len(noises), figsize=(3.2 * len(noises), 3.4),
                             sharey=True)
    axes = np.atleast_1d(axes)

    truths = sorted(df["e_true"].unique())
    cmap = plt.get_cmap("viridis")
    for ax, sigma in zip(axes, noises):
        sub = df[df["noise_cm"] == sigma]
        for i, e in enumerate(truths):
            g = sub[sub["e_true"] == e].sort_values("coverage_deg")
            col = cmap(i / max(1, len(truths) - 1))
            ax.plot(g["coverage_deg"], g["e_median"], "-o", ms=3.5, lw=1.3,
                    color=col, label=f"{e:.1f}")
            ax.axhline(e, color=col, lw=0.6, ls=":", alpha=0.7)
        ax.set_xlabel("arc coverage (°)")
        ax.set_title(f"noise σ = {sigma:g} cm", loc="left", fontsize=8.5)
        ax.grid(alpha=0.4)
    axes[0].set_ylabel("recovered eccentricity (median)")
    axes[-1].legend(title="true e", fontsize=7, title_fontsize=7,
                    loc="upper right")
    fig.suptitle("Dotted lines mark the true value; the gap above them is noise-driven inflation",
                 fontsize=8, y=1.02, x=0.01, ha="left", color=PALETTE["muted"])
    fig.tight_layout()
    return save_figure(fig, out_path, dpi=dpi, data=df)
