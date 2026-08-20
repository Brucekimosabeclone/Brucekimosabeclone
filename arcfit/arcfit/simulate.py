"""Synthetic photographs and the simulation study.

Two jobs, both about knowing the right answer in advance.

``render_scene`` builds a photograph that imitates the real ones -- an oblique
view of a fragmentary object lying on sand beside a checkerboard scale card,
with the object's outline genuinely raised above the card's plane so the
parallax effect is present rather than assumed. Because the ellipse that went
in is known exactly, the whole chain (detection, rectification, fitting,
statistics) can be checked end to end against truth. These images are also the
demo dataset shipped with the supplement, so a reviewer can run everything
without the 2.5 GB of field photographs.

``simulation_study`` answers the question ground truth cannot: how well is
*eccentricity* recovered from a partial arc? Caliper measurements can validate a
length, but nobody measured the eccentricity of an object that is missing half
its outline, so its reliability has to be established by simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .fitting import EllipseParams, FitError, fit_ellipse, fit_ellipse_algebraic
from .scalecard import CardSpec

__all__ = [
    "SceneTruth",
    "make_camera",
    "project",
    "render_scene",
    "make_demo_dataset",
    "simulation_study",
    "arc_points",
]


# --------------------------------------------------------------------------
# camera
# --------------------------------------------------------------------------

def make_camera(height_cm: float = 150.0, offset_cm: float = 90.0,
                f_px: float = 3000.0, img_w: int = 2400, img_h: int = 1800,
                yaw_deg: float = 0.0):
    """A camera at height ``height_cm`` looking down at the world origin.

    ``offset_cm`` is its horizontal displacement, so 0 gives a plan view and
    larger values give more oblique shots -- the standing-height field geometry.
    """
    L = float(np.hypot(offset_cm, height_cm))
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -height_cm / L, -offset_cm / L],
        [0.0, offset_cm / L, -height_cm / L],
    ])
    if yaw_deg:
        c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
        R = R @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    t = np.array([0.0, 0.0, L])
    K = np.array([[f_px, 0.0, img_w / 2.0],
                  [0.0, f_px, img_h / 2.0],
                  [0.0, 0.0, 1.0]])
    tilt = float(np.degrees(np.arctan2(offset_cm, height_cm)))
    return K, R, t, tilt


def project(K, R, t, pts_cm, z: float = 0.0) -> np.ndarray:
    """Project world points on the plane Z = z into image pixels."""
    pts = np.atleast_2d(np.asarray(pts_cm, float))
    X = np.column_stack([pts, np.full(len(pts), float(z))])
    cam = (np.asarray(R) @ X.T).T + np.asarray(t)
    img = (np.asarray(K) @ cam.T).T
    return img[:, :2] / img[:, 2:3]


def arc_points(ell: EllipseParams, n: int, coverage_deg: float,
               start_deg: float = 0.0) -> np.ndarray:
    """``n`` points spanning ``coverage_deg`` of an ellipse, in world cm."""
    t = np.radians(start_deg) + np.linspace(0.0, np.radians(coverage_deg), int(n))
    ct, st = np.cos(ell.theta), np.sin(ell.theta)
    qx, qy = ell.a * np.cos(t), ell.b * np.sin(t)
    return np.column_stack([ell.cx + ct * qx - st * qy,
                            ell.cy + st * qx + ct * qy])


# --------------------------------------------------------------------------
# scene rendering
# --------------------------------------------------------------------------

@dataclass
class SceneTruth:
    """Everything the renderer knew, for checking what the pipeline recovers."""

    object_id: str
    a_cm: float
    b_cm: float
    major_axis_cm: float
    minor_axis_cm: float
    eccentricity: float
    theta_rad: float
    cx_cm: float
    cy_cm: float
    coverage_deg: float
    start_deg: float
    thickness_cm: float
    outline_height_cm: float
    camera_height_cm: float
    camera_offset_cm: float
    tilt_deg: float
    card_w_cm: float
    card_h_cm: float
    card_corners_px: List[List[float]]
    parallax_inflation: float
    fragment_max_cm: float
    width_across_cm: float

    def to_dict(self) -> dict:
        return asdict(self)


def _ground_texture(rng, h: int, w: int) -> np.ndarray:
    """Sand and dry grass: mid-grey, heavily textured, low contrast."""
    import cv2
    base = rng.normal(150, 18, (h, w)).astype(np.float32)
    # A few octaves of blurred noise gives clumping like real ground.
    for sigma, amp in ((31, 22.0), (11, 12.0), (5, 6.0)):
        n = rng.normal(0, 1, (h, w)).astype(np.float32)
        base += amp * cv2.GaussianBlur(n, (0, 0), sigma)
    # Dry grass: thin darker streaks.
    for _ in range(rng.integers(40, 90)):
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        ang, ln = rng.uniform(0, np.pi), rng.integers(20, 90)
        x1 = int(np.clip(x0 + ln * np.cos(ang), 0, w - 1))
        y1 = int(np.clip(y0 + ln * np.sin(ang), 0, h - 1))
        cv2.line(base, (int(x0), int(y0)), (x1, y1),
                 float(rng.uniform(105, 145)), int(rng.integers(1, 3)))
    return np.clip(base, 0, 255)


def render_scene(object_id: str = "SIM_0001", seed: int = 0,
                 a_cm: float = 9.0, b_cm: float = 6.0, theta_deg: float = 25.0,
                 coverage_deg: float = 150.0, start_deg: float = 20.0,
                 thickness_cm: float = 4.5, camera_height_cm: float = 150.0,
                 camera_offset_cm: float = 90.0, spec: CardSpec = CardSpec(),
                 img_w: int = 2400, img_h: int = 1800,
                 card_origin_cm: Tuple[float, float] = (-16.0, -7.0),
                 jpeg_quality: int = 92,
                 contrast: float = 1.0) -> Tuple[np.ndarray, SceneTruth]:
    """Render one synthetic field photograph plus its ground truth.

    The object's outline is drawn at half its thickness above the ground plane,
    matching the "widest visible silhouette" that gets digitised in practice, so
    the rendered image carries the real parallax offset rather than pretending
    the object is flat.
    """
    import cv2

    rng = np.random.default_rng(seed)
    K, R, t, tilt = make_camera(camera_height_cm, camera_offset_cm,
                                img_w=img_w, img_h=img_h)

    img = _ground_texture(rng, img_h, img_w)

    # --- the object -------------------------------------------------------
    ell = EllipseParams(0.0, 0.0, a_cm, b_cm, np.radians(theta_deg))
    outline_h = 0.5 * thickness_cm
    outline = arc_points(ell, 400, coverage_deg, start_deg)
    poly_px = project(K, R, t, outline, z=outline_h)

    # Cast shadow, offset along the ground away from the light.
    shadow = project(K, R, t, outline + np.array([1.6, -1.1]), z=0.0)
    cv2.fillPoly(img, [np.round(shadow).astype(np.int32)], 96.0)
    cv2.GaussianBlur(img, (0, 0), 3, dst=img)

    stone = float(np.clip(150 - 22 * contrast, 60, 200))
    cv2.fillPoly(img, [np.round(poly_px).astype(np.int32)], stone)

    # Give the stone its own speckle and a slightly darker rim, so the edge is
    # findable but not trivially so -- the real difficulty of grey on grey.
    mask = np.zeros((img_h, img_w), np.uint8)
    cv2.fillPoly(mask, [np.round(poly_px).astype(np.int32)], 255)
    speckle = rng.normal(0, 7, (img_h, img_w)).astype(np.float32)
    speckle = cv2.GaussianBlur(speckle, (0, 0), 1.5)
    img = np.where(mask > 0, img + speckle, img)
    cv2.polylines(img, [np.round(poly_px).astype(np.int32)], True,
                  float(stone - 12 * contrast), 2)

    # --- the scale card ---------------------------------------------------
    ox, oy = card_origin_cm
    card_cm = np.array([[ox, oy],
                        [ox + spec.width_cm, oy],
                        [ox + spec.width_cm, oy + spec.height_cm],
                        [ox, oy + spec.height_cm]])
    card_px = project(K, R, t, card_cm, z=0.0)

    cell = 16
    patch = np.zeros((spec.rows * cell, spec.cols * cell), np.float32)
    for r in range(spec.rows):
        for c in range(spec.cols):
            patch[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = (
                242.0 if (r + c) % 2 == 0 else 18.0)
    src = np.array([[0, 0], [patch.shape[1] - 1, 0],
                    [patch.shape[1] - 1, patch.shape[0] - 1],
                    [0, patch.shape[0] - 1]], np.float32)
    M = cv2.getPerspectiveTransform(src, card_px.astype(np.float32))
    warped = cv2.warpPerspective(patch, M, (img_w, img_h))
    card_mask = cv2.warpPerspective(np.ones_like(patch), M, (img_w, img_h))
    img = np.where(card_mask > 0.5, warped, img)

    # --- camera realism ---------------------------------------------------
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    img = img + rng.normal(0, 3.0, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # A warm field cast, so the demo images are not suspiciously neutral.
    bgr = np.clip(bgr * np.array([0.93, 0.99, 1.06]), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if ok:
        bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # --- ground truth -----------------------------------------------------
    inflation = camera_height_cm / (camera_height_cm - outline_h)
    frag = float(np.max(np.hypot(*(outline[:, None, :] - outline[None, :, :]).T)))
    truth = SceneTruth(
        object_id=object_id, a_cm=a_cm, b_cm=b_cm,
        major_axis_cm=2 * a_cm, minor_axis_cm=2 * b_cm,
        eccentricity=ell.eccentricity, theta_rad=float(ell.theta),
        cx_cm=0.0, cy_cm=0.0, coverage_deg=coverage_deg, start_deg=start_deg,
        thickness_cm=thickness_cm, outline_height_cm=outline_h,
        camera_height_cm=camera_height_cm, camera_offset_cm=camera_offset_cm,
        tilt_deg=tilt, card_w_cm=spec.width_cm, card_h_cm=spec.height_cm,
        card_corners_px=[[float(x), float(y)] for x, y in card_px],
        parallax_inflation=float(inflation),
        fragment_max_cm=frag, width_across_cm=2 * b_cm,
    )
    return bgr, truth


def make_demo_dataset(outdir, n: int = 12, seed: int = 20260820,
                      spec: CardSpec = CardSpec()) -> List[SceneTruth]:
    """Render a small demo assemblage spanning the real range of preservation.

    Deliberately mixes near-circular and elongated objects across a range of arc
    coverage, so the demo exercises the interesting cases rather than only easy
    ones.
    """
    import cv2

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    truths: List[SceneTruth] = []

    for i in range(n):
        # Half one-hand-ish (small, round), half two-hand-ish (long, oval).
        if i % 2 == 0:
            a = rng.uniform(5.5, 7.5)
            b = a * rng.uniform(0.92, 1.0)
        else:
            a = rng.uniform(9.5, 13.0)
            b = a * rng.uniform(0.45, 0.68)
        oid = f"SIM_{i + 1:04d}"
        img, truth = render_scene(
            object_id=oid, seed=int(rng.integers(0, 1 << 30)),
            a_cm=float(a), b_cm=float(b),
            theta_deg=float(rng.uniform(0, 180)),
            coverage_deg=float(rng.uniform(95, 185)),
            start_deg=float(rng.uniform(0, 360)),
            thickness_cm=float(rng.uniform(3.0, 6.0)),
            camera_height_cm=float(rng.uniform(135, 165)),
            camera_offset_cm=float(rng.uniform(50, 120)),
            spec=spec,
        )
        cv2.imwrite(str(outdir / f"{oid}.jpg"), img)
        truths.append(truth)
    return truths


# --------------------------------------------------------------------------
# the simulation study
# --------------------------------------------------------------------------

def simulation_study(eccentricities: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8),
                     coverages_deg: Sequence[float] = (60, 90, 120, 150, 180, 270),
                     noise_cm: Sequence[float] = (0.05, 0.10, 0.20),
                     n_points: int = 30, replicates: int = 300,
                     a_cm: float = 10.0, seed: int = 20260820) -> "object":
    """Recovery of ellipse parameters as a function of arc coverage and noise.

    Returns a tidy DataFrame with one row per cell. This is what calibrates the
    reliability tiers and what tells the reader which eccentricity estimates in
    the results table can carry weight.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows = []
    for e_true in eccentricities:
        b = a_cm * float(np.sqrt(max(1e-12, 1.0 - e_true ** 2)))
        truth = EllipseParams(0.0, 0.0, a_cm, b, 0.0)
        for cov in coverages_deg:
            for sigma in noise_cm:
                est_e, est_a, est_b, fails = [], [], [], 0
                for _ in range(replicates):
                    pts = arc_points(truth, n_points, float(cov),
                                     float(rng.uniform(0, 360)))
                    pts = pts + rng.normal(0, sigma, pts.shape)
                    try:
                        fit = fit_ellipse(pts[:, 0], pts[:, 1])
                    except (FitError, np.linalg.LinAlgError):
                        fails += 1
                        continue
                    if not np.isfinite(fit.a) or fit.a > 50 * a_cm:
                        fails += 1
                        continue
                    est_e.append(fit.eccentricity)
                    est_a.append(2 * fit.a)
                    est_b.append(2 * fit.b)

                est_e = np.asarray(est_e)
                est_a = np.asarray(est_a)
                rows.append({
                    "e_true": e_true,
                    "coverage_deg": cov,
                    "noise_cm": sigma,
                    "n_points": n_points,
                    "n_ok": int(est_e.size),
                    "fail_rate": fails / replicates,
                    "e_median": float(np.median(est_e)) if est_e.size else np.nan,
                    "e_bias": float(np.median(est_e) - e_true) if est_e.size else np.nan,
                    "e_iqr": float(np.subtract(*np.percentile(est_e, [75, 25]))) if est_e.size else np.nan,
                    "e_rmse": float(np.sqrt(np.mean((est_e - e_true) ** 2))) if est_e.size else np.nan,
                    "major_median": float(np.median(est_a)) if est_a.size else np.nan,
                    "major_rel_bias": float(np.median(est_a) / (2 * a_cm) - 1.0) if est_a.size else np.nan,
                })
    return pd.DataFrame(rows)


def digitise_synthetically(image_path, truth: SceneTruth, spec: CardSpec = CardSpec(),
                           n_points: int = 34, click_noise_px: float = 2.5,
                           seed: int = 0, workdir=None):
    """Produce a digitisation record for a rendered scene, without an operator.

    Deliberately does *not* shortcut the pipeline: the scale card is located by
    the real detector and the calibration is built from those detected corners.
    Only the operator's clicking is simulated, by sampling the true outline and
    adding click noise. So an end-to-end run over these records exercises
    detection, rectification, parallax and fitting against known geometry --
    which is what makes the demo dataset a test and not just an illustration.
    """
    from .calibrate import Calibration
    from .records import ObjectRecord
    from .scalecard import detect_card

    image_path = Path(image_path)
    rng = np.random.default_rng(seed)
    rec = ObjectRecord.for_image(image_path, workdir)

    det = detect_card(image_path, spec)
    if det is None:
        rec.notes = "scale card not detected"
        return rec
    rec.calibration = Calibration.from_rect(
        det.corners_px, spec.width_cm, spec.height_cm, source="auto",
        image_path=image_path, image_size_px=rec.image_size_px,
        detection_score=det.score,
        extra_px=det.interior_px, extra_cm=det.interior_cm)

    K, R, t, _ = make_camera(truth.camera_height_cm, truth.camera_offset_cm,
                             img_w=rec.image_size_px[0], img_h=rec.image_size_px[1])
    ell = EllipseParams(truth.cx_cm, truth.cy_cm, truth.a_cm, truth.b_cm,
                        truth.theta_rad)
    world = arc_points(ell, n_points, truth.coverage_deg, truth.start_deg)
    px = project(K, R, t, world, z=truth.outline_height_cm)
    noisy = px + rng.normal(0, click_noise_px, px.shape)

    rec.points_px = [[float(a), float(b)] for a, b in noisy]
    rec.points_px_raw = [[float(a), float(b)] for a, b in noisy]
    rec.snap_used = False
    rec.operator = "simulated"
    return rec


def make_demo_workdir(outdir, n: int = 12, seed: int = 20260820,
                      spec: CardSpec = CardSpec(), n_points: int = 34,
                      click_noise_px: float = 2.5) -> dict:
    """Render a demo assemblage and digitise it, ready for ``arcfit analyze``.

    Also writes a caliper measurement sheet derived from the true geometry, with
    realistic measurement error, so the validation and typology steps have
    something to work on.
    """
    import pandas as pd

    outdir = Path(outdir)
    images = outdir / "images"
    truths = make_demo_dataset(images, n=n, seed=seed, spec=spec)

    rng = np.random.default_rng(seed + 1)
    rows = []
    for i, truth in enumerate(truths):
        img = images / f"{truth.object_id}.jpg"
        rec = digitise_synthetically(img, truth, spec, n_points=n_points,
                                     click_noise_px=click_noise_px,
                                     seed=seed + 100 + i, workdir=outdir)
        rec.save(outdir)
        # Caliper readings: the true values plus plausible measurement error.
        rows.append({
            "object_id": truth.object_id,
            "image": img.name,
            "fragment_max_cm": round(truth.fragment_max_cm + rng.normal(0, 0.05), 2),
            "width_across_cm": round(truth.width_across_cm + rng.normal(0, 0.05), 2),
            "width_is_at_widest": 1,
            "notes": "",
        })

    pd.DataFrame(rows).to_csv(outdir / "measurements.csv", index=False)
    pd.DataFrame([t.to_dict() for t in truths]).to_csv(outdir / "ground_truth.csv",
                                                       index=False)
    return {"workdir": outdir, "images": images, "truths": truths,
            "measurements": outdir / "measurements.csv",
            "ground_truth": outdir / "ground_truth.csv"}
