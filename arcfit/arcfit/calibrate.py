"""Turning image pixels into centimetres on the ground plane.

The photographs are oblique, so a single pixels-per-centimetre scalar is not
enough: a circle viewed off-axis projects to an ellipse, and scaling alone
cannot undo that. Instead the four corners of the scale card -- a rectangle of
known size lying in the ground plane -- define a homography that maps image
pixels to centimetres on that plane. Fitting then happens in real units, and
the projective distortion is removed rather than absorbed into the result.

Two-point scaling is retained as a fallback for photographs where the card
cannot be located, but records calibrated that way are flagged so they never
silently mix with rectified ones.

A note on what this does *not* fix: the outline being traced sits above the
card's plane by roughly half the object's thickness. Rectifying it onto the
ground plane is a central projection between parallel planes, which is a
homothety -- a uniform scaling about the camera's nadir. It inflates lengths by
H / (H - h) but leaves shape alone, so eccentricity is unaffected. See
``parallax_inflation``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Calibration",
    "CalibrationError",
    "homography_from_rect",
    "homography_from_points",
    "apply_homography",
    "two_point_scale",
    "intrinsics_from_exif",
    "decompose_homography",
    "parallax_inflation",
    "estimate_card_aspect",
]


class CalibrationError(ValueError):
    """Raised when a calibration cannot be formed from the given input."""


# --------------------------------------------------------------------------
# homography
# --------------------------------------------------------------------------

def _normalise(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hartley normalisation: centroid to origin, mean distance sqrt(2)."""
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    if d < 1e-12:
        raise CalibrationError("degenerate point set (all coincident)")
    s = np.sqrt(2.0) / d
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
    q = (T @ np.column_stack([pts, np.ones(len(pts))]).T).T
    return q[:, :2], T


def homography_from_points(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, float]:
    """DLT homography mapping ``src`` to ``dst``, with reprojection RMS.

    Hartley normalisation is applied to both point sets. Without it the DLT
    design matrix mixes terms of order 1 with terms of order 1e7 for
    pixel coordinates on a 4000px image, and the solution degrades badly.
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    if src.shape[0] < 4 or src.shape != dst.shape:
        raise CalibrationError("need >= 4 matched point pairs")

    ns, Ts = _normalise(src)
    nd, Td = _normalise(dst)

    rows = []
    for (x, y), (u, v) in zip(ns, nd):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.array(rows, float)
    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)

    H = np.linalg.inv(Td) @ Hn @ Ts
    if abs(H[2, 2]) < 1e-15:
        raise CalibrationError("degenerate homography")
    H = H / H[2, 2]

    pred = apply_homography(H, src)
    rms = float(np.sqrt(((pred - dst) ** 2).sum(axis=1).mean()))
    return H, rms


def homography_from_rect(corners_px: Sequence[Sequence[float]],
                         width_cm: float, height_cm: float) -> Tuple[np.ndarray, float]:
    """Homography mapping image pixels to centimetres on the card's plane.

    ``corners_px`` are the four card corners in order (any consistent winding),
    starting from the corner that maps to the origin. The card is taken to span
    ``width_cm`` x ``height_cm`` in the plane.
    """
    corners = np.asarray(corners_px, float)
    if corners.shape != (4, 2):
        raise CalibrationError(f"expected 4 corners, got shape {corners.shape}")
    if width_cm <= 0 or height_cm <= 0:
        raise CalibrationError("card dimensions must be positive")
    dst = np.array([[0.0, 0.0], [width_cm, 0.0],
                    [width_cm, height_cm], [0.0, height_cm]], float)
    return homography_from_points(corners, dst)


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map an (N, 2) array of points through a 3x3 homography."""
    pts = np.atleast_2d(np.asarray(pts, float))
    hom = np.column_stack([pts, np.ones(len(pts))])
    out = (np.asarray(H, float) @ hom.T).T
    w = out[:, 2:3]
    if np.any(np.abs(w) < 1e-15):
        raise CalibrationError("points map to the horizon (w = 0)")
    return out[:, :2] / w


def two_point_scale(p1: Sequence[float], p2: Sequence[float],
                    known_cm: float) -> float:
    """Pixels per centimetre from two clicked points a known distance apart."""
    if known_cm <= 0:
        raise CalibrationError("known distance must be positive")
    d = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    if d < 1e-9:
        raise CalibrationError("the two calibration points are coincident")
    return d / known_cm


# --------------------------------------------------------------------------
# camera geometry
# --------------------------------------------------------------------------

def _f35_from_tags(tags) -> Optional[float]:
    """35mm-equivalent focal length from EXIF tags, or None.

    ``FocalLengthIn35mmFilm`` is preferred because it encodes focal length and
    sensor size together. Plenty of cameras never write it -- Canon DSLRs among
    them -- but do write the focal-plane resolution, from which the sensor width
    follows directly. Without this fallback such a camera looks uncalibratable
    and every check that needs K is silently skipped.
    """
    f35 = tags.get("FocalLengthIn35mmFilm")
    if f35:
        try:
            value = float(f35)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0:
            return value

    focal_mm = tags.get("FocalLength")
    x_res = tags.get("FocalPlaneXResolution")
    width_px = tags.get("ExifImageWidth")
    if not (focal_mm and x_res and width_px):
        return None
    try:
        focal_mm = float(focal_mm)
        x_res = float(x_res)
        width_px = float(width_px)
    except (TypeError, ValueError):
        return None
    if focal_mm <= 0 or x_res <= 0 or width_px <= 0:
        return None

    # FocalPlaneResolutionUnit: 2 = inch, 3 = cm. Inch is the near-universal
    # choice and the sensible default when the tag is missing.
    unit = tags.get("FocalPlaneResolutionUnit") or 2
    try:
        mm_per_unit = {2: 25.4, 3: 10.0}[int(unit)]
    except (TypeError, ValueError, KeyError):
        mm_per_unit = 25.4
    sensor_width_mm = width_px / x_res * mm_per_unit

    # Refuse to guess from an implausible sensor. Anything outside roughly
    # phone-sensor to medium-format is a misread tag, and a wrong K is worse
    # than no K: it produces a confident, wrong tilt.
    if not 1.0 < sensor_width_mm < 100.0:
        return None
    return focal_mm * 36.0 / sensor_width_mm


def intrinsics_from_exif(image_path, image_width_px: int,
                         image_height_px: int) -> Optional[np.ndarray]:
    """Approximate camera matrix K from EXIF, or None if unavailable.

    Uses the 35mm-equivalent focal length, taken from EXIF directly where the
    camera records it and otherwise derived from the focal-plane resolution --
    see ``_f35_from_tags``. The principal point
    is assumed to be the image centre. This is good enough for a tilt estimate
    and a camera-height estimate; it is not a substitute for calibrating the
    camera, and nothing that affects a reported measurement depends on it.
    """
    try:
        from PIL import Image, ExifTags
    except ImportError:  # pragma: no cover - Pillow is a hard dependency
        return None

    try:
        with Image.open(image_path) as im:
            exif = im.getexif()
            if not exif:
                return None
            tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
            ifd = exif.get_ifd(0x8769)
            tags.update({ExifTags.TAGS.get(k, k): v for k, v in ifd.items()})
    except Exception:
        return None

    f35 = _f35_from_tags(tags)
    if f35 is None:
        return None

    # 35mm frame is 36mm wide; scale by the longer image side, which is the one
    # that corresponds to the 36mm dimension regardless of orientation.
    long_side = max(image_width_px, image_height_px)
    f_px = (f35 / 36.0) * long_side
    return np.array([[f_px, 0.0, image_width_px / 2.0],
                     [0.0, f_px, image_height_px / 2.0],
                     [0.0, 0.0, 1.0]], float)


def decompose_homography(H_px_to_cm: np.ndarray, K: np.ndarray) -> dict:
    """Recover tilt and camera height from a plane homography.

    Returns ``{"tilt_deg", "camera_height_cm"}``. ``tilt_deg`` is the angle
    between the camera's optical axis and the ground-plane normal: 0 means a
    perfectly perpendicular (plan) view, larger values mean a more oblique shot.
    """
    H_cm_to_px = np.linalg.inv(np.asarray(H_px_to_cm, float))
    M = np.linalg.inv(np.asarray(K, float)) @ H_cm_to_px

    n1 = np.linalg.norm(M[:, 0])
    n2 = np.linalg.norm(M[:, 1])
    if n1 < 1e-12 or n2 < 1e-12:
        raise CalibrationError("degenerate homography decomposition")
    scale = np.sqrt(n1 * n2)
    M = M / scale

    r1, r2, t = M[:, 0], M[:, 1], M[:, 2]
    # Re-orthonormalise: the DLT solution is only approximately a rotation.
    r1n = r1 / np.linalg.norm(r1)
    r2n = r2 - (r2 @ r1n) * r1n
    r2n = r2n / np.linalg.norm(r2n)
    r3 = np.cross(r1n, r2n)

    # A plane in front of the camera has t_z > 0; flip the sign convention if not.
    if t[2] < 0:
        r1n, r2n, t = -r1n, -r2n, -t
        r3 = np.cross(r1n, r2n)

    tilt = np.degrees(np.arccos(np.clip(abs(r3[2]), 0.0, 1.0)))
    height = abs(float(r3 @ t))
    return {"tilt_deg": float(tilt), "camera_height_cm": height}


def parallax_inflation(camera_height_cm: float, object_height_cm: float) -> float:
    """Factor by which rectifying an outline at height h onto the ground inflates it.

    Rectifying a point at height ``h`` with the ground-plane homography recovers
    where its camera ray crosses the ground. That is a central projection between
    parallel planes -- a homothety about the camera's nadir with ratio
    ``H / (H - h)``.

    Because a homothety is a *uniform* scaling, it maps an ellipse to a similar
    ellipse: lengths inflate but eccentricity and orientation are untouched. The
    circular-vs-oval decision is therefore immune to this effect; only the axis
    lengths carry it.
    """
    if camera_height_cm <= 0:
        raise CalibrationError("camera height must be positive")
    if object_height_cm < 0:
        raise CalibrationError("object height cannot be negative")
    if object_height_cm >= camera_height_cm:
        raise CalibrationError("object height must be below the camera")
    return float(camera_height_cm / (camera_height_cm - object_height_cm))


# --------------------------------------------------------------------------
# the stored calibration
# --------------------------------------------------------------------------

@dataclass
class Calibration:
    """How one photograph's pixels map to centimetres.

    ``mode`` is "homography" (rectified, the normal case), "two_point" (a plain
    scalar scale, used only where the card could not be located) or "none".
    """

    mode: str = "none"
    H: Optional[list] = None
    corners_px: Optional[list] = None
    card_cm: Optional[list] = None
    px_per_cm: Optional[float] = None
    two_point_px: Optional[list] = None
    known_cm: Optional[float] = None
    reproj_rms_px: Optional[float] = None
    reproj_rms_cm: Optional[float] = None
    tilt_deg: Optional[float] = None
    camera_height_cm: Optional[float] = None
    source: str = "manual"
    detection_score: Optional[float] = None
    n_calibration_points: int = 4
    notes: str = ""

    @property
    def rectified(self) -> bool:
        return self.mode == "homography"

    @property
    def is_calibrated(self) -> bool:
        return self.mode in ("homography", "two_point")

    def to_cm(self, points_px: np.ndarray) -> np.ndarray:
        """Map image pixel coordinates to centimetres on the ground plane."""
        pts = np.atleast_2d(np.asarray(points_px, float))
        if pts.size == 0:
            return pts.reshape(0, 2)
        if self.mode == "homography":
            if self.H is None:
                raise CalibrationError("homography calibration has no matrix")
            return apply_homography(np.asarray(self.H, float), pts)
        if self.mode == "two_point":
            if not self.px_per_cm:
                raise CalibrationError("two-point calibration has no scale")
            # Anchored at the image origin: only differences are meaningful,
            # and every downstream quantity is a difference.
            return pts / float(self.px_per_cm)
        raise CalibrationError("record is not calibrated")

    @classmethod
    def from_rect(cls, corners_px, width_cm, height_cm, source="manual",
                  image_path=None, image_size_px=None,
                  detection_score=None, extra_px=None, extra_cm=None) -> "Calibration":
        """Build a rectified calibration from four card corners.

        ``extra_px``/``extra_cm`` optionally add further correspondences -- in
        practice the checkerboard's interior corners. Four points alone exactly
        determine a homography, leaving a reprojection residual of zero that
        says nothing about quality; the extra points both improve the fit and
        turn that residual into a usable calibration check.
        """
        corners = np.asarray(corners_px, float)
        if extra_px is not None and extra_cm is not None and len(extra_px) >= 1:
            dst_corners = np.array([[0.0, 0.0], [width_cm, 0.0],
                                    [width_cm, height_cm], [0.0, height_cm]], float)
            src = np.vstack([corners, np.asarray(extra_px, float)])
            dst = np.vstack([dst_corners, np.asarray(extra_cm, float)])
            H, rms_cm = homography_from_points(src, dst)
        else:
            H, rms_cm = homography_from_rect(corners_px, width_cm, height_cm)
        cal = cls(
            mode="homography",
            H=[list(map(float, r)) for r in H],
            corners_px=[[float(a), float(b)] for a, b in np.asarray(corners_px, float)],
            card_cm=[float(width_cm), float(height_cm)],
            reproj_rms_cm=float(rms_cm),
            source=source,
            detection_score=None if detection_score is None else float(detection_score),
            n_calibration_points=4 + (0 if extra_px is None else len(extra_px)),
        )
        # An equivalent scale, for drawing scale bars and for reporting.
        corners = np.asarray(corners_px, float)
        edge_px = float(np.hypot(*(corners[1] - corners[0])))
        cal.px_per_cm = edge_px / float(width_cm) if width_cm else None

        if image_path is not None and image_size_px is not None:
            K = intrinsics_from_exif(image_path, int(image_size_px[0]), int(image_size_px[1]))
            if K is not None:
                try:
                    geom = decompose_homography(H, K)
                    cal.tilt_deg = geom["tilt_deg"]
                    cal.camera_height_cm = geom["camera_height_cm"]
                except (CalibrationError, np.linalg.LinAlgError):
                    pass
        return cal

    @classmethod
    def from_two_points(cls, p1, p2, known_cm, source="manual") -> "Calibration":
        """Build an unrectified fallback calibration from a known distance."""
        return cls(
            mode="two_point",
            px_per_cm=two_point_scale(p1, p2, known_cm),
            two_point_px=[[float(p1[0]), float(p1[1])], [float(p2[0]), float(p2[1])]],
            known_cm=float(known_cm),
            source=source,
            notes="not rectified: oblique perspective is NOT corrected",
        )

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


def estimate_card_aspect(corners_px, K) -> float:
    """Measure a rectangle's true width:height ratio from one view.

    Independent of what the operator declared the card to be, which is the
    point: a wrong declared height stretches the rectified plane in one
    direction and corrupts eccentricity while leaving everything else looking
    fine. Nothing else in the pipeline can catch that, because the homography is
    *built* from the declared numbers and so always reproduces them.

    For a homography H taking the unit square to the imaged rectangle,
    ``K^-1 H = [w*r1, h*r2, t]`` with r1, r2 unit vectors, so the ratio of the
    first two column norms is the rectangle's aspect.

    Accuracy depends on perspective: a perfectly head-on view carries no depth
    information and the estimate becomes ill-conditioned, so a single photograph
    can be noisy. Take the median across many.
    """
    corners = np.asarray(corners_px, float).reshape(4, 2)
    unit = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    H, _ = homography_from_points(unit, corners)

    M = np.linalg.inv(np.asarray(K, float)) @ H
    n1 = float(np.linalg.norm(M[:, 0]))
    n2 = float(np.linalg.norm(M[:, 1]))
    if n2 < 1e-12:
        return float("nan")
    return n1 / n2
