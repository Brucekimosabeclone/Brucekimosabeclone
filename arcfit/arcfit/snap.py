"""Sub-pixel edge snapping along a curve's normal.

Used in two places: pulling a hand-placed click onto the nearest real image
edge, and refining the scale card's corners by fitting lines to its sides.

The method is the same in both cases. Walk a short distance along the local
normal, sample the gradient magnitude, take the strongest response, and
interpolate its position with a parabola through the peak and its neighbours.

The search radius is deliberately short. These photographs are grey stone on
grey sand with dry grass across them, and a generous search window will happily
snap to a grass blade or a shadow edge instead of the object. A conservative
radius that sometimes fails to improve a click is much safer than one that
confidently moves it somewhere wrong, and the raw clicks are kept regardless so
the operator's intent is never lost.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = [
    "gradient_magnitude",
    "sample_bilinear",
    "snap_points",
    "refine_polyline_edge",
]


def gradient_magnitude(gray: np.ndarray, smooth: float = 1.5) -> np.ndarray:
    """Smoothed Sobel gradient magnitude.

    The pre-smoothing matters on this material: unsmoothed, sand grain and JPEG
    noise produce a gradient field with no dominant edge to find.
    """
    from scipy import ndimage

    g = np.asarray(gray, float)
    if smooth and smooth > 0:
        g = ndimage.gaussian_filter(g, smooth)
    gx = ndimage.sobel(g, axis=1, mode="nearest")
    gy = ndimage.sobel(g, axis=0, mode="nearest")
    return np.hypot(gx, gy)


def sample_bilinear(img: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinear sample of ``img`` at (x, y) coordinates, edge-clamped."""
    img = np.asarray(img, float)
    h, w = img.shape[:2]
    xy = np.atleast_2d(np.asarray(xy, float))
    x = np.clip(xy[:, 0], 0, w - 1.001)
    y = np.clip(xy[:, 1], 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = x0 + 1, y0 + 1
    fx, fy = x - x0, y - y0
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
            + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)


def _peak_offsets(profiles: np.ndarray, offsets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sub-pixel peak position and strength for each row of ``profiles``."""
    k = np.argmax(profiles, axis=1)
    n = profiles.shape[1]
    ki = np.clip(k, 1, n - 2)
    rows = np.arange(profiles.shape[0])
    fm, f0, fp = profiles[rows, ki - 1], profiles[rows, ki], profiles[rows, ki + 1]

    denom = fm - 2.0 * f0 + fp
    frac = np.where(np.abs(denom) > 1e-12, 0.5 * (fm - fp) / np.where(np.abs(denom) > 1e-12, denom, 1.0), 0.0)
    frac = np.clip(frac, -1.0, 1.0)
    # A peak pinned to the window edge is not a peak; refuse to interpolate it.
    frac = np.where((k == 0) | (k == n - 1), 0.0, frac)

    step = offsets[1] - offsets[0]
    pos = offsets[ki] + frac * step
    return pos, f0


def snap_points(gray: np.ndarray, points: np.ndarray, normals: np.ndarray,
                search_px: float = 6.0, samples: int = 25,
                min_contrast: float = 0.0,
                smooth: float = 1.5,
                mag: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Move each point onto the strongest nearby edge along its normal.

    Returns ``(snapped, moved)`` where ``moved`` marks the points that actually
    found a peak. Points whose response never exceeds ``min_contrast`` are left
    exactly where they were, which is the desired behaviour on the low-contrast
    stretches where there is no edge to find.
    """
    if mag is None:
        mag = gradient_magnitude(gray, smooth=smooth)
    pts = np.atleast_2d(np.asarray(points, float))
    nrm = np.atleast_2d(np.asarray(normals, float))
    if pts.shape[0] == 0:
        return pts.copy(), np.zeros(0, bool)

    norm = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.where(norm > 1e-12, norm, 1.0)

    offsets = np.linspace(-search_px, search_px, int(samples))
    probes = pts[:, None, :] + offsets[None, :, None] * nrm[:, None, :]
    profiles = sample_bilinear(mag, probes.reshape(-1, 2)).reshape(pts.shape[0], len(offsets))

    pos, strength = _peak_offsets(profiles, offsets)
    ok = strength > min_contrast
    snapped = pts + np.where(ok[:, None], pos[:, None], 0.0) * nrm
    return snapped, ok


def _fit_line_tls(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Total-least-squares line fit; returns (point_on_line, unit_direction)."""
    c = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - c)
    return c, Vt[0]


def _intersect(p1, d1, p2, d2) -> Optional[np.ndarray]:
    A = np.column_stack([d1, -d2])
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    ts = np.linalg.solve(A, p2 - p1)
    return p1 + ts[0] * d1


def refine_polyline_edge(gray: np.ndarray, quad: np.ndarray, search_px: float = 6.0,
                         samples_per_edge: int = 40, trim_frac: float = 0.15,
                         robust_iters: int = 2) -> Tuple[np.ndarray, float]:
    """Refine a quadrilateral by fitting lines to its four edges.

    Far more precise than refining each corner in isolation: every corner is
    determined by the hundreds of pixels along its two edges rather than by one
    small patch, and a corner is exactly where a corner detector is least
    reliable because both edges are turning there.

    Returns ``(corners, rms_residual_px)``.
    """
    quad = np.asarray(quad, float).reshape(4, 2)
    mag = gradient_magnitude(gray)

    lines = []
    residuals = []
    for i in range(4):
        p0, p1 = quad[i], quad[(i + 1) % 4]
        edge = p1 - p0
        length = np.hypot(*edge)
        if length < 4:
            return quad, float("nan")
        # Skip the ends: near a corner the perpendicular search straddles the
        # adjoining edge and pulls the fit off the line.
        ts = np.linspace(trim_frac, 1.0 - trim_frac, samples_per_edge)
        base = p0[None, :] + ts[:, None] * edge[None, :]
        normal = np.array([-edge[1], edge[0]]) / length
        normals = np.repeat(normal[None, :], len(base), axis=0)

        snapped, ok = snap_points(gray, base, normals, search_px=search_px, mag=mag)
        pts = snapped[ok]
        if len(pts) < 6:
            return quad, float("nan")

        c, d = _fit_line_tls(pts)
        for _ in range(robust_iters):
            perp = np.abs((pts - c) @ np.array([-d[1], d[0]]))
            keep = perp <= max(1.0, 2.5 * np.median(perp) + 1e-9)
            if keep.sum() < 6:
                break
            pts = pts[keep]
            c, d = _fit_line_tls(pts)
        residuals.append(np.abs((pts - c) @ np.array([-d[1], d[0]])))
        lines.append((c, d))

    corners = []
    for i in range(4):
        # Corner i is where edge (i-1) meets edge i.
        pt = _intersect(*lines[(i - 1) % 4], *lines[i])
        if pt is None:
            return quad, float("nan")
        corners.append(pt)
    corners = np.array(corners)

    # Reject a refinement that wandered to a different feature entirely.
    if np.abs(corners - quad).max() > 4.0 * search_px:
        return quad, float("nan")

    rms = float(np.sqrt(np.mean(np.concatenate(residuals) ** 2)))
    return corners, rms
