"""Circle and ellipse fitting for partially preserved arcs.

The estimator used throughout is the *geometric* (orthogonal-distance) fit: it
minimises true perpendicular distances from the digitised points to the curve,
which is the maximum-likelihood estimator under isotropic Gaussian point noise.
Algebraic fits are used only to seed it, because algebraic distance is biased
towards small, low-eccentricity ellipses -- exactly the bias that would corrupt
a circular-vs-oval decision on a short arc.

Everything here is *batched*: the public fitters accept a stack of B independent
problems and solve them simultaneously with a damped Gauss-Newton iteration
sharing one vectorised inner loop. This matters. The bootstrap in ``stats``
needs several thousand refits per object across 167 objects; looping
``scipy.optimize.least_squares`` over them would take hours, whereas the batched
solver does the same work in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.special import ellipeinc

__all__ = [
    "EllipseParams",
    "CircleParams",
    "fit_ellipse_algebraic",
    "fit_ellipse",
    "fit_ellipse_batch",
    "fit_circle_algebraic",
    "fit_circle",
    "fit_circle_batch",
    "ellipse_residuals",
    "circle_residuals",
    "ellipse_points",
    "conic_to_params",
    "params_to_conic",
    "arc_coverage",
    "FitError",
]

_TINY = 1e-12


class FitError(ValueError):
    """Raised when a fit cannot be formed at all (e.g. too few points)."""


@dataclass(frozen=True)
class EllipseParams:
    """An ellipse in world coordinates, with ``a >= b > 0``.

    ``theta`` is the orientation of the major axis, in radians, wrapped to
    [-pi/2, pi/2).
    """

    cx: float
    cy: float
    a: float
    b: float
    theta: float

    @property
    def eccentricity(self) -> float:
        return float(np.sqrt(max(0.0, 1.0 - (self.b / self.a) ** 2)))

    @property
    def major_axis(self) -> float:
        """Maximum length through the centre."""
        return 2.0 * self.a

    @property
    def minor_axis(self) -> float:
        return 2.0 * self.b

    @property
    def perimeter(self) -> float:
        e2 = 1.0 - (self.b / self.a) ** 2
        return float(4.0 * self.a * ellipeinc(np.pi / 2.0, e2))

    def as_array(self) -> np.ndarray:
        return np.array([self.cx, self.cy, self.a, self.b, self.theta], float)

    @staticmethod
    def from_array(p: np.ndarray) -> "EllipseParams":
        return EllipseParams(float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]))


@dataclass(frozen=True)
class CircleParams:
    cx: float
    cy: float
    r: float

    @property
    def diameter(self) -> float:
        return 2.0 * self.r

    def as_array(self) -> np.ndarray:
        return np.array([self.cx, self.cy, self.r], float)


# --------------------------------------------------------------------------
# conic <-> geometric parameter conversion
# --------------------------------------------------------------------------

def conic_to_params(coeffs: np.ndarray) -> EllipseParams:
    """Convert conic ``A x^2 + B xy + C y^2 + D x + E y + F = 0`` to geometry.

    Both the axis lengths and the orientation are taken from a single
    eigendecomposition of the quadratic part, so they cannot disagree.

    The sign normalisation on the first line is load-bearing. A conic is defined
    only up to scale *including sign*, and closed-form axis-length formulas are
    invariant to that flip while orientation formulas are not -- so a negated
    conic can otherwise yield the major axis length paired with the minor axis
    direction, which fits catastrophically badly but looks plausible.

    Raises FitError if the conic is not a real ellipse.
    """
    coeffs = np.asarray(coeffs, float).ravel()
    if coeffs[0] + coeffs[2] < 0:
        coeffs = -coeffs
    A, B, C, D, E, F = (float(v) for v in coeffs)

    if B * B - 4.0 * A * C >= -_TINY:
        raise FitError("conic is not elliptical (discriminant >= 0)")

    M = np.array([[A, B / 2.0], [B / 2.0, C]], float)
    L = np.array([D, E], float)
    try:
        centre = -0.5 * np.linalg.solve(M, L)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - degenerate input
        raise FitError("conic has no unique centre") from exc

    # Constant term after translating the conic to its centre.
    f_centred = 0.5 * float(L @ centre) + F
    if -f_centred <= _TINY:
        raise FitError("degenerate or imaginary ellipse")

    evals, evecs = np.linalg.eigh(M)
    if np.any(evals <= _TINY):
        raise FitError("quadratic part is not positive definite")

    # Centred conic is lambda_1 u^2 + lambda_2 v^2 = -f_centred, so the semi-axis
    # along eigenvector i is sqrt(-f_centred / lambda_i). The *smaller*
    # eigenvalue therefore gives the *longer* axis.
    semi = np.sqrt(-f_centred / evals)
    i_major = int(np.argmin(evals))
    i_minor = 1 - i_major
    major_vec = evecs[:, i_major]

    return EllipseParams(
        float(centre[0]),
        float(centre[1]),
        float(semi[i_major]),
        float(semi[i_minor]),
        _wrap_theta(float(np.arctan2(major_vec[1], major_vec[0]))),
    )


def params_to_conic(ell: EllipseParams) -> np.ndarray:
    """Inverse of :func:`conic_to_params`, normalised so that ``A^2+B^2+C^2 = 1``.

    Provided so the round trip can be asserted in tests rather than trusted.
    """
    ct, st = np.cos(ell.theta), np.sin(ell.theta)
    a2, b2 = ell.a ** 2, ell.b ** 2
    A = ct * ct / a2 + st * st / b2
    B = 2.0 * ct * st * (1.0 / a2 - 1.0 / b2)
    C = st * st / a2 + ct * ct / b2
    D = -2.0 * A * ell.cx - B * ell.cy
    E = -B * ell.cx - 2.0 * C * ell.cy
    F = A * ell.cx ** 2 + B * ell.cx * ell.cy + C * ell.cy ** 2 - 1.0
    coeffs = np.array([A, B, C, D, E, F], float)
    return coeffs / np.linalg.norm(coeffs[:3])


def _wrap_theta(theta: float) -> float:
    """Wrap a major-axis orientation into [-pi/2, pi/2)."""
    t = (float(theta) + np.pi / 2.0) % np.pi - np.pi / 2.0
    return t


# --------------------------------------------------------------------------
# algebraic seeds
# --------------------------------------------------------------------------

def fit_ellipse_algebraic(x: np.ndarray, y: np.ndarray) -> EllipseParams:
    """Halir & Flusser (1998) numerically stable direct least-squares fit.

    Guarantees an elliptical (not hyperbolic) solution. Coordinates are
    normalised to unit scale first and the conic un-normalised afterwards;
    without that step the design matrix conditioning degrades badly for
    pixel-scale coordinates in the thousands.
    """
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    if x.size < 5:
        raise FitError(f"need >= 5 points for an ellipse, got {x.size}")

    mx, my = x.mean(), y.mean()
    s = float(np.sqrt(((x - mx) ** 2 + (y - my) ** 2).mean()))
    if s < _TINY:
        raise FitError("all points are coincident")
    u, v = (x - mx) / s, (y - my) / s

    D1 = np.column_stack([u * u, u * v, v * v])
    D2 = np.column_stack([u, v, np.ones_like(u)])
    S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2

    try:
        T = -np.linalg.solve(S3, S2.T)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - degenerate input
        raise FitError("singular scatter matrix") from exc

    M = S1 + S2 @ T
    # Pre-multiply by inv(C1) where C1 is the ellipse constraint matrix
    # [[0,0,2],[0,-1,0],[2,0,0]]; done by row shuffling rather than an inverse.
    M = np.array([M[2] / 2.0, -M[1], M[0] / 2.0])

    evals, evecs = np.linalg.eig(M)
    cond = 4.0 * evecs[0] * evecs[2] - evecs[1] ** 2
    valid = np.where(cond > 0)[0]
    if valid.size == 0:
        raise FitError("no elliptical solution in the eigenspace")
    a1 = np.real(evecs[:, valid[0]])
    a2 = T @ a1
    conic_n = np.concatenate([a1, a2])  # in normalised (u, v) coordinates

    return conic_to_params(_unnormalise_conic(conic_n, mx, my, s))


def _unnormalise_conic(c: np.ndarray, mx: float, my: float, s: float) -> np.ndarray:
    """Map a conic in u=(x-mx)/s, v=(y-my)/s back to x, y coordinates."""
    A, B, C, D, E, F = (float(v) for v in c)
    s2 = s * s
    nA = A / s2
    nB = B / s2
    nC = C / s2
    nD = (D / s) - (2.0 * A * mx + B * my) / s2
    nE = (E / s) - (2.0 * C * my + B * mx) / s2
    nF = (
        F
        - (D * mx + E * my) / s
        + (A * mx * mx + B * mx * my + C * my * my) / s2
    )
    return np.array([nA, nB, nC, nD, nE, nF], float)


def fit_circle_algebraic(x: np.ndarray, y: np.ndarray) -> CircleParams:
    """Pratt (1987) algebraic circle fit, used to seed the geometric fit."""
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    if x.size < 3:
        raise FitError(f"need >= 3 points for a circle, got {x.size}")
    mx, my = x.mean(), y.mean()
    u, v = x - mx, y - my
    # Kasa/Coope linear system on the centred points; adequate as a seed and
    # immune to the sign pathologies of the general Pratt eigen-formulation.
    A = np.column_stack([2.0 * u, 2.0 * v, np.ones_like(u)])
    rhs = u * u + v * v
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    cx, cy = sol[0], sol[1]
    r2 = sol[2] + cx * cx + cy * cy
    if r2 <= 0:
        raise FitError("degenerate circle fit")
    return CircleParams(float(cx + mx), float(cy + my), float(np.sqrt(r2)))


# --------------------------------------------------------------------------
# closest point on an ellipse (Eberly), vectorised
# --------------------------------------------------------------------------

def _foot_canonical(a: np.ndarray, b: np.ndarray, y0: np.ndarray, y1: np.ndarray,
                    iters: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    """Closest point on ``(x/a)^2 + (y/b)^2 = 1`` to a first-quadrant point.

    Vectorised implementation of Eberly's robust bisection. Inputs broadcast to
    a common shape; ``a >= b > 0`` and ``y0, y1 >= 0`` are required. Bisection
    is used rather than Newton because it cannot diverge, which matters when
    the bootstrap drives points into near-degenerate configurations.

    40 halvings resolve the bracket to ~1e-12 of its width, far below the
    precision the outer Gauss-Newton iteration needs; this loop runs tens of
    thousands of times per object, so the count is worth not inflating.
    """
    a, b, y0, y1 = np.broadcast_arrays(a, b, y0, y1)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    y0 = np.asarray(y0, float)
    y1 = np.asarray(y1, float)

    x0 = np.empty_like(y0)
    x1 = np.empty_like(y1)

    # A near-circular ellipse makes a^2 - b^2 vanish, so the general branch's
    # denominators blow up. Radial projection is exact in that case anyway.
    circle = (a - b) <= 1e-9 * np.maximum(a, _TINY)
    if np.any(circle):
        rad = np.hypot(y0, y1)
        safe = np.maximum(rad, _TINY)
        cx0 = np.where(rad > _TINY, a * y0 / safe, a)
        cx1 = np.where(rad > _TINY, a * y1 / safe, 0.0)
        x0 = np.where(circle, cx0, x0)
        x1 = np.where(circle, cx1, x1)

    # y1 == 0 needs its own branch: the bisection bracket collapses there.
    on_axis = (~circle) & (y1 <= _TINY)
    if np.any(on_axis):
        denom0 = a * a - b * b
        numer0 = a * y0
        inside = numer0 < denom0
        xde0 = np.where(inside, numer0 / np.where(denom0 > _TINY, denom0, 1.0), 0.0)
        xde0 = np.clip(xde0, 0.0, 1.0)
        ax0 = np.where(inside, a * xde0, a)
        ax1 = np.where(inside, b * np.sqrt(np.maximum(0.0, 1.0 - xde0 * xde0)), 0.0)
        x0 = np.where(on_axis, ax0, x0)
        x1 = np.where(on_axis, ax1, x1)

    general = (~circle) & (~on_axis)
    if np.any(general):
        # Work on the full arrays with safe substitutes off-mask, then select.
        a_g = np.where(general, a, 1.0)
        b_g = np.where(general, b, 0.5)
        y0_g = np.where(general, y0, 1.0)
        y1_g = np.where(general, y1, 1.0)

        z0 = y0_g / a_g
        z1 = y1_g / b_g
        g = z0 * z0 + z1 * z1 - 1.0
        r0 = (a_g / b_g) ** 2
        n0 = r0 * z0

        # Eberly bisects on s, then divides by (s + 1). For a point near the
        # ellipse centre the root sits at s = z1 - 1 with z1 ~ 0, so s ~ -1 and
        # (s + 1) loses almost all its significant digits to cancellation.
        # Substituting w = s + 1 and bisecting on w keeps the small quantity
        # exact: no subtraction ever forms it. Then s + r0 = w + (r0 - 1), and
        # r0 >= 1 here because the near-circular case is handled separately.
        w0 = z1
        w1 = np.where(g < 0.0, 1.0, np.sqrt(n0 * n0 + z1 * z1))
        rm1 = r0 - 1.0

        # Geometric rather than arithmetic midpoint. The bracket endpoints are
        # both strictly positive and the function is monotone in w, so this is
        # still a valid bisection -- but it converges in *relative* terms. That
        # matters because the root can legitimately sit at w ~ 1e-10 (a point at
        # the ellipse centre), where halving an O(1) bracket arithmetically
        # would need far more iterations to resolve any significant digits.
        w = np.sqrt(w0 * w1)
        for _ in range(iters):
            w = np.sqrt(w0 * w1)
            ratio0 = n0 / (w + rm1)
            ratio1 = z1 / w
            gg = ratio0 * ratio0 + ratio1 * ratio1 - 1.0
            w0 = np.where(gg > 0.0, w, w0)
            w1 = np.where(gg < 0.0, w, w1)

        gx0 = r0 * y0_g / (w + rm1)
        gx1 = y1_g / w
        x0 = np.where(general, gx0, x0)
        x1 = np.where(general, gx1, x1)

    return x0, x1


def _ellipse_foot_and_anomaly(params: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Foot points and eccentric anomalies for a batch of ellipse problems.

    ``params`` is (B, 5); ``x``/``y`` are (B, n). Returns canonical-frame foot
    coordinates, the eccentric anomaly t of each foot, and the signed distance
    (positive when the point lies outside the ellipse).
    """
    cx = params[:, 0:1]
    cy = params[:, 1:2]
    a = params[:, 2:3]
    b = params[:, 3:4]
    th = params[:, 4:5]

    ct, st = np.cos(th), np.sin(th)
    dx, dy = x - cx, y - cy
    # rotate into the ellipse frame
    px = ct * dx + st * dy
    py = -st * dx + ct * dy

    sx = np.sign(px)
    sy = np.sign(py)
    sx = np.where(sx == 0, 1.0, sx)
    sy = np.where(sy == 0, 1.0, sy)

    qx, qy = _foot_canonical(a, b, np.abs(px), np.abs(py))
    qx = qx * sx
    qy = qy * sy

    dist = np.hypot(px - qx, py - qy)
    outside = (px / a) ** 2 + (py / b) ** 2 > 1.0
    signed = np.where(outside, dist, -dist)

    t = np.arctan2(qy / b, qx / a)
    return qx, qy, t, signed


def ellipse_residuals(ell: EllipseParams, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Signed orthogonal distances from points to an ellipse (outside > 0)."""
    p = ell.as_array()[None, :]
    x = np.asarray(x, float).ravel()[None, :]
    y = np.asarray(y, float).ravel()[None, :]
    _, _, _, signed = _ellipse_foot_and_anomaly(p, x, y)
    return signed[0]


def circle_residuals(circ: CircleParams, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Signed radial distances from points to a circle (outside > 0)."""
    return np.hypot(np.asarray(x, float) - circ.cx, np.asarray(y, float) - circ.cy) - circ.r


def ellipse_points(ell: EllipseParams, n: int = 512, t0: float = 0.0,
                   t1: float = 2.0 * np.pi) -> np.ndarray:
    """Sample ``n`` points along the ellipse between eccentric anomalies t0, t1."""
    t = np.linspace(t0, t1, int(n))
    ct, st = np.cos(ell.theta), np.sin(ell.theta)
    qx, qy = ell.a * np.cos(t), ell.b * np.sin(t)
    return np.column_stack([ell.cx + ct * qx - st * qy, ell.cy + st * qx + ct * qy])


# --------------------------------------------------------------------------
# batched geometric fitting
# --------------------------------------------------------------------------

def _canonicalise(params: np.ndarray) -> np.ndarray:
    """Enforce a >= b > 0 and theta in [-pi/2, pi/2) for a (B, 5) stack."""
    p = params.copy()
    p[:, 2] = np.abs(p[:, 2])
    p[:, 3] = np.abs(p[:, 3])
    swap = p[:, 3] > p[:, 2]
    if np.any(swap):
        a_old = p[swap, 2].copy()
        p[swap, 2] = p[swap, 3]
        p[swap, 3] = a_old
        p[swap, 4] += np.pi / 2.0
    p[:, 2] = np.maximum(p[:, 2], _TINY)
    p[:, 3] = np.maximum(p[:, 3], _TINY)
    p[:, 4] = (p[:, 4] + np.pi / 2.0) % np.pi - np.pi / 2.0
    return p


def _ellipse_jacobian(params: np.ndarray, t: np.ndarray, signed: np.ndarray,
                      x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Analytic Jacobian of the signed orthogonal residual, shape (B, n, 5).

    By the envelope theorem the derivative of a minimised distance with respect
    to the curve parameters is just the normal component of the foot point's
    motion -- the derivative of the foot's *position along* the curve drops out
    at the minimum. That makes this exact and cheap, which is what lets the
    Gauss-Newton loop converge in a handful of iterations.
    """
    a = params[:, 2:3]
    b = params[:, 3:4]
    th = params[:, 4:5]
    ct, st = np.cos(th), np.sin(th)
    cost, sint = np.cos(t), np.sin(t)

    # foot point in world coordinates
    qx, qy = a * cost, b * sint
    fx = params[:, 0:1] + ct * qx - st * qy
    fy = params[:, 1:2] + st * qx + ct * qy

    d = np.hypot(x - fx, y - fy)
    d_safe = np.where(d > _TINY, d, _TINY)
    sgn = np.sign(signed)
    sgn = np.where(sgn == 0, 1.0, sgn)
    # outward unit normal at the foot point
    nx = sgn * (x - fx) / d_safe
    ny = sgn * (y - fy) / d_safe

    # dC/dp for p = (cx, cy, a, b, theta)
    dC = np.empty(t.shape + (5, 2), float)
    dC[..., 0, 0] = 1.0
    dC[..., 0, 1] = 0.0
    dC[..., 1, 0] = 0.0
    dC[..., 1, 1] = 1.0
    dC[..., 2, 0] = ct * cost
    dC[..., 2, 1] = st * cost
    dC[..., 3, 0] = -st * sint
    dC[..., 3, 1] = ct * sint
    dC[..., 4, 0] = -st * qx - ct * qy
    dC[..., 4, 1] = ct * qx - st * qy

    return -(dC[..., 0] * nx[..., None] + dC[..., 1] * ny[..., None])


def fit_ellipse_batch(x: np.ndarray, y: np.ndarray, init: np.ndarray,
                      iters: int = 40, tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """Geometric ellipse fit for a stack of B problems.

    Damped Gauss-Newton (Levenberg) with a per-problem damping factor, so one
    badly conditioned replicate cannot stall the rest of the batch.

    Replicates drop out of the working set as they converge or stall. Bootstrap
    replicates start from the point estimate and most converge within a handful
    of iterations, so without compaction nearly all the work would be spent
    re-solving already-converged problems while waiting for the slowest few.

    Returns ``(params, converged)`` with shapes (B, 5) and (B,).
    """
    x = np.atleast_2d(np.asarray(x, float))
    y = np.atleast_2d(np.asarray(y, float))
    p = _canonicalise(np.atleast_2d(np.asarray(init, float)).copy())
    B = p.shape[0]
    if x.shape[0] == 1 and B > 1:
        x = np.broadcast_to(x, (B, x.shape[1]))
        y = np.broadcast_to(y, (B, y.shape[1]))

    lam = np.full(B, 1e-3)
    _, _, t, signed = _ellipse_foot_and_anomaly(p, x, y)
    cost = np.sum(signed ** 2, axis=1)
    converged = np.zeros(B, bool)
    active = np.ones(B, bool)
    eye = np.eye(5)[None, :, :]

    for _ in range(iters):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break

        p_a, x_a, y_a = p[idx], x[idx], y[idx]
        t_a, signed_a, cost_a, lam_a = t[idx], signed[idx], cost[idx], lam[idx]

        J = _ellipse_jacobian(p_a, t_a, signed_a, x_a, y_a)     # (k, n, 5)
        JTJ = np.einsum("bni,bnj->bij", J, J)
        JTr = np.einsum("bni,bn->bi", J, signed_a)

        # Scale-aware damping: add lambda * diag(JTJ) rather than lambda * I so
        # the damping is invariant to the units the coordinates are in.
        diag = np.maximum(np.einsum("bii->bi", JTJ), _TINY)
        Aug = JTJ + lam_a[:, None, None] * diag[:, :, None] * eye

        try:
            step = np.linalg.solve(Aug, -JTr[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            step = np.zeros_like(p_a)
            for i in range(idx.size):
                try:
                    step[i] = np.linalg.lstsq(Aug[i], -JTr[i], rcond=None)[0]
                except np.linalg.LinAlgError:
                    step[i] = 0.0

        cand = _canonicalise(p_a + step)
        _, _, t_c, signed_c = _ellipse_foot_and_anomaly(cand, x_a, y_a)
        cost_c = np.sum(signed_c ** 2, axis=1)

        better = cost_c < cost_a
        p_a = np.where(better[:, None], cand, p_a)
        t_a = np.where(better[:, None], t_c, t_a)
        signed_a = np.where(better[:, None], signed_c, signed_a)
        improvement = cost_a - np.where(better, cost_c, cost_a)
        cost_a = np.where(better, cost_c, cost_a)
        lam_a = np.where(better, np.maximum(lam_a / 3.0, 1e-12),
                         np.minimum(lam_a * 3.0, 1e8))

        done = better & (improvement <= tol * np.maximum(cost_a, _TINY))
        stalled = lam_a >= 1e8

        p[idx], t[idx], signed[idx] = p_a, t_a, signed_a
        cost[idx], lam[idx] = cost_a, lam_a
        converged[idx] |= done
        active[idx] = ~(done | stalled)

    return p, converged


def fit_ellipse(x: np.ndarray, y: np.ndarray,
                init: EllipseParams | None = None) -> EllipseParams:
    """Geometric ellipse fit for a single problem, seeded algebraically."""
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    if x.size < 5:
        raise FitError(f"need >= 5 points for an ellipse, got {x.size}")
    seed = init if init is not None else fit_ellipse_algebraic(x, y)
    p, _ = fit_ellipse_batch(x[None, :], y[None, :], seed.as_array()[None, :])
    return EllipseParams.from_array(p[0])


def fit_circle_batch(x: np.ndarray, y: np.ndarray, init: np.ndarray,
                     iters: int = 50, tol: float = 1e-12) -> np.ndarray:
    """Geometric circle fit for a stack of B problems. ``init`` is (B, 3)."""
    x = np.atleast_2d(np.asarray(x, float))
    y = np.atleast_2d(np.asarray(y, float))
    p = np.atleast_2d(np.asarray(init, float)).copy()

    for _ in range(iters):
        dx = x - p[:, 0:1]
        dy = y - p[:, 1:2]
        d = np.hypot(dx, dy)
        d_safe = np.where(d > _TINY, d, _TINY)
        r = d - p[:, 2:3]
        J = np.stack([-dx / d_safe, -dy / d_safe, -np.ones_like(d)], axis=-1)
        JTJ = np.einsum("bni,bnj->bij", J, J)
        JTr = np.einsum("bni,bn->bi", J, r)
        JTJ = JTJ + 1e-12 * np.eye(3)[None, :, :]
        try:
            step = np.linalg.solve(JTJ, -JTr[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate input
            break
        p = p + step
        p[:, 2] = np.abs(p[:, 2])
        if np.all(np.abs(step) <= tol * (1.0 + np.abs(p))):
            break
    return p


def fit_circle(x: np.ndarray, y: np.ndarray) -> CircleParams:
    """Geometric circle fit for a single problem, seeded algebraically."""
    seed = fit_circle_algebraic(x, y)
    p = fit_circle_batch(np.asarray(x, float)[None, :], np.asarray(y, float)[None, :],
                         seed.as_array()[None, :])
    return CircleParams(float(p[0, 0]), float(p[0, 1]), float(p[0, 2]))


# --------------------------------------------------------------------------
# arc coverage
# --------------------------------------------------------------------------

def _arclen_to(t: np.ndarray, a: float, e2: float) -> np.ndarray:
    """Arc length along (a cos t, b sin t) from t=0 to t, for e2 = 1 - b^2/a^2."""
    return a * (ellipeinc(np.pi / 2.0, e2) - ellipeinc(np.pi / 2.0 - t, e2))


def arc_coverage(ell: EllipseParams, x: np.ndarray, y: np.ndarray) -> dict:
    """How much of the reconstructed outline the digitised points actually span.

    Returned as both an angular span in degrees and a fraction of the true
    perimeter. The perimeter fraction is the honest measure for an elongated
    ellipse, where equal angular steps cover very unequal arc lengths.

    Both are computed as ``total - largest gap``, so a fragment digitised as one
    continuous run is measured across that run rather than around the long way.
    """
    p = ell.as_array()[None, :]
    xs = np.asarray(x, float).ravel()[None, :]
    ys = np.asarray(y, float).ravel()[None, :]
    _, _, t, _ = _ellipse_foot_and_anomaly(p, xs, ys)
    t = np.sort(np.mod(t[0], 2.0 * np.pi))

    if t.size < 2:
        return {"coverage_deg": 0.0, "coverage_perimeter_frac": 0.0,
                "t_start": float(t[0]) if t.size else 0.0,
                "t_end": float(t[0]) if t.size else 0.0}

    gaps = np.diff(np.concatenate([t, t[:1] + 2.0 * np.pi]))
    k = int(np.argmax(gaps))
    coverage_rad = 2.0 * np.pi - gaps[k]

    # The arc runs from the point after the largest gap round to the point before it.
    t_start = t[(k + 1) % t.size]
    t_end = t[k]

    e2 = 1.0 - (ell.b / ell.a) ** 2
    span = np.mod(t_end - t_start, 2.0 * np.pi)
    covered = float(_arclen_to(np.array(t_start + span), ell.a, e2)
                    - _arclen_to(np.array(t_start), ell.a, e2))
    perim = ell.perimeter

    return {
        "coverage_deg": float(np.degrees(coverage_rad)),
        "coverage_perimeter_frac": float(np.clip(covered / perim, 0.0, 1.0)),
        "t_start": float(t_start),
        "t_end": float(t_end),
    }


def foot_frame(ell: EllipseParams, x: np.ndarray, y: np.ndarray) -> dict:
    """Foot points, outward normals and residuals for points against an ellipse.

    The residual bootstrap needs to rebuild points as ``foot + r * normal``, so
    it needs the geometry of the projection rather than just its magnitude.
    """
    p = ell.as_array()[None, :]
    xs = np.asarray(x, float).ravel()[None, :]
    ys = np.asarray(y, float).ravel()[None, :]
    qx, qy, t, signed = _ellipse_foot_and_anomaly(p, xs, ys)
    qx, qy, t, signed = qx[0], qy[0], t[0], signed[0]

    ct, st = np.cos(ell.theta), np.sin(ell.theta)
    foot = np.column_stack([ell.cx + ct * qx - st * qy,
                            ell.cy + st * qx + ct * qy])

    # Outward normal of (x/a)^2 + (y/b)^2 = 1 is the gradient (x/a^2, y/b^2).
    nx_c, ny_c = qx / ell.a ** 2, qy / ell.b ** 2
    norm = np.hypot(nx_c, ny_c)
    norm = np.where(norm > _TINY, norm, 1.0)
    nx_c, ny_c = nx_c / norm, ny_c / norm
    normal = np.column_stack([ct * nx_c - st * ny_c, st * nx_c + ct * ny_c])

    return {"foot": foot, "normal": normal, "t": t, "residual": signed}
