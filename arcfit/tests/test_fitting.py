"""Correctness tests for the geometric fitting core.

These check the estimator against ground truth and against independently
computed quantities (brute-force distance minimisation, numerical quadrature)
rather than against previously recorded outputs, so a regression in the maths
fails here rather than being frozen in.
"""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from arcfit.fitting import (
    CircleParams,
    EllipseParams,
    FitError,
    arc_coverage,
    conic_to_params,
    ellipse_residuals,
    fit_circle,
    fit_ellipse,
    fit_ellipse_algebraic,
    fit_ellipse_batch,
    params_to_conic,
)


def sample_arc(ell, n, t0, t1, seed=None, noise=0.0):
    """Points along an ellipse between eccentric anomalies t0 and t1."""
    rng = np.random.default_rng(seed)
    t = np.linspace(t0, t1, n)
    ct, st = np.cos(ell.theta), np.sin(ell.theta)
    qx, qy = ell.a * np.cos(t), ell.b * np.sin(t)
    x = ell.cx + ct * qx - st * qy
    y = ell.cy + st * qx + ct * qy
    if noise:
        x = x + rng.normal(0, noise, n)
        y = y + rng.normal(0, noise, n)
    return x, y


class TestConicRoundTrip:
    @pytest.mark.parametrize("theta", [0.0, 0.3, -0.7, 1.2, np.pi / 2 - 1e-6])
    @pytest.mark.parametrize("ab", [(5.0, 3.0), (10.0, 9.9), (2.0, 0.5)])
    def test_params_to_conic_and_back(self, theta, ab):
        ell = EllipseParams(3.0, -2.0, ab[0], ab[1], theta)
        back = conic_to_params(params_to_conic(ell))
        assert back.cx == pytest.approx(ell.cx, abs=1e-8)
        assert back.cy == pytest.approx(ell.cy, abs=1e-8)
        assert back.a == pytest.approx(ell.a, rel=1e-9)
        assert back.b == pytest.approx(ell.b, rel=1e-9)
        # orientation is defined modulo pi
        d = (back.theta - ell.theta + np.pi / 2) % np.pi - np.pi / 2
        assert d == pytest.approx(0.0, abs=1e-7)

    def test_circle_conic_center(self):
        # x^2 + y^2 - 2x = 0  ->  centre (1, 0), radius 1
        ell = conic_to_params(np.array([1.0, 0.0, 1.0, -2.0, 0.0, 0.0]))
        assert ell.cx == pytest.approx(1.0)
        assert ell.cy == pytest.approx(0.0)
        assert ell.a == pytest.approx(1.0)
        assert ell.b == pytest.approx(1.0)

    @pytest.mark.parametrize("theta", [0.9, -0.3, 1.4])
    @pytest.mark.parametrize("sign", [1.0, -1.0])
    def test_negated_conic_is_identical(self, theta, sign):
        """A conic is defined only up to scale *including sign*.

        Regression test: an earlier version took axis lengths from a
        sign-invariant closed form but orientation from a sign-sensitive one, so
        a negated conic returned the major axis length with the minor axis
        direction -- a fit that is catastrophically wrong but looks plausible.
        """
        ell = EllipseParams(4.0, -3.0, 20.0, 4.0, theta)
        got = conic_to_params(sign * params_to_conic(ell))
        assert got.a == pytest.approx(20.0, rel=1e-9)
        assert got.b == pytest.approx(4.0, rel=1e-9)
        d = (got.theta - theta + np.pi / 2) % np.pi - np.pi / 2
        assert d == pytest.approx(0.0, abs=1e-9)

    def test_hyperbola_rejected(self):
        with pytest.raises(FitError):
            conic_to_params(np.array([1.0, 0.0, -1.0, 0.0, 0.0, -1.0]))


class TestExactRecovery:
    """On noise-free points the geometric fit must recover the truth exactly."""

    @pytest.mark.parametrize("a,b,theta", [
        (10.0, 6.0, 0.4),
        (7.5, 7.5, 0.0),      # a true circle
        (12.0, 11.8, -1.1),   # nearly circular
        (20.0, 4.0, 0.9),     # strongly elongated
    ])
    def test_full_perimeter(self, a, b, theta):
        truth = EllipseParams(4.0, -3.0, a, b, theta)
        x, y = sample_arc(truth, 60, 0, 2 * np.pi)
        got = fit_ellipse(x, y)
        assert got.a == pytest.approx(truth.a, rel=1e-6)
        assert got.b == pytest.approx(truth.b, rel=1e-6)
        assert got.cx == pytest.approx(truth.cx, abs=1e-6)
        assert got.cy == pytest.approx(truth.cy, abs=1e-6)

    @pytest.mark.parametrize("span_deg", [360, 180, 120, 90])
    def test_partial_arc_noise_free(self, span_deg):
        """Even a 90-degree arc is exactly determined when there is no noise.

        This isolates the estimator from the identifiability problem: short arcs
        are hard because of *noise*, not because the geometry is ambiguous.
        """
        truth = EllipseParams(1.0, 2.0, 9.0, 5.0, 0.35)
        span = np.radians(span_deg)
        x, y = sample_arc(truth, 40, 0.2, 0.2 + span)
        got = fit_ellipse(x, y)
        assert got.a == pytest.approx(truth.a, rel=1e-5)
        assert got.b == pytest.approx(truth.b, rel=1e-5)
        assert got.eccentricity == pytest.approx(truth.eccentricity, abs=1e-5)

    def test_major_axis_is_max_length_through_centre(self):
        truth = EllipseParams(0.0, 0.0, 8.0, 3.0, 0.0)
        x, y = sample_arc(truth, 200, 0, 2 * np.pi)
        r = np.hypot(x, y)
        assert r.max() == pytest.approx(truth.a, rel=1e-9)
        assert truth.major_axis == pytest.approx(2 * r.max(), rel=1e-9)


class TestOrthogonalDistance:
    """The Eberly foot-point solver against brute-force minimisation."""

    @pytest.mark.parametrize("px,py", [
        (12.0, 7.0), (0.0, 0.0), (0.1, 0.0), (9.0, 0.0), (0.0, 5.0),
        (-3.0, -8.0), (1e-9, 1e-9), (100.0, 100.0), (4.999, 0.0),
    ])
    def test_matches_brute_force(self, px, py):
        ell = EllipseParams(0.0, 0.0, 9.0, 5.0, 0.0)

        def dist_at(t):
            return np.hypot(ell.a * np.cos(t) - px, ell.b * np.sin(t) - py)

        # Dense grid, then polish around the best node. A bounded search over
        # fixed quadrants would miss minima that land exactly on an interval
        # boundary, such as t = pi/2 for a point on the minor axis.
        grid = np.linspace(0, 2 * np.pi, 20001)
        k = int(np.argmin(dist_at(grid)))
        lo, hi = grid[max(k - 2, 0)], grid[min(k + 2, grid.size - 1)]
        best = minimize_scalar(dist_at, bounds=(lo, hi), method="bounded",
                               options={"xatol": 1e-14}).fun
        got = abs(ellipse_residuals(ell, np.array([px]), np.array([py]))[0])
        assert got == pytest.approx(best, abs=1e-7)

    def test_sign_convention(self):
        ell = EllipseParams(0.0, 0.0, 9.0, 5.0, 0.0)
        r = ellipse_residuals(ell, np.array([20.0, 0.0]), np.array([0.0, 0.0]))
        assert r[0] > 0, "point outside must have positive residual"
        assert r[1] < 0, "point inside must have negative residual"

    def test_near_circular_does_not_divide_by_zero(self):
        ell = EllipseParams(0.0, 0.0, 5.0, 5.0 - 1e-13, 0.0)
        r = ellipse_residuals(ell, np.array([3.0, 0.0, 7.0]), np.array([0.0, 0.0, 0.0]))
        assert np.all(np.isfinite(r))
        assert r[0] == pytest.approx(-2.0, abs=1e-6)
        assert r[2] == pytest.approx(2.0, abs=1e-6)


class TestArcCoverage:
    @pytest.mark.parametrize("n", [60, 120, 360])
    def test_complete_outline_reads_one_spacing_short(self, n):
        """Coverage is 'total minus the largest gap', which is the right rule.

        For a fragment that rule measures the preserved span exactly. For a
        *complete* outline there is no break, so the largest gap is just the
        sample spacing and coverage necessarily reads 360 - 360/n. Asserting
        that exact relationship is stronger than asserting "about 360", and it
        documents that a complete outline is not a special case.
        """
        ell = EllipseParams(0.0, 0.0, 5.0, 5.0, 0.0)
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x, y = 5.0 * np.cos(t), 5.0 * np.sin(t)
        cov = arc_coverage(ell, x, y)
        assert cov["coverage_deg"] == pytest.approx(360.0 - 360.0 / n, abs=0.01)
        assert cov["coverage_perimeter_frac"] == pytest.approx(1.0 - 1.0 / n, abs=1e-3)

    def test_half_circle(self):
        ell = EllipseParams(0.0, 0.0, 5.0, 5.0, 0.0)
        x, y = sample_arc(ell, 50, 0, np.pi)
        cov = arc_coverage(ell, x, y)
        assert cov["coverage_deg"] == pytest.approx(180.0, abs=1.0)
        assert cov["coverage_perimeter_frac"] == pytest.approx(0.5, abs=0.02)

    def test_perimeter_fraction_differs_from_angle_when_elongated(self):
        """On an elongated ellipse equal angles cover unequal arc length.

        The arc near the ends of the major axis is much longer than the same
        angular span near the flanks, so the two measures must disagree -- that
        disagreement is precisely why both are reported.
        """
        ell = EllipseParams(0.0, 0.0, 20.0, 4.0, 0.0)
        x, y = sample_arc(ell, 40, -np.pi / 4, np.pi / 4)
        cov = arc_coverage(ell, x, y)
        assert cov["coverage_deg"] == pytest.approx(90.0, abs=1.0)
        assert cov["coverage_perimeter_frac"] < 0.25

    def test_perimeter_matches_quadrature(self):
        ell = EllipseParams(0.0, 0.0, 9.0, 5.0, 0.3)
        t = np.linspace(0, 2 * np.pi, 200001)
        speed = np.hypot(-ell.a * np.sin(t), ell.b * np.cos(t))
        assert ell.perimeter == pytest.approx(np.trapezoid(speed, t), rel=1e-8)


class TestCircleFit:
    def test_exact_recovery(self):
        truth = CircleParams(3.0, -4.0, 7.0)
        t = np.linspace(0.5, 2.5, 30)
        x = truth.cx + truth.r * np.cos(t)
        y = truth.cy + truth.r * np.sin(t)
        got = fit_circle(x, y)
        assert got.cx == pytest.approx(truth.cx, abs=1e-8)
        assert got.cy == pytest.approx(truth.cy, abs=1e-8)
        assert got.r == pytest.approx(truth.r, rel=1e-9)


class TestBatching:
    def test_batch_matches_single(self):
        truth = EllipseParams(1.0, 1.0, 8.0, 5.0, 0.2)
        rng = np.random.default_rng(0)
        xs, ys, seeds = [], [], []
        for i in range(12):
            x, y = sample_arc(truth, 30, 0.0, np.pi, seed=i, noise=0.05)
            xs.append(x)
            ys.append(y)
            seeds.append(fit_ellipse_algebraic(x, y).as_array())
        batched, _ = fit_ellipse_batch(np.array(xs), np.array(ys), np.array(seeds))
        for i in range(12):
            single = fit_ellipse(xs[i], ys[i])
            assert batched[i, 2] == pytest.approx(single.a, rel=1e-6)
            assert batched[i, 3] == pytest.approx(single.b, rel=1e-6)

    def test_geometric_beats_algebraic_on_noisy_arc(self):
        """The reason the algebraic fit is only ever used as a seed.

        Averaged over many noise realisations the algebraic fit is measurably
        worse on a short arc; that bias is what would otherwise leak into the
        eccentricity estimate.
        """
        truth = EllipseParams(0.0, 0.0, 10.0, 6.0, 0.0)
        alg_err, geo_err = [], []
        for seed in range(60):
            x, y = sample_arc(truth, 30, 0.0, np.radians(140), seed=seed, noise=0.08)
            try:
                alg = fit_ellipse_algebraic(x, y)
                geo = fit_ellipse(x, y, init=alg)
            except FitError:
                continue
            alg_err.append(abs(alg.a - truth.a))
            geo_err.append(abs(geo.a - truth.a))
        assert np.median(geo_err) < np.median(alg_err)

    def test_too_few_points_raises(self):
        with pytest.raises(FitError):
            fit_ellipse(np.arange(4.0), np.arange(4.0))
