"""Statistics: interval behaviour, and the circular-versus-oval decision.

The decisive tests here are about *calibration* -- that the test rejects a true
circle about as often as it claims to, and that the intervals contain the truth
about as often as they claim to. A method that produced confident wrong answers
would pass a smoke test and fail these.
"""

import numpy as np
import pytest

from arcfit.fitting import EllipseParams
from arcfit.simulate import arc_points
from arcfit.stats import (
    DEFAULT_TIERS, analyse_points, assign_tier, bca_interval, classify_shape,
)


def sample(e_true, coverage_deg, sigma=0.10, n=30, a=10.0, seed=0, start=None):
    b = a * np.sqrt(max(1e-12, 1 - e_true ** 2))
    truth = EllipseParams(0.0, 0.0, a, b, 0.0)
    rng = np.random.default_rng(seed)
    s = rng.uniform(0, 360) if start is None else start
    pts = arc_points(truth, n, coverage_deg, s) + rng.normal(0, sigma, (n, 2))
    return pts, truth


class TestNaiveEccentricityIsMisleading:
    """The finding the whole method is built around.

    A perfect circle, digitised with ordinary noise on a partial arc, produces a
    substantially non-zero fitted eccentricity. Any rule of the form
    "e > threshold means oval" is therefore reading the noise.
    """

    @pytest.mark.parametrize("coverage,floor", [(100, 0.25), (140, 0.15)])
    def test_true_circles_fit_as_clearly_elliptical(self, coverage, floor):
        ecc = []
        for seed in range(40):
            pts, _ = sample(0.0, coverage, seed=seed)
            fit = analyse_points(pts[:, 0], pts[:, 1], n_boot=0, seed=seed)
            if fit.fit_ok:
                ecc.append(fit.eccentricity)
        assert np.median(ecc) > floor, (
            "a true circle should still fit as visibly elliptical on a short arc; "
            "if this fails the simulation no longer reproduces the problem the "
            "bootstrap test exists to solve"
        )

    def test_inflation_grows_as_the_arc_shortens(self):
        med = {}
        for coverage in (90, 140, 200, 300):
            vals = []
            for seed in range(30):
                pts, _ = sample(0.0, coverage, seed=seed)
                fit = analyse_points(pts[:, 0], pts[:, 1], n_boot=0, seed=seed)
                if fit.fit_ok:
                    vals.append(fit.eccentricity)
            med[coverage] = np.median(vals)
        assert med[90] > med[140] > med[200] > med[300]


class TestCircularityTestCalibration:
    @pytest.mark.slow
    def test_size_under_a_true_circle(self):
        """Rejection rate on true circles must sit near the nominal 5%."""
        p = []
        for seed in range(80):
            pts, _ = sample(0.0, 150.0, seed=seed)
            fit = analyse_points(pts[:, 0], pts[:, 1], n_boot=300, seed=seed)
            if np.isfinite(fit.p_bootstrap):
                p.append(fit.p_bootstrap)
        rate = float(np.mean(np.asarray(p) < 0.05))
        # 3 standard errors around 0.05 at n=80.
        assert rate < 0.12, f"test is anti-conservative: rejects {rate:.1%} of true circles"

    @pytest.mark.slow
    def test_has_power_against_a_strongly_oval_object(self):
        p = []
        for seed in range(40):
            pts, _ = sample(0.8, 200.0, sigma=0.06, seed=500 + seed)
            fit = analyse_points(pts[:, 0], pts[:, 1], n_boot=300, seed=seed)
            if np.isfinite(fit.p_bootstrap):
                p.append(fit.p_bootstrap)
        assert np.mean(np.asarray(p) < 0.05) > 0.5

    def test_null_median_is_reported_for_transparency(self):
        pts, _ = sample(0.0, 140.0, seed=3)
        fit = analyse_points(pts[:, 0], pts[:, 1], n_boot=200, seed=3)
        assert np.isfinite(fit.null_e_median)
        assert fit.null_e_median > 0, "the null itself must show the inflation"


class TestIntervals:
    @pytest.mark.slow
    def test_major_axis_interval_covers_truth_about_95_percent(self):
        hits = 0
        total = 0
        for seed in range(60):
            pts, truth = sample(0.5, 200.0, sigma=0.08, seed=seed)
            fit = analyse_points(pts[:, 0], pts[:, 1], n_boot=400, seed=seed)
            if not fit.fit_ok or not np.isfinite(fit.major_axis_lo):
                continue
            total += 1
            hits += fit.major_axis_lo <= truth.major_axis <= fit.major_axis_hi
        assert total > 40
        assert 0.80 <= hits / total <= 1.0, f"interval coverage {hits / total:.2f}"

    def test_bca_falls_back_to_percentile_when_degenerate(self):
        boot = np.full(200, 5.0)
        lo, hi = bca_interval(boot, 5.0, np.full(30, 5.0))
        assert lo == pytest.approx(5.0) and hi == pytest.approx(5.0)

    def test_bca_returns_nan_with_too_few_replicates(self):
        lo, hi = bca_interval(np.arange(5.0), 2.0, np.arange(5.0))
        assert np.isnan(lo) and np.isnan(hi)


class TestClassification:
    def test_rejecting_a_circle_gives_elliptical(self):
        assert classify_shape(0.001, 0.9) == "elliptical"

    def test_tight_interval_near_zero_gives_circular(self):
        assert classify_shape(0.7, 0.2) == "circular"

    def test_wide_interval_gives_indeterminate_not_circular(self):
        """Failing to reject is not evidence of circularity when underpowered."""
        assert classify_shape(0.7, 0.85) == "indeterminate"

    def test_missing_p_value_is_indeterminate(self):
        assert classify_shape(float("nan"), 0.1) == "indeterminate"


class TestDiagnostics:
    def test_extant_chord_bounds_the_reconstruction(self):
        pts, truth = sample(0.4, 160.0, sigma=0.03, seed=2)
        fit = analyse_points(pts[:, 0], pts[:, 1], n_boot=0, seed=2)
        assert fit.extant_chord_cm <= fit.major_axis_cm * 1.02
        assert fit.extrapolation_factor >= 0.98

    def test_short_arc_has_larger_extrapolation_factor(self):
        long_arc = analyse_points(*sample(0.3, 300.0, seed=5)[0].T, n_boot=0, seed=5)
        short_arc = analyse_points(*sample(0.3, 100.0, seed=5)[0].T, n_boot=0, seed=5)
        assert short_arc.extrapolation_factor > long_arc.extrapolation_factor

    def test_too_few_points_fails_cleanly(self):
        fit = analyse_points(np.arange(4.0), np.arange(4.0), n_boot=10)
        assert not fit.fit_ok and "6 points" in fit.error


class TestTiering:
    def test_good_object_reaches_tier_a(self):
        assert assign_tier(180, 0.10, 0.5, True, True, DEFAULT_TIERS,
                           extrapolation=1.05, size_ci_frac=0.10) == "A"

    def test_runaway_extrapolation_is_demoted(self):
        """Coverage alone cannot catch a runaway fit, because it is measured
        against the runaway ellipse."""
        assert assign_tier(180, 0.10, 0.5, True, True, DEFAULT_TIERS,
                           extrapolation=2.5, size_ci_frac=0.10) == "C"

    def test_useless_interval_is_demoted(self):
        assert assign_tier(180, 0.10, 0.5, True, True, DEFAULT_TIERS,
                           extrapolation=1.05, size_ci_frac=3.0) == "C"

    def test_unrectified_cannot_reach_tier_a(self):
        assert assign_tier(180, 0.10, 0.5, True, False, DEFAULT_TIERS,
                           extrapolation=1.05, size_ci_frac=0.10) == "B"

    def test_failed_fit_is_tier_c(self):
        assert assign_tier(180, 0.01, 0.1, False, True) == "C"
