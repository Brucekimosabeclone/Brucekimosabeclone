"""Validation against caliper measurements."""

import numpy as np
import pytest

from arcfit.truth import bland_altman, deming_regression, qc_reconstruction


class TestBlandAltman:
    def test_detects_a_known_constant_bias(self):
        rng = np.random.default_rng(0)
        ref = rng.uniform(8, 25, 80)
        method = ref + 0.30 + rng.normal(0, 0.05, 80)
        ag = bland_altman(method, ref)
        assert ag.bias == pytest.approx(0.30, abs=0.03)
        assert ag.bias_lo < 0.30 < ag.bias_hi
        assert ag.loa_lower < ag.bias < ag.loa_upper

    def test_reports_no_bias_when_there_is_none(self):
        rng = np.random.default_rng(1)
        ref = rng.uniform(8, 25, 120)
        method = ref + rng.normal(0, 0.08, 120)
        ag = bland_altman(method, ref)
        assert ag.bias_lo <= 0.0 <= ag.bias_hi

    def test_detects_proportional_bias(self):
        """A scale error and a constant offset have different causes; the
        regression of difference on mean is what separates them."""
        rng = np.random.default_rng(2)
        ref = rng.uniform(8, 25, 120)
        method = ref * 1.05 + rng.normal(0, 0.05, 120)
        ag = bland_altman(method, ref)
        assert ag.proportional_p < 0.01
        assert ag.proportional_slope == pytest.approx(0.05, abs=0.02)

    def test_ignores_missing_pairs(self):
        m = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        r = np.array([1.1, 2.1, 3.1, np.nan, 5.1])
        assert bland_altman(m, r).n == 3

    def test_requires_enough_pairs(self):
        with pytest.raises(ValueError):
            bland_altman(np.array([1.0]), np.array([1.0]))


class TestDeming:
    def test_recovers_slope_of_one_when_methods_agree(self):
        rng = np.random.default_rng(3)
        true = rng.uniform(8, 25, 150)
        x = true + rng.normal(0, 0.1, 150)
        y = true + rng.normal(0, 0.1, 150)
        d = deming_regression(x, y, n_boot=300, seed=0)
        assert d["slope"] == pytest.approx(1.0, abs=0.08)
        assert d["slope_includes_1"]

    def test_less_biased_than_ols_when_x_has_error(self):
        """Ordinary least squares assumes an error-free predictor; when both
        carry error it attenuates the slope towards zero, which would read as a
        spurious scale error."""
        rng = np.random.default_rng(4)
        true = rng.uniform(8, 25, 400)
        x = true + rng.normal(0, 0.8, 400)
        y = true + rng.normal(0, 0.8, 400)
        ols = np.polyfit(x, y, 1)[0]
        dem = deming_regression(x, y, n_boot=200, seed=0)["slope"]
        assert abs(dem - 1.0) < abs(ols - 1.0)

    def test_detects_a_real_scale_error(self):
        rng = np.random.default_rng(5)
        true = rng.uniform(8, 25, 150)
        x = true + rng.normal(0, 0.05, 150)
        y = 1.10 * true + rng.normal(0, 0.05, 150)
        d = deming_regression(x, y, n_boot=300, seed=0)
        assert not d["slope_includes_1"]
        assert d["slope"] == pytest.approx(1.10, abs=0.05)


class TestReconstructionQC:
    def test_flags_reconstruction_shorter_than_its_own_fragment(self):
        major = np.array([20.0, 10.0, 15.0])
        frag = np.array([18.0, 14.0, 15.2])
        flags = qc_reconstruction(major, frag, tolerance_cm=0.5)
        assert list(flags) == [False, True, False]

    def test_tolerance_absorbs_measurement_error(self):
        assert not qc_reconstruction(np.array([14.8]), np.array([15.0]))[0]

    def test_missing_values_are_not_flagged(self):
        assert not qc_reconstruction(np.array([np.nan]), np.array([15.0]))[0]
        assert not qc_reconstruction(np.array([15.0]), np.array([np.nan]))[0]
