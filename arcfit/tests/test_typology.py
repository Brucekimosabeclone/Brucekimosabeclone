"""Bimodality testing: it must find two groups when they exist and not otherwise."""

import numpy as np
import pytest

from arcfit.typology import (
    analyse_typology, antimode_interval, assign_groups, bootstrap_lrt,
    count_kde_modes, critical_bandwidth, find_antimode, fit_mixture, silverman_test,
)


def two_groups(seed=0, n1=60, n2=70):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(12.0, 1.3, n1), rng.normal(22.0, 2.0, n2)])


def one_group(seed=0, n=130):
    return np.random.default_rng(seed).normal(17.0, 3.5, n)


class TestMixture:
    def test_recovers_known_components(self):
        m = fit_mixture(two_groups(1), k=2, seed=0).ordered()
        assert m.means[0] == pytest.approx(12.0, abs=0.8)
        assert m.means[1] == pytest.approx(22.0, abs=1.2)

    def test_components_are_ordered_by_mean(self):
        m = fit_mixture(two_groups(2), k=2, seed=0)
        assert m.means[0] < m.means[1]

    def test_single_component_is_the_sample_moments(self):
        x = one_group(3)
        m = fit_mixture(x, k=1)
        assert m.means[0] == pytest.approx(x.mean())
        assert m.sds[0] == pytest.approx(x.std(ddof=1))

    def test_refuses_too_few_observations(self):
        with pytest.raises(ValueError):
            fit_mixture(np.arange(3.0), k=2)

    def test_variance_floor_prevents_collapse(self):
        """A component collapsing onto one point sends the likelihood to
        infinity, which is a numerical artefact and not a fit."""
        x = np.concatenate([np.full(20, 5.0), np.random.default_rng(0).normal(15, 2, 40)])
        m = fit_mixture(x, k=2, seed=1)
        assert np.all(m.sds > 0) and np.isfinite(m.loglik)


class TestBimodality:
    @pytest.mark.slow
    def test_detects_two_real_groups(self):
        r = analyse_typology(two_groups(4), n_boot=200, seed=1)
        assert r.bimodal
        assert r.lrt["p_value"] < 0.05
        assert r.lrt["delta_bic"] > 0

    @pytest.mark.slow
    def test_does_not_invent_groups_in_unimodal_data(self):
        r = analyse_typology(one_group(5), n_boot=200, seed=1)
        assert not r.bimodal
        assert r.lrt["p_value"] > 0.05
        assert r.lrt["delta_bic"] < 0

    def test_too_few_objects_declines_to_answer(self):
        r = analyse_typology(np.arange(5.0), n_boot=50)
        assert "too few" in r.verdict


class TestSilverman:
    def test_mode_counting(self):
        x = two_groups(6)
        assert count_kde_modes(x, 0.2) >= 2
        assert count_kde_modes(x, 50.0) == 1

    def test_critical_bandwidth_is_the_boundary(self):
        x = two_groups(7)
        h = critical_bandwidth(x, k_modes=1)
        assert count_kde_modes(x, h * 1.05) <= 1
        assert count_kde_modes(x, h * 0.80) > 1

    def test_finds_bimodality(self):
        assert silverman_test(two_groups(8), n_boot=150, seed=0)["p_value"] < 0.10


class TestAntimode:
    def test_lies_between_the_two_means(self):
        m = fit_mixture(two_groups(9), k=2, seed=0).ordered()
        a = find_antimode(m)
        assert m.means[0] < a < m.means[1]

    def test_interval_brackets_the_estimate(self):
        r = antimode_interval(two_groups(10), n_boot=200, seed=0)
        assert r["lo"] <= r["antimode"] <= r["hi"]

    def test_requires_two_components(self):
        with pytest.raises(ValueError):
            find_antimode(fit_mixture(one_group(0), k=1))


class TestAssignment:
    def test_uncertain_objects_are_not_forced_into_a_group(self):
        """An object whose interval straddles the boundary must be reported as
        uncertain; classifying point estimates alone hides exactly this case."""
        x = two_groups(11)
        m = fit_mixture(x, k=2, seed=0).ordered()
        boundary = find_antimode(m)
        values = np.array([12.0, 22.0, boundary])
        # tight, tight, very wide
        lo = np.array([11.8, 21.7, boundary - 6.0])
        hi = np.array([12.2, 22.3, boundary + 6.0])
        out = assign_groups(values, lo, hi, m, seed=0)
        assert out["group_reported"][0] == "one-hand"
        assert out["group_reported"][1] == "two-hand"
        assert out["group_reported"][2] == "uncertain"

    def test_probabilities_are_in_range(self):
        x = two_groups(12)
        m = fit_mixture(x, k=2, seed=0)
        out = assign_groups(x, x - 0.5, x + 0.5, m, seed=0)
        assert np.all((out["p_two_hand"] >= 0) & (out["p_two_hand"] <= 1))
