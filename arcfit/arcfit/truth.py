"""Validating the photogrammetric measurements against the calipers.

With every object measured directly, accuracy can be *measured* rather than
argued for -- which is a much stronger position than a simulation study alone.

Only comparable quantities are compared. The fitted minor axis is checked
against the preserved width, because both describe the same surviving dimension.
The fitted major axis is not checked against the fragment's longest dimension,
because those are different things: the original object extended past the break.
That measurement is still useful, though, as a one-sided consistency check -- a
reconstruction shorter than a piece of the object it reconstructs is wrong.

Systematic bias measured on the width transfers to the major axis. The dominant
biases here -- calibration scale error, and the parallax from the outline
sitting above the card's plane -- are both *isotropic*: they scale the whole
outline uniformly. So one verifiable dimension calibrates both.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sps

__all__ = [
    "bland_altman",
    "deming_regression",
    "qc_reconstruction",
    "validate_against_truth",
    "AgreementResult",
]


@dataclass
class AgreementResult:
    """Bland-Altman agreement between two ways of measuring the same thing."""

    n: int
    bias: float
    bias_lo: float
    bias_hi: float
    sd_diff: float
    loa_lower: float
    loa_upper: float
    loa_lower_lo: float
    loa_lower_hi: float
    loa_upper_lo: float
    loa_upper_hi: float
    bias_pct: float
    proportional_slope: float
    proportional_p: float

    def to_dict(self) -> dict:
        return asdict(self)


def bland_altman(method: np.ndarray, reference: np.ndarray,
                 alpha: float = 0.05) -> AgreementResult:
    """Agreement between a method and a reference (Bland & Altman 1986).

    Also tests for *proportional* bias by regressing the difference on the mean.
    A constant offset and an offset that grows with size have different causes --
    the first suggests a fixed measurement convention difference, the second a
    scale error -- and only the regression distinguishes them.
    """
    m = np.asarray(method, float)
    r = np.asarray(reference, float)
    ok = np.isfinite(m) & np.isfinite(r)
    m, r = m[ok], r[ok]
    n = len(m)
    if n < 3:
        raise ValueError(f"need >= 3 paired observations, got {n}")

    diff = m - r
    avg = 0.5 * (m + r)
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1))

    tcrit = float(sps.t.ppf(1 - alpha / 2, n - 1))
    se_bias = sd / np.sqrt(n)
    loa_lo, loa_hi = bias - 1.96 * sd, bias + 1.96 * sd
    # Bland & Altman's standard error for the limits themselves.
    se_loa = sd * np.sqrt(1.0 / n + (1.96 ** 2) / (2.0 * (n - 1)))

    if np.ptp(avg) > 0:
        lr = sps.linregress(avg, diff)
        slope, pval = float(lr.slope), float(lr.pvalue)
    else:
        slope, pval = float("nan"), float("nan")

    denom = float(np.mean(r))
    return AgreementResult(
        n=n, bias=bias, bias_lo=bias - tcrit * se_bias, bias_hi=bias + tcrit * se_bias,
        sd_diff=sd, loa_lower=loa_lo, loa_upper=loa_hi,
        loa_lower_lo=loa_lo - tcrit * se_loa, loa_lower_hi=loa_lo + tcrit * se_loa,
        loa_upper_lo=loa_hi - tcrit * se_loa, loa_upper_hi=loa_hi + tcrit * se_loa,
        bias_pct=float(100.0 * bias / denom) if denom else float("nan"),
        proportional_slope=slope, proportional_p=pval,
    )


def deming_regression(x: np.ndarray, y: np.ndarray, lambda_ratio: float = 1.0,
                      n_boot: int = 2000, seed: int = 0,
                      alpha: float = 0.05) -> Dict[str, float]:
    """Errors-in-variables regression of y on x.

    Ordinary least squares assumes the predictor is measured without error. Here
    both the caliper reading and the photogrammetric estimate carry error, and
    ignoring that biases the slope towards zero -- which would show up as a
    spurious scale error. Deming regression accounts for error in both.

    ``lambda_ratio`` is var(y-error) / var(x-error); 1.0 assumes comparable
    precision. Intervals are bootstrapped, since the closed forms rely on
    normality assumptions that need not hold across a heterogeneous assemblage.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3:
        raise ValueError(f"need >= 3 paired observations, got {n}")

    def fit(xx, yy):
        mx, my = xx.mean(), yy.mean()
        sxx = float(np.sum((xx - mx) ** 2)) / (len(xx) - 1)
        syy = float(np.sum((yy - my) ** 2)) / (len(yy) - 1)
        sxy = float(np.sum((xx - mx) * (yy - my))) / (len(xx) - 1)
        if abs(sxy) < 1e-15:
            return float("nan"), float("nan")
        term = syy - lambda_ratio * sxx
        slope = (term + np.sqrt(term ** 2 + 4.0 * lambda_ratio * sxy ** 2)) / (2.0 * sxy)
        return float(slope), float(my - slope * mx)

    slope, intercept = fit(x, y)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.array([fit(x[i], y[i]) for i in idx], float)
    boots = boots[np.isfinite(boots).all(axis=1)]

    def ci(col):
        if boots.shape[0] < 20:
            return (float("nan"), float("nan"))
        return (float(np.percentile(boots[:, col], 100 * alpha / 2)),
                float(np.percentile(boots[:, col], 100 * (1 - alpha / 2))))

    slope_lo, slope_hi = ci(0)
    int_lo, int_hi = ci(1)
    return {
        "n": n, "slope": slope, "slope_lo": slope_lo, "slope_hi": slope_hi,
        "intercept": intercept, "intercept_lo": int_lo, "intercept_hi": int_hi,
        "slope_includes_1": bool(np.isfinite(slope_lo) and slope_lo <= 1.0 <= slope_hi),
        "intercept_includes_0": bool(np.isfinite(int_lo) and int_lo <= 0.0 <= int_hi),
        "lambda_ratio": lambda_ratio,
    }


def qc_reconstruction(major_axis_cm: np.ndarray, fragment_max_cm: np.ndarray,
                      tolerance_cm: float = 0.5) -> np.ndarray:
    """Flag reconstructions shorter than the surviving fragment.

    The reconstructed whole cannot be smaller than a piece of it. Any object
    failing this has a real error somewhere -- a mis-scaled calibration, points
    digitised on the wrong feature, a runaway fit -- and it is caught without
    needing any extra measurement, using the fragment maximum that is otherwise
    not comparable to anything.
    """
    major = np.asarray(major_axis_cm, float)
    frag = np.asarray(fragment_max_cm, float)
    both = np.isfinite(major) & np.isfinite(frag)
    return both & (major < frag - tolerance_cm)


def validate_against_truth(results, measurements, alpha: float = 0.05,
                           seed: int = 0, n_boot: int = 2000) -> Dict[str, object]:
    """Full accuracy assessment of the fitted results against caliper data.

    Returns the merged table plus agreement statistics, overall and split by
    reliability tier and by arc coverage -- which is what shows whether error is
    driven by preservation, by obliquity, or by neither.
    """
    import pandas as pd

    res = results if hasattr(results, "columns") else pd.DataFrame(results)
    meas = measurements if hasattr(measurements, "columns") else pd.DataFrame(measurements)

    merged = res.merge(meas, on="object_id", how="outer", indicator=True,
                       suffixes=("", "_meas"))
    out: Dict[str, object] = {
        "merged": merged,
        "n_results_only": int((merged["_merge"] == "left_only").sum()),
        "n_measurements_only": int((merged["_merge"] == "right_only").sum()),
        "n_matched": int((merged["_merge"] == "both").sum()),
    }

    paired = merged[(merged["_merge"] == "both")].copy()
    if "minor_axis_cm" in paired and "width_across_cm" in paired:
        # Only widths taken at the widest point are comparable to the minor axis.
        usable = paired[paired["width_across_cm"].notna()
                        & paired["minor_axis_cm"].notna()]
        if "width_is_at_widest" in usable.columns:
            flagged = usable["width_is_at_widest"]
            at_widest = usable[flagged.isna() | (flagged != 0)]
        else:
            at_widest = usable
        out["n_width_comparable"] = int(len(at_widest))

        if len(at_widest) >= 3:
            m = at_widest["minor_axis_cm"].to_numpy(float)
            r = at_widest["width_across_cm"].to_numpy(float)
            out["agreement_width"] = bland_altman(m, r, alpha=alpha)
            out["deming_width"] = deming_regression(r, m, seed=seed, n_boot=n_boot,
                                                    alpha=alpha)

            # A width taken off the widest point reads low in a way that tracks
            # arc coverage; a genuine scale error does not. Reporting the
            # correlation lets the reader tell the two apart.
            if "coverage_deg" in at_widest:
                cov = at_widest["coverage_deg"].to_numpy(float)
                d = m - r
                ok = np.isfinite(cov) & np.isfinite(d)
                if ok.sum() >= 4 and np.ptp(cov[ok]) > 0:
                    lr = sps.linregress(cov[ok], d[ok])
                    out["diff_vs_coverage_slope"] = float(lr.slope)
                    out["diff_vs_coverage_p"] = float(lr.pvalue)

            by_tier = {}
            for tier, grp in at_widest.groupby("tier", dropna=True):
                if len(grp) >= 3:
                    by_tier[str(tier)] = bland_altman(
                        grp["minor_axis_cm"].to_numpy(float),
                        grp["width_across_cm"].to_numpy(float), alpha=alpha)
            out["agreement_by_tier"] = by_tier

    if "major_axis_cm" in paired and "fragment_max_cm" in paired:
        bad = qc_reconstruction(paired["major_axis_cm"].to_numpy(float),
                                paired["fragment_max_cm"].to_numpy(float))
        out["qc_shorter_than_fragment"] = paired.loc[bad, "object_id"].tolist()
    return out
