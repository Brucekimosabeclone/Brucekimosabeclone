"""One-hand versus two-hand manos: is the size distribution actually two groups?

The typological question is about size, so the reconstructed major axis is the
variable of interest. But "are there two size classes?" is a claim about the
distribution, and eyeballing a histogram for two humps is not evidence -- a
single skewed distribution produces convincing-looking humps in samples of this
size all the time.

Two independent tests are therefore applied:

* A **parametric bootstrap likelihood-ratio test** comparing a one-component
  against a two-component Gaussian mixture. The bootstrap is not optional here:
  the usual chi-square reference distribution for a likelihood-ratio test is
  invalid for mixtures, because the null sits on the boundary of the parameter
  space and the extra component's parameters are unidentifiable under it.
  Quoting a chi-square p-value would substantially overstate the evidence.

* **Silverman's bandwidth test**, which asks the same question without assuming
  the groups are Gaussian at all. Agreement between a parametric and a
  distribution-free test is much more persuasive than either alone.

Objects are then assigned to groups with their *own* measurement uncertainty
propagated, so a fragment whose reconstruction is uncertain is reported as
uncertain rather than being pushed to whichever side of the boundary its point
estimate happens to land on.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sps
from scipy.optimize import brentq

__all__ = [
    "GaussianMixture",
    "fit_mixture",
    "bootstrap_lrt",
    "silverman_test",
    "count_kde_modes",
    "critical_bandwidth",
    "find_antimode",
    "antimode_interval",
    "assign_groups",
    "TypologyResult",
]

_VAR_FLOOR = 1e-6


@dataclass
class GaussianMixture:
    """A fitted 1-D Gaussian mixture."""

    weights: np.ndarray
    means: np.ndarray
    sds: np.ndarray
    loglik: float
    n_iter: int
    converged: bool

    @property
    def k(self) -> int:
        return len(self.means)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        return sum(w * sps.norm.pdf(x, m, s)
                   for w, m, s in zip(self.weights, self.means, self.sds))

    def responsibilities(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)[:, None]
        comp = self.weights[None, :] * sps.norm.pdf(x, self.means[None, :], self.sds[None, :])
        total = comp.sum(axis=1, keepdims=True)
        return comp / np.where(total > 0, total, 1.0)

    def ordered(self) -> "GaussianMixture":
        """Components sorted by mean, so 'component 0' is always the smaller."""
        o = np.argsort(self.means)
        return GaussianMixture(self.weights[o], self.means[o], self.sds[o],
                               self.loglik, self.n_iter, self.converged)


def _loglik(x: np.ndarray, w, m, s) -> float:
    comp = w[None, :] * sps.norm.pdf(x[:, None], m[None, :], s[None, :])
    total = comp.sum(axis=1)
    return float(np.sum(np.log(np.maximum(total, 1e-300))))


def fit_mixture(x: np.ndarray, k: int = 2, n_init: int = 10, max_iter: int = 500,
                tol: float = 1e-8, seed: int = 0) -> GaussianMixture:
    """Fit a k-component 1-D Gaussian mixture by EM.

    Restarted from several initialisations because EM finds local optima, and a
    mixture likelihood on real data reliably has several. Variances are floored
    to stop a component collapsing onto a single point, which sends the
    likelihood to infinity and is not a solution to anything.
    """
    x = np.asarray(x, float).ravel()
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2 * k:
        raise ValueError(f"need >= {2 * k} observations to fit {k} components, got {n}")

    if k == 1:
        m = np.array([x.mean()])
        s = np.array([max(x.std(ddof=1), np.sqrt(_VAR_FLOOR))])
        w = np.array([1.0])
        return GaussianMixture(w, m, s, _loglik(x, w, m, s), 0, True)

    rng = np.random.default_rng(seed)
    var_floor = max(_VAR_FLOOR, (np.ptp(x) / (100.0 * k)) ** 2)
    best: Optional[GaussianMixture] = None

    for attempt in range(n_init):
        if attempt == 0:
            m = np.quantile(x, np.linspace(0.25, 0.75, k))
        else:
            m = rng.choice(x, size=k, replace=False)
        m = np.sort(m).astype(float)
        s = np.full(k, max(x.std(ddof=1), np.sqrt(var_floor)))
        w = np.full(k, 1.0 / k)

        prev = -np.inf
        converged = False
        it = 0
        for it in range(1, max_iter + 1):
            comp = w[None, :] * sps.norm.pdf(x[:, None], m[None, :], s[None, :])
            total = comp.sum(axis=1, keepdims=True)
            resp = comp / np.where(total > 0, total, 1.0)

            nk = resp.sum(axis=0)
            if np.any(nk < 1e-8):
                break
            w = nk / n
            m = (resp * x[:, None]).sum(axis=0) / nk
            var = (resp * (x[:, None] - m[None, :]) ** 2).sum(axis=0) / nk
            s = np.sqrt(np.maximum(var, var_floor))

            ll = _loglik(x, w, m, s)
            if np.isfinite(ll) and abs(ll - prev) < tol * max(1.0, abs(prev)):
                prev = ll
                converged = True
                break
            prev = ll

        if np.isfinite(prev) and (best is None or prev > best.loglik):
            best = GaussianMixture(w.copy(), m.copy(), s.copy(), prev, it, converged)

    if best is None:
        raise ValueError("mixture fitting failed from every initialisation")
    return best.ordered()


def bootstrap_lrt(x: np.ndarray, n_boot: int = 500, seed: int = 0,
                  n_init: int = 6) -> Dict[str, float]:
    """Parametric bootstrap likelihood-ratio test for one versus two components.

    Simulates from the fitted single Gaussian, refits both models to each
    simulated sample, and places the observed likelihood ratio in that null
    distribution -- sidestepping the invalid chi-square reference entirely.
    """
    x = np.asarray(x, float).ravel()
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return {"lr": float("nan"), "p_value": float("nan"), "n_boot": 0, "n": n}

    m1 = fit_mixture(x, k=1, seed=seed)
    m2 = fit_mixture(x, k=2, n_init=n_init, seed=seed)
    lr_obs = 2.0 * (m2.loglik - m1.loglik)

    rng = np.random.default_rng(seed + 1)
    null = []
    for _ in range(n_boot):
        sim = rng.normal(m1.means[0], m1.sds[0], n)
        try:
            a = fit_mixture(sim, k=1, seed=0)
            b = fit_mixture(sim, k=2, n_init=n_init, seed=int(rng.integers(1 << 30)))
        except ValueError:
            continue
        val = 2.0 * (b.loglik - a.loglik)
        if np.isfinite(val):
            null.append(val)

    null = np.asarray(null)
    # BIC = -2*loglik + n_params*ln(n); 2 params for one component (mean, sd),
    # 5 for two (two means, two sds, one free weight). Positive favours two.
    bic1 = -2.0 * m1.loglik + 2 * np.log(n)
    bic2 = -2.0 * m2.loglik + 5 * np.log(n)

    out = {
        "lr": float(lr_obs),
        "n": n,
        "delta_bic": float(bic1 - bic2),
        "loglik_k1": float(m1.loglik),
        "loglik_k2": float(m2.loglik),
    }
    if null.size < 20:
        out.update({"p_value": float("nan"), "n_boot": int(null.size)})
        return out
    out.update({
        "p_value": float((1 + np.sum(null >= lr_obs)) / (1 + null.size)),
        "n_boot": int(null.size),
        "null_lr_p95": float(np.percentile(null, 95)),
    })
    return out


def count_kde_modes(x: np.ndarray, h: float, grid: int = 2048) -> int:
    """Number of local maxima of a Gaussian KDE with bandwidth ``h``."""
    x = np.asarray(x, float)
    lo, hi = x.min() - 4 * h, x.max() + 4 * h
    g = np.linspace(lo, hi, grid)
    dens = sps.norm.pdf((g[:, None] - x[None, :]) / h).sum(axis=1)
    return int(np.sum((dens[1:-1] > dens[:-2]) & (dens[1:-1] >= dens[2:])))


def critical_bandwidth(x: np.ndarray, k_modes: int = 1, tol: float = 1e-4) -> float:
    """Smallest bandwidth whose KDE has at most ``k_modes`` modes.

    Well defined because, for a Gaussian kernel, the mode count is
    non-increasing in the bandwidth -- so bisection is valid.
    """
    x = np.asarray(x, float)
    spread = float(np.ptp(x))
    if spread <= 0:
        return tol
    lo, hi = tol * spread, 2.0 * spread
    while count_kde_modes(x, hi) > k_modes and hi < 100 * spread:
        hi *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if count_kde_modes(x, mid) > k_modes:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * spread:
            break
    return hi


def silverman_test(x: np.ndarray, k_modes: int = 1, n_boot: int = 500,
                   seed: int = 0) -> Dict[str, float]:
    """Silverman's (1981) bandwidth test for more than ``k_modes`` modes.

    Makes no assumption that the groups are Gaussian, which is exactly what the
    mixture test does assume -- so the two together are a much stronger argument
    than either on its own. The smoothed bootstrap rescales resamples to preserve
    the sample variance, without which the test is badly conservative.
    """
    x = np.asarray(x, float).ravel()
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return {"critical_bandwidth": float("nan"), "p_value": float("nan"), "n": n}

    h_crit = critical_bandwidth(x, k_modes=k_modes)
    sd = float(x.std(ddof=1))
    if sd <= 0:
        return {"critical_bandwidth": h_crit, "p_value": float("nan"), "n": n}

    rng = np.random.default_rng(seed)
    xbar = x.mean()
    scale = 1.0 / np.sqrt(1.0 + (h_crit ** 2) / (sd ** 2))
    count = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        eps = rng.normal(0.0, 1.0, n)
        sim = xbar + scale * (x[idx] - xbar + h_crit * eps)
        if count_kde_modes(sim, h_crit) > k_modes:
            count += 1
    return {
        "critical_bandwidth": float(h_crit),
        "p_value": float((1 + count) / (1 + n_boot)),
        "n_boot": n_boot,
        "n": n,
    }


def find_antimode(mix: GaussianMixture) -> float:
    """The density minimum between two components: the group boundary."""
    mix = mix.ordered()
    if mix.k != 2:
        raise ValueError("antimode is defined for a two-component mixture")
    lo, hi = mix.means[0], mix.means[1]
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float("nan")

    grid = np.linspace(lo, hi, 1001)
    dens = mix.pdf(grid)
    j = int(np.argmin(dens))
    if j in (0, len(grid) - 1):
        return float(grid[j])

    def dprime(v):
        eps = (hi - lo) * 1e-5
        return float(mix.pdf(np.array([v + eps]))[0] - mix.pdf(np.array([v - eps]))[0])

    a, b = grid[j - 1], grid[j + 1]
    try:
        if dprime(a) * dprime(b) < 0:
            return float(brentq(dprime, a, b, xtol=1e-10))
    except (ValueError, RuntimeError):
        pass
    return float(grid[j])


def antimode_interval(x: np.ndarray, n_boot: int = 1000, seed: int = 0,
                      alpha: float = 0.05) -> Dict[str, float]:
    """Bootstrap confidence interval for the group boundary.

    A boundary quoted without an interval invites the reader to treat objects
    near it as classified, when in a sample this size the boundary itself often
    moves by more than the gap between adjacent objects.
    """
    x = np.asarray(x, float).ravel()
    x = x[np.isfinite(x)]
    n = len(x)
    try:
        point = find_antimode(fit_mixture(x, k=2, seed=seed))
    except ValueError:
        return {"antimode": float("nan"), "lo": float("nan"), "hi": float("nan")}

    rng = np.random.default_rng(seed + 7)
    vals = []
    for _ in range(n_boot):
        sim = x[rng.integers(0, n, n)]
        try:
            v = find_antimode(fit_mixture(sim, k=2, n_init=4,
                                          seed=int(rng.integers(1 << 30))))
        except ValueError:
            continue
        if np.isfinite(v):
            vals.append(v)
    vals = np.asarray(vals)
    if vals.size < 20:
        return {"antimode": point, "lo": float("nan"), "hi": float("nan"),
                "n_boot": int(vals.size)}
    return {
        "antimode": float(point),
        "lo": float(np.percentile(vals, 100 * alpha / 2)),
        "hi": float(np.percentile(vals, 100 * (1 - alpha / 2))),
        "n_boot": int(vals.size),
    }


def assign_groups(values: np.ndarray, ci_lo: np.ndarray, ci_hi: np.ndarray,
                  mix: GaussianMixture, n_draws: int = 2000,
                  seed: int = 0, confident_at: float = 0.9) -> Dict[str, np.ndarray]:
    """Assign objects to size groups, carrying each object's own uncertainty.

    Each object's major axis is resampled from a normal implied by its
    confidence interval, and the mixture posterior is averaged over those draws.
    An object whose interval straddles the boundary comes out near 0.5 and is
    labelled uncertain -- which is the honest answer, and is invisible if only
    point estimates are classified.
    """
    values = np.asarray(values, float)
    lo = np.asarray(ci_lo, float)
    hi = np.asarray(ci_hi, float)
    mix = mix.ordered()

    # Interval half-width back to a standard deviation, assuming a 95% interval.
    sd = np.where(np.isfinite(hi) & np.isfinite(lo), (hi - lo) / (2 * 1.959964), np.nan)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 0.0)

    rng = np.random.default_rng(seed)
    draws = values[:, None] + sd[:, None] * rng.standard_normal((len(values), n_draws))
    draws = np.where(np.isfinite(draws), draws, np.nan)

    flat = draws.ravel()
    ok = np.isfinite(flat)
    post = np.full(flat.shape, np.nan)
    if ok.any():
        post[ok] = mix.responsibilities(flat[ok])[:, 1]
    post = post.reshape(draws.shape)

    p_large = np.nanmean(post, axis=1)
    group = np.where(p_large >= 0.5, "two-hand", "one-hand").astype(object)
    confident = np.maximum(p_large, 1 - p_large) >= confident_at
    group[~np.isfinite(p_large)] = "unassigned"
    return {
        "p_two_hand": p_large,
        "group": group,
        "confident": confident,
        "group_reported": np.where(confident, group, "uncertain"),
    }


@dataclass
class TypologyResult:
    """Everything the typological analysis concluded."""

    n: int = 0
    lrt: Dict[str, float] = field(default_factory=dict)
    silverman: Dict[str, float] = field(default_factory=dict)
    antimode: Dict[str, float] = field(default_factory=dict)
    mixture_k1: Optional[GaussianMixture] = None
    mixture_k2: Optional[GaussianMixture] = None
    bimodal: bool = False
    verdict: str = ""

    def to_dict(self) -> dict:
        d = {"n": self.n, "bimodal": self.bimodal, "verdict": self.verdict}
        for name, sub in (("lrt", self.lrt), ("silverman", self.silverman),
                          ("antimode", self.antimode)):
            for k, v in (sub or {}).items():
                d[f"{name}_{k}"] = v
        for name, m in (("k1", self.mixture_k1), ("k2", self.mixture_k2)):
            if m is not None:
                d[f"{name}_loglik"] = m.loglik
                for i in range(m.k):
                    d[f"{name}_w{i}"] = float(m.weights[i])
                    d[f"{name}_mean{i}"] = float(m.means[i])
                    d[f"{name}_sd{i}"] = float(m.sds[i])
        return d


def analyse_typology(major_axis_cm: np.ndarray, n_boot: int = 500,
                     seed: int = 0, alpha: float = 0.05) -> TypologyResult:
    """Run both bimodality tests and locate the group boundary."""
    x = np.asarray(major_axis_cm, float).ravel()
    x = x[np.isfinite(x)]
    res = TypologyResult(n=len(x))
    if len(x) < 10:
        res.verdict = f"too few objects ({len(x)}) to assess bimodality"
        return res

    res.mixture_k1 = fit_mixture(x, k=1, seed=seed)
    res.mixture_k2 = fit_mixture(x, k=2, seed=seed)
    res.lrt = bootstrap_lrt(x, n_boot=n_boot, seed=seed)
    res.silverman = silverman_test(x, k_modes=1, n_boot=n_boot, seed=seed)
    res.antimode = antimode_interval(x, n_boot=n_boot, seed=seed, alpha=alpha)

    p_lrt = res.lrt.get("p_value", np.nan)
    p_sil = res.silverman.get("p_value", np.nan)
    both = [p for p in (p_lrt, p_sil) if np.isfinite(p)]
    res.bimodal = bool(both) and all(p < alpha for p in both)

    if not both:
        res.verdict = "bimodality could not be assessed"
    elif res.bimodal:
        res.verdict = (f"two size groups supported (mixture LRT p={p_lrt:.3f}, "
                       f"Silverman p={p_sil:.3f})")
    elif any(p < alpha for p in both):
        res.verdict = (f"mixed evidence for two size groups (mixture LRT p={p_lrt:.3f}, "
                       f"Silverman p={p_sil:.3f}); treat any boundary as provisional")
    else:
        res.verdict = (f"no support for two size groups (mixture LRT p={p_lrt:.3f}, "
                       f"Silverman p={p_sil:.3f}); a single distribution is adequate")
    return res


__all__.append("analyse_typology")
