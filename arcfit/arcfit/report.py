"""Results tables, the data dictionary, and the methods text.

Every column that reaches the results table is defined in the data dictionary
with its units, because a supplementary table whose columns have to be guessed
at is not usable by anyone but its author.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

__all__ = ["DATA_DICTIONARY", "SLIM_COLUMNS", "results_table", "write_tables",
           "methods_text", "summary_counts"]


# The article's table: the measurements that were actually asked for, each with
# its interval, plus the shape call and how far it can be trusted. The full
# table keeps every diagnostic alongside it -- this is a view, not a
# replacement, so nothing is discarded by having it.
SLIM_COLUMNS = [
    "object_id",
    "major_axis_cm", "major_axis_lo", "major_axis_hi",
    "minor_axis_cm", "minor_axis_lo", "minor_axis_hi",
    "eccentricity", "eccentricity_lo", "eccentricity_hi",
    "shape_class", "tier",
]


DATA_DICTIONARY: List[Dict[str, str]] = [
    ("object_id", "", "Object identifier, matching the photograph filename stem."),
    ("n_points", "count", "Number of digitised points on the surviving outline."),
    ("fit_ok", "boolean", "Whether a valid ellipse was obtained. False rows carry no measurements."),
    ("error", "", "Why the fit failed, when it did."),
    ("major_axis_cm", "cm", "Reconstructed major axis, 2a: the maximum length through the centre of the reconstructed whole object."),
    ("major_axis_lo", "cm", "Lower bound of the 95% bootstrap interval for the major axis."),
    ("major_axis_hi", "cm", "Upper bound of the 95% bootstrap interval for the major axis."),
    ("minor_axis_cm", "cm", "Reconstructed minor axis, 2b."),
    ("minor_axis_lo", "cm", "Lower bound of the 95% bootstrap interval for the minor axis."),
    ("minor_axis_hi", "cm", "Upper bound of the 95% bootstrap interval for the minor axis."),
    ("eccentricity", "", "Eccentricity of the reconstructed ellipse, sqrt(1 - (b/a)^2). 0 is a circle."),
    ("eccentricity_lo", "", "Lower bound of the 95% bootstrap interval for eccentricity."),
    ("eccentricity_hi", "", "Upper bound of the 95% bootstrap interval for eccentricity."),
    ("axis_ratio", "", "b/a. Reported alongside eccentricity because it is the more directly interpretable measure of elongation."),
    ("axis_ratio_lo", "", "Lower bound of the 95% bootstrap interval for b/a."),
    ("axis_ratio_hi", "", "Upper bound of the 95% bootstrap interval for b/a."),
    ("orientation_deg", "degrees", "Orientation of the major axis in the rectified ground plane."),
    ("coverage_deg", "degrees", "Angular span of the surviving outline about the reconstructed centre."),
    ("coverage_perimeter_frac", "fraction", "Proportion of the reconstructed perimeter that survives. The honest measure for elongated objects, where equal angles cover unequal arc length."),
    ("rms_residual_mm", "mm", "Root-mean-square perpendicular distance from the digitised points to the fitted ellipse."),
    ("max_residual_mm", "mm", "Largest perpendicular distance from any digitised point to the fitted ellipse."),
    ("extant_chord_cm", "cm", "Widest separation of the digitised points: a hard lower bound on the reconstructed major axis, requiring no extra measurement."),
    ("extrapolation_factor", "", "Reconstructed major axis divided by the extant chord. Near 1 the outline nearly spans the object; large values mean most of the reconstruction is extrapolation."),
    ("size_ci_frac", "", "Width of the major-axis confidence interval as a fraction of the estimate. A direct measure of how precisely the size is known."),
    ("circle_diameter_cm", "cm", "Diameter of the best-fitting circle, for comparison with the major axis."),
    ("p_bootstrap", "", "Headline test. Probability of an eccentricity at least this large if the object were truly circular, simulated at this object's own arc coverage and noise. Small values are evidence of a genuinely oval object."),
    ("p_f_test", "", "Nested F-test comparing ellipse against circle. Reported for completeness; anti-conservative here because digitised points along a curve are spatially autocorrelated."),
    ("delta_aicc", "", "AICc for the circle minus AICc for the ellipse. Positive favours the ellipse."),
    ("delta_bic", "", "BIC for the circle minus BIC for the ellipse. Positive favours the ellipse."),
    ("null_e_median", "", "Median eccentricity recovered from simulated true circles matching this object. Quantifies how much of the observed eccentricity is noise."),
    ("null_e_p95", "", "95th percentile of eccentricity recovered from simulated true circles matching this object."),
    ("shape_class", "", "circular / elliptical / indeterminate. 'circular' requires both failing to reject a circle and an interval excluding meaningful elongation; 'indeterminate' means the data cannot separate the two."),
    ("tier", "", "Reliability tier: A reliable, B marginal, C indeterminate. Based on arc coverage, interval width, residuals and calibration quality."),
    ("rectified", "boolean", "Whether perspective was removed via a scale-card homography. False means only a scalar scale was available and oblique distortion is NOT corrected."),
    ("tilt_deg", "degrees", "Angle between the camera axis and the ground-plane normal. 0 is a plan view."),
    ("camera_height_cm", "cm", "Camera height above the ground plane, estimated from EXIF focal length. Used only for the parallax diagnostic."),
    ("reproj_rms_cm", "cm", "Reprojection residual of the scale-card homography. A calibration quality check."),
    ("card_detection_score", "", "Confidence of automatic scale-card detection, 0-1. Blank where calibration was manual."),
    ("card_on_object", "boolean", "True if the scale card rested on the object rather than the ground. This flips the sign of the parallax term: card on the ground puts the reference plane below the traced outline and sizes read slightly large, card on the object puts it at or above and they read small."),
    ("calibration_mode", "", "homography (rectified) or two_point (scalar scale only)."),
    ("n_boot", "count", "Bootstrap replicates used for intervals and for the circularity test."),
    ("seed", "", "Random seed, recorded so every interval and p-value is exactly reproducible."),
    ("fragment_max_cm", "cm", "Caliper measurement: longest dimension of the surviving fragment. A lower bound on the original, not comparable to the major axis."),
    ("width_across_cm", "cm", "Caliper measurement: fully preserved width perpendicular to the break. Directly comparable to the minor axis."),
    ("qc_shorter_than_fragment", "boolean", "True if the reconstructed major axis is shorter than the surviving fragment, which is impossible and indicates an error."),
    ("group_reported", "", "Size-group assignment (one-hand / two-hand / uncertain), with each object's own measurement uncertainty propagated."),
    ("p_two_hand", "", "Posterior probability of belonging to the larger size group, averaged over the object's measurement uncertainty."),
]


def summary_counts(df) -> Dict[str, object]:
    """Headline counts for the run, for the console and the report."""
    out: Dict[str, object] = {"n_objects": int(len(df))}
    if "fit_ok" in df:
        out["n_fitted"] = int(df["fit_ok"].sum())
        out["n_failed"] = int((~df["fit_ok"]).sum())
    for col, key in (("shape_class", "shape"), ("tier", "tier"),
                     ("group_reported", "group")):
        if col in df:
            out[key] = df[col].value_counts().to_dict()
    if "rectified" in df:
        out["n_unrectified"] = int((~df["rectified"].fillna(False).astype(bool)).sum())
    return out


def results_table(fits: Sequence, extra: Optional[Dict[str, Dict]] = None) -> "object":
    """Assemble the per-object results table from a list of ObjectFit."""
    import pandas as pd

    rows = []
    for fit in fits:
        row = fit.to_row()
        if extra and fit.object_id in extra:
            row.update(extra[fit.object_id])
        rows.append(row)
    df = pd.DataFrame(rows)

    # Order known columns as documented; keep any others at the end rather than
    # dropping them, so nothing is lost silently.
    known = [c for c, _, _ in ((n, u, d) for n, u, d in DATA_DICTIONARY)]
    ordered = [c for c in known if c in df.columns]
    rest = [c for c in df.columns if c not in ordered]
    return df[ordered + rest]


def write_tables(df, outdir, typology=None, truth=None, study=None,
                 write_excel: bool = True) -> List[Path]:
    """Write results.csv, the data dictionary, and a formatted workbook."""
    import pandas as pd

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = [outdir / "results.csv"]
    df.to_csv(written[0], index=False)

    slim = df[[c for c in SLIM_COLUMNS if c in df.columns]].copy()
    slim_path = outdir / "table_s1.csv"
    slim.to_csv(slim_path, index=False)
    written.append(slim_path)

    dd = pd.DataFrame(
        [{"column": c, "units": u, "description": d} for c, u, d in DATA_DICTIONARY]
    )
    dd = dd[dd["column"].isin(df.columns)]
    dd_path = outdir / "data_dictionary.csv"
    dd.to_csv(dd_path, index=False)
    written.append(dd_path)

    if study is not None:
        p = outdir / "simulation_study.csv"
        study.to_csv(p, index=False)
        written.append(p)

    if write_excel:
        try:
            xlsx = outdir / "results.xlsx"
            with pd.ExcelWriter(xlsx, engine="openpyxl") as xl:
                # Table S1 first: it is the one a reader opens the file for.
                slim.to_excel(xl, sheet_name="Table S1", index=False)
                df.to_excel(xl, sheet_name="results_full", index=False)
                dd.to_excel(xl, sheet_name="data_dictionary", index=False)
                if typology is not None:
                    pd.DataFrame([typology.to_dict()]).T.reset_index().rename(
                        columns={"index": "quantity", 0: "value"}
                    ).to_excel(xl, sheet_name="typology", index=False)
                if truth is not None:
                    rows = []
                    for key in ("n_matched", "n_width_comparable",
                                "diff_vs_coverage_slope", "diff_vs_coverage_p"):
                        if key in truth:
                            rows.append({"quantity": key, "value": truth[key]})
                    ag = truth.get("agreement_width")
                    if ag is not None:
                        rows += [{"quantity": f"agreement_{k}", "value": v}
                                 for k, v in ag.to_dict().items()]
                    dm = truth.get("deming_width")
                    if dm is not None:
                        rows += [{"quantity": f"deming_{k}", "value": v}
                                 for k, v in dm.items()]
                    if rows:
                        pd.DataFrame(rows).to_excel(xl, sheet_name="validation", index=False)
                if study is not None:
                    study.to_excel(xl, sheet_name="simulation", index=False)
            written.append(xlsx)
        except ImportError:
            pass
    return written


def methods_text(df, typology=None, truth=None, card_spec=None,
                 n_boot: int = 2000, alpha: float = 0.05,
                 e_practical: float = 0.5) -> str:
    """A methods section drafted from what the run actually did.

    Generated from the results rather than written by hand, so the numbers in
    the prose cannot drift from the numbers in the table.
    """
    counts = summary_counts(df)
    n = counts.get("n_objects", 0)
    n_fit = counts.get("n_fitted", 0)
    shape = counts.get("shape", {})
    tier = counts.get("tier", {})

    cw = f"{card_spec.width_cm:g} x {card_spec.height_cm:g} cm" if card_spec else "the scale card"
    if card_spec:
        # Describe the rows as printed. Summarising a mixed card as "N x M
        # squares of k cm" would misstate the reference the scale rests on.
        groups, layout = [], card_spec.row_layout
        for height_cm, n_cells in sorted(set(layout), key=lambda r: -r[0]):
            n_rows = sum(1 for r in layout if r == (height_cm, n_cells))
            side = card_spec.width_cm / n_cells
            groups.append(f"{n_rows} row{'s' if n_rows > 1 else ''} of "
                          f"{n_cells} squares of {side:g} cm")
        card_desc = f"{cw}, " + " and ".join(groups)
    else:
        card_desc = "10 x 2 cm, 2 rows of 10 squares of 1 cm"

    parts = [
        "## Methods (draft)",
        "",
        "### Image calibration",
        "",
        f"Each object was photographed in the field beside a checkerboard scale card "
        f"({card_desc}). "
        "Because the photographs were taken obliquely from standing height, a scalar "
        "pixels-per-centimetre conversion is not sufficient: a circular object viewed "
        "off-axis projects to an ellipse, which would produce spurious evidence of "
        "elongation. The four corners of the scale card were therefore located "
        "automatically (and confirmed by the operator) and used to compute a "
        "homography mapping image pixels to centimetres on the ground plane. All "
        "subsequent measurement was carried out in those rectified coordinates.",
        "",
        f"Of {n} objects, "
        f"{counts.get('n_unrectified', 0)} could not be rectified and were calibrated "
        "with a scalar scale only; these are flagged in the results table.",
        "",
        "### Digitisation",
        "",
        "The surviving outline of each object was digitised by hand in a "
        "purpose-built interface, with each click refined to the nearest sub-pixel "
        "image-gradient maximum along the local curve normal. Raw and refined "
        "coordinates were both retained. Digitised points, the calibration, and "
        "provenance are archived as one JSON record per object; all results below "
        "regenerate from those records without reference to the original images.",
        "",
        "### Reconstruction",
        "",
        "An ellipse was fitted to each set of digitised points by minimising "
        "perpendicular (orthogonal) distances, seeded by the numerically stable "
        "direct least-squares method of Halir and Flusser (1998) and refined by "
        "damped Gauss-Newton iteration; perpendicular distance to the ellipse was "
        "evaluated by Eberly's robust bisection. Orthogonal-distance fitting was "
        "used in preference to algebraic fitting because algebraic distance is "
        "biased towards small, low-eccentricity ellipses, and that bias falls "
        "directly on the quantity of interest. The maximum length through the "
        "centre of the reconstructed object is the major axis, 2a.",
        "",
        "### Uncertainty",
        "",
        f"Confidence intervals were obtained by a residual bootstrap with "
        f"{n_boot} replicates and bias-corrected and accelerated (BCa) intervals "
        "(Efron 1987). Residuals were resampled at fixed positions along the fitted "
        "curve rather than resampling the points themselves, because resampling "
        "points varies the arc coverage between replicates and arc coverage is "
        "itself the main determinant of how precisely an ellipse can be recovered.",
        "",
        "### Circular or elliptical",
        "",
        "Eccentricity is bounded below by zero, so measurement noise can only "
        "increase it: a truly circular object yields a positive fitted "
        "eccentricity, and increasingly so as less of the outline survives. A "
        "fitted eccentricity above zero is therefore not evidence of an oval "
        "object. Each object was instead assessed by comparing the five-parameter "
        "ellipse against a three-parameter circle, referred to a null distribution "
        "generated by simulating truly circular objects carrying that object's own "
        "arc coverage, point spacing and residual distribution. Any inflation "
        "caused by a short arc is then present in the null as well and cancels.",
        "",
        f"Objects were classified as elliptical where this test rejected circularity "
        f"at alpha = {alpha}, as circular where it did not reject *and* the upper "
        f"confidence bound on eccentricity fell below {e_practical} (a minor axis "
        "about 87% of the major, an elongation that is plainly visible), and as "
        "indeterminate otherwise. The third category matters: failing to reject a "
        "circle is not evidence of circularity when the surviving arc is too short "
        "to have detected an ellipse.",
        "",
        f"Of {n_fit} successfully reconstructed objects, "
        f"{shape.get('circular', 0)} were classified as circular, "
        f"{shape.get('elliptical', 0)} as elliptical and "
        f"{shape.get('indeterminate', 0)} as indeterminate.",
        "",
        "### Reliability tiers",
        "",
        f"Objects were tiered by arc coverage, interval width, fit residual and "
        f"calibration quality: tier A (reliable) n = {tier.get('A', 0)}, tier B "
        f"(marginal) n = {tier.get('B', 0)}, tier C (indeterminate) n = "
        f"{tier.get('C', 0)}.",
    ]

    if truth and truth.get("agreement_width") is not None:
        ag = truth["agreement_width"]
        dm = truth.get("deming_width", {})
        parts += [
            "",
            "### Validation against direct measurement",
            "",
            f"All objects were also measured directly with calipers. The "
            f"reconstructed minor axis was compared against the fully preserved "
            f"width across (n = {ag.n} comparable pairs); the fragment's maximum "
            "dimension is not comparable to the reconstructed major axis, since the "
            "original extended past the break, but it provides a one-sided "
            "consistency check.",
            "",
            f"Mean difference (photogrammetric minus caliper) was "
            f"{ag.bias:+.3f} cm (95% CI {ag.bias_lo:+.3f} to {ag.bias_hi:+.3f}), "
            f"i.e. {ag.bias_pct:+.1f}% of the reference, with 95% limits of "
            f"agreement from {ag.loa_lower:+.3f} to {ag.loa_upper:+.3f} cm. "
            f"Deming regression (which accounts for error in both measurements, "
            f"unlike ordinary least squares) gave a slope of "
            f"{dm.get('slope', float('nan')):.3f} "
            f"({dm.get('slope_lo', float('nan')):.3f} to {dm.get('slope_hi', float('nan')):.3f}).",
            "",
            "Because the residual systematic effects here -- calibration scale error, "
            "and the parallax arising from the outline sitting above the plane of "
            "the scale card -- are isotropic, they scale the whole outline uniformly. "
            "The bias measured on the width therefore applies equally to the major "
            "axis, and leaves eccentricity unaffected. Axis lengths are reported "
            "uncorrected, with this measured bias as a stated systematic term.",
        ]

    if typology is not None and typology.n:
        am = typology.antimode or {}
        parts += [
            "",
            "### Size typology",
            "",
            "Whether the reconstructed major axes form one size group or two was "
            "assessed by a parametric bootstrap likelihood-ratio test comparing "
            "one- and two-component Gaussian mixtures (the usual chi-square "
            "reference distribution is invalid for mixtures, since the null lies on "
            "the boundary of the parameter space), and independently by Silverman's "
            "bandwidth test, which assumes no parametric form. "
            f"{typology.verdict.capitalize()}.",
        ]
        if np.isfinite(am.get("antimode", np.nan)):
            parts.append(
                f"\nThe boundary between groups was estimated at "
                f"{am['antimode']:.2f} cm (95% CI {am.get('lo', float('nan')):.2f} to "
                f"{am.get('hi', float('nan')):.2f}). Objects were assigned to groups "
                "with their own measurement uncertainty propagated, so those whose "
                "intervals straddle the boundary are reported as uncertain rather "
                "than forced into a class."
            )

    parts += [
        "",
        "### References",
        "",
        "Bland, J.M. & Altman, D.G. (1986) Statistical methods for assessing "
        "agreement between two methods of clinical measurement. *Lancet* 1, 307-310.",
        "",
        "Efron, B. (1987) Better bootstrap confidence intervals. *Journal of the "
        "American Statistical Association* 82, 171-185.",
        "",
        "Fitzgibbon, A., Pilu, M. & Fisher, R.B. (1999) Direct least square fitting "
        "of ellipses. *IEEE Transactions on Pattern Analysis and Machine "
        "Intelligence* 21, 476-480.",
        "",
        "Halir, R. & Flusser, J. (1998) Numerically stable direct least squares "
        "fitting of ellipses. *Proceedings of WSCG* 6, 125-132.",
        "",
        "Hartley, R. & Zisserman, A. (2003) *Multiple View Geometry in Computer "
        "Vision*, 2nd edn. Cambridge University Press.",
        "",
        "Pratt, V. (1987) Direct least-squares fitting of algebraic surfaces. "
        "*ACM SIGGRAPH Computer Graphics* 21, 145-152.",
        "",
        "Silverman, B.W. (1981) Using kernel density estimates to investigate "
        "multimodality. *Journal of the Royal Statistical Society B* 43, 97-99.",
    ]
    return "\n".join(parts)
