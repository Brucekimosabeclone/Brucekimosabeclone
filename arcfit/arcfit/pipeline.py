"""Orchestration: records in, tables and figures out.

Kept separate from the CLI so the whole analysis can be driven from a script or
a notebook, and separate from the statistics so that the science is testable
without going through file I/O.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .calibrate import CalibrationError
from .records import ObjectRecord, load_records
from .scalecard import CardSpec
from .stats import ObjectFit, TierThresholds, analyse_points

__all__ = ["analyse_records", "run_analysis", "AnalysisConfig"]


@dataclass
class AnalysisConfig:
    """Everything that affects the numbers, in one place and written to disk.

    Recorded alongside the results so a run can be reproduced exactly, including
    the seed -- every interval and p-value here is deterministic given this.
    """

    n_boot: int = 2000
    seed: int = 20260820
    alpha: float = 0.05
    e_practical: float = 0.5
    jobs: int = 0                      # 0 = all cores
    card_width_cm: float = 10.0
    card_height_cm: float = 2.0
    card_cols: int = 10
    card_rows: int = 2

    def card_spec(self) -> CardSpec:
        return CardSpec(self.card_width_cm, self.card_height_cm,
                        self.card_cols, self.card_rows)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _analyse_one(args) -> Tuple[ObjectFit, Dict[str, object]]:
    """Worker: fit one record. Module level so it can be pickled."""
    rec_dict, cfg_dict, thresholds_dict = args
    rec = ObjectRecord.from_dict(rec_dict)
    cfg = AnalysisConfig(**cfg_dict)
    thresholds = TierThresholds(**thresholds_dict)

    cal = rec.calibration
    meta: Dict[str, object] = {
        "calibration_mode": cal.mode,
        "rectified": bool(cal.rectified),
        "tilt_deg": cal.tilt_deg,
        "camera_height_cm": cal.camera_height_cm,
        "reproj_rms_cm": cal.reproj_rms_cm,
        "card_detection_score": cal.detection_score,
        "n_points_raw": len(rec.points_px_raw),
        "snap_used": rec.snap_used,
        "card_on_object": bool(rec.card_on_object),
        "image": rec.image,
    }

    if not rec.is_digitised:
        fit = ObjectFit(object_id=rec.object_id, n_points=rec.n_points,
                        error=f"record status: {rec.status()}")
        return fit, meta

    try:
        pts = rec.points_cm()
    except CalibrationError as exc:
        return ObjectFit(object_id=rec.object_id, n_points=rec.n_points,
                         error=str(exc)), meta

    # Seed per object so results do not depend on how the work is parallelised.
    seed = (cfg.seed + abs(hash(rec.object_id))) % (2 ** 31)
    fit = analyse_points(pts[:, 0], pts[:, 1], object_id=rec.object_id,
                         n_boot=cfg.n_boot, seed=seed, alpha=cfg.alpha,
                         e_practical=cfg.e_practical,
                         rectified=cal.rectified, thresholds=thresholds)
    return fit, meta


def _pool_context():
    """Pick a multiprocessing start method that will actually work here.

    ``spawn`` re-imports the parent's ``__main__`` in each worker, which does
    not exist when the analysis is driven from a notebook, a REPL, or a piped
    script -- every worker then dies on import. ``fork`` needs no re-import, so
    it covers those cases on POSIX. Returns None when neither is safe, and the
    caller falls back to running serially.
    """
    import multiprocessing as mp

    try:
        import __main__
        has_main_file = hasattr(__main__, "__file__")
    except ImportError:  # pragma: no cover - defensive
        has_main_file = False

    available = mp.get_all_start_methods()
    if has_main_file and "spawn" in available:
        return mp.get_context("spawn")
    if "fork" in available:
        return mp.get_context("fork")
    if has_main_file and available:
        return mp.get_context(available[0])
    return None


def _run_parallel(payload, n_jobs: int, progress: bool):
    """Fit in parallel, returning [] if the pool cannot be used at all."""
    ctx = _pool_context()
    if ctx is None:
        return []
    results = []
    try:
        with ctx.Pool(n_jobs) as pool:
            for i, out in enumerate(pool.imap(_analyse_one, payload, chunksize=1)):
                results.append(out)
                if progress:
                    print(f"\r  fitting {i + 1}/{len(payload)} "
                          f"({n_jobs} workers)", end="", flush=True)
    except Exception as exc:
        # Never lose the analysis to a parallelism problem; serial is slower but
        # produces identical numbers, because each object is seeded from its id.
        if progress:
            print(f"\r  parallel execution unavailable ({type(exc).__name__}); "
                  f"running serially" + " " * 20)
        return []
    return results


def analyse_records(records: Sequence[ObjectRecord], config: AnalysisConfig,
                    thresholds: TierThresholds = TierThresholds(),
                    progress: bool = True) -> Tuple[List[ObjectFit], Dict[str, Dict]]:
    """Fit every record, in parallel across objects."""
    payload = [(r.to_dict(), config.to_dict(), thresholds.__dict__) for r in records]
    n_jobs = config.jobs or (os.cpu_count() or 1)
    n_jobs = max(1, min(n_jobs, len(payload))) if payload else 1

    results: List[Tuple[ObjectFit, Dict]] = []
    if n_jobs > 1 and len(payload) >= 2:
        results = _run_parallel(payload, n_jobs, progress)
    if not results:
        for i, item in enumerate(payload):
            results.append(_analyse_one(item))
            if progress:
                print(f"\r  fitting {i + 1}/{len(payload)}", end="", flush=True)
    if progress and payload:
        print()

    fits = [r[0] for r in results]
    meta = {f.object_id: m for f, (_, m) in zip(fits, results)}
    return fits, meta


def run_analysis(workdir, outdir=None, config: Optional[AnalysisConfig] = None,
                 measurements=None, image_dir=None, make_figures: bool = True,
                 per_object_figures: bool = True, study=None,
                 progress: bool = True) -> Dict[str, object]:
    """The whole analysis: fit, validate, classify, tabulate, plot."""
    import pandas as pd

    from . import figures as figmod
    from . import report as repmod
    from .measurements import load_measurements
    from .truth import validate_against_truth, qc_reconstruction
    from .typology import analyse_typology, assign_groups, fit_mixture

    workdir = Path(workdir)
    outdir = Path(outdir) if outdir else workdir / "output"
    outdir.mkdir(parents=True, exist_ok=True)
    config = config or AnalysisConfig()

    records = load_records(workdir)
    if not records:
        raise FileNotFoundError(
            f"no digitisation records in {workdir / 'records'}; run 'arcfit digitize' first"
        )
    if progress:
        print(f"Analysing {len(records)} records with {config.n_boot} bootstrap replicates")

    fits, meta = analyse_records(records, config, progress=progress)
    df = repmod.results_table(fits, extra=meta)

    out: Dict[str, object] = {"config": config, "fits": fits, "records": records}

    # --- caliper validation ---------------------------------------------
    truth = None
    if measurements is not None:
        meas = (measurements if hasattr(measurements, "columns")
                else load_measurements(measurements))
        truth = validate_against_truth(df, meas, alpha=config.alpha, seed=config.seed)
        cols = [c for c in ("object_id", "fragment_max_cm", "width_across_cm")
                if c in meas.columns]
        df = df.merge(meas[cols], on="object_id", how="left")
        if "fragment_max_cm" in df:
            df["qc_shorter_than_fragment"] = qc_reconstruction(
                df["major_axis_cm"].to_numpy(float),
                df["fragment_max_cm"].to_numpy(float))
        out["truth"] = truth

    # --- size typology ---------------------------------------------------
    typology = None
    fitted = df[df["fit_ok"]] if "fit_ok" in df else df
    major = fitted["major_axis_cm"].to_numpy(float) if len(fitted) else np.zeros(0)
    if np.isfinite(major).sum() >= 10:
        typology = analyse_typology(major, n_boot=min(500, config.n_boot),
                                    seed=config.seed, alpha=config.alpha)
        out["typology"] = typology
        if typology.mixture_k2 is not None:
            assign = assign_groups(
                fitted["major_axis_cm"].to_numpy(float),
                fitted["major_axis_lo"].to_numpy(float),
                fitted["major_axis_hi"].to_numpy(float),
                typology.mixture_k2, seed=config.seed)
            g = pd.DataFrame({"object_id": fitted["object_id"].to_numpy(),
                              "p_two_hand": assign["p_two_hand"],
                              "group_reported": assign["group_reported"]})
            df = df.merge(g, on="object_id", how="left")

    out["results"] = df

    # --- tables -----------------------------------------------------------
    written = repmod.write_tables(df, outdir, typology=typology, truth=truth, study=study)
    (outdir / "config.json").write_text(json.dumps(config.to_dict(), indent=2))
    methods = repmod.methods_text(df, typology=typology, truth=truth,
                                  card_spec=config.card_spec(),
                                  n_boot=config.n_boot, alpha=config.alpha,
                                  e_practical=config.e_practical)
    (outdir / "METHODS_DRAFT.md").write_text(methods)
    written += [outdir / "config.json", outdir / "METHODS_DRAFT.md"]

    # --- figures ----------------------------------------------------------
    if make_figures:
        figdir = outdir / "figures"
        figdir.mkdir(parents=True, exist_ok=True)
        am = (typology.antimode or {}) if typology else {}
        try:
            figmod.fig_size_vs_shape(
                df, figdir / "fig1_size_vs_shape",
                antimode=am.get("antimode"),
                antimode_ci=(am.get("lo"), am.get("hi")) if am.get("lo") else None)
            figmod.fig_major_axis_density(df, figdir / "fig2_size_distribution",
                                          typology=typology)
            figmod.fig_eccentricity_vs_coverage(df, figdir / "fig3_shape_vs_coverage")
            figmod.fig_eccentricity_forest(df, figdir / "fig4_eccentricity_by_object")
        except Exception as exc:  # a figure failing must not lose the tables
            print(f"  warning: a summary figure failed to render: {exc}")

        if truth and truth.get("agreement_width") is not None:
            try:
                figmod.fig_bland_altman(truth["merged"], truth["agreement_width"],
                                        figdir / "fig5_validation")
            except Exception as exc:
                print(f"  warning: validation figure failed: {exc}")

        if study is not None:
            try:
                figmod.fig_simulation_recovery(study, figdir / "fig6_simulation_recovery")
            except Exception as exc:
                print(f"  warning: simulation figure failed: {exc}")

        if per_object_figures:
            objdir = figdir / "objects"
            objdir.mkdir(parents=True, exist_ok=True)
            by_id = {r.object_id: r for r in records}
            for i, fit in enumerate(fits):
                if not fit.fit_ok:
                    continue
                rec = by_id.get(fit.object_id)
                if rec is None:
                    continue
                img = None
                if image_dir:
                    cand = Path(image_dir) / rec.image
                    img = cand if cand.exists() else None
                try:
                    figmod.object_figure(
                        fit, rec.points_cm(), objdir / fit.object_id,
                        image_path=img, calibration=rec.calibration,
                        card_corners_px=rec.calibration.corners_px,
                        points_px=rec.points_px)
                except Exception as exc:
                    print(f"  warning: figure for {fit.object_id} failed: {exc}")
                if progress:
                    print(f"\r  object figures {i + 1}/{len(fits)}", end="", flush=True)
            if progress:
                print()
        out["figures_dir"] = figdir

    out["written"] = written
    out["summary"] = repmod.summary_counts(df)
    return out
