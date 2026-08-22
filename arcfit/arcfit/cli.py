"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .pipeline import AnalysisConfig, run_analysis
from .scalecard import CardSpec


def _parse_card_layout(text: str):
    """Parse ``"1x10,1x10,2x5"`` into ``((1.0, 10), (1.0, 10), (2.0, 5))``.

    Each row is ``HEIGHT_CMxCELLS``, given along the card's short side. This
    exists because real scale cards mix square sizes -- a row of 2 cm squares
    beside two rows of 1 cm squares is a common archaeological pattern -- and
    ``--card-rows``/``--card-cols`` can only describe a uniform grid.
    """
    rows = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        bits = part.lower().split("x")
        if len(bits) != 2:
            raise argparse.ArgumentTypeError(
                f"bad card row {part!r}: expected HEIGHTxCELLS, for example 2x5")
        try:
            rows.append((float(bits[0]), int(bits[1])))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"bad card row {part!r}: expected HEIGHTxCELLS, for example 2x5")
    if not rows:
        raise argparse.ArgumentTypeError("empty card layout")
    return tuple(rows)


def _card_from_args(a) -> CardSpec:
    layout = getattr(a, "card_layout", None)
    declared = getattr(a, "card_height", None)
    if layout:
        height = sum(h for h, _ in layout)
        # The layout already states the short side. Letting --card-height
        # disagree would leave two sources of truth for the number that governs
        # eccentricity, so a conflict is an error rather than a silent winner.
        if declared is not None and abs(declared - height) > 1e-6:
            raise SystemExit(
                f"--card-height {declared:g} contradicts --card-layout, which "
                f"sums to {height:g} cm; drop one of them")
        spec = CardSpec(a.card_width, height, a.card_cols, len(layout),
                        row_spec=layout)
    else:
        spec = CardSpec(a.card_width, 2.0 if declared is None else declared,
                        a.card_cols, a.card_rows)
    spec.validate()
    return spec


def _add_card_args(p) -> None:
    g = p.add_argument_group("scale card (set once to match your card)")
    g.add_argument("--card-width", type=float, default=10.0,
                   help="long side of the scale card in cm (default: 10)")
    g.add_argument("--card-height", type=float, default=None,
                   help="short side of the scale card in cm (default: 2)")
    g.add_argument("--card-cols", type=int, default=10,
                   help="squares along the long side (default: 10)")
    g.add_argument("--card-rows", type=int, default=2,
                   help="squares along the short side (default: 2)")
    g.add_argument("--card-layout", type=_parse_card_layout, default=None,
                   metavar="ROWS",
                   help="rows of unequal squares, as HEIGHTxCELLS separated by "
                        "commas, e.g. 1x10,1x10,2x5 for two rows of 1 cm "
                        "squares and one of 2 cm. Overrides --card-height "
                        "and --card-rows.")


# --------------------------------------------------------------------------

def cmd_demo(a) -> int:
    from .simulate import make_demo_workdir

    out = Path(a.outdir)
    print(f"Rendering {a.n} synthetic photographs into {out} ...")
    info = make_demo_workdir(out, n=a.n, seed=a.seed, spec=_card_from_args(a))
    print(f"  images:       {info['images']}")
    print(f"  records:      {out / 'records'}")
    print(f"  measurements: {info['measurements']}")
    print(f"  ground truth: {info['ground_truth']}")
    print(f"\nNow run:  arcfit analyze --workdir {out} --images {info['images']} "
          f"--measurements {info['measurements']}")
    return 0


def cmd_detect(a) -> int:
    from .records import ObjectRecord, scan_images
    from .scalecard import detect_card
    from .calibrate import (Calibration, CalibrationError, estimate_card_aspect,
                            intrinsics_from_exif)

    spec = _card_from_args(a)
    images = scan_images(a.images)
    if not images:
        print(f"no images found in {a.images}", file=sys.stderr)
        return 1

    workdir = Path(a.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    found = low = failed = 0
    rows = []
    for i, path in enumerate(images, 1):
        rec = ObjectRecord.for_image(path, workdir)
        try:
            det = detect_card(path, spec)
        except (RuntimeError, FileNotFoundError) as exc:
            print(f"\n{exc}", file=sys.stderr)
            return 1
        status = "none"
        score = float("nan")
        aspect = float("nan")
        if det is not None:
            score = det.score
            # Measure the card's real aspect ratio, independently of what was
            # declared. Nothing else can catch a wrong --card-height: the
            # homography is built from the declared numbers, so it always
            # reproduces them however wrong they are.
            if rec.image_size_px:
                K = intrinsics_from_exif(path, int(rec.image_size_px[0]),
                                         int(rec.image_size_px[1]))
                if K is not None:
                    try:
                        aspect = estimate_card_aspect(det.corners_px, K)
                    except Exception:
                        aspect = float("nan")
            try:
                rec.calibration = Calibration.from_rect(
                    det.corners_px, spec.width_cm, spec.height_cm, source="auto",
                    image_path=path, image_size_px=rec.image_size_px,
                    detection_score=det.score,
                    extra_px=det.interior_px, extra_cm=det.interior_cm)
                status = "ok" if det.is_confident else "low"
                found += status == "ok"
                low += status == "low"
                if a.save:
                    rec.save(workdir)
            except CalibrationError:
                status = "bad"
        if status in ("none", "bad"):
            failed += 1
        rows.append({"object_id": rec.object_id, "status": status, "score": score,
                     "tilt_deg": rec.calibration.tilt_deg,
                     "measured_aspect": aspect,
                     "implied_height_cm": (spec.width_cm / aspect
                                           if np.isfinite(aspect) and aspect > 0
                                           else float("nan"))})
        print(f"\r  {i}/{len(images)}  confident {found}  low {low}  failed {failed}",
              end="", flush=True)
    print()

    import pandas as pd
    out = workdir / "detection_report.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    # The card-geometry check. Individually noisy -- a near head-on view carries
    # little perspective information -- so judge it on the median across the set.
    #
    # Only confident detections count. A low-scoring row is a quad that failed
    # to verify as a checkerboard, so its aspect describes whatever was found
    # instead of the card; pooling those in would let the failures outvote the
    # successes on exactly the question the check exists to answer.
    measured = np.array([r["measured_aspect"] for r in rows
                         if r["status"] == "ok"], float)
    measured = measured[np.isfinite(measured)]
    if measured.size >= 5:
        med = float(np.median(measured))
        implied = spec.width_cm / med if med > 0 else float("nan")
        declared = spec.aspect
        print(f"\nCard geometry: declared {spec.width_cm:g} x {spec.height_cm:g} cm "
              f"(aspect {declared:.2f}); measured aspect {med:.2f} "
              f"(n={measured.size})")
        if np.isfinite(implied) and abs(med / declared - 1.0) > 0.08:
            print("\n" + "!" * 68)
            print("WARNING: the card does not appear to be the size you declared.")
            print(f"  The four corners across {measured.size} photographs imply a card")
            print(f"  about {spec.width_cm:g} x {implied:.2f} cm, not "
                  f"{spec.width_cm:g} x {spec.height_cm:g} cm.")
            print("  Measure the card with callipers and re-run with the right")
            print("  --card-layout (or --card-height for a uniform card).")
            print("  A wrong height stretches the rectified plane in one direction,")
            print("  so ECCENTRICITY IS WRONG until this is right - and nothing")
            print("  downstream will look obviously broken.")
            print("!" * 68)
        else:
            print("  Consistent with the declared size.")
    elif measured.size:
        print(f"\nCard geometry: too few EXIF-bearing photographs "
              f"({measured.size}) to check the declared card size.")
    else:
        print("\nCard geometry: not checked (no usable EXIF focal length).")

    print(f"\nConfident: {found}   Low confidence: {low}   Failed: {failed}")
    if low or failed:
        print("Low-confidence and failed photographs still need attention in the\n"
              "digitiser: press d to retry, m to click the four corners yourself.")
    print(f"Report written to {out}")
    return 0


def cmd_digitize(a) -> int:
    from .gui import run_digitizer

    try:
        run_digitizer(a.images, a.workdir, card_spec=_card_from_args(a),
                      operator=a.operator, only_missing=a.only_missing,
                      display_max_px=a.display_px)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


def cmd_analyze(a) -> int:
    # Go through _card_from_args rather than the raw namespace: it resolves
    # --card-layout into a full row layout and supplies the default height when
    # none was given. Reading a.card_height directly would drop the layout and
    # pass None straight into the config.
    spec = _card_from_args(a)
    config = AnalysisConfig(
        n_boot=a.boot, seed=a.seed, alpha=a.alpha, e_practical=a.e_practical,
        jobs=a.jobs, card_width_cm=spec.width_cm, card_height_cm=spec.height_cm,
        card_cols=spec.cols, card_rows=spec.rows, card_row_spec=spec.row_spec)

    study = None
    if a.simulation:
        import pandas as pd
        study = pd.read_csv(a.simulation)

    out = run_analysis(a.workdir, outdir=a.outdir, config=config,
                       measurements=a.measurements, image_dir=a.images,
                       make_figures=not a.no_figures,
                       per_object_figures=not a.no_object_figures,
                       study=study)

    s = out["summary"]
    print("\n" + "=" * 62)
    print(f"  {s['n_objects']} objects, {s.get('n_fitted', 0)} reconstructed, "
          f"{s.get('n_failed', 0)} failed")
    if s.get("shape"):
        print("  shape:  " + ",  ".join(f"{k} {v}" for k, v in sorted(s["shape"].items())))
    if s.get("tier"):
        print("  tier:   " + ",  ".join(f"{k} {v}" for k, v in sorted(s["tier"].items())))
    if s.get("group"):
        print("  group:  " + ",  ".join(f"{k} {v}" for k, v in sorted(s["group"].items())))
    if s.get("n_unrectified"):
        print(f"  WARNING: {s['n_unrectified']} objects were not rectified; "
              "perspective is uncorrected for those")

    truth = out.get("truth")
    if truth and truth.get("agreement_width") is not None:
        ag = truth["agreement_width"]
        print(f"\n  Validation against calipers (n={ag.n}):")
        print(f"    bias {ag.bias:+.3f} cm [{ag.bias_lo:+.3f}, {ag.bias_hi:+.3f}]  "
              f"({ag.bias_pct:+.1f}%)")
        print(f"    95% limits of agreement {ag.loa_lower:+.3f} to {ag.loa_upper:+.3f} cm")
        bad = truth.get("qc_shorter_than_fragment") or []
        if bad:
            print(f"    QC FAILURES ({len(bad)}): reconstruction shorter than the "
                  f"surviving fragment: {', '.join(bad[:6])}")
        else:
            print("    QC: no reconstruction is shorter than its own fragment")

    typ = out.get("typology")
    if typ is not None:
        print(f"\n  Typology: {typ.verdict}")

    print(f"\n  Output written to {Path(a.outdir or Path(a.workdir) / 'output')}")
    print("=" * 62)
    return 0


def cmd_validate(a) -> int:
    from .simulate import simulation_study

    print("Running the simulation study (this establishes what arc coverage "
          "can support)...")
    df = simulation_study(replicates=a.replicates, n_points=a.n_points, seed=a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\nWritten to {out}\n")
    print("Median recovered eccentricity when the object is TRULY CIRCULAR")
    print("(anything above zero here is noise, not shape):\n")
    circ = df[df["e_true"] == 0.0]
    if len(circ):
        piv = circ.pivot_table(index="coverage_deg", columns="noise_cm",
                               values="e_median")
        print(piv.round(3).to_string())
        print("\n  Read this as: a perfect circle digitised at this arc coverage and")
        print("  noise level will *appear* this elliptical. It is why a raw")
        print("  eccentricity threshold cannot answer the question.")
    return 0


def cmd_init_measurements(a) -> int:
    from .measurements import init_template

    try:
        path = init_template(a.images, a.out, overwrite=a.overwrite)
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Wrote {path}")
    print(f"Notes on each column: {Path(path).with_name(Path(path).stem + '_README.txt')}")
    return 0


def cmd_check_measurements(a) -> int:
    from .measurements import check_measurements, load_measurements
    from .records import load_records

    df = load_measurements(a.measurements)
    ids = None
    if a.workdir:
        ids = [r.object_id for r in load_records(a.workdir, include_excluded=True)]
    rep = check_measurements(df, record_ids=ids)
    print(rep.summary())
    return 0 if rep.ok else 1


def cmd_prep_samples(a) -> int:
    from .samples import prepare_samples

    names = [x for x in (a.names or "").split(",") if x.strip()] or None
    rep = prepare_samples(a.images, a.out, n=a.n, quality=a.quality,
                          names=names, keep_exif=a.keep_exif)
    if not rep.rows:
        print("Nothing prepared.", file=sys.stderr)
        if rep.missing:
            print("None of the named files were found: "
                  + ", ".join(rep.missing), file=sys.stderr)
        return 1

    print(rep.summary())
    print(f"\nWritten to {Path(a.out).resolve()}")
    print("\nTo push these to a NEW PRIVATE repo, run one line at a time:")
    print("  (create the empty private repo on github.com first)")
    print(f"  cd {a.out}")
    print("  git init")
    print("  git add .")
    print('  git commit -m "sample photographs for detection check"')
    print("  git branch -M main")
    print("  git remote add origin https://github.com/<you>/<private-repo>.git")
    print("  git push -u origin main")
    print("\nDo not drag these onto github.com in a browser -- the browser "
          "upload has a much\nlower size limit than git push does.")
    return 0


def cmd_package(a) -> int:
    from .supplement import build_supplement

    info = build_supplement(a.workdir, a.out, image_dir=a.images,
                            include_images=a.include_images, lite=a.lite)
    print(f"Supplement built at {info['root']}")
    mode = "lite (per-object figures omitted)" if info.get("lite") else "full"
    print(f"  {info['n_files']} files, {info['size_mb']:.1f} MB  [{mode}]")
    if info.get("zip"):
        print(f"  archive: {info['zip']}")
    if not info.get("lite") and info["size_mb"] > 100:
        print("\n  This is large for sharing or syncing. `--lite` omits the")
        print("  per-object figures, which regenerate from the records, and is")
        print("  typically an order of magnitude smaller.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="arcfit",
        description="Reconstruct fragmentary objects from scaled photographs and "
                    "decide whether they were originally circular or oval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Typical run:\n"
               "  arcfit detect     --images photos/ --workdir work/\n"
               "  arcfit digitize   --images photos/ --workdir work/\n"
               "  arcfit init-measurements --images photos/ --out work/measurements.csv\n"
               "  arcfit analyze    --workdir work/ --images photos/ "
               "--measurements work/measurements.csv\n"
               "  arcfit package    --workdir work/ --out supplement/\n")
    p.add_argument("--version", action="version", version=f"arcfit {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="render a synthetic dataset and digitise it")
    d.add_argument("--outdir", default="demo_run")
    d.add_argument("--n", type=int, default=12)
    d.add_argument("--seed", type=int, default=20260820)
    _add_card_args(d)
    d.set_defaults(func=cmd_demo)

    d = sub.add_parser("detect", help="batch-locate the scale card in every photograph")
    d.add_argument("--images", required=True)
    d.add_argument("--workdir", required=True)
    d.add_argument("--save", action="store_true", default=True,
                   help="write the calibration into each record (default: on)")
    _add_card_args(d)
    d.set_defaults(func=cmd_detect)

    d = sub.add_parser("digitize", help="open the click-based digitiser")
    d.add_argument("--images", required=True)
    d.add_argument("--workdir", required=True)
    d.add_argument("--operator", default="")
    d.add_argument("--only-missing", action="store_true",
                   help="skip objects that are already finished")
    d.add_argument("--display-px", type=int, default=1600)
    _add_card_args(d)
    d.set_defaults(func=cmd_digitize)

    d = sub.add_parser("analyze", help="fit, validate, classify, tabulate and plot")
    d.add_argument("--workdir", required=True)
    d.add_argument("--images", default=None, help="needed only for per-object figures")
    d.add_argument("--outdir", default=None)
    d.add_argument("--measurements", default=None, help="caliper CSV")
    d.add_argument("--simulation", default=None, help="simulation study CSV from 'validate'")
    d.add_argument("--boot", type=int, default=2000, help="bootstrap replicates")
    d.add_argument("--seed", type=int, default=20260820)
    d.add_argument("--alpha", type=float, default=0.05)
    d.add_argument("--e-practical", type=float, default=0.5,
                   help="eccentricity above which an object counts as meaningfully oval")
    d.add_argument("--jobs", type=int, default=0, help="parallel workers (0 = all cores)")
    d.add_argument("--no-figures", action="store_true")
    d.add_argument("--no-object-figures", action="store_true")
    _add_card_args(d)
    d.set_defaults(func=cmd_analyze)

    d = sub.add_parser("validate", help="simulation study: what is recoverable")
    d.add_argument("--out", default="simulation_study.csv")
    d.add_argument("--replicates", type=int, default=300)
    d.add_argument("--n-points", type=int, default=30)
    d.add_argument("--seed", type=int, default=20260820)
    d.set_defaults(func=cmd_validate)

    d = sub.add_parser("init-measurements", help="write a caliper sheet pre-filled with filenames")
    d.add_argument("--images", required=True)
    d.add_argument("--out", default="measurements.csv")
    d.add_argument("--overwrite", action="store_true")
    d.set_defaults(func=cmd_init_measurements)

    d = sub.add_parser("check-measurements", help="validate a caliper sheet")
    d.add_argument("--measurements", required=True)
    d.add_argument("--workdir", default=None)
    d.set_defaults(func=cmd_check_measurements)

    d = sub.add_parser("prep-samples",
                       help="make smaller shareable copies of a few photographs")
    d.add_argument("--images", required=True)
    d.add_argument("--out", default="samples")
    d.add_argument("--n", type=int, default=8,
                   help="how many to take, spread across the folder (default: 8)")
    d.add_argument("--quality", type=int, default=92,
                   help="JPEG quality 1-100; resolution is never changed (default: 92)")
    d.add_argument("--names", default=None,
                   help="comma-separated filenames to use instead of --n")
    d.add_argument("--keep-exif", action="store_true",
                   help="keep all EXIF, INCLUDING any GPS coordinates")
    d.set_defaults(func=cmd_prep_samples)

    d = sub.add_parser("package", help="assemble the journal supplement")
    d.add_argument("--workdir", required=True)
    d.add_argument("--out", default="supplement")
    d.add_argument("--images", default=None)
    d.add_argument("--include-images", action="store_true",
                   help="copy the photographs in as well (large)")
    d.add_argument("--lite", action="store_true",
                   help="omit per-object figures (~380 MB for 167 objects); "
                        "they regenerate from the records")
    d.set_defaults(func=cmd_package)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
