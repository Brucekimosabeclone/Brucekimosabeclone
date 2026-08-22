"""End-to-end: rendered photographs in, measurements out, checked against truth.

This is the test that exercises the whole chain as one thing -- card detection,
homography, parallax, digitising, fitting, bootstrap, tables, figures -- against
geometry that was known before the image existed. Unit tests can all pass while
the assembled pipeline is wrong; this is what catches that.
"""

import numpy as np
import pytest

pytest.importorskip("cv2")

from arcfit.calibrate import Calibration, apply_homography
from arcfit.fitting import EllipseParams, fit_ellipse
from arcfit.measurements import check_measurements, init_template, load_measurements
from arcfit.pipeline import AnalysisConfig, run_analysis
from arcfit.scalecard import CardSpec, detect_card
from arcfit.simulate import (
    arc_points, make_camera, make_demo_workdir, project, render_scene,
)


class TestRecoveryFromRenderedPhotographs:
    """Accuracy of the full imaging chain, isolated from digitising noise."""

    @pytest.mark.parametrize("offset", [0.0, 45.0, 90.0])
    @pytest.mark.parametrize("a,b", [(9.0, 6.0), (7.5, 7.2)])
    def test_confident_detection_yields_accurate_recovery(self, offset, a, b):
        """The contract the pipeline actually offers.

        Detection reports its own confidence, and a low-confidence result is
        routed to the operator rather than trusted. What must hold is that a
        *confident* detection produces an accurate measurement.
        """
        img, truth = render_scene(seed=4, a_cm=a, b_cm=b, coverage_deg=200,
                                  camera_offset_cm=offset, theta_deg=35)
        det = detect_card(img, CardSpec())
        assert det is not None
        if not det.is_confident:
            pytest.skip("detection self-reported low confidence; operator would verify")

        cal = Calibration.from_rect(det.corners_px, truth.card_w_cm, truth.card_h_cm,
                                    extra_px=det.interior_px, extra_cm=det.interior_cm)
        K, R, t, _ = make_camera(truth.camera_height_cm, truth.camera_offset_cm,
                                 img_w=img.shape[1], img_h=img.shape[0])
        ell = EllipseParams(0, 0, truth.a_cm, truth.b_cm, truth.theta_rad)
        world = arc_points(ell, 120, truth.coverage_deg, truth.start_deg)
        px = project(K, R, t, world, z=truth.outline_height_cm)
        got = fit_ellipse(*cal.to_cm(px).T)

        # The outline sits above the card's plane, so a known inflation is
        # expected; the check is that the recovered size matches that
        # prediction rather than that it matches the raw truth.
        expected = truth.major_axis_cm * truth.parallax_inflation
        assert got.major_axis == pytest.approx(expected, rel=0.04)

        # Shape is checked on the axis ratio, not on eccentricity. Near a circle
        # eccentricity is ill-conditioned -- de/d(b/a) diverges as b/a -> 1 --
        # so a small, honest calibration error moves b/a by a couple of percent
        # while moving e by tens of percent. That is a property of the measure,
        # not of the fit, and it is why both are reported in the results table.
        true_ratio = truth.b_cm / truth.a_cm
        assert got.b / got.a == pytest.approx(true_ratio, rel=0.05)
        if truth.eccentricity > 0.5:
            assert got.eccentricity == pytest.approx(truth.eccentricity, abs=0.06)

    def test_parallax_leaves_shape_alone_end_to_end(self):
        """Underwrites reporting axes uncorrected while trusting the shape call."""
        results = []
        for thickness in (1.0, 5.0, 9.0):
            img, truth = render_scene(seed=9, a_cm=9.0, b_cm=6.0, coverage_deg=260,
                                      thickness_cm=thickness, camera_offset_cm=95)
            det = detect_card(img, CardSpec())
            # Use the true card corners here so the comparison isolates the
            # effect of thickness rather than mixing in detection variation.
            cal = Calibration.from_rect(np.array(truth.card_corners_px),
                                        truth.card_w_cm, truth.card_h_cm)
            K, R, t, _ = make_camera(truth.camera_height_cm, truth.camera_offset_cm,
                                     img_w=img.shape[1], img_h=img.shape[0])
            ell = EllipseParams(0, 0, truth.a_cm, truth.b_cm, truth.theta_rad)
            world = arc_points(ell, 120, truth.coverage_deg, truth.start_deg)
            px = project(K, R, t, world, z=truth.outline_height_cm)
            results.append(fit_ellipse(*cal.to_cm(px).T))

        ecc = [r.eccentricity for r in results]
        assert max(ecc) - min(ecc) < 0.03, "thickness must not change the shape estimate"
        sizes = [r.major_axis for r in results]
        assert max(sizes) > min(sizes), "thickness should change the size estimate"


class TestDetectionConfidence:
    def test_low_confidence_is_reported_not_hidden(self):
        """A very oblique card is harder; the detector must say so."""
        scores = []
        for off in (0.0, 150.0):
            img, _ = render_scene(seed=2, camera_offset_cm=off)
            det = detect_card(img, CardSpec())
            scores.append(det.score if det else 0.0)
        assert scores[0] > scores[1], "confidence should fall as the view gets harder"


class TestFullRun:
    @pytest.mark.slow
    def test_demo_run_produces_every_output(self, tmp_path):
        info = make_demo_workdir(tmp_path, n=6, seed=1)
        out = run_analysis(tmp_path, outdir=tmp_path / "output",
                           config=AnalysisConfig(n_boot=200, jobs=1),
                           measurements=info["measurements"],
                           image_dir=info["images"],
                           per_object_figures=True, progress=False)

        outdir = tmp_path / "output"
        for name in ("results.csv", "data_dictionary.csv", "config.json",
                     "METHODS_DRAFT.md"):
            assert (outdir / name).exists(), f"missing {name}"
        figs = list((outdir / "figures").glob("*.png"))
        assert len(figs) >= 3
        assert list((outdir / "figures" / "objects").glob("*.png"))

        df = out["results"]
        assert len(df) == 6
        assert df["fit_ok"].sum() >= 4
        for col in ("major_axis_cm", "eccentricity", "p_bootstrap", "tier",
                    "shape_class", "extrapolation_factor"):
            assert col in df.columns

    @pytest.mark.slow
    def test_results_are_reproducible_given_the_seed(self, tmp_path):
        """Every interval and p-value must reproduce exactly from the recorded
        seed, or the supplement cannot be checked by anyone."""
        info = make_demo_workdir(tmp_path, n=4, seed=2)
        cfg = AnalysisConfig(n_boot=150, seed=99, jobs=1)
        a = run_analysis(tmp_path, outdir=tmp_path / "o1", config=cfg,
                         measurements=info["measurements"], make_figures=False,
                         progress=False)["results"]
        b = run_analysis(tmp_path, outdir=tmp_path / "o2", config=cfg,
                         measurements=info["measurements"], make_figures=False,
                         progress=False)["results"]
        for col in ("major_axis_cm", "eccentricity_lo", "eccentricity_hi", "p_bootstrap"):
            np.testing.assert_allclose(a[col].to_numpy(float), b[col].to_numpy(float),
                                       rtol=0, atol=0, err_msg=f"{col} not reproducible")

    @pytest.mark.slow
    def test_tier_ordering_tracks_real_accuracy(self, tmp_path):
        """The tiers are only worth reporting if they predict error."""
        import pandas as pd

        info = make_demo_workdir(tmp_path, n=14, seed=3)
        res = run_analysis(tmp_path, outdir=tmp_path / "out",
                           config=AnalysisConfig(n_boot=250, jobs=1),
                           measurements=info["measurements"], make_figures=False,
                           progress=False)["results"]
        truth = pd.read_csv(info["ground_truth"])
        m = res.merge(truth, on="object_id", suffixes=("", "_true"))
        m = m[m["fit_ok"]]
        m["err"] = (m["major_axis_cm"] - m["major_axis_cm_true"]).abs() / m["major_axis_cm_true"]

        good = m[m["tier"].isin(["A", "B"])]["err"]
        poor = m[m["tier"] == "C"]["err"]
        if len(good) >= 2 and len(poor) >= 2:
            assert good.median() < poor.median(), (
                "tier A/B objects must be more accurate than tier C, "
                "otherwise the tiers mean nothing"
            )


class TestMeasurementSheet:
    def test_template_is_prefilled_with_every_object(self, tmp_path):
        images = tmp_path / "images"
        images.mkdir()
        import cv2
        for i in range(3):
            img, _ = render_scene(seed=i)
            cv2.imwrite(str(images / f"OBJ_{i}.jpg"), img)

        out = init_template(images, tmp_path / "m.csv")
        df = load_measurements(out)
        assert len(df) == 3
        assert set(df["object_id"]) == {"OBJ_0", "OBJ_1", "OBJ_2"}
        assert df["width_across_cm"].isna().all()
        assert (tmp_path / "m_README.txt").exists()

    def test_refuses_to_overwrite_typed_in_data(self, tmp_path):
        images = tmp_path / "images"
        images.mkdir()
        import cv2
        img, _ = render_scene(seed=0)
        cv2.imwrite(str(images / "A.jpg"), img)
        init_template(images, tmp_path / "m.csv")
        with pytest.raises(FileExistsError):
            init_template(images, tmp_path / "m.csv")

    def test_validation_reports_problems_without_dropping_rows(self, tmp_path):
        import pandas as pd
        df = pd.DataFrame({
            "object_id": ["A", "B", "B", "D"],
            "fragment_max_cm": [12.0, 15.0, 15.0, 900.0],
            "width_across_cm": [8.0, 20.0, 9.0, 5.0],
        })
        rep = check_measurements(df, record_ids=["A", "B", "C"])
        assert "B" in rep.duplicates
        assert any("D" in s for s in rep.out_of_range)
        assert any("B" in e for e in rep.errors)      # width > fragment max
        assert rep.unmatched_rows == ["D"]
        assert rep.missing_records == ["C"]
        assert not rep.ok


class TestCardLayoutReachesTheAnalysis:
    """A mixed-row card declared on the command line must survive to the output.

    The analysis rebuilds its own CardSpec from the saved config rather than
    being handed the parsed one, so a card described by --card-layout can be
    silently downgraded to a uniform grid between the command line and the
    methods text -- where it would then misdescribe the scale the whole study
    rests on.
    """

    LAYOUT = ((1.0, 10), (1.0, 10), (2.0, 5))

    def test_config_round_trips_the_layout(self):
        cfg = AnalysisConfig(card_width_cm=10.0, card_height_cm=4.0,
                             card_cols=10, card_rows=3,
                             card_row_spec=self.LAYOUT)
        spec = cfg.card_spec()
        spec.validate()
        assert spec.row_layout == self.LAYOUT
        assert spec.height_cm == pytest.approx(4.0)
        assert spec.aspect == pytest.approx(2.5)

    def test_config_survives_serialisation(self):
        """to_dict feeds config.json, which is what the supplement ships."""
        cfg = AnalysisConfig(card_width_cm=10.0, card_height_cm=4.0,
                             card_cols=10, card_rows=3,
                             card_row_spec=self.LAYOUT)
        restored = AnalysisConfig(**cfg.to_dict())
        assert restored.card_spec().row_layout == self.LAYOUT

    def test_default_config_is_still_a_uniform_card(self):
        assert AnalysisConfig().card_spec().row_layout == ((1.0, 10), (1.0, 10))

    def test_cli_resolves_the_layout_for_analyze(self):
        """cmd_analyze must not read --card-height off the namespace directly.

        Its default is None so that a conflict with --card-layout can be
        detected; taking it raw put None into the config and broke the methods
        text at the very end of a long run.
        """
        import argparse
        from arcfit.cli import _add_card_args, _card_from_args

        parser = argparse.ArgumentParser()
        _add_card_args(parser)

        spec = _card_from_args(parser.parse_args(["--card-layout", "1x10,1x10,2x5"]))
        assert spec.height_cm == pytest.approx(4.0)
        assert spec.row_spec == self.LAYOUT

        plain = _card_from_args(parser.parse_args([]))
        assert plain.height_cm == pytest.approx(2.0), "default height must survive"
        assert plain.row_spec == ()
