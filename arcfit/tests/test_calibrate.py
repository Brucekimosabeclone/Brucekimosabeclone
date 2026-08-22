"""Calibration tests against a synthetic camera with a known pose.

A virtual camera is built at a known height and tilt, a known rectangle and a
known ellipse are projected through it, and the pipeline is asked to recover
what was put in. This exercises the actual premise of the method: that an
oblique photograph can be rectified well enough to measure shape.
"""

import numpy as np
import pytest

from arcfit.calibrate import (
    Calibration,
    _f35_from_tags,
    CalibrationError,
    apply_homography,
    decompose_homography,
    homography_from_rect,
    parallax_inflation,
    two_point_scale,
)
from arcfit.fitting import EllipseParams, fit_ellipse


def make_camera(height_cm=150.0, offset_cm=100.0, f_px=3000.0,
                img_w=4000, img_h=3000):
    """A camera at (0, -offset, height) looking at the world origin.

    Returns (K, R, t) with world-to-camera X_cam = R @ X_world + t, and the
    tilt/height that the decomposition should recover.
    """
    L = np.hypot(offset_cm, height_cm)
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -height_cm / L, -offset_cm / L],
        [0.0, offset_cm / L, -height_cm / L],
    ])
    t = np.array([0.0, 0.0, L])
    K = np.array([[f_px, 0, img_w / 2], [0, f_px, img_h / 2], [0, 0, 1.0]])
    expected_tilt = np.degrees(np.arctan2(offset_cm, height_cm))
    return K, R, t, expected_tilt


def project(K, R, t, pts_cm, z=0.0):
    """Project world points on the plane Z = z into image pixels."""
    pts_cm = np.atleast_2d(np.asarray(pts_cm, float))
    X = np.column_stack([pts_cm, np.full(len(pts_cm), z)])
    cam = (R @ X.T).T + t
    img = (K @ cam.T).T
    return img[:, :2] / img[:, 2:3]


CARD_W, CARD_H = 10.0, 5.0
CARD = np.array([[0.0, 0.0], [CARD_W, 0.0], [CARD_W, CARD_H], [0.0, CARD_H]])


class TestHomography:
    def test_recovers_known_plane_points(self):
        K, R, t, _ = make_camera()
        corners_px = project(K, R, t, CARD)
        H, rms = homography_from_rect(corners_px, CARD_W, CARD_H)
        assert rms < 1e-8

        probe = np.array([[3.0, 2.0], [-7.5, 11.0], [25.0, -4.0]])
        got = apply_homography(H, project(K, R, t, probe))
        assert got == pytest.approx(probe, abs=1e-6)

    def test_scaling_alone_cannot_replace_rectification(self):
        """The reason a plain px/cm scalar is not enough on an oblique shot.

        A true circle in the ground plane is measurably elliptical in the image.
        This is the false-oval failure mode the homography exists to remove.
        """
        K, R, t, _ = make_camera()
        circle = EllipseParams(0.0, 0.0, 8.0, 8.0, 0.0)
        th = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        world = np.column_stack([8 * np.cos(th), 8 * np.sin(th)])
        px = project(K, R, t, world)

        raw = fit_ellipse(px[:, 0], px[:, 1])
        assert raw.eccentricity > 0.3, "an oblique circle should look clearly oval in pixels"

        H, _ = homography_from_rect(project(K, R, t, CARD), CARD_W, CARD_H)
        cm = apply_homography(H, px)
        fixed = fit_ellipse(cm[:, 0], cm[:, 1])
        assert fixed.eccentricity < 1e-4, "rectification should restore circularity"
        assert fixed.a == pytest.approx(circle.a, rel=1e-6)

    @pytest.mark.parametrize("a,b,theta", [(9.0, 5.0, 0.4), (12.0, 11.0, -0.9)])
    def test_recovers_ellipse_through_oblique_view(self, a, b, theta):
        K, R, t, _ = make_camera()
        truth = EllipseParams(2.0, -3.0, a, b, theta)
        tt = np.linspace(0.3, 0.3 + np.radians(150), 40)
        ct, st = np.cos(theta), np.sin(theta)
        world = np.column_stack([
            truth.cx + ct * a * np.cos(tt) - st * b * np.sin(tt),
            truth.cy + st * a * np.cos(tt) + ct * b * np.sin(tt),
        ])
        H, _ = homography_from_rect(project(K, R, t, CARD), CARD_W, CARD_H)
        cm = apply_homography(H, project(K, R, t, world))
        got = fit_ellipse(cm[:, 0], cm[:, 1])
        assert got.a == pytest.approx(a, rel=1e-5)
        assert got.b == pytest.approx(b, rel=1e-5)
        assert got.eccentricity == pytest.approx(truth.eccentricity, abs=1e-5)

    def test_rejects_bad_input(self):
        with pytest.raises(CalibrationError):
            homography_from_rect(np.zeros((3, 2)), 10.0, 5.0)
        with pytest.raises(CalibrationError):
            homography_from_rect(CARD, -1.0, 5.0)


class TestCameraGeometry:
    @pytest.mark.parametrize("height,offset", [(150.0, 100.0), (160.0, 0.0), (140.0, 200.0)])
    def test_recovers_tilt_and_height(self, height, offset):
        K, R, t, expected_tilt = make_camera(height_cm=height, offset_cm=offset)
        H, _ = homography_from_rect(project(K, R, t, CARD), CARD_W, CARD_H)
        geom = decompose_homography(H, K)
        assert geom["tilt_deg"] == pytest.approx(expected_tilt, abs=0.05)
        assert geom["camera_height_cm"] == pytest.approx(height, rel=1e-3)


class TestParallax:
    """The claim that thickness biases size but not shape.

    This underwrites the decision to report axes uncorrected while still
    trusting the circular-vs-oval result, so it is verified numerically rather
    than asserted from the geometry alone.
    """

    @pytest.mark.parametrize("h_obj", [2.0, 5.0, 8.0])
    def test_inflates_size_by_predicted_factor(self, h_obj):
        cam_h = 150.0
        K, R, t, _ = make_camera(height_cm=cam_h, offset_cm=100.0)
        H, _ = homography_from_rect(project(K, R, t, CARD), CARD_W, CARD_H)

        truth = EllipseParams(0.0, 0.0, 9.0, 6.0, 0.3)
        tt = np.linspace(0, 2 * np.pi, 240, endpoint=False)
        ct, st = np.cos(truth.theta), np.sin(truth.theta)
        world = np.column_stack([
            truth.cx + ct * truth.a * np.cos(tt) - st * truth.b * np.sin(tt),
            truth.cy + st * truth.a * np.cos(tt) + ct * truth.b * np.sin(tt),
        ])
        # Outline lifted to height h_obj, but rectified with the ground-plane H.
        px = project(K, R, t, world, z=h_obj)
        got = fit_ellipse(*apply_homography(H, px).T)

        expected = parallax_inflation(cam_h, h_obj)
        assert got.a / truth.a == pytest.approx(expected, rel=1e-3)
        assert got.b / truth.b == pytest.approx(expected, rel=1e-3)

    @pytest.mark.parametrize("h_obj", [2.0, 5.0, 8.0])
    def test_leaves_eccentricity_untouched(self, h_obj):
        cam_h = 150.0
        K, R, t, _ = make_camera(height_cm=cam_h, offset_cm=100.0)
        H, _ = homography_from_rect(project(K, R, t, CARD), CARD_W, CARD_H)
        truth = EllipseParams(0.0, 0.0, 9.0, 6.0, 0.3)
        tt = np.linspace(0, 2 * np.pi, 240, endpoint=False)
        ct, st = np.cos(truth.theta), np.sin(truth.theta)
        world = np.column_stack([
            ct * truth.a * np.cos(tt) - st * truth.b * np.sin(tt),
            st * truth.a * np.cos(tt) + ct * truth.b * np.sin(tt),
        ])
        got = fit_ellipse(*apply_homography(H, project(K, R, t, world, z=h_obj)).T)
        assert got.eccentricity == pytest.approx(truth.eccentricity, abs=2e-3)

    def test_inflation_formula(self):
        assert parallax_inflation(150.0, 5.0) == pytest.approx(150.0 / 145.0)
        assert parallax_inflation(150.0, 0.0) == pytest.approx(1.0)
        with pytest.raises(CalibrationError):
            parallax_inflation(150.0, 150.0)


class TestCalibrationObject:
    def test_round_trip_serialisation(self):
        K, R, t, _ = make_camera()
        cal = Calibration.from_rect(project(K, R, t, CARD), CARD_W, CARD_H)
        back = Calibration.from_dict(cal.to_dict())
        assert back.rectified and back.is_calibrated
        pts = np.array([[100.0, 200.0], [300.0, 400.0]])
        assert back.to_cm(pts) == pytest.approx(cal.to_cm(pts))

    def test_two_point_fallback_is_flagged_unrectified(self):
        cal = Calibration.from_two_points([0, 0], [200, 0], 10.0)
        assert cal.px_per_cm == pytest.approx(20.0)
        assert cal.is_calibrated
        assert not cal.rectified, "two-point mode must never claim to be rectified"

    def test_uncalibrated_record_refuses_to_convert(self):
        with pytest.raises(CalibrationError):
            Calibration().to_cm(np.array([[1.0, 2.0]]))

    def test_two_point_scale_rejects_coincident(self):
        with pytest.raises(CalibrationError):
            two_point_scale([5, 5], [5, 5], 10.0)


class TestFocalLengthFromExif:
    """Recovering a 35mm-equivalent focal length across camera conventions.

    Nothing reported to the user depends on K, but the card-geometry check does,
    and that check is the only thing standing between a mis-declared card and a
    silently wrong eccentricity. A camera whose EXIF cannot be read therefore
    loses that protection entirely, without saying so.
    """

    # Canon EOS 5D Mark II, as written in the field photographs: no
    # FocalLengthIn35mmFilm, but focal-plane resolution is present.
    CANON = {
        "FocalLength": 28.0,
        "FocalPlaneXResolution": 3849.2117888965045,
        "FocalPlaneResolutionUnit": 2,
        "ExifImageWidth": 5616,
    }

    def test_prefers_the_explicit_35mm_tag(self):
        tags = dict(self.CANON, FocalLengthIn35mmFilm=50)
        assert _f35_from_tags(tags) == pytest.approx(50.0)

    def test_falls_back_to_focal_plane_resolution(self):
        """A full-frame body: the equivalent focal length is the focal length."""
        assert _f35_from_tags(self.CANON) == pytest.approx(28.0, rel=0.05)

    def test_handles_centimetre_resolution_unit(self):
        tags = dict(self.CANON, FocalPlaneResolutionUnit=3,
                    FocalPlaneXResolution=3849.2117888965045 / 2.54)
        assert _f35_from_tags(tags) == pytest.approx(28.0, rel=0.05)

    def test_returns_none_when_nothing_is_available(self):
        assert _f35_from_tags({}) is None
        assert _f35_from_tags({"FocalLength": 28.0}) is None

    def test_ignores_a_zero_or_unparseable_35mm_tag(self):
        """A camera writing 0 should fall through, not be believed."""
        assert _f35_from_tags(dict(self.CANON, FocalLengthIn35mmFilm=0)) ==             pytest.approx(28.0, rel=0.05)
        assert _f35_from_tags({"FocalLengthIn35mmFilm": "wide"}) is None

    @pytest.mark.parametrize("x_res", [1e9, 1e-9])
    def test_refuses_an_implausible_sensor_size(self, x_res):
        """A wrong K is worse than none: it yields a confident, wrong tilt."""
        assert _f35_from_tags(dict(self.CANON, FocalPlaneXResolution=x_res)) is None
