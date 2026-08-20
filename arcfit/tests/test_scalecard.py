"""Scale-card detection against rendered scenes with known card corners."""

import numpy as np
import pytest

from arcfit.scalecard import CardSpec, detect_card, order_card_corners
from arcfit.simulate import render_scene
from arcfit.calibrate import homography_from_rect, apply_homography

pytest.importorskip("cv2")


def align_error(detected, truth):
    """Max corner error, minimised over winding and starting corner."""
    best = np.inf
    for rev in (False, True):
        q = detected[::-1] if rev else detected
        for k in range(4):
            best = min(best, np.abs(np.roll(q, k, axis=0) - truth).max())
    return best


class TestDetection:
    @pytest.mark.parametrize("offset", [0.0, 60.0, 90.0, 120.0])
    def test_finds_card_across_view_angles(self, offset):
        img, truth = render_scene(seed=11, camera_offset_cm=offset)
        det = detect_card(img, CardSpec())
        assert det is not None, f"no card found at offset {offset}"
        err = align_error(det.corners_px, np.array(truth.card_corners_px))
        assert err < 3.0, f"corner error {err:.2f}px too large"

    def test_subpixel_refinement_improves_corners(self):
        img, truth = render_scene(seed=5)
        T = np.array(truth.card_corners_px)
        coarse = detect_card(img, CardSpec(), refine=False)
        fine = detect_card(img, CardSpec(), refine=True)
        assert fine.subpixel
        assert align_error(fine.corners_px, T) <= align_error(coarse.corners_px, T)

    def test_detection_feeds_an_accurate_homography(self):
        """What detection is actually for: recovering true lengths."""
        img, truth = render_scene(seed=7, a_cm=9.0, b_cm=6.0, camera_offset_cm=80)
        det = detect_card(img, CardSpec())
        H, _ = homography_from_rect(det.corners_px, truth.card_w_cm, truth.card_h_cm)
        # The card's own diagonal is a known length; recovering it tests the scale.
        corners_cm = apply_homography(H, det.corners_px)
        diag = np.hypot(*(corners_cm[2] - corners_cm[0]))
        expect = np.hypot(truth.card_w_cm, truth.card_h_cm)
        assert diag == pytest.approx(expect, rel=0.02)

    def test_returns_none_when_no_card_present(self):
        img, _ = render_scene(seed=3)
        # Blank out the card region entirely.
        img[:] = 150
        assert detect_card(img, CardSpec()) is None

    def test_low_score_flags_uncertain_detection(self):
        """Confidence must track reality, so weak detections get reviewed."""
        img, truth = render_scene(seed=3)
        det = detect_card(img, CardSpec())
        assert 0.0 <= det.score <= 1.0
        assert isinstance(det.is_confident, bool)


class TestCornerOrdering:
    def test_long_edge_first(self):
        spec = CardSpec(width_cm=10.0, height_cm=2.0)
        quad = np.array([[0.0, 0.0], [200.0, 0.0], [200.0, 40.0], [0.0, 40.0]])
        for k in range(4):
            ordered = order_card_corners(np.roll(quad, k, axis=0), spec)
            e0 = np.hypot(*(ordered[1] - ordered[0]))
            e1 = np.hypot(*(ordered[2] - ordered[1]))
            assert e0 > e1, "edge 0->1 must be the card's long side"

    def test_ordering_is_deterministic(self):
        """Same quad, any input order, same output -- so orientations are comparable."""
        spec = CardSpec()
        quad = np.array([[10.0, 5.0], [210.0, 8.0], [208.0, 48.0], [12.0, 45.0]])
        ref = order_card_corners(quad, spec)
        for k in range(4):
            assert order_card_corners(np.roll(quad, k, axis=0), spec) == pytest.approx(ref)
        assert order_card_corners(quad[::-1], spec) == pytest.approx(ref)


class TestCardSpec:
    def test_rejects_impossible_specs(self):
        with pytest.raises(ValueError):
            CardSpec(width_cm=0).validate()
        with pytest.raises(ValueError):
            CardSpec(width_cm=2.0, height_cm=10.0).validate()
        with pytest.raises(ValueError):
            CardSpec(cols=1).validate()
