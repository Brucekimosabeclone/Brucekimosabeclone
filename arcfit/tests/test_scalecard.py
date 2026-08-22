"""Scale-card detection against rendered scenes with known card corners."""

import numpy as np
import pytest

from arcfit.scalecard import (CardSpec, _checkerboard_score, detect_card,
                              order_card_corners)
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


REAL_CARD = CardSpec(10.0, 4.0, 10, 3, row_spec=((1.0, 10), (1.0, 10), (2.0, 5)))
"""The card actually used in the field: 10 x 4 cm, two rows of 1 cm squares
against one row of 2 cm squares."""


def render_card_patch(rows, cell_px=24, flip=False):
    """A clean, face-on rectified card image built from a row layout."""
    total_cm = sum(h for h, _ in rows)
    width_cm = 10.0
    h_px = int(round(total_cm * cell_px))
    w_px = int(round(width_cm * cell_px))
    patch = np.full((h_px, w_px), 235, np.uint8)
    y = 0.0
    for height_cm, n_cells in rows:
        y0 = int(round(y / total_cm * h_px))
        y1 = int(round((y + height_cm) / total_cm * h_px))
        edges = np.linspace(0, w_px, n_cells + 1).astype(int)
        for c in range(n_cells):
            if c % 2 == 0:
                patch[y0:y1, edges[c]:edges[c + 1]] = 30
        y += height_cm
    return patch[::-1] if flip else patch


class TestMixedRowCards:
    """Cards whose rows differ in square size.

    A single (rows, cols) pair cannot describe them, and resampling such a card
    onto a uniform grid scores it at chance -- which is how a genuine card came
    to lose to bare ground on real photographs.
    """

    def test_layout_defaults_to_a_uniform_grid(self):
        assert CardSpec().row_layout == ((1.0, 10), (1.0, 10))

    def test_layout_reports_the_declared_rows(self):
        assert REAL_CARD.row_layout == ((1.0, 10), (1.0, 10), (2.0, 5))
        assert REAL_CARD.aspect == pytest.approx(2.5)

    def test_rejects_rows_that_disagree_with_the_height(self):
        with pytest.raises(ValueError, match="sum to"):
            CardSpec(10.0, 4.0, 10, 2, row_spec=((1.0, 10), (1.0, 10))).validate()

    def test_rejects_a_degenerate_row(self):
        with pytest.raises(ValueError):
            CardSpec(10.0, 2.0, 10, 2, row_spec=((1.0, 10), (1.0, 1))).validate()

    def test_scores_its_own_layout_highly(self):
        patch = render_card_patch(REAL_CARD.row_layout)
        assert _checkerboard_score(patch, REAL_CARD) > 0.95

    def test_uniform_card_still_scores_highly(self):
        """The generalisation must not cost anything on an ordinary bar."""
        spec = CardSpec()
        assert _checkerboard_score(render_card_patch(spec.row_layout), spec) > 0.95

    def test_a_mixed_card_scores_poorly_under_a_uniform_model(self):
        """The regression that broke the real run, pinned down."""
        patch = render_card_patch(REAL_CARD.row_layout)
        assert _checkerboard_score(patch, CardSpec()) < 0.8

    def test_reads_the_card_either_way_up(self):
        """Nothing constrains which end of the card faces the camera."""
        flipped = render_card_patch(REAL_CARD.row_layout, flip=True)
        assert _checkerboard_score(flipped, REAL_CARD) > 0.95

    def test_flat_and_empty_patches_score_zero(self):
        assert _checkerboard_score(np.full((40, 100), 128, np.uint8), REAL_CARD) == 0.0
        assert _checkerboard_score(np.zeros((0, 0), np.uint8), REAL_CARD) == 0.0
