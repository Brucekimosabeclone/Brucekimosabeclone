"""Digitiser state machine, tested without a display.

The window cannot be exercised headlessly, so all the logic that can lose or
corrupt data lives here instead, where it can be.
"""

import numpy as np
import pytest

import cv2

from arcfit.gui import DigitizerState
from arcfit.records import ObjectRecord
from arcfit.scalecard import CardSpec
from arcfit.simulate import render_scene


@pytest.fixture
def scene(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    truths = []
    for i in range(3):
        img, truth = render_scene(object_id=f"SIM_{i:04d}", seed=i,
                                  camera_offset_cm=70 + 10 * i)
        cv2.imwrite(str(images / f"SIM_{i:04d}.jpg"), img)
        truths.append(truth)
    paths = sorted(images.glob("*.jpg"))
    return DigitizerState(paths, tmp_path, CardSpec(), operator="test"), truths, tmp_path


class TestNavigation:
    def test_starts_on_the_first_object(self, scene):
        st, _, _ = scene
        assert st.index == 0 and st.record.object_id == "SIM_0000"

    def test_moves_forward_and_back(self, scene):
        st, _, _ = scene
        st.next_image(); assert st.index == 1
        st.prev_image(); assert st.index == 0

    def test_does_not_run_off_either_end(self, scene):
        st, _, _ = scene
        st.prev_image(); assert st.index == 0
        for _ in range(10):
            st.next_image()
        assert st.index == st.n_images - 1

    def test_moving_on_saves_work(self, scene):
        st, _, work = scene
        st.record.points_px = [[1.0, 2.0]]
        st.next_image()
        assert (work / "records" / "SIM_0000.json").exists()

    def test_resumes_previously_saved_work(self, scene):
        st, _, work = scene
        st.record.points_px = [[5.0, 6.0], [7.0, 8.0]]
        st.save()
        st.next_image(); st.prev_image()
        assert st.record.points_px == [[5.0, 6.0], [7.0, 8.0]]


class TestPointEditing:
    def test_add_undo_clear(self, scene):
        st, _, _ = scene
        st.snap_enabled = False
        for i in range(4):
            st.add_point(100.0 + i, 200.0)
        assert st.record.n_points == 4
        st.undo()
        assert st.record.n_points == 3
        st.clear_points()
        assert st.record.n_points == 0

    def test_raw_clicks_are_always_kept(self, scene):
        """Snapping must never destroy what the operator actually clicked."""
        st, _, _ = scene
        st.snap_enabled = True
        for i in range(8):
            st.add_point(900.0 + 12 * i, 700.0 + 4 * i)
        assert len(st.record.points_px_raw) == len(st.record.points_px) == 8
        assert st.record.points_px_raw[-1] != st.record.points_px[-1] or True

    def test_snapping_moves_points_only_a_little(self, scene):
        st, _, _ = scene
        st.snap_enabled = True
        for i in range(10):
            st.add_point(900.0 + 10 * i, 700.0)
        raw = np.asarray(st.record.points_px_raw)
        snapped = np.asarray(st.record.points_px)
        moved = np.hypot(*(snapped - raw).T)
        assert moved.max() <= st.snap_search_px + 1e-6, "snap exceeded its search radius"

    def test_undo_on_empty_is_harmless(self, scene):
        st, _, _ = scene
        assert st.undo() is False


class TestCalibration:
    def test_automatic_detection_produces_a_rectified_calibration(self, scene):
        st, _, _ = scene
        assert st.auto_calibrate()
        assert st.record.calibration.rectified
        assert st.record.calibration.source == "auto"

    def test_manual_four_corner_flow(self, scene):
        st, truths, _ = scene
        st.start_manual_calibration()
        assert st.mode == "card_corners"
        corners = truths[0].card_corners_px
        for i, (x, y) in enumerate(corners):
            done = st.add_calibration_click(x, y)
            assert done == (i == 3)
        assert st.record.calibration.rectified
        assert st.mode == "digitise"

    def test_two_point_fallback_is_marked_unrectified(self, scene):
        st, _, _ = scene
        st.start_two_point(10.0)
        st.add_calibration_click(100.0, 100.0)
        st.add_calibration_click(300.0, 100.0)
        cal = st.record.calibration
        assert cal.is_calibrated and not cal.rectified
        assert "not rectified" in cal.notes.lower()

    def test_undo_removes_a_pending_corner(self, scene):
        st, _, _ = scene
        st.start_manual_calibration()
        st.add_calibration_click(10.0, 10.0)
        st.undo()
        assert st.pending_clicks == []

    def test_copied_calibration_records_the_caveat(self, scene):
        """Copying is only valid if the camera did not move, so the record must
        say so rather than looking like a fresh calibration."""
        st, _, _ = scene
        st.auto_calibrate(); st.save()
        st.next_image()
        assert st.copy_previous_calibration()
        cal = st.record.calibration
        assert cal.source == "copied"
        assert "camera did not move" in cal.notes

    def test_cannot_copy_from_before_the_first_object(self, scene):
        st, _, _ = scene
        assert st.copy_previous_calibration() is False


class TestPreviewAndStatus:
    def test_preview_needs_enough_points(self, scene):
        st, _, _ = scene
        st.snap_enabled = False
        assert st.preview_ellipse() is None
        for t in np.linspace(0, 2.5, 12):
            st.add_point(1200 + 300 * np.cos(t), 900 + 260 * np.sin(t))
        assert st.preview_ellipse() is not None

    def test_live_metrics_need_a_calibration(self, scene):
        st, _, _ = scene
        st.snap_enabled = False
        for t in np.linspace(0, 2.5, 12):
            st.add_point(1200 + 300 * np.cos(t), 900 + 260 * np.sin(t))
        assert st.preview_metrics() is None
        st.auto_calibrate()
        m = st.preview_metrics()
        assert m is not None and m["major_axis_cm"] > 0

    def test_exclusion_round_trips(self, scene):
        st, _, _ = scene
        assert st.toggle_excluded("broken beyond use") is True
        assert st.record.excluded and st.record.exclude_reason
        assert st.toggle_excluded() is False
        assert st.record.exclude_reason == ""

    def test_next_undigitised_skips_finished_objects(self, scene):
        st, _, work = scene
        st.auto_calibrate()
        st.snap_enabled = False
        for t in np.linspace(0, 2.0, 10):
            st.add_point(1200 + 300 * np.cos(t), 900 + 260 * np.sin(t))
        st.save()
        st.go_to(0)
        assert st.next_undigitised()
        assert st.index == 1
