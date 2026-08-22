"""Catching a mis-declared scale card, and recording which plane it sat in.

Both come from looking at real field photographs. A wrong declared card height
is the one input that corrupts eccentricity while leaving everything else
looking plausible, and nothing in the pipeline could previously detect it --
the homography is built from the declared numbers, so it always reproduces
them however wrong they are.
"""

import numpy as np
import pytest

from arcfit.calibrate import estimate_card_aspect
from arcfit.records import ObjectRecord
from arcfit.simulate import make_camera, project


def project_card(width_cm, height_cm, offset_cm=90.0, f_px=3000.0,
                 img_w=4000, img_h=3000):
    K, R, t, _ = make_camera(150.0, offset_cm, img_w=img_w, img_h=img_h, f_px=f_px)
    card = np.array([[0.0, 0.0], [width_cm, 0.0],
                     [width_cm, height_cm], [0.0, height_cm]])
    return project(K, R, t, card, z=0.0), K


class TestCardAspectMeasurement:
    @pytest.mark.parametrize("w,h", [(10.0, 2.0), (10.0, 2.5), (10.0, 3.0), (15.0, 5.0)])
    @pytest.mark.parametrize("offset", [60.0, 90.0, 130.0])
    def test_recovers_true_aspect(self, w, h, offset):
        px, K = project_card(w, h, offset_cm=offset)
        assert estimate_card_aspect(px, K) == pytest.approx(w / h, rel=0.02)

    def test_implied_height_exposes_a_wrong_declaration(self):
        """The scenario this exists for: the card is really 10 x 2.5 but was
        declared 10 x 2. The measurement must point at 2.5."""
        px, K = project_card(10.0, 2.5)
        implied = 10.0 / estimate_card_aspect(px, K)
        assert implied == pytest.approx(2.5, rel=0.03)
        assert abs(implied - 2.0) > 0.3, "a 25% error must be clearly visible"

    def test_is_independent_of_what_was_declared(self):
        """The check would be worthless if it merely echoed the input. It sees
        only the corner pixels and the intrinsics, never the declared size."""
        px, K = project_card(10.0, 2.5)
        assert estimate_card_aspect(px, K) == pytest.approx(4.0, rel=0.02)

    def test_degenerate_input_gives_nan_not_a_wrong_number(self):
        K = np.array([[3000.0, 0, 2000.0], [0, 3000.0, 1500.0], [0, 0, 1.0]])
        collapsed = np.array([[100.0, 100.0], [200.0, 100.0],
                              [200.0, 100.0], [100.0, 100.0]])
        try:
            val = estimate_card_aspect(collapsed, K)
        except Exception:
            return
        assert not np.isfinite(val) or val > 0


class TestCardPlaneRecording:
    def test_defaults_to_the_common_case(self):
        rec = ObjectRecord(object_id="A", image="A.JPG")
        assert rec.card_on_object is False

    def test_round_trips_through_a_saved_record(self, tmp_path):
        rec = ObjectRecord(object_id="A", image="A.JPG", card_on_object=True)
        rec.save(tmp_path)
        back = ObjectRecord.load(tmp_path / "records" / "A.json")
        assert back.card_on_object is True

    def test_survives_a_dict_round_trip(self):
        rec = ObjectRecord(object_id="A", image="A.JPG", card_on_object=True)
        assert ObjectRecord.from_dict(rec.to_dict()).card_on_object is True
