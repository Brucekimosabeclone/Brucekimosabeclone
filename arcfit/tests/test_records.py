"""Record serialisation: the durable artifact the whole supplement rests on."""

import json
import numpy as np
import pytest

from arcfit.calibrate import Calibration
from arcfit.records import SCHEMA_VERSION, ObjectRecord, load_records, sha256_file


def a_record(**kw):
    base = dict(
        object_id="IMG_5443", image="IMG_5443.JPG",
        calibration=Calibration.from_two_points([0, 0], [200, 0], 10.0),
        points_px=[[100.0, 200.0], [140.0, 210.0], [180.0, 230.0],
                   [210.0, 260.0], [230.0, 300.0]],
        points_px_raw=[[101.0, 201.0], [141.0, 211.0], [181.0, 231.0],
                       [211.0, 261.0], [231.0, 301.0]],
        snap_used=True,
    )
    base.update(kw)
    return ObjectRecord(**base)


class TestRoundTrip:
    def test_save_and_load(self, tmp_path):
        rec = a_record()
        rec.save(tmp_path)
        back = ObjectRecord.load(tmp_path / "records" / "IMG_5443.json")
        assert back.object_id == rec.object_id
        assert back.snap_used is True
        assert back.calibration.px_per_cm == pytest.approx(20.0)
        assert back.points_cm() == pytest.approx(rec.points_cm())

    def test_raw_clicks_are_preserved_alongside_snapped(self):
        """Snapping must stay auditable, not baked in irreversibly."""
        rec = a_record()
        assert rec.points_px != rec.points_px_raw
        assert len(rec.points_px) == len(rec.points_px_raw)

    def test_json_is_human_readable(self, tmp_path):
        path = a_record().save(tmp_path)
        data = json.loads(path.read_text())
        assert data["schema_version"] == SCHEMA_VERSION
        assert "points_px" in data and "calibration" in data

    def test_atomic_save_leaves_no_temp_file(self, tmp_path):
        a_record().save(tmp_path)
        assert not list((tmp_path / "records").glob("*.tmp"))


class TestStatus:
    def test_ready_when_calibrated_with_enough_points(self):
        assert a_record().status() == "ready"

    def test_incomplete_with_too_few_points(self):
        assert a_record(points_px=[[1.0, 2.0]]).status() == "incomplete"

    def test_uncalibrated_without_calibration(self):
        assert a_record(calibration=Calibration()).status() == "uncalibrated"

    def test_excluded_wins(self):
        assert a_record(excluded=True).status() == "excluded"


class TestLoading:
    def test_load_records_skips_excluded_by_default(self, tmp_path):
        a_record().save(tmp_path)
        a_record(object_id="IMG_5444", excluded=True).save(tmp_path)
        assert len(load_records(tmp_path)) == 1
        assert len(load_records(tmp_path, include_excluded=True)) == 2

    def test_missing_directory_is_empty_not_an_error(self, tmp_path):
        assert load_records(tmp_path / "nope") == []

    def test_future_schema_is_refused(self, tmp_path):
        """Better to stop than to silently misread a newer record format."""
        rec = a_record()
        d = rec.to_dict()
        d["schema_version"] = SCHEMA_VERSION + 1
        with pytest.raises(ValueError, match="newer"):
            ObjectRecord.from_dict(d)


def test_sha256_of_file(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"arcfit")
    import hashlib
    assert sha256_file(p) == hashlib.sha256(b"arcfit").hexdigest()
