"""Preparing shareable copies of photographs.

The central question is whether a re-encoded copy still tests what the original
would have tested. If re-encoding moved the detected scale-card corners, then
samples prepared this way would be measuring the encoder rather than the
photograph, and the whole exercise would be worthless.
"""

import numpy as np
import pytest

pytest.importorskip("cv2")
import cv2
from PIL import Image
from PIL.TiffImagePlugin import IFDRational

from arcfit.samples import prepare_samples, select_spread
from arcfit.scalecard import CardSpec, detect_card
from arcfit.simulate import render_scene

GPS_IFD, EXIF_IFD = 0x8825, 0x8769
ORIENTATION, FOCAL35 = 0x0112, 0xA405


@pytest.fixture
def photos(tmp_path):
    """A folder of scenes carrying realistic EXIF, including GPS."""
    d = tmp_path / "photos"
    d.mkdir()
    for i in range(6):
        img, _ = render_scene(object_id=f"IMG_{5400 + i}", seed=i,
                              camera_offset_cm=60 + 12 * i)
        path = d / f"IMG_{5400 + i}.JPG"
        cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
        with Image.open(path) as im:
            ex = im.getexif()
            ex[ORIENTATION] = 1
            ex.get_ifd(EXIF_IFD)[FOCAL35] = 27
            # Realistic coordinates: Pillow needs IFDRational for GPS rationals,
            # not plain tuples.
            ex.get_ifd(GPS_IFD)[1] = "S"
            ex.get_ifd(GPS_IFD)[2] = (IFDRational(0), IFDRational(17), IFDRational(30))
            ex.get_ifd(GPS_IFD)[3] = "E"
            ex.get_ifd(GPS_IFD)[4] = (IFDRational(36), IFDRational(4), IFDRational(12))
            im.save(path, "JPEG", quality=98, exif=ex.tobytes())
    return d


class TestReEncodingPreservesTheMeasurement:
    def test_recovered_scale_is_unchanged(self, photos, tmp_path):
        """The load-bearing test, and it asserts the right quantity.

        An earlier version compared detected corner *pixel* positions and failed
        at 1.6 px. Measuring across qualities showed why that was the wrong
        target: the shift is not monotonic in quality -- 0.54 px at q98, 0.10 at
        q92, 1.61 at q90, 0.63 at q75 -- so it is not encoding damage at all. It
        is the corner detector occasionally settling on a different local
        optimum when any pixel changes, which happens even at near-lossless
        quality.

        What actually matters is the recovered scale, since that is what
        propagates into every measurement. Measured across these scenes at the
        default quality, it moves by at most 0.12%.
        """
        from arcfit.calibrate import Calibration

        spec = CardSpec()

        def recovered_scale(path):
            det = detect_card(path, spec)
            if det is None:
                return None
            cal = Calibration.from_rect(det.corners_px, spec.width_cm, spec.height_cm,
                                        extra_px=det.interior_px,
                                        extra_cm=det.interior_cm)
            probe = np.array([[1000.0, 800.0], [1400.0, 1100.0]])
            cm = cal.to_cm(probe)
            return float(np.hypot(*(cm[1] - cm[0])))

        rep = prepare_samples(photos, tmp_path / "out", n=6)
        assert rep.rows

        errors = []
        for row in rep.rows:
            before = recovered_scale(photos / row["name"])
            after = recovered_scale(tmp_path / "out" / row["name"])
            if before is None or after is None:
                continue
            errors.append(abs(after - before) / before)

        assert len(errors) >= 3, "too few detections to draw a conclusion"
        assert np.mean(errors) < 0.005, (
            f"re-encoding shifted the recovered scale by {np.mean(errors):.3%} "
            "on average; samples would not represent the originals"
        )
        # Loose per-image bound: a single image can flip the detector to another
        # local optimum, which happens at any quality including near-lossless.
        assert max(errors) < 0.02

    def test_resolution_is_identical(self, photos, tmp_path):
        rep = prepare_samples(photos, tmp_path / "out", n=4, quality=85)
        for row in rep.rows:
            with Image.open(photos / row["name"]) as a, \
                 Image.open(tmp_path / "out" / row["name"]) as b:
                assert a.size == b.size
                assert (b.size[0], b.size[1]) == (row["width"], row["height"])

    def test_files_get_materially_smaller(self, photos, tmp_path):
        rep = prepare_samples(photos, tmp_path / "out", n=4, quality=85)
        assert rep.total_after_mb < rep.total_before_mb
        for row in rep.rows:
            assert row["after_mb"] < row["before_mb"]


class TestExif:
    def test_focal_length_survives_and_gps_does_not(self, photos, tmp_path):
        """Focal length feeds the tilt and camera-height diagnostics, so it has
        to survive. Location data has no analytical use and should not leave
        the machine by default."""
        rep = prepare_samples(photos, tmp_path / "out", n=3)
        assert rep.any_gps_dropped
        for row in rep.rows:
            with Image.open(tmp_path / "out" / row["name"]) as im:
                ex = im.getexif()
                assert ex.get_ifd(EXIF_IFD).get(FOCAL35) == 27
                assert not ex.get_ifd(GPS_IFD), "GPS survived into the copy"

    def test_keep_exif_preserves_everything(self, photos, tmp_path):
        rep = prepare_samples(photos, tmp_path / "out", n=2, keep_exif=True)
        assert rep.kept_exif
        for row in rep.rows:
            with Image.open(tmp_path / "out" / row["name"]) as im:
                assert im.getexif().get_ifd(GPS_IFD)

    def test_summary_states_what_happened_to_gps(self, photos, tmp_path):
        dropped = prepare_samples(photos, tmp_path / "a", n=2).summary()
        assert "REMOVED" in dropped and "originals are untouched" in dropped
        kept = prepare_samples(photos, tmp_path / "b", n=2, keep_exif=True).summary()
        assert "INCLUDING any GPS" in kept


class TestSelection:
    def test_spread_covers_the_range(self):
        """Not the first n: the opening frames of a shoot are the least
        informative sample you could send."""
        paths = [f"IMG_{i:04d}.JPG" for i in range(100)]
        picked = select_spread(paths, 5)
        assert len(picked) == 5
        assert picked[0] == paths[0]
        assert picked[-1] != paths[4], "selection clustered at the start"
        assert paths.index(picked[-1]) > 70

    def test_spread_handles_short_sequences(self):
        paths = ["a.JPG", "b.JPG"]
        assert select_spread(paths, 8) == paths
        assert select_spread(paths, 0) == []

    def test_explicit_names_override_n(self, photos, tmp_path):
        rep = prepare_samples(photos, tmp_path / "out", n=6,
                              names=["IMG_5401.JPG", "IMG_5404.JPG"])
        assert [r["name"] for r in rep.rows] == ["IMG_5401.JPG", "IMG_5404.JPG"]

    def test_names_are_case_insensitive(self, photos, tmp_path):
        rep = prepare_samples(photos, tmp_path / "out", names=["img_5401.jpg"])
        assert [r["name"] for r in rep.rows] == ["IMG_5401.JPG"]

    def test_missing_names_are_reported_not_swallowed(self, photos, tmp_path):
        rep = prepare_samples(photos, tmp_path / "out",
                              names=["IMG_5401.JPG", "NOPE.JPG"])
        assert rep.missing == ["NOPE.JPG"]
        assert len(rep.rows) == 1
        assert "NOT FOUND" in rep.summary()


class TestGuards:
    def test_rejects_impossible_quality(self, photos, tmp_path):
        with pytest.raises(ValueError):
            prepare_samples(photos, tmp_path / "out", quality=0)
        with pytest.raises(ValueError):
            prepare_samples(photos, tmp_path / "out", quality=101)

    def test_empty_source_folder_is_an_error(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError):
            prepare_samples(tmp_path / "empty", tmp_path / "out")

    def test_originals_are_never_modified(self, photos, tmp_path):
        before = {p.name: p.read_bytes() for p in photos.iterdir()}
        prepare_samples(photos, tmp_path / "out", n=6, quality=60)
        for p in photos.iterdir():
            assert p.read_bytes() == before[p.name], f"{p.name} was modified"
