"""The per-object digitisation record: one JSON file per photograph.

These records are the durable output of the manual work. Everything downstream
-- fitting, statistics, figures, tables -- is derived from them and can be
regenerated at any time without the GUI and without the original photographs.
That is what makes the analysis reproducible for a reviewer who will never have
access to 2.5 GB of field images.

Records therefore store *raw inputs*, never derived results: clicked pixel
coordinates, the calibration, and provenance. Both the raw clicks and the
edge-snapped coordinates are kept, so the effect of snapping stays auditable
rather than being baked in irreversibly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

from . import __version__
from .calibrate import Calibration

__all__ = [
    "ObjectRecord",
    "SCHEMA_VERSION",
    "sha256_file",
    "scan_images",
    "records_dir",
    "load_records",
    "utc_now",
]

SCHEMA_VERSION = 1
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path, chunk_size: int = 1 << 20) -> str:
    """Hash a file's contents.

    Recorded so a reviewer can confirm which image produced which measurement
    even if files are renamed. Read in chunks: these are ~15 MB each and there
    are 167 of them.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def scan_images(image_dir) -> List[Path]:
    """All images in a directory, sorted by name for a stable object ordering."""
    d = Path(image_dir)
    if not d.is_dir():
        raise NotADirectoryError(f"not a directory: {d}")
    found = [p for p in d.iterdir() if p.is_file() and p.suffix in IMAGE_SUFFIXES]
    return sorted(found, key=lambda p: p.name)


def records_dir(workdir) -> Path:
    return Path(workdir) / "records"


@dataclass
class ObjectRecord:
    """One digitised object."""

    object_id: str
    image: str
    image_sha256: str = ""
    image_size_px: Optional[List[int]] = None
    calibration: Calibration = field(default_factory=Calibration)
    points_px: List[List[float]] = field(default_factory=list)
    points_px_raw: List[List[float]] = field(default_factory=list)
    snap_used: bool = False
    operator: str = ""
    timestamp_utc: str = field(default_factory=utc_now)
    notes: str = ""
    excluded: bool = False
    exclude_reason: str = ""
    schema_version: int = SCHEMA_VERSION
    arcfit_version: str = __version__

    # ------------------------------------------------------------------
    @property
    def n_points(self) -> int:
        return len(self.points_px)

    @property
    def is_digitised(self) -> bool:
        """Enough points and a calibration to attempt a fit."""
        return self.n_points >= 5 and self.calibration.is_calibrated

    def points_cm(self) -> np.ndarray:
        """Digitised points in centimetres on the rectified ground plane."""
        if not self.points_px:
            return np.zeros((0, 2))
        return self.calibration.to_cm(np.asarray(self.points_px, float))

    def status(self) -> str:
        if self.excluded:
            return "excluded"
        if not self.calibration.is_calibrated:
            return "uncalibrated"
        if self.n_points < 5:
            return "incomplete"
        return "ready"

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["calibration"] = self.calibration.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ObjectRecord":
        d = dict(d)
        version = d.get("schema_version", SCHEMA_VERSION)
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"record schema v{version} is newer than this arcfit "
                f"(v{SCHEMA_VERSION}); upgrade arcfit rather than downgrading data"
            )
        d["calibration"] = Calibration.from_dict(d.get("calibration") or {})
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, workdir) -> Path:
        """Write atomically, so an interrupted save cannot corrupt a record."""
        out = records_dir(workdir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{self.object_id}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path) -> "ObjectRecord":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def for_image(cls, image_path, workdir=None, hash_image: bool = True) -> "ObjectRecord":
        """Load the existing record for an image, or start a fresh one."""
        image_path = Path(image_path)
        object_id = image_path.stem
        if workdir is not None:
            existing = records_dir(workdir) / f"{object_id}.json"
            if existing.exists():
                return cls.load(existing)

        size = None
        try:
            from PIL import Image
            with Image.open(image_path) as im:
                size = list(im.size)
        except Exception:
            pass

        return cls(
            object_id=object_id,
            image=image_path.name,
            image_sha256=sha256_file(image_path) if hash_image else "",
            image_size_px=size,
        )


def load_records(workdir, include_excluded: bool = False) -> List[ObjectRecord]:
    """Every record in a working directory, ordered by object id."""
    d = records_dir(workdir)
    if not d.is_dir():
        return []
    out = []
    for path in sorted(d.glob("*.json")):
        rec = ObjectRecord.load(path)
        if rec.excluded and not include_excluded:
            continue
        out.append(rec)
    return out
