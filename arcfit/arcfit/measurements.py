"""Caliper measurements: entry template, schema, and validation.

The caliper data has not been digitised yet, so the tool supplies the sheet
rather than expecting one. ``init_template`` scans the photograph folder and
writes a CSV already carrying every object id, so only numbers get typed --
which removes the most common source of join failures, a mistyped filename.

Two measurements are recorded per object, and they play very different roles:

``width_across_cm``  A dimension perpendicular to the break that survives
                     intact. This is directly comparable to the fitted minor
                     axis and is the basis of the accuracy validation.

``fragment_max_cm``  The longest dimension of the surviving piece. This is *not*
                     comparable to the reconstructed major axis -- the original
                     extended past the break -- but it does give a hard lower
                     bound, and checking the reconstruction against it catches
                     gross errors for free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "MEASUREMENT_COLUMNS",
    "COLUMN_HELP",
    "init_template",
    "load_measurements",
    "check_measurements",
    "MeasurementReport",
]

MEASUREMENT_COLUMNS = [
    "object_id",
    "image",
    "fragment_max_cm",
    "width_across_cm",
    "width_is_at_widest",
    "notes",
]

COLUMN_HELP = {
    "object_id": "Object identifier; matches the photograph filename stem. Pre-filled.",
    "image": "Photograph filename. Pre-filled.",
    "fragment_max_cm": "Longest dimension of the surviving fragment, cm. Lower bound on the original.",
    "width_across_cm": "Fully preserved width perpendicular to the break, cm. Blank if not preserved.",
    "width_is_at_widest": "1 if the width was taken at the object's widest point, 0 if not, blank if unsure.",
    "notes": "Free text; anything unusual about the object or the measurement.",
}

# Anything outside this is far more likely a typo or a unit slip than a real
# mano, so it is flagged for a second look rather than silently analysed.
PLAUSIBLE_CM = (2.0, 60.0)


@dataclass
class MeasurementReport:
    """Outcome of validating a measurement sheet."""

    n_rows: int = 0
    n_complete: int = 0
    missing_width: List[str] = None
    missing_fragment: List[str] = None
    out_of_range: List[str] = None
    duplicates: List[str] = None
    unmatched_rows: List[str] = None
    missing_records: List[str] = None
    errors: List[str] = None

    def __post_init__(self):
        for f in ("missing_width", "missing_fragment", "out_of_range",
                  "duplicates", "unmatched_rows", "missing_records", "errors"):
            if getattr(self, f) is None:
                setattr(self, f, [])

    @property
    def ok(self) -> bool:
        return not (self.errors or self.duplicates or self.out_of_range)

    def summary(self) -> str:
        lines = [f"{self.n_rows} rows, {self.n_complete} with both measurements present."]
        def report(label, items, cap=8):
            if items:
                shown = ", ".join(items[:cap])
                more = f" (+{len(items) - cap} more)" if len(items) > cap else ""
                lines.append(f"  {label}: {len(items)} -- {shown}{more}")
        report("ERRORS", self.errors)
        report("duplicate object_id", self.duplicates)
        report("implausible values", self.out_of_range)
        report("rows with no matching record", self.unmatched_rows)
        report("records with no measurement row", self.missing_records)
        report("missing width_across_cm", self.missing_width)
        report("missing fragment_max_cm", self.missing_fragment)
        if self.ok and not self.unmatched_rows and not self.missing_records:
            lines.append("  No blocking problems found.")
        return "\n".join(lines)


def init_template(image_dir, out_csv, overwrite: bool = False) -> Path:
    """Write a measurement CSV pre-filled with one row per photograph."""
    import pandas as pd
    from .records import scan_images

    out_csv = Path(out_csv)
    if out_csv.exists() and not overwrite:
        raise FileExistsError(
            f"{out_csv} already exists; pass overwrite=True to replace it "
            "(this would discard any measurements already typed in)"
        )
    images = scan_images(image_dir)
    if not images:
        raise FileNotFoundError(f"no images found in {image_dir}")

    df = pd.DataFrame({
        "object_id": [p.stem for p in images],
        "image": [p.name for p in images],
        "fragment_max_cm": [""] * len(images),
        "width_across_cm": [""] * len(images),
        "width_is_at_widest": [""] * len(images),
        "notes": [""] * len(images),
    })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    help_path = out_csv.with_name(out_csv.stem + "_README.txt")
    help_path.write_text(
        "Caliper measurement sheet for arcfit\n"
        "====================================\n\n"
        f"One row per photograph; {len(images)} rows pre-filled.\n"
        "Fill in the numeric columns. Leave a cell blank if not measurable --\n"
        "blank is recorded as missing, which is handled; a guess is not.\n\n"
        + "\n".join(f"{c}\n    {COLUMN_HELP[c]}" for c in MEASUREMENT_COLUMNS)
        + "\n\nValidate with:  arcfit check-measurements --measurements "
        + out_csv.name + "\n"
    )
    return out_csv


def load_measurements(path) -> "object":
    """Read a measurement sheet, coercing numeric columns."""
    import pandas as pd

    df = pd.read_csv(path, dtype={"object_id": str, "image": str})
    missing = [c for c in ("object_id", "fragment_max_cm", "width_across_cm")
               if c not in df.columns]
    if missing:
        raise ValueError(f"measurement sheet is missing columns: {missing}")
    for col in ("fragment_max_cm", "width_across_cm", "width_is_at_widest"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["object_id"] = df["object_id"].astype(str).str.strip()
    return df


def check_measurements(measurements, record_ids: Optional[Sequence[str]] = None
                       ) -> MeasurementReport:
    """Validate a measurement sheet and report every problem found.

    Reports rather than raises, and never drops a row silently: an unmatched row
    is a fact the operator needs to see, not something to quietly skip.
    """
    import pandas as pd

    df = measurements if hasattr(measurements, "columns") else load_measurements(measurements)
    rep = MeasurementReport(n_rows=len(df))

    if "object_id" not in df.columns:
        rep.errors.append("no object_id column")
        return rep

    dup = df["object_id"][df["object_id"].duplicated(keep=False)].unique().tolist()
    rep.duplicates = sorted(dup)

    for col in ("fragment_max_cm", "width_across_cm"):
        vals = df[col]
        bad = df.loc[vals.notna() & ((vals < PLAUSIBLE_CM[0]) | (vals > PLAUSIBLE_CM[1])),
                     "object_id"].tolist()
        rep.out_of_range.extend(f"{oid}:{col}" for oid in bad)

    rep.missing_width = df.loc[df["width_across_cm"].isna(), "object_id"].tolist()
    rep.missing_fragment = df.loc[df["fragment_max_cm"].isna(), "object_id"].tolist()
    rep.n_complete = int((df["width_across_cm"].notna() & df["fragment_max_cm"].notna()).sum())

    # A width recorded as larger than the fragment's longest dimension is
    # self-contradictory and almost always two columns swapped.
    both = df["width_across_cm"].notna() & df["fragment_max_cm"].notna()
    swapped = df.loc[both & (df["width_across_cm"] > df["fragment_max_cm"] + 1e-9),
                     "object_id"].tolist()
    for oid in swapped:
        rep.errors.append(f"{oid}: width_across_cm exceeds fragment_max_cm (columns swapped?)")

    if record_ids is not None:
        rec = set(record_ids)
        sheet = set(df["object_id"])
        rep.unmatched_rows = sorted(sheet - rec)
        rep.missing_records = sorted(rec - sheet)
    return rep
