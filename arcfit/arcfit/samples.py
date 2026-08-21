"""Preparing a small set of photographs to share for a detection check.

The scale-card detector is developed against rendered scenes, so a handful of
real photographs is worth more than any amount of further synthetic testing.
Getting them somewhere they can be looked at runs into two practical problems,
and this module solves both.

**Size.** Camera JPEGs from these shoots are around 15 MB each. Eight of them is
a repository slow to push and slow to clone. Re-encoding the same pixels at a
lower JPEG quality typically cuts that several-fold.

The default quality of 92 was chosen by measurement rather than taste. Across
rendered scenes, re-encoding at 92 changed the *recovered scale* -- the quantity
that propagates into every measurement -- by at most 0.12%, while roughly
halving file size. (Real camera JPEGs shrink considerably more than these
synthetic scenes do, because rendered sand and grass texture is unusually
expensive to encode at any quality.)

**Resolution must survive it.** Downscaling would be the obvious way to shrink a
photograph and the wrong one here: it changes how many pixels the scale card
subtends, which is the single variable the detection test exists to probe. A
card that fails to detect at 200 px tells you nothing about the same card at
400 px. So the pixel dimensions are held exactly and only the encoding quality
changes -- and ``tests/test_samples.py`` checks that detected corner positions
are unmoved by it, rather than taking that on trust.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

__all__ = ["SampleReport", "prepare_samples", "select_spread"]

# EXIF tag numbers, named so the intent is readable at the call site.
_TAG_ORIENTATION = 0x0112
_IFD_EXIF = 0x8769
_IFD_GPS = 0x8825
_TAG_FOCAL_35MM = 0xA405

# Above this, a repository starts being unpleasant to push and clone.
_SIZE_WARN_MB = 50.0


@dataclass
class SampleReport:
    """What ``prepare_samples`` did, per file and in total."""

    rows: List[Dict[str, object]] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    out_dir: Optional[Path] = None
    kept_exif: bool = False

    @property
    def total_before_mb(self) -> float:
        return sum(r["before_mb"] for r in self.rows)

    @property
    def total_after_mb(self) -> float:
        return sum(r["after_mb"] for r in self.rows)

    @property
    def any_gps_dropped(self) -> bool:
        return any(r["gps_dropped"] for r in self.rows)

    def summary(self) -> str:
        if not self.rows:
            return "No photographs were prepared."
        lines = [
            f"{'file':<28}{'before':>10}{'after':>10}{'saving':>9}   dimensions",
            "-" * 74,
        ]
        for r in self.rows:
            saving = 1.0 - (r["after_mb"] / r["before_mb"]) if r["before_mb"] else 0.0
            lines.append(
                f"{r['name']:<28}{r['before_mb']:>8.1f} MB{r['after_mb']:>8.1f} MB"
                f"{saving * 100:>8.0f}%   {r['width']} x {r['height']}"
            )
        lines.append("-" * 74)
        factor = (self.total_before_mb / self.total_after_mb) if self.total_after_mb else 0.0
        lines.append(
            f"{len(self.rows)} files: {self.total_before_mb:.1f} MB -> "
            f"{self.total_after_mb:.1f} MB ({factor:.1f}x smaller)"
        )
        lines.append("Pixel dimensions are unchanged; only the JPEG quality differs.")

        if self.kept_exif:
            lines.append("EXIF: kept in full, INCLUDING any GPS coordinates.")
        elif self.any_gps_dropped:
            n = sum(1 for r in self.rows if r["gps_dropped"])
            lines.append(
                f"EXIF: kept focal length and orientation. GPS coordinates were "
                f"present in {n} of these and have been REMOVED from the copies "
                f"(your originals are untouched)."
            )
        else:
            lines.append("EXIF: kept focal length and orientation. No GPS was present.")

        if self.missing:
            lines.append("")
            lines.append(f"NOT FOUND ({len(self.missing)}): " + ", ".join(self.missing))
        if self.total_after_mb > _SIZE_WARN_MB:
            lines.append("")
            lines.append(
                f"That is still {self.total_after_mb:.0f} MB. Consider fewer files, "
                "or a lower --quality."
            )
        return "\n".join(lines)


def select_spread(paths: Sequence[Path], n: int) -> List[Path]:
    """``n`` paths spread evenly across the sequence.

    Deliberately not the first ``n``. The opening frames of a shoot tend to be
    one object in one light, which is the least informative sample you could
    send; spreading across the listing picks up whatever changed during the day.
    """
    paths = list(paths)
    if n >= len(paths):
        return paths
    if n <= 0:
        return []
    step = len(paths) / float(n)
    picked = [paths[min(int(i * step), len(paths) - 1)] for i in range(n)]
    # Guard against duplicates from rounding on short sequences.
    seen, out = set(), []
    for p in picked:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _clean_exif(exif):
    """A minimal EXIF keeping only what the analysis needs.

    Focal length is kept because ``calibrate.intrinsics_from_exif`` reads it to
    estimate camera tilt and height; without it those diagnostics go blank and
    the sample stops resembling the real thing. Orientation is kept so the copy
    displays as the original does. Everything else, GPS included, is dropped.

    Returns ``(exif_bytes_or_None, gps_was_present)``.
    """
    from PIL import Image

    gps_present = False
    try:
        gps_present = bool(exif.get_ifd(_IFD_GPS))
    except Exception:
        gps_present = False

    clean = Image.Exif()
    try:
        orientation = exif.get(_TAG_ORIENTATION)
        if orientation:
            clean[_TAG_ORIENTATION] = orientation
        focal = exif.get_ifd(_IFD_EXIF).get(_TAG_FOCAL_35MM)
        if focal:
            clean.get_ifd(_IFD_EXIF)[_TAG_FOCAL_35MM] = focal
    except Exception:
        pass

    try:
        return clean.tobytes(), gps_present
    except Exception:
        return None, gps_present


def prepare_samples(image_dir, out_dir, n: int = 8, quality: int = 92,
                    names: Optional[Sequence[str]] = None,
                    keep_exif: bool = False) -> SampleReport:
    """Write smaller, shareable copies of a few photographs.

    ``names`` selects specific files by name and overrides ``n``; otherwise
    ``n`` files are taken spread across the folder. Originals are never
    modified.
    """
    from PIL import Image

    from .records import scan_images

    if not 1 <= int(quality) <= 100:
        raise ValueError(f"quality must be between 1 and 100, got {quality}")

    image_dir = Path(image_dir)
    out_dir = Path(out_dir)
    available = scan_images(image_dir)
    if not available:
        raise FileNotFoundError(f"no images found in {image_dir}")

    report = SampleReport(out_dir=out_dir, kept_exif=bool(keep_exif))

    if names:
        by_name = {p.name: p for p in available}
        lower = {p.name.lower(): p for p in available}
        chosen = []
        for raw in names:
            key = str(raw).strip()
            hit = by_name.get(key) or lower.get(key.lower())
            if hit is None:
                report.missing.append(key)
            else:
                chosen.append(hit)
    else:
        chosen = select_spread(available, int(n))

    if not chosen:
        return report

    out_dir.mkdir(parents=True, exist_ok=True)
    for src in chosen:
        before = src.stat().st_size
        with Image.open(src) as im:
            size = im.size
            exif_bytes, gps_present = (None, False)
            if keep_exif:
                raw = im.info.get("exif")
                exif_bytes = raw if raw else None
                try:
                    gps_present = bool(im.getexif().get_ifd(_IFD_GPS))
                except Exception:
                    gps_present = False
            else:
                exif_bytes, gps_present = _clean_exif(im.getexif())

            dst = out_dir / src.name
            # Convert only when the mode actually needs it. Pillow's
            # subsampling="keep" preserves the camera's own chroma layout, but
            # is only available while the object is still the decoded JPEG --
            # calling convert() first produces a new image with no format and
            # makes it an error.
            out_im = im if im.mode in ("RGB", "L") else im.convert("RGB")
            save_kw = {"quality": int(quality), "optimize": True}
            if exif_bytes:
                save_kw["exif"] = exif_bytes
            if im.format == "JPEG" and out_im is im:
                save_kw["subsampling"] = "keep"
            try:
                out_im.save(dst, "JPEG", **save_kw)
            except (ValueError, OSError):
                save_kw.pop("subsampling", None)
                out_im.save(dst, "JPEG", **save_kw)

        # Resolution is the whole point; refuse to ship a copy that lost it.
        with Image.open(dst) as check:
            if check.size != size:
                dst.unlink(missing_ok=True)
                raise RuntimeError(
                    f"{src.name}: dimensions changed {size} -> {check.size}; "
                    "refusing to write a resized copy"
                )

        report.rows.append({
            "name": src.name,
            "before_mb": before / 1e6,
            "after_mb": dst.stat().st_size / 1e6,
            "width": size[0],
            "height": size[1],
            "gps_dropped": bool(gps_present and not keep_exif),
        })
    return report
