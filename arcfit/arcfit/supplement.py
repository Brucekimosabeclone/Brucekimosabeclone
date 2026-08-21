"""Assembling the journal supplement.

The aim is that someone who has only this folder can reproduce every number and
every figure. That means the code, the pinned dependencies, the raw
digitisation records, the caliper sheet, the outputs, and a manifest of
checksums -- plus a demo dataset, so a reviewer without the original
photographs can still run the pipeline end to end and see it work.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

__all__ = ["build_supplement"]

RUNME = """# Reproducing this analysis

Everything below regenerates from the raw digitisation records in `records/`.
The original photographs are not needed.

## 1. Install

    python -m venv .venv
    . .venv/bin/activate          # Windows: .venv\\Scripts\\activate
    pip install -r code/requirements.txt
    pip install -e code

Python 3.9 or newer. On Linux the interactive digitiser also needs Tk
(`sudo apt install python3-tk`); the analysis below does not.

## 2. Reproduce every table and figure

    arcfit analyze --workdir . --measurements measurements.csv --outdir output_reproduced

Results are deterministic: the random seed is recorded in `output/config.json`
and re-used, so bootstrap intervals and p-values reproduce exactly.

## 3. Check it against a known answer

`demo/` holds synthetic photographs generated with known geometry, together
with the true values in `demo/ground_truth.csv`. Running

    arcfit analyze --workdir demo --images demo/images --measurements demo/measurements.csv --outdir demo/output

and comparing against `ground_truth.csv` shows what the method recovers when
the right answer is known in advance.

## 4. What is where

| Path | Contents |
|---|---|
| `code/` | The `arcfit` package and its tests |
| `records/` | One JSON per object: the digitised points, calibration and provenance |
| `measurements.csv` | Caliper measurements |
| `output/` | Results tables, data dictionary, figures, drafted methods text |
| `demo/` | Synthetic dataset with known ground truth |
| `MANIFEST.txt` | SHA256 of every file |
"""


LITE_NOTE = """

## A note on this archive

It was built in **lite mode**: the per-object figures are not included, because
at roughly 2.3 MB each they would dominate the download.

Nothing is missing. Every per-object figure regenerates from the digitisation
records in `records/`, together with everything else, using the command in
step 2 above. The summary figures, all tables, and the complete raw data are
present.
"""


def _copy(src: Path, dst: Path, ignore=None) -> int:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(*(ignore or [])))
        return sum(1 for p in dst.rglob("*") if p.is_file())
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return 1


def build_supplement(workdir, outdir, image_dir=None, include_images: bool = False,
                     make_zip: bool = True, lite: bool = False) -> Dict[str, object]:
    """Assemble a self-contained supplement directory (and a zip of it).

    ``lite`` omits the per-object figures, which dominate the size: at roughly
    2.3 MB each (600 dpi raster plus vector), 167 objects is about 380 MB, and
    the resulting archive is awkward to share or sync. Nothing is lost by
    omitting them -- every one regenerates from the digitisation records with a
    single command, which RUNME and the manifest both state.
    """
    workdir = Path(workdir)
    root = Path(outdir)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    pkg_root = Path(__file__).resolve().parent.parent

    _copy(pkg_root / "arcfit", root / "code" / "arcfit",
          ignore=["__pycache__", "*.pyc"])
    for name in ("pyproject.toml", "requirements.txt", "LICENSE", "README.md"):
        src = pkg_root / name
        if src.exists():
            _copy(src, root / "code" / name)
    if (pkg_root / "tests").is_dir():
        _copy(pkg_root / "tests", root / "code" / "tests",
              ignore=["__pycache__", "*.pyc"])
    if (pkg_root / "docs").is_dir():
        _copy(pkg_root / "docs", root / "docs")

    if (workdir / "records").is_dir():
        _copy(workdir / "records", root / "records")
    for name in ("measurements.csv", "measurements_README.txt",
                 "detection_report.csv", "simulation_study.csv"):
        if (workdir / name).exists():
            _copy(workdir / name, root / name)
    if (workdir / "output").is_dir():
        # Per-object figures are excluded in lite mode; everything else in
        # output/ (tables, summary figures, methods draft, config) is small.
        _copy(workdir / "output", root / "output",
              ignore=["objects"] if lite else None)

    # A runnable known-answer example, so the supplement can be verified by a
    # reader who will never have the field photographs.
    demo_src = workdir / "demo"
    if demo_src.is_dir():
        _copy(demo_src, root / "demo")
    else:
        try:
            from .simulate import make_demo_workdir
            make_demo_workdir(root / "demo", n=6)
        except Exception:
            pass

    if include_images and image_dir:
        _copy(Path(image_dir), root / "images")

    runme = RUNME
    if lite:
        runme += LITE_NOTE
    (root / "RUNME.md").write_text(runme)

    # Freeze the environment that actually produced these numbers.
    try:
        import importlib.metadata as md
        pins = []
        for dist in ("numpy", "scipy", "matplotlib", "pandas", "pillow",
                     "openpyxl", "opencv-python", "opencv-python-headless"):
            try:
                pins.append(f"{dist}=={md.version(dist)}")
            except md.PackageNotFoundError:
                continue
        (root / "code" / "requirements-frozen.txt").write_text(
            "# Exact versions used to produce the results in output/.\n"
            f"# Python {sys.version.split()[0]}\n" + "\n".join(pins) + "\n")
    except Exception:
        pass

    files = sorted(p for p in root.rglob("*") if p.is_file())
    lines = ["# SHA256 checksums of every file in this supplement", ""]
    if lite:
        lines[1:1] = [
            "# Built in lite mode: the per-object figures are not included.",
            "# Regenerate all of them from the records with:",
            "#     arcfit analyze --workdir . --measurements measurements.csv \\",
            "#                    --outdir output_regenerated",
            "",
        ]
    total = 0
    for p in files:
        if p.name == "MANIFEST.txt":
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        total += p.stat().st_size
        lines.append(f"{h}  {p.relative_to(root).as_posix()}")
    (root / "MANIFEST.txt").write_text("\n".join(lines) + "\n")

    info: Dict[str, object] = {
        "root": root,
        "n_files": len(files) + 1,
        "size_mb": total / 1e6,
        "lite": lite,
    }
    if make_zip:
        archive = shutil.make_archive(str(root), "zip", root_dir=root)
        info["zip"] = archive
    return info
