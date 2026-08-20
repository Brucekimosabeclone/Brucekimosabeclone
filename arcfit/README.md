# arcfit

Reconstruct fragmentary objects from scaled photographs, measure the maximum
length through the centre of the reconstructed whole, and decide whether each
object was originally **circular or oval** — with confidence intervals,
publication-ready figures, a documented data table, and a packaged supplement.

Built for a specific problem: 167 field photographs of fragmentary manos, each
shot obliquely on sand beside a checkerboard scale card, each preserving roughly
a quarter to a half of its original outline.

---

## Why not just fit an ellipse and read off the eccentricity

Two things make the obvious approach give confidently wrong answers.

**A circle photographed off-axis projects to an ellipse.** These photographs are
taken from standing height, so the ground plane is oblique. Converting with a
single pixels-per-centimetre number cannot undo that, and the residual
distortion looks exactly like a genuinely oval object. `arcfit` therefore
rectifies every photograph through a homography built from the scale card,
and measures in centimetres on the ground plane.

**Eccentricity cannot go below zero, so noise can only push it up.** A perfect
circle, digitised with ordinary care on a partial arc, fits as visibly
elliptical. Measured on this tool's own simulations:

| surviving arc | fitted eccentricity of a *true circle* |
|---|---|
| 180° | 0.19 (median), 0.27 at the 90th percentile |
| 140° | 0.28 (median), 0.39 at the 90th percentile |
| 100° | **0.43** (median), **0.61** at the 90th percentile |

A rule like "e > 0.3 means oval" would classify almost every circle in the
100° group as oval. So `arcfit` does not threshold eccentricity. It compares a
five-parameter ellipse against a three-parameter circle, referred to a null
distribution simulated at *that object's own* arc coverage, point spacing and
noise — so the short-arc inflation is present in the null too and cancels.

Getting the null's *noise level* right turned out to matter as much as the
geometry. Estimating it from the circle fit is the textbook choice and gives
exact size, but on a genuinely oval object those residuals are dominated by the
circle's own lack of fit; feeding that back in as noise widens the null by as
much as the signal being tested for. Measured over 300 replicates:

| noise estimated from | size (nominal 0.05) | power at e = 0.6, 150° arc |
|---|---|---|
| circle fit | 0.050 | 0.45 |
| ellipse fit | 0.073 | 1.00 |
| **pooled** (used) | **0.063** | **0.84** |

The pooled estimate keeps size within Monte Carlo error of nominal while
recovering most of the power. Erring conservative is deliberate: a false
rejection here means calling a circular object oval, which is the exact error
the method exists to prevent.

---

## Install

```bash
python -m venv .venv
. .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Python 3.9+. The interactive digitiser needs Tk — bundled with python.org and
Homebrew builds; on Linux `sudo apt install python3-tk`. Everything else,
including the whole analysis, runs headless.

## Try it without any data

```bash
arcfit demo --outdir demo_run
arcfit analyze --workdir demo_run --images demo_run/images \
               --measurements demo_run/measurements.csv
```

This renders synthetic photographs with **known** geometry, runs the full
pipeline, and writes `demo_run/ground_truth.csv` so you can check what was
recovered against what went in.

## The real workflow

```bash
# 1. Locate the scale card in every photograph, up front, so failures are known
#    before you start clicking.
arcfit detect --images photos/ --workdir work/

# 2. Digitise. Resumable; press n to advance, and it remembers where you were.
arcfit digitize --images photos/ --workdir work/ --operator BK

# 3. A caliper sheet, pre-filled with all 167 filenames so you only type numbers.
arcfit init-measurements --images photos/ --out work/measurements.csv
arcfit check-measurements --measurements work/measurements.csv --workdir work/

# 4. What is recoverable at your arc coverage? (Calibrates the reliability tiers.)
arcfit validate --out work/simulation_study.csv

# 5. Fit, validate against the calipers, classify, tabulate, plot.
arcfit analyze --workdir work/ --images photos/ \
               --measurements work/measurements.csv \
               --simulation work/simulation_study.csv

# 6. Assemble the journal supplement.
arcfit package --workdir work/ --out supplement/
```

If your scale card is not 10 × 2 cm with 1 cm squares, set it once — every
command takes `--card-width --card-height --card-cols --card-rows`, and nothing
is hard-coded, so a correction later does not mean re-digitising.

**Running this on a real assemblage for the first time?** `docs/RUNBOOK.md` is a
step-by-step operational guide — getting the images off Dropbox, installing on
Windows, and the order to do things in so a mistake surfaces in ten minutes
rather than after seven hours of digitising.

## Digitiser keys

| key | action |
|---|---|
| left click | add an outline point (snapped to the nearest edge) |
| right click | undo the last point |
| `n` / `p` | next / previous object |
| `d` | re-run automatic card detection |
| `m` | manual calibration: click the 4 card corners |
| `t` | two-point scale fallback (**not** rectified) |
| `k` | copy the previous object's calibration |
| `s` | toggle edge snapping |
| `e` / `g` | edge overlay / cycle display contrast |
| `x` | exclude this object |
| `q` | save and quit |

Display aids affect only what you see; stored coordinates are always
full-resolution and both raw and snapped clicks are kept.

## What comes out

```
output/
  results.csv / results.xlsx     one row per object
  data_dictionary.csv            every column defined, with units
  METHODS_DRAFT.md               methods text generated from this run's numbers
  config.json                    including the seed, so results reproduce exactly
  figures/
    fig1_size_vs_shape           headline: size against shape
    fig2_size_distribution       one-hand / two-hand mixture, with the boundary
    fig3_shape_vs_coverage       the limitations figure — put this in the paper
    fig4_eccentricity_by_object  per-object intervals
    fig5_validation              Bland–Altman against the calipers
    objects/<id>.png             photograph, reconstruction, residuals
```

Each figure is written at 600 dpi **and** as vector PDF **and** with the CSV of
its own source data.

## How much to trust each object

Every object gets a tier from its arc coverage, interval width, residual,
calibration quality, and how far the reconstruction extrapolates beyond the
digitised points. On the demo data the tiers track real error:

| tier | median error in reconstructed size |
|---|---|
| A / B | 3.7% |
| C | 10.2% (worst case 78%) |

The runaway cases are caught by `extrapolation_factor` — the reconstructed
major axis divided by the widest separation of the digitised points. Arc
coverage alone misses them, because coverage is measured against the runaway
ellipse and so looks unremarkable.

## Two things the tool does not fix, and reports instead

**Parallax.** The outline sits above the plane of the scale card by about half
the object's thickness. Rectifying it onto the ground plane is a uniform scaling
about the camera's nadir, `H / (H − h)` — roughly 3% at standing height. It
inflates lengths but, being uniform, **leaves eccentricity and orientation
untouched**, so the circular-vs-oval result is immune. Axis lengths are reported
uncorrected; because every object also has a caliper measurement, the bias is
measured empirically rather than modelled, and reported with a confidence
interval.

**Short arcs are ill-conditioned.** With a quarter of an outline, a 0.2%
calibration error can become a 3% size error. That is a property of the geometry,
not a defect, and it is why every quantity carries an interval and why the
limitations figure belongs in the paper rather than the supplement.

## Reproducibility

The digitisation records in `records/` are the durable artifact — raw clicked
coordinates, calibration and provenance, one JSON per object. Every table and
figure regenerates from them with no GUI and no access to the original images.
Seeds are recorded, so bootstrap intervals and p-values reproduce exactly.

## Tests

```bash
pytest                    # everything
pytest -m "not slow"      # skip the statistical calibration tests
```

The slow ones check that the circularity test holds its nominal error rate on
true circles, that intervals cover the truth about 95% of the time, and that the
bimodality test does not invent size groups in unimodal data.

## Licence

MIT.
