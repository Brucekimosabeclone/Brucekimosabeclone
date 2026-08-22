# Handoff — `arcfit`

**Project:** reconstructing original mano dimensions from 167 field photographs
of broken fragments.
**Package version:** 1.1.0 · **Branch:** `claude/broken-object-ellipse-analysis-acud5i`
**Status:** software complete and tested; awaiting real-image detection run.

This document is for whoever picks the work up next — a co-author, an assistant
doing the digitising, or the author returning after a gap. `QUICKSTART.md` is
the command sequence, `RUNBOOK.md` the reasoning, `METHODS.md` the text for the
article. This is the map.

---

## 1. What the tool does

Each photograph shows one broken mano with a printed cm scale card. A fragment
preserves part of its original rounded outline. From that surviving arc the tool
reconstructs the whole object:

1. **Rectify.** The scale card is detected, giving a homography from image
   pixels to the ground plane in cm. Everything downstream is measured in that
   rectified plane, not in pixels.
2. **Digitise.** The operator clicks 30–40 points along the surviving edge, with
   optional snapping to a detected image gradient.
3. **Fit.** An ellipse is fitted to those points by orthogonal-distance
   (geometric) least squares — the maximum-likelihood fit under isotropic point
   noise. An algebraic fit seeds it.
4. **Quantify.** Major axis, minor axis, eccentricity, each with a 95%
   bias-corrected-and-accelerated bootstrap interval.
5. **Classify.** Circle-versus-ellipse is decided by a *test*, not a threshold:
   a parametric bootstrap under a fitted-circle null. Each object comes back
   `circular`, `elliptical`, or `indeterminate`.

Output is a publication table, per-object diagnostic figures, generated methods
text, and a zipped supplement.

## 2. Why it is built this way

Three findings shaped the design. Each was measured, not assumed, and each has a
test guarding it.

**Pixels lie about shape.** A circle photographed obliquely fits with
eccentricity above 0.3 in raw pixels and below 0.0001 after rectification. The
photographs are "some angled", so rectification is not optional — it is the
difference between measuring the object and measuring the camera angle.

**No eccentricity threshold can work.** A *true circle* fits with eccentricity
0.43 when only 100° of arc survives, 0.28 at 140°, 0.19 at 180°. Eccentricity
has a hard floor at zero, so noise can only push it up. Any fixed cutoff would
therefore call short-arc circles "oval" at a rate set by how much of the object
happened to break off. Hence the bootstrap test against a circle null, which
absorbs arc coverage into the null distribution.

**Parallax changes size, not shape.** When the card sits on the ground and the
object stands above it, the object images larger by the factor `H/(H−h)`. This
is a homothety: it scales both axes equally and leaves eccentricity and
orientation untouched. So it biases lengths and nothing else, and the decision
was to report it rather than correct it.

## 3. Current state

**Complete and tested.** 6,400 lines across 18 modules, 201 fast tests passing
at HEAD `188cdb7` (2 skipped, 11 slow deselected). The demo pipeline runs end to
end on synthetic data with known ground truth.

**Documentation.** `QUICKSTART.md` (commands only), `RUNBOOK.md` (~500 lines,
full reasoning), `METHODS.md` (article text), `USER_GUIDE.md` (the digitising
GUI), this file. All ship inside the supplement zip.

**Not yet done.** No real photograph has been through the pipeline. Card
detection has been validated against synthetic scenes and eyeballed on five real
field photos, but the 167-image `arcfit detect` run has not happened.

## 4. The two open risks

Both were spotted in the five sample photographs the author sent, and both must
be settled **before** digitising begins, because both corrupt data that would
then have to be re-collected.

### Risk 1 — the scale card may not be 10 × 2 cm

The card in the samples reads roughly **3.5–4:1**, not the 5:1 that a 10 × 2 cm
card implies. It may be nearer 10 × 2.5 cm.

This matters more than any other single number. A wrong declared height stretches
the rectified plane along one axis only. That is a shear applied to every
outline, so it changes the fitted **eccentricity** — the study's headline
variable — while lengths, figures, and diagnostics all still look entirely
plausible. There is no downstream symptom.

*Resolution:* callipers on the card, then `arcfit detect`, which now measures the
imaged card's aspect ratio against the declared one and prints a **"Card
geometry"** line either confirming it or warning loudly. Do not digitise past a
warning.

### Risk 2 — the card is on the object in some photos and on the ground in others

In one of the five samples the card sits on top of the mano; in the others it
lies on the ground beside it.

With consistent placement, parallax is a fixed bias that can be stated in the
methods and corrected or bounded. With *mixed* placement the sign flips between
photographs, converting a correctable bias into irreducible scatter in the size
estimates.

*Resolution:* the digitiser has a per-object flag. Press **`o`** when the card is
resting on the object. It is stored in the record as `card_on_object` and
reported. This costs one keystroke and cannot be recovered afterwards without
revisiting every photograph — so it must be done during digitising, not after.

## 5. Decisions already made

These were settled with the author and are baked into defaults. Change them only
deliberately.

| Decision | Choice |
|---|---|
| View angle | Some angled → homography rectification always on |
| Digitising | Clicks plus edge snapping |
| Circular vs oval | Statistical test against a circle null, not a threshold |
| Expected arc survival | A quarter to a half of the outline |
| Outline traced | The widest visible silhouette |
| Parallax | Report, do not correct |
| Ground truth | All 167 measured with callipers |
| What was measured | Fragment max dimension, plus width across where fully preserved |
| Research aim | One-hand versus two-hand manos |
| Supplement | Slim `table_s1.csv` for the article, full table alongside; `--lite` zip |

## 6. What to do next, in order

The full commands are in `QUICKSTART.md`. The shape of it:

1. **Calliper the card.** Long side, short side, square size.
2. **`arcfit detect`** on all 167. Read the "Card geometry" line. If it warns,
   re-measure and re-run with the right `--card-height` before anything else.
3. **Ten-object pilot**, end to end, including analysis. Four checkpoints: the
   card measures its true length against the cm axes in panel *b*; bias is a few
   percent not tens; no reconstruction is shorter than its own fragment; the
   major axes look like real manos. About 45 minutes, and it protects the seven
   hours that follow.
4. **Calliper sheet** for all 167, then `check-measurements`.
5. **Digitise all 167.** 5–8 hours, resumable — `--only-missing` skips finished
   objects and every `n` keypress saves. Press `o` for card-on-object.
6. **`arcfit validate`** — the simulation study, runnable unattended.
7. **`arcfit analyze`** — about 15 minutes for the full bootstrap.
8. **`arcfit package --lite`** and share.

## 7. Reading the output

`output/table_s1.csv` is the article table. `results.csv` is everything.

- **`major_axis_cm` / `minor_axis_cm`** with 95% CIs — the reconstructed whole
  object. The maximum length through the centre is the major axis.
- **`eccentricity`** with CI, and **axis ratio** alongside it. Both are reported
  because eccentricity is ill-conditioned near a circle: its derivative with
  respect to axis ratio diverges as the ratio approaches 1, so a small
  measurement wobble on a near-circular object moves eccentricity a lot. Axis
  ratio does not have that pathology and is the more stable thing to plot.
- **`shape_class`** — `circular`, `elliptical`, or `indeterminate`.
  `indeterminate` is an honest answer, usually meaning too few points or points
  bunched at one end of the arc. It is not a failure to be tuned away.
- **`tier`** — A/B/C data quality. Tiers track real accuracy: A/B objects showed
  3.7% median error against known geometry, C objects 10.2%. Consider reporting
  A/B as the primary sample with C as a sensitivity check.
- **`extrapolation_factor`** — how far beyond the surviving fragment the
  reconstruction reaches. A conspicuously large object usually has a large value
  here, and that is where to look first when a number seems wrong.

## 8. Traps for whoever continues

- **Windows PowerShell 5.1 has no `&&`.** Run one line per prompt. Every code
  block in the docs is written this way.
- **The virtualenv must be active in every new terminal.** `arcfit` not
  recognised means it is not.
- **The card size must be passed to *every* command**, not just `detect`, if it
  differs from the 10 × 2 cm default.
- **Do not put the field photographs in a public repository.** They are
  unpublished archaeological data and camera EXIF can carry GPS pointing at the
  site. Use a private repo. `arcfit` itself never reads or stores GPS —
  `intrinsics_from_exif` takes only the 35 mm-equivalent focal length, and
  `records.py` stores no EXIF at all. `prep-samples` drops GPS by default when
  making copies to send elsewhere, with `--keep-exif` as the opt-out.
- **Build the supplement on local disk, then copy into the synced folder.**
  Building directly inside Dropbox makes the sync client fight the writes.
- **`--lite` omits per-object figures** (~380 MB), which regenerate from the
  records. Roughly 10–15 MB instead of ~450.

## 9. Where things live

| Module | Responsibility |
|---|---|
| `fitting.py` | Ellipse and circle fitting; point-to-ellipse distance; arc coverage |
| `stats.py` | Bootstrap, BCa intervals, circularity test, tier assignment |
| `calibrate.py` | Homography, rectification, parallax, card-aspect recovery |
| `scalecard.py` | Card detection and corner ordering |
| `gui.py` | Digitiser — state class is fully testable without matplotlib |
| `pipeline.py` | Orchestration and parallelism |
| `figures.py` / `report.py` | Publication figures, tables, methods text |
| `supplement.py` | Zip building, full and `--lite` |
| `simulate.py` / `truth.py` | Simulation study; agreement against callipers |

A note on the fitting code, because it is the part most likely to be
misunderstood by a future reader: the algebraic (Halíř–Flusser) fit is a *seed
only*. The reported parameters always come from the geometric fit. Two
subtleties are load-bearing and have regression tests — the conic is normalised
in **sign** before extracting parameters (a conic is defined up to scale
*including* sign, and getting this wrong pairs the major axis length with the
minor axis direction), and the distance solver is reformulated on `w = s + 1` to
avoid catastrophic cancellation for points near the ellipse centre.

## 10. One bug worth knowing about

An earlier version classified strongly oval objects (eccentricity 0.80 and 0.86,
with confidence intervals nowhere near zero) as `indeterminate`. The p-value
contradicted the confidence interval computed from the same data.

The cause: the parametric bootstrap null pooled residual variance from the circle
fit and the ellipse fit. On an oval object the circle's residuals are dominated
by its own lack of fit — which *is the signal being tested for* — so the null was
inflated 1.5–2.3×, and only for elongated objects, exactly where power was
needed.

Fixed to use ellipse residuals alone. Measured at 300 replicates against a
nominal 0.05: circle-only residuals gave correct size but 0.45 power; pooled gave
0.063 size and 0.84 power; ellipse-only gives 0.073 size and **1.00** power. The
slight size inflation is the accepted cost. Both objects now return `elliptical`
(p = 0.007 and 0.016) and near-circular objects are unchanged. A test fails on
the old behaviour.

This is worth knowing because it is the failure mode to watch for generally: if a
p-value and a confidence interval built from the same data disagree, one of them
is being computed under the wrong model.
