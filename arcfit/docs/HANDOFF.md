# Handoff — `arcfit`

**Project:** reconstructing original mano dimensions from 167 field photographs
of broken fragments.
**Package version:** 1.1.0 · **Branch:** `claude/broken-object-ellipse-analysis-acud5i`
**Status:** calibration corrected against real photographs; ready to digitise.
**Card:** 10 x 4 cm -- always pass `--card-layout 1x10,1x10,2x5`.

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

**Complete and tested.** The demo pipeline runs end to end on synthetic data
with known ground truth.

**Documentation.** `QUICKSTART.md` (commands only), `RUNBOOK.md` (~500 lines,
full reasoning), `METHODS.md` (article text), `USER_GUIDE.md` (the digitising
GUI), this file. All ship inside the supplement zip.

**The real-image run has now happened (19-photograph pilot subset).** It failed
completely on the first attempt -- 0 of 19 confident, every row `low` -- because
the card model was wrong (see Risk 1, now resolved). The confidence gate held
and no bad calibration reached the data.

With the card declared correctly via `--card-layout 1x10,1x10,2x5`:

| Outcome | Count |
|---|---|
| Confident, verified on the card | 6 |
| Flagged `low` for the operator | 13 |
| Confident but wrong | 0 |

The six confident detections score 0.68-0.94 and were confirmed by eye to sit on
the card. The thirteen others are photographs on brightly lit sandy ground, where
the white card carries little contrast against the soil and no candidate scores
above threshold. **Press `m` on those** -- manual four-corner calibration is
exactly as accurate, since the operator's corners get the same subpixel
refinement the detector's do.

An earlier version of this run produced four *confident but wrong* detections on
specks of gravel a few pixels across. `detect_card` now discards any candidate
too small to verify (`min_cell_px`, default 6 px per printed cell in the
detector's working image). Below that, the per-cell means are noise and sign
agreement reaches threshold by chance. Genuine cards here run 25-30 px per cell
and the synthetic test card 10-12, so the floor is not tuned to either.

**Still not done.** The remaining photographs beyond these 19, and no object has
been digitised yet.

## 4. The two open risks

Both were spotted in the five sample photographs the author sent, and both must
be settled **before** digitising begins, because both corrupt data that would
then have to be re-collected.

### Risk 1 — RESOLVED: the card is 10 × 4 cm, not 10 × 2 cm

**Settled from the photographs (2026-08-22); still worth one calliper check.**

The checkered block is **10 × 4 cm**, aspect 2.5:1 — ten columns of 1 cm, and
three rows at heights 2:1:1 (two rows of ten 1 cm squares, one row of five 2 cm
squares). Measured on `IMG_5308` and `IMG_5540`. The square *count* is the
strong evidence: counting squares is independent of perspective. Rectifying to
an assumed 10 × 4 then returns cell widths of 0.97 and 1.97 cm — clean
centimetre multiples — and printed cells are square, so the row heights are
1 cm and 2 cm and the block is 4 cm tall.

Declare it with `--card-layout 1x10,1x10,2x5` on **every** command. Measure the
checkered block, not the white border: the border is arbitrary and unrecoverable
from a photograph.

The earlier guess in this document — "roughly 3.5–4:1, maybe 10 × 2.5 cm" — was
an under-estimate of the error. The real discrepancy is a factor of two.

This matters more than any other single number. A wrong declared height stretches
the rectified plane along one axis only. That is a shear applied to every
outline, so it changes the fitted **eccentricity** — the study's headline
variable — while lengths, figures, and diagnostics all still look entirely
plausible. There is no downstream symptom.

*Resolution:* callipers on the card, then `arcfit detect`, which measures the
imaged card's aspect against the declared one and prints a **"Card geometry"**
line. Do not digitise past a warning.

Note that this check was **inoperative on these photographs** until
2026-08-22. It needs camera intrinsics, and `intrinsics_from_exif` required
`FocalLengthIn35mmFilm` — a tag Canon DSLRs do not write, so it returned `None`
for all 19 files and the check silently never ran. It now falls back to
`FocalLength` with the sensor width derived from `FocalPlaneXResolution`, and
recovers intrinsics for 19/19 (f35 ≈ 27 mm on the 5D Mark II).

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

1. **Calliper the card**, to confirm the 10 x 4 cm read off the photographs.
   Measure the checkered block, not the white border.
2. **`arcfit detect`** on everything, with `--card-layout 1x10,1x10,2x5`. Read
   the "Card geometry" line -- it works now. Expect roughly a third confident
   and the rest flagged; that is the bright-ground contrast problem, not a
   regression. Do not continue past a card-geometry warning.
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
- **The card size must be passed to *every* command**, not just `detect`. It
  does differ from the 10 × 2 cm default: use `--card-layout 1x10,1x10,2x5`.
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

## 11. A second bug, on the manual-calibration path

Pressing `m` and clicking the card's four corners fed those clicks straight into
the homography, which maps the first click to the origin and the second to
`(width, 0)` **literally**. Starting the four clicks on a short side therefore
assigned the card's 10 cm dimension to its 4 cm side.

The consequence is not a wrong scale but a wrong *shape*: the rectified plane is
stretched by the aspect ratio along one axis, so a true circle came out with an
axis ratio of 25 on a 10 x 2 cm card, and 6.25 on the real 10 x 4 cm one. Two of
the four possible starting corners produced this; the other two were correct.

`USER_GUIDE.md` had always promised the tool worked the ordering out itself, and
`order_card_corners` does exactly that -- but it was only ever wired into the
automatic path. It is now used for manual clicks too.

This one is worth remembering for the same reason as the bug above: the card
cannot reveal it. The homography is built from the declared numbers, so
rectifying the card's own corners reproduces 10 x 4 cm whichever way round the
clicks went in. Only an independent shape shows the distortion, which is what
the regression test uses.
