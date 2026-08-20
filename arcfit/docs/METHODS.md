# Methods

A ready-to-edit methods section is **generated from each run** at
`output/METHODS_DRAFT.md`, so the numbers in the prose always match the numbers
in the table. This file records the reasoning behind the choices it describes.

## Why rectification, not a scale factor

The photographs are oblique. Under perspective projection a circle in the ground
plane images as an ellipse, and no single pixels-per-centimetre factor can undo
that — the distortion is not a scaling. Measuring in pixels and converting would
therefore produce eccentricity that is partly a record of the camera position.

Four coplanar points of known geometry determine a homography from image pixels
to the ground plane, and the scale card supplies them. All measurement happens
after that mapping, in centimetres.

The tool verifies this on synthetic scenes: a true circle photographed at a 31°
tilt fits with eccentricity above 0.3 in raw pixels and below 0.0001 after
rectification.

## Why orthogonal-distance fitting

Algebraic ellipse fitting minimises a quantity with no geometric meaning and is
biased towards small, low-eccentricity ellipses. That bias falls directly on the
quantity of interest, so the algebraic fit is used only to seed a geometric fit
that minimises true perpendicular distances — the maximum-likelihood estimator
under isotropic point noise.

## Why the circularity decision is a model comparison

Eccentricity is bounded below by zero. Measurement noise can therefore only
raise it, and the shorter the surviving arc the more it raises it. The tool's own
simulations put the median fitted eccentricity of a *true circle* at 0.19 for a
180° arc, 0.28 at 140°, and 0.43 at 100°.

Any threshold rule inherits that bias. Comparing the five-parameter ellipse to a
three-parameter circle against a null simulated at the object's own arc coverage
and noise does not, because the null carries the same inflation. Simulation
confirms the test holds its nominal 5% error rate at 100°, 140° and 180°.

The three-way outcome matters. Failing to reject a circle is not evidence of
circularity when the arc is too short to have detected an ellipse, so "circular"
additionally requires the interval to exclude meaningful elongation, and objects
meeting neither condition are reported as indeterminate.

## Why a residual bootstrap

Resampling points would vary the arc coverage between replicates, and arc
coverage is the main determinant of how precisely an ellipse can be recovered.
Holding the geometry fixed and resampling orthogonal residuals keeps each
replicate an honest repeat of the same measurement. Residuals are inflated by
sqrt(n/(n−p)) before resampling, since fitted residuals understate the true
errors.

## Why eccentricity is reported alongside the axis ratio

Near a circle, eccentricity is ill-conditioned: de/d(b/a) diverges as b/a → 1. A
2–3% error in the axis ratio can move eccentricity by tens of percent. The axis
ratio is the better-behaved measure and both are reported.

## Parallax

The digitised outline sits above the plane of the scale card. Rectifying it with
the ground-plane homography recovers where each camera ray crosses the ground —
a central projection between parallel planes, which is a homothety about the
camera's nadir with ratio H/(H−h).

Being a *uniform* scaling, it maps an ellipse to a similar ellipse: axis lengths
inflate by roughly 3% at standing height, while eccentricity and orientation are
unchanged. The circular-versus-oval conclusion is therefore unaffected. Axis
lengths are reported uncorrected and the bias is quantified empirically against
the caliper measurements. Verified numerically in `tests/test_calibrate.py`.

## References

Bland, J.M. & Altman, D.G. (1986) Statistical methods for assessing agreement
between two methods of clinical measurement. *Lancet* 1, 307–310.

Eberly, D. *Distance from a Point to an Ellipse, an Ellipsoid, or a
Hyperellipsoid.* Geometric Tools.

Efron, B. (1987) Better bootstrap confidence intervals. *JASA* 82, 171–185.

Fitzgibbon, A., Pilu, M. & Fisher, R.B. (1999) Direct least square fitting of
ellipses. *IEEE TPAMI* 21, 476–480.

Halíř, R. & Flusser, J. (1998) Numerically stable direct least squares fitting
of ellipses. *Proceedings of WSCG* 6, 125–132.

Hartley, R. & Zisserman, A. (2003) *Multiple View Geometry in Computer Vision*,
2nd edn. Cambridge University Press.

Pratt, V. (1987) Direct least-squares fitting of algebraic surfaces.
*ACM SIGGRAPH Computer Graphics* 21, 145–152.

Silverman, B.W. (1981) Using kernel density estimates to investigate
multimodality. *JRSS B* 43, 97–99.
