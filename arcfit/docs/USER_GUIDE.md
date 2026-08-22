# Digitising 167 objects: a practical guide

## Before you start

Run the detection pre-pass first. It takes a few minutes and tells you which
photographs will need manual attention, so you meet those in a known batch
rather than as interruptions:

```bash
arcfit detect --images photos/ --workdir work/
```

It writes `work/detection_report.csv` with a confidence score per photograph.
A low score does not mean the detection is wrong — it means it is worth a look,
which costs one keystroke in the digitiser.

Check your scale card's real dimensions before anything else, and pass them to
*every* command. The built-in default is a uniform 10 × 2 cm card of 1 cm
squares, which is **not** the card used in this project.

The field card is **10 × 4 cm**: two rows of ten 1 cm squares against one row of
five 2 cm squares. Describe it with `--card-layout`:

```bash
arcfit detect --images photos/ --workdir work/ --card-layout 1x10,1x10,2x5
```

Each entry is `HEIGHTxCELLS` along the short side, so the layout also fixes the
card height (4 cm here) and `--card-height` becomes redundant.

Measurements are taken against the **checkered block** — the printed squares —
not the white border around them. The border's width is arbitrary and cannot be
recovered from a photograph, so it plays no part in the calibration.

Getting the height wrong does not merely rescale the results. It stretches the
rectified plane along one axis, which changes fitted **eccentricity** — the
thing the study is about — while lengths and figures still look plausible. There
is no downstream symptom, so measure the card with callipers once.

## The rhythm of one object

1. The photograph opens with the card outlined in blue, already detected.
   Glance at it. If the outline is not on the card, press `d` to retry or `m`
   to click the four corners yourself.
2. Zoom to the object with the toolbar. **Clicks are ignored while a pan or zoom
   tool is active**, so you cannot accidentally add points while navigating --
   the title line says `PAN ACTIVE - clicks ignored` whenever that is why
   nothing is happening. Click the same toolbar button again to leave the tool.

   The digitiser's own keys take priority while its window is open: matplotlib
   normally binds `p` to pan, `o` to zoom, `s` to save and `c`/`h` to its view
   history, and those are suppressed so they cannot fight the keys below.
3. Click along the surviving outline. Watch the dashed preview: it is the
   reconstruction updating live, and it is the fastest way to notice a bad point.
4. Check the live readout — 2a, 2b, e, and arc coverage.
5. Press `n`. The record saves automatically.

## How many points, and where

More points help, but *spread* helps far more than *density*. Twenty points
spanning the whole surviving arc beat sixty crowded into one end. The estimator
is limited by the angular span of the evidence, not by the number of clicks.

Thirty to forty points across the full preserved outline is a good target.

Place them on the same feature throughout — the widest visible silhouette. Being
consistent matters more than which feature you choose, because a systematic
offset is measurable against the calipers and a *variable* one is not.

## When the object is hard to see

Grey stone on grey sand is genuinely difficult. In order of usefulness:

- `g` cycles the display contrast — the second setting (contrast stretch) is
  usually the one.
- `e` overlays detected edges in orange. Very effective for finding where the
  outline actually is, but turn it off before clicking; it can bias you.
- Zoom in further than feels necessary. Snapping searches only ±6 pixels, so
  your click needs to be close for it to help.

If the outline genuinely cannot be seen, press `x` to exclude the object and
move on. An excluded object is recorded with its reason, not deleted.

## Snapping

Every click is pulled to the nearest strong image gradient along the local
normal, within a short radius. This reduces hand jitter without inventing an
outline.

It is deliberately conservative and will sometimes decline to move a point.
That is correct behaviour on a low-contrast stretch. Both the raw click and the
snapped position are stored, so the effect is auditable and reversible.

Press `s` to turn it off in cluttered spots where it is grabbing grass.

## Interruptions

Every object saves to its own file. Close the window whenever you like and
resume with:

```bash
arcfit digitize --images photos/ --workdir work/ --only-missing
```

which skips everything already finished.

## Repeatability

Digitise a random 20 objects a second time into a separate working directory,
then compare. This gives you an operator-repeatability figure for the paper and
takes about half an hour:

```bash
arcfit digitize --images photos/ --workdir work_repeat/ --operator BK
```

## Common problems

**"no interactive matplotlib backend"** — install Tk (`sudo apt install
python3-tk` on Linux) or `pip install PySide6`.

**Detection keeps failing on one photograph** — press `m` and click the four
corners of the checkered block, going round consistently. Any corner may be the
first: the tool identifies the long side itself from the four positions, so
starting on a short side is safe.

Automatic detection is unreliable on brightly lit sandy ground, where the white
card carries little contrast against the soil. On those photographs `m` is the
expected route rather than a fallback, and is no less accurate — the operator's
corners are refined the same way the detector's are.

**No scale card visible at all** — press `t` for a two-point scale against any
object of known length. This does **not** correct perspective, and the record is
flagged so those objects never silently mix with the rectified ones.

**The live 2a looks far too big** — usually a point placed on a shadow edge or
on the wrong side of the object. Right-click to undo. If the reconstruction is
much larger than the fragment, the analysis will flag it later via
`extrapolation_factor`, but catching it now is cheaper.
