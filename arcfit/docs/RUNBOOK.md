# Runbook: analysing the 167 mano photographs

A start-to-finish operational guide for the real run, on Windows, with the
photographs downloaded from Dropbox as a ZIP.

`README.md` explains what the tool does and why; `USER_GUIDE.md` covers
digitising technique. This covers the order to do things in, so that a mistake
surfaces in ten minutes rather than after seven hours of clicking.

**Assumed here:** scale card 10 × 2 cm, 1 cm squares in 2 rows. That is the
tool's default, so no card flags appear in any command below. If your card
differs, add `--card-width --card-height --card-cols --card-rows` to *every*
command — see step 4.

---

## Time budget

| Step | Time | Attended? |
|---|---|---|
| 0–3 Install and smoke test | 30–60 min | yes |
| 4 **Card geometry check** | 10 min | yes |
| 5 Batch card detection | 10–20 min | no |
| 6 **Pilot: 10 objects end to end** | 45 min | yes |
| 7 Caliper sheet | 1–2 h typing | yes |
| 8 Digitise all 167 | 5–8 h, split over sessions | yes |
| 9 Simulation study | 10–30 min | no (run overnight) |
| 10 Full analysis | ~15 min | no |
| 11 Repeatability re-digitise (20) | ~45 min | yes |
| 12 Package supplement | 5 min | no |

Steps 4 and 6 are checkpoints. Skipping them to save an hour risks discovering a
systematic error only after step 8.

---

## Step 0 — Prerequisites

Install **Python from python.org**, 3.9 or newer, ticking *"Add python.exe to
PATH"*. Prefer this over the Microsoft Store build: the python.org installer
bundles Tk, which the digitiser window needs.

```powershell
py --version
```

Free disk: about **8 GB** (2.5 GB zip, 2.5 GB extracted, ~1 GB outputs, headroom).

**A note on your shell.** Every command here is written for PowerShell, one per
line. If your shell banner says *"Install the latest PowerShell for new
features"* you are on Windows PowerShell 5.1, which does not support `&&`
between commands — run the lines separately rather than joining them. Nothing in
this guide requires PowerShell 7.

## Step 1 — Get the photographs out of Dropbox

In the browser, open the shared folder, then **⋯ → Download**. 167 files at
~15 MB each sits comfortably inside Dropbox's folder-download limits.

Extract to a **short path with no spaces, not inside Dropbox or OneDrive**:

```
C:\manos\photos\
```

Both parts matter. A folder a sync client is watching will have files moving
under the tool while it reads them, and Windows' 260-character path limit is
easy to hit once figure subfolders nest inside a `C:\Users\...\Dropbox\...`
path.

Verify before going further:

```powershell
(Get-ChildItem C:\manos\photos -Filter *.JPG).Count
Get-ChildItem C:\manos\photos | Measure-Object -Property Length -Sum
```

Expect **167** files totalling roughly 2.5 GB. A correct count with a tiny total
size means placeholder files rather than real ones.

> Uppercase `.JPG` is handled; the tool matches both cases.

## Step 2 — Install arcfit

> **Paste one line at a time, not the whole block.** Windows PowerShell 5.1 —
> the version that ships with Windows, and the one you have if the shell greets
> you with *"Install the latest PowerShell for new features"* — does **not**
> support `&&` between commands. It was added in PowerShell 7. Joining these
> with `&&` fails with *"The token '&&' is not a valid statement separator in
> this version"*. Run them as separate lines, or use `;` between them.

```powershell
mkdir C:\manos
cd C:\manos
git clone https://github.com/Brucekimosabeclone/Brucekimosabeclone.git code
cd code
git checkout claude/broken-object-ellipse-analysis-acud5i
cd arcfit

py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

`mkdir C:\manos` is harmless if step 1 already created it — but this step does
not depend on step 1, so it is here explicitly. If you already cloned somewhere
else, move it rather than cloning again:

```powershell
mkdir C:\manos
Move-Item C:\Users\<you>\code C:\manos\code
cd C:\manos\code
```

No git? Download the branch as a ZIP from the pull request page, extract to
`C:\manos\code`, then run the last four lines from `C:\manos\code\arcfit`.

Confirm:

```powershell
arcfit --version
```

**Every command below assumes `.venv\Scripts\activate` has been run in that
shell.** A new terminal means activating again.

## Step 3 — Smoke test on synthetic data

Proves the install works before any real data is involved:

```powershell
arcfit demo --outdir C:\manos\demo --n 8
arcfit analyze --workdir C:\manos\demo --images C:\manos\demo\images --measurements C:\manos\demo\measurements.csv --boot 500
```

You should get a summary table and files under `C:\manos\demo\output`. Open
`output\figures\objects\SIM_0001.png`; if that renders, the whole chain works.

`C:\manos\demo\ground_truth.csv` holds the values the synthetic scenes were
drawn from, so you can see what the method recovers when the answer is known.

## Step 4 — Confirm the card geometry (do not skip)

This is the one input that can silently invalidate the headline result. The
homography maps the four card corners onto a rectangle of the size you declare.
Declare 10 × 2 cm for a card that is really 10 × 1 cm and the rectified plane is
stretched two-fold in one direction — a perfectly circular mano then comes out
as a 2:1 ellipse, and nothing downstream will look obviously wrong.

Measure the card with calipers once. Then calibrate a small sample:

```powershell
mkdir C:\manos\sample
Get-ChildItem C:\manos\photos\*.JPG | Select-Object -First 8 | Copy-Item -Destination C:\manos\sample

arcfit detect --images C:\manos\sample --workdir C:\manos\work_sample
```

Open `C:\manos\work_sample\detection_report.csv`. Most rows should read `ok`
with a `score` above 0.55. `low` is not a failure — it means "confirm this one
by eye", which is a single keystroke in the digitiser.

The decisive visual check comes in step 6.

## Step 5 — Detect the card in all 167

```powershell
arcfit detect --images C:\manos\photos --workdir C:\manos\work
```

Each image is also hashed once here (~2.5 GB of reading, a minute or two from
local disk — one reason the ZIP route beats syncing). Results go to
`C:\manos\work\detection_report.csv`.

Expect a fair number of `low` scores. These shots are oblique, and past roughly
40° of tilt the detector reports lower confidence even where the corners are
accurate to well under a pixel. Count the outcomes:

```powershell
Import-Csv C:\manos\work\detection_report.csv | Group-Object status | Select-Object Name,Count
```

Only `none` and `bad` need manual calibration later.

## Step 6 — Pilot: ten objects end to end

The real checkpoint. Digitise ten objects, analyse them, and check four things
before committing to the full set.

```powershell
arcfit digitize --images C:\manos\sample --workdir C:\manos\work_sample --operator BK
```

Per object: glance at the blue card outline (`d` re-detects, `m` lets you click
the four corners), zoom in, click 30–40 points spread across the **whole**
surviving outline, press `n`.

Then enter rough caliper numbers for those ten and analyse:

```powershell
arcfit init-measurements --images C:\manos\sample --out C:\manos\work_sample\measurements.csv
notepad C:\manos\work_sample\measurements.csv
arcfit analyze --workdir C:\manos\work_sample --images C:\manos\sample --measurements C:\manos\work_sample\measurements.csv --boot 500
```

**Check all four:**

1. **Card scale.** Open any figure in `output\figures\objects\`. Panel *b* is
   drawn in real centimetre axes, and the scale card is visible in it. Its long
   side must measure **10 cm** against the axis ticks and its short side **2 cm**.
   If the short side reads 1 cm, stop: your card is single-row, and every
   command from here needs `--card-height 1 --card-rows 1`.
2. **Caliper agreement.** The console prints `bias ... cm (...%)`. A few percent
   is expected. Tens of percent points at a calibration problem, card geometry
   first.
3. **QC line.** Should read *"no reconstruction is shorter than its own
   fragment"*. A failure means points landed on the wrong feature.
4. **Plausible sizes.** `major_axis_cm` in `output\results.csv` should look like
   real manos, not 2 cm or 80 cm.

Only when all four look right, continue.

## Step 7 — Caliper measurement sheet

```powershell
arcfit init-measurements --images C:\manos\photos --out C:\manos\work\measurements.csv
```

Writes a CSV pre-filled with all 167 object ids, plus a
`measurements_README.txt` describing each column. Fill in:

- `fragment_max_cm` — longest dimension of the surviving piece
- `width_across_cm` — the fully preserved width across the break
- `width_is_at_widest` — `1` if that width was taken at the widest point, else `0`

Leave a cell **blank** where you could not measure. Blank is handled as missing;
a guess is not. Then validate:

```powershell
arcfit check-measurements --measurements C:\manos\work\measurements.csv --workdir C:\manos\work
```

Fix whatever it reports — especially duplicate ids, and any row where the width
exceeds the fragment maximum, which is almost always two columns swapped.

## Step 8 — Digitise all 167

```powershell
arcfit digitize --images C:\manos\photos --workdir C:\manos\work --operator BK --only-missing
```

`--only-missing` skips finished objects, so this is the command for every
session including the first. Each object saves the moment you press `n`; close
the window whenever you like.

Keys you will actually use: `n` next · right-click undo · `d` re-detect card ·
`m` manual four corners · `g` cycle contrast · `e` edge overlay · `s` snapping
on/off · `x` exclude · `q` save and quit.

Two points worth repeating from the user guide. **Spread beats density** — twenty
points across the whole surviving arc are worth more than sixty crowded into one
end, because the estimator is limited by the angular span of the evidence rather
than the number of clicks. And place points on the **same feature every time**
(the widest visible silhouette): a consistent offset is measurable against the
calipers, a variable one is not.

Progress at any time:

```powershell
(Get-ChildItem C:\manos\work\records\*.json).Count
```

## Step 9 — Simulation study

Run this while doing something else. It establishes what your arc coverage can
support and calibrates the reliability tiers:

```powershell
arcfit validate --out C:\manos\work\simulation_study.csv --replicates 300
```

It prints what a *truly circular* object's fitted eccentricity looks like at each
arc coverage and noise level. That table belongs in the paper — it is the
clearest justification for not thresholding eccentricity.

## Step 10 — Full analysis

```powershell
arcfit analyze --workdir C:\manos\work --images C:\manos\photos --measurements C:\manos\work\measurements.csv --simulation C:\manos\work\simulation_study.csv --outdir C:\manos\work\output
```

Roughly 15 minutes for 167 objects at 2000 bootstrap replicates including
per-object figures. Add `--no-object-figures` for fast iteration (~2 min) while
you are still checking things.

Read the console summary: shape counts, tier counts, caliper bias and limits of
agreement, QC failures, typology verdict.

Outputs in `C:\manos\work\output`:

| File | What it is |
|---|---|
| `results.csv` / `results.xlsx` | one row per object |
| `data_dictionary.csv` | every column defined, with units |
| `METHODS_DRAFT.md` | methods text generated from this run's own numbers |
| `config.json` | settings and seed, so the run reproduces exactly |
| `figures\` | five summary figures plus one per object |

## Step 11 — Repeatability (recommended)

Re-digitise about 20 objects into a *separate* working directory, for an
operator-repeatability figure:

```powershell
mkdir C:\manos\repeat_photos
Get-ChildItem C:\manos\photos\*.JPG | Get-Random -Count 20 | Copy-Item -Destination C:\manos\repeat_photos
arcfit digitize --images C:\manos\repeat_photos --workdir C:\manos\work_repeat --operator BK
arcfit analyze --workdir C:\manos\work_repeat --outdir C:\manos\work_repeat\output --no-object-figures
```

Compare `major_axis_cm` between the two `results.csv` files on the shared ids.

## Step 12 — Package the supplement

```powershell
arcfit package --workdir C:\manos\work --out C:\manos\supplement
```

Produces `C:\manos\supplement\` plus a zip: code, pinned dependency versions, the
raw digitisation records, the caliper sheet, results, figures, a runnable
synthetic demo with known answers, and a SHA256 manifest of every file.

`--include-images` adds the photographs and makes it very large. Most journals
would rather have the records, since the entire analysis regenerates from those
without the images.

---

## Troubleshooting

**`arcfit` is not recognised.** The virtualenv is not active. Run
`.venv\Scripts\activate`, or use `py -m arcfit.cli ...` instead.

**`no interactive matplotlib backend`.** Tk is missing. Reinstall Python from
python.org, or `pip install PySide6`.

**Activate fails with an execution-policy error.** In that PowerShell window:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate
again. Or use `cmd.exe` with `.venv\Scripts\activate.bat`.

**Paths with spaces.** Quote them: `--images "C:\My Photos\manos"`.

**Detection keeps failing on one photograph.** Press `m` and click the four card
corners, starting at one end of the long side and going round consistently — the
ordering is worked out for you. If the card is not visible at all, press `t` for
a two-point scale. That does **not** correct perspective, and such objects are
flagged in the results so they never silently mix with the rectified ones.

**Analysis feels slow.** `--jobs` defaults to all cores. `--boot 500` is fine
while iterating; do the final run at the 2000 default.

**Everything came out "indeterminate".** Usually too few points, or points
bunched at one end of the arc. Check `coverage_deg` and `n_points` in
`results.csv`. Note that "indeterminate" is a real answer — it means the
surviving arc is too short to distinguish a circle from an oval — so some of
these are expected and honest.

**A reconstruction looks far too large.** Check `extrapolation_factor` in
`results.csv`. Values well above 1.5 mean the fit is projecting far beyond
anything actually digitised, which is the known failure mode on short arcs.
Those objects are automatically demoted to tier C.
