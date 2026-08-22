# Quickstart — the commands, in order

Just the sequence. `RUNBOOK.md` has the reasoning behind each step and a fuller
troubleshooting section.

Assumes photographs in `C:\manos\photos` and the clone at `C:\manos\code`.

> **Run one line at a time.** Windows PowerShell 5.1 does not support `&&`
> between commands.

---

## 0. Update and activate

```powershell
cd C:\manos\code
git pull
cd arcfit
.venv\Scripts\activate
pip install -e .
arcfit --version
```

**Check:** reports `1.1.0` or later. Every step below needs this shell with the
virtualenv active — a new terminal means activating again.

## 1. Measure the scale card

Put callipers on the card. Note its long side, its short side, and the size of
one square.

**This is the one number that can invalidate the study.** A wrong declared
height stretches the rectified plane in one direction and corrupts eccentricity,
while everything else still looks fine.

The built-in default is a uniform **10 × 2 cm** card of 1 cm squares. The card
used in this project is **not** that: it is **10 × 4 cm** — two rows of ten 1 cm
squares against one row of five 2 cm squares. Add

    --card-layout 1x10,1x10,2x5

to **every** command below. Entries are `HEIGHTxCELLS` along the short side, so
the layout sets the card height too. Measure the **checkered block**, not the
white border. For a uniform card use `--card-width --card-height --card-cols
--card-rows` instead.

## 2. Locate the card in all 167

```powershell
arcfit detect --images C:\manos\photos --workdir C:\manos\work
```

**Check the "Card geometry" line.** It measures the card's real proportions from
the photographs and compares them with what you declared. If it warns, stop,
re-measure, and re-run with the right `--card-height`. Do not continue past a
warning.

Some `low` scores are normal — they mean "confirm by eye", one keystroke later.
Only `none` and `bad` need manual work.

## 3. Pilot: ten objects, end to end

```powershell
mkdir C:\manos\sample
Get-ChildItem C:\manos\photos\*.JPG | Select-Object -First 10 | Copy-Item -Destination C:\manos\sample
arcfit digitize --images C:\manos\sample --workdir C:\manos\work_sample --operator BK
```

Per object: check the blue card outline (`d` re-detect, `m` click the corners
yourself), press **`o` if the card is resting on the object rather than the
ground**, zoom in, click 30–40 points across the whole surviving edge, press `n`.

Then, with rough calliper numbers for those ten:

```powershell
arcfit init-measurements --images C:\manos\sample --out C:\manos\work_sample\measurements.csv
notepad C:\manos\work_sample\measurements.csv
arcfit analyze --workdir C:\manos\work_sample --images C:\manos\sample --measurements C:\manos\work_sample\measurements.csv --boot 500
```

**Check all four:**

1. In `output\figures\objects\*.png`, panel *b* — the card's long side measures
   its true length against the centimetre axes.
2. Console `bias` is a few percent, not tens.
3. Console says no reconstruction is shorter than its own fragment.
4. `major_axis_cm` in `output\results.csv` looks like real manos.

Only continue when all four are right. Forty-five minutes here protects seven
hours later.

## 4. Calliper sheet

```powershell
arcfit init-measurements --images C:\manos\photos --out C:\manos\work\measurements.csv
```

Fill in `fragment_max_cm`, `width_across_cm`, and `width_is_at_widest` (1 or 0).
Leave a cell blank where you could not measure — blank is handled as missing, a
guess is not.

```powershell
arcfit check-measurements --measurements C:\manos\work\measurements.csv --workdir C:\manos\work
```

**Check:** fix duplicates, and any row where the width exceeds the fragment
maximum (almost always two columns swapped).

## 5. Digitise all 167

```powershell
arcfit digitize --images C:\manos\photos --workdir C:\manos\work --operator BK --only-missing
```

The same command every session — `--only-missing` skips what is already done.
Saves on every `n`, so close the window whenever. Budget 5–8 hours in total.

Keys: `n` next · right-click undo · `d` re-detect card · `m` manual corners ·
`o` card-on-object · `g` cycle contrast · `e` edge overlay · `s` snapping ·
`x` exclude · `q` save and quit.

Progress at any time:

```powershell
(Get-ChildItem C:\manos\work\records\*.json).Count
```

## 6. Simulation study

Run it while doing something else.

```powershell
arcfit validate --out C:\manos\work\simulation_study.csv --replicates 300
```

## 7. Full analysis

```powershell
arcfit analyze --workdir C:\manos\work --images C:\manos\photos --measurements C:\manos\work\measurements.csv --simulation C:\manos\work\simulation_study.csv --outdir C:\manos\work\output
```

About 15 minutes. Read the console summary — shape counts, tiers, calliper bias
and limits of agreement, QC failures, typology verdict.

Outputs in `C:\manos\work\output`:

| File | What it is |
|---|---|
| `table_s1.csv` | **the article table** — major axis, minor axis and eccentricity, each with 95% CI, plus the circular/elliptical call |
| `results.csv` / `results.xlsx` | the full table; the workbook opens on Table S1 |
| `data_dictionary.csv` | every column defined, with units |
| `METHODS_DRAFT.md` | methods text generated from this run's own numbers |
| `figures\` | five summary figures plus one per object, 600 dpi and vector |

## 8. Package and share

```powershell
arcfit package --workdir C:\manos\work --out C:\manos\supplement --lite
Copy-Item C:\manos\supplement.zip "$env:USERPROFILE\Dropbox\manos_supplement.zip"
```

`--lite` leaves out the per-object figures — about 380 MB that regenerate from
the records — giving roughly 10–15 MB instead of 450. Build on local disk and
then copy into the synced folder, so the sync client is not uploading files
while the packager is still writing them.

---

## If something breaks

| Symptom | Cause |
|---|---|
| `arcfit` is not recognised | virtualenv not active — run `.venv\Scripts\activate` |
| `py` is not recognised | use `python` instead, or install from python.org and **reopen** the shell |
| `'&&' is not a valid statement separator` | PowerShell 5.1 — run one line at a time |
| `no interactive matplotlib backend` | `pip install PySide6` |
| `the module '.venv' could not be loaded` | the venv was never created; fix the line above it |
| everything came out "indeterminate" | too few points, or points bunched at one end of the arc |
| a reconstruction looks far too large | check `extrapolation_factor` in `results.csv` |

Anything not covered here is in `RUNBOOK.md`.
