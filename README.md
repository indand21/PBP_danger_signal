# PBP_danger_signal

A within-host quantitative systems pharmacology model of the danger signal
(free endotoxin and other pathogen-associated molecular patterns) liberated
during antibiotic killing of Gram-negative bacteria, with an agent-specific
release yield tied to penicillin-binding protein (PBP) specificity.

This repository holds the model source, the digitized input data with their
provenance, the analysis pipeline, and the stored results from which every
reported number and figure is generated. It accompanies a manuscript in
preparation; the manuscript itself is not part of this repository.

## What the model contains

| Layer | Content |
|---|---|
| Pharmacokinetic | two-compartment, analytic effect-site concentration from a dosing history |
| Bacterial | replicating, persister and small-colony-variant pools with immune effectors and saturating concentration-dependent killing |
| Danger signal | release proportional to drug-induced killing with an agent-specific per-CFU yield, plus a kill-independent term confined to a smooth sub-MIC window; first-order plasma clearance |
| Cytokine | interleukin-6 and tumour necrosis factor downstream of receptor occupancy. Calibrated, but **computed and not reported** - it does not transfer from healthy volunteers to septic patients |
| Injury | used only to test whether a ranking is invariant across seven monotone maps; no injury magnitude is ever reported |

No parameter in the pharmacokinetic or bacterial layer is tuned to a
danger-signal result. Both were fixed before the danger-signal layer was
built, and while the bacterial layer was later refitted (see below), its
objective is a root-mean-square error in log10 CFU and contains no
danger-signal quantity. Their calibration status is not uniform, though, and
an earlier version of this file overstated it:

- The **pharmacokinetic** layer is fitted to published concentration-time
  data. Ceftazidime's resulting half-life, 1.18 h, is shorter than the
  reported 1.6-2.0 h; that is a consequence of the one-compartment
  simplification this study uses.
- The **bacterial** layer's maximum kill rate is fitted to a meropenem
  time-kill curve and a neutropenic-thigh dose-response
  (`scripts/ds_bacterial_refit.py`, written to
  `results/danger_signal/bacterial_refit.json`). The half-maximal
  concentration is **anchored rather than fitted**: killing depends on
  concentration only through the ratio `C/kill_C50`, so MIC-normalised data
  cannot identify it, and it is instead set so that the model's broth stasis
  concentration equals the calibration isolate's measured MIC of 1 mg/L. The
  Hill exponent is not identified either - the fit prefers 8 and cannot
  separate that from the 4 in use - and that one choice decides the residual
  kill difference between the two agents: 0.10 log10 CFU at 8 against the
  0.24 reported at 4. The larger is reported.
- The **immune** layer's recruitment term is not on the effector state's
  scale, so it behaves as a decaying pool rather than a recruited response.
  This is the largest remaining structural defect, and it matters more at the
  refitted kill rate than it did before: slower drug killing leaves immune
  killing a larger share of the total, so a mis-scaled immune layer has more
  leverage.

`scripts/ds_structural_sensitivity.py` recomputes the whole analysis under
six variants - the study as reported, the parameters it ran on before the
bacterial layer was refitted, the refit's own formal optimum, and three
immune-layer extremes - and writes the comparison to
`results/danger_signal/structural_sensitivity.json`, so what does and does
not depend on these choices can be read off directly rather than taken on
trust. The agent ranking holds in every one.

Measuring the half-maximal concentration rather than anchoring it is the
outstanding piece of work on the bacterial layer: it needs a time-kill
exposure near that concentration, where the Hill function is sensitive to
it. `data/external_validation/WPD_SETUP.md` holds the digitisation protocol
for the figure such a curve would come from.

## Installation

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements-lock.txt   # to reproduce the committed results
pip install -r requirements.txt        # or just to run the model
```

Python 3.10 or newer. The core dependencies are NumPy, SciPy, pandas,
Matplotlib, PyYAML, SALib and packaging. `requirements-lock.txt` pins the
versions the committed results were generated and verified with (NumPy 2.4.6
with SciPy 1.15.3, under Python 3.12.4); `requirements.txt` states the
supported ranges.

## Reproducing the analysis

Each stage writes a JSON file to `results/danger_signal/`. The files are
committed, so the figures can be regenerated without rerunning the pipeline.

```bash
python scripts/ds_pilot_acceptance.py    # feasibility and acceptance screen
python scripts/ds_phase2_build.py        # calibrate the danger-signal layer
python scripts/ds_phase3_validate.py     # held-out validation, scale invariance
python scripts/ds_phase4_analyses.py     # yields, matched exposure, cost indices
python scripts/ds_phase5_tolerance.py    # endotoxin-tolerance term (ships disabled)
python scripts/ds_phase4_figures.py      # figures, read back from the stored JSON
```

Three further stages write their own JSON alongside the pipeline's without
feeding into it:

```bash
python scripts/ds_numerical_assurance.py       # analytical benchmark, tolerance study
python scripts/ds_mass_balance_convergence.py  # order of the mass-balance residual
python scripts/ds_structural_sensitivity.py    # kill-parameter and immune-layer variants
python scripts/ds_bacterial_refit.py           # bacterial-layer fit and identifiability
```

`ds_bacterial_refit.py` reports the fit the bacterial parameters come from
and the profiles behind the identifiability statements above; it changes no
parameter itself. Its objective and its profiles reproduce, but its argmin
does not: the surface is flat enough that changing only the NumPy version
relocates the optimum while the objective moves under 1e-3. The argmin's
coordinates are therefore not quoted anywhere, and the parameters in use are
not read off it - `argmin_stability` in its output records this.

Two further entry points cover the layers underneath:

```bash
python scripts/model_qualification.py      # PK/PD index qualification
python scripts/run_external_validation.py  # curve-level comparison to published data
```

Figures are always generated from the stored results rather than recomputed,
so a figure and the number in the text cannot drift apart.

Every results file carries a `provenance` block recording the code revision,
the Python, NumPy and SciPy versions, the yield configuration, the reported
endpoint and the solver settings it was produced with. It holds no timestamp,
so an unchanged rerun leaves the file byte-identical. The `git_head` it names
is the revision of the study repository these files were generated in, not a
commit in this one; the payload itself regenerates unchanged here. The
pre-registration and the phase reports that some script docstrings cite
(`docs/danger_signal_study/`) accompany the manuscript and are not part of
this repository.

## Tests

```bash
python -m pytest -q
```

532 tests pass, covering the pharmacokinetic conventions and the conservation
of drug mass, the bacterial dynamics, mass conservation across the persister
and small-colony-variant switches, the danger-signal layer and its
mass-balance identity, the unbound-fraction convention, the held-out
validation data, solver failure handling, the provenance block, the tolerance
term, and the acceptance criteria. One test deliberately rejects a saturated
danger-signal pathway, so the criterion that a predecessor model failed can
still fail here.

One further test is a deliberate strict `xfail`. It pins the non-conserving
legacy equations in `pk_models.ode_rhs`, which the combination- and
sequential-therapy modules still drive killing from. Those modules are
outside this study, whose own simulations no longer integrate those
equations, and the strict marker means the defect can be neither silently
fixed nor silently spread.

## Data and provenance

`data/danger_signal/` holds values digitized from published figures and
tables, and `registry.json` records, for each source, the quantities
extracted and the values stated in the published abstract against which the
extraction was checked. `data/external_validation/` holds the
concentration-time, time-kill and dose-fractionation curves used for the
layers below the danger signal, with their own registry.

No copyrighted source PDF is included in this repository, and no new
experimental data were generated. Every dataset is published and cited in the
accompanying manuscript.

## Uncalibrated parameters

Two parameters of the danger-signal layer are not identified by the available
data and are reported as such: the infection focus volume and the fraction of
locally released signal that reaches the circulation. They move absolute
levels by about two orders of magnitude. Results are therefore reported as
rankings and as ranges, never as absolute magnitudes or point estimates. The
bacterial layer's half-maximal concentration and Hill exponent are not
identified either, for the separate reasons given above.

The per-CFU yields themselves come from three published studies that measured
different estimands, so the range they span is an **assumed scenario range**,
not a measured confidence interval, and is treated as one throughout. See the
`final_yields_fg_per_CFU` entry in any results file's `provenance` block for
the configuration in force, and `src/analysis/danger_signal.py` for how it was
set.

## Licence

MIT. See `LICENSE`.
