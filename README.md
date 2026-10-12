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

The pharmacokinetic and bacterial layers were calibrated first, against
published concentration-time, time-kill and neutropenic-thigh
dose-fractionation data, and were not subsequently adjusted, so no parameter
in them is tuned to a danger-signal result.

## Installation

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10 or newer. The core dependencies are NumPy, SciPy, pandas,
Matplotlib, PyYAML and SALib.

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

Two further entry points cover the layers underneath:

```bash
python scripts/model_qualification.py      # PK/PD index qualification
python scripts/run_external_validation.py  # curve-level comparison to published data
```

Figures are always generated from the stored results rather than recomputed,
so a figure and the number in the text cannot drift apart.

## Tests

```bash
python -m pytest -q
```

474 tests, covering the pharmacokinetic conventions, the bacterial dynamics,
mass conservation across the persister and small-colony-variant switches, the
danger-signal layer, the held-out validation data, the tolerance term, and the
acceptance criteria. One test deliberately rejects a saturated danger-signal
pathway, so the criterion that a predecessor model failed can still fail here.

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

Two parameters are not identified by the available data and are reported as
such: the infection focus volume and the fraction of locally released signal
that reaches the circulation. They move absolute levels by about two orders of
magnitude. Results are therefore reported as rankings and as intervals over
the measured yield range, never as absolute magnitudes or point estimates.

## Licence

MIT. See `LICENSE`.
