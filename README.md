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
| Pharmacokinetic | two-compartment code path, one-compartment as used: the simulation reads clearance, central volume and the penetration ratio only, so the peripheral parameters never enter |
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
  data where such a curve has been digitized, which is true for meropenem and
  not for ceftazidime. Meropenem's clearance and central volume are fitted to
  the digitized 1 g plasma curve and meet all three of that dataset's
  acceptance targets. Ceftazidime's central volume is **anchored, not
  fitted**: it is set so the half-life equals the 1.57 h measured by Leroy,
  which puts the volume on that paper's independently measured value without
  being made to. That is a weaker basis, and the manuscript's supplement says
  so.
- The **bacterial** layer's maximum kill rate is fitted to a meropenem
  time-kill curve and a neutropenic-thigh dose-response
  (`scripts/ds_bacterial_refit.py`, written to
  `results/danger_signal/bacterial_refit.json`). The half-maximal
  concentration is **anchored rather than fitted**: killing depends on
  concentration only through the ratio `C/kill_C50`, so expressing an exposure
  in multiples of the model's own effective MIC cancels it exactly - the
  objective moves 5e-10 log10 over a 120-fold range, which is zero to machine
  precision - and MIC-normalised data therefore cannot identify it. It is
  instead set so that the model's broth stasis
  concentration equals the calibration isolate's measured MIC of 1 mg/L. The
  Hill exponent is not identified either - the fit prefers 8 and cannot
  separate that from the 4 in use - and that one choice decides the residual
  kill difference between the two agents: 0.03 log10 CFU at 8 against the
  0.07 reported at 4. The larger is reported.
- The **immune** layer is illustrative, and nothing here identifies it: both
  datasets above are immune-free by construction, one a broth culture and
  one a *neutropenic* animal model. Its recruitment term was also
  dimensionally wrong until 2026-10-13 - a rate constant in 1/h where the
  equation needs cells/mL/h, six orders of magnitude too small, so the pool
  could only decay - and it is now homeostatic and recruited by bacterial
  burden, with production on the effector state's own scale. What the repair
  cannot supply is the **size** of the response. That plateau is an
  assumption, bounded below by homeostasis without recruitment and above by
  the requirement that an untreated infection still progress; the default
  sits mid-band, and the sensitivity run below covers both ends.

`scripts/ds_structural_sensitivity.py` recomputes the whole analysis under
seven variants - the study as reported, the parameters it ran on before the
bacterial layer was refitted, the refit's own formal optimum, and four
immune structures spanning no immune killing at all to the top of the
recruited band - and writes the comparison to
`results/danger_signal/structural_sensitivity.json`, so what does and does
not depend on these choices can be read off directly rather than taken on
trust. The agent ranking holds in every one. Across the whole admissible
band of the recruited plateau the cost ratio tops out at 8.2-8.8 and the
kill difference spans 0.05-0.14 log10, so that choice is not load-bearing;
removing immune killing altogether, the neutropenic limit rather than a
plausible host, is what moves them, to 10.8 and 0.31.

The effective MIC is resolved with the effector pool switched off, so it is a
property of the drug and the organism rather than of the host, and it is
identical across every immune variant. Setting `N_eff_0 = 0` is what imposes
that condition: it zeroes the homeostatic target and the recruited increment
together. A zeroed initial condition alone is **not** enough, since
homeostasis refills it - see `immune_free_parameters` in
`src/core/parameters.py` and the tests in `tests/test_immune_layer.py`.

The three places that compute an effective MIC now agree to 0.003%. One of
them used to interpolate the stasis crossing on a 25-point grid and was about
4% high; because the kill function is steep near stasis, holding that
concentration for 24 h gave a net change of -0.52 log10 CFU rather than the
zero the quantity is defined as. It now solves continuously, and the test in
`tests/test_external_validation.py` checks the returned value against the
definition rather than against a stored number.

The stasis threshold depends on the host, so both are recorded under
`stasis_thresholds` in `results/danger_signal/phase4_analyses.json`:
about 22% fT>MIC (ceftazidime) and 21% (meropenem) in an immunocompetent
host, and 34% and 32% with the effector pool off, which is the condition the
published neutropenic-animal thresholds were measured in.

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

Seven further stages write their own JSON alongside the pipeline's without
feeding into it:

```bash
python scripts/ds_numerical_assurance.py       # analytical benchmark, tolerance study
python scripts/ds_mass_balance_convergence.py  # order of the mass-balance residual
python scripts/ds_structural_sensitivity.py    # kill-parameter and immune-layer variants
python scripts/ds_bacterial_refit.py           # bacterial-layer fit and identifiability
python scripts/ds_validation_targets.py        # curve-level validation vs registry targets
python scripts/ds_pk_refit.py                  # pharmacokinetic fit (adopted)
python scripts/ds_multiarm_refit.py            # seven-arm C50 refit from one figure
python scripts/ds_multiarm_downstream.py       # what that refit would do downstream
```

`ds_validation_targets.py` scores the digitized curves against the numeric
acceptance targets recorded in `data/external_validation/registry.json`, which
were previously computed by nothing. Both curves it scores are registered as
**calibration**, not validation: the adopted parameters were estimated from
them, so meeting a target there measures fit, not prediction, and a test
enforces that the dataset a fit was estimated from cannot be registered as
held out. **One of those targets is still not
met** and the script says so rather than failing quietly: the neutropenic-thigh
stasis threshold, which the model places at 37.1% fT>MIC against a published
17.3%.

`ds_pk_refit.py` fits the two identifiable pharmacokinetic parameters to the
digitized plasma curve and meets all three of its targets. **Its fit has now
been adopted** as the meropenem default (`CL` 12.157 L/h, `Vc` 0.270 L/kg),
which is why those three targets are met above; ceftazidime, which has no
digitized curve here, is instead anchored to a published half-life. Running
it also shows that the thigh threshold is *not* a pharmacokinetic problem —
the refit leaves it where it was — so the two failures have different causes,
one in the PK layer and one in the bacterial layer.

`ds_multiarm_refit.py` refits the kill-potency parameter to all seven
concentration arms of a single published time-kill figure, read from the
raster images shipped in `data/external_validation/source_figures/`, and
`ds_multiarm_downstream.py` reports what adopting that fit would change.
Neither changes a default. The primary fit is dominated by two 2 h readings
that disagree with each other; dropping that pair gives the best residual of
any scenario and a potency within 2.2% of the one in use, by a route that
uses no MIC anchor. The refit is treated as corroboration, not as a
replacement, and the Hill exponent stays unidentified across its scenarios.

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

559 tests pass, covering the pharmacokinetic conventions and the conservation
of drug mass, the bacterial dynamics, mass conservation across the persister
and small-colony-variant switches, the danger-signal layer and its
mass-balance identity, the immune layer's units, fixed points and
immune-free invariant, the unbound-fraction convention, the held-out
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
accompanying manuscript. One exception runs the other way:
`data/external_validation/source_figures/` ships two raster renderings of the
published figure the seven-arm time-kill digitisation reads its coordinates
from. That paper is open access under CC BY 4.0, so redistributing the images
with attribution is permitted, and a digitisation whose source image is absent
cannot be checked by a reader. The folder's own README carries the citation
and the licence.

## Uncalibrated parameters

Two parameters of the danger-signal layer are not identified by the available
data and are reported as such: the infection focus volume and the fraction of
locally released signal that reaches the circulation. They move absolute
levels by about two orders of magnitude. Results are therefore reported as
rankings and as ranges, never as absolute magnitudes or point estimates. The
bacterial layer's half-maximal concentration and Hill exponent are not
identified either, nor is the size of the immune response, for the separate
reasons given above.

The nominal per-CFU coefficients are informed by studies that measured
different estimands, so the range used is an **assumed scenario range**,
not a measured confidence interval, and is treated as one throughout. See the
`final_yields_fg_per_CFU` entry in any results file's `provenance` block for
the configuration in force, and `src/analysis/danger_signal.py` for how it was
set.

Their **absolute** values are a scale convention rather than a measurement,
and the full text of the source that anchors them says why. It inoculates at
1e6 CFU/mL, not the 1e8 the conversion assumed; over the density actually
present its 493 ng/mL works out at some hundreds of fg per CFU, which is more
lipopolysaccharide than one cell contains, and its untreated control already
carries about two thirds of that concentration at the same timepoint. So that
experiment does not identify a per-CFU mass at all. It does identify the
ordering of four agents, which is what this model takes from it. Nothing
computed here is an absolute magnitude, so the scale cancels; the findings
are recorded against the dataset in `data/danger_signal/registry.json`.

One source reports viable counts beside the LPS endpoint, which is what lets
the agent contrast be read at a matched bacterial outcome rather than at equal
MIC multiple. Three such pairs give ceftazidime:carbapenem ratios of 3.2, 11.6
and 6.5, bracketing the assumed 3.5-7 range, which is not refitted to them.
What is matched is the NET change in viable count, which is NOT cumulative
drug-induced killing: that is the integral of the kill flux and cannot be
recovered from a net count, and in one of the three pairs BOTH ARMS GREW.
`net_change_matched_ratio` in `src/analysis/yield_observations.py` is named for
what it does and refuses to form a ratio unless both observations carry counts
and their net change agrees within a tolerance.

Regimen-level release modelling is not new: Thorsted et al. (Antimicrob Agents
Chemother 2020;64:e02070-19) fitted growth, killing and filament formation to
cefuroxime endotoxin release under static and dynamic exposure. They estimate
0.00636 EU per antibiotic-killed cell and 0.295 per killed filament, 46x the
susceptible value, so the per-CFU coefficients this project treats as a scale
convention are unidentified by ITS sources rather than unidentifiable. Their
regimen predictions also reach this project's exposure-response conclusion
independently: release is lowest when exposure clears the organism rapidly.

The same source undercuts a structural assumption of the model. The
kill-independent release term is given to the PBP-3 agent alone, but
carbapenems filament below the MIC and release heavily there: 10x to 0.1x MIC
raises free LPS 88-fold for doripenem and 48-fold for meropenem, against
5.7-fold for ceftazidime. `scripts/ds_filament_structure.py` sweeps a
meropenem filament coefficient from zero to the full ceftazidime value: the
matched-exposure cost ratio moves from 5.07-6.51 to 4.26-5.04 and the 0.5x MIC
ratio collapses from 35.2 to 1.13, while the ordering survives throughout. No
coefficient is estimated and no default changes. Below the MIC this assumption
is load-bearing, and the evidence for it is weak.

## PD reassessment on 14 October 2026

The thigh stasis target remains unmet: the historical diagnostic is 37.1%
against 17.3% published. A new fixed-MIC, strain-specific context analysis
uses untreated growth over the actual 10 h interval and gives a midpoint
mean of 22.2% with a 14.6-30.6% growth envelope. This is a transport sensitivity,
not validated murine calibration; paired controls and regimen-linked counts
are missing.

The first Figure 2B/C broth windows are now included in a joint concentration
response check. Its best h is 1.97, but residuals remain too large and the
concentration ordering reverses in one arm. No PD default is replaced.

Source observations use their actual point or sampling-mean operators.
Isepamicin's 2/4/6 h mean is not compared as a yield against 4 h beta-lactam
points. The direct CAZ:MERO ELISA endpoint ratio is 4.77; it does not measure
latent release per killed cell. Joint fixed-MIC stress tests include ten PD
scenarios, equal yields and zero filament release, and report ranking tipping
points rather than a confidence interval.

From the pinned environment, run:

```bash
python scripts/ds_joint_broth_refit.py
python scripts/ds_pd_context.py
python scripts/ds_joint_robustness.py
python scripts/ds_filament_structure.py
python scripts/ds_validation_targets.py
python scripts/ds_pd_review_figures.py
```

See `data/external_validation/PD_CONTEXT_README.md` for transcription rules,
assumptions and remaining data requirements. Original source PDFs are not
distributed. Optional re-extraction of thigh pixels requires a locally
obtained PDF, PyMuPDF and Pillow; the numerical analyses use the supplied CSVs.
Nominal Table 1 remains an explicitly conditional model-MIC scenario.

## Licence

MIT. See `LICENSE`.
