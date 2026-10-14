# Source context for the PD reassessment

The current adopted simulation is retained. The new analyses are diagnostic
and conditional sensitivity analyses; none identifies a confidence interval
or justifies adopting the optimizer's best point as a measured parameter.

## Broth first exposure windows

`meropenem_figure2bc_initial.csv` supplements the existing Figure 2A readings
from Nussbaumer-Pröll et al. (2022), doi:10.3389/fphar.2022.840692. Figure 2
is available under CC BY in `source_figures/figure2_embedded0.png` (1978×1276
pixels). Coordinates in the CSV refer to that image. The axes are linear in
time and log10 CFU/mL: panel B uses log10=1 at approximately y=700 and
32.7 pixels per log; panel C uses log10=1 at y=1122 and 37.5 pixels per log.
Values are rounded readings with a ±0.3 log10 sensitivity allowance, not SDs.

Only times at or before the first removal are used: A 9.6 h, B 4.8 h and
C 3.2 h. Concentrations are absolute mg/L because ATCC-27853 has MIC 1 mg/L.
Each panel has one shared baseline and its own growth nuisance parameter.
Coincident markers are not independent replicates. Panel B's unresolved
0.25×MIC arm and off-plot higher-dose markers are excluded rather than
invented as censored measurements. Panel C's high-dose cluster is flagged;
an additional fit excludes it. The panel B 2×MIC arm falls then regrows
before removal. An omission sensitivity is a model-adequacy diagnostic and
does not establish that the source observations are wrong.

Run `python scripts/ds_joint_broth_refit.py`. It profiles h with other
parameters reoptimized and reports residuals and omission sensitivities.
These raster summaries do not supply a valid sampling-error likelihood.

## Thigh strain and timing context

`sabet2020_strains.csv` transcribes Tables 1 and 3 and the six starting counts
from Sabet et al. (2020), doi:10.1128/AAC.01665-19. The published stasis
threshold 17.3% is the rounded arithmetic mean of six strain-specific
thresholds, not a crossing of a pooled response cloud. Table 3 is retained
when its values conflict slightly with labels in Figures 5 and 6.

Untreated animals were killed 12 h after infection; treatment started at 2 h.
The reported 1.16–2.08 log growth therefore spans **10 h**, not the 24 h
treated endpoint. The source does not pair those growth increments to strains.
The entire range is applied separately to each starting count, with a 12 h
elapsed-time sensitivity retained. The capacity of 1e10 CFU/thigh is an
explicit context assumption above the possible observed count range; it is
not the broth CFU/mL capacity and is not estimated from the endpoint.

`tools/digitize_sabet.py` recovers visible blue marker clusters per strain
from the locally obtained source PDF. It does not recover animal replicates
hidden by overlap or dose/interval identities. The CSV includes quality and
pixel coordinates; the provenance JSON records image calibration and hashes.
Source images remain in gitignored `tmp/pd_source_review/`.

Run `python scripts/ds_pd_context.py`. It holds each measured strain MIC
fixed, scales C50 by MIC as an explicit cross-organism transfer assumption,
and derives growth from controls without using 17.3% as a fitting target.
The PK is the canonical human effect-site model, **not** the original murine
PK (reported half-lives 0.09–0.32 h). Intervals of 2, 4 and 6 h are transport
sensitivities. Missing per-point regimens and paired controls prevent a
definitive mechanistic replay. An envelope containing 17.3% does not satisfy
the original validation target.

The legacy index sweep treats its driving concentration as fully active.
With linear PK, introducing a constant unbound fraction and rescaling all
synthetic doses inversely leaves this index-response curve unchanged. The
sweep therefore carries no inference about administered murine doses or
species-specific binding. The main matched-dose study applies its explicit
agent-specific free fractions separately.

Re-extracting thigh pixels requires the locally obtained PDF plus optional
PyMuPDF and Pillow. The committed numerical CSVs and context simulations do
not require the source PDF or PyMuPDF.

## Joint downstream uncertainty

Run `python scripts/ds_joint_robustness.py`. Ten PD scenarios are propagated
with MIC fixed at 1 mg/L (measured for meropenem's broth isolate and assumed
for ceftazidime in this synthetic comparison), while in-vivo growth, immune
and PK assumptions are held fixed. This is distinct from the nominal Table 1
and older `multiarm_downstream.json`, which use each model's own stasis MIC.
The yield grid spans 0.25–40 and filament release spans zero to twice nominal.
Equal-yield, zero-filament tests and ranking tipping points are reported.
All ranges are stress-test envelopes, not confidence intervals. A direct
4.77-fold CAZ:MERO ELISA observation is included as an endpoint-motivated
scenario; it is not treated as a measured latent lysis-yield ratio.

To resolve the remaining questions, obtain raw broth replicates, matched
drug concentrations and MICs, strain-specific untreated trajectories, and
murine regimen-linked counts. A paired CAZ/MERO experiment must measure viable
counts, exposure and LPS with the same assay and sampling times, with controls
and a clearance/background observation model. Yield denominators must use
integrated drug-kill flux rather than inoculum or net CFU loss.
