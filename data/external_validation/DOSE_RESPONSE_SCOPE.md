# Scope: `dose_response` external-validation mode (neutropenic-thigh PK/PD index)

**Goal.** Validate the model's central immune-dependence claim against the canonical
neutropenic mouse-thigh dose-fractionation literature: that a bactericidal drug
achieves exposure-proportional kill in the absence of neutrophils, while a
bacteriostatic drug only stabilises. Reproduce the published PK/PD **index thresholds**
— meropenem stasis at ~17% fT>MIC and 1-log kill at ~26% fT>MIC (Sabet 2020), and the
bacteriostatic fAUC/MIC ~24–25-for-stasis pattern (LaPlante 2008).

This is the single dataset most likely to move the paper into JAC's wheelhouse, because
%fT>MIC / fAUC-MIC dose-fractionation is the Craig–Andes paradigm JAC reviewers know best.

---

## What it validates

A 24-h endpoint (Δlog₁₀ CFU from inoculum) as a function of the PK/PD index, in a
**neutropenic host** (immune effectors off). The model passes if:
1. the cidal drug shows a monotonic, sigmoidal kill-vs-fT>MIC curve;
2. the predicted **stasis threshold** lands near the published value (target: within
   ~5–10 percentage-points of fT>MIC);
3. the static drug produces **no net kill** at any achievable exposure (stasis at best),
   confirming immune-dependence of bacteriostasis.

---

## The central technical problem: the model is not MIC-parameterised

The PD kill terms use **absolute concentration** (mg/L), not multiples of MIC:
- cidal direct kill `8·C/(C+1)`, damage `k_dmg = 12·C`, `Damage50 = 3.0`;
- static `H = 1 − C^1.2/(0.1^1.2 + C^1.2)` (EC50 = 0.1 mg/L).

To compute fT>MIC we must first define an **effective MIC** for the model's
representative organism. Recommended definition (principled, standard, disclosable):

> **MIC_eff = the constant concentration that produces net 24-h stasis (Δlog₁₀ CFU = 0)
> in the neutropenic host**, found by a 1-D root-solve on a constant-concentration run.

Compute this once per drug class. fT>MIC and fAUC/MIC are then taken relative to MIC_eff.
This makes the validation self-consistent and avoids smuggling in an organism-specific MIC
the model was never fit to.

---

## Two build approaches

**Approach A — index-driven exposure (RECOMMENDED).**
Generate a grid of dosing regimens (vary dose × interval at a fixed representative
half-life), simulate each in the neutropenic host for 24 h, compute the achieved fT>MIC
from the concentration trajectory (reusing `compute_ft_mic`), and record Δlog₁₀ CFU.
Plot endpoint vs index. *Validates the index→response relationship directly, needs no
mouse PK.* Cleanest and most faithful to what the experiments actually report.

**Approach B — full murine PK.**
Add mouse meropenem PK parameters and reproduce the actual q3/q6/q12/q24h fractionation
schedules. More literally faithful to the experiment, but requires sourcing murine PK
(meropenem t½ in mice ≈ 0.3 h, very different from human) and adds moving parts without
changing the scientific conclusion. Not recommended for the first pass.

> Recommendation: **Approach A.** The experiments are summarised by their PK/PD index, not
> their mouse PK; matching the index is the validation.

---

## Components to build

| # | Component | Where | Reuse / new | Est. |
|---|-----------|-------|-------------|------|
| 1 | `effective_mic(drug_class, host)` — root-solve stasis concentration | new helper in `external_validation.py` | new (~30 lines) | 0.5 d |
| 2 | `predict_dose_response(ds)` — regimen grid → (fT>MIC, Δlog₁₀CFU) points in neutropenic host | `external_validation.py` | reuses `run_simulation`, `compute_ft_mic`/`compute_auc_mic` | 1.0 d |
| 3 | `kind: "dose_response"` dispatch + registry schema (host=neutropenic, index=fT>MIC\|fAUC_MIC, regimen grid spec) | `external_validation.py`, `registry.json` | extend existing dispatch | 0.5 d |
| 4 | Metrics: stasis-threshold error + RMSE on the Δlog₁₀-vs-index curve; sigmoid (Emax) fit | `external_validation.py` | new (~40 lines) | 0.5 d |
| 5 | Overlay plot (model sigmoid + digitised points, stasis line) | `external_validation.py` `plot_overlay` | extend | 0.25 d |
| 6 | Tests for the new mode | `tests/test_external_validation.py` | extend | 0.25 d |
| 7 | Digitise Sabet 2020 Table 3 / dose-response fig (+ LaPlante 2008) into CSV | data (user) | — | 0.5 d |
| 8 | Manuscript: new Table + Figure + narrative; abstract/Results/Discussion | `MANUSCRIPT_framework.md` | — | 0.5 d |

**Total: ~3–4 focused days** (≈2.5 d engineering + user digitisation + manuscript).

The PK/PD index helpers (`compute_ft_mic`, `compute_auc_mic` in
`src/analysis/dosing_optimization.py`) already exist and are directly reusable — this is
the main reason the estimate is days, not weeks.

---

## Data to digitise

- **Sabet 2020** (AAC; PMC7179318), Table 3 + dose-response figure: meropenem vs
  *A. baumannii*, neutropenic thigh. Stasis 17.3% fT>MIC, 1-log kill 26.0%. → cidal anchor.
- **LaPlante 2008** (AAC; PMC2415789), Table 3 / Fig 3: doxycycline neutropenic CA-MRSA
  thigh, in-vivo bacteriostatic, fAUC/MIC ~24–25 for stasis. → static anchor.

CSV format: `index,delta_log10_cfu` (index = %fT>MIC for cidal, fAUC/MIC for static).

---

## Decision points (need user input)

1. **Approach A vs B** — recommend A (index-driven; no mouse PK).
2. **Scope** — cidal-only (Sabet) for a first, clean result, or also static (LaPlante)?
   Recommend both: the static "no net kill" arm is the more striking immune-dependence
   evidence and is cheap to add once the harness exists.
3. **Effective-MIC definition** — confirm the stasis-concentration definition above.

---

## Risks / honest caveats

- **Effective-MIC choice is a modelling decision** that shifts the absolute index axis;
  must be stated plainly in Methods. Mitigated by the principled stasis definition.
- **The model's cidal kill is concentration-dependent (direct + damage), so it may not
  collapse perfectly onto fT>MIC** — it could show residual AUC/Cmax dependence. This is
  itself a reportable finding (does the model agree that fT>MIC is the governing index for
  the beta-lactam?), but it could blur the stasis-threshold comparison. Plan: hold half-life
  fixed in Approach A so the index is well-defined, and report index-collapse as a secondary
  result.
- **Single representative organism** — the model is not *A. baumannii*-specific; the
  comparison is of the index *threshold*, not strain-specific magnitudes.
- **Neutropenic definition** — set N_eff to near-zero and disable recruitment; confirm the
  static arm then shows no clearance (expected) before trusting the cidal numbers.

---

## Success criteria

- Cidal stasis threshold within ~5–10 pp fT>MIC of Sabet 2020 (17.3%).
- Monotonic sigmoidal cidal kill curve; static arm net Δlog₁₀ ≥ 0 (no kill) throughout.
- A new manuscript Figure (kill vs index, model + digitised) and one-paragraph Results
  subsection placing the model in the Craig–Andes dose-fractionation tradition.
