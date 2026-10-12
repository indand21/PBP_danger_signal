# External validation datasets

This folder holds **digitized data points extracted from independent published
figures**, used by `src/analysis/external_validation.py` to validate model
*trajectories* (not just ranges) with quantitative metrics (RMSE, bias, R²) and
VPC-style overlays.

This is the genuine external-validation layer. `literature_validation.py` only
checks whether a model output falls inside a published min–max *range* (face
validity). This layer checks whether the model reproduces a published *curve*.

## Independence (calibration vs validation)

Every dataset declares a `role`:

- `calibration` — a source whose data was used (directly or indirectly) to tune
  model constants. Reported for context only.
- `validation` — a source held out entirely from model building. **These are the
  only datasets that count toward the reported external-validation metrics.**

`run_external_validation()` defaults to `roles=("validation",)`. Keep that
boundary honest: if you ever tune a constant to improve fit on a dataset, change
its role to `calibration`.

## How to add a dataset

1. **Pick a figure** from a published paper (see `registry.json` `notes` for the
   target list, e.g. a meropenem plasma concentration–time profile or a
   time-kill curve).
2. **Digitize it** with a tool such as WebPlotDigitizer
   (https://automeris.io/WebPlotDigitizer). Export the (x, y) points.
3. **Save** them as `data/external_validation/<id>.csv` with a header row:
   ```
   time_h,value
   0.5,42.1
   1.0,28.7
   ...
   ```
   - For **time-kill** data, `value` is **log10 CFU/mL** (set `y_space:"log10"`).
   - For **PK** data, `value` is the plasma concentration in **mg/L**
     (`y_space:"linear"`).
   - For **cytokine** data, `value` is **pg/mL** (`y_space:"linear"`).
   Blank rows, `#` comments, and the header line are ignored, so a stub CSV with
   only a header is harmless — the dataset simply reports `NO_DATA` until filled.
4. **Register** it: add an entry to `registry.json` (see schema below).
5. **Run** `python scripts/run_external_validation.py`.

## registry.json schema

```jsonc
{
  "datasets": [
    {
      "id": "meropenem_pk_1g_30min",   // unique; also the CSV/plot basename
      "kind": "pk",                     // 'pk' | 'timekill' | 'cytokine'
      "drug": "meropenem",              // must resolve in get_drug_pk_parameters
      "drug_class": "cidal",            // 'cidal' | 'static'
      "role": "validation",             // 'calibration' | 'validation'
      "csv": "meropenem_pk_1g_30min.csv",
      "y_unit": "mg/L",
      "y_space": "linear",              // 'linear' | 'log10' (metric space)
      "organism": null,
      "source": "Author Year, Journal, DOI — Fig N",
      "notes": "what to digitize / any caveats",
      "scenario": { /* kind-specific, see below */ }
    }
  ]
}
```

### Scenario fields by kind

- **pk**: `dose_mg`, `interval_hours`, `n_doses`, `infusion_min`, `weight_kg`.
  The model's central-compartment (plasma) concentration is reconstructed and
  compared to the digitized profile. (Effect-site `Kp` scaling is divided out so
  the comparison is against plasma.)
- **timekill**: `concentration_mgL` (the constant drug concentration of the
  experiment, e.g. 4×MIC in mg/L), `initial_burden`, `initial_persister`,
  `initial_scv`, `immune` (default 0 — broth has no neutrophils). The PD ODE is
  integrated at constant concentration, isolating intrinsic drug kill.
- **cytokine**: `dose_mg`, `interval_hours`, `n_doses`, `infusion_min`,
  `initial_burden`, `immune`, `marker` (`"IL6"` or `"TNF"`).

## Output

- `results/trial_data/external_validation_report.json` — metrics per dataset +
  pooled RMSE.
- `results/figures/external_validation/<id>.png` — per-dataset overlay.
