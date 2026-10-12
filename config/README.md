# config/

**None of these files is the active configuration for the danger-signal
study.** Editing them does not change any reported result.

| File | What it is | Read by |
|---|---|---|
| `model_parameters.yaml` | Parameters for an older damage-driven killing formulation (`damage50`, `dmg_rate`) | legacy `src/utils/config_manager.py` only |
| `config_template.yaml` | Template for the above | legacy config manager |
| `drug_library.yaml` | Drug descriptors | legacy modules |
| `calibrated_parameters.json`, `calibrated_parameters_cidal_FIXED.json` | Outputs of the legacy calibration scripts in `src/calibration/`, fitted for the damage-driven formulation | nothing in the study pipeline |

The study takes its parameters from:

- **Bacterial, immune and PK parameters:** the dataclass defaults in
  `src/core/parameters.py` (including the saturating Hill kill: `k_kill_max`,
  `kill_C50`, `kill_hill`) and the per-drug values from
  `get_drug_pk_parameters()`.
- **Danger-signal yields:** `FINAL_YIELDS_FG_PER_CFU` in
  `src/analysis/danger_signal.py`.

Every results file under `results/danger_signal/` records the yield
configuration it was produced with, together with the code revision and
library versions, in its `provenance` block.

These files were documented rather than deleted because the legacy therapy,
resistance and machine-learning modules, which are outside the study's
scientific claims, still use them. An external review (R12) noted the
mismatch; the fitted chain for the active Hill parameters is a separate open
item (R7).
