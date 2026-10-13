"""
Pharmacodynamic model: bacterial dynamics, immune response, cytokine production
"""

import numpy as np
from typing import Dict, Tuple

from src.core.parameters import HostDamageParameters

class BacterialPopulationODE:
    """
    Multi-population bacterial ODE system:
      - B_rep: replicating cells
      - B_pers: persister cells
      - B_SCV: small-colony variants (heteroresistant)
      - N_eff: effective immune effectors (neutrophils/macrophages)
      - Damage: accumulated cidal damage (unitless)
      - IL6, TNF: inflammatory cytokines
      - PAMP: pathogen-associated molecular patterns released during killing
    """

    def __init__(self, params: Dict):
        """
        Parameters:
            params: dict with keys 'bacteria', 'immune', 'cytokine'
        """
        self.p_bact = params['bacteria']
        self.p_imm = params['immune']
        self.p_cyto = params['cytokine']
        # Host-damage params; default keeps backward compatibility with callers
        # that build a params dict without a 'damage' key.
        self.p_damage = params.get('damage', HostDamageParameters())

    def h_static(self, C: float, EC50: float = 1.0, hill: float = 1.0) -> float:
        """
        Hill-type inhibition for bacteriostatic drugs
        H = 1 - C^hill / (EC50^hill + C^hill)
        Lower growth multiplier as concentration increases

        Units: C and EC50 are in mg/L (consistent with MIC)
        EC50=1.0 mg/L corresponds to typical MIC for susceptible strains
        """
        return 1.0 - (C**hill) / (EC50**hill + C**hill)

    def cidal_kill_rate(self, C: float) -> float:
        """Instantaneous bactericidal kill rate (per hour) at concentration C.

        Extracted from ``rhs`` unchanged so that the danger-signal layer
        (``src/analysis/danger_signal.py``) can reuse the *same* kill function
        rather than restating the Hill expression: danger-signal release is
        proportional to drug-induced killing, so the two must not be able to
        drift apart.
        """
        Ch = np.asarray(C, dtype=float) ** self.p_bact.kill_hill
        return (self.p_bact.k_kill_max * Ch
                / (Ch + self.p_bact.kill_C50 ** self.p_bact.kill_hill))

    def f_cidal_mechanism(self, C: float, Damage: float) -> float:
        """
        Cidal killing as function of accumulated damage
        f_cidal = Damage^n / (Damage50^n + Damage^n)
        Higher damage -> higher killing rate

        Units: C in mg/L, Damage dimensionless
        Damage50 calibrated for mg/L concentration scale
        """
        n = 2.0
        Damage50 = 3.0  # calibrated for mg/L scale
        return (Damage**n) / (Damage50**n + Damage**n)

    def rhs(self, t: float, y: np.ndarray, C_effect: float, 
            drug_class: str = 'cidal', is_static: bool = False) -> np.ndarray:
        """
        Right-hand side of the bacterial + immune ODE

        State vector y:
            y[0] = B_rep (replicating cells, CFU/mL)
            y[1] = B_pers (persisters, CFU/mL)
            y[2] = B_SCV (small-colony variants, CFU/mL)
            y[3] = N_eff (neutrophil/macrophage equivalents)
            y[4] = Damage (accumulated cidal damage, unitless)
            y[5] = IL6 (pg/mL)
            y[6] = TNF (pg/mL)

        Parameters:
            t: time (hours)
            y: state vector
            C_effect: drug concentration at effect site (mg/L)
            drug_class: 'cidal' or 'static'
            is_static: legacy parameter, True if drug is bacteriostatic
        """
        B_rep = max(y[0], 1e-6)
        B_pers = max(y[1], 0)
        B_SCV = max(y[2], 0)
        N_eff = max(y[3], 0)
        Damage = max(y[4], 0)
        IL6 = max(y[5], 0)
        TNF = max(y[6], 0)

        # PAMP state (index 7) — new
        PAMP = max(y[7], 0) if len(y) > 7 else 0.0

        # Host-damage state (index 8) — new
        D_host = max(y[8], 0) if len(y) > 8 else 0.0

        B_total = B_rep + B_pers + B_SCV

        dydt = np.zeros(9)  # 7 PD states + PAMP + D_host

        # --- Replicating population ---
        # Logistic growth with carrying capacity
        growth_term = self.p_bact.k_growth * (1.0 - B_total / self.p_bact.B_max) * B_rep

        # Bacteriostatic effect: reduces growth proportionally
        static_kill = 0.0
        if is_static or drug_class == 'static':
            H_static = self.h_static(C_effect, EC50=0.1, hill=1.2)  # EC50=0.1 mg/L for clinical potency
            growth_term *= H_static
            # Path B: modest DIRECT bacteriostatic-associated killing, scaled by the
            # degree of target engagement (1 - H_static). Real bacteriostatic agents
            # produce a slow net decline in vitro even without immune cells, which a
            # pure growth-inhibition model cannot reproduce. Default 0 keeps the
            # original (immune-dependent-only) bacteriostatic mechanism. This kill
            # does NOT release PAMPs: protein-synthesis inhibitors cause little
            # lysis/endotoxin release compared with cell-wall-active agents.
            static_kill = self.p_bact.static_kill_max * (1.0 - H_static) * B_rep

        # Immune-mediated killing
        immune_kill = self.p_imm.k_kill_base * N_eff * B_rep

        # Cidal drug killing: a single saturating (Hill) function of effect-site
        # concentration (C_effect in mg/L). The earlier two-term (direct +
        # damage-accumulation) kill was replaced during recalibration; the Damage
        # state below is retained as a diagnostic and no longer drives killing.
        if drug_class == 'cidal' and not is_static:
            # Saturating (Hill) concentration-response: the rate plateaus at
            # k_kill_max and is half-maximal at kill_C50, so beyond a few multiples
            # of kill_C50 additional concentration does not raise the instantaneous
            # rate. The steep exponent (kill_hill=4) places half-maximal killing
            # near the MIC, so with realistic (fast-clearing) PK the net burden
            # reduction is driven by the time the concentration spends above the
            # MIC -- i.e. it behaves time-dependently (%fT>MIC): more-fractionated
            # regimens out-kill a single large dose (manuscript Section 2.7). This
            # arises from the kill/growth structure, NOT from the persister
            # reservoir (the fractionation signature persists, and strengthens,
            # with zero persisters). kill_C50 sets the stasis concentration ~1 mg/L.
            kill_rate = float(self.cidal_kill_rate(C_effect))
            cidal_kill = kill_rate * B_rep
        else:
            kill_rate = 0.0
            cidal_kill = 0.0

        # Transition to persisters
        to_pers = self.p_bact.k_pers * B_rep
        # Reactivation returns persisters to the replicating pool. The same term
        # is removed from B_pers below, so the switch conserves bacterial mass
        # (previously exit_pers left B_pers but was never added back to B_rep).
        exit_pers = 0.05 * B_pers  # slow reactivation

        dydt[0] = growth_term - immune_kill - cidal_kill - static_kill - to_pers + exit_pers

        # --- Persister population ---
        # Persisters are relatively protected from drugs
        from_rep = self.p_bact.k_pers * B_rep
        immune_kill_pers = 0.1 * self.p_imm.k_kill_base * N_eff * B_pers  # slower immune kill

        dydt[1] = from_rep - immune_kill_pers - exit_pers

        # --- Small-colony variant population ---
        # SCVs emerge under sustained static inhibition. The emergence probability
        # is a SMOOTH sigmoid of the static-inhibition level (no hard threshold):
        # strong inhibition (H_static_check -> 0) gives near-full emergence, weak
        # inhibition (-> 1) gives ~0. Mutation only occurs under static pressure.
        if is_static or drug_class == 'static':
            H_static_check = self.h_static(C_effect, EC50=0.1, hill=1.2)
            scv_switch = 1.0 / (1.0 + np.exp(
                (H_static_check - self.p_bact.scv_switch_midpoint)
                / self.p_bact.scv_switch_width))
            mutation_rate = self.p_bact.mu_mut * B_rep * scv_switch
        else:
            mutation_rate = 0.0

        immune_kill_scv = 0.05 * self.p_imm.k_kill_base * N_eff * B_SCV

        dydt[2] = mutation_rate - immune_kill_scv
        # Mass balance: cells that convert to small-colony variants leave the
        # replicating pool. (mu_mut ~ 1e-6 makes this numerically negligible, but
        # the conversion is now conserved rather than creating cells from nothing.)
        dydt[0] -= mutation_rate

        # --- Immune effectors (neutrophils/macrophages) ---
        # Homeostatic pool, recruited by burden: the effectors relax towards a
        # target that rises from N_eff_0 (no infection) to recruit_fold_max x
        # N_eff_0 (saturating burden). Production is k_deg_immune * target, in
        # cells/mL/h, which is the state's own scale; see ImmuneParameters for
        # what this replaced (review R10) and why the plateau is a fold.
        # N_eff_0 = 0 gives dN/dt = -k_deg * N, so a neutropenic host started
        # at zero stays at exactly zero.
        hill_immune = B_total / (self.p_imm.EC50_immune + B_total)
        target = self.p_imm.N_eff_0 * (
            1.0 + (self.p_imm.recruit_fold_max - 1.0) * hill_immune)

        dydt[3] = self.p_imm.k_deg_immune * (target - N_eff)

        # --- Cidal damage accumulation ---
        # Units: C_effect in mg/L
        # At C=5 mg/L: k_dmg = 12.0 * 5 = 60/h → Damage_eq = 200 → f_cidal ≈ 1.0
        if drug_class == 'cidal' and not is_static:
            k_dmg = 12.0 * C_effect  # damage accumulation rate (mg/L scale)
            k_repair = self.p_bact.k_repair
            dydt[4] = k_dmg - k_repair * Damage
        else:
            dydt[4] = -self.p_bact.k_repair * Damage  # passive repair

        # --- IL-6 production ---
        # Two sources: (1) burden-dependent baseline, (2) PAMP-mediated burst during killing
        # Cidal drugs trigger more IL-6 via TLR9 (DNA release) — captured in PAMP burst
        if drug_class == 'cidal' and not is_static:
            alpha_cyto = self.p_cyto.alpha_cidal
        else:
            alpha_cyto = self.p_cyto.alpha_static

        # Baseline IL-6 from bacterial burden
        IL6_prod_baseline = alpha_cyto * self.p_cyto.k_IL6_prod * (B_rep + 0.5*B_pers) / 1e6
        # PAMP-mediated IL-6 burst (much stronger production during active killing)
        IL6_prod_pamp = self.p_cyto.k_IL6_prod * 5000.0 * PAMP / (PAMP + 1e6)
        IL6_prod = IL6_prod_baseline + IL6_prod_pamp
        IL6_clear = self.p_cyto.k_IL6_clear * IL6

        dydt[5] = IL6_prod - IL6_clear

        # --- TNF production (linked to positive IL-6 production, not net IL-6 change) ---
        TNF_prod = self.p_cyto.TNF_IL6_ratio * IL6_prod
        TNF_clear = 0.3 * TNF

        dydt[6] = TNF_prod - TNF_clear

        # --- PAMP dynamics ---
        # PAMPs are released during cidal bacterial killing (cell lysis releases DNA/LPS)
        # This drives the IL-6 burst observed during active bacterial clearance
        # PAMPs/DAMPs are released in proportion to killed biomass. Bactericidal
        # (cell-wall-active, lytic) killing releases the full yield; bacteriostatic
        # killing releases a fraction static_pamp_fraction of it (0 = fully
        # non-lytic idealization). Making this graded -- rather than an on/off
        # switch tied to drug_class -- is what lets the comparison be decided by a
        # measurable quantity instead of by a structural assumption.
        pamp_release = 1e7 * (cidal_kill
                              + self.p_bact.static_pamp_fraction * static_kill)

        pamp_clear = 2.0 * PAMP  # PAMPs cleared rapidly (t1/2 ≈ 20 min)
        dydt[7] = pamp_release - pamp_clear

        # --- Host damage (damage-response framework) ---
        # Injury from pathogen burden (saturating) and from normalised inflammatory
        # intensity (IL-6/TNF fold-change over healthy baseline), with recovery.
        infl_IL6 = max(IL6 / self.p_damage.IL6_ref - 1.0, 0.0)
        infl_TNF = max(TNF / self.p_damage.TNF_ref - 1.0, 0.0)
        infl_intensity = infl_IL6 + self.p_damage.w_TNF * infl_TNF
        pathogen_injury = self.p_damage.k_path * B_total / (B_total + self.p_damage.B50)
        # Hill saturation: injury rate plateaus at k_infl as inflammation grows,
        # so an extreme (and only semi-quantitative) cytokine burst cannot produce
        # unbounded host damage.
        inflammation_injury = (self.p_damage.k_infl * infl_intensity
                               / (infl_intensity + self.p_damage.I50))
        dydt[8] = pathogen_injury + inflammation_injury - self.p_damage.k_heal * D_host

        return dydt


def create_ode_system(params: Dict) -> BacterialPopulationODE:
    """Factory to create ODE system"""
    return BacterialPopulationODE(params)
