"""
Configuration manager for QSP model parameters.
Loads parameters from YAML/JSON files instead of hardcoded values.
"""

import yaml
import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

from src.core.parameters import (
    BacterialParameters,
    ImmuneParameters,
    CytokineParameters,
    PKParameters,
    get_default_parameters,
    get_drug_pk_parameters,
)


# Default config directory
DEFAULT_CONFIG_DIR = Path(__file__).parent / "config"


@dataclass
class PDMechanismParameters:
    """Pharmacodynamic mechanism parameters."""
    static_EC50: float = 1.0
    static_hill: float = 1.0
    cidal_damage50: float = 6.0
    cidal_hill: float = 2.0
    cidal_kill_rate: float = 2.0
    cidal_dmg_rate: float = 2.0
    concentration_dependent: bool = True


@dataclass
class PersisterParameters:
    """Persister dynamics parameters."""
    immune_kill_factor: float = 0.1
    reactivation_rate: float = 0.05


@dataclass
class SCVParameters:
    """SCV dynamics parameters."""
    mutation_threshold: float = 0.3
    immune_kill_factor: float = 0.05


@dataclass
class SimulationParameters:
    """Simulation control parameters."""
    default_t_span: tuple = (0, 96)
    max_step: float = 0.1
    rtol: float = 1e-6
    atol: float = 1e-8
    default_method: str = "RK45"


@dataclass
class ClinicalThresholds:
    """Clinical endpoint thresholds."""
    eradication_threshold: float = 100.0
    microbiologic_threshold_log: float = 3.0
    resistance_threshold_fraction: float = 0.10
    toxicity_threshold_IL6: float = 500.0


@dataclass
class QSPConfiguration:
    """Complete QSP model configuration."""
    bacteria: BacterialParameters = field(default_factory=BacterialParameters)
    immune: ImmuneParameters = field(default_factory=ImmuneParameters)
    cytokine: CytokineParameters = field(default_factory=CytokineParameters)
    pd_mechanism: PDMechanismParameters = field(default_factory=PDMechanismParameters)
    persister: PersisterParameters = field(default_factory=PersisterParameters)
    scv: SCVParameters = field(default_factory=SCVParameters)
    simulation: SimulationParameters = field(default_factory=SimulationParameters)
    clinical: ClinicalThresholds = field(default_factory=ClinicalThresholds)


class ConfigManager:
    """
    Manages loading and access to QSP model configuration.
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize config manager.

        Args:
            config_dir: Directory containing config files.
                        Defaults to ./config
        """
        self.config_dir = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
        self._model_config: Optional[QSPConfiguration] = None
        self._drug_library: Optional[Dict] = None

    def load_model_parameters(self, filename: str = "model_parameters.yaml") -> QSPConfiguration:
        """
        Load model parameters from YAML file.

        Args:
            filename: Config file name (relative to config_dir)

        Returns:
            QSPConfiguration object with loaded parameters
        """
        filepath = self.config_dir / filename

        if not filepath.exists():
            # Fall back to defaults if file not found
            self._model_config = QSPConfiguration()
            return self._model_config

        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        # Parse bacterial parameters
        bact_data = data.get('bacteria', {})
        bacteria = BacterialParameters(
            k_growth=bact_data.get('k_growth', 0.5),
            B_max=bact_data.get('B_max', 1e9),
            k_pers=bact_data.get('k_pers', 0.01),
            mu_mut=bact_data.get('mu_mut', 1e-6),
            k_repair=bact_data.get('k_repair', 0.1),
            MIC_baseline=bact_data.get('MIC_baseline', 1.0),
        )

        # Parse immune parameters
        imm_data = data.get('immune', {})
        immune = ImmuneParameters(
            N_eff_0=imm_data.get('N_eff_0', 1e7),
            k_prod=imm_data.get('k_prod', 0.5),
            EC50_immune=imm_data.get('EC50_immune', 1e5),
            k_deg_immune=imm_data.get('k_deg_immune', 0.05),
            k_kill_base=imm_data.get('k_kill_base', 1e-8),
        )

        # Parse cytokine parameters
        cyto_data = data.get('cytokine', {})
        cytokine = CytokineParameters(
            k_IL6_prod=cyto_data.get('k_IL6_prod', 4.0),
            alpha_static=cyto_data.get('alpha_static', 1.0),
            alpha_cidal=cyto_data.get('alpha_cidal', 3.0),
            k_IL6_clear=cyto_data.get('k_IL6_clear', 0.2),
            TNF_IL6_ratio=cyto_data.get('TNF_IL6_ratio', 0.3),
        )

        # Parse PD mechanism parameters
        pd_data = data.get('pharmacodynamics', {})
        static_data = pd_data.get('static', {})
        cidal_data = pd_data.get('cidal', {})
        pd_mechanism = PDMechanismParameters(
            static_EC50=static_data.get('EC50', 1.0),
            static_hill=static_data.get('hill_coefficient', 1.0),
            cidal_damage50=cidal_data.get('damage50', 6.0),
            cidal_hill=cidal_data.get('hill_coefficient', 2.0),
            cidal_kill_rate=cidal_data.get('kill_rate', 2.0),
            cidal_dmg_rate=cidal_data.get('dmg_rate', 2.0),
            concentration_dependent=cidal_data.get('concentration_dependent', True),
        )

        # Parse persister parameters
        pers_data = data.get('persister', {})
        persister = PersisterParameters(
            immune_kill_factor=pers_data.get('immune_kill_factor', 0.1),
            reactivation_rate=pers_data.get('reactivation_rate', 0.05),
        )

        # Parse SCV parameters
        scv_data = data.get('scv', {})
        scv = SCVParameters(
            mutation_threshold=scv_data.get('mutation_threshold', 0.3),
            immune_kill_factor=scv_data.get('immune_kill_factor', 0.05),
        )

        # Parse simulation parameters
        sim_data = data.get('simulation', {})
        t_span = sim_data.get('default_t_span', [0, 96])
        simulation = SimulationParameters(
            default_t_span=(t_span[0], t_span[1]),
            max_step=sim_data.get('max_step', 0.1),
            rtol=sim_data.get('rtol', 1e-6),
            atol=sim_data.get('atol', 1e-8),
            default_method=sim_data.get('default_method', 'RK45'),
        )

        # Parse clinical thresholds
        clin_data = data.get('clinical', {})
        clinical = ClinicalThresholds(
            eradication_threshold=clin_data.get('eradication_threshold', 100.0),
            microbiologic_threshold_log=clin_data.get('microbiologic_threshold_log', 3.0),
            resistance_threshold_fraction=clin_data.get('resistance_threshold_fraction', 0.10),
            toxicity_threshold_IL6=clin_data.get('toxicity_threshold_IL6', 500.0),
        )

        self._model_config = QSPConfiguration(
            bacteria=bacteria,
            immune=immune,
            cytokine=cytokine,
            pd_mechanism=pd_mechanism,
            persister=persister,
            scv=scv,
            simulation=simulation,
            clinical=clinical,
        )

        return self._model_config

    def load_drug_library(self, filename: str = "drug_library.yaml") -> Dict[str, Any]:
        """
        Load drug library from YAML file.

        Args:
            filename: Drug library file name

        Returns:
            Dictionary of drug configurations
        """
        filepath = self.config_dir / filename

        if not filepath.exists():
            self._drug_library = {}
            return self._drug_library

        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        self._drug_library = data.get('drugs', {})
        return self._drug_library

    def get_drug_config(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific drug.

        Args:
            drug_name: Name of the drug

        Returns:
            Drug configuration dictionary or None
        """
        if self._drug_library is None:
            self.load_drug_library()

        # Case-insensitive lookup
        for name, config in self._drug_library.items():
            if name.lower() == drug_name.lower():
                return config
        return None

    def get_model_parameters_dict(self) -> Dict[str, Any]:
        """
        Get model parameters in dictionary format compatible with existing code.

        Returns:
            Dictionary with 'bacteria', 'immune', 'cytokine' keys
        """
        if self._model_config is None:
            self.load_model_parameters()

        return {
            'bacteria': self._model_config.bacteria,
            'immune': self._model_config.immune,
            'cytokine': self._model_config.cytokine,
        }

    def save_model_parameters(self, config: QSPConfiguration, filename: str = "model_parameters.yaml"):
        """
        Save model parameters to YAML file.

        Args:
            config: Configuration to save
            filename: Output file name
        """
        filepath = self.config_dir / filename

        data = {
            'bacteria': {
                'k_growth': config.bacteria.k_growth,
                'B_max': config.bacteria.B_max,
                'k_pers': config.bacteria.k_pers,
                'mu_mut': config.bacteria.mu_mut,
                'k_repair': config.bacteria.k_repair,
                'MIC_baseline': config.bacteria.MIC_baseline,
            },
            'immune': {
                'N_eff_0': config.immune.N_eff_0,
                'k_prod': config.immune.k_prod,
                'EC50_immune': config.immune.EC50_immune,
                'k_deg_immune': config.immune.k_deg_immune,
                'k_kill_base': config.immune.k_kill_base,
            },
            'cytokine': {
                'k_IL6_prod': config.cytokine.k_IL6_prod,
                'alpha_static': config.cytokine.alpha_static,
                'alpha_cidal': config.cytokine.alpha_cidal,
                'k_IL6_clear': config.cytokine.k_IL6_clear,
                'TNF_IL6_ratio': config.cytokine.TNF_IL6_ratio,
            },
            'pharmacodynamics': {
                'static': {
                    'EC50': config.pd_mechanism.static_EC50,
                    'hill_coefficient': config.pd_mechanism.static_hill,
                },
                'cidal': {
                    'damage50': config.pd_mechanism.cidal_damage50,
                    'hill_coefficient': config.pd_mechanism.cidal_hill,
                    'kill_rate': config.pd_mechanism.cidal_kill_rate,
                    'dmg_rate': config.pd_mechanism.cidal_dmg_rate,
                    'concentration_dependent': config.pd_mechanism.concentration_dependent,
                },
            },
            'persister': {
                'immune_kill_factor': config.persister.immune_kill_factor,
                'reactivation_rate': config.persister.reactivation_rate,
            },
            'scv': {
                'mutation_threshold': config.scv.mutation_threshold,
                'immune_kill_factor': config.scv.immune_kill_factor,
            },
            'simulation': {
                'default_t_span': list(config.simulation.default_t_span),
                'max_step': config.simulation.max_step,
                'rtol': config.simulation.rtol,
                'atol': config.simulation.atol,
                'default_method': config.simulation.default_method,
            },
            'clinical': {
                'eradication_threshold': config.clinical.eradication_threshold,
                'microbiologic_threshold_log': config.clinical.microbiologic_threshold_log,
                'resistance_threshold_fraction': config.clinical.resistance_threshold_fraction,
                'toxicity_threshold_IL6': config.clinical.toxicity_threshold_IL6,
            },
        }

        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def get_pd_mechanism_params(self) -> PDMechanismParameters:
        """Get PD mechanism parameters."""
        if self._model_config is None:
            self.load_model_parameters()
        return self._model_config.pd_mechanism

    def get_simulation_params(self) -> SimulationParameters:
        """Get simulation control parameters."""
        if self._model_config is None:
            self.load_model_parameters()
        return self._model_config.simulation

    def get_clinical_thresholds(self) -> ClinicalThresholds:
        """Get clinical endpoint thresholds."""
        if self._model_config is None:
            self.load_model_parameters()
        return self._model_config.clinical


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get global config manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_config() -> QSPConfiguration:
    """Convenience function to load default configuration."""
    return get_config_manager().load_model_parameters()


def get_parameters_from_config() -> Dict[str, Any]:
    """Get parameters dictionary from config (backward compatible)."""
    return get_config_manager().get_model_parameters_dict()


if __name__ == "__main__":
    # Test config loading
    config_mgr = ConfigManager()
    config = config_mgr.load_model_parameters()
    print("Model parameters loaded successfully!")
    print(f"  Growth rate: {config.bacteria.k_growth}")
    print(f"  Cidal damage50: {config.pd_mechanism.cidal_damage50}")
    print(f"  Eradication threshold: {config.clinical.eradication_threshold}")

    drugs = config_mgr.load_drug_library()
    print(f"\nLoaded {len(drugs)} drugs from library")
    for drug_name in drugs:
        print(f"  - {drug_name}")
