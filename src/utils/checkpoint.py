"""
Checkpoint and resume functionality for the QSP Antibiotic Model.

Provides:
- Save/load simulation state to disk (JSON + numpy arrays)
- Save/load cohort results
- Resume interrupted simulations
- Checkpoint management (list, delete, verify)

Use cases:
- Long-running sensitivity analyses
- Interrupted parallel cohort simulations
- Experiment reproducibility
- State sharing between machines
"""

import numpy as np
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import hashlib
import shutil

from src.core.simulation import SimulationResult
from src.utils.parallel_sim import PatientResult, VirtualPatient


# ---------------------------------------------------------------------------
# Checkpoint metadata
# ---------------------------------------------------------------------------

@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint file."""
    checkpoint_id: str
    created_at: str
    description: str
    checkpoint_type: str  # "simulation", "cohort", "sensitivity", "custom"
    n_items: int = 0
    status: str = "in_progress"  # "in_progress", "completed", "failed"
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "checkpoint_id": self.checkpoint_id,
            "created_at": self.created_at,
            "description": self.description,
            "checkpoint_type": self.checkpoint_type,
            "n_items": self.n_items,
            "status": self.status,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CheckpointMetadata":
        return cls(**d)


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """
    Manages checkpoint files for saving/loading simulation state.

    Each checkpoint is stored in a subdirectory under `base_dir`:
        base_dir/
            checkpoint_001/
                metadata.json
                data.npz  (or data.json for non-array data)
            checkpoint_002/
                ...

    Parameters:
        base_dir: root directory for all checkpoints.
    """

    def __init__(self, base_dir: str = "checkpoints"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _generate_id(self, prefix: str = "ckpt") -> str:
        """Generate a unique checkpoint ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rand_suffix = hashlib.md5(str(time.time_ns()).encode()).hexdigest()[:6]
        return f"{prefix}_{timestamp}_{rand_suffix}"

    def _get_dir(self, checkpoint_id: str) -> Path:
        """Get the directory path for a checkpoint."""
        return self.base_dir / checkpoint_id

    # -------------------------------------------------------------------
    # Save simulation result
    # -------------------------------------------------------------------

    def save_simulation(
        self,
        result: SimulationResult,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Save a single SimulationResult to a checkpoint.

        Parameters:
            result: SimulationResult object.
            description: human-readable description.
            tags: optional key-value tags.

        Returns:
            checkpoint_id of the saved checkpoint.
        """
        ckpt_id = self._generate_id("sim")
        ckpt_dir = self._get_dir(ckpt_id)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save arrays
        np.savez_compressed(
            ckpt_dir / "data.npz",
            t=result.t,
            y=result.y,
        )

        # Save metadata
        meta = CheckpointMetadata(
            checkpoint_id=ckpt_id,
            created_at=datetime.now().isoformat(),
            description=description,
            checkpoint_type="simulation",
            n_items=len(result.t),
            status="completed",
            tags=tags or {},
        )

        # Save state names and params as JSON
        extra = {
            "state_names": result.state_names,
            "params": result.params,
        }

        with open(ckpt_dir / "metadata.json", "w") as f:
            json.dump(meta.to_dict(), f, indent=2)

        with open(ckpt_dir / "extra.json", "w") as f:
            json.dump(extra, f, indent=2, default=str)

        return ckpt_id

    def load_simulation(self, checkpoint_id: str) -> SimulationResult:
        """
        Load a SimulationResult from a checkpoint.

        Parameters:
            checkpoint_id: the checkpoint ID.

        Returns:
            SimulationResult object.
        """
        ckpt_dir = self._get_dir(checkpoint_id)

        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")

        # Load arrays
        data = np.load(ckpt_dir / "data.npz")
        t = data["t"]
        y = data["y"]

        # Load extra
        with open(ckpt_dir / "extra.json", "r") as f:
            extra = json.load(f)

        state_names = extra["state_names"]
        params = extra["params"]

        return SimulationResult(t, y, state_names, params)

    # -------------------------------------------------------------------
    # Save cohort results
    # -------------------------------------------------------------------

    def save_cohort(
        self,
        results: List[PatientResult],
        patients: List[VirtualPatient],
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
        save_trajectories: bool = True,
    ) -> str:
        """
        Save a cohort of PatientResults to a checkpoint.

        Parameters:
            results: list of PatientResult objects.
            patients: list of VirtualPatient objects.
            description: human-readable description.
            tags: optional key-value tags.
            save_trajectories: if True, save full trajectories (larger files).

        Returns:
            checkpoint_id of the saved checkpoint.
        """
        ckpt_id = self._generate_id("cohort")
        ckpt_dir = self._get_dir(ckpt_id)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save patient specifications
        patient_dicts = []
        for p in patients:
            patient_dicts.append({
                "patient_id": p.patient_id,
                "weight_kg": p.weight_kg,
                "immune_level": p.immune_level,
                "initial_burden": p.initial_burden,
                "drug_name": p.drug_name,
                "drug_class": p.drug_class,
                "param_overrides": p.param_overrides,
                "dose_mg": p.dose_mg,
                "interval_hours": p.interval_hours,
                "n_doses": p.n_doses,
                "infusion_min": p.infusion_min,
            })

        with open(ckpt_dir / "patients.json", "w") as f:
            json.dump(patient_dicts, f, indent=2)

        # Save results (metrics and optionally trajectories)
        result_dicts = []
        for r in results:
            rd = {
                "patient_id": r.patient_id,
                "success": r.success,
                "error_message": r.error_message,
                "metrics": r.metrics,
            }
            result_dicts.append(rd)

        with open(ckpt_dir / "results.json", "w") as f:
            json.dump(result_dicts, f, indent=2, default=str)

        # Save trajectories if requested
        if save_trajectories:
            traj_dir = ckpt_dir / "trajectories"
            traj_dir.mkdir(exist_ok=True)

            for r in results:
                if r.success and r.sim_result is not None:
                    np.savez_compressed(
                        traj_dir / f"patient_{r.patient_id:04d}.npz",
                        t=r.sim_result.t,
                        y=r.sim_result.y,
                    )

        # Save metadata
        n_success = sum(1 for r in results if r.success)
        meta = CheckpointMetadata(
            checkpoint_id=ckpt_id,
            created_at=datetime.now().isoformat(),
            description=description,
            checkpoint_type="cohort",
            n_items=len(results),
            status="completed",
            tags={
                **(tags or {}),
                "n_success": str(n_success),
                "n_failed": str(len(results) - n_success),
            },
        )

        with open(ckpt_dir / "metadata.json", "w") as f:
            json.dump(meta.to_dict(), f, indent=2)

        return ckpt_id

    def load_cohort_metrics(self, checkpoint_id: str) -> List[Dict[str, Any]]:
        """
        Load cohort metrics (without trajectories) from a checkpoint.

        Returns:
            list of dicts with patient_id, success, metrics, etc.
        """
        ckpt_dir = self._get_dir(checkpoint_id)
        with open(ckpt_dir / "results.json", "r") as f:
            return json.load(f)

    def load_cohort_trajectories(self, checkpoint_id: str) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Load cohort trajectories from a checkpoint.

        Returns:
            dict mapping patient_id to {"t": array, "y": array}.
        """
        ckpt_dir = self._get_dir(checkpoint_id)
        traj_dir = ckpt_dir / "trajectories"

        if not traj_dir.exists():
            raise FileNotFoundError(f"No trajectories in checkpoint: {checkpoint_id}")

        trajectories = {}
        for f in traj_dir.glob("patient_*.npz"):
            pid = int(f.stem.split("_")[1])
            data = np.load(f)
            trajectories[pid] = {"t": data["t"], "y": data["y"]}

        return trajectories

    # -------------------------------------------------------------------
    # Save arbitrary data
    # -------------------------------------------------------------------

    def save_custom(
        self,
        data: Dict[str, Any],
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Save arbitrary JSON-serializable data to a checkpoint.

        Parameters:
            data: dict of data to save.
            description: human-readable description.
            tags: optional key-value tags.

        Returns:
            checkpoint_id.
        """
        ckpt_id = self._generate_id("custom")
        ckpt_dir = self._get_dir(ckpt_id)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        with open(ckpt_dir / "data.json", "w") as f:
            json.dump(data, f, indent=2, default=str)

        meta = CheckpointMetadata(
            checkpoint_id=ckpt_id,
            created_at=datetime.now().isoformat(),
            description=description,
            checkpoint_type="custom",
            n_items=len(data),
            status="completed",
            tags=tags or {},
        )

        with open(ckpt_dir / "metadata.json", "w") as f:
            json.dump(meta.to_dict(), f, indent=2)

        return ckpt_id

    def load_custom(self, checkpoint_id: str) -> Dict[str, Any]:
        """Load custom data from a checkpoint."""
        ckpt_dir = self._get_dir(checkpoint_id)
        with open(ckpt_dir / "data.json", "r") as f:
            return json.load(f)

    # -------------------------------------------------------------------
    # Checkpoint management
    # -------------------------------------------------------------------

    def list_checkpoints(
        self,
        checkpoint_type: Optional[str] = None,
    ) -> List[CheckpointMetadata]:
        """
        List all checkpoints, optionally filtered by type.

        Parameters:
            checkpoint_type: if provided, filter to this type.

        Returns:
            list of CheckpointMetadata objects, sorted by creation time.
        """
        checkpoints = []

        for d in self.base_dir.iterdir():
            if not d.is_dir():
                continue
            meta_file = d / "metadata.json"
            if not meta_file.exists():
                continue

            try:
                with open(meta_file, "r") as f:
                    meta_dict = json.load(f)
                meta = CheckpointMetadata.from_dict(meta_dict)

                if checkpoint_type is None or meta.checkpoint_type == checkpoint_type:
                    checkpoints.append(meta)
            except (json.JSONDecodeError, KeyError):
                continue

        # Sort by creation time (newest first)
        checkpoints.sort(key=lambda m: m.created_at, reverse=True)
        return checkpoints

    def get_metadata(self, checkpoint_id: str) -> CheckpointMetadata:
        """Get metadata for a specific checkpoint."""
        ckpt_dir = self._get_dir(checkpoint_id)
        meta_file = ckpt_dir / "metadata.json"

        if not meta_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")

        with open(meta_file, "r") as f:
            return CheckpointMetadata.from_dict(json.load(f))

    def delete_checkpoint(self, checkpoint_id: str) -> None:
        """Delete a checkpoint and all its files."""
        ckpt_dir = self._get_dir(checkpoint_id)
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)

    def get_size(self, checkpoint_id: str) -> int:
        """Get the total size of a checkpoint in bytes."""
        ckpt_dir = self._get_dir(checkpoint_id)
        total = 0
        for f in ckpt_dir.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total

    def cleanup(self, max_age_days: float = 7.0) -> int:
        """
        Delete checkpoints older than max_age_days.

        Returns:
            number of checkpoints deleted.
        """
        cutoff = datetime.now().timestamp() - (max_age_days * 86400)
        deleted = 0

        for meta in self.list_checkpoints():
            try:
                created = datetime.fromisoformat(meta.created_at).timestamp()
                if created < cutoff:
                    self.delete_checkpoint(meta.checkpoint_id)
                    deleted += 1
            except (ValueError, OSError):
                continue

        return deleted


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def save_simulation_checkpoint(
    result: SimulationResult,
    base_dir: str = "checkpoints",
    description: str = "",
    tags: Optional[Dict[str, str]] = None,
) -> str:
    """Save a simulation result to a checkpoint. Returns checkpoint_id."""
    mgr = CheckpointManager(base_dir)
    return mgr.save_simulation(result, description, tags)


def load_simulation_checkpoint(
    checkpoint_id: str,
    base_dir: str = "checkpoints",
) -> SimulationResult:
    """Load a simulation result from a checkpoint."""
    mgr = CheckpointManager(base_dir)
    return mgr.load_simulation(checkpoint_id)


def save_cohort_checkpoint(
    results: List[PatientResult],
    patients: List[VirtualPatient],
    base_dir: str = "checkpoints",
    description: str = "",
    tags: Optional[Dict[str, str]] = None,
) -> str:
    """Save cohort results to a checkpoint. Returns checkpoint_id."""
    mgr = CheckpointManager(base_dir)
    return mgr.save_cohort(results, patients, description, tags)


def list_all_checkpoints(base_dir: str = "checkpoints") -> List[CheckpointMetadata]:
    """List all checkpoints."""
    mgr = CheckpointManager(base_dir)
    return mgr.list_checkpoints()
