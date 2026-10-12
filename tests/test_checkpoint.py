"""
Tests for the checkpoint/resume module (checkpoint.py).

Covers:
- CheckpointManager creation
- Save/load simulation results
- Save/load cohort results
- Save/load custom data
- Checkpoint listing and metadata
- Checkpoint deletion
- Size calculation
"""

import numpy as np
import pytest
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.checkpoint import (
    CheckpointManager,
    CheckpointMetadata,
    save_simulation_checkpoint,
    load_simulation_checkpoint,
    save_cohort_checkpoint,
    list_all_checkpoints,
)
from src.core.simulation import SimulationResult
from src.utils.parallel_sim import (
    VirtualPatient,
    PatientResult,
    generate_cohort,
    run_cohort_sequential,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ckpt_dir(tmp_path):
    """Temporary checkpoint directory."""
    return str(tmp_path / "checkpoints")


@pytest.fixture
def mgr(ckpt_dir):
    """CheckpointManager with temporary directory."""
    return CheckpointManager(ckpt_dir)


@pytest.fixture
def sample_sim_result():
    """A simple SimulationResult for testing."""
    t = np.linspace(0, 96, 100)
    y = np.random.rand(100, 11)
    state_names = [
        "A_central", "A_peripheral", "A_absorption", "A_effect",
        "B_rep", "B_pers", "B_SCV", "N_eff", "Damage", "IL6", "TNF",
    ]
    params = {"drug_class": "cidal", "weight": 70.0}
    return SimulationResult(t, y, state_names, params)


@pytest.fixture
def sample_cohort():
    """A small cohort with results."""
    patients = generate_cohort(n_patients=3, seed=42)
    results = run_cohort_sequential(patients)
    return patients, results


# ---------------------------------------------------------------------------
# CheckpointMetadata tests
# ---------------------------------------------------------------------------

class TestCheckpointMetadata:

    def test_creation(self):
        meta = CheckpointMetadata(
            checkpoint_id="test_001",
            created_at="2026-04-25T10:00:00",
            description="Test checkpoint",
            checkpoint_type="simulation",
            n_items=100,
            status="completed",
        )
        assert meta.checkpoint_id == "test_001"
        assert meta.status == "completed"

    def test_to_dict(self):
        meta = CheckpointMetadata(
            checkpoint_id="test_001",
            created_at="2026-04-25T10:00:00",
            description="Test",
            checkpoint_type="simulation",
        )
        d = meta.to_dict()
        assert d["checkpoint_id"] == "test_001"
        assert d["checkpoint_type"] == "simulation"

    def test_from_dict(self):
        d = {
            "checkpoint_id": "test_002",
            "created_at": "2026-04-25T10:00:00",
            "description": "Test",
            "checkpoint_type": "cohort",
            "n_items": 50,
            "status": "completed",
            "tags": {"key": "value"},
        }
        meta = CheckpointMetadata.from_dict(d)
        assert meta.checkpoint_id == "test_002"
        assert meta.tags["key"] == "value"


# ---------------------------------------------------------------------------
# CheckpointManager creation tests
# ---------------------------------------------------------------------------

class TestCheckpointManager:

    def test_creates_directory(self, ckpt_dir):
        mgr = CheckpointManager(ckpt_dir)
        assert os.path.isdir(ckpt_dir)

    def test_existing_directory(self, ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
        mgr = CheckpointManager(ckpt_dir)
        assert os.path.isdir(ckpt_dir)


# ---------------------------------------------------------------------------
# Save/load simulation tests
# ---------------------------------------------------------------------------

class TestSaveLoadSimulation:

    def test_save_and_load(self, mgr, sample_sim_result):
        ckpt_id = mgr.save_simulation(sample_sim_result, description="Test sim")
        assert ckpt_id.startswith("sim_")

        loaded = mgr.load_simulation(ckpt_id)
        np.testing.assert_array_equal(loaded.t, sample_sim_result.t)
        np.testing.assert_array_equal(loaded.y, sample_sim_result.y)
        assert loaded.state_names == sample_sim_result.state_names

    def test_metadata_exists(self, mgr, sample_sim_result):
        ckpt_id = mgr.save_simulation(sample_sim_result, description="Test")
        meta = mgr.get_metadata(ckpt_id)
        assert meta.checkpoint_type == "simulation"
        assert meta.status == "completed"
        assert meta.n_items == len(sample_sim_result.t)

    def test_load_nonexistent_raises(self, mgr):
        with pytest.raises(FileNotFoundError):
            mgr.load_simulation("nonexistent_ckpt_id")

    def test_with_tags(self, mgr, sample_sim_result):
        ckpt_id = mgr.save_simulation(
            sample_sim_result,
            description="Tagged",
            tags={"drug": "meropenem", "dose": "1000mg"},
        )
        meta = mgr.get_metadata(ckpt_id)
        assert meta.tags["drug"] == "meropenem"


# ---------------------------------------------------------------------------
# Convenience function tests
# ---------------------------------------------------------------------------

class TestConvenienceFunctions:

    def test_save_and_load_checkpoint(self, tmp_path, sample_sim_result):
        base_dir = str(tmp_path / "convenience_checkpoints")
        ckpt_id = save_simulation_checkpoint(
            sample_sim_result,
            base_dir=base_dir,
            description="Convenience test",
        )
        loaded = load_simulation_checkpoint(ckpt_id, base_dir=base_dir)
        np.testing.assert_array_equal(loaded.t, sample_sim_result.t)


# ---------------------------------------------------------------------------
# Save/load cohort tests
# ---------------------------------------------------------------------------

class TestSaveLoadCohort:

    def test_save_cohort(self, mgr, sample_cohort):
        patients, results = sample_cohort
        ckpt_id = mgr.save_cohort(results, patients, description="Test cohort")
        assert ckpt_id.startswith("cohort_")

    def test_load_cohort_metrics(self, mgr, sample_cohort):
        patients, results = sample_cohort
        ckpt_id = mgr.save_cohort(results, patients, description="Test")

        metrics = mgr.load_cohort_metrics(ckpt_id)
        assert len(metrics) == 3
        assert all("patient_id" in m for m in metrics)

    def test_load_cohort_trajectories(self, mgr, sample_cohort):
        patients, results = sample_cohort
        ckpt_id = mgr.save_cohort(
            results, patients, description="Test", save_trajectories=True,
        )

        trajs = mgr.load_cohort_trajectories(ckpt_id)
        assert len(trajs) == 3
        for pid, traj in trajs.items():
            assert "t" in traj
            assert "y" in traj
            assert len(traj["t"]) > 0

    def test_save_cohort_no_trajectories(self, mgr, sample_cohort):
        patients, results = sample_cohort
        ckpt_id = mgr.save_cohort(
            results, patients, description="No traj", save_trajectories=False,
        )

        with pytest.raises(FileNotFoundError):
            mgr.load_cohort_trajectories(ckpt_id)

    def test_cohort_metadata(self, mgr, sample_cohort):
        patients, results = sample_cohort
        ckpt_id = mgr.save_cohort(results, patients, description="Test")

        meta = mgr.get_metadata(ckpt_id)
        assert meta.checkpoint_type == "cohort"
        assert meta.n_items == 3

    def test_save_cohort_convenience(self, tmp_path, sample_cohort):
        patients, results = sample_cohort
        base_dir = str(tmp_path / "cohort_checkpoints")
        ckpt_id = save_cohort_checkpoint(
            results, patients, base_dir=base_dir, description="Conv test",
        )
        assert ckpt_id.startswith("cohort_")


# ---------------------------------------------------------------------------
# Custom data tests
# ---------------------------------------------------------------------------

class TestCustomData:

    def test_save_and_load(self, mgr):
        data = {"key1": "value1", "key2": [1, 2, 3], "key3": {"nested": True}}
        ckpt_id = mgr.save_custom(data, description="Custom test")

        loaded = mgr.load_custom(ckpt_id)
        assert loaded["key1"] == "value1"
        assert loaded["key2"] == [1, 2, 3]
        assert loaded["key3"]["nested"] is True


# ---------------------------------------------------------------------------
# Checkpoint listing tests
# ---------------------------------------------------------------------------

class TestListCheckpoints:

    def test_list_empty(self, mgr):
        checkpoints = mgr.list_checkpoints()
        assert len(checkpoints) == 0

    def test_list_after_save(self, mgr, sample_sim_result):
        import time
        mgr.save_simulation(sample_sim_result, description="First")
        time.sleep(0.01)  # Ensure distinct timestamps
        mgr.save_simulation(sample_sim_result, description="Second")

        checkpoints = mgr.list_checkpoints()
        assert len(checkpoints) == 2

    def test_list_filter_by_type(self, mgr, sample_sim_result, sample_cohort):
        mgr.save_simulation(sample_sim_result, description="Sim")

        patients, results = sample_cohort
        mgr.save_cohort(results, patients, description="Cohort")

        sims = mgr.list_checkpoints(checkpoint_type="simulation")
        cohorts = mgr.list_checkpoints(checkpoint_type="cohort")

        assert len(sims) == 1
        assert len(cohorts) == 1

    def test_list_sorted_newest_first(self, mgr, sample_sim_result):
        import time
        mgr.save_simulation(sample_sim_result, description="First")
        time.sleep(0.01)  # Ensure distinct timestamps
        mgr.save_simulation(sample_sim_result, description="Second")

        checkpoints = mgr.list_checkpoints()
        # Newest first
        assert checkpoints[0].created_at >= checkpoints[1].created_at

    def test_list_all_checkpoints_convenience(self, tmp_path, sample_sim_result):
        base_dir = str(tmp_path / "list_checkpoints")
        mgr = CheckpointManager(base_dir)
        mgr.save_simulation(sample_sim_result, description="Test")

        checkpoints = list_all_checkpoints(base_dir)
        assert len(checkpoints) == 1


# ---------------------------------------------------------------------------
# Delete tests
# ---------------------------------------------------------------------------

class TestDeleteCheckpoint:

    def test_delete(self, mgr, sample_sim_result):
        ckpt_id = mgr.save_simulation(sample_sim_result, description="To delete")
        assert os.path.isdir(mgr._get_dir(ckpt_id))

        mgr.delete_checkpoint(ckpt_id)
        assert not os.path.isdir(mgr._get_dir(ckpt_id))

    def test_delete_nonexistent(self, mgr):
        # Should not raise
        mgr.delete_checkpoint("nonexistent_id")

    def test_list_after_delete(self, mgr, sample_sim_result):
        ckpt_id = mgr.save_simulation(sample_sim_result, description="To delete")
        mgr.delete_checkpoint(ckpt_id)

        checkpoints = mgr.list_checkpoints()
        assert len(checkpoints) == 0


# ---------------------------------------------------------------------------
# Size tests
# ---------------------------------------------------------------------------

class TestGetSize:

    def test_size_positive(self, mgr, sample_sim_result):
        ckpt_id = mgr.save_simulation(sample_sim_result, description="Size test")
        size = mgr.get_size(ckpt_id)
        assert size > 0


# ---------------------------------------------------------------------------
# Metadata tests
# ---------------------------------------------------------------------------

class TestGetMetadata:

    def test_get_metadata(self, mgr, sample_sim_result):
        ckpt_id = mgr.save_simulation(sample_sim_result, description="Meta test")
        meta = mgr.get_metadata(ckpt_id)
        assert isinstance(meta, CheckpointMetadata)
        assert meta.description == "Meta test"

    def test_get_metadata_nonexistent(self, mgr):
        with pytest.raises(FileNotFoundError):
            mgr.get_metadata("nonexistent_id")
