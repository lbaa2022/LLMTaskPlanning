"""Unit tests for the ground-truth evaluator module.

These tests cover pure logic functions (data loading, validation, selection)
and do NOT require AI2-THOR or the simulator to be installed.
"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# Mock heavy AI2-THOR dependencies so we can import the evaluator module
# without needing the simulator installed.
_thor_mocks = {}
for mod_name in [
    "env", "env.thor_env",
    "gen", "gen.constants", "gen.utils", "gen.utils.game_util",
    "alfred", "alfred.utils", "alfred.data", "alfred.data.preprocess",
    "scipy", "scipy.spatial",
]:
    _thor_mocks[mod_name] = MagicMock()
with patch.dict(sys.modules, _thor_mocks):
    from src.alfred.gt_evaluator import (
        load_gt_entries,
        build_split_index,
        validate_portion,
        select_entries,
    )
    from src.alfred.gt_report import GroundTruthEntry, SplitEntry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_gt_data():
    """Minimal GT entries matching alfred_examples_for_prompt.json format."""
    return [
        {
            "task id": "trial_T20190907_000001_000001",
            "task type": "pick_and_place_simple",
            "task description": "Put a mug on the desk",
            "step description": "pick up a mug then put it on the desk",
            "NL steps": ["find a mug", "pick up the mug", "find a desk", "put down the mug on the desk"],
        },
        {
            "task id": "trial_T20190907_000002_000002",
            "task type": "look_at_obj_in_light",
            "task description": "Examine a pen under a lamp",
            "step description": "find a pen and turn on the lamp",
            "NL steps": ["find a pen", "pick up the pen", "find a desk lamp", "turn on the desk lamp"],
        },
        {
            "task id": "trial_T20190907_000003_000003",
            "task type": "pick_and_place_simple",
            "task description": "Empty NL steps entry",
            "step description": "",
            "NL steps": [],
        },
    ]


@pytest.fixture
def gt_data_file(sample_gt_data, tmp_path):
    """Write sample GT data to a temp file."""
    path = tmp_path / "gt_examples.json"
    path.write_text(json.dumps(sample_gt_data))
    return str(path)


@pytest.fixture
def sample_splits_data():
    """Minimal splits data matching oct21.json format."""
    return {
        "train": [
            {"task": "pick_and_place_simple-Mug-None-Desk-1/trial_T20190907_000001_000001", "repeat_idx": 0},
            {"task": "pick_and_place_simple-Mug-None-Desk-1/trial_T20190907_000001_000001", "repeat_idx": 1},
            {"task": "look_at_obj_in_light-Pen-None-DeskLamp-2/trial_T20190907_000002_000002", "repeat_idx": 0},
        ],
        "valid_seen": [],
        "valid_unseen": [],
    }


@pytest.fixture
def splits_file(sample_splits_data, tmp_path):
    """Write sample splits data to a temp file."""
    path = tmp_path / "splits.json"
    path.write_text(json.dumps(sample_splits_data))
    return str(path)


# ---------------------------------------------------------------------------
# T006: Tests for load_gt_entries
# ---------------------------------------------------------------------------

class TestLoadGtEntries:
    def test_loads_valid_entries(self, gt_data_file):
        entries = load_gt_entries(gt_data_file)
        assert len(entries) == 2  # third entry has empty NL steps, should be skipped
        assert entries[0].task_id == "trial_T20190907_000001_000001"
        assert entries[0].task_type == "pick_and_place_simple"
        assert entries[0].entry_index == 0
        assert len(entries[0].nl_steps) == 4

    def test_skips_empty_nl_steps(self, gt_data_file):
        entries = load_gt_entries(gt_data_file)
        task_ids = [e.task_id for e in entries]
        assert "trial_T20190907_000003_000003" not in task_ids

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_gt_entries(str(tmp_path / "nonexistent.json"))

    def test_corrupted_json_raises(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{invalid json")
        with pytest.raises(ValueError, match="Failed to parse"):
            load_gt_entries(str(bad_file))

    def test_preserves_entry_index(self, gt_data_file):
        entries = load_gt_entries(gt_data_file)
        # First valid entry has index 0, second has index 1
        assert entries[0].entry_index == 0
        assert entries[1].entry_index == 1

    def test_skips_missing_task_id(self, tmp_path):
        data = [{"task type": "test", "task description": "test", "NL steps": ["step1"]}]
        path = tmp_path / "no_id.json"
        path.write_text(json.dumps(data))
        entries = load_gt_entries(str(path))
        assert len(entries) == 0


# ---------------------------------------------------------------------------
# T006: Tests for build_split_index
# ---------------------------------------------------------------------------

class TestBuildSplitIndex:
    def test_builds_index(self, splits_file):
        index = build_split_index(splits_file)
        assert len(index) == 2  # two unique trial IDs
        assert "trial_T20190907_000001_000001" in index
        assert "trial_T20190907_000002_000002" in index

    def test_first_occurrence_kept(self, splits_file):
        index = build_split_index(splits_file)
        entry = index["trial_T20190907_000001_000001"]
        assert entry.repeat_idx == 0  # first occurrence

    def test_task_path_correct(self, splits_file):
        index = build_split_index(splits_file)
        entry = index["trial_T20190907_000001_000001"]
        assert "pick_and_place_simple" in entry.task
        assert entry.trial_id == "trial_T20190907_000001_000001"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            build_split_index(str(tmp_path / "nonexistent.json"))


# ---------------------------------------------------------------------------
# T007: Tests for validate_portion
# ---------------------------------------------------------------------------

class TestValidatePortion:
    def test_valid_portions(self):
        validate_portion(1)
        validate_portion(50)
        validate_portion(100)
        validate_portion(0.5)

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match="must be > 0"):
            validate_portion(0)

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="must be > 0"):
            validate_portion(-10)

    def test_rejects_over_100(self):
        with pytest.raises(ValueError, match="must be <= 100"):
            validate_portion(101)

    def test_rejects_large_value(self):
        with pytest.raises(ValueError, match="must be <= 100"):
            validate_portion(200)


# ---------------------------------------------------------------------------
# T008: Tests for select_entries
# ---------------------------------------------------------------------------

class TestSelectEntries:
    @pytest.fixture
    def entries(self):
        return [
            GroundTruthEntry(
                task_id=f"trial_T2019_{i:04d}",
                task_type="test",
                task_description=f"Task {i}",
                nl_steps=["step1"],
                entry_index=i,
            )
            for i in range(100)
        ]

    def test_correct_count(self, entries):
        selected = select_entries(entries, 10, seed=42)
        assert len(selected) == 10

    def test_at_least_one(self, entries):
        # 0.1% of 100 = 0.1 → should round up to 1
        selected = select_entries(entries, 0.1, seed=42)
        assert len(selected) >= 1

    def test_full_portion(self, entries):
        selected = select_entries(entries, 100, seed=42)
        assert len(selected) == 100

    def test_rounding(self, entries):
        # 3% of 100 = 3
        selected = select_entries(entries, 3, seed=42)
        assert len(selected) == 3

    # T027: Reproducibility — same seed same result
    def test_same_seed_same_selection(self, entries):
        sel1 = select_entries(entries, 10, seed=42)
        sel2 = select_entries(entries, 10, seed=42)
        assert [e.entry_index for e in sel1] == [e.entry_index for e in sel2]

    # T028: Different seeds different result
    def test_different_seed_different_selection(self, entries):
        sel1 = select_entries(entries, 10, seed=42)
        sel2 = select_entries(entries, 10, seed=99)
        ids1 = [e.entry_index for e in sel1]
        ids2 = [e.entry_index for e in sel2]
        assert ids1 != ids2

    # T029: Default seed behavior (tested via config, but ensure function works with any int)
    def test_seed_zero_works(self, entries):
        selected = select_entries(entries, 10, seed=0)
        assert len(selected) == 10


# ---------------------------------------------------------------------------
# T009: Tests for GroundTruthEvaluator.__init__ (config parsing)
# ---------------------------------------------------------------------------

class TestGroundTruthEvaluatorInit:
    """Test that __init__ properly validates config parameters.

    Note: Full __init__ test requires Hydra config objects which are complex
    to mock. We test the individual validation functions instead, which is
    what __init__ calls.
    """

    def test_validate_portion_called_on_init(self):
        """validate_portion should raise for invalid portion."""
        with pytest.raises(ValueError):
            validate_portion(0)
        with pytest.raises(ValueError):
            validate_portion(150)

    def test_load_gt_entries_called_on_init(self, tmp_path):
        """load_gt_entries should raise for missing file."""
        with pytest.raises(FileNotFoundError):
            load_gt_entries(str(tmp_path / "missing.json"))
