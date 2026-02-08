"""Unit tests for the ground-truth report module."""

import json
import os

import pytest

from src.alfred.gt_report import (
    FailureCategory,
    TaskResult,
    EvaluationReport,
    categorize_failure,
    generate_report,
    save_report,
    print_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def successful_result():
    return TaskResult(
        task_id="trial_T20190907_000001",
        task_type="pick_and_place_simple",
        task_description="Put a mug on the desk",
        entry_index=0,
        success=True,
        executed_steps=["find a mug", "pick up the mug", "find a desk", "put down the mug"],
        total_steps=4,
        goal_satisfied=True,
        scene_name="FloorPlan301",
    )


@pytest.fixture
def failed_result():
    return TaskResult(
        task_id="trial_T20190907_000002",
        task_type="look_at_obj_in_light",
        task_description="Look at pen under lamp",
        entry_index=1,
        success=False,
        executed_steps=["find a pen"],
        total_steps=4,
        goal_satisfied=False,
        scene_name="FloorPlan201",
        failure_step=2,
        failure_action="pick up the pen",
        failure_message="Cannot find Pen to pick up",
        failure_category="object_not_found",
    )


@pytest.fixture
def mixed_results(successful_result, failed_result):
    return [successful_result, failed_result]


@pytest.fixture
def sample_config():
    return {
        "eval_portion_in_percent": 10,
        "random_seed": 42,
        "gt_data_file": "resource/alfred_examples_for_prompt.json",
        "x_display": "0",
    }


# ---------------------------------------------------------------------------
# T015: Tests for categorize_failure
# ---------------------------------------------------------------------------

class TestCategorizeFailure:
    def test_object_not_found(self):
        assert categorize_failure("Cannot find mug") == "object_not_found"
        assert categorize_failure("Cannot find mug to pick up") == "object_not_found"
        assert categorize_failure("Cannot find DeskLamp to turn on") == "object_not_found"
        assert categorize_failure("Cannot find Knife to slice") == "object_not_found"

    def test_navigation_failure(self):
        assert categorize_failure("Cannot move to mug") == "navigation_failure"
        assert categorize_failure("Cannot move to DeskLamp") == "navigation_failure"

    def test_inventory_error(self):
        assert categorize_failure("Robot is not holding any object") == "inventory_error"
        assert categorize_failure("Robot is currently holding Mug") == "inventory_error"

    def test_visibility_error(self):
        assert categorize_failure("mug is not visible because it is in Fridge") == "visibility_error"
        assert categorize_failure("Pen is not visible because it is in Drawer") == "visibility_error"

    def test_interaction_failure(self):
        assert categorize_failure("Open action failed") == "interaction_failure"
        assert categorize_failure("Close action failed") == "interaction_failure"
        assert categorize_failure("Slice action failed") == "interaction_failure"
        assert categorize_failure("Turn on action failed") == "interaction_failure"
        assert categorize_failure("Turn off action failed") == "interaction_failure"
        assert categorize_failure("Drop action failed") == "interaction_failure"
        assert categorize_failure("Putting the object on CounterTop failed") == "interaction_failure"
        assert categorize_failure("put down failed") == "interaction_failure"

    def test_exception(self):
        assert categorize_failure("Exception: KeyError('missing_key')") == "exception"
        assert categorize_failure("Exception: RuntimeError('crash')") == "exception"

    def test_unknown(self):
        assert categorize_failure("some random error message") == "unknown"
        assert categorize_failure("") == "unknown"

    def test_case_insensitive(self):
        assert categorize_failure("CANNOT FIND mug") == "object_not_found"
        assert categorize_failure("Robot Is Not Holding Any Object") == "inventory_error"


# ---------------------------------------------------------------------------
# T016: Tests for failure detail capture in TaskResult
# ---------------------------------------------------------------------------

class TestTaskResultFailureDetails:
    def test_success_has_no_failure_details(self, successful_result):
        assert successful_result.success is True
        assert successful_result.failure_step is None
        assert successful_result.failure_action is None
        assert successful_result.failure_message is None
        assert successful_result.failure_category is None

    def test_failure_has_all_details(self, failed_result):
        assert failed_result.success is False
        assert failed_result.failure_step == 2
        assert failed_result.failure_action == "pick up the pen"
        assert failed_result.failure_message == "Cannot find Pen to pick up"
        assert failed_result.failure_category == "object_not_found"

    def test_failure_step_is_1_indexed(self, failed_result):
        assert failed_result.failure_step >= 1


# ---------------------------------------------------------------------------
# T020: Tests for generate_report
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_aggregate_stats(self, mixed_results, sample_config):
        report = generate_report(mixed_results, sample_config)
        assert report.total_evaluated == 2
        assert report.total_success == 1
        assert report.total_failure == 1
        assert report.success_rate == 50.0

    def test_by_task_type(self, mixed_results, sample_config):
        report = generate_report(mixed_results, sample_config)
        assert "pick_and_place_simple" in report.by_task_type
        assert "look_at_obj_in_light" in report.by_task_type
        pnp = report.by_task_type["pick_and_place_simple"]
        assert pnp["total"] == 1
        assert pnp["success"] == 1
        assert pnp["success_rate"] == 100.0

    def test_by_failure_category(self, mixed_results, sample_config):
        report = generate_report(mixed_results, sample_config)
        assert "object_not_found" in report.by_failure_category
        assert report.by_failure_category["object_not_found"] == 1

    def test_empty_results(self, sample_config):
        report = generate_report([], sample_config)
        assert report.total_evaluated == 0
        assert report.success_rate == 0.0

    def test_all_success(self, successful_result, sample_config):
        report = generate_report([successful_result] * 3, sample_config)
        assert report.success_rate == 100.0
        assert report.by_failure_category == {}

    def test_config_preserved(self, mixed_results, sample_config):
        report = generate_report(mixed_results, sample_config)
        assert report.config["random_seed"] == 42
        assert report.config["eval_portion_in_percent"] == 10


# ---------------------------------------------------------------------------
# T021: Tests for save_report
# ---------------------------------------------------------------------------

class TestSaveReport:
    def test_saves_valid_json(self, mixed_results, sample_config, tmp_path):
        report = generate_report(mixed_results, sample_config)
        path = str(tmp_path / "report.json")
        save_report(report, path)

        with open(path, "r") as f:
            data = json.load(f)

        # Verify required fields from contracts/report-schema.json
        assert "timestamp" in data
        assert "config" in data
        assert "total_evaluated" in data
        assert "total_success" in data
        assert "total_failure" in data
        assert "success_rate" in data
        assert "results" in data
        assert "by_task_type" in data
        assert "by_failure_category" in data

    def test_results_format(self, mixed_results, sample_config, tmp_path):
        report = generate_report(mixed_results, sample_config)
        path = str(tmp_path / "report.json")
        save_report(report, path)

        with open(path, "r") as f:
            data = json.load(f)

        assert len(data["results"]) == 2
        result = data["results"][0]
        assert "task_id" in result
        assert "task_type" in result
        assert "success" in result
        assert "executed_steps" in result
        assert "total_steps" in result
        assert "goal_satisfied" in result
        assert "scene_name" in result

    def test_by_task_type_format(self, mixed_results, sample_config, tmp_path):
        report = generate_report(mixed_results, sample_config)
        path = str(tmp_path / "report.json")
        save_report(report, path)

        with open(path, "r") as f:
            data = json.load(f)

        for tt, stats in data["by_task_type"].items():
            assert "total" in stats
            assert "success" in stats
            assert "failure" in stats
            assert "success_rate" in stats


# ---------------------------------------------------------------------------
# T022: Tests for print_summary
# ---------------------------------------------------------------------------

class TestPrintSummary:
    def test_prints_without_error(self, mixed_results, sample_config):
        report = generate_report(mixed_results, sample_config)
        # Should not raise
        print_summary(report)

    def test_prints_with_empty_results(self, sample_config):
        report = generate_report([], sample_config)
        print_summary(report)

    def test_prints_all_success(self, successful_result, sample_config):
        report = generate_report([successful_result], sample_config)
        print_summary(report)
