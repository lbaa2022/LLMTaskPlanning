"""Ground-truth evaluation report: data types, failure categorization, and report generation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FailureCategory(str, Enum):
    OBJECT_NOT_FOUND = "object_not_found"
    NAVIGATION_FAILURE = "navigation_failure"
    INVENTORY_ERROR = "inventory_error"
    VISIBILITY_ERROR = "visibility_error"
    INTERACTION_FAILURE = "interaction_failure"
    EXCEPTION = "exception"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GroundTruthEntry:
    task_id: str
    task_type: str
    task_description: str
    nl_steps: list[str]
    entry_index: int


@dataclass
class SplitEntry:
    task: str
    repeat_idx: int

    @property
    def trial_id(self) -> str:
        return self.task.split("/")[-1]


@dataclass
class TaskResult:
    task_id: str
    task_type: str
    task_description: str
    entry_index: int
    success: bool
    executed_steps: list[str]
    total_steps: int
    goal_satisfied: bool
    scene_name: str
    failure_step: Optional[int] = None
    failure_action: Optional[str] = None
    failure_message: Optional[str] = None
    failure_category: Optional[str] = None


@dataclass
class EvaluationReport:
    timestamp: str
    config: dict
    total_evaluated: int
    total_success: int
    total_failure: int
    success_rate: float
    results: list[TaskResult]
    by_task_type: dict
    by_failure_category: dict


# ---------------------------------------------------------------------------
# Failure categorization
# ---------------------------------------------------------------------------

def categorize_failure(message: str) -> str:
    """Categorize a failure message into a FailureCategory value.

    Uses substring/prefix matching on error messages returned by
    ThorConnector.llm_skill_interact().
    """
    if not message:
        return FailureCategory.UNKNOWN.value

    msg = message.lower()

    # Navigation failure
    if "cannot move to" in msg:
        return FailureCategory.NAVIGATION_FAILURE.value

    # Visibility error (must check before object_not_found since both may contain "not visible")
    if "is not visible because it is in" in msg:
        return FailureCategory.VISIBILITY_ERROR.value

    # Inventory error
    if "robot is not holding any object" in msg or "robot is currently holding" in msg:
        return FailureCategory.INVENTORY_ERROR.value

    # Object not found
    if "cannot find" in msg:
        return FailureCategory.OBJECT_NOT_FOUND.value

    # Interaction failure
    interaction_keywords = [
        "action failed",
        "putting the object on",
        "put down failed",
    ]
    for keyword in interaction_keywords:
        if keyword in msg:
            return FailureCategory.INTERACTION_FAILURE.value

    # Exception
    if msg.startswith("exception:"):
        return FailureCategory.EXCEPTION.value

    return FailureCategory.UNKNOWN.value


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    results: list[TaskResult],
    config: dict,
) -> EvaluationReport:
    """Build an EvaluationReport from a list of TaskResult objects."""
    total = len(results)
    successes = sum(1 for r in results if r.success)
    failures = total - successes
    success_rate = (successes / total * 100) if total > 0 else 0.0

    # Per-task-type breakdown
    by_task_type: dict[str, dict] = {}
    for r in results:
        tt = r.task_type
        if tt not in by_task_type:
            by_task_type[tt] = {"total": 0, "success": 0, "failure": 0, "success_rate": 0.0}
        by_task_type[tt]["total"] += 1
        if r.success:
            by_task_type[tt]["success"] += 1
        else:
            by_task_type[tt]["failure"] += 1

    for tt_stats in by_task_type.values():
        tt_stats["success_rate"] = (
            tt_stats["success"] / tt_stats["total"] * 100 if tt_stats["total"] > 0 else 0.0
        )

    # Per-failure-category breakdown
    by_failure_category: dict[str, int] = {}
    for r in results:
        if not r.success and r.failure_category:
            cat = r.failure_category
            by_failure_category[cat] = by_failure_category.get(cat, 0) + 1

    return EvaluationReport(
        timestamp=datetime.now().isoformat(),
        config=config,
        total_evaluated=total,
        total_success=successes,
        total_failure=failures,
        success_rate=round(success_rate, 2),
        results=results,
        by_task_type=by_task_type,
        by_failure_category=by_failure_category,
    )


# ---------------------------------------------------------------------------
# Report serialization
# ---------------------------------------------------------------------------

def _task_result_to_dict(r: TaskResult) -> dict:
    d = asdict(r)
    return d


def save_report(report: EvaluationReport, output_path: str) -> None:
    """Save an EvaluationReport as a JSON file matching contracts/report-schema.json."""
    data = {
        "timestamp": report.timestamp,
        "config": report.config,
        "total_evaluated": report.total_evaluated,
        "total_success": report.total_success,
        "total_failure": report.total_failure,
        "success_rate": report.success_rate,
        "results": [_task_result_to_dict(r) for r in report.results],
        "by_task_type": report.by_task_type,
        "by_failure_category": report.by_failure_category,
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    log.info(f"Report saved to {output_path}")


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------

def print_summary(report: EvaluationReport) -> None:
    """Log a human-readable summary of the evaluation report."""
    log.info("=" * 60)
    log.info("GROUND-TRUTH EVALUATION SUMMARY")
    log.info("=" * 60)
    log.info(f"Total evaluated: {report.total_evaluated}")
    log.info(f"Success: {report.total_success}  |  Failure: {report.total_failure}")
    log.info(f"Success rate: {report.success_rate:.2f}%")
    log.info("")

    # Per task type
    if report.by_task_type:
        log.info("--- Success Rate by Task Type ---")
        for tt, stats in sorted(report.by_task_type.items()):
            log.info(
                f"  {tt}: {stats['success_rate']:.1f}% "
                f"({stats['success']}/{stats['total']})"
            )
        log.info("")

    # Per failure category
    if report.by_failure_category:
        log.info("--- Failure Breakdown by Category ---")
        for cat, count in sorted(report.by_failure_category.items(), key=lambda x: -x[1]):
            log.info(f"  {cat}: {count}")
        log.info("")

    log.info("=" * 60)
