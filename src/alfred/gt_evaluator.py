"""Ground-truth plan evaluator for ALFRED tasks.

Loads pre-defined NL action plans from the examples JSON, executes them in
AI2-THOR via ThorConnector, and produces a structured evaluation report.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
import datetime
import traceback

from tqdm import tqdm

from src.alfred.gt_report import (
    GroundTruthEntry,
    SplitEntry,
    TaskResult,
    categorize_failure,
    generate_report,
    save_report,
    print_summary,
)
from src.alfred.thor_connector import ThorConnector
from src.alfred.utils import load_task_json, dotdict
from src.evaluator import Evaluator

log = logging.getLogger(__name__)

SPLITS_FILE = "alfred/data/splits/oct21.json"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_gt_entries(filepath: str) -> list[GroundTruthEntry]:
    """Load ground-truth entries from the examples JSON file.

    Skips entries with empty NL steps and logs a warning for each.
    Raises FileNotFoundError if the file does not exist, and
    ValueError if the file cannot be parsed as JSON.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Ground-truth data file not found: {filepath}")

    try:
        with open(filepath, "r") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse GT data file {filepath}: {e}") from e

    entries: list[GroundTruthEntry] = []
    skipped = 0
    for idx, item in enumerate(raw):
        nl_steps = item.get("NL steps", [])
        if not nl_steps:
            log.warning(f"Skipping entry {idx}: empty NL steps (task_id={item.get('task id', 'unknown')})")
            skipped += 1
            continue

        task_id = item.get("task id", "")
        task_type = item.get("task type", "")
        task_description = item.get("task description", "")

        if not task_id:
            log.warning(f"Skipping entry {idx}: missing task id")
            skipped += 1
            continue

        entries.append(
            GroundTruthEntry(
                task_id=task_id,
                task_type=task_type,
                task_description=task_description,
                nl_steps=nl_steps,
                entry_index=idx,
            )
        )

    log.info(f"Loaded {len(entries)} GT entries from {filepath} (skipped {skipped})")
    return entries


def build_split_index(splits_file: str = SPLITS_FILE) -> dict[str, SplitEntry]:
    """Build task_id → SplitEntry index from the train split for O(1) lookups.

    Returns a dict mapping trial_id (e.g. 'trial_T20190907_174127_043461')
    to the first matching SplitEntry with its full task path and repeat_idx.
    """
    if not os.path.exists(splits_file):
        raise FileNotFoundError(f"Splits file not found: {splits_file}")

    with open(splits_file, "r") as f:
        splits = json.load(f)

    train_entries = splits.get("train", [])
    index: dict[str, SplitEntry] = {}

    for entry in train_entries:
        task_path = entry["task"]
        repeat_idx = entry["repeat_idx"]
        trial_id = task_path.split("/")[-1]

        # Keep the first occurrence for each trial_id
        if trial_id not in index:
            index[trial_id] = SplitEntry(task=task_path, repeat_idx=repeat_idx)

    log.info(f"Built split index with {len(index)} unique task IDs from {len(train_entries)} train entries")
    return index


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_portion(portion: float) -> None:
    """Validate that portion is in the range (0, 100].

    Raises ValueError for invalid values.
    """
    if portion <= 0:
        raise ValueError(f"eval_portion_in_percent must be > 0, got {portion}")
    if portion > 100:
        raise ValueError(f"eval_portion_in_percent must be <= 100, got {portion}")


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_entries(
    entries: list[GroundTruthEntry],
    portion: float,
    seed: int,
) -> list[GroundTruthEntry]:
    """Randomly select a portion of GT entries with a given seed.

    Guarantees at least 1 entry is selected. Uses a local Random instance
    to avoid mutating the global random state.
    """
    n_total = len(entries)
    n_sample = max(1, round(n_total * portion / 100))
    n_sample = min(n_sample, n_total)

    rng = random.Random(seed)
    selected = rng.sample(entries, n_sample)
    log.info(f"Selected {len(selected)} / {n_total} entries ({portion}%, seed={seed})")
    return selected


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class GroundTruthEvaluator(Evaluator):
    """Evaluator for ground-truth NL action plans in AI2-THOR."""

    def __init__(self, cfg):
        self.cfg = cfg

        gt_cfg = cfg.gt
        self.gt_data_file = gt_cfg.gt_data_file
        self.portion = gt_cfg.eval_portion_in_percent
        self.seed = gt_cfg.random_seed
        self.x_display = gt_cfg.x_display
        self.save_path = cfg.out_dir

        log.info(f"GT Evaluator config: portion={self.portion}%, seed={self.seed}, data={self.gt_data_file}")

        # Validate
        validate_portion(self.portion)

        # Load data
        self.all_entries = load_gt_entries(self.gt_data_file)
        self.split_index = build_split_index()

        # Select subset
        self.selected_entries = select_entries(self.all_entries, self.portion, self.seed)

    def evaluate(self):
        """Run GT evaluation on selected entries."""
        log.info(f"Starting ground-truth evaluation: {len(self.selected_entries)} entries")
        start = time.time()

        args_dict = {
            "data": "alfred/data/json_2.1.0",
            "pframe": 300,
            "fast_epoch": False,
            "use_templated_goals": False,
            "dout": "exp/model",
            "pp_folder": "pp",
            "reward_config": "alfred/models/config/rewards.json",
            "max_steps": 1000,
        }
        model_args = dotdict(args_dict)

        env = ThorConnector(x_display=self.x_display)
        results: list[TaskResult] = []

        for entry in tqdm(self.selected_entries, desc="GT Evaluation"):
            result = self.evaluate_single_entry(entry, env, model_args)
            results.append(result)

            # Save per-task JSON
            self._save_per_task_result(result)

        # Report
        config_dict = {
            "eval_portion_in_percent": self.portion,
            "random_seed": self.seed,
            "gt_data_file": self.gt_data_file,
            "x_display": self.x_display,
        }
        report = generate_report(results, config_dict)
        report_path = os.path.join(self.save_path, "gt_evaluation_report.json")
        save_report(report, report_path)
        print_summary(report)

        elapsed = str(datetime.timedelta(seconds=int(time.time() - start)))
        log.info(f"GT evaluation complete in {elapsed}")

    def evaluate_single_entry(
        self,
        entry: GroundTruthEntry,
        env: ThorConnector,
        model_args: dotdict,
    ) -> TaskResult:
        """Execute a single GT plan entry in the simulator and return the result."""
        task_id = entry.task_id
        scene_name = "unknown"
        executed_steps: list[str] = []

        # Resolve split entry
        split_entry = self.split_index.get(task_id)
        if split_entry is None:
            log.warning(f"No split entry for task_id={task_id}, skipping execution")
            return TaskResult(
                task_id=task_id,
                task_type=entry.task_type,
                task_description=entry.task_description,
                entry_index=entry.entry_index,
                success=False,
                executed_steps=[],
                total_steps=len(entry.nl_steps),
                goal_satisfied=False,
                scene_name="unknown",
                failure_step=None,
                failure_action=None,
                failure_message="No split entry found for task_id",
                failure_category=categorize_failure("Cannot find task data"),
            )

        try:
            # Load scene data
            task_dict = {"task": split_entry.task, "repeat_idx": split_entry.repeat_idx}
            traj_data = load_task_json(task_dict, split="train")

            # Setup scene
            scene_num = traj_data["scene"]["scene_num"]
            scene_name = f"FloorPlan{scene_num}"
            object_poses = traj_data["scene"]["object_poses"]
            dirty_and_empty = traj_data["scene"]["dirty_and_empty"]
            object_toggles = traj_data["scene"]["object_toggles"]

            env.reset(scene_name)
            env.restore_scene(object_poses, object_toggles, dirty_and_empty)
            env.step(dict(traj_data["scene"]["init_action"]))
            env.set_task(traj_data, model_args, reward_type="dense")

            log.info(f"Evaluating {task_id} in {scene_name} ({len(entry.nl_steps)} steps)")

            # Execute NL steps
            for step_idx, step_text in enumerate(entry.nl_steps):
                try:
                    action_ret = env.llm_skill_interact(step_text)
                except Exception as e:
                    log.warning(f"Exception during step {step_idx + 1}: {e}")
                    return TaskResult(
                        task_id=task_id,
                        task_type=entry.task_type,
                        task_description=entry.task_description,
                        entry_index=entry.entry_index,
                        success=False,
                        executed_steps=executed_steps,
                        total_steps=len(entry.nl_steps),
                        goal_satisfied=False,
                        scene_name=scene_name,
                        failure_step=step_idx + 1,
                        failure_action=step_text,
                        failure_message=f"Exception: {e}",
                        failure_category=categorize_failure(f"Exception: {e}"),
                    )

                executed_steps.append(step_text)

                if not action_ret["success"]:
                    msg = action_ret.get("message", "")
                    log.warning(f"Step {step_idx + 1} failed: {msg}")
                    return TaskResult(
                        task_id=task_id,
                        task_type=entry.task_type,
                        task_description=entry.task_description,
                        entry_index=entry.entry_index,
                        success=False,
                        executed_steps=executed_steps,
                        total_steps=len(entry.nl_steps),
                        goal_satisfied=False,
                        scene_name=scene_name,
                        failure_step=step_idx + 1,
                        failure_action=step_text,
                        failure_message=msg,
                        failure_category=categorize_failure(msg),
                    )

                # Track reward
                env.get_transition_reward()

            # All steps executed — check goal
            goal_satisfied = env.get_goal_satisfied()
            log.info(f"Task {task_id}: all steps executed, goal_satisfied={goal_satisfied}")

            return TaskResult(
                task_id=task_id,
                task_type=entry.task_type,
                task_description=entry.task_description,
                entry_index=entry.entry_index,
                success=goal_satisfied,
                executed_steps=executed_steps,
                total_steps=len(entry.nl_steps),
                goal_satisfied=goal_satisfied,
                scene_name=scene_name,
            )

        except Exception as e:
            log.error(f"Exception evaluating {task_id}: {e}")
            log.error(traceback.format_exc())
            return TaskResult(
                task_id=task_id,
                task_type=entry.task_type,
                task_description=entry.task_description,
                entry_index=entry.entry_index,
                success=False,
                executed_steps=executed_steps,
                total_steps=len(entry.nl_steps),
                goal_satisfied=False,
                scene_name=scene_name,
                failure_step=None,
                failure_action=None,
                failure_message=f"Exception: {e}",
                failure_category=categorize_failure(f"Exception: {e}"),
            )

    def _save_per_task_result(self, result: TaskResult) -> None:
        """Save individual task result as JSON, consistent with existing format."""
        filename = f"{result.task_id}_{result.entry_index}"
        log_entry = {
            "trial": result.task_id,
            "type": result.task_type,
            "entry_index": result.entry_index,
            "goal_instr": result.task_description,
            "scene": result.scene_name,
            "success": result.success,
            "goal_satisfied": result.goal_satisfied,
            "executed_steps": result.executed_steps,
            "total_steps": result.total_steps,
        }
        if not result.success:
            log_entry["failure_step"] = result.failure_step
            log_entry["failure_action"] = result.failure_action
            log_entry["failure_message"] = result.failure_message
            log_entry["failure_category"] = result.failure_category

        path = os.path.join(self.save_path, f"{filename}.json")
        with open(path, "w") as f:
            json.dump(log_entry, f, indent=2)
