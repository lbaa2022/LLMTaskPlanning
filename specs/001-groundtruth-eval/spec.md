# Feature Specification: Ground-Truth Plan Evaluation

**Feature Branch**: `001-groundtruth-eval`
**Created**: 2026-02-08
**Status**: Draft
**Input**: User description: "Run ground-truth plans and report the success rate with causes of failures for failed plans. The ground-truth NL steps are in resource/alfred_examples_for_prompt.json (with task IDs linking to scene data in alfred/data/). It should be possible to specify how much portion of the plans will be executed e.g. 10%, 20% etc. The plans to execute should be selected at random for diversity of tasks."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run Ground-Truth Evaluation with Configurable Portion (Priority: P1)

A researcher wants to validate how well the AI2-THOR simulator executes known-correct (ground-truth) action plans from the ALFRED training dataset. The ground-truth NL steps and task IDs are sourced from `resource/alfred_examples_for_prompt.json` (17,469 entries across 5,466 unique tasks), and the corresponding scene data (object poses, toggles, initial state) is loaded from `alfred/data/` using the task ID. The researcher specifies a percentage of plans to evaluate (e.g., 10%), and the system randomly selects that portion, executes each plan in the simulator, and reports the overall success rate.

**Why this priority**: This is the core capability of the feature. Without configurable portion-based execution, the researcher cannot efficiently validate the simulator against ground-truth data at varying scales.

**Independent Test**: Can be tested by running the evaluator with a small percentage (e.g., 1%) and verifying that the correct number of tasks are selected, executed, and a success rate is reported.

**Acceptance Scenarios**:

1. **Given** the ground-truth dataset is loaded and a portion of 10% is specified, **When** the evaluation is started, **Then** approximately 1,747 plans are randomly selected from the 17,469 entries, executed in the simulator using their linked scene data, and a success rate is printed at the end.
2. **Given** no portion is specified, **When** the evaluation is started, **Then** the system defaults to evaluating 100% of the ground-truth plans.
3. **Given** a portion of 0% or a negative value is specified, **When** the evaluation is started, **Then** the system rejects the input with a clear error message.
4. **Given** a portion greater than 100% is specified, **When** the evaluation is started, **Then** the system rejects the input with a clear error message.

---

### User Story 2 - Failure Cause Reporting (Priority: P1)

A researcher wants to understand why ground-truth plans fail in the simulator. After execution completes, the system produces a detailed report that categorizes each failed plan by its failure cause (e.g., navigation failure, object not found, action precondition not met, unexpected exception) along with the specific error message and the step at which failure occurred.

**Why this priority**: Without failure diagnostics, the success rate number alone is insufficient for identifying simulator bugs, environment setup issues, or plan format mismatches. This is equally critical to the execution itself.

**Independent Test**: Can be tested by examining the output report for failed tasks and verifying each entry contains the task identifier, the failing step number, the failing action, and the categorized failure reason.

**Acceptance Scenarios**:

1. **Given** an evaluation run completes with some failed plans, **When** the report is generated, **Then** each failed plan entry includes: task identifier, instruction text, step number where failure occurred, the action that failed, and the error message from the simulator.
2. **Given** an evaluation run completes with all plans succeeding, **When** the report is generated, **Then** the report indicates 100% success and no failure entries.
3. **Given** a plan fails due to an unexpected exception (not a simulator action failure), **When** the report is generated, **Then** the exception is captured and categorized as "exception" with the traceback information.

---

### User Story 3 - Summary Report with Aggregate Statistics (Priority: P2)

A researcher wants a high-level summary of the evaluation results, including the overall success rate, breakdown of failure causes by category, breakdown by task type (e.g., pick-and-place, examine-in-light, toggle), and the total number of tasks evaluated.

**Why this priority**: Aggregate statistics help the researcher quickly identify systemic issues (e.g., navigation always failing, or a specific task type consistently failing) without manually inspecting individual failure records.

**Independent Test**: Can be tested by running a small evaluation and verifying the summary section includes overall success rate, per-category failure counts, and per-task-type success rates.

**Acceptance Scenarios**:

1. **Given** an evaluation run completes, **When** the summary report is generated, **Then** it includes: total tasks evaluated, number of successes, number of failures, overall success rate as a percentage, and a breakdown of failure causes with counts.
2. **Given** an evaluation run includes tasks of multiple types, **When** the summary report is generated, **Then** the success rate is also broken down by task type.

---

### User Story 4 - Random Selection with Reproducibility (Priority: P2)

A researcher wants the random selection of plans to be diverse (covering different task types and objects) but also reproducible by specifying a random seed.

**Why this priority**: Reproducibility is essential for scientific benchmarking. Different runs with the same seed and portion should produce identical results.

**Independent Test**: Can be tested by running the evaluation twice with the same seed and portion, and verifying that the same set of tasks is selected both times.

**Acceptance Scenarios**:

1. **Given** a random seed of 42 and a portion of 5% are specified, **When** the evaluation is run twice, **Then** the same set of tasks is selected both times.
2. **Given** no random seed is specified, **When** the evaluation is run, **Then** a default seed is used and logged so results can be reproduced later.
3. **Given** two different seeds are specified with the same portion, **When** the evaluations are run, **Then** different sets of tasks are selected (with high probability).

---

### Edge Cases

- What happens when the ground-truth data file is missing or corrupted? The system reports a clear error indicating the file path and the nature of the problem.
- How does the system handle a plan whose actions reference objects not present in the corresponding scene? The action fails in the simulator and is recorded as a failure with the simulator's error message.
- What happens if the AI2-THOR simulator crashes mid-evaluation for a single task? The exception is caught, the task is recorded as failed with the exception details, and evaluation continues with the next task.
- How does the system behave when a ground-truth plan entry has an empty NL steps list? The entry is skipped and logged as a warning.
- What happens when the specified portion results in fewer than 1 task (e.g., 0.001% of 17,469)? The system rounds up to at least 1 task or reports that the portion is too small.
- How does the system handle ground-truth entries whose task ID cannot be matched to scene data in the ALFRED dataset? The entry is skipped and logged as unmatched.
- How does the system handle multiple entries sharing the same task ID (same scene, different annotations)? Each entry is treated as an independent evaluation using its own NL steps in the same scene.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST load ground-truth plans from `resource/alfred_examples_for_prompt.json`, using each entry's `NL steps` field as the action sequence, `task id` to link to scene data in `alfred/data/`, and `task type` for categorized reporting.
- **FR-002**: System MUST accept a configurable portion parameter (as a percentage, greater than 0 and up to 100) that determines what fraction of the total plans to evaluate.
- **FR-003**: System MUST randomly select the specified portion of plans from the dataset.
- **FR-004**: System MUST accept an optional random seed parameter for reproducible task selection, with a documented default value.
- **FR-005**: System MUST execute each selected ground-truth plan in the AI2-THOR simulator by mapping each plan step to the corresponding simulator action via the existing action execution mechanism.
- **FR-006**: System MUST determine the correct ALFRED scene and initial state for each ground-truth plan by using the `task id` to locate the trajectory data (scene number, object poses, object toggles, dirty/empty state, initial action) in the `alfred/data/` directory via the dataset splits.
- **FR-007**: System MUST record the execution outcome (success/failure) for each plan, and for failures, capture: the step number, the failed action text, and the error message returned by the simulator.
- **FR-008**: System MUST generate a structured report containing: overall success rate, per-task results, and failure details, saved both as log output and as a persistent file.
- **FR-009**: System MUST categorize failures into meaningful categories (e.g., navigation failure, object interaction failure, object not found, precondition not met, exception/crash).
- **FR-010**: System MUST provide per-task-type success rate breakdown in the summary report.
- **FR-011**: System MUST validate the portion parameter and reject values outside the valid range with a clear error message.
- **FR-012**: System MUST gracefully handle individual task failures (exceptions, crashes) without aborting the entire evaluation run.
- **FR-013**: System MUST integrate with the existing configuration system for specifying evaluation parameters (portion, seed, output path).

### Key Entities

- **Ground-Truth Plan**: A single entry from `resource/alfred_examples_for_prompt.json` containing a `task id` (e.g., `trial_T20190907_174127_043461`), `task type` (e.g., `look_at_obj_in_light`), `task description` (natural language instruction), and `NL steps` (ordered list of executable actions like `["find an alarm clock", "pick up the alarm clock", "find a desk lamp", "turn on the desk lamp"]`). The dataset contains 17,469 entries covering 5,466 unique tasks across multiple task types.
- **Evaluation Result**: The outcome of executing a single ground-truth plan, including success/failure status, the list of executed steps, and failure details if applicable.
- **Failure Record**: A structured record for a failed plan containing the task identifier, failing step number, failing action text, error message, and failure category.
- **Evaluation Report**: An aggregate document containing overall statistics (success rate, total evaluated), per-task-type breakdowns, individual task results, and failure cause analysis.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Researchers can evaluate any percentage (1-100%) of ground-truth plans with a single command, and the correct number of tasks (within rounding) are executed.
- **SC-002**: Every failed plan in the report includes the specific step and action that caused the failure, along with the simulator's error message, enabling the researcher to diagnose the root cause.
- **SC-003**: Two evaluation runs with the same seed and portion produce identical task selections and results.
- **SC-004**: The evaluation report includes an overall success rate and per-task-type breakdown that a researcher can use to identify systematic failure patterns without manual log inspection.
- **SC-005**: A single task failure (crash or exception) does not prevent the remaining tasks from being evaluated; the run completes with partial results.
- **SC-006**: The evaluation run time scales linearly with the percentage of tasks selected (i.e., 10% of tasks completes proportionally faster than 100%).

## Assumptions

- The ground-truth plans are sourced from `resource/alfred_examples_for_prompt.json` (17,469 entries). Each entry has: `task id`, `task type`, `task description`, `step description` (human-readable), and `NL steps` (executable action list). The `NL steps` field contains the actions to execute (e.g., `["find an alarm clock", "pick up the alarm clock", ...]`).
- Scene data for each task is available in `alfred/data/json_2.1.0/train/{task_type_dir}/{trial_id}/traj_data.json`, containing scene configuration (scene number, object poses, object toggles, dirty/empty state, and initial agent action). All 5,466 unique task IDs have matching scene data on disk.
- The ALFRED dataset splits file (`alfred/data/splits/oct21.json`) maps task IDs to their full paths in the data directory (format: `{task_type_dir}/{trial_id}`) with repeat indices.
- The existing ThorConnector and its `llm_skill_interact` method can execute the action strings from `NL steps` without modification, as the format (e.g., "find a mug", "pick up the mug") matches the expected input format.
- AI2-THOR simulator (version 5.x) is available and properly installed on the evaluation machine.
- The existing evaluation infrastructure (Hydra configuration, image logging, result saving) will be reused and extended rather than replaced.
- The existing `eval_set: train` mode in `alfred_evaluator.py` already demonstrates loading ground-truth steps keyed by `task id` for training tasks, and this feature extends that pattern to a dedicated ground-truth evaluation mode.
- Multiple entries in the examples file may share the same `task id` (same scene, different task descriptions/annotations). When evaluating, the system uses the `NL steps` from each selected entry and the scene data from the shared `task id`.
