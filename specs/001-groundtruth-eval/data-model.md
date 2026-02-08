# Data Model: Ground-Truth Plan Evaluation

## Entities

### GroundTruthEntry

Represents a single entry loaded from `resource/alfred_examples_for_prompt.json`.

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `task_id` | string | Trial identifier (e.g., `trial_T20190907_174127_043461`) | `"task id"` field |
| `task_type` | string | ALFRED task category (e.g., `look_at_obj_in_light`) | `"task type"` field |
| `task_description` | string | Human-readable instruction | `"task description"` field |
| `nl_steps` | list[string] | Ordered executable actions | `"NL steps"` field |
| `entry_index` | int | Position in the original dataset (for reproducibility) | Array index |

**Validation rules**:
- `task_id` must be non-empty and match pattern `trial_T\d+_\d+_\d+`
- `nl_steps` must be a non-empty list of strings
- `task_type` must be non-empty

### SplitEntry

Represents a task entry from `alfred/data/splits/oct21.json` train split.

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `task` | string | Full path: `{task_type_dir}/{trial_id}` | `"task"` field |
| `repeat_idx` | int | Annotation index (0, 1, or 2) | `"repeat_idx"` field |

**Derived field**:
- `trial_id`: Extracted from `task.split('/')[-1]`

### TaskResult

Represents the outcome of executing one ground-truth plan entry.

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | Trial identifier |
| `task_type` | string | ALFRED task category |
| `task_description` | string | Human-readable instruction |
| `entry_index` | int | Position in original dataset |
| `success` | bool | Whether all steps completed and goal was satisfied |
| `executed_steps` | list[string] | Steps that were actually executed |
| `total_steps` | int | Total number of NL steps in the plan |
| `failure_step` | int or null | Step number where failure occurred (1-indexed), null if success |
| `failure_action` | string or null | The action text that failed, null if success |
| `failure_message` | string or null | Error message from simulator, null if success |
| `failure_category` | string or null | Categorized failure type, null if success |
| `goal_satisfied` | bool | Whether the ALFRED goal checker reports success |
| `scene_name` | string | Scene identifier (e.g., `FloorPlan301`) |

### FailureCategory

Enumeration of failure categories derived from ThorConnector error message patterns.

| Value | Description | Example Messages |
|-------|-------------|-----------------|
| `object_not_found` | Target object not in scene or not locatable | "Cannot find mug", "Cannot find mug to pick up" |
| `navigation_failure` | Agent cannot navigate to target object | "Cannot move to mug" |
| `inventory_error` | Inventory state prevents action | "Robot is not holding any object", "Robot is currently holding Mug" |
| `visibility_error` | Object hidden inside closed receptacle | "mug is not visible because it is in Fridge" |
| `interaction_failure` | Simulator action failed despite finding object | "Open action failed", "Putting the object on CounterTop failed" |
| `exception` | Unexpected Python exception during execution | Any caught Exception |
| `unknown` | Error message doesn't match known patterns | Fallback category |

### EvaluationReport

Aggregate output of a complete evaluation run.

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | string | ISO 8601 timestamp of the run |
| `config` | dict | Copy of the evaluation configuration used |
| `total_evaluated` | int | Number of plans executed |
| `total_success` | int | Number of plans that succeeded |
| `total_failure` | int | Number of plans that failed |
| `success_rate` | float | `total_success / total_evaluated` as percentage |
| `random_seed` | int | Seed used for task selection |
| `eval_portion_percent` | float | Portion of dataset evaluated |
| `results` | list[TaskResult] | Per-task results |
| `by_task_type` | dict | Success rate breakdown by `task_type` |
| `by_failure_category` | dict | Count of failures by category |

## Relationships

```
GroundTruthEntry (17,469)
    │
    ├──[task_id]──▶ SplitEntry (21,023 in train)
    │                   │
    │                   └──[task path]──▶ traj_data.json on disk
    │                                       (scene, object_poses, etc.)
    │
    └──[evaluated as]──▶ TaskResult
                            │
                            ├──[aggregated into]──▶ EvaluationReport
                            │
                            └──[failure_category]──▶ FailureCategory (enum)
```

## State Transitions

### Task Execution States

```
PENDING ──[scene loaded]──▶ EXECUTING ──[all steps done + goal satisfied]──▶ SUCCESS
                              │
                              ├──[step fails]──▶ FAILED (with failure details)
                              │
                              └──[exception]──▶ FAILED (category: exception)
```
