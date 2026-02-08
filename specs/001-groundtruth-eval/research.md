# Research: Ground-Truth Plan Evaluation

## R1: Data Source and Task Matching Strategy

**Decision**: Use `resource/alfred_examples_for_prompt.json` as the primary data source, with `alfred/data/splits/oct21.json` as the mapping layer to locate scene data on disk.

**Rationale**:
- The examples file contains 17,469 entries with structured fields: `task id`, `task type`, `task description`, `step description`, and `NL steps` — all needed for execution and reporting.
- All 5,466 unique task IDs in the examples file are present in the `train` split of `oct21.json`, which provides the full filesystem path (`{task_type_dir}/{trial_id}`) needed by `load_task_json`.
- All matched task IDs have trajectory data (`traj_data.json`) on disk containing scene configuration (scene number, object poses, toggles, dirty/empty state, init action).
- The existing code in `alfred_evaluator.py` (lines 126-134) already demonstrates this exact loading pattern: `train_gt_steps[ex['task id']] = ex['NL steps']`.

**Alternatives considered**:
- `alfred_train_in_alpaca_instruct_data_v5.json` (17,468 entries): Lacks `task id` and `task type` fields, making scene matching impossible. Same content as examples file but in alpaca instruction-tuning format. Rejected.
- Loading directly from `traj_data.json` files on disk: Would require parsing PDDL plan format instead of pre-extracted NL steps. More complex and fragile. Rejected.

## R2: Task-to-Scene Matching Pipeline

**Decision**: Build a pre-execution index that maps `task_id` → split entry → `traj_data.json` path, then use existing `load_task_json` for loading.

**Rationale**:
- The split file maps trial IDs to full task paths: e.g., `trial_T20190909_070538_437648` → `pick_cool_then_place_in_recep-LettuceSliced-None-DiningTable-17/trial_T20190909_070538_437648`
- `load_task_json` accepts a dict `{'task': path, 'repeat_idx': N}` and tries three locations: `pp/ann_N.json`, `train/{path}/traj_data.json`, `{path}/traj_data.json`
- Multiple examples share the same task_id (same scene, different human annotations). The split entries include `repeat_idx` (0, 1, 2) for these variations.
- Pre-building the index avoids repeated O(n) scans of the 21,023-entry train split.

**Alternatives considered**:
- Linear scan of splits per task: O(n²) for 17,469 × 21,023. Rejected for performance.
- Glob-based filesystem scan: Fragile and slow. Rejected.

## R3: Failure Categorization Strategy

**Decision**: Categorize failures by parsing the error message string returned by `ThorConnector.llm_skill_interact()` using prefix/substring matching.

**Rationale**: The ThorConnector returns structured error messages in `ret_dict['message']` with consistent patterns:

| Category | Message Patterns |
|----------|-----------------|
| `object_not_found` | `"Cannot find {obj}"`, `"Cannot find {obj} to pick up/open/close/slice/turn on/turn off"` |
| `navigation_failure` | `"Cannot move to {obj}"` |
| `inventory_error` | `"Robot is not holding any object"`, `"Robot is currently holding {obj}"` |
| `visibility_error` | `"{obj} is not visible because it is in {recep}"` |
| `interaction_failure` | `"Open/Close/Slice/Turn on/Turn off/Drop action failed"`, `"Putting the object on {recep} failed"`, `"put down failed"` |
| `exception` | Any error caught in the try/except wrapper |

**Alternatives considered**:
- Modifying ThorConnector to return structured error codes: Would require changing existing code, violating the constraint of reusing existing infrastructure. Rejected.
- AI-based error classification: Over-engineered for predictable, pattern-based messages. Rejected.

## R4: Report Format

**Decision**: Generate both a human-readable log summary (via Python logging) and a machine-readable JSON report file.

**Rationale**:
- Log summary follows existing pattern in `alfred_evaluator.py` (`log.info(f'success rate: ...')`)
- JSON report enables programmatic analysis, visualization, and comparison across runs
- JSON includes all individual results plus aggregate statistics
- Consistent with existing per-task JSON output pattern (`{trial}_{repeat_idx}.json`)

**Alternatives considered**:
- CSV output: Less structured, harder to represent nested failure details. Rejected.
- Markdown report: Nice for reading but harder to parse programmatically. Rejected.
- Only log output: Not persistent or machine-readable. Rejected.

## R5: Hydra Configuration Approach

**Decision**: Create a new config file `conf/config_alfred_gt.yaml` that inherits from `planner/default.yaml` and adds GT-specific parameters, with dispatch via `cfg.name == 'alfred_gt'` in `evaluate.py`.

**Rationale**:
- Follows existing pattern: `config_alfred.yaml` for LLM evaluation, `config_wah.yaml` for WAH evaluation
- Reuses Hydra defaults infrastructure (hydra config, output directory timestamping)
- GT evaluation doesn't need a planner (no LLM), but keeping the planner defaults avoids config merge errors
- Command: `python src/evaluate.py --config-name=config_alfred_gt gt.eval_portion_in_percent=10`

**Alternatives considered**:
- Adding a flag to existing `config_alfred.yaml`: Clutters the existing config with GT-specific options. Would require conditional logic. Rejected.
- Separate script outside Hydra: Loses output directory management, config overrides, and consistency. Rejected.

## R6: Handling Multiple Annotations per Task ID

**Decision**: Each example entry is treated as an independent evaluation. When multiple entries share a `task_id`, each is evaluated with its own NL steps but in the same scene (re-initialized each time).

**Rationale**:
- The examples file has 17,469 entries but only 5,466 unique task IDs (~3.2 entries per task on average)
- Different annotations have the same NL steps (same ground-truth plan) but different task descriptions
- Since NL steps are identical for same-task entries, evaluating all of them tests the same plan in the same scene — which is redundant but harmless
- When sampling a portion, random selection naturally handles this (some duplicates may be selected, some not)

**Alternatives considered**:
- Deduplicate by task_id before sampling: Would change the effective dataset size from 17,469 to 5,466. Could add as an option later but not needed for MVP. Noted for future.
