# Tasks: Ground-Truth Plan Evaluation

**Input**: Design documents from `/specs/001-groundtruth-eval/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/report-schema.json

**Tests**: Included (TDD approach specified in plan.md constitution check)

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization — Hydra config, new file scaffolding, and shared data types

- [x] T001 Create Hydra config file for GT evaluation at conf/config_alfred_gt.yaml with `name: alfred_gt`, defaults from `planner/default.yaml`, and GT-specific parameters (`gt.eval_portion_in_percent`, `gt.random_seed`, `gt.gt_data_file`, `gt.x_display`)
- [x] T002 [P] Create FailureCategory enum and GroundTruthEntry/TaskResult/EvaluationReport dataclasses in src/alfred/gt_report.py (data model only, no logic yet)
- [x] T003 [P] Add `alfred_gt` dispatch branch in src/evaluate.py to import and call GroundTruthEvaluator when `cfg.name == 'alfred_gt'`

**Checkpoint**: Config loads via `python src/evaluate.py --config-name=config_alfred_gt --cfg job` and dispatch branch exists

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Data loading and task-to-scene index — MUST be complete before any user story can execute plans

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T004 Implement `load_gt_entries(filepath) -> list[GroundTruthEntry]` in src/alfred/gt_evaluator.py to load and validate entries from `resource/alfred_examples_for_prompt.json` (parse `task id`, `task type`, `task description`, `NL steps`; skip entries with empty NL steps with warning)
- [x] T005 Implement `build_split_index(splits_file) -> dict[str, SplitEntry]` in src/alfred/gt_evaluator.py to pre-build task_id → split entry mapping from `alfred/data/splits/oct21.json` train split for O(1) lookups
- [x] T006 Write unit tests for `load_gt_entries` and `build_split_index` in tests/test_gt_evaluator.py (test valid loading, missing file error, empty NL steps skip, split index correctness)

**Checkpoint**: GT entries load correctly and every task_id resolves to a split path. Tests pass with `pytest tests/test_gt_evaluator.py -v`

---

## Phase 3: User Story 1 — Run GT Evaluation with Configurable Portion (Priority: P1) 🎯 MVP

**Goal**: Researcher can run `python src/evaluate.py --config-name=config_alfred_gt gt.eval_portion_in_percent=10` and the correct number of plans are randomly selected, executed in the simulator, and a success rate is reported.

**Independent Test**: Run with 1% portion and verify correct task count selected, plans execute, and success rate printed.

### Tests for User Story 1 (TDD — Red Phase) ⚠️

> **NOTE: TDD MANDATORY — Write tests FIRST, ensure they FAIL (Red), then implement (Green)**

- [x] T007 [P] [US1] Unit test for portion validation (reject <=0 and >100) in tests/test_gt_evaluator.py
- [x] T008 [P] [US1] Unit test for random selection (`select_entries`) verifying correct count (within rounding) and at-least-1 guarantee in tests/test_gt_evaluator.py
- [x] T009 [P] [US1] Unit test for `GroundTruthEvaluator.__init__` config parsing and parameter validation in tests/test_gt_evaluator.py

### Implementation for User Story 1

- [x] T010 [US1] Implement `validate_portion(portion: float) -> None` in src/alfred/gt_evaluator.py to reject values <=0 or >100 with clear error messages
- [x] T011 [US1] Implement `select_entries(entries, portion, seed) -> list[GroundTruthEntry]` in src/alfred/gt_evaluator.py for random sampling with seed, ensuring at least 1 task selected
- [x] T012 [US1] Implement `GroundTruthEvaluator.__init__(self, cfg)` in src/alfred/gt_evaluator.py to parse config, call `load_gt_entries`, `build_split_index`, `validate_portion`, and `select_entries`
- [x] T013 [US1] Implement `GroundTruthEvaluator.evaluate_single_entry(self, entry) -> TaskResult` in src/alfred/gt_evaluator.py to initialize scene via `load_task_json` + `ThorConnector`, execute each NL step via `llm_skill_interact`, catch exceptions, and return TaskResult
- [x] T014 [US1] Implement `GroundTruthEvaluator.evaluate(self)` in src/alfred/gt_evaluator.py to iterate over selected entries with tqdm, call `evaluate_single_entry`, collect results, log overall success rate, and save per-task JSON results

**Checkpoint**: Running `python src/evaluate.py --config-name=config_alfred_gt gt.eval_portion_in_percent=1` executes the correct number of tasks and prints success rate. Tests pass.

---

## Phase 4: User Story 2 — Failure Cause Reporting (Priority: P1)

**Goal**: Each failed plan in the output includes the task ID, failing step number, failing action, error message, and categorized failure reason.

**Independent Test**: Examine output for failed tasks and verify each failure entry contains all required fields with a meaningful failure category.

### Tests for User Story 2 (TDD — Red Phase) ⚠️

- [x] T015 [P] [US2] Unit test for `categorize_failure(message)` with all known error patterns (object_not_found, navigation_failure, inventory_error, visibility_error, interaction_failure, exception, unknown) in tests/test_gt_report.py
- [x] T016 [P] [US2] Unit test for failure detail capture in TaskResult (failure_step, failure_action, failure_message populated on failure; null on success) in tests/test_gt_report.py

### Implementation for User Story 2

- [x] T017 [US2] Implement `categorize_failure(message: str) -> FailureCategory` in src/alfred/gt_report.py using prefix/substring matching per research.md R3 patterns
- [x] T018 [US2] Update `GroundTruthEvaluator.evaluate_single_entry` in src/alfred/gt_evaluator.py to call `categorize_failure` on simulator error messages and populate failure_step, failure_action, failure_message, failure_category in TaskResult
- [x] T019 [US2] Update per-task JSON output in src/alfred/gt_evaluator.py to include failure details (step number, action, message, category) for each failed task

**Checkpoint**: Failed tasks in output include categorized failure details. `pytest tests/test_gt_report.py -v` passes.

---

## Phase 5: User Story 3 — Summary Report with Aggregate Statistics (Priority: P2)

**Goal**: After evaluation, a structured JSON report and human-readable log summary are generated with overall success rate, per-task-type breakdown, and per-failure-category counts.

**Independent Test**: Run evaluation and verify `gt_evaluation_report.json` matches the schema in `contracts/report-schema.json` and contains aggregate statistics.

### Tests for User Story 3 (TDD — Red Phase) ⚠️

- [x] T020 [P] [US3] Unit test for `generate_report(results, config) -> EvaluationReport` verifying correct aggregate computation (success rate, by_task_type, by_failure_category) in tests/test_gt_report.py
- [x] T021 [P] [US3] Unit test for `save_report(report, path)` verifying JSON output matches contracts/report-schema.json in tests/test_gt_report.py
- [x] T022 [P] [US3] Unit test for `print_summary(report)` verifying log output includes success rate, per-task-type, and per-failure-category in tests/test_gt_report.py

### Implementation for User Story 3

- [x] T023 [US3] Implement `generate_report(results: list[TaskResult], config: dict) -> EvaluationReport` in src/alfred/gt_report.py to compute overall stats, per-task-type breakdown, and per-failure-category counts
- [x] T024 [US3] Implement `save_report(report: EvaluationReport, output_path: str) -> None` in src/alfred/gt_report.py to serialize report as JSON matching contracts/report-schema.json
- [x] T025 [US3] Implement `print_summary(report: EvaluationReport) -> None` in src/alfred/gt_report.py to log human-readable summary (success rate, per-task-type table, failure category breakdown)
- [x] T026 [US3] Update `GroundTruthEvaluator.evaluate()` in src/alfred/gt_evaluator.py to call `generate_report`, `save_report`, and `print_summary` after all tasks complete

**Checkpoint**: `gt_evaluation_report.json` is generated and validates against the report schema. `pytest tests/test_gt_report.py -v` passes.

---

## Phase 6: User Story 4 — Random Selection with Reproducibility (Priority: P2)

**Goal**: Same seed + same portion = identical task selection and results. Default seed is logged for later reproduction.

**Independent Test**: Run evaluation twice with `gt.random_seed=42 gt.eval_portion_in_percent=5` and verify identical task selections.

### Tests for User Story 4 (TDD — Red Phase) ⚠️

- [x] T027 [P] [US4] Unit test verifying `select_entries` with same seed produces identical selection in tests/test_gt_evaluator.py
- [x] T028 [P] [US4] Unit test verifying `select_entries` with different seeds produces different selections in tests/test_gt_evaluator.py
- [x] T029 [P] [US4] Unit test verifying default seed is used and logged when no seed specified in tests/test_gt_evaluator.py

### Implementation for User Story 4

- [x] T030 [US4] Add default seed value (42) in conf/config_alfred_gt.yaml and ensure `GroundTruthEvaluator.__init__` logs the active seed in src/alfred/gt_evaluator.py
- [x] T031 [US4] Include `random_seed` and `eval_portion_in_percent` in the saved report config section via `generate_report` in src/alfred/gt_report.py

**Checkpoint**: Two runs with same seed and portion select identical tasks. Default seed is visible in both log output and report JSON.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Edge cases, robustness, and validation across all stories

- [x] T032 [P] Handle missing/corrupted GT data file with clear error message in src/alfred/gt_evaluator.py
- [x] T033 [P] Handle unmatched task_id (no split entry) — skip and log warning in src/alfred/gt_evaluator.py
- [x] T034 [P] Handle simulator crash for single task — catch exception, record as failed with traceback, continue with next task in src/alfred/gt_evaluator.py
- [x] T035 [P] Handle empty NL steps entries — skip and log warning during loading in src/alfred/gt_evaluator.py
- [x] T036 [P] Handle portion yielding <1 task (e.g., 0.001% of 17,469) — round up to 1 in src/alfred/gt_evaluator.py
- [x] T037 Run quickstart.md validation — verify all CLI commands documented in specs/001-groundtruth-eval/quickstart.md work correctly

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 (needs config and data types) — BLOCKS all user stories
- **US1 (Phase 3)**: Depends on Phase 2 (needs data loading and split index)
- **US2 (Phase 4)**: Depends on Phase 3 T013 (needs `evaluate_single_entry` to exist for updating)
- **US3 (Phase 5)**: Depends on Phase 3 T014 (needs `evaluate()` to exist for adding report calls)
- **US4 (Phase 6)**: Depends on Phase 3 T011 (needs `select_entries` to exist for seed behavior)
- **Polish (Phase 7)**: Can start after Phase 3 — edge cases can be added incrementally

### User Story Dependencies

- **US1 (P1)**: Foundation only — no dependencies on other stories
- **US2 (P1)**: Depends on US1 T013 (`evaluate_single_entry`) — extends failure handling
- **US3 (P2)**: Depends on US1 T014 (`evaluate()`) — adds report generation after evaluation
- **US4 (P2)**: Depends on US1 T011 (`select_entries`) — seed behavior already partial in US1

### Within Each User Story

- Tests MUST be written and FAIL before implementation (Red → Green)
- Data types before logic
- Loading/validation before execution
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

**Within Phase 1:**
- T002 and T003 can run in parallel (different files)

**Within Phase 2:**
- T004 and T005 can run in parallel initially (both in gt_evaluator.py but independent functions)

**Within Phase 3 (US1) Tests:**
- T007, T008, T009 can all run in parallel (independent test cases)

**Within Phase 4 (US2):**
- T015, T016 can run in parallel (different test aspects)

**Within Phase 5 (US3) Tests:**
- T020, T021, T022 can all run in parallel (independent test aspects)

**Within Phase 6 (US4) Tests:**
- T027, T028, T029 can all run in parallel (independent test cases)

**Within Phase 7 (Polish):**
- T032, T033, T034, T035, T036 can all run in parallel (independent edge cases)

**Cross-Story Parallelism:**
- US3 and US4 can be started in parallel once US1 is complete (they depend on different US1 tasks)

---

## Parallel Example: User Story 1

```bash
# Launch all US1 tests in parallel (Red phase):
Task: "T007 - Unit test for portion validation in tests/test_gt_evaluator.py"
Task: "T008 - Unit test for random selection in tests/test_gt_evaluator.py"
Task: "T009 - Unit test for GroundTruthEvaluator init in tests/test_gt_evaluator.py"

# Then implement sequentially:
Task: "T010 - Implement validate_portion in src/alfred/gt_evaluator.py"
Task: "T011 - Implement select_entries in src/alfred/gt_evaluator.py"
Task: "T012 - Implement GroundTruthEvaluator.__init__ in src/alfred/gt_evaluator.py"
Task: "T013 - Implement evaluate_single_entry in src/alfred/gt_evaluator.py"
Task: "T014 - Implement evaluate() in src/alfred/gt_evaluator.py"
```

## Parallel Example: User Story 3

```bash
# Launch all US3 tests in parallel (Red phase):
Task: "T020 - Unit test for generate_report in tests/test_gt_report.py"
Task: "T021 - Unit test for save_report in tests/test_gt_report.py"
Task: "T022 - Unit test for print_summary in tests/test_gt_report.py"

# Then implement sequentially:
Task: "T023 - Implement generate_report in src/alfred/gt_report.py"
Task: "T024 - Implement save_report in src/alfred/gt_report.py"
Task: "T025 - Implement print_summary in src/alfred/gt_report.py"
Task: "T026 - Update evaluate() to call report functions in src/alfred/gt_evaluator.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001–T003)
2. Complete Phase 2: Foundational (T004–T006)
3. Complete Phase 3: User Story 1 (T007–T014)
4. **STOP and VALIDATE**: Run with 1% portion, verify correct count and success rate output
5. Functional GT evaluator ready for basic use

### Incremental Delivery

1. Setup + Foundational → Config loads, data loads, split index works
2. Add US1 → Evaluator runs, success rate printed (MVP!)
3. Add US2 → Failure causes categorized and reported
4. Add US3 → Full JSON report + human-readable summary
5. Add US4 → Reproducible with seed logging
6. Polish → Edge cases, robustness, quickstart validation

### Suggested MVP Scope

**User Story 1 (Phase 3)** is the MVP. After completing Phases 1–3 (tasks T001–T014), the researcher can:
- Run GT evaluation with any portion percentage
- See success rate output
- Get per-task JSON results

This is sufficient for initial use while US2–US4 add diagnostics and reporting.

---

## Notes

- [P] tasks = different files or independent functions, no dependencies
- [Story] label maps task to specific user story for traceability
- All new code goes in 2 new files: `src/alfred/gt_evaluator.py` and `src/alfred/gt_report.py`
- All new tests go in 2 new files: `tests/test_gt_evaluator.py` and `tests/test_gt_report.py`
- One new config: `conf/config_alfred_gt.yaml`
- One modified file: `src/evaluate.py` (add dispatch branch only)
- Existing files (`alfred_evaluator.py`, `thor_connector.py`, `utils.py`, `evaluator.py`) remain unchanged
- TDD: Verify tests fail (Red) before implementing (Green)
- Tidy First: Commit structural changes separately from behavioral changes
- Commit after each task or logical group with appropriate prefix (`feat:`, `test:`, `refactor:`)
