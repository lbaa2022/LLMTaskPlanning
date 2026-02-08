# Implementation Plan: Ground-Truth Plan Evaluation

**Branch**: `001-groundtruth-eval` | **Date**: 2026-02-08 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-groundtruth-eval/spec.md`

## Summary

Build a dedicated ground-truth evaluation mode that executes pre-defined NL action plans from `resource/alfred_examples_for_prompt.json` in the AI2-THOR simulator and produces a detailed success/failure report with categorized failure causes and per-task-type breakdowns. The feature extends the existing `AlfredEvaluator` by adding a new `GroundTruthEvaluator` class that reuses the existing `ThorConnector`, `load_task_json`, and Hydra configuration infrastructure, with a new Hydra config file for ground-truth-specific parameters.

## Technical Context

**Language/Version**: Python 3.10+ (matches existing codebase)
**Primary Dependencies**: ai2thor>=5.0.0, hydra-core==1.3.2, omegaconf==2.3.0, Pillow>=9.4.0, tqdm>=4.65.0
**Storage**: JSON files for input data and output reports; PNG for trajectory images
**Testing**: pytest (existing test framework in `tests/`)
**Target Platform**: macOS / Linux with AI2-THOR simulator
**Project Type**: Single project (extending existing `src/` structure)
**Performance Goals**: Linear scaling — 10% of tasks takes ~10% of full evaluation time
**Constraints**: Must reuse existing ThorConnector, load_task_json, and Hydra config patterns
**Scale/Scope**: 17,469 ground-truth entries across 5,466 unique tasks; evaluable in configurable portions (1-100%)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Verify compliance with TDD and Tidy First principles:

- [x] **TDD Cycle**: Plan includes test-first approach — unit tests for data loading, plan parsing, failure categorization, and report generation before implementing each module
- [x] **Tidy First**: Structural changes (extracting shared code from AlfredEvaluator) separated from behavioral changes (new GT evaluation logic)
- [x] **Commit Discipline**: Plan supports small, atomic commits: `refactor:` for extracted helpers, `feat:` for new GT evaluator, `test:` for new tests
- [x] **Code Quality**: Design follows single responsibility (separate loader, evaluator, reporter), DRY (reuses existing ThorConnector/load_task_json), YAGNI (no speculative abstractions)
- [x] **Refactoring**: Any code extraction from existing `alfred_evaluator.py` happens in separate `refactor:` commits before new feature work
- [x] **Simplicity**: Extends existing patterns (Evaluator base class, Hydra config, ThorConnector) rather than building new infrastructure

## Project Structure

### Documentation (this feature)

```text
specs/001-groundtruth-eval/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── report-schema.json
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
src/
├── alfred/
│   ├── alfred_evaluator.py          # Existing (unchanged)
│   ├── gt_evaluator.py              # NEW: GroundTruthEvaluator class
│   ├── gt_report.py                 # NEW: Report generation & failure categorization
│   ├── thor_connector.py            # Existing (reused, unchanged)
│   └── utils.py                     # Existing (reused, unchanged)
├── evaluate.py                      # MODIFIED: Add GT evaluator dispatch
└── evaluator.py                     # Existing base class (unchanged)

conf/
├── config_alfred_gt.yaml            # NEW: Hydra config for GT evaluation
└── planner/default.yaml             # Existing (unchanged)

tests/
├── test_gt_evaluator.py             # NEW: Unit tests for GT evaluator
├── test_gt_report.py                # NEW: Unit tests for report generation
└── test_ai2thor_compatibility.py    # Existing (unchanged)
```

**Structure Decision**: Extends the existing single-project `src/` layout. New files are added alongside existing ALFRED modules to maintain consistency. The GT evaluator is a separate file (`gt_evaluator.py`) rather than modifying `alfred_evaluator.py` to preserve existing functionality and follow single responsibility.

## Complexity Tracking

No constitution violations. The design uses the simplest approach: extending existing patterns without new abstractions.
