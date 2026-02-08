# Quickstart: Ground-Truth Plan Evaluation

## Prerequisites

- Python 3.10+
- AI2-THOR 5.x simulator installed (`pip install ai2thor>=5.0.0`)
- ALFRED dataset downloaded in `alfred/data/json_2.1.0/` with train split data
- Ground-truth examples file at `resource/alfred_examples_for_prompt.json`

## Running the Evaluation

### Basic Usage

Evaluate 5% of ground-truth plans (default):

```bash
python src/evaluate.py --config-name=config_alfred_gt
```

### Custom Portion

Evaluate 10% of plans:

```bash
python src/evaluate.py --config-name=config_alfred_gt gt.eval_portion_in_percent=10
```

Evaluate all plans (100%):

```bash
python src/evaluate.py --config-name=config_alfred_gt gt.eval_portion_in_percent=100
```

### Custom Random Seed

Use a specific seed for reproducible task selection:

```bash
python src/evaluate.py --config-name=config_alfred_gt gt.eval_portion_in_percent=10 gt.random_seed=42
```

### macOS vs Linux

On Linux with X11 display:

```bash
python src/evaluate.py --config-name=config_alfred_gt gt.x_display='0'
```

On macOS (no X display needed):

```bash
python src/evaluate.py --config-name=config_alfred_gt
```

## Output

Results are saved to `outputs/alfred_gt/{timestamp}/`:

| File | Description |
|------|-------------|
| `gt_evaluation_report.json` | Full structured report with all results |
| `gt_summary.log` | Human-readable summary |
| `{trial_id}_{repeat_idx}.json` | Per-task result (same as existing format) |
| `{trial_id}_{repeat_idx}_{success\|fail}.png` | Trajectory visualization |

## Report Contents

The `gt_evaluation_report.json` includes:

- **Overall**: success rate, total evaluated, total success/failure
- **By task type**: success rate per task type (e.g., `look_at_obj_in_light: 85%`)
- **By failure category**: count per failure type (e.g., `navigation_failure: 12`)
- **Per-task details**: each task's result with failure step/action/message if failed

## Running Tests

Unit tests (no simulator needed):

```bash
pytest tests/test_gt_evaluator.py tests/test_gt_report.py -v
```

Integration tests (requires AI2-THOR):

```bash
pytest tests/test_gt_evaluator.py -v -m integration
```
