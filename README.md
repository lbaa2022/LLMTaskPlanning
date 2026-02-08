# LoTa-Bench: Benchmarking Language-oriented Task Planners for Embodied Agents

> **Fork Notice**: This is a fork of the original [LoTa-Bench](https://github.com/lbaa2022/LLMTaskPlanning) repository with significant updates for modern compatibility and extended LLM support.

### [Paper (ICLR 2024)](https://arxiv.org/abs/2402.08178) | [Project Page](https://choi-jaewoo.github.io/LoTa-Bench/) | [Original Repository](https://github.com/lbaa2022/LLMTaskPlanning)

[Jae-Woo Choi](https://choi-jaewoo.github.io/)<sup>1*</sup>, [Youngwoo Yoon](https://sites.google.com/view/youngwoo-yoon/)<sup>1*</sup>, Hyobin Ong<sup>1, 2</sup>, Jaehong Kim<sup>1</sup>, Minsu Jang<sup>1, 2</sup> (*equal contribution)

<sup>1</sup> Electronics and Telecommunications Research Institute, <sup>2</sup> University of Science and Technology

We introduce a system for automatically quantifying performance of task planning for home-service agents. Task planners are tested on two pairs of datasets and simulators: 1) [ALFRED](https://github.com/askforalfred/alfred) and [AI2-THOR](https://ai2thor.allenai.org/), 2) an extension of [Watch-And-Help](https://github.com/xavierpuigf/watch_and_help) and [VirtualHome](http://virtual-home.org/). Using the proposed benchmark system, we perform extensive experiments with LLMs and prompts, and explore several extentions of the baseline planner.

---

## Changes in This Fork

This fork introduces the following improvements over the original repository:

### AI2-THOR 5.x Compatibility
- **Updated from AI2-THOR 2.x to 5.0.0+** - The original codebase used AI2-THOR 2.x which is no longer maintained. This fork includes all necessary API changes:
  - `TeleportFull` action now uses Vector3 rotation format (`{'x': 0, 'y': 90, 'z': 0}`) instead of scalar rotation
  - Removed deprecated `rotateOnTeleport` parameter
  - Added required `standing` parameter for teleport actions
  - Replaced removed `SetStateOfAllObjects` with individual object state actions
  - Fixed `visibilityDistance` parameter naming (camelCase)
  - Platform-specific controller initialization (no `x_display` on macOS)

### OpenAI API Compatible LLM Support
- **Unified LLM provider interface** - Refactored to support any OpenAI API compatible endpoint:
  - Native OpenAI support (GPT-4, GPT-4-turbo, GPT-5.x, o1, o3)
  - **vLLM support** - Run evaluations with locally hosted models via vLLM server
  - Easy extensibility for other OpenAI-compatible providers (Ollama, LM Studio, etc.)
- **Reasoning model support** - Added `reasoning_effort` parameter for o1/o3/GPT-5.x models

### Ground-Truth Plan Evaluation
- **New `config_alfred_gt` evaluation mode** — Execute known-correct ground-truth plans from the ALFRED training set in AI2-THOR and produce a structured report:
  - Configurable evaluation portion (1-100%) with reproducible random selection via seed
  - Automatic failure categorization (object not found, navigation failure, inventory error, visibility error, interaction failure, exception)
  - Per-task-type success rate breakdown and aggregate statistics
  - Machine-readable JSON report alongside human-readable log summary
  - 47 unit tests covering all evaluation logic (no simulator required to run tests)

### Other Improvements
- Modernized Python compatibility (3.8 - 3.12)
- Environment variable configuration via `.env` file
- Improved cross-platform support (removed Linux-specific dependencies)
- Added comprehensive test suite for AI2-THOR compatibility

---

## Environment

- **OS**: Ubuntu 14.04+, macOS, or Windows with WSL
- **Python**: 3.8 - 3.12
- **AI2-THOR**: 5.0.0+ (updated for compatibility)

## Install

1. Clone the repository:
    ```bash
    git clone {repo_url}
    cd LLMTaskPlanning
    ```

2. Setup a virtual environment:
    ```bash
    conda create -n llmtaskplanning python=3.10
    conda activate llmtaskplanning
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Configure API keys by creating a `.env` file:
    ```bash
    cp .env.example .env
    # Edit .env and add your API keys
    ```

## LLM Provider Configuration

The system supports OpenAI API compatible LLM providers through a unified interface:

| Provider | Models | Default Port | Configuration |
|----------|--------|--------------|---------------|
| **OpenAI** | gpt-4, gpt-4-turbo, gpt-5.2, o1, o3 | N/A | `OPENAI_API_KEY` |
| **vLLM** | Any model served locally | 8000 | `VLLM_API_BASE` |
| **Ollama** | llama2, llama3, mistral, codellama, etc. | 11434 | `OLLAMA_API_BASE` |
| **LM Studio** | Any loaded model | 1234 | `LMSTUDIO_API_BASE` |

### Environment Variables (.env file)

```bash
# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# vLLM (local server)
VLLM_API_BASE=http://localhost:8000/v1

# Ollama (local server)
OLLAMA_API_BASE=http://localhost:11434/v1

# LM Studio (local server)
LMSTUDIO_API_BASE=http://localhost:1234/v1
```

## Benchmarking on ALFRED

### Download ALFRED dataset
```bash
cd alfred/data
bash download_data.sh json
cd ../..
```

### Run Benchmarking

**With OpenAI GPT-4:**
```bash
python src/evaluate.py --config-name=config_alfred \
    planner.provider=openai \
    planner.model_name=gpt-4
```

**With OpenAI GPT-5.2:**
```bash
python src/evaluate.py --config-name=config_alfred \
    planner.provider=openai \
    planner.model_name=gpt-5.2
```

**With local vLLM server:**
```bash
# First, start vLLM server
vllm serve meta-llama/Llama-2-7b-chat-hf --port 8000

# Then run evaluation
python src/evaluate.py --config-name=config_alfred \
    planner.provider=vllm \
    planner.model_name=meta-llama/Llama-2-7b-chat-hf
```

**With Ollama:**
```bash
# First, start Ollama and pull a model
ollama serve
ollama pull llama2

# Then run evaluation
python src/evaluate.py --config-name=config_alfred \
    planner.provider=ollama \
    planner.model_name=llama2
```

**With LM Studio:**
```bash
# Start LM Studio and load a model, then run evaluation
python src/evaluate.py --config-name=config_alfred \
    planner.provider=lmstudio \
    planner.model_name=your-loaded-model

# With custom endpoint (e.g., remote LM Studio server)
python src/evaluate.py --config-name=config_alfred \
    planner.provider=lmstudio \
    planner.model_name=openai/gpt-oss-20b \
    planner.api_base=http://10.254.90.90:1234/v1
```

### Configuration Options

We use [Hydra](https://hydra.cc/) for configuration management. Key options:

```bash
# Change evaluation portion
python src/evaluate.py --config-name=config_alfred alfred.eval_portion_in_percent=10

# Change number of prompt examples
python src/evaluate.py --config-name=config_alfred prompt.num_examples=8

# Change X display for headless servers
python src/evaluate.py --config-name=config_alfred alfred.x_display='1'

# Adjust max planning steps
python src/evaluate.py --config-name=config_alfred planner.max_steps=30
```

### Planning Modes

- **API providers** (OpenAI): Use `plan_whole` mode - generates complete action sequence in one API call
- **Local providers** (vLLM): Use `plan_step_by_step` mode - generates actions iteratively

### Reasoning Models (o1, o3, GPT-5.x)

For reasoning models, you can control the thinking depth with `reasoning_effort`:

```bash
# Low effort - faster responses, less thorough reasoning
python src/evaluate.py --config-name=config_alfred \
    planner.model_name=gpt-5.2 \
    planner.reasoning_effort=low

# Medium effort - balanced
python src/evaluate.py --config-name=config_alfred \
    planner.model_name=gpt-5.2 \
    planner.reasoning_effort=medium

# High effort - thorough reasoning but slower
python src/evaluate.py --config-name=config_alfred \
    planner.model_name=gpt-5.2 \
    planner.reasoning_effort=high
```

### Headless Server Setup

For headless servers, run `startx.py` before evaluation:

```bash
sudo python3 alfred/scripts/startx.py 1
```


## Ground-Truth Plan Evaluation

The ground-truth evaluation mode executes known-correct (ground-truth) action plans from the ALFRED training dataset in the AI2-THOR simulator and produces a detailed report with success rates, failure cause categorization, and per-task-type breakdowns. This is useful for validating that the simulator correctly executes reference plans and for diagnosing systematic failure patterns.

### Data Sources

The evaluator uses two data files:

| File | Contents |
|------|----------|
| `resource/alfred_examples_for_prompt.json` | 17,469 ground-truth entries with `task id`, `task type`, `task description`, and `NL steps` (the executable action sequence) |
| `alfred/data/splits/oct21.json` | Maps task IDs to filesystem paths for loading scene data (object poses, toggles, initial state) |

### Running the Evaluation

**Evaluate 5% of plans (default):**
```bash
python src/evaluate.py --config-name=config_alfred_gt
```

**Evaluate a custom portion:**
```bash
# 10% of all ground-truth plans
python src/evaluate.py --config-name=config_alfred_gt gt.eval_portion_in_percent=10

# All plans (full dataset)
python src/evaluate.py --config-name=config_alfred_gt gt.eval_portion_in_percent=100
```

**Reproducible runs with a specific random seed:**
```bash
python src/evaluate.py --config-name=config_alfred_gt \
    gt.eval_portion_in_percent=10 \
    gt.random_seed=42
```

Running the same command twice with the same seed and portion will select and evaluate the identical set of tasks.

**On Linux with X11 display:**
```bash
python src/evaluate.py --config-name=config_alfred_gt gt.x_display='0'
```

### Configuration Parameters

All parameters live under the `gt` key in `conf/config_alfred_gt.yaml` and can be overridden from the command line:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gt.eval_portion_in_percent` | `5` | Percentage of plans to evaluate (must be >0 and <=100) |
| `gt.random_seed` | `42` | Seed for reproducible random task selection |
| `gt.gt_data_file` | `resource/alfred_examples_for_prompt.json` | Path to the ground-truth examples file |
| `gt.x_display` | `'0'` | X11 display number for AI2-THOR (Linux only) |

### Output

Results are saved to `outputs/alfred_gt/{timestamp}/` (Hydra-managed output directory):

| File | Description |
|------|-------------|
| `gt_evaluation_report.json` | Full structured report with all results and aggregate statistics |
| `{trial_id}_{entry_index}.json` | Individual task result with execution details |

### Report Structure

The `gt_evaluation_report.json` contains:

**Top-level fields:**
- `timestamp` — ISO 8601 timestamp of the run
- `config` — The evaluation configuration used (portion, seed, data file)
- `total_evaluated` — Number of plans executed
- `total_success` / `total_failure` — Count of successes and failures
- `success_rate` — Overall success rate as a percentage

**Per-task-type breakdown** (`by_task_type`):
```json
{
  "pick_and_place_simple": {"total": 50, "success": 42, "failure": 8, "success_rate": 84.0},
  "look_at_obj_in_light": {"total": 30, "success": 25, "failure": 5, "success_rate": 83.33}
}
```

**Failure category breakdown** (`by_failure_category`):
```json
{
  "object_not_found": 5,
  "navigation_failure": 3,
  "interaction_failure": 3,
  "visibility_error": 1,
  "inventory_error": 1
}
```

**Per-task results** (`results`): Each entry includes the task ID, task type, description, whether it succeeded, the list of executed steps, and — for failures — the step number, action text, error message, and categorized failure reason.

### Failure Categories

Failed plans are automatically categorized based on the simulator error message:

| Category | Meaning | Example Error |
|----------|---------|---------------|
| `object_not_found` | Target object not in scene or not locatable | "Cannot find mug" |
| `navigation_failure` | Agent cannot navigate to target | "Cannot move to mug" |
| `inventory_error` | Inventory state prevents action | "Robot is not holding any object" |
| `visibility_error` | Object hidden inside a closed receptacle | "mug is not visible because it is in Fridge" |
| `interaction_failure` | Simulator action failed despite finding object | "Open action failed" |
| `exception` | Unexpected Python exception during execution | Any caught Exception |
| `unknown` | Error doesn't match known patterns | Fallback |

### Example: Interpreting a Failed Task Result

```json
{
  "task_id": "trial_T20190907_174127_043461",
  "task_type": "look_at_obj_in_light",
  "task_description": "Examine an alarm clock under a desk lamp",
  "success": false,
  "executed_steps": ["find an alarm clock", "pick up the alarm clock"],
  "total_steps": 4,
  "failure_step": 3,
  "failure_action": "find a desk lamp",
  "failure_message": "Cannot find DeskLamp",
  "failure_category": "object_not_found",
  "goal_satisfied": false,
  "scene_name": "FloorPlan301"
}
```

This tells you: the plan failed at step 3 ("find a desk lamp") because the DeskLamp object was not found in the scene. Steps 1-2 executed successfully.

### Running Tests

Unit tests for the GT evaluator do not require AI2-THOR and can run anywhere:

```bash
pytest tests/test_gt_evaluator.py tests/test_gt_report.py -v
```

This runs 47 tests covering data loading, portion validation, random selection, failure categorization, report generation, and JSON serialization.

---

## Benchmarking on Watch-And-Help

### Download the VirtualHome Simulator
```bash
cd virtualhome/simulation/unity_simulator/
wget http://virtual-home.org//release/simulator/v2.0/v2.2.2/linux_exec.zip
unzip linux_exec.zip
cd ../../..
```

### Run Benchmarking

1. Start VirtualHome simulator in one terminal:
    ```bash
    ./virtualhome/simulation/unity_simulator/linux_exec.x86_64
    ```

2. Run evaluation in another terminal:
    ```bash
    python src/evaluate.py --config-name=config_wah \
        planner.provider=openai \
        planner.model_name=gpt-4
    ```

### Headless Server Setup

1. Start Xserver:
    ```bash
    cd virtualhome
    sudo python helper_scripts/startx.py $display_num
    ```

2. Start simulator in batchmode:
    ```bash
    DISPLAY=:$display_num ./simulation/unity_simulator/linux_exec.x86_64 -batchmode
    ```

3. Run evaluation:
    ```bash
    python src/evaluate.py --config-name=config_wah_headless \
        planner.provider=openai \
        planner.model_name=gpt-4
    ```


## Extensions

### In-context Example Selection
```bash
python src/evaluate.py --config-name=config_wah prompt.select_method=same_task
python src/evaluate.py --config-name=config_wah prompt.select_method=topk
```

### Replanning with Feedback
```bash
python src/evaluate.py --config-name=config_alfred planner.use_predefined_prompt=True
```


## Customizing Prompts

The system uses externalized prompt templates that can be easily customized without modifying code.

### Prompt Template Files

All prompt templates are stored in `src/prompts/templates/`:

| File | Purpose |
|------|---------|
| `action_selection_system.txt` | System message for action selection |
| `action_selection_user.txt` | User message template for action selection |
| `plan_generation_system.txt` | System message for plan generation |
| `plan_generation_user.txt` | User message template for plan generation |
| `step_by_step_format.txt` | Format for initial step-by-step prompt |
| `step_continuation.txt` | Format for continuing steps |
| `step_with_failure.txt` | Format for steps that failed |

### Customization Examples

**Change the robot's persona** by editing `plan_generation_system.txt`:
```
You are a helpful home assistant robot. When given a task, you carefully plan
each step to accomplish it safely and efficiently.
```

**Change action selection behavior** by editing `action_selection_system.txt`:
```
You are a robot operating in a home environment. Given the current situation
and a list of possible actions, select the single best action to take next.
Respond with ONLY the exact action text.
```

### Template Variables

Templates use Python's `str.format()` syntax for variable substitution:

| Template | Variables |
|----------|-----------|
| `action_selection_user.txt` | `{prompt}`, `{candidates}` |
| `plan_generation_user.txt` | `{examples}`, `{skills}`, `{query}` |
| `step_by_step_format.txt` | `{query}` |
| `step_continuation.txt` | `{step}`, `{next_step_num}` |
| `step_with_failure.txt` | `{step}`, `{failure_message}`, `{next_step_num}` |


## WAH-NL Dataset

You can find the WAH-NL data, which is our extension of WAH, in `./dataset` folder.


## Project Structure

```
LLMTaskPlanning/
├── src/
│   ├── evaluate.py             # Main entry point (dispatches by config name)
│   ├── evaluator.py            # Base Evaluator class
│   ├── task_planner.py         # Base task planner
│   ├── llm/                    # LLM provider abstraction
│   │   ├── base.py             # Abstract base class
│   │   ├── factory.py          # Provider factory
│   │   ├── openai_provider.py  # OpenAI implementation
│   │   └── vllm_provider.py    # vLLM implementation
│   ├── alfred/                 # ALFRED evaluators
│   │   ├── alfred_evaluator.py # LLM-based evaluation
│   │   ├── gt_evaluator.py     # Ground-truth plan evaluation
│   │   ├── gt_report.py        # Failure categorization & report generation
│   │   ├── thor_connector.py   # AI2-THOR simulator interface
│   │   └── utils.py            # Shared utilities (load_task_json, etc.)
│   └── wah/                    # Watch-And-Help evaluator
├── conf/                       # Hydra configurations
│   ├── config_alfred.yaml      # LLM-based ALFRED evaluation
│   ├── config_alfred_gt.yaml   # Ground-truth plan evaluation
│   └── config_wah.yaml         # Watch-And-Help evaluation
├── tests/                      # Test suite
│   ├── test_gt_evaluator.py    # GT evaluator unit tests (24 tests)
│   └── test_gt_report.py       # GT report unit tests (23 tests)
├── alfred/                     # ALFRED environment & dataset
├── virtualhome/                # VirtualHome environment
└── dataset/                    # WAH-NL dataset
```


## FAQ

**Q: I encounter 'cannot find X server with xdpyinfo' in running ALFRED experiments.**

A: Try a different x_display number:
```bash
python src/evaluate.py --config-name=config_alfred alfred.x_display='1'
```

**Q: How do I use a custom API endpoint?**

A: Set the API base URL in your `.env` file:
```bash
OPENAI_API_BASE=https://your-custom-endpoint.com/v1
```

**Q: Which planning mode should I use?**

A: The system automatically selects the appropriate mode:
- API providers (OpenAI) → `plan_whole` (faster, single API call)
- Local providers (vLLM) → `plan_step_by_step` (better for smaller models)


## Citation

```bibtex
@inproceedings{choi2024lota,
  title={LoTa-Bench: Benchmarking Language-oriented Task Planners for Embodied Agents},
  author={Choi, Jae-Woo and Yoon, Youngwoo and Ong, Hyobin and Kim, Jaehong and Jang, Minsu},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```
