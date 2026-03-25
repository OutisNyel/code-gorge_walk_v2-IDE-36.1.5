# code-gorge_walk_v2-IDE-36.1.5

This repository is a `gorge_walk_v2` reinforcement learning code package for the Tencent Kaiwu platform. It provides five algorithm packages:

- Dynamic Programming
- Monte Carlo
- Q-Learning
- SARSA
- DIY template

It is not a complete standalone Python project. It is better understood as a platform-facing algorithm package plus a local smoke-test entrypoint.

## What This Repository Is

The official Tencent Arena documentation describes a standard KaiwuRL project as:

- one agent package
- shared configuration
- one `train_test.py` validation script

This repository is a variant of that template:

- it keeps one package per algorithm: `agent_dynamic_programming`, `agent_monte_carlo`, `agent_q_learning`, `agent_sarsa`, `agent_diy`
- it relies on external scripts to switch the active algorithm
- `train_test.py` is used as a correctness smoke test before launching a real platform training task

That design matters because static config files do not always represent the true runtime selection.

## Repository Structure

```text
.
|-- train_test.py
|-- kaiwu.json
|-- conf/
|   |-- algo_conf_gorge_walk_v2.toml
|   |-- app_conf_gorge_walk_v2.toml
|   |-- configure_app.toml
|   `-- map_data/
|-- agent_dynamic_programming/
|-- agent_monte_carlo/
|-- agent_q_learning/
|-- agent_sarsa/
`-- agent_diy/
```

Important files and directories:

- `train_test.py`
  - smoke-test entrypoint
  - selects an algorithm
  - starts learner and aisrv
  - checks logs and model artifacts to decide success or failure
- `conf/algo_conf_gorge_walk_v2.toml`
  - maps algorithm names to agent classes and workflows
- `conf/app_conf_gorge_walk_v2.toml`
  - policy-side application config
- `conf/map_data/F_level_*.json`
  - transition graph data used by dynamic programming
- `agent_<name>/agent.py`
  - agent implementation
- `agent_<name>/algorithm/algorithm.py`
  - learning rule
- `agent_<name>/workflow/train_workflow.py`
  - training loop
- `agent_<name>/feature/definition.py`
  - sample conversion and reward shaping
- `agent_<name>/conf/train_env_conf.toml`
  - environment configuration for training

## Environment and Task

From the official docs:

- map size: `64 x 64`
- action space:
  - `0 = UP`
  - `1 = DOWN`
  - `2 = LEFT`
  - `3 = RIGHT`
- agent view: centered `5 x 5`
- goal:
  - reach the destination
  - use fewer steps
  - optionally collect treasures

Default training environment config:

- start: `[29, 9]`
- end: `[11, 55]`
- `treasure_random = false`
- `treasure_count = 0`
- `treasure_id = []`
- `max_step = 2000`

Scoring rules from the docs:

- reaching the end gives terminal score
- fewer steps gives step bonus
- each treasure adds score
- timeout leads to total score `0`

Important distinction:

- `score` is an environment-side performance metric
- `reward` is the learning signal and may be shaped separately in code

## Supported Algorithms

| Algorithm | State Encoding | Learning Style | Requires Training Samples | Saved Artifact |
| --- | --- | --- | --- | --- |
| Dynamic Programming | Position only | Value iteration or policy iteration | No online sample-heavy training | Policy table |
| Monte Carlo | Position only | First-Visit full-episode control | Yes | Policy table |
| Q-Learning | Position + treasure bits | Off-policy TD | Yes | Q-table |
| SARSA | Position + treasure bits | On-policy TD | Yes | Q-table |
| DIY | User-defined | User-defined | User-defined | User-defined |

Notes:

- `dynamic_programming` uses known transition data from `conf/map_data/F_level_1.json`
- `monte_carlo`, `q_learning`, and `sarsa` rely on environment interaction
- `diy` is a skeleton, not a ready-to-run algorithm

## State and Feature Design

The official docs describe three key concepts:

- `ObsData` for prediction input
- `ActData` for prediction output
- `SampleData` for training input

This repository implements those ideas in a simplified tabular form.

### Position-only state

Used by:

- `agent_dynamic_programming`
- `agent_monte_carlo`

Encoding:

- `pos_feature = x * 64 + z`
- state count: `64 * 64 = 4096`

### Position plus treasure state

Used by:

- `agent_q_learning`
- `agent_sarsa`

Encoding:

- derive `pos_feature = x * 64 + z`
- derive a 10-bit treasure mask from `organs`
- combine them as:
  - `feature = 1024 * pos_feature + treasure_binary`

State count:

- `64 * 64 * 1024`

This follows the official environment docs and agent docs closely.

## Training and Evaluation Semantics

The official framework docs define the contract as:

- `predict`
  - used during training
  - may include exploration
- `exploit`
  - used during evaluation
  - should usually be greedy or deterministic
- `learn`
  - consumes `SampleData`
- `observation_process`
  - converts raw environment output to `ObsData`
- `action_process`
  - converts `ActData` to action format accepted by the environment
- `save_model`
  - writes files containing `model.ckpt-{id}` in the filename
- `load_model`
  - reloads saved model state

In this repository:

- `predict` and `exploit` are explicitly separated in each algorithm package
- evaluation behavior differs from training behavior, especially for epsilon-greedy methods

## Training Workflow

The official workflow docs say the training loop should:

1. reset environment with `usr_conf`
2. obtain observations
3. call `agent.predict`
4. call `agent.action_process`
5. call `env.step`
6. compute reward
7. collect trajectory frames
8. call `sample_process`
9. call `agent.learn`
10. periodically save models

This repository follows that pattern, with algorithm-specific variations:

- `dynamic_programming`
  - skips online trajectory training and learns from map transition data
- `monte_carlo`
  - trains on full trajectories
- `q_learning`
  - trains step by step with Q-table updates
- `sarsa`
  - trains step by step with actual next action

## Runtime Selection Details

There is an important runtime mismatch to understand:

- `train_test.py` currently sets `algorithm_name = "q_learning"`
- `conf/app_conf_gorge_walk_v2.toml` still has `dynamic_programming` as the static initial value

This is not necessarily a bug by itself. The intended flow is that external scripts rewrite the runtime config before training starts.

Do not infer the actual algorithm from one config file alone.

## External Dependencies

This repository depends on a larger Kaiwu environment, including:

- `kaiwudrl`
- `common_python`
- `tools` modules imported from workflows and entry scripts
- shell scripts such as:
  - `tools/stop.sh`
  - `tools/modelpool_start.sh`
  - `tools/change_sample_server.sh`
  - `/root/tools/change_algorithm_all.sh`

Because of that, cloning this repository alone is usually not enough for a successful run.

## Single-Machine vs Distributed Training

According to the official `gorge_walk_v2` docs:

- `dynamic_programming`
- `monte_carlo`
- `q_learning`
- `sarsa`

are provided as single-machine algorithms.

If you want distributed training, the official expectation is to implement it yourself under `agent_diy`.

That also means:

- the standard distributed `SampleData2NumpyData` and `NumpyData2SampleData` path is not implemented in these built-in tabular packages
- adding distributed support is not just a config change; it requires interface work

## Monitoring and Logging

The official docs define four monitor groups:

- `basic`
- `algorithm`
- `env`
- `diy`

Examples of important built-in metrics:

- `train_global_step`
- `predict_succ_cnt`
- `sample_production_and_consumption_ratio`
- `episode_cnt`
- `load_model_succ_cnt`
- `sample_receive_cnt`

Repository-level guidance:

- keep the existing monitor reporting in workflows
- do not replace the platform logger with a custom logging system
- if you need extra metrics, prefer the `diy_1` to `diy_5` slots documented by the platform

## Model Save Limits

Official platform constraints:

- max save frequency: `2 times / minute`
- total save count:
  - Dynamic Programming: `10`
  - Monte Carlo: `100`
  - Q-Learning: `100`
  - SARSA: `100`

Do not add high-frequency save logic to workflows.

## Evaluation Constraints

Official evaluation notes:

- evaluation calls `agent.exploit`
- evaluation keeps start and end fixed at:
  - start: `[29, 9]`
  - end: `[11, 55]`
- treasure settings and `max_step` may vary at evaluation time

## Development Guidance

- If you change feature encoding, also update:
  - `STATE_SIZE`
  - Q-table or policy dimensions
  - `observation_process`
  - persistence format
- If you change reward shaping, also revisit:
  - convergence logic
  - metric interpretation
  - the distinction between environment score and RL reward
- If you change workflows, preserve:
  - `usr_conf` handling
  - sample collection
  - disaster recovery
  - monitor reporting
  - periodic model saving
- If you add a new algorithm, update:
  - `conf/algo_conf_gorge_walk_v2.toml`
  - runtime switching logic
  - documentation

## Current Limitations

- no complete dependency manifest such as `requirements.txt` or `pyproject.toml`
- external runtime scripts are not included here
- `agent_diy` is unfinished
- no self-contained automated test suite was found
- most model-related framework docs assume neural models, while this repository mainly uses tabular methods

## Recommended Use

Use this repository when you already have a working Kaiwu environment and want to:

- compare tabular RL methods on `gorge_walk_v2`
- understand how the official environment maps to concrete agent code
- bootstrap a custom algorithm from the provided template

If you want a standalone project, you will first need to supply the missing platform runtime, modules, and scripts.
