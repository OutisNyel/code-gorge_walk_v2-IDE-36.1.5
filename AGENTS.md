# AGENTS.md

## Repository Identity

This repository is a `gorge_walk_v2` reinforcement learning code package intended for the Tencent KaiwuDRL platform.

It is not a self-contained standalone project. It should be treated as a platform-facing algorithm package with local smoke-test support.

The repository currently contains five parallel algorithm packages:

- `agent_dynamic_programming`
- `agent_monte_carlo`
- `agent_q_learning`
- `agent_sarsa`
- `agent_diy`

## Platform Context

Official Tencent Arena documentation shows the standard KaiwuRL package layout as one `agent/` directory plus shared `conf/` and `train_test.py`.

This repository is a variant of that layout:

- it keeps one agent package per algorithm instead of one active `agent/` folder
- it relies on external scripts to switch the active algorithm at runtime
- `train_test.py` is the validation entrypoint that should complete one minimal training cycle before a platform training task is launched

Do not assume the static TOML files alone describe the actual runtime state. The runtime algorithm is expected to be synchronized by external tooling.

## External Dependencies

The code depends on platform modules and scripts not present in this repository, including:

- `kaiwudrl.*`
- `common_python.*`
- `tools.*`
- `/root/tools/change_algorithm_all.sh`
- `tools/change_sample_server.sh`
- `tools/modelpool_start.sh`
- `tools/stop.sh`

Without those dependencies, local execution cannot be treated as representative of the platform runtime.

## High-Level Training Flow

According to the official docs, the platform training loop is:

1. environment returns observation and reward
2. agent converts raw observation to `ObsData`
3. agent predicts `ActData`
4. agent converts `ActData` to environment action
5. environment advances one step
6. workflow collects trajectory data
7. `sample_process` converts trajectory data to `SampleData`
8. `agent.learn` consumes `SampleData`
9. model is periodically saved and later reloaded for prediction/evaluation

This repository follows that pattern, but mostly with tabular methods rather than neural models.

## Important Runtime Facts

- `train_test.py` currently defaults to `algorithm_name = "q_learning"`.
- `conf/app_conf_gorge_walk_v2.toml` still has a static initial algorithm of `dynamic_programming`.
- Runtime switching is expected to happen through external scripts invoked by `train_test.py`.
- `train_test.py` also chooses sample transport between `reverb` and `zmq` based on CPU architecture.
- The script clears logs and checkpoint files under `/data/ckpt/...` before the smoke test run.
- Training success is inferred from generated `model.ckpt-*` files.

## Environment Facts From Official Docs

- Map size: `64 x 64`
- Action space:
  - `0 = UP`
  - `1 = DOWN`
  - `2 = LEFT`
  - `3 = RIGHT`
- Default training env config:
  - start: `[29, 9]`
  - end: `[11, 55]`
  - `treasure_random = false`
  - `treasure_count = 0`
  - `treasure_id = []`
  - `max_step = 2000`
- Task goal:
  - reach the end
  - minimize steps
  - optionally collect treasures
- Important distinction:
  - `score` measures environment performance
  - `reward` is the RL training signal and may be shaped independently

## Algorithm Facts

### dynamic_programming

- Uses position-only state encoding.
- `STATE_SIZE = 64 * 64`.
- Reads `conf/map_data/F_level_1.json` directly.
- Solves from known transition data.
- Does not rely on online sample-heavy training in the same way as the other algorithms.
- Saves a policy array as `.npy`.

### monte_carlo

- Uses position-only state encoding.
- `STATE_SIZE = 64 * 64`.
- Collects full-episode trajectories.
- Applies First-Visit Monte Carlo control.
- Saves a policy array as `.npy`.

### q_learning

- Encodes position plus 10 treasure states into a single integer feature.
- `STATE_SIZE = 64 * 64 * 1024`.
- Updates Q-values with `max Q(s', a')`.
- Saves a Q-table as `.npy`.

### sarsa

- Uses the same combined state encoding as Q-Learning.
- Updates Q-values with the actual `next_action`.
- Saves a Q-table as `.npy`.

### diy

- Platform template only.
- Core methods are placeholders.
- Use it only as a starting point for custom algorithms.

## Framework Interface Contracts

Official framework docs define the following responsibilities.

### `feature/definition.py`

- define `ObsData`, `ActData`, and optionally `SampleData`
- implement `sample_process`
- implement `reward_shaping`
- for distributed training, also implement:
  - `SampleData2NumpyData`
  - `NumpyData2SampleData`
  - both with `@attached`

Note:

- this repository focuses on single-machine tabular algorithms
- distributed converters are not implemented here because the provided four standard algorithms are documented as single-machine only

### `agent.py`

- `predict`
  - training-time decision path
- `exploit`
  - evaluation-time decision path
  - receives raw environment observation in platform evaluation
- `learn`
  - consumes training samples
- `observation_process`
  - converts raw observation to `ObsData`
- `action_process`
  - converts `ActData` to environment action
- `load_model`
  - loads latest or specified model
- `save_model`
  - writes files whose names must include `model.ckpt-{id}`

### `workflow/train_workflow.py`

- owns the interaction loop between environment and agent
- resets the environment
- runs episode loops
- collects `Frame` trajectory items
- calls `sample_process`
- calls `agent.learn`
- periodically saves models
- should keep logging and monitor reporting in place

## Current Repository-Specific Implementation Choices

- The tabular algorithms use simple handcrafted features in `agent.py` instead of a separate heavy `preprocessor.py`.
- `agent_q_learning` and `agent_sarsa` follow the official combined feature idea:
  - `pos_feature = x * 64 + z`
  - `feature = 1024 * pos_feature + treasure_binary`
- `agent_dynamic_programming` and `agent_monte_carlo` intentionally keep only position features.
- `agent_q_learning/model/model.py` is effectively a placeholder; the real learning object is the Q-table in `algorithm.py`.
- `agent_diy` is only a scaffold and should not be described as production-ready.

## Monitoring and Logging

Official docs describe four monitor groups:

- `basic`
- `algorithm`
- `env`
- `diy`

Important built-in basic metrics include:

- `train_global_step`
- `predict_succ_cnt`
- `sample_production_and_consumption_ratio`
- `episode_cnt`
- `load_model_succ_cnt`
- `sample_receive_cnt`

Repository guidance:

- preserve monitor reporting in workflows
- do not replace the platform logger with a custom logging system, otherwise the monitor panel may stop showing error counts correctly
- if custom metrics are needed, prefer the documented `diy_1` to `diy_5` slots

## Model Save Constraints

Official docs define save restrictions on the platform:

- max save frequency: `2 times / minute`
- total save count limit:
  - Dynamic Programming: `10`
  - Monte Carlo: `100`
  - Q-Learning: `100`
  - SARSA: `100`

Do not add high-frequency saving logic.

## Evaluation Constraints

- Evaluation uses `agent.exploit`, not `agent.predict`.
- Evaluation keeps start and end fixed at `[29, 9]` and `[11, 55]`.
- Evaluation configuration mainly varies treasure settings and `max_step`.

## Safe Change Rules

- If you change state encoding, also update:
  - `Config.STATE_SIZE`
  - table sizes or policy structure
  - `observation_process`
  - sample fields
  - persistence logic
- If you change reward shaping, also review:
  - convergence conditions
  - monitoring outputs
  - the distinction between environment score and RL reward
- If you change workflows, preserve:
  - env reset with `usr_conf`
  - episode collection
  - disaster recovery handling
  - sample conversion
  - monitor reporting
  - periodic save behavior
- If you add distributed training support, you must add the SampleData/Numpy conversion path and revisit the workflow semantics.

## Documentation Expectations

- State clearly that the provided four standard algorithms are single-machine only.
- Do not claim platform training was verified unless the actual external Kaiwu runtime was used.
- If a new algorithm is added, update:
  - `conf/algo_conf_gorge_walk_v2.toml`
  - runtime selection logic
  - the new algorithm package
  - `README.md`
