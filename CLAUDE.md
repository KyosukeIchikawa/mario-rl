# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

A collection of value-based deep RL agents (DQN variants) that learn to play Super Mario Bros via `gym-super-mario-bros`. Built on TensorFlow 2.13 + standalone `keras` 2.13, pinned to Python 3.8.

## Commands

```bash
make setup          # create .venv, install requirements.txt, then `pip install -e .`
make test           # pytest tests/   (run from repo root)
make clean          # remove .venv

# single test
.venv/bin/pytest tests/test_mario_categorical_dqn.py::test_extract_action_probs

# training — MUST be launched from inside mario_rl/ (see gotcha below)
cd mario_rl && python main.py --rl ddqn --world_stage 1-1 --episodes 5000
```

CI (`.github/workflows/pytest.yaml`) runs `make setup` + `make test` on Python 3.8.

## Architecture

**Entry point** `mario_rl/main.py` (`Main` class): builds the Gym wrapper stack, dynamically imports a `Mario` agent class based on `--rl`, then runs the train/test episode loop. Outputs land in `data/{world_stage}_{rl}_{timestamp}/` (gitignored): `train.log`, `test.log`, `.mp4` videos with an action-value bar-graph overlay, and zlib-compressed `.npz` experience dumps written on test episodes.

**Env wrapper stack** (order matters): `gym_super_mario_bros.make` → `JoypadSpace(SIMPLE_MOVEMENT)` (7 actions) → `SkipFrame(4)` → `GrayScaleObservation` → `ResizeObservation(84×84)` → `FrameStack(4)`. Custom wrappers live in `mario_rl/gym_util/`.

**Dual-input state.** State is a tuple `(stacked_frames, last_action_history)` — 4 stacked grayscale frames *and* a one-hot encoding of the last 4 actions (`rl_util/OneHotDeque`). Every agent's Keras network therefore has two `Input` layers (a conv stack for the image, a dense branch for the action history) that are concatenated. This threads through the whole pipeline; preserve it when editing networks or replay logic.

**Interchangeable agents.** Each `mario_rl/mario_*.py` defines a `Mario` class implementing the same interface, which is all `main.py` depends on:
- `__init__(state_shape, action_dim)` where `state_shape = ((frames_shape), (action_history_shape))`
- `act(state, train) -> (int action, Optional[np.ndarray] action_values)`
- `cache(exp: Experience)`, `learn() -> Optional[loss]`
- properties `.exploration_rate`, `.memory` (supports `len()`), `.cnt_called_learn`

Variants: `mario_ddqn` (baseline Double DQN), `mario_categorical_dqn` (C51 distributional, exposes `_V_MAX`), `mario_dueling_ddqn`, `mario_noisy_dqn` (NoisyDense layers replace epsilon-greedy exploration), `mario_prioritized_ddqn` (PER; its `memory.sample` returns `(indices, probabilities, experiences)` and it calls `memory.update_priorities`).

**`mario_rl/rl_util/`** — shared RL building blocks: `Experience` (NamedTuple), `ReplayBuffer` (FIFO), `PrioritizedReplayBuffer` (`SumTree`-backed) plus a slow `PrioritizedReplayBufferWithoutSumTree` reference impl, `SumTree`/`SumTreeForSpiky` (capacity must be a power of 2), and `layers/NoisyDense`. Replay buffers pickle+zlib-compress every experience to cut memory use (`compress=True`).

Supporting: `math_util/OnlineStats` (streaming mean/std), `visual/` (bar-graph overlay + mp4 export), `validation/` (standalone speed-benchmark scripts, not part of the test suite).

## Gotchas

- **Split import convention.** Training code (`main.py` and `mario_ddqn`/`mario_dueling_ddqn`/`mario_noisy_dqn`/`mario_prioritized_ddqn`) uses *bare* imports like `from rl_util import ...` / `from gym_util import ...`, so it only resolves when **cwd is `mario_rl/`** — always launch training from there. Tests, `validation/`, and `mario_categorical_dqn.py` use *package* imports (`from mario_rl.rl_util import ...`) and run from the repo root (where `pip install -e .` makes `mario_rl` importable). Match the surrounding file's style when editing.
- **Adding an agent touches two places in `main.py`.** The argparse `--rl` `choices` list and the dispatch block in `Main.__init__` (which maps each name to a `from mario_* import Mario`) must stay in sync — update **both**, or the new value is either rejected by argparse or accepted but unhandled (raises `ValueError`).
- Uses standalone `keras` (imported as `import keras`), not `tf.keras`.
