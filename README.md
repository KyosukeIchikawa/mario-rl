# mario-rl

Value-based deep reinforcement learning agents that learn to play **Super Mario Bros** from raw pixels, built on [`gym-super-mario-bros`](https://github.com/Kautenja/gym-super-mario-bros) and TensorFlow/Keras.

The project implements several DQN variants behind a single, interchangeable agent interface so they can be compared on the same environment and training loop.

## Implemented algorithms

| `--rl` value        | Algorithm                                              | Notes |
| ------------------- | ------------------------------------------------------ | ----- |
| `ddqn`              | Double DQN                                             | Baseline |
| `dueling_ddqn`      | Dueling Double DQN                                     | Separate value/advantage streams |
| `noisy_dqn`         | Noisy DQN                                              | `NoisyDense` layers replace ε-greedy exploration |
| `prioritized_ddqn`  | Double DQN + Prioritized Experience Replay             | `SumTree`-backed PER |
| `categorical_dqn`   | Categorical DQN (C51) — [paper](https://arxiv.org/abs/1707.06887) | Distributional value learning |

## How it works

- **Environment pipeline.** The raw NES environment is wrapped as: `JoypadSpace` (restricted to the 7 `SIMPLE_MOVEMENT` actions) → skip every 4th frame (summing reward) → grayscale → resize to 84×84 → stack 4 frames.
- **Dual-input state.** Each agent observes both the stacked grayscale frames *and* a one-hot history of its last 4 actions. The Q-network has two input branches (a CNN over the frames, a dense branch over the action history) that are concatenated before the Q-value head.
- **Training loop.** `main.py` runs training episodes with ε-greedy (or noisy) exploration, periodically runs a greedy test episode, and logs metrics, records videos, and dumps experiences.

## Setup

Requires **Python 3.8**.

```bash
make setup        # creates .venv, installs requirements.txt, then `pip install -e .`
```

## Usage

Training must be launched from inside the `mario_rl/` directory:

```bash
cd mario_rl
python main.py --rl ddqn --world_stage 1-1 --episodes 5000
```

### Options

| Flag                    | Default  | Description |
| ----------------------- | -------- | ----------- |
| `--rl`                  | `ddqn`   | RL algorithm (see table above) |
| `--world_stage`         | `1-1`    | World and stage, e.g. `1-1`, `4-2` |
| `--episodes`            | `5000`   | Number of training episodes |
| `--test_frequency`      | `10`     | Run a greedy test episode every N episodes |
| `--video_frequency`     | `1`      | Record a video every N test episodes |
| `--use_test_experience` | `True`   | Also store experience collected during test episodes |

### Outputs

Each run writes to `data/{world_stage}_{rl}_{timestamp}/`:

- `train.log` / `test.log` — per-episode metrics (loss, reward, training step)
- `*.mp4` — test-episode gameplay with a per-action Q-value bar-graph overlay
- `*.npz` — compressed dumps of test-episode experiences

The `data/` directory is gitignored.

## Tests

```bash
make test         # runs pytest tests/ (from the repo root)
```

## Project layout

```
mario_rl/
  main.py            # entry point: env setup, agent dispatch, train/test loop
  mario_*.py         # one Mario agent per DQN variant (shared interface)
  gym_util/          # frame skip / grayscale / resize Gym wrappers
  rl_util/           # Experience, replay buffers (incl. PER + SumTree), NoisyDense
  math_util/         # streaming statistics
  visual/            # action-value overlay + mp4 export
tests/               # pytest suite
validation/          # standalone performance benchmarks
```
