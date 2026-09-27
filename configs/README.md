# Training configs

YAML configs for the training scripts in `train/`. Replaces previous
hardcoded hyperparameter dicts.

## Layout

Configs are grouped by algorithm:

| Path | Algorithm | Notes |
|------|-----------|-------|
| `ppo/maskable_ppo.yaml` | MaskablePPO | Default recommended setup |
| `ppo/ppo_baseline.yaml` | PPO | Vanilla baseline, no masking |
| `ppo/bootstrap.yaml` | MaskablePPO | Full multi-stage curriculum (entry point for production training) |
| `ppo/bootstrap_sweep/v*.yaml` | MaskablePPO | Per-axis sweep variants of `ppo/bootstrap.yaml` (entropy schedule, etc.) |
| `feudal/feudal_rl.yaml` | Feudal RL | Manager-Worker hierarchy |
| `self_play/self_play.yaml` | Self-play | With opponent pool |
| `alphazero/alphazero.yaml` | AlphaZero | MCTS + policy/value network |
| `imitation/bc_scenarios.yaml` | Behavior cloning | Demonstration scenario mix for BC warm-start |

## Usage

Training scripts accept `--config` and any CLI flag overrides:

```bash
python scripts/train/train_feudal_rl.py --config configs/ppo/maskable_ppo.yaml
python scripts/train/train_feudal_rl.py --config configs/ppo/maskable_ppo.yaml \
    --total-timesteps 50000 --seed 42
```

Load programmatically:

```python
from reinforcetactics.rl.config import load_config, apply_overrides

cfg = load_config("configs/ppo/maskable_ppo.yaml")
cfg = apply_overrides(cfg, {"ppo.learning_rate": 1e-4})
cfg.validate()
```

## Schema

See dataclasses in `reinforcetactics/rl/config.py` for the full schema.
Top-level sections: `env`, `ppo`, `feudal`, `self_play`, `alphazero`,
`curriculum`, `eval`, `logging`. Validation is strict:

- Unknown keys raise `ValueError`.
- Every value is coerced to its field's type (`learning_rate: 3e-4`, which
  PyYAML reads as a string, becomes a float) and range-checked.
- `reward_config` keys must be ones the env reads (`KNOWN_REWARD_KEYS`).
- Opponents must be named exactly as `bot_registry.accepted_names()` lists
  them (plus `self` for `env.opponent`), and `opponent_kwargs` must be
  constructor arguments of that bot, coerced to its types and range-checked
  (RandomBot's `max_actions` is an integer >= 1).
- `env.enabled_units` is `null` (all units) or a non-empty list of unit codes,
  and `env.engine_overrides` is resolved the way every game will resolve it.

No training script reads every field. Each one reports the fields a config
sets away from their defaults that the run ignores (for example
`total_timesteps` or `logging.*` for `train_bootstrap.py`, `feudal.*` for
`train_feudal_rl.py --mode flat`, the pool settings for a
`train_self_play.py` run without `use_opponent_pool`), as a warning by default
and as an error with `--strict`. The report follows the run's mode:
`train_self_play.py` reads `env.opponent` / `self_play.bot_ratio` only in
mixed mode, and `train_feudal_rl.py` reads `self_play.*` only with
`--opponent self`. `run_curriculum` (the notebook path) warns too. Every
config here except the `ppo/bootstrap_sweep` archive (which keeps an
informational `total_timesteps`) runs under `--strict`.

Command-line flags that override a config value are validated like the
config itself, so `--gamma 1.5` is a usage error. In
`train_bootstrap.py`, `--set KEY=null` (or `~`) unsets a field.
