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
- `reward_config` keys must be ones the env reads (`KNOWN_REWARD_KEYS`), and
  opponents / `opponent_kwargs` must be ones the bot registry knows.

No training script reads every field. Each one reports the fields a config
sets away from their defaults that it ignores (for example `total_timesteps`
or `logging.*` for `train_bootstrap.py`, `feudal.*` for
`train_feudal_rl.py --mode flat`), as a warning by default and as an error
with `--strict`. Every config here except the `ppo/bootstrap_sweep` archive
(which keeps an informational `total_timesteps`) runs under `--strict`.
