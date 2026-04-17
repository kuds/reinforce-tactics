# Training configs

YAML configs for the training scripts in `train/`. Replaces previous
hardcoded hyperparameter dicts.

## Files

| File | Algorithm | Notes |
|------|-----------|-------|
| `maskable_ppo.yaml` | MaskablePPO | Default recommended setup |
| `ppo_baseline.yaml` | PPO | Vanilla baseline, no masking |
| `feudal_rl.yaml` | Feudal RL | Manager-Worker hierarchy |
| `self_play.yaml` | Self-play | With opponent pool |
| `alphazero.yaml` | AlphaZero | MCTS + policy/value network |
| `curriculum/easy.yaml` | MaskablePPO | Random opponent, generous shaping |
| `curriculum/medium.yaml` | MaskablePPO | Scripted bot, standard shaping |
| `curriculum/hard.yaml` | MaskablePPO | Long episodes, harsh penalties |

## Usage

Training scripts accept `--config` and any CLI flag overrides:

```bash
python train/train_feudal_rl.py --config configs/maskable_ppo.yaml
python train/train_feudal_rl.py --config configs/maskable_ppo.yaml \
    --total-timesteps 50000 --seed 42
```

Load programmatically:

```python
from reinforcetactics.rl.config import load_config, apply_overrides

cfg = load_config("configs/maskable_ppo.yaml")
cfg = apply_overrides(cfg, {"ppo.learning_rate": 1e-4})
cfg.validate()
```

## Schema

See dataclasses in `reinforcetactics/rl/config.py` for the full schema.
Top-level sections: `env`, `ppo`, `feudal`, `self_play`, `alphazero`,
`eval`, `logging`. Validation is strict — unknown keys raise
`ValueError`.
