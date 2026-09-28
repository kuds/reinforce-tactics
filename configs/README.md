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

### Inheritance and stage addressing

A config can be a diff against another one (`load_config` resolves it; the
run's `resolved_config.yaml` records the fully expanded result):

```yaml
extends: ../bootstrap.yaml          # relative to this file (or absolute); chains up to 8 deep
env:
  reward_config: {turn_penalty: -0.5}   # mappings merge key by key
  engine_overrides: {__replace__: true, damage_model: hp_scaled}   # replace instead of merging
curriculum:
  stage_defaults: {max_retries: 2}      # every stage that leaves the field unset or null
  stages:                               # merged by name: a known name deep-merges, a new one is appended
    - {name: skirmish_random_15, promotion_win_rate: 0.6}
  drop_stages: [starter_simple]         # applied (and removed) at this file's level
  stage_order: [...]                    # optional: a permutation of the remaining stages
```

Anything that is not a mapping (lists included) replaces, and `null` sets
null. A stage without a `name`, a repeated name, an unknown name in
`drop_stages`, or a `stage_order` that is not a permutation is an error.
`stage_defaults` fills field by field: a stage's own `reward_config` is kept
whole, not merged with the default's.

`train_bootstrap.py --set` addresses stages and keys inside mapping fields:

```bash
--set 'curriculum.stages[beginner_simple].promotion_win_rate=0.6'
--set 'curriculum.stages[*].patience=1'
--set env.reward_config.turn_penalty=-0.5         # null deletes the key
--set ppo.policy_kwargs.features_extractor_kwargs.pool=flatten
```

An unknown stage name is an error that lists the stages; every value is then
validated like the file's (reward keys included). `--seed N` is `--set seed=N`.
