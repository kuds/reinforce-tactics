# SB3 training diagnostics

`notebooks/ppo_training.ipynb` writes its resolved `config.json` before
training. `training_metrics.csv` is appended after each optimizer update;
`eval_results.jsonl` is appended after each completed development evaluation.
The compatibility JSON/CSV summaries are saved immediately after training,
including partial results if training raises, before plotting or videos.

## Read the plots together

| Plot | Useful question |
| --- | --- |
| `eval_curves.png` | Do outcomes improve, and how uncertain is each measurement? |
| `reward_by_outcome.png` | Are action rewards paying for draws or losses? |
| `episode_distributions.png` | Which outcomes hit the turn/step limits, and how long is the tail? |
| `optimization_diagnostics.png` | Are updates clipping heavily, and is the policy concentrating despite having choices? |
| `action_space_diagnostics.png` | Is the action table truncating, or is the agent declining available attacks/captures? |

The existing action, build, combat, reward, and end-reason plots remain available.
The new helpers tolerate older result files that lack the additional fields.
Missing outcome groups or action opportunities are gaps, not zero-valued results.
Episode distribution bands show the interquartile range. Reward/length bands in
the main evaluation figure show episode standard deviation, rather than a
confidence interval for the mean.

Normalized masked entropy is measured for flat-discrete policies as
`H(policy) / log(n_offered)` only where `n_offered > 1`. Forced decisions and empty
masks are reported separately. The callback uses an evenly spaced sample of at
most 256 rollout observations before the update, without shuffling the rollout
buffer or consuming the global Torch, NumPy, or Python random streams,
including when a custom extractor raises. Set
`TrainingMetricsCallback(policy_diagnostic_samples=0)` to disable that extra
forward pass. The same CSV row pairs these collection-policy measurements with
the subsequent optimizer metrics; they describe different phases of the update.

`seize_choice_rate` and `attack_choice_rate` divide valid selections by decision
points where that action was offered. They distinguish reaching an opportunity
from choosing to act on it. `seize_available_rate` is the fraction of all
decision points offering a seize action. Captures and first-capture timing then
show whether those attempts finish. A policy may still be correct to decline an
available action; these rates are diagnostic, not objectives.

## Evaluation and saved checkpoints

The notebook's development evaluation uses a fixed seed set, both player seats,
and greedy plus sampled play. Episodes are configured **per seat and mode**;
both seats and both modes cost four times a single-seat greedy evaluation.
Repeated development evaluations are paired measurements of the same problems.
Pointwise confidence intervals do not adjust for selecting the best checkpoint
from many evaluations, or measure variation between training seeds.

The final optimizer update can happen after the last periodic evaluation.
The notebook reloads `best_model.zip` and `final_model.zip` and evaluates both
on the same independent seed batch, writing each completed mode immediately to
`heldout_results.jsonl`. Report per-seat and per-mode results, and use multiple
independent training seeds when comparing configurations. Best/final videos use
a common recorded replay seed.

`include_episode_records=True` opts into richer evaluator results. Each episode
retains seed, seat, outcome, end reason, return, turns/steps, combat counters,
and (with `track_breakdown=True`) action and reward components. The notebook
persists these records inside the corresponding evaluation JSONL row so paired
comparisons and failure analysis do not require replaying a model. Other callers
retain compact results by default.

Failure traces can include `max_turns_draw` as well as `max_steps_truncate` via
`PeriodicEvalCallback(trace_end_reasons=(...))`. The notebook caps them at three
episodes per evaluation using `max_traces=3`; traces are stored beneath
`traces/eval_<timestep>/`. Full step traces can be large, so keep this limit small.

TensorBoard evaluation scalars use the evaluation's actual environment step,
without flushing pending optimizer metrics. JSONL rows also retain exact steps,
evaluation duration, and wall-clock timestamps. Periodic evaluations happen
during rollout collection, so an evaluation at step X and the optimizer row at
step X need not describe the same weights.

## Interpretation

Low KL can reflect convergence or conservative updates. Explained variance
depends on the return distribution; rising value loss alone does not identify a
critic failure. Frequent end turns can be forced by the mask, and a concentrated
build mix can be effective. Check outcomes, legal choices, reward scale, and
opportunities together before changing training settings.

`episode_stats["reward"]` now includes the terminal bonus, matching the sum of
rewards returned by the environment. Evaluation also retains invalid-action,
unit-loss, and counterattack totals that the environment already measured.
