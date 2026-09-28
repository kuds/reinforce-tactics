# Full codebase review — 2026-09-26

**Scope:** the whole repository at `8f51132`: the core engine; rule-based, LLM, model and AlphaZero bots; the Gymnasium env; the PPO bootstrap and self-play pipeline; Feudal RL, AlphaZero/MCTS and behaviour cloning; the pygame shell and renderer; animations, replay playback and video export; menus, map editor, i18n and settings; persistence, tournaments, CLI and cloud; tests and CI. It also re-audits every earlier review in `docs/` against the current code.

**Companion file:** [`REVIEW_full_2026-09-26_findings.md`](REVIEW_full_2026-09-26_findings.md) has all 351 verified findings, each with its location, impact and fix. IDs in this document (for example `core-1`) point there.

## Status

| Section | State | Where |
|---|---|---|
| §1 P0 (1.1–1.9) | **Done.** Each item implemented with regression tests that fail on the old code, adversarially reviewed, and merged. | 1.1 `57036e2`; 1.2 merge `8d3e114`; 1.3 `1b946be`; 1.4–1.5 `666e0c7`; 1.6–1.8 `64f3741`; 1.9 `0944446`; integration fixes `ac6f999` (GUI offers only legal actions) and `f462e62` (LLM quota/retry follow-ups) |
| Core engine findings, phase 1 (`core-4`–`core-13`, `core-17`, `core-18`, `core-20`, `core-21`, `core-23`, `core-25`; `core-1`–`core-3` were §1) | **Done.** Three workstreams, each reviewed and merged, then integration-tested together: seeded RNG, pathfinding and terrain rules; fog-of-war knowledge and complete saves; teams, elimination, engine-side haste and turn start. | W3 merge `8dced54`; W2 merge `0293c5b`; W1 merge `6719287`; follow-ups `60cdb92` |
| Core engine findings, phase 2 (`core-14`, `core-15`, `core-16`, `core-19`, `core-22`, `core-24`) | **Done.** Behaviour-preserving refactors, each gated on byte-identical seeded bot games (plus gym env, replays and saves where they apply): one action API (`GameState.apply_action` / `is_legal`) that MCTS, the bots, the GUI, the gym env and the LLM and model bots use; one definition per rule (ability ranges, range scans, status ticks, ability methods); `GameState` split into `EngineConfig`, the serializer, the legal-action generator and a `FogOfWar` component (`game_state.py` 2934 → 2071 lines); dead padding plumbing and dead helpers removed; `constants.py` split into `rules.py` and `ui/assets.py` (the old module re-exports every name). Review fixes along the way: teammates no longer trigger fog-of-war ambushes, cancelled moves leave no scouting trace (in other units' attack snapshots or across a reload), `haste_refreshed` is saved, and re-selecting a unit refreshes the GUI's action list. | core-24 `c118fb2`; core-15 `08b4aeb`; core-19/22 `6908fbb`; core-16 part 1 `011f872`; core-14 `af8c591`; core-16 part 2 `d83d508`; review fixes `1829197` |
| Post-core triage (all non-core findings re-checked at `48856af`) | 57 closed by the §1/core work, 282 open or partial (0 critical, 19 high). Recommended order: GUI regression, rule-bot legality, MDP/config validation, eval gate and resumable runs, 3-seed validation run; GUI session work in parallel. | — |
| GUI stayed unresponsive with only bots to move (regression from §1.6 + core-7) | **Done.** One bot turn per frame, then events; only pause/save while a bot's seat is to move; a bot that does not end its turn has it ended instead of handing its seat to the human. | `302893d` |
| Rule-bot legality (`rulebots-1`, `-6`, `-8`, `-9`, `-14`, `-19`) | **Done.** Scripted bots attempt only accepted actions (refusals were 43–70% of moves), with a CI legality harness. **Before the next curriculum run:** the ladder shifted (MediumBot now beats AdvancedBot/MasterBot on beginner/intermediate), so re-validate the stage order; see the note under `rulebots-1` in the findings file. | `63d9c81` |
| MDP freeze and config validation (`rlenv-5`, `-6`, `-8`, `-9`, `-10`, `-11`, `-16`, `prior-13`, `rulebots-7`, `-11`, `rltrain-9`, `-10`, `-13`, `-22`) | **Done.** End reason from the engine; combat shaping on HP actually removed, symmetric for both sides; budget-gated masks; versioned flat decode tables (old checkpoints decode unchanged, mismatches raise); reward keys, opponents and config values validated; fog/engine overrides/obs scales reach every env builder; ignored fields warn (fail with `--strict`); run records match the envs. Self-play's latest-vs-pool opponent is now the explicit `self_play.latest_opponent_prob` (default keeps today's pool-only behaviour: **a decision for the next self-play run**). | `127b333` |
| Eval gate and resumable runs (`rltrain-4`, `-5`, `-6`, `-7`, `-8`, `-11`, `-12`, `-18`, `-19`, `-20`, `-21`, `prior-3`, `-5`, `-6`, `-16`, `critic-gaps-2`, `critic-integration-5`) | **Done.** Stages are gated on the stochastic policy (greedy recorded alongside; draws separate) with point/Wilson/rolling criteria; `train_bootstrap.py --resume` continues a killed run with its step count, eval timeline and promotion state; stalls retry once from the stage's best; LR schedules honoured; `agent_seat` and per-seat eval everywhere; batched eval. Defaults keep today's difficulty (point criterion, seat 1) except the stochastic gate. `docs/bot_ladder_2026-09.md` flags 5 of 33 bootstrap stages whose opponent is weaker than the stage before, with a recommended order: **the curriculum retune and 3-seed validation run come next.** | `5cd3871`, ladder `811ecca`, `c21b99f` |
| Seed replication for §2.1 (`rltrain-15`, tooling part) | **Done; the run itself is next.** `scripts/train/run_seeds.py` launches one `train_bootstrap.py --seed N --resume-if-exists` per seed (local, sequential or CPU-pinned parallel, or one resubmittable Vertex job per seed) and records the group; `scripts/eval/summarize_seeds.py` reports per seed and across seeds (stochastic and greedy W/D/L, captures by type, shaping share, steps, retries) and against v52a, from new or archived run dirs. Evals now also log reward components per outcome, opponent captures and wall time. Config: `extends:` with stage merge by name, `curriculum.stage_defaults`, `--set curriculum.stages[<name>\|*].<field>` and dotted keys into mappings. The sweep archive is not yet rewritten onto `extends`. | [`validation_run.md`](validation_run.md) |
| Curriculum retune and reward back-port for §2.1 (`rltrain-14`, `prior-4`; the `rulebots-1` re-order) | **Done; the run itself is next.** `bootstrap.yaml` carries the v52a reward values with every key explicit, `max_actions_per_turn` 60 and `max_steps` scaled so truncation never precedes the clock; 24 stages ordered by the scripted-bot ladder (no stage flagged on fresh seeds); a 95% Wilson gate on the stochastic policy pooled over both seats. `bootstrap_validation.yaml` is its first 8 stages at half budget. `self_play.yaml` trains the bootstrap MDP with `latest_opponent_prob` 0.5 and a pool that fills. | [`validation_run_config.md`](validation_run_config.md) |

Behaviour changes from §1 that matter when comparing against older runs:

- **Opponents no longer freeze.** A passing agent no longer freezes RandomBot / MixedBot(random), which changes every random-opponent curriculum stage. Archived results predate this.
- **The engine refuses illegal actions.** It checks turn, action budget, paralysis, range, ownership and game over. `multi_discrete` policies now see those refusals as invalid actions.
- **No more phantom counterattacks.** Out-of-range defenders no longer deal 1 damage back.
- **Tests and scenarios set up units with `GameState.place_unit()`,** not `create_unit()`.

Behaviour changes from core phase 1:

- **Teams and elimination.** The bundled 2v2 map now plays players 1 and 3 against 2 and 4. Teammates never attack or seize each other. In a free-for-all, losing your last HQ or last unit, or resigning, eliminates you and the game goes on. Replays recorded under the old end rules play back by them (`legacy_end_rules`, set automatically).
- **Haste is an engine rule.** The extra action now reaches the RL env, MCTS and LLM bots, which never got it before. The GUI no longer gets a third action.
- **Fog of war hides what it should.** Hidden enemies no longer shape the move mask; running into one is an ambush. Observations, the renderer and LLM prompts show out-of-sight structures as last seen. Every HQ is known from the start. Cancelling a move takes back what it revealed.
- **Every game owns a seeded RNG.** Tournaments, evals, AlphaZero and BC datasets are reproducible from their seeds. Archived seeded runs drew Rogue evades from the global `random` and will not match.
- **Saves are complete** (format version 2). A reloaded game offers the same legal actions as the unsaved one.
- **API notes for code outside the repo (phase 2):** `constants.UNIT_DATA` entries no longer carry `static_path`, `animation_path` or `color` (see `ui.assets.UNIT_ASSETS`); `Tile.get_color`/`color_for` moved to `ui.assets.tile_color`; `GameState.set_map_metadata` and the padded/original coordinate converters are gone (they were never set, so every game used one frame).
- **Unchanged by default:** paralysis still costs the victim two turns (the constant now says 2, the value it always had in effect), and Player 1 still plays turn 0 without income (`begin_first_turn` opts in). The new terrain rules are all off.

---

## How the review was done

- **13 subsystem reviewers** each read their whole scope (17–87 files each). They reproduced suspected bugs with scratch scripts: real env rollouts, headless pygame, cProfile, and round-trips of every shipped save and replay.
- **Every finding went to an adversarial verifier** whose instructions were to refute it against HEAD. No finding was refuted outright, but 70 of the 351 were narrowed ("partially": real, but with a smaller scope or a different mechanism). Many severities were lowered. **All ratings below are the verifiers'.**
- **Two critics** then looked for what the reviewers missed. One checked for files no reviewer had read; the other checked for inconsistencies *between* subsystems. They added 25 findings.
- The headline bug (§1.1) was reproduced again, independently, for this write-up.
- The Google Drive training archive was checked for runs newer than the July review (§0).

**Baseline at HEAD**

| Check | Result |
|---|---|
| ruff check / ruff format | Clean |
| mypy | Clean, but 10 modules are `ignore_errors` and hide 100+ errors (`tests-8`) |
| Tests | 1439 pass in about 2 minutes |
| Coverage | 75% as configured; **66%** once `ui/`, `app/` and `cli/` stop being excluded (`tests-6`) |

**Result:** 8 critical, 39 high, 135 medium, 148 low and 21 info findings. That is about **264 distinct issues** once duplicates are merged (the same bug is often reported by two or three reviewers).

---

## TL;DR

**The core is in good shape.**

- `game_state.py` and `mechanics.py` have 95–97% test coverage.
- A 120-game random-play fuzz across every 1v1 map, with fog of war on and off, held every engine invariant.
- Seeded play is reproducible across processes.
- The `flat_discrete` masks are exact, and gymnasium's `check_env` passes.
- The July review's correctness fixes really landed, except the stochastic eval gate, which landed only half (§2.2).
- The recent consolidation pass shows: one observation builder, one mask layout, a bot registry, and a shared callback base.

**The problems sit at the boundaries, where one subsystem trusts another.**

1. **The engine trusts its callers, and one of its caches goes stale.**
   - `GameState.end_turn()` never invalidates the legal-action cache. If the agent passes a turn, a RandomBot opponent reads a stale, empty action list and does nothing. This lasts for as long as the agent keeps passing. Reproduced: with the bug the opponent ends with 2 units and 3,200 gold banked; with a one-line fix it has 12 units.
   - RandomBot is the opponent in 11 of the 33 stages of `bootstrap.yaml`, and the "easy" side of 3 more `mixed` stages. So the archived bootstrap runs trained against an opponent that a passive policy can freeze. That is a free draw, and it feeds the draw attractor that the last two reviews tried to fix through reward shaping.
   - Separately, `create_unit`, `attack`, `seize`, `heal` and the other action methods do almost no legality checking: none for whose turn it is, the per-turn action budget, range or tile ownership. Any caller that doesn't pre-filter can make illegal moves that the engine applies and records. Those callers are multi_discrete policies, LLM bots, the rule bots' knight charge and the GUI.
2. **Every training path except bootstrap is broken in a way that makes its results meaningless.**
   - Self-play isn't self-play. The seat swap lands on a wrapper, the opponent is unmasked, subprocess envs never get an opponent, and the documented trainer crashes on its first rollout.
   - Feudal runs use 4× their timestep budget, and with more than one env the self-play opponent never moves.
   - AlphaZero can only ever build Warriors and cannot represent about 52% of legal moves.
   - The BC recorder labels bot actions before the engine accepts them; 18–53% of its demonstrations are illegal.
   - The README's `--mode train` crashes on an undeclared dependency.
3. **The GUI has crash and data-loss paths in normal use.**
   - Picking the offered 1v1v1 mode crashes the app.
   - One random-map replay makes "Watch Replay" crash every time from then on.
   - Random-map saves never load.
   - "Load Game" asks for the save twice.
   - "Custom Model" can never start a game.
   - Bot turns run synchronously inside the click handler. The window freezes during LLM turns, bot moves are never shown, and the walking animation the project built is never called.
4. **Nothing has tested the current pipeline.** The last reinforce-tactics run in Drive dates from 2026-06-03. Since the commit behind the deepest runs, 22 commits have changed the RL, core and bot code. All 109 archived runs used seed 42 and identical PPO hyperparameters.

**Recommended order**

1. Fix the P0 list (§1, about 3–4 days, mostly small fixes).
2. Run a 3-seed validation of `bootstrap.yaml` on the fixed code before any new sweep variant (§2).
3. Make the GUI trustworthy and bot turns visible (§4–5).
4. Build the engine-owned action layer that removes the dispatch-duplication class for good (§6).

---

## 0. What the training archive says

The run archive is in `MyDrive/reinforce-tactics/benchmarks/bootstrap/`. `runs_summary.csv` was last regenerated on 2026-07-12.

| Fact | Value |
|---|---|
| Runs | 109, created 2026-05-08 → **2026-06-03** (none since) |
| Runs with at least one eval / at least one stage cleared | 56 / 48 |
| Deepest | 20 of 33 stages (`v50_hp_scaled_damage`, `v52a_maxturn_scaled_draw`), stopping at `skirmish_random_15` |
| `flat_discrete` runs | 58: max 20 stages cleared, mean 4.0 |
| `multi_discrete` runs (v33 BC warm-start, `skirmish_bc_selfplay`) | 14: **0 stages cleared** |
| Seeds | 42 in every run that recorded one |
| PPO hyperparameters | Identical in every run: γ 0.99, lr 3e-4, n_steps 2048, batch 256, clip 0.2, 8 envs, `SpatialFeatureExtractor` |
| Code since the deepest runs (`078313e`, 2026-06-01) | 22 commits to `rl/`, `core/`, `game/`, including `568fbc9`, `7fedf3e` and `c9e8212` ("Fix four RL correctness defects that distort what every run measures") |

What this means for the review:

- **The archived results were produced with the opponent-freeze bug (§1.1).** `end_turn` has no cache invalidation at the deepest runs' commit (`078313e`) or the latest run's (`59f6917`). The May commits aren't in this shallow clone, so they weren't checked. This bug sits under the conclusions in `bootstrap_lessons_learned.md`, `REVIEW_ppo_training.md` and `REVIEW_rl_pipeline_2026-07-24.md`. Part of the reward-shaping story they tell may really be a story about this bug. Re-baseline before drawing further conclusions from the archive.
- **One seed with fixed hyperparameters cannot tell a configuration effect from seed variance.** This is the July review's §2.14, and it still stands.
- **The legality gap (§1.2) matters most where the archive never learned.** `flat_discrete` masks are exact, so the production runs were largely protected. The 14 `multi_discrete` runs were exposed, and none of them cleared a stage. Correlation, not proof, but it is cheap to rule out once §1.2 lands.

---

## 1. P0 — fix now

These invalidate training results, crash in normal use, lose data, or leak credentials. Almost all are small.

| # | Problem | Where | Effort | IDs |
|---|---|---|---|---|
| 1.1 | `end_turn()` never invalidates the legal-action cache | `core/game_state.py:1192`, cache read at `:1305` | S | `core-1` `rlenv-1` `prior-1` |
| 1.2 | Engine action methods skip turn, action-budget, range and ownership checks, and out-of-range fights deal phantom 1-damage | `core/game_state.py:651` (create), `:787` (attack), `:1025` (seize); `core/mechanics.py:223`, `:445` | M | `core-2` `core-3` `rlenv-2` `aibots-1` `rulebots-2` `prior-11` |
| 1.3 | Self-play pipeline isn't self-play, and its trainer crashes | `rl/self_play.py:332`, `:545`, `:552`, `:652`; `scripts/train/train_self_play.py:101`, `:149` | M | `prior-2` `rltrain-1` `rltrain-2` `critic-gaps-1` `rlenv-12` `consolidate-4` |
| 1.4 | GUI crashes: 1v1v1 mode, Watch Replay after a random-map game | `ui/menus/game_setup/player_config_menu.py:43`; `ui/menus/save_load/replay_selection_menu.py:111` | S | `pygame-1` `menus-2` `menus-1` |
| 1.5 | Random-map saves can never be loaded | `app/game_loop.py:329`; `core/game_state.py:1446`; `core/grid.py:86` | S | `pygame-2` |
| 1.6 | `tqdm`/`rich` used via `progress_bar=True` but never declared; default Docker CMD and CLI train crash | `cli/commands.py:113`; `scripts/train/train_self_play.py:260`, `:398`; `scripts/train/train_feudal_rl.py:231`; `examples/` | S | `tests-1` |
| 1.7 | `.dockerignore` lets `docker/tournament/.env` (LLM API keys) into images via `COPY . .` | `.dockerignore` | S | `tests-2` |
| 1.8 | A preempted or cancelled Vertex bootstrap job loses its whole run directory | `scripts/train/train_bootstrap.py:417` | S | `rltrain-3` `prior-10` |
| 1.9 | ClaudeBot always sends an assistant prefill, which returns a 400 on every current Claude model | `game/llm_bot.py:1336`; model list at `:40` | S | `aibots-3` |

### 1.1 Stale legal-action cache after `end_turn()`

`end_turn()` changes almost everything the cache depends on. It resets `can_move`/`can_attack`, ticks paralysis and cooldowns, pays income, heals, and changes `current_player`. But it never calls `_invalidate_cache()`. `get_legal_actions()` returns `_legal_actions_cache[player]` whenever the flag is set.

As a result, whenever a turn passes with no state change, the next player acts on stale data:

- **RandomBot freezes.** Its last turn ended on "no legal actions", which cached an empty list. If the agent then passes, RandomBot keeps reading that empty list. This was reproduced in the real env with `flat_discrete`: RandomBot made `[2, 0, 0, 0, 0, 0, 0, 0, 0, 0]` actions per turn. With a one-line fix it made `[2, 3, 4, 5, 7, 11, 9, 11, 11, 16]`.
- **The agent can soft-lock itself.** Against `noop`, a turn that ends with an end-turn-only mask keeps the agent end-turn-only for the rest of the episode. One verifier measured 284 of 300 steps stale.
- **The freeze is not permanent.** It lasts only while the agent keeps passing whole turns. That is exactly the behaviour it rewards.
- **`MixedBot` freezes the same way when it draws RandomBot.**

**Who is exposed.** Any reader whose last `get_legal_actions` call wasn't followed by a state change can get a stale answer: the agent's own mask, ModelBot, the LLM bots and MCTS. BalancedRandomBot and SimpleBot were measured unaffected in the same scenario, because they always act after their last read.

Two tests hide the bug by calling `_invalidate_cache()` by hand: `tests/test_game_state.py:96-132` and `tests/test_rl_masking.py:164-206`.

**Fix**

- Call `self._invalidate_cache()` right after the `game_over` guard in `end_turn`.
- Harden it: replace the boolean flag with a state-version counter that every mutator bumps.
- Add an `RT_CHECK_CACHE=1` debug mode that recomputes the actions and asserts they equal the cache, and run it in CI (`core-17`).
- Then re-run one noop sanity stage and one `random_15` seed to measure how much the draw rate moves.

### 1.2 Make the engine the only enforcer of the rules

`GameState.create_unit` checks only the unit cap, occupancy, a known unit type and gold. It does not check that the tile is an owned building, is in bounds, or allows that unit type. `attack`, `seize`, `heal` and the other abilities do not check turn, `can_attack`, paralysis, range or `game_over`.

An out-of-range attack still deals 1 damage, because `apply_defence_reduction` returns `max(1, ...)` (`mechanics.py:223`). For the same reason, every counterattack from a defender who can't reach deals at least 1 (`mechanics.py:445`, `core-3`, first raised in `REVIEW_ppo_training.md` §2.7).

The illegal moves actually happen:

- **multi_discrete policies** (the `TrainingConfig` default, used by 6 shipped configs) can spawn units anywhere and attack across the map. Their per-dimension masks over-approximate the legal set, so these combinations are reachable.
- **LLM bots** can capture a 50-HP HQ in one turn by repeating SEIZE (reproduced), and can hit their own units.
- **Advanced and Master bots** make about 9 illegal out-of-range melee attacks per game, because the knight charge and rogue flank ignore the result of `move_unit` (`rulebots-2`).

**Fix**

- Every action method rejects the action when:
  - the game is over;
  - it isn't the actor's player's turn;
  - the actor is paralyzed, or the action it needs is already spent;
  - the target is out of range or on the wrong side (for `create_unit`: the tile isn't an owned, empty building, or the unit type is disabled).
- Reuse the predicates `get_legal_actions` already uses.
- Skip the counterattack when base counter damage is ≤ 0.
- This is also the first step of the action layer in §6.1.

### 1.3 Self-play: fix it, or refuse to run until fixed

All of the following are verified at HEAD:

- `self.env.agent_player = 2` (`self_play.py:545`) writes to the `ActionMaskedEnv` wrapper, which has `__getattr__` but no `__setattr__`. The base env stays player 1. In swapped episodes, rewards, masks and potential are scored for the wrong seat.
- The opponent's `predict()` gets no action masks (`:332`). In `flat_discrete` it resolves indices against the agent's action list. Measured: 0 valid opponent actions in 300 steps.
- Under the default `--n-envs 8` `SubprocVecEnv`, the envs are discovered through `hasattr(self.env, "envs")`, which finds 0 envs (`:652`, `train_self_play.py:101`). No opponent weights are ever pushed.
- `SelfPlayEnv.action_masks()` returns the per-dimension mask tuple, so MaskablePPO crashes on its first rollout (`critic-gaps-1`). A unit test even asserts the tuple.
- `--mode mixed` builds the bot envs and then throws them away (`consolidate-4`).
- The ROADMAP and `REVIEW_advancedbot.md` both say the swap fix landed (`prior-7`).

**Fix:** follow `rltrain-1`:

- set the seat through `env.unwrapped`, before `reset`;
- build the opponent's observation and action list for its own seat, and pass masks to `predict`;
- push weights with `env_method`, or refuse to start under `SubprocVecEnv`;
- return the concatenated mask.

Add the regression tests in `tests-3` first, marked `xfail(strict=True)`.

### 1.4–1.5 GUI crash and data-loss paths

- **1v1v1 crashes.** `GameModeMenu` lists every `maps/` subfolder, so it offers 1v1v1. `PlayerConfigMenu` then raises `ValueError` for anything but 1v1/2v2, and nothing catches it. Short term, whitelist the supported modes. Real fix: derive the player count from the map's HQ owners.
- **Watch Replay crashes.** Random-map replays save `"map_file": null`. `os.path.basename(None)` then raises in the replay picker on every open, and the replay is auto-saved on every game over. Use `game_info.get("map_file") or ""`.
- **Random-map saves never load.** `to_dict` always writes `map_file` (null for random maps), and `Grid.to_dict` stores only capturable tiles, so the terrain is gone. Persist the full terrain whenever `map_file_used is None`. While there, also persist `max_turns` and `end_reason` (`core-6`).
- **Any bot or menu exception kills the whole app.** Wrap the menu and session loop in a top-level error dialog that autosaves a crash save and the replay (`pygame-14`).

### 1.6–1.9 Shipping and credentials

- **1.6** Declare `tqdm` and `rich`, or depend on `stable-baselines3[extra]`. Add a test that scans imports and fails on undeclared third-party packages (or run `deptry`).
- **1.7** Add `**/.env`, `docker/tournament/output/` and `**/*credentials*.json` to `.dockerignore`. Better still, switch to an allow-list.
- **1.8** Add a SIGTERM handler in `train_bootstrap.main` so its `finally` upload runs, and sync `benchmarks/bootstrap` periodically. Also return a non-zero exit code on a stall.
- **1.9** Remove the prefill and get JSON through structured outputs (`output_config.format`) or system-prompt instructions. Update `ANTHROPIC_MODELS`:
  - it labels `claude-opus-4-6` "latest";
  - it still lists `claude-opus-4-1-20250805`, which retired on 2026-08-05;
  - it lists no 5.x models.

  Pair this with `aibots-4` (§4.7). As things stand, every failure is silently turned into a passed turn, so a tournament records an API error as a loss for the model.

---

## 2. P1 — RL pipeline: make the next run count

The bootstrap path is fundamentally sound. The problem is that runs can't yet be trusted or survive what kills them. Do these, then run the validation experiment in 2.1.

### 2.1 The validation run (after P0)

Run `bootstrap.yaml` with the §2.4 backport, **3 seeds**, on the fixed code. Report per seed:

- win / draw / loss per stage, measured with the stochastic policy;
- captures by type;
- the shaping share of return.

Compare against v52a. Until this exists, no new sweep variant is interpretable. It needs seed replication in the sweep tooling; see `rltrain-15` in §6.

### 2.2 Promotion, evaluation and run lifecycle

| Item | Where | Effort | IDs |
|---|---|---|---|
| Gates still score the deterministic argmax policy. Gate on the stochastic policy (or record both), and use a Wilson lower bound or a rolling mean instead of raw consecutive crossings | `rl/callbacks.py:276`, `:458` | S | `rltrain-4` `prior-5` `rltrain-12` |
| A stall ends the whole run. Retry once from `best_model.zip`, add a within-stage regression guard, add `--resume <run_dir>` / start-at-stage-K, and exit non-zero on a stall | `rl/bootstrap.py:624`, `:915` | M | `rltrain-5` `rltrain-7` `prior-3` |
| The canonical `bootstrap.yaml` still ships the reward terms the sweep showed cause the stall. Back-port the v52a/v54 values, set `max_actions_per_turn`, scale `max_steps` with `max_turns` | `configs/ppo/bootstrap.yaml:157` | S | `rltrain-14` `prior-4` |
| `ppo.lr_schedule` is dropped before SB3. Add an `LRScheduleCallback(ScheduledAttrCallback)` | `rl/config.py:137` | S | `prior-6` `rltrain-8` |
| The promoting eval never reaches TensorBoard; the stage's first rollout metrics describe the previous stage; chart normalisation is misleading | `rl/callbacks.py:472`; `rl/bootstrap.py:817`; `rl/viz.py:475` | S | `rltrain-19` `rltrain-18` `rltrain-20` |
| Eval runs one env at a time in the main process. Vectorise it over a fixed seed queue | `rl/evaluation.py:215` | M | `rltrain-11` |

### 2.3 MDP correctness

| Item | Where | Effort | IDs |
|---|---|---|---|
| **No Player-2 seat.** Every checkpoint is trained and gated as the first mover only, but tournaments and the GUI deploy it in both seats. Add `agent_seat: 1 \| 2 \| "random"`, and evaluate both seats | `rl/gym_env.py:490` | M | `critic-gaps-2` `critic-integration-5` |
| **Fog of war leaks.** Shrouded structures expose their live owner and HP in `to_numpy`; the move mask reveals invisible enemies; the last-seen memory is written but never read; the LLM serializer leaks the same way | `core/game_state.py:1536`; `game/llm_bot.py:636` | M | `core-5` `rlenv-14` `consolidate-1` `critic-integration-3` |
| **An unknown opponent string (including `"master"`) silently means no opponent.** Validate against the registry | `rl/gym_env.py:1655`; `rl/config.py:200` | S | `rulebots-7` `rlenv-10` `rltrain-22` |
| **Combat shaping ignores counter-damage and attacker death;** `damage_taken` is netted against turn-start healing | `rl/gym_env.py:1138` | S | `rlenv-9` |
| **`end_reason` is inferred from the board,** so HQ wins are labelled elimination and get the wrong terminal reward | `rl/gym_env.py:1439` | S | `rlenv-5` |
| **`flat_discrete` truncation drops attacks, heals and casts before moves;** truncation is invisible to diagnostics | `rl/gym_env.py:306` | S | `rlenv-11` `prior-13` |
| `structured_action_masks()` ignores `max_actions_per_turn` | `rl/gym_env.py:849` | S | `rlenv-8` |
| Config values are neither type- nor range-checked (`'3e-4'` loads as a string, `eval_freq=0` passes); known keys are silently ignored by their consumers | `rl/config.py:624`; `rl/bootstrap.py:723` | S–M | `rltrain-10` `rltrain-9` |
| `render_mode='rgb_array'` returns None; human mode never flips | `rl/gym_env.py:1677` | S | `rlenv-7` `tests-7` |

### 2.4 Throughput

- **The opponent's pathfinding dominates `step()`.** It takes 64–81% of wall time: `can_move_to_position` scans every unit on each BFS probe (`core/mechanics.py:58`). Build an occupancy map once per legal-action pass and cache reachable sets per state version. Expect 2–4× faster opponent turns. `rlenv-3` `core-20` `prior-17`
- **`find_killable_targets` runs a BFS per (enemy, unit) pair** and again after every kill. Hoisting it gives a 1.8–2.7× speedup with byte-identical games. `rulebots-9`

### 2.5 Experiments the archive never ran

Once 2.1 exists, these are the cheapest high-information runs (`rltrain-23`, `prior-27`), each one knob off v52a at 3 seeds:

- `features_extractor_kwargs.pool: flatten`, which could explain why HQ captures stay at 0;
- `gamma: 0.997` with `max_actions_per_turn: 25`;
- value normalisation (`clip_range_vf`, or `VecNormalize(norm_reward=True)`).

The long-term unblocker is to move the autoregressive head out of `feudal_rl.py` into a structured/pointer PPO policy (`prior-28`, L). That removes the positional flat head, its aliasing, and its 512/1024 truncation ceiling.

---

## 3. P1b — Alternative algorithms: fix cheaply or freeze

Each of these pipelines has clean core parts: shared mask builders, a correctly masked AR head, and a consistent MCTS sign convention. But none of the training drivers is tested (`rlalt-19`), and each has bugs that make its output meaningless.

| Pipeline | Must-fix before trusting any result | Effort |
|---|---|---|
| **Feudal RL** | **Timestep budget:** runs use n_envs× the budget, and a linear LR sits at 0 for the last 75% (`rlalt-1`) | S |
| | **Self-play:** the opponent factory is installed only on the first env, so with n_envs > 1 the opponent never moves (`rlalt-2`) | S |
| | **Deterministic eval:** the legacy 6-head worker deadlocks, with 297 of 300 steps invalid. Make `autoregressive_worker: true` the default (`rlalt-3`) | S |
| | **`--config` runs:** crash at the end on `json.dump(vars(args))` (`rlalt-15`), and the default `--mode flat` ignores `algorithm: feudal` (`rlalt-16`) | S |
| | **Returns:** the manager reward leaks across rollout boundaries, truncation is treated as terminal, and the manager target is undiscounted (`rlalt-9` `rlalt-11` `prior-8`) | S–M |
| **AlphaZero / MCTS** | **Action encoding:** the flat index drops unit type and source, so only Warriors can be built and ~52% of legal moves are unreachable (`rlalt-4`) | M |
| | **Evaluation:** the candidate is evaluated in train mode, corrupting BatchNorm (`rlalt-6`); each "epoch" is one minibatch (`rlalt-7`); self-play has no `max_turns` and all-draw evals return 0.5 (`rlalt-8`) | S each |
| | **Resume** is broken (`rlalt-5`) | S |
| | **AlphaZeroBot** ignores the saved architecture and isn't registered anywhere (`aibots-10` `rlalt-14`) | S |
| **Behaviour cloning** | **Recorder:** it saves labels before the engine accepts the action; 18–53% of demonstrations are illegal (`critic-integration-1`) | S |
| | **Action space:** it supports only multi_discrete, the space that never learned (`rlalt-21` `prior-12` `rltrain-17`) | L |
| | **Validation:** there is no held-out set (`rlalt-20`) | S |
| **Cross-cutting** | Heal and cure share action index 4: the env and ModelBot resolve it as cure-first, MCTS as heal-first, and BC labels round-trip wrongly (`critic-integration-9`) | S |

**Recommendation**

- **Feudal:** make the S-sized fixes now, about a day. It is 91% covered by tests and is the path to the structured policy in §2.5.
- **AlphaZero:** mark it experimental in the README and ROADMAP until the action encoding is fixed. It isn't reachable from the GUI or tournaments anyway.
- **BC:** decide between porting it to `flat_discrete` and retiring it. Fix the recorder either way.

---

## 4. P2 — Game correctness and pygame UX

### 4.1 Input and session state machine (`app/`)

| Bug | Where | Effort | IDs |
|---|---|---|---|
| SPACE during target selection leaves target mode armed, so the next player's click makes the previous player's unit act | `app/input_handler.py:118` | S | `pygame-3` |
| Bot turns run only after a human ends a turn. A Player-1 bot's first turn is played by the human, and all-bot games stall | `app/input_handler.py:393` | S | `pygame-5` |
| With a unit selected, clicking your own empty building opens the purchase menu, so units can never move onto their own buildings | `app/input_handler.py:372` | S | `pygame-7` |
| Double-clicking End Turn (or input queued during a bot turn) silently skips the human's next turn | `app/input_handler.py:157` | S | `pygame-11` |
| Pressing S shrinks the game window to 900×700 for the rest of the session | `app/game_loop.py:163` | S | `pygame-4` `menus-12` |
| Save & Quit quits even when the save was cancelled or failed | `app/game_loop.py:145` | S | `pygame-20` |
| Load Game makes the player pick the save twice | `ui/menus/main_menu.py:100`; `cli/commands.py:273` | S | `menus-3` `consolidate-10` |
| Enter confirms a dialog even when Cancel has keyboard focus | `ui/menus/in_game/confirmation_dialog.py:34` | S | `menus-4` |
| The HUD is drawn over the playfield, and Resign/End Turn capture clicks meant for tiles. Reserve a HUD strip and route all screen→grid math through one `screen_to_grid()` | `ui/renderer.py:369` | M | `pygame-8` |

### 4.2 The GUI's rules differ from the engine's

The unit action menu builds its own target lists instead of reading `get_legal_actions()`. As a result:

- attacks ignore the fog-of-war "move to discover, then attack" rule (`pygame-9`);
- Paralyze accepts adjacent targets only, skips the cooldown and already-paralyzed checks, and still ends the Mage's turn when the cast fails (`critic-integration-2`);
- the purchase menu prices units from the global `UNIT_DATA` and ignores the unit cap (`critic-gaps-9`).

**Fix:** build every menu entry from `game.get_legal_actions(unit.player)`, filtered to that unit.

### 4.3 Bots in the GUI

- **ModelBot can't be used from the GUI.**
  - `_validate_model` passes a list of lists where a DataFrame is expected, so it always fails (`menus-5`).
  - If you get past that, the padded UI maps fail the checkpoint size check, and `bot_factory` silently substitutes SimpleBot (`aibots-2`).
  - Record padding metadata (`set_map_metadata` is never called in production) and have ModelBot crop to the original window.
- **LLM bots are hidden when their keys come from environment variables** (`menus-9`).
- **Bot turns block the main thread** (`pygame-6` `anim-2` `aibots-5`); see §5.2 item 1.

### 4.4 Rule-based bots (the curriculum's opponents)

The root cause of most of these is one helper. `BotUnitMixin.get_reachable` (`game/bot_base.py:227`) returns pass-through tiles, including tiles occupied by allies, and ignores `can_move`. Callers never check what `move_unit` returns.

| Effect | Effort | IDs |
|---|---|---|
| 68% of SimpleBot moves are rejected, and 19 of 20 SimpleBot mirror games end in a max-turns draw | S | `rulebots-1` |
| Illegal out-of-range melee attacks from knight charge and rogue flank (fixed by §1.2 plus a return-value check) | S | `rulebots-2` |
| No-progress recursion after failed moves; every level claims another capture target | S | `rulebots-6` |
| Damage predictions ignore defence, buffs and `hp_scaled`, so 24–42% of "sure kills" fail. Needs an engine preview (§6.2) | M | `rulebots-3` `consolidate-3` |
| Knight charge and flank are scored from the pre-move tile; ranged units walk to melee range and don't shoot | S | `rulebots-4` `rulebots-5` |
| Scripted bots see and target units hidden by fog of war | M | `rulebots-12` |
| Haste re-entry never fires for live units | S | `rulebots-8` |

**Note for RL:** fixing `rulebots-1` makes SimpleBot noticeably stronger. The curriculum thresholds for SimpleBot, MediumBot and AdvancedBot stages will need to be re-baselined, so land this together with the 2.1 validation run rather than in the middle of a sweep. Measured on the fix, AdvancedBot and MasterBot also fall further behind MediumBot on the beginner and intermediate maps (AdvancedBot loses 45 of 50 seeded games to MediumBot on beginner, up from 33), and MasterBot is weaker on skirmish. So the re-baseline must also re-check the stage order in `bootstrap.yaml`, not only the thresholds. Figures are under `rulebots-1` in the findings file.

### 4.5 Modes and maps

- **2v2 has no teams.** `Tile.team` is parsed and never read, the only 2v2 map gives players 3 and 4 nothing, and friendly fire applies (`core-4`, `persist-18`). Hide 2v2 until teams exist.
- **1v1v1 end rules are inconsistent:** any HQ capture ends the whole game, while eliminated players keep taking turns (`core-7`).
- **No map is validated against `num_players`.** A spare HQ on a 3-player map becomes a free win in the 2-player env (`critic-integration-4`).
- **7 of the 20 `maps/1v1` maps have no production buildings,** so a GUI game on them never ends. Move them to `maps/scenarios/` (`pygame-22`, `persist-18`).
- **Map editor:**
  - Esc discards unsaved edits (`menus-6`);
  - you can't pan horizontally (`menus-7`);
  - validation lets unplayable maps through (`menus-14`);
  - saving crops and overwrites shipped maps (`persist-23`).

### 4.6 Settings, i18n and fonts

- **A corrupt `settings.json` silently wipes every setting,** API keys included, and writes aren't atomic (`menus-8`, S).
- **Korean and Chinese render as tofu on typical Linux,** because DejaVu is tried before Noto CJK (`pygame-19`, S).
- **Translations have drifted from the UI:** 22–34 referenced keys exist in no language, es/zh lack the whole map-editor section, and about 50 English literals bypass translation (`menus-13` `consolidate-21`, M).

### 4.7 LLM bots

- **Error handling** (`aibots-4`, M): classify errors, fail fast on non-retryable ones, and record errored games separately from losses.
- **History and context** (`aibots-6`, S): stateful history is unbounded and overflows the context window around turns 20–30. Positional unit IDs go stale.
- **Prompts contradict the engine constants** (`aibots-8`, M): buff percentages, cooldowns, HQ income and Cleric/Archer ranges. Generate the prompts from constants (§6.4).
- **Sorcerer abilities are advertised but unreachable** (`aibots-9`, S).
- **Truncated or refused responses lose the whole turn** (`aibots-15`, S).
- **Token use** (`aibots-7`, M): no prompt caching; every legal move is sent as indented JSON.

### 4.8 Persistence and tournaments

| Bug | Effort | IDs |
|---|---|---|
| Tournament resume replays everything outside multi-map `all` mode, and the resumed results drop the games already played | M | `persist-1` |
| Setup failures abort the tournament (sequential) or vanish (concurrent); in-game errors count as draws in Elo | S | `persist-2` |
| Replays re-run the economy under default constants: `engine_overrides`, `max_turns` and `enabled_units` are never recorded | M | `persist-3` |
| Replays of games loaded from a save have no starting state | M | `persist-4` |
| `cycle` mode (the CLI default) puts a matchup's two mirror games on different maps, tying side to map | S | `persist-5` |
| Elo depends on game order and varies between concurrent runs | M | `persist-7` |
| Saves drop `max_turns`/`end_reason` and have no version field; writes are non-atomic; save names aren't sanitised (path traversal) | S | `core-6` `persist-9` `persist-10` `persist-11` |
| Both committed v1 tournament replays silently drop actions (60/262 and 100/528), and no divergence check exists | M | `persist-13` |

---

## 5. Animations and presentation

The asset side is solid. Team palettes are swapped once at load time, so recolouring costs nothing per frame and memory stays bounded. The fallback chain (animated → static → letter) is robust, and the replay viewer and video export share one action dispatcher.

**The problem is that almost none of it reaches the player.** The walking and path animations are fully built and never called (`pygame-17`, `anim-7`, `consolidate-17`). Bot turns never render, and the interactive game doesn't use the bundled pixel art by default (`pygame-16`).

### 5.1 Bugs

| Bug | Where | Effort | IDs |
|---|---|---|---|
| Exporting a video from the replay viewer leaves `SDL_VIDEODRIVER=dummy` set, so the next main menu opens as an invisible window | `utils/video.py:27` | S | `anim-1` |
| The team palette swap misses whole colour ramps: red, green and yellow Rogues look blue, and Sorcerer hats and Archer trousers stay blue | `constants.py:80` | S | `anim-3` |
| The bundled pixel art is off by default; new players see coloured rectangles | `ui/renderer.py:204`; `utils/settings.py:23` | S | `pygame-16` |
| The animation clock is wall-clock time, so video exports are non-deterministic. Pass `dt=1/fps` from the exporter | `ui/renderer.py:380` | S | `anim-4` |
| The OpenCV fallback never runs when ffmpeg fails; the game-over overlay's `convert()` crashes headless | `utils/video.py:596`, `:515` | S | `anim-5` `anim-6` |
| Human-game replays are padded twice; small-map exports are mostly ocean | `app/game_loop.py:253` | S | `anim-8` |
| Replay seeking re-simulates from action 0 on every step back and every scrub event | `utils/replay_player.py:329` | M | `anim-10` |
| Animator state is keyed by `id(unit)` and never cleaned up; `get_frame` advances at most one frame per call | `ui/sprite_animator.py:287`, `:315` | S | `anim-12` `anim-13` |
| Replay speeds of 8× and 10× are nearly identical (overshoot dropped, one action per frame); replay panels cover the last map row | `utils/replay_player.py:256`, `:736` | S | `anim-15` `anim-9` |
| The 32×32 centre crop cuts every sprite's drop shadow and some weapons | `ui/sprite_animator.py:180` | S | `anim-20` |

### 5.2 Development roadmap: ranked by value for effort

The sprite sheets have no attack, hurt or death frames, so everything below is procedural and needs no new art.

| # | Feature | How it plugs in | Effort | IDs |
|---|---|---|---|---|
| 1 | **Paced, visible bot turns.** The foundation for everything else | Add an action listener called right after `record_action` (`core/game_state.py:649`). The session renders, flips, pumps events and waits ~150–250 ms per bot action. LLM bots plan on a worker thread and show a "thinking…" overlay with cancel; actions are applied on the main thread | M | `anim-2` `pygame-6` `aibots-5` |
| 2 | **Movement tweening along the real path.** Wires up the existing, unused walk animation | Add a `find_path` helper with a parent map (the BFS records no parents today), compute the path before `move_unit` mutates x/y, route drawing through a single `Renderer._unit_px(unit)`, and tween at 0.12–0.18 s per tile | M | `pygame-17` `anim-7` `anim-23` |
| 3 | **Effects layer** (`ui/effects.py`) | `on_action(action)`, `update(dt)` and `draw()`, called after `_draw_units`. Attack records already carry damage, evade, flank, kill and HP-after data. In order: floating damage/heal numbers with EVADE/FLANK tags; hit flash; death fade from a last-drawn snapshot; turn banner; capture flash | S each | `anim-22` |
| 4 | **Procedural attack lunge** | Offset the attacker 6–8 px toward the target for about 100 ms, then trigger the flash and number | S | `anim-23` |
| 5 | **Idle polish** | De-synchronise idle phases by `unit_id`, face units by their last horizontal move, show a focus reticle on each bot action | S | `anim-24` |
| 6 | **Damage-forecast tooltip; show buffs, cooldowns and structure-heal costs** | Reads the engine preview API (§6.2), the same one the bots and the LLM serializer should use | M | `critic-integration-12` `critic-integration-8` |
| 7 | **Evaluation videos that show the opponent's turn** | Render from the saved replay (`record_replay_to_video` already emits a frame per action) instead of one frame per agent step | S | `anim-25` |
| 8 | **Turn-flow quality of life** | Spectate mode, next-idle-unit hotkey, end-turn warning, hotseat hand-off screen | M | `pygame-25` |
| 9 | **Terrain pre-render cache** | One pre-rendered terrain surface, re-blit only dirty structures. Frame time is measured at 3.7 ms (24×24) and 21 ms (54×54) today; the estimate after the change is about 1 ms | S–M | `pygame-15` |
| 10 | **Camera and viewport** | Pan and zoom, draw only visible tiles, window sized to the desktop. Needed once editor maps exceed about 26 tiles | L | `pygame-13` |

---

## 6. Consolidation and cleanup

The July consolidation pass removed real duplication: one observation builder, one BFS, one registry, one mask layout. What remains is mostly *parallel implementations of engine rules outside the engine*, and several of those have already drifted into live bugs. Ordered by payoff against risk:

| # | Consolidation | What it replaces | Fixes bugs? | Effort |
|---|---|---|---|---|
| 1 | **Engine-owned action layer** `core/actions.py`: an `ActionType` IntEnum, `is_legal(state, action)` and `apply_action(state, action) -> ActionResult`. The env, ModelBot, MCTS, RandomBot, LLM bots, GUI and replay all call it | 5–8 hand-written dispatch tables, including ModelBot's 220-line duplicate executor and its dead 95-line mask builder (`core-14` `consolidate-9` `aibots-13` `prior-14` `pygame-18` `rulebots-20`) | §1.2, `critic-integration-9`, `critic-integration-2` | M–L |
| 2 | **`GameMechanics.preview_attack()`**: deterministic expected damage, counter and evade. `attack_unit` uses the same core, so the two cannot drift | Bot damage estimates; LLM, GUI and diagnostics forecasts (`rulebots-3` `consolidate-3` `critic-integration-12`) | 24–42% failed "sure kills" | M |
| 3 | **Bot registry as the single source of truth**, with metadata (display name, GUI-selectable, ladder, kwargs) | 8+ hand-maintained bot lists; the GUI can't select MasterBot; the Docker runner skips unknown types; the env accepts typos (`rulebots-10` `menus-25` `persist-15` `consolidate-18` `tests-13` `aibots-17`) | `rlenv-10` class | S–M |
| 4 | **Rules text generated from constants** | Six prompt blocks, the docs-site mechanics page and the README, all hand-copied. A generator already exists but only tests use it (`aibots-8` `critic-integration-7` `core-23`) | Prompt/engine contradictions | M |
| 5 | **One map loader** returning `LoadedMap(df, offsets, original)`; `set_map_metadata` actually gets called | Padding in FileIO, ReplayPlayer and video export (`consolidate-11` `anim-18`). Padding metadata is dead in production today (`core-19` `aibots-18` `persist-22`) | `anim-8`, `aibots-2`, LLM coordinate conversion | M |
| 6 | **Replay/save schema v4**: one `build_replay_game_info()` records overrides, `max_turns`, enabled units and fog; saves get a version field and atomic writes | Two different `game_info` layouts under the same schema version 3 (`persist-14`) | `persist-3` `persist-9` `persist-10` | M |
| 7 | **`UnitType`/`TileType` enums and `TILE_TYPE_TO_IDX`** | Canonical lists retyped 13+ times, kept in sync by comments (`consolidate-12` `prior-22`) | — | M |
| 8 | **`units_in_range()` and `tick_statuses()`**; ability ranges moved into constants | 9 range helpers, 5 status-tick helpers, 6 ability wrappers (`core-15` `prior-15`) | — | S–M |
| 9 | **Bot tiers:** template hooks plus a `_continue_if_hasted()` helper | ~194 identical lines, 29 haste re-entry checks, 13 enemy comprehensions (`rulebots-13`) | `rulebots-8` | M |
| 10 | **UI:** `run_loop()` in `ScreenBootstrapMixin`, an `OverlayPopupMenu`, one icon generator | 4 non-Menu event loops, popup duplication, 11 copy-pasted icon generators, map-preview code in 3 menus plus a script (`menus-19` `critic-gaps-11` `consolidate-25`) | `menus-10` | S–M |
| 11 | **Delete about 800 LOC of confirmed dead code;** add vulture to CI with a whitelist | `consolidate-15` `prior-20` and the per-subsystem dead-code items | — | S |
| 12 | **Replace ~160 `print()` calls in library code with logging;** the CLI configures logging | `consolidate-20` `prior-18` | Hidden failures | M |
| 13 | **Config inheritance** (`extends:`) plus seed replication; archive the 59 sweep YAMLs (49k lines) | `rltrain-15` `consolidate-14` | Enables §2.1 | M |
| 14 | **Break up the `GameState` god object:** `EngineConfig`, serializer, legal-action generator, visibility facade. Do this after #1 | `core-16` | Makes `core-6`-style omissions impossible | L |
| 15 | **Docs:** merge the seven review and planning docs into one status tracker and archive the superseded ones; fix the wrong statuses; fix README drift (the Lint badge points at a nonexistent `lint.yml`, unit stats are outdated) | `prior-26` `prior-7` `prior-23` `prior-24` `prior-25` `consolidate-22` | — | S |

---

## 7. Tests, CI, tooling and security

**Tests that would have caught this review's headline bugs**

- A cache-coherence debug assertion plus a random-play invariant test (`core-17`) would have caught §1.1.
- A legality harness running every bot tier on every map (`rulebots-14`) would have caught §1.2 and `rulebots-1`/`rulebots-2`.
- Headless UI smoke tests. Rendering every shipped map headless takes under 1 s, and each GUI finding above reproduces headless. Remove `ui/*`, `app/*` and `cli/*` from the coverage `omit` (`tests-6`, `pygame-24`, `menus-22`).
- Self-play regression tests (`tests-3`), trainer smoke tests (`rlalt-19`, `tests-9`), and entry-point `--help`/short-run tests (`tests-1`).

**Test hygiene**

- The Settings and Language singletons write to `./settings.json`, so a shuffled test order fails 3 tests every time (`tests-4`).
- 8 tests pass without executing a single assertion (`tests-5`).
- Only 5 of the 67 training YAMLs are load-tested (`tests-23`).
- One test takes 21% of suite wall time (`tests-18`).

**CI**

- An import scan or `deptry` for undeclared dependencies (§1.6).
- A wheel-install test. The built wheel ships no assets, maps or fonts (`tests-14`, `consolidate-13`).
- Job and test timeouts (`tests-10`).
- A constraints file or lock (`tests-11`).
- Pre-commit in CI, and fix the Lint badge (`tests-17`).
- Build the docs site on PRs, since `onBrokenLinks` is `throw` (`tests-16`).
- Remove mypy `ignore_errors` module by module, starting with `self_play.py`. One of the hidden errors sits on the exact line of the §1.3 wrapper bug (`tests-8`, `consolidate-24`).

**Security**

| Issue | IDs |
|---|---|
| `.env` baked into Docker images | §1.7 |
| API keys stored in plaintext `settings.json` with default 0644 permissions, shown unmasked | `menus-18` `prior-19` |
| The W&B API key is printed and stored in plaintext in the Vertex job spec | `critic-gaps-5` |
| Loading an SB3 `.zip` unpickles arbitrary code, and tournament discovery loads every file in a directory. Document "trusted checkpoints only", and restrict discovery | `critic-gaps-7` |
| Save names allow path traversal | `persist-11` |
| `scripts/stop_training.sh` *deletes*, not stops, every VM whose name contains `rl-trainer` | `critic-gaps-4` |

---

## 8. Suggested sequencing

| Week | Work | Exit criterion |
|---|---|---|
| 1 | **P0** (§1.1–1.9) with a regression test for each; the cache-coherence debug flag; the bot legality harness | CI green with `RT_CHECK_CACHE=1`; the harness shows 0 illegal actions |
| 2 | **RL lifecycle:** stochastic gate, stall retry and resume, `bootstrap.yaml` backport, LR schedule, opponent validation (§2.2–2.3). Feudal S-fixes (§3). **Start the 3-seed validation run** | Three seeds finish or stall with a resumable run record |
| 3 | **GUI correctness batch** (§4.1–4.3); **visible bot turns and movement tweening** (§5.2 items 1–2); pixel art on by default | Human vs SimpleBot/LLM plays without freezes; bot moves visible |
| 4 | **Action layer and preview API** (§6.1–6.2); rule-bot fixes (§4.4); re-baseline curriculum thresholds against the stronger SimpleBot | ModelBot, env, MCTS and GUI share one executor |
| Then | Effects layer (§5.2 #3–7); Player-2 seat and fog-of-war obs fixes; persistence/tournament batch (§4.8); docs consolidation; the AlphaZero/BC decision; the structured policy (§2.5) | — |

---

## 9. What's in good shape

- **The engine core.** `game_state` 95% and `mechanics` 97% coverage. A 120-game random-legal-action fuzz across all 20 1v1 maps (fog of war included) held every invariant. There is a single game-over chokepoint. Replay schema v3 records outcomes rather than re-rolling. Engine overrides resolve into a per-game deep copy and fail loudly on unknown fields.
- **Determinism.** `reset(seed)` is fully deterministic for all scripted opponents. Bot action-history hashes were identical across three processes with different `PYTHONHASHSEED` values.
- **Env contract.** `check_env` passes for the multi_discrete, flat_discrete, fog-of-war and padded variants. `flat_discrete` masks were exact over about 1,500 random-legal steps. Potential-based shaping is now correct at terminal and truncated boundaries.
- **ModelBot parity.** For the same state, ModelBot's observation, masks and flat decode table are identical to the training env's in both action modes. The loader rejects mismatched checkpoints with actionable messages.
- **Bootstrap observability.** Incremental `eval_results.jsonl`, `train_metrics.csv`, `run_status.json` with best-checkpoint provenance, and `resolved_config.yaml`. The config loader rejects unknown keys at every level, and all 64 non-BC YAMLs still load.
- **UI foundations.** One geometry source for hit-testing and drawing. Overlay surfaces are pre-allocated, text is cached, and `theme.py` centralises every colour and timing constant. The widget library (Dialog, TextInput, ellipsis) is solid.
- **Tooling.** A well-engineered CI workflow (CPU-only torch, a push/PR dedupe guard, least-privilege permissions), exec-form Vertex entrypoint with SIGTERM forwarding, and secrets redacted from exported tournament results.

---

## Appendix A — Coverage of this review

| Reviewer | Files read | Crit | High | Med | Low | Info | Total |
|---|---|---|---|---|---|---|---|
| Core game engine | 33 | 1 | 3 | 9 | 11 | 1 | 25 |
| Rule-based bots | 18 | 0 | 3 | 11 | 8 | 3 | 25 |
| LLM, model and AlphaZero bots | 23 | 0 | 4 | 9 | 10 | 1 | 24 |
| RL environment (Gymnasium) | 17 | 1 | 2 | 7 | 11 | 1 | 22 |
| PPO bootstrap / self-play pipeline | 28 | 0 | 3 | 8 | 11 | 1 | 23 |
| Feudal RL, AlphaZero/MCTS, imitation | 28 | 0 | 6 | 14 | 5 | 1 | 26 |
| Pygame shell and renderer | 37 | 2 | 5 | 11 | 6 | 1 | 25 |
| Animations, replay, video export | 20 | 0 | 2 | 3 | 16 | 4 | 25 |
| Menus, map editor, i18n, settings | 47 | 2 | 2 | 12 | 10 | 1 | 27 |
| Save/load, replays, tournament, CLI, cloud | 44 | 0 | 2 | 16 | 7 | 1 | 26 |
| Consolidation, dead code, repo hygiene | 87 | 0 | 2 | 12 | 11 | 0 | 25 |
| Tests, CI and tooling | 62 | 0 | 3 | 6 | 15 | 1 | 25 |
| Prior-review status audit | 42 | 2 | 0 | 11 | 13 | 2 | 28 |
| Critic: coverage gaps | 52 | 0 | 1 | 2 | 8 | 2 | 13 |
| Critic: cross-subsystem integration | 37 | 0 | 1 | 4 | 6 | 1 | 12 |

The coverage-gap critic confirmed that every Python file had at least one reader. The only exceptions were seven re-export-only `__init__.py` files, which it read and found clean. It also reviewed the cloud shell scripts.

## Appendix B — Status of earlier reviews

| Document | Status at HEAD |
|---|---|
| `REVIEW_rl_pipeline_2026-07-24.md` | The five §4 items marked "Landed" are in the code: terminal −Φ charge, truncation semantics, fixed eval seed set, baseline-only stage-entry eval (`best_eligible_after`), sanity eval through `make_stage_env`. One exception: item 3 landed only half. The seed set is fixed, but the gate still evaluates the deterministic policy. Items 6–11 and these sections are still open: §2.2(a) (deterministic gate), §2.7 (LR schedule), §2.9 (stall and resume), §2.12 (BC), §2.13 (self-play), §2.16 (alternative algorithms), the §3 `bootstrap.yaml` back-port and preemption |
| `REVIEW_ppo_training.md` | Still open: §2.1 / recommendation #1 (canonical config), §2.6 / #6 (truncation still drops attacks; the seize and end-turn protection landed), §2.7 (phantom counter), and recommendations #2, #3 and #5 |
| `REVIEW_maintainability.md` | At least 10 of its 32 items are fixed, but it has no status markers. Still open: #4/#5 (dispatch duplication), #7 (range helpers), #11–13 (enums and encodings), #14/#16 (paths and padding), #15 (print vs logging), #22 (settings merge), #29 (CSV escaping) |
| `REVIEW_advancedbot.md` | The resolution table is wrong in both directions. The swap_players fix (§16) did not land; the cache item was marked fixed but `end_turn` invalidation was never added |
| `feudal_rl_review.md` | Every "Still open" item is still open |
| `ROADMAP.md` | Statuses are stale in both directions. It claims the self-play swap fix landed, and it doesn't cover the bootstrap pipeline or its blockers |

Recommendation (`prior-26`): keep this document and the ROADMAP as the live trackers, add a status line to the top of each older review pointing here, and move them to `docs/archive/`.
