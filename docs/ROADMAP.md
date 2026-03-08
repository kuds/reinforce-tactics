# Reinforce Tactics — Development Roadmap

A prioritized, phased plan for growing Reinforce Tactics from a solid alpha into a
complete RL education platform. Each phase builds on the previous one.

> **Last updated:** February 2026
> Status legend: ✅ Complete | 🟡 Partial | ⬚ Not Started

---

## Progress Summary

| Phase | Description | Progress |
|-------|-------------|----------|
| **Phase 1** | Fix Foundations & Quick Wins | **75%** — 3 of 4 items complete |
| **Phase 2** | Educational Scaffolding | **0%** — 0 of 4 items complete |
| **Phase 3** | RL Depth & Model Zoo | **25%** — 1 of 4 items complete, 1 partial |
| **Phase 4** | LLM & Benchmark Polish | **12%** — 0 of 4 items complete, 2 partial |
| **Phase 5** | Platform Expansion | **10%** — 0 of 5 items complete, 2 partial |

---

## Completed Work (Outside Original Roadmap)

Significant features shipped since the roadmap was created that were not originally
tracked here:

### Sprite & Animation System ✅
- Coordinate-based sprite animation with movement path transitions
- Per-team palette swap for unit sprites and team-coloured structure tiles
- Tile variant auto-discovery for visual terrain variety
- Sprite sheet splitting utility and unified `sprites_path` with auto-discovery

### AlphaZero Implementation ✅
- Full AlphaZero with MCTS (`alphazero_net.py`, `alphazero_trainer.py`, `mcts.py`)
- Dual-head CNN: residual blocks with policy head and value head
- MCTS with PUCT selection, Dirichlet noise, neural evaluation, action masking
- Self-play data generation, replay buffer, LR scheduling, checkpointing
- Game-compatible `AlphaZeroBot` for tournament play
- 22 comprehensive tests in `test_alphazero.py`
- Training script: `train/train_alphazero.py`

### Feudal RL (Hierarchical) Implementation ✅
- Manager-Worker PPO with variable-step GAE and intrinsic rewards
- `FeudalRolloutBuffer` with separate manager/worker storage
- Full training loop with TensorBoard logging (`train/train_feudal_rl.py`)
- CLI arguments: `--manager-horizon`, `--worker-reward-alpha`, `--manager-lr-scale`

### AdvancedBot & RL Bug Fixes ✅
- `max_turns` enforcement in `game_state.py` (games no longer run indefinitely)
- BFS pathfinding fix: `moving_unit` passed to `get_legal_actions()`
- Cache interleaving fix: separate flags for `unit_count` and `legal_actions`
- Bot recursion guard (`MAX_RECURSION_DEPTH=10`) preventing haste stack overflow
- Model bot masking: real masks from legal actions instead of `np.ones`
- Self-play weight swap: `try/finally` guards for safe weight restoration
- Improved exception handling with structured logging

### Critical Bug Fix Round 2 ✅
- Dead attacker state mutation: guard `can_move`/`can_attack` with `attacker_alive` check
- ClaudeBot `max_tokens` crash: Anthropic API requires `max_tokens`; defaults to 4096 when None
- Bare exception catch in `_execute_action`: `TypeError`/`AttributeError` now re-raised instead of silenced
- Mountain vision bonus: `calculate_vision_radius()` now wired into `PlayerVisibility.update()` (mountain +1 range works)
- Self-play `swap_players` fix: `agent_player` attribute added to `StrategyGameEnv`; `_execute_action`, `_get_obs`, `_compute_potential`, and `step()` use it instead of hardcoded player 1; `SelfPlayEnv` propagates `agent_player` to base env on reset

### Save/Load/Replay Improvements ✅
- Save-before-quit prompt when exiting an active game
- Replay auto-save after game completion
- Load menu status display with timestamps and game info

### Tournament Improvements ✅
- `enabled_units` config to restrict unit types per tournament
- `ClaudeBot` JSON serialization support (`to_dict()` / `from_dict()`)
- Token tracking in `LLMBot` (`total_input_tokens`, `total_output_tokens`, `get_token_usage()`)

### Package & Infrastructure ✅
- `pyproject.toml` for pip install support (version 0.2.0)
- Dependency split: headless base, `[gui]`, `[llm]`, `[dev]`, `[all]` extras
- Fog of War: HQ always visible, buildings/towers hidden until scouted
- 2v2 FoW perspective support
- Documentation site sync with current codebase

---

## Phase 1: Fix Foundations & Quick Wins (1–2 weeks)

These are high-impact, low-effort items that unblock everything else.

### 1.1 Fix `SelfPlayCallback` SB3 Compatibility ✅
**Priority:** Critical
- `SelfPlayCallback` now inherits from `BaseCallback` via dynamic class creation
  in `_make_callback_class()`.
- Implements `_on_step()` and `_init_callback()` per the SB3 API.
- Tested with `MaskablePPO` callback integration in `test_self_play.py`.

### 1.2 Add Baseline Training Benchmarks ⬚
**Priority:** High — users currently have no way to know if their training is working.
- Run PPO (with action masking) against `SimpleBot` on `maps/1v1/beginner.csv` for
  10K, 50K, 200K, and 1M timesteps. Record win rate, avg reward, avg episode length.
- Save results as a markdown table and a TensorBoard log.
- Commit as `docs-site/docs/training-benchmarks.md` and `benchmarks/` directory with
  the training scripts and logs.
- Goal: a user can run the same script and compare their curve to the reference.

### 1.3 Unify Action Mask Logic ✅
**Priority:** Medium
- `_build_masks()` method in `gym_env.py` computes `get_legal_actions()` once and
  derives both flat and per-dimension masks from a shared intermediate structure.
- Both `_get_action_mask()` (flat) and `action_masks()` (tuple) use the unified method.

### 1.4 Add RL-Specific Tests ✅
**Priority:** Medium
- 73+ tests in `test_gym_env.py` covering:
  - Observation shapes match `observation_space` after `reset()` and `step()`
  - `action_masks()` never returns all-zero for any dimension
  - Reward stays within expected bounds
  - Self-play environment properly alternates players

**Milestone:** ~~At the end of Phase 1, the RL training pipeline is correct, tested, and
has documented expected results.~~ **Status: 3 of 4 items complete.** The RL pipeline is
correct and tested. Baseline benchmarks remain the final gap — once published, users will
have a reference point to validate their own training runs.

---

## Phase 2: Educational Scaffolding (2–4 weeks)

This is the highest-leverage work. The code exists; it just needs to be made accessible.

### 2.1 Beginner RL Environment ⬚
**Priority:** High — single most impactful change for newcomers.
- Create `StrategyGameEnvSimple` (or a config preset): 6x6 map (the existing
  `maps/1v1/beginner.csv`), 2 unit types (Warrior + Archer), no special abilities,
  smaller action space.
- PPO should converge to >70% win rate against `SimpleBot` in <50K steps on this env.
- Ship with a script: `examples/train_beginner.py` that runs in ~5 minutes on CPU.
- Document expected output in the docs site.

### 2.2 Core RL Documentation Pages ⬚
**Priority:** High — the docs site has almost no RL content.
Add these pages to `docs-site/docs/`:
- **"Training Your First Agent"** — step-by-step PPO tutorial, what to expect, how to
  read TensorBoard output.
- **"Understanding the Environment"** — observation space walkthrough with diagrams,
  action space breakdown, reward config explanation.
- **"Action Masking Explained"** — why it matters, how it works in this game, the
  over-approximation tradeoff.
- **"Reward Engineering Guide"** — sparse vs. dense, potential-based shaping (cite
  Ng et al. 1999), risks of direct action bonuses, how to tune `reward_config`.
- **"Self-Play Training Guide"** — opponent pool, selection strategies, detecting
  policy cycling, recommended hyperparameters.

### 2.3 Learning Path / Curriculum Page ⬚
**Priority:** High — gives the project a clear narrative arc.
- Add a "Learning Path" page to the docs site with a progressive curriculum:
  1. Play the game manually to understand mechanics
  2. Train PPO on beginner env
  3. Understand and tune reward shaping
  4. Add action masking
  5. Self-play training
  6. AlphaZero deep-dive
  7. Feudal RL (hierarchical)
  8. LLM bots and RL-vs-LLM comparison
- Each step links to the relevant doc page, example script, and expected results.

### 2.4 Expand Example Scripts ⬚
**Priority:** Medium
- `examples/train_self_play.py` — end-to-end self-play with opponent pool
- `examples/train_alphazero.py` — AlphaZero with MCTS visualization
- `examples/evaluate_and_compare.py` — pit trained model against bots, print stats
- `examples/reward_shaping_experiment.py` — train with different reward configs, plot comparison

> **Note:** Training scripts exist in `train/` (train_self_play.py, train_alphazero.py,
> train_feudal_rl.py) but these are full training pipelines, not simplified educational
> examples. The `examples/` directory still needs beginner-friendly versions.

**Milestone:** At the end of Phase 2, a user with zero RL experience can follow the
learning path from "What is RL?" to "I trained an AlphaZero agent."

---

## Phase 3: RL Depth & Model Zoo (3–6 weeks)

Deepen the RL capabilities and ship pre-trained artifacts.

### 3.1 Pre-trained Model Zoo ⬚
**Priority:** High — immediate hands-on experience without waiting for training.
- Ship models at: random baseline, 50K steps, 200K steps, 1M steps, self-play champion.
- Store as downloadable `.zip` files (or Git LFS / release artifacts).
- Add `examples/play_against_model.py` that loads a model and renders a game.
- Users can immediately see what different training budgets produce.

### 3.2 Auto-Regressive Action Decomposition ⬚
**Priority:** **High — prerequisite before training on 10x14+ and 20x20 maps.**

The current MultiDiscrete space (10 × 8 × W × H × W × H) with independent per-dimension
masking suffers from combinatorial explosion on larger maps:

| Map size | Combinations | Per-dim mask over-approx |
|----------|-------------|--------------------------|
| 6×6      | 288K        | Manageable               |
| 10×14    | 1.6M        | Significant              |
| 20×20    | 12.8M       | Severe                   |

Auto-regressive decomposition keeps each step small (≤20 choices) regardless of map size
and enables *exact* conditional masking that eliminates all invalid action combinations.

**Decomposition order:**
```
action_type(10) → from_x(W) → from_y(H) → unit_type(8) → to_x(W) → to_y(H)
```

Each dimension is sampled conditioned on all prior choices:
`P(at) → P(fx|at) → P(fy|at,fx) → P(ut|at,fx,fy) → P(tx|...) → P(ty|...)`

**Implementation steps (6 sub-tasks):**

1. **Conditional mask builder** (`gym_env.py`) — precompute a nested lookup from
   `get_legal_actions()` so that, given choices so far, the exact valid mask for the
   next dimension can be retrieved in O(1). Same single `get_legal_actions()` call per
   step as the current `_build_masks()`. (~80 lines)

2. **Auto-regressive policy network** (`reinforcetactics/rl/autoregressive.py`) — new
   module. Shared CNN+MLP backbone produces hidden state; 6 sequential MLP heads each
   take the hidden state plus learned embeddings of all previously sampled dimensions.
   Conditional masks applied as logit masking before sampling. Includes `sample_action()`
   and `evaluate_action()` for PPO. (~250 lines)

3. **Feudal RL integration** (`feudal_rl.py`) — add `autoregressive=True` flag to
   `FeudalRLAgent`. When enabled, `AutoRegressiveWorker` replaces the independent-head
   `WorkerNetwork`. Per-dimension masks stored in the rollout buffer alongside actions
   (small cost: 6 bool arrays per step vs. 20×20×3 float32 observation tensors). (~150
   lines modified)

4. **Standalone PPO mode** — add `action_space_type='autoregressive'` to
   `StrategyGameEnv`, create SB3-compatible policy wrapper so auto-regressive
   decomposition works without the feudal hierarchy. Update `masking.py`. (~150 lines)

5. **Training CLI** — add `--autoregressive`, `--ar-embedding-dim`, `--ar-head-hidden`
   flags to `train/train_feudal_rl.py`. Add `--mode autoregressive` for standalone
   non-feudal training. (~50 lines)

6. **Tests** (`tests/test_autoregressive.py`) — conditional mask exactness, network
   output shapes, gradient flow, round-trip validity (sampled action ∈ legal actions),
   evaluate/sample log-prob consistency, 20×20 smoke test. (~200 lines)

**Scaling note:** Sequential sampling (6 small MLP forwards) is negligible overhead
for a turn-based game. The CNN backbone dominates inference time regardless.

**Prerequisite for:** Training on maps larger than 6×6 at competitive sample efficiency.
Should be completed before large-map curriculum stages in Phase 5.1.

### 3.3 AlphaZero & Feudal RL Documentation 🟡
**Priority:** Medium — these are impressive but invisible.
- **"AlphaZero for Reinforce Tactics"** doc page: architecture diagram, MCTS explained
  in context of this game, training loop walkthrough, results.
- **"Hierarchical RL with Feudal Networks"** doc page: Manager-Worker hierarchy,
  when/why it helps, spatial feature extraction, training guide.
- Add Jupyter notebooks that walk through each step interactively.

> **Note:** Both AlphaZero and Feudal RL are **fully implemented and tested** (see
> "Completed Work" above). What remains is writing the documentation and interactive
> notebooks to make these accessible.

### 3.4 Training Debug Overlay ⬚
**Priority:** Medium — visual learning aid.
- When evaluating a trained model with `render_mode='human'`, show an optional panel:
  - Top 5 actions with probabilities
  - Current accumulated reward
  - Valid action count
  - Value function estimate (if available)
- Toggle with a keyboard shortcut during evaluation.

**Milestone:** At the end of Phase 3, the project has a model zoo, auto-regressive action
decomposition enabling large-map training (10×14, 20×20), and documentation for advanced
topics.

---

## Phase 4: LLM & Benchmark Polish (4–8 weeks)

Formalize the RL-vs-LLM comparison that makes this project unique.

### 4.1 RL vs. LLM Benchmark Suite ⬚
**Priority:** Medium-High — unique value proposition.
- Define a standardized benchmark: fixed maps, fixed starting conditions, N games per
  matchup.
- Metrics: win rate, avg game length, gold efficiency (gold spent per kill),
  unit survival rate, structure control over time.
- Automate: `python -m reinforcetactics.benchmark run` produces a JSON results file.
- Publish results on docs site as "Tactical Reasoning Leaderboard."

### 4.2 Local LLM Support (Ollama) ⬚
**Priority:** Medium — removes cost barrier for experimentation.
- Add `OllamaBot` subclass of `LLMBot` that talks to a local Ollama server.
- Support models like Llama 3, Mistral, etc.
- Great for students who can't afford API costs.

### 4.3 LLM Cost Tracking 🟡
**Priority:** Low-Medium
- Track tokens (input + output) and estimated cost per game.
- Log to the conversation JSON files that already exist.
- Add a summary command: `python main.py --mode stats --llm-costs`

> **Note:** Token tracking is partially implemented — `LLMBot` already tracks
> `total_input_tokens` and `total_output_tokens` with a `get_token_usage()` method.
> What remains is dollar-cost estimation per model and the CLI summary command.

### 4.4 Tournament CI Automation 🟡
**Priority:** Low-Medium
- GitHub Actions workflow that runs a tournament on each release.
- Publishes updated ELO ratings and RL-vs-LLM results to the docs site.

> **Note:** CI infrastructure exists (deploy-docusaurus.yml, python-package.yml,
> pylint.yml) but no tournament-specific automation is in place yet.

**Milestone:** At the end of Phase 4, "How does GPT-5 compare to a 1M-step PPO agent
at tactical reasoning?" has a quantified, reproducible answer.

---

## Phase 5: Platform Expansion (8+ weeks, longer-term)

Bigger efforts that grow the project's reach and research utility.

### 5.1 Curriculum Learning System 🟡
**Priority:** Medium — makes the project course-ready.
- Formalize the progressive environments from Phase 2 into a `CurriculumEnv` that
  automatically advances difficulty when the agent reaches a win-rate threshold.
- Stages: 6×6 map → 10×14 map → 20×20 map; warriors only → all units; no fog → fog of war.
- Track and visualize progress across stages.

> **Note:** `make_curriculum_env()` in `masking.py` provides difficulty presets
> (easy/medium/hard), but auto-advancement based on win-rate thresholds is not
> implemented.
>
> **Dependency:** The 10×14 and 20×20 map stages require Phase 3.2 (Auto-Regressive
> Action Decomposition) to be completed first. Without it, the combinatorial action
> space makes training on large maps impractical.

### 5.2 Multi-Agent RL (3+ Players) 🟡
**Priority:** Medium — opens up MARL research.
- The 1v1v1 and 2v2 maps exist but `StrategyGameEnv` is hardcoded for 2 players.
- Extend to PettingZoo-compatible multi-agent env.
- Enables research on coalition formation, opponent modeling, etc.

> **Note:** PettingZoo is listed as a dependency but not yet used. The 2v2 FoW
> perspective is implemented, providing groundwork for multi-agent support.

### 5.3 Web-Based Interface ⬚
**Priority:** Lower — largest effort, broadest reach.
- Browser-based version using either Pyodide (Python in browser) or a React frontend
  with WebSocket to a Python backend.
- The headless mode already separates logic from rendering, so architecture supports this.
- Would let people try the game without any installation.

### 5.4 Sound & Music ⬚
**Priority:** Lower for RL use, higher for game polish.
- Listed as "High Priority" in implementation-status.md.
- Add sound effects for combat, unit creation, structure capture.
- Background music tracks.
- Matters for demos and the "game" experience.

> **Note:** Settings infrastructure exists (audio volume config in `settings.py`,
> language keys for `settings.sound`), but no actual audio files or `pygame.mixer`
> integration.

### 5.5 Campaign / Story Mode ⬚
**Priority:** Lower — nice to have.
- Scripted scenarios that teach game mechanics through guided play.
- Could double as an RL curriculum: each campaign mission is a training scenario.

---

## Visual Timeline

```
Week  1-2   ████ Phase 1: Fix Foundations        [██████████████░░░░░░] 75%
Week  3-6   ████████ Phase 2: Educational         [░░░░░░░░░░░░░░░░░░░░]  0%
Week  7-12  ████████████ Phase 3: RL Depth         [█████░░░░░░░░░░░░░░░] 25%
Week 13-20  ████████████████ Phase 4: LLM Polish   [██░░░░░░░░░░░░░░░░░░] 12%
Week 21+    ████████████████████ Phase 5: Platform  [██░░░░░░░░░░░░░░░░░░] 10%
```

Phases can overlap — Phase 2 documentation can start while Phase 1 fixes land.
Items within a phase can be parallelized across contributors.

---

## Recommended Next Steps

Based on current progress, the highest-impact work to tackle next:

1. **Phase 1.2 — Baseline Training Benchmarks**: The last remaining Phase 1 item.
   Publishing reference training curves unblocks users from validating their own runs.

2. **Phase 2.1 — Beginner RL Environment**: The single most impactful change for
   newcomers. A simplified 6×6 environment with fast convergence lowers the barrier
   to entry dramatically.

3. **Phase 2.2 — Core RL Documentation**: Five documentation pages that make the
   existing RL infrastructure accessible. The code is solid; the docs are the gap.

4. **Phase 3.2 — Auto-Regressive Action Decomposition**: The critical path item for
   scaling beyond 6×6 maps. Must be completed before training on 10×14 or 20×20 maps
   (Phase 5.1 curriculum stages). Current per-dimension masking over-approximation
   becomes severe at larger map sizes (12.8M combinations at 20×20).

5. **Phase 3.3 — AlphaZero & Feudal RL Docs**: Both algorithms are fully implemented
   and tested but have zero documentation. Writing these pages would surface work
   that's already done.

---

## How to Use This Roadmap

1. **Solo developer?** Work through Phase 1 → 2 → 3 in order. Phase 2 has the highest
   impact-per-hour.
2. **Small team?** One person on Phase 1 fixes, another starts Phase 2 docs in parallel.
3. **Looking for contributors?** The Phase 2 doc pages and Phase 3 example scripts are
   great first-contributor tasks — well-scoped, high-impact, and don't require deep
   familiarity with the codebase.
4. **Academic use?** Phase 2.3 (Learning Path) is your priority — it turns the project
   into a course module.
