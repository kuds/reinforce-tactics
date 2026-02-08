# Reinforce Tactics — Development Roadmap

A prioritized, phased plan for growing Reinforce Tactics from a solid alpha into a
complete RL education platform. Each phase builds on the previous one.

---

## Phase 1: Fix Foundations & Quick Wins (1–2 weeks)

These are high-impact, low-effort items that unblock everything else.

### 1.1 Fix `SelfPlayCallback` SB3 Compatibility
**Priority:** Critical — current code won't actually work with `model.learn(callback=...)`
- `SelfPlayCallback` in `rl/self_play.py` implements `_init_callback` and `_on_step`
  but does not inherit from `stable_baselines3.common.callbacks.BaseCallback`.
- Fix: subclass `BaseCallback`, rename `_init_callback` to `_on_training_start` (or
  keep `_init_callback` per SB3's API), and ensure `self.model` is set by the parent.
- Add a test that creates a MaskablePPO model and runs `model.learn(callback=SelfPlayCallback(...))`
  for 100 steps without error.

### 1.2 Add Baseline Training Benchmarks
**Priority:** High — users currently have no way to know if their training is working.
- Run PPO (with action masking) against `SimpleBot` on `maps/1v1/beginner.csv` for
  10K, 50K, 200K, and 1M timesteps. Record win rate, avg reward, avg episode length.
- Save results as a markdown table and a TensorBoard log.
- Commit as `docs-site/docs/training-benchmarks.md` and `benchmarks/` directory with
  the training scripts and logs.
- Goal: a user can run the same script and compare their curve to the reference.

### 1.3 Unify Action Mask Logic
**Priority:** Medium — reduces maintenance burden and potential for subtle bugs.
- `gym_env.py` has `_get_action_mask()` (flat) and `action_masks()` (per-dimension tuple)
  that iterate over the same `legal_actions` independently.
- Refactor: compute legal actions once, build a shared intermediate structure, derive
  both formats from it.

### 1.4 Add RL-Specific Tests
**Priority:** Medium — prevents regressions as you build on the RL layer.
- Verify observation shapes match `observation_space` after `reset()` and `step()`.
- Verify `action_masks()` never returns all-zero for any dimension.
- Verify reward doesn't diverge (stays within [-2000, 2000]) over a 500-step episode.
- Verify self-play env properly alternates players.

**Milestone:** At the end of Phase 1, the RL training pipeline is correct, tested, and
has documented expected results.

---

## Phase 2: Educational Scaffolding (2–4 weeks)

This is the highest-leverage work. The code exists; it just needs to be made accessible.

### 2.1 Beginner RL Environment
**Priority:** High — single most impactful change for newcomers.
- Create `StrategyGameEnvSimple` (or a config preset): 8x8 map, 2 unit types
  (Warrior + Archer), no special abilities, smaller action space.
- PPO should converge to >70% win rate against `SimpleBot` in <50K steps on this env.
- Ship with a script: `examples/train_beginner.py` that runs in ~5 minutes on CPU.
- Document expected output in the docs site.

### 2.2 Core RL Documentation Pages
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

### 2.3 Learning Path / Curriculum Page
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

### 2.4 Expand Example Scripts
**Priority:** Medium
- `examples/train_self_play.py` — end-to-end self-play with opponent pool
- `examples/train_alphazero.py` — AlphaZero with MCTS visualization
- `examples/evaluate_and_compare.py` — pit trained model against bots, print stats
- `examples/reward_shaping_experiment.py` — train with different reward configs, plot comparison

**Milestone:** At the end of Phase 2, a user with zero RL experience can follow the
learning path from "What is RL?" to "I trained an AlphaZero agent."

---

## Phase 3: RL Depth & Model Zoo (3–6 weeks)

Deepen the RL capabilities and ship pre-trained artifacts.

### 3.1 Pre-trained Model Zoo
**Priority:** High — immediate hands-on experience without waiting for training.
- Ship models at: random baseline, 50K steps, 200K steps, 1M steps, self-play champion.
- Store as downloadable `.zip` files (or Git LFS / release artifacts).
- Add `examples/play_against_model.py` that loads a model and renders a game.
- Users can immediately see what different training budgets produce.

### 3.2 Auto-Regressive Action Head
**Priority:** Medium-High — the single biggest RL architecture improvement.
- Replace flat MultiDiscrete with sequential: predict `action_type` → sample →
  predict `from_pos` conditioned on action_type → sample → predict `to_pos` → sample.
- Each sub-head gets its own mask (exact, not over-approximated).
- This is a significant effort (~1 week of focused work) but will dramatically
  improve training sample efficiency.
- Document the architecture and comparison against flat action space.

### 3.3 AlphaZero & Feudal RL Documentation
**Priority:** Medium — these are impressive but invisible.
- **"AlphaZero for Reinforce Tactics"** doc page: architecture diagram, MCTS explained
  in context of this game, training loop walkthrough, results.
- **"Hierarchical RL with Feudal Networks"** doc page: Manager-Worker hierarchy,
  when/why it helps, spatial feature extraction, training guide.
- Add Jupyter notebooks that walk through each step interactively.

### 3.4 Training Debug Overlay
**Priority:** Medium — visual learning aid.
- When evaluating a trained model with `render_mode='human'`, show an optional panel:
  - Top 5 actions with probabilities
  - Current accumulated reward
  - Valid action count
  - Value function estimate (if available)
- Toggle with a keyboard shortcut during evaluation.

**Milestone:** At the end of Phase 3, the project has a model zoo, improved RL architecture,
and documentation for advanced topics.

---

## Phase 4: LLM & Benchmark Polish (4–8 weeks)

Formalize the RL-vs-LLM comparison that makes this project unique.

### 4.1 RL vs. LLM Benchmark Suite
**Priority:** Medium-High — unique value proposition.
- Define a standardized benchmark: fixed maps, fixed starting conditions, N games per
  matchup.
- Metrics: win rate, avg game length, gold efficiency (gold spent per kill),
  unit survival rate, structure control over time.
- Automate: `python -m reinforcetactics.benchmark run` produces a JSON results file.
- Publish results on docs site as "Tactical Reasoning Leaderboard."

### 4.2 Local LLM Support (Ollama)
**Priority:** Medium — removes cost barrier for experimentation.
- Add `OllamaBot` subclass of `LLMBot` that talks to a local Ollama server.
- Support models like Llama 3, Mistral, etc.
- Great for students who can't afford API costs.

### 4.3 LLM Cost Tracking
**Priority:** Low-Medium
- Track tokens (input + output) and estimated cost per game.
- Log to the conversation JSON files that already exist.
- Add a summary command: `python main.py --mode stats --llm-costs`

### 4.4 Tournament CI Automation
**Priority:** Low-Medium
- GitHub Actions workflow that runs a tournament on each release.
- Publishes updated ELO ratings and RL-vs-LLM results to the docs site.

**Milestone:** At the end of Phase 4, "How does GPT-5 compare to a 1M-step PPO agent
at tactical reasoning?" has a quantified, reproducible answer.

---

## Phase 5: Platform Expansion (8+ weeks, longer-term)

Bigger efforts that grow the project's reach and research utility.

### 5.1 Curriculum Learning System
**Priority:** Medium — makes the project course-ready.
- Formalize the progressive environments from Phase 2 into a `CurriculumEnv` that
  automatically advances difficulty when the agent reaches a win-rate threshold.
- Stages: tiny map → small map → full map; warriors only → all units; no fog → fog of war.
- Track and visualize progress across stages.

### 5.2 Multi-Agent RL (3+ Players)
**Priority:** Medium — opens up MARL research.
- The 1v1v1 and 2v2 maps exist but `StrategyGameEnv` is hardcoded for 2 players.
- Extend to PettingZoo-compatible multi-agent env.
- Enables research on coalition formation, opponent modeling, etc.

### 5.3 Web-Based Interface
**Priority:** Lower — largest effort, broadest reach.
- Browser-based version using either Pyodide (Python in browser) or a React frontend
  with WebSocket to a Python backend.
- The headless mode already separates logic from rendering, so architecture supports this.
- Would let people try the game without any installation.

### 5.4 Sound & Music
**Priority:** Lower for RL use, higher for game polish.
- Listed as "High Priority" in implementation-status.md.
- Add sound effects for combat, unit creation, structure capture.
- Background music tracks.
- Matters for demos and the "game" experience.

### 5.5 Campaign / Story Mode
**Priority:** Lower — nice to have.
- Scripted scenarios that teach game mechanics through guided play.
- Could double as an RL curriculum: each campaign mission is a training scenario.

---

## Visual Timeline

```
Week  1-2   ████ Phase 1: Fix Foundations
Week  3-6   ████████ Phase 2: Educational Scaffolding
Week  7-12  ████████████ Phase 3: RL Depth & Model Zoo
Week 13-20  ████████████████ Phase 4: LLM & Benchmark Polish
Week 21+    ████████████████████ Phase 5: Platform Expansion
```

Phases can overlap — Phase 2 documentation can start while Phase 1 fixes land.
Items within a phase can be parallelized across contributors.

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
