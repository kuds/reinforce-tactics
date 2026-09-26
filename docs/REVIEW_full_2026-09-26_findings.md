# Full codebase review — findings appendix (2026-09-26)

Companion to [`REVIEW_full_2026-09-26.md`](REVIEW_full_2026-09-26.md). This file lists all **351 findings** that survived adversarial verification (about 264 distinct issues once duplicate reports are merged), grouped by the reviewer that raised them.

How to read an entry:

- **ID** is `<reviewer>-<n>`. Findings raised by more than one reviewer are cross-linked under *Same issue as*.
- **Severity** is the verifier's corrected severity; the reviewer's original rating is shown when it differed.
- **Verdict**: `confirmed` (real as stated) or `partially` (real, but the verifier narrowed the scope or mechanism; see the verifier note).
- **Effort**: S (< 2h), M (half day to 2 days), L (> 2 days).
- Locations are `path:line` at commit `8f51132`. Line numbers will drift as the code changes.
- Entries are the reviewers' and verifiers' own wording. Some claims were re-measured while the main report was written. Where this appendix and the main report disagree, the main report is authoritative. Example: `core-1` lists BalancedRandomBot as affected by the stale cache, but it was measured unaffected; only RandomBot and MixedBot(random) freeze.

Critical, high and medium findings have full entries. Low and info findings are listed in compact tables at the end of each section.

| Severity | Count |
|---|---|
| critical | 8 |
| high | 39 |
| medium | 135 |
| low | 148 |
| info | 21 |

## Contents

- [Core game engine](#core-core-game-engine) (25)
- [Rule-based bots](#rulebots-rule-based-bots) (25)
- [LLM, model and AlphaZero bots](#aibots-llm-model-and-alphazero-bots) (24)
- [RL environment (Gymnasium)](#rlenv-rl-environment-gymnasium) (22)
- [PPO bootstrap / self-play training pipeline](#rltrain-ppo-bootstrap--self-play-training-pipeline) (23)
- [Feudal RL, AlphaZero/MCTS and imitation learning](#rlalt-feudal-rl-alphazeromcts-and-imitation-learning) (26)
- [Pygame application shell and renderer](#pygame-pygame-application-shell-and-renderer) (25)
- [Animations, replay playback and video export](#anim-animations-replay-playback-and-video-export) (25)
- [Menus, widgets, map editor, i18n and settings](#menus-menus-widgets-map-editor-i18n-and-settings) (27)
- [Save/load, replays, tournament, CLI and cloud](#persist-saveload-replays-tournament-cli-and-cloud) (26)
- [Cross-cutting consolidation, dead code and repo hygiene](#consolidate-cross-cutting-consolidation-dead-code-and-repo-hygiene) (25)
- [Tests, CI and tooling](#tests-tests-ci-and-tooling) (25)
- [Prior-review status audit](#prior-prior-review-status-audit) (28)
- [Coverage-gap critic](#critic-gaps-coverage-gap-critic) (13)
- [Cross-subsystem integration critic](#critic-integration-cross-subsystem-integration-critic) (12)

## core Core game engine

### `core-1` — end_turn() never invalidates the legal-actions cache, so masks and bots after a pass run on stale action sets (RandomBot does nothing on every turn after the agent passes)

**critical** · rl-correctness · confirmed · effort S · `reinforcetactics/core/game_state.py:1192`

Also: `reinforcetactics/core/game_state.py:1304`, `reinforcetactics/game/bot.py:88`, `reinforcetactics/rl/mcts.py:93`, `reinforcetactics/rl/gym_env.py:1065`, `reinforcetactics/core/unit.py:220`

Same issue as: `rlenv-1`, `prior-1`

- **Impact.** Any turn in which the opponent makes no state change leaves the next player's mask stale. Against 'random' opponents (about 580 stage entries across configs), a passing agent freezes the opponent completely. Passivity is safe and draws replace losses, which likely feeds the draw/end-turn attractor. The noop stage soft-locks the agent to end_turn. MCTS end_turn->end_turn lines inherit the deepcopied stale cache (rl/mcts.py:93). Win rates evaluated against random are biased.
- **Fix.** Call `self._invalidate_cache()` in end_turn() right after the game_over guard, and again before returning. Better still, replace the boolean flag with a `_state_version` counter that every mutator bumps, and key the cache by (player, version). Route Unit-level mutations (end_unit_turn, cancel_move) through GameState so they bump it too. Add a regression test: end_turn twice with no actions, then assert get_legal_actions(p) equals a fresh recompute.

### `core-2` — GameState action methods do not check legality (action budget, whose turn, range, tile/ownership, enabled units, game_over); multi_discrete env and LLM bots exploit this

**high** · bug · confirmed · effort M · `reinforcetactics/core/game_state.py:651`

Also: `reinforcetactics/core/game_state.py:787`, `reinforcetactics/core/game_state.py:894`, `reinforcetactics/core/game_state.py:1025`, `reinforcetactics/core/game_state.py:1273`, `reinforcetactics/rl/gym_env.py:1004`, `reinforcetactics/game/llm_bot.py:1076`

Same issue as: `rlenv-2`, `aibots-1`

- **Impact.** multi_discrete is the TrainingConfig default (rl/config.py:50) and 6 shipped configs use it. Those policies can learn spawn-anywhere, heal spam and extra attacks by exhausted units, so they are trained on a different game. LLM tournaments can capture an HQ in one turn by repeating SEIZE. Changes made after game over make the replay's final_unit_counts/final_hp_totals disagree with the recorded actions.
- **Fix.** Make the engine the single enforcer. Each action method first rejects when `self.game_over`, when `actor.player != self.current_player`, when the actor is paralyzed, when `can_attack`/`can_move` is already spent, on target alignment, and when out of range (reuse the same predicates get_legal_actions uses). create_unit must also require an in-bounds tile with type 'b' owned by the player and `unit_type in self.enabled_units`. Return the same failure shapes callers already handle. A failed seize should not consume the unit's actions or be recorded.

### `core-3` — Out-of-range defenders always deal 1 phantom counterattack damage (the min-1 damage floor applies to 0 base damage)

**high** · bug · confirmed · effort S · `reinforcetactics/core/mechanics.py:445`

Also: `reinforcetactics/core/mechanics.py:219`, `reinforcetactics/core/mechanics.py:332`

Prior review: REVIEW_ppo_training.md §2.7 ("min-1-damage clamp grants phantom counter-attacks")

Same issue as: `prior-11`

- **Impact.** Every ranged harass by a Mage or Sorcerer, and every melee hit on an Archer, costs the attacker 1 HP it should not lose. This can kill 1-HP units, skews balance and bot heuristics, and adds a phantom term to the RL damage_taken reward.
- **Fix.** Only counter when the defender can reach: `if can_counter and target.get_attack_damage(attacker.x, attacker.y, target_on_mountain) > 0`. Alternatively make apply_defence_reduction return 0 when base_damage <= 0 (which also stops out-of-range primary attacks from dealing 1). Add tests: Mage at range 2 vs Warrior, and melee vs adjacent Archer.

### `core-4` — 2v2 team mode does not work: the team suffix is parsed and ignored, players 3-4 own nothing, and the engine has no concept of allies

**high** · bug · confirmed · effort L · `reinforcetactics/core/tile.py:38`

Also: `maps/2v2/beginner.csv:1`, `reinforcetactics/app/game_loop.py:240`, `reinforcetactics/core/mechanics.py:81`, `reinforcetactics/core/game_state.py:430`

Same issue as: `persist-18`

- **Impact.** The README advertises "2v2 (team) maps" and the UI game-mode menu offers it, but in practice it is a 2-player game with two idle seats and friendly fire between intended teammates.
- **Fix.** Settle on one tile encoding (e.g. type_player, plus a separate teams map in map metadata or a `teams: {player: team}` GameState argument). Add `GameState.are_allies(p1, p2)` and use it in every hostility check (attack, flank, paralyze, heal/buff targeting, legal actions, elimination/HQ win conditions). Fix maps/2v2/*.csv so each of the 4 players owns an HQ. Until then, hide 2v2 from the mode menu.

### `core-5` — Fog of war leaks hidden information: shrouded structures show their live owner/HP in to_numpy, and the move mask reveals invisible enemies

**medium (reviewer: high)** · rl-correctness · confirmed · effort M · `reinforcetactics/core/game_state.py:1536`

Also: `reinforcetactics/core/mechanics.py:58`, `reinforcetactics/core/visibility.py:184`, `reinforcetactics/rl/observation.py:209`, `reinforcetactics/ui/renderer.py:443`

Same issue as: `rlenv-14`, `consolidate-1`, `critic-integration-3`, `pygame-12`

- **Impact.** Any FOW training run or evaluation (via observation.py:209 `to_numpy(for_player=...)`) gets remote structure state and enemy positions through both the observation and the mask, so the policy learns to rely on hidden information. The UI has the same leak (renderer.py:443 picks team-coloured variants from live tile.player in fog). No shipped config enables FOW yet, which is why this is high rather than critical.
- **Fix.** In to_numpy, for SHROUDED tiles write owner/HP from `vis_map.last_seen_structures` and zero units, and replace the Python double loop with boolean-mask indexing. For movement under FOW, pathfind treating non-visible enemies as passable and resolve collisions at execution time (for example, stop on the last free tile, an 'ambush' rule), or document the leak as accepted. Add a test that the FOW obs and mask are invariant to hidden enemy placement.
- **Verifier note.** Both mechanisms are real. In to_numpy (1535-1548), non-VISIBLE tiles are masked only when `visibility_state[y, x] == 0`, so SHROUDED (1) structures keep live owner and HP. get_legal_actions builds moves with can_move_to_position over all self.units (game_state.py ~1343-1354; mechanics.py:58-70), so hidden enemies block paths and destinations. last_seen_units/last_seen_structures are only read by get_last_seen_* and clear_stale_unit_memory, and get_last_seen_* is never called anywhere.

### `core-6` — Save/load drops max_turns, unit stat overrides, has_moved, end_reason and healing totals; a loaded unit can have health > max_health

**medium (reviewer: high)** · bug · partially · effort S · `reinforcetactics/core/game_state.py:1436`

Also: `reinforcetactics/core/game_state.py:1713`, `reinforcetactics/core/unit.py:307`, `reinforcetactics/core/unit.py:283`

Prior review: REVIEW_advancedbot.md §1 (marked FIXED, but only the from_dict read landed; to_dict still never writes max_turns)

Same issue as: `persist-9`

- **Impact.** A saved game with a turn limit becomes unlimited after loading. Balance-sweep games reload under default stats, with HP above max (HP% > 100 in observations). Replay integrity totals (healing_totals, winning_action_index) are wrong for continued games.
- **Fix.** Persist every field from_dict reads (max_turns, end_reason, winning_action_index, healing_totals, has_moved, padding metadata, original_map_data) and restore healing_totals. Pass stats through: `Unit.from_dict(d, stats=game.unit_data[d['type']])`. Copy input containers (`list(...)`, `dict(...)`). Add a round-trip property test: `from_dict(json(to_dict(g))).to_dict() == to_dict(g)` over random mid-game states.
- **Verifier note.** All serialization gaps are real. to_dict omits max_turns, end_reason, winning_action_index and healing_totals, although from_dict reads the first three (1734, 1755, 1756). Unit.from_dict (unit.py:307-310) builds without stats. has_moved is not in Unit.to_dict. from_dict aliases action_history and falls back to the module-level ALL_UNIT_TYPES list. The practical impact is overstated.

### `core-8` — Haste works differently in the engine/RL path and the bot/UI path; hasting a unit that has not acted, or a paralyzed ally, is wasted in RL

**medium** · rl-correctness · confirmed · effort M · `reinforcetactics/core/mechanics.py:587`

Also: `reinforcetactics/core/unit.py:220`, `reinforcetactics/core/mechanics.py:226`, `reinforcetactics/core/game_state.py:725`, `reinforcetactics/core/game_state.py:1408`

- **Impact.** Sorcerer value differs between RL agents and scripted/human play (unfair comparisons, bot-derived imitation data mismatched to env dynamics). The mask includes wasted haste actions, and paralysis can be bypassed through the direct API.
- **Fix.** Put haste in the engine: add a `GameState._consume_action(unit)` used by attack/seize/heal/etc. that calls end_unit_turn semantics (if is_hasted, refresh once). Stop Haste from directly setting can_move/can_attack on units that haven't acted. Exclude paralyzed targets from get_hasteable_allies and haste_unit. Make move_unit/attack reject paralyzed actors.

### `core-9` — Unit.cancel_move bypasses GameState: the recorded move stays in replay history and the vision it revealed is kept (FOW scout-and-cancel)

**medium** · bug · confirmed · effort S · `reinforcetactics/core/unit.py:206`

Also: `reinforcetactics/app/action_executor.py:25`, `reinforcetactics/app/action_executor.py:69`, `reinforcetactics/core/game_state.py:783`

- **Impact.** Human games with cancels produce replays that move units the live game reverted (replay/state divergence). Human players can scout for free under fog of war.
- **Fix.** Add `GameState.cancel_move(unit)`. It restores position, records an 'undo_move' (or pops the last matching move record when it is the latest action), invalidates the cache, and recomputes visibility using the pre-move snapshot. Either disallow cancel after new tiles were revealed, or keep the revealed tiles SHROUDED. Point the UI at it.

### `core-10` — Combat randomness defaults to the global `random`; the tournament, AlphaZero and imitation paths build GameState without an rng

**medium** · bug · confirmed · effort S · `reinforcetactics/tournament/runner.py:310`

Also: `reinforcetactics/core/mechanics.py:429`, `reinforcetactics/tournament/runner.py:310`, `reinforcetactics/rl/alphazero_trainer.py:529`, `reinforcetactics/rl/imitation.py:607`

Same issue as: `persist-6`

- **Impact.** Rogue evade rolls make tournaments, AlphaZero evaluation and BC dataset generation non-reproducible even when seeds are set, and the global RNG stream is shared with other consumers.
- **Fix.** Make GameState own a `random.Random` by default (`rng if rng is not None else random.Random(seed)`, with an explicit `seed` argument that is recorded in the replay game_info), and pass a per-game seed at the three call sites. Keep the global fallback only behind an explicit opt-in.

### `core-14` — Five parallel action-dispatch tables outside the engine; add a single GameState.apply(action) / is_legal(action)

**medium** · consolidation · confirmed · effort M · `reinforcetactics/core/game_state.py:1294`

Also: `reinforcetactics/rl/mcts.py:191`, `reinforcetactics/game/bot.py:107`, `reinforcetactics/rl/gym_env.py:981`, `reinforcetactics/utils/replay_actions.py:282`, `reinforcetactics/game/llm_bot.py:1022`

Same issue as: `consolidate-9`, `aibots-13`, `prior-14`, `pygame-18`, `rulebots-20`

- **Impact.** Validation drift between paths is the direct cause of the multi_discrete and LLM exploits. Every new ability requires editing 5+ files.
- **Fix.** Add `GameState.apply_action(key, payload) -> ActionResult` together with `is_legal(key, payload)`, backed by the same predicates get_legal_actions uses (see the validation finding). Have MCTS, RandomBot, the gym env, the LLM bots and replay v3 call it; replay can keep its recorded-outcome override. This is a concrete seam that also makes the god object smaller.

### `core-15` — Duplicated rule logic: 9 range helpers, 5 status-tick helpers, 6 ability wrappers, and two sources of truth for attack ranges

**medium** · consolidation · confirmed · effort M · `reinforcetactics/core/mechanics.py:74`

Also: `reinforcetactics/core/game_state.py:875`, `reinforcetactics/core/unit.py:83`, `reinforcetactics/core/unit.py:118`, `reinforcetactics/core/mechanics.py:671`

Prior review: REVIEW_maintainability.md §7; REVIEW_advancedbot.md §21

Same issue as: `prior-15`

- **Impact.** Every mask-vs-execution drift bug the comments warn about (mechanics.py:478-485, 512-518) comes from this duplication, and a balance change to a range has to touch 3-4 sites.
- **Fix.** Add ABILITY_RANGES / PARALYZE_RANGE / HASTE_RANGE / BUFF_RANGE to constants and a single `units_in_range(center, units, lo, hi, pred)`. Derive get_attack_damage from get_attack_range. Replace the ticks with one `tick_statuses(units, player)` driven by a table of (attr, unit_type or None). Collapse the six wrappers into `_support_action(kind, actor, target, fn)`. Delete get_adjacent_allies and get_adjacent_paralyzed_allies (used only in tests).

### `core-16` — GameState god object (1793 lines): concrete seams are EngineConfig, the serializer, the legal-action generator and a visibility facade

**medium** · design · confirmed · effort L · `reinforcetactics/core/game_state.py:40`

Also: `reinforcetactics/core/game_state.py:1294`, `reinforcetactics/core/game_state.py:1436`, `reinforcetactics/core/game_state.py:1713`

Prior review: REVIEW_advancedbot.md §19; REVIEW_maintainability.md §24

- **Impact.** Hard to test in isolation (legal actions, serialization and overrides all need a full GameState), and it accounts for most of the drift and missed-field bugs above.
- **Fix.** Extract, in order: (1) a frozen `EngineConfig` dataclass (unit_data, income_rates, starting_gold, damage_model, structure_health, max_units) with `from_overrides()` and `to_dict()`, so persistence is automatic; (2) `rules/legal_actions.py:enumerate(state, player)` sharing predicates with mechanics; (3) `serialization.py` (to_dict/from_dict/save_replay/_get_player_type); (4) a `FogOfWar` object owning visibility_maps, the snapshots and the refresh hooks. Keep GameState as the state container plus the apply_action facade.

### `core-17` — No tests check cache coherence, full save/load round-trips or rule invariants at the engine boundary

**medium** · test-gap · confirmed · effort M · `reinforcetactics/core/game_state.py:1305`

Also: `tests/test_game_state.py:1`, `tests/test_mechanics.py:606`

- **Impact.** Regressions in the most safety-critical RL surface (masks) go unnoticed until they show up as training pathologies.
- **Fix.** Add (a) a debug flag (env var RT_CHECK_CACHE=1) that makes get_legal_actions recompute and assert equality with the cached value, enabled in CI; (b) a hypothesis/random-play test that drives RandomBot vs RandomBot for N turns and checks the invariants (no unit on a non-walkable tile, health <= max_health, each unit acts at most once per turn unless hasted, a fresh mask equals the cached mask, JSON round-trip equality); (c) targeted tests for out-of-range counters and multi-player end conditions.

#### Low and info findings (12)

| ID | Sev | Title | Location | Effort | Fix (summary) |
|---|---|---|---|---|---|
| `core-7` | low | Free-for-all (1v1v1) end rules are inconsistent: any HQ capture ends the whole game, while eliminated or resigned players keep taking turns, collecting income … | `reinforcetactics/core/game_state.py:1054` | M | Add `self.eliminated_players: set[int]`. HQ capture in games with 3+ players eliminates the HQ's owner: remove their units, and either neutralise their structures or hand them to the capturer. Resign and zero-units elimination feed the same set. |
| `core-11` | low | Paralysis has an off-by-one: PARALYZE_DURATION=3 costs the victim 2 turns (README says 3), unlike buffs where 3 means 3 turns | `reinforcetactics/core/game_state.py:1227` | S | Pick one semantic: "N of the affected unit's own turns". Decrement paralysis at the end of the victim's turn (in end_turn, before switching to the outgoing player's units), or set PARALYZE_DURATION=2 and update the docs. Add a test that counts lost turns. |
| `core-12` | low | FOW visibility is not refreshed after create_unit, capture, own-unit death or cancel, and GameState never initializes it itself | `reinforcetactics/core/game_state.py:651` | S | Call `self.update_visibility(player)` from create_unit, after a successful capture in seize, and after unit removal in attack (for the dead unit's owner). Initialize all visibility maps at the end of __init__ and from_dict, and remove the four external calls. |
| `core-13` | low | Player 1 skips start-of-turn processing on turn 0, so Player 2 is always one income tick ahead | `reinforcetactics/core/game_state.py:1261` | S | Extract the start-of-turn block into `_begin_turn(player)`, call it from end_turn, and add an explicit, documented choice for turn 0 (either call `_begin_turn(1)` in __init__, or skip P2's first income). |
| `core-18` ≈ rlalt-18 | low | MCTS clones full GameState with deepcopy, including action_history and caches (5-11 ms per node) | `reinforcetactics/rl/mcts.py:426` | M | Add `GameState.clone(for_search=True)`. It copies units, tile owner/HP/regenerating, gold, cooldowns and turn fields; shares the immutable grid terrain and engine config; |
| `core-19` ≈ aibots-18, persist-22, anim-8, anim-18, consolidate-11 | low | Padded/original coordinate plumbing is dead (set_map_metadata is never called), and record_action's conversion is incomplete (sorcerer_pos) and depends on … | `reinforcetactics/core/game_state.py:604` | S | Either delete the padding plumbing (set_map_metadata, original_map_*, conversion loop, llm_bot conversions), or make it table-driven: `_COORD_PAIRS=(('x','y'),('from_x','from_y'),('to_x','to_y'))` and `_POS_KEYS` derived from a single list that includes … |
| `core-20` ≈ rlenv-3, prior-17 | low | Pathfinding scans the whole unit list for every BFS cell; unit lookup by position is O(n) | `reinforcetactics/core/mechanics.py:59` | S | Build `occ = {(u.x,u.y): u for u in self.units}` once per get_legal_actions and move_unit and pass it to a `can_enter(x, y, occ, mover, is_destination)` helper. |
| `core-21` | low | VisibilityMap.update and to_numpy use per-tile Python loops where numpy slicing works | `reinforcetactics/core/visibility.py:151` | S | Use the slice assignment. Iterate `grid.get_capturable_tiles(player)` (or a cached list of structure positions) instead of the full grid. In to_numpy use `grid_state[visibility_state == UNEXPLORED] = 0`. |
| `core-22` | low | Dead or unreachable code in core | `reinforcetactics/core/game_state.py:568` | S | Delete the unused helpers, or wire get_unit_count into create_unit and get_legal_actions. Either accept lists in TileGrid (`np.asarray(map_data, dtype=object)`) or drop the branch. |
| `core-23` | low | Rules documentation disagrees with the constants and the code comments | `README.md:148` | S | Generate the README rules table from UNIT_DATA/constants (a small script or test that fails on drift), fix the comments, and switch tile.py to logger.warning. |
| `core-24` | low | constants.py mixes engine rules with UI assets and repeats data (UNIT_COLORS, duplicate TileType keys, TileType unused in core) | `reinforcetactics/constants.py:163` | M | Split into `rules.py` (UNIT_STATS with cost/movement/health/attack/defence only, ranges, economy) and `ui/theme.py` (colors, sprite paths, ANIMATION_CONFIG). Delete UNIT_COLORS and the duplicate keys. Use TileType / a UnitType enum in core. |
| `core-25` | info | Documented terrain rules are not implemented: road speed, forest stealth, and 'enemy HQ always visible' (movement is uniform cost) | `reinforcetactics/core/unit.py:184` | M | Add a TERRAIN_MOVE_COST table (road 0.5 or a +1 movement bonus, forest/mountain 2) and switch BFS to Dijkstra (grid <= 30x30, cheap). Add forest concealment (units in forest are visible only to adjacent enemies) in VisibilityMap.update. |

## rulebots Rule-based bots

### `rulebots-1` — get_reachable returns ally-occupied tiles and ignores can_move; ~68% of SimpleBot moves are rejected by the engine

**high** · bug · confirmed · effort S · `reinforcetactics/game/bot_base.py:227`

Also: `reinforcetactics/game/bot_base.py:295`, `reinforcetactics/game/bot.py:551`, `reinforcetactics/game/bot.py:558`, `reinforcetactics/core/game_state.py:752`, `reinforcetactics/core/game_state.py:1351`

- **Impact.** SimpleBot, the main curriculum opponent, jams its army behind its own units: SimpleBot mirrors ended in max_turns_draw in 19 of 20 games across 5 maps. Filtering occupied destinations (scratch monkeypatch) turned 4 of those 20 into HQ captures. This is also the root cause of the illegal attacks (knight charge) and the capture-claim inflation findings below. The retreat, capture and interrupt paths all waste actions the same way.
- **Fix.** Make get_reachable return only legal destinations: `if not unit.can_move: return []`, then drop tiles failing can_move_to_position(..., is_destination=True). This matches get_legal_actions at game_state.py:1351. If a caller truly needs path semantics, add a separate get_pass_through_reachable. Also have pick_capture_target skip structures occupied by an ally, and have continue_active_seizes add seized tiles to _capture_assigned.
- **Status: fixed in the rule-bot legality package (with `rulebots-6`, `-8`, `-14` and `-19`). Consequences accepted pending the §8 week-4 curriculum re-baseline, which must come before any curriculum run on this code.**
  - The tier ladder no longer rises on the curriculum maps. The figures below are seeded head-to-heads: 25 seeds × both seats, 75 turns, fog off. Each is W/D/L for the first tier, measured at 302893d and then on the package.

    | Pairing | Map | 302893d | Package |
    |---|---|---|---|
    | AdvancedBot vs MediumBot | beginner | 17/0/33 | 5/0/45 |
    | MasterBot vs MediumBot | beginner | 17/0/33 | 6/0/44 |
    | MasterBot vs MediumBot | intermediate | 11/0/39 | 0/0/50 |
    | MediumBot vs SimpleBot | skirmish | 32/2/16 | 28/1/21 |
    | AdvancedBot vs MediumBot | skirmish | 21/0/29 | 22/0/28 |

    `configs/ppo/bootstrap.yaml`'s `beginner_mixed_med_adv_50` and `beginner_advanced` stages assume AdvancedBot is harder than MediumBot on beginner, and so do the `bootstrap_sweep/` variants that copy them. Re-check the stage order there, not only the thresholds.
  - MasterBot is weaker on skirmish. Against 302893d's MasterBot it wins 176 and loses 264 of 440 seeded 120-turn games (seeds 30–249, both seats).

### `rulebots-2` — Knight charge / Rogue flank attack without checking the move succeeded; the engine applies illegal out-of-range melee attacks

**high** · bug · confirmed · effort S · `reinforcetactics/game/bot.py:2138`

Also: `reinforcetactics/game/bot.py:2193`, `reinforcetactics/core/game_state.py:787`, `reinforcetactics/core/mechanics.py:223`

- **Impact.** Rules-violating ranged melee attacks (1 dmg each way) are applied to game state and written into replays, about 9 per game for Advanced/Master. This happens in GUI games, tournaments and RL training against advanced opponents. Telemetry also counts these as successful charges/flanks.
- **Fix.** Use `if not self.game_state.move_unit(...): return False`, then re-check get_attackable_enemies before attacking, as _execute_focus_fire already does at 1042. For defence in depth, make GameState.attack return its no-op dict when `not attacker.can_attack` or `attacker.get_attack_damage(target.x, target.y, on_mountain) == 0`, mirroring move_unit's guards.

### `rulebots-6` — No-progress recursion after failed moves in act_with_unit_enhanced; each level claims another capture target

**high** · bug · confirmed · effort S · `reinforcetactics/game/bot.py:1959`

Also: `reinforcetactics/game/bot.py:1944`, `reinforcetactics/game/bot.py:1907`, `reinforcetactics/game/bot.py:2059`, `reinforcetactics/game/bot.py:2089`, `reinforcetactics/game/bot.py:1242`

Prior review: REVIEW_advancedbot.md §12 (depth guard added; no-progress recursion still occurs within the cap)

- **Impact.** Sibling units are starved of capture targets because pick_capture_target skips claimed tiles. That directly undermines the EXPAND/CONQUER capture strategy and makes early turns look like stalls. About half of per-unit CPU goes to wasted recursion.
- **Fix.** Check unit.can_move before any move branch and check move_unit's return value. Claim a structure only after a successful move toward it. Recurse only through end_unit_turn() (see the haste finding). Add an invariant test: recursion depth <= 2 unless the unit is hasted, and claimed structures <= live units.

### `rulebots-3` — Bot damage/kill predictions ignore defence, buffs and hp_scaled; 24% (flat) to 42% (hp_scaled) of 'sure kills' fail

**medium (reviewer: high)** · bug · confirmed · effort M · `reinforcetactics/game/bot.py:1095`

Also: `reinforcetactics/game/bot.py:905`, `reinforcetactics/game/bot.py:919`, `reinforcetactics/game/bot.py:1285`, `reinforcetactics/game/bot.py:1926`, `reinforcetactics/game/bot.py:2035`, `reinforcetactics/game/bot.py:2809`

Same issue as: `consolidate-3`, `critic-integration-12`

- **Impact.** The kill-confirm path returns 1000+ and skips the suicide guard (1161), so bots make suicidal attacks. Focus-fire groups under-commit because kill sets are planned with inflated damage. Opponent strength in RL sweeps that use hp_scaled is distorted.
- **Fix.** Add an engine-owned deterministic preview, e.g. GameMechanics.preview_attack(attacker, target, grid, units, damage_model, from_pos=None, move_distance=None) -> {damage, counter_damage, kills, evade_chance}. Build it by factoring the non-random part out of attack_unit, and route every bot estimate through it. LLM prompts and UI tooltips could use it too.
- **Verifier note.** The mechanism is confirmed. calculate_attack_value (bot.py:1095) uses raw get_attack_damage. The counter estimate (1117, 1126) is raw x0.8 with no defence reduction, attack/defence buffs or hp_scaled scaling. The same raw pattern is at 905, 919, 1285, 1926, 2035-2037 and bot_base.py:397. The kill branch returns 1000+dmg before the suicide guard at 1161.

### `rulebots-4` — Knight charge and Rogue flank-move targets are scored from the unit's pre-move tile

**medium (reviewer: high)** · bug · confirmed · effort S · `reinforcetactics/game/bot.py:2131`

Also: `reinforcetactics/game/bot.py:2136`, `reinforcetactics/game/bot.py:2181`, `reinforcetactics/game/bot.py:2192`, `reinforcetactics/game/bot.py:1093`

- **Impact.** Charge/flank selection effectively becomes 'most expensive melee target adjacent to a reachable tile', regardless of the trade. 350g knights and rogues throw themselves away. MasterBot inherits this: its docstring claims a threat-aware charge override that doesn't exist.
- **Fix.** Score from the landing tile: temporarily set unit.x/y = pos (the pattern find_killable_targets already uses at 914-924), or pass from_pos to the preview API from the damage-prediction finding. Apply a `best_value > 0` gate in the flank-move branch as well.
- **Verifier note.** Confirmed. _try_knight_charge (bot.py:2131) scores each landing pos with calculate_attack_value(unit, enemy, move_distance) while unit.x/y are still at the origin. The enemy is at least 2 from the origin, so melee damage is 0, there is no kill branch, and the counter is 0 for non-ranged targets. The value is therefore target_cost/100 > 0 and the best_value > 0 gate (2136) never filters a suicidal charge. The rogue flank-move branch (2181) has the same bug and commits with no value gate (2192).

### `rulebots-5` — Ranged units in focus fire, interrupts and approach move to Manhattan-closest tile: Archers end adjacent and can't shoot

**medium (reviewer: high)** · bug · confirmed · effort S · `reinforcetactics/game/bot.py:1038`

Also: `reinforcetactics/game/bot.py:1256`, `reinforcetactics/game/bot.py:1905`, `reinforcetactics/game/bot.py:2057`, `reinforcetactics/game/bot.py:517`, `reinforcetactics/game/bot_base.py:295`

- **Impact.** Planned kills fail, and the other committed attackers take counters for nothing. Archers are 20% of AdvancedBot's composition target, so Medium+ ranged play is badly degraded.
- **Fix.** Promote SimpleBot._find_ranged_attack_position (517-539) into BotUnitMixin as find_attack_position(unit, target). It should consider legal destinations with min_r <= dist <= max_r (using the range from that tile, including the mountain bonus) and prefer max damage, then max distance. Use it at all four sites and inside find_killable_targets so planned and executed positions agree.
- **Verifier note.** Confirmed. _execute_focus_fire (bot.py:1038), the MediumBot/AdvancedBot interrupts (1256, 1905) and the Priority 7 approach (2057) all use find_best_move_position, which minimises Manhattan distance (bot_base.py:313-317). Archers (min range 2) land adjacent. Mages land at distance 1 (8 damage) even though find_killable_targets (905-924) planned the max-damage tile (12 at distance 2).

### `rulebots-7` — StrategyGameEnv silently runs with NO opponent for opponent='master' or any unknown string

**medium (reviewer: high)** · rl-correctness · confirmed · effort S · `reinforcetactics/rl/gym_env.py:1655`

Also: `reinforcetactics/rl/gym_env.py:71`, `reinforcetactics/rl/gym_env.py:1627`, `reinforcetactics/rl/config.py:200`, `scripts/ab_feudal_ar.py:151`

Prior review: REVIEW_rl_pipeline_2026-07-24.md §2.13 (same silent-None fallback, raised for 'self')

Same issue as: `rlenv-10`, `rltrain-22`

- **Impact.** Training or evaluating against MasterBot, or making any typo, silently trains against a pass-bot and reports inflated win rates. This is the scripted-bot analogue of the 'self' silent-None issue already on record.
- **Fix.** Derive the accepted set from the registry (set(SCRIPTED_BOTS) | {'bot','self'}), validate in __init__ with a ValueError that lists valid names, and build every non-'self' opponent via canonical_name/build_scripted. Replace config.py's _CURRICULUM_OPPONENTS with the same derived set.
- **Verifier note.** Confirmed. _BOT_OPPONENT_TYPES (gym_env.py:71) omits 'master', although bot_registry.SCRIPTED_BOTS has it. reset() builds a bot only for names in that set (1627) and otherwise sets self.opponent = None (1655-1656). _opponent_turn (1255) then no-ops. __init__ (467) does no validation. Curriculum configs are validated against _CURRICULUM_OPPONENTS (config.py:200, 274), which also lacks 'master'.

### `rulebots-8` — Haste re-entry checks never fire for live units (haste wasted in 3 tiers) and do fire for dead attackers (no-op recursion)

**medium** · bug · partially · effort S · `reinforcetactics/game/bot.py:1951`

Also: `reinforcetactics/core/game_state.py:868`, `reinforcetactics/core/unit.py:235`, `reinforcetactics/game/bot.py:2739`, `reinforcetactics/app/action_executor.py:113`

Prior review: REVIEW_advancedbot.md §3 (attacker_alive guard leaves dead attackers' flags set) and §12

- **Impact.** Sorcerer haste is wasted for Simple/Medium/Advanced; AdvancedBot's own sorcerer_haste rarely buys anything. Humans in the GUI do get the extra action because action_executor.py calls end_unit_turn after each action, so bots are handicapped relative to humans. CPU is wasted on dead-unit loops.
- **Fix.** Replace the 29 checks with one helper, e.g. `_continue_if_hasted(unit, depth, act)`: return if `unit not in self.game_state.units`, else `if unit.end_unit_turn(): act(unit, depth + 1)`. MasterBot's post-pass then becomes unnecessary. Consider also clearing can_move/can_attack for dead attackers in GameState.attack.
- **Verifier note.** Both mechanisms are real, but the 'haste wasted in 3 tiers' claim is overstated. (1) Live units: GameState.attack (game_state.py:868-870), seize (1057-1058), heal, paralyze and haste all zero can_move/can_attack without consuming is_hasted. So `if unit.can_move or unit.can_attack` after a terminal action never re-enters. Only end_unit_turn() (unit.py:235) consumes haste. Haste is only ever applied by AdvancedBot._try_sorcerer_abilities (bot.py:2318, 2328) and MasterBot.

### `rulebots-9` — find_killable_targets runs a BFS per (enemy, unit) pair and re-runs it after every kill; hoisting gives a 1.8-2.7x speedup with identical games

**medium** · performance · confirmed · effort S · `reinforcetactics/game/bot.py:911`

Also: `reinforcetactics/game/bot.py:891`, `reinforcetactics/game/bot.py:999`, `reinforcetactics/game/bot.py:711`, `reinforcetactics/core/mechanics.py:59`

Prior review: REVIEW_advancedbot.md §26-27 (repeated BFS and grid scans per turn; still open)

- **Impact.** The scripted opponent's turn cost sits on the RL env step path (once per agent end_turn) and makes GUI bot turns in 3-player games stall visibly (~0.25 s).
- **Fix.** Compute `reach = {id(u): self.get_reachable(u) for u in available}` once per call, or once per turn with invalidation when a unit moves. Precompute an occupied-position set so the BFS predicate is O(1). Cache enemies, capturable tiles and our HQ in a small per-turn TurnContext (find_our_hq is currently a full grid scan per structure, per unit).

### `rulebots-10` — Registry is not the single source of truth: GUI can't select MasterBot; 8+ hand-maintained bot lists remain

**medium** · consolidation · confirmed · effort M · `reinforcetactics/ui/menus/game_setup/player_config_menu.py:344`

Also: `reinforcetactics/app/bot_factory.py:24`, `reinforcetactics/tournament/bots.py:21`, `reinforcetactics/tournament/bots.py:227`, `reinforcetactics/cli/main.py:75`, `scripts/eval_agent.py:161`, `scripts/train/train_feudal_rl.py:689`

Same issue as: `menus-25`

- **Impact.** New or strongest bots silently fail to appear in the GUI, CLIs and tournaments, and the lists drift. That drift is what caused the silent no-opponent env bug.
- **Fix.** Attach metadata to registry entries (display_name, gui_selectable, ladder, accepts_kwargs) and generate every list from it. CLIs should use choices=sorted(SCRIPTED_BOTS) + aliases. Forward BotDescriptor.extra_kwargs in create_bot_instance. Remove the stale language keys.

### `rulebots-11` — MixedBot validates only the coin-flip winner's bot name; a typo crashes a curriculum run at a random Nth reset

**medium** · bug · confirmed · effort S · `reinforcetactics/game/bot.py:1420`

Also: `reinforcetactics/game/bot.py:1435`, `reinforcetactics/game/bot.py:1409`

- **Impact.** A misconfigured bridge stage (opponent_kwargs) passes validation and kills a long training run mid-curriculum, which is the failure mode the class comment says it prevents.
- **Fix.** Validate both easy and hard against _BOT_NAMES (and 0 <= p_hard <= 1) before the coin flip. Ideally also validate mixed opponent_kwargs in CurriculumStage validation (config.py:274).

### `rulebots-12` — Scripted bots ignore fog of war: they see and target hidden units

**medium** · design · confirmed · effort M · `reinforcetactics/game/bot.py:394`

Also: `reinforcetactics/game/bot.py:889`, `reinforcetactics/game/bot_base.py:369`, `reinforcetactics/core/game_state.py:1369`, `reinforcetactics/game/llm_bot.py:714`, `reinforcetactics/app/game_loop.py:265`

- **Impact.** In fog-of-war games scripted opponents are omniscient, which is unfair to humans and makes LLM/RL vs scripted comparisons under FOW invalid.
- **Fix.** Add a visible_enemies() helper in BotUnitMixin that uses game_state.get_visible_units_for_player(bot_player, include_own=False) when fog_of_war is on, and route all enemy lists through it. This also consolidates the 13 duplicated enemy list comprehensions.

### `rulebots-13` — Cross-tier duplication: ~194 identical lines plus 29 re-entry checks, 13 enemy comprehensions and parallel *_enhanced APIs

**medium** · consolidation · confirmed · effort M · `reinforcetactics/game/bot.py:2637`

Also: `reinforcetactics/game/bot.py:966`, `reinforcetactics/game/bot.py:649`, `reinforcetactics/game/bot.py:1702`, `reinforcetactics/game/bot.py:2596`, `reinforcetactics/game/bot.py:302`, `reinforcetactics/game/bot.py:823`

Prior review: REVIEW_advancedbot.md §20 (partially addressed by bot_base; tier duplication still open)

- **Impact.** Fixes land in one tier and not the others: the ranged-positioning helper exists only in SimpleBot, and paralyze telemetry differs between tiers. Calling advanced_bot.act_with_unit silently runs Medium logic.
- **Fix.** Use template methods: coordinate_attacks with an _order_attackers() hook; find_retreat_tile with a _retreat_score() hook; one purchase loop with a _choose_purchase(affordable) hook and a shared _apply_warrior_cap. AdvancedBot should override act_with_unit, purchase_units and move_and_act_units instead of adding *_enhanced methods. Delete _enemy_reachable_positions. Estimated -250 lines.

### `rulebots-14` — No bot legality test harness and no direct bot_registry tests; every defect above passes the 215 bot tests

**medium** · test-gap · confirmed · effort M · `reinforcetactics/game/bot_registry.py:93`

Also: `reinforcetactics/game/bot_registry.py:114`, `reinforcetactics/game/bot.py:226`

- **Impact.** Regressions in engine/bot contracts (illegal attacks, idle units, lost haste) go undetected, and registry alias/classification drift is untested.
- **Fix.** (a) Add test_bot_registry.py covering aliases, class names, BotType members, the KeyError message, and a player_type table that includes AlphaZeroBot. (b) Add a parametrized legality harness: every scripted tier on every maps/1v1 and 1v1v1 map for N turns, with move_unit/attack/seize wrapped to assert no rejected moves, no out-of-range or can_attack=False attacks, no actions by dead units, recursion depth <= 2 unless hasted, and claimed captures <= units. Mark it slow if needed.

#### Low and info findings (11)

| ID | Sev | Title | Location | Effort | Fix (summary) |
|---|---|---|---|---|---|
| `rulebots-15` | low | Purchase loops recompute the full legal-action set after every buy and ignore create_unit failure | `reinforcetactics/game/bot.py:285` | S | Compute buildable actions directly (owned, unoccupied buildings x enabled types, affordable per game_state.unit_data), or add GameState.get_create_actions(player). Break out of the loop when create_unit returns None. |
| `rulebots-16` | low | MasterBot rebuilds its threat map every turn but only retreat uses it; docstring claims a threat-aware knight-charge override that doesn't exist | `reinforcetactics/game/bot.py:2590` | S | Compute the threat map lazily on first use each turn. Either implement the threat-aware charge landing (score landing tiles with threat) or correct the docstring. |
| `rulebots-17` | low | Dead code in AdvancedBot/MasterBot: analyze_map state, threat_at, MAX_PERMUTE_ATTACKERS | `reinforcetactics/game/bot.py:1631` | S | Delete these along with the tests that pin them, or wire defensive_positions into the Priority 5 mountain search, which re-scans reachable tiles today. |
| `rulebots-18` | low | Bots read global UNIT_DATA costs instead of the per-game engine_overrides table | `reinforcetactics/game/bot.py:295` | S | Add a mixin helper unit_cost(t) returning self.game_state.unit_data[t]['cost'] and use it everywhere. |
| `rulebots-19` | low | Capability telemetry miscounts: paralyze under-recorded, charges/flanks recorded even when the move failed, MixedBot always empty | `reinforcetactics/game/bot.py:2268` | S | Delete the duplicate paralyze branch, since try_mage_paralyze already prioritises capturers. Record events only after the action succeeds. Have MixedBot proxy get_capabilities_fired to _inner. |
| `rulebots-20` ≈ core-14, consolidate-9, aibots-13, prior-14, pygame-18 | low | RandomBot/BalancedRandomBot swallow all exceptions; their executor duplicates MCTS's and gym_env's action tables | `reinforcetactics/game/bot.py:101` | S | Narrow the except to expected game-logic errors or remove it. Put one apply_structured_action(game_state, key, action, player) next to get_legal_actions and use it from RandomBot, MCTS and AlphaZeroBot. Derive the actor fields from ACTION_KEY_MAP. |
| `rulebots-21` | low | Simple/Medium/Advanced/Master take_turn don't honour the game_over contract; create_unit mutates state without a replay record | `reinforcetactics/game/bot.py:271` | S | Add an early `if self.game_state.game_over: return` via a BaseBot.take_turn template that calls an abstract _play_turn, and add a game_over guard to create_unit. |
| `rulebots-22` | low | Bots bypass BaseBot.__init__, forcing lazy/hasattr state; the package exports only 3 of 8 scripted bots and no registry API | `reinforcetactics/game/__init__.py:9` | S | Give BaseBot.__init__(game_state, player, rng=None) responsibility for bot_player, _rng, capabilities_fired and per-turn sets, and have subclasses call super(). Let NoopBot accept rng. Export all tiers and the registry from reinforcetactics.game. |
| `rulebots-23` ≈ aibots-10, rlalt-14, consolidate-16 | info | Registry covers only scripted bots: AlphaZeroBot is unreachable from any factory and player_type labels it 'bot' | `reinforcetactics/game/bot_registry.py:66` | M | Extend the registry with model-driven entries (name -> constructor taking model_path/device kwargs) for ModelBot and AlphaZeroBot, classify both as 'rl', and add a tournament descriptor kind for them. Together these make 'train -> ladder' a config-only step. |
| `rulebots-24` | info | Multi-player/team handling: first-in-scan HQ targeting in FFA, team field ignored, 9 of 12 1v1v1 bot mirrors draw | `reinforcetactics/game/bot.py:2696` | L | Add GameState.are_enemies(p, q) (team-aware) plus a mixin is_enemy(u) used by every enemy query. Pick the HQ-snipe target by distance or threat, and consider leader-targeting heuristics for FFA. |
| `rulebots-25` | info | Purchases ignore spawn location: the building is chosen by scan order | `reinforcetactics/game/bot.py:319` | S | After choosing the unit type, pick the building tile that minimises distance to the nearest unclaimed or contested structure or enemy (break ties with the rng). This takes a few lines in a shared _choose_spawn helper. |

## aibots LLM, model and AlphaZero bots

### `aibots-1` — LLM bot runs SEIZE/ATTACK/HEAL/CURE/PARALYZE without checking legality or remaining actions: HQ capturable in one turn, friendly fire, repeated attacks

**high (reviewer: critical)** · bug · confirmed · effort M · `reinforcetactics/game/llm_bot.py:1211`

Also: `reinforcetactics/game/llm_bot.py:1070`, `reinforcetactics/game/llm_bot.py:1100`, `reinforcetactics/game/llm_bot.py:1134`, `reinforcetactics/game/llm_bot.py:1046`, `reinforcetactics/core/game_state.py:787`, `reinforcetactics/core/game_state.py:1025`

Same issue as: `core-2`, `rlenv-2`

- **Impact.** LLM benchmark and tournament results can be decided by illegal instant HQ captures or repeated attacks. LLMs often repeat or hallucinate actions, so this happens in normal play, not only adversarially. FoW games can also be exploited by guessing hidden enemy coordinates. In multi_discrete RL training, over-approximated per-dimension masks can reach the same double-seize path.
- **Fix.** Fix the root cause in the engine: return a failure from GameState.seize/attack/heal/cure/paralyze/haste/buffs when the actor can't act (not (can_move or can_attack) after acting, or is paralyzed), when the target's owner is wrong, or when it is out of range. In LLMBot, check every action against get_legal_actions(self.bot_player) re-queried after each executed action (key on actor + target), reject anything that doesn't match, and cross-check MOVE's `from` against the unit's position (it is currently ignored). Add regression tests for repeated SEIZE, friendly-fire ATTACK and double ATTACK.
- **Verifier note.** Real as described. LLMBot checks only CREATE_UNIT against get_legal_actions (llm_bot.py:1035-1037). _execute_seize (1202-1212), _execute_attack (1070-1098), _execute_heal/_cure/_paralyze call the engine directly. The engine has no action-economy check: GameState.seize (game_state.py:1025-1061) only guards `if unit not in self.units`, and attack (787) only guards stale references.

### `aibots-2` — ModelBot can never play in the pygame GUI: padded UI maps fail the checkpoint size check and it silently falls back to SimpleBot

**high** · rl-correctness · partially · effort M · `reinforcetactics/app/bot_factory.py:147`

Also: `reinforcetactics/game/model_bot.py:182`, `reinforcetactics/app/game_loop.py:253`, `reinforcetactics/ui/menus/game_setup/player_config_menu.py:143`, `reinforcetactics/core/game_state.py:355`

Same issue as: `menus-5`

- **Impact.** The human-vs-RL feature doesn't work for any checkpoint: users who pick a trained model play SimpleBot, and the only notice is a console print. Padded checkpoints that do load play blind to the real board layout.
- **Fix.** Have the GUI record padding metadata: use load_map_with_metadata and call GameState.set_map_metadata with offset = pad offset + border_size (it is never called in production today). Then have ModelBot crop grid/units to the original window before build_observation and shift decoded action coordinates via original_to_padded_coords, as LLMBot does. Validate in player_config_menu against the selected map, accept `.pt`, and show a visible in-game warning instead of printing when falling back.
- **Verifier note.** Real for every file-based map, but 'can never play' is overstated. game_loop.py:253 (and 330 for saved games) loads with for_ui=True, border_size=2. That pads to MIN_MAP_SIZE=20 (constants.py:42) plus a 4-tile border, so every 1v1 map becomes 24x24 or larger (29x29 for the 25x25 map). ModelBot raises ValueError when obs dims < live dims (model_bot.py:182-187), and bot_factory.py:147-153 catches it and falls back to SimpleBot with only a print.

### `aibots-3` — ClaudeBot always sends an assistant prefill, which returns a 400 on Claude Opus 4.6 (listed as 'latest') and all 4.6+/5 models; model list is stale

**high** · bug · confirmed · effort S · `reinforcetactics/game/llm_bot.py:1336`

Also: `reinforcetactics/game/llm_bot.py:1359`, `reinforcetactics/game/llm_bot.py:40`

- **Impact.** Choosing the top-listed Claude model, or any current one, makes every request 400. Combined with the retry-then-pass behavior, the bot ends every turn with no actions, and a tournament records that as the model losing.
- **Fix.** Remove the prefill. Request JSON through structured outputs (`output_config={"format": {...json_schema...}}`) or rely on the system-prompt instruction plus _extract_json. Read text by iterating content blocks for `type == "text"`. Update the model list, drop retired IDs, and consider validating `self.model` against `client.models.list()` at construction.

### `aibots-4` — LLM errors are all retried the same way and then silently become a passed turn; missing SDK, bad key or bad parameters never surface

**high** · bug · confirmed · effort M · `reinforcetactics/game/llm_bot.py:495`

Also: `reinforcetactics/game/llm_bot.py:436`, `reinforcetactics/game/llm_bot.py:1250`, `reinforcetactics/app/bot_factory.py:154`, `reinforcetactics/tournament/runner.py:355`

- **Impact.** An API outage, expired key, removed model or missing package looks like a passive opponent. Tournament results count infrastructure failures as model losses, and GUI users get no feedback.
- **Fix.** Sort errors into non-retryable (ImportError, auth, permission, not-found, bad-request) and retryable (429, 5xx, timeouts, connection). Fail fast on the first kind with a typed LLMBotError. Import and build the SDK client in __init__ so a missing dependency raises there. Track consecutive failed turns and let the runner mark the game `error`/forfeit instead of a normal loss. Add a jittered backoff that honors Retry-After.

### `aibots-5` — LLM (and AlphaZero) turns run synchronously on the pygame main loop with no timeout, so the window can freeze for minutes

**medium (reviewer: high)** · ux · partially · effort M · `reinforcetactics/app/input_handler.py:402`

Also: `reinforcetactics/app/input_handler.py:393`, `reinforcetactics/game/llm_bot.py:1255`, `reinforcetactics/game/llm_bot.py:1324`, `reinforcetactics/game/alphazero_bot.py:132`

Same issue as: `pygame-6`, `anim-2`, `pygame-11`

- **Impact.** Each LLM turn freezes the GUI, and the OS shows 'Not Responding' after a few seconds. A network stall can hang the app for a very long time, and the user can't cancel or save.
- **Fix.** Split LLMBot into plan_turn() (serialize and network call on a worker thread; it only reads state) and apply_turn(actions) (runs on the main thread). Have GameSession poll a Future while rendering a 'thinking...' overlay with a cancel option. Set explicit client timeouts (e.g. 60-120s) and max_retries, and make the outer retry budget time-bounded.
- **Verifier note.** The LLM part is real. input_handler._process_bot_turns (393-405) calls current_bot.take_turn() synchronously from the SPACE key / End Turn click handlers (lines 125, 161), for up to num_players*2 consecutive bot turns. There is no threading anywhere in app/ui/game, and no event pumping or rendering in between. OpenAI and Anthropic clients are built with no timeout (llm_bot.py:1255, 1324).

### `aibots-6` — Stateful mode keeps unbounded history and overflows the context window by about turn 20-30; positional unit IDs go stale across turns

**medium** · bug · confirmed · effort S · `reinforcetactics/game/llm_bot.py:444`

Also: `reinforcetactics/game/llm_bot.py:481`, `reinforcetactics/game/llm_bot.py:1012`

- **Impact.** Long stateful games start failing with context-length errors, which the retry logic turns into passed turns. Cost grows quadratically, and the history misleads the model about unit identity.
- **Fix.** Keep a bounded window: the last N exchanges, with older turns' state JSON replaced by a short summary or just the action list. Use the persistent `unit.unit_id` as the LLM-facing ID. Count tokens before sending and trim when over a budget.

### `aibots-8` — LLM prompts and serializer contradict engine constants (buff %, cooldowns, HQ income, Cleric/Archer ranges)

**medium** · bug · confirmed · effort M · `reinforcetactics/game/llm_prompts.py:56`

Also: `reinforcetactics/game/llm_prompts.py:280`, `reinforcetactics/game/llm_prompts.py:330`, `reinforcetactics/game/llm_prompts.py:599`, `reinforcetactics/game/llm_bot.py:633`, `reinforcetactics/game/llm_bot.py:757`

- **Impact.** The LLM plans against wrong rules (misjudging buff value, HQ importance, heal reach), which weakens results and makes LLM benchmarks harder to interpret, especially in balance sweeps with engine_overrides.
- **Fix.** Generate the unit and building rules section at runtime from game_state.unit_data, income_rates and the constants (a single render_rules(game_state) used by every prompt), and delete the copy-pasted blocks. Serialize income from game_state.income_rates. Compute move_then_heal with CLERIC_HEAL_RANGE. Add a test that checks prompt numbers against the constants.

### `aibots-9` — LLM bots can't use Sorcerer abilities: HASTE, DEFENCE_BUFF and ATTACK_BUFF are advertised but never serialized or executed

**medium** · bug · confirmed · effort S · `reinforcetactics/game/llm_bot.py:785`

Also: `reinforcetactics/game/llm_bot.py:957`, `reinforcetactics/game/llm_prompts.py:74`

Same issue as: `consolidate-2`

- **Impact.** A 350-gold unit loses its main purpose for every LLM player, and the prompt tells the model it can do things the executor silently drops.
- **Fix.** Serialize the three lists (unit_id and target_position), add them to the response schema, and route them through game_state.haste/defence_buff/attack_buff after the legality check from finding 1. Or remove them from the prompts until they're supported.

### `aibots-10` — AlphaZeroBot ignores the saved architecture and doesn't validate the grid; it isn't registered in the GUI, tournaments or the bot registry

**medium** · bug · confirmed · effort S · `reinforcetactics/game/alphazero_bot.py:99`

Also: `reinforcetactics/game/alphazero_bot.py:106`, `reinforcetactics/game/alphazero_bot.py:132`, `reinforcetactics/rl/alphazero_trainer.py:574`, `reinforcetactics/tournament/bots.py:414`, `reinforcetactics/game/bot_registry.py:66`

Same issue as: `rlalt-14`, `consolidate-16`, `rulebots-23`

- **Impact.** Non-default-architecture checkpoints can't be deployed, grid mismatches crash mid-game, and the docstring's claim of tournament/GUI integration isn't true.
- **Fix.** Pass num_res_blocks/channels from the config. Raise ValueError at load when (grid_width, grid_height) != live dims. Catch exceptions in take_turn and end the turn. Add BotType.ALPHAZERO plus a 'alphazerobot' entry in _RL_TYPES and a GUI option. Make discover_model_bots tell AlphaZero checkpoints apart by the presence of the 'model_state_dict' + 'config' keys.

### `aibots-11` — Tournament model discovery tests every checkpoint on a hard-coded 6x6 map and silently drops valid MultiDiscrete, feudal and FoW checkpoints

**medium** · bug · partially · effort S · `reinforcetactics/tournament/bots.py:511`

Also: `reinforcetactics/tournament/runner.py:330`, `reinforcetactics/game/model_bot.py:213`

Same issue as: `consolidate-8`

- **Impact.** Tournament rosters are missing valid RL agents without clear notice. A checkpoint that passes discovery can still raise at game setup for a mismatched map, outside the per-game error handling.
- **Fix.** Validate each checkpoint against the tournament's actual map pool and FoW setting, and record which maps each checkpoint supports so the scheduler skips unsupported pairings. Resolve the map path relative to the package or config, and move bot creation inside the runner's try.
- **Verifier note.** The core bug is real, but parts of the finding are wrong or overstated. - Confirmed: _test_model_file loads the cwd-relative 'maps/1v1/beginner.csv' (6x6) and builds a non-FoW GameState. MultiDiscrete checkpoints from other grid sizes and feudal checkpoints are dropped with only a logger.warning. From any other working directory, load_map returns None and every model is dropped. - Overstated: 'any checkpoint trained on a non-6x6 map fails' is false for flat_discrete checkpoints.

### `aibots-12` — ModelBot's deterministic MultiDiscrete decoding often picks illegal combinations; the first invalid action ends the turn, and failed moves count as valid

**medium** · rl-correctness · partially · effort M · `reinforcetactics/game/model_bot.py:660`

Also: `reinforcetactics/game/model_bot.py:342`, `reinforcetactics/game/model_bot.py:653`, `reinforcetactics/rl/gym_env.py:99`

- **Impact.** Deployed MultiDiscrete policies play far fewer actions per turn than they could, so tournament and GUI results understate the policy's strength. Invalid moves waste up to 50 predict calls per turn.
- **Fix.** At inference, enumerate the exact legal set with build_flat_actions and pick the legal action with the highest joint log-probability under model.policy.get_distribution(obs) (sum of per-dimension log-probs). At minimum, fall back to the best legal action instead of ending the turn. Return move_unit's bool.
- **Verifier note.** The mechanism is real; 'often' is not measured and the scope is narrower than stated. - Confirmed: in multi_discrete mode, _predict_sb3 passes the concatenated per-dimension union masks from build_per_dim_masks (gym_env.py:98-165) to a deterministic predict. The six dimensions are masked independently, so a combination of them can be illegal. - Confirmed: take_turn breaks on the first invalid action (l.342-344). The training env does not end the turn on an invalid action;

### `aibots-13` — ModelBot reimplements the env's action executor with differences; _compute_action_mask is dead code duplicating ACTION_KEY_MAP

**medium** · consolidation · confirmed · effort M · `reinforcetactics/game/model_bot.py:563`

Also: `reinforcetactics/game/model_bot.py:467`, `reinforcetactics/rl/gym_env.py:981`, `reinforcetactics/rl/self_play.py:437`

Same issue as: `core-14`, `consolidate-9`, `prior-14`, `pygame-18`, `rulebots-20`

- **Impact.** Env and ModelBot can disagree on what a decoded action does, which is the kind of train/deploy gap the shared build_* helpers were meant to remove, and there is about 220 lines of extra code to maintain.
- **Fix.** Move execute_game_action into a module-level function execute_game_action(game_state, action_dict, player) next to build_per_dim_masks, used by StrategyGameEnv, SelfPlayEnv and ModelBot (ModelBot keeps only the 6-vector to action_dict encoding). Delete _compute_action_mask and NUM_ACTION_TYPES.

### `aibots-15` — Truncated or empty LLM responses lose the whole turn with no retry or partial recovery (max_tokens, reasoning-token exhaustion, refusals)

**medium** · bug · confirmed · effort S · `reinforcetactics/game/llm_bot.py:1279`

Also: `reinforcetactics/game/llm_bot.py:438`, `reinforcetactics/game/llm_bot.py:986`

- **Impact.** Turns are silently skipped exactly when the state is complex and the model reasons most, and the cause is invisible unless conversation logging is on.
- **Fix.** Check the stop/finish reason. On length or max_tokens, log a warning and retry once with a larger budget or lower reasoning effort, or salvage the complete action objects from the truncated `actions` array. Treat None content or a refusal as a classified error rather than an empty string.

#### Low and info findings (11)

| ID | Sev | Title | Location | Effort | Fix (summary) |
|---|---|---|---|---|---|
| `aibots-7` | low | LLM prompts use too many tokens: indent=2 JSON of every legal move plus move_then combos, no prompt caching, full state resent in two-phase planning | `reinforcetactics/game/llm_bot.py:905` | M | Use `separators=(",", ":")`. Group moves per unit as `reachable: [[x,y],...]` instead of repeating `from` in every entry. Emit move_then targets as per-unit sets rather than one entry per destination and enemy. |
| `aibots-14` | low | OpenAI/Anthropic clients are created per call with no timeout or connection reuse; no rate-limit coordination across concurrent tournament games | `reinforcetactics/game/llm_bot.py:1255` | S | Move _get_client() into LLMBot with a provider factory, cache the client per bot or process, and pass explicit timeout and max_retries. |
| `aibots-16` | low | Token accounting double-counts stale usage and has no cost or cache breakdown | `reinforcetactics/game/llm_bot.py:490` | S | Reset the _last_* fields at the start of _call_llm, return usage as a small dataclass from each provider call, record cache_read/creation and reasoning tokens, and add a per-model price table to report estimated cost per game. |
| `aibots-17` ≈ consolidate-19 | low | LLM provider-to-class mapping and default model names are duplicated in three places; _test_anthropic_key never checks the key | `reinforcetactics/tournament/bots.py:470` | S | Add `LLM_PROVIDERS = {"openai": OpenAIBot, "anthropic": ClaudeBot, "google": GeminiBot}` and `build_llm_bot(provider, game_state, **kw)` in llm_bot.py (mirroring bot_registry.build_scripted), with a `validate_key()` classmethod per provider (Anthropic: … |
| `aibots-18` ≈ core-19, persist-22, anim-8, anim-18, consolidate-11 | low | Padding metadata is never set in production, so LLMBot's padded/original coordinate conversion does nothing in the GUI | `reinforcetactics/game/llm_bot.py:581` | S | Have game_loop use FileIO.load_map_with_metadata and call set_map_metadata with offsets that include border_size, which fixes the LLM prompt coordinates and enables ModelBot cropping. Otherwise remove the conversion layer. |
| `aibots-19` | low | ModelBot uses default observation scales and a hard 50-action cap; SB3 checkpoints don't carry their env config | `reinforcetactics/game/model_bot.py:460` | M | Save an env-config sidecar (`<ckpt>.env.json` from bootstrap's env-kwargs helper, or embed it in the SB3 zip) and have ModelBot read it for scales, max_actions_per_turn, enabled_units and FoW, warning when the live game differs. |
| `aibots-20` | low | Dead code and aliases in the bot layer | `reinforcetactics/game/llm_bot.py:70` | S | Delete them, or wire the dynamic-prompt helpers into _get_effective_system_prompt as part of the rules-from-constants refactor. |
| `aibots-21` | low | Conversation logger re-reads and rewrites the whole JSON file every turn | `reinforcetactics/game/llm_bot.py:352` | S | Write JSONL: a header line followed by one appended line per turn. Or write to a temp file and replace it atomically. |
| `aibots-22` | low | Feudal self-play reloads the opponent checkpoint from disk on every env reset via ModelBot | `reinforcetactics/game/model_bot.py:65` | S | Add `ModelBot.from_loaded(game_state, player, model=..., feudal_agent=...)` or a module-level LRU cache keyed on (path, mtime), plus a `rebind(game_state)` method, so the factory reuses loaded snapshots. |
| `aibots-23` | low | examples/llm_bot_demo.py points to a missing map and ends turns twice | `examples/llm_bot_demo.py:57` | S | Use maps/1v1/beginner.csv, call end_turn only for the non-bot player, and build the bot through the shared provider registry. |
| `aibots-24` | info | Opportunity: schema-constrained LLM output plus a bounded repair loop for rejected actions | `reinforcetactics/game/llm_bot.py:926` | M | Define one JSON Schema for the action list (enum of types, integer unit_id, 2-int coordinate arrays) and pass it through each provider's structured-output feature (OpenAI json_schema strict, Anthropic output_config.format, Gemini response_schema). |

## rlenv RL environment (Gymnasium)

### `rlenv-1` — GameState.end_turn never invalidates the legal-action cache, so masks and cache-reading opponents act on stale pre-end_turn state

**critical** · bug · confirmed · effort S · `reinforcetactics/core/game_state.py:1192`

Also: `reinforcetactics/core/game_state.py:1305`, `reinforcetactics/rl/gym_env.py:1197`, `reinforcetactics/game/bot.py:82`

Same issue as: `core-1`, `prior-1`

- **Impact.** The noop sanity stage is unplayable because the agent can never act after turn 1. In the 11 `opponent: random` stages of configs/ppo/bootstrap.yaml, RandomBot stops playing on any turn in which the agent changed nothing. A passive or end-turn-spamming policy is therefore rewarded with a frozen opponent (a draw instead of a loss) while that opponent banks gold. This is exactly the attractor the curriculum fights. Any other caller that reads get_legal_actions after a no-change opponent turn (LLM bot, UI) is also affected.
- **Fix.** Call self._invalidate_cache() in GameState.end_turn after the unit flag reset, income and healing. As a defence in the env, also invalidate after `_opponent_turn()` and the safety-net end_turn (gym_env.py:1197-1202). Add a regression test: vs noop, the turn-2 mask must contain moves and creates. Separately, re-baseline any results from random-opponent stages.

### `rlenv-2` — execute_game_action runs illegal actions as valid: create anywhere, out-of-range 1-dmg attacks, repeat attacks/seizes by exhausted or paralyzed units

**high (reviewer: critical)** · rl-correctness · confirmed · effort M · `reinforcetactics/rl/gym_env.py:1006`

Also: `reinforcetactics/rl/gym_env.py:1018`, `reinforcetactics/rl/gym_env.py:1028`, `reinforcetactics/core/game_state.py:651`, `reinforcetactics/core/game_state.py:787`, `reinforcetactics/core/mechanics.py:223`

Same issue as: `core-2`, `aibots-1`

- **Impact.** multi_discrete is the env default, the make_maskable_env default, used by 6 configs (maskable_ppo, ppo_baseline with vanilla unmasked PPO, v33, v48, skirmish_bc_selfplay, feudal) and by the CLI trainer. Per-dim masks are a union over-approximation, so these exploit paths can be reached and paid for. A policy can learn to spawn at the enemy HQ, snipe across the map, and farm kills or captures several times per turn. The damage lands in the game state, not just in the reward.
- **Fix.** In execute_game_action, build the canonical key (atype, ut, fx, fy, tx, ty) as in build_flat_actions and reject anything outside a legal-key set. Build that set once per legal-action cache version so each step costs an O(1) lookup. Also harden the engine: create_unit should require a player-owned empty building and an enabled type; attack, seize and the abilities should require can_attack, not paralyzed, and a positive range damage.
- **Verifier note.** Mechanism confirmed. GameState.create_unit (651-704) checks only the unit cap, occupancy, a known type and gold. There is no building-ownership check and no enabled_units check. GameState.attack (787+) checks neither can_attack, paralysis nor range. mechanics.apply_defence_reduction returns `max(1, int(reduced_damage))` (223), so an out-of-range hit (get_attack_damage returns 0) still deals 1. Seize has no can_attack check; the env only checks ownership.

### `rlenv-12` — ActionMaskedEnv.__getattr__ breaks pickle/deepcopy, and without __setattr__ writes land on the wrapper

**high (reviewer: medium)** · bug · confirmed · effort S · `reinforcetactics/rl/masking.py:61`

Also: `reinforcetactics/rl/self_play.py:542`

Prior review: REVIEW_rl_pipeline_2026-07-24.md §2.13(b) (agent_player write still lands on wrapper)

Same issue as: `prior-2`, `rltrain-1`, `rltrain-2`, `critic-gaps-1`, `tests-3`, `consolidate-4`, `rlenv-4`

- **Impact.** The wrapped env cannot be copied or serialized. On every swapped self-play episode the base env scores rewards, masks and the potential for the wrong seat.
- **Fix.** Guard it: `env = self.__dict__.get('env'); if env is None or name.startswith('__'): raise AttributeError(name)`. Set seat state through env.unwrapped (or add a set_agent_player method), or drop the custom __getattr__ in favour of get_wrapper_attr.
- **Verifier note.** Both mechanisms confirmed, and the self-play impact is understated. (1) masking.py:61 has an unguarded `__getattr__` that returns getattr(self.env, name). Under gymnasium 1.3.0, pickle and deepcopy of ActionMaskedEnv both hit RecursionError; the bare StrategyGameEnv pickles fine. Nothing in the repo pickles or deepcopies an ActionMaskedEnv instance (SubprocVecEnv pickles env_fns), so this part is latent.

### `rlenv-3` — step() throughput is dominated by the opponent's O(units) pathfinding scan (64-81% of wall time)

**medium (reviewer: high)** · performance · partially · effort M · `reinforcetactics/core/mechanics.py:58`

Also: `reinforcetactics/core/unit.py:159`, `reinforcetactics/core/game_state.py:1343`, `reinforcetactics/core/game_state.py:741`, `reinforcetactics/rl/gym_env.py:1197`

Same issue as: `core-20`, `prior-17`

- **Impact.** Simulation, not the env or the network, sets rollout throughput for every PPO run. Beginner and skirmish episodes run 800-3000 steps.
- **Fix.** Build an occupancy map `{(x, y): unit}` once per get_legal_actions or bot-turn call and pass an O(1) closure. In move_unit, skip the BFS re-validation when the move came from a legal list. Cache reachable sets per unit per cache version. Let RandomBot update its legal list incrementally instead of recomputing after each action. A 2-4x speedup on opponent turns is plausible.
- **Verifier note.** The hotspot is real, but its size is overstated and it is an optimisation opportunity, not a defect. can_move_to_position scans `for unit in units:` (line 58, not 59) on every BFS probe. It is the top tottime entry. Callers are get_legal_actions (env masks and bots), move_unit's BFS re-validation (game_state.py:741-743) and bot_base.py:232. RandomBot does recompute get_legal_actions after each mutation (bot.py:89).

### `rlenv-7` — Render modes are broken: rgb_array returns None; human mode never flips or pumps; renderer rebuilt every reset

**medium** · bug · confirmed · effort S · `reinforcetactics/rl/gym_env.py:1677`

Also: `reinforcetactics/rl/gym_env.py:684`, `reinforcetactics/rl/gym_env.py:1659`, `reinforcetactics/ui/renderer.py:395`

Same issue as: `tests-7`

- **Impact.** `scripts/eval_agent.py --render`, the CLI and examples/train_with_action_masking.py show a window that never updates. RecordVideo and other rgb_array consumers get None.
- **Fix.** For rgb_array, construct Renderer(game_state, headless=True, viewing_player=self.agent_player) and return renderer.render() followed by get_rgb_array(). For human, call pygame.event.pump() and display.flip(), and render from step/reset per the Gymnasium convention. On reset, rebind renderer.game_state instead of rebuilding the renderer.

### `rlenv-8` — structured_action_masks() ignores max_actions_per_turn

**medium** · bug · confirmed · effort S · `reinforcetactics/rl/gym_env.py:849`

Also: `reinforcetactics/rl/feudal_rl.py:1383`

- **Impact.** The feudal autoregressive worker (feudal_rl.py:1383) samples from structured masks, and train_feudal_rl forwards max_actions_per_turn. The 'never end the turn' safety net is therefore silently off in AR mode.
- **Fix.** Apply the same budget gate in _build_structured_masks: atype=[5] only, source[5,0,0], and target[(5,0,0)].

### `rlenv-9` — Combat shaping ignores counter-damage and attacker death; damage_taken is netted against start-of-turn healing

**medium** · rl-correctness · confirmed · effort S · `reinforcetactics/rl/gym_env.py:1138`

- **Impact.** damage_taken_scale: -0.002 is set in 9 configs to make combat shaping 'net-zero-sum' (comment at 530-536). That property does not hold: an agent-initiated trade that loses more HP to the counter than it deals still nets positive, and suicide attacks cost nothing except the unit_diff potential.
- **Fix.** Forward counter_damage and attacker_alive. Charge counter_damage * damage_taken_scale on the attack step. Snapshot post_hp before the agent's turn-start healing, or subtract the healing_totals delta. Optionally add a symmetric 'unit_lost' term.

### `rlenv-10` — Unknown opponent strings silently produce an env with no opponent

**medium** · bug · confirmed · effort S · `reinforcetactics/rl/gym_env.py:1655`

Also: `reinforcetactics/rl/gym_env.py:71`, `reinforcetactics/rl/config.py:200`

Same issue as: `rulebots-7`, `rltrain-22`

- **Impact.** A typo in a script or notebook trains against nothing, with no error.
- **Fix.** In __init__, accept only None, 'self', or anything bot_registry.canonical_name resolves, and raise ValueError otherwise. Derive _BOT_OPPONENT_TYPES and the config/CLI lists from the registry.

### `rlenv-11` — flat_discrete truncation still drops attacks, heals and casts before moves; diagnostics cannot see truncation

**medium** · rl-correctness · confirmed · effort S · `reinforcetactics/rl/gym_env.py:306`

Also: `reinforcetactics/rl/gym_env.py:1507`, `reinforcetactics/rl/gym_env.py:288`

Prior review: REVIEW_ppo_training.md §2.6 (seize/end_turn now protected; attacks/casts still dropped first)

Same issue as: `prior-13`

- **Impact.** The largest armies lose their combat actions first. The 'guardrail' metric cannot detect this, and truncated states flood the worker logs.
- **Fix.** Also protect attack, heal, cure and cast types, and truncate moves first (e.g. round-robin per unit). Record the pre-truncation count and a truncated flag in info and episode_stats. Rate-limit the warning. Compute the diagnostics from the same pre-step legal set in both modes.

### `rlenv-14` — Fog of war: shrouded tiles leak live ownership/HP, unexplored cells encode as plains, and the extractor ignores visibility

**medium** · rl-correctness · confirmed · effort M · `reinforcetactics/core/game_state.py:1536`

Also: `reinforcetactics/core/game_state.py:1545`, `reinforcetactics/rl/extractors.py:199`

Same issue as: `core-5`, `consolidate-1`, `critic-integration-3`, `pygame-12`

- **Impact.** FoW is a supported flag, although no config enables it yet. Training with it would leak hidden information, and the policy could not tell unexplored cells from plains.
- **Fix.** For shrouded cells, emit the last-seen owner and HP from the visibility map. Encode unexplored cells as an all-zero tile one-hot or a dedicated 'unknown' channel. Feed a 3-channel visibility one-hot into the extractor when the key is present.

#### Low and info findings (12)

| ID | Sev | Title | Location | Effort | Fix (summary) |
|---|---|---|---|---|---|
| `rlenv-4` ≈ prior-2, rltrain-1, rltrain-2, critic-gaps-1, rlenv-12, tests-3, consolidate-4 | low | With opponent=None (SelfPlayEnv) the env skips all opponent-turn accounting, and the wrapper re-implements terminals incompletely | `reinforcetactics/rl/self_play.py:497` | M | Have SelfPlayEnv use the env's own `opponent='self'` plus `set_self_play_opponent_factory` path, which already runs the opponent inside step() with full accounting. |
| `rlenv-5` | low | end_reason is inferred from board state and mislabels HQ-capture wins as elimination (wrong terminal reward) | `reinforcetactics/rl/gym_env.py:1439` | S | Use `end_reason = self.game_state.end_reason` and keep the heuristic only as a fallback when it is None. |
| `rlenv-6` | low | flat_discrete step() decodes indices against whatever list the last action_masks() call built; out-of-range becomes a free end_turn | `reinforcetactics/rl/gym_env.py:1381` | S | Record the step and cache version at which _current_actions was built and rebuild inside step() when stale. Treat out-of-range indices as an invalid no-op with the invalid_action penalty (or raise), not as end_turn. Clear the list in reset(). |
| `rlenv-13` | low | Purchase exploration stores log pi(a) for actions drawn from a mixture policy (biased PPO gradient) | `reinforcetactics/rl/purchase_exploration.py:249` | S | For every create row, store log mu: log pi(a) - log pi(ut) + log((1-eps)*pi(ut) + eps/\|L\|). The PPO ratio pi_theta/mu is then the proper importance weight. Otherwise, document the bias explicitly and keep eps annealed to 0. |
| `rlenv-15` | low | Observation has no deadline features (turns remaining, per-turn action budget) | `reinforcetactics/rl/observation.py:230` | S | Add turns_remaining_frac = 1 - turn/max_turns (1.0 when unlimited), plus optionally actions_this_turn/max_actions_per_turn, as global features. Put this behind a flag or version bump because it changes GLOBAL_FEATURES_DIM and breaks checkpoints. |
| `rlenv-16` | low | reward_config keys are not validated, and rc.get fallbacks contradict the real defaults | `reinforcetactics/rl/gym_env.py:558` | S | Define a KNOWN_REWARD_KEYS frozenset that includes the dynamic *_capture and win_by_* keys, and raise or warn on unknown keys. Index rc[key] directly instead of using divergent fallbacks. |
| `rlenv-17` | low | Shaping gamma must match the trainer's by hand; feudal and self-play scripts never forward it | `scripts/train/train_feudal_rl.py:48` | S | Forward gamma in both scripts, and add a one-time check at learn start that model.gamma equals env.get_attr('gamma'), warning otherwise. |
| `rlenv-18` | low | validate_action_mask is broken, and ActionMaskedEnv carries dead stats | `reinforcetactics/rl/masking.py:436` | S | Use ACTION_KEY_MAP to map types to keys, pass agent_player, and branch on action_space_type. Remove the dead stat and no-op override, and use importlib.util.find_spec for the install check. |
| `rlenv-19` | low | Dead or fake API surface: hierarchical action space, flat-mask accessors, duplicate unit list | `reinforcetactics/rl/gym_env.py:642` | S | Delete hierarchical and goal_space_size (feudal has its own design), plus the unused mask accessors. Use ALL_UNIT_TYPES in _encode_action and fix the docstring. |
| `rlenv-20` | low | rl/__init__ eagerly imports torch and SB3 for numpy-only modules | `reinforcetactics/rl/__init__.py:12` | S | Replace the eager imports with a PEP 562 lazy `__getattr__` that maps exported names to submodules. |
| `rlenv-21` | low | map_file=None draws its random map from the global RNG in __init__ (not reproducible by reset(seed)) | `reinforcetactics/rl/gym_env.py:431` | S | Generate the map in reset() from self.np_random, or require map_file. |
| `rlenv-22` | info | Opportunity: register the env with Gymnasium and support a seeded map pool per reset | `reinforcetactics/rl/gym_env.py:1575` | M | Accept `map_file: str \| list[str]` and sample one per reset with self.np_random, with an `options={'map_file': ...}` override that requires pad_to_size. Add `gym.register('ReinforceTactics-v0', entry_point='reinforcetactics.rl.gym_env:StrategyGameEnv')`. |

## rltrain PPO bootstrap / self-play training pipeline

### `rltrain-1` — SelfPlayEnv is not self-play: the seat swap never reaches the game, the flat_discrete opponent executes the agent's action list, and agent wins are never counted

**high** · rl-correctness · confirmed · effort M · `reinforcetactics/rl/self_play.py:541-547`

Also: `reinforcetactics/rl/self_play.py:332`, `reinforcetactics/rl/self_play.py:420-435`, `reinforcetactics/rl/self_play.py:503-521`, `reinforcetactics/rl/self_play.py:317-340`, `reinforcetactics/rl/masking.py:61`

Prior review: REVIEW_rl_pipeline_2026-07-24.md §2.13 (a),(b); win accounting, flat_discrete opponent action list and swap cost measurement are new

Same issue as: `prior-2`, `rltrain-2`, `critic-gaps-1`, `rlenv-12`, `tests-3`, `consolidate-4`, `rlenv-4`

- **Impact.** Half of all episodes (swap_players defaults to True) score rewards, potential and terminals for the wrong seat. In flat_discrete the opponent is effectively a pass-bot. `_add_to_pool` gates on `min_win_rate_for_pool=0.55` against a win rate that can never get there, so `use_opponent_pool: true` in self_play.yaml leaves the pool empty. Any self-play training curve is meaningless, and nothing warns about it. This is not the production bootstrap path.
- **Fix.** (a) In SelfPlayEnv, set the seat on the base env: `self.env.unwrapped.agent_player = p`, and do it before `self.env.reset()` so `_prev_potential` is computed for the right seat. (b) Build the opponent's legal list with `build_flat_actions(game_state, opponent_player, max_flat_actions)`, then resolve and mask against that list. Pass `action_masks=` to `predict`. (c) Record game outcomes wherever `terminated or truncated` is true, including when the agent's own step ends the game. Charge the terminal shaping term and set `episode_stats['winner']` and `end_reason` when the game ends during the opponent's turn. (d) Seed from `self.env.unwrapped.np_random`.

### `rltrain-2` — train_self_play.py with default settings never wires an opponent, ignores the env config, and its 'mixed' mode does nothing

**high** · bug · confirmed · effort M · `scripts/train/train_self_play.py:149-164`

Also: `scripts/train/train_self_play.py:101-111`, `scripts/train/train_self_play.py:147-157`, `scripts/train/train_self_play.py:312-329`, `scripts/train/train_self_play.py:458`, `scripts/train/train_self_play.py:181-183`, `configs/self_play/self_play.yaml:6-9`

Prior review: REVIEW_rl_pipeline_2026-07-24.md §2.13 (c) (the same SubprocVecEnv discovery failure, now in the script-level helper); ignored config, dead mixed mode and CLI defects are new

Same issue as: `prior-2`, `rltrain-1`, `critic-gaps-1`, `rlenv-12`, `tests-3`, `consolidate-4`, `rlenv-4`

- **Impact.** The documented self-play entry point trains against a random or no-op opponent on a different game from the one the config describes, while its logs say 'self-play'. Mixed training is plain self-play on half the envs.
- **Fix.** Make the envs self-play from inside the worker, e.g. a `set_opponent_params` method called via `vec_env.env_method`, or register a ModelBot-based opponent factory through the base env's `opponent='self'` path. Or refuse to start with SubprocVecEnv until that exists. Pass through env.map_file, action_space_type, max_flat_actions, reward_config, max_turns and pad_to_size, ideally by building envs from `TrainingConfig` instead of argparse. Delete `train_mixed` and `MixedTrainingCallback` or implement them. Use `BooleanOptionalAction` for `--swap-players`. Convert `*_freq` values to per-call units with `max(1, freq // n_envs)`. Evaluate against a fixed bot through `evaluate_model`.

### `rltrain-3` — A Vertex bootstrap run loses its whole run directory on preemption or cancel

**high** · bug · confirmed · effort S · `scripts/train/train_bootstrap.py:417`

Also: `scripts/cloud/vertex_train.py:104-111`, `reinforcetactics/cloud/storage.py:24`, `docs/vertex_training.md:9-13`

Prior review: REVIEW_rl_pipeline_2026-07-24.md §3 (Preemption)

Same issue as: `prior-10`

- **Impact.** For preempted, cancelled or wall-clock-killed Vertex jobs, all checkpoints, eval JSONL and train_metrics.csv are lost, even though the per-eval persistence exists to prevent exactly this. The project's deepest runs have died to wall-clock limits.
- **Fix.** In `train_bootstrap.main`, add `signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))` so `finally` runs. Also let vertex_train sync extra directories (a `GCS_SYNC_DIRS` env var, or add `benchmarks` to the defaults) so periodic syncs cover the run directory during training. The doc then holds for both termination paths.

### `rltrain-4` — The promotion, best-model and stall gates still score the deterministic argmax policy, not the stochastic policy PPO trains

**medium (reviewer: high)** · rl-correctness · confirmed · effort S · `reinforcetactics/rl/callbacks.py:276`

Also: `reinforcetactics/rl/evaluation.py:97`, `reinforcetactics/rl/bootstrap.py:720-750`, `reinforcetactics/rl/config.py:467-493`

Prior review: REVIEW_rl_pipeline_2026-07-24.md §2.2(a), §4 item 3

Same issue as: `prior-5`, `rltrain-12`

- **Impact.** Every promotion, every `best_model.zip` and every stall verdict measures a policy the optimizer is not improving. Argmax over the positional flat_discrete action list can flip completely after a small weight change, which is the 1.00 -> 0.21 -> 0.99 swing the review documented. Now that the seed set is fixed, this is the main remaining source of gate noise.
- **Fix.** Add `deterministic: bool` to `PeriodicEvalCallback` and an `EvalConfig.eval_deterministic` field (default False for the gate), and pass it from bootstrap.py:720. Optionally run a second cheap eval with the other mode and log both as `win_rate_stochastic` and `win_rate_greedy` in the eval row and in TensorBoard. Correct the review doc's status line.
- **Verifier note.** The facts are correct. `PeriodicEvalCallback._do_eval` (callbacks.py:276-283) calls `evaluate_model` with no `deterministic` argument, and `evaluate_model` defaults to `deterministic=True` (evaluation.py:96). PeriodicEvalCallback.__init__ (lines 186-199) and EvalConfig (config.py:467-493) have no knob for it. The prior review's status line says items 1-5 are implemented, but the Landed table covers only the fixed-seed half of item 3; the 'gate on stochastic or report both' half is missing.

### `rltrain-5` — A crashed or killed curriculum cannot resume: no start-stage option, no run manifest, no mid-stage checkpoint

**medium (reviewer: high)** · design · confirmed · effort M · `reinforcetactics/rl/bootstrap.py:624`

Also: `reinforcetactics/rl/bootstrap.py:543-551`, `reinforcetactics/rl/config.py:473`, `reinforcetactics/rl/callbacks.py:298-304`

Prior review: REVIEW_rl_pipeline_2026-07-24.md §2.9 fix 3, §4 item 10, §5 'Rec #2 not landed'

Same issue as: `rltrain-7`, `prior-3`

- **Impact.** The 33-stage ladder needs about 108M steps against about 6M per session. A kill at 4.9M of a 5M-step stage loses everything since the last best-eval improvement. The only workaround is hand-editing the YAML to delete finished stages and pointing `warm_start_path` at a checkpoint, which resets `num_timesteps` and the TensorBoard continuity.
- **Fix.** Add `run_curriculum(..., resume: bool)` and a `--resume <run_dir>` flag in train_bootstrap. Read `run_status.json` and each stage's `config.json` (`extra.promoted`) to skip finished stages, then load the last promoted stage's best or `stage_final.zip` with `MaskablePPO.load(..., env=vec_env)` so `num_timesteps` and the optimizer state are kept. Add a rolling `latest.zip` saved every `eval.checkpoint_freq` steps inside a stage, and resume a partial stage from it with the remaining budget.
- **Verifier note.** Verified. `run_curriculum` always iterates `for stage in cfg.curriculum.stages:` from the first stage (bootstrap.py:624), and train_bootstrap has no resume or start-stage flag (the parser at lines 52-86 has none). The only checkpoints are best_model.zip (on an eligible improvement, callbacks.py:366-373), stage_final.zip (at stage end, bootstrap.py:834-835) and final_model.zip (on stall or completion). `EvalConfig.checkpoint_freq` (config.py:473) is never read anywhere in the bootstrap path;

### `rltrain-7` — A stall ends the whole run: no retry from the stage's best checkpoint, no within-stage regression guard, a misleading message and exit code 0

**medium** · design · confirmed · effort M · `reinforcetactics/rl/bootstrap.py:915`

Also: `reinforcetactics/rl/bootstrap.py:102-106`, `reinforcetactics/rl/bootstrap.py:978-994`, `scripts/train/train_bootstrap.py:404-420`

Prior review: REVIEW_rl_pipeline_2026-07-24.md §2.9 fixes 1-2, §3 (message; exit code), §4 items 9-10 and 13

Same issue as: `rltrain-5`, `prior-3`

- **Impact.** The review found 28/41 archived stalls peaked at or above the gate before collapsing. The recovery checkpoint is on disk, but nothing loads it. CI and Vertex record stalled runs as successful.
- **Fix.** On stall, if `best_model.zip` exists and `stage.max_retries` (new field, default 1) allows, call `model.set_parameters(best)`, re-apply the stage's entropy start, and rerun `learn` with a fresh callback set before raising. Add a within-stage guard: after N evals more than X below the stage best, restore best and continue. Word the message differently when `achieved >= threshold` ('peaked at X but never held it for patience=N'). Return 3 from train_bootstrap on stall.

### `rltrain-9` — Known config keys are silently ignored by their consumers, and there are three different n_eval_episodes defaults

**medium** · design · partially · effort S · `reinforcetactics/rl/bootstrap.py:723`

Also: `reinforcetactics/rl/config.py:255`, `reinforcetactics/rl/config.py:472-473`, `reinforcetactics/rl/config.py:733-757`, `scripts/train/train_self_play.py:408-439`, `reinforcetactics/rl/bootstrap.py:1036-1078`

Prior review: REVIEW_rl_pipeline_2026-07-24.md §3 (cfg.eval.n_eval_episodes never read); the other ignored keys are new

- **Impact.** Setting a value in YAML does not guarantee it takes effect. A new bootstrap stage that omits `n_eval_episodes` silently runs 30 episodes while the config says 80. Every stage in bootstrap.yaml repeats `n_eval_episodes: 80` to work around this.
- **Fix.** Make `CurriculumStage.n_eval_episodes: int | None = None` and resolve it against `cfg.eval.n_eval_episodes`. Add a `consumed_fields` declaration per entry point and a helper that warns, or errors under `--strict`, when a loaded config sets a non-default value for a field the entry point does not consume. Either forward `fog_of_war` through `_stage_env_kwargs` or reject it for the bootstrap path.
- **Verifier note.** Accurate: bootstrap.py:723 passes `n_eval_episodes=stage.n_eval_episodes` (stage default 30, config.py:255) and never reads cfg.eval.n_eval_episodes (default 10, config.py:472). bootstrap.yaml's own comment at 209-218 says the stage value 'overrides this default', which is false, and all 33 stages repeat `n_eval_episodes: 80`. bootstrap.py never reads eval.checkpoint_freq, cfg.logging.*, cfg.algorithm, ppo.use_action_masking or ppo.lr_schedule. tensorboard_log is hard-coded at bootstrap.py:272.

### `rltrain-10` — Config values are neither type-checked nor range-checked: '3e-4' loads as a string, eval_freq=0 passes, reward_config keys are unchecked

**medium** · bug · confirmed · effort M · `reinforcetactics/rl/config.py:624`

Also: `reinforcetactics/rl/config.py:534-562`, `reinforcetactics/rl/config.py:692-714`, `reinforcetactics/rl/gym_env.py:558-559`, `reinforcetactics/rl/bootstrap.py:670-674`, `reinforcetactics/rl/bootstrap.py:716-717`

Prior review: REVIEW_rl_pipeline_2026-07-24.md §3 (purchase-exploration hook no-op)

- **Impact.** Typos and YAML float quirks either fail late with cryptic SB3 errors or, worse, silently change the reward function by orders of magnitude.
- **Fix.** Coerce each field with `typing.get_type_hints(cls)` in `_build_section` (float/int/bool/tuple), and range-check eval_freq > 0, n_eval_episodes > 0, max_flat_actions > 0, len(pad_to_size) == 2, lr > 0 and 0 <= clip_range. Export `KNOWN_REWARD_KEYS` from gym_env (defaults plus the `{tower,building,hq}_capture`, `win_by_*`, `truncation` and `damage_taken_scale` keys) and reject unknown keys in `TrainingConfig.validate` and `CurriculumStage.validate`. Raise when any stage sets `purchase_explore_eps > 0` with `action_space_type == 'flat_discrete'`. Check that `warm_start_path` exists inside `validate()` rather than after the envs are built.

### `rltrain-11` — Eval runs one environment at a time in the main process, calling the model once per step, and probably rivals training for wall-clock time

**medium** · performance · partially · effort M · `reinforcetactics/rl/evaluation.py:215`

Also: `reinforcetactics/rl/evaluation.py:223`, `reinforcetactics/rl/callbacks.py:253-283`

Prior review: REVIEW_rl_pipeline_2026-07-24.md §3 (eval sequential; trace buffer for every episode)

- **Impact.** Wall-clock death is one of the two ways runs end, and eval time is spent between every 3 PPO updates. The per-step trace dicts add allocation work across 160k steps per eval.
- **Fix.** Add a vectorised `evaluate_model_vec(model, make_env_fn, n_envs, seeds)`: a DummyVecEnv or SubprocVecEnv of K stage envs, each consuming a queue of fixed episode seeds, with batched `predict(obs, action_masks=vec.env_method('action_masks'))`. Keep identical per-episode seeding so results match the serial path, and assert that equivalence in a test. Only buffer traces after the episode passes `max_steps - N`, or use a ring buffer.
- **Verifier note.** The core claim holds. evaluate_model (evaluation.py:215-240) runs a strictly sequential loop over one env with one model.predict per step, and PeriodicEvalCallback._do_eval calls it synchronously from _on_step (callbacks.py:253-283) while the SubprocVecEnv workers sit idle. Inference dominates: on this 4-vCPU box, env-only stepping with random legal actions ran at 1,307 steps/s and model+env at 442 steps/s, with predict taking about 60% of wall time. The reviewer measured 76%.

### `rltrain-14` — The canonical configs/ppo/bootstrap.yaml still ships the reward terms and guards the review isolated as causes of the wall

**medium** · rl-correctness · confirmed · effort S · `configs/ppo/bootstrap.yaml:157`

Also: `configs/ppo/bootstrap.yaml:32`, `configs/ppo/bootstrap.yaml:49`, `configs/ppo/bootstrap.yaml:80-91`, `configs/ppo/bootstrap.yaml:149`, `configs/ppo/bootstrap.yaml:171`, `configs/ppo/bootstrap_validation.yaml:56-60`

Prior review: REVIEW_rl_pipeline_2026-07-24.md §3 (bootstrap.yaml back-port), §2.10, §4 item 12

Same issue as: `prior-4`

- **Impact.** The default config, which the README, Vertex doc and notebook point to, reproduces known failure modes, and its comments steer tuning the wrong way.
- **Fix.** Back-port the v52a/v54 reward values. Set `max_actions_per_turn` (e.g. 25-60). Scale `max_steps` per stage, either as a stage override of about max_turns*max_actions_per_turn*1.5 or as a derived default in `CurriculumStage.resolve_max_steps`. Rewrite the shaping comment to refer to review §2.1.

### `rltrain-17` — --build-bc only accepts multi_discrete, the action space that has never learned in the archive

**medium** · design · confirmed · effort L · `reinforcetactics/rl/imitation.py:25`

Prior review: REVIEW_rl_pipeline_2026-07-24.md §2.12, §4 item 14

Same issue as: `rlalt-21`, `prior-12`

- **Impact.** Warm-starting the production flat_discrete curriculum from bot demonstrations is impossible. The BC path is wired to an action space with measured peak WR 0.00 across three runs, according to the prior review.
- **Fix.** Port BC labelling to flat_discrete: the label is the index of the demonstrated action in `build_flat_actions(game_state, player, max_flat_actions)`, the same function ModelBot uses. Build the template env through `make_stage_env(first_stage, cfg.env, seed=..., gamma=cfg.ppo.gamma)`. If the port is not planned, delete the gate and the three BC configs.

#### Low and info findings (12)

| ID | Sev | Title | Location | Effort | Fix (summary) |
|---|---|---|---|---|---|
| `rltrain-6` | low | A stage can promote on evals that cannot claim best_model.zip, leaving best_win_rate at -1.0 in the run record for a promoted stage | `reinforcetactics/rl/callbacks.py:456-463` | S | Track `peak_win_rate` over all evals separately from `best_win_rate`, and record `None` instead of -1.0 when no eligible eval exists. |
| `rltrain-8` ≈ prior-6 | low | ppo.lr_schedule is silently dropped, and a native SB3 schedule would anneal wrongly under reset_num_timesteps=False | `reinforcetactics/rl/config.py:137` | S | Add `CurriculumStage.learning_rate: float \| {start, end, schedule}` and an `LRScheduleCallback(ScheduledAttrCallback)` that sets `model.lr_schedule = get_schedule_fn(v)` (SB3 reads `lr_schedule` in `_update_learning_rate`, not `learning_rate`). |
| `rltrain-12` ≈ rltrain-4, prior-5 | low | The promotion gate uses raw consecutive point estimates, so a stage near its threshold is a coin flip | `reinforcetactics/rl/callbacks.py:458` | S | Add `PromotionCallback(criterion='point'\|'wilson'\|'rolling', k=3)`. For 'wilson', compare the lower bound of `wins/episodes` at z=1.0-1.64 to the threshold (lowering thresholds to match), or gate on the mean of the last k evals. |
| `rltrain-13` | low | The run record cannot rebuild the observation space: resolved_config.yaml is written before pad_to_size is resolved, and config.json hand-copies the env kwargs | `scripts/train/train_bootstrap.py:387` | S | Resolve pad_to_size in a pure `resolve_config(cfg)` helper called before the dump, or rewrite resolved_config.yaml from inside `run_curriculum` after resolution. |
| `rltrain-15` ≈ consolidate-14 | low | The sweep configs are 59 full copies (49k lines) because the loader has no inheritance and --set cannot address stages; seed replication is manual | `reinforcetactics/rl/config.py:632` | M | Support `extends: <path>` with a deep merge (with stages matched by `name`), plus `curriculum.stage_defaults: {...}` applied under every stage. Extend `--set` to accept `curriculum.stages[<name>].<field>` and `curriculum.stage_defaults.<field>`. |
| `rltrain-16` | low | scripts/eval_agent.py is orphaned and cannot evaluate any checkpoint the pipeline produces; the example evaluators count draws as losses | `scripts/eval_agent.py:38` | S | Replace it with `scripts/eval_checkpoint.py --run-dir <dir> --stage <name> [--checkpoint best\|final] [--opponent X] [--episodes N] [--deterministic/--stochastic]`. |
| `rltrain-18` | low | ep_info_buffer carries over across stages, so each stage's first rollout metrics describe the previous stage | `reinforcetactics/rl/bootstrap.py:817` | S | Before `model.learn(...)` add `if getattr(model, 'ep_info_buffer', None) is not None: model.ep_info_buffer.clear()`, and do the same for `ep_success_buffer`. |
| `rltrain-19` | low | The eval that promotes a stage never reaches TensorBoard | `reinforcetactics/rl/callbacks.py:472` | S | In `PromotionCallback._on_step`, call `self.logger.dump(self.num_timesteps)` before `return False`. Alternatively, dump in `PeriodicEvalCallback._on_training_end`. |
| `rltrain-20` | low | Curriculum charts: the stage-entry eval bar hides the promoting eval, raw sums are not normalised per episode, and -1.0 WR and a fixed 70% line mislead | `reinforcetactics/rl/viz.py:475` | S | Draw carry-in (`best_eligible == False`) evals with a distinct hatch, or drop them from the stacked bars. Divide reward_components and outcome counts by `r['episodes']`. Render `best_win_rate` of None or -1 as 'n/a'. |
| `rltrain-21` | low | Eight bare `except Exception: pass` blocks in bootstrap.py hide why metadata or checkpoint writes failed | `reinforcetactics/rl/bootstrap.py:859` | S | Keep the best-effort behaviour but log with `logger.warning('...: %s', exc, exc_info=True)` or a one-line print, and count the failures into run_status.json (`metadata_write_failures`). |
| `rltrain-22` ≈ rulebots-7, rlenv-10 | low | The curriculum opponent list duplicates the canonical bot registry and has already drifted: 'master' is rejected, and opponent_kwargs are dropped for … | `reinforcetactics/rl/config.py:200` | S | Derive both sets from `SCRIPTED_BOTS` plus `_ALIASES` (e.g. `bot_registry.accepted_names()`). In `CurriculumStage.validate`, reject `opponent_kwargs` for non-stochastic bots, and validate MixedBot keys (`easy`, `hard`, `p_hard`, `easy_kwargs`, `hard_kwargs`). |
| `rltrain-23` | info | Development opportunity: the two untested axes (gamma, representation) are still untested, and there is no reward/value normalisation | `configs/ppo/bootstrap_sweep/v54_uncapped_frontier.yaml:1` | M | Once the extends/seed infrastructure exists, run three one-knob variants off v52a at 3 seeds each: `features_extractor_kwargs.pool: flatten`; `gamma: 0.997` plus `max_actions_per_turn: 25`; |

## rlalt Feudal RL, AlphaZero/MCTS and imitation learning

### `rlalt-1` — Feudal vec path ignores n_envs in the update count: runs n_envs x the timestep budget, and a linear LR sits at 0 for the last 75% of updates

**high** · bug · confirmed · effort S · `scripts/train/train_feudal_rl.py:456`

Also: `scripts/train/train_feudal_rl.py:474`, `scripts/train/train_feudal_rl.py:488`, `scripts/train/train_feudal_rl.py:462`, `scripts/train/train_feudal_rl.py:452`, `scripts/gcp_launch.sh:61`

- **Impact.** Every default or GCP feudal run trains 4x the requested steps (40M instead of 10M). With `--lr-schedule linear`, the learning rate is 0 for the last 75% of updates, so that compute is wasted. Resumed runs start at the wrong update index and with a shrunken base LR.
- **Fix.** Compute `num_updates = args.total_timesteps // (args.n_steps * max(args.n_envs, 1))` and drive the loop and `start_update` off `total_timesteps` (e.g. `while total_timesteps < args.total_timesteps`). Stash `initial_lr` only when it is absent after `load_checkpoint`, or persist `initial_lr` / base LR in `training_state` and restore it from there.

### `rlalt-2` — `--opponent self` with n_envs>1 trains against an opponent that never moves: the snapshot factory is installed only on the single env

**high** · bug · confirmed · effort S · `scripts/train/train_feudal_rl.py:392`

Also: `scripts/train/train_feudal_rl.py:309`, `reinforcetactics/rl/gym_env.py:1642`, `reinforcetactics/rl/gym_env.py:1255`

- **Impact.** Self-play feudal runs with the default n_envs=4 (and every flat self-play run) train against a passive opponent. The run logs look normal while the agent learns to beat nothing.
- **Fix.** Install the factory on every env in `vec_envs` (and take the initial snapshot before building them). Reject `--opponent self` in `train_flat_baseline`, or wire a factory there too. Add a smoke test that checks `env.opponent is not None` after reset for each vec env in self-play mode.

### `rlalt-3` — The default 6-head feudal worker gets stuck in deterministic eval: the joint argmax is illegal, the state never changes, and the same invalid action repeats until max_steps

**high** · rl-correctness · confirmed · effort S · `reinforcetactics/rl/feudal_rl.py:1159`

Also: `reinforcetactics/rl/feudal_rl.py:1849`, `reinforcetactics/game/model_bot.py:429`, `scripts/ab_feudal_ar.py:110`, `configs/feudal/feudal_rl.yaml`

- **Impact.** For the default worker, `eval/win_rate`, best-model selection and the ab_feudal_ar.py verdict measure the deadlock, not the policy, and the A/B is biased toward AR by construction. Feudal self-play snapshots and tournament entries built on the legacy head are close to passive.
- **Fix.** Make `autoregressive_worker: true` the default (the AR path is always legal under structured masks). For the legacy head: on an invalid step in deterministic mode, fall back to sampling or pick the highest joint-probability legal action from `build_flat_actions`. Also log `invalid_actions` per eval episode so this shows up in metrics.

### `rlalt-4` — MCTS/AlphaZero flat action index drops unit type and source: only Warriors can ever be built and ~52% of legal moves are unreachable

**high** · rl-correctness · confirmed · effort M · `reinforcetactics/rl/mcts.py:100`

Also: `reinforcetactics/rl/gym_env.py:134`, `reinforcetactics/rl/alphazero_net.py:72`, `reinforcetactics/rl/gym_env.py:1046`

- **Impact.** Any AlphaZero agent (self-play, AlphaZeroBot) can only ever build Warriors, and many moves are unrepresentable (for a given target cell, only the first unit that can reach it is selectable). The policy and value targets are learned over a crippled game, so AlphaZero results say nothing about the algorithm.
- **Fix.** Widen the flat space. Cheapest fix (S): give create_unit one plane per unit type (8 create planes), so the index becomes (plane, y, x). For moves and abilities, encode the source (e.g. AlphaZero-chess-style source-cell x offset planes), or build the policy over the legal-action list, or reuse `AutoregressiveActionHead`. Update `build_per_dim_masks(flat_action_size=...)` and `AlphaZeroNet.num_action_types` together, and add a test asserting a bijection between legal structured actions and flat indices.

### `rlalt-5` — AlphaZero resume is broken: map_file/enabled_units/hyperparams are not persisted, and resuming on a sub-20x20 map crashes on a state_dict size mismatch

**high** · bug · confirmed · effort S · `reinforcetactics/rl/alphazero_trainer.py:574`

Also: `reinforcetactics/rl/alphazero_trainer.py:612`, `scripts/train/train_alphazero.py:237`, `reinforcetactics/rl/alphazero_trainer.py:295`

Prior review: REVIEW_rl_pipeline_2026-07-24.md §2.16 (AlphaZero checkpoint config)

- **Impact.** `--resume` either crashes (maps under 20x20) or silently switches to random maps with default c_puct/lr/buffer/enabled_units. The iteration counter and checkpoint names restart, and old checkpoints are overwritten.
- **Fix.** Persist every constructor argument (map_file, enabled_units, c_puct, dirichlet_*, lr, weight_decay, buffer size, max_game_steps, temperature_threshold, eval params) plus `iteration` and `best_network_state` in the checkpoint. Have `train()` start from `self.start_iteration`, and optionally save the replay buffer to a separate .npz. In train_alphazero.py, forward CLI overrides only when the user supplies them explicitly.

### `rlalt-8` — AlphaZero self-play truncates at 400 actions with no max_turns and labels the game a draw; eval gating returns 0.5 on all-draws and plays ~2 distinct games

**high** · rl-correctness · confirmed · effort S · `reinforcetactics/rl/alphazero_trainer.py:94`

Also: `reinforcetactics/rl/alphazero_trainer.py:148`, `reinforcetactics/rl/alphazero_trainer.py:522`, `reinforcetactics/rl/alphazero_trainer.py:529`, `reinforcetactics/rl/alphazero_trainer.py:541`

Prior review: REVIEW_rl_pipeline_2026-07-24.md §2.16

Same issue as: `rlalt-6`, `prior-9`

- **Impact.** The value head is trained mostly toward 0. Every candidate is rejected whenever eval games time out. When games do finish, the acceptance decision rests on 2 effective samples.
- **Fix.** Pass `max_turns` into GameState for both self-play and eval, and score a max-turns ending with a material or structure heuristic, or exclude it from value targets. Treat 'no decided games' as undecided (keep the candidate, or fall back to a tiebreak). For eval, use a small temperature or root noise, or a set of seeded maps, so the games actually differ.

### `rlalt-6` — AlphaZero evaluates the candidate network in train mode: batch-of-1 BatchNorm stats decide acceptance, and the evaluation overwrites the network's running statistics

**medium (reviewer: high)** · bug · confirmed · effort S · `reinforcetactics/rl/alphazero_trainer.py:315`

Also: `reinforcetactics/rl/alphazero_trainer.py:500`, `reinforcetactics/rl/alphazero_trainer.py:327`

Prior review: REVIEW_rl_pipeline_2026-07-24.md §2.16

Same issue as: `prior-9`, `rlalt-8`

- **Impact.** The gate compares a different function (per-sample BatchNorm stats) against the best net in eval mode. Every 5th iteration, thousands of single-sample forward passes corrupt the accepted network's running stats, which self-play then uses.
- **Fix.** Call `self.network.eval()` at the start of `_evaluation_phase` (and in `_self_play_phase`, which already does). Add a regression test that BatchNorm buffers are unchanged after `_evaluation_phase`.
- **Verifier note.** The mechanism is confirmed: `self.network.train()` (315) is never undone before `_evaluation_phase` (327/500), and MCTS inference is only `@torch.no_grad()` (mcts.py:293), which does not freeze BN running stats. I downgraded severity for three reasons. On rejection, `load_state_dict(best_network_state)` restores the stats, so persistent drift happens only on acceptance. The drift comes from real game positions, so running_mean stays roughly right while running_var is biased low.

### `rlalt-7` — AlphaZero 'epochs_per_iteration' runs one minibatch per epoch: 2,560 samples per iteration against up to 10,000 new examples

**medium (reviewer: high)** · rl-correctness · confirmed · effort S · `reinforcetactics/rl/alphazero_trainer.py:407`

Also: `scripts/train/train_alphazero.py:133`, `configs/alphazero/alphazero.yaml`

- **Impact.** Each example is seen about 0.26 times on average before it ages out of the buffer. The network badly under-fits its own MCTS targets, and ReduceLROnPlateau reacts to noise from just 10 batches.
- **Fix.** Either do real epochs (`n_batches = epochs * len(buffer) // batch_size`) or rename the setting to `train_steps_per_iteration` and derive a default from a target sample-reuse ratio (e.g. 4 passes over the newest data). Log samples-seen per iteration.
- **Verifier note.** The facts are accurate. `for epoch in range(self.epochs_per_iteration): batch = self.replay_buffer.sample(self.batch_size)` (407-408) takes one gradient step per "epoch". The defaults are 10 x 256 = 2,560 samples per iteration, while 25 games x up to 400 steps produce up to 10k examples (the yaml matches). At steady state (100k buffer, about 10 iterations of residency) each example is sampled about 0.26 times.

### `rlalt-9` — Manager reward carries over across single-env rollout boundaries, and evaluate() leaks eval-episode goals into training

**medium** · rl-correctness · confirmed · effort S · `reinforcetactics/rl/feudal_rl.py:1362`

Also: `reinforcetactics/rl/feudal_rl.py:1283`, `reinforcetactics/rl/feudal_rl.py:1459`, `reinforcetactics/rl/feudal_rl.py:1512`, `reinforcetactics/rl/feudal_rl.py:1838`

Prior review: REVIEW_rl_pipeline_2026-07-24.md §2.16 (Feudal, first bullet)

Same issue as: `rlalt-11`, `prior-8`

- **Impact.** One segment per rollout is credited with reward from a previous decision and discounted as gamma^(k>horizon). After each eval, training resumes with a goal chosen in a different game. The effect is small at n_steps=2048/horizon=10 but grows with horizon.
- **Fix.** Unify the two collectors: implement `collect_rollout(env)` as `collect_rollout_vec([env])` with per-env state persisted on the agent, including manager_open/reward/steps. Otherwise, always reset `manager_reward/steps` when a new goal opens, and snapshot and restore `current_goal/goal_step_counter` around `evaluate()`. Add a test asserting `m_segment_lengths <= manager_horizon`.

### `rlalt-10` — Manager PPO ratio is contaminated by encoder drift from worker updates within the same update()

**medium** · rl-correctness · partially · effort S · `reinforcetactics/rl/feudal_rl.py:1710`

Also: `reinforcetactics/rl/feudal_rl.py:1644`, `reinforcetactics/rl/feudal_rl.py:1656`

- **Impact.** The manager's importance ratio reflects representation changes it cannot control. Clipping zeroes the gradient for those samples, so manager learning is throttled at random and the trust region is not what it claims to be.
- **Fix.** Manager features are detached anyway, so compute them once at the top of `update()` under `torch.no_grad()` (before any worker step) and reuse them for all manager epochs. This also saves n_epochs encoder forward passes over the manager data. Optionally log the manager clip fraction.
- **Verifier note.** The drift is real and large. `update()` runs every worker minibatch of an epoch, and those steps update the shared encoder through worker_optimizer (1644-1692). The manager then recomputes `features = self.feature_extractor(b_obs).detach()` (1710) with the updated encoder, while `m_old_lp` comes from rollout time. There are no BatchNorm or Dropout layers, so the drift at epoch 0 is exactly 0. Parts of the framing are overstated, though.

### `rlalt-11` — Feudal return targets: truncation is treated as terminal, and the manager critic regresses on an undiscounted segment sum

**medium** · rl-correctness · partially · effort S · `reinforcetactics/rl/feudal_rl.py:1400`

Also: `reinforcetactics/rl/feudal_rl.py:731`, `reinforcetactics/rl/feudal_rl.py:1433`, `reinforcetactics/rl/feudal_rl.py:1437`

Prior review: REVIEW_rl_pipeline_2026-07-24.md §2.16 (Feudal, bullets 2-3)

Same issue as: `rlalt-9`, `prior-8`

- **Impact.** Values near the step limit are pushed toward 0 for states that are not actually terminal. The observation carries no step counter, so this adds irreducible value noise. The manager's return is inconsistent with its own discounting (about 5% bias at k=10, larger for longer horizons).
- **Fix.** Store `terminated` for GAE, and on `truncated` bootstrap from `V(final_obs)` (keep `next_obs` before `env.reset()`). Accumulate `manager_reward += gamma**manager_steps * ext_reward` so the segment return is the SMDP-discounted sum.
- **Verifier note.** Both mechanisms are present at HEAD. `done = terminated or truncated` (feudal_rl.py:1400) goes into `w_dones` and into `m_dones` (through `end_manager_segment(..., done=True)` at 1438), and `_compute_gae` sets `non_terminal = 1 - dones[t]`, which zeroes the bootstrap (731-734). The env deliberately relies on the learner bootstrapping at truncation (gym_env.py:1476-1491 charges 0 at truncation for exactly that reason), so feudal gets this wrong.

### `rlalt-12` — worker_reward_alpha does not mean what the CLI says; the intrinsic reward is unnormalized, paid every step, and partly outside the worker's control

**medium** · rl-correctness · confirmed · effort M · `reinforcetactics/rl/feudal_rl.py:826`

Also: `scripts/train/train_feudal_rl.py:732`, `reinforcetactics/rl/feudal_rl.py:834`, `reinforcetactics/rl/feudal_rl.py:1916`, `reinforcetactics/rl/feudal_rl.py:1934`

Prior review: REVIEW_rl_pipeline_2026-07-24.md §2.16 (Feudal, bullet 4)

- **Impact.** An alpha sweep (feudal_rl_review 'Still open' #1) cannot reach extrinsic-only. Paying a level reward every step encourages camping on the goal cell over finishing it, and part of the worker's advantage is noise from enemy positions it cannot affect. goal_reached_rate is inflated late in the game.
- **Fix.** Either use `(1-alpha)*intrinsic + alpha*extrinsic` or fix the help text (and the notebook). Make the intrinsic reward potential-based (previous distance minus new distance to the goal, plus a one-time reach bonus), cap the count-based bonuses, and record `reached` directly from `unit_at_goal` rather than thresholding the reward.

### `rlalt-13` — AR head: unit_type is a free 'don't-care' stage for every non-create action, adding log-prob noise to PPO and ~2 nats of spurious entropy

**medium** · rl-correctness · confirmed · effort S · `reinforcetactics/rl/feudal_rl.py:93`

Also: `reinforcetactics/rl/feudal_rl.py:544`, `reinforcetactics/rl/feudal_rl.py:599`, `reinforcetactics/rl/imitation.py:403`

- **Impact.** The PPO ratio carries variance from an irrelevant head, and `ent_coef=0.05` rewards maximal ut entropy on most steps. This is the same leak BC already fixes by narrowing placeholder dims (imitation.py:403-416).
- **Fix.** In `unit_type_mask`, return a one-hot at index 0 when `int(atype.item()) != 0` (log p = 0, entropy = 0). Apply the same narrowing to the stored masks so `evaluate` matches. Add a test that non-create steps contribute zero ut entropy.

### `rlalt-14` — AlphaZeroBot ignores the checkpoint's architecture params, so any non-default net fails to load; AlphaZero is also not reachable from the tournament

**medium** · bug · confirmed · effort S · `reinforcetactics/game/alphazero_bot.py:99`

Also: `reinforcetactics/rl/alphazero_trainer.py:574`, `reinforcetactics/game/model_bot.py:82`, `reinforcetactics/tournament/bots.py:414`

Same issue as: `aibots-10`, `consolidate-16`, `rulebots-23`

- **Impact.** Only checkpoints with the default 6x128 architecture load. A map-size mismatch crashes at forward time instead of producing a clear error. Trained AlphaZero agents cannot enter the ladder, which is ROADMAP's stated goal.
- **Fix.** Pass `num_res_blocks=config.get('num_res_blocks', 6), channels=config.get('channels', 128)`, and raise a clear error when the checkpoint grid dims differ from the live grid. Register 'alphazero' in the bot registry, and have ModelBot or tournament discovery route a .pt with a `model_state_dict` + `config` payload to AlphaZeroBot.

### `rlalt-15` — train_feudal_rl crashes at the end of every --config run: json.dump(vars(args)) hits the non-serializable args._cfg

**medium** · bug · confirmed · effort S · `scripts/train/train_feudal_rl.py:618`

Also: `scripts/train/train_feudal_rl.py:239`, `scripts/train/train_feudal_rl.py:811`

- **Impact.** Any run started with `--config` (the documented way) exits with a traceback after the final save. config.json is never written, `writer.close()` and `wandb.finish()` are skipped (the last TensorBoard/W&B points can be lost), and the non-zero exit code makes orchestration treat the run as failed.
- **Fix.** Dump `{k: v for k, v in vars(args).items() if not k.startswith('_')}` plus `dataclasses.asdict(args._cfg)` (or the resolved YAML), and put `writer.close()` in a `finally` block.

### `rlalt-16` — train_feudal_rl.py defaults to --mode flat, and the config's `algorithm: feudal` is not mapped, so the documented feudal commands train flat PPO

**medium** · ux · confirmed · effort S · `scripts/train/train_feudal_rl.py:685`

Also: `README.md:104`, `configs/feudal/feudal_rl.yaml:8`

- **Impact.** `--config configs/feudal/feudal_rl.yaml` and the README command both train a flat PPO baseline while the user believes a feudal run is in progress, which wastes a whole training budget.
- **Fix.** Derive the default `--mode` from `cfg.algorithm` when a config is given (map 'feudal' to feudal, 'ppo'/'maskable_ppo' to flat), fail loudly on a conflict, and fix the README command to include `--mode feudal`. Longer term, split the flat baseline into its own script.

### `rlalt-17` — AlphaZero replay buffer holds ~75 KB of dense float32 per example (~7.5 GB at the default 100k on 20x20)

**medium** · performance · partially · effort S · `reinforcetactics/rl/alphazero_trainer.py:116`

Also: `reinforcetactics/rl/alphazero_trainer.py:51`, `reinforcetactics/rl/alphazero_trainer.py:60`, `configs/alphazero/alphazero.yaml`

- **Impact.** Default AlphaZero runs on the default random 20x20 maps will run out of memory or swap on typical machines well before the buffer fills.
- **Fix.** Store the mask as packed bits or legal indices, the policy as sparse (indices, probs) over legal actions, and the observations as float16 (grid/units are one-hot or bounded). Sample with `np.random.randint` over a preallocated ring buffer of numpy arrays instead of a deque of tuples.
- **Verifier note.** The per-example memory figure is exact. Each example holds grid (20,20,11) f32 = 17,600 B, units (20,20,16) f32 = 25,600 B, gf (5,) f32 = 20 B, a dense mask (4000,) f32 = 16,000 B and a dense policy (4000,) f32 = 16,000 B, for 75,220 B total, or about 7.5 GB at the default buffer_size 100000 on the default 20x20 random map. Both the mask and the policy are extremely sparse (measured nnz of 3 and 2).

### `rlalt-18` — MCTS hot path: one network call per simulation, no subtree reuse, and full-GameState deepcopy including the growing action_history

**medium** · performance · partially · effort M · `reinforcetactics/rl/mcts.py:426`

Also: `reinforcetactics/rl/mcts.py:329`, `reinforcetactics/rl/mcts.py:360`, `reinforcetactics/core/game_state.py:649`

Same issue as: `core-18`

- **Impact.** Self-play cost grows with game length. Roughly 100 sims x 400 moves x 25 games per iteration of serial batch-1 inference makes the default config take days on CPU.
- **Fix.** Add a `GameState.clone_for_search()` (or `__deepcopy__` memo override) that skips action_history, initial_map_data and replay metadata. Keep the chosen child as the next root (subtree reuse). Batch leaf evaluation with virtual loss (e.g. 8-16 leaves per forward pass).
- **Verifier note.** The structural claims hold. search() deep-copies the incoming state and builds a fresh root on every move (329), so there is no subtree reuse. Each simulation evaluates a single leaf with a batch-1 forward pass via `_evaluate(node.game_state)` (360). Lazy expansion deep-copies the whole GameState (426). GameState has no __deepcopy__/__getstate__ override and no clone helper anywhere in the repo, and action_history records carry an ISO timestamp (game_state.py:643-649).

### `rlalt-19` — The default and GCP training paths, and every trainer/script driver, have no tests; each high-severity bug above sits in untested code

**medium** · test-gap · confirmed · effort M · `reinforcetactics/rl/feudal_rl.py:1465`

Also: `tests/test_alphazero.py`, `tests/test_feudal_rl_integration.py`, `tests/test_imitation.py`

- **Impact.** The n_envs accounting, vec self-play, config dump, eval train-mode, resume and AlphaZeroBot arch bugs all shipped undetected, and future consolidation passes have no safety net here.
- **Fix.** Add fast smoke tests: (1) `train_feudal_rl(args)` with n_envs=2, n_steps=16, total=64, and asserts on the update count, the self-play opponent being set, and config.json being written; (2) `AlphaZeroTrainer(map=beginner, 1 block, 8 ch, 2 sims).train()` for 1 iteration, followed by `load_checkpoint` and an AlphaZeroBot load; (3) `collect_rollout_vec` with 2 envs, checking merged lengths/masks; (4) a `build_bc_warmstart.main()` run with a 1-episode scenario, then `set_parameters(exact_match=True)` into a bootstrap model.

### `rlalt-21` — BC subsystem gaps from §2.12 are still open at HEAD: multi_discrete only, demos on the default engine, and the env's obs scales/padding are not threaded through

**medium** · rl-correctness · confirmed · effort M · `reinforcetactics/rl/imitation.py:607`

Also: `reinforcetactics/rl/imitation.py:352`, `reinforcetactics/rl/imitation.py:1231`, `scripts/train/train_bootstrap.py:136`, `reinforcetactics/rl/gym_env.py:769`

Prior review: REVIEW_rl_pipeline_2026-07-24.md §2.12

Same issue as: `prior-12`, `rltrain-17`

- **Impact.** BC on a config with engine overrides or non-default obs scales clones behaviour from a different game, or scores observations on a different scale, with no error. BC also still targets the action space that REVIEW 2.12 measured as unable to learn.
- **Fix.** Decide between porting and retiring (the label for a flat_discrete port is the index of the demonstrated action in `build_flat_actions`). If keeping BC, add `engine_overrides`, `rng` seeding, `pad_to_size` and the obs-scale fields to `DemonstrationScenario`, and pass them through from `cfg.env` in `_bc_build`.

#### Low and info findings (6)

| ID | Sev | Title | Location | Effort | Fix (summary) |
|---|---|---|---|---|---|
| `rlalt-20` | low | BC has no held-out validation: accuracy is measured in-sample after the optimizer step, and weighted upsampling would leak duplicates across a naive split | `reinforcetactics/rl/imitation.py:1297` | S | Split by episode, per scenario, before resampling (keep the episode id on each Demonstration). Report val loss / val atype / full accuracy in BCStats, and early-stop on val loss. |
| `rlalt-22` | low | The A/B harness verdict is statistically meaningless (1 seed, 10 eval episodes, 0.05 threshold), and all of feudal_rl_review's 'Still open' items remain open | `scripts/ab_feudal_ar.py:144` | S | Run at least 3 seeds per arm, 50+ eval episodes on a fixed seed set, and report a Wilson or bootstrap CI on the win-rate difference. Fix the legacy eval first. Record invalid-action rate and eval goal-reach alongside win rate. |
| `rlalt-23` | low | Goal encoding feeds raw integer coordinates and an ordinal goal_type into Linear(3, 64); goals are not masked to the playable map | `reinforcetactics/rl/feudal_rl.py:242 (also :638)` | S | Embed goal_type with `nn.Embedding(4, d)`, normalize x/y by (W-1, H-1) (or use a learned positional embedding), and mask manager x/y logits to cells within the original map bounds. |
| `rlalt-24` | low | MCTS edge semantics: the 'terminal loss' on a failed action is dead code, and stochastic transitions are frozen at their first sample | `reinforcetactics/rl/mcts.py:439-445` | S | Set a dedicated `failed` flag and back up -1 for the acting player (or prune the edge). For stochastic actions, either re-sample the transition on each visit (open-loop MCTS) or add chance nodes keyed by outcome. |
| `rlalt-26` | low | train_alphazero.py hygiene: no seeding, wrong sys.path root, and a config key whose comment claims it is used when it isn't | `scripts/train/train_alphazero.py:32` | S | Add `--seed` (mapped to `seed`) and seed python/numpy/torch plus the GameState rng. Use `parent.parent.parent` or drop the hack. Delete or map `env.max_steps`, and add `auto` device resolution as in train_feudal_rl.py. |
| `rlalt-25` | info | AlphaZeroNet policy head is a 51M-parameter dense layer tied to grid size; a conv policy head would be ~1000x smaller and size-agnostic | `reinforcetactics/rl/alphazero_net.py:97` | M | Replace the policy head with `Conv2d(channels, num_action_types, 1)` flattened to (A*H*W), which matches the flat layout exactly. Use global average pooling in the value head. |

## pygame Pygame application shell and renderer

### `pygame-1` — Selecting the offered '1v1v1' game mode crashes the whole application

**critical** · bug · confirmed · effort S · `reinforcetactics/ui/menus/game_setup/player_config_menu.py:43`

Also: `reinforcetactics/ui/menus/game_setup/game_mode_menu.py:28`, `reinforcetactics/ui/menus/main_menu.py:82`, `reinforcetactics/cli/commands.py:256`

Same issue as: `menus-2`, `consolidate-6`

- **Impact.** A user who picks the third menu entry ('1v1v1') and then a map gets a traceback and the app exits. None of the 4 shipped 3-player maps can be played from the GUI.
- **Fix.** Short term: have GameModeMenu offer only the modes PlayerConfigMenu supports, or derive num_players from the mode (the HQ count of the chosen map, or '1v1v1' -> 3). Then let PlayerConfigMenu accept any 2-4 player count; start_new_game already uses len(player_configs). Also wrap main_menu.run() in play_mode with an error dialog so a menu bug can't kill the process.

### `pygame-2` — Saves of random-map games can never be loaded (and the fallback would build a different random map)

**critical** · bug · confirmed · effort S · `reinforcetactics/app/game_loop.py:329`

Also: `reinforcetactics/core/game_state.py:1446`, `reinforcetactics/core/grid.py:86`, `reinforcetactics/app/game_loop.py:249`

- **Impact.** Data loss in normal use. 'random' is the first entry in the map list. Every save made in such a game, whether via S, Pause > Save or Save & Quit, fails to load with 'Error loading game'.
- **Fix.** Persist the full terrain (padded map codes) in the save whenever map_file_used is None. Reasonable to always do it, which also guards against edited map files. Change the load check to `if save_data.get("map_file"):`, otherwise rebuild map_data from the saved terrain. While there, also persist max_turns and end_reason: from_dict reads them but to_dict never writes them. Add a save/load round-trip test for a random map.

### `pygame-3` — Ending the turn with SPACE during target selection leaves stale target mode armed: the next player's click fires the old unit's action

**high** · bug · confirmed · effort S · `reinforcetactics/app/input_handler.py:118`

Also: `reinforcetactics/app/input_handler.py:70`, `reinforcetactics/app/input_handler.py:142`, `reinforcetactics/app/action_executor.py:34`

- **Impact.** Corrupts game state and replays in normal play. The previous player's unit acts during another player's turn, against stale targets (GameState.attack does no range, turn or can_attack check), and the unit's next-turn action is consumed.
- **Fix.** Add one `_clear_selection_state(cancel_move: bool)` and call it from every end-turn path (SPACE, End Turn button, before _process_bot_turns). Alternatively block SPACE while target_selection_mode is set. Make ESC reopen the UnitActionMenu the same way the click-outside path does. Add a headless regression test that selects Attack, presses SPACE and clicks a target.

### `pygame-5` — Bot turns only run after a human ends a turn: a Player-1 bot's first turn is played by the human, and all-bot games stall

**high** · bug · confirmed · effort S · `reinforcetactics/app/input_handler.py:393`

Also: `reinforcetactics/app/game_loop.py:79`, `reinforcetactics/app/input_handler.py:125`, `reinforcetactics/app/input_handler.py:161`, `reinforcetactics/ui/menus/game_setup/player_config_menu.py:280`

- **Impact.** Human players control the AI's units and spend the AI's gold on turn 1. A bot-vs-bot spectate game cannot be watched.
- **Fix.** At the start of GameSession.run, and each frame, if `game.current_player in bots` and the game isn't over, run the bot turn (ideally via the non-blocking playback in the next finding) instead of waiting for input. Drop the max_bot_turns cap-and-return behaviour for all-bot games.

### `pygame-7` — With a unit selected, clicking your own empty building opens the purchase menu, so units can never be moved onto own buildings

**high** · bug · confirmed · effort S · `reinforcetactics/app/input_handler.py:372`

- **Impact.** Human players can't garrison or heal units on their own buildings (heal_units_on_structures) or block a building. Bots and the RL env can. The overlay advertises a move that can't be made.
- **Fix.** If `self.selected_unit` can move and (grid_x, grid_y) is among its reachable positions, move it. Otherwise fall through to the purchase menu. Optionally let a second click on the building open the purchase menu after deselecting.

### `pygame-8` — HUD is drawn over the playfield; the Resign/End Turn buttons capture clicks on playable tiles and the HUD hides units

**high** · ux · confirmed · effort M · `reinforcetactics/ui/renderer.py:369`

Also: `reinforcetactics/ui/renderer.py:770`, `reinforcetactics/app/input_handler.py:157`, `reinforcetactics/app/game_loop.py:249`

- **Impact.** Clicking a unit or tile in those corners opens the resign dialog or ends the turn instead. Units and structures there are invisible. Localized labels widen the boxes further.
- **Fix.** Reserve a dedicated HUD strip (e.g. 48 px top bar or a right side panel): grow the window by that amount and draw the grid at an offset. Route all screen->grid conversion (input_handler lines 224, 257, 342; renderer.draw_unit_tooltip) through one `renderer.screen_to_grid()` helper that subtracts the offset. That helper is also the hook a camera needs. Give generate_random_map UI output the same 2-tile border as file maps.

### `pygame-10` — CLI --mode train/evaluate (the README's documented entry point) is stale: DQN crashes, shaping zeroed, no masking, self-play vs nothing, --render window never flips

**high** · rl-correctness · confirmed · effort M · `reinforcetactics/cli/commands.py:60`

Also: `reinforcetactics/cli/main.py:82`, `reinforcetactics/cli/commands.py:92`, `reinforcetactics/cli/commands.py:181`, `reinforcetactics/rl/gym_env.py:1671`, `README.md:81`

Same issue as: `consolidate-5`, `persist-17`

- **Impact.** Anyone following README's 'Train an RL Agent' section gets a crash (dqn), silently degenerate training (self), or a run that diverges from the maintained pipeline (scripts/train/*, MaskablePPO). The README's Vertex example submits exactly this command.
- **Fix.** Either delegate `--mode train` to the maintained entry point (rl.bootstrap / scripts/train/train_bootstrap.py with --config) and `--mode evaluate` to rl.evaluation.evaluate_model with masks, or deprecate both with a pointer. Remove 'dqn' from choices unless flat_discrete is used. Default the reward flags to None and only override when given. Reject `--opponent self` without a factory. In gym_env human mode, flip and pump events and reuse the Renderer across resets. Update README.md:81-95 and 225.

### `pygame-4` — Pressing S in-game resizes the game window to 900x700 for the rest of the session

**medium (reviewer: high)** · bug · confirmed · effort S · `reinforcetactics/app/game_loop.py:163`

Also: `reinforcetactics/app/game_loop.py:146`, `reinforcetactics/ui/menus/base.py:93`, `reinforcetactics/ui/menus/in_game/pause_menu.py:91`

Same issue as: `menus-12`

- **Impact.** After a save via the advertised 'Press S to save game' control, the map is cropped: the bottom 2 rows on 24x24 UI grids, 7 rows on the 29x29 demo map. The window gains a dead strip on the right, and the End Turn/Resign buttons are no longer at the window edge. The Save & Quit path (line 146) goes through the same call.
- **Fix.** Use `SaveGameMenu(self.game, self.renderer.screen)`. Add an assert or test that a GameSession's display size is unchanged after each in-game sub-menu.
- **Verifier note.** The mechanism is confirmed exactly. game_loop.py:163 calls SaveGameMenu(self.game) with no screen, so ScreenBootstrapMixin._init_screen calls pygame.display.set_mode((900,700)) (base.py:93), which resizes the display surface the Renderer draws on. PauseMenu passes self.screen (pause_menu.py:91). The Save & Quit path (game_loop.py:146) goes through the same call but quits right after, so it only matters cosmetically there. The impact is overstated, so severity drops to medium.

### `pygame-6` — Bot turns block the main thread inside the event handler: window freezes, bot moves never render, LLM retries sleep on the UI thread

**medium (reviewer: high)** · design · confirmed · effort L · `reinforcetactics/app/input_handler.py:402`

Also: `reinforcetactics/game/llm_bot.py:501`, `reinforcetactics/app/game_loop.py:81`

Same issue as: `anim-2`, `aibots-5`, `pygame-11`

- **Impact.** With LLM bots the OS shows 'Not Responding'. With any bot the human only sees the end state, never what the opponent did, and input made during the freeze is queued and replayed afterwards (see the double-click finding).
- **Fix.** Run bot.take_turn() in a worker thread (or step the bot action by action). Keep the loop pumping events and drawing a 'Player N is thinking...' banner with a cancel/resign option. Apply bot actions on the main thread via a queue, with a short per-action delay so moves are visible. That is also where movement animations belong (see the animations finding). At minimum, call pygame.event.pump() and render between bot turns, and draw a blocking overlay before starting an LLM turn.
- **Verifier note.** Confirmed as a design issue. current_bot.take_turn() runs synchronously inside the keyboard and mouse handlers (input_handler.py:399-405). Nothing pumps events, renders or flips during bot turns; the only display.flip in app/ is game_loop.py:194. LLMBot._call_llm_with_retry sleeps on the same thread (llm_bot.py:501, time.sleep(2**attempt), max_retries=3 so 1 s + 2 s), and two-phase planning adds a second call (llm_bot.py:419-420). Events queued during the freeze are handled on later frames.

### `pygame-9` — GUI ignores the fog-of-war 'move to discover, then attack' rule the engine enforces

**medium (reviewer: high)** · bug · confirmed · effort S · `reinforcetactics/ui/menus/in_game/unit_action_menu.py:63`

Also: `reinforcetactics/app/input_handler.py:367`, `reinforcetactics/core/game_state.py:1369`, `reinforcetactics/app/action_executor.py:62`

- **Impact.** Human players get an ability that bots, LLM bots and RL agents are denied, so GUI games are unfair and human-vs-model results can't be compared with engine rules.
- **Fix.** Filter the menu's targets (attack/paralyze and the other enemy-targeted actions) with `game_state.is_enemy_attackable_by_unit(unit, e)`. Better: build the menu from `game.get_legal_actions()` so the GUI can never diverge from the engine again. Also re-capture the snapshot on haste re-activation in action_executor, as input_handler.py:296 already does.
- **Verifier note.** UnitActionMenu._calculate_available_actions builds its attack targets from `GameMechanics.get_attackable_enemies(...)` (unit_action_menu.py:63) with no FOW check. action_executor and _handle_target_selection_click then call game.attack(), and game_state.attack() (game_state.py:~790-870) does not check is_enemy_attackable_by_unit either. The snapshot captured on selection (input_handler.py:367, or lazily in move_unit at game_state.py:760) is therefore never consulted on the GUI path.

### `pygame-11` — A double-click on End Turn (or input queued during a bot turn) silently skips the human's next turn

**medium (reviewer: high)** · bug · confirmed · effort S · `reinforcetactics/app/input_handler.py:157`

Also: `reinforcetactics/app/game_loop.py:81`, `reinforcetactics/app/game_loop.py:101`, `reinforcetactics/app/input_handler.py:118`

Same issue as: `pygame-6`, `anim-2`, `aibots-5`

- **Impact.** The player loses a whole turn with no feedback, and stale clicks land wherever the cursor happens to be.
- **Fix.** After `_process_bot_turns()` returns, drop queued input with `pygame.event.clear((pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.KEYDOWN))`, or keep drain_events semantics that preserve QUIT. Add a short (~300 ms) debounce on End Turn. Use `event.pos` for click and motion events.
- **Verifier note.** The mechanism is real. End Turn (input_handler.py:157-162) and SPACE (input_handler.py:118-125) call end_turn() and then _process_bot_turns() synchronously. Nothing flushes queued input afterwards and End Turn has no debounce; the 200 ms guard at input_handler.py:148 applies only to open menus.

### `pygame-12` — Fog-of-war information leaks in rendering and input: structure owners, hidden-unit holes in the move overlay, right-click preview, hotseat perspective

**medium** · bug · confirmed · effort M · `reinforcetactics/ui/renderer.py:443`

Also: `reinforcetactics/core/tile.py:75`, `reinforcetactics/ui/renderer.py:853`, `reinforcetactics/app/input_handler.py:232`, `reinforcetactics/ui/renderer.py:416`

Same issue as: `core-5`, `rlenv-14`, `consolidate-1`, `critic-integration-3`

- **Impact.** FOW games reveal enemy captures anywhere on the map and the positions of hidden units. In hotseat, each player sees the other's view.
- **Fix.** Keep a per-player 'last seen owner' for structures in the visibility map and render that (or neutral) when the tile isn't VISIBLE. Build the movement overlay from units visible to the viewer (the engine still resolves collisions on move). Ignore right-clicks on non-visible enemies. Add a 'Pass to Player N' interstitial for hotseat FOW games.

### `pygame-13` — No camera or viewport: window equals the map size; editor-made maps above ~26 tiles don't fit a 1080p screen

**medium** · ux · confirmed · effort L · `reinforcetactics/ui/renderer.py:97`

Also: `reinforcetactics/ui/menus/map_editor/new_map_dialog.py:82`, `reinforcetactics/utils/settings.py:17`, `reinforcetactics/app/game_loop.py:118`

- **Impact.** Custom maps from the shipped editor are unplayable on laptops. Window size jumps between menus (900x700) and games.
- **Fix.** Add a camera offset and zoom to Renderer: draw only visible tiles, provide `screen_to_grid()` and `grid_to_screen()`, and support edge/arrow-key/middle-drag scrolling. Size the window from `pygame.display.get_desktop_sizes()` clamped to the map. Honour video.fullscreen and video.fps (replace the hard-coded clock.tick(60)). Pair this with the HUD strip from the HUD finding.

### `pygame-14` — A bot exception ends the session: game lost, no replay saved, only a console traceback; ModelBot FileNotFoundError escapes the fallback

**medium** · bug · partially · effort M · `reinforcetactics/app/input_handler.py:402`

Also: `reinforcetactics/app/input_handler.py:402`, `reinforcetactics/app/bot_factory.py:147`, `reinforcetactics/game/model_bot.py:80`

- **Impact.** One bad LLM response or a moved model file drops the user back to the main menu without explanation, and a long game is lost.
- **Fix.** Wrap each bot turn: on exception, log it, show an in-game toast, and end that bot's turn (or swap in SimpleBot) so the game continues. In start_new_game/load_saved_game use try/finally to autosave a crash save plus replay and restore the display. Catch `Exception` in create_bots_from_config and show a ConfirmationDialog ('Model not found, use SimpleBot?').
- **Verifier note.** Real: - _process_bot_turns has no try/except (input_handler.py:399-405). - An exception unwinds GameSession.run into start_new_game's `except Exception` (game_loop.py:308-312) or load_saved_game's (~386). That returns None, skips pygame.quit and save_replay_to_file, and play_mode simply loops back to the main menu (commands.py:250-283).

### `pygame-18` — Targeted-action dispatch and haste continuation duplicated between input_handler and action_executor, glued with list-as-pointer out-params

**medium** · consolidation · partially · effort M · `reinforcetactics/app/input_handler.py:264`

Also: `reinforcetactics/app/action_executor.py:87`, `reinforcetactics/app/input_handler.py:93`, `reinforcetactics/app/input_handler.py:323`, `reinforcetactics/app/bot_factory.py:147`, `reinforcetactics/app/game_loop.py:68`

Same issue as: `core-14`, `consolidate-9`, `aibots-13`, `prior-14`, `rulebots-20`

- **Impact.** The copies already drift: only the target-click path re-captures the FOW snapshot after haste. Any new targeted ability must be added in two places, and none of these paths validate against engine legality.
- **Fix.** Add a `TARGETED_ACTIONS = {"attack": (GameState.attack, "attacked"), ...}` table and one `apply_targeted_action(game, kind, unit, target)` that checks the target against `game.get_legal_actions()` and returns can_still_act. Move handle_action_menu_result/execute_unit_action onto InputHandler as methods that mutate self (removing the ref-lists). Collapse the bot_factory excepts into `except (ValueError, ImportError, OSError)`. Extract `_run_session(game, bots)` shared by start_new_game and load_saved_game.
- **Verifier note.** The duplication is real: - The 7-branch attack/paralyze/heal/cure/haste/defence_buff/attack_buff dispatch appears at input_handler.py:264-284 and action_executor.py:93-113. - The same dispatch also exists in game/bot.py (around lines 117-128) and rl/mcts.py:207-220, which the reviewer missed. That makes the case for a shared table stronger. - The haste/end_unit_turn continuation appears at action_executor.py:58-66, 78-85, 114-121 and input_handler.py:287-300.

### `pygame-19` — CJK fallback picks DejaVu Sans (no CJK glyphs) ahead of Noto Sans CJK, so Korean and Chinese render as tofu on typical Linux

**medium** · bug · confirmed · effort S · `reinforcetactics/utils/fonts.py:41`

Also: `reinforcetactics/utils/fonts.py:185`

- **Impact.** Korean and Chinese UI text is unreadable on Linux even when a CJK font is installed. The macOS order also prefers a Korean font (Apple SD Gothic Neo) for Chinese.
- **Fix.** Order candidates per language (Korean: Noto Sans CJK KR, Malgun Gothic, Apple SD Gothic Neo; Chinese: Noto Sans CJK SC, Microsoft YaHei, PingFang SC) and move DejaVu, Helvetica and FreeSans after all CJK fonts. Better: validate coverage by rendering a sample glyph ('한' or '中') and rejecting fonts whose output matches the .notdef box. Add a test that asserts the chosen font isn't DejaVu when a CJK candidate is present.

### `pygame-20` — 'Save & Quit' quits the app even when the save was cancelled or failed

**medium** · bug · confirmed · effort S · `reinforcetactics/app/game_loop.py:145`

Also: `reinforcetactics/app/game_loop.py:161`, `reinforcetactics/cli/commands.py:281`

- **Impact.** A user who presses ESC in the save-name prompt, or whose save fails, loses the game state (only the auto-replay survives) after explicitly choosing to save.
- **Fix.** Have _handle_save_game return the path, and on None go back to the pause menu (or show 'Save failed/cancelled. Quit anyway?') instead of quitting.

### `pygame-24` — app/, ui/ and cli/ are excluded from coverage and effectively untested, yet every bug above is reproducible headlessly

**medium** · test-gap · partially · effort M · `pyproject.toml:139`

Also: `tests/test_input_handler.py:47`

Same issue as: `menus-22`, `tests-6`, `anim-21`

- **Impact.** The main user-facing surface regresses without CI noticing. The consolidation work in ui/ (ListDetailMenu, ScreenBootstrapMixin) has no safety net.
- **Fix.** Add a `headless_session` pytest fixture (dummy SDL, tmp cwd for settings/replays, GameSession with a frame cap) plus an event-posting helper, and turn each finding here into a regression test. Add a render smoke test that renders one frame per shipped map in both pixel modes and asserts no exception and frame time under a budget. Once there are tests, bring app/ and ui/renderer.py back into coverage.
- **Verifier note.** True: [tool.coverage.run] omit (pyproject.toml:139-144) excludes ui/*, cli/* and app/*. No test references Renderer (test_gym_env only asserts env.renderer is None and render() returns None with no render_mode), GameSession, game_loop or reinforcetactics.cli. tests/test_input_handler.py has 4 tests, all mock-based on _process_bot_turns. Overstated: ui/ is not 'effectively untested', and the claim that the ui/ consolidation work 'has no safety net' is false.

#### Low and info findings (7)

| ID | Sev | Title | Location | Effort | Fix (summary) |
|---|---|---|---|---|---|
| `pygame-15` | low | Renderer redraws static terrain tile by tile every frame (3.7 ms/frame on 24x24, 21 ms on 54x54 with defaults) | `reinforcetactics/ui/renderer.py:397` | M | Pre-render the terrain (tiles, borders, forest dots, structure fills) into one `convert()`ed surface at init. Re-blit only structure tiles when owner or health changes, keyed on a cheap dirty set or version counter from GameState. |
| `pygame-16` | low | The interactive game never uses the bundled pixel art by default; it falls back to coloured rects and letters | `reinforcetactics/ui/renderer.py:204` | S | Make the bundled assets the default when settings don't name a sprites path: in settings.get_sprites_path, fall back to the bundled directory, or treat pixel_art=None as 'bundled if present'. Keep graphics.disable_* as the opt-outs. |
| `pygame-17` ≈ anim-7, consolidate-17 | low | Movement and path animations are never triggered: units teleport, and the animator's per-unit state is never cleaned up | `reinforcetactics/ui/renderer.py:1011` | M | When game.move_unit succeeds (GUI and bot playback), compute the path via a BFS parent map, call queue_movement_path_animation, and interpolate the draw position over ~120 ms per tile. Block input during the tween, or queue it. |
| `pygame-21` | low | Every game session ends with pygame.quit() and play_mode re-inits pygame: window destroyed and recreated, and error paths skip cleanup | `reinforcetactics/app/game_loop.py:303` | S | Keep pygame initialised for the life of play_mode and just `set_mode` the menu size when a session returns. Do teardown once in a try/finally in play_mode. Remove Renderer.close()'s global pygame.quit, or make it opt-in. |
| `pygame-22` | low | Scenario-only maps (7 of 20 1v1 maps) have no production buildings; a GUI New Game on them can never end | `reinforcetactics/app/game_loop.py:265` | S | Hide maps without a building per player from MapSelectionMenu, or tag them as scenarios and route them through the load-scenario path. Offer a max_turns option in PlayerConfigMenu, defaulting to something like 100, so every GUI game can end. |
| `pygame-23` | low | Dead or contradictory code in renderer, icons, CLI and bot_factory | `reinforcetactics/ui/renderer.py:1069` | S | Delete the unused wrappers, or wire them (animations finding). Give FOW an explicit OMNISCIENT sentinel if replays want it. Replace the icon boilerplate with a `@_cached_icon("name")` decorator taking a draw fn. |
| `pygame-25` | info | Turn-flow QoL features: spectate mode, next-idle-unit hotkey, end-turn warning, hotseat hand-off | `reinforcetactics/app/input_handler.py:118` | M | (1) Tab / Shift-Tab cycles to the next unit that can still act, and centres the camera on it once the viewport exists. (2) Show a 'N units can still act, end turn?' confirmation (with a 'don't ask again' setting). |

## anim Animations, replay playback and video export

### `anim-1` — Video export leaks SDL_VIDEODRIVER=dummy into the process, so the main menu opens as an invisible window after exporting from the replay viewer

**high** · bug · confirmed · effort S · `reinforcetactics/utils/video.py:27`

Also: `reinforcetactics/ui/renderer.py:107`, `reinforcetactics/utils/replay_player.py:492`, `reinforcetactics/app/game_loop.py:433`, `reinforcetactics/cli/commands.py:252`

- **Impact.** On a normal desktop (SDL_VIDEODRIVER unset), exporting a replay video and then exiting the replay makes the app re-open its main menu on the dummy driver. No window is visible and the process appears to hang or vanish.
- **Fix.** In _ensure_headless_pygame, set the variable only when `not pygame.display.get_init()` and no display surface exists; otherwise leave the environment alone. Or save the previous value and restore it in a try/finally around the export. In Renderer.__init__ headless branch, set the variable only inside the `existing is None` branch before display.init(). Add a regression test that asserts os.environ is unchanged after record_replay_to_video when a display surface already exists.

### `anim-2` — Bot turns run synchronously inside the click/key handler with no rendering or event pumping: the board jumps, LLM turns freeze the window, and queued clicks hit the human's next turn

**high** · ux · confirmed · effort M · `reinforcetactics/app/input_handler.py:393`

Also: `reinforcetactics/app/input_handler.py:118`, `reinforcetactics/app/input_handler.py:157`, `reinforcetactics/app/game_loop.py:81`, `reinforcetactics/app/game_loop.py:101`, `reinforcetactics/core/game_state.py:649`

Same issue as: `pygame-6`, `aibots-5`, `pygame-11`

- **Impact.** The human never sees any bot action; the screen jumps from the end of their turn to the start of their next one. With LLM bots the window is unresponsive for tens of seconds (the OS shows 'Not Responding'). Clicks queued during the freeze are replayed afterwards at the current cursor position, so a second End Turn click can end the human's next turn unseen (confidence medium for that sub-case).
- **Fix.** Short term (S): after _process_bot_turns call pygame.event.clear(pygame.MOUSEBUTTONDOWN / KEYDOWN), and use event.pos in game_loop. Better (M): make bot turns visible and responsive. Add a per-action presentation hook, e.g. `GameState.action_listeners` invoked after the append at game_state.py:649. GameSession registers a callback that renders a frame, flips, pumps events and waits about 150-250 ms, so each bot action is shown. Run LLM bots in a worker thread and poll it from the main loop; the main loop only renders while the worker holds the state, and the worker must not touch pygame. This hook is also what the effects layer uses (see the effects-layer finding).

### `anim-3` — Team palette swap misses whole colour ramps: red/green/yellow Rogues are identical to blue, and Sorcerer hats and Archer trousers stay blue

**medium** · bug · confirmed · effort S · `reinforcetactics/constants.py:80`

Also: `reinforcetactics/ui/sprite_animator.py:204`, `assets/sprites/units/rogue_sheet.png`, `assets/sprites/units/sorcerer_sheet.png`, `assets/sprites/units/archer_sheet.png`

- **Impact.** Team identity for 3 of the 8 unit types depends only on the 2px border that _draw_unit_sprite draws. In 3-4 player games and at a glance, red, green and yellow Rogues, Sorcerers and Archers read as blue units.
- **Fix.** Add the missing base shades ((63,80,110), (72,135,194), (30,70,119), (40,115,176), (28,87,156)) with per-team replacements for players 1, 3 and 4. Put them in a unit-only list (e.g. UNIT_EXTRA_BASE_COLORS) passed to _generate_team_variants, so structure-tile recolouring (renderer.py:263-283) is unaffected unless verified. Leave spell-FX cyan (12,230,242) as is if it is intentional. Add a test asserting that no BASE colour survives in a non-blue team frame, and that the count of bluish pixels in team 1 frames is below a small threshold.

### `anim-8` — Replays of human (UI) games are padded twice and exports of small maps are mostly ocean, because set_map_metadata is never called (with a coupled latent sorcerer_pos bug)

**medium** · bug · confirmed · effort S · `reinforcetactics/app/game_loop.py:253`

Also: `reinforcetactics/core/game_state.py:345`, `reinforcetactics/core/game_state.py:631`, `reinforcetactics/core/game_state.py:1644`, `reinforcetactics/utils/replay_player.py:98`, `reinforcetactics/utils/video.py:367`

Same issue as: `core-19`, `aibots-18`, `persist-22`, `anim-18`, `consolidate-11`

- **Impact.** The in-game Export of a small-map human game yields a video in which the board fills about 5% of the frame. The replay window is larger than the game window, with a double ocean border. If someone fixes the padding by calling set_map_metadata, haste, defence_buff and attack_buff replays will look up the sorcerer at a doubly-offset tile and silently no-op, because replay_actions.py only acts `if sorcerer and target`.
- **Fix.** In start_new_game and load_saved_game use FileIO.load_map_with_metadata(..., for_ui=True) and call game.set_map_metadata(original_width, original_height, padding_offset_x, padding_offset_y, map_file, original_map_data). In the same PR add 'sorcerer_pos' to the tuple-coordinate list at game_state.py:631. Add a round-trip test: a UI-padded game with a haste action is saved, then replayed, and the haste is applied.

### `anim-10` — Replay seeking re-simulates from action 0 on every step back and on every scrub motion event

**medium** · performance · confirmed · effort M · `reinforcetactics/utils/replay_player.py:329`

Also: `reinforcetactics/utils/replay_player.py:402`, `reinforcetactics/utils/replay_player.py:609`, `reinforcetactics/utils/replay_player.py:322`

- **Impact.** Scrubbing and holding Left drop frames already at about 500 actions, and cost grows linearly (long LLM or RL games reach thousands of actions). Several motion events in one frame multiply the cost. Forward seeks needlessly restart from zero.
- **Fix.** (1) Coalesce scrubbing: record the latest motion x and seek once per frame in update(). (2) For forward seeks (target >= current index) execute only actions[current:target]. (3) Keep keyframe snapshots (e.g. GameState.to_dict or deepcopy every 25-50 actions, built lazily) and restore the nearest keyframe at or before the target, so backward seeks cost O(K).

#### Low and info findings (20)

| ID | Sev | Title | Location | Effort | Fix (summary) |
|---|---|---|---|---|---|
| `anim-4` | low | Animation clock is wall-clock time between render() calls, so video exports are non-deterministic and idle animation barely plays in them | `reinforcetactics/ui/renderer.py:380` | S | Give Renderer.render an optional `dt: float \| None = None`. When it is None, use time.perf_counter(); when it is given, use the provided value. In video.py pass dt=1/fps from every _capture_frame (the held extra frames can reuse the last frame). |
| `anim-5` | low | The OpenCV fallback never runs for ffmpeg failures: imageio starts ffmpeg lazily in append_data, outside the try block | `reinforcetactics/utils/video.py:596` | S | Probe the backend inside the try: call `self._writer.append_data(first_conformed_frame)` inside _lazy_init, and on any exception close the imageio writer and fall back to OpenCV (or raise a clear RuntimeError naming the install fix). |
| `anim-6` | low | The game-over overlay calls Surface.convert(), which raises when the headless renderer fell back to a plain Surface, so the whole recording fails at the last … | `reinforcetactics/utils/video.py:515` | S | Drop `.convert()`. The frombuffer surface is already 24-bit RGB and blitting the SRCALPHA band onto it works without a display format. Alternatively use `pygame.surfarray.make_surface(frame.swapaxes(0, 1))`. |
| `anim-7` ≈ pygame-17, consolidate-17 | low | Walking and path-transition animation is built but never wired up, and would misbehave if it were (no position interpolation, 0.8 s per tile, advancement tied … | `reinforcetactics/ui/sprite_animator.py:377` | M | Either delete the queue API, or finish it as part of the movement-tween opportunity. Make segments time-based (for example 0.15 s per tile via an explicit elapsed-per-segment timer, not frame wrap). |
| `anim-9` | low | Replay control and info panels are drawn over the board and hide the last real map row on 20x20 maps | `reinforcetactics/utils/replay_player.py:736` | S | Reserve the panel height instead of overlaying it. After the Renderer is built, call pygame.display.set_mode((w, h + 100)) and point renderer.screen at a board subsurface. Alternatively add a `bottom_margin` parameter to Renderer. |
| `anim-11` | low | Replay and video GameStates ignore recorded max_turns and fog_of_war | `reinforcetactics/utils/replay_player.py:88` | S | Pass max_turns=game_info.get('max_turns') (and enabled_units) in all three constructors. To support fog-of-war replays, also pass fog_of_war, add an explicit omniscient default for replays (a sentinel rather than None, because _get_fow_player maps None to … |
| `anim-12` | low | Animation state is keyed by id(unit) and never cleaned up: the dicts grow without bound and new units inherit dead units' state | `reinforcetactics/ui/sprite_animator.py:287` | S | Key by the engine's stable unit.unit_id (already used by replay v3). Prune entries whose ids were not drawn this frame: collect the drawn ids in _draw_units and drop the rest, or clear the animator state when renderer.game_state is reassigned … |
| `anim-13` | low | get_frame advances at most one frame per call regardless of dt | `reinforcetactics/ui/sprite_animator.py:315` | S | Use `steps, timer['current_time'] = divmod(timer['current_time'], frame_duration)` and advance `int(steps)` frames in a loop, calling _advance_movement_queue on each wrap (or better, make segment timing independent of frames, per the unwired-animation … |
| `anim-14` | low | Replay step_forward leaves a stale action description on screen | `reinforcetactics/utils/replay_player.py:312` | S | Add `self.current_action_description = self._get_action_description(action)` in step_forward. Better, route update(), step_forward and _replay_to_action through one `_apply(index)` helper. |
| `anim-15` | low | Replay playback timing drops overshoot and caps at one action per frame; 8x and 10x are nearly the same speed | `reinforcetactics/utils/replay_player.py:256` | S | Use time.perf_counter(). Accumulate `self.last_action_time += time_per_action` inside a `while` loop capped at N actions per frame, and reset the anchor on pause/seek (already done in toggle_pause/_replay_to_action). |
| `anim-16` | low | Sprite fallbacks and defaults: static sprites point at missing files (and would not be team-coloured), and bundled sprites are off by default | `reinforcetactics/ui/renderer.py:319` | S | Build static fallbacks from `animator.team_sheets[(type, player)]['idle'][0]` (or at least base idle[0]) instead of static_path, and remove static_path from UNIT_DATA. |
| `anim-17` | low | Dead animation, renderer and export APIs | `reinforcetactics/utils/file_io.py:555` | S | Delete export_replay_video, or make it a thin wrapper over video.record_replay_to_video. For the renderer passthroughs, either wire them (see the tween and replay max_turns/fog findings) or remove them in the same PR that adds the effects layer. |
| `anim-18` ≈ core-19, aibots-18, persist-22, anim-8, consolidate-11 | low | Map-padding logic now exists in three copies (ReplayPlayer, video export, FileIO) | `reinforcetactics/utils/replay_player.py:98` | S | Replace the body of _pad_map_for_replay with `df, ox, oy = FileIO._pad_map(df, MIN_MAP_SIZE, MIN_MAP_SIZE); df = FileIO.add_water_border(df, REPLAY_BORDER_SIZE); |
| `anim-19` | low | Duplicated recolour helper, and sprite scripts hard-code ANIMATION_CONFIG instead of importing it | `scripts/generate_unit_gifs.py:30` | S | Move the recolour into a module-level `recolor_surface(surface, base, palette)` in sprite_animator and reuse it in renderer. Add a `load_sprite(path, headless)` helper. Have generate_unit_gifs derive names, frames, crop and duration from constants. |
| `anim-20` | low | The 32x32 centre-crop cuts every sprite's drop shadow and some weapons | `reinforcetactics/ui/sprite_animator.py:180` | S | Support `crop_offset_y` (for example +3, since rows 16-17 are empty for most units) and per-unit crop overrides in ANIMATION_CONFIG['units']. Or crop 40x40 and blit centred so the sprite may overflow the tile (draw units in y order). |
| `anim-21` ≈ pygame-24, menus-22, tests-6 | low | No tests cover SpriteAnimator or renderer animation, and ui/* is excluded from coverage | `reinforcetactics/ui/sprite_animator.py:255` | S | Add tests/test_sprite_animator.py (SDL_VIDEODRIVER=dummy). Cover: all 8 sheets load 5 states with 4/8/8/8/8 frames; the move_right frame is a mirror of move_left; large-dt catch-up; the queue order right, down, idle for a path; |
| `anim-22` | info | Opportunity: an action-driven effects layer (floating damage/heal numbers, hit flash, death fade, capture flash, buff particles, turn banner) | `reinforcetactics/ui/renderer.py:390` | M | Add ui/effects.py `EffectsLayer` with on_action(action_dict), update(dt) and draw(screen), and call it in render() after _draw_units. Feed it by tailing game_state.action_history (live games; |
| `anim-23` | info | Opportunity: movement tweening and a procedural attack lunge (the sheets have no attack, hurt or death frames) | `reinforcetactics/core/unit.py:159` | M | (1) Add `came_from` tracking to get_reachable_positions, or a `find_path(unit, to, can_move)` helper in the renderer. Compute the path before move_unit mutates x/y, or in replay from the pre-action state, falling back to an L-shaped path when BFS fails (v3 … |
| `anim-24` | info | Opportunity: cheap idle polish (desynchronised idle phase, unit facing, bot-action focus reticle) | `reinforcetactics/ui/sprite_animator.py:305` | S | (1) Initialise the timer with `current_frame = unit_id % len(frames)` and `current_time = (unit_id * 0.037) % duration`. (2) Add 'idle_left': 'idle' to mirror_states and track facing per unit_id: set it from the last horizontal move or attack dx, defaulting … |
| `anim-25` | info | Opportunity: evaluation videos compress the opponent's entire turn into one frame | `reinforcetactics/utils/video.py:226` | S | After the episode, render the video from the saved replay via record_replay_to_video, which already emits a frame for every action including the opponent's. Keep step_stats from the live loop. |

## menus Menus, widgets, map editor, i18n and settings

### `menus-1` — Watch Replay crashes (TypeError) once any random-map replay exists

**critical** · bug · confirmed · effort S · `reinforcetactics/ui/menus/save_load/replay_selection_menu.py:111`

Also: `reinforcetactics/ui/menus/save_load/replay_selection_menu.py:133`, `reinforcetactics/ui/menus/save_load/load_game_menu.py:66`, `reinforcetactics/ui/menus/save_load/load_game_menu.py:88`, `reinforcetactics/ui/menus/save_load/load_game_menu.py:152`, `reinforcetactics/app/game_loop.py:250`

- **Impact.** Replays are auto-saved on every game over and on every mid-game quit, so after one random-map game, clicking "Watch Replay" crashes the whole application every time until the user deletes the file by hand.
- **Fix.** Use `map_file = game_info.get("map_file") or ""` and label an empty value "Random Map". Parse metadata defensively: check `isinstance(data, dict)`, catch `Exception` per file, and fall back to the minimal-metadata dict (already written at L134-147 / load_game_menu.py:153-174). Add a test that opens both menus over a random-map replay and a malformed JSON file.

### `menus-2` — New Game → '1v1v1' (bundled maps folder) crashes the app in PlayerConfigMenu

**critical** · bug · confirmed · effort S · `reinforcetactics/ui/menus/game_setup/game_mode_menu.py:31`

Also: `reinforcetactics/ui/menus/game_setup/player_config_menu.py:43`, `reinforcetactics/ui/menus/main_menu.py:82`, `reinforcetactics/app/game_loop.py:239`

Same issue as: `pygame-1`, `consolidate-6`

- **Impact.** A visible main-menu option crashes the game. Three-player maps ship in the repo but cannot be played from the GUI.
- **Fix.** Short term: whitelist the modes PlayerConfigMenu supports in GameModeMenu._load_modes. Better: derive num_players from the mode (or from the map's HQ count via MapPreviewGenerator metadata) and make PlayerConfigMenu accept 2 to 4 players. Also pass num_players explicitly to start_new_game instead of inferring it from `mode == "2v2"`.

### `menus-3` — Load Game makes the player pick the save twice (menu result is discarded)

**high** · bug · confirmed · effort S · `reinforcetactics/ui/menus/main_menu.py:100`

Also: `reinforcetactics/cli/commands.py:273`, `reinforcetactics/app/game_loop.py:315`, `reinforcetactics/ui/menus/save_load/load_game_menu.py:434`

Same issue as: `consolidate-10`

- **Impact.** Every load goes through the save picker twice, and a completed save shows the "Load anyway?" dialog twice. The mislabelled `save_path` key hides the problem.
- **Fix.** Rename the key to `save_data`. Change `load_saved_game(save_data=None)` to open the menu only when nothing was passed, mirroring `watch_replay(replay_path)`, and pass `menu_result["save_data"]` from commands.py. Add a test driving MainMenu._load_game → commands dispatch with a stubbed menu.

### `menus-5` — 'Custom Model' (ModelBot) can never be started: _validate_model always fails

**high** · bug · confirmed · effort M · `reinforcetactics/ui/menus/game_setup/player_config_menu.py:144`

Also: `reinforcetactics/ui/menus/game_setup/player_config_menu.py:135`, `reinforcetactics/ui/menus/game_setup/player_config_menu.py:161`, `reinforcetactics/ui/menus/game_setup/player_config_menu.py:181`, `reinforcetactics/game/model_bot.py:196`, `reinforcetactics/ui/menus/main_menu.py:82`

Same issue as: `aibots-2`

- **Impact.** Playing against a trained RL agent from the GUI, the main showcase of the RL pipeline, is broken for everyone.
- **Fix.** Pass the selected map path and fog_of_war into PlayerConfigMenu (MainMenu already has both) and validate against `FileIO.load_map(selected_map, for_ui=True)` with the chosen fog setting, or defer validation to game start. Accept `.zip` and `.pt`. Show the full error with wrap_text. Re-validate when the fog toggle changes. Add a regression test with a mocked `MaskablePPO.load`.

### `menus-4` — ConfirmationDialog: Enter confirms even when Cancel has keyboard focus

**medium (reviewer: high)** · bug · confirmed · effort S · `reinforcetactics/ui/menus/in_game/confirmation_dialog.py:34`

Also: `reinforcetactics/ui/widgets/dialog.py:155`, `reinforcetactics/ui/menus/in_game/pause_menu.py:116`, `reinforcetactics/ui/menus/main_menu.py:153`

- **Impact.** Keyboard users who deliberately pick Cancel still quit the game (MainMenu quit confirm), lose unsaved progress ("Return to Main Menu"), or load a finished game. These are the destructive actions the dialog exists to guard.
- **Fix.** Drop K_RETURN from the keymap and let Enter act on the focused button. If Enter-to-confirm with nothing focused is wanted, pre-focus the confirm button (selected_index = index of confirm). Add a test that sends RIGHT then RETURN and expects False.
- **Verifier note.** The mechanism is real. The keymap {K_RETURN: True, K_y: True, K_n: False} is checked at dialog.py:155-156, before the focus-aware Enter handling at L164-166, so Enter always resolves True no matter which button has focus. Only K_KP_ENTER respects focus. From the default -1, RIGHT/DOWN/TAB focuses Cancel (index 0).

### `menus-6` — Map editor discards unsaved edits on Esc/close; save and validation results only go to stdout

**medium (reviewer: high)** · ux · confirmed · effort S · `reinforcetactics/ui/menus/map_editor/map_editor.py:142`

Also: `reinforcetactics/ui/menus/map_editor/map_editor.py:90`, `reinforcetactics/ui/menus/map_editor/map_editor.py:240`, `reinforcetactics/ui/menus/map_editor/map_editor.py:136`

- **Impact.** A GUI user who presses Ctrl+S on an invalid map sees nothing happen, and one Esc throws away all their work. That is data loss of user-created content.
- **Fix.** Before leaving with `modified` set, show the existing QuitConfirmDialog (Save & Quit / Quit / Cancel). Show validation errors and the saved path on screen as a status line or Dialog. Remove or implement the Ctrl+N/Ctrl+O TODO branches.
- **Verifier note.** Every mechanism checks out. Esc sets running=False with no check of self.modified (L142-143). The QUIT branch has `if self.modified: # TODO ... pass` (L90-92). A validation failure in _save_map only prints (L241-245). Ctrl+N and Ctrl+O are empty TODO branches (L136-141). One overstatement: success is not purely print-only. After a save, draw() shows the filename in the title and clears the ' *' dirty marker (L317-320).

### `menus-7` — Map editor cannot pan horizontally, so columns past the canvas width are unreachable

**medium (reviewer: high)** · bug · confirmed · effort S · `reinforcetactics/ui/menus/map_editor/map_editor.py:216`

Also: `reinforcetactics/ui/menus/map_editor/editor_canvas.py:111`, `reinforcetactics/ui/menus/map_editor/editor_canvas.py:130`, `reinforcetactics/ui/menus/map_editor/map_editor_menu.py:75`

- **Impact.** Larger maps, which the New Map dialog explicitly allows (up to 100x100), cannot be fully edited.
- **Fix.** Handle `pygame.MOUSEWHEEL` (event.x for horizontal, Shift+wheel mapped to dx), arrow/WASD panning and middle-mouse drag via `canvas.handle_scroll(dx, dy)`. Clamp offsets after zoom. Consider a zoom-to-fit default.
- **Verifier note.** Real, with a small off-by-one. The only handle_scroll calls are (0, -32) and (0, 32) for buttons 4 and 5 (L216-220). grep finds no other caller and no other assignment to offset_x, and there is no MOUSEWHEEL, arrow-key or drag handling, so offset_x stays 0. MapEditorMenu passes the 900x700 menu surface, which makes the canvas 590x550 (map_editor.py:58-59).

### `menus-8` — Corrupt or partially invalid settings.json silently wipes all settings, including API keys; writes are non-atomic

**medium (reviewer: high)** · bug · confirmed · effort S · `reinforcetactics/utils/settings.py:52`

Also: `reinforcetactics/utils/settings.py:62`, `reinforcetactics/utils/settings.py:81`

Prior review: REVIEW_maintainability.md #22 (merge robustness; still one-level and type-unsafe)

- **Impact.** Users permanently lose stored API keys and every other preference with no warning beyond a stdout print.
- **Fix.** Write atomically (temp file in the same dir + `os.replace`). On a parse error, copy the bad file to `settings.json.bak` before any later save. Merge per key with type checks: keep defaults for sub-sections that are not dicts instead of discarding everything. Emit a logging warning instead of print.
- **Verifier note.** The mechanism is exactly as described. load() catches any exception and returns deepcopy(DEFAULT_SETTINGS) (L52-55). _merge_with_defaults does `result[key].update(value)` (L69) with no type check, so `"graphics": null` or `"paths": "x"` raises and throws away the whole file. save() is a plain `open(self.settings_file, "w")` + json.dump (L81-82), so the write is not atomic, and every set()/set_api_key()/toggle_unit() calls save().

### `menus-9` — LLM bots hidden in PlayerConfigMenu when keys come from environment variables

**medium (reviewer: high)** · bug · confirmed · effort S · `reinforcetactics/ui/menus/game_setup/player_config_menu.py:101`

Also: `reinforcetactics/ui/menus/settings/api_keys_menu.py:145`, `reinforcetactics/app/bot_factory.py:77`, `reinforcetactics/game/llm_bot.py:198`

- **Impact.** Users who follow the on-screen advice and export env vars never see OpenAI, Claude or Gemini in the bot cycle.
- **Fix.** Resolve availability as `settings.get_api_key(p) or os.getenv(ENV_VAR)`, reusing the bot classes' `_env_var_name`. Better: add a `has_api_key(provider)` helper in Settings or bot_factory and use it from both places.
- **Verifier note.** player_config_menu.py:100-104 builds available_llm_bots only from `bool(settings.get_api_key(...))`. _available_bot_types (L345) adds only the providers marked available, so env-var-only providers never show up in the cycle. The bots themselves do fall back to env vars: bot_factory.py:77-85 passes `settings.get_api_key(p) or None`, llm_bot.py:136 does `self.api_key = api_key or self._get_api_key_from_env()`, and _env_var_name is OPENAI_API_KEY/ANTHROPIC_API_KEY/GOOGLE_API_KEY (L1235/1304/1386).

### `menus-10` — Menu.run's blanket event.clear() swallows window-close in multi-step flows

**medium** · bug · confirmed · effort S · `reinforcetactics/ui/menus/base.py:622`

Also: `reinforcetactics/ui/menus/main_menu.py:75`, `reinforcetactics/ui/menus/map_editor/map_editor_menu.py:98`, `reinforcetactics/ui/menus/game_setup/player_config_menu.py:644`, `reinforcetactics/ui/menus/settings/api_keys_menu.py:396`, `reinforcetactics/ui/menus/map_editor/map_editor.py:85`

- **Impact.** Clicking the window's close button on the map picker, player config or the 1v1 edit picker just shows the previous screen. This is the bug drain_events() was written to fix, reintroduced by the base loop.
- **Fix.** Replace the clears with `drain_events()`, or have run() return immediately if `pygame.event.peek(pygame.QUIT)`. Move the loop into ScreenBootstrapMixin so all screens share one implementation (see the consolidation finding).

### `menus-11` — Game-over screen: Esc exits the whole app; 'Save Replay' navigates away and writes a duplicate

**medium** · bug · confirmed · effort S · `reinforcetactics/ui/menus/in_game/game_over_menu.py:45`

Also: `reinforcetactics/app/game_loop.py:205`, `reinforcetactics/app/game_loop.py:214`, `reinforcetactics/ui/menus/base.py:236`

- **Impact.** One stray Esc after a match closes the game. "Save Replay" silently bounces to the main menu and creates a second replay file.
- **Fix.** Override `_on_result` so `_save_replay` shows a "Replay saved to …" status and stays (or remove the option, since the replay is auto-saved). Map Esc to "main_menu" and suppress or override footer_hint. Render a draw banner when winner is falsy.

### `menus-12` — In-game 'S' save resizes the game window to 900x700; saves fail or overwrite silently

**medium** · bug · confirmed · effort S · `reinforcetactics/app/game_loop.py:163`

Also: `reinforcetactics/ui/menus/base.py:91`, `reinforcetactics/ui/menus/save_load/save_game_menu.py:53`, `reinforcetactics/utils/file_io.py:371`

Same issue as: `pygame-4`

- **Impact.** Pressing S (or Save & Quit via _handle_save_game) mid-game corrupts the game window layout (map-sized maps get cropped or offset), and failed or overwritten saves go unnoticed.
- **Fix.** Pass `self.renderer.screen` at game_loop.py:163. Consider making `screen` a required argument for sub-screens that are never top-level. In SaveGameMenu, strip path separators and illegal characters, confirm overwrite via ConfirmationDialog, and show the error or saved path on screen.

### `menus-14` — Map editor validation lets unplayable maps through and misfiles 3-player maps

**medium** · bug · confirmed · effort M · `reinforcetactics/ui/menus/map_editor/map_editor.py:294`

Also: `reinforcetactics/ui/menus/map_editor/map_editor.py:253`, `reinforcetactics/ui/menus/map_editor/tile_palette.py:92`, `reinforcetactics/ui/menus/map_editor/map_editor_menu.py:59`

- **Impact.** Users can save and play maps that softlock or break player counts; the editor gives false confidence.
- **Fix.** Extend validation: exactly one `h_N` per player and no bare `h`, at least one `b_N` per player, no structures owned by players above num_players, and a BFS over non-ocean/non-water tiles (or unit move costs) showing all HQs are mutually reachable. Save to `maps/{'1v1'|'1v1v1'|'2v2'}` by player count. Show the errors on screen (see the Esc/stdout finding).

### `menus-19` — Remaining UI duplication after the ScreenBootstrapMixin/ListDetailMenu refactor

**medium** · consolidation · confirmed · effort M · `reinforcetactics/ui/menus/base.py:607`

Also: `reinforcetactics/ui/menus/game_setup/player_config_menu.py:630`, `reinforcetactics/ui/menus/settings/api_keys_menu.py:385`, `reinforcetactics/ui/menus/map_editor/map_editor.py:75`, `reinforcetactics/ui/menus/in_game/unit_action_menu.py:119`, `reinforcetactics/ui/menus/in_game/unit_purchase_menu.py:56`, `reinforcetactics/ui/components/map_preview.py:230`

- **Impact.** Each copy has already drifted (e.g. only some loops preserve QUIT, only one overlay has keyboard navigation, the preview colours differ), and every fix has to be applied several times.
- **Fix.** Add `ScreenBootstrapMixin.run_loop()` with `handle_input` / `draw` / `_on_quit_event` / `_on_result` hooks, and port the 4 non-Menu loops to it. Extract `OverlayPopupMenu` (rect placement, overlay, header, close button, arrow/number navigation). Add `render_tile_grid(grid2d, size, units=None)` plus `pretty_map_name(path)` in components/map_preview.py. Add a `TextPromptScreen(Menu)`. Let NewMapDialog reuse the base option drawing.

### `menus-20` — Map previews and the editor canvas use per-cell DataFrame.iloc (650 ms menu stall; 21-68 ms editor frames)

**medium** · performance · confirmed · effort M · `reinforcetactics/ui/components/map_preview.py:253`

Also: `reinforcetactics/ui/components/map_preview.py:136`, `reinforcetactics/ui/menus/game_setup/map_selection_menu.py:79`, `reinforcetactics/ui/menus/map_editor/editor_canvas.py:199`

- **Impact.** A visible freeze when entering map selection, and a sluggish editor on larger maps.
- **Fix.** Convert once with `grid = map_data.to_numpy(dtype=str)` and iterate the ndarray, or build previews via `pygame.surfarray` / a colour lookup table and `pygame.transform.scale`. Parse each CSV once and derive every thumbnail size from one rendered base surface. Cache metadata separately from surfaces, keyed on mtime. In the editor, keep a numpy mirror of the tiles or pre-render a tile surface updated only on paint.

### `menus-22` — UI flows are excluded from coverage; none of the crash/regression paths above are tested

**medium** · test-gap · confirmed · effort M · `pyproject.toml:139`

Also: `tests/test_menus.py`, `tests/test_map_editor.py`

Same issue as: `pygame-24`, `tests-6`, `anim-21`

- **Impact.** Crashes in the main navigation paths shipped unnoticed, and the 75% coverage figure overstates UI safety.
- **Fix.** Include ui/menus, ui/widgets and ui/components in coverage (they run fine under the dummy driver) with a separate lower threshold. Add regression tests for the findings above, plus a smoke test that walks MainMenu → every sub-menu via posted events.

#### Low and info findings (11)

| ID | Sev | Title | Location | Effort | Fix (summary) |
|---|---|---|---|---|---|
| `menus-13` ≈ consolidate-21 | low | i18n: 34 referenced keys exist in no language, es/zh miss all 34 map_editor keys, ~50 hardcoded English literals | `reinforcetactics/utils/language.py:519` | M | Add the 34 missing keys to all five tables and route the literals through `lang.get`. Add a unit test asserting identical key sets across languages and that every `lang.get("x.y")` literal in the source exists in English (the scratch script is ~40 lines). |
| `menus-15` | low | Edit Existing Map: confusing two-picker flow, 1v1v1 not editable, in-place overwrite shrinks bundled maps | `reinforcetactics/ui/menus/map_editor/map_editor_menu.py:94` | M | Use one `MapSelectionMenu(game_mode=None)` that scans all mode folders, including 1v1v1 (currently hardcoded to ['1v1','2v2'] at map_selection_menu.py:61). Add a filename prompt (TextInput) for first save and Save As, and confirm before overwriting. |
| `menus-16` | low | List/detail screens: hover overrides keyboard selection, and the wheel leaves hover stale | `reinforcetactics/ui/menus/list_detail.py:150` | S | On keyboard navigation set `hover_index = -1`. On wheel events recompute hover from `pygame.mouse.get_pos()` against the refreshed rects. Alternatively track a single 'active' index updated by whichever input moved last. |
| `menus-17` | low | Map editor Ctrl tracking via KEYDOWN/KEYUP flag can stick and swallow all shortcuts | `reinforcetactics/ui/menus/map_editor/map_editor.py:129` | S | Drop the flag and test `event.mod & (pygame.KMOD_CTRL \| pygame.KMOD_META)` on each KEYDOWN, as TextInput already does (text_input.py:66). |
| `menus-18` ≈ prior-19 | low | API keys stored in plaintext in a world-readable settings.json and shown unmasked | `reinforcetactics/utils/settings.py:81` | S | Create the file with 0600 (`os.open(..., 0o600)` in the atomic-write helper) or store keys via the `keyring` package with the JSON as a fallback. Keep the field masked while editing, with a show/hide toggle, and mask regardless of length. |
| `menus-21` | low | TextInput/clipboard: no IME/TEXTINPUT support, AltGr characters blocked, deprecated scrap API | `reinforcetactics/ui/widgets/text_input.py:74` | M | Consume `pygame.TEXTINPUT` events for insertion (call `pygame.key.start_text_input()` and `set_text_input_rect` while focused) and keep KEYDOWN only for editing keys. Add a cursor index with Left/Right/Home/End/Delete. |
| `menus-23` | low | Replay picker lists non-replay JSON and fully parses every file on open | `reinforcetactics/ui/menus/save_load/replay_selection_menu.py:60` | S | Accept files by schema (`isinstance(data, dict) and "game_info" in data and "actions" in data`, or `replay_schema_version`). Take the search dirs from Settings paths. Lazy-load metadata for visible rows, or cache it keyed by mtime. |
| `menus-24` | low | APIKeysMenu: synchronous network test can freeze the window; inputs are not keyboard-reachable | `reinforcetactics/ui/menus/settings/api_keys_menu.py:312` | S | Pass `timeout=10, max_retries=0` (and the equivalent for google-genai) and run the test in a `threading.Thread`, polling a result each frame. Make Tab cycle `active_input` through providers. Import `_default_model_name` from the bot classes. |
| `menus-25` ≈ rulebots-10 | low | PlayerConfigMenu hardcodes bot types and names outside bot_registry (MasterBot not offered) | `reinforcetactics/ui/menus/game_setup/player_config_menu.py:344` | S | Build the list from the registry (a GUI-visible subset flag plus display names on the registry entries) and add MasterBot. Delete the dead branch and the unused attributes. |
| `menus-26` | low | Settings paths/video/audio sections are dead config; menus hardcode relative dirs; fullscreen not persisted; footer hint goes stale | `reinforcetactics/utils/settings.py:16` | S | Either wire `get_settings().get_path(...)` into the menus and FileIO or delete the unused sections. Persist fullscreen and apply it at startup, checking toggle_fullscreen's return value. Refresh `footer_hint` in `_refresh_options`. |
| `menus-27` | info | Map editor: add undo/redo, fill tool, Save As and an on-screen validation panel | `reinforcetactics/ui/menus/map_editor/editor_canvas.py:101` | M | Record (x, y, old, new) per stroke, grouping on mouse release, onto undo/redo stacks bound to Ctrl+Z/Ctrl+Y. Add flood fill (BFS on equal tiles) and rectangle tools. |

## persist Save/load, replays, tournament, CLI and cloud

### `persist-1` — Tournament resume replays everything outside multi-map 'all' mode, and the resumed results never include the games already played

**high** · bug · confirmed · effort M · `reinforcetactics/tournament/schedule.py:246`

Also: `reinforcetactics/tournament/schedule.py:172`, `reinforcetactics/tournament/runner.py:113`, `reinforcetactics/tournament/runner.py:154`, `docker/tournament/run_tournament.py:337`

- **Impact.** Resuming an interrupted LLM tournament (the documented reason --resume exists) either pays for every API game again, or produces a results JSON/CSV with empty or partial standings that overwrites the real outcome.
- **Fix.** Give every scheduled game a stable key: (map_stem, sorted pair, side, repetition index). Filter pending games by that key in ONE place, after building the full schedule for any mode. Write the key into replay game_info. On resume, feed the recovered games back through `results.add_game_result` in schedule order before running the new ones. Better still, write an append-only results.jsonl, one line per finished game, and rebuild TournamentResults from it on resume. Add a test: run, resume, and check that the totals equal an uninterrupted run.

### `persist-2` — Setup failures abort the whole tournament (sequential) or vanish silently (concurrent), and in-game errors count as draws

**high** · bug · confirmed · effort S · `reinforcetactics/tournament/runner.py:309`

Also: `reinforcetactics/tournament/runner.py:238`, `reinforcetactics/tournament/runner.py:403`, `reinforcetactics/tournament/results.py:228`

Prior review: REVIEW_advancedbot.md §18 (error path in the concurrent loop)

- **Impact.** A single incompatible model or missing LLM key kills a long run with no results. Crashes inside games move Elo as if they were draws.
- **Fix.** Move setup inside the try, or wrap it in its own try, and always return a GameResult with `error` set. Record errored games with a distinct status and skip them in bot_stats and Elo (report them in a separate 'errors' column). In the concurrent path, build an error GameResult from the ScheduledGame. Add a pre-flight step in `run()` that builds every (model bot, map) pair once and rejects incompatible pairs before game 1.

### `persist-3` — Replay playback re-runs the economy under default constants: engine_overrides are never recorded, and max_turns/enabled_units are not restored

**medium (reviewer: high)** · bug · confirmed · effort M · `reinforcetactics/utils/replay_player.py:88`

Also: `reinforcetactics/utils/replay_player.py:287`, `reinforcetactics/utils/video.py:374`, `reinforcetactics/core/game_state.py:1683`, `reinforcetactics/tournament/runner.py:490`

- **Impact.** Replays of balance-sweep or engine_overrides games, and every stored v3 replay once constants.py changes (the repo does this often), silently diverge. Draw replays never reach a game-over state.
- **Fix.** Write `engine_overrides`, `enabled_units`, `fog_of_war`, `max_turns` and `run_config._full_engine_constants_hash()` into every replay's game_info. Build the playback GameState from them in one shared factory (`replay_actions.make_replay_state(game_info, map)`) used by ReplayPlayer and video. Longer term (schema v4), record `gold_after` on create_unit and end_turn and apply it like the v2 HP outcomes, so replays no longer depend on economy constants.
- **Verifier note.** Confirmed. ReplayPlayer (88, 287) and video.record_replay_to_video (374) build `GameState(map, num_players=...)` with no engine_overrides and no max_turns. Neither save_replay_to_file (game_state.py:1683-1706) nor runner._save_replay (490-537) records engine_overrides. execute_replay_action re-runs create_unit and end_turn on the engine (replay_actions.py:305-315, 402-407).

### `persist-4` — Replays of games loaded from a save (including the shipped scenarios) contain no starting state and are unwatchable

**medium (reviewer: high)** · bug · partially · effort M · `reinforcetactics/core/game_state.py:1644`

Also: `reinforcetactics/app/game_loop.py:330`, `reinforcetactics/app/game_loop.py:206`

- **Impact.** Every scenario game (the bundled demo content) gives a replay/video of an empty board with actions that silently fail.
- **Fix.** When a game starts from a save, write an `initial_state` snapshot (to_dict() at load time, without action_history) into game_info, and have playback build the state with `GameState.from_dict(initial_state, map)` when it is present. Alternatively, refuse to auto-save replays for loaded games until this exists. Add a round-trip test: load a scenario, play, save the replay, replay it, compare checksums.
- **Verifier note.** True for the bundled scenarios, overstated as 'games loaded from a save'. Ordinary in-game saves persist action_history from turn 0 (to_dict, game_state.py:1458), and from_dict restores it (1790). So a normal save/load game still produces a complete replay from the empty starting map. What breaks is any save authored without history, which means all 8 saves/*_scenario.json (action_history is empty; they contain units, tiles and gold).

### `persist-5` — 'cycle' map-pool mode (the CLI default) puts the two mirror games of every matchup on different maps, tying side to map

**medium (reviewer: high)** · rl-correctness · confirmed · effort S · `reinforcetactics/tournament/schedule.py:254`

Also: `reinforcetactics/tournament/schedule.py:300`, `scripts/tournament.py:56`

- **Impact.** First-move advantage and map asymmetry mix into every head-to-head result in multi-map CLI tournaments (for example `--map-dir maps/1v1`).
- **Fix.** In cycle and random modes, pick one map per (matchup, repetition) and play both sides on it. Draw random maps from `random.Random(config.rng_seed)`. Add a schedule test checking that each (pair, map) has equal P1 counts for both bots.
- **Verifier note.** Mechanism confirmed. bot1-as-P1 games use map index map_idx+game_num (254) and bot2-as-P1 games use map_idx+games_per_side+game_num (271), so mirror games land on different maps unless len(maps) divides games_per_side. `random` mode uses the global unseeded random.choice (300). 'Every head-to-head result' is overstated. With 2 maps and games_per_side=2 (the CLI default gps) every (pair, map) cell is balanced, and the CLI default of a single map is unaffected.

### `persist-6` — rng_seed does not seed the engine RNG, so seeded tournaments are not reproducible (Rogue evade uses the global random module)

**medium** · bug · confirmed · effort S · `reinforcetactics/tournament/runner.py:310`

Also: `reinforcetactics/core/mechanics.py:429`, `reinforcetactics/tournament/config.py:65`

Same issue as: `core-10`

- **Impact.** Balance-analysis tournaments that depend on seeded reproducibility get different trajectories on every run, whenever Rogues are enabled (the default).
- **Fix.** Derive a third per-game seed in `_make_per_game_rngs` (for example `p0|engine`) and pass `rng=random.Random(seed)` to GameState. When rng_seed is None, still pass a per-game Random so threads never share state. Extend the existing same-seed test to a Rogue-enabled map.

### `persist-7` — Elo depends on game order: it varies between concurrent runs, and single-map schedules are not interleaved

**medium** · rl-correctness · confirmed · effort M · `reinforcetactics/tournament/runner.py:224`

Also: `reinforcetactics/tournament/runner.py:224`, `reinforcetactics/tournament/schedule.py:246`, `reinforcetactics/tournament/elo.py:75`

- **Impact.** The published Elo standings (tournament_results) are not a function of the games played. Close ranks can flip between reruns.
- **Fix.** Buffer each round's results and apply Elo in game_id order. Better: compute final ratings in one batch pass (Bradley-Terry/Elo MLE over all games, or the average of sequential Elo over N random permutations) and report bootstrap confidence intervals next to win rates. Keep the online history only for plotting.

### `persist-8` — The resume scan counts every replay twice, and renumbered game_ids reuse the seeds of games already played

**medium** · bug · partially · effort S · `docker/tournament/run_tournament.py:149`

Also: `reinforcetactics/tournament/schedule.py:169`, `reinforcetactics/tournament/runner.py:269`

- **Impact.** Resumed tournaments play too few games, repeat already-played seeded games instead of new samples, and cannot reproduce an uninterrupted run.
- **Fix.** De-duplicate with a set of resolved paths, or scan only the configured replay_dir. Assign game_id from the full, uncompleted schedule (stable key), and skip completed keys without renumbering.
- **Verifier note.** Double counting is real. search_paths = [resume_path, resume_path/'replays', resume_path/'output'/'replays'], and each is rglob'd with no de-duplication. The README's documented usage (`--resume /app/output`, README.md:228, 375) hits exactly this case, because the default replay_dir is /app/output/replays. With games_per_side>=2, _get_pending_games (schedule.py:329-343) then reports 0 pending after one game per side.

### `persist-11` — The save-game name is used unsanitised in a path: traversal, silent overwrite, Windows-invalid characters

**medium** · security · confirmed · effort S · `reinforcetactics/ui/menus/save_load/save_game_menu.py:54`

Also: `reinforcetactics/utils/file_io.py:371`

- **Impact.** Local data loss (settings or other saves) and silent save failures on Windows. This is not a remote exposure.
- **Fix.** Add `FileIO.safe_filename(name)`: strip path separators and `<>:"/\\|?*`, reject reserved names, and resolve the result, asserting it stays inside the saves dir. Ask before overwriting an existing file. Show save failures in the UI.

### `persist-12` — Bot names are not validated: '|' crashes the export after the whole tournament, commas corrupt the CSVs, duplicate names merge

**medium** · bug · confirmed · effort S · `reinforcetactics/tournament/results.py:271`

Also: `reinforcetactics/tournament/results.py:199`, `reinforcetactics/tournament/results.py:398`, `reinforcetactics/tournament/runner.py:435`

Prior review: REVIEW_maintainability.md §29 (CSV export doesn't escape values)

- **Impact.** Lost results at the very end of long runs, and malformed CSVs fed into notebooks.
- **Fix.** Use tuple keys (frozenset or sorted tuple) instead of joined strings. Write the CSVs with the `csv` module. In validate()/run(), reject duplicate names and slugify names for filenames (keep display names in JSON).

### `persist-13` — Legacy v1 replays silently diverge, and replay errors are swallowed with no divergence check

**medium** · bug · confirmed · effort M · `reinforcetactics/utils/replay_actions.py:411`

Also: `reinforcetactics/utils/replay_player.py:248`, `tournament_results/0.1.1/replays`

- **Impact.** Users watch plausible but wrong replays with no warning. Balance notebooks re-simulating old replays get corrupted statistics.
- **Fix.** Track failed or no-op actions in execute_replay_action (return a status). Add `verify_replay(replay) -> first_divergent_index | None` that compares the recorded checksums at the end. Show a 'replay diverged at action N (recorded with vX, engine hash Y)' banner in ReplayPlayer and fail loudly in video export. Add a strict mode for tests. Label v1 replays best-effort.

### `persist-14` — Two different game_info layouts for the same replay_schema_version 3

**medium** · consolidation · confirmed · effort M · `reinforcetactics/tournament/runner.py:489`

Also: `reinforcetactics/core/game_state.py:1683`, `docker/tournament/run_tournament.py:172`

- **Impact.** Any checksum verifier, resume scanner or analysis notebook has to special-case the producer. Fields such as enabled_units exist in only one variant.
- **Fix.** Move game_info construction into one `build_replay_game_info(game_state, *, extra)` in replay_actions (or a new replay_schema module). The runner passes only tournament extras (bot names, capabilities). Bump to schema 4 with a documented TypedDict and a loader that normalises v1-v3.

### `persist-15` — The Docker runner re-implements config, bot parsing and GCS upload, and has drifted: rng_seed dropped, unknown bot types (incl. 'master') silently skipped

**medium** · consolidation · confirmed · effort S · `docker/tournament/run_tournament.py:213`

Also: `docker/tournament/run_tournament.py:48`, `docker/tournament/run_tournament.py:293`, `reinforcetactics/tournament/config.py:295`, `docker/tournament/config.schema.json`, `docker/tournament/Dockerfile:55`

Same issue as: `consolidate-18`, `tests-13`

- **Impact.** Docker tournaments can't be seeded, can't include MasterBot, and behave differently from the library given the same config file.
- **Fix.** Replace lines 213-308 with `TournamentConfig.from_dict(raw)` plus a shared `parse_bots_from_config` (add 'master' via the bot registry and raise on unknown types). Import `reinforcetactics.cloud.GCSUploader`. Add rng_seed, enabled_units and master to config.schema.json and validate the config against the schema at startup.

### `persist-16` — scripts/tournament.py fails on its default invocation (nonexistent default map) and has no way to seed

**medium** · bug · confirmed · effort S · `scripts/tournament.py:102`

Also: `reinforcetactics/tournament/__init__.py:29`, `reinforcetactics/tournament/runner.py:54`

Same issue as: `consolidate-7`

- **Impact.** The documented entry point is broken out of the box, and CLI tournaments double-count identical games.
- **Fix.** Default to maps/1v1/beginner.csv. Add --seed (passed to rng_seed), --enabled-units and --concurrent validation. Fix the docstrings. Add a smoke test running main() with --no-llm --no-models --games-per-side 1 --max-turns 5.

### `persist-17` — The CLI always exits 0, its train/evaluate modes are stale (DQN crashes, evaluate uses a different random map), and it has no argument validation

**medium** · bug · partially · effort M · `reinforcetactics/cli/main.py:107`

Also: `reinforcetactics/cli/commands.py:91`, `reinforcetactics/cli/commands.py:181`, `reinforcetactics/cli/commands.py:217`, `reinforcetactics/cli/main.py:37`

Same issue as: `pygame-10`, `consolidate-5`

- **Impact.** Scripts and CI treat failures as success. The CLI training path produces models unrelated to the maintained pipeline.
- **Fix.** Have each mode return an int and `sys.exit(rc)`. Use argparse type checks (positive int). Remove dqn/a2c, or route train to scripts/train/train_bootstrap.py and MaskablePPO. Pass map_file to evaluate and load models through ModelBot. Move ensure_directories after parsing, and only for modes that write.
- **Verifier note.** Mostly real, but 'always exits 0' is overstated. Handled errors in the mode handlers `return` and main prints '✅ Done!' with exit 0. Unhandled exceptions still exit 1: the DQN construction (commands.py:92-94, outside any try) and `wins / args.episodes` (217) both give a traceback and exit 1. check_dependencies failure also calls sys.exit(1).

### `persist-18` — Map data problems: the only 2v2 map leaves players 3 and 4 with nothing, and scenario-only maps without barracks sit in maps/1v1

**medium** · bug · confirmed · effort S · `maps/2v2/beginner.csv:1`

Also: `reinforcetactics/core/tile.py:36`, `reinforcetactics/tournament/config.py:272`, `scripts/tournament.py:92`

Same issue as: `core-4`

- **Impact.** UI 2v2 mode is unplayable for players 3 and 4. Directory-based tournaments get many meaningless draws.
- **Fix.** Rewrite the 2v2 map with per-player codes (h_1..h_4, b_1..b_4) until teams are implemented. Move scenario maps to maps/scenarios/ (or exclude maps with no barracks in add_maps_from_directory). Add `FileIO.validate_map(path, expected_players)` (rectangular, known codes, one HQ per player, at least one barracks per player) and a parametrised test over maps/**/*.csv.

### `persist-19` — GCS upload retries client creation for every file on credential failure, reports failure like 'nothing to upload', and only uploads at the end

**medium** · design · partially · effort S · `reinforcetactics/cloud/storage.py:107`

Also: `reinforcetactics/cloud/storage.py:122`, `reinforcetactics/cloud/storage.py:112`, `scripts/train/train_bootstrap.py:343`

- **Impact.** Training artifacts can be lost silently on ephemeral runners. Big output trees generate thousands of slow credential probes and log lines.
- **Fix.** Initialise the client once, outside the per-file loop. On failure raise, or return a result object with `failed` and `errors`, and stop the loop. Warn when credentials_file is missing. Add a periodic sync (SB3 callback or timer using the existing manifest support) and have scripts return non-zero when a configured upload fails.
- **Verifier note.** The per-file retry is real. _get_bucket (107-119) builds storage.Client() inside upload_file's try/except (121-132). When construction raises, _client stays None and every later file retries, each logging a warning. sync_directories returns {} and upload_tree returns 0 in that case, the same values as 'nothing configured'. A missing credentials_file falls back to default credentials, logged only at INFO (112-117). One claim is wrong: train_bootstrap._maybe_upload is not the only caller.

### `persist-20` — concurrent_games uses threads, so CPU-bound scripted and model games get no speedup; ModelBot reloads its checkpoint every game

**medium** · performance · confirmed · effort M · `reinforcetactics/tournament/runner.py:221`

Also: `reinforcetactics/tournament/bots.py:251`, `reinforcetactics/game/model_bot.py:102`

- **Impact.** Ladders with scripted and RL bots (the ROADMAP's main goal) run at single-core speed no matter the setting.
- **Fix.** Use ProcessPoolExecutor when no LLM bot is in the round (each game is independent and results are small), keep threads for I/O-bound LLM games, and add a per-process LRU cache of loaded SB3 policies keyed by (path, mtime).

#### Low and info findings (8)

| ID | Sev | Title | Location | Effort | Fix (summary) |
|---|---|---|---|---|---|
| `persist-9` ≈ core-6 | low | Save format drops max_turns, end_reason and winning_action_index, and has no version field | `reinforcetactics/core/game_state.py:1436` | S | Write the missing fields plus `save_format_version` and `library_version` in to_dict. Add a small migration step in from_dict (v0 -> v1: assign unit_ids and next_unit_id). |
| `persist-10` | low | Non-atomic JSON writes leave truncated files on errors and can destroy the previous save | `reinforcetactics/utils/file_io.py:377` | S | Add `FileIO.write_json_atomic(path, data)`: serialise to a string first with `default=` handling numpy scalars and tuples, write to `path.with_suffix('.tmp')`, fsync, then `os.replace`. |
| `persist-21` | low | run_config's git provenance depends on the current directory, and 'dirty' ignores untracked files | `reinforcetactics/utils/run_config.py:27` | S | Pass `cwd=Path(__file__).resolve().parents[2]` and use `git status --porcelain` for dirty. Fall back to importlib.metadata's version plus an optional `RT_GIT_SHA` env var baked into Docker images. |
| `persist-22` ≈ core-19, aibots-18, anim-8, anim-18, consolidate-11 | low | Replay coordinate frames are inconsistent: padding metadata is never set, sorcerer_pos is never converted, the attack helper assumes 2 players | `reinforcetactics/core/game_state.py:631` | S | Add 'sorcerer_pos' to the conversion list, or better, convert every `*_pos` key. Either call set_map_metadata from start_new_game and load_saved_game or delete the translation machinery. Use `target.player` when the target is found. |
| `persist-23` | low | The map editor crops and overwrites shipped maps on save, and puts 3-player maps in maps/2v2 | `reinforcetactics/utils/file_io.py:544` | S | Record the padding offsets from load_map_with_metadata and strip exactly those, instead of every ocean edge. Default to Save As for files under version control. Map player counts to their directories (2->1v1, 3->1v1v1, 4->2v2). |
| `persist-24` | low | FileIO/replay cleanup: dead helpers, duplicated padding, a random-map fallback for replays, print-and-swallow errors | `reinforcetactics/utils/file_io.py:555` | S | Delete the stub and unused helpers. Have ReplayPlayer and video call FileIO._pad_map/add_water_border. Fall back to loading the recorded map path and raise if it is missing. Switch FileIO to logging and raise typed errors (SaveFormatError) that the UI catches. |
| `persist-25` | low | LLM tournament plumbing: colliding conversation-log session IDs, a per-game (not per-call) API delay, a no-op Anthropic key test | `reinforcetactics/tournament/runner.py:306` | S | Add game_id to session_id. Move the delay into LLMBot._call_llm, or rename it. Make the Anthropic test call `models.list()`. Count forfeited turns per game in GameResult and mark games with forfeits above a threshold as errors. |
| `persist-26` | info | Opportunity: a versioned replay/save schema with built-in integrity checking, plus end-to-end determinism tests over real tournaments | `tests/test_replay_determinism.py:36` | M | (1) Add `reinforcetactics/utils/replay_schema.py` with `CURRENT_VERSION`, `build_game_info`, `normalize(game_info)` for v1-v3, and `verify(replay) -> Divergence \| None` using the recorded checksums. |

## consolidate Cross-cutting consolidation, dead code and repo hygiene

### `consolidate-5` — CLI `train` is an outdated third training pipeline: DQN crashes and `--opponent self` trains against a passive opponent

**high** · bug · confirmed · effort M · `reinforcetactics/cli/commands.py:92`

Also: `reinforcetactics/cli/main.py:68`, `reinforcetactics/cli/commands.py:31`, `reinforcetactics/rl/gym_env.py:1178`, `README.md:84`

Same issue as: `pygame-10`, `persist-17`

- **Impact.** Of the README's first three training commands, the self-play one silently trains against a do-nothing opponent, and `--algorithm dqn` crashes on construction. The CLI keeps drifting from scripts/train and rl/bootstrap.
- **Fix.** Have `train` delegate to the config pipeline (e.g. `--config configs/ppo/maskable_ppo.yaml` with MaskablePPO and masking). Build DQN with `action_space_type='flat_discrete'` or drop it. Route `--opponent self` through `make_self_play_env` or remove the choice. Delete the `rl_gym_env` fallbacks and the sys.path hack.

### `consolidate-6` — GUI offers a 1v1v1 mode that crashes PlayerConfigMenu; 2v2 'team' info is parsed and then ignored

**high** · bug · partially · effort M · `reinforcetactics/ui/menus/game_setup/player_config_menu.py:43`

Also: `reinforcetactics/ui/menus/game_setup/game_mode_menu.py:28`, `reinforcetactics/ui/menus/main_menu.py:82`, `reinforcetactics/ui/menus/game_setup/map_selection_menu.py:61`, `reinforcetactics/ui/menus/map_editor/map_editor.py:253`, `reinforcetactics/core/tile.py:38`

Same issue as: `pygame-1`, `menus-2`

- **Impact.** Choosing an offered game mode crashes the game. The README advertises '2v2 (team) maps', but they play as a 4-player free-for-all.
- **Fix.** Share one table, MODE_PLAYERS = {'1v1': 2, '1v1v1': 3, '2v2': 4}, across the four sites, or derive the player count from the HQ owners in the map. Either implement teams (ally checks in mechanics and win conditions) or label 2v2 as FFA and drop `Tile.team`.
- **Verifier note.** The crash is real. GameModeMenu._load_modes lists every maps/ subfolder with CSVs, giving ['1v1', '1v1v1', '2v2']. MapSelectionMenu(game_mode='1v1v1') loads the four 1v1v1 maps, then MainMenu._new_game step 3 calls PlayerConfigMenu(game_mode='1v1v1'), which raises ValueError at line 43. Neither the menu base nor cli.commands.play_mode (which catches only ImportError) handles it, so the game exits with a traceback.

### `consolidate-1` — Fog of war shows live owner and HP of structures on shrouded tiles; last-seen memory is written but never read

**medium (reviewer: high)** · rl-correctness · partially · effort M · `reinforcetactics/core/game_state.py:1545`

Also: `reinforcetactics/core/visibility.py:159`, `reinforcetactics/core/visibility.py:252`, `reinforcetactics/ui/renderer.py:444`, `reinforcetactics/rl/observation.py:209`

Same issue as: `core-5`, `rlenv-14`, `critic-integration-3`, `pygame-12`

- **Impact.** Agents trained with fog of war see enemy captures and structure damage anywhere they have ever explored, which inflates FOW results. GUI players can see enemy ownership through fog, contradicting the README claim that 'buildings and towers are hidden until scouted'.
- **Fix.** In `GameState.to_numpy(for_player)`, fill channels 1-2 of SHROUDED structure tiles from `vis_map.last_seen_structures` (owner/health at turn_seen). Zero owner/HP for UNEXPLORED structures. In `Renderer._draw_tile`, use the last-seen owner when `vis_state != VISIBLE`. Add the repro above as a regression test. If last-seen semantics are not wanted, delete the write-only `_update_memory` and the `last_seen_*` code (~60 LOC) instead.
- **Verifier note.** The RL observation leak is real. In GameState.to_numpy(for_player), the FOW block (1537-1548) zeroes grid channels only when `visibility_state[y, x] == 0` (UNEXPLORED). SHROUDED tiles pass through the live tile owner and HP. VisibilityMap._update_memory writes last_seen_structures and last_seen_units (visibility.py:159-192).

### `consolidate-2` — LLM bots cannot run HASTE/DEFENCE_BUFF/ATTACK_BUFF, but their prompts advertise them (with outdated numbers)

**medium (reviewer: high)** · bug · confirmed · effort M · `reinforcetactics/game/llm_bot.py:957`

Also: `reinforcetactics/game/llm_bot.py:802`, `reinforcetactics/game/llm_bot.py:909`, `reinforcetactics/game/llm_prompts.py:55`, `reinforcetactics/game/llm_prompts.py:74`, `reinforcetactics/game/llm_prompts.py:175`, `reinforcetactics/constants.py:304`

Same issue as: `aibots-9`

- **Impact.** An LLM bot that buys a Sorcerer (350 gold) gets a unit that can only attack; every buff it tries is silently dropped. LLM-vs-X tournament results published on the docs site are biased, and the prompts contradict each other and the engine.
- **Fix.** Add the three Sorcerer actions to the legal-action serialization, the format block and the dispatcher, ideally through the shared action applier proposed in the dispatch finding. Generate the unit-stat and ability text in prompts from UNIT_DATA and constants instead of hand-written literals, so prompts also stay correct under engine_overrides. Add a test asserting that every action type named in each registered prompt has a dispatcher branch.
- **Verifier note.** _execute_actions (llm_bot.py:957-978) dispatches only CREATE_UNIT, MOVE, ATTACK, PARALYZE, HEAL, CURE, SEIZE, END_TURN and RESIGN. Everything else hits `logger.warning("Unknown action type: %s", action_type)`. _format_legal_actions (782-873) has no haste/defence_buff/attack_buff keys, and the response-format block (909-918) omits them too, with the example `"unit_type": "W|M|C|A"`. grep finds no 'haste' or 'buff' anywhere in llm_bot.py.

### `consolidate-3` — Bot attack valuation re-implements damage without defence reduction or buffs, so it predicts kills that don't happen

**medium (reviewer: high)** · bug · confirmed · effort M · `reinforcetactics/game/bot.py:1095`

Also: `reinforcetactics/core/mechanics.py:336`, `reinforcetactics/game/bot.py:905`, `reinforcetactics/game/bot.py:1285`, `reinforcetactics/game/bot.py:2809`, `reinforcetactics/game/bot_base.py:397`

Same issue as: `rulebots-3`, `critic-integration-12`

- **Impact.** Medium, Advanced and Master bots (curriculum opponents and BC demonstrators) consistently overvalue attacks on high-defence units and mis-plan finishing blows. Bot-strength numbers and imitation data inherit the error.
- **Fix.** Add `GameMechanics.preview_attack(attacker, target, grid, units, damage_model, attacker_pos=None)` that returns deterministic expected damage and counter-damage (evade as an expectation). Have `attack_unit` use the same helper so the two cannot drift, and replace the 8 raw `get_attack_damage` kill checks with it. Re-baseline bot tournaments afterwards, since bot behaviour will change.
- **Verifier note.** MediumBot.calculate_attack_value uses raw `attacker.get_attack_damage(...)` plus its own charge and flank multipliers, then `if damage_dealt >= target.health: return 1000 + damage_dealt`. It never calls GameMechanics.apply_defence_reduction (defined at mechanics.py:206 and applied in attack_unit at mechanics.py:399), the attack/defence buffs, or the hp_scaled damage model. Its counter estimate (1117-1130) also skips the attacker's defence reduction and ignores a paralyzed target not countering.

### `consolidate-4` — `train_self_play.py --mode mixed` is plain self-play: the bot envs are built and thrown away

**medium (reviewer: high)** · rl-correctness · confirmed · effort S · `scripts/train/train_self_play.py:323`

Also: `scripts/train/train_self_play.py:62`, `scripts/train/train_self_play.py:396`, `scripts/train/train_self_play.py:458`, `reinforcetactics/rl/config.py:168`, `configs/self_play/self_play.yaml:33`

Same issue as: `prior-2`, `rltrain-1`, `rltrain-2`, `critic-gaps-1`, `rlenv-12`, `tests-3`, `rlenv-4`

- **Impact.** Every 'mixed' run is really self-play on half the requested envs, while the logs report a bot mix, so the experiment's results mean something other than what it claims.
- **Fix.** Either implement it: build a single VecEnv with round(n_envs*bot_ratio) bot-opponent workers and the rest self-play (SB3 cannot swap envs during `learn`). Or delete `train_mixed`, `MixedTrainingCallback`, `mixed_training`, `bot_ratio` and the 'mixed' choice (~150 LOC). Use `argparse.BooleanOptionalAction` for `--swap-players`.
- **Verifier note.** Line 323 is `_ = make_maskable_vec_env(n_envs=args.n_envs // 2, opponent="bot", ...)`, whose result is discarded. The model is built on `VecMonitor(self_play_vec_env)`, which has only n_envs//2 envs. MixedTrainingCallback (62-94) is never instantiated, and all its _on_step does is flip `self.using_bots`. Line 396 logs a bot ratio that is never applied. SelfPlayConfig.mixed_training (rl/config.py:168) and configs/self_play/self_play.yaml:33 have no reader.

### `consolidate-7` — Tournament script's default map, the example shell script and many docs point at map files that don't exist

**medium** · bug · confirmed · effort S · `scripts/tournament.py:102`

Also: `scripts/run_tournament.sh:11`, `reinforcetactics/tournament/__init__.py:29`, `reinforcetactics/tournament/runner.py:54`, `README.md:133`, `examples/llm_bot_demo.py:57`, `reinforcetactics/rl/masking.py:12`

Same issue as: `persist-16`

- **Impact.** The documented zero-argument tournament run and the example shell script fail immediately, and copy-pasted examples fail too.
- **Fix.** Default to maps/1v1/beginner.csv and fix the other references. Add a cheap test that collects `maps/**/*.csv` string literals from .py/.sh/.md/.ipynb files and asserts each exists.

### `consolidate-8` — Tournament model discovery tests every checkpoint on a 6x6 map and silently drops ones trained on other sizes

**medium** · bug · confirmed · effort S · `reinforcetactics/tournament/bots.py:511`

Also: `reinforcetactics/game/model_bot.py:213`, `reinforcetactics/game/model_bot.py:270`, `scripts/tournament.py:132`, `tests/test_model_bot_feudal.py:176`

Same issue as: `aibots-11`

- **Impact.** Feudal and MultiDiscrete checkpoints trained on any non-6x6 map silently disappear from tournaments.
- **Fix.** At discovery time, check only that the checkpoint loads (SB3 load or torch.load plus a sanity check of the spaces). Check map compatibility per match in the runner, skipping incompatible model×map pairs with a logged reason. Resolve the probe map through a package-relative path.

### `consolidate-9` — Action dispatch is written separately in 8 places, the implementations already disagree, and action-type ints are magic numbers

**medium** · consolidation · partially · effort L · `reinforcetactics/rl/gym_env.py:980`

Also: `reinforcetactics/game/model_bot.py:563`, `reinforcetactics/rl/mcts.py:191`, `reinforcetactics/game/bot.py:107`, `reinforcetactics/utils/replay_actions.py:282`, `reinforcetactics/app/action_executor.py:87`, `reinforcetactics/app/input_handler.py:260`

Prior review: REVIEW_maintainability.md §4, §6

Same issue as: `core-14`, `aibots-13`, `prior-14`, `pygame-18`, `rulebots-20`

- **Impact.** Every rule change has to be made in 8 places, and the same policy action counts as valid in one consumer and invalid in another (for example, ModelBot's multi-discrete play differs from the training env).
- **Fix.** Create reinforcetactics/core/actions.py with `ActionType(IntEnum)`, `NUM_ACTION_TYPES`, `apply_legal_action(game_state, key, action) -> ActionResult` (used by RandomBot, MCTS, LLMBot, input handler and replay) and `apply_vector_action(game_state, vec, player)` (used by StrategyGameEnv and ModelBot). Make ACTION_KEY_MAP the single table. Migrate the lowest-risk callers (mcts, RandomBot) first.
- **Verifier note.** The core claim is real. gym_env.execute_game_action (980-1113) and ModelBot._execute_action plus its helpers (563-780) disagree. ModelBot._move_unit ignores the move_unit bool (`self.game_state.move_unit(unit, to_x, to_y)` / `return True`, 660-661), while gym_env checks it (1009). gym_env marks seize invalid on damage<=0, while ModelBot checks tile.is_capturable/owner instead. ModelBot requires can_attack and gym_env does not, and game_state.attack does not check it either.

### `consolidate-10` — Load Game shows the save picker twice; the first choice is thrown away

**medium** · ux · confirmed · effort S · `reinforcetactics/cli/commands.py:273`

Also: `reinforcetactics/ui/menus/main_menu.py:97`, `reinforcetactics/app/game_loop.py:315`, `reinforcetactics/ui/menus/save_load/load_game_menu.py:434`

Same issue as: `menus-3`

- **Impact.** Players have to pick the save twice (and confirm the 'completed game' dialog twice), and the first selection is discarded.
- **Fix.** Change to `load_saved_game(save_data=None)`, pass `menu_result["save_data"]` (and rename the misleading key), and only show the menu when nothing was passed.

### `consolidate-11` — Map loading and padding are duplicated and inconsistent; a missing map returns None and gives an obscure AttributeError

**medium** · consolidation · confirmed · effort M · `reinforcetactics/utils/file_io.py:109`

Also: `reinforcetactics/utils/replay_player.py:98`, `reinforcetactics/app/game_loop.py:253`, `reinforcetactics/core/game_state.py:345`, `reinforcetactics/game/llm_bot.py:663`, `reinforcetactics/rl/gym_env.py:429`

Prior review: REVIEW_maintainability.md §16

Same issue as: `core-19`, `aibots-18`, `persist-22`, `anim-8`, `anim-18`

- **Impact.** Mistyped map paths give confusing tracebacks. GUI and tournament LLM games see different coordinate spaces, and the offset logic can't be wired up safely because the two padders disagree.
- **Fix.** Write one `load_map(path, for_ui) -> LoadedMap(df, offset_x, offset_y, original)` that raises FileNotFoundError/ValueError, plus a `GameState.from_loaded_map()` that applies the metadata (offsets including the border). ReplayPlayer should call the same pad function. Fix the grass/ocean comment.

### `consolidate-12` — Unit, tile and action codes are plain strings or ints: canonical lists are retyped 13+ times and encodings kept in sync by comment

**medium** · consolidation · confirmed · effort M · `reinforcetactics/constants.py:156`

Also: `reinforcetactics/rl/gym_env.py:976`, `reinforcetactics/core/grid.py:66`, `reinforcetactics/core/game_state.py:1509`, `reinforcetactics/rl/observation.py:88`, `reinforcetactics/core/tile.py:43`

Prior review: REVIEW_maintainability.md §11, §12, §13

Same issue as: `prior-22`

- **Impact.** Adding a unit or tile type means editing about 20 sites, some of them silent (the observation encoding). A typo in a map becomes impassable ocean while the log says grass.
- **Fix.** Add a `UnitType(str, Enum)` (str-valued, so JSON/CSV stay compatible) and derive ALL_UNIT_TYPES and UNIT_TYPE_TO_IDX from it. Add `TILE_TYPE_TO_IDX` in constants derived from TILE_TYPE_ORDER (mapping 'o' to water) and use it in grid.to_numpy, and a 0-based unit index in game_state.to_numpy. Replace the literals mechanically, and add the ActionType enum from the dispatch finding. Fix the grass/ocean message or behaviour.

### `consolidate-13` — Installed wheel has no assets or maps; all runtime paths depend on the CWD; the Settings 'paths' section does nothing

**medium** · design · confirmed · effort M · `pyproject.toml:83`

Also: `reinforcetactics/utils/file_io.py:517`, `reinforcetactics/ui/renderer.py:53`, `reinforcetactics/utils/fonts.py:97`, `reinforcetactics/utils/settings.py:173`, `reinforcetactics/cli/main.py:37`

Prior review: REVIEW_maintainability.md §14

Same issue as: `tests-14`

- **Impact.** A non-editable install (`pip install .`, which docs-site/docs/intro.md:59 recommends) gives a console script with no maps, sprites or fonts unless launched from the repo root. Paths set in settings.json are ignored.
- **Fix.** Move assets/ and maps/ under reinforcetactics/data/ (or declare package-data) and resolve them with importlib.resources. Route saves/replays/models through `Settings.get_path`. Call ensure_directories only in play mode. Add a CI step that installs the built wheel in a temp dir and loads one map.

### `consolidate-18` — Docker tournament runner re-implements bot-config parsing (silently skips unknown types) and copies GCSUploader

**medium** · consolidation · confirmed · effort S · `docker/tournament/run_tournament.py:213`

Also: `reinforcetactics/tournament/config.py:278`, `reinforcetactics/cloud/storage.py:85`, `docker/tournament/run_tournament.py:48`

Same issue as: `persist-15`, `tests-13`

- **Impact.** A misconfigured tournament silently runs with fewer bots, and fixes to the uploader (such as the manifest skip) don't reach the tournament image.
- **Fix.** Import parse_bots_from_config (resolving scripted types through bot_registry.canonical_name, with the env-key check as an option) and cloud.storage.GCSUploader. That removes about 130 LOC.

#### Low and info findings (11)

| ID | Sev | Title | Location | Effort | Fix (summary) |
|---|---|---|---|---|---|
| `consolidate-14` ≈ rltrain-15 | low | 59 sweep YAMLs are 49k lines of mostly copied config with no inheritance and no load test | `configs/ppo/bootstrap_sweep/v54_uncapped_frontier.yaml:1` | M | Add a `base:` key (deep-merge overlay) to load_config and express live variants as small diffs against configs/ppo/bootstrap.yaml. Move v01-v5x to configs/ppo/archive/ or delete them; git history and docs/bootstrap_lessons_learned.md already record them. |
| `consolidate-15` ≈ prior-20 | low | About 800 LOC of confirmed dead code (vulture plus repo-wide grep) | `reinforcetactics/game/model_bot.py:467` | S | Delete them; tests that only exercise dead helpers go too. Add vulture to CI at --min-confidence 60 with a checked-in whitelist for framework hooks (SB3 `_on_step`, `__exit__` args). |
| `consolidate-16` ≈ aibots-10, rlalt-14, rulebots-23 | low | AlphaZeroBot is not wired in anywhere and can't load checkpoints trained with non-default architecture | `reinforcetactics/game/alphazero_bot.py:20` | M | Either wire it up (read the full arch config, add BotType.ALPHAZERO and a registry entry or sniff the checkpoint type in ModelBot, add a one-game smoke test) or delete the module and the docs checkbox. |
| `consolidate-17` ≈ pygame-17, anim-7 | low | Movement-path animation API is never called; units never walk and stay idle | `reinforcetactics/ui/renderer.py:1040` | M | Feature: have the BFS return a predecessor map, rebuild the path in GameState.move_unit (return it or emit a moved event), and have GameSession call queue_movement_path_animation and tween the sprite between tiles. |
| `consolidate-19` ≈ aibots-17 | low | LLM provider setup is spread over 3 modules; the tournament's Anthropic key check accepts any string | `reinforcetactics/tournament/bots.py:470` | S | Add reinforcetactics/game/llm_providers.py with a PROVIDERS table {key: (bot_cls, env_var, settings_key, default_model, test_fn)} used by the GUI factory, tournament discovery and the API-keys menu. |
| `consolidate-20` ≈ prior-18 | low | ~160 print() calls in library code, and the CLI never configures logging | `reinforcetactics/utils/file_io.py:110` | M | Add reinforcetactics/utils/log.py with `setup_logging(level, fmt)`, called from cli.main and the scripts. Convert library prints to logger calls; keep prints only for CLI and script user output. Consider ruff rule T20 for reinforcetactics/ outside cli/. |
| `consolidate-21` ≈ menus-13 | low | Translation tables and UI have drifted apart: 22 keys used in code are missing, 58 defined keys are unused, es/zh lack 34 keys | `reinforcetactics/utils/language.py:5` | M | Move the tables into per-language JSON files. Add a test that extracts `lang.get` literals and asserts each exists in every table and that no table key is orphaned. |
| `consolidate-22` ≈ tests-17 | low | README and docs no longer match the code: broken Lint badge, outdated unit stats, removed files, wrong output names | `README.md:6` | S | Fix or remove the badge. Generate the unit tables for README, docs-site and llm_prompts from UNIT_DATA with a small script, plus a test that fails when they drift. Add a docs link-check test that asserts every repo path in .md/.py/.sh/.ipynb exists. |
| `consolidate-23` | low | Dependency hygiene: unused pettingzoo dependency, requirements.txt duplicates pyproject, ruff unpinned in CI | `pyproject.toml:30` | S | Drop pettingzoo, or move it to an optional [marl] extra once a PettingZoo env exists. Replace requirements.txt with `-e .[gui]` (or delete it) and have CI/Docker install `.[gui,dev]`. Pin ruff in [dev] to the pre-commit rev. |
| `consolidate-24` ≈ tests-8 | low | mypy ignore_errors overrides hide 111 errors (the comment says 67); several modules are cheap to un-ignore | `pyproject.toml:119` | M | Fix and un-ignore bot.py, player_config_menu, mcts and api_keys_menu first (11 errors). Add a `LegalActions` TypedDict, or per-kind dataclasses to pair with the dispatch consolidation, to clear most of game_state. |
| `consolidate-25` | low | Map preview drawing and tile colour tables are duplicated across 3 menus and a script that copies constants | `scripts/generate_map_previews.py:18` | S | Add reinforcetactics/utils/map_raster.py with `tile_color_grid(map_2d) -> np.ndarray[H, W, 3]` (pure numpy). The menus blit it via pygame.surfarray and the script via PIL. Import constants in the script (constants.py imports no pygame). |

## tests Tests, CI and tooling

### `tests-1` — Default Docker CMD and CLI train mode crash: tqdm/rich used via progress_bar=True but never declared as dependencies

**high (reviewer: critical)** · bug · confirmed · effort S · `reinforcetactics/cli/commands.py:113`

Also: `reinforcetactics/cli/commands.py:113`, `scripts/train/train_self_play.py:260`, `scripts/train/train_feudal_rl.py:231`, `scripts/eval_agent.py:11`, `scripts/cloud/submit_vertex_job.sh:75`, `tests/test_packaging.py:62`

- **Impact.** The image's default command, the README's `--mode train`, and the cloud scripts' suggested Vertex job all fail after the image is pulled and the env and model are built. Paid Vertex jobs die at startup. eval_agent.py is unusable on a stock install.
- **Fix.** 1. Add `tqdm` and `rich` to [project].dependencies and requirements.txt, or depend on `stable-baselines3[extra]`, or pass progress_bar only when importable. 2. Add tests/test_entrypoints.py: - a subprocess run of `main.py --mode train --timesteps 64` in tmp_path (mark slow, ~10s); - `--help` for every scripts/**/*.py. 3. Add an import-scan test, or `deptry` in CI, that fails when a third-party top-level import in reinforcetactics/ or scripts/ is not declared. test_packaging.py today only diffs requirements.txt against pyproject.
- **Verifier note.** Real as stated. tqdm and rich are declared nowhere: not in pyproject [project].dependencies (lines 28-38), not in any extra, not in requirements.txt. `import tqdm` and `import rich` both raise ModuleNotFoundError in this fully provisioned env. Confirmed call sites: - commands.py:113 `model.learn(..., progress_bar=True)` - train_self_play.py:260 and :398 - train_feudal_rl.py:231 - examples/train_with_action_masking.py:75/115 and examples/train_with_bc_warmstart.py:134 (also affected) - …

### `tests-2` — .dockerignore does not exclude docker/tournament/.env, so LLM API keys get baked into pushed images

**high** · security · confirmed · effort S · `.dockerignore:1`

Also: `Dockerfile:22`, `docker/tournament/Dockerfile:34`, `docker/tournament/README.md:12`, `docker/tournament/README.md:319`, `scripts/gcp_launch.sh:30`

- **Impact.** Anyone who can pull the image, whether from a shared registry or a public GCR, can extract the provider API keys from the layer. Tournament output (LLM conversation logs, replays) is baked in as well.
- **Fix.** 1. Add `**/.env`, `**/.env.*`, `!**/.env.example`, `docker/tournament/output/`, `**/*service-account*.json` and `**/credentials*.json` to .dockerignore. Better, switch to an allow-list .dockerignore (`*` then `!reinforcetactics/ !scripts/ !maps/ !configs/ !main.py !pyproject.toml !requirements.txt !README.md !LICENSE`). 2. Add `.env` to the root .gitignore. 3. Add a small test asserting those patterns stay in .dockerignore.

### `tests-3` — Self-play tests avoid the production configuration, and the still-open §2.13 defects have no regression tests

**high** · rl-correctness · confirmed · effort M · `tests/test_self_play.py:38`

Also: `reinforcetactics/rl/self_play.py:332`, `reinforcetactics/rl/self_play.py:545`, `reinforcetactics/rl/self_play.py:652`, `tests/test_self_play.py:73`

Prior review: REVIEW_rl_pipeline_2026-07-24.md §2.13 / §4 item 15

Same issue as: `prior-2`, `rltrain-1`, `rltrain-2`, `critic-gaps-1`, `rlenv-12`, `consolidate-4`, `rlenv-4`

- **Impact.** Self-play training silently runs against a mask-less or random opponent, with rewards scored for the wrong seat. The suite gives false confidence because it only ever exercises the configuration where the bugs don't show.
- **Fix.** Add regression tests, marked `xfail(strict=True)` until §2.13 item 15 lands: - swap_players=True, asserting `env.unwrapped.agent_player == 2` after a swapped reset; - make_self_play_vec_env(use_subprocess=True, n_envs=2) plus the callback, asserting opponent updates reach the workers; - a tiny MaskablePPO opponent with a spy asserting predict receives action_masks; - a real OpponentPool.add_model round-trip; - caplog assertions that the opponent path logs no 'Error getting opponent action' warnings.

### `tests-4` — Tests read and write the developer's ./settings.json and leak the global Language, causing order-dependent failures

**medium** · bug · confirmed · effort S · `reinforcetactics/utils/language.py:939`

Also: `reinforcetactics/utils/language.py:939`, `reinforcetactics/utils/settings.py:272`, `tests/test_menus.py:220`, `tests/test_map_editor.py:24`

- **Impact.** Test outcomes depend on execution order and on the developer's local config. Running tests overwrites the user's saved preferences. This also blocks adopting pytest-xdist or pytest-randomly.
- **Fix.** In conftest.py: - add an autouse session fixture that sets `reinforcetactics.utils.settings._settings_instance = Settings(str(tmp_path_factory.mktemp('cfg') / 'settings.json'))` and `language._language_instance = None`; - add an autouse function fixture that calls `reset_language('en')` after each test. In the language fixture, restore the language in the teardown, not only at setup.

### `tests-6` — Coverage omits ui/app/cli, which hides 66%-actual coverage and near-zero coverage of the game loop, CLI and input handling

**medium** · test-gap · confirmed · effort M · `pyproject.toml:139`

Also: `reinforcetactics/cli/commands.py:91`, `reinforcetactics/app/game_loop.py:58`, `reinforcetactics/app/input_handler.py:266`, `tests/test_input_handler.py:14`

Same issue as: `pygame-24`, `menus-22`, `anim-21`

- **Impact.** Rendering, animation, the human input flow and the CLI can regress with no signal. These are the paths human players and new users hit first.
- **Fix.** Add cheap headless smoke tests: 1. Parametrize over all 25 maps/**/*.csv: GameState, a few SimpleBot turns, then `Renderer(gs, headless=True).render()` and `get_rgb_array()` shape checks. I ran this; it takes under 1s and all maps pass. 2. SpriteAnimator: queue_movement_path_animation plus a tick loop. 3. InputHandler with a real GameState and headless Renderer, fed synthetic MOUSEBUTTONDOWN events for select, move, attack and purchase. 4. `reinforce-tactics --help` and `--mode stats` in a tmp cwd. 5. GameSession.run with a patched event source that posts QUIT after N frames. Then remove ui/app/cli from omit and set fail_under from the measured value.

### `tests-7` — StrategyGameEnv render_mode='rgb_array' always returns None; no test covers it

**medium** · bug · confirmed · effort S · `reinforcetactics/rl/gym_env.py:684`

Also: `reinforcetactics/rl/gym_env.py:343`, `reinforcetactics/rl/gym_env.py:1677`

Same issue as: `rlenv-7`

- **Impact.** Gymnasium RecordVideo and any user capturing frames get None, which breaks the Gymnasium render contract for RL users.
- **Fix.** In __init__ and reset, create `Renderer(self.game_state, headless=True)` when render_mode == 'rgb_array'. Add a test asserting render() returns a uint8 (H, W, 3) array after reset and after a step. Headless Renderer works on every shipped map.

### `tests-8` — mypy ignore_errors overrides hide 100 errors in 10 modules, one on the exact line of the known self-play wrapper bug

**medium** · tooling · confirmed · effort M · `pyproject.toml:115`

Also: `reinforcetactics/rl/self_play.py:545`, `reinforcetactics/rl/feudal_rl.py:1084`, `reinforcetactics/rl/feudal_rl.py:1015`, `reinforcetactics/core/game_state.py:1128`, `reinforcetactics/tournament/bots.py:296`

Prior review: REVIEW_rl_pipeline_2026-07-24.md §2.13(b)

Same issue as: `consolidate-24`

- **Impact.** The 'mypy clean' status is weak. The override suppressed a warning that pointed straight at a training-corrupting bug.
- **Fix.** 1. Start with self_play.py: `self.base_env = cast(StrategyGameEnv, env.unwrapped)` and use it for every attribute read and write. This fixes the bug too. 2. Annotate `dict[str, Any]` (or TypedDicts) for the event/heal dicts and the MCTS/LLM kwargs. 3. Declare `self.worker: WorkerNetwork | AutoregressiveWorkerNetwork` and add the two dataclass fields. 4. Drop modules from the override one PR at a time, then set check_untyped_defs = true (36 errors). 5. Update the stale comment.

### `tests-9` — AlphaZero trainer/bot, bootstrap warm-start and all scripts/train/* entry points are essentially untested

**medium** · test-gap · confirmed · effort M · `reinforcetactics/rl/alphazero_trainer.py:277`

Also: `reinforcetactics/game/alphazero_bot.py:8`, `reinforcetactics/rl/bootstrap.py:645`, `reinforcetactics/rl/bootstrap.py:1203`, `scripts/train/train_bootstrap.py:1`

Prior review: REVIEW_rl_pipeline_2026-07-24.md §2.16

- **Impact.** The third trainer and its tournament bot, and the BC warm-start path used by v33_production_bc_warmstart.yaml, can break silently. §2.16 of the RL review already found correctness bugs in this code.
- **Fix.** - Add one ~5s test: AlphaZeroTrainer on a 6x6 map (1 iteration, 1 self-play game, num_simulations=2, 1 epoch), then train(), load_checkpoint(), and AlphaZeroBot(...).take_turn(). - Add a warm-start test that saves a tiny MaskablePPO and resumes via cfg.warm_start_path. - Add a parametrized `--help` subprocess smoke test for scripts/train/*.py and scripts/*.py.

### `tests-13` — Docker tournament runner re-implements bot dispatch, silently drops unknown types, and doesn't support MasterBot

**medium** · consolidation · confirmed · effort S · `docker/tournament/run_tournament.py:213`

Also: `docker/tournament/config.schema.json:151`, `reinforcetactics/tournament/config.py:278`, `reinforcetactics/tournament/bots.py:27`

Same issue as: `persist-15`, `consolidate-18`

- **Impact.** The strongest scripted bot cannot be entered in config-driven or Docker tournaments. The recent bot-registry consolidation did not reach these two dispatch tables.
- **Fix.** - Have run_tournament call parse_bots_from_config and apply only its API-key filter on top. - Derive the accepted types from the canonical bot registry or BotType, and add 'master' to the schema enum. - Raise or log on unknown types. - Add a test that loads config.json and config.simple.json through the runner and validates them against config.schema.json.

#### Low and info findings (16)

| ID | Sev | Title | Location | Effort | Fix (summary) |
|---|---|---|---|---|---|
| `tests-5` | low | 8 tests pass without executing a single assertion (measured with the pytest_assertion_pass hook) | `tests/test_save_replay.py:328` | S | Turn the guards into asserted preconditions: - set player_gold before creating units, then `assert ally is not None`; - fix `paralyzed_turns`; - end the AlphaZero game via resign to get a known winner, and assert per-player value signs; |
| `tests-10` | low | No job or per-test timeouts in CI; one test relies on a timeout that doesn't exist | `.github/workflows/python-package.yml:26` | S | - Add `timeout-minutes: 20` to the build job and 15 to the docs jobs. - Add `pytest-timeout` to the dev extra with `timeout = 120` in [tool.pytest.ini_options], plus `@pytest.mark.timeout(...)` for the known heavy tests. |
| `tests-11` | low | No dependency lock or constraints: CI, pre-commit and Docker resolve different, drifting toolchains | `.github/workflows/python-package.yml:59` | M | - Generate a constraints file (`uv pip compile pyproject.toml --all-extras -o constraints.txt`) and use `-c constraints.txt` in CI and both Dockerfiles. - Install tools via `pip install -e .[dev] -c constraints.txt` instead of an ad-hoc list. |
| `tests-12` | low | Cloud training runs can't be traced to a commit: git is installed in the image but .git is excluded | `reinforcetactics/utils/run_config.py:28` | S | - Add `ARG GIT_SHA` plus `ENV GIT_COMMIT=$GIT_SHA` to the Dockerfile. - Pass `--build-arg GIT_SHA=$(git rev-parse HEAD)` in gcp_launch.sh, and use `$COMMIT_SHA` in the Cloud Build config for build_image.sh. |
| `tests-14` ≈ consolidate-13 | low | Built wheel ships no fonts, sprites, maps or configs, and packaging tests only check metadata | `pyproject.toml:83` | M | Either move runtime assets under reinforcetactics/ and load them via importlib.resources with package-data, or document that source checkouts are required. |
| `tests-15` | low | Training Dockerfile: duplicated CUDA stack, GUI deps, poor layer caching, root user, unbuffered logs missing | `Dockerfile:19` | M | - Base on python:3.12-slim (the CUDA torch wheel brings its own runtime), or keep the CUDA base and install torch from the matching cu126 index. - Split requirements into train and gui, and use opencv-python-headless. |
| `tests-16` | low | Docs site is built only after merge to main, although onBrokenLinks is 'throw' | `.github/workflows/deploy-docusaurus.yml:3` | S | - Add a `pull_request` trigger (paths docs-site/**) that runs the build job only, with `if: github.event_name != 'pull_request'` on deploy. - Bump Node to 22 or 24. - Scope the pages/id-token permissions to the deploy job. |
| `tests-17` ≈ consolidate-22 | low | pre-commit hooks never run in CI, and the README Lint badge points to a nonexistent lint.yml | `README.md:6` | S | Add .github/workflows/lint.yml that runs `pre-commit run --all-files --show-diff-on-failure` once on 3.12, with the pre-commit cache keyed on .pre-commit-config.yaml. That also makes the badge valid. Drop the ruff steps from the test matrix. |
| `tests-18` | low | One test takes 21% of suite wall time because it builds 2000 GameStates | `tests/test_mixed_bot.py:269` | S | Build one GameState outside the loop; MixedBot construction doesn't mutate it. That brings the test to about 50ms. Drop the misleading comment. Optionally module-scope the MapSelectionMenu preview generation in test_menus.py (~0.7s per test, x6). |
| `tests-19` | low | Global RNG is reset to OS entropy by fixtures; several tests use unseeded maps and action sampling | `tests/test_gym_env.py:30` | S | - Add an autouse conftest fixture that seeds random, np.random and torch from a hash of request.node.nodeid (and prints the seed on failure), or adopt pytest-randomly, which also shuffles order and would have caught the Settings/Language leak. |
| `tests-20` | low | conftest.py carries 6 unused fixtures while map, game and pygame fixtures are copy-pasted across 16 files | `tests/conftest.py:34` | S | - Delete the unused fixtures. - Move a single `pygame_headless` fixture (monkeypatch.setenv for the SDL vars, init/quit) and `seeded_map`, `game_state` factory fixtures into conftest.py. - Replace direct os.environ writes with monkeypatch.setenv. |
| `tests-21` | low | Class-scoped fixture defined as an instance method will break under pytest 10; the MixedBot branch coverage it claims is never asserted | `tests/test_bootstrap.py:878` | S | Make shipped_cfg a module-level fixture, or add @classmethod. Record `env.unwrapped.opponent.use_hard` per (stage, seed) and assert both True and False were observed for p_hard stages. |
| `tests-22` | low | Coverage gate has 10 points of slack and is declared twice | `pyproject.toml:135` | S | - Set fail_under = 74 in one place ([tool.coverage.report]). - Move `--cov` into the CI command, or keep it and add `testpaths = ["tests"]`. - Add `--strict-markers` and `xfail_strict = true`. |
| `tests-23` | low | Only 5 of 67 shipped training YAMLs are load/validate-tested | `tests/test_rl_config.py:369` | S | Parametrize over `sorted(Path('configs').rglob('*.yaml'))`: route configs/imitation/* through load_scenarios_from_yaml and everything else through load_config(...).validate(). It adds about 2s. |
| `tests-24` | low | macOS clipboard tests are fully mocked but skipped on Linux CI | `tests/test_clipboard.py:103` | S | Replace the skipif with `monkeypatch.setattr('reinforcetactics.utils.clipboard.sys.platform', 'darwin')` (or whatever the module checks) so the tests run on every platform. Separately, migrate clipboard.py off the deprecated pygame.scrap init API. |
| `tests-25` | info | Opportunity: add packaging, container and cross-platform checks to CI | `.github/workflows/python-package.yml:35` | M | - Add a `docker` job that builds docker/tournament/Dockerfile (CPU-only, cached with buildx gha cache) and runs `--help`. Optionally lint both Dockerfiles with hadolint. - Add a `wheel` job: build, clean-venv install, entry-point smoke. |

## prior Prior-review status audit

### `prior-1` — GameState.end_turn() never clears the legal-actions cache: masks go stale, and a passing agent freezes RandomBot/MixedBot opponents

**critical** · bug · confirmed · effort S · `reinforcetactics/core/game_state.py:1192`

Also: `reinforcetactics/core/game_state.py:1305`, `reinforcetactics/core/game_state.py:400`, `reinforcetactics/game/bot.py:88`, `reinforcetactics/rl/gym_env.py:258`, `tests/test_bootstrap.py:725`

Prior review: REVIEW_advancedbot.md §2 (marked FIXED: the flag split landed, but end_turn invalidation was never added)

Same issue as: `core-1`, `rlenv-1`

- **Impact.** This changes game dynamics in normal training. On random/mixed-random stages, the majority of the curriculum, a policy that passes a full turn while the bot has used up its legal actions freezes the bot permanently. That makes a riskless draw available, which is plausibly one contributor to the draw attractor behind the random_10/15 stalls. Against NoopBot, every agent turn after the first starts with the previous turn's moved-unit mask. That is a concrete mechanism for the 'noop stages broke PPO learning' note (tests/test_bootstrap.py:725-731). ModelBot, LLM bots and MCTS read the same cache.
- **Fix.** Call `self._invalidate_cache()` in end_turn right after the `if self.game_over` guard. Nothing inside end_turn reads the cache, so one call covers the max_turns early return too. Add a regression test: create a unit, end_turn twice, and assert the unit's moves appear. Then re-run a noop sanity stage and one random_15 seed to see whether the draw rate moves.

### `prior-2` — Self-play is still not self-play: the seat swap never reaches the base env, the opponent is unmasked and resolves against the agent's action list, and SubprocVecEnv yields zero envs

**critical** · rl-correctness · confirmed · effort M · `reinforcetactics/rl/self_play.py:545`

Also: `reinforcetactics/rl/self_play.py:332`, `reinforcetactics/rl/self_play.py:431`, `reinforcetactics/rl/self_play.py:511`, `reinforcetactics/rl/self_play.py:652`, `scripts/train/train_self_play.py:101`, `scripts/train/train_self_play.py:153`

Prior review: REVIEW_rl_pipeline_2026-07-24.md §2.13 (a)-(c) and RNG; still open

Same issue as: `rltrain-1`, `rltrain-2`, `critic-gaps-1`, `rlenv-12`, `tests-3`, `consolidate-4`, `rlenv-4`

- **Impact.** Every self-play run trains against a pass-bot or random opponent, with rewards and observations for the wrong seat in about half of all episodes. No self-play result can be trusted, and the Jul-24 fix list did not touch any of this.
- **Fix.** (1) Set `self.env.unwrapped.agent_player` and rebuild `_prev_potential` after the swap. (2) Build the opponent's obs and flat action list with player=opponent, and pass action_masks to predict. (3) Keep a separate frozen opponent policy object instead of swapping state_dicts. (4) Push opponent weights via `vec_env.env_method('set_opponent_params', ...)` so SubprocVecEnv works, or fail loudly when 0 envs are found. (5) Seed from np_random. (6) Route opponent-turn game ends through the base env's terminal logic. Add a test asserting the base env agent_player after a swap.

### `prior-3` — A stall still ends the whole run: no retry from best_model.zip, no within-stage regression restore, no start-at-stage-K resume

**medium (reviewer: high)** · rl-correctness · confirmed · effort M · `reinforcetactics/rl/bootstrap.py:915`

Also: `reinforcetactics/rl/bootstrap.py:543`, `reinforcetactics/rl/bootstrap.py:978`

Prior review: REVIEW_rl_pipeline_2026-07-24.md §2.9 / §4 items 9-10; REVIEW_ppo_training.md rec #2

Same issue as: `rltrain-5`, `rltrain-7`

- **Impact.** This is still the main reason no run has finished the 33-stage ladder. In 28 of 41 archived stalls the stage peaked above its gate and the good checkpoint sat on disk. Wall-clock deaths re-pay 1-2M already-solved steps every session.
- **Fix.** Add `run_curriculum(start_stage=K)` plus a manifest written after each stage (model path, stage index, entropy schedule position). On stall, `set_parameters(best_model.zip)`, re-warm entropy and retry the stage once. Add a within-stage guard: after N consecutive evals more than X below the stage best, restore best_model.zip. The best_eligible_after plumbing already exists.
- **Verifier note.** This is accurate as a capability gap: - On stall the code writes run_status with best_model_path and raises CurriculumStalled (915-962). It never loads best_model.zip or retries. - restore_best_checkpoint_between_stages runs only on promotion (978). - run_curriculum (543-551) has no start_stage, and it loops over every stage (624). - There is no within-stage regression restore; there are no start_stage, resume or retry hits in bootstrap.py, config.py, callbacks.py or train_bootstrap.py.

### `prior-4` — Canonical bootstrap.yaml still ships the reward terms the sweep showed cause the stall, plus disabled guards and stale comments

**medium (reviewer: high)** · rl-correctness · confirmed · effort S · `configs/ppo/bootstrap.yaml:157`

Also: `configs/ppo/bootstrap.yaml:27`, `configs/ppo/bootstrap.yaml:49`, `reinforcetactics/rl/gym_env.py:1492`

Prior review: REVIEW_ppo_training.md §2.1 / rec #1; REVIEW_rl_pipeline_2026-07-24.md §3 and §4 item 12

Same issue as: `rltrain-14`

- **Impact.** This is the default config for train_bootstrap.py. A fresh run re-runs the configuration the v26/v27 bisection showed stalls. The comments also misdescribe the current truncation semantics.
- **Fix.** Back-port v52a/v54: win_speed_bonus 0, enemy_owned_capture 0, turn_penalty -0.5..-1.0, turn-scaled draw, max_actions_per_turn around 60, max_flat_actions 1024, max_steps scaled to max_turns. Rewrite the truncation comment. Re-anchor with 3 seeds.
- **Verifier note.** Every value is at the cited line: - win_speed_bonus 50.0 (157) - enemy_neutral_capture -8 / enemy_owned_capture -15 (170-171) - turn_penalty 0.0 (149) - max_actions_per_turn null (49) - max_steps 3000 (32) - gamma 0.99 (178) - no max_flat_actions, so the EnvConfig default of 512 applies (config.py:51) The file was last changed in 026b8a0. The comment at 27 still says truncation pays reward_config.draw, but gym_env.py:1492 now pays reward_config.get('truncation', 0.0), so the comment is stale.

### `prior-5` — Promotion gate still measures the deterministic policy with raw consecutive crossings and counts draws as losses

**medium (reviewer: high)** · rl-correctness · confirmed · effort S · `reinforcetactics/rl/callbacks.py:276`

Also: `reinforcetactics/rl/evaluation.py:97`, `reinforcetactics/rl/evaluation.py:366`, `reinforcetactics/rl/callbacks.py:457`

Prior review: REVIEW_rl_pipeline_2026-07-24.md §2.2(a), §2.11; REVIEW_ppo_training.md rec #3

Same issue as: `rltrain-4`, `rltrain-12`

- **Impact.** Promotion, best_model.zip and stall verdicts are still made on an argmax policy that PPO does not optimize. Near the gate, two consecutive 80-episode crossings are close to a coin flip. This is the '±10pp lottery' behind the stalls.
- **Fix.** Pass `deterministic=False` for the gate metric, or record both and gate on the stochastic one. Gate on a Wilson lower bound or a rolling mean of the last K evals. Report draws separately and allow a win+0.5*draw score option per stage.
- **Verifier note.** Verified: - PeriodicEvalCallback._do_eval calls evaluate_model with no deterministic argument (276-283), and the default is deterministic=True (evaluation.py:97). - PromotionCallback uses a raw consecutive streak (457-462). - win_rate = wins/n_episodes (evaluation.py:366), so for the gate a draw counts the same as a loss. - bootstrap.yaml uses n_eval_episodes 80 and patience 2.

### `prior-6` — The optimizer axis is still untouched: lr_schedule is dropped before SB3, entropy anneals over max_timesteps, gamma is 0.99 in all 64 configs

**medium (reviewer: high)** · rl-correctness · confirmed · effort S · `reinforcetactics/rl/config.py:137`

Also: `reinforcetactics/rl/callbacks.py:476`, `reinforcetactics/rl/bootstrap.py:763`, `configs/ppo/bootstrap.yaml:178`

Prior review: REVIEW_rl_pipeline_2026-07-24.md §2.7, §2.3; REVIEW_ppo_training.md rec #5

Same issue as: `rltrain-8`

- **Impact.** The cheapest counters to the documented Warrior-to-Mage bistability (late-stage LR anneal, entropy annealed over expected steps to promote, gamma 0.997) are still impossible from config. Every sweep keeps varying only reward shaping.
- **Fix.** Add `LRScheduleCallback(ScheduledAttrCallback)` that writes `model.learning_rate` and refreshes `model.lr_schedule`. Honour `ppo.lr_schedule` and add `CurriculumStage.learning_rate`. Allow an ent/LR anneal horizon smaller than max_timesteps.
- **Verifier note.** Verified: - as_sb3_kwargs skips lr_schedule (137). - EntropyScheduleCallback is built with total_timesteps=stage.max_timesteps (bootstrap.py:764-767). - ScheduledAttrCallback exists at callbacks.py:476. - There is no LR callback and no per-stage learning_rate on CurriculumStage. Two corrections: - The tally is gamma 0.99 x66 and clip_range 0.2 x66 across configs/, not 64. The 'all configs' point holds. - lr_schedule is not dead everywhere.

### `prior-8` — Feudal rollout: manager reward misattributed across rollout boundaries, time-limit truncation treated as terminal, undiscounted manager target

**medium** · rl-correctness · confirmed · effort M · `reinforcetactics/rl/feudal_rl.py:1364`

Also: `reinforcetactics/rl/feudal_rl.py:1283`, `reinforcetactics/rl/feudal_rl.py:1400`, `reinforcetactics/rl/feudal_rl.py:1433`, `reinforcetactics/rl/feudal_rl.py:1838`

Prior review: REVIEW_rl_pipeline_2026-07-24.md §2.16 (Feudal); feudal_rl_review.md does not list these

Same issue as: `rlalt-9`, `rlalt-11`

- **Impact.** The manager's credit assignment is biased at every rollout boundary, and value bootstraps are zeroed on time-limit episodes. This produces plausible-looking but wrong feudal curves and would confound the pending AR A/B (ROADMAP 3.7).
- **Fix.** Carry manager_open, manager_steps and manager_reward on the agent (or in a persistent `_EnvRolloutState`) across rollouts. Store `terminated` for GAE and bootstrap on `truncated`. Accumulate `gamma**k * r` for the manager target. Call reset_goal() after evaluate().

### `prior-9` — AlphaZero: BatchNorm left in train mode during eval, 0.5 returned when all eval games are draws, no max_turns, incomplete checkpoint config

**medium** · rl-correctness · partially · effort S · `reinforcetactics/rl/alphazero_trainer.py:315`

Also: `reinforcetactics/rl/alphazero_trainer.py:327`, `reinforcetactics/rl/alphazero_trainer.py:522`, `reinforcetactics/rl/alphazero_trainer.py:94`, `reinforcetactics/rl/alphazero_trainer.py:574`, `reinforcetactics/rl/mcts.py:293`

Prior review: REVIEW_rl_pipeline_2026-07-24.md §2.16 (AlphaZero); REVIEW_maintainability.md §9 is fixed

Same issue as: `rlalt-6`, `rlalt-8`

- **Impact.** Eval-time forwards overwrite BatchNorm running statistics. An all-draw eval permanently rejects new networks, and rejection reverts weights but not the Adam or scheduler state. A resumed run silently trains on a different map and learning rate.
- **Fix.** Call network.eval() before _evaluation_phase and restore train() afterwards. Return None and skip the accept/reject when total_decided == 0, or count draws as 0.5. Pass max_turns to both GameState constructors. Persist the full trainer config.
- **Verifier note.** Confirmed: `self.network.train()` (315) runs before `_evaluation_phase` (327). Neither MCTS._evaluate (mcts.py:293-312, @torch.no_grad only) nor AlphaZeroNet.predict switches to eval(), so BatchNorm uses per-sample batch statistics and overwrites its running statistics during evaluation, while the opponent network is in eval() — an asymmetric match. `if total_decided == 0: return 0.5` (522-524) is below the 0.55 threshold, so an all-draw evaluation always rejects the candidate.

### `prior-10` — train_bootstrap exits 0 after a stall, and a preempted run's directory is never uploaded

**medium** · bug · confirmed · effort S · `scripts/train/train_bootstrap.py:404`

Also: `reinforcetactics/cloud/storage.py:24`, `scripts/cloud/vertex_train.py:105`

Prior review: REVIEW_rl_pipeline_2026-07-24.md §3 (non-zero exit; preemption) / §4 item 13

Same issue as: `rltrain-3`

- **Impact.** Schedulers record stalled runs as successes, and a Vertex/Colab preemption loses the whole run record. Wall-clock death is one of the two ways runs end.
- **Fix.** Return a distinct non-zero code (e.g. 3) on stall. Install a SIGTERM handler that raises KeyboardInterrupt/SystemExit so the finally block uploads, and add benchmarks/bootstrap to the synced dirs, or pass --output-dir under a synced root.

### `prior-11` — Units that cannot reach the attacker still deal a 1-damage counter-attack

**medium** · bug · confirmed · effort S · `reinforcetactics/core/mechanics.py:445`

Also: `reinforcetactics/core/mechanics.py:223`, `reinforcetactics/core/unit.py:97`

Prior review: REVIEW_ppo_training.md §2.7 ('min-1-damage clamp grants phantom counter-attacks')

Same issue as: `core-3`

- **Impact.** Ranged units (M, S, and A against A/M/S) take phantom counter damage from defenders that cannot legally attack back. This is a rules bug in human play, bots and every training run, and it skews the value of ranged units.
- **Fix.** Skip the counter when base_counter_damage <= 0 (e.g. `if can_counter and base_counter_damage > 0`), or move the max(1) clamp so it applies only to positive base damage. Add a unit test.

### `prior-12` — BC/imitation supports multi_discrete only and records demos on a different engine

**medium** · rl-correctness · confirmed · effort M · `reinforcetactics/rl/imitation.py:607`

Also: `reinforcetactics/rl/imitation.py:25`, `reinforcetactics/rl/imitation.py:1231`, `scripts/build_bc_warmstart.py:160`

Prior review: REVIEW_rl_pipeline_2026-07-24.md §2.12 / §4 item 14

Same issue as: `rlalt-21`, `rltrain-17`

- **Impact.** About 1,500 lines of BC infrastructure target an action space that archived runs show cannot learn (0.00 WR). Configs from v50 on clone a bot playing different rules (hp_scaled, W cost 300). ROADMAP 3.5 depends on this subsystem.
- **Fix.** Port the recorder to flat_discrete (the label is the index in build_flat_actions) and forward engine_overrides and rng. Regress the value head on demo returns before handoff. Otherwise retire the subsystem and its 3 configs.

### `prior-13` — Flat action truncation still drops attacks and heals before moves

**medium** · design · confirmed · effort S · `reinforcetactics/rl/gym_env.py:303`

Also: `reinforcetactics/rl/gym_env.py:77`, `reinforcetactics/rl/gym_env.py:360`

Prior review: REVIEW_ppo_training.md §2.6 / rec #6 (partially addressed: seize/end_turn protection landed)

Same issue as: `rlenv-11`

- **Impact.** On skirmish and corner_points boards with 728-744 legal actions, combat and support actions are the first to be dropped. An army large enough to win cannot attack. Only v54 raises the cap.
- **Fix.** Protect action types 2, 4 and 6-9 as well, or truncate moves first (e.g. keep the K closest moves per unit). Set max_flat_actions: 1024 in the canonical config. Log how often truncation fires in episode_stats.

### `prior-14` — ModelBot carries a dead 95-line mask builder and a 220-line duplicate action dispatcher; MCTS has a third

**medium** · consolidation · confirmed · effort M · `reinforcetactics/game/model_bot.py:467`

Also: `reinforcetactics/game/model_bot.py:563`, `reinforcetactics/rl/mcts.py:191`, `reinforcetactics/rl/gym_env.py:981`

Prior review: REVIEW_maintainability.md §4, §5, §6 (masks: partially fixed; paralyze: consistent now)

Same issue as: `core-14`, `consolidate-9`, `aibots-13`, `pygame-18`, `rulebots-20`

- **Impact.** Three dispatchers can drift, as with the historical paralyze inconsistency. The dead mask code misleads readers into thinking ModelBot has its own layout.
- **Fix.** Delete _compute_action_mask. Extract `execute_game_action` into a free function in rl/actions.py (or core) taking (game_state, action_dict, player), and use it from the env, ModelBot and MCTS (via ACTION_KEY_MAP). Fix the comments.

#### Low and info findings (15)

| ID | Sev | Title | Location | Effort | Fix (summary) |
|---|---|---|---|---|---|
| `prior-7` | low | ROADMAP and REVIEW_advancedbot claim the swap_players/agent_player fix landed; it did not | `docs/ROADMAP.md:92` | S | Mark #16 as open (regressed or never effective through the wrapper) in both docs and link §2.13. Add a test so the claim is enforced. |
| `prior-15` ≈ core-15 | low | Nine near-identical 'units in range' helpers remain in GameMechanics, with hard-coded ranges | `reinforcetactics/core/mechanics.py:75` | S | Add `units_in_range(center, units, *, player, same_team, min_r, max_r, predicate)` and express each helper as a one-liner. Move the ranges into constants/unit_data. |
| `prior-16` | low | Bootstrap run-record and eval hygiene items from the Jul-24 review are still open | `reinforcetactics/rl/bootstrap.py:723` | S | Make stage.n_eval_episodes `int \| None`, resolved against cfg.eval. Add the missing env fields to config.json. Replace `pass` with `logger.warning(..., exc_info=True)`. Update best_win_rate regardless of save_dir. Clear model.ep_info_buffer at stage start. |
| `prior-17` ≈ rlenv-3, core-20 | low | get_legal_actions is O(units × reachable × units): can_move_to_position scans every unit per BFS node | `reinforcetactics/core/mechanics.py:35` | M | Keep a `{(x, y): unit}` position index on GameState, updated in create/move/remove, and pass it to can_move_to_position and get_unit_at_position. |
| `prior-18` ≈ consolidate-20 | low | FileIO ignores the configured Settings paths, and library modules still print() instead of logging | `reinforcetactics/utils/file_io.py:365` | M | Have FileIO take a base dir resolved from Settings (or env vars), or delete the paths block. Replace print() with module loggers. |
| `prior-19` ≈ menus-18 | low | LLM API keys are stored in plaintext settings.json with default file permissions | `reinforcetactics/utils/settings.py:81` | S | Create the file with `os.open(..., 0o600)` (or chmod after writing). Prefer env vars or the OS keyring, and warn in the API-keys menu. |
| `prior-20` ≈ consolidate-15 | low | Dead code and duplicate data called out in March are still present | `reinforcetactics/utils/file_io.py:555` | S | Delete export_replay_video and have ReplayPlayer call FileIO._pad_map/add_water_border. Derive the renderer colours from UNIT_DATA and drop UNIT_COLORS. Drop the duplicate TILE_COLORS keys. |
| `prior-21` | low | Small open items: missing translations, hand-rolled CSV, per-call LLM clients, unused PettingZoo dependency, fog_of_war silently ignored on the PPO path | `reinforcetactics/utils/language.py:519` | S | Fill or flag the missing keys (add a CI test for key parity). Use csv.writer. Create LLM clients once in __init__. Move pettingzoo to an optional [marl] extra. |
| `prior-22` ≈ consolidate-12 | low | There is still no UnitType enum, and raw tile/unit string comparisons dominate | `reinforcetactics/core/grid.py:66` | M | Add `UnitType(str, Enum)` (a str subclass, so existing comparisons keep working) and move both encodings into constants/observation. Migrate incrementally, core first. |
| `prior-23` | low | ROADMAP.md statuses are stale in both directions and it ignores the bootstrap pipeline and its blockers | `docs/ROADMAP.md:32` | S | Refresh the date and statuses. Add a 'Current RL blockers' section linking the review docs' open items. Replace the PPO row with the bootstrap pipeline, and retire or fix ppo_training.ipynb as REVIEW_ppo_training rec #8 says. |
| `prior-24` | low | REVIEW_maintainability.md has no status markers even though at least 10 of its 32 items are fixed | `docs/REVIEW_maintainability.md:12` | S | Add a Status column (Fixed / Partial / Open, with commit), or fold the open items into a single tracker and move the file to docs/archive/. |
| `prior-25` | low | REVIEW_advancedbot.md resolution table is wrong in both directions | `docs/REVIEW_advancedbot.md:144` | S | Update the table, or archive the doc after moving its truly open items (#5, #19, #21-29, #16, and the end_turn cache defect) into the consolidated tracker. |
| `prior-26` | low | The seven review/planning docs overlap and should be consolidated into one status tracker, with superseded ones archived | `docs/bootstrap_runs_review.md:1` | S | Create docs/reviews/STATUS.md with one row per open item (source doc §, current file:line, owner, status). Move bootstrap_runs_review.md, REVIEW_maintainability.md and REVIEW_advancedbot.md to docs/archive/ with a header pointing to STATUS.md. |
| `prior-27` | info | Opportunity: run the two never-tested axes and get a trained checkpoint onto the ladder | `configs/ppo/bootstrap.yaml:205` | M | After the end_turn cache fix, add v55a (v54 + pool: flatten) and v55b (v54 + gamma 0.997 + max_actions_per_turn 25), 3 seeds each, with the best-model gate fixed. Enter the best stage checkpoint into a tournament round via ModelBot (ROADMAP next step #1). |
| `prior-28` | info | Opportunity: pull AutoregressiveActionHead out of feudal and build a pointer/per-cell PPO policy to replace the positional flat head | `reinforcetactics/rl/feudal_rl.py:364` | L | Extract the AR head to rl/autoregressive.py. Write an SB3 MaskableActorCriticPolicy subclass that scores (source cell, target cell) from per-cell features (pool: flatten trunk) using build_structured_masks. |

## critic-gaps Coverage-gap critic

### `critic-gaps-1` — train_self_play.py cannot train at all: SelfPlayEnv.action_masks() returns the per-dimension mask tuple, and MaskablePPO crashes on its first rollout

**high** · bug · confirmed · effort S · `reinforcetactics/rl/self_play.py:552`

Also: `scripts/train/train_self_play.py:149`, `scripts/train/train_self_play.py:312`, `scripts/train/train_self_play.py:260`, `tests/test_self_play.py:223`, `reinforcetactics/rl/masking.py:75`, `README.md:107`

Same issue as: `prior-2`, `rltrain-1`, `rltrain-2`, `rlenv-12`, `tests-3`, `consolidate-4`, `rlenv-4`

- **Impact.** The documented entry point (README.md:107 `python scripts/train/train_self_play.py`, listed as ✅ in docs/vertex_training.md:89) cannot produce a single PPO update in either self-play or mixed mode. The semantic self-play defects already reported (rltrain-1, prior-2) sit behind this hard crash.
- **Fix.** Make SelfPlayEnv.action_masks() return the concatenated mask (delegate to the ActionMaskedEnv's action_masks()), and keep the tuple behind a separately named method for the opponent's own use. Add `--action-space` to train_self_play.py and forward it to make_self_play_vec_env/make_self_play_env. Replace test_action_masks_method with a smoke test that runs `MaskablePPO(...).learn(n_steps)` on make_self_play_vec_env in both action spaces. Also resolve the progress_bar=True dependency (tests-1) at :260 and :398.

### `critic-gaps-2` — StrategyGameEnv has no working Player-2 seat: all PPO training and evaluation is first-mover only, and setting agent_player=2 produces an illegal game

**medium** · rl-correctness · confirmed · effort M · `reinforcetactics/rl/gym_env.py:490`

Also: `reinforcetactics/rl/gym_env.py:1575`, `reinforcetactics/rl/gym_env.py:1197`, `reinforcetactics/rl/evaluation.py:283`, `reinforcetactics/rl/viz.py:776`, `reinforcetactics/rl/config.py:50`

Prior review: REVIEW_rl_pipeline_2026-07-24.md §2.13(b) (covers only the SelfPlayEnv wrapper not propagating the seat)

Same issue as: `critic-integration-5`

- **Impact.** Every bootstrap curriculum stage trains, and every promotion gate measures, the agent only as first mover. Tournaments, the GUI's ModelBot and any ladder play the same checkpoint as Player 2 half the time, a seat it has never seen (turn parity and the core-13 income offset differ). The agent_player attribute looks like a knob, but anyone who sets it on the base env (for example while fixing the SelfPlayEnv propagation bug) gets an agent that takes an extra opening move and a permanent +1 income tick.
- **Fix.** Add `agent_seat: 1 | 2 | "random"` to StrategyGameEnv and EnvConfig, drawing it from np_random. In reset, when the agent is P2, run the opponent's opening turn and end_turn before returning the first observation. Replace the blind safety-net end_turn with a loop that ends turns until current_player == agent_player. Report win rate per seat in PeriodicEvalCallback and evaluate both seats before promotion. Fix viz.py:776 to use agent_player. Add a regression test that seat-2 gold and turn flow mirror seat 1.

### `critic-gaps-3` — StrategyGameEnv.action_masks() breaks sb3-contrib's mask contract in multi_discrete mode, so record_evaluation_to_video crashes on a raw env

**medium** · design · confirmed · effort M · `reinforcetactics/rl/gym_env.py:931`

Also: `reinforcetactics/utils/video.py:224`, `reinforcetactics/rl/evaluation.py:228`, `reinforcetactics/rl/masking.py:75`, `reinforcetactics/rl/self_play.py:552`, `reinforcetactics/rl/feudal_rl.py:1355`

- **Impact.** Each consumer has to remember to wrap or concatenate; two already forgot (SelfPlayEnv, video export). Any new script, notebook or test that uses the env directly with MaskablePPO fails in the default action space.
- **Fix.** Make action_masks() return the concatenated vector the sb3-contrib contract expects, and expose `action_masks_per_dim()` for the per-dimension consumers (feudal_rl.py:1170/1355, ModelBot, purchase exploration). Then reduce ActionMaskedEnv to a compatibility shim and delete the concatenation workarounds. Add a contract test that `MaskablePPO.predict(obs, action_masks=env.action_masks())` works on the raw env in both action spaces.

#### Low and info findings (10)

| ID | Sev | Title | Location | Effort | Fix (summary) |
|---|---|---|---|---|---|
| `critic-gaps-4` | low | Legacy GCE scripts cannot train anything, and stop_training.sh deletes (not stops) every VM whose name contains 'rl-trainer' | `scripts/stop_training.sh:22` | S | Delete gcp_launch.sh, monitor_training.sh and stop_training.sh (the Vertex path supersedes them) and drop the reference in docs/vertex_training.md. |
| `critic-gaps-5` | low | submit_vertex_job.sh prints the W&B API key to the terminal and stores it in plain text in the Vertex job spec | `scripts/cloud/submit_vertex_job.sh:125` | S | Never echo env values: redact `value:` lines when printing, or print only the args. Store the key in Secret Manager, pass only the secret's resource name (e.g. WANDB_API_KEY_SECRET) and have vertex_train.py fetch it at startup with the job's service account. |
| `critic-gaps-6` | low | The default bootstrap output dir (benchmarks/bootstrap/<ts>) is not gitignored, and with no .gcloudignore every Cloud Build uploads old runs | `scripts/train/train_bootstrap.py:372` | S | Add `/benchmarks/bootstrap/` to .gitignore (or move the default output to a gitignored `runs/`), and add a .gcloudignore that mirrors .dockerignore. |
| `critic-gaps-7` | low | Loading an SB3 .zip unpickles arbitrary code (cloudpickle), and ModelBot and tournament discovery do so for every file they are given, contradicting the … | `reinforcetactics/game/model_bot.py:102` | M | Load SB3 zips with custom_objects that override every serialized key (resolve policy_class from an allowlist by name, and pass lr_schedule and clip_range as constants), or rebuild the policy from a stored JSON config and load policy.pth with … |
| `critic-gaps-8` | low | Eval stall traces record only the first action component, so the dumped JSONL cannot show what the agent actually did | `reinforcetactics/rl/evaluation.py:258` | S | Have the env put the decoded game action in info (for example `decoded_action: {type, unit_type, from, to}` from `_decode_action`/`_current_actions`) and write that plus the full MultiDiscrete vector to the trace. |
| `critic-gaps-9` | low | UnitPurchaseMenu prices units from the global UNIT_DATA and ignores the unit cap, so affordability can disagree with the engine | `reinforcetactics/ui/menus/in_game/unit_purchase_menu.py:231` | S | Read `self.game_state.unit_data[ut]` for name and cost. Disable all buttons and show 'unit cap reached' when get_unit_count(player) >= max_units_per_player. Surface create_unit failures in the menu instead of only logging them. |
| `critic-gaps-10` | low | OpponentPool's restore path is dead and would be wrong if wired up: lexicographic snapshot order, unpruned disk snapshots, stale positional indices | `reinforcetactics/rl/self_play.py:187` | S | Either delete the unused API, or wire load_from_disk into --resume-from with a numeric sort (`key=lambda p: int(p.stem.split('_')[1])`), delete evicted zips, drop the positional 'index' field, and refuse to add empty parameter dicts. |
| `critic-gaps-11` | low | icons.py: 11 copy-pasted cached icon generators, each of which calls the unnecessary full pygame.init() | `reinforcetactics/ui/icons.py:14` | S | Add a `@_cached_icon("name")` decorator that owns the key, cache and surface creation, so each icon is a 3-5 line draw function. Remove `_ensure_pygame_init`, since surface drawing needs no init. |
| `critic-gaps-12` | info | Opportunity: a persistent rating ladder; the Elo persistence APIs are unused and merge_from duplicates history | `reinforcetactics/tournament/elo.py:233` | M | Replace the incremental API with a ladder store of all game results (already in the results and replay JSON). Fit ratings in batch (Bradley-Terry or BayesElo MLE with bootstrap CIs, which also removes the order dependence in persist-7). |
| `critic-gaps-13` | info | Opportunity: replace the five cloud shell scripts with one Python launcher that expands the config × seed matrix into Vertex jobs | `scripts/cloud/submit_vertex_job.sh:75` | M | Add `scripts/cloud/launch.py` with subcommands `submit --config ... --seeds 0-4` and `status`/`collect`. Each job gets its own GCS prefix and labels (config hash, git SHA baked at build time), W&B credentials come via Secret Manager, the default command is … |

## critic-integration Cross-subsystem integration critic

### `critic-integration-1` — BC recorder saves bot actions as labels before they run and never checks the engine accepted them; 18–53% of demonstrations are illegal

**high** · rl-correctness · confirmed · effort S · `reinforcetactics/rl/imitation.py:455`

Also: `reinforcetactics/rl/imitation.py:367`, `reinforcetactics/rl/imitation.py:481`, `configs/imitation/bc_beginner_warmstart.yaml:40`

- **Impact.** Behaviour cloning trains on moves the engine rejected and on illegal attacks. A rejected move leaves the state unchanged, so the same observation is then labelled again with the bot's next action, giving contradictory supervision. The warm-started policy learns to emit actions the env rejects, each costing the invalid_action penalty. This is separate from rulebots-1/2, which cover the bots' own bugs: even with those fixed, the recorder has no legality check of its own.
- **Fix.** Keep taking the observation and masks before the call. Append the demo only if the delegated call succeeded: move_unit True, create_unit not None, heal > 0, paralyze/cure/haste/buff True, seize damage > 0, and for attack the (attacker, target) pair was in get_legal_actions at snapshot time. Alternatively, test the full action tuple against the legal set, as the harness does. Add a regression test asserting every recorded demo is in the legal set.

### `critic-integration-2` — GUI Paralyze follows its own rules: adjacent targets only, no cooldown or already-paralyzed check, and a failed cast still ends the Mage's turn

**medium** · bug · confirmed · effort S · `reinforcetactics/ui/menus/in_game/unit_action_menu.py:69`

Also: `reinforcetactics/app/action_executor.py:96`, `reinforcetactics/app/input_handler.py:253`, `reinforcetactics/core/game_state.py:1382`

- **Impact.** Human players cannot use the Mage's safest paralyze (range 2), which every AI player can use. Pressing P while on cooldown silently wastes the Mage's whole turn. Humans can also re-paralyze an already-paralyzed target, which no AI path allows.
- **Fix.** Build each menu entry's targets from `game.get_legal_actions(unit.player)[kind]`, filtered to this unit. That also brings in the fog-of-war and cooldown rules for attack, heal, cure, haste and buffs. In action_executor and input_handler, call end_unit_turn only when the engine call returned success; otherwise keep the menu open and show a message.

### `critic-integration-3` — The FOW (fog-of-war) serializer sends LLM bots the live owner of shrouded structures and ignores the stored last-seen memory; three consumers disagree on what is known about structures

**medium** · bug · confirmed · effort S · `reinforcetactics/game/llm_bot.py:636`

Also: `reinforcetactics/core/visibility.py:190`, `reinforcetactics/ui/renderer.py:499`, `README.md:161`

Same issue as: `core-5`, `rlenv-14`, `consolidate-1`, `pygame-12`

- **Impact.** In FOW games LLM bots learn about enemy captures in real time, which humans and RL agents are meant not to see. This skews LLM-vs-human and LLM-vs-RL fog-of-war comparisons, and the enemy-HQ rule differs between renderer, observation and serializer. This site is not covered by core-5, pygame-12 or consolidate-1, which cover to_numpy and the renderer.
- **Fix.** For explored but non-visible tiles, report owner and HP from `vis_map.get_last_seen_structure(x, y)` plus `turn_seen`. Decide once, in VisibilityMap, whether enemy HQs are always known (e.g. seed a snapshot at game start), and have to_numpy, the renderer and the serializer all read that one facade.

### `critic-integration-4` — Nothing checks a map's owners against num_players: 3-player maps and editor-made neutral or extra HQs load into the 2-player env and tournament, where capturing the spare HQ wins instantly

**medium** · bug · confirmed · effort S · `reinforcetactics/rl/gym_env.py:456`

Also: `reinforcetactics/tournament/runner.py:310`, `reinforcetactics/core/mechanics.py:799`, `reinforcetactics/ui/menus/map_editor/map_editor.py:295`, `reinforcetactics/ui/menus/map_editor/tile_palette.py:92`

- **Impact.** Running a tournament with --map-dir on a 3-player or editor-made map, or pointing the env at one, silently produces games where an undefended HQ is a free win. The policy cannot tell that HQ from a neutral one. core-7 is related but covers FFA end rules, not missing map validation.
- **Fix.** Validate once when GameState is built: the set of HQ owners must equal {1..num_players}, every structure owner must be ≤ num_players, and neutral HQs are rejected (or seizing one becomes a plain capture). Replace the always-false env guard with this check and have the map editor's validation call the same function.

### `critic-integration-5` — Every PPO and feudal checkpoint is trained and promotion-gated only as Player 1, but the tournament and GUI deploy it in both seats

**medium** · rl-correctness · confirmed · effort M · `reinforcetactics/rl/gym_env.py:490`

Also: `reinforcetactics/tournament/schedule.py:253`, `reinforcetactics/ui/menus/game_setup/player_config_menu.py:72`

Prior review: REVIEW_ppo_training.md §2.6

Same issue as: `critic-gaps-2`

- **Impact.** A checkpoint that clears a 0.70 gate as P1 has never played a P2 game: its start is spatially mirrored, it moves second and its income timing differs. Win-rate gates overstate strength, and ladder or GUI results will be much worse than training curves suggest.
- **Fix.** Add an EnvConfig `agent_seat: 1 | 2 | 'random'`, drawn per reset from the env's seeded np_random. Set agent_player before binding the opponent, and play the opponent's first turn inside reset() when the agent is P2. Evaluate in both seats and log win rate per seat in evaluation.py and the promotion gate.

#### Low and info findings (7)

| ID | Sev | Title | Location | Effort | Fix (summary) |
|---|---|---|---|---|---|
| `critic-integration-6` | low | LLM prompt and state omit the rules and state that decide games: multi-turn HP-based capture, capture reset, paid structure healing, max_turns and statuses; | `reinforcetactics/game/llm_bot.py:630` | S | Add hp, max_hp and turns_to_capture per adjacent friendly unit to structure entries; add paralysis, buffs and cooldowns to unit entries; add max_turns and turns_remaining. Add a short capture-and-economy rules block generated from constants. |
| `critic-integration-7` | low | The rules text has four hand-kept copies (six prompt blocks, the docs-site mechanics page, the README, code comments), and the generator for it is test-only; | `reinforcetactics/game/llm_prompts.py:613` | M | Make one rules renderer driven by constants and GameMechanics (a unit table plus ability, capture and economy sections). Use it for the LLM prompts (wiring get_dynamic_prompt into LLMBot) and to generate the docs-site table and the README section. |
| `critic-integration-8` | low | The GUI never shows defence/attack buffs, ability cooldowns or structure-heal gold costs, although they swing combat by ±50% and the RL observation includes … | `reinforcetactics/ui/renderer.py:938` | S | Add 'D'/'A' badges with turns remaining next to the H/P badges, list active buffs and cooldowns in the tooltip, and show a turn-start line such as '+150 income, −13 repairs' in the HUD. |
| `critic-integration-9` | low | Heal and cure share action index 4: the env and ModelBot resolve it as cure-first, MCTS as heal-first, and BC labels a heal as index 4 that replays as a cure | `reinforcetactics/rl/mcts.py:97` | M | Either give cure its own action type (index 5 is end_turn, so append index 10 and bump NUM_ACTION_TYPES in one place), or document cure-first as the single rule and make MCTS's decode follow it. |
| `critic-integration-10` | low | Map CSV loader and Tile disagree on blank and invalid cells: trailing commas add a phantom grass column, while Tile turns blanks into ocean and logs … | `reinforcetactics/utils/file_io.py:66` | S | Drop all-NaN rows and columns before filling, pick one fill value (ocean, to match Tile), fix the comments and messages, and reject unknown tile codes with the file and cell location instead of silently substituting. |
| `critic-integration-11` | low | FOW saves drop explored terrain and last-seen memory; a reloaded game re-fogs the whole map | `reinforcetactics/core/game_state.py:1436` | S | Serialise each VisibilityMap's state array and snapshot dicts (plus a schema version) in to_dict, and restore them in from_dict before calling update_visibility. |
| `critic-integration-12` ≈ rulebots-3, consolidate-3 | info | Opportunity: one pure GameMechanics.forecast_attack() shared by bots, the LLM serializer, a GUI damage preview and RL diagnostics | `reinforcetactics/core/mechanics.py:337` | M | Factor attack_unit into `forecast_attack(attacker, target, from_pos=None, grid, units, damage_model) -> {damage, counter_damage, kill_prob, counter_kill_prob, bonuses}`, treating Rogue evade as a probability. |
