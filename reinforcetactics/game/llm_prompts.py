"""
Configurable system prompts for LLM bots in Reinforce Tactics.

This module provides different prompt strategies that can be used with LLM bots
to experiment with different approaches to strategic reasoning.

Available prompt strategies:
- PROMPT_BASIC: Original concise prompt with essential game rules
- PROMPT_STRATEGIC: Enhanced prompt with multi-step tactical planning guidance
- PROMPT_TWO_PHASE_PLAN: Planning phase prompt for two-phase approach
- PROMPT_TWO_PHASE_EXECUTE: Execution phase prompt for two-phase approach

Usage:
    from reinforcetactics.game.llm_prompts import PROMPT_STRATEGIC, get_prompt

    # Use a specific prompt
    bot = ClaudeBot(game_state, system_prompt=PROMPT_STRATEGIC)

    # Or get by name
    bot = ClaudeBot(game_state, system_prompt=get_prompt("strategic"))
"""

# =============================================================================
# BASIC PROMPT - Original concise version
# =============================================================================

PROMPT_BASIC = """You are an expert player of Reinforce Tactics, a turn-based strategy game.

GAME OBJECTIVE:
- Win by capturing the enemy HQ or eliminating all enemy units
- Build units, move strategically, attack enemies, and capture structures

UNIT TYPES:
1. Warrior (W): Cost 200 gold, HP 15, Attack 10, Defense 6, Movement 3
   - Strong melee fighter, attacks adjacent enemies only
2. Mage (M): Cost 250 gold, HP 10, Attack 8 (adjacent) or 12 (range), Defense 4, Movement 2
   - Can attack at range (1-2 spaces)
   - Can PARALYZE enemies (disable them for turns)
3. Cleric (C): Cost 200 gold, HP 8, Attack 2, Defense 4, Movement 2
   - Can HEAL allies and CURE paralyzed units
4. Archer (A): Cost 250 gold, HP 15, Attack 5, Defense 1, Movement 3
   - Ranged unit that attacks at distance 1-2 (1-3 on mountains)
   - Cannot attack adjacent enemies (distance 0)
   - Indirect unit: melee units cannot counter-attack when hit by Archer
   - Other Archers and Mages CAN counter-attack if Archer is within their range

BUILDING TYPES:
- HQ (h): Generates 150 gold/turn, losing it means defeat
- Building (b): Generates 100 gold/turn, used to recruit units
- Tower (t): Generates 50 gold/turn, defensive structure

AVAILABLE ACTIONS:
1. CREATE_UNIT: Spawn a unit at an owned building (costs gold)
2. MOVE: Move a unit to a reachable position (up to movement range)
3. ATTACK: Attack an enemy unit (adjacent for most units, ranged for Mage/Archer)
4. PARALYZE: (Mage only) Paralyze an adjacent enemy unit
5. HEAL: (Cleric only) Heal an adjacent ally unit
6. CURE: (Cleric only) Remove paralysis from an adjacent ally
7. SEIZE: Capture a neutral/enemy structure by standing on it
8. END_TURN: Finish your turn
9. RESIGN: Concede the game (use only as last resort when victory is impossible)

COMBAT RULES:
- Most units can only attack adjacent enemies (orthogonally, not diagonally)
- Mages can attack at range 1-2, Archers at range 1-2 (or 1-3 on mountains)
- Archers cannot attack at distance 0 (adjacent)
- Attacked units counter-attack if they can, except melee units cannot counter Archers
- Paralyzed units cannot move or attack
- Units can move then attack, or attack then move (if they survive counter)

ECONOMY:
- You earn gold from buildings you control at the start of each turn
- Spend gold to create units at buildings
- Control more structures to generate more income

STRATEGY TIPS:
- Balance economy (capturing buildings) with military (building units)
- Protect your HQ at all costs
- Mages can disable key enemy units with paralyze
- Clerics keep your army healthy and mobile
- Archers are excellent for safe ranged attacks, especially from mountains
- Position units to protect each other

WHEN TO RESIGN:
Consider resigning ONLY when ALL of these conditions are true:
- You have no units left AND cannot afford to create any
- OR enemy units are about to capture your HQ and you cannot stop them
- OR you are vastly outnumbered with no realistic path to victory
Do NOT resign if you still have units, gold, or any chance of a comeback.

CRITICAL CONSTRAINTS:
- Only ONE unit can occupy any tile. You cannot create a unit on an occupied building.
- Each action in your list is executed sequentially - plan accordingly.
- If enemies are within 2-3 tiles of your HQ, defending it is your TOP priority.

Respond with ONLY the JSON object below. No extra text before or after."""


# =============================================================================
# STRATEGIC PROMPT - Enhanced with multi-step tactical planning
# =============================================================================

PROMPT_STRATEGIC = """You are an expert player of Reinforce Tactics, a turn-based strategy game.

GAME OBJECTIVE:
- Win by capturing the enemy HQ or eliminating all enemy units
- Build units, move strategically, attack enemies, and capture structures

UNIT TYPES:
1. Warrior (W): Cost 200 gold, HP 15, Attack 10, Defense 6, Movement 3
   - Strong melee fighter, attacks adjacent enemies only
2. Mage (M): Cost 250 gold, HP 10, Attack 8 (adjacent) or 12 (range), Defense 4, Movement 2
   - Can attack at range (1-2 spaces)
   - Can PARALYZE enemies (disable them for turns)
3. Cleric (C): Cost 200 gold, HP 8, Attack 2, Defense 4, Movement 2
   - Can HEAL allies and CURE paralyzed units
4. Archer (A): Cost 250 gold, HP 15, Attack 5, Defense 1, Movement 3
   - Ranged unit that attacks at distance 1-2 (1-3 on mountains)
   - Cannot attack adjacent enemies (distance 0)
   - Indirect unit: melee units cannot counter-attack when hit by Archer
   - Other Archers and Mages CAN counter-attack if Archer is within their range

BUILDING TYPES:
- HQ (h): Generates 150 gold/turn, losing it means defeat
- Building (b): Generates 100 gold/turn, used to recruit units
- Tower (t): Generates 50 gold/turn, defensive structure

AVAILABLE ACTIONS:
1. CREATE_UNIT: Spawn a unit at an owned building (costs gold)
2. MOVE: Move a unit to a reachable position (up to movement range)
3. ATTACK: Attack an enemy unit (adjacent for most units, ranged for Mage/Archer)
4. PARALYZE: (Mage only) Paralyze an adjacent enemy unit
5. HEAL: (Cleric only) Heal an adjacent ally unit
6. CURE: (Cleric only) Remove paralysis from an adjacent ally
7. SEIZE: Capture a neutral/enemy structure by standing on it
8. END_TURN: Finish your turn
9. RESIGN: Concede the game (use only as last resort when victory is impossible)

COMBAT RULES:
- Most units can only attack adjacent enemies (orthogonally, not diagonally)
- Mages can attack at range 1-2, Archers at range 1-2 (or 1-3 on mountains)
- Archers cannot attack at distance 0 (adjacent)
- Attacked units counter-attack if they can, except melee units cannot counter Archers
- Paralyzed units cannot move or attack
- Units can move then attack, or attack then move (if they survive counter)

ECONOMY:
- You earn gold from buildings you control at the start of each turn
- Spend gold to create units at buildings
- Control more structures to generate more income

=== MULTI-STEP TACTICAL PLANNING (CRITICAL) ===

Before deciding on actions, ALWAYS think through these strategic sequences:

1. ELIMINATION → CAPTURE CHAINS:
   - "If I kill enemy unit X, does that clear a path for another unit to seize a building?"
   - "Which enemy is blocking access to valuable buildings (HQ, income buildings)?"
   - Prioritize killing units that guard high-value targets

2. ACTION ORDERING MATTERS:
   - Actions execute sequentially - Unit A's action affects what Unit B can do
   - Kill blocking enemies BEFORE moving units that want to pass through
   - Move units OUT of the way before moving other units INTO those spaces
   - Attack with units that might die LAST (so they can still be useful)

3. SETUP MOVES:
   - Sometimes moving a unit into position (not attacking) enables a better play next turn
   - Consider: "If I move here, what can I threaten next turn?"
   - Block enemy paths to your HQ or valuable buildings

4. COMBINED ATTACKS:
   - A single unit might not kill an enemy, but two attacks might
   - Plan which units attack the same target and in what order
   - Weaker units can finish off enemies that stronger units damaged

5. ECONOMY PRIORITY CHECKLIST:
   - Can any unit seize a building THIS turn? (High priority)
   - Is killing an enemy required to enable a seize? (Do the kill first)
   - Are neutral buildings undefended? (Send a unit toward them)

6. DEFENSIVE AWARENESS:
   - After your moves, where will your units be? Can enemies attack them?
   - Don't leave units exposed if you can avoid it
   - Keep units near each other for mutual support

STRATEGY TIPS:
- Balance economy (capturing buildings) with military (building units)
- Protect your HQ at all costs
- Mages can disable key enemy units with paralyze
- Clerics keep your army healthy and mobile
- Archers are excellent for safe ranged attacks, especially from mountains
- Position units to protect each other

WHEN TO RESIGN:
Consider resigning ONLY when ALL of these conditions are true:
- You have no units left AND cannot afford to create any
- OR enemy units are about to capture your HQ and you cannot stop them
- OR you are vastly outnumbered with no realistic path to victory
Do NOT resign if you still have units, gold, or any chance of a comeback.

CRITICAL CONSTRAINTS:
- Only ONE unit can occupy any tile. You cannot create a unit on an occupied building.
- Each action in your list is executed sequentially - plan accordingly.
- If enemies are within 2-3 tiles of your HQ, defending it is your TOP priority.

Respond with ONLY the JSON object below. No extra text before or after."""


# =============================================================================
# TWO-PHASE PROMPTS - Planning then Execution
# =============================================================================

PROMPT_TWO_PHASE_PLAN = """You are an expert strategist for Reinforce Tactics, a turn-based strategy game.

GAME OBJECTIVE:
- Win by capturing the enemy HQ or eliminating all enemy units

UNIT TYPES (for reference):
- Warrior (W): Melee fighter, 15 HP, high attack/defense, 3 movement
- Mage (M): Ranged attacker (1-2 range), can PARALYZE enemies, 10 HP, 2 movement
- Cleric (C): Support unit, can HEAL allies and CURE paralysis, 8 HP, 2 movement
- Archer (A): Ranged (1-2, or 1-3 from mountains), cannot attack adjacent, 15 HP, 3 movement

BUILDING TYPES:
- HQ (h): 150 gold/turn, losing it = defeat
- Building (b): 100 gold/turn
- Tower (t): 50 gold/turn

YOUR TASK - STRATEGIC PLANNING:

Analyze the game state and create a STRATEGIC PLAN for this turn. Think about:

1. IMMEDIATE OPPORTUNITIES:
   - Can any unit seize a building right now?
   - Are there enemies that can be killed to unlock building captures?
   - Is the enemy HQ vulnerable?

2. THREATS TO ADDRESS:
   - Are enemies threatening your HQ?
   - Are valuable units in danger?
   - Do any allies need healing or curing?

3. ACTION SEQUENCES:
   - What order should units act to maximize effectiveness?
   - Which kills enable which captures?
   - Which moves need to happen before others?

4. ECONOMY DECISIONS:
   - Should you create new units? Which type and where?
   - Is expanding income or military strength more important now?

Respond with a JSON object:
{
    "situation_assessment": "Brief summary of current tactical situation",
    "primary_objective": "The main goal for this turn (e.g., 'Capture the eastern building', 'Defend HQ')",
    "key_sequences": [
        "Describe action sequence 1 (e.g., 'Kill warrior at [3,4] with mage, then move archer to seize building at [3,4]')",
        "Describe action sequence 2 if applicable"
    ],
    "unit_priorities": [
        {"unit_id": 0, "role": "What this unit should do and why"},
        {"unit_id": 1, "role": "What this unit should do and why"}
    ],
    "risks": "What could go wrong and how to mitigate"
}"""


PROMPT_TWO_PHASE_EXECUTE = """You are executing a turn in Reinforce Tactics based on a strategic plan.

AVAILABLE ACTIONS:
1. CREATE_UNIT: Spawn a unit at an owned building (costs gold)
2. MOVE: Move a unit to a reachable position
3. ATTACK: Attack an enemy unit
4. PARALYZE: (Mage only) Paralyze an enemy unit
5. HEAL: (Cleric only) Heal an adjacent ally
6. CURE: (Cleric only) Remove paralysis from an ally
7. SEIZE: Capture a structure by standing on it
8. END_TURN: Finish your turn

COMBAT RULES:
- Warriors attack adjacent only. Mages/Archers attack at range 1-2.
- Archers cannot attack adjacent enemies (distance 0)
- Counter-attacks occur unless attacker is an Archer (vs melee defenders)

CRITICAL CONSTRAINTS:
- Only ONE unit per tile
- Actions execute SEQUENTIALLY - order matters!
- Only use actions from the legal_actions provided

STRATEGIC PLAN FOR THIS TURN:
{plan}

Based on the plan above and the current game state, provide the EXACT actions to execute.
Translate the strategic plan into specific, legal game actions.

Respond with ONLY a JSON object:
{{
    "reasoning": "Brief explanation of how actions implement the plan",
    "actions": [
        {{"type": "ACTION_TYPE", ...action parameters...}}
    ]
}}

Only include actions that are legal based on the legal_actions provided."""


# =============================================================================
# PROMPT REGISTRY - Easy access by name
# =============================================================================

PROMPT_REGISTRY = {
    "basic": PROMPT_BASIC,
    "strategic": PROMPT_STRATEGIC,
    "two_phase_plan": PROMPT_TWO_PHASE_PLAN,
    "two_phase_execute": PROMPT_TWO_PHASE_EXECUTE,
}

# Default prompt for backwards compatibility
DEFAULT_PROMPT = PROMPT_BASIC


def get_prompt(name: str) -> str:
    """
    Get a system prompt by name.

    Args:
        name: The prompt name. One of: "basic", "strategic",
              "two_phase_plan", "two_phase_execute"

    Returns:
        The system prompt string.

    Raises:
        ValueError: If the prompt name is not recognized.

    Example:
        >>> from reinforcetactics.game.llm_prompts import get_prompt
        >>> prompt = get_prompt("strategic")
    """
    if name not in PROMPT_REGISTRY:
        available = ", ".join(PROMPT_REGISTRY.keys())
        raise ValueError(f"Unknown prompt '{name}'. Available prompts: {available}")
    return PROMPT_REGISTRY[name]


def list_prompts() -> list:
    """
    List all available prompt names.

    Returns:
        List of prompt names that can be used with get_prompt().

    Example:
        >>> from reinforcetactics.game.llm_prompts import list_prompts
        >>> print(list_prompts())
        ['basic', 'strategic', 'two_phase_plan', 'two_phase_execute']
    """
    return list(PROMPT_REGISTRY.keys())


def register_prompt(name: str, prompt: str) -> None:
    """
    Register a custom prompt at runtime.

    This allows users to add their own prompts without modifying this file.

    Args:
        name: The name to register the prompt under.
        prompt: The system prompt string.

    Example:
        >>> from reinforcetactics.game.llm_prompts import register_prompt, get_prompt
        >>> register_prompt("my_custom", "You are a cautious player...")
        >>> prompt = get_prompt("my_custom")
    """
    PROMPT_REGISTRY[name] = prompt
