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
2. Mage (M): Cost 300 gold, HP 10, Attack 8 (adjacent) or 12 (range), Defense 4, Movement 2
   - Can attack at range (1-2 spaces)
   - Can PARALYZE enemies (disable them for 3 turns, 2-turn cooldown)
3. Cleric (C): Cost 200 gold, HP 8, Attack 2, Defense 4, Movement 2
   - Can HEAL allies (range 1-2) and CURE paralyzed units (range 1-2)
4. Archer (A): Cost 250 gold, HP 15, Attack 5, Defense 1, Movement 3
   - Ranged unit that attacks at distance 2-3 (2-4 on mountains)
   - Cannot attack adjacent enemies (distance 0-1)
   - Indirect unit: melee units cannot counter-attack when hit by Archer
   - Other Archers, Mages, and Sorcerers CAN counter-attack if Archer is within their range
5. Knight (K): Cost 350 gold, HP 18, Attack 8, Defense 5, Movement 4
   - Heavy cavalry unit with high mobility
   - CHARGE: +50% damage if moved 3+ tiles before attacking
6. Rogue (R): Cost 350 gold, HP 12, Attack 9, Defense 3, Movement 4
   - Fast melee assassin
   - FLANK: +50% damage if target is adjacent to another friendly unit
   - EVADE: 15% chance to dodge counter-attacks (30% in forest)
7. Sorcerer (S): Cost 400 gold, HP 10, Attack 6 (adjacent) or 8 (range), Defense 3, Movement 2
   - Support caster with ranged attacks (1-2 spaces)
   - HASTE: Grant an ally an extra action (3-turn cooldown)
   - DEFENCE BUFF: Give ally -35% damage taken for 3 turns (3-turn cooldown)
   - ATTACK BUFF: Give ally +35% damage dealt for 3 turns (3-turn cooldown)
8. Barbarian (B): Cost 400 gold, HP 20, Attack 10, Defense 2, Movement 5
   - High HP glass cannon with excellent mobility
   - Best for rapid strikes and flanking maneuvers

BUILDING TYPES:
- HQ (h): Generates 150 gold/turn, losing it means defeat
- Building (b): Generates 100 gold/turn, used to recruit units
- Tower (t): Generates 50 gold/turn, defensive structure

AVAILABLE ACTIONS:
1. CREATE_UNIT: Spawn a unit at an owned building (costs gold)
2. MOVE: Move a unit to a reachable position (up to movement range)
3. ATTACK: Attack an enemy unit (adjacent for most units, ranged for Mage/Archer/Sorcerer)
4. PARALYZE: (Mage only) Paralyze an enemy unit within range 1-2
5. HEAL: (Cleric only) Heal an ally unit within range 1-2
6. CURE: (Cleric only) Remove paralysis from an ally within range 1-2
7. HASTE: (Sorcerer only) Grant an ally an extra action this turn
8. DEFENCE_BUFF: (Sorcerer only) Give an ally 35% damage reduction for 3 turns
9. ATTACK_BUFF: (Sorcerer only) Give an ally 35% damage boost for 3 turns
10. SEIZE: Capture a neutral/enemy structure by standing on it
11. END_TURN: Finish your turn
12. RESIGN: Concede the game (use only as last resort when victory is impossible)

COMBAT RULES:
- Most units can only attack adjacent enemies (orthogonally, not diagonally)
- Mages and Sorcerers can attack at range 1-2, Archers at range 2-3 (or 2-4 on mountains)
- Archers cannot attack at distance 0-1 (adjacent or very close)
- Attacked units counter-attack if they can, except melee units cannot counter Archers
- Paralyzed units cannot move or attack
- Units can move then attack, but NOT attack then move (action ends unit's turn)
- Knight's Charge: +50% damage if moved 3+ tiles before attacking
- Rogue's Flank: +50% damage if target is adjacent to friendly unit
- Rogue's Evade: 15% dodge chance (30% in forest)

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
- Knights excel at charging into battle after long moves
- Rogues work best when flanking enemies with allies
- Sorcerers can buff allies or haste them for extra actions
- Barbarians have high HP and mobility for rapid strikes
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
2. Mage (M): Cost 300 gold, HP 10, Attack 8 (adjacent) or 12 (range), Defense 4, Movement 2
   - Can attack at range (1-2 spaces)
   - Can PARALYZE enemies (disable them for 3 turns, 2-turn cooldown)
3. Cleric (C): Cost 200 gold, HP 8, Attack 2, Defense 4, Movement 2
   - Can HEAL allies (range 1-2) and CURE paralyzed units (range 1-2)
4. Archer (A): Cost 250 gold, HP 15, Attack 5, Defense 1, Movement 3
   - Ranged unit that attacks at distance 2-3 (2-4 on mountains)
   - Cannot attack adjacent enemies (distance 0-1)
   - Indirect unit: melee units cannot counter-attack when hit by Archer
   - Other Archers, Mages, and Sorcerers CAN counter-attack if Archer is within their range
5. Knight (K): Cost 350 gold, HP 18, Attack 8, Defense 5, Movement 4
   - Heavy cavalry unit with high mobility
   - CHARGE: +50% damage if moved 3+ tiles before attacking
6. Rogue (R): Cost 350 gold, HP 12, Attack 9, Defense 3, Movement 4
   - Fast melee assassin
   - FLANK: +50% damage if target is adjacent to another friendly unit
   - EVADE: 15% chance to dodge counter-attacks (30% in forest)
7. Sorcerer (S): Cost 400 gold, HP 10, Attack 6 (adjacent) or 8 (range), Defense 3, Movement 2
   - Support caster with ranged attacks (1-2 spaces)
   - HASTE: Grant an ally an extra action (3-turn cooldown)
   - DEFENCE BUFF: Give ally -35% damage taken for 3 turns (3-turn cooldown)
   - ATTACK BUFF: Give ally +35% damage dealt for 3 turns (3-turn cooldown)
8. Barbarian (B): Cost 400 gold, HP 20, Attack 10, Defense 2, Movement 5
   - High HP glass cannon with excellent mobility
   - Best for rapid strikes and flanking maneuvers

BUILDING TYPES:
- HQ (h): Generates 150 gold/turn, losing it means defeat
- Building (b): Generates 100 gold/turn, used to recruit units
- Tower (t): Generates 50 gold/turn, defensive structure

AVAILABLE ACTIONS:
1. CREATE_UNIT: Spawn a unit at an owned building (costs gold)
2. MOVE: Move a unit to a reachable position (up to movement range)
3. ATTACK: Attack an enemy unit (adjacent for most units, ranged for Mage/Archer/Sorcerer)
4. PARALYZE: (Mage only) Paralyze an enemy unit within range 1-2
5. HEAL: (Cleric only) Heal an ally unit within range 1-2
6. CURE: (Cleric only) Remove paralysis from an ally within range 1-2
7. HASTE: (Sorcerer only) Grant an ally an extra action this turn
8. DEFENCE_BUFF: (Sorcerer only) Give an ally 35% damage reduction for 3 turns
9. ATTACK_BUFF: (Sorcerer only) Give an ally 35% damage boost for 3 turns
10. SEIZE: Capture a neutral/enemy structure by standing on it
11. END_TURN: Finish your turn
12. RESIGN: Concede the game (use only as last resort when victory is impossible)

COMBAT RULES:
- Most units can only attack adjacent enemies (orthogonally, not diagonally)
- Mages and Sorcerers can attack at range 1-2, Archers at range 2-3 (or 2-4 on mountains)
- Archers cannot attack at distance 0-1 (adjacent or very close)
- Attacked units counter-attack if they can, except melee units cannot counter Archers
- Paralyzed units cannot move or attack
- Units can move then attack, but NOT attack then move (action ends unit's turn)
- Knight's Charge: +50% damage if moved 3+ tiles before attacking
- Rogue's Flank: +50% damage if target is adjacent to friendly unit
- Rogue's Evade: 15% dodge chance (30% in forest)

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
- Knights excel at charging into battle after long moves
- Rogues work best when flanking enemies with allies
- Sorcerers can buff allies or haste them for extra actions
- Barbarians have high HP and mobility for rapid strikes
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
- Mage (M): Ranged attacker (1-2 range), can PARALYZE enemies (2-turn cooldown), 10 HP, 2 movement
- Cleric (C): Support unit, can HEAL/CURE allies (range 1-2), 8 HP, 2 movement
- Archer (A): Ranged (2-3, or 2-4 from mountains), cannot attack at range 0-1, 15 HP, 3 movement
- Knight (K): Heavy cavalry, CHARGE (+50% dmg if moved 3+ tiles), 18 HP, 4 movement
- Rogue (R): Assassin, FLANK (+50% dmg if target adjacent to ally), EVADE (15% dodge, 30% in forest), 12 HP, 4 movement
- Sorcerer (S): Support caster (1-2 range), HASTE/DEFENCE_BUFF/ATTACK_BUFF (+35%) allies, 10 HP, 2 movement
- Barbarian (B): Glass cannon, high HP and mobility, 20 HP, 5 movement

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


# =============================================================================
# DYNAMIC UNIT DESCRIPTIONS - For filtering based on enabled units
# =============================================================================

# Full descriptions for each unit type
UNIT_DESCRIPTIONS = {
    'W': """Warrior (W): Cost 200 gold, HP 15, Attack 10, Defense 6, Movement 3
   - Strong melee fighter, attacks adjacent enemies only""",
    'M': """Mage (M): Cost 300 gold, HP 10, Attack 8 (adjacent) or 12 (range), Defense 4, Movement 2
   - Can attack at range (1-2 spaces)
   - Can PARALYZE enemies (disable them for 3 turns, 2-turn cooldown)""",
    'C': """Cleric (C): Cost 200 gold, HP 8, Attack 2, Defense 4, Movement 2
   - Can HEAL allies (range 1-2) and CURE paralyzed units (range 1-2)""",
    'A': """Archer (A): Cost 250 gold, HP 15, Attack 5, Defense 1, Movement 3
   - Ranged unit that attacks at distance 2-3 (2-4 on mountains)
   - Cannot attack adjacent enemies (distance 0-1)
   - Indirect unit: melee units cannot counter-attack when hit by Archer
   - Other Archers, Mages, and Sorcerers CAN counter-attack if Archer is within their range""",
    'K': """Knight (K): Cost 350 gold, HP 18, Attack 8, Defense 5, Movement 4
   - Heavy cavalry unit with high mobility
   - CHARGE: +50% damage if moved 3+ tiles before attacking""",
    'R': """Rogue (R): Cost 350 gold, HP 12, Attack 9, Defense 3, Movement 4
   - Fast melee assassin
   - FLANK: +50% damage if target is adjacent to another friendly unit
   - EVADE: 15% chance to dodge counter-attacks (30% in forest)""",
    'S': """Sorcerer (S): Cost 400 gold, HP 10, Attack 6 (adjacent) or 8 (range), Defense 3, Movement 2
   - Support caster with ranged attacks (1-2 spaces)
   - HASTE: Grant an ally an extra action (3-turn cooldown)
   - DEFENCE BUFF: Give ally -35% damage taken for 3 turns (3-turn cooldown)
   - ATTACK BUFF: Give ally +35% damage dealt for 3 turns (3-turn cooldown)""",
    'B': """Barbarian (B): Cost 400 gold, HP 20, Attack 10, Defense 2, Movement 5
   - High HP glass cannon with excellent mobility
   - Best for rapid strikes and flanking maneuvers"""
}

# Short descriptions for two-phase prompts
UNIT_DESCRIPTIONS_SHORT = {
    'W': "Warrior (W): Melee fighter, 15 HP, high attack/defense, 3 movement",
    'M': "Mage (M): Ranged attacker (1-2 range), can PARALYZE enemies (2-turn cooldown), 10 HP, 2 movement",
    'C': "Cleric (C): Support unit, can HEAL/CURE allies (range 1-2), 8 HP, 2 movement",
    'A': "Archer (A): Ranged (2-3, or 2-4 from mountains), cannot attack at range 0-1, 15 HP, 3 movement",
    'K': "Knight (K): Heavy cavalry, CHARGE (+50% dmg if moved 3+ tiles), 18 HP, 4 movement",
    'R': "Rogue (R): Assassin, FLANK (+50% dmg if target adjacent to ally), EVADE (15% dodge, 30% in forest), 12 HP, 4 movement",
    'S': "Sorcerer (S): Support caster (1-2 range), HASTE/DEFENCE_BUFF/ATTACK_BUFF (+35%) allies, 10 HP, 2 movement",
    'B': "Barbarian (B): Glass cannon, high HP and mobility, 20 HP, 5 movement"
}

# Strategy tips specific to each unit
UNIT_STRATEGY_TIPS = {
    'M': "Mages can disable key enemy units with paralyze",
    'C': "Clerics keep your army healthy and mobile",
    'A': "Archers are excellent for safe ranged attacks, especially from mountains",
    'K': "Knights excel at charging into battle after long moves",
    'R': "Rogues work best when flanking enemies with allies",
    'S': "Sorcerers can buff allies or haste them for extra actions",
    'B': "Barbarians have high HP and mobility for rapid strikes"
}


def get_unit_types_section(enabled_units: list) -> str:
    """
    Generate the UNIT TYPES section of the prompt based on enabled units.

    Args:
        enabled_units: List of enabled unit type codes (e.g., ['W', 'M', 'C'])

    Returns:
        Formatted string with unit type descriptions
    """
    if not enabled_units:
        enabled_units = ['W', 'M', 'C', 'A', 'K', 'R', 'S', 'B']

    lines = ["UNIT TYPES:"]
    for i, unit_type in enumerate(enabled_units, 1):
        if unit_type in UNIT_DESCRIPTIONS:
            lines.append(f"{i}. {UNIT_DESCRIPTIONS[unit_type]}")

    return "\n".join(lines)


def get_unit_types_section_short(enabled_units: list) -> str:
    """
    Generate a short UNIT TYPES section for two-phase prompts.

    Args:
        enabled_units: List of enabled unit type codes

    Returns:
        Formatted string with short unit descriptions
    """
    if not enabled_units:
        enabled_units = ['W', 'M', 'C', 'A', 'K', 'R', 'S', 'B']

    lines = ["UNIT TYPES (for reference):"]
    for unit_type in enabled_units:
        if unit_type in UNIT_DESCRIPTIONS_SHORT:
            lines.append(f"- {UNIT_DESCRIPTIONS_SHORT[unit_type]}")

    return "\n".join(lines)


def get_strategy_tips(enabled_units: list) -> str:
    """
    Generate strategy tips based on enabled units.

    Args:
        enabled_units: List of enabled unit type codes

    Returns:
        Formatted string with relevant strategy tips
    """
    if not enabled_units:
        enabled_units = ['W', 'M', 'C', 'A', 'K', 'R', 'S', 'B']

    lines = [
        "STRATEGY TIPS:",
        "- Balance economy (capturing buildings) with military (building units)",
        "- Protect your HQ at all costs"
    ]

    for unit_type in enabled_units:
        if unit_type in UNIT_STRATEGY_TIPS:
            lines.append(f"- {UNIT_STRATEGY_TIPS[unit_type]}")

    lines.append("- Position units to protect each other")

    return "\n".join(lines)


def get_available_actions_section(enabled_units: list) -> str:
    """
    Generate the AVAILABLE ACTIONS section based on enabled units.

    Args:
        enabled_units: List of enabled unit type codes

    Returns:
        Formatted string with available actions
    """
    if not enabled_units:
        enabled_units = ['W', 'M', 'C', 'A', 'K', 'R', 'S', 'B']

    lines = [
        "AVAILABLE ACTIONS:",
        "1. CREATE_UNIT: Spawn a unit at an owned building (costs gold)",
        "2. MOVE: Move a unit to a reachable position (up to movement range)",
        "3. ATTACK: Attack an enemy unit (adjacent for most units, ranged for Mage/Archer/Sorcerer)"
    ]

    action_num = 4

    # Add unit-specific actions
    if 'M' in enabled_units:
        lines.append(f"{action_num}. PARALYZE: (Mage only) Paralyze an enemy unit within range 1-2")
        action_num += 1

    if 'C' in enabled_units:
        lines.append(f"{action_num}. HEAL: (Cleric only) Heal an adjacent ally unit")
        action_num += 1
        lines.append(f"{action_num}. CURE: (Cleric only) Remove paralysis from an adjacent ally")
        action_num += 1

    if 'S' in enabled_units:
        lines.append(f"{action_num}. HASTE: (Sorcerer only) Grant an ally an extra action this turn")
        action_num += 1
        lines.append(f"{action_num}. DEFENCE_BUFF: (Sorcerer only) Give an ally 35% damage reduction for 3 turns")
        action_num += 1
        lines.append(f"{action_num}. ATTACK_BUFF: (Sorcerer only) Give an ally 35% damage boost for 3 turns")
        action_num += 1

    lines.append(f"{action_num}. SEIZE: Capture a neutral/enemy structure by standing on it")
    action_num += 1
    lines.append(f"{action_num}. END_TURN: Finish your turn")
    action_num += 1
    lines.append(f"{action_num}. RESIGN: Concede the game (use only as last resort when victory is impossible)")

    return "\n".join(lines)


def get_dynamic_prompt(base_prompt_name: str, enabled_units: list) -> str:
    """
    Get a prompt with unit sections dynamically filtered based on enabled units.

    Args:
        base_prompt_name: Name of the base prompt ('basic', 'strategic', etc.)
        enabled_units: List of enabled unit type codes

    Returns:
        Prompt string with only enabled units mentioned
    """
    base_prompt = get_prompt(base_prompt_name)

    # If all units are enabled, return the original prompt
    all_units = ['W', 'M', 'C', 'A', 'K', 'R', 'S', 'B']
    if set(enabled_units) == set(all_units):
        return base_prompt

    # Generate dynamic sections
    unit_types_section = get_unit_types_section(enabled_units)
    available_actions_section = get_available_actions_section(enabled_units)
    strategy_tips_section = get_strategy_tips(enabled_units)

    # For basic and strategic prompts, we need to replace sections
    # This is a simplified approach - for production, consider using templates
    disabled_units_note = ""
    disabled = [u for u in all_units if u not in enabled_units]
    if disabled:
        from reinforcetactics.constants import UNIT_DATA
        disabled_names = [UNIT_DATA[u]['name'] for u in disabled]
        disabled_units_note = f"\nNOTE: The following units are DISABLED for this game and cannot be created: {', '.join(disabled_names)}\n"

    # Add the disabled units note near the beginning of the prompt
    if disabled_units_note:
        # Insert after the game objective section
        if "GAME OBJECTIVE:" in base_prompt:
            parts = base_prompt.split("UNIT TYPES:", 1)
            if len(parts) == 2:
                return parts[0] + disabled_units_note + "\n" + unit_types_section + "\n" + parts[1].split("\n\n", 1)[-1]

    return base_prompt
