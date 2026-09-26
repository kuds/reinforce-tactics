"""
Action Executor for Reinforce Tactics.

This module handles unit action execution from the unit action menu.
"""

# Targeted actions: GameState method, past-tense verb for the console log.
# The engine refuses an illegal action without changing anything (review
# §1.2); each entry's result is read by ``_accepted`` below.
TARGETED_ACTIONS = {
    "attack": ("attack", "attacked"),
    "paralyze": ("paralyze", "paralyzed"),
    "heal": ("heal", "healed"),
    "cure": ("cure", "cured"),
    "haste": ("haste", "hasted"),
    "defence_buff": ("defence_buff", "granted defence buff to"),
    "attack_buff": ("attack_buff", "granted attack buff to"),
}


def _accepted(kind, result):
    """Whether the engine carried out ``kind``, judged from its return value.

    attack returns its result dict with ``damage`` 0 when refused (an executed
    attack always deals at least 1); heal returns the HP healed (0 when
    refused); the other abilities return a bool.
    """
    if kind == "attack":
        return result["damage"] > 0
    if kind == "heal":
        return result > 0
    return bool(result)


def apply_targeted_action(game, kind, unit, target):
    """Run the targeted action ``kind`` from ``unit`` on ``target``.

    Returns:
        True if the engine carried it out. On False nothing changed, and the
        caller must not end the unit's turn: the refused action did not use it.
    """
    method, verb = TARGETED_ACTIONS[kind]
    result = getattr(game, method)(unit, target)
    if _accepted(kind, result):
        print(f"{unit.type} {verb} {target.type}")
        return True
    print(f"{unit.type} can't {kind.replace('_', ' ')} {target.type} right now (not a legal action)")
    return False


def handle_action_menu_result(game, menu_result, active_menu_ref, target_selection_unit_ref, selected_unit_ref):
    """
    Handle result from UnitActionMenu interaction.

    Args:
        game: The GameState instance
        menu_result: Result dictionary from menu interaction
        active_menu_ref: Reference list for active_menu [active_menu]
        target_selection_unit_ref: Reference list for target_selection_unit [target_selection_unit]
        selected_unit_ref: Reference list for selected_unit [selected_unit]

    Returns:
        Tuple of (target_selection_mode, target_selection_action) or None
    """
    if menu_result["type"] == "cancel":
        # Cancel move if unit has moved
        if target_selection_unit_ref[0] and target_selection_unit_ref[0].has_moved:
            game.cancel_move(target_selection_unit_ref[0])
            print(f"Cancelled move for {target_selection_unit_ref[0].type}")
        target_selection_unit_ref[0] = None
        active_menu_ref[0] = None
        return None

    if menu_result["type"] == "action_selected":
        # Process the selected action
        action = menu_result["action"]
        active_menu_ref[0] = None

        # Execute action using helper function
        result = execute_unit_action(game, action, target_selection_unit_ref[0], selected_unit_ref)
        target_selection_mode, target_selection_action, target_selection_unit_ref[0] = result
        return (target_selection_mode, target_selection_action)

    return None


def execute_unit_action(game, action, unit, selected_unit_ref):
    """
    Execute a unit action from the menu.

    Args:
        game: The GameState instance
        action: The action dictionary from the menu
        unit: The unit performing the action
        selected_unit_ref: Reference list to clear selection [selected_unit]

    Returns:
        Tuple of (target_selection_mode, target_selection_action, unit or None)
    """
    if action["type"] == "wait":
        can_still_act = game.end_unit_turn(unit)
        if can_still_act:
            print(f"{unit.type} used haste action (can act again)")
            # Keep unit selected for another action
            selected_unit_ref[0] = unit
            return (False, None, unit)
        print(f"{unit.type} ended turn")
        selected_unit_ref[0] = None
        return (False, None, None)

    if action["type"] == "cancel_move":
        game.cancel_move(unit)
        print(f"Cancelled move for {unit.type}")
        selected_unit_ref[0] = None
        return (False, None, None)

    if action["type"] == "capture":
        result = game.seize(unit)
        if "damage" not in result:
            # Refused by the engine (see GameState.seize): nothing happened,
            # so the unit keeps its action.
            print(f"{unit.type} can't capture here right now (not a legal action)")
            selected_unit_ref[0] = unit
            return (False, None, unit)
        if result["captured"]:
            print(f"{unit.type} captured structure!")
        can_still_act = game.end_unit_turn(unit)
        if can_still_act:
            print(f"{unit.type} used haste action (can act again)")
            # Keep unit selected for another action
            selected_unit_ref[0] = unit
            return (False, None, unit)
        selected_unit_ref[0] = None
        return (False, None, None)

    if action["type"] in ["attack", "paralyze", "heal", "cure", "haste", "defence_buff", "attack_buff"]:
        # Enter target selection mode
        targets = action["targets"]
        if len(targets) == 1:
            # Only one target, execute immediately
            target = targets[0]
            if not apply_targeted_action(game, action["type"], unit, target):
                selected_unit_ref[0] = unit
                return (False, None, unit)
            can_still_act = game.end_unit_turn(unit)
            if can_still_act:
                print(f"{unit.type} used haste action (can act again)")
                # Keep unit selected for another action
                selected_unit_ref[0] = unit
                return (False, None, unit)
            selected_unit_ref[0] = None
            return (False, None, None)

        # Multiple targets, enter target selection mode
        print(f"Select target for {action['type']}")
        return (True, action, unit)

    return (False, None, unit)
