"""
Action Executor for Reinforce Tactics.

This module handles unit action execution from the unit action menu.
"""

from reinforcetactics.core.actions import ACTOR_KEYS

# Targeted actions (get_legal_actions kinds): past-tense verb for the console
# log. The engine refuses an illegal action without changing anything
# (review §1.2) and apply_action reports whether it did.
TARGETED_ACTIONS = {
    "attack": "attacked",
    "paralyze": "paralyzed",
    "heal": "healed",
    "cure": "cured",
    "haste": "hasted",
    "defence_buff": "granted defence buff to",
    "attack_buff": "granted attack buff to",
}


def apply_targeted_action(game, kind, unit, target):
    """Run the targeted action ``kind`` from ``unit`` on ``target``.

    Returns:
        True if the engine carried it out. On False nothing changed, and the
        caller must not end the unit's turn: the refused action did not use it.
    """
    verb = TARGETED_ACTIONS[kind]
    if game.apply_action(kind, {ACTOR_KEYS[kind]: unit, "target": target}).accepted:
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
            if game.cancel_move(target_selection_unit_ref[0]):
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
        if game.cancel_move(unit):
            print(f"Cancelled move for {unit.type}")
        selected_unit_ref[0] = None
        return (False, None, None)

    if action["type"] == "capture":
        seized = game.apply_action("seize", {"unit": unit})
        if not seized.accepted:
            # Refused by the engine (see GameState.seize): nothing happened,
            # so the unit keeps its action.
            print(f"{unit.type} can't capture here right now (not a legal action)")
            selected_unit_ref[0] = unit
            return (False, None, unit)
        if seized.result["captured"]:
            print(f"{unit.type} captured structure!")
        # The engine already spent the action, or refreshed a hasted unit;
        # end_unit_turn closes the former and keeps the latter's extra action.
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
            # As for capture: True only if haste refreshed the unit.
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
