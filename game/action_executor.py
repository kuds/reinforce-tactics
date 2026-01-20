"""
Action Executor for Reinforce Tactics.

This module handles unit action execution from the unit action menu.
"""


def handle_action_menu_result(game, menu_result, active_menu_ref, target_selection_unit_ref,
                               selected_unit_ref):
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
    if menu_result['type'] == 'cancel':
        # Cancel move if unit has moved
        if target_selection_unit_ref[0] and target_selection_unit_ref[0].has_moved:
            target_selection_unit_ref[0].cancel_move()
            print(f"Cancelled move for {target_selection_unit_ref[0].type}")
        target_selection_unit_ref[0] = None
        active_menu_ref[0] = None
        return None

    if menu_result['type'] == 'action_selected':
        # Process the selected action
        action = menu_result['action']
        active_menu_ref[0] = None

        # Execute action using helper function
        result = execute_unit_action(
            game, action, target_selection_unit_ref[0], selected_unit_ref
        )
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
    if action['type'] == 'wait':
        unit.end_unit_turn()
        print(f"{unit.type} ended turn")
        selected_unit_ref[0] = None
        return (False, None, None)

    if action['type'] == 'cancel_move':
        unit.cancel_move()
        print(f"Cancelled move for {unit.type}")
        selected_unit_ref[0] = None
        return (False, None, None)

    if action['type'] == 'capture':
        result = game.seize(unit)
        if result['captured']:
            print(f"{unit.type} captured structure!")
        unit.end_unit_turn()
        selected_unit_ref[0] = None
        return (False, None, None)

    if action['type'] in ['attack', 'paralyze', 'heal', 'cure', 'haste', 'defence_buff', 'attack_buff']:
        # Enter target selection mode
        targets = action['targets']
        if len(targets) == 1:
            # Only one target, execute immediately
            target = targets[0]
            if action['type'] == 'attack':
                game.attack(unit, target)
                print(f"{unit.type} attacked {target.type}")
            elif action['type'] == 'paralyze':
                game.paralyze(unit, target)
                print(f"{unit.type} paralyzed {target.type}")
            elif action['type'] == 'heal':
                game.heal(unit, target)
                print(f"{unit.type} healed {target.type}")
            elif action['type'] == 'cure':
                game.cure(unit, target)
                print(f"{unit.type} cured {target.type}")
            elif action['type'] == 'haste':
                game.haste(unit, target)
                print(f"{unit.type} hasted {target.type}")
            elif action['type'] == 'defence_buff':
                game.defence_buff(unit, target)
                print(f"{unit.type} granted defence buff to {target.type}")
            elif action['type'] == 'attack_buff':
                game.attack_buff(unit, target)
                print(f"{unit.type} granted attack buff to {target.type}")
            unit.end_unit_turn()
            selected_unit_ref[0] = None
            return (False, None, None)

        # Multiple targets, enter target selection mode
        print(f"Select target for {action['type']}")
        return (True, action, unit)

    return (False, None, unit)
