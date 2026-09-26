"""The engine's action vocabulary: what ``GameState.apply_action`` and ``is_legal`` take and return.

An action is named the way ``GameState.get_legal_actions`` lists it: a kind
(one of its keys) and a payload (one of that key's entries). The GUI, the
rule bots, MCTS, the gym env and the LLM bots used to each keep their own
table from that shape to the engine method to call and to how its return
value says whether the engine carried the action out (review core-14).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# The kinds of action, in ``get_legal_actions`` order.
ACTION_KINDS: tuple[str, ...] = (
    "create_unit",
    "move",
    "attack",
    "paralyze",
    "heal",
    "cure",
    "haste",
    "defence_buff",
    "attack_buff",
    "seize",
    "end_turn",
)

# The payload key naming the acting unit, per kind (``create_unit`` and
# ``end_turn`` have none). Attack and the abilities also name their target
# unit, under ``"target"``, and their action records hold the actor's
# position under ``"<actor key>_pos"`` (replays find the actor by it).
ACTOR_KEYS: dict[str, str] = {
    "move": "unit",
    "attack": "attacker",
    "paralyze": "paralyzer",
    "heal": "healer",
    "cure": "curer",
    "haste": "sorcerer",
    "defence_buff": "sorcerer",
    "attack_buff": "sorcerer",
    "seize": "unit",
}


@dataclass(frozen=True)
class ActionResult:
    """What ``GameState.apply_action`` did.

    Attributes:
        kind: The action kind applied.
        accepted: Whether the engine carried the action out. When False,
            nothing changed and nothing was recorded.
        result: Exactly what the underlying ``GameState`` method returned
            (the attack or seize result dict, the HP healed, the created
            unit or None, ``end_turn``'s income breakdown, or a bool), for
            callers that report outcomes such as damage or captures.
    """

    kind: str
    accepted: bool
    result: Any
