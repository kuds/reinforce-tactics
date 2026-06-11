"""Reusable UI widgets shared by menu screens and dialogs.

Centralizes button, text-input, and modal-dialog rendering so every screen
draws them the same way (and theme changes take effect everywhere at once).
"""

from reinforcetactics.ui.widgets.button import (
    CANCEL,
    CONFIRM,
    MENU_OPTION,
    MENU_OPTION_SMALL,
    PANEL_BUTTON,
    QUIT,
    Button,
    ButtonStyle,
    CloseButton,
)
from reinforcetactics.ui.widgets.dialog import Dialog
from reinforcetactics.ui.widgets.text import ellipsize, wrap_text
from reinforcetactics.ui.widgets.text_input import TextInput

__all__ = [
    "Button",
    "ButtonStyle",
    "CloseButton",
    "Dialog",
    "TextInput",
    "wrap_text",
    "ellipsize",
    "MENU_OPTION",
    "MENU_OPTION_SMALL",
    "PANEL_BUTTON",
    "CONFIRM",
    "CANCEL",
    "QUIT",
]
