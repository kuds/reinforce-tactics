"""
User interface module.

``Renderer`` is imported on first use, so the pygame-free data modules here
(``assets``, ``theme``) load without pygame -- ``reinforcetactics.constants``
re-exports ``assets`` and must stay importable in headless installs.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reinforcetactics.ui.renderer import Renderer

__all__ = ["Renderer"]


def __getattr__(name: str) -> Any:
    if name == "Renderer":
        from reinforcetactics.ui.renderer import Renderer

        return Renderer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
