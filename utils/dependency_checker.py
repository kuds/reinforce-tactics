"""
Dependency checker utility for Reinforce Tactics.

This module checks if required dependencies are installed.
"""


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []

    try:
        import pygame  # noqa: F401
    except ImportError:
        missing.append("pygame")

    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")

    try:
        import pandas  # noqa: F401
    except ImportError:
        missing.append("pandas")

    if missing:
        print(f"❌ Missing required dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False

    return True
