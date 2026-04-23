"""Reinforce Tactics - Turn-Based Strategy Game"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("reinforcetactics")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
