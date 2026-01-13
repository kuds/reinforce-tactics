"""Cross-platform clipboard utility with macOS fallback."""
import sys
import subprocess
from typing import Optional

import pygame


def get_clipboard_text() -> Optional[str]:
    """
    Get text from the system clipboard.

    Uses pygame.scrap on Windows/Linux, falls back to pbpaste on macOS.

    Returns:
        Clipboard text or None if unavailable
    """
    # Try pygame.scrap first
    try:
        if pygame.scrap.get_init():
            clipboard_text = pygame.scrap.get(pygame.SCRAP_TEXT)
            if clipboard_text:
                # Decode bytes to string and strip null characters
                return clipboard_text.decode('utf-8').rstrip('\x00')
    except (pygame.error, UnicodeDecodeError, AttributeError):
        pass

    # Fallback for macOS
    if sys.platform == 'darwin':
        try:
            result = subprocess.run(
                ['pbpaste'],
                capture_output=True,
                text=True,
                timeout=1,
                check=False
            )
            if result.returncode == 0:
                return result.stdout
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    return None


def set_clipboard_text(text: str) -> bool:
    """
    Set text to the system clipboard.

    Uses pygame.scrap on Windows/Linux, falls back to pbcopy on macOS.

    Args:
        text: Text to copy to clipboard

    Returns:
        True if successful, False otherwise
    """
    # Try pygame.scrap first
    try:
        if pygame.scrap.get_init():
            pygame.scrap.put(pygame.SCRAP_TEXT, text.encode('utf-8'))
            return True
    except (pygame.error, AttributeError):
        pass

    # Fallback for macOS
    if sys.platform == 'darwin':
        try:
            subprocess.run(
                ['pbcopy'],
                input=text.encode('utf-8'),
                timeout=1,
                check=True
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    return False
