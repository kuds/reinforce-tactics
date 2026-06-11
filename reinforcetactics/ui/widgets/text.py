"""Text layout helpers shared by the widget library and menu screens."""

from typing import List

import pygame

ELLIPSIS = "..."


def wrap_text(text: str, font: pygame.font.Font, max_width: int) -> List[str]:
    """Wrap ``text`` into lines that each fit within ``max_width`` pixels.

    Splits on whitespace and preserves explicit newlines. Words wider than
    ``max_width`` are broken mid-word so no line ever exceeds the limit.

    Args:
        text: The text to wrap. May contain ``\\n`` for forced breaks.
        font: Font used to measure line widths.
        max_width: Maximum line width in pixels.

    Returns:
        List of lines. Always contains at least one (possibly empty) line.
    """
    lines: List[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split(" ")
        current = ""
        for word in words:
            candidate = f"{current} {word}" if current else word
            if font.size(candidate)[0] <= max_width:
                current = candidate
                continue
            if current:
                lines.append(current)
                current = ""
            # Word alone is too wide: hard-break it.
            while font.size(word)[0] > max_width and len(word) > 1:
                cut = len(word)
                while cut > 1 and font.size(word[:cut])[0] > max_width:
                    cut -= 1
                lines.append(word[:cut])
                word = word[cut:]
            current = word
        lines.append(current)
    return lines if lines else [""]


def ellipsize(text: str, font: pygame.font.Font, max_width: int, keep_end: bool = False) -> str:
    """Shorten ``text`` with an ellipsis so it fits within ``max_width``.

    Args:
        text: The text to shorten.
        font: Font used to measure widths.
        max_width: Maximum width in pixels.
        keep_end: When True, keep the tail of the string and put the
            ellipsis at the front (useful for file paths where the end
            matters most).

    Returns:
        The original text if it fits, otherwise a shortened version.
    """
    if font.size(text)[0] <= max_width:
        return text

    for cut in range(len(text) - 1, 0, -1):
        candidate = ELLIPSIS + text[-cut:] if keep_end else text[:cut] + ELLIPSIS
        if font.size(candidate)[0] <= max_width:
            return candidate
    return ELLIPSIS
