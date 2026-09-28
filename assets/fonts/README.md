# Bundled Fonts

These fonts ship with the game so the UI renders identically on every
platform. They are loaded by `reinforcetactics/utils/fonts.py`.

| File | Family | Role | License |
|------|--------|------|---------|
| `NotoSans-Regular.ttf` | [Noto Sans](https://notofonts.github.io/) | Body/UI text (`get_font`) | [OFL 1.1](OFL-NotoSans.txt) |
| `Jersey15-Regular.ttf` | [Jersey 15](https://fonts.google.com/specimen/Jersey+15) | Titles/headings (`get_display_font`) | [OFL 1.1](OFL-Jersey15.txt) |

Both fonts cover Latin (including the accented characters used by the
French and Spanish translations) but not CJK. When the active language is
Korean or Chinese, the font loader falls back to a system font with CJK
coverage.

The display font must have no standard ligatures (`liga`): SDL_ttf shapes
text with HarfBuzz and applies them, and pygame offers no switch to turn
them off. The previous display font merged "fi", "fl" and "ff" into single
glyphs that read as "A" or "F" at game sizes ("Configure" came out as
"ConAgure"). Jersey 15 has no ligature lookups; `tests/test_fonts.py`
checks that.
