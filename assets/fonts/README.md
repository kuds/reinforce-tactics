# Bundled Fonts

These fonts ship with the game so the UI renders identically on every
platform. They are loaded by `reinforcetactics/utils/fonts.py`.

| File | Family | Role | License |
|------|--------|------|---------|
| `NotoSans-Regular.ttf` | [Noto Sans](https://notofonts.github.io/) | Body/UI text (`get_font`) | [OFL 1.1](OFL-NotoSans.txt) |
| `PixelifySans-Regular.ttf` | [Pixelify Sans](https://fonts.google.com/specimen/Pixelify+Sans) | Titles/headings (`get_display_font`) | [OFL 1.1](OFL-PixelifySans.txt) |

Both fonts cover Latin (including the accented characters used by the
French and Spanish translations) but not CJK. When the active language is
Korean or Chinese, the font loader falls back to a system font with CJK
coverage.
