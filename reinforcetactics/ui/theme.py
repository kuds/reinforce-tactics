"""
Shared UI color theme and dimension constants for the Pygame interface.

All menu screens and the in-game renderer should import colors from here
so the visual style stays consistent across the entire game.
"""

# ── Menu chrome ──────────────────────────────────────────────────────
BG = (30, 30, 40)
TEXT = (255, 255, 255)
TEXT_SECONDARY = (200, 200, 210)
TEXT_MUTED = (160, 160, 170)
TITLE = (100, 200, 255)
SELECTED = (255, 200, 50)
HOVER = (200, 180, 100)
DISABLED = (100, 100, 120)

OPTION_BG = (50, 50, 65)
OPTION_BG_HOVER = (70, 70, 90)
OPTION_BG_SELECTED = (80, 80, 100)

PANEL_BG = (40, 40, 50)
PANEL_BORDER = (80, 80, 100)
BORDER = (100, 150, 200)

# ── Buttons ──────────────────────────────────────────────────────────
BTN_CONFIRM = (80, 150, 80)
BTN_CONFIRM_HOVER = (100, 180, 100)
BTN_CONFIRM_BORDER_HOVER = (150, 255, 150)
BTN_CANCEL = (150, 80, 80)
BTN_CANCEL_HOVER = (180, 100, 100)
BTN_CANCEL_BORDER_HOVER = (255, 150, 150)
BTN_CLOSE = (200, 50, 50)
BTN_CLOSE_HOVER = (255, 80, 80)
BTN_QUIT = (150, 120, 50)
BTN_QUIT_HOVER = (180, 150, 70)
BTN_QUIT_BORDER_HOVER = (255, 220, 120)

BTN_END_TURN = (70, 120, 70)
BTN_END_TURN_HOVER = (100, 150, 100)
BTN_RESIGN = (120, 50, 50)
BTN_RESIGN_HOVER = (150, 70, 70)
BTN_RESIGN_BORDER = (200, 100, 100)

# Disabled-option styling (used by Menu base class for enabled=False options)
OPTION_BG_DISABLED = (45, 45, 55)
TEXT_DISABLED = DISABLED

# ── Status indicator colors ──────────────────────────────────────────
STATUS_PARALYSIS = (148, 0, 211)
STATUS_PARALYSIS_TINT = (200, 150, 255)
STATUS_HASTE = (0, 200, 255)
STATUS_DISABLED_TINT = (128, 128, 128)

STATUS_VALID = (100, 255, 100)
STATUS_INVALID = (255, 100, 100)
STATUS_WARNING = (255, 200, 100)

# ── Health bars ──────────────────────────────────────────────────────
HEALTH_GOOD = (0, 200, 0)
HEALTH_INJURED = (255, 165, 0)
HEALTH_CRITICAL = (255, 0, 0)
HEALTH_BAR_BG = (100, 0, 0)
HEALTH_STRUCTURE_BG = (100, 100, 100)

HEALTH_STRUCTURE_HQ = (255, 200, 0)
HEALTH_STRUCTURE_BUILDING = (0, 200, 200)
HEALTH_STRUCTURE_TOWER = (200, 200, 200)

# ── In-game HUD ──────────────────────────────────────────────────────
HUD_GOLD_TEXT = (255, 215, 0)
HUD_TURN_BG = (50, 50, 65)
HUD_TURN_BORDER = (100, 150, 200)
HUD_FOW_TEXT = (180, 180, 220)
HUD_FOW_BG = (40, 40, 60)
HUD_FOW_BORDER = (80, 80, 120)
TOOLTIP_BG = (30, 30, 45)

# ── Overlay colors ───────────────────────────────────────────────────
OVERLAY_MOVEMENT = (255, 255, 255)
OVERLAY_MOVEMENT_BORDER = (255, 255, 0)
OVERLAY_TARGET = (255, 100, 100)
OVERLAY_TARGET_BORDER = (255, 0, 0)
OVERLAY_ATTACK_RANGE = (255, 150, 50)
OVERLAY_ATTACK_RANGE_BORDER = (255, 100, 0)
OVERLAY_FOG = (0, 0, 0, 128)

# ── Dimensions ───────────────────────────────────────────────────────
BORDER_RADIUS = 8
BORDER_RADIUS_SMALL = 4
BORDER_RADIUS_DIALOG = 12
BORDER_WIDTH_HOVER = 2
BORDER_WIDTH_DIALOG = 3
HEALTH_BAR_UNIT_HEIGHT = 5
HEALTH_BAR_STRUCTURE_HEIGHT = 4
HEALTH_BAR_MARGIN = 3
MENU_OPTION_SPACING = 60
OPTION_PADDING_X = 40
OPTION_PADDING_Y = 10

# ── Font sizes ───────────────────────────────────────────────────────
FONT_SIZE_TITLE = 48
FONT_SIZE_HEADING = 32
FONT_SIZE_OPTION = 36
FONT_SIZE_BODY = 24
FONT_SIZE_HINT = 18
FONT_SIZE_INDICATOR = 24

# ── Dialog overlay (modal dim) ───────────────────────────────────────
DIALOG_OVERLAY_COLOR = (0, 0, 0, 150)

# ── Frame rate (menus/dialogs unified) ───────────────────────────────
MENU_FRAMERATE = 30
