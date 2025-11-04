# constants.py

# Defaults if nothing is uploaded in the sidebar.
PRIMARY_JSON_PATH = "seed.json"
SECONDARY_JSON_PATH = ""

# Backwards-compatible alias used by older parts of the app;
# the app will still merge PRIMARY + SECONDARY by default.
DEFAULT_JSON_PATH = PRIMARY_JSON_PATH

# Personalisation keys seen across datasets
AVATAR_KEYS = ["avatar", "avatar_name", "selected_avatar", "active_avatar", "avatarId", "avatar_id"]
FONT_KEYS = ["font", "font_name", "selected_font", "text_font"]
BACKGROUND_KEYS = ["background", "background_name", "background_theme", "bg_theme", "bg", "selected_background"]
