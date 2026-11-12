"""Configuration: ~/.config/mouthpiece/config.toml (+ secrets.toml for API keys, mode 0600)."""

from __future__ import annotations

import copy
import json
import os
import re
import tomllib
from pathlib import Path

DEFAULT_CONFIG_TEXT = """\
# Mouthpiece — push-to-talk dictation for Linux.
# Edit this file, then run `mouthpiece reload` (or use `mouthpiece config set ...`).

[hotkey]
# Any key or combination. Examples: "RIGHTCTRL", "CTRL+SPACE", "SUPER+ALT", "F9",
# "SHIFT+F7", "MOUSE_SIDE" (mouse thumb button). Pick one interactively: mouthpiece set-hotkey
key = "RIGHTCTRL"
# hold   = talk while holding, release to paste (push-to-talk)
# toggle = press once to start, press again to stop and paste
# hybrid = hold for push-to-talk, or quick-tap to start hands-free (tap again to stop)
mode = "hold"
# Press while recording to throw the recording away ("" disables).
cancel_key = "ESC"
# On GNOME, stop that Esc from also reaching the app while you dictate (WhatsApp would close
# the chat, Claude Code would interrupt). Uses a temporary GNOME shortcut, removed when idle.
swallow_cancel_key = true
# In hold mode, presses shorter than this are treated as accidental and ignored.
min_hold_ms = 250
# Hybrid mode: a tap shorter than this switches to hands-free.
tap_ms = 300
# Another key pressed this soon after the hotkey means you were typing a shortcut: cancel.
shortcut_guard_ms = 400

[transcription]
# auto = Groq if you've set a Groq key, else OpenAI if set, else the offline local model.
# local | groq | openai | custom
engine = "auto"
# "parakeet-v3" (25 European languages incl. English, most accurate here) or "parakeet-v2" (English only)
local_model = "parakeet-v3"
# CPU threads for the local model (0 = automatic).
threads = 0
# Spoken language code for cloud engines ("" = auto-detect). The local model auto-detects.
language = "en"
groq_model = "whisper-large-v3"
openai_model = "gpt-4o-transcribe"
custom_base_url = ""
custom_model = ""
# If the cloud engine fails (offline, quota), retry with the local model.
fallback_to_local = true
# Transcribe in chunks at natural pauses while you talk: faster results + live preview.
live_segments = true

[text]
# Words and names you use often, spelled the way you want them (fixes capitalisation and
# guides cloud models).
vocabulary = ["Claude Code", "Codex", "WhatsApp", "GitHub", "JavaScript", "TypeScript", "Python"]
# The languages and dialects you dictate in. AI formatting keeps each one as you spoke it, and
# with "Nigerian Pidgin" listed it writes your Pidgin as proper Pidgin (wetin, dey, abeg...)
# instead of the English sound-alikes the recogniser hears ("what in", "they", "a bag").
# Example: ["English", "Nigerian Pidgin"]
languages = ["English"]
remove_fillers = true
# Start each dictation with a capital letter.
capitalize_first = true
# Add a space after each dictation so the next one doesn't stick to it.
append_space = true

[text.replacements]
# "what you hear it write" = "what you want"
"cloud code" = "Claude Code"
"clod code" = "Claude Code"

[formatting]
# AI formatting: adds punctuation (commas, full stops, apostrophes, question marks), breaks
# run-on speech into sentences and paragraphs, and drops fillers, repeated words and false
# starts ("if you have you are building a project" -> "if you're building a project"), while
# keeping what you said. Off = the text is pasted exactly as recognised.
enabled = true
# auto        = Claude through your Claude Code login if the `claude` command is installed
#               (no API key needed), otherwise Groq / OpenAI if you've added a key
# claude-code = always Claude Code
# groq | openai | custom = an OpenAI-compatible chat API (needs a key; custom also needs base_url)
provider = "auto"
# "" = the provider's default: Claude Code "opus" (best at keeping your meaning), Groq
# "openai/gpt-oss-120b", OpenAI "gpt-4.1-mini". Claude Code also accepts "sonnet" or "haiku".
model = ""
# Claude only: how much the model thinks first. "low" is the fastest and is plenty for formatting.
effort = "low"
# If formatting fails or takes longer than this (plus 1 s per 40 words), the plain transcript is
# pasted instead.
timeout_s = 10
# Full path to the `claude` command, if Mouthpiece can't find it ("" = search the usual places).
claude_path = ""
# Base URL for provider = "custom" (any OpenAI-compatible API).
base_url = ""

[paste]
# auto = Ctrl+Shift+V in terminals, Shift+Insert in code editors and unknown apps, Ctrl+V elsewhere.
# Or force one: "ctrl+v" | "ctrl+shift+v" | "shift+insert" | "type" (simulated typing)
# | "copy" (only put the text on the clipboard; you paste it yourself)
method = "auto"
# Put your previous clipboard back after pasting.
preserve_clipboard = true
restore_delay_ms = 900
# Wait between setting the clipboard and sending the paste keys.
delay_ms = 70

[paste.apps]
# Per-app override, by the app name shown in `mouthpiece status` (lower-case):
# "ptyxis" = "ctrl+shift+v"

[audio]
# "" = system default microphone (as chosen in GNOME Settings). See: mouthpiece devices
input_device = ""
max_seconds = 600
sounds = true
sound_volume = 0.35

[ui]
overlay = true
# "bottom" or "top"
overlay_position = "bottom"
# Show the words as they are recognised (needs live_segments).
live_preview = true

[history]
enabled = true
"""

DEFAULTS: dict = tomllib.loads(DEFAULT_CONFIG_TEXT)


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return Path(base) / "mouthpiece"


def config_path() -> Path:
    return config_dir() / "config.toml"


def secrets_path() -> Path:
    return config_dir() / "secrets.toml"


def data_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(base) / "mouthpiece"


def runtime_dir() -> Path:
    return Path(os.environ.get("XDG_RUNTIME_DIR") or f"/tmp/mouthpiece-{os.getuid()}")


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in ("replacements", "apps"):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def ensure_config() -> Path:
    p = config_path()
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(DEFAULT_CONFIG_TEXT)
    return p


def load_config() -> dict:
    p = ensure_config()
    with p.open("rb") as f:
        user = tomllib.load(f)
    return _merge(DEFAULTS, user)


def load_secrets() -> dict[str, str]:
    out: dict[str, str] = {}
    p = secrets_path()
    if p.exists():
        with p.open("rb") as f:
            data = tomllib.load(f)
        out.update({k: str(v) for k, v in data.items() if isinstance(v, str) and v})
    for name, env in (("groq", "GROQ_API_KEY"), ("openai", "OPENAI_API_KEY"), ("custom", "MOUTHPIECE_CUSTOM_API_KEY")):
        if os.environ.get(env):
            out.setdefault(name, os.environ[env])
    return out


def save_secret(name: str, value: str) -> Path:
    p = secrets_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, str] = {}
    if p.exists():
        with p.open("rb") as f:
            data = {k: v for k, v in tomllib.load(f).items() if isinstance(v, str)}
    if value:
        data[name] = value
    else:
        data.pop(name, None)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write("# Mouthpiece API keys (keep private)\n")
        for k, v in sorted(data.items()):
            f.write(f"{k} = {json.dumps(v)}\n")
    os.chmod(p, 0o600)
    return p


def toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    return json.dumps(str(value), ensure_ascii=False)


def parse_cli_value(raw: str, current):
    """Interpret a CLI string using the type of the current/default value."""
    if isinstance(current, bool):
        if raw.lower() in ("1", "true", "yes", "on"):
            return True
        if raw.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"expected true/false, got {raw!r}")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        try:
            v = json.loads(raw)
            if isinstance(v, list):
                return v
        except json.JSONDecodeError:
            pass
        return [s.strip() for s in raw.split(",") if s.strip()]
    return raw


def set_value(dotted: str, value) -> None:
    """Set section.key in config.toml, editing the line in place so comments survive."""
    p = ensure_config()
    if "." not in dotted:
        raise KeyError("use section.key, e.g. hotkey.key")
    *sections, key = dotted.split(".")
    section = ".".join(sections)
    lines = p.read_text().splitlines()
    header = f"[{section}]"
    start = next((i for i, l in enumerate(lines) if l.strip() == header), None)
    rendered_key = key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else json.dumps(key)
    new_line = f"{rendered_key} = {toml_value(value)}"
    if start is None:
        lines += ["", header, new_line]
    else:
        end = next((i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("[")), len(lines))
        pat = re.compile(rf"^\s*{re.escape(rendered_key)}\s*=")
        for i in range(start + 1, end):
            if pat.match(lines[i]):
                lines[i] = new_line
                break
        else:
            insert_at = end
            while insert_at > start + 1 and not lines[insert_at - 1].strip():
                insert_at -= 1
            lines.insert(insert_at, new_line)
    text = "\n".join(lines) + "\n"
    tomllib.loads(text)  # refuse to write a broken file
    p.write_text(text)


def get_value(cfg: dict, dotted: str):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    return node
