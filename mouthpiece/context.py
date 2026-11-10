"""Which app has focus? (to choose the right paste keys and clean-up style)

Wayland hides the focused window from ordinary clients, so we ask the
accessibility bus (AT-SPI) which window is ACTIVE. GTK/Qt apps (Ptyxis, GNOME Terminal,
Nautilus...) always answer; Chromium/Electron answer when accessibility is on. The probe runs
in a short-lived subprocess with a timeout so a frozen app can never hang the daemon.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
from dataclasses import dataclass

log = logging.getLogger(__name__)

TERMINALS = {
    "ptyxis", "gnome-terminal-server", "gnome-terminal", "kgx", "console", "tilix", "terminator",
    "konsole", "kitty", "alacritty", "wezterm", "wezterm-gui", "foot", "footclient", "xterm", "uxterm",
    "urxvt", "rxvt", "st", "guake", "tilda", "blackbox", "terminology", "xfce4-terminal",
    "mate-terminal", "lxterminal", "qterminal", "cool-retro-term", "contour", "ghostty", "warp",
    "warp-terminal", "rio", "yakuake", "deepin-terminal", "terminal", "sakura", "kermit", "hyper",
    "tabby", "termius", "black box", "gnome console",
}
CODE_EDITORS = {
    "code", "code-oss", "vscodium", "codium", "cursor", "windsurf", "zed", "zed-editor", "kiro",
    "jetbrains-idea", "jetbrains-pycharm", "jetbrains-webstorm", "idea", "pycharm", "webstorm",
    "android-studio", "sublime_text", "sublime text", "void", "trae",
}
BROWSERS = {
    "google chrome", "google-chrome", "chromium", "chromium-browser", "firefox", "firefox-esr",
    "brave", "brave-browser", "microsoft edge", "microsoft-edge", "vivaldi", "opera", "librewolf",
    "zen", "zen browser", "epiphany", "web", "floorp", "thorium",
}
CHAT = {
    "whatsapp", "whatsapp for linux", "zapzap", "wasistlos", "telegram", "telegram desktop",
    "telegramdesktop", "signal", "signal-desktop", "discord", "vesktop", "slack", "element",
    "teams-for-linux", "microsoft teams", "fractal", "dino", "beeper",
}
CHAT_TITLES = ("whatsapp", "messenger", "facebook", "telegram", "slack", "discord", "instagram",
               "x.com", "twitter", "linkedin", "teams", "signal", "gmail", "outlook")
AI_TITLES = ("claude", "chatgpt", "gemini", "copilot", "perplexity", "deepseek", "grok")

_PROBE = r"""
import json, sys
try:
    import gi
    gi.require_version("Atspi", "2.0")
    from gi.repository import Atspi
except Exception as e:
    print(json.dumps({"error": "no-atspi: %s" % e})); sys.exit(0)
desktop = Atspi.get_desktop(0)
found = None
for i in range(desktop.get_child_count()):
    try:
        app = desktop.get_child_at_index(i)
    except Exception:
        continue
    if app is None:
        continue
    try:
        n = app.get_child_count()
    except Exception:
        continue
    for j in range(min(n, 64)):
        try:
            win = app.get_child_at_index(j)
            if win is not None and win.get_state_set().contains(Atspi.StateType.ACTIVE):
                try:
                    pid = app.get_process_id()
                except Exception:
                    pid = 0
                found = {"app": app.get_name() or "", "title": (win.get_name() or "")[:200], "pid": pid}
                break
        except Exception:
            continue
    if found:
        break
print(json.dumps(found or {}))
"""


@dataclass
class AppInfo:
    app: str = ""
    title: str = ""
    pid: int = 0
    kind: str = "unknown"  # terminal | code | browser | chat | ai | other | unknown

    def describe(self) -> str:
        if not self.app:
            return "unknown app"
        return f"{self.app} ({self.kind})"


def _proc_name(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/comm") as f:
            return f.read().strip().lower()
    except OSError:
        return ""


def classify(app: str, title: str = "", pid: int = 0) -> str:
    a = app.strip().lower()
    t = title.lower()
    comm = _proc_name(pid) if pid else ""
    if not a and not comm:
        return "unknown"
    if a in TERMINALS or comm in TERMINALS:
        return "terminal"
    if a in CODE_EDITORS or comm in CODE_EDITORS:
        return "code"
    if a in CHAT or comm in CHAT:
        return "chat"
    if a in BROWSERS or comm in BROWSERS or "chrome" in a or "firefox" in a:
        if any(k in t for k in CHAT_TITLES):
            return "chat"
        if any(k in t for k in AI_TITLES):
            return "ai"
        return "browser"
    if a in ("claude", "chatgpt") or comm == "claude":
        return "ai"
    return "other"


def probe_active_app(timeout: float = 1.5) -> AppInfo:
    try:
        p = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=timeout)
        data = json.loads(p.stdout.strip() or "{}")
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as e:
        log.debug("AT-SPI probe failed: %s", e)
        return AppInfo()
    if not data or "error" in data:
        if data.get("error"):
            log.debug("AT-SPI probe: %s", data["error"])
        return AppInfo()
    info = AppInfo(app=data.get("app", ""), title=data.get("title", ""), pid=int(data.get("pid") or 0))
    info.kind = classify(info.app, info.title, info.pid)
    return info


class AppProbe:
    """Start probing in the background at hotkey press; collect the answer at paste time."""

    def __init__(self) -> None:
        self.result: AppInfo | None = None
        self._thread = threading.Thread(target=self._run, name="app-probe", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        self.result = probe_active_app()

    def get(self, wait: float = 1.6) -> AppInfo:
        self._thread.join(wait)
        return self.result or AppInfo()
