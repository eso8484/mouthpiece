"""Deliver text into the focused app: clipboard + a paste keystroke from a virtual keyboard.

Choosing the keystroke matters: Ctrl+V everywhere would fail in terminals, where it means
"literal next char", and in Claude Code, where it means "paste image". So:

    terminal (Ptyxis, GNOME Terminal, Kitty, ...)  -> Ctrl+Shift+V
    code editors (VS Code: editor *or* terminal)   -> Shift+Insert
    unknown app (no accessibility info)            -> Shift+Insert (works in terminals and GUIs,
                                                      because we set CLIPBOARD *and* PRIMARY)
    everything else (browsers, chat, docs)         -> Ctrl+V
"""

from __future__ import annotations

import logging
import subprocess
import time

from .clipboard import Clipboard
from .context import AppInfo
from .keycodes import CODES as C
from .linux_input import VirtualKeyboard

log = logging.getLogger(__name__)

CHORDS: dict[str, tuple[list[int], int]] = {
    "ctrl+v": ([C["KEY_LEFTCTRL"]], C["KEY_V"]),
    "ctrl+shift+v": ([C["KEY_LEFTCTRL"], C["KEY_LEFTSHIFT"]], C["KEY_V"]),
    "shift+insert": ([C["KEY_LEFTSHIFT"]], C["KEY_INSERT"]),
}
METHODS = ("auto", *CHORDS, "type", "copy")


def choose_method(method: str, app: AppInfo, overrides: dict[str, str]) -> str:
    if method != "auto":
        return method
    name = app.app.strip().lower()
    if name and name in overrides:
        return overrides[name]
    if app.kind == "terminal":
        return "ctrl+shift+v"
    if app.kind in ("code", "unknown"):
        return "shift+insert"
    return "ctrl+v"


# --------------------------------------------------------------------------- typing


def _keyboard_layout() -> str:
    try:
        out = subprocess.run(["gsettings", "get", "org.gnome.desktop.input-sources", "sources"],
                             capture_output=True, text=True, timeout=2).stdout
        if "'gb'" in out.split(")")[0]:
            return "gb"
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "us"


def _char_map(layout: str) -> dict[str, tuple[int, bool]]:
    m: dict[str, tuple[int, bool]] = {}
    for ch in "abcdefghijklmnopqrstuvwxyz":
        m[ch] = (C[f"KEY_{ch.upper()}"], False)
        m[ch.upper()] = (C[f"KEY_{ch.upper()}"], True)
    for d, sym in zip("1234567890", "!@#$%^&*()"):
        m[d] = (C[f"KEY_{d}"], False)
        m[sym] = (C[f"KEY_{d}"], True)
    plain = {" ": "SPACE", "-": "MINUS", "=": "EQUAL", "[": "LEFTBRACE", "]": "RIGHTBRACE",
             ";": "SEMICOLON", "'": "APOSTROPHE", "`": "GRAVE", "\\": "BACKSLASH", ",": "COMMA",
             ".": "DOT", "/": "SLASH", "\t": "TAB"}
    shifted = {"_": "MINUS", "+": "EQUAL", "{": "LEFTBRACE", "}": "RIGHTBRACE", ":": "SEMICOLON",
               '"': "APOSTROPHE", "~": "GRAVE", "|": "BACKSLASH", "<": "COMMA", ">": "DOT", "?": "SLASH"}
    for ch, k in plain.items():
        m[ch] = (C[f"KEY_{k}"], False)
    for ch, k in shifted.items():
        m[ch] = (C[f"KEY_{k}"], True)
    if layout == "gb":
        m['"'] = (C["KEY_2"], True)
        m["@"] = (C["KEY_APOSTROPHE"], True)
        m["#"] = (C["KEY_BACKSLASH"], False)
        m["~"] = (C["KEY_BACKSLASH"], True)
        m["\\"] = (C["KEY_102ND"], False)
        m["|"] = (C["KEY_102ND"], True)
        m["£"] = (C["KEY_3"], True)
        m["¬"] = (C["KEY_GRAVE"], True)
    return m


class Injector:
    def __init__(self, cfg: dict, wait_released=None):
        self.cfg = cfg
        self.wait_released = wait_released  # callable(timeout) -> bool, from the hotkey listener
        self.clipboard = Clipboard()
        self.keyboard: VirtualKeyboard | None = None
        self.error: str | None = None
        self._charmap: dict[str, tuple[int, bool]] | None = None
        try:
            self.keyboard = VirtualKeyboard()
        except OSError as e:
            self.error = f"cannot create virtual keyboard (/dev/uinput): {e.strerror or e}"
            log.error(self.error)

    def close(self) -> None:
        if self.keyboard:
            self.keyboard.close()

    def _settle_modifiers(self) -> None:
        if self.wait_released and not self.wait_released(1.5):
            log.warning("modifier keys still held; pasting anyway")

    def deliver(self, text: str, app: AppInfo) -> dict:
        p = self.cfg["paste"]
        method = choose_method(p.get("method", "auto"), app, {k.lower(): v for k, v in (p.get("apps") or {}).items()})
        if method == "type" and not self._typable(text):
            log.info("text has characters the virtual keyboard can't type; pasting instead")
            method = "shift+insert" if app.kind in ("unknown", "code") else (
                "ctrl+shift+v" if app.kind == "terminal" else "ctrl+v")
        result = {"method": method, "ok": False, "app": app.describe()}
        if method == "copy":  # copy-only mode: leave it on the clipboard, press nothing
            result["ok"] = self.clipboard.set_text(text)
            return result
        if self.keyboard is None:
            # Can't press keys: leave the text on the clipboard so the user can paste manually.
            result["ok"] = self.clipboard.set_text(text)
            result["error"] = self.error or "no virtual keyboard"
            return result
        try:
            if method == "type":
                self._settle_modifiers()
                self._type(text)
                result["ok"] = True
                return result
            snap = self.clipboard.snapshot() if p.get("preserve_clipboard", True) else None
            if not self.clipboard.set_text(text, primary=True):
                result["error"] = f"could not set clipboard ({self.clipboard.backend})"
                return result
            self._settle_modifiers()
            time.sleep(max(0, int(p.get("delay_ms", 70))) / 1000)
            mods, key = CHORDS[method]
            self.keyboard.chord(mods, key)
            result["ok"] = True
            if snap is not None:
                self.clipboard.restore_later(snap, text, max(200, int(p.get("restore_delay_ms", 900))) / 1000)
        except OSError as e:
            result["error"] = f"keystroke injection failed: {e}"
            log.error(result["error"])
        return result

    # ------------------------------------------------------------------ typing
    def _typable(self, text: str) -> bool:
        if self._charmap is None:
            self._charmap = _char_map(_keyboard_layout())
        return all(ch in self._charmap or ch == "\n" for ch in text)

    def _type(self, text: str) -> None:
        assert self.keyboard is not None
        kb = self.keyboard
        kb.wait_ready()
        shift = C["KEY_LEFTSHIFT"]
        cm = self._charmap or _char_map(_keyboard_layout())
        for ch in text:
            if ch == "\n":
                ch = " "  # Enter would send the message / run the command
            code, needs_shift = cm[ch]
            if needs_shift:
                kb.press(shift)
            kb.press(code)
            time.sleep(0.004)
            kb.release(code)
            if needs_shift:
                kb.release(shift)
            time.sleep(0.006)
