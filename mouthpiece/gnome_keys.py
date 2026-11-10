"""Swallow the cancel key (Esc) while dictating, on GNOME.

Reading evdev sees Esc, but the focused app sees it too — WhatsApp Web would close the chat
and Claude Code would interrupt its current task. So while a recording is in progress we
register Escape as a GNOME custom shortcut (GNOME then consumes the key press) and remove it
as soon as we're idle again.

Safety: the shortcut's command is `mouthpiece unstick-escape`, which removes the binding if
the daemon isn't running — so even after a crash, the next Esc press heals itself. Leftovers
are also removed at daemon start and exit.
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading

log = logging.getLogger(__name__)

SCHEMA = "org.gnome.settings-daemon.plugins.media-keys"
CUSTOM = "org.gnome.settings-daemon.plugins.media-keys.custom-keybinding"
PATH = "/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/mouthpiece-cancel/"


def is_gnome() -> bool:
    return "gnome" in (os.environ.get("XDG_CURRENT_DESKTOP", "") + os.environ.get("ORIGINAL_XDG_CURRENT_DESKTOP", "")).lower()


def _settings():
    import gi

    from gi.repository import Gio

    src = Gio.SettingsSchemaSource.get_default()
    if src is None or src.lookup(SCHEMA, True) is None:
        raise RuntimeError("GNOME media-keys schema not available")
    return Gio, Gio.Settings.new(SCHEMA)


def set_binding(active: bool, command: str = "") -> None:
    Gio, media = _settings()
    paths = list(media.get_strv("custom-keybindings"))
    if active:
        custom = Gio.Settings.new_with_path(CUSTOM, PATH)
        custom.set_string("name", "Mouthpiece: cancel dictation (temporary)")
        custom.set_string("command", command)
        custom.set_string("binding", "Escape")
        if PATH not in paths:
            media.set_strv("custom-keybindings", paths + [PATH])
    elif PATH in paths:
        media.set_strv("custom-keybindings", [p for p in paths if p != PATH])
    Gio.Settings.sync()


def remove_leftover() -> bool:
    """Remove the binding if present. Returns True if something was removed."""
    try:
        _Gio, media = _settings()
        if PATH in media.get_strv("custom-keybindings"):
            set_binding(False)
            return True
    except Exception as e:
        log.debug("escape binding cleanup skipped: %s", e)
    return False


class EscapeSwallower(threading.Thread):
    """Applies 'Esc bound / unbound' requests in order, off the main loop (gsettings is slowish)."""

    def __init__(self) -> None:
        super().__init__(name="esc-swallow", daemon=True)
        self.q: queue.Queue = queue.Queue()
        self.active = False
        exe = os.path.join(os.path.dirname(sys.executable), "mouthpiece")
        self.command = f"'{exe}' unstick-escape" if os.path.exists(exe) else f"'{sys.executable}' -m mouthpiece unstick-escape"
        if remove_leftover():
            log.info("removed a leftover Escape binding from a previous run")

    def want(self, on: bool) -> None:
        self.q.put(on)

    def run(self) -> None:
        while True:
            on = self.q.get()
            if on is None:
                break
            while not self.q.empty():  # coalesce bursts; keep the latest wish
                nxt = self.q.get()
                if nxt is None:
                    on = None
                    break
                on = nxt
            if on is None:
                break
            if on != self.active:
                try:
                    set_binding(on, self.command)
                    self.active = on
                except Exception as e:
                    log.warning("could not %s the Escape binding: %s", "add" if on else "remove", e)
        if self.active:
            try:
                set_binding(False)
            except Exception:
                pass
            self.active = False

    def stop(self) -> None:
        self.q.put(None)
        self.join(3)
