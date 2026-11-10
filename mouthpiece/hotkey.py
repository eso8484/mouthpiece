"""Global hotkeys from raw evdev events.

A hotkey is a '+'-separated list of tokens, each matching one or more key codes:

    RIGHTCTRL            hold the right Ctrl key alone (default)
    CTRL+SPACE           either Ctrl plus Space
    SUPER+ALT            two modifiers together
    F9, SHIFT+F7, PAUSE  function keys / combos
    MOUSE_SIDE           a mouse thumb button (BTN_SIDE)

Reading /dev/input sees every key on every keyboard, including key releases, which is what
makes true push-to-talk possible on GNOME Wayland (where apps can't grab global keys).
"""

from __future__ import annotations

import errno
import logging
import select
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .keycodes import CODES
from .linux_input import (
    BTN_MISC,
    EV_KEY,
    VIRTUAL_KEYBOARD_NAME,
    InputDevice,
    list_event_nodes,
)

log = logging.getLogger(__name__)

C = CODES
MODIFIER_GROUPS: dict[str, frozenset[int]] = {
    "CTRL": frozenset({C["KEY_LEFTCTRL"], C["KEY_RIGHTCTRL"]}),
    "SHIFT": frozenset({C["KEY_LEFTSHIFT"], C["KEY_RIGHTSHIFT"]}),
    "ALT": frozenset({C["KEY_LEFTALT"], C["KEY_RIGHTALT"]}),
    "SUPER": frozenset({C["KEY_LEFTMETA"], C["KEY_RIGHTMETA"]}),
}
ALIASES: dict[str, str] = {
    "CONTROL": "CTRL", "META": "SUPER", "WIN": "SUPER", "WINDOWS": "SUPER", "CMD": "SUPER",
    "LOGO": "SUPER", "OPTION": "ALT",
    "RCTRL": "RIGHTCTRL", "LCTRL": "LEFTCTRL", "RSHIFT": "RIGHTSHIFT", "LSHIFT": "LEFTSHIFT",
    "RALT": "RIGHTALT", "LALT": "LEFTALT", "ALTGR": "RIGHTALT", "RSUPER": "RIGHTMETA",
    "LSUPER": "LEFTMETA", "RIGHTSUPER": "RIGHTMETA", "LEFTSUPER": "LEFTMETA",
    "RIGHTCONTROL": "RIGHTCTRL", "LEFTCONTROL": "LEFTCTRL",
    "ESCAPE": "ESC", "RETURN": "ENTER", "INS": "INSERT", "DEL": "DELETE", "PGUP": "PAGEUP",
    "PGDN": "PAGEDOWN", "PRINTSCREEN": "SYSRQ", "PRTSC": "SYSRQ", "PRINT": "SYSRQ",
    "MENU": "COMPOSE", "CAPS": "CAPSLOCK", "SCROLL": "SCROLLLOCK",
    "MOUSE_SIDE": "BTN_SIDE", "MOUSE_EXTRA": "BTN_EXTRA", "MOUSE_BACK": "BTN_SIDE",
    "MOUSE_FORWARD": "BTN_EXTRA", "MOUSE_MIDDLE": "BTN_MIDDLE", "MOUSE4": "BTN_SIDE", "MOUSE5": "BTN_EXTRA",
}
MODIFIER_CODES = frozenset().union(*MODIFIER_GROUPS.values())

_PRETTY = {
    "LEFTCTRL": "Left Ctrl", "RIGHTCTRL": "Right Ctrl", "LEFTSHIFT": "Left Shift",
    "RIGHTSHIFT": "Right Shift", "LEFTALT": "Left Alt", "RIGHTALT": "Right Alt",
    "LEFTMETA": "Left Super", "RIGHTMETA": "Right Super", "ESC": "Esc", "SYSRQ": "PrtSc",
    "BTN_SIDE": "Mouse Back", "BTN_EXTRA": "Mouse Forward", "BTN_MIDDLE": "Middle Click",
}
_CODE_NAMES: dict[int, str] = {}
for _name, _code in CODES.items():
    _CODE_NAMES.setdefault(_code, _name.removeprefix("KEY_") if _name.startswith("KEY_") else _name)


def code_name(code: int) -> str:
    return _CODE_NAMES.get(code, f"CODE_{code}")


class HotkeyError(ValueError):
    pass


@dataclass(frozen=True)
class Hotkey:
    spec: str
    groups: tuple[frozenset[int], ...]
    labels: tuple[str, ...]

    @classmethod
    def parse(cls, spec: str) -> "Hotkey":
        tokens = [t.strip().upper().replace(" ", "") for t in spec.replace("-", "+").split("+")]
        tokens = [t for t in tokens if t]
        if not tokens:
            raise HotkeyError("empty hotkey")
        groups, labels = [], []
        for tok in tokens:
            tok = ALIASES.get(tok, tok)
            if tok in MODIFIER_GROUPS:
                groups.append(MODIFIER_GROUPS[tok])
                labels.append(tok.title())
                continue
            code = CODES.get(tok) or CODES.get("KEY_" + tok) or CODES.get(tok.removeprefix("KEY_"))
            if code is None and tok.startswith("CODE_") and tok[5:].isdigit():
                code = int(tok[5:])
            if code is None:
                raise HotkeyError(f"unknown key '{tok}' in hotkey '{spec}'")
            groups.append(frozenset({code}))
            name = code_name(code)
            labels.append(_PRETTY.get(name, name.title() if len(name) > 1 else name))
        if len(set(groups)) != len(groups):
            raise HotkeyError(f"hotkey '{spec}' repeats a key")
        return cls(spec="+".join(tokens), groups=tuple(groups), labels=tuple(labels))

    @classmethod
    def from_codes(cls, codes: list[int]) -> "Hotkey":
        """Build a hotkey from captured physical keys (keeps left/right distinction)."""
        names = []
        for code in sorted(codes, key=lambda c: (c not in MODIFIER_CODES, c)):
            name = code_name(code)
            names.append(name)
        return cls.parse("+".join(names))

    @property
    def codes(self) -> frozenset[int]:
        return frozenset().union(*self.groups)

    @property
    def modifier_only(self) -> bool:
        return self.codes <= MODIFIER_CODES

    @property
    def uses_mouse(self) -> bool:
        return any(BTN_MISC <= c < 0x160 for c in self.codes)

    def satisfied(self, held: set[int]) -> bool:
        return all(g & held for g in self.groups)

    def pretty(self) -> str:
        return " + ".join(self.labels)


class Action(Enum):
    PRESS = "press"            # hotkey became active
    RELEASE = "release"        # hotkey no longer active
    INTERRUPT = "interrupt"    # another key went down while the hotkey was held
    CANCEL = "cancel"          # the cancel key (Esc) went down
    KEY = "key"                # any key down (only reported when `report_keys` is set)


def _tracked(code: int, hotkey: Hotkey) -> bool:
    # keyboard keys plus the hotkey's own buttons; ignore clicks, touchpad finger "buttons", etc.
    return code < BTN_MISC or code >= 0x160 or code in hotkey.codes


class HotkeyStateMachine:
    """Pure logic: feed key transitions, get actions. No I/O, so it's unit-testable."""

    def __init__(self, hotkey: Hotkey, cancel_codes: frozenset[int] = frozenset()):
        self.hotkey = hotkey
        self.cancel_codes = cancel_codes
        self.counts: dict[int, int] = {}  # same key held on two keyboards counts twice
        self.active = False

    @property
    def held(self) -> set[int]:
        return set(self.counts)

    def feed(self, code: int, value: int) -> list[tuple[Action, int]]:
        if value == 2 or not _tracked(code, self.hotkey):  # autorepeat / irrelevant buttons
            return []
        out: list[tuple[Action, int]] = []
        if value == 1:
            first = code not in self.counts
            self.counts[code] = self.counts.get(code, 0) + 1
            if not first:
                return out
            if code in self.cancel_codes:
                out.append((Action.CANCEL, code))
            if not self.active:
                held = self.held
                # Extra keys held (e.g. Shift+RightCtrl) mean the user is typing a shortcut.
                if self.hotkey.satisfied(held) and held <= self.hotkey.codes:
                    self.active = True
                    out.append((Action.PRESS, code))
            elif code not in self.hotkey.codes:
                out.append((Action.INTERRUPT, code))
        else:
            n = self.counts.get(code, 0)
            if n <= 1:
                self.counts.pop(code, None)
            else:
                self.counts[code] = n - 1
            if self.active and not self.hotkey.satisfied(self.held):
                self.active = False
                out.append((Action.RELEASE, code))
        return out

    def drop_codes(self, codes: set[int]) -> list[tuple[Action, int]]:
        """A device vanished: forget its keys (and release the hotkey if it depended on them)."""
        out: list[tuple[Action, int]] = []
        for code in codes:
            if code in self.counts:
                out += self.feed(code, 0)
        return out


Callback = Callable[[Action, int], None]


class HotkeyListener(threading.Thread):
    """Watches every keyboard (and mice, for mouse-button hotkeys), hot-plug aware."""

    RESCAN_SECONDS = 2.0

    def __init__(self, hotkey: Hotkey, callback: Callback, cancel_codes: frozenset[int] = frozenset()):
        super().__init__(name="hotkey-listener", daemon=True)
        self._lock = threading.Lock()
        self.machine = HotkeyStateMachine(hotkey, cancel_codes)
        self.callback = callback
        self.devices: dict[str, InputDevice] = {}
        self.device_keys: dict[str, set[int]] = {}
        self._skip: dict[str, float] = {}
        self._stop = threading.Event()
        self.open_errors: dict[str, str] = {}
        self.last_scan = 0.0

    # ------------------------------------------------------------------ public
    def set_hotkey(self, hotkey: Hotkey, cancel_codes: frozenset[int]) -> None:
        with self._lock:
            held = self.machine.held
            self.machine = HotkeyStateMachine(hotkey, cancel_codes)
            for code in held:
                self.machine.counts[code] = 1
            self.machine.active = False
            self.last_scan = 0.0  # mouse devices may now be needed

    def held_keys(self) -> set[int]:
        with self._lock:
            return self.machine.held

    def wait_keys_released(self, codes: frozenset[int] = MODIFIER_CODES, timeout: float = 1.5) -> bool:
        """Block until none of `codes` is physically held (so a paste chord isn't merged with them)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not (self.held_keys() & codes):
                return True
            time.sleep(0.015)
        return False

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------ internals
    def _wanted(self, dev: InputDevice) -> bool:
        if dev.name == VIRTUAL_KEYBOARD_NAME:
            return False  # never react to our own paste keystrokes
        hk = self.machine.hotkey
        return dev.is_keyboard or bool(dev.keys & hk.codes)

    def _rescan(self) -> None:
        self.last_scan = time.monotonic()
        present = set(list_event_nodes())
        for path in list(self.devices):
            if path not in present:
                self._drop(path)
        for path in present:
            if path in self.devices or self._skip.get(path, 0) > time.monotonic():
                continue
            try:
                dev = InputDevice.open(path)
            except OSError as e:
                self.open_errors[path] = errno.errorcode.get(e.errno, str(e))
                self._skip[path] = time.monotonic() + 30
                continue
            self.open_errors.pop(path, None)
            if not self._wanted(dev):
                dev.close()
                self._skip[path] = time.monotonic() + 30
                continue
            self.devices[path] = dev
            self.device_keys[path] = set()
            log.info("listening on %s (%s)", path, dev.name)
            # keys already held when the device appeared are not "presses"
        if not self.devices and self.open_errors:
            log.warning("no readable keyboards: %s", self.open_errors)

    def _drop(self, path: str) -> None:
        dev = self.devices.pop(path, None)
        keys = self.device_keys.pop(path, set())
        if dev:
            dev.close()
            log.info("keyboard gone: %s", path)
        with self._lock:
            actions = self.machine.drop_codes(keys)
        for act, code in actions:
            self._emit(act, code)

    def _emit(self, action: Action, code: int) -> None:
        try:
            self.callback(action, code)
        except Exception:  # never let a callback kill the listener
            log.exception("hotkey callback failed")

    def run(self) -> None:
        while not self._stop.is_set():
            if time.monotonic() - self.last_scan > self.RESCAN_SECONDS:
                self._rescan()
            fds = {dev.fd: path for path, dev in self.devices.items()}
            if not fds:
                self._stop.wait(0.5)
                continue
            try:
                ready, _, _ = select.select(list(fds), [], [], 0.5)
            except (OSError, ValueError):
                self.last_scan = 0.0
                continue
            for fd in ready:
                path = fds[fd]
                dev = self.devices.get(path)
                if dev is None:
                    continue
                try:
                    events = dev.read_events()
                except OSError:
                    self._drop(path)
                    continue
                for etype, code, value in events:
                    if etype != EV_KEY:
                        continue
                    keys = self.device_keys[path]
                    if value == 1:
                        if code in keys:
                            continue
                        keys.add(code)
                    elif value == 0:
                        if code not in keys:
                            continue  # release of a key pressed before we started listening
                        keys.discard(code)
                    with self._lock:
                        actions = self.machine.feed(code, value)
                    for act, c in actions:
                        self._emit(act, c)
        for path in list(self.devices):
            self.devices.pop(path).close()


def capture_hotkey(timeout: float = 15.0) -> list[int]:
    """Wait for the user to press and release a key combination; return the codes pressed."""
    devices = []
    for path in list_event_nodes():
        try:
            dev = InputDevice.open(path)
        except OSError:
            continue
        if dev.name == VIRTUAL_KEYBOARD_NAME or not (dev.is_keyboard or dev.keys & {C["BTN_SIDE"], C["BTN_EXTRA"]}):
            dev.close()
            continue
        devices.append(dev)
    if not devices:
        raise PermissionError("can't read any keyboard in /dev/input (are you in the 'input' group?)")
    held: set[int] = set()
    combo: set[int] = set()
    deadline = time.monotonic() + timeout
    try:
        # drain anything pending (e.g. the Enter that launched this command)
        for d in devices:
            d.read_events()
        time.sleep(0.3)
        for d in devices:
            d.read_events()
        while time.monotonic() < deadline:
            ready, _, _ = select.select([d.fd for d in devices], [], [], 0.25)
            for d in devices:
                if d.fd not in ready:
                    continue
                for etype, code, value in d.read_events():
                    if etype != EV_KEY or value == 2:
                        continue
                    if code in (C["BTN_LEFT"], C["BTN_RIGHT"], C["BTN_TOUCH"]) or 0x140 <= code < 0x160:
                        continue
                    if value == 1:
                        held.add(code)
                        combo.add(code)
                    else:
                        held.discard(code)
                        if not held and combo:
                            return sorted(combo)
        raise TimeoutError("no key pressed")
    finally:
        for d in devices:
            d.close()
