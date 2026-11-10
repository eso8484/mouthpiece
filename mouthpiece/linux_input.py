"""Pure-Python access to Linux evdev (reading keyboards) and uinput (a virtual keyboard).

Works the same on Wayland and X11 because it sits below the display server. Needs read
access to /dev/input/event* and write access to /dev/uinput (the `input` group on most
distros, or a uaccess udev rule).
"""

from __future__ import annotations

import errno
import fcntl
import os
import struct
import time
from dataclasses import dataclass, field

from .keycodes import CODES

EV_SYN, EV_KEY, EV_REL, EV_MSC = 0x00, 0x01, 0x02, 0x04
SYN_REPORT = 0
BUS_VIRTUAL = 0x06
KEY_MAX = 0x2FF
BTN_MISC = 0x100

_EVENT = struct.Struct("llHHi")  # struct input_event on 64-bit Linux (timeval + type, code, value)
EVENT_SIZE = _EVENT.size

VIRTUAL_KEYBOARD_NAME = "Mouthpiece Virtual Keyboard"


def _ioc(direction: int, kind: str, nr: int, size: int) -> int:
    return (direction << 30) | (size << 16) | (ord(kind) << 8) | nr


_READ, _WRITE = 2, 1


def EVIOCGNAME(length: int) -> int:
    return _ioc(_READ, "E", 0x06, length)


def EVIOCGBIT(ev: int, length: int) -> int:
    return _ioc(_READ, "E", 0x20 + ev, length)


def EVIOCGKEY(length: int) -> int:
    return _ioc(_READ, "E", 0x18, length)


UI_SET_EVBIT = _ioc(_WRITE, "U", 100, 4)
UI_SET_KEYBIT = _ioc(_WRITE, "U", 101, 4)
UI_DEV_SETUP = _ioc(_WRITE, "U", 3, 92)  # sizeof(struct uinput_setup)
UI_DEV_CREATE = _ioc(0, "U", 1, 0)
UI_DEV_DESTROY = _ioc(0, "U", 2, 0)


def UI_GET_SYSNAME(length: int) -> int:
    return _ioc(_READ, "U", 44, length)


_KEY_BYTES = (KEY_MAX + 8) // 8


def _bits(buf: bytes) -> set[int]:
    out = set()
    for i, byte in enumerate(buf):
        if byte:
            for b in range(8):
                if byte >> b & 1:
                    out.add(i * 8 + b)
    return out


@dataclass
class InputDevice:
    """An opened /dev/input/eventN node (non-blocking)."""

    path: str
    fd: int
    name: str
    keys: set[int] = field(default_factory=set)

    @classmethod
    def open(cls, path: str) -> "InputDevice":
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
        try:
            buf = bytearray(256)
            fcntl.ioctl(fd, EVIOCGNAME(len(buf)), buf, True)
            name = bytes(buf).split(b"\0", 1)[0].decode("utf-8", "replace")
            evbits = bytearray(4)
            fcntl.ioctl(fd, EVIOCGBIT(0, len(evbits)), evbits, True)
            keys: set[int] = set()
            if _bits(evbits) & {EV_KEY}:
                kb = bytearray(_KEY_BYTES)
                fcntl.ioctl(fd, EVIOCGBIT(EV_KEY, len(kb)), kb, True)
                keys = _bits(kb)
            return cls(path, fd, name, keys)
        except Exception:
            os.close(fd)
            raise

    @property
    def is_keyboard(self) -> bool:
        # A real keyboard has letters and space; this skips power buttons, lid switches, etc.
        return {CODES["KEY_A"], CODES["KEY_Z"], CODES["KEY_SPACE"]} <= self.keys

    def pressed_now(self) -> set[int]:
        """Keys currently held on this device (kernel state), used to sync at open time."""
        buf = bytearray(_KEY_BYTES)
        try:
            fcntl.ioctl(self.fd, EVIOCGKEY(len(buf)), buf, True)
        except OSError:
            return set()
        return _bits(buf)

    def read_events(self) -> list[tuple[int, int, int]]:
        """Return pending (type, code, value) events. Raises OSError(ENODEV) when unplugged."""
        events = []
        while True:
            try:
                data = os.read(self.fd, EVENT_SIZE * 64)
            except BlockingIOError:
                break
            if not data:
                break
            for off in range(0, len(data) - EVENT_SIZE + 1, EVENT_SIZE):
                _s, _us, etype, code, value = _EVENT.unpack_from(data, off)
                events.append((etype, code, value))
            if len(data) < EVENT_SIZE * 64:
                break
        return events

    def close(self) -> None:
        try:
            os.close(self.fd)
        except OSError:
            pass


def list_event_nodes() -> list[str]:
    try:
        names = os.listdir("/dev/input")
    except OSError:
        return []
    nodes = [n for n in names if n.startswith("event") and n[5:].isdigit()]
    return [f"/dev/input/{n}" for n in sorted(nodes, key=lambda n: int(n[5:]))]


def describe_devices() -> list[dict]:
    """Best-effort listing for `mouthpiece doctor` / `devices`."""
    out = []
    for path in list_event_nodes():
        try:
            dev = InputDevice.open(path)
        except OSError as e:
            out.append({"path": path, "error": errno.errorcode.get(e.errno, str(e))})
            continue
        out.append({"path": path, "name": dev.name, "keyboard": dev.is_keyboard, "keys": len(dev.keys)})
        dev.close()
    return out


# --------------------------------------------------------------------------- uinput


class VirtualKeyboard:
    """A uinput keyboard used to send paste chords or type text into the focused app."""

    STEP_DELAY = 0.012  # seconds between key transitions; some apps miss events sent back-to-back

    def __init__(self, name: str = VIRTUAL_KEYBOARD_NAME, keys: set[int] | None = None):
        self.name = name
        self.fd = os.open("/dev/uinput", os.O_WRONLY | os.O_NONBLOCK | os.O_CLOEXEC)
        try:
            fcntl.ioctl(self.fd, UI_SET_EVBIT, EV_KEY)
            fcntl.ioctl(self.fd, UI_SET_EVBIT, EV_SYN)
            if keys is None:
                keys = {c for c in CODES.values() if c < BTN_MISC}
            for code in sorted(keys):
                fcntl.ioctl(self.fd, UI_SET_KEYBIT, code)
            setup = struct.pack("HHHH80sI", BUS_VIRTUAL, 0x1209, 0x4D50, 1, name.encode()[:79], 0)
            fcntl.ioctl(self.fd, UI_DEV_SETUP, setup)
            fcntl.ioctl(self.fd, UI_DEV_CREATE)
        except Exception:
            os.close(self.fd)
            raise
        self.created_at = time.monotonic()
        self._down: set[int] = set()

    def sysname(self) -> str | None:
        buf = bytearray(64)
        try:
            fcntl.ioctl(self.fd, UI_GET_SYSNAME(len(buf)), buf, True)
        except OSError:
            return None
        return bytes(buf).split(b"\0", 1)[0].decode()

    def wait_ready(self, settle: float = 0.25) -> None:
        """The compositor needs a moment to pick up a freshly created device."""
        left = settle - (time.monotonic() - self.created_at)
        if left > 0:
            time.sleep(left)

    def _write(self, code: int, value: int) -> None:
        payload = _EVENT.pack(0, 0, EV_KEY, code, value) + _EVENT.pack(0, 0, EV_SYN, SYN_REPORT, 0)
        written = os.write(self.fd, payload)
        if written != len(payload):
            raise OSError(errno.EIO, f"short uinput write {written}/{len(payload)}")
        if value:
            self._down.add(code)
        else:
            self._down.discard(code)

    def press(self, code: int) -> None:
        self._write(code, 1)

    def release(self, code: int) -> None:
        self._write(code, 0)

    def chord(self, modifiers: list[int], key: int) -> None:
        """Press modifiers, tap key, release modifiers (reverse order), paced."""
        self.wait_ready()
        try:
            for m in modifiers:
                self.press(m)
                time.sleep(self.STEP_DELAY)
            self.press(key)
            time.sleep(self.STEP_DELAY)
            self.release(key)
            time.sleep(self.STEP_DELAY)
        finally:
            for m in reversed(modifiers):
                if m in self._down:
                    self.release(m)
                    time.sleep(self.STEP_DELAY / 2)

    def release_all(self) -> None:
        for code in list(self._down):
            try:
                self.release(code)
            except OSError:
                pass

    def close(self) -> None:
        if self.fd < 0:
            return
        self.release_all()
        try:
            fcntl.ioctl(self.fd, UI_DEV_DESTROY)
        except OSError:
            pass
        os.close(self.fd)
        self.fd = -1

    def __enter__(self) -> "VirtualKeyboard":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
