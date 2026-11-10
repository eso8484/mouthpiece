"""Clipboard access that never steals keyboard focus.

GNOME's compositor (Mutter) offers no data-control protocol, so `wl-copy` would have to pop up
a tiny window to grab focus. Instead we own the X11 CLIPBOARD/PRIMARY selections through
XWayland with xclip; Mutter mirrors them to Wayland apps instantly. Compositors with ext/wlr
data-control (KDE, Sway, Hyprland) use wl-clipboard directly.
"""

from __future__ import annotations

import logging
import os
import select
import shutil
import socket
import struct
import subprocess
import threading
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)

TEXT_TARGETS = ("UTF8_STRING", "text/plain;charset=utf-8", "text/plain", "STRING", "TEXT")


def wayland_globals(timeout: float = 1.0) -> set[str]:
    """List compositor globals with a minimal Wayland wire-protocol client (no dependencies)."""
    disp = os.environ.get("WAYLAND_DISPLAY")
    rt = os.environ.get("XDG_RUNTIME_DIR")
    if not disp or not rt:
        return set()
    path = disp if disp.startswith("/") else os.path.join(rt, disp)
    names: set[str] = set()
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(path)
        msg = lambda obj, op, payload=b"": struct.pack("<IHH", obj, op, 8 + len(payload)) + payload  # noqa: E731
        s.sendall(msg(1, 1, struct.pack("<I", 2)) + msg(1, 0, struct.pack("<I", 3)))
        buf, done = b"", False
        deadline = time.monotonic() + timeout
        while not done and time.monotonic() < deadline:
            r, _, _ = select.select([s], [], [], max(0.0, deadline - time.monotonic()))
            if not r:
                break
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
            while len(buf) >= 8:
                obj, op, size = struct.unpack("<IHH", buf[:8])
                if len(buf) < size:
                    break
                body, buf = buf[8:size], buf[size:]
                if obj == 2 and op == 0:
                    slen = struct.unpack("<I", body[4:8])[0]
                    names.add(body[8:8 + slen - 1].decode())
                elif obj == 3:
                    done = True
        s.close()
    except OSError:
        pass
    return names


@dataclass
class Snapshot:
    mime: str | None
    data: bytes | None


class Clipboard:
    def __init__(self, backend: str = "auto"):
        self.backend = self._detect() if backend == "auto" else backend
        self._lock = threading.Lock()
        self._restore_timer: threading.Timer | None = None
        self.last_set: str | None = None
        log.info("clipboard backend: %s", self.backend)

    @staticmethod
    def _detect() -> str:
        wayland = bool(os.environ.get("WAYLAND_DISPLAY"))
        have_xclip = bool(shutil.which("xclip")) and bool(os.environ.get("DISPLAY"))
        if wayland:
            g = wayland_globals()
            if ({"ext_data_control_manager_v1", "zwlr_data_control_manager_v1"} & g) and shutil.which("wl-copy"):
                return "wl"
            if have_xclip:
                return "xclip"
            if shutil.which("wl-copy"):
                return "wl"  # last resort: may flash a tiny focus-grabbing surface on GNOME
            return "none"
        return "xclip" if have_xclip else "none"

    # ------------------------------------------------------------------ primitives
    def _run(self, args: list[str], data: bytes | None = None, timeout: float = 1.5) -> bytes | None:
        try:
            p = subprocess.run(args, input=data, capture_output=True, timeout=timeout,
                               stdin=None if data is not None else subprocess.DEVNULL)
        except (subprocess.TimeoutExpired, OSError) as e:
            log.debug("%s failed: %s", args[0], e)
            return None
        return p.stdout if p.returncode == 0 else None

    def _set(self, data: bytes, mime: str | None, primary: bool) -> bool:
        if self.backend == "xclip":
            args = ["xclip", "-selection", "primary" if primary else "clipboard", "-i"]
            if mime and mime not in TEXT_TARGETS:
                args += ["-t", mime]
            # xclip forks a child that keeps serving the selection; give it no stdout/stderr
            # pipes, or our call would wait for that child to exit.
            try:
                p = subprocess.run(args, input=data, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
                return p.returncode == 0
            except (subprocess.TimeoutExpired, OSError):
                return False
        if self.backend == "wl":
            args = ["wl-copy"] + (["--primary"] if primary else [])
            if mime and mime not in TEXT_TARGETS:
                args += ["--type", mime]
            try:
                p = subprocess.run(args, input=data, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
                return p.returncode == 0
            except (subprocess.TimeoutExpired, OSError):
                return False
        return False

    def targets(self) -> list[str]:
        if self.backend == "xclip":
            out = self._run(["xclip", "-selection", "clipboard", "-t", "TARGETS", "-o"])
        elif self.backend == "wl":
            out = self._run(["wl-paste", "--list-types"])
        else:
            out = None
        return [t.strip() for t in (out or b"").decode("utf-8", "replace").splitlines() if t.strip()]

    def read(self, mime: str | None = None) -> bytes | None:
        if self.backend == "xclip":
            args = ["xclip", "-selection", "clipboard", "-o"] + (["-t", mime] if mime else [])
        elif self.backend == "wl":
            args = ["wl-paste", "--no-newline"] + (["--type", mime] if mime else [])
        else:
            return None
        return self._run(args)

    def get_text(self) -> str | None:
        data = self.read()
        return None if data is None else data.decode("utf-8", "replace")

    # ------------------------------------------------------------------ high level
    def set_text(self, text: str, primary: bool = True) -> bool:
        data = text.encode("utf-8")
        with self._lock:
            self.cancel_restore()
            ok = self._set(data, None, primary=False)
            if primary:
                self._set(data, None, primary=True)
            if ok:
                self.last_set = text
            return ok

    def snapshot(self) -> Snapshot | None:
        targets = self.targets()
        if not targets:
            return None
        if any(t in targets for t in TEXT_TARGETS):
            data = self.read()
            return Snapshot(None, data) if data is not None else None
        for preferred in ("image/png", "image/jpeg", "text/uri-list", "text/html"):
            if preferred in targets:
                data = self.read(preferred)
                return Snapshot(preferred, data) if data else None
        return None

    def restore_later(self, snap: Snapshot | None, ours: str, delay: float) -> None:
        if snap is None or snap.data is None:
            return

        def restore() -> None:
            with self._lock:
                current = self.get_text()
                if current is not None and current != ours:
                    log.debug("clipboard changed since paste; not restoring")
                    return
                self._set(snap.data, snap.mime, primary=False)
                log.debug("clipboard restored (%s, %d bytes)", snap.mime or "text", len(snap.data))

        self.cancel_restore()
        self._restore_timer = threading.Timer(delay, restore)
        self._restore_timer.daemon = True
        self._restore_timer.start()

    def cancel_restore(self) -> None:
        if self._restore_timer is not None:
            self._restore_timer.cancel()
            self._restore_timer = None
