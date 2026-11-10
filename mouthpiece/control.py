"""Unix-socket control channel: `mouthpiece toggle|start|stop|cancel|status|...` talk to the daemon.

This also lets you drive Mouthpiece from a GNOME custom shortcut (Settings → Keyboard →
Custom Shortcuts → command `mouthpiece toggle`) if you'd rather not use the evdev hotkey.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
from typing import Callable

from .config import runtime_dir

log = logging.getLogger(__name__)


def socket_path() -> str:
    return str(runtime_dir() / "mouthpiece.sock")


class ControlServer(threading.Thread):
    def __init__(self, handler: Callable[[str, list[str]], dict]):
        super().__init__(name="control", daemon=True)
        self.handler = handler
        self.path = socket_path()
        if os.path.exists(self.path):
            if ping():
                raise RuntimeError("Mouthpiece is already running (mouthpiece quit to stop it)")
            os.unlink(self.path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(self.path)
        os.chmod(self.path, 0o600)
        self.sock.listen(8)

    def run(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        with conn:
            conn.settimeout(5)
            try:
                data = b""
                while not data.endswith(b"\n"):
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    data += chunk
                req = json.loads(data.decode() or "{}")
                reply = self.handler(str(req.get("cmd", "")), list(req.get("args", [])))
            except Exception as e:  # report, don't crash the server
                log.exception("control request failed")
                reply = {"ok": False, "error": str(e)}
            try:
                conn.sendall((json.dumps(reply) + "\n").encode())
            except OSError:
                pass

    def close(self) -> None:
        try:
            self.sock.close()
        finally:
            try:
                os.unlink(self.path)
            except OSError:
                pass


def send(cmd: str, *args: str, timeout: float = 10.0) -> dict:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(socket_path())
        s.sendall((json.dumps({"cmd": cmd, "args": list(args)}) + "\n").encode())
        data = b""
        while not data.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
        return json.loads(data.decode() or "{}")
    finally:
        s.close()


def ping() -> bool:
    try:
        return bool(send("ping", timeout=1.0).get("ok"))
    except OSError:
        return False
