"""Floating status pill ("Listening… / Transcribing… / Pasted").

Runs as its own process under XWayland (GDK_BACKEND=x11) as an override-redirect popup: it
floats above every window, never takes keyboard focus (so the paste lands where you were
typing) and lets mouse clicks pass straight through (an empty X Shape input region). The
daemon drives it with JSON lines on stdin.

The pill is drawn with plain pycairo into an image and shown in a GTK window, so it needs no
python3-gi-cairo bridge (not installed by default on Ubuntu).
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import threading
import time

log = logging.getLogger(__name__)


class OverlayClient:
    """Daemon-side handle; restarts the overlay process if it dies (max 3 times)."""

    def __init__(self, enabled: bool, position: str = "bottom"):
        self.enabled = enabled and bool(os.environ.get("DISPLAY"))
        self.position = position
        self.proc: subprocess.Popen | None = None
        self.restarts = 0
        self._lock = threading.Lock()
        self._last_level = 0.0
        if enabled and not self.enabled:
            log.warning("overlay disabled: no X display (XWayland) available")

    def _spawn(self) -> None:
        # GDK_SCALE=1: GTK works in real pixels and we draw at the true HiDPI scale ourselves
        # (a 1x image upscaled by GTK would look blurry on 2x screens).
        env = dict(os.environ, GDK_BACKEND="x11", NO_AT_BRIDGE="1", GDK_SCALE="1", GDK_DPI_SCALE="1")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "mouthpiece.overlay", self.position],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env, text=True,
        )
        threading.Thread(target=self._drain, args=(self.proc,), daemon=True).start()

    def _drain(self, proc: subprocess.Popen) -> None:
        for line in proc.stderr or []:
            line = line.strip()
            if line and "Gtk-WARNING" not in line:
                log.warning("overlay: %s", line)

    def start(self) -> None:
        if self.enabled:
            with self._lock:
                self._spawn()

    def send(self, **msg) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self.proc is None or self.proc.poll() is not None:
                if self.restarts >= 3:
                    return
                self.restarts += 1
                self._spawn()
            try:
                assert self.proc and self.proc.stdin
                self.proc.stdin.write(json.dumps(msg) + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError):
                self.proc = None

    def level(self, v: float) -> None:
        if abs(v - self._last_level) > 0.01:
            self._last_level = v
            self.send(cmd="level", v=round(v, 3))

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.send(cmd="quit")
                self.proc.wait(1)
            except Exception:
                self.proc.kill()


# =========================================================================== drawing (pycairo only)

H = 44
W_MIN, W_MAX = 230, 560
FONT = ("Ubuntu", "Cantarell", "Noto Sans", "DejaVu Sans", "Sans")
FONT_SIZE = 14.0
COLORS = {
    "recording": (0.93, 0.27, 0.27),
    "handsfree": (0.95, 0.45, 0.20),
    "transcribing": (0.98, 0.75, 0.25),
    "formatting": (0.68, 0.56, 0.98),
    "done": (0.30, 0.80, 0.45),
    "error": (0.75, 0.75, 0.78),
}


class PillRenderer:
    def __init__(self) -> None:
        import cairo

        self.cairo = cairo
        self.state = "hidden"
        self.label = ""
        self.preview = ""
        self.level = 0.0
        self.shown_level = 0.0
        self.bars = [0.0] * 9
        self.started = time.monotonic()
        self._scratch = cairo.Context(cairo.ImageSurface(cairo.FORMAT_ARGB32, 4, 4))
        self._font(self._scratch)

    def _font(self, cr) -> None:
        cr.select_font_face(FONT[0], self.cairo.FONT_SLANT_NORMAL, self.cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(FONT_SIZE)

    @property
    def meter(self) -> bool:
        return self.state in ("recording", "handsfree")

    def text(self, now: float) -> str:
        if self.preview:
            return self.preview
        if self.meter:
            secs = int(now - self.started)
            return f"{self.label}  {secs // 60}:{secs % 60:02d}"
        return self.label

    def width(self, now: float) -> int:
        tw = self._scratch.text_extents(self.text(now)).x_advance
        want = 44 + tw + (112 if self.meter else 26)
        return int(max(W_MIN, min(W_MAX, want)))

    def _fit(self, cr, text: str, max_w: float) -> str:
        if cr.text_extents(text).x_advance <= max_w:
            return text
        lo, hi = 0, len(text)
        while lo < hi:  # keep the end of the text (newest words), drop the start
            mid = (lo + hi) // 2
            if cr.text_extents("…" + text[mid:]).x_advance <= max_w:
                hi = mid
            else:
                lo = mid + 1
        tail = text[lo:]
        cut = tail.find(" ")
        if 0 <= cut < 16:  # start on a whole word
            tail = tail[cut:]
        return "…" + tail.lstrip(" ,.;:")

    def render(self, now: float | None = None, scale: float = 1.0):
        cairo = self.cairo
        now = time.monotonic() if now is None else now
        w = self.width(now)
        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, math.ceil(w * scale), math.ceil(H * scale))
        cr = cairo.Context(surf)
        cr.scale(scale, scale)
        r = H / 2
        cr.new_sub_path()
        cr.arc(w - r, r, r - 1, -math.pi / 2, math.pi / 2)
        cr.arc(r, r, r - 1, math.pi / 2, 3 * math.pi / 2)
        cr.close_path()
        cr.set_source_rgba(0.09, 0.09, 0.11, 0.93)
        cr.fill_preserve()
        cr.set_source_rgba(1, 1, 1, 0.12)
        cr.set_line_width(1)
        cr.stroke()

        color = COLORS.get(self.state, COLORS["error"])
        cx, cy = 24, H / 2
        if self.state in ("transcribing", "formatting"):
            for i in range(3):
                a = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(now * 7 - i * 0.9))
                cr.set_source_rgba(*color, a)
                cr.arc(cx - 7 + i * 7, cy, 2.6, 0, 2 * math.pi)
                cr.fill()
        elif self.state == "done":
            cr.set_source_rgb(*color)
            cr.set_line_width(2.4)
            cr.set_line_cap(cairo.LINE_CAP_ROUND)
            cr.move_to(cx - 6, cy)
            cr.line_to(cx - 1.5, cy + 4.5)
            cr.line_to(cx + 6.5, cy - 5)
            cr.stroke()
        else:
            pulse = 0.75 + 0.25 * math.sin(now * 5) if self.meter else 1.0
            cr.set_source_rgba(*color, 0.25 * pulse)
            cr.arc(cx, cy, 9, 0, 2 * math.pi)
            cr.fill()
            cr.set_source_rgba(*color, pulse)
            cr.arc(cx, cy, 5, 0, 2 * math.pi)
            cr.fill()

        self._font(cr)
        right = 112 if self.meter else 22
        text = self._fit(cr, self.text(now), w - 44 - right)
        ext = cr.font_extents()  # ascent, descent, height, ...
        cr.set_source_rgba(1, 1, 1, 0.85 if self.preview else 0.96)
        cr.move_to(42, H / 2 + (ext[0] - ext[1]) / 2)
        cr.show_text(text)

        if self.meter:
            bx = w - 104
            for i, v in enumerate(self.bars):
                bh = 4 + min(1.0, v * 1.4) * (H - 18)
                cr.set_source_rgba(1, 1, 1, 0.35 + 0.6 * min(1.0, v * 1.5))
                self._rounded_bar(cr, bx + i * 10, (H - bh) / 2, 5, bh)
        surf.flush()
        return surf

    @staticmethod
    def _rounded_bar(cr, x, y, w, h) -> None:
        r = w / 2
        cr.new_sub_path()
        cr.arc(x + r, y + r, r, math.pi, 0)
        cr.arc(x + r, y + h - r, r, 0, math.pi)
        cr.close_path()
        cr.fill()

    def tick(self) -> None:
        self.shown_level += (self.level - self.shown_level) * 0.35
        self.bars = self.bars[1:] + [self.shown_level]


def surface_to_rgba(surf) -> tuple[bytes, int, int]:
    """cairo ARGB32 (premultiplied, BGRA in memory) -> straight RGBA bytes for GdkPixbuf."""
    import numpy as np

    w, h, stride = surf.get_width(), surf.get_height(), surf.get_stride()
    a = np.ndarray((h, stride // 4, 4), np.uint8, surf.get_data())[:, :w, :].astype(np.float32)
    b, g, r, alpha = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    scale = np.where(alpha > 0, 255.0 / np.maximum(alpha, 1), 0)
    rgba = np.stack([r * scale, g * scale, b * scale, alpha], axis=-1)
    return np.clip(rgba, 0, 255).astype(np.uint8).tobytes(), w, h


def _make_click_through(xid: int) -> None:
    """Empty X Shape input region: clicks fall through to the window underneath."""
    import ctypes
    import ctypes.util

    try:
        xlib = ctypes.CDLL(ctypes.util.find_library("X11") or "libX11.so.6")
        xext = ctypes.CDLL(ctypes.util.find_library("Xext") or "libXext.so.6")
        xlib.XOpenDisplay.restype = ctypes.c_void_p
        xlib.XOpenDisplay.argtypes = [ctypes.c_char_p]
        xlib.XFlush.argtypes = [ctypes.c_void_p]
        xlib.XCloseDisplay.argtypes = [ctypes.c_void_p]
        xext.XShapeCombineRectangles.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int,
                                                 ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        dpy = xlib.XOpenDisplay(None)
        if not dpy:
            return
        SHAPE_INPUT, SHAPE_SET, UNSORTED = 2, 0, 0
        xext.XShapeCombineRectangles(dpy, xid, SHAPE_INPUT, 0, 0, None, 0, SHAPE_SET, UNSORTED)
        xlib.XFlush(dpy)
        xlib.XCloseDisplay(dpy)
    except OSError as e:
        print(f"click-through unavailable: {e}", file=sys.stderr)


# =========================================================================== overlay process


def main(position: str = "bottom", snapshot_dir: str | None = None) -> None:  # pragma: no cover - needs a display
    pill = PillRenderer()

    if snapshot_dir:  # render each state to PNG without a display (tests / docs)
        os.makedirs(snapshot_dir, exist_ok=True)
        demo = [
            ("recording", "Listening", "", [0.1, 0.3, 0.6, 0.9, 0.5, 0.7, 0.4, 0.2, 0.6]),
            ("recording", "Listening", "Hello, I'd like to book a table for two people tonight at the place on",
             [0.2, 0.5, 0.8, 0.4, 0.6, 0.3, 0.7, 0.5, 0.3]),
            ("handsfree", "Hands-free — press Right Ctrl to stop", "", [0.3] * 9),
            ("transcribing", "Transcribing…", "", [0] * 9),
            ("formatting", "Formatting…", "", [0] * 9),
            ("done", "Pasted", "", [0] * 9),
            ("error", "No speech heard — check your microphone", "", [0] * 9),
        ]
        for i, (state, label, preview, bars) in enumerate(demo):
            pill.state, pill.label, pill.preview, pill.bars = state, label, preview, bars
            pill.started = 100.0
            scale = float(os.environ.get("MOUTHPIECE_SNAPSHOT_SCALE", "1"))
            pill.render(now=103.4, scale=scale).write_to_png(os.path.join(snapshot_dir, f"{i}-{state}.png"))
        return

    import gi

    gi.require_version("Gtk", "3.0")
    gi.require_version("Gdk", "3.0")
    gi.require_version("GdkPixbuf", "2.0")
    gi.require_version("GdkX11", "3.0")
    from gi.repository import Gdk, GdkPixbuf, GdkX11, GLib, Gtk

    win = Gtk.Window(type=Gtk.WindowType.POPUP)
    win.set_app_paintable(True)
    win.set_decorated(False)
    win.set_accept_focus(False)
    win.set_focus_on_map(False)
    win.set_type_hint(Gdk.WindowTypeHint.NOTIFICATION)
    visual = win.get_screen().get_rgba_visual()
    if visual is not None:
        win.set_visual(visual)
    css = Gtk.CssProvider()
    css.load_from_data(b"window, image { background-color: transparent; }")
    Gtk.StyleContext.add_provider_for_screen(win.get_screen(), css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
    image = Gtk.Image()
    win.add(image)

    def on_realize(_w) -> None:
        _make_click_through(GdkX11.X11Window.get_xid(win.get_window()))

    win.connect("realize", on_realize)
    # We run with GDK_SCALE=1, so GDK speaks real pixels; the desktop's scale comes from its DPI.
    scale = max(1.0, min(4.0, Gdk.Screen.get_default().get_resolution() / 96.0)) \
        if os.environ.get("GDK_SCALE") == "1" else 1.0
    state = {"hide_at": 0.0, "width": 0, "height": 0}

    def place(width: int, height: int) -> None:
        display = Gdk.Display.get_default()
        mon = display.get_primary_monitor() or display.get_monitor(0)
        g = mon.get_workarea()
        margin = int(56 * scale) if position != "top" else int(48 * scale)
        x = g.x + (g.width - width) // 2
        y = g.y + g.height - height - margin if position != "top" else g.y + margin
        win.resize(width, height)
        win.move(x, y)

    def redraw() -> None:
        surf = pill.render(scale=scale)
        data, w, h = surface_to_rgba(surf)
        pb = GdkPixbuf.Pixbuf.new_from_bytes(GLib.Bytes.new(data), GdkPixbuf.Colorspace.RGB, True, 8, w, h, w * 4)
        image.set_from_pixbuf(pb)
        if (w, h) != (state["width"], state["height"]):
            state["width"], state["height"] = w, h
            place(w, h)

    def handle(msg: dict) -> bool:
        cmd = msg.get("cmd")
        if cmd == "quit":
            Gtk.main_quit()
        elif cmd == "level":
            pill.level = float(msg.get("v", 0))
        elif cmd == "preview":
            pill.preview = msg.get("text", "")
        elif cmd == "hide":
            pill.state = "hidden"
            win.hide()
        elif cmd == "show":
            pill.state = msg.get("state", "recording")
            pill.label = msg.get("text", "")
            if pill.meter and msg.get("reset", True):
                pill.started = time.monotonic()
                pill.preview = ""
            if pill.state not in ("recording", "handsfree", "transcribing"):
                pill.preview = ""
            state["hide_at"] = time.monotonic() + float(msg["timeout"]) if msg.get("timeout") else 0.0
            redraw()
            win.show_all()
            place(state["width"], state["height"])
        return False

    def tick() -> bool:
        if pill.state == "hidden":
            return True
        if state["hide_at"] and time.monotonic() > state["hide_at"]:
            pill.state = "hidden"
            win.hide()
            return True
        pill.tick()
        redraw()
        return True

    GLib.timeout_add(40, tick)

    def reader() -> None:
        for line in sys.stdin:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            GLib.idle_add(handle, msg)
        GLib.idle_add(Gtk.main_quit)

    threading.Thread(target=reader, daemon=True).start()
    Gtk.main()


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) > 2 and sys.argv[1] == "--snapshot":
        main(snapshot_dir=sys.argv[2])
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "bottom")
