"""The Mouthpiece daemon: hotkey → microphone → speech-to-text → paste into the focused app."""

from __future__ import annotations

import itertools
import logging
import queue
import signal
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from . import history
from .audio import Recorder, Segment, Segmenter, normalize_for_asr
from .config import load_config, load_secrets
from .context import AppInfo, AppProbe
from .control import ControlServer
from .engines import CloudEngine, EngineError, LocalEngine, build_engine
from .formatting import ClaudeUnavailable, FormatError, Formatter, speaks_pidgin
from .gnome_keys import EscapeSwallower, is_gnome
from .hotkey import MODIFIER_CODES, Action, Hotkey, HotkeyError, HotkeyListener
from .injector import Injector
from .overlay import OverlayClient
from .sounds import Sounds
from .textproc import finalize_spacing, join_segments, postprocess

log = logging.getLogger(__name__)

_ids = itertools.count(1)

# After the cloud engine fails (offline, slow, out of quota), seconds to use the local model before
# trying the cloud again.
CLOUD_RETRY_AFTER = 120.0

# Whisper famously "hears" these in silence/noise. Only applied to cloud Whisper results when
# almost no speech was detected, so a real "Thank you" reply still gets through.
_HALLUCINATIONS = {
    "thank you", "thanks for watching", "thank you for watching", "thank you very much",
    "please subscribe", "subtitles by the amara.org community", "you", ".", "...",
}


def _is_hallucination(text: str) -> bool:
    return text.strip().strip(".!?,").lower() in _HALLUCINATIONS


@dataclass
class Session:
    id: int
    press_at: float
    probe: AppProbe
    recording: bool = True
    handsfree: bool = False
    cancelled: bool = False
    finished: bool = False
    feedback_shown: bool = False
    final_index: int | None = None
    speech_seconds: float = 0.0
    audio_seconds: float = 0.0
    held_segments: list[Segment] = field(default_factory=list)
    results: dict[int, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    engine_used: str = ""
    stopped_at: float = 0.0
    fmt_job: object | None = None      # Claude Code process started early for formatting
    fmt_requested: bool = False
    skip_format: bool = False

    def text_so_far(self) -> str:
        return join_segments([self.results[i] for i in sorted(self.results)])

    def complete(self) -> bool:
        return self.final_index is not None and all(i in self.results for i in range(self.final_index + 1))


class Daemon:
    TICK = 0.05

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.secrets = load_secrets()
        self.events: queue.Queue = queue.Queue()
        self.hotkey = self._parse_hotkey()
        self.recorder = Recorder(self.cfg["audio"].get("input_device") or None)
        self.overlay = OverlayClient(self.cfg["ui"].get("overlay", True), self.cfg["ui"].get("overlay_position", "bottom"))
        self.sounds = Sounds(self.cfg["audio"].get("sounds", True), float(self.cfg["audio"].get("sound_volume", 0.35)))
        self.listener = HotkeyListener(self.hotkey, self._on_hotkey, self._cancel_codes())
        self.injector = Injector(self.cfg, wait_released=self._wait_released)
        self.engine = None
        self.engine_error: str | None = None
        self.fallback: LocalEngine | None = None
        self._cloud_down_until = 0.0
        self._build_engine()
        self.formatter = Formatter(self.cfg, self.secrets)
        self.session: Session | None = None        # the one currently recording
        self.pending: list[Session] = []           # recorded, awaiting transcription/paste
        self.jobs: queue.Queue = queue.Queue()
        self.deliveries: queue.Queue = queue.Queue()
        self.last_result: dict = {}
        self.last_app: AppInfo = AppInfo()
        self.running = True
        self.control: ControlServer | None = None
        self._last_level_push = 0.0
        self._busy = False
        self.esc: EscapeSwallower | None = None
        if (is_gnome() and self.cfg["hotkey"].get("swallow_cancel_key", True)
                and self._cancel_codes() == Hotkey.parse("ESC").codes):
            try:
                self.esc = EscapeSwallower()
            except Exception as e:
                log.warning("Esc swallowing unavailable: %s", e)

    # ------------------------------------------------------------------ setup helpers
    def _parse_hotkey(self) -> Hotkey:
        spec = self.cfg["hotkey"].get("key", "RIGHTCTRL")
        try:
            return Hotkey.parse(spec)
        except HotkeyError as e:
            log.error("%s; falling back to RIGHTCTRL", e)
            return Hotkey.parse("RIGHTCTRL")

    def _cancel_codes(self) -> frozenset[int]:
        spec = (self.cfg["hotkey"].get("cancel_key") or "").strip()
        if not spec:
            return frozenset()
        try:
            return Hotkey.parse(spec).codes
        except HotkeyError:
            return frozenset()

    def _build_engine(self) -> None:
        self.engine_error = None
        try:
            self.engine = build_engine(self.cfg, self.secrets)
        except EngineError as e:
            self.engine_error = str(e)
            log.error("engine: %s — using the local model", e)
            self.engine = LocalEngine(self.cfg["transcription"]["local_model"])
        self._cloud_down_until = 0.0
        threading.Thread(target=self._load_engine, args=(self.engine,), name="engine-load", daemon=True).start()
        t = self.cfg["transcription"]
        if isinstance(self.engine, CloudEngine) and t.get("fallback_to_local", True):
            # loaded in the background now, so losing the connection doesn't add a 10-30 s model load
            if self.fallback is None or self.fallback.model_key != t["local_model"]:
                self.fallback = LocalEngine(t["local_model"], int(t.get("threads", 0) or 0))
            threading.Thread(target=self._load_fallback, args=(self.fallback,), name="fallback-load",
                             daemon=True).start()
        else:
            self.fallback = None  # ~700 MB of memory back

    @staticmethod
    def _load_fallback(engine: LocalEngine) -> None:
        try:
            engine.load()
        except Exception as e:
            log.warning("local fallback model failed to load: %s", e)

    def _load_engine(self, engine) -> None:
        try:
            engine.load()
        except Exception as e:
            self.engine_error = str(e)
            log.error("engine failed to load: %s", e)

    def _wait_released(self, timeout: float) -> bool:
        return self.listener.wait_keys_released(MODIFIER_CODES | self.hotkey.codes, timeout)

    # ------------------------------------------------------------------ event plumbing
    def _on_hotkey(self, action: Action, code: int) -> None:
        self.events.put(("hotkey", action, code, time.monotonic()))

    def post(self, *event) -> None:
        self.events.put(event)

    def call(self, fn, timeout: float = 5.0):
        """Run fn on the main loop and wait for its result (used by the control socket)."""
        box: queue.Queue = queue.Queue(maxsize=1)
        self.events.put(("call", fn, box))
        ok, value = box.get(timeout=timeout)
        if not ok:
            raise value
        return value

    # ------------------------------------------------------------------ recording lifecycle
    def start_recording(self, now: float) -> None:
        if not self.engine:
            return
        s = Session(id=next(_ids), press_at=now, probe=AppProbe())
        seg = Segmenter(on_segment=lambda segment, s=s: self.post("segment", s, segment))
        try:
            self.recorder.start(seg)
        except Exception as e:
            log.error("microphone failed: %s", e)
            self.sounds.play("error")
            self.overlay.send(cmd="show", state="error", text="Microphone unavailable", timeout=3)
            return
        self.session = s
        log.info("recording #%d", s.id)

    def _show_recording_feedback(self, s: Session) -> None:
        s.feedback_shown = True
        self.sounds.play("start")
        mode = self.cfg["hotkey"].get("mode", "hold")
        label = "Listening" if mode != "toggle" else f"Listening — press {self.hotkey.pretty()} to stop"
        self.overlay.send(cmd="show", state="recording", text=label)

    def stop_recording(self, reason: str = "") -> None:
        s = self.session
        if s is None:
            return
        self.session = None
        s.recording = False
        s.stopped_at = time.monotonic()
        seg = self.recorder.stop()
        if seg is None:
            return
        seg.flush()  # emits the final segment through the queue
        s.speech_seconds = seg.speech_frames_total * 0.03
        s.audio_seconds = len(seg.buf) * 0.03
        log.info("recording #%d stopped%s: %.1fs audio, %.1fs speech", s.id, f" ({reason})" if reason else "",
                 s.audio_seconds, s.speech_seconds)
        if not s.feedback_shown:
            self._show_recording_feedback(s)
        self.pending.append(s)
        if s.speech_seconds < 0.15:
            s.cancelled = True
            self._drop_format_job(s)
            self.sounds.play("cancel")
            self.overlay.send(cmd="show", state="error", text="No speech heard — check your microphone", timeout=2.2)
            return
        self.sounds.play("stop")
        self.overlay.send(cmd="show", state="transcribing", text="Transcribing…", reset=False)

    def cancel(self, why: str, quiet: bool = False) -> None:
        target = self.session
        if target is not None:
            self.session = None
            self.recorder.stop(discard=True)
            target.recording = False
        else:
            target = next((p for p in reversed(self.pending) if not p.finished and not p.cancelled), None)
        if target is None:
            return
        target.cancelled = True
        self._drop_format_job(target)
        log.info("recording #%d cancelled (%s)", target.id, why)
        if quiet:
            self.overlay.send(cmd="hide")
            return
        self.sounds.play("cancel")
        self.overlay.send(cmd="show", state="error", text="Cancelled", timeout=0.9)

    # ------------------------------------------------------------------ hotkey handling
    def handle_hotkey(self, action: Action, now: float) -> None:
        h = self.cfg["hotkey"]
        mode = h.get("mode", "hold")
        s = self.session
        if action is Action.CANCEL:
            if s is not None or any(not p.finished and not p.cancelled for p in self.pending):
                self.cancel("cancel key")
            return
        if action is Action.INTERRUPT:
            if s is not None and not s.handsfree and now - s.press_at < h.get("shortcut_guard_ms", 400) / 1000:
                self.cancel("keyboard shortcut", quiet=True)
            return
        if action is Action.PRESS:
            if s is None:
                self.start_recording(now)
            elif mode == "toggle" or s.handsfree:
                self.stop_recording("pressed again")
            return
        if action is Action.RELEASE and s is not None:
            held = now - s.press_at
            if mode == "hold":
                if held < h.get("min_hold_ms", 250) / 1000:
                    self.cancel("released too quickly", quiet=True)
                else:
                    self.stop_recording()
            elif mode == "hybrid" and not s.handsfree:
                if held < h.get("tap_ms", 300) / 1000:
                    s.handsfree = True
                    if not s.feedback_shown:
                        s.feedback_shown = True
                    self.sounds.play("lock")
                    self.overlay.send(cmd="show", state="handsfree",
                                      text=f"Hands-free — press {self.hotkey.pretty()} to stop", reset=False)
                else:
                    self.stop_recording()

    # ------------------------------------------------------------------ transcription
    def _submit(self, s: Session, segment: Segment) -> None:
        if segment.final:
            s.final_index = segment.index
        live = self.cfg["transcription"].get("live_segments", True)
        if live or segment.final:
            for held in s.held_segments:
                self.jobs.put((s, held))
            s.held_segments.clear()
            self.jobs.put((s, segment))
        else:
            s.held_segments.append(segment)

    def _prompt_for(self, s: Session) -> str:
        vocab = self.cfg["text"].get("vocabulary") or []
        parts = []
        if speaks_pidgin(self.cfg["text"].get("languages") or []):
            # cloud Whisper copies the spelling style of its prompt
            parts.append("Abeg, wetin dey happen? Make una no vex, I dey come now now.")
        if vocab:
            parts.append("Vocabulary: " + ", ".join(vocab) + ".")
        prev = s.text_so_far()
        if prev:
            parts.append(prev[-300:])
        return " ".join(parts)

    def _transcribe_worker(self) -> None:
        while True:
            s, segment = self.jobs.get()
            if s is None:
                return
            text, err, used = "", None, ""
            if not s.cancelled and segment.speech_seconds >= 0.1:
                audio = normalize_for_asr(segment.audio)
                text, err, used = self._run_engine(s, audio)
                if (text and segment.speech_seconds < 0.5 and isinstance(self.engine, CloudEngine)
                        and "fallback" not in used and _is_hallucination(text)):
                    log.info("dropped a likely hallucination on near-silence")
                    text = ""
            self.post("result", s, segment.index, text, err, used)

    def _run_engine(self, s: Session, audio: np.ndarray) -> tuple[str, str | None, str]:
        t = self.cfg["transcription"]
        engine = self.engine
        use_fallback = isinstance(engine, CloudEngine) and t.get("fallback_to_local", True)
        if use_fallback and time.monotonic() < self._cloud_down_until:
            return self._run_fallback(audio, "")
        try:
            t0 = time.monotonic()
            text = engine.transcribe(audio, prompt=self._prompt_for(s), language=t.get("language", ""))
            log.info("segment (%.1fs audio) transcribed by %s in %.2fs", len(audio) / 16000, engine.label,
                     time.monotonic() - t0)
            return text, None, engine.label
        except Exception as e:
            log.error("%s failed: %s", getattr(engine, "label", "engine"), e)
            if use_fallback:
                # offline, slow or out of quota: the next chunks go straight to the local model for a
                # while instead of each waiting for the cloud to fail again
                self._cloud_down_until = time.monotonic() + CLOUD_RETRY_AFTER
                return self._run_fallback(audio, str(e))
            return "", str(e), ""

    def _run_fallback(self, audio: np.ndarray, cloud_error: str) -> tuple[str, str | None, str]:
        try:
            if self.fallback is None:
                self.fallback = LocalEngine(self.cfg["transcription"]["local_model"])
            text = self.fallback.transcribe(audio)
            return text, None, self.fallback.label + " (fallback)"
        except Exception as e2:
            return "", f"{cloud_error or 'cloud engine unavailable'}; local fallback: {e2}", ""

    def handle_result(self, s: Session, index: int, text: str, err: str | None, used: str) -> None:
        if err:
            s.errors.append(err)
        if used:
            s.engine_used = used
        s.results[index] = text
        if s.cancelled:
            s.finished = True
            self._gc()
            return
        if self.cfg["ui"].get("live_preview", True) and text:
            self.overlay.send(cmd="preview", text=s.text_so_far())
        if s.complete():
            self.deliveries.put(s)

    @staticmethod
    def _drop_format_job(s: Session) -> None:
        job, s.fmt_job = s.fmt_job, None
        if job is not None:
            job.kill()

    def _gc(self) -> None:
        self.pending = [p for p in self.pending if not p.finished]

    # ------------------------------------------------------------------ delivery
    def _delivery_worker(self) -> None:
        while True:
            s = self.deliveries.get()
            if s is None:
                return
            try:
                self._deliver(s)
            except Exception as e:
                log.exception("delivery failed")
                self.post("delivered", s, {"ok": False, "error": str(e)}, "")

    def _deliver(self, s: Session) -> None:
        raw = s.text_so_far()
        if s.cancelled:
            self.post("delivered", s, {"ok": False, "cancelled": True}, "")
            return
        if not raw.strip():
            err = s.errors[-1] if s.errors else ""
            self.post("delivered", s, {"ok": False, "error": err or "Didn't catch any words — try again",
                                       "empty": not err}, "")
            return
        app = s.probe.get()
        text = postprocess(raw, self.cfg)
        fmt: dict = {}
        formatter = self.formatter
        job, s.fmt_job = s.fmt_job, None
        if formatter.active and not s.skip_format:
            fmt = {"formatted": False}
            self.overlay.send(cmd="show", state="formatting", text="Formatting…")
            t0 = time.monotonic()
            try:
                plain, text = text, formatter.format(text, app, job)
                fmt = {"formatted": True, "raw": plain, "format_ms": int((time.monotonic() - t0) * 1000)}
                log.info("formatted #%d by %s in %.2fs (%d -> %d chars)", s.id, formatter.last_used,
                         time.monotonic() - t0, len(plain), len(text))
            except FormatError as e:
                fmt["format_error"] = str(e)
                if isinstance(e, ClaudeUnavailable):
                    fmt["format_note"] = e.short
                if not s.cancelled:
                    log.warning("formatting skipped for #%d: %s", s.id, e)
            except Exception as e:  # never lose a dictation because of formatting
                fmt["format_error"] = str(e)
                log.exception("formatting failed for #%d", s.id)
        elif job is not None:
            job.kill()
        final = finalize_spacing(text, self.cfg["text"].get("append_space", True))
        if s.cancelled:
            self.post("delivered", s, {"ok": False, "cancelled": True}, "")
            return
        result = self.injector.deliver(final, app)
        log.info("delivered #%d: %d chars via %s into %s (%s)", s.id, len(final), result.get("method"),
                 app.describe(), "ok" if result.get("ok") else result.get("error"))
        self.post("delivered", s, {**result, **fmt}, text)

    def handle_delivered(self, s: Session, result: dict, text: str) -> None:
        s.finished = True
        self._drop_format_job(s)
        self._gc()
        if result.get("cancelled"):
            return
        self.last_result = {"text": text, **result, "engine": s.engine_used,
                            "latency_ms": int((time.monotonic() - s.stopped_at) * 1000) if s.stopped_at else None}
        self.last_app = s.probe.get(0)
        if result.get("ok"):
            if self.cfg["history"].get("enabled", True):
                entry = {"text": text, "app": self.last_app.app, "engine": s.engine_used,
                         "seconds": round(s.audio_seconds, 1), "method": result.get("method")}
                if result.get("formatted"):
                    entry["raw"] = result.get("raw")
                elif result.get("format_error"):
                    entry["unformatted"] = result.get("format_note") or result["format_error"][:200]
                history.append(entry)
            if self.session is None:
                if result.get("format_note"):  # e.g. the Claude usage limit: say so, it lasts a while
                    self.overlay.send(cmd="show", state="done", text=f"Pasted unformatted: {result['format_note']}",
                                      timeout=4.5)
                elif result.get("format_error"):
                    self.overlay.send(cmd="show", state="done", text="Pasted (not formatted)", timeout=1.8)
                else:
                    self.overlay.send(cmd="show", state="done", text="Pasted", timeout=0.9)
        else:
            err = result.get("error") or "failed"
            self.sounds.play("cancel" if result.get("empty") else "error")
            hint = "Couldn't paste — text is on the clipboard (Ctrl+V)" if text and "clipboard" not in err else err
            self.overlay.send(cmd="show", state="error", text=hint[:90], timeout=2.5 if result.get("empty") else 4)

    # ------------------------------------------------------------------ control commands
    def handle_command(self, cmd: str, args: list[str]) -> dict:
        now = time.monotonic()
        if cmd == "ping":
            return {"ok": True}
        if cmd == "toggle":
            if self.session is None:
                self.start_recording(now)
                return {"ok": True, "state": "recording"}
            self.stop_recording("toggle command")
            return {"ok": True, "state": "transcribing"}
        if cmd == "start":
            if self.session is None:
                self.start_recording(now)
                if self.session:
                    self.session.handsfree = True
            return {"ok": True}
        if cmd == "stop":
            self.stop_recording("stop command")
            return {"ok": True}
        if cmd == "cancel":
            self.cancel("cancel command")
            return {"ok": True}
        if cmd == "paste-last":
            text = history.last_text()
            if not text:
                return {"ok": False, "error": "no transcript in history"}
            s = Session(id=next(_ids), press_at=now, probe=AppProbe())
            s.results = {0: text}
            s.final_index = 0
            s.speech_seconds = 1.0
            s.skip_format = True
            self.pending.append(s)
            self.deliveries.put(s)
            return {"ok": True}
        if cmd == "status":
            return {"ok": True, **self.status()}
        if cmd == "reload":
            self.reload()
            return {"ok": True, **self.status()}
        if cmd == "quit":
            self.running = False
            return {"ok": True}
        return {"ok": False, "error": f"unknown command '{cmd}'"}

    def status(self) -> dict:
        eng = self.engine
        loaded = eng is not None and (not isinstance(eng, LocalEngine) or eng._model is not None)
        return {
            "state": "recording" if self.session else ("transcribing" if any(not p.finished for p in self.pending) else "idle"),
            "hotkey": self.hotkey.pretty(),
            "hotkey_spec": self.hotkey.spec,
            "mode": self.cfg["hotkey"].get("mode"),
            "engine": getattr(eng, "label", "?"),
            "engine_ready": loaded,
            "engine_error": self.engine_error,
            "keyboards": [d.name for d in self.listener.devices.values()],
            "input_errors": self.listener.open_errors,
            "paste": self.cfg["paste"].get("method"),
            "clipboard_backend": self.injector.clipboard.backend,
            "virtual_keyboard": self.injector.keyboard is not None,
            "injector_error": self.injector.error,
            "formatting": self.formatter.describe(),
            "last": self.last_result,
            "last_app": {"app": self.last_app.app, "kind": self.last_app.kind, "title": self.last_app.title},
        }

    def reload(self) -> None:
        old_engine_cfg = dict(self.cfg["transcription"])
        self.cfg = load_config()
        self.secrets = load_secrets()
        self.hotkey = self._parse_hotkey()
        self.listener.set_hotkey(self.hotkey, self._cancel_codes())
        self.injector.cfg = self.cfg
        self.sounds = Sounds(self.cfg["audio"].get("sounds", True), float(self.cfg["audio"].get("sound_volume", 0.35)))
        self.recorder.device = self.cfg["audio"].get("input_device") or None
        if dict(self.cfg["transcription"]) != old_engine_cfg or self.engine_error:
            self._build_engine()
        self.formatter = Formatter(self.cfg, self.secrets)
        log.info("reloaded: hotkey %s (%s), engine %s, formatting %s", self.hotkey.pretty(),
                 self.cfg["hotkey"]["mode"], getattr(self.engine, "label", "?"), self.formatter.describe())

    # ------------------------------------------------------------------ main loop
    def _sync_busy(self) -> None:
        busy = self.session is not None or any(not p.finished and not p.cancelled for p in self.pending)
        if busy != self._busy:
            self._busy = busy
            if self.esc is not None:
                self.esc.want(busy)

    def tick(self, now: float) -> None:
        self._sync_busy()
        s = self.session
        if s is None:
            return
        # the mic is already recording; the beep + pill wait a moment so keyboard shortcuts
        # (Right Ctrl+C...) that get cancelled don't flash anything
        if not s.feedback_shown and now - s.press_at > 0.18:
            self._show_recording_feedback(s)
        if not s.fmt_requested and now - s.press_at > 0.45:
            s.fmt_requested = True
            s.fmt_job = self.formatter.prepare()
        if now - self._last_level_push > 0.06:
            self._last_level_push = now
            self.overlay.level(self.recorder.level)
        if self.recorder.elapsed() > float(self.cfg["audio"].get("max_seconds", 600)):
            self.stop_recording("max length reached")

    def dispatch(self, ev: tuple) -> None:
        kind = ev[0]
        if kind == "hotkey":
            self.handle_hotkey(ev[1], ev[3])
        elif kind == "segment":
            s, segment = ev[1], ev[2]
            if not s.cancelled:
                self._submit(s, segment)
            elif segment.final:
                s.finished = True
                self._gc()
        elif kind == "result":
            self.handle_result(*ev[1:])
        elif kind == "delivered":
            self.handle_delivered(*ev[1:])
        elif kind == "call":
            fn, box = ev[1], ev[2]
            try:
                box.put((True, fn()))
            except Exception as e:
                box.put((False, e))

    def run(self) -> None:
        self.control = ControlServer(lambda cmd, args: self.call(lambda: self.handle_command(cmd, args)))
        self.control.start()
        self.overlay.start()
        self.listener.start()
        if self.esc is not None:
            self.esc.start()
        threading.Thread(target=self._transcribe_worker, name="transcriber", daemon=True).start()
        threading.Thread(target=self._delivery_worker, name="delivery", daemon=True).start()
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, lambda *_: setattr(self, "running", False))
            signal.signal(signal.SIGHUP, lambda *_: self.post("call", self.reload, queue.Queue(1)))
        log.info("Mouthpiece ready: hold %s to talk (%s mode), engine %s, formatting %s", self.hotkey.pretty(),
                 self.cfg["hotkey"].get("mode"), getattr(self.engine, "label", "?"), self.formatter.describe())
        try:
            while self.running:
                try:
                    ev = self.events.get(timeout=self.TICK)
                except queue.Empty:
                    ev = None
                if ev is not None:
                    try:
                        self.dispatch(ev)
                    except Exception:
                        log.exception("event %s failed", ev[0])
                self.tick(time.monotonic())
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        log.info("shutting down")
        if self.esc is not None:
            self.esc.stop()
        self.listener.stop()
        self.recorder.stop(discard=True)
        for s in [self.session, *self.pending]:
            if s is not None:
                self._drop_format_job(s)
        self.jobs.put((None, None))
        self.deliveries.put(None)
        self.injector.close()
        self.overlay.stop()
        if self.control:
            self.control.close()
