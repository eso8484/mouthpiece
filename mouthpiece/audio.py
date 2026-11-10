"""Microphone capture + a small energy-based voice-activity segmenter.

The segmenter cuts the recording at natural pauses while you are still talking, so earlier
chunks are transcribed in the background and only the last few seconds remain to be
processed when you release the hotkey (and so the overlay can show a live preview).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME = 480  # 30 ms


def rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    v = float(np.sqrt(np.mean(np.square(x.astype(np.float32)))))
    return 20.0 * np.log10(max(v, 1e-6))


@dataclass
class Segment:
    index: int
    audio: np.ndarray  # float32 mono 16 kHz
    final: bool
    speech_seconds: float


@dataclass
class Segmenter:
    """Cut float32 audio into utterance-sized segments at pauses."""

    min_segment: float = 6.0       # don't cut before this many seconds (more context = better accuracy)
    max_segment: float = 45.0      # force a cut eventually, at the quietest point
    pause: float = 0.7             # silence needed to cut
    pad: float = 0.25              # silence kept around speech
    on_segment: Callable[[Segment], None] | None = None

    buf: list[np.ndarray] = field(default_factory=list)
    frames_db: list[float] = field(default_factory=list)
    _pending: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    seg_start: int = 0               # frame index where the current segment starts
    index: int = 0
    noise_db: float = -60.0
    speech_frames_total: int = 0

    def threshold(self) -> float:
        # lenient on purpose: dropping real speech is far worse than transcribing a little noise
        return max(self.noise_db + 7.0, -55.0)

    def _is_speech(self, db: float) -> bool:
        return db > self.threshold()

    def feed(self, chunk: np.ndarray) -> None:
        data = np.concatenate([self._pending, chunk]) if self._pending.size else chunk
        n = len(data) // FRAME
        for i in range(n):
            frame = data[i * FRAME:(i + 1) * FRAME]
            self.buf.append(frame)
            db = rms_db(frame)
            self.frames_db.append(db)
            # noise floor = 10th percentile of the last ~10 s (gaps between words keep it honest)
            if len(self.frames_db) >= 8:
                self.noise_db = min(float(np.percentile(self.frames_db[-333:], 10)), -30.0)
            if self._is_speech(db):
                self.speech_frames_total += 1
            self._maybe_cut()
        self._pending = data[n * FRAME:].copy()

    def _frames_to_s(self, frames: int) -> float:
        return frames * FRAME / SAMPLE_RATE

    def _maybe_cut(self) -> None:
        total = len(self.buf)
        length = self._frames_to_s(total - self.seg_start)
        if length < self.min_segment:
            return
        # the longer the chunk, the shorter the pause we accept to cut it, so that what is
        # left to transcribe after you release the key stays short
        pause = self.pause if length < 2 * self.min_segment else self.pause * (0.6 if length < 3 * self.min_segment else 0.45)
        pause_frames = max(6, int(pause * SAMPLE_RATE / FRAME))
        tail = self.frames_db[-pause_frames:]
        if len(tail) == pause_frames and all(not self._is_speech(d) for d in tail):
            cut = total - pause_frames // 2
            self._emit(cut, final=False)
        elif length >= self.max_segment:
            # no pause found: cut at the quietest frame of the last 3 seconds
            window = int(3.0 * SAMPLE_RATE / FRAME)
            min_frames = int(self.min_segment * SAMPLE_RATE / FRAME)
            lo = max(self.seg_start + min_frames, total - window)
            # quietest frame; on ties the latest, so a steady signal isn't cut into slivers
            quiet = min(range(lo, total), key=lambda k: (self.frames_db[k], -k)) if lo < total else total
            self._emit(quiet, final=False)

    def _emit(self, cut: int, final: bool) -> None:
        frames = self.buf[self.seg_start:cut]
        dbs = self.frames_db[self.seg_start:cut]
        speech = [i for i, d in enumerate(dbs) if self._is_speech(d)]
        start_frame = self.seg_start
        self.seg_start = cut
        if not frames:
            audio = np.zeros(0, np.float32)
        else:
            pad = int(self.pad * SAMPLE_RATE / FRAME)
            if speech:
                a, b = max(0, speech[0] - pad), min(len(frames), speech[-1] + 1 + pad)
                frames = frames[a:b]
            audio = np.concatenate(frames)
        seg = Segment(self.index, audio, final, self._frames_to_s(len(speech)))
        self.index += 1
        log.debug("segment %d: frames %d..%d, %.1fs audio, %.1fs speech%s", seg.index, start_frame, cut,
                  len(audio) / SAMPLE_RATE, seg.speech_seconds, " (final)" if final else "")
        if self.on_segment:
            self.on_segment(seg)

    def flush(self) -> None:
        if self._pending.size:
            pad = np.zeros(FRAME - self._pending.size, np.float32)
            self.feed(pad)  # completes the partial frame
            self._pending = np.zeros(0, np.float32)
        self._emit(len(self.buf), final=True)

    def all_audio(self) -> np.ndarray:
        return np.concatenate(self.buf) if self.buf else np.zeros(0, np.float32)


def resolve_input_device(wanted: str | int | None):
    """Pick the capture device. Default: PortAudio's PulseAudio "Default Source" (PipeWire):
    it opens in ~25 ms and follows the mic chosen in GNOME Settings. The plain ALSA "default"
    device opens slower and emits a loud start-up transient on PipeWire systems."""
    import sounddevice as sd

    if wanted not in (None, ""):
        if isinstance(wanted, str) and wanted.isdigit():
            return int(wanted)
        return wanted
    try:
        devices = sd.query_devices()
        apis = sd.query_hostapis()
    except Exception:
        return None
    for preferred_api, preferred_name in (("PulseAudio", "Default Source"), ("ALSA", "pipewire")):
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0 and d["name"] == preferred_name \
                    and apis[d["hostapi"]]["name"] == preferred_api:
                return i
    return None


class Recorder:
    """Opens the microphone only while recording (so GNOME's mic indicator is honest)."""

    def __init__(self, device: str | int | None = None):
        self.device = device or None
        self._resolved = None
        self._resolved_for: object = object()
        self.stream = None
        self.segmenter: Segmenter | None = None
        self.level = 0.0
        self.started_at = 0.0
        self.first_audio_at = 0.0
        self._lock = threading.Lock()
        self.overflows = 0

    def start(self, segmenter: Segmenter) -> None:
        import sounddevice as sd

        self.stop(discard=True)
        self.segmenter = segmenter
        self.level = 0.0
        self.overflows = 0
        self.started_at = time.monotonic()
        self.first_audio_at = 0.0
        if self._resolved_for != self.device:
            self._resolved = resolve_input_device(self.device)
            self._resolved_for = self.device
        try:
            stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=FRAME,
                                    latency="low", device=self._resolved, callback=self._callback)
        except Exception:
            if self._resolved is None:
                raise
            log.warning("input device %r failed; using the system default", self._resolved)
            self._resolved = None
            stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=FRAME,
                                    latency="low", device=None, callback=self._callback)
        self.stream = stream
        self.stream.start()

    def _callback(self, indata, frames, _time, status) -> None:
        if status and status.input_overflow:
            self.overflows += 1
        chunk = indata[:, 0].copy()
        if not self.first_audio_at:
            self.first_audio_at = time.monotonic()
        peak = float(np.max(np.abs(chunk))) if chunk.size else 0.0
        self.level = 0.6 * self.level + 0.4 * min(1.0, peak * 1.6)
        with self._lock:
            seg = self.segmenter
            if seg is not None:
                try:
                    seg.feed(chunk)
                except Exception:
                    log.exception("segmenter failed")

    @property
    def active(self) -> bool:
        return self.stream is not None

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at if self.stream else 0.0

    def stop(self, discard: bool = False) -> Segmenter | None:
        stream, self.stream = self.stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                log.exception("closing microphone failed")
        with self._lock:
            seg, self.segmenter = self.segmenter, None
        if seg is None or discard:
            return None
        if self.overflows:
            log.warning("microphone overflowed %d times", self.overflows)
        return seg


def normalize_for_asr(audio: np.ndarray) -> np.ndarray:
    """Remove DC offset and bring quiet recordings up to a sane level (models are trained on it)."""
    if audio.size == 0:
        return audio
    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio)))
    if 1e-4 < peak < 0.5:
        audio = audio * min(0.9 / peak, 20.0)
    return audio.astype(np.float32)


def to_wav_bytes(audio: np.ndarray) -> bytes:
    import io
    import wave

    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return out.getvalue()


def load_audio_file(path: str) -> np.ndarray:
    """Decode any audio file to 16 kHz mono float32 via ffmpeg (falls back to soundfile)."""
    import shutil
    import subprocess

    if shutil.which("ffmpeg"):
        raw = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
            capture_output=True, check=True,
        ).stdout
        return np.frombuffer(raw, dtype="<f4").copy()
    import soundfile as sf

    data, rate = sf.read(path, dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if rate != SAMPLE_RATE:
        idx = np.linspace(0, len(data) - 1, int(len(data) * SAMPLE_RATE / rate))
        data = np.interp(idx, np.arange(len(data)), data).astype(np.float32)
    return data
