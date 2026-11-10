"""Short, soft audio cues (generated once, played with pw-play/aplay without blocking)."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

RATE = 44100


def _tone(freqs: list[float], dur: float, vol: float) -> np.ndarray:
    parts = []
    seg = dur / len(freqs)
    for f in freqs:
        t = np.arange(int(RATE * seg)) / RATE
        env = np.minimum(1.0, t / 0.008) * np.exp(-t * 9.0)  # quick attack, soft decay
        parts.append(np.sin(2 * np.pi * f * t) * env + 0.25 * np.sin(4 * np.pi * f * t) * env)
    return np.concatenate(parts) * vol


CUES = {
    "start": ([660.0, 990.0], 0.16),
    "stop": ([880.0, 587.0], 0.16),
    "cancel": ([440.0, 330.0], 0.18),
    "error": ([220.0, 196.0], 0.30),
    "lock": ([660.0, 880.0, 1100.0], 0.20),
}


class Sounds:
    def __init__(self, enabled: bool, volume: float = 0.35):
        self.enabled = enabled
        self.volume = max(0.0, min(1.0, volume))
        base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
        self.dir = Path(base) / "mouthpiece" / "sounds"
        self.player = shutil.which("pw-play") or shutil.which("paplay") or shutil.which("aplay")
        if enabled:
            self._ensure()

    def _ensure(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        for name, (freqs, dur) in CUES.items():
            p = self.dir / f"{name}.wav"
            if p.exists():
                continue
            pcm = (np.clip(_tone(freqs, dur, 0.9), -1, 1) * 32767).astype("<i2").tobytes()
            with wave.open(str(p), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(RATE)
                w.writeframes(pcm)

    def play(self, name: str) -> None:
        if not self.enabled or not self.player or name not in CUES:
            return
        path = str(self.dir / f"{name}.wav")
        args = [self.player, path]
        if self.player.endswith("pw-play"):
            args = [self.player, f"--volume={self.volume:.2f}", "--media-role=Notification", path]
        elif self.player.endswith("aplay"):
            args = [self.player, "-q", path]
        try:
            subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        except OSError as e:
            log.debug("sound failed: %s", e)
