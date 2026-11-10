"""Speech-to-text engines.

local   NVIDIA Parakeet TDT 0.6B (v3 multilingual or v2 English) via onnx-asr, fully offline.
groq    Whisper large-v3 / large-v3-turbo on Groq.
openai  gpt-4o-transcribe / gpt-4o-mini-transcribe / whisper-1.

Cloud engines speak the OpenAI-compatible /audio/transcriptions API, so any compatible
server (e.g. a self-hosted whisper server) works by setting a custom base_url.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE, to_wav_bytes

log = logging.getLogger(__name__)

LOCAL_MODELS = {
    "parakeet-v3": ("nemo-parakeet-tdt-0.6b-v3", "parakeet-tdt-0.6b-v3-int8"),
    "parakeet-v2": ("nemo-parakeet-tdt-0.6b-v2", "parakeet-tdt-0.6b-v2-int8"),
}


class EngineError(RuntimeError):
    pass


def models_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(base) / "mouthpiece" / "models"


def default_threads() -> int:
    return max(1, min(4, (os.cpu_count() or 2) - 1))


class LocalEngine:
    name = "local"

    def __init__(self, model: str = "parakeet-v3", threads: int = 0, download: bool = True):
        if model not in LOCAL_MODELS:
            raise EngineError(f"unknown local model '{model}' (choose: {', '.join(LOCAL_MODELS)})")
        self.model_key = model
        self.threads = threads or default_threads()
        self.download = download
        self._model = None
        self._lock = threading.Lock()
        self._loaded = threading.Event()
        self.load_error: str | None = None
        self.load_seconds = 0.0

    @property
    def label(self) -> str:
        return f"local {self.model_key}"

    def model_path(self) -> Path:
        return models_dir() / LOCAL_MODELS[self.model_key][1]

    def is_downloaded(self) -> bool:
        p = self.model_path()
        return (p / "encoder-model.int8.onnx").exists() and (p / "vocab.txt").exists()

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            import onnx_asr
            import onnxruntime as rt

            if not self.is_downloaded() and not self.download:
                raise EngineError(f"model {self.model_key} is not downloaded (run: mouthpiece download-model)")
            t = time.monotonic()
            so = rt.SessionOptions()
            so.intra_op_num_threads = self.threads
            so.inter_op_num_threads = 1
            name, folder = LOCAL_MODELS[self.model_key]
            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
            try:
                self._model = onnx_asr.load_model(name, str(models_dir() / folder), quantization="int8", sess_options=so)
                self._model.recognize(np.zeros(SAMPLE_RATE // 2, dtype=np.float32), sample_rate=SAMPLE_RATE)  # warm-up
            except Exception as e:
                self.load_error = str(e)
                raise EngineError(f"could not load {self.model_key}: {e}") from e
            self.load_seconds = time.monotonic() - t
            self._loaded.set()
            log.info("local model %s ready in %.1fs (%d threads)", self.model_key, self.load_seconds, self.threads)

    def transcribe(self, audio: np.ndarray, prompt: str = "", language: str = "") -> str:
        if self._model is None:
            self.load()
        with self._lock:
            text = self._model.recognize(audio.astype(np.float32), sample_rate=SAMPLE_RATE)
        return str(text).strip()


def _multipart(fields: dict[str, str], file_field: str, filename: str, data: bytes, ctype: str) -> tuple[bytes, str]:
    boundary = f"----mouthpiece{uuid.uuid4().hex}"
    parts = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n".encode() + data + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def http_json(url: str, *, data: bytes, headers: dict[str, str], timeout: float) -> dict:
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:400]
        try:
            body = json.loads(body).get("error", {}).get("message", body)
        except Exception:
            pass
        raise EngineError(f"HTTP {e.code}: {body}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise EngineError(f"network error: {getattr(e, 'reason', e)}") from e


class CloudEngine:
    """OpenAI-compatible /audio/transcriptions client."""

    def __init__(self, name: str, base_url: str, api_key: str, model: str, timeout: float = 30.0):
        if not api_key:
            raise EngineError(f"no API key for {name} (run: mouthpiece set-key {name})")
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @property
    def label(self) -> str:
        return f"{self.name} {self.model}"

    def load(self) -> None:
        pass

    def transcribe(self, audio: np.ndarray, prompt: str = "", language: str = "") -> str:
        fields = {"model": self.model, "response_format": "json", "temperature": "0"}
        if language:
            fields["language"] = language
        if prompt:
            fields["prompt"] = prompt[-800:]
        body, ctype = _multipart(fields, "file", "speech.wav", to_wav_bytes(audio), "audio/wav")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": ctype,
            "User-Agent": "mouthpiece/1.0",
        }
        t = time.monotonic()
        out = http_json(f"{self.base_url}/audio/transcriptions", data=body, headers=headers, timeout=self.timeout)
        log.debug("%s answered in %.2fs", self.label, time.monotonic() - t)
        return str(out.get("text", "")).strip()


def build_engine(cfg, secrets: dict[str, str], name: str | None = None):
    """Create the engine named in config ('auto' prefers Groq when a key is set)."""
    t = cfg["transcription"]
    name = name or t["engine"]
    if name == "auto":
        name = "groq" if secrets.get("groq") else "openai" if secrets.get("openai") else "local"
    if name == "local":
        return LocalEngine(t["local_model"], int(t.get("threads", 0) or 0))
    if name == "groq":
        return CloudEngine("groq", t.get("groq_base_url") or "https://api.groq.com/openai/v1",
                           secrets.get("groq", ""), t["groq_model"])
    if name == "openai":
        return CloudEngine("openai", t.get("openai_base_url") or "https://api.openai.com/v1",
                           secrets.get("openai", ""), t["openai_model"])
    if name == "custom":
        return CloudEngine("custom", t["custom_base_url"], secrets.get("custom", "") or "none", t["custom_model"])
    raise EngineError(f"unknown engine '{name}'")
