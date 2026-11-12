import numpy as np
import pytest

from mouthpiece import config as cfgmod
from mouthpiece.audio import FRAME, SAMPLE_RATE, Segmenter
from mouthpiece.context import AppInfo, classify
from mouthpiece.injector import _char_map, choose_method
from mouthpiece.textproc import (
    apply_replacements,
    apply_vocabulary,
    finalize_spacing,
    join_segments,
    postprocess,
    remove_fillers,
)


@pytest.mark.parametrize("raw,clean", [
    ("Um, so I think we should go.", "So I think we should go."),
    ("I was, uh, going to say hello", "I was, going to say hello"),
    ("Let's meet, um.", "Let's meet."),
    ("Hmm okay", "Okay"),
    ("Umbrella and human stay", "Umbrella and human stay"),
])
def test_remove_fillers(raw, clean):
    assert remove_fillers(raw) == clean


def test_vocabulary_casing_and_joins():
    vocab = ["Claude Code", "WhatsApp", "GitHub", "Next.js", "JavaScript"]
    s = "open claude code and whats app, push to git hub, use next js and java script"
    assert apply_vocabulary(s, vocab) == "open Claude Code and WhatsApp, push to GitHub, use Next.js and JavaScript"


def test_vocabulary_whole_words_only():
    assert apply_vocabulary("the codex was githubby", ["GitHub", "Codex"]) == "the Codex was githubby"


def test_replacements():
    assert apply_replacements("Open cloud code please", {"cloud code": "Claude Code"}) == "Open Claude Code please"


def test_postprocess_defaults():
    cfg = cfgmod.DEFAULTS
    assert postprocess("  um  open   cloud code  ", cfg) == "Open Claude Code"
    assert postprocess("Um, open cloud code.", cfg) == "Open Claude Code."
    no_caps = {**cfg, "text": {**cfg["text"], "capitalize_first": False}}
    assert postprocess("um open cloud code", no_caps) == "open Claude Code"


def test_join_segments_lowercases_continuations():
    assert join_segments(["I went to the shop", "And bought milk.", "Then I left."]) == \
        "I went to the shop and bought milk. Then I left."
    assert join_segments(["Hello.", "", "I am here"]) == "Hello. I am here"
    assert join_segments(["It was stressful these days.", "you know, it was hard."]) == \
        "It was stressful these days. You know, it was hard."


def test_finalize_spacing():
    assert finalize_spacing("hi", True) == "hi "
    assert finalize_spacing("hi", False) == "hi"
    assert finalize_spacing("", True) == ""


def test_classify_apps():
    assert classify("ptyxis") == "terminal"
    assert classify("gnome-terminal-server") == "terminal"
    assert classify("code") == "code"
    assert classify("Google Chrome", "(3) WhatsApp - Google Chrome") == "chat"
    assert classify("Google Chrome", "Claude - Google Chrome") == "ai"
    assert classify("Google Chrome", "Dashboard - Google Chrome") == "browser"
    assert classify("") == "unknown"


def test_choose_method():
    assert choose_method("auto", AppInfo("ptyxis", kind="terminal"), {}) == "ctrl+shift+v"
    assert choose_method("auto", AppInfo("code", kind="code"), {}) == "shift+insert"
    assert choose_method("auto", AppInfo(), {}) == "shift+insert"
    assert choose_method("auto", AppInfo("Google Chrome", kind="chat"), {}) == "ctrl+v"
    assert choose_method("auto", AppInfo("ptyxis", kind="terminal"), {"ptyxis": "shift+insert"}) == "shift+insert"
    assert choose_method("ctrl+v", AppInfo("ptyxis", kind="terminal"), {}) == "ctrl+v"


def test_gb_layout_symbols():
    gb, us = _char_map("gb"), _char_map("us")
    assert gb['"'] != us['"'] and gb["@"] != us["@"]
    assert us["A"][1] is True and us["a"][1] is False


def _tone(seconds, amp=0.3, f=220.0):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def _silence(seconds):
    rng = np.random.default_rng(0)
    return (0.001 * rng.standard_normal(int(seconds * SAMPLE_RATE))).astype(np.float32)


def test_segmenter_cuts_at_pause_and_flushes_final():
    segs = []
    s = Segmenter(min_segment=2.0, on_segment=segs.append)
    audio = np.concatenate([_silence(0.3), _tone(2.5), _silence(1.0), _tone(1.5), _silence(0.2)])
    for i in range(0, len(audio), FRAME * 3):
        s.feed(audio[i:i + FRAME * 3])
    assert len(segs) == 1 and not segs[0].final
    assert 2.2 < segs[0].speech_seconds < 2.8
    s.flush()
    assert len(segs) == 2 and segs[1].final and segs[1].index == 1
    assert 1.2 < segs[1].speech_seconds < 1.8
    # silence trimmed around speech (pad 0.25s each side)
    assert len(segs[1].audio) / SAMPLE_RATE < 1.5 + 0.6


def test_segmenter_no_speech():
    segs = []
    s = Segmenter(on_segment=segs.append)
    s.feed(_silence(2.0))
    s.flush()
    assert len(segs) == 1 and segs[0].speech_seconds < 0.2


def test_segmenter_forced_cut_without_pause():
    segs = []
    s = Segmenter(min_segment=1.0, max_segment=3.0, on_segment=segs.append)
    s.feed(_tone(7.0))
    assert len(segs) == 2
    assert all(len(sg.audio) / SAMPLE_RATE >= 1.0 for sg in segs)  # no slivers



def test_segmenter_cuts_long_talk_without_pauses():
    # someone talking for 50 s without a clear pause: what's left after release stays short
    segs = []
    s = Segmenter(on_segment=segs.append)
    s.feed(_tone(50.0))
    s.flush()
    lengths = [len(sg.audio) / SAMPLE_RATE for sg in segs]
    assert len(segs) == 3 and max(lengths) <= 20.5
    assert abs(sum(lengths) - 50.0) < 0.2  # nothing lost or doubled


def test_segmenter_forced_cut_lands_in_a_gap():
    # 25 s of talk with a short (0.2 s) breath at 18.6 s: too short to cut on, but the forced
    # cut at 20 s goes there rather than into a word
    audio = np.concatenate([_tone(18.6), _silence(0.2), _tone(6.2)])
    segs = []
    s = Segmenter(on_segment=segs.append)
    for i in range(0, len(audio), FRAME):
        s.feed(audio[i:i + FRAME])
    assert len(segs) == 1 and abs(len(segs[0].audio) / SAMPLE_RATE - 18.7) < 0.3

def test_config_set_value_preserves_comments(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    p = cfgmod.ensure_config()
    cfgmod.set_value("hotkey.key", "CTRL+SPACE")
    cfgmod.set_value("paste.apps.kitty", "shift+insert")
    cfgmod.set_value("text.vocabulary", ["Eneojo", "Lagos"])
    text = p.read_text()
    assert '# Any key or combination' in text
    cfg = cfgmod.load_config()
    assert cfg["hotkey"]["key"] == "CTRL+SPACE"
    assert cfg["paste"]["apps"]["kitty"] == "shift+insert"
    assert cfg["text"]["vocabulary"] == ["Eneojo", "Lagos"]
    assert cfg["hotkey"]["mode"] == "hold"  # untouched defaults survive


def test_secrets_file_is_private(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    p = cfgmod.save_secret("groq", "gsk_test")
    assert oct(p.stat().st_mode & 0o777) == "0o600"
    assert cfgmod.load_secrets()["groq"] == "gsk_test"


def test_parse_cli_value():
    assert cfgmod.parse_cli_value("off", True) is False
    assert cfgmod.parse_cli_value("12", 3) == 12
    assert cfgmod.parse_cli_value("a, b", []) == ["a", "b"]


# --------------------------------------------------------------------------- cloud transcription

def test_cloud_timeout_scales_with_audio(monkeypatch):
    import mouthpiece.engines as eng

    seen = []
    monkeypatch.setattr(eng, "http_json", lambda url, data, headers, timeout: seen.append(timeout) or {"text": "hi"})
    e = eng.CloudEngine("groq", "https://example.invalid/v1", "key", "whisper-large-v3")
    e.transcribe(np.zeros(4 * SAMPLE_RATE, np.float32))
    e.transcribe(np.zeros(120 * SAMPLE_RATE, np.float32))
    assert seen == [10.0, 30.0]


def test_offline_cloud_falls_back_once_then_skips_ahead(monkeypatch):
    import types

    from mouthpiece import daemon as dm
    from mouthpiece.engines import CloudEngine, EngineError

    class DeadCloud(CloudEngine):
        calls = 0

        def transcribe(self, audio, prompt="", language=""):
            DeadCloud.calls += 1
            raise EngineError("network error: timed out")

    class Local:
        label = "local parakeet-v3"

        def transcribe(self, audio, prompt="", language=""):
            return "hello from the local model"

    fake = types.SimpleNamespace(cfg={"transcription": {"fallback_to_local": True, "language": "en"}},
                                 engine=DeadCloud("groq", "https://example.invalid", "k", "m"),
                                 fallback=Local(), _cloud_down_until=0.0, _prompt_for=lambda s: "")
    fake._run_fallback = lambda audio, err: dm.Daemon._run_fallback(fake, audio, err)
    audio = np.zeros(SAMPLE_RATE, np.float32)
    for _ in range(3):
        text, err, used = dm.Daemon._run_engine(fake, None, audio)
        assert text == "hello from the local model" and err is None and used.endswith("(fallback)")
    assert DeadCloud.calls == 1  # the second and third chunks didn't wait for the cloud again
    fake._cloud_down_until = 0.0  # two minutes later: the cloud is tried again
    dm.Daemon._run_engine(fake, None, audio)
    assert DeadCloud.calls == 2
