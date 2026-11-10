"""Deterministic transcript clean-up (always on, instant, offline)."""

from __future__ import annotations

import re
from functools import lru_cache

_FILLER = r"(?:u+m+|u+h+m*|e+r+m+|h+m+|m+h+m+)"
_FILLER_RE = re.compile(
    rf"(?i)(?P<pre>^|(?<=[\s,.;:!?\"'(]))(?P<f>{_FILLER})(?P<post>[,.]?)(?=\s|$|[,.;:!?\"')])"
)


def remove_fillers(text: str) -> str:
    starts_upper = text[:1].isupper()

    def repl(m: re.Match) -> str:
        # keep a sentence-ending period that followed the filler ("... the thing, um." -> "... the thing.")
        return m.group("post") if m.group("post") == "." else ""

    out = _FILLER_RE.sub(repl, text)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r"([,;:])\s*([,.;:!?])", r"\2", out)   # ", ." -> "."
    out = re.sub(r"^[\s,;:.]+", "", out)
    out = re.sub(r"[ \t]{2,}", " ", out).strip()
    if starts_upper and out[:1].islower():
        out = out[0].upper() + out[1:]
    return out


# Short words people stutter on. "had had", "that that" and "bye bye" can be correct, so only
# these are collapsed ("to to" -> "to", "I I think" -> "I think").
_REPEAT_SAFE = frozenset(
    "i a an the to and of in on at for with from we you he she it they me my your our is are was be "
    "can will just this but or if what when how".split()
)
_REPEAT_RE = re.compile(r"(?i)\b([a-z]+)(?:[ \t]*,?[ \t]+\1\b)+")


def collapse_repeats(text: str) -> str:
    return _REPEAT_RE.sub(lambda m: m.group(1) if m.group(1).lower() in _REPEAT_SAFE else m.group(0), text)


def _term_pattern(term: str) -> str:
    # "WhatsApp" -> whats[\s-]?app ; "Claude Code" -> claude[\s-]?code ; "Next.js" -> next\.?\s?js
    parts = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+|[^\sA-Za-z\d]", term)
    pieces = []
    for p in parts:
        if re.fullmatch(r"[^\sA-Za-z\d]", p):
            pieces.append(re.escape(p) + "?")
        else:
            pieces.append(re.escape(p))
    return r"[\s-]?".join(pieces)


@lru_cache(maxsize=256)
def _vocab_regex(term: str) -> re.Pattern:
    return re.compile(rf"(?i)(?<![\w]){_term_pattern(term)}(?![\w])")


def apply_vocabulary(text: str, vocabulary: list[str]) -> str:
    for term in sorted({t.strip() for t in vocabulary if t and t.strip()}, key=len, reverse=True):
        text = _vocab_regex(term).sub(term, text)
    return text


def apply_replacements(text: str, replacements: dict[str, str]) -> str:
    for src, dst in sorted(replacements.items(), key=lambda kv: len(kv[0]), reverse=True):
        if not src.strip():
            continue
        pat = re.compile(rf"(?i)(?<![\w]){re.escape(src.strip())}(?![\w])")
        text = pat.sub(lambda _m, d=dst: d, text)
    return text


def join_segments(parts: list[str]) -> str:
    out = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not out:
            out = p
            continue
        # a chunk boundary is a pause; if the previous chunk ended mid-sentence keep the flow,
        # and if it ended a sentence make sure the next one starts with a capital
        if out[-1] not in ".!?" and p[:1].isupper() and not re.match(r"(I\b|I'|[A-Z]{2,})", p):
            p = p[0].lower() + p[1:]
        elif out[-1] in ".!?" and p[:1].islower():
            p = p[0].upper() + p[1:]
        out += " " + p
    return out


def postprocess(text: str, cfg: dict) -> str:
    t = cfg["text"]
    text = re.sub(r"\s+", " ", text).strip()
    if t.get("remove_fillers", True):
        text = remove_fillers(text)
        text = collapse_repeats(text)
    text = apply_replacements(text, t.get("replacements") or {})
    text = apply_vocabulary(text, t.get("vocabulary") or [])
    if t.get("capitalize_first", True) and text[:1].islower():
        text = text[0].upper() + text[1:]
    return text


def finalize_spacing(text: str, append_space: bool) -> str:
    if append_space and text and not text[-1].isspace():
        return text + " "
    return text
