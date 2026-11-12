"""mouthpiece — push-to-talk dictation for Linux (GNOME/Wayland friendly)."""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import logging.handlers
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import __version__, control, history
from .config import (
    DEFAULTS,
    config_path,
    ensure_config,
    get_value,
    load_config,
    load_secrets,
    parse_cli_value,
    save_secret,
    set_value,
)

APP_DIR = Path(__file__).resolve().parent.parent
SERVICE = "mouthpiece.service"


def setup_logging(verbose: bool) -> None:
    state = Path(os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")) / "mouthpiece"
    state.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    root.addHandler(console)
    fileh = logging.handlers.RotatingFileHandler(state / "mouthpiece.log", maxBytes=2_000_000, backupCount=2)
    fileh.setFormatter(fmt)
    root.addHandler(fileh)


def daemon_running() -> bool:
    return control.ping()


def _send(cmd: str, *args: str) -> dict:
    try:
        return control.send(cmd, *args)
    except (FileNotFoundError, ConnectionRefusedError):
        print("Mouthpiece isn't running. Start it with: mouthpiece run   (or: mouthpiece install)", file=sys.stderr)
        sys.exit(2)


# --------------------------------------------------------------------------- commands


def cmd_run(a) -> None:
    setup_logging(a.verbose)
    from .daemon import Daemon

    cfg = load_config()
    if a.no_overlay:
        cfg["ui"]["overlay"] = False
    try:
        Daemon(cfg).run()
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(1)


def cmd_simple(a) -> None:
    r = _send(a.command)
    if not r.get("ok"):
        print(r.get("error", "failed"), file=sys.stderr)
        sys.exit(1)
    if a.command in ("toggle", "start", "stop", "cancel"):
        print(r.get("state", "ok"))


def cmd_status(a) -> None:
    r = _send("status")
    if a.json:
        print(json.dumps(r, indent=2))
        return
    print(f"state        {r['state']}")
    print(f"hotkey       {r['hotkey']}  ({r['mode']} mode)")
    ready = "ready" if r["engine_ready"] else "loading…"
    print(f"engine       {r['engine']}  [{ready}]" + (f"  error: {r['engine_error']}" if r.get("engine_error") else ""))
    print(f"keyboards    {', '.join(r['keyboards']) or 'NONE (check input group)'}")
    print(f"paste        {r['paste']}  via {r['clipboard_backend']} clipboard, virtual keyboard "
          f"{'ok' if r['virtual_keyboard'] else 'MISSING: ' + str(r.get('injector_error'))}")
    if "formatting" in r:
        print(f"formatting   {r['formatting']}")
    if r.get("last_app", {}).get("app"):
        la = r["last_app"]
        print(f"last app     {la['app']} ({la['kind']})")
    last = r.get("last") or {}
    if last:
        ok = "ok" if last.get("ok") else f"failed: {last.get('error')}"
        print(f"last result  {ok}, {last.get('method')}, {last.get('engine')}, {last.get('latency_ms')} ms after release")
        if last.get("formatted"):
            print(f"             formatted in {last.get('format_ms')} ms")
        elif last.get("format_error"):
            print(f"             not formatted: {last['format_error'][:100]}")
        if last.get("text"):
            print(f"             “{last['text'][:120]}”")


def cmd_set_hotkey(a) -> None:
    from .hotkey import Hotkey, HotkeyError, capture_hotkey

    if a.spec:
        try:
            hk = Hotkey.parse(a.spec)
        except HotkeyError as e:
            sys.exit(str(e))
    else:
        print("Press the key (or key combination) you want to hold for dictation, then let go…")
        try:
            codes = capture_hotkey()
        except (PermissionError, TimeoutError) as e:
            sys.exit(str(e))
        hk = Hotkey.from_codes(codes)
    set_value("hotkey.key", hk.spec)
    print(f"Hotkey set to: {hk.pretty()}   ({hk.spec})")
    if not hk.modifier_only:
        print("Note: the key also reaches the focused app. Modifier-only keys (like Right Ctrl) are the most "
              "invisible choice.")
    if daemon_running():
        _send("reload")
        print("Mouthpiece reloaded.")


def cmd_config(a) -> None:
    if a.action == "path":
        print(ensure_config())
    elif a.action == "show":
        print(ensure_config().read_text())
    elif a.action == "edit":
        p = ensure_config()
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
        if editor and sys.stdin.isatty():
            subprocess.call([editor, str(p)])
        else:
            subprocess.call(["gio", "open", str(p)])
    elif a.action == "get":
        print(json.dumps(get_value(load_config(), a.key), ensure_ascii=False))
    elif a.action == "set":
        try:
            current = get_value(DEFAULTS, a.key)
        except KeyError:
            current = ""
        value = parse_cli_value(a.value, current)
        set_value(a.key, value)
        print(f"{a.key} = {json.dumps(value, ensure_ascii=False)}")
        if daemon_running():
            _send("reload")
            print("Mouthpiece reloaded.")


def _check_key(provider: str, key: str) -> str:
    import urllib.request

    base = {"groq": "https://api.groq.com/openai/v1", "openai": "https://api.openai.com/v1"}.get(provider)
    if not base:
        return "saved (not verified)"
    req = urllib.request.Request(f"{base}/models", headers={"Authorization": f"Bearer {key}", "User-Agent": "mouthpiece"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            ids = sorted(m["id"] for m in json.load(r).get("data", []))
        stt = [i for i in ids if "whisper" in i or "transcribe" in i]
        return "verified ✓  speech models: " + (", ".join(stt) or "none listed")
    except Exception as e:
        return f"saved, but verification failed: {e}"


def cmd_set_key(a) -> None:
    key = a.key or getpass.getpass(f"Paste your {a.provider} API key (input hidden): ").strip()
    p = save_secret(a.provider, key)
    if not key:
        print(f"Removed {a.provider} key from {p}")
        return
    print(f"Stored in {p} (permissions 600) — {_check_key(a.provider, key)}")
    if daemon_running():
        _send("reload")
        print("Mouthpiece reloaded.")


def cmd_devices(a) -> None:
    from .linux_input import describe_devices

    print("Keyboards / input devices:")
    for d in describe_devices():
        if "error" in d:
            print(f"  {d['path']}: cannot open ({d['error']})")
        elif d["keyboard"] or d["keys"]:
            print(f"  {d['path']}: {d['name']}{'  [keyboard]' if d['keyboard'] else ''}")
    try:
        import sounddevice as sd

        print("\nMicrophones (use the name or number for audio.input_device):")
        default = sd.default.device[0]
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                mark = "*" if i == default else " "
                print(f" {mark}{i:3d}  {dev['name']}  ({sd.query_hostapis(dev['hostapi'])['name']})")
    except Exception as e:
        print(f"\nMicrophones: unavailable ({e})")


def cmd_download_model(a) -> None:
    setup_logging(False)
    from .engines import LOCAL_MODELS, LocalEngine

    names = list(LOCAL_MODELS) if a.model == "all" else [a.model]
    for n in names:
        e = LocalEngine(n)
        print(f"{n}: {'already downloaded, loading' if e.is_downloaded() else 'downloading (~670 MB)'}…")
        e.load()
        print(f"{n}: ready ({e.load_seconds:.1f}s to load) at {e.model_path()}")


def cmd_transcribe(a) -> None:
    from .audio import load_audio_file, normalize_for_asr
    from .engines import build_engine
    from .textproc import postprocess

    logging.basicConfig(level=logging.WARNING)
    cfg = load_config()
    eng = build_engine(cfg, load_secrets(), a.engine)
    audio = normalize_for_asr(load_audio_file(a.file))
    t = time.monotonic()
    eng.load()
    t_load = time.monotonic() - t
    t = time.monotonic()
    text = eng.transcribe(audio, language=cfg["transcription"].get("language", ""))
    dt = time.monotonic() - t
    if not a.raw:
        text = postprocess(text, cfg)
    print(f"[{eng.label}: {len(audio) / 16000:.1f}s audio in {dt:.2f}s, load {t_load:.1f}s]", file=sys.stderr)
    if a.format and not a.raw:
        text = _format_or_exit(text, cfg, a.dest)
    print(text)


DESTINATIONS = ("chat", "terminal", "ai", "code", "email", "other")


def _destination_app(dest: str):
    from .context import AppInfo

    return {
        "chat": AppInfo(app="Google Chrome", title="WhatsApp", kind="chat"),
        "terminal": AppInfo(app="ptyxis", kind="terminal"),
        "ai": AppInfo(app="Google Chrome", title="Claude", kind="ai"),
        "code": AppInfo(app="code", kind="code"),
        "email": AppInfo(app="Google Chrome", title="Inbox - Gmail", kind="chat"),
    }.get(dest, AppInfo())


def _format_or_exit(text: str, cfg: dict, dest: str) -> str:
    from .formatting import FormatError, Formatter

    f = Formatter({**cfg, "formatting": {**cfg["formatting"], "enabled": True}}, load_secrets())
    if not f.active:
        sys.exit(f"AI formatting unavailable: {f.problem}")
    t = time.monotonic()
    try:
        out = f.format(text, _destination_app(dest))
    except FormatError as e:
        sys.exit(f"AI formatting failed: {e}")
    print(f"[formatted by {f.last_used} in {time.monotonic() - t:.1f}s]", file=sys.stderr)
    return out


def cmd_format(a) -> None:
    from .textproc import postprocess

    cfg = load_config()
    text = " ".join(a.text) if a.text else sys.stdin.read()
    if not text.strip():
        sys.exit("nothing to format")
    print(_format_or_exit(postprocess(text, cfg), cfg, a.dest))


def cmd_test_paste(a) -> None:
    from .context import probe_active_app
    from .injector import Injector

    cfg = load_config()
    if a.method:
        cfg["paste"]["method"] = a.method
    inj = Injector(cfg)
    if inj.error:
        sys.exit(inj.error)
    for i in range(a.delay, 0, -1):
        print(f"Click into the app you want to test… pasting in {i}", end="\r", flush=True)
        time.sleep(1)
    app = probe_active_app()
    r = inj.deliver(a.text + " ", app)
    time.sleep(1.2)  # let the clipboard restore run
    inj.close()
    print(f"\nTarget: {app.describe()}  method: {r['method']}  -> {'sent' if r['ok'] else r.get('error')}")


def cmd_history(a) -> None:
    for e in history.recent(a.n):
        print(f"{e.get('ts', '')}  [{e.get('app') or '?'}]  {e.get('text', '')}")


def cmd_doctor(a) -> None:
    import grp

    ok = lambda b: "✓" if b else "✗"  # noqa: E731
    print(f"Mouthpiece {__version__}\n")
    groups = {grp.getgrgid(g).gr_name for g in os.getgroups()}
    print(f" {ok('input' in groups)} user in 'input' group (read keyboards)")
    from .linux_input import describe_devices

    devs = describe_devices()
    kbds = [d for d in devs if d.get("keyboard")]
    print(f" {ok(bool(kbds))} keyboards readable: {', '.join(d['name'] for d in kbds) or 'none'}")
    print(f" {ok(os.access('/dev/uinput', os.W_OK))} /dev/uinput writable (virtual keyboard for pasting)")
    from .clipboard import Clipboard

    cb = Clipboard()
    print(f" {ok(cb.backend != 'none')} clipboard backend: {cb.backend}")
    session = os.environ.get("XDG_SESSION_TYPE", "?")
    print(f"   session: {session}, desktop: {os.environ.get('XDG_CURRENT_DESKTOP', '?')}")
    try:
        import sounddevice as sd

        dev = sd.query_devices(kind="input")
        print(f" ✓ microphone: {dev['name']}")
    except Exception as e:
        print(f" ✗ microphone: {e}")
    from .context import probe_active_app

    t = time.monotonic()
    app = probe_active_app()
    print(f" {ok(bool(app.app))} focused-app detection (AT-SPI): {app.describe()} in {1000 * (time.monotonic() - t):.0f} ms")
    cfg = load_config()
    secrets = load_secrets()
    from .engines import LocalEngine

    le = LocalEngine(cfg["transcription"]["local_model"])
    print(f" {ok(le.is_downloaded())} local model {le.model_key}: {'downloaded' if le.is_downloaded() else 'missing — mouthpiece download-model'}")
    from .formatting import Formatter

    fm = Formatter(cfg, secrets)
    if not fm.enabled:
        print("   AI formatting: off (formatting.enabled = false)")
    elif fm.provider == "none":
        print(f" ✗ AI formatting: {fm.problem}")
    elif fm.provider == "claude-code":
        signed_in = _claude_signed_in(fm.claude)
        state = {True: "signed in", False: "NOT signed in (run `claude` and use /login)", None: "sign-in unknown"}[signed_in]
        print(f" {ok(signed_in is not False)} AI formatting: {fm.describe()} via {fm.claude} ({state})")
    else:
        print(f" ✓ AI formatting: {fm.describe()}")
    for prov in ("groq", "openai"):
        print(f"   {prov} API key: {'set' if secrets.get(prov) else 'not set'}")
    print(f" {ok(daemon_running())} daemon running")
    print(f"\nConfig: {config_path()}")


def _claude_signed_in(claude: str) -> bool | None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE_CODE_")}
    try:
        p = subprocess.run([claude, "auth", "status"], capture_output=True, text=True, timeout=20, env=env)
        return bool(json.loads(p.stdout).get("loggedIn"))
    except (OSError, subprocess.TimeoutExpired, ValueError, AttributeError):
        return None


# --------------------------------------------------------------------------- install


def _unit_text() -> str:
    py = sys.executable
    return f"""[Unit]
Description=Mouthpiece push-to-talk dictation
Documentation=file://{APP_DIR}/README.md
PartOf=graphical-session.target
After=graphical-session.target

[Service]
Type=simple
ExecStart={py} -m mouthpiece run
WorkingDirectory={APP_DIR}
Restart=on-failure
RestartSec=3
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=graphical-session.target
"""


def cmd_install(a) -> None:
    home = Path.home()
    bin_dir = home / ".local" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    launcher = bin_dir / "mouthpiece"
    target = Path(sys.executable).parent / "mouthpiece"
    if launcher.is_symlink() or not launcher.exists():
        if launcher.is_symlink():
            launcher.unlink()
        launcher.symlink_to(target)
        print(f"✓ command: {launcher} -> {target}")
    unit_dir = home / ".config" / "systemd" / "user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    (unit_dir / SERVICE).write_text(_unit_text())
    print(f"✓ service: {unit_dir / SERVICE}")
    apps = home / ".local" / "share" / "applications"
    apps.mkdir(parents=True, exist_ok=True)
    (apps / "mouthpiece-settings.desktop").write_text(
        "[Desktop Entry]\nType=Application\nName=Mouthpiece Settings\n"
        "Comment=Push-to-talk dictation: edit hotkey, engine and options\n"
        f"Exec={target} config edit\nIcon=audio-input-microphone\nCategories=Utility;Accessibility;\nTerminal=false\n"
    )
    print(f"✓ menu entry: {apps / 'mouthpiece-settings.desktop'}")
    ensure_config()
    if daemon_running() and not a.no_start:
        _send("quit")
        time.sleep(1.0)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    subprocess.run(["systemctl", "--user", "enable", SERVICE], check=False, capture_output=True)
    if not a.no_start:
        subprocess.run(["systemctl", "--user", "restart", SERVICE], check=False)
        print("✓ started (and will start automatically when you log in)")


def cmd_uninstall(a) -> None:
    subprocess.run(["systemctl", "--user", "disable", "--now", SERVICE], check=False, capture_output=True)
    for p in (Path.home() / ".config/systemd/user" / SERVICE, Path.home() / ".local/share/applications/mouthpiece-settings.desktop"):
        if p.exists():
            p.unlink()
            print(f"removed {p}")
    link = Path.home() / ".local/bin/mouthpiece"
    if link.is_symlink():
        link.unlink()
        print(f"removed {link}")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    print("Config and history were kept in ~/.config/mouthpiece and ~/.local/share/mouthpiece.")


def cmd_unstick_escape(a) -> None:
    """Run by the temporary GNOME Esc shortcut: if the daemon is gone, remove that shortcut."""
    if not daemon_running():
        from .gnome_keys import remove_leftover

        remove_leftover()


def cmd_logs(a) -> None:
    if shutil.which("journalctl"):
        os.execvp("journalctl", ["journalctl", "--user", "-u", SERVICE, "-n", "60"] + (["-f"] if a.follow else []))


# --------------------------------------------------------------------------- parser


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="mouthpiece", description=__doc__)
    p.add_argument("--version", action="version", version=f"mouthpiece {__version__}")
    sub = p.add_subparsers(dest="command")

    r = sub.add_parser("run", help="start the dictation daemon (foreground)")
    r.add_argument("-v", "--verbose", action="store_true")
    r.add_argument("--no-overlay", action="store_true")
    r.set_defaults(fn=cmd_run)

    for name, hlp in (("toggle", "start/stop recording"), ("start", "start recording (hands-free)"),
                      ("stop", "stop recording and paste"), ("cancel", "discard the current recording"),
                      ("paste-last", "paste the previous transcript again"), ("reload", "re-read the config"),
                      ("quit", "stop the daemon")):
        sp = sub.add_parser(name, help=hlp)
        sp.set_defaults(fn=cmd_simple)

    s = sub.add_parser("status", help="show what the daemon is doing")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_status)

    h = sub.add_parser("set-hotkey", help="choose the dictation hotkey (press it, or give e.g. RIGHTCTRL)")
    h.add_argument("spec", nargs="?")
    h.set_defaults(fn=cmd_set_hotkey)

    c = sub.add_parser("config", help="show/edit settings")
    csub = c.add_subparsers(dest="action", required=True)
    csub.add_parser("path")
    csub.add_parser("show")
    csub.add_parser("edit")
    g = csub.add_parser("get")
    g.add_argument("key")
    st = csub.add_parser("set")
    st.add_argument("key")
    st.add_argument("value")
    c.set_defaults(fn=cmd_config)

    k = sub.add_parser("set-key", help="store an API key for a cloud engine (groq, openai, custom)")
    k.add_argument("provider", choices=["groq", "openai", "custom"])
    k.add_argument("key", nargs="?")
    k.set_defaults(fn=cmd_set_key)

    sub.add_parser("devices", help="list keyboards and microphones").set_defaults(fn=cmd_devices)
    sub.add_parser("doctor", help="check permissions, audio, clipboard, models").set_defaults(fn=cmd_doctor)

    d = sub.add_parser("download-model", help="download the offline model")
    d.add_argument("model", nargs="?", default="parakeet-v3", choices=["parakeet-v3", "parakeet-v2", "all"])
    d.set_defaults(fn=cmd_download_model)

    t = sub.add_parser("transcribe", help="transcribe an audio file")
    t.add_argument("file")
    t.add_argument("--engine", choices=["auto", "local", "groq", "openai", "custom"])
    t.add_argument("--raw", action="store_true", help="skip text post-processing")
    t.add_argument("--format", action="store_true", help="also apply AI formatting")
    t.add_argument("--for", dest="dest", choices=DESTINATIONS, default="chat", help="where the text is going")
    t.set_defaults(fn=cmd_transcribe)

    fo = sub.add_parser("format", help="AI-format some text (argument or stdin) the way dictation would")
    fo.add_argument("text", nargs="*")
    fo.add_argument("--for", dest="dest", choices=DESTINATIONS, default="chat", help="where the text is going")
    fo.set_defaults(fn=cmd_format)

    tp = sub.add_parser("test-paste", help="paste a test sentence into whatever app you click into")
    tp.add_argument("--delay", type=int, default=5)
    tp.add_argument("--method", choices=["auto", "ctrl+v", "ctrl+shift+v", "shift+insert", "type"])
    tp.add_argument("--text", default="Hello from Mouthpiece — paste test ✓")
    tp.set_defaults(fn=cmd_test_paste)

    hi = sub.add_parser("history", help="show recent transcripts")
    hi.add_argument("-n", type=int, default=10)
    hi.set_defaults(fn=cmd_history)

    ins = sub.add_parser("install", help="install the command, autostart service and menu entry")
    ins.add_argument("--no-start", action="store_true")
    ins.set_defaults(fn=cmd_install)
    sub.add_parser("uninstall", help="remove the service/command (keeps config)").set_defaults(fn=cmd_uninstall)

    sub.add_parser("unstick-escape", help=argparse.SUPPRESS).set_defaults(fn=cmd_unstick_escape)

    lg = sub.add_parser("logs", help="show daemon logs")
    lg.add_argument("-f", "--follow", action="store_true")
    lg.set_defaults(fn=cmd_logs)

    a = p.parse_args(argv)
    if not a.command:
        p.print_help()
        return
    a.fn(a)


if __name__ == "__main__":
    main()
