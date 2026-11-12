# Mouthpiece

Push-to-talk dictation for Linux. **Hold a hotkey, speak, let go — the text is pasted into
whatever app you're in**: WhatsApp, Facebook, Claude Code / Codex in a terminal, VS Code,
Claude Desktop, your browser, anything with a text cursor.

Built for GNOME on Wayland (where most dictation tools struggle), and works on X11 and other
desktops too. Speech recognition runs offline on your computer by default (Groq or OpenAI
Whisper are optional), and AI formatting turns what you said into clean, punctuated text.

## Install

Mouthpiece is developed on Ubuntu with GNOME and needs Python 3.11 or newer. On other
distributions, install the equivalent packages.

1. System packages:

   ```bash
   sudo apt install git python3-venv python3-gi python3-cairo gir1.2-atspi-2.0 gir1.2-gtk-3.0 xclip libportaudio2
   ```

2. Let your user read the keyboard (for the hotkey) and create a virtual keyboard (for pasting),
   then restart your computer:

   ```bash
   sudo usermod -aG input "$USER"
   echo 'KERNEL=="uinput", GROUP="input", MODE="0660", OPTIONS+="static_node=uinput"' \
     | sudo tee /etc/udev/rules.d/70-mouthpiece-uinput.rules
   ```

   Being in the `input` group lets programs you run read the keyboard directly. That is how the
   hotkey works in every app without a desktop shortcut.

3. Mouthpiece itself:

   ```bash
   git clone https://github.com/eso8484/mouthpiece.git ~/mouthpiece
   cd ~/mouthpiece
   python3 -m venv --system-site-packages .venv
   .venv/bin/pip install -e .
   .venv/bin/mouthpiece download-model   # offline speech model, about 670 MB
   .venv/bin/mouthpiece doctor           # checks keyboard, pasting, microphone and model
   ```

4. Optional, for AI formatting: install [Claude Code](https://claude.com/claude-code) and sign in
   once by running `claude`. Without it, formatting can use a Groq or OpenAI key instead (see
   [AI formatting](#ai-formatting)).

## Quick start

```bash
cd ~/mouthpiece
.venv/bin/mouthpiece run          # start it (Ctrl+C to stop)
```

Then click into any text box, **hold Right Ctrl, talk, release**. A small pill at the bottom
of the screen shows *Listening → Transcribing → Formatting → Pasted*.

To have it start automatically at login and put `mouthpiece` on your PATH:

```bash
.venv/bin/mouthpiece install      # systemd user service + ~/.local/bin/mouthpiece
```

`mouthpiece uninstall` removes it again (your settings and history are kept).

## Everyday use

| Action | How |
|---|---|
| Dictate | Hold **Right Ctrl**, speak, release |
| Cancel while dictating | **Esc** (on GNOME the Esc is swallowed, so it won't close your WhatsApp chat or interrupt Claude Code) |
| Re-paste the last text | `mouthpiece paste-last` |
| Change the hotkey | `mouthpiece set-hotkey` then press the key/combination you want |
| Toggle instead of hold | `mouthpiece config set hotkey.mode toggle` (or `hybrid`: hold = push-to-talk, quick tap = hands-free) |
| What's it doing? | `mouthpiece status` |

Any key or combination works as the hotkey: `RIGHTCTRL` (default), `CTRL+SPACE`,
`SUPER+ALT`, `F9`, `SHIFT+F7`, `MOUSE_SIDE` (mouse thumb button)…
Modifier-only keys like Right Ctrl are the most invisible choice — other keys also reach the
app you're typing in. If you press another key within 400 ms of the hotkey (e.g. Right Ctrl+C),
Mouthpiece assumes you were typing a shortcut and silently cancels.

## AI formatting

Speech recognition writes down exactly what you said, the way you said it: run-on sentences,
missing commas, false starts, repeated words. Mouthpiece then has an AI rewrite it the way you
would have typed it:

- **Punctuation:** commas, full stops, question marks, apostrophes, quotes.
- **Sentences and paragraphs:** run-on speech is split into sentences, and longer messages into paragraphs.
- **Cleaned speech:** fillers ("um", filler "you know"), stutters ("to to") and false starts are
  removed. Self-corrections are applied: "if you have you are building a project" → "if you're
  building a project", and "Tuesday, sorry, Monday" → "Monday".
- **Your meaning and voice stay:** every point and detail is kept, along with your expressions
  ("How far?"). Nothing is summarised or added.
- **The style fits the app:** a chat message in WhatsApp, a prompt in Claude Code/Codex, an email
  in Gmail. Commands, file names and code are left exactly as you said them.

It is on by default. If [Claude Code](https://claude.com/claude-code) is installed and signed in,
it uses **Claude through your Claude Code login**, so no API key is needed; otherwise it uses
Groq or OpenAI once you add a key (`mouthpiece set-key groq`). Mouthpiece starts Claude while you
are still talking, so after you release the key only the formatting itself remains:

| Dictation | Formatting time (Claude Opus, low effort) |
|---|---|
| a sentence or two | about 1.5–2 s |
| a minute of speech | about 3.5–4.5 s |

If formatting fails or takes too long, the plain transcript is pasted instead and the pill says
*Pasted (not formatted)*. The same happens when the AI's result doesn't look like your text, for
example an answer to a question you dictated. Your original wording is kept in the history.

Formatting through Claude Code shares your Claude plan's usage limits with everything else you do
in Claude. When the limit is reached, or Claude Code is signed out, the pill says so, for example
*Pasted unformatted: Claude usage limit, resets 8:15am*. Mouthpiece then stops asking Claude until
the reset time, so dictation isn't slowed down by calls that can't succeed. If you have also added a
Groq or OpenAI key (`mouthpiece set-key groq`; Groq has a free tier), that key formats in the
meantime, so your text is still punctuated and cleaned up. Groq (`openai/gpt-oss-120b`) formats
in about a second, a little less faithfully than Claude: in tests it sometimes left a misheard
word or dropped a small Pidgin word that Claude kept.

| Setting | Meaning |
|---|---|
| `formatting.enabled` | `false` = paste exactly what was recognised |
| `formatting.provider` | `auto` (Claude Code if installed, else Groq/OpenAI if you've added a key), `claude-code`, `groq`, `openai`, `custom` |
| `formatting.model` | `""` = default (`opus` for Claude Code); `sonnet` or `haiku` also work |
| `formatting.effort` | Claude's thinking effort; `low` is fastest |
| `formatting.timeout_s` | paste the plain transcript if formatting takes longer (plus 1 s per 40 words) |
| `text.languages` | the languages and dialects you speak, e.g. `["English", "Nigerian Pidgin"]`; each is kept as you spoke it |

Try it without speaking: `mouthpiece format --for chat "so um i was thinking we we could meet
tomorrow at at five"`. Formatting sends the transcribed text (not your audio) to Anthropic, or
to Groq/OpenAI if you use those. Through Claude Code, each dictation uses a little of your
Claude plan.

### Nigerian Pidgin

The offline speech models only know standard English, so Pidgin comes out as English
sound-alikes: "what in you they do" for "wetin you dey do". Tell Mouthpiece you speak Pidgin:

```bash
mouthpiece config set text.languages '["English", "Nigerian Pidgin"]'
```

Formatting then keeps Pidgin as Pidgin (it never "corrects" it into English), spells it the way
people type it in chats, and puts back the Pidgin words the recogniser misheard. English stays
English, and you can mix both in one message. "my phone they ring since morning but I no fit
pick am" becomes "My phone dey ring since morning, but I no fit pick am."

Measured on 40 Pidgin recordings from 5 speakers ([Nigerian Pidgin ASR dataset](https://huggingface.co/datasets/asr-nigerian-pidgin/nigerian-pidgin-1.0), CC BY 4.0):

| | Words right | Pidgin grammar words kept (dey, don, na, wey…) |
|---|---|---|
| Without the setting | 48.5% | 18% |
| With the setting | 56.2% | 42% |
| With the setting and the English-only model | 63.5% | 51% |

The English-only model (`mouthpiece config set transcription.local_model parakeet-v2`) hears
Pidgin better but is slightly less accurate on accented English (see the table below). Fast or
mumbled speech can still come out garbled.

## Accuracy

| Engine | How to enable | Clean read speech | Nigerian-accented English (conversational) | Speed on a 2015 dual-core laptop |
|---|---|---|---|---|
| **Parakeet TDT 0.6B v3** (local, default) | nothing — offline | **97.4%** | **86.2%** (92.1% end-to-end on a 75 s mixed clip) | 0.15× real time |
| Parakeet TDT 0.6B v2 (local, English) | `config set transcription.local_model parakeet-v2` | 97.0% | 84.3% | 0.17× real time |
| Whisper large-v3 on **Groq** (cloud) | `mouthpiece set-key groq` | — | same as Parakeet v3 (see below) | ~1.5 s per chunk, however busy the laptop is |
| OpenAI gpt-4o-transcribe (cloud) | `mouthpiece set-key openai` + `config set transcription.engine openai` | — | — | ~1 s per request |

(Word accuracy measured on LibriSpeech and on the EdAcc corpus's Nigerian English speakers.
EdAcc is unscripted two-person conversation, much harder than dictation.)

Tips for 95%+:
- **Raise your microphone input volume** (Settings → Sound → Input). A quiet microphone is a common cause of mistakes.
- Put names and jargon you use in `text.vocabulary` (fixes spelling/capitalisation, and guides
  cloud models), and recurring mis-hearings in `[text.replacements]`.
- A Groq key gives you Whisper large-v3 (free tier: https://console.groq.com/keys). On 25
  Nigerian English clips from EdAcc it was as accurate as the local model (65.2% vs 64.5% of
  words right, scored the same way), on 50 Nigerian Pidgin clips clearly better (54.0% vs 43.6%,
  before formatting), and it isn't slowed down by a busy laptop. Your audio is then sent to Groq.
  If the connection drops, the local model takes over (it's kept loaded for that) and the cloud is
  tried again two minutes later.
- AI formatting (on by default, see above) also fixes words the recogniser clearly misheard when
  the context makes the right word obvious ("let's get the J" → "let's get the gist").

Long dictation is fast too: while you talk, Mouthpiece cuts the audio at natural pauses and
transcribes those chunks in the background, so after you release the key only the last few
seconds remain (≈1–1.8 s wait measured for 25–75 s of speech on the same laptop). If you talk
without pausing, a chunk is cut at the quietest moment after 20 seconds anyway, so the wait after
release stays short however long you talk.

## How pasting works (and why it works in terminals)

1. When you press the hotkey, Mouthpiece asks the accessibility bus (AT-SPI) which app has
   focus — e.g. `ptyxis` (terminal), `code` (VS Code), `Google Chrome` + window title.
2. The text goes onto the clipboard **and** the primary selection through XWayland (`xclip`).
   GNOME has no clipboard protocol for background apps, and `wl-copy` would briefly steal focus.
3. A virtual keyboard (`/dev/uinput`) presses the right paste keys for that app:

| App | Keys |
|---|---|
| Terminals (Ptyxis, GNOME Terminal, Kitty, Alacritty, WezTerm…) — Claude Code, Codex | Ctrl+Shift+V |
| VS Code / Cursor / Zed (editor *and* integrated terminal) | Shift+Insert |
| Unknown app | Shift+Insert (works in both terminals and GUI apps) |
| Everything else (Chrome, WhatsApp, Facebook, Slack, Claude Desktop…) | Ctrl+V |

4. Your previous clipboard is restored about a second later.

Before pressing the paste keys, Mouthpiece waits until you've physically let go of all modifier
keys (it reads your keyboard directly), so the paste never turns into Ctrl+Shift+Insert or similar.

Override per app: `mouthpiece config set paste.apps.kitty shift+insert`, or force one method
for everything: `paste.method = "ctrl+v" | "ctrl+shift+v" | "shift+insert" | "type" | "copy"`.

## Files

- `~/.config/mouthpiece/config.toml` — all settings (commented). `mouthpiece config edit`
- `~/.config/mouthpiece/secrets.toml` — API keys (mode 600). `mouthpiece set-key groq`
- `~/.local/share/mouthpiece/models/` — offline models (~670 MB each)
- `~/.local/share/mouthpiece/history.jsonl` — transcript history, with the unformatted wording
  as `raw`, or why a text wasn't formatted as `unformatted` (`history.enabled = false` to stop)
- `~/.local/state/mouthpiece/mouthpiece.log` — log (`mouthpiece logs -f` when installed as a service)

## Troubleshooting

- `mouthpiece doctor` checks keyboard access (`input` group), `/dev/uinput`, clipboard,
  microphone, focused-app detection and the model.
- Nothing pastes, but `mouthpiece status` shows the text → the app may not accept that key
  combination; try `mouthpiece test-paste --method shift+insert` (click into the app within 5 s).
- "No speech heard" → wrong/quiet microphone: `mouthpiece devices`, then
  `mouthpiece config set audio.input_device "<name or number>"`.
- Prefer a GNOME shortcut instead of the evdev hotkey? Bind a custom shortcut to `mouthpiece toggle`.
- Text isn't formatted → `mouthpiece doctor` shows whether Claude Code was found and is signed
  in, and `mouthpiece status` shows why the last dictation wasn't formatted. After signing in
  again or switching Claude accounts, `mouthpiece reload` makes it try Claude again straight away.
- The machine is busy (a build, many browser tabs, another AI tool working) → the offline model
  shares the CPU and can get several times slower. A Groq key moves transcription to the cloud.

## Development

```bash
.venv/bin/pip install -e '.[dev]'   # adds pytest
.venv/bin/python -m pytest          # unit tests
.venv/bin/mouthpiece run -v         # debug logging
.venv/bin/mouthpiece transcribe some.wav --engine local
```

Layout:
- `hotkey.py`: evdev listener and state machine
- `audio.py`: microphone and pause segmenter
- `engines.py`: Parakeet and cloud Whisper
- `injector.py` and `clipboard.py`: paste
- `context.py`: AT-SPI focused-app detection
- `formatting.py`: AI formatting (Claude Code / OpenAI-compatible) and its safety checks
- `overlay.py`: status pill
- `gnome_keys.py`: Esc swallowing
- `daemon.py`: orchestration

## License

MIT. See [LICENSE](LICENSE).
