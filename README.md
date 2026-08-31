# Local TTS

Turn written scripts into spoken voice-over on your own Mac — no
account, no per-word billing, and no text or audio leaving the machine.

**[Install](#install)** and **[Run](#run)** are the two steps that get
you going; everything from
**[Run it as a menu bar app](#run-it-as-a-menu-bar-app-no-terminal)**
onward is reference material.

## What it's for

One job: **narrating training content.** Course scripts are many short
lines, revised constantly, that all have to sound like one narrator.
That combination is what makes hand-recording painful and most cloud
services awkward:

- **Scripts change.** Renders are content-addressed, so an edited line
  is the only thing that re-synthesises — everything else is served
  from cache, instantly.
- **Consistency matters more than perfection.** A voice is defined once
  in `voices.yaml` (preset, language, style, pacing) and every line
  rendered through it matches. Loudness is normalised to a broadcast
  target, so clips don't jump in volume played back to back.
- **Some words are always read wrong.** Numbers, dates, phone numbers
  and acronyms are normalised before the model sees them, and anything
  still wrong gets a respelling in `pronunciations.yaml` — applied
  everywhere, once, instead of fought line by line.
- **The material is often confidential.** Nothing is uploaded; the
  model runs in the server's own process, reachable only on
  `127.0.0.1`.

It is a small appliance, not a platform: a web UI for writing a line
and hearing it, plus an HTTP API for scripting a whole module. **Not
for:** real-time or streaming speech, telephony, cloning a specific
real person's voice, or high-volume batch rendering on a shared server.
It renders a finished WAV per request, one at a time, on one machine.

## Current iteration targets Apple silicon only

**As shipped, this targets Apple Silicon Macs (M1–M4) and nothing
else** — deliberately, and deeper than a dependency:
`requirements.txt` pins [`mlx-audio`](https://github.com/Blaizzy/mlx-audio),
which runs the model on MLX, Apple's array framework; the default
checkpoint (`mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit`) is an
MLX-specific conversion other runtimes can't load; and the menu bar
launcher is a macOS `rumps`/py2app app. On this hardware everything
works with no configuration — nothing to tune, no GPU to select.

**Running it elsewhere is a supported port, not a config change.** On
NVIDIA/Linux: swap `mlx-audio` for `qwen-tts` (plus `flash-attn`), set
`TTS_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`, and point a
`Backend` subclass at the `qwen_tts` API (`load` + `generate`, ~20
lines). Everything outside that class — manifest, cache, lexicon, API,
UI — is unchanged. See
**[What survives a model swap](#what-survives-a-model-swap)**.

## Requirements

- **An Apple Silicon Mac** (M1–M4) — see above
- **Python 3.10+**
- **`ffmpeg` on PATH** (`brew install ffmpeg`) for speed control and
  loudness normalisation. Optional, but the two degrade differently:
  without it loudness normalisation is skipped silently (clip volumes
  vary between takes), while any speed other than `1.0` fails with a
  message telling you to install it.
- **~3 GB of disk** for the model checkpoint, plus ~700 MB for the
  Python environment

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No build step. **Every command in this README assumes that activated
venv** — macOS ships no bare `python`, so unactivated commands fail
with *command not found*. In a fresh terminal, activate first or prefix
with `.venv/bin/` (`.venv/bin/python engine.py check`).

Nothing is downloaded yet — the model is fetched on first use, and the
page asks before it starts.

## Run

```bash
source .venv/bin/activate
uvicorn app:app --host 127.0.0.1 --port 8123
```

Open **http://127.0.0.1:8123/** — pick a voice, paste text, Generate.
If the checkpoint isn't on the machine yet the page asks first
(`Download the voice model?`) and shows bytes-so-far with a working
Cancel; a cancelled download is kept and resumed next time. Once
downloaded, the first generation loads the model in a minute or so;
after that it's seconds, and identical re-requests are instant
(cached). Ctrl-C stops the server.

Or via the API:

```bash
curl -s -X POST localhost:8123/speak \
  -H 'content-type: application/json' \
  -d '{"text":"Welcome to module one.","voice":"narrator_en","speed":1.0}' \
  -o out.wav
```

## Run it as a menu bar app (no Terminal)

For anyone who shouldn't have to think about a virtualenv: a microphone
icon on the **right-hand side of the macOS menu bar**, next to Wi-Fi
and the clock. Launching it starts the server and opens the page;
clicking it gives Open / Stop / Show Log. Build once:

```bash
pip install -r requirements-app.txt && python setup_app.py py2app
```

Drag `dist/Text to Speech.app` to `/Applications` or the Dock. It is a
*launcher*, not a second copy of the service — 36 MB, no model, no venv
inside; it runs `.venv/bin/uvicorn` from this folder as a child, so the
rest of this README applies unchanged.

- **No Dock icon, no windows** — an `LSUIElement` agent app; the status
  item is the entire interface.
- **Quitting it stops the server**, and so do logout, restart and Force
  Quit (hooked via `NSApplicationWillTerminateNotification`), so the
  child can't be orphaned holding the port and 4 GB of model.
- **A server already on 8123 is adopted, not duplicated** — no second
  model load. If 8123 is taken by something else, it moves to the next
  free port and the menu says which.
- **`Choose Project Folder…`** repoints it if this folder moves (moving
  just the `.app` needs nothing — the build stamps the path into
  `Info.plist`).
- Server output goes to `~/Library/Logs/LocalTTS.log` (**Show Log**),
  with the access log off so errors aren't buried by the two-second
  health poll.

Two macOS specifics: the app is **unsigned**, so a copy that arrives by
download or email is quarantined — right-click → **Open** once trusts
it (AirDrop/USB copies avoid the quarantine entirely). And on a Mac
**with a notch**, a crowded menu bar hides items underneath it — if the
icon seems missing, check that first.

### Icons

`assets/iconTemplate.png`/`@2x.png` are the menu bar pair (20/40 px) —
*template* images (pure black on transparency) so macOS can recolour
them for light/dark bars and the clicked state; rumps uses the 40 px
file at 20 pt. `assets/TTS.icns` is the Finder/Gatekeeper icon. All are
drawn from one glyph:

```bash
python make_icons.py            # or: python make_icons.py your-art.png
```

Supplied artwork is used for its silhouette only — black-on-transparent
at 1024 px or larger.

## Models, offline & privacy

**Every checkpoint comes from [Hugging Face](https://huggingface.co)**,
by repo id, into the shared `~/.cache/huggingface` — outside this
folder, so it survives a re-clone. `TTS_MODEL_ID` and `TTS_MODELS` are
Hugging Face repo ids and nothing else; no vendored copy, no fallback
host. The default checkpoint is ~3 GB, fetched the first time you
generate with it, and **the page asks before starting**.

That per-checkpoint fetch is the only time the service touches the
network. After it, everything runs locally — no text, audio, or
telemetry ever leaves the machine; the "API" is your browser talking to
`127.0.0.1:8123`. To guarantee it (and skip Hugging Face's startup
update check):

```bash
HF_HUB_OFFLINE=1 uvicorn app:app --host 127.0.0.1 --port 8123
```

Verified: synthesis works with networking to Hugging Face fully
disabled. Switching to a not-yet-downloaded checkpoint still needs the
internet for that one download.

## Voices & languages

The picker is grouped by language; rows read "label — gender". Voices
are defined in `voices.yaml` and map onto the model's nine speaker
presets:

| Speaker | Gender | Native language |
|---|---|---|
| Ryan | male | English |
| Aiden | male | English |
| Vivian | female | Chinese |
| Serena | female | Chinese |
| Uncle_Fu | male | Chinese |
| Dylan | male | Chinese (Beijing) |
| Eric | male | Chinese (Sichuan) |
| Ono_Anna | female | Japanese |
| Sohee | female | Korean |

**Any preset can speak any supported language** — English, Chinese,
German, French, Spanish, Italian, Portuguese, Russian, Japanese,
Korean, plus `auto` — so adding a language is a manifest entry, not
code. `python engine.py list-languages` prints what the live checkpoint
accepts.

**There is no English-native female preset** — Ryan and Aiden are both
male. Options: point a female preset at English (`speaker: Serena`,
`language: English`) and audition the accent, or use **Kokoro-82M**,
which ships genuine English female voices and is already supported —
uncomment the example at the foot of `voices.yaml`, reload, and pick
it. It downloads on first use, takes voice packs instead of presets,
and ignores `style`.

## What survives a model swap

The service is built around a `Backend` — one class per model family,
declaring what that family can do. Everything outside it is transport,
so settings fall into three tiers:

### Fully agnostic — change the model, keep these

| Setting / file | Why it doesn't care |
|---|---|
| `pronunciations.yaml` | text → text, applied before anything model-specific (the *format* is portable; the entries are tuned to what your model gets wrong, so audition after a swap) |
| `TTS_CACHE_DIR`, `TTS_CACHE_MAX_MB` | a folder of WAVs keyed by content — the key includes the model id, so families can't serve each other's audio |
| `TTS_MAX_CHARS`, `TTS_MAX_TEXT`, `TTS_CHUNK_GAP_MS` | chunking splits plain text on sentence boundaries and stitches WAVs |
| `TTS_LOUDNESS_LUFS`, `TTS_LOUDNESS_TP` | ffmpeg `loudnorm` on the finished render |
| `TTS_VOICES_FILE`, `TTS_LEXICON_FILE` | file paths |
| `speed` (manifest, `/speak`, UI slider) | ffmpeg `atempo` on the cached base render, never a model argument |
| manifest keys `id`, `label`, `gender`, `tags` | the caller-facing contract and the picker; no model sees them |
| manifest key `model` | names the checkpoint a voice needs — the mechanism is the same whatever that checkpoint is |

### Declared by the backend

Honoured where the family supports it, and **silently left out of the
cache key where it doesn't** — so an ignored field can't split the
cache into duplicate renders of identical audio.

| Setting | Needs capability | Ignored by, today |
|---|---|---|
| `style` (manifest) | `instruct` | Kokoro; also the Qwen 0.6B builds |
| `language` (manifest) | `language` | a family with no per-call language argument |
| `TTS_TEMPERATURE`, `TTS_REP_PENALTY` | `sampling` | Kokoro and any non-autoregressive family |
| `speaker:` / `voice_pack:` / `ref_audio:` | listed in `sources` | each other — a `voice_pack` on a Qwen checkpoint is refused, not mis-synthesised |

`GET /health` reports the active family's capabilities and sources;
`python engine.py list-backends` prints them for all families.

### Model-specific — revisit when you change family

| Thing | Why it's tied to the model |
|---|---|
| `TTS_MODEL_ID`, `TTS_MODELS` | Hugging Face repo ids; family inferred from the id, or append `#backend` (`acme/Mystery-TTS#kokoro`) when it can't be |
| `speaker:` values (Ryan, Aiden, …) | CustomVoice's nine presets; another family has different ones, or none |
| English number expansion | exists *because* Qwen reads digits poorly; a family with a real G2P front-end sets `expands_numbers = False` and gets raw text |
| Stutter warning threshold | `Backend.stutter_secs_per_char` — a sampling model repeats words, a deterministic one doesn't |
| `_NORMALIZE_EN` in `engine.py` | English rules; other languages get the typography pass only |

### Adding a family

Subclass `Backend` in `engine.py`, declare `caps` and `sources`,
implement `generate()`, add it to `BACKENDS` — the two that ship
(`QwenCustomVoice`, `Kokoro`) are ~20 lines each. Override `prefetch`
only if the family fetches its checkpoint differently. Then it's a
manifest entry:

```yaml
- id: narrator_en_f
  label: "Course Narrator"
  voice_pack: af_heart          # this family's voice source
  gender: female
  language: English
  model: mlx-community/Kokoro-82M-4bit
```

A voice that pins a `model` makes that checkpoint selectable in the UI
even if it isn't in `TTS_MODELS`, and `/speak` switches to it for that
voice. Voices without one follow the model selector.

## Day-to-day

- **Voices**: edit `voices.yaml` (no Python), then "Reload config" in
  the UI settings panel.
- **Mispronunciations**: add a respelling to `pronunciations.yaml`
  (e.g. `nginx: "engine-ex"`), reload, regenerate. An ALL-CAPS key
  matches only all-caps text, so `VAT` doesn't rewrite "vat"; unknown
  acronyms are spelled out automatically, so most need no entry.
  `POST /normalize` shows exactly what the model will be asked to say.
- **Awkward delivery**: the Regenerate button re-rolls the take.
- **Script-writing guidance**: `TTS-BEST-PRACTICES.md`.
- **Models**: the top-right dropdown switches checkpoints — the 1.7B
  CustomVoice at 8-bit (default, best quality) and 4-bit (smaller, more
  mispronunciations and stutters). An unused one costs a download; the
  page asks first, shows progress, and Cancel keeps the part-download.
  The 0.6B builds are supported but deliberately unlisted: they ignore
  each voice's `style` text, so the voice set silently flattens into
  one delivery. Add their ids to `TTS_MODELS` if you want them anyway.
- **Checking the manifest**: `python engine.py check` reports voices
  the active checkpoint can't render ("Reload config" shows the same).
- **Cache**: rendered clips accumulate in `cache/`, kept under
  `TTS_CACHE_MAX_MB` (500 MB default; `off` = unbounded) by dropping
  the least recently used after a render — never the one just made.
  Settings shows size against cap; "Clear cache" (or
  `POST /cache/clear`) empties it. Everything re-synthesises on demand,
  so losing a clip costs render time only — but it *is* a cache: use
  the Download button for anything you mean to keep.
- Architecture, API reference, and config env vars: see `claude.md`.

## Uninstall

Nothing installs outside these; delete them and it's gone.

| What | Where | Size |
|---|---|---|
| The project (venv, render cache included) | this folder | ~800 MB, mostly `.venv` |
| The menu bar app | `dist/Text to Speech.app` + any copy in `/Applications` | 36 MB |
| Launcher preferences | `~/Library/Application Support/LocalTTS/` | bytes |
| Launcher log | `~/Library/Logs/LocalTTS.log` (+ `.old`) | ~1 MB |
| **The model checkpoints** | `~/.cache/huggingface/hub/models--mlx-community--Qwen3-TTS-*` (and `--Kokoro-*`) | **2 GB+ each** |

The last row is the big one and the easy one to forget. The cache is
shared with anything else on the machine that uses Hugging Face, so
delete these checkpoints' `models--…` folders rather than the whole
thing — unless nothing else uses it. No launch agents, no login items,
no receipts: the `.app` is unsigned and never touched the system.
