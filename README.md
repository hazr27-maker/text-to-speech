# Local TTS

Turn written scripts into spoken voice-over on your own Mac — no
account, no per-word billing, and no text or audio leaving the machine.

Read this file top to bottom to go from a fresh clone to a rendered WAV.
**[What it's for](#what-its-for)** and
**[Designed for Apple Silicon](#designed-for-apple-silicon)** are
context; **[Install](#install)** and **[Run](#run)** are the two steps
that actually get you running. Everything after
**[Run it as a menu bar app](#run-it-as-a-menu-bar-app-no-terminal)** is
reference material you can read when you need it.

## What it's for

This exists for one job: **narrating training content.** Course scripts
are written as many short lines rather than one long take, they get
revised constantly, and every line has to sound like the same narrator
as the one before it. That combination is what makes hand-recording
painful and most cloud services awkward:

- **Scripts change.** Fix one sentence in module three and only that
  line should be re-rendered, not the whole course re-recorded. Renders
  are content-addressed, so an edited line is the only thing that
  re-synthesises — everything else is served from cache, instantly.
- **Consistency matters more than perfection.** A voice is defined once
  in `voices.yaml` (preset, language, style, pacing) and every line
  rendered through it matches. Loudness is normalised to a broadcast
  target, so clips don't jump in volume when played back to back in a
  finished module.
- **Some words are always read wrong.** Product names and acronyms get a
  respelling in `pronunciations.yaml`, applied everywhere, once —
  instead of being fought line by line.
- **The material is often confidential.** Nothing is uploaded. The model
  runs in the same process as the server, reachable only on
  `127.0.0.1`.

It is a small appliance, not a platform: a web UI for writing a line and
hearing it, plus an HTTP API for scripting a whole module.

**What it is not for:** real-time or streaming speech, telephony,
cloning a specific real person's voice, or high-volume batch rendering
on a shared server. It renders a finished WAV per request, one at a
time, on one machine.

## Designed for Apple Silicon

**As shipped, this targets Apple Silicon Macs (M1–M4) and nothing
else.** That is a deliberate choice, not an oversight, and it goes
deeper than a dependency:

- `requirements.txt` pins [`mlx-audio`](https://github.com/Blaizzy/mlx-audio),
  which runs the model on **MLX**, Apple's own array framework, against
  the Mac's GPU and unified memory.
- The default checkpoint is an **MLX-specific conversion**
  (`mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit`). It is not a
  portable format that other runtimes can load.
- The **menu bar launcher** (`menubar.py`, `setup_app.py`) is macOS-only
  by nature — a `rumps` status-item app packaged with py2app.

On this hardware everything works with no configuration: install, run,
generate. There is nothing to tune and no GPU to select.

**Running it elsewhere is a supported port, not a config change.**
On NVIDIA or Linux, swap `mlx-audio` in `requirements.txt` for
`qwen-tts` (plus `flash-attn`), set
`TTS_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`, and point a
`Backend` subclass at the `qwen_tts` API (`load` + `generate`, ~20
lines). Everything outside that class — the manifest, cache, lexicon,
API and UI — is unchanged. See
**[What survives a model swap](#what-survives-a-model-swap)**.

## Requirements

- **An Apple Silicon Mac** (M1–M4) — see above
- **Python 3.10+**
- **`ffmpeg` on PATH** (`brew install ffmpeg`) for speed control and
  loudness normalisation. Strictly optional, but the two degrade
  differently: loudness normalisation is skipped silently, so clip
  volumes vary between takes, while asking for any speed other than
  `1.0` fails outright with a message telling you to install it.
- **~3 GB of disk** for the model checkpoint, plus ~700 MB for the
  Python environment

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No build step.

**Every command in this README assumes that activated venv.** `python`,
`pip` and `uvicorn` mean the venv's own, not the system's — macOS ships
no bare `python` at all, so without the activation they fail with
*command not found* rather than doing anything surprising. In a fresh
terminal, `source .venv/bin/activate` first, or prefix the command with
`.venv/bin/` (`.venv/bin/python engine.py check`).

Nothing is downloaded yet — the model is fetched on first use, and the
page asks before it starts. See
**[Where the models come from](#where-the-models-come-from)**.

## Run

```bash
source .venv/bin/activate
uvicorn app:app --host 127.0.0.1 --port 8123
```

Open **http://127.0.0.1:8123/** — pick a voice, paste text, Generate.
If the checkpoint isn't on the machine yet the page asks before fetching
it (`Download the voice model?`) and shows bytes-so-far with a working
Cancel; cancelling keeps the part-download and resumes next time. Once
it's there, the first generation loads it in a minute or so; after that
it's seconds, and identical re-requests are instant (cached). Stop the
server with Ctrl-C.

Or via the API:

```bash
curl -s -X POST localhost:8123/speak \
  -H 'content-type: application/json' \
  -d '{"text":"Welcome to module one.","voice":"narrator_en","speed":1.0}' \
  -o out.wav
```

## Run it as a menu bar app (no Terminal)

For anyone who shouldn't have to think about a virtualenv, there's a
launcher: a microphone icon in the **right-hand side of the macOS menu
bar**, alongside Wi-Fi and the clock. Launching it starts the server and
opens the page; clicking the icon gives Open / Stop / Show Log.

Build it once:

```bash
pip install -r requirements-app.txt && python setup_app.py py2app
```

That produces `dist/Text to Speech.app` — drag it to `/Applications`, or
straight to the Dock. It is a *launcher*, not a second copy of the
service: 36 MB, with no model and no venv inside it. It starts
`.venv/bin/uvicorn` from this folder as a child process, so everything
else in this README still applies unchanged.

- **No Dock icon, no windows.** It's an `LSUIElement` agent app; the
  status item is the entire interface.
- **Quitting it stops the server** — and so do logout, restart and Force
  Quit, which terminate the app without running the menu's Quit. The
  child is killed on `NSApplicationWillTerminateNotification` as well, so
  it can't be orphaned holding the port and 4 GB of model.
- **A server already on 8123 is adopted, not duplicated.** A leftover
  from a Terminal run won't trigger a second model load. If 8123 is
  taken by something that isn't us, it moves to the next free port and
  the menu says which.
- **`Choose Project Folder…`** repoints it if this folder moves. The
  build stamps the current path into `Info.plist`, so moving just the
  `.app` needs nothing.
- Server output goes to `~/Library/Logs/LocalTTS.log` (**Show Log**),
  with uvicorn's access log off — a health poll every two seconds would
  otherwise bury the errors the log exists to show.

Two macOS specifics worth knowing:

- It's **unsigned**, so a copy that arrives by download or email is
  quarantined by Gatekeeper: right-click → **Open** once and it's
  trusted from then on. Copying the folder by AirDrop or USB avoids it.
- On a Mac **with a notch**, a crowded menu bar pushes items underneath
  it, where they're genuinely invisible. If the icon seems missing,
  check that before anything else.

### Icons

`assets/iconTemplate.png` and `@2x.png` are the menu bar pair (20 and
40 px). They are *template* images — pure black on transparency — which
is what lets macOS recolour them itself for the light bar, the dark bar
and the clicked-and-highlighted state; rumps pins the status item to
20 pt and does no `@2x` lookup, so the 40 px file is the one handed to
it. `assets/TTS.icns` is the Finder / Gatekeeper / Login Items icon.
All three are drawn from one glyph:

```bash
python make_icons.py            # or: python make_icons.py your-art.png
```

Supplied artwork is used for its silhouette only, so give it
black-on-transparent at 1024 px or larger.

## Where the models come from

**Every checkpoint comes from [Hugging Face](https://huggingface.co)**,
by repo id, into the shared `~/.cache/huggingface` — outside this
folder, so it survives a re-clone and is shared with anything else on
the machine that uses it. `TTS_MODEL_ID` and `TTS_MODELS` are Hugging
Face repo ids and nothing else; there is no vendored copy and no
fallback host.

The default checkpoint is about 3 GB on disk. It's fetched the first
time you generate with it and **the page asks before starting** —
nothing large is downloaded behind your back. That fetch is the only
time the service touches the network at all.

## Offline & privacy

Internet is needed **once per checkpoint**, to fetch it from Hugging
Face. After that everything runs locally — no text, audio, or telemetry
ever leaves the machine. The "API" is just your browser talking to the
local server on `127.0.0.1:8123`; the model runs in that same process.

To guarantee it (and skip Hugging Face's update check on startup):

```bash
HF_HUB_OFFLINE=1 uvicorn app:app --host 127.0.0.1 --port 8123
```

Verified: synthesis works with networking to Hugging Face fully disabled.
Note that switching to a checkpoint you haven't downloaded yet does need
the internet for that one download.

## Voices & languages

The picker is grouped by language and each row reads "label — gender".
Voices are defined in `voices.yaml`; they map onto the nine speaker
presets the model ships with:

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
German, French, Spanish, Italian, Portuguese, Russian, Japanese and
Korean, plus `auto` to let the model detect it — so adding a language is
just a manifest entry, not code. `python engine.py list-languages`
prints the list the live checkpoint actually accepts.

**Note: there is no English-native female preset.** Both English natives
(Ryan, Aiden) are male. For a female English voice you can point a
female preset at English (`speaker: Serena`, `language: English`) —
audition it first, as a non-native accent may or may not suit your
content. If none of them fits, **Kokoro-82M** ships genuine English
female voices and is already supported: uncomment the example at the
foot of `voices.yaml`, reload, and pick it. It downloads on first use,
takes voice packs instead of presets, and ignores `style`.

## What survives a model swap

The service is built around a `Backend` — one class per model family,
declaring what that family can do. Everything outside it is transport.
So most settings are portable, some are honoured only where the family
supports them, and a few are genuinely tied to the model. Which is
which:

### Fully agnostic

Change the model, keep these exactly as they are.

| Setting / file | Why it doesn't care |
|---|---|
| `pronunciations.yaml` | plain text → text, applied before anything model-specific. (The *format* is portable; the entries themselves are tuned to whatever your model gets wrong, so audition them after a swap.) |
| `TTS_CACHE_DIR` | the cache stores WAVs, keyed by content — and the key includes the model id, so two families can't serve each other's audio |
| `TTS_CACHE_MAX_MB` | a byte ceiling on a folder of WAVs; what produced them is irrelevant |
| `TTS_MAX_CHARS`, `TTS_MAX_TEXT`, `TTS_CHUNK_GAP_MS` | chunking splits plain text on sentence boundaries and stitches WAVs |
| `TTS_LOUDNESS_LUFS`, `TTS_LOUDNESS_TP` | ffmpeg `loudnorm`, applied to the finished render |
| `TTS_VOICES_FILE`, `TTS_LEXICON_FILE` | file paths |
| `speed` — manifest, `/speak`, and the UI slider | ffmpeg `atempo` on the cached base render, never a model argument |
| manifest keys `id`, `label`, `gender`, `tags` | the caller-facing contract and the picker; no model sees them |
| manifest key `model` | names the checkpoint a voice needs — the mechanism is the same whatever that checkpoint is |

### Declared by the backend

Honoured where the family supports it, and **silently left out of the
cache key where it doesn't** — so a field your model ignores can't split
the cache into duplicate renders of identical audio.

| Setting | Needs capability | Ignored by, today |
|---|---|---|
| `style` (manifest) | `instruct` | Kokoro; also the Qwen 0.6B builds |
| `language` (manifest) | `language` | a family with no per-call language argument |
| `TTS_TEMPERATURE`, `TTS_REP_PENALTY` | `sampling` | Kokoro and any other non-autoregressive family — the knobs don't exist there |
| `speaker:` / `voice_pack:` / `ref_audio:` | must be one the family lists in `sources` | each other — a `voice_pack` on a Qwen checkpoint is refused, not mis-synthesised |

`GET /health` reports the active family's capabilities and accepted
voice sources; `python engine.py list-backends` prints them for all
families.

### Model-specific

These are the ones to revisit when you change family.

| Thing | Why it's tied to the model |
|---|---|
| `TTS_MODEL_ID`, `TTS_MODELS` | Hugging Face repo ids. The family is inferred from the id; append `#backend` (e.g. `acme/Mystery-TTS#kokoro`) when it can't be |
| `speaker:` values (Ryan, Aiden, …) | CustomVoice's nine presets. Another family has different ones, or none at all |
| English number expansion | it exists *because* Qwen reads digits poorly. A family with a real G2P front-end sets `expands_numbers = False` and gets the raw text — pre-expanding makes those models worse, not better |
| The stutter warning threshold | `Backend.stutter_secs_per_char`. A sampling model repeats words; a deterministic one doesn't, and its threshold only has to allow for a slow voice |
| `_NORMALIZE_EN` in `engine.py` | English rules. Other languages get the typography pass only |

### Adding a family

Subclass `Backend` in `engine.py`, declare `caps` and `sources`,
implement `generate()`, and add it to `BACKENDS` — the two that ship
(`QwenCustomVoice`, `Kokoro`) are about twenty lines each. Override
`prefetch` only if the family fetches its checkpoint some other way;
both that ship resolve through mlx_audio's loader and share the
default. Then it is
a manifest entry:

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
voice. Voices without a `model` follow the model selector.

## Day-to-day

- **Voices** live in `voices.yaml` — add/edit entries (no Python),
  then press "Reload config" in the UI settings panel.
- **Mispronunciations**: add a respelling to `pronunciations.yaml`
  (e.g. `nginx: "engine-ex"`), reload config, regenerate.
- **Awkward delivery**: the Regenerate button re-rolls the take.
- **Script-writing guidance**: see `TTS-BEST-PRACTICES.md`.
- **Models**: the top-right dropdown switches checkpoints — the 1.7B
  CustomVoice at 8-bit (the default, best quality) and 4-bit (smaller,
  but more mispronunciations and stutters). Picking one you haven't used
  before costs a download from Hugging Face — the page asks first, shows
  bytes-so-far, and Cancel keeps the part-download so a retry resumes.
  When more than one model *family* is on offer, each row names its
  backend. The 0.6B builds are supported but deliberately not listed:
  they ignore each voice's `style` text, so the voice set would flatten
  into one delivery with nothing appearing to fail. Add their ids to
  `TTS_MODELS` if you want them anyway.
- **Checking the manifest**: `python engine.py check` reports voices the
  active checkpoint can't render; "Reload config" shows the same thing.
  `python engine.py list-backends` prints what each family accepts.
- **Cache**: rendered clips accumulate in `cache/`, kept under
  `TTS_CACHE_MAX_MB` (500 MB by default, `off` for no limit): once past
  it, the least recently used clips are dropped after a render, never
  the one just made. Settings shows the current size against the cap.
  "Clear cache" there (or `POST /cache/clear`) empties it outright.
  Everything re-synthesises on demand, so losing a cached clip costs
  render time and nothing else — but `cache/` is a cache, not storage:
  use the Download button for anything you mean to keep.
- Architecture, API reference, and config env vars: see `claude.md`.

## Uninstall

Nothing installs outside the places below; delete them and it's gone.

| What | Where | Size |
|---|---|---|
| The project (venv, render cache included) | this folder | ~800 MB, mostly `.venv` |
| The menu bar app | `dist/Text to Speech.app`, plus any copy you dragged to `/Applications` | 36 MB |
| Launcher preferences | `~/Library/Application Support/LocalTTS/` | bytes |
| Launcher log | `~/Library/Logs/LocalTTS.log` (+ `.old`) | ~1 MB |
| **The model checkpoints** | `~/.cache/huggingface/hub/models--mlx-community--Qwen3-TTS-*` (and `--Kokoro-*` if used) | **2 GB+ each** |

The last row is the big one and the easy one to forget. It's shared
with anything else on the machine that uses Hugging Face, so delete the
`models--…` folders for *these* checkpoints rather than the whole cache
— unless nothing else uses it, in which case `~/.cache/huggingface` can
go entirely. No launch agents, no login items, no receipts: the `.app`
is unsigned and never touched the system.
