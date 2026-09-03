"""TTS engine: voice manifest, lexicon, cache, chunking, and backends.

Everything model-specific lives in a `Backend` subclass. The rest of
this file — lexicon, normalisation, chunking, cache, loudness, speed —
talks only to that interface, so a new model family is a new subclass
plus manifest entries, not a rewrite.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

# A checkpoint is named "repo/id", optionally "repo/id#backend" to say
# which model family drives it. Without the suffix the family is
# inferred from the id (see _BACKEND_HINTS), so the common cases need no
# extra config and older TTS_MODELS values keep working verbatim.
_BACKEND_OVERRIDE: dict[str, str] = {}


def _parse_model(spec: str) -> str:
    """"repo/id#backend" -> "repo/id", recording the backend override."""
    model_id, _, backend_name = spec.strip().partition("#")
    model_id = model_id.strip()
    if backend_name.strip():
        _BACKEND_OVERRIDE[model_id] = backend_name.strip()
    return model_id


MODEL_ID = _parse_model(
    os.environ.get(
        "TTS_MODEL_ID", "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
    )
)
# selectable checkpoints (comma-separated env override); the active one
# always appears even if not in the list.
#
# 1.7B only. The 0.6B builds are a real option -- add them back through
# TTS_MODELS if you want them -- but they don't belong in the default
# picker: they largely ignore the manifest's `style` text, and every
# voice here relies on it, so switching to one quietly flattens the
# whole voice set into the same delivery rather than failing visibly.
MODELS = [
    _parse_model(m)
    for m in os.environ.get(
        "TTS_MODELS",
        ",".join(
            [
                "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
                "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit",
            ]
        ),
    ).split(",")
    if m.strip()
]
if MODEL_ID not in MODELS:
    MODELS.insert(0, MODEL_ID)
CACHE_DIR = Path(os.environ.get("TTS_CACHE_DIR", "./cache"))


# The render cache is content-addressed, so nothing ever overwrites
# anything: without a cap a year of renders simply accumulates. A byte
# ceiling with least-recently-used eviction bounds it without ever
# costing the take just made (see `evict_cache`); "off" or 0 restores
# the old unbounded behaviour.
def _parse_cache_cap(raw: str) -> int | None:
    val = raw.strip().lower()
    if val in ("", "off", "none", "no", "0"):
        return None
    return int(float(val) * 1_000_000)


CACHE_MAX_BYTES = _parse_cache_cap(os.environ.get("TTS_CACHE_MAX_MB", "500"))

MAX_CHARS = int(os.environ.get("TTS_MAX_CHARS", "350"))
# per-request ceiling. Generation is serial (~1k chars/min), so a huge
# paste blocks the server for many minutes with no way to see progress;
# prosody is also better section by section. 5k chars is roughly five
# minutes of speech — a generous module section, not a whole course.
MAX_TEXT = int(os.environ.get("TTS_MAX_TEXT", "5000"))
CHUNK_GAP_MS = int(os.environ.get("TTS_CHUNK_GAP_MS", "120"))
VOICES_FILE = Path(os.environ.get("TTS_VOICES_FILE", "voices.yaml"))
LEXICON_FILE = Path(os.environ.get("TTS_LEXICON_FILE", "pronunciations.yaml"))
# sampling knobs — lower temperature / higher repetition penalty reduce
# stutters ("28th 28th") at some cost in liveliness
TEMPERATURE = float(os.environ.get("TTS_TEMPERATURE", "0.9"))
REPETITION_PENALTY = float(os.environ.get("TTS_REP_PENALTY", "1.1"))
# EBU R128 loudness target. Lines are synthesised independently and come
# out 3+ dB apart, which is audible when clips play back-to-back in a
# course. -16 LUFS is the spoken-word/e-learning convention; -1 dBTP
# leaves headroom so MP3/AAC encoding can't overshoot into clipping.
LOUDNESS_LUFS = os.environ.get("TTS_LOUDNESS_LUFS", "-16")
LOUDNESS_TP = os.environ.get("TTS_LOUDNESS_TP", "-1.0")
# The model emits ~80ms of silence at each end of every chunk. Left in,
# it is dead air at the head of every clip, and it stacks with the gap
# between chunks — a 120ms setting produced a ~280ms join. Trimming per
# chunk is what makes TTS_CHUNK_GAP_MS mean what it says. `off` keeps
# the model's own edges.
TRIM_SILENCE = os.environ.get("TTS_TRIM_SILENCE", "on").strip().lower() not in (
    "off", "0", "false", "no",
)
# Silence kept either side of speech after trimming: enough that an
# onset never sounds clipped, far less than the model's own padding.
TRIM_PAD_MS = int(os.environ.get("TTS_TRIM_PAD_MS", "20"))
#: Bumped whenever the render pipeline changes the audio for identical
#: input. It is part of the cache key, so old renders retire instead of
#: being served alongside new ones that sound different.
RENDER_REV = "2"


class VoiceError(KeyError):
    """Requested voice id is not in the manifest."""


class ManifestError(ValueError):
    """A voices.yaml entry the active backend cannot render."""


class RenderCancelled(Exception):
    """A render abandoned because the caller asked it to stop."""


class _Renders:
    """In-flight renders, by caller-supplied id, so one can be stopped.

    Cancellation is *cooperative and lands on a chunk boundary*, and the
    honesty of that is the whole design. A checkpoint download could be
    killed outright because it runs in its own process (see _Download);
    a render cannot -- it is one model call inside this process, holding
    the accelerator, with no interruption point of its own. What the
    render does have is the seam between chunks: long text is already
    synthesised piece by piece, so a Stop takes effect at the next one.

    So this stops real work rather than only the waiting, which is the
    line _Download drew and the reason a Cancel is offered at all. The
    price is that it is not instant, and for text short enough to be a
    single chunk there is nothing to stop early -- the caller is told
    "stopping", never "stopped", and the UI keeps its wait on screen
    until the render actually lets go.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._live: set[str] = set()
        self._cancelled: set[str] = set()

    @contextmanager
    def track(self, render_id: str | None):
        """Mark a render live for its duration. No id, no tracking."""
        if render_id is None:
            yield
            return
        with self._lock:
            self._live.add(render_id)
        try:
            yield
        finally:
            # both sets, or a cancel that arrived for a render already
            # finished would sit here forever
            with self._lock:
                self._live.discard(render_id)
                self._cancelled.discard(render_id)

    def cancel(self, render_id: str) -> bool:
        """Ask a render to stop; True if it was still running.

        A cancel is recorded even for an id not yet live: the client
        sends its id with the request, so a Stop can beat the render to
        the threadpool. Recorded early it is seen at the first
        checkpoint; recorded late it is discarded by `track`.
        """
        with self._lock:
            self._cancelled.add(render_id)
            return render_id in self._live

    def cancelled(self, render_id: str | None) -> bool:
        if render_id is None:
            return False
        with self._lock:
            return render_id in self._cancelled


renders = _Renders()


# ---------------------------------------------------------------- manifest
#
# A manifest entry names a voice *source* — the thing that tells the
# model who is speaking. Exactly one of:
#
#   speaker:    a preset baked into the checkpoint      (Qwen CustomVoice)
#   voice_pack: a separately downloaded voice file      (Kokoro, Piper)
#   ref_audio:  a clip to clone                         (Chatterbox, F5)
#   (none)      described by `style` text alone         (Qwen VoiceDesign)
#
# Which of these a checkpoint accepts is declared by its Backend, so an
# entry that can't be rendered is reported rather than mis-synthesised.

VOICE_SOURCES = ("speaker", "voice_pack", "ref_audio")

_voices: dict[str, dict] = {}
_voice_problems: list[str] = []


def voice_source(voice: dict) -> tuple[str, str]:
    """(kind, value) identifying this voice to the model. A voice with
    no source at all is ("style", "") — legitimate for a backend that
    can build a voice from its description."""
    found = [k for k in VOICE_SOURCES if voice.get(k)]
    if len(found) > 1:
        raise ManifestError(
            f"voice {voice.get('id', '?')!r} sets {' and '.join(found)} — pick one"
        )
    return (found[0], str(voice[found[0]])) if found else ("style", "")


def check_voice(voice: dict, backend: Backend | None = None) -> None:
    """Raise if `backend` can't render this entry. Called by /speak
    against the backend that will actually run, and at load time for
    voices that pin their own model (where the answer is knowable)."""
    backend = backend or backend_for(voice.get("model") or runtime.model_id)
    kind, value = voice_source(voice)
    vid = voice.get("id", "?")
    if kind == "style":
        if "instruct" not in backend.caps:
            raise ManifestError(
                f"voice {vid!r} names no {' / '.join(backend.sources)}, and backend "
                f"{backend.name!r} can't build a voice from style text alone"
            )
    elif kind not in backend.sources:
        raise ManifestError(
            f"voice {vid!r} uses {kind!r}, but backend {backend.name!r} takes "
            f"{' / '.join(backend.sources)}"
        )
    if kind == "ref_audio" and not Path(value).exists():
        raise ManifestError(f"voice {vid!r}: ref_audio {value!r} not found")


def load_voices() -> list[dict]:
    """(Re)read voices.yaml. Called at startup and by /voices/reload.
    Entries are checked as far as is knowable without loading a model;
    problems are collected rather than raised, so one bad line can't
    take the service down — /speak still refuses that voice."""
    global _voices, _voice_problems
    with open(VOICES_FILE, encoding="utf-8") as f:
        entries = yaml.safe_load(f) or []
    _voices = {v["id"]: v for v in entries}
    _voice_problems = []
    for entry in entries:
        try:
            # a voice that pins a model can be checked now; one that
            # rides the active checkpoint is checked at /speak, since
            # the active checkpoint can change under it
            check_voice(entry) if entry.get("model") else voice_source(entry)
        except ManifestError as exc:
            _voice_problems.append(str(exc))
    return entries


def voice_problems() -> list[str]:
    return list(_voice_problems)


def voices() -> list[dict]:
    if not _voices:
        load_voices()
    return list(_voices.values())


def get_voice(voice_id: str) -> dict:
    if not _voices:
        load_voices()
    try:
        return _voices[voice_id]
    except KeyError:
        raise VoiceError(voice_id) from None


# ---------------------------------------------------------------- lexicon

_lexicon: dict[str, str] = {}
_lexicon_re: re.Pattern | None = None
_lexicon_cased_re: re.Pattern | None = None


def _is_acronym_key(key: str) -> bool:
    """An all-caps key is a name, not a word: VAT the tax, not vat the
    tub. Matching those case-insensitively rewrote the ordinary noun
    too, so they match case-sensitively and everything else doesn't."""
    return key.isupper() and any(c.isalpha() for c in key)


def load_lexicon() -> dict[str, str]:
    """(Re)read pronunciations.yaml — a flat map of word/phrase ->
    respelling, applied to text before synthesis. Missing file = empty."""
    global _lexicon, _lexicon_re, _lexicon_cased_re
    if LEXICON_FILE.exists():
        with open(LEXICON_FILE, encoding="utf-8") as f:
            _lexicon = {str(k): str(v) for k, v in (yaml.safe_load(f) or {}).items()}
    else:
        _lexicon = {}

    def compile_keys(keys: list[str], flags: int = 0) -> re.Pattern | None:
        # longest keys first so "check point" wins over "check"
        keys = sorted(keys, key=len, reverse=True)
        if not keys:
            return None
        return re.compile(
            r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b", flags
        )

    _lexicon_cased_re = compile_keys([k for k in _lexicon if _is_acronym_key(k)])
    _lexicon_re = compile_keys(
        [k for k in _lexicon if not _is_acronym_key(k)], re.IGNORECASE
    )
    return _lexicon


def _lexicon_loaded() -> bool:
    return bool(_lexicon) or _lexicon_re is not None or _lexicon_cased_re is not None


def lexicon() -> dict[str, str]:
    if not _lexicon_loaded():
        load_lexicon()
    return dict(_lexicon)


def apply_lexicon(text: str) -> str:
    if not _lexicon_loaded():
        load_lexicon()
    if _lexicon_cased_re is not None:
        text = _lexicon_cased_re.sub(lambda m: _lexicon[m.group(0)], text)
    if _lexicon_re is not None:
        lowered = {k.lower(): v for k, v in _lexicon.items()}
        text = _lexicon_re.sub(lambda m: lowered[m.group(0).lower()], text)
    return text


# ---------------------------------------------------------------- normalization
#
# Deterministic English text normalization: expand numbers, times,
# currency, percentages, ordinals, and years to words BEFORE the model
# sees them — its own digit handling is weak and not controllable.
# Applied only to English voices; other languages pass through (plus the
# typography cleanup, which is language-neutral).

from num2words import num2words


def _n2w(n: int) -> str:
    return num2words(n)


def _year_words(y: int) -> str:
    if 2000 <= y <= 2009:
        return _n2w(y)
    a, b = divmod(y, 100)
    if b == 0:
        return f"{_n2w(a)} hundred"
    return f"{_n2w(a)} {'oh ' + _n2w(b) if b < 10 else _n2w(b)}"


_CURRENCY = {
    "€": ("euro", "euros", "cent", "cents"),
    "$": ("dollar", "dollars", "cent", "cents"),
    "£": ("pound", "pounds", "penny", "pence"),
}


def _time_repl(m: re.Match) -> str:
    h, mi, ap = int(m[1]), int(m[2]), m[3]
    if not (0 <= h <= 23 and 0 <= mi <= 59):
        return m[0]
    parts = [_n2w(h)]
    if mi == 0:
        if not ap:
            parts.append("o'clock")
    elif mi < 10:
        parts.append("oh " + _n2w(mi))
    else:
        parts.append(_n2w(mi))
    if ap:
        parts.append("ay em" if ap[0].lower() == "a" else "pee em")
    return " ".join(parts)


def _currency_repl(m: re.Match) -> str:
    one, many, cent_one, cent_many = _CURRENCY[m[1]]
    whole = int(m[2].replace(",", ""))
    out = f"{_n2w(whole)} {one if whole == 1 else many}"
    if m[3]:
        cents = int(m[3])
        if cents:
            out += f" and {_n2w(cents)} {cent_one if cents == 1 else cent_many}"
    return out


def _number_repl(m: re.Match) -> str:
    whole = _n2w(int(m[1].replace(",", "")))
    if m[2]:
        return f"{whole} point {' '.join(_n2w(int(d)) for d in m[2])}"
    return whole


def _digits(s: str) -> str:
    """Digit by digit, the way a reference code or phone number is read.
    Zero is "oh" — "zero" is how you read a quantity, not a sequence."""
    return " ".join("oh" if c == "0" else _n2w(int(c)) for c in s if c.isdigit())


def _sequence_repl(m: re.Match) -> str:
    """A run of digits that is an identifier rather than a quantity, so
    the grouping is preserved as pauses: 0800 123 4567 is three groups,
    not four thousand five hundred and sixty-seven."""
    groups = re.split(r"[\s.–-]+", m[0].strip())
    return ", ".join(_digits(g) for g in groups if g)


def _version_repl(m: re.Match) -> str:
    return " point ".join(_n2w(int(p)) for p in m[0].split("."))


def _year_repl(m: re.Match) -> str:
    return _year_words(int(m[1]))


def _meridiem(word: str, m: re.Match) -> str:
    """"ay em" / "pee em", keeping a full stop that was doing double
    duty. In "9 a.m. He agreed." one dot both abbreviates and ends the
    sentence; consuming it silently ran the two sentences together."""
    said = "ay em" if word.lstrip()[0].lower() == "a" else "pee em"
    if word.endswith("."):
        rest = m.string[m.end():]
        if not rest.strip() or re.match(r"\s+[\"'“‘(]?[A-Z]", rest):
            return said + "."
    return said


def _hour_ampm_repl(m: re.Match) -> str:
    """"9 a.m." — an hour with a meridiem but no minutes, which the
    clock rule (which requires :MM) never sees."""
    return f"{_n2w(int(m[1]))} {_meridiem(m[2], m)}"


#: All-caps tokens left alone. Two kinds: acronyms everyone says as a
#: word, and ordinary words that appear in capitals for emphasis —
#: lettering "N-O-T" out of "This is NOT allowed" is the failure this
#: second group exists to prevent.
_SAID_AS_WORD = {
    "NASA", "NATO", "SCUBA", "RADAR", "LASER", "ASAP", "OPEC", "AIDS",
    "SIM", "PIN", "RAM", "ROM", "GIF", "JPEG", "PNG", "WIFI", "OSHA",
    "FAQ", "ZIP", "SWOT", "OKR", "SMART", "OK",
    "NOT", "ALL", "NEW", "NOW", "YES", "YOU", "THE", "AND", "FOR", "BUT",
    "CAN", "ARE", "IS", "IT", "AN", "AS", "AT", "BE", "BY", "DO", "GO",
    "IF", "IN", "NO", "OF", "ON", "OR", "SO", "TO", "UP", "WE", "WILL",
    "MUST", "ONLY", "MORE", "LESS", "BEST", "FREE", "HERE", "THIS",
    "THAT", "WHEN", "WHY", "HOW", "WHO", "ANY", "ONE", "TWO", "SEE",
    "USE", "ADD", "GET", "PUT", "END", "TOP", "OWN", "DAY", "WAY",
}


def _acronym_repl(m: re.Match) -> str:
    """Spell out an unglossed acronym. Left to itself the model guesses
    between lettering and word-ing, and guesses differently between
    takes; hyphens are the same lever pronunciations.yaml already uses.
    Anything with a lexicon entry never reaches here — the lexicon runs
    first and its replacement is no longer all-caps."""
    word = m[0]
    if word in _SAID_AS_WORD:
        return word
    return "-".join(word)


_TYPOGRAPHY = [
    (re.compile(r"[“”]"), '"'),
    (re.compile(r"[‘’]"), "'"),
    (re.compile(r"\s*[—–]\s*"), ", "),  # em/en dash -> spoken pause
    (re.compile(r"…"), "..."),
]

_NUM = r"(\d(?:[\d,]*\d)?)"  # digits with inner commas; never eats a trailing comma

_RANGE_RE = re.compile(r"\b(\d+)\s*[–—-]\s*(\d+)\b")

# Rules run in order over the whole string, so each one only ever sees
# what its predecessors left behind. Times and money are consumed early
# (they own their digits); the bare-number rule is deliberately last.
_NORMALIZE_EN = [
    # clock times, then an hour carrying a meridiem on its own
    (re.compile(r"\b(\d{1,2}):(\d{2})\s*(a\.m\.|p\.m\.|am\b|pm\b)?", re.IGNORECASE), _time_repl),
    (re.compile(r"\b(\d{1,2})\s*(a\.m\.|p\.m\.|am|pm)\b", re.IGNORECASE), _hour_ampm_repl),
    # a bare "am" is the verb far more often than the meridiem, and only
    # the dotted spelling is unambiguous enough to rewrite on its own
    (re.compile(r"\b(a\.m\.|p\.m\.)(?=\s|$)", re.IGNORECASE), lambda m: _meridiem(m[1], m)),
    (re.compile(r"([€$£])\s?" + _NUM + r"(?:\.(\d{1,2}))?"), _currency_repl),
    (re.compile(_NUM + r"(?:\.(\d+))?\s?%"), lambda m: _number_repl(m) + " percent"),
    # before the digit/letter split below, which would strip the suffix
    (re.compile(r"\b(\d+)(?:st|nd|rd|th)\b"), lambda m: num2words(int(m[1]), to="ordinal")),
    # phone numbers and reference codes. Kept narrow on purpose — only
    # shapes no one writes for a quantity: a leading zero, nine or more
    # unbroken digits, or three separated groups.
    (re.compile(r"\b0\d[\d\s–-]{5,}\d\b"), _sequence_repl),
    (re.compile(r"\b\d{9,}\b"), _sequence_repl),
    (re.compile(r"\b\d{3,4}[\s–-]\d{3,4}[\s–-]\d{3,4}\b"), _sequence_repl),
    (re.compile(r"\b\d+(?:\.\d+){2,}\b"), _version_repl),  # 4.2.1
    (re.compile(r"\b(\d+):(\d+)\b"), lambda m: f"{_n2w(int(m[1]))} to {_n2w(int(m[2]))}"),
    (re.compile(r"(?<![\w-])-(?=\d)"), "minus "),
    # 5GB / 60mph / F1 / IPv6 — a digit welded to letters is read as one
    # nonsense word ("fiveGB"), so give the two halves a boundary
    (re.compile(r"(?<=\d)(?=[A-Za-z])|(?<=[A-Za-z])(?=\d)"), " "),
    (re.compile(r"\b([12]\d{3})\b"), _year_repl),
    (re.compile(_NUM + r"(?:\.(\d+))?"), _number_repl),
    (re.compile(r"\b[A-Z]{2,5}\b"), _acronym_repl),
    # a semicolon or colon is a real breath; the model gives neither one
    (re.compile(r"\s*[;:]\s+"), ", "),
]


def normalize_text(
    text: str, language: str = "English", expand_numbers: bool = True
) -> str:
    """Typography cleanup always (it is language- and model-neutral),
    plus the English number expansion when the model needs it. Pass
    expand_numbers=False for a family whose own front-end reads digits
    properly — pre-expanding makes those *worse*, not better."""
    english = language.strip().lower() in ("english", "auto", "")
    numbers = english and expand_numbers
    if numbers:
        # before typography turns the dash into a pause comma
        text = _RANGE_RE.sub(r"\1 to \2", text)
    for pattern, repl in _TYPOGRAPHY:
        text = pattern.sub(repl, text)
    if numbers:
        for pattern, repl in _NORMALIZE_EN:
            text = pattern.sub(repl, text)
    return text


# ---------------------------------------------------------------- backends
#
# A Backend adapts one model family to the pipeline. Adding a family is
# a subclass: declare what it can do, say which voice sources it takes,
# implement generate(). Nothing above or below this section moves.


def _probe(model, *names) -> list[str]:
    """First of `names` the model exposes, called if it's a method.
    mlx-audio spells these differently per family."""
    for name in names:
        val = getattr(model, name, None)
        if callable(val):
            try:
                val = val()
            except Exception:
                continue
        if val:
            return list(val)
    return []


class Backend:
    """One model family's API, behind a fixed interface."""

    #: registry key, and the "#suffix" in a TTS_MODELS entry
    name = "base"
    #: optional features this family has. Anything not listed is never
    #: sent to the model — and is left out of the cache key too, so a
    #: field the model ignores can't split the cache.
    #:   instruct - free-text style steering (the manifest's `style`)
    #:   sampling - temperature / repetition_penalty
    #:   language - an explicit per-call language
    caps: frozenset[str] = frozenset()
    #: voice-source fields (see VOICE_SOURCES) this family accepts
    sources: tuple[str, ...] = ("speaker",)
    #: whether digits must be spelled out before the model sees them.
    #: Qwen's own number handling is weak; a family with a real G2P
    #: front-end does it better itself, and pre-expanding hurts.
    expands_numbers = True
    #: /speak's stutter heuristic — seconds of audio per non-space
    #: character above which a render is suspiciously long. Sampling
    #: models repeat words; deterministic ones don't.
    stutter_secs_per_char = 0.14

    #: Statement run in a throwaway process to fetch this family's
    #: checkpoint *without* loading it; `model_id` is in scope. It must
    #: fetch exactly what `load()` will ask for, or the load re-downloads.
    #: Both current families resolve through mlx_audio's loader, so the
    #: default covers them.
    prefetch = "from mlx_audio.utils import get_model_path; get_model_path(model_id)"

    def load(self, model_id: str):
        from mlx_audio.tts.utils import load_model

        return load_model(model_id)

    def generate(self, model, text: str, source: tuple[str, str],
                 language: str, style: str):
        """Iterable of segments, each with `.audio` (and `.sample_rate`)."""
        raise NotImplementedError

    def speakers(self, model) -> list[str]:
        return []

    def languages(self, model) -> list[str]:
        return []


class QwenCustomVoice(Backend):
    """Qwen3-TTS CustomVoice — nine presets baked into the checkpoint,
    free-text `instruct` styling, explicit language. The default."""

    name = "qwen-customvoice"
    caps = frozenset({"instruct", "sampling", "language"})
    sources = ("speaker",)

    def generate(self, model, text, source, language, style):
        kwargs = dict(
            text=text,
            speaker=source[1],
            instruct=style or None,
            temperature=TEMPERATURE,
            repetition_penalty=REPETITION_PENALTY,
        )
        try:
            return model.generate_custom_voice(
                language=language.lower() if language else "auto", **kwargs
            )
        except TypeError:
            # some mlx-audio versions don't take `language` here
            return model.generate_custom_voice(**kwargs)

    def speakers(self, model):
        return _probe(model, "get_supported_speakers", "supported_speakers", "speakers")

    def languages(self, model):
        return _probe(model, "get_supported_languages", "supported_languages")


class Kokoro(Backend):
    """Kokoro-82M — voice packs rather than presets (`voice_pack:
    af_heart`), a misaki G2P front-end that reads numbers correctly on
    its own, and no style steering. Non-sampling, so it doesn't stutter;
    the threshold only has to allow for a slow pack."""

    name = "kokoro"
    caps = frozenset({"language"})
    sources = ("voice_pack",)
    expands_numbers = False
    stutter_secs_per_char = 0.12

    # Kokoro takes a one-letter code, not a language name
    LANG_CODES = {
        "english": "a", "american english": "a", "british english": "b",
        "spanish": "e", "french": "f", "hindi": "h", "italian": "i",
        "japanese": "j", "portuguese": "p", "chinese": "z",
    }

    def generate(self, model, text, source, language, style):
        return model.generate(
            text=text,
            voice=source[1],
            # pacing is applied after synthesis, from the cached base
            # render — never baked into the model call
            speed=1.0,
            lang_code=self.LANG_CODES.get((language or "").strip().lower(), "a"),
        )

    def languages(self, model):
        return sorted({k.title() for k in self.LANG_CODES})


class Silence(Backend):
    """A generator with no checkpoint behind it — tone where there would
    be speech, and the same near-silent padding the real families put on
    every chunk.

    This is the third adapter, and it is what makes the seam a seam: with
    only Qwen and Kokoro behind it, nothing could exercise the render
    pipeline without two gigabytes and a minute of loading, so nothing
    did. The padding is not decoration — trimming it is a pipeline step,
    and a fake that emitted clean audio would let a broken trim pass.

    Deterministic, so a test can assert on durations. Select it with
    `TTS_MODEL_ID=local/silence#silence`.
    """

    name = "silence"
    #: no style, no language, no sampling — the pipeline is the subject
    caps = frozenset()
    sources = ("speaker", "voice_pack", "ref_audio")
    expands_numbers = True
    #: deterministic: it cannot repeat itself, so nothing should trip
    stutter_secs_per_char = 1e9
    prefetch = "pass"

    SAMPLE_RATE = 24000
    #: what the real families pad each chunk with, and what _trim removes
    PAD_MS = 80
    #: speech duration per non-space character — near the middle of the
    #: 12-18 chars/sec the stutter heuristic assumes
    SECS_PER_CHAR = 0.07

    class _Model:
        sample_rate = 24000

    class _Segment:
        def __init__(self, audio, sample_rate):
            self.audio = audio
            self.sample_rate = sample_rate

    def load(self, model_id: str):
        return self._Model()

    def generate(self, model, text, source, language, style):
        sr = self.SAMPLE_RATE
        chars = max(1, len(text.replace(" ", "")))
        n = int(sr * chars * self.SECS_PER_CHAR)
        t = np.arange(n, dtype=np.float32) / sr
        speech = 0.25 * np.sin(2 * np.pi * 220.0 * t, dtype=np.float32)
        pad = np.zeros(int(sr * self.PAD_MS / 1000), dtype=np.float32)
        return [self._Segment(np.concatenate([pad, speech, pad]), sr)]

    def speakers(self, model):
        return ["Silence"]

    def languages(self, model):
        return ["English"]


BACKENDS: dict[str, type[Backend]] = {
    b.name: b for b in (QwenCustomVoice, Kokoro, Silence)
}
DEFAULT_BACKEND = QwenCustomVoice
# substring of a checkpoint id -> family, for entries that don't say
_BACKEND_HINTS = (
    ("kokoro", "kokoro"),
    ("qwen3-tts", "qwen-customvoice"),
    ("silence", "silence"),
)


def backend_for(model_id: str) -> Backend:
    """Which family drives this checkpoint. An explicit "#name" in
    TTS_MODEL_ID/TTS_MODELS wins; otherwise the id is matched against
    the known families; otherwise the default."""
    name = _BACKEND_OVERRIDE.get(model_id)
    if not name:
        low = model_id.lower()
        name = next(
            (b for hint, b in _BACKEND_HINTS if hint in low), DEFAULT_BACKEND.name
        )
    try:
        return BACKENDS[name]()
    except KeyError:
        raise ValueError(
            f'There\'s no backend called "{name}". '
            f"Known: {', '.join(sorted(BACKENDS))}."
        ) from None


class _Runtime:
    """Holds the one loaded checkpoint and the backend driving it.
    Generation is lock-guarded: one checkpoint, one accelerator — scale
    with a queue, not threads."""

    def __init__(self) -> None:
        self._model = None
        # reentrant: speak() holds it across a whole render (model
        # resolution -> last chunk) and synthesize() re-takes it per
        # chunk on the same thread. A plain Lock would deadlock there,
        # and without the outer hold a /model switch (or another voice's
        # pinned checkpoint) landing between chunks would splice two
        # models into one clip and file it under the first one's key.
        self._lock = threading.RLock()
        self.model_id = MODEL_ID
        self.backend = backend_for(MODEL_ID)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self):
        self._model = self.backend.load(self.model_id)
        return self._model

    def switch(self, model_id: str, register: bool = False) -> None:
        """Swap checkpoints; the new one loads lazily on the next call.
        `register` admits an id that isn't in TTS_MODELS — used when a
        manifest voice pins a checkpoint, so adding such a voice is
        enough to make it selectable."""
        if model_id not in MODELS:
            if not register:
                raise ValueError(
                    f'There\'s no model called "{model_id}". Add it to TTS_MODELS.'
                )
            MODELS.append(model_id)
        with self._lock:
            if model_id != self.model_id:
                self._model = None  # release before the next load
                self.model_id = model_id
                self.backend = backend_for(model_id)

    def ensure(self, model_id: str) -> None:
        if model_id != self.model_id:
            self.switch(model_id, register=True)

    def synthesize(
        self, text: str, source: tuple[str, str], language: str, style: str
    ) -> tuple[np.ndarray, int]:
        """One model call: text in, mono float32 audio + sample rate out."""
        with self._lock:
            model = self._model or self._load()
            results = list(
                self.backend.generate(model, text, source, language, style)
            )
            segments = [
                np.asarray(getattr(seg, "audio", seg), dtype=np.float32).reshape(-1)
                for seg in results
            ]
            # families report the rate on the result; fall back to the model
            sr = int(
                next((getattr(s, "sample_rate", 0) or 0 for s in results), 0)
                or model.sample_rate
            )
        return (segments[0] if len(segments) == 1 else np.concatenate(segments)), sr

    def speakers(self) -> list[str]:
        with self._lock:
            model = self._model or self._load()
        return self.backend.speakers(model)

    def languages(self) -> list[str]:
        with self._lock:
            model = self._model or self._load()
        return self.backend.languages(model)


runtime = _Runtime()


# ------------------------------------------------- first-run model download
#
# Loading a checkpoint that's already on disk takes about a minute;
# fetching one the machine has never seen is gigabytes over the network.
# The UI has to tell those two waits apart, or its "can take a minute"
# message is simply untrue -- which is exactly what switching models in
# the picker produces.


def _repo_dir(model_id: str) -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}"


def model_cached(model_id: str | None = None) -> bool:
    """Is this checkpoint downloaded in full?

    A part-finished download still leaves a snapshots/ directory behind,
    so it's the leftover *.incomplete blobs that separate "ready to
    load" from "resume where the last attempt gave up".
    """
    try:
        repo = _repo_dir(model_id or runtime.model_id)
    except Exception:  # huggingface_hub absent or restructured
        return True  # don't let a UI hint invent a download that isn't happening
    snaps = repo / "snapshots"
    if not (snaps.is_dir() and any(snaps.iterdir())):
        return False
    blobs = repo / "blobs"
    return not (blobs.is_dir() and any(blobs.glob("*.incomplete")))


def model_cache_bytes(model_id: str | None = None) -> int:
    """Bytes on disk for this checkpoint, part-downloaded ones included.
    Polled by the UI so a long download shows movement rather than an
    indeterminate shimmer. Snapshots are symlinks into blobs/, so they
    are skipped to avoid counting every file twice."""
    try:
        repo = _repo_dir(model_id or runtime.model_id)
    except Exception:
        return 0
    if not repo.is_dir():
        return 0
    total = 0
    for f in repo.rglob("*"):
        try:
            if f.is_file() and not f.is_symlink():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def downloadable_models() -> set[str]:
    """Ids /model/download may fetch: the selector list plus any
    checkpoint a manifest voice pins. Anything else is refused — an
    unauthenticated endpoint must not fetch arbitrary repos."""
    return set(MODELS) | {v["model"] for v in voices() if v.get("model")}


def explain_load_failure(exc: BaseException | str) -> str:
    """Turn a model-load failure into a sentence someone can act on.

    Takes either an exception (the in-process load) or captured stderr
    (the download subprocess). Everything that realistically goes wrong
    here is environmental -- no network on a first download, a full
    disk, a checkpoint name that doesn't exist -- and the raw text names
    none of them. The fallback still carries the original wording, so a
    genuine bug stays reportable rather than being smoothed away.
    """
    name = "" if isinstance(exc, str) else type(exc).__name__
    msg = exc if isinstance(exc, str) else str(exc)
    blob = f"{name} {msg}".lower()
    if "no space left" in blob or (
        isinstance(exc, OSError) and exc.errno == errno.ENOSPC
    ):
        return "Not enough disk space to download the voice model."
    if any(
        k in blob
        for k in (
            "localentrynotfound", "offline", "connection", "max retries",
            "getaddrinfo", "name resolution", "timed out", "network",
        )
    ):
        return (
            "Couldn't download the voice model. "
            "Check your internet connection and try again."
        )
    if any(k in blob for k in ("repositorynotfound", "gatedrepo", "401", "403", "404")):
        return f"Couldn't download the voice model — {runtime.model_id} isn't available."
    # a traceback's last line is the part that names the failure
    tail = next((l.strip() for l in reversed(msg.splitlines()) if l.strip()), "")
    return f"Couldn't load the voice model ({f'{name}: ' if name else ''}{tail or 'no detail given'})."


# ---------------------------------------------------------------- chunking

#: Abbreviations whose full stop does not end a sentence. Splitting
#: after "Dr." puts a chunk boundary — and its silence gap — inside a
#: phrase, and restarts the model's prosody mid-thought.
_ABBREV = (
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "e.g", "i.e", "fig", "no", "vol", "approx", "inc", "ltd", "co",
    "dept", "est", "min", "max", "p", "pp", "ch", "sec",
)

_SENTENCE_RE = re.compile(
    r"(?<=[.!?。！？])"
    + "".join(rf"(?<!\b{re.escape(a)}\.)" for a in _ABBREV)
    + r"\s+",
    re.IGNORECASE,
)


def chunk_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split on sentence boundaries into chunks of at most max_chars.
    Feeding a whole page in one call degrades prosody."""
    text = " ".join(text.split())
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    cur = ""
    for sentence in _SENTENCE_RE.split(text):
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(sentence[:cut])
            sentence = sentence[cut:].lstrip()
        if cur and len(cur) + 1 + len(sentence) > max_chars:
            chunks.append(cur)
            cur = sentence
        else:
            cur = f"{cur} {sentence}".strip()
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------- cache + speak


def voice_backend(voice: dict, rt: "_Runtime | None" = None) -> Backend:
    """The backend that will render this voice — the one it pins, or
    whichever is active. Resolved from the id alone, so callers can ask
    about a voice without loading a model."""
    return backend_for(voice.get("model") or (rt or runtime).model_id)


def spoken_text(text: str, voice: dict, rt: "_Runtime | None" = None) -> str:
    """Exactly what the model will be asked to say: lexicon first, then
    normalisation tuned to that model's front-end. /speak, /normalize
    and the stutter check all go through here so none of them can
    disagree about what was sent."""
    return normalize_text(
        apply_lexicon(text),
        voice.get("language", "English"),
        expand_numbers=voice_backend(voice, rt).expands_numbers,
    )


def stutter_threshold(voice: dict, rt: "_Runtime | None" = None) -> float:
    """Seconds of audio per character above which a render looks like a
    repetition — a property of the model family, not of the text."""
    return voice_backend(voice, rt).stutter_secs_per_char


def cache_key(
    voice: dict,
    text: str,
    rt: "_Runtime | None" = None,
    pipe: RenderPipeline | None = None,
) -> str:
    """Everything that can change the audio, and nothing that can't: a
    field the active backend ignores is left out, so it doesn't split
    the cache. A plain `speaker` hashes to its bare name.

    The post-processing settings are in here too, because two renders of
    the same words are not the same clip if one was trimmed or levelled
    differently — but this function does not know what they are. It asks
    the pipeline for its fingerprint, so a step added there cannot be
    forgotten here.
    """
    rt = rt or runtime
    pipe = pipe or pipeline
    kind, value = voice_source(voice)
    caps = rt.backend.caps
    h = hashlib.sha256()
    for part in (
        rt.model_id,
        value if kind == "speaker" else f"{kind}:{value}",
        voice.get("language", "") if "language" in caps else "",
        voice.get("style", "") if "instruct" in caps else "",
        # every setting that changes the audio, named by the module that
        # applies it — retargeting loudness or changing the gap must
        # invalidate the renders it would no longer describe
        *pipe.fingerprint(),
        text,
    ):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def clear_cache() -> tuple[int, int]:
    """Delete every rendered WAV (speed variants included); returns
    (files, bytes freed). The cache is a convenience, not a store —
    anything deleted re-synthesises on demand. A render in flight is
    unharmed: its atomic rename simply recreates its file."""
    import time

    files = freed = 0
    if not CACHE_DIR.is_dir():
        return 0, 0
    for f in cache_files():  # atomic-write temps handled below
        try:
            size = f.stat().st_size
            f.unlink()
        except OSError:
            continue
        files += 1
        freed += size
    # .part files are a render's atomic-write temps; one being written
    # this second must survive, so only stale leftovers go
    for f in CACHE_DIR.glob("*.part*"):
        try:
            if time.time() - f.stat().st_mtime > 3600:
                f.unlink()
        except OSError:
            pass
    return files, freed


def cache_files() -> list[Path]:
    """Every finished render — the atomic-write temps are not cache."""
    if not CACHE_DIR.is_dir():
        return []
    return [f for f in CACHE_DIR.glob("*.wav") if ".part" not in f.name]


def cache_bytes() -> int:
    """Bytes the render cache currently occupies."""
    total = 0
    for f in cache_files():
        try:
            total += f.stat().st_size
        except OSError:  # evicted or cleared under us
            continue
    return total


def _touch(*paths: Path) -> None:
    """Mark renders as used, so LRU eviction ranks them by when they
    were last wanted rather than when they were first made. A cache hit
    is otherwise invisible to the filesystem."""
    for f in paths:
        try:
            os.utime(f, None)
        except OSError:
            pass


_evict_lock = threading.Lock()


def evict_cache(keep: Iterable[Path] = ()) -> tuple[int, int]:
    """Delete least-recently-used renders until the cache fits under
    CACHE_MAX_BYTES; returns (files, bytes freed).

    "Recently used" is mtime, which `speak` touches on a cache *hit* —
    without that the order would be creation order, and a line you
    re-render every day would be evicted for being old rather than kept
    for being wanted. `keep` is the render just served: it is the newest
    thing here and the one file whose loss would be felt, so it is never
    a candidate even if it alone exceeds the cap.

    Runs after a render rather than on a timer or at startup: the cache
    can only grow by rendering, so that is the only moment the cap can
    be newly exceeded.
    """
    if CACHE_MAX_BYTES is None:
        return 0, 0
    spared = {f.resolve() for f in keep}
    with _evict_lock:
        entries = []
        total = 0
        for f in cache_files():
            try:
                st = f.stat()
            except OSError:
                continue
            total += st.st_size
            if f.resolve() not in spared:
                entries.append((st.st_mtime, st.st_size, f))
        if total <= CACHE_MAX_BYTES:
            return 0, 0
        files = freed = 0
        for _, size, f in sorted(entries):
            try:
                f.unlink()
            except OSError:  # a concurrent clear got there first
                continue
            files += 1
            freed += size
            total -= size
            if total <= CACHE_MAX_BYTES:
                break
        return files, freed


# ------------------------------------------------------- render pipeline


class RenderPipeline:
    """Everything done to the model's audio after generation, and the
    fingerprint that says which "everything" it was.

    The two halves are one module deliberately. The cache is keyed on
    the fingerprint, so a step that changes the audio for identical
    input *must* change the key — otherwise old and new renders are
    served side by side and the voice changes between lines, which is
    the one failure this service exists to prevent. While the steps and
    the key lived in separate places that was a thing a person had to
    remember; here every setting is declared once, and the fingerprint
    is derived from the same values the steps run on.

    `rev` covers the steps that have no setting to hash — a change to
    the trim algorithm, the join, or the ffmpeg filter chain. Bump it
    and every stale render retires at once.
    """

    def __init__(
        self,
        *,
        trim: bool = TRIM_SILENCE,
        trim_pad_ms: int = TRIM_PAD_MS,
        gap_ms: int = CHUNK_GAP_MS,
        loudness_lufs: str = LOUDNESS_LUFS,
        loudness_tp: str = LOUDNESS_TP,
        rev: str = RENDER_REV,
    ) -> None:
        self.trim = trim
        self.trim_pad_ms = trim_pad_ms
        self.gap_ms = gap_ms
        self.loudness_lufs = str(loudness_lufs)
        self.loudness_tp = str(loudness_tp)
        self.rev = rev

    # ------------------------------------------------------- interface

    def render(self, pieces: list[np.ndarray], sr: int, dst: Path) -> None:
        """Model chunks in, finished base render at `dst`: trimmed,
        joined with the configured gap, written atomically, levelled.

        Atomic because concurrent requests for the same cache key would
        otherwise tear the file: temp file in the destination's own
        directory, then rename.
        """
        joined = self.join([self.trim_silence(p, sr) for p in pieces], sr)
        fd, tmp = tempfile.mkstemp(dir=dst.parent, suffix=".part")
        os.close(fd)
        try:
            sf.write(tmp, joined, sr, format="WAV")
            target = self.loudness_target()
            if target is not None:
                fd2, tmp2 = tempfile.mkstemp(dir=dst.parent, suffix=".part.wav")
                os.close(fd2)
                try:
                    if self._normalize_loudness(Path(tmp), Path(tmp2), sr, target):
                        os.replace(tmp2, tmp)
                finally:
                    if os.path.exists(tmp2):
                        os.unlink(tmp2)
            os.replace(tmp, dst)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def stretch(self, src: Path, dst: Path, speed: float) -> None:
        """Pitch-preserving tempo change via ffmpeg atempo (speed<1 =
        slower). Outside the fingerprint on purpose: the speed is in the
        filename, so variants sit beside the base render rather than
        replacing it."""
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                'Speed control needs ffmpeg — install it with "brew install ffmpeg".'
            )
        fd, tmp = tempfile.mkstemp(dir=dst.parent, suffix=".part.wav")
        os.close(fd)
        try:
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                     "-filter:a", f"atempo={speed}", str(tmp)],
                    check=True, capture_output=True, timeout=120,
                )
            except (subprocess.SubprocessError, OSError):
                # a wedged or failing ffmpeg must not hang the request; the
                # base render is untouched, so 1.0x still works
                raise RuntimeError(
                    "Couldn't apply the speed change. Try again, or set speed to 1."
                ) from None
            os.replace(tmp, dst)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def fingerprint(self) -> tuple[str, ...]:
        """The parts of the cache key this pipeline owns. Every setting
        that changes the audio appears here, which is the whole point of
        the class: `render` and this method read the same fields.

        Two parts rather than one because that is the shape the key
        already had — a single joined string would rehash every render
        on disk for no change in behaviour.
        """
        return (
            f"{self.loudness_lufs}/{self.loudness_tp}",
            f"rev{self.rev}/gap{self.gap_ms}/"
            f"trim{self.trim_pad_ms if self.trim else 'off'}",
        )

    def loudness_target(self) -> float | None:
        """None when normalisation is switched off."""
        val = self.loudness_lufs.strip().lower()
        if val in ("", "off", "none", "no"):
            return None
        return float(val)

    # -------------------------------------------------- implementation

    def join(self, pieces: list[np.ndarray], sr: int) -> np.ndarray:
        """Chunks with `gap_ms` of silence between them. Exact, because
        the model's own padding was trimmed first — stacked with it, a
        120 ms setting produced a 278 ms join."""
        gap = np.zeros(int(sr * self.gap_ms / 1000), dtype=np.float32)
        joined = pieces[0]
        for piece in pieces[1:]:
            joined = np.concatenate([joined, gap, piece])
        return joined

    def trim_silence(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Drop the near-silence the model pads each chunk with, keeping
        `trim_pad_ms` either side.

        Done in numpy rather than through ffmpeg's `silenceremove` on
        purpose: ffmpeg is an optional dependency here, and chunk spacing
        is not a nicety that may quietly stop working when it is absent.
        It also avoids a subprocess per chunk.

        The threshold is relative to the clip's own peak, because trimming
        happens *before* loudness normalisation — an absolute dBFS floor
        would cut a quiet render to nothing and leave a loud one untouched.
        """
        if not self.trim or audio.size == 0:
            return audio
        mag = np.abs(audio)
        peak = float(mag.max())
        if peak <= 0:
            return audio  # digital silence: nothing to find
        loud = np.flatnonzero(mag > peak * 0.01)  # -40 dB relative to peak
        if loud.size == 0:
            return audio
        pad = int(sr * self.trim_pad_ms / 1000)
        start = max(0, int(loud[0]) - pad)
        end = min(audio.size, int(loud[-1]) + 1 + pad)
        return audio[start:end]

    def _normalize_loudness(
        self, src: Path, dst: Path, sr: int, target: float
    ) -> bool:
        """Two-pass EBU R128 normalisation. The first pass measures, the
        second applies a *linear* gain using those measurements, so the
        delivery keeps its natural dynamics instead of being compressed.
        Returns False if it couldn't run — normalisation is a nicety, never
        a reason to fail a render."""
        if shutil.which("ffmpeg") is None:
            return False
        # dual_mono matters because every render here is mono. EBU R128 sums
        # channel energies, so the same audio measures 3 LU louder the moment
        # an editor lays it on a stereo timeline — which is where these clips
        # always end up. Without this, a clip normalised to -16 plays at -13.
        filt = f"loudnorm=I={target}:TP={self.loudness_tp}:LRA=11:dual_mono=true"
        try:
            probe = subprocess.run(
                ["ffmpeg", "-hide_banner", "-nostats", "-i", str(src),
                 "-af", filt + ":print_format=json", "-f", "null", "-"],
                capture_output=True, text=True, timeout=120,
            )
            start = probe.stderr.rfind("{")
            end = probe.stderr.rfind("}")
            if start == -1 or end == -1:
                return False
            m = json.loads(probe.stderr[start:end + 1])
            if "inf" in str(m.get("input_i", "")).lower():
                return False  # silence: nothing to normalise
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
                 "-af", (f"{filt}:measured_I={m['input_i']}:measured_TP={m['input_tp']}"
                         f":measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}"
                         f":offset={m['target_offset']}:linear=true"),
                 "-ar", str(sr), "-c:a", "pcm_s16le", str(dst)],
                check=True, capture_output=True, timeout=120,
            )
            return True
        except (subprocess.SubprocessError, json.JSONDecodeError, KeyError, OSError):
            return False


#: The pipeline the service renders through, built from the environment.
#: Tests construct their own with different settings; `speak` takes one.
pipeline = RenderPipeline()


def loudness_target() -> float | None:
    """The active pipeline's target — /health reports it."""
    return pipeline.loudness_target()


def speak(
    text: str,
    voice_id: str,
    use_cache: bool = True,
    speed: float | None = None,
    rt: "_Runtime | None" = None,
    pipe: RenderPipeline | None = None,
    render_id: str | None = None,
) -> Path:
    """Render text with a manifest voice; returns the path to a WAV.
    Cached by content hash — a line is synthesised once, ever. Speed
    variants (0.5–1.5, 1.0 = as generated) are derived from the cached
    base render with ffmpeg, never by re-synthesising.

    `rt` and `pipe` default to the service's own runtime and pipeline.
    They are arguments so a caller can supply others — a Silence backend
    and a pipeline with different settings render the whole path here in
    milliseconds, which is how any of this is tested.

    `render_id` names this render so `renders.cancel(id)` can stop it;
    it raises RenderCancelled at the next chunk boundary.
    """
    rt = rt or runtime
    pipe = pipe or pipeline
    voice = get_voice(voice_id)
    if speed is None:
        speed = float(voice.get("speed", 1.0))
    if not 0.5 <= speed <= 1.5:
        raise ValueError("Speed must be between 0.5 and 1.5.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # the runtime lock is held from model resolution through the last
    # chunk: the model id is baked into the cache key, so nothing may
    # swap the checkpoint while a render that key describes is running
    with renders.track(render_id), rt._lock:
        # a render can wait here behind another one, and a Stop clicked
        # during that wait must not be spent synthesising anyway
        if renders.cancelled(render_id):
            raise RenderCancelled
        # a voice may pin its own checkpoint; do this first, because the
        # model id is part of the cache key and of what the backend can do
        if voice.get("model"):
            rt.ensure(voice["model"])
        check_voice(voice, rt.backend)
        # cache is keyed on the rewritten text, so a lexicon or normalizer
        # change invalidates exactly the lines it touches
        text = spoken_text(text, voice, rt)
        # MAX_CHARS read here, not bound into chunk_text's default at
        # import — otherwise changing it has no effect on a render
        chunks = chunk_text(text, MAX_CHARS)
        if not chunks:
            raise ValueError("Enter some text first.")

        key = cache_key(voice, text, rt, pipe)
        path = CACHE_DIR / f"{key}.wav"
        final = path if speed == 1.0 else CACHE_DIR / f"{key}_x{speed:g}.wav"
        if use_cache and final.exists():
            _touch(final, path)
            return final
        if use_cache and path.exists():
            _touch(path)
            pipe.stretch(path, final, speed)
            evict_cache(keep=(final, path))
            return final

        source = voice_source(voice)
        caps = rt.backend.caps
        pieces: list[np.ndarray] = []
        sr = 0
        for chunk in chunks:
            # the seam the whole cancellation rests on: the model call
            # below cannot be interrupted, the gap before it can
            if renders.cancelled(render_id):
                raise RenderCancelled
            audio, sr = rt.synthesize(
                chunk,
                source,
                voice.get("language", "English") if "language" in caps else "",
                voice.get("style", "") if "instruct" in caps else "",
            )
            pieces.append(audio)

    # everything past the model call belongs to the pipeline, including
    # the trim: it is numpy on audio already generated, so holding the
    # runtime lock through it would only make other renders wait
    pipe.render(pieces, sr, path)
    if final != path:
        pipe.stretch(path, final, speed)
    evict_cache(keep=(final, path))
    return final


class _Download:
    """A checkpoint fetch running in its own process.

    A subprocess rather than a thread for exactly one reason: it can be
    killed. The transfer happens inside huggingface_hub, which offers no
    cancellation hook, and a Python thread cannot be interrupted -- so a
    Cancel button backed by a thread would be a lie, and aborting the
    browser's fetch would stop only the waiting, not the downloading.

    Part-finished blobs stay on disk as *.incomplete and huggingface_hub
    resumes from them, so cancelling costs nothing but the chunk in
    flight. Splitting the fetch out of `/speak` is also what makes the
    two waits separately honest: this one has real bytes and a Cancel,
    and the load that follows it is local and about a minute.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._err: Path | None = None
        self.model_id: str | None = None
        self.state = "idle"  # idle | downloading | done | cancelled | failed
        self.error: str | None = None

    def start(self, model_id: str) -> None:
        with self._lock:
            # settle a finished transfer first, or its leftover
            # "downloading" would wrongly refuse the next one
            self._reap()
            # already here: record it, so a poller that reads status()
            # straight after doesn't see a stale state for another model
            # and wait for a transfer that will never begin
            if model_cached(model_id):
                self.model_id, self.state, self.error = model_id, "done", None
                return
            if self.state == "downloading":
                if self.model_id == model_id:
                    return  # already fetching this one; join it
                raise RuntimeError("Another download is already running.")
            code = f"import sys\nmodel_id = sys.argv[1]\n{backend_for(model_id).prefetch}\n"
            fd, path = tempfile.mkstemp(prefix="tts-fetch-", suffix=".log")
            self._err = Path(path)
            self._proc = subprocess.Popen(
                [sys.executable, "-c", code, model_id],
                stdout=subprocess.DEVNULL,
                stderr=fd,
                # the progress bars would otherwise be the whole of
                # stderr, burying the traceback we report on failure
                env={**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1"},
            )
            os.close(fd)
            self.model_id, self.state, self.error = model_id, "downloading", None

    def cancel(self) -> None:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            if self.state == "downloading":
                self.state = "cancelled"
            self._drop_err()

    def status(self) -> dict:
        with self._lock:
            self._reap()
            return {
                "state": self.state,
                "model": self.model_id,
                "bytes": model_cache_bytes(self.model_id) if self.model_id else 0,
                "error": self.error,
            }

    def _reap(self) -> None:
        if self.state != "downloading" or self._proc is None:
            return
        rc = self._proc.poll()
        if rc is None:
            return
        text = ""
        if rc != 0 and self._err and self._err.exists():
            text = self._err.read_text(errors="replace")[-4000:]
        self._drop_err()
        if rc == 0:
            self.state = "done"
        else:
            self.state, self.error = "failed", explain_load_failure(text)

    def _drop_err(self) -> None:
        """The stderr capture has served its purpose once the transfer
        settles; left behind, one temp file leaked per attempt."""
        if self._err:
            try:
                self._err.unlink(missing_ok=True)
            except OSError:
                pass
            self._err = None


download = _Download()


# ---------------------------------------------------------------- CLI

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "list-speakers":
        for name in runtime.speakers():
            print(name)
    elif cmd == "list-languages":
        for name in runtime.languages():
            print(name)
    elif cmd == "list-backends":
        for name, cls in sorted(BACKENDS.items()):
            print(f"{name}\t sources: {'/'.join(cls.sources) or '-'}"
                  f"\t caps: {'/'.join(sorted(cls.caps)) or '-'}")
    elif cmd == "check":
        problems = voice_problems()
        for entry in voices():
            try:
                check_voice(entry)
            except ManifestError as exc:
                problems.append(str(exc))
        # Language detection is checked here too: it is the other thing
        # that can be wrong without anything failing, and one command
        # for "is the config sound" is easier to remember than two.
        # Imported late so the engine keeps no dependency on it.
        import languages
        problems.extend(languages.selftest())
        for line in dict.fromkeys(problems):
            print(line, file=sys.stderr)
        print(f"{len(voices())} voices, {len(languages.CORPUS)} detection "
              f"cases, {len(set(problems))} problem(s) "
              f"against backend {runtime.backend.name!r}")
        sys.exit(1 if problems else 0)
    else:
        print("usage: python engine.py "
              "list-speakers|list-languages|list-backends|check", file=sys.stderr)
        sys.exit(2)
