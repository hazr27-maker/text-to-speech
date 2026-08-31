"""Script check: what the model will be asked to say, and what will
read badly — decided before a render rather than discovered after one.

Pure functions over `engine` and `languages`. Nothing here loads a
model, writes a cache entry, or changes the text: it is advisory, and a
person can ignore every word of it and press Generate.

Two rules hold the whole thing together.

**A finding may only describe behaviour that will actually happen.**
Normalisation is per model family, so a finding about it is true for one
backend and a lie for the next. Every such finding is therefore gated on
what the *live* backend does — obtained by asking it (`_expands_numbers`)
rather than by re-deriving the engine's condition here, because a copy
of that condition is a second source of truth and the way it fails is by
promising a rewrite that never happens.

**A finding is written for someone writing a script**, not someone
editing configuration: no filenames, no environment variables, no
milliseconds. The exception in `engine.check_voice` — which names
`voice_pack` and `ref_audio` on purpose — exists because its reader
really is editing the manifest. This reader is not.
"""

from __future__ import annotations

import difflib
import re

import engine
import languages

#: Punctuation that ends a sentence, plus the closers that may follow it.
_TERMINAL = ".!?…\"')]"

_CAPS_RE = re.compile(r"\b[A-Z]{2,5}\b")
_WORD_RE = re.compile(r"\S+")
#: Punctuation whose coming and going is not worth reporting. The
#: hyphen is deliberately absent: it is the whole mechanism behind
#: "A-P-I" and "check-point", so a change that only adds hyphens is the
#: most meaningful kind there is, not the least.
_NOISE_RE = re.compile(r"""[\s,;:.!?…"'“”‘’()\[\]—–]+""")


def _finding(level: str, kind: str, title: str, detail: str, token=None) -> dict:
    """level is one of warn / info / ok — the order they are shown in."""
    return {"level": level, "kind": kind, "title": title,
            "detail": detail, "token": token}


def _expands_numbers(voice: dict) -> bool:
    """Does this voice get the number and acronym pass, or typography
    only? Probed rather than re-derived — see the module docstring."""
    return "one" in engine.spoken_text("Test 1", voice)


def _diff(before: str, after: str) -> tuple[list[dict], list[list[int]]]:
    """Word-level rewrite pairs, and the character spans in the rewritten
    text that they produced.

    Spans rather than a string search, because a replacement can be a
    single comma and searching for that highlights every comma in the
    script.

    Rewrites that only move punctuation around are dropped. An em dash
    becoming a comma is real — it is how the model is made to pause —
    but nobody needs telling about it, and listing it crowds out the
    rewrites that change what a word sounds like.
    """
    a = before.split()
    spans = [m.span() for m in _WORD_RE.finditer(after)]
    b = [after[i:j] for i, j in spans]

    pairs: list[dict] = []
    marks: list[list[int]] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag == "equal":
            continue
        src, dst = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        if _NOISE_RE.sub("", src) == _NOISE_RE.sub("", dst):
            continue  # punctuation moved around; nothing to hear about
        pairs.append({"from": src, "to": dst})
        if j2 > j1:
            marks.append([spans[j1][0], spans[j2 - 1][1]])
    return pairs, marks


def _language_finding(text: str, language: str, expands: bool) -> dict | None:
    """The one failure that ruins a whole clip rather than a word.

    The engine decides how to normalise from the *voice's* language and
    never looks at the text, so German typed against an English voice
    really is read with English numbers — and the only way to discover
    that today is to listen to the take.
    """
    guessed = languages.detect(text)
    if not guessed or guessed == language:
        return None

    consequence = (
        f"It will be read with {languages.article(language)} accent"
        + (f", and its numbers and capitals read as {language}"
           if expands else "")
        + "."
    )

    offered = {v.get("language", "English") for v in engine.voices()}
    if guessed in offered:
        return _finding(
            "warn", "language",
            f"This looks like {guessed}, but the voice is {language}",
            consequence + f" Switch to {languages.article(guessed)} voice, or "
            "ignore this if the text really is meant for this voice.",
        )
    # Nothing to switch to. Say so plainly rather than recommending a
    # voice that doesn't exist — and don't imply the language can't be
    # spoken, because the gap is which voices we offer, not the model.
    return _finding(
        "warn", "language",
        f"This looks like {guessed}, and there's no {guessed} voice",
        consequence + f" There's no {guessed} voice to switch to yet.",
    )


def check(text: str, voice: dict) -> dict:
    """Everything the panel shows, for one piece of text and one voice."""
    lexicon = engine.lexicon()
    lexed = engine.apply_lexicon(text)
    spoken = engine.spoken_text(text, voice)
    chunks = engine.chunk_text(spoken)

    language = voice.get("language", "English")
    expands = _expands_numbers(voice)
    findings: list[dict] = []

    # --- the voice's own pronunciations, confirmed ------------------------
    # Not a problem: the point is to show that a respelling you
    # configured is being used, since the alternative is wondering.
    for word, said in sorted(lexicon.items()):
        pattern = rf"(?<!\w){re.escape(word)}(?!\w)"
        if re.search(pattern, text, 0 if word.isupper() else re.IGNORECASE):
            findings.append(_finding(
                "ok", "lexicon", "Pronunciation applied",
                "Your pronunciation list respells this, so it's said the way "
                "you want.",
                token=f"{word} → {said}",
            ))

    # --- does the script match the voice? ---------------------------------
    if (mismatch := _language_finding(text, language, expands)):
        findings.append(mismatch)

    # --- findings that describe the normalisation pass --------------------
    # Gated, not dimmed: for a voice that doesn't get this pass they are
    # false rather than merely irrelevant.
    if expands:
        seen: list[str] = []
        for match in _CAPS_RE.finditer(lexed):
            word = match.group(0)
            if word not in seen and word not in engine._SAID_AS_WORD:
                seen.append(word)
        if seen:
            findings.append(_finding(
                "info", "acronym",
                f"{len(seen)} acronym{'s' if len(seen) > 1 else ''} "
                "will be spelled out",
                "Capitals the voice doesn't know are read letter by letter, "
                "so they sound the same in every take. Add one to your "
                "pronunciation list if it should be said as a word.",
                token="  ·  ".join(f"{w} → {'-'.join(w)}" for w in seen),
            ))

        if (left := sorted(set(re.findall(r"\d[\d.,:/]*", spoken)))):
            findings.append(_finding(
                "warn", "digits", "Some numbers are left to the voice",
                "How these are read is left to the voice, and it can change "
                "between takes. Write them out in words if the reading "
                "matters.",
                token=", ".join(left[:8]),
            ))
    elif re.search(r"\d", text):
        # Not a warning: for this voice, digits reaching the model is the
        # designed behaviour. But it is the one thing about this voice a
        # writer needs to know.
        findings.append(_finding(
            "info", "digits-raw", "Numbers are sent as digits",
            f"{language} numbers are read by the voice as they're written, "
            "and how it says them can change between takes. Write one out in "
            "words if the reading matters.",
        ))

    # --- length and joins -------------------------------------------------
    if len(text) > engine.MAX_TEXT:
        findings.append(_finding(
            "warn", "length", "Too long to record",
            "This is too long to record in one go, and won't render. Split "
            "it into shorter passages.",
        ))

    if len(chunks) > 1:
        mid_sentence = any(
            not c.rstrip().endswith(tuple(_TERMINAL)) for c in chunks
        )
        findings.append(_finding(
            "warn" if mid_sentence else "info", "chunks",
            f"Recorded in {len(chunks)} parts",
            "Long text is recorded in parts and joined with a short pause, "
            "so the delivery restarts at every join." + (
                " One join falls in the middle of a sentence, which you will "
                "hear. Shorten that sentence, or split it with a full stop."
                if mid_sentence else
                " Every join lands at the end of a sentence."
            ),
        ))

    for sentence in engine._SENTENCE_RE.split(" ".join(spoken.split())):
        if len(sentence) > engine.MAX_CHARS:
            findings.append(_finding(
                "warn", "sentence", "This sentence is too long",
                "It's too long to be spoken in one piece, so it will be cut "
                "somewhere in the middle. Split it with a full stop.",
                token=sentence[:70] + "…",
            ))

    # --- punctuation and spacing ------------------------------------------
    stripped = text.strip()
    if stripped and stripped[-1] not in _TERMINAL:
        findings.append(_finding(
            "info", "punctuation", "No closing punctuation",
            "Punctuation is what sets the pacing. A line without it often "
            "trails off, or runs straight into whatever you record next.",
        ))

    if re.search(r"\S {2,}\S", text):
        findings.append(_finding(
            "info", "spacing", "Double spaces",
            "Harmless, but they don't buy you a pause — they're removed "
            "before recording. Use a comma or a full stop for that.",
        ))

    order = {"warn": 0, "info": 1, "ok": 2}
    findings.sort(key=lambda f: order[f["level"]])

    changes, spans = _diff(text, spoken)
    return {
        "spoken": spoken,
        "spans": spans,
        "changes": changes,
        "chunks": chunks,
        "findings": findings,
    }
