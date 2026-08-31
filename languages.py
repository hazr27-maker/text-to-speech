"""Per-language data, and nothing else.

This module knows about *languages*. It does not know which model is
loaded, which backend is active, or what any of them can do — that
belongs to `Backend`, and joining the two is `check.py`'s job. Keeping
the split means a model swap can never make this file wrong.

Today each record carries what language detection needs. The number and
acronym rules for a language land here too when they are written, beside
the words that identify it, so adding a language stays one entry rather
than edits scattered across the engine.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    #: display name, matching the manifest's `language` values
    name: str
    #: the most frequent function words, which is what identifies a
    #: language in a short sample. Written accent-folded, to match what
    #: `_fold` produces at lookup time.
    stopwords: frozenset[str] = frozenset()
    #: letters that are strong evidence on their own. Shared accents
    #: (é, ü, à) are deliberately absent: they corroborate nothing.
    marks: str = ""


def _words(*text: str) -> frozenset[str]:
    return frozenset(" ".join(text).split())


#: Detection is deliberately wider than the voice manifest. A script in
#: a language we have no voice for is *more* wrong than one we do, not
#: less, so it is worth naming — see check.py's language finding.
#:
#:
#: The lists overlap freely, and that is not a flaw to be pruned. A word
#: shared by three languages raises all three scores by the same amount,
#: so it never moves them relative to each other — it only separates
#: them collectively from the languages that don't share it. Removing
#: shared words was tried and is strictly worse: it strips "la", "le"
#: and "il" from the Romance languages, which are the most frequent
#: words they have. Absolute score says "this is a language I know";
#: the margin over the runner-up says which one.
_FREQUENT: dict[str, str] = {
    "English": """the and is are was were be been of to for with from by on
        at this that these those it its you your we our they their he she
        his her not but or if then than as have has had will would can
        could should does did what when where which who how any some more
        most such only same too very just about there here""",
    "German": """der die das den dem des ein eine einen einem und oder aber
        nicht ist sind war waren sein seine ihr ihre wir uns sie ihnen mit
        von zu fur auf aus bei nach uber unter durch werden wurde haben
        hatte kann muss soll auch nur noch schon wenn dann als wie was wer
        wo dass diese dieser dieses jetzt hier um am im vom zum zur man es
        an ich du sich nicht mehr sehr""",
    "French": """le la les un une des du de au aux et ou mais ne pas est
        sont etait etre avoir ont avec pour dans sur sous par vers chez
        nous vous ils elles leur leurs ce cette ces qui que dont plus
        moins tres bien aussi tout tous toute toutes ici""",
    "Spanish": """el la los las un una unos unas y o pero no es son era ser
        haber han con para por en sobre entre desde hasta nosotros ustedes
        ellos ellas su sus este esta estos estas que quien donde cuando
        como mas menos muy tambien todo todos toda todas aqui de del al
        se lo su si ya""",
    "Italian": """il lo la i gli le un una uno e o ma non sono era essere
        hanno con per da in su tra fra noi voi loro questo questa questi
        queste che chi dove quando come piu meno molto anche tutto tutti
        tutta tutte della delle dello degli nel nella qui di se gia
        stesso""",
    "Portuguese": """o a os as um uma uns umas e ou mas nao sao era ser tem
        com para por em sobre entre desde ate nos voces eles elas seu sua
        seus suas este esta estes estas que quem onde quando como mais
        menos muito tambem todo todos toda todas do da dos das no na nas
        ao aos aqui de se lo ja mesmo""",
}

def _weights() -> dict[str, float]:
    """How much one word is worth as evidence: 1 divided by the number
    of languages that share it.

    A word in every Romance list still says "this is Romance", which is
    worth something — but "der" says "German" outright and should not
    count the same. Without this, a runner-up that shares most of the
    winner's words trails by only the handful of unique ones, and one
    short sentence never clears the margin.
    """
    shared: dict[str, int] = {}
    for words in _FREQUENT.values():
        for w in set(words.split()):
            shared[w] = shared.get(w, 0) + 1
    return {w: 1 / n for w, n in shared.items()}


WEIGHTS = _weights()

LANGUAGES: dict[str, Language] = {
    name: Language(name, frozenset(_FREQUENT[name].split()), marks)
    for name, marks in (
        ("English", ""),
        ("German", "ß"),
        ("French", "çœ"),
        ("Spanish", "ñ¿¡"),
        ("Italian", ""),
        ("Portuguese", "ãõ"),
    )
}

#: Scripts that identify a language outright, so no scoring is needed.
#: Kept even though nothing here speaks them: recognising pasted
#: Japanese is what lets the check say there is no voice for it, rather
#: than letting an English voice read it.
_SCRIPTS = (
    ("Japanese", re.compile(r"[぀-ヿ]")),
    ("Korean", re.compile(r"[가-힯]")),
    ("Chinese", re.compile(r"[一-鿿]")),
)

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# Detection stays silent unless the evidence is clear. A wrong "this
# looks like French" on a short line teaches people to ignore the panel,
# which costs more than the check is worth — so the failure mode is
# silence, never a confident wrong answer.
MIN_WORDS = 6         # below this there is nothing to score
MIN_SCORE = 1.75      # needs real evidence, not one coincidence
MIN_MARGIN = 0.75     # and a clear winner, not a photo finish


def _fold(word: str) -> str:
    """Strip accents, so "não" matches the Portuguese list's "nao"."""
    return "".join(
        c for c in unicodedata.normalize("NFD", word.lower())
        if unicodedata.category(c) != "Mn"
    )


def detect(text: str) -> str | None:
    """The language this text appears to be written in, or None when the
    evidence is too thin to say."""
    for name, pattern in _SCRIPTS:
        if pattern.search(text):
            return name

    words = [_fold(w) for w in _WORD_RE.findall(text)]
    if len(words) < MIN_WORDS:
        return None

    lowered = text.lower()
    scores = {
        lang.name: sum(WEIGHTS[w] for w in words if w in lang.stopwords)
        + (2.0 if any(m in lowered for m in lang.marks) else 0)
        for lang in LANGUAGES.values()
    }

    (best, top), (_, runner) = sorted(scores.items(), key=lambda kv: -kv[1])[:2]
    if top < MIN_SCORE or top - runner < MIN_MARGIN:
        return None
    return best


def article(word: str) -> str:
    """"a German accent" but "an English one". Only ever applied to
    language names, so the vowel test is enough — no need for the
    silent-h and long-u exceptions a general rule would want."""
    return f"{'an' if word[:1].upper() in 'AEIOU' else 'a'} {word}"


# ---------------------------------------------------------------- self-test
#
# The corpus lives beside the lists it exercises, because the two move
# together: widening a list to catch one sentence is exactly the change
# that can quietly lose another. Run by `python engine.py check`.

#: One line per language, plus text that must stay silent. The silent
#: cases are the important half — a confident wrong answer teaches
#: people to ignore the panel, which costs more than the check is worth.
CORPUS: tuple[tuple[str, str | None], ...] = (
    ("Der schnelle braune Fuchs springt über den faulen Hund.", "German"),
    ("Die Sitzung beginnt um neun Uhr und endet am Nachmittag.", "German"),
    ("Die API ist nicht optional und der Preis liegt bei 1.250 Euro.", "German"),
    ("Welcome to module one. In this session we will cover the rules.", "English"),
    ("The quick brown fox jumps over the lazy dog in the garden.", "English"),
    ("Le renard brun rapide saute par-dessus le chien paresseux.", "French"),
    ("La session commence à neuf heures et se termine dans la journée.", "French"),
    ("El zorro marrón salta sobre el perro perezoso y cuenta.", "Spanish"),
    ("Usted encontrará que esta es la mejor manera de hacerlo.", "Spanish"),
    ("La volpe marrone salta sopra il cane pigro nella tasca.", "Italian"),
    ("La sessione inizia alle nove e finisce nel pomeriggio.", "Italian"),
    ("A raposa castanha salta sobre o cao preguicoso no bolso.", "Portuguese"),
    ("A sessao comeca as nove e termina no fim da tarde.", "Portuguese"),
    ("これは日本語のテキストです。", "Japanese"),
    ("Short line.", None),
    ("Click here.", None),
    ("Step 1: go now.", None),
    ("Module 4", None),
)


def selftest() -> list[str]:
    """Detection failures, as printable lines. Empty when all pass."""
    return [
        f"detection: {text[:44]!r} -> {detect(text)}, expected {want}"
        for text, want in CORPUS
        if detect(text) != want
    ]
