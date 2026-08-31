# TTS best practices

Practical guidance for getting clean voice-overs out of this service
(Qwen3-TTS CustomVoice). Everything here runs locally.

## Writing the script

- **Write for the ear, not the eye.** Short sentences. One idea per
  sentence. Read it aloud yourself first — if you stumble, the model will.
- **Punctuation is your intonation control.** Commas force short pauses;
  full stops force sentence-final falls; question marks get rising
  intonation. A paragraph with no commas gets read in one breathless run.
- **Numbers are normalized automatically** (English voices): clock
  times, €/$/£ currency, percentages, ordinals (28th), years, digit
  ranges (3–5), negatives (−5), ratios and scores (16:9, 3–1), version
  numbers (4.2.1), phone numbers and long reference codes (read digit by
  digit), and plain numbers are all expanded to words before the model
  sees them. A digit stuck to letters is separated too, so "5GB" and
  "F1" are read rather than mangled into one word. Check what the model
  will actually be asked to say with `POST /normalize` if a reading
  surprises you.
- **Acronyms are spelled out automatically.** Any unfamiliar 2–5 letter
  all-caps word (API, NHS, SLA) becomes "A-P-I". Ones said as words
  (NASA, SCUBA) are left alone. If yours is read the wrong way, put it in
  `pronunciations.yaml` once — `SQL: "sequel"` for a pronounceable one —
  rather than respelling it in every script.
- **Still spell out yourself:**
  - Units and symbols: "&" → "and", "/" → "or" (or "per"), "km" → "kilometres".
  - URLs and file paths: rewrite as you'd say them ("app dot py").
- **Keep emphasis in the wording,** not in formatting — asterisks and
  markdown do nothing. Avoid ALL CAPS for emphasis in particular: a short
  capitalised word is treated as an acronym, so "this is NOT allowed"
  risks being read "N-O-T". Common words are exempted, but italics-style
  emphasis belongs in the sentence, not the casing.

## Fixing mispronunciations

- Add a line to `pronunciations.yaml` (`word: "respelling"`), press
  **Reload config**, regenerate. The fix is permanent and applies to all
  future scripts; the cache invalidates exactly the lines it touches.
- **Case matters for one kind of key.** An ALL-CAPS entry matches only
  all-caps text, so `VAT: "V-A-T"` fixes the tax without touching "a vat
  of oil". Everything else matches regardless of case.
- Respelling tricks: hyphens force syllable breaks ("check-point"),
  spaces split compounds, sounds-like spellings fix odd words
  ("koob-control"), stress can often be nudged by doubling a letter or
  hyphenating ("REE-cord" style respellings sometimes help — test).
- This model has no phoneme/SSML input — respelling is the lever.

## Fixing awkward delivery

- **Regenerate.** Synthesis is sampled, so the same line reads slightly
  differently each run. An awkward phrase often fixes itself in one or
  two re-rolls (the Regenerate button bypasses the cache). It wakes up
  once you have a take on screen, and greys out again if you edit the
  text or switch voice — at that point what you want is Generate.
- **The repeated-word warning.** Sampling occasionally makes the model
  say a word twice ("28th 28th"). When a render comes back longer than
  its text should need, the page says so — *The model may have repeated
  a word — listen back, or Regenerate.* It is deliberately cautious: it
  catches gross repetitions, not subtle ones, so your ear stays the
  final check.
- **Style text** (per voice, in `voices.yaml`) steers overall delivery:
  pace ("speak slowly and deliberately, with clear pauses"), energy,
  emotion, register. It changes the model's actual prosody — prefer it
  over post-processing when the whole voice should change.
  Note: the 0.6B model builds ignore style text.
- **Speed slider** for exact pacing (pitch-preserving, 0.5–1.5×).
  Small moves — 0.9–0.95× — usually sound best; below ~0.8× the pauses
  start to feel stretched.
- **Restructure before you fight.** If a sentence reads badly after a
  few re-rolls, reword it — it's faster than hunting a perfect take.

## Levels and mastering

- **Loudness is handled for you.** Every render is normalised to -16 LUFS
  with a -1 dBTP ceiling, so clips sit at a consistent volume when played
  back-to-back and won't clip when encoded into a video. You should not
  need to touch gain in your editor.
- **Don't be alarmed if your meter reads about -19.** These files are
  mono, and the loudness standard counts a mono file differently from a
  stereo one. They are levelled so that they hit -16 LUFS *once they are
  on a stereo timeline*, which is where they end up. Measured on its own
  a clip reads roughly 3 LU lower; that is correct, not quiet.
- **Clips start and end tight.** The model pads each render with about
  80ms of silence; that is trimmed to a 20ms handle, so you can butt
  clips up against each other without trimming dead air by hand.
- **Very short, punchy clips** (a single word like "Stop.") can come out
  slightly quieter than the rest. They hit the peak ceiling before
  reaching the loudness target — the only way louder is compression,
  which would change the delivery. If such a line must match, record it
  as part of a longer sentence and cut it in your editor.
- **Don't normalise again downstream.** Stacking another pass on top
  undoes the headroom and can introduce pumping.

## Known limitations

- **Homographs can't be fixed in the lexicon.** Words like "read",
  "live", "lead", and "wound" change pronunciation with context, and the
  lexicon is a whole-word find-and-replace — respelling one sense breaks
  the other. If one is read wrongly, reword the sentence ("has read" →
  "has finished reading"). Regenerating sometimes lands the right sense
  too, since the model uses context.
- **Takes are not reproducible.** Synthesis is sampled, so the same line
  rendered twice differs slightly, and a colleague on another machine
  won't get your exact take. What you download is the artefact — keep
  approved WAVs; don't assume you can re-create one identically later.
  (Locally, the cache does return the same file until the text changes.)
- **Joins between chunks are uniform.** Text over ~350 characters is
  split on sentence boundaries and rejoined with a fixed short pause
  (120ms, and now exactly that — chunk edges are trimmed first, so the
  model's own padding no longer stacks on top of it). Uniform is the
  limitation: a break between two paragraphs sounds the same as one
  mid-paragraph. For a long section where pacing matters, render
  paragraph by paragraph and assemble in your editor, where you control
  the gaps.

## Workflow

- **Chunking is automatic** (≤350 chars, sentence boundaries) — but the
  model still does best with paragraph-sized inputs. Render a long
  script section by section rather than as one wall of text.
- **One render is capped at 5,000 characters** (~5 minutes of speech).
  That is a deliberate ceiling, not a technical limit: generation is
  serial at roughly a thousand characters a minute, so a bigger paste
  ties up the service for a long time with no progress to watch. Work in
  sections and assemble in your editor.
- **The cache is your friend.** Identical text + voice + speed returns
  instantly. Iterate on the one line you're editing; the rest of the
  script costs nothing to re-render.
- **Lock the wording before polishing delivery.** Every text edit is a
  new render; every delivery edit on stable text is nearly free.
- **Listen on the target device.** Laptop speakers hide sibilance and
  low-end artifacts that headphones or a phone will expose.
- **Keep masters at 1.0×.** If you need a slow variant, render at 1.0
  and let the speed control derive it — you can always re-derive, and
  the base render stays reusable.
