"""The render pipeline, exercised without a checkpoint.

Every test here goes through `engine.speak` — the same interface the
service calls — using the `Silence` backend and a pipeline built for the
test. That is the point of both: the seam is only real if something
other than a 2 GB model can sit behind it.
"""

import wave

import numpy as np
import pytest

import engine


SR = engine.Silence.SAMPLE_RATE


@pytest.fixture
def rt():
    """A runtime driving the Silence backend, isolated from the service's."""
    runtime = engine._Runtime()
    runtime.switch("local/silence", register=True)
    assert runtime.backend.name == "silence"
    return runtime


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """A cache directory of this test's own."""
    monkeypatch.setattr(engine, "CACHE_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def calls(rt):
    """Counts model calls — the only way to tell a cache hit from a
    re-render that happens to produce the same bytes."""
    class Counter:
        n = 0
    counter = Counter()
    original = rt.backend.generate

    def counted(*a, **kw):
        counter.n += 1
        return original(*a, **kw)

    rt.backend.generate = counted
    return counter


@pytest.fixture
def voice():
    return engine.get_voice("narrator_en")


def wav(path):
    """(samples, sample rate) of a rendered file."""
    with wave.open(str(path)) as f:
        return f.getnframes(), f.getframerate()


def seconds(path):
    frames, rate = wav(path)
    return frames / rate


# ----------------------------------------------------------- the seam


def test_speak_renders_without_a_checkpoint(rt, cache, voice):
    out = engine.speak("Hello there.", "narrator_en", rt=rt)
    assert out.exists() and out.parent == cache
    assert seconds(out) > 0


def test_silence_backend_pads_like_a_real_one(rt):
    """The fake reproduces the ~80 ms the real families emit — a fake
    with clean edges would let a broken trim pass."""
    audio, sr = rt.synthesize("Hello", ("speaker", "Ryan"), "", "")
    assert sr == SR
    lead = np.flatnonzero(np.abs(audio) > 0.01)[0]
    assert lead == pytest.approx(SR * engine.Silence.PAD_MS / 1000, rel=0.02)


# ------------------------------------------------------ the pipeline


def test_trim_removes_the_model_padding():
    pipe = engine.RenderPipeline(trim=True, trim_pad_ms=20)
    pad = np.zeros(int(SR * 0.08), dtype=np.float32)
    speech = np.full(int(SR * 0.5), 0.3, dtype=np.float32)
    trimmed = pipe.trim_silence(np.concatenate([pad, speech, pad]), SR)
    # 0.5 s of speech plus the 20 ms kept either side
    assert trimmed.size / SR == pytest.approx(0.54, abs=0.005)


def test_trim_off_keeps_the_model_edges():
    pipe = engine.RenderPipeline(trim=False)
    audio = np.concatenate([
        np.zeros(int(SR * 0.08), dtype=np.float32),
        np.full(int(SR * 0.5), 0.3, dtype=np.float32),
    ])
    assert pipe.trim_silence(audio, SR).size == audio.size


def test_join_gap_is_exactly_what_was_asked_for():
    """The bug this trim exists for: the model's own padding stacked
    with the configured gap, so 120 ms produced a 278 ms join."""
    pipe = engine.RenderPipeline(gap_ms=120)
    piece = np.full(int(SR * 0.2), 0.3, dtype=np.float32)
    joined = pipe.join([piece, piece], SR)
    quiet = joined.size - 2 * piece.size
    assert quiet / SR == pytest.approx(0.120, abs=0.001)


def test_multi_chunk_render_spends_exactly_the_gap(rt, cache, voice, monkeypatch):
    """Same words, same chunks, two gap settings: the whole difference
    in duration must be the gap and nothing else."""
    monkeypatch.setattr(engine, "MAX_CHARS", 30)
    two = "First sentence here. Second sentence here."
    assert len(engine.chunk_text(two, engine.MAX_CHARS)) == 2
    tight = engine.speak("t " + two, "narrator_en", rt=rt,
                         pipe=engine.RenderPipeline(gap_ms=0, loudness_lufs="off"))
    roomy = engine.speak("t " + two, "narrator_en", rt=rt,
                         pipe=engine.RenderPipeline(gap_ms=300, loudness_lufs="off"))
    assert seconds(roomy) - seconds(tight) == pytest.approx(0.300, abs=0.002)


# --------------------------------------------------- the fingerprint
#
# The invariant the cache rests on: a setting that changes the audio
# must change the key. These are the tests that make it enforced rather
# than remembered.


@pytest.mark.parametrize("field,value", [
    ("trim", False),
    ("trim_pad_ms", 40),
    ("gap_ms", 250),
    ("loudness_lufs", "-14"),
    ("loudness_tp", "-2.0"),
    ("rev", "99"),
])
def test_every_audio_affecting_setting_reaches_the_key(field, value, voice):
    base = engine.RenderPipeline()
    changed = engine.RenderPipeline(**{field: value})
    assert base.fingerprint() != changed.fingerprint(), field
    assert cache_keys_differ(voice, base, changed), field


def cache_keys_differ(voice, a, b):
    return engine.cache_key(voice, "same words", pipe=a) != engine.cache_key(
        voice, "same words", pipe=b
    )


def test_identical_settings_give_the_same_key(voice):
    a, b = engine.RenderPipeline(), engine.RenderPipeline()
    assert engine.cache_key(voice, "x", pipe=a) == engine.cache_key(voice, "x", pipe=b)


def test_key_covers_the_text_and_the_model(rt, voice):
    assert engine.cache_key(voice, "one") != engine.cache_key(voice, "two")
    other = engine._Runtime()
    other.switch("local/silence-2", register=True)
    assert engine.cache_key(voice, "x", rt) != engine.cache_key(voice, "x", other)


# -------------------------------------------------------- the cache


def test_second_render_is_served_from_cache(rt, cache, voice, calls):
    first = engine.speak("Cache me.", "narrator_en", rt=rt)
    again = engine.speak("Cache me.", "narrator_en", rt=rt)
    assert again == first
    assert calls.n == 1, "a cache hit must not reach the model"
    assert len(list(cache.glob("*.wav"))) == 1


def test_cache_false_rerenders_in_place(rt, cache, voice, calls):
    first = engine.speak("Re-roll me.", "narrator_en", rt=rt)
    again = engine.speak("Re-roll me.", "narrator_en", rt=rt, use_cache=False)
    assert again == first
    assert calls.n == 2, "Regenerate must reach the model again"
    assert len(list(cache.glob("*.wav"))) == 1


def test_a_pipeline_change_does_not_serve_the_old_render(rt, cache, voice):
    """The failure this whole design prevents: two renders that sound
    different must not share a filename."""
    quiet = engine.RenderPipeline(gap_ms=120, loudness_lufs="off")
    roomy = engine.RenderPipeline(gap_ms=400, loudness_lufs="off")
    a = engine.speak("Spaced out.", "narrator_en", rt=rt, pipe=quiet)
    b = engine.speak("Spaced out.", "narrator_en", rt=rt, pipe=roomy)
    assert a != b
    assert len(list(cache.glob("*.wav"))) == 2


def test_eviction_drops_the_least_recently_used(rt, cache, voice, monkeypatch):
    monkeypatch.setattr(engine, "CACHE_MAX_BYTES", 1)   # keep only the newest
    old = engine.speak("First line.", "narrator_en", rt=rt)
    new = engine.speak("Second line.", "narrator_en", rt=rt)
    assert new.exists(), "the render just served is never evicted"
    assert not old.exists()


def test_eviction_spares_the_render_just_served(rt, cache, voice, monkeypatch):
    monkeypatch.setattr(engine, "CACHE_MAX_BYTES", 1)
    only = engine.speak("Bigger than the cap on its own.", "narrator_en", rt=rt)
    assert only.exists()


# -------------------------------------------------------- the speed


@pytest.mark.skipif(not engine.shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_speed_variant_is_derived_not_resynthesised(rt, cache, voice):
    base = engine.speak("Slow this down.", "narrator_en", rt=rt)
    slow = engine.speak("Slow this down.", "narrator_en", rt=rt, speed=0.8)
    assert slow != base and slow.exists() and base.exists()
    assert slow.name.endswith("_x0.8.wav")
    assert seconds(slow) == pytest.approx(seconds(base) / 0.8, rel=0.05)


def test_speed_out_of_range_is_refused(rt, cache, voice):
    with pytest.raises(ValueError, match="between 0.5 and 1.5"):
        engine.speak("Too fast.", "narrator_en", rt=rt, speed=3.0)


# ------------------------------------------------------ the loudness


@pytest.mark.skipif(not engine.shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_loudness_normalisation_changes_the_level(rt, cache, voice):
    off = engine.speak("Level me.", "narrator_en", rt=rt,
                       pipe=engine.RenderPipeline(loudness_lufs="off"))
    on = engine.speak("Level me.", "narrator_en", rt=rt,
                      pipe=engine.RenderPipeline(loudness_lufs="-16"))
    assert peak(off) != pytest.approx(peak(on), rel=0.01)


def test_loudness_off_is_read_from_the_pipeline():
    assert engine.RenderPipeline(loudness_lufs="off").loudness_target() is None
    assert engine.RenderPipeline(loudness_lufs="-16").loudness_target() == -16.0


def peak(path):
    with wave.open(str(path)) as f:
        data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    return float(np.abs(data).max())
