"""HTTP transport for the local TTS engine. All logic lives in engine.py."""

import logging
import re
from datetime import datetime
from pathlib import Path

from urllib.parse import urlsplit

import soundfile as sf

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import engine

app = FastAPI(title="Local TTS")

# The service is personal and unauthenticated, so the loopback address
# is the whole perimeter -- and the browser can be tricked across it. A
# Host header that isn't loopback is DNS rebinding (a hostile page
# reading a "same-origin" server that is really this one); a cross-site
# Origin on a state-changing request is CSRF (a hostile page firing
# no-preflight POSTs at a known local port -- /model/download is a
# multi-gigabyte transfer one such POST away). Requests without an
# Origin (curl, the menu bar's health poll) are untouched.
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _hostname(netloc_or_url: str) -> str | None:
    try:
        value = netloc_or_url if "//" in netloc_or_url else f"//{netloc_or_url}"
        return urlsplit(value).hostname
    except ValueError:
        return None


@app.middleware("http")
async def _local_only(request: Request, call_next):
    if _hostname(request.headers.get("host", "")) not in _LOCAL_HOSTS:
        return JSONResponse(
            {"detail": "This service answers local requests only."}, status_code=400
        )
    origin = request.headers.get("origin")
    if (
        origin
        and request.method not in ("GET", "HEAD", "OPTIONS")
        and _hostname(origin) not in _LOCAL_HOSTS
    ):
        return JSONResponse(
            {"detail": "Cross-site requests aren't allowed."}, status_code=403
        )
    return await call_next(request)


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=engine.MAX_TEXT)
    voice: str
    cache: bool = True
    # 1.0 = as generated; <1 slower, >1 faster (pitch preserved via ffmpeg).
    # None falls back to the voice's manifest `speed`, then 1.0.
    speed: float | None = Field(default=None, ge=0.5, le=1.5)


# sync def on purpose: FastAPI runs it in the threadpool, keeping the
# event loop free while the (lock-guarded) model generates
@app.post("/speak")
def speak(req: SpeakRequest):
    try:
        path = engine.speak(req.text, req.voice, use_cache=req.cache, speed=req.speed)
    except engine.VoiceError:
        raise HTTPException(
            status_code=404,
            detail=f'There\'s no voice called "{req.voice}". '
            "Reload the page to refresh the list.",
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        # A checkpoint that won't download or load raises none of the
        # above -- huggingface_hub has its own exception hierarchy -- so
        # without this the UI showed FastAPI's bare "Internal Server
        # Error" for the single most likely failure there is. Logged as
        # well as reported: the sentence is for whoever is waiting, the
        # traceback is for whoever has to fix it.
        logging.getLogger("uvicorn.error").exception("/speak failed")
        raise HTTPException(status_code=503, detail=engine.explain_load_failure(exc))
    # voice + first words + yymmdd-hhmm, matching the UI's download names
    slug = "-".join(re.sub(r"[^a-z0-9\s]", "", req.text.lower()).split()[:4]) or "speech"
    stamp = datetime.now().strftime("%y%m%d-%H%M")
    headers = {}
    note = _stutter_check(path, req)
    if note:
        headers["X-TTS-Note"] = note
    return FileResponse(
        path,
        media_type="audio/wav",
        filename=f"{req.voice}-{slug}-{stamp}.wav",
        headers=headers,
    )


# ~12–18 chars/sec is normal English TTS; much slower suggests the model
# repeated itself. The threshold belongs to the model family (a sampling
# model stutters, a deterministic one doesn't), so the backend sets it.
# Deliberately conservative — a miss is fine, the human is still the
# judge; this only nudges them to listen closely.


def _stutter_check(path: Path, req: SpeakRequest) -> str | None:
    voice = engine.get_voice(req.voice)
    chars = len(engine.spoken_text(req.text, voice).replace(" ", ""))
    if chars < 20:
        return None
    speed = req.speed if req.speed is not None else float(voice.get("speed", 1.0))
    duration = sf.info(path).duration * speed  # undo the tempo stretch
    if duration / chars > engine.stutter_threshold(voice):
        return "The model may have repeated a word — listen back, or Regenerate."
    return None


@app.get("/voices")
def voices():
    """The manifest, plus `model_cached` on any voice that pins its own
    checkpoint -- that render downloads a model the header's selector
    knows nothing about, and the UI must ask before gigabytes, not
    discover them behind a Generating label."""
    out = []
    for v in engine.voices():
        v = dict(v)
        if v.get("model"):
            v["model_cached"] = engine.model_cached(v["model"])
        out.append(v)
    return out


class ModelRequest(BaseModel):
    model: str


@app.get("/models")
def models():
    """Each checkpoint with the backend that drives it — the UI shows
    the family so a Kokoro entry isn't mistaken for a Qwen one."""
    return {
        "available": [
            {"id": m, "backend": engine.backend_for(m).name} for m in engine.MODELS
        ],
        "active": engine.runtime.model_id,
        "backend": engine.runtime.backend.name,
    }


@app.post("/model")
def set_model(req: ModelRequest):
    try:
        engine.runtime.switch(req.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "active": engine.runtime.model_id,
        "backend": engine.runtime.backend.name,
        "loaded": engine.runtime.loaded,
    }


class DownloadRequest(BaseModel):
    # None = the active checkpoint, which is all the endpoint ever
    # fetched before the body existed -- old callers are unchanged
    model: str | None = None


@app.post("/model/download")
def start_download(req: DownloadRequest | None = None):
    """Fetch a checkpoint without loading it — the active one, or a
    named one (how the UI pre-fetches a voice's pinned model, which the
    selector may not list).

    Split out of /speak so the wait can be cancelled: the transfer runs
    in a killable subprocess, and the /speak that follows is a local
    load. Idempotent — asking for one already in progress joins it.
    """
    model = (req.model if req else None) or engine.runtime.model_id
    if model not in engine.downloadable_models():
        raise HTTPException(
            status_code=400,
            detail=f'There\'s no model called "{model}". Add it to TTS_MODELS.',
        )
    try:
        engine.download.start(model)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return engine.download.status()


@app.get("/model/download")
def download_status():
    return engine.download.status()


@app.delete("/model/download")
def cancel_download():
    """Stop the transfer. Part-downloaded blobs are kept — huggingface_hub
    resumes from them, so cancelling is cheap to undo."""
    engine.download.cancel()
    return engine.download.status()


@app.post("/voices/reload")
def reload_voices():
    """Re-reads voices.yaml AND pronunciations.yaml; model stays warm.
    `problems` lists manifest entries that won't render — reported here
    rather than raised, so one bad line can still be edited and
    reloaded without restarting the service."""
    count = len(engine.load_voices())
    return {
        "voices": count,
        "pronunciations": len(engine.load_lexicon()),
        "problems": engine.voice_problems(),
    }


@app.post("/cache/clear")
def clear_cache():
    """Empty the render cache. Renders are cheap to rebuild and the
    cache otherwise only ever grows; this is the drain."""
    files, freed = engine.clear_cache()
    return {"files": files, "bytes": freed}


@app.get("/pronunciations")
def pronunciations():
    return engine.lexicon()


class NormalizeRequest(BaseModel):
    text: str
    voice: str


@app.post("/normalize")
def normalize(req: NormalizeRequest):
    """Preview what the model will actually be asked to say."""
    try:
        voice = engine.get_voice(req.voice)
    except engine.VoiceError:
        raise HTTPException(
            status_code=404,
            detail=f'There\'s no voice called "{req.voice}". '
            "Reload the page to refresh the list.",
        )
    return {"text": engine.spoken_text(req.text, voice)}


@app.get("/health")
def health():
    _cached = engine.model_cached()
    return {
        "ok": True,
        "model": engine.runtime.model_id,
        "model_loaded": engine.runtime.loaded,
        # "loaded" is a minute's wait; "not yet downloaded" is gigabytes
        # and worth asking about first. Progress lives on
        # /model/download, which is the only thing that needs it.
        "model_cached": _cached,
        "backend": engine.runtime.backend.name,
        "capabilities": sorted(engine.runtime.backend.caps),
        "voice_sources": list(engine.runtime.backend.sources),
        "max_chars": engine.MAX_CHARS,
        "max_text": engine.MAX_TEXT,
        "chunk_gap_ms": engine.CHUNK_GAP_MS,
        "loudness_lufs": engine.loudness_target(),
        # "./cache" rather than "cache", which reads like a stray word
        "cache_dir": str(engine.CACHE_DIR) if engine.CACHE_DIR.is_absolute()
        else f"./{engine.CACHE_DIR}",
        # what the cache holds now, and the ceiling it is kept under
        # (null = unbounded) — the settings panel shows both, so the
        # cap is visible before it silently drops an old render
        "cache_bytes": engine.cache_bytes(),
        "cache_max_bytes": engine.CACHE_MAX_BYTES,
    }


# the UI — mounted last so API routes take precedence
app.mount(
    "/",
    StaticFiles(directory=Path(__file__).parent / "static", html=True),
    name="ui",
)
