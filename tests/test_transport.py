"""The HTTP surface, without a checkpoint.

engine.speak is replaced here: these tests are about what the transport
does with what it gets back — the headers it sets, and the two ways a
render can end — not about audio. The render pipeline has its own file.
"""

import pytest
from fastapi.testclient import TestClient

import app as service
import engine


@pytest.fixture
def client():
    # loopback base_url on purpose: the Host check in app.py refuses
    # anything else, TestClient's own "testserver" included
    return TestClient(service.app, base_url="http://127.0.0.1")


# ------------------------------------------------------- the note header
#
# A header is latin-1 on the wire and the note is house-style prose, so
# these two facts met badly: an em dash in the message raised
# UnicodeEncodeError while the response was being built, and a render
# that had just succeeded came back as a 500. The advice may never cost
# the audio it is about.


@pytest.mark.parametrize("note", [
    "The model may have repeated a word — listen back, or Regenerate.",
    "Curly “quotes” and an en dash – too.",
])
def test_a_note_never_breaks_the_response(note, client, monkeypatch, tmp_path):
    wav = tmp_path / "out.wav"
    wav.write_bytes(b"RIFF0000WAVEfmt ")
    monkeypatch.setattr(service.engine, "speak", lambda *a, **k: wav)
    monkeypatch.setattr(service, "_stutter_check", lambda *a: note)
    res = client.post("/speak", json={"text": "Hello there.", "voice": "narrator_en"})
    assert res.status_code == 200
    assert res.headers["x-tts-note"].isascii()


def test_the_shipped_note_says_the_same_thing_in_ascii():
    """The message itself, not just the escape hatch — the fix was to
    write it in ASCII, and the sanitiser is the belt to that's braces."""
    assert service._header_text("word — listen") == "word - listen"


# ------------------------------------------------------- stopping a render


def test_a_stopped_render_answers_409_not_an_error(client, monkeypatch):
    def stopped(*a, **k):
        raise engine.RenderCancelled

    monkeypatch.setattr(service.engine, "speak", stopped)
    res = client.post("/speak", json={"text": "Stop me.", "voice": "narrator_en"})
    assert res.status_code == 409
    assert res.json()["detail"] == "Render stopped."


def test_cancel_says_whether_anything_was_running(client):
    res = client.post("/speak/cancel", json={"render_id": "nothing-doing"})
    assert res.status_code == 200 and res.json() == {"stopping": False}


def test_cancel_is_refused_from_another_site(client):
    """It is a state-changing POST with no body a page could not send,
    which is exactly what the Origin check exists for."""
    res = client.post(
        "/speak/cancel",
        json={"render_id": "x"},
        headers={"origin": "https://evil.example"},
    )
    assert res.status_code == 403
