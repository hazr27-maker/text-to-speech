#!/usr/bin/env python3
"""Menu bar front end for the local TTS service.

The status item *is* the app: no Dock icon, no window, and no Terminal
left open (`LSUIElement`, set in setup_app.py). It supervises a
`uvicorn` child running the same `app:app` the CLI runs, so the service
is untouched -- this is a launcher, not a second implementation.

Deliberately *not* a py2app bundle of the whole environment: MLX ships
Metal libraries that don't survive being copied into an .app, and the
venv is 660 MB. The bundle stays small and shells out to the project's
own `.venv/bin/uvicorn`.

    python menubar.py             # run in place, for development
    python setup_app.py py2app    # build "Text to Speech.app"
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import urllib.request
import webbrowser
from pathlib import Path

import rumps
from Foundation import NSNotificationCenter

HOST = "127.0.0.1"
BASE_PORT = 8123
PORT_SCAN = 20  # how far past BASE_PORT to look for a free one
POLL_SECS = 2.0

HERE = Path(__file__).resolve().parent
# 40px art displayed at 20pt: rumps pins the status-item image to 20x20
# points (rumps.py:128) and does no @2x lookup of its own, so handing it
# the @2x file is what keeps the icon crisp on a Retina display.
ICON = HERE / "assets" / "iconTemplate@2x.png"
SUPPORT = Path.home() / "Library" / "Application Support" / "LocalTTS"
CONFIG = SUPPORT / "config.json"
LOG = Path.home() / "Library" / "Logs" / "LocalTTS.log"


# --------------------------------------------------------------------
# preferences


def _config() -> dict:
    try:
        return json.loads(CONFIG.read_text())
    except (OSError, ValueError):
        return {}


def _save(**kw) -> None:
    cfg = _config() | kw
    SUPPORT.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(cfg, indent=2))


# --------------------------------------------------------------------
# locating the project


def _is_project(p: Path) -> bool:
    return (p / "app.py").is_file() and (p / ".venv" / "bin" / "uvicorn").is_file()


def _baked_root() -> Path | None:
    """Path stamped into Info.plist at build time, so the .app still
    works after being dragged to /Applications."""
    try:
        from Foundation import NSBundle

        raw = NSBundle.mainBundle().objectForInfoDictionaryKey_("TTSProjectRoot")
    except Exception:
        return None
    return Path(str(raw)) if raw else None


def find_project_root() -> Path | None:
    """An explicit choice wins; then the folder we're sitting in; then
    the build-time path. Returns None if none of them still look like
    the project, which is the cue to ask."""
    saved = _config().get("project_root")
    if saved and _is_project(Path(saved)):
        return Path(saved)
    # dev: this file is in the project root. Bundled: .../Contents/
    # Resources, so the walk up still finds it when the .app is left in
    # dist/ or dropped into the project folder.
    for cand in (HERE, *HERE.parents):
        if _is_project(cand):
            return cand
    baked = _baked_root()
    return baked if baked and _is_project(baked) else None


def alert(title: str, message: str) -> None:
    """rumps.alert from an agent app, brought to the front. Preferred
    over a notification, which needs a signed bundle and the user's
    permission and otherwise fails silently."""
    try:
        from AppKit import NSApp

        NSApp.activateIgnoringOtherApps_(True)
    except Exception:
        pass
    rumps.alert(title, message)


def ask_for_project_root() -> Path | None:
    """NSOpenPanel — an agent app has no windows, so it must be pulled
    to the front by hand or the panel opens behind everything."""
    from AppKit import NSApp, NSOpenPanel

    NSApp.activateIgnoringOtherApps_(True)
    panel = NSOpenPanel.openPanel()
    panel.setCanChooseFiles_(False)
    panel.setCanChooseDirectories_(True)
    panel.setAllowsMultipleSelection_(False)
    panel.setPrompt_("Choose")
    panel.setMessage_("Select the text-to-speech folder (the one containing app.py)")
    if panel.runModal() != 1:  # NSModalResponseOK
        return None
    chosen = Path(panel.URLs()[0].path())
    if not _is_project(chosen):
        alert(
            "That folder doesn't look right",
            "It should contain app.py and a .venv folder. "
            "If .venv is missing, run the setup step first.",
        )
        return None
    _save(project_root=str(chosen))
    return chosen


# --------------------------------------------------------------------
# the port


def health(port: int) -> dict | None:
    """/health from *our* service, or None if this port is someone else."""
    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}/health", timeout=0.6) as r:
            body = json.load(r)
        return body if body.get("backend") else None
    except Exception:
        return None


def _free(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex((HOST, port)) != 0


def _child_env() -> dict[str, str]:
    """The environment for the uvicorn child.

    py2app points PYTHONHOME/PYTHONPATH at the bundle's own trimmed
    interpreter. Inherited, that half-stdlib shadows the venv's and the
    child dies on `No module named 'logging.config'` -- a failure that
    only ever appears in the built .app, never when run from source. So
    the bundle's Python wiring is stripped back out here."""
    keep = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("PYTHON")
        and k not in ("RESOURCEPATH", "ARGVZERO", "EXECUTABLEPATH")
    }
    return keep | {"PYTHONUNBUFFERED": "1"}


def _pids_on(port: int) -> list[int]:
    out = subprocess.run(
        ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"], capture_output=True, text=True
    ).stdout
    return [int(p) for p in out.split()]


# --------------------------------------------------------------------


class TTSApp(rumps.App):
    def __init__(self) -> None:
        super().__init__(
            "Text to Speech", icon=str(ICON), template=True, quit_button=None
        )
        self.proc: subprocess.Popen | None = None
        self.port = BASE_PORT
        self.adopted = False  # a server we found running, not one we spawned
        self.open_when_ready = _config().get("open_at_launch", True)
        self.ready = False
        self.reported_crash = False
        self.health_misses = 0  # consecutive failed polls of an adopted server

        self.state = rumps.MenuItem("Starting…")  # no callback => greyed
        self.power = rumps.MenuItem("Stop Server", callback=self.toggle)
        self.autoopen = rumps.MenuItem(
            "Open Page When App Starts", callback=self.toggle_autoopen
        )
        self.autoopen.state = int(self.open_when_ready)
        self.menu = [
            self.state,
            rumps.MenuItem("Open Text to Speech", callback=self.open_ui, key="o"),
            None,
            self.power,
            self.autoopen,
            None,
            rumps.MenuItem("Show Log", callback=self.show_log),
            rumps.MenuItem("Choose Project Folder…", callback=self.pick_root),
            None,
            rumps.MenuItem("Quit Text to Speech", callback=self.quit_app, key="q"),
        ]

        self.start()
        # Quit from the menu runs quit_app, but a logout, a restart or
        # Force Quit terminates us without it -- and the child would
        # outlive us holding the port and 4 GB. Both hooks are
        # idempotent: stop() is a no-op once the child is gone.
        self._term_observer = (
            NSNotificationCenter.defaultCenter().addObserverForName_object_queue_usingBlock_(
                "NSApplicationWillTerminateNotification", None, None,
                lambda _note: self.stop(),
            )
        )
        # Python signal handlers only run between bytecodes, so this
        # lands on the next timer tick rather than instantly
        signal.signal(signal.SIGTERM, self._on_signal)

        # retained on purpose: rumps keeps timers in a WeakKeyDictionary
        # (rumps.py:31), so an unreferenced Timer is collected and
        # silently stops firing
        self.poll_timer = rumps.Timer(self.poll, POLL_SECS)
        self.poll_timer.start()

    # ---------------- lifecycle

    def start(self, _=None) -> None:
        root = find_project_root() or ask_for_project_root()
        if root is None:
            self.state.title = "Can't find the project folder"
            self.power.title = "Start Server"
            return

        # Someone is already serving in our port range. If it's us -- an
        # orphan from a hard quit, or a copy started from Terminal --
        # adopt it rather than starting a second copy of a model that
        # wants 4 GB of memory. The whole scan range is checked, not
        # just 8123: a previous run that had to move up a port leaves
        # its orphan there, exactly where a new spawn would never look.
        # (_free first: probing a closed local port fails instantly, so
        # only ports with an actual listener cost a health call.)
        for p in range(BASE_PORT, BASE_PORT + PORT_SCAN):
            if not _free(p) and health(p):
                self.port, self.adopted, self.proc = p, True, None
                self.on_ready()
                return

        self.port = next(
            (p for p in range(BASE_PORT, BASE_PORT + PORT_SCAN) if _free(p)), BASE_PORT
        )
        LOG.parent.mkdir(parents=True, exist_ok=True)
        # append-only with a health poll's lifetime of restarts behind
        # it, the log would grow forever -- roll it at ~1 MB, keeping
        # one previous file so a crash that caused the restart is kept
        try:
            if LOG.exists() and LOG.stat().st_size > 1_000_000:
                LOG.replace(LOG.with_name(LOG.name + ".old"))
        except OSError:
            pass
        log = LOG.open("a", buffering=1)
        log.write(f"\n=== {root} :{self.port} ===\n")
        self.proc = subprocess.Popen(
            [
                str(root / ".venv" / "bin" / "uvicorn"),
                "app:app",
                "--host", HOST,
                "--port", str(self.port),
                # keep the log about problems: a health poll every
                # POLL_SECS would otherwise bury real errors
                "--no-access-log",
            ],
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=_child_env(),
        )
        self.adopted = False
        self.ready = False
        self.reported_crash = False
        self.health_misses = 0
        self.open_when_ready = bool(self.autoopen.state)
        self.state.title = "Starting…"
        self.power.title = "Stop Server"

    def stop(self, _=None) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        elif self.adopted:
            # not our child, so no handle to it -- go via the port
            for pid in _pids_on(self.port):
                try:
                    os.kill(pid, 15)
                except OSError:
                    pass
        self.proc = None
        self.adopted = False
        self.ready = False
        self.state.title = "Not running"
        self.power.title = "Start Server"

    def toggle(self, _) -> None:
        self.stop() if (self.proc or self.adopted) else self.start()

    def _on_signal(self, *_) -> None:
        self.stop()
        rumps.quit_application()

    def quit_app(self, _) -> None:
        self.stop()
        rumps.quit_application()

    # ---------------- state

    def on_ready(self) -> None:
        self.ready = True
        self.state.title = f"Ready — {HOST}:{self.port}"
        self.power.title = "Stop Server"
        if self.open_when_ready:
            self.open_when_ready = False
            self.open_ui(None)

    def poll(self, _) -> None:
        if self.proc and self.proc.poll() is not None:  # died on its own
            if not self.reported_crash:
                self.reported_crash = True
                tail = ""
                try:
                    tail = "\n".join(LOG.read_text().splitlines()[-6:])
                except OSError:
                    pass
                alert(
                    "Text to Speech stopped",
                    "The server quit unexpectedly.\n\n"
                    + (tail or "Choose Show Log for details."),
                )
            self.proc = None
            self.state.title = "Not running"
            self.power.title = "Start Server"
            return

        if not (self.proc or self.adopted):
            return
        info = health(self.port)
        if info:
            self.health_misses = 0
            if not self.ready:
                self.on_ready()
            else:
                # the model loads lazily on the first render, so this line
                # is the only place the warm/cold distinction is visible
                warm = " · model loaded" if info.get("model_loaded") else ""
                self.state.title = f"Ready — {HOST}:{self.port}{warm}"
        elif self.adopted:
            # an adopted server left no process handle, so its death is
            # only visible here -- without this the menu says "Ready"
            # forever over a port nobody is listening on. Three misses,
            # not one: a poll can time out while the model is busy. No
            # alert either -- we didn't start it, and stopping it from
            # the Terminal it came from is a normal thing to do.
            self.health_misses += 1
            if self.health_misses >= 3 and _free(self.port):
                self.adopted = False
                self.ready = False
                self.health_misses = 0
                self.state.title = "Not running"
                self.power.title = "Start Server"

    # ---------------- menu actions

    def open_ui(self, _) -> None:
        webbrowser.open(f"http://{HOST}:{self.port}/")

    def toggle_autoopen(self, item) -> None:
        item.state = 0 if item.state else 1
        _save(open_at_launch=bool(item.state))

    def show_log(self, _) -> None:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        LOG.touch(exist_ok=True)
        subprocess.run(["open", "-a", "Console", str(LOG)])

    def pick_root(self, _) -> None:
        if ask_for_project_root():
            self.stop()
            self.start()


if __name__ == "__main__":
    TTSApp().run()
