"""py2app build for the menu bar launcher.

    python setup_app.py py2app        ->  dist/Text to Speech.app

A thin bundle. It does *not* contain the model, the venv, or MLX -- it
runs the project's own .venv/bin/uvicorn as a child process. The project
path is stamped into Info.plist so the .app keeps working if it's
dragged to /Applications, and menubar.py can still be pointed somewhere
else at runtime (Choose Project Folder…).
"""

from pathlib import Path

from setuptools import setup

ROOT = Path(__file__).resolve().parent

setup(
    app=["menubar.py"],
    options={
        "py2app": {
            "iconfile": str(ROOT / "assets" / "TTS.icns"),
            "resources": [str(ROOT / "assets")],
            "packages": ["rumps"],
            "plist": {
                "CFBundleName": "Text to Speech",
                "CFBundleDisplayName": "Text to Speech",
                "CFBundleIdentifier": "local.tts.menubar",
                "CFBundleShortVersionString": "1.0",
                "CFBundleVersion": "1.0",
                # agent app: status item only -- no Dock icon, no app menus
                "LSUIElement": True,
                # the floor is MLX's, not ours; the menu bar code is happy
                # much further back
                "LSMinimumSystemVersion": "13.5",
                "NSHighResolutionCapable": True,
                "TTSProjectRoot": str(ROOT),
            },
        }
    },
    setup_requires=["py2app"],
)
