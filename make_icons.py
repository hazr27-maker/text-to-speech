#!/usr/bin/env python3
"""Regenerate the menu bar and app-bundle icons from a single glyph.

Two different objects, drawn from one shape:

  assets/iconTemplate.png     18x18   menu bar, @1x
  assets/iconTemplate@2x.png  36x36   menu bar, @2x
  assets/TTS.icns             16..1024 Finder / Gatekeeper / Login Items

The menu bar pair are *template* images -- pure black on transparency,
no colour. That is what lets macOS recolour them itself for the light
bar, the dark bar and the clicked-and-highlighted state; a colour icon
would be wrong in at least one of the three. The `Template` filename
suffix is the signal AppKit looks for.

    python make_icons.py             # built-in microphone glyph
    python make_icons.py art.png     # or refit your own artwork

A supplied PNG is used for its *silhouette* only (its alpha channel is
the shape), so give it black-on-transparent at 1024px or larger.
"""

import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
RES = 2048  # supersample, then downsample once with LANCZOS

MENUBAR_PT = 20  # rumps pins the status-item image to 20x20pt (rumps.py:128)
BG = (24, 26, 30, 255)  # app-icon squircle fill


def _glyph(res: int = RES) -> Image.Image:
    """Studio microphone on a stand, black on transparency, cropped to its ink.

    The previous mark was a speaker cone with sound waves, which reads as
    a volume control at menu bar size. A mic capsule + cradle + stand is
    the one silhouette everyone parses as "speech" at 18px.
    """
    img = Image.new("RGBA", (res, res), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    u = lambda v: v * res
    ink = (0, 0, 0, 255)

    # capsule -- a solid pill; any grille detail vanishes at 18px
    d.rounded_rectangle(
        [u(0.37), u(0.06), u(0.63), u(0.58)], radius=u(0.13), fill=ink
    )
    # cradle arc hugging the capsule's lower half
    cx, cy, rad = u(0.50), u(0.44), u(0.245)
    d.arc(
        [cx - rad, cy - rad, cx + rad, cy + rad],
        -15, 195, fill=ink, width=int(u(0.07)),
    )
    # stem down from the cradle, then the base bar
    d.rectangle([u(0.465), u(0.66), u(0.535), u(0.88)], fill=ink)
    d.rounded_rectangle(
        [u(0.33), u(0.88), u(0.67), u(0.95)], radius=u(0.035), fill=ink
    )
    return img.crop(img.getbbox())


def _fit(glyph: Image.Image, size: int, pad: int) -> Image.Image:
    """Centre `glyph` in a size x size canvas, inset by `pad`."""
    g = glyph.copy()
    g.thumbnail((size - 2 * pad, size - 2 * pad), Image.LANCZOS)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(g, ((size - g.width) // 2, (size - g.height) // 2), g)
    return out


def _tint(img: Image.Image, rgb: tuple[int, int, int]) -> Image.Image:
    solid = Image.new("RGBA", img.size, rgb + (255,))
    solid.putalpha(img.getchannel("A"))
    return solid


def _app_icon(glyph: Image.Image, size: int = 1024) -> Image.Image:
    """Squircle plate with the glyph knocked out in white."""
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    inset = round(size * 0.098)  # macOS leaves the plate short of the edge
    plate = size - 2 * inset
    ImageDraw.Draw(out).rounded_rectangle(
        [inset, inset, size - inset, size - inset],
        radius=round(plate * 0.2237),  # Apple's continuous-corner ratio
        fill=BG,
    )
    mark = _tint(_fit(glyph, plate, round(plate * 0.23)), (255, 255, 255))
    out.paste(mark, (inset, inset), mark)
    return out


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    if len(sys.argv) > 1:
        src = Image.open(sys.argv[1]).convert("RGBA")
        glyph = src.crop(src.getbbox())
    else:
        glyph = _glyph()

    # --- menu bar: template images, @1x and @2x (macOS has no @3x) ---
    for scale, name in ((1, "iconTemplate.png"), (2, "iconTemplate@2x.png")):
        size = MENUBAR_PT * scale
        icon = _tint(_fit(glyph, size, round(size * 0.06)), (0, 0, 0))
        icon.save(ASSETS / name)
        print(f"  {name}  {size}x{size}")

    # --- app bundle: .icns via iconutil ---
    base = _app_icon(glyph)
    iconset = ASSETS / "TTS.iconset"
    shutil.rmtree(iconset, ignore_errors=True)
    iconset.mkdir()
    for pt in (16, 32, 128, 256, 512):
        for scale in (1, 2):
            px = pt * scale
            suffix = "@2x" if scale == 2 else ""
            base.resize((px, px), Image.LANCZOS).save(
                iconset / f"icon_{pt}x{pt}{suffix}.png"
            )
    subprocess.run(
        ["iconutil", "-c", "icns", str(iconset), "-o", str(ASSETS / "TTS.icns")],
        check=True,
    )
    shutil.rmtree(iconset)
    print(f"  TTS.icns  16..1024")


if __name__ == "__main__":
    main()
