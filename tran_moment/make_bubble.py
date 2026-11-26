"""Launcher bubble loop (/avatar_bubble.mp4 = prepared/bubble.mp4): the puppet's own portrait, 200x200, 12 s at 24 fps,
muted h264, ~40-70 KB. Gentle breathing (a 0.5% scale and a 1 px rise, 3 cycles), two blinks from the puppet's
eyes_half / eyes_closed layers, no warp of the face, a seamless loop (every motion is periodic over the 12 s).

    uv run --no-project --python 3.13 --with numpy --with pillow python tran_moment/make_bubble.py
    (needs ffmpeg on PATH; run after generate_mouths.py and generate_reactions.py)
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
FPS, SECONDS, SIZE = 24, 12, 200
BLINKS = (2.6, 8.1)          # s: blink starts
BLINK_LEN = 0.16             # s: close + open


def main() -> int:
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found", file=sys.stderr)
        return 1
    base = Image.open(HERE / "path_b" / "mouths" / "base.png").convert("RGBA")
    react = HERE / "path_b" / "reactions"
    half = Image.open(react / "eyes_half.png").convert("RGBA")
    closed = Image.open(react / "eyes_closed.png").convert("RGBA")
    p = json.loads((HERE / "prepared" / "face.json").read_text(encoding="utf-8"))["points"]
    eye_y = (p["eye_l_outer"][1] + p["eye_r_outer"][1]) / 2
    cx = (p["eye_l_outer"][0] + p["eye_r_outer"][0]) / 2
    side = (p["chin"][1] - eye_y) * 3.0                       # head with some hair and neck
    cy = eye_y + (p["chin"][1] - eye_y) * 0.42
    tmp = Path(tempfile.mkdtemp(prefix="bubble_"))
    try:
        n = FPS * SECONDS
        for i in range(n):
            t = i / FPS
            im = base.copy()
            for b in BLINKS:
                d = t - b
                if 0 <= d < BLINK_LEN:
                    k = math.sin(math.pi * d / BLINK_LEN)        # 0 -> 1 -> 0
                    im.alpha_composite(closed if k > 0.7 else half)
            breath = math.sin(2 * math.pi * t / 4.0)              # 3 cycles in 12 s
            s = side * (1 - 0.005 * breath)
            y = cy - 1.0 * breath
            box = (cx - s / 2, y - s / 2, cx + s / 2, y + s / 2)
            fr = im.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS, box=box)
            fr.save(tmp / f"{i:04d}.png")
        out = HERE / "prepared" / "bubble.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(tmp / "%04d.png"),
                        "-c:v", "libx264", "-preset", "slow", "-crf", "28", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        "-an", str(out)], check=True)
        print("wrote", out, out.stat().st_size // 1024, "KB")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
