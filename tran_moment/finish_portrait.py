"""After prepare_portrait.py on a flattened portrait: the stills the card and the launcher use.

    uv run --no-project --python 3.12 --with mediapipe --with numpy --with pillow \
        python tran_moment/finish_portrait.py --source tran_moment/prepared/face_v4_source.png --flat tran_moment/prepared/face_v4_flat.png

    prepared/face.png        RGBA cut-out, framed exactly like face.jpg (Simli face generation, designers)
    prepared/face_black.jpg  the still served as /avatar.jpg (launcher disc, header thumbnail, card before the puppet
                             loads). The name is historical: it is the portrait on the card's own background, which for
                             the 2026-11-25 portrait is the soft studio background of face.jpg, not black.

The puppet's base (path_b/mouths/base.png) stays opaque (generate_mouths.py): with an opaque base no mouth patch can
carry a different background over the jaw line. path_b/apply_alpha.py is the older cut-out-on-black variant.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare_portrait as pp  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True, help="levelled RGBA portrait (level_portrait.py)")
    ap.add_argument("--flat", type=Path, required=True, help="the flattened copy prepare_portrait.py was run on")
    a = ap.parse_args()
    src = ImageOps.exif_transpose(Image.open(a.source)).convert("RGBA")
    flat = ImageOps.exif_transpose(Image.open(a.flat)).convert("RGB")
    p = pp.landmarks(flat)                                   # the same points prepare_portrait.py framed with
    framed_flat = pp.frame(flat, p)
    face_jpg = Image.open(HERE / "prepared" / "face.jpg").convert("RGB")
    diff = float(np.abs(np.asarray(framed_flat, np.int16) - np.asarray(face_jpg, np.int16)).mean())
    if diff > 4:
        print(f"framing does not match prepared/face.jpg (mean diff {diff:.1f}); re-run prepare_portrait.py on --flat", file=sys.stderr)
        return 1
    colour = pp.frame(src.convert("RGB"), p)
    alpha = pp.frame(src.getchannel("A").convert("RGB"), p).getchannel("R")
    cut = colour.copy()
    cut.putalpha(alpha)
    cut.save(HERE / "prepared" / "face.png")
    framed_flat.save(HERE / "prepared" / "face_black.jpg", quality=92)
    print(f"wrote prepared/face.png (RGBA) and prepared/face_black.jpg (/avatar.jpg); framing match {diff:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
