#!/usr/bin/env python3
"""Assemble the hero GIF from eight real Control Room frames (S24, T15).

`scripts/capture_screenshots.py --hero-output DIR` writes the eight ordered frames; this
script fits each one into a 1280x800 canvas, quantises the palette and writes an
optimised, looping GIF at 1.2 s per frame with a 3 MB budget::

    uv run --group capture python scripts/capture_screenshots.py --hero-output /tmp/mll-hero
    uv run --group capture python scripts/make_gif.py --frames-dir /tmp/mll-hero
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "assets" / "hero.gif"
DEFAULT_FRAMES = ROOT / "docs" / "assets" / "screenshots"
CANVAS = (1280, 800)
FRAME_DURATION_MS = 1200
GIF_BUDGET_BYTES = 3_000_000
EXPECTED_FRAMES = 8


def _fit(image: Image.Image) -> Image.Image:
    """Resize into the hero canvas without distorting or clipping the frame."""
    rgb = image.convert("RGB")
    contained = ImageOps.contain(rgb, CANVAS, Image.Resampling.LANCZOS)
    background = contained.getpixel((0, 0))
    canvas = Image.new("RGB", CANVAS, background)
    canvas.paste(
        contained,
        ((CANVAS[0] - contained.width) // 2, (CANVAS[1] - contained.height) // 2),
    )
    return canvas


def _frame_paths(frames_dir: Path, *, limit: int = EXPECTED_FRAMES) -> list[Path]:
    paths = sorted(frames_dir.glob("*.png"))
    if len(paths) < limit:
        raise SystemExit(
            f"need {limit} PNG frames in {frames_dir}, found {len(paths)}; run "
            "`scripts/capture_screenshots.py --hero-output DIR` first"
        )
    return paths[:limit]


def _save(frames: list[Image.Image], output: Path, colors: int) -> int:
    palette = [frame.quantize(colors=colors, method=Image.Quantize.MEDIANCUT) for frame in frames]
    output.parent.mkdir(parents=True, exist_ok=True)
    palette[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=palette[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return output.stat().st_size


def build(frames_dir: Path, output: Path = DEFAULT_OUTPUT) -> Path:
    """Build the GIF and keep reducing the palette until it fits the size budget."""
    paths = _frame_paths(frames_dir)
    with Image.open(paths[0]) as first:
        frames = [_fit(first)] + [_fit(Image.open(path).convert("RGB")) for path in paths[1:]]
    for colors in (256, 128, 64):
        size = _save(frames, output, colors)
        if size <= GIF_BUDGET_BYTES:
            print(f"wrote {output} ({len(frames)} frames, {size} bytes, {colors} colours)")
            return output
    raise SystemExit(f"{output} exceeds the {GIF_BUDGET_BYTES}-byte GIF budget")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build(args.frames_dir, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
