"""Generate the seamless Ubuntu-mark background tile used by the dashboard.

Run from the repository root with:
    .venv/Scripts/python.exe dashboard/assets/make_bg_tile.py
"""

from pathlib import Path
import random

from PIL import Image, ImageEnhance


ASSETS_DIR = Path(__file__).resolve().parent
MARK_PATH = ASSETS_DIR / "ubuntu_mark.png"
OUTPUT_PATH = ASSETS_DIR / "ubuntu_bg_tile.png"

TILE_SIZE = 900
MARK_SIZES = [72, 104, 142]
OPACITIES = [10, 14, 18]

random.seed(42)


def softened_mark(mark: Image.Image, size: int, opacity: int) -> Image.Image:
    """Resize the supplied mark without changing its shape or brand colors."""
    aspect_ratio = mark.height / mark.width
    resized = mark.resize(
        (size, round(size * aspect_ratio)),
        Image.Resampling.LANCZOS,
    )
    resized = ImageEnhance.Color(resized).enhance(0.82)
    alpha = resized.getchannel("A").point(lambda value: value * opacity // 100)
    resized.putalpha(alpha)
    return resized


def stamp_wrapped(canvas: Image.Image, sprite: Image.Image, cx: int, cy: int) -> None:
    """Stamp across opposite edges so the finished image tiles seamlessly."""
    width, height = sprite.size
    x, y = cx - width // 2, cy - height // 2
    for x_offset in (-TILE_SIZE, 0, TILE_SIZE):
        for y_offset in (-TILE_SIZE, 0, TILE_SIZE):
            canvas.alpha_composite(sprite, (x + x_offset, y + y_offset))


def main() -> None:
    mark = Image.open(MARK_PATH).convert("RGBA")
    canvas = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))

    grid_size = 4
    cell_size = TILE_SIZE / grid_size
    for row in range(grid_size):
        for column in range(grid_size):
            sprite = softened_mark(
                mark,
                random.choice(MARK_SIZES),
                random.choice(OPACITIES),
            )
            sprite = sprite.rotate(
                random.uniform(-32, 32),
                expand=True,
                resample=Image.Resampling.BICUBIC,
            )
            center_x = int(
                (column + 0.5) * cell_size
                + random.uniform(-cell_size * 0.25, cell_size * 0.25)
            )
            center_y = int(
                (row + 0.5) * cell_size
                + random.uniform(-cell_size * 0.25, cell_size * 0.25)
            )
            stamp_wrapped(canvas, sprite, center_x, center_y)

    canvas.save(OUTPUT_PATH, optimize=True)
    print(f"Saved tile: {OUTPUT_PATH} ({TILE_SIZE}x{TILE_SIZE})")


if __name__ == "__main__":
    main()
