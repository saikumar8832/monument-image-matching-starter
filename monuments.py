"""Local monument image-matching experiment, not a trained recognition model."""
from __future__ import annotations

import argparse
import json
import io
import sqlite3
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageDraw, UnidentifiedImageError

SIZE = 32
MAX_IMAGE_PIXELS = 20_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

# Fixed, hand-designed filters are illustrative; they are NOT learned weights.
FILTERS = np.array([
    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
], dtype=np.float32)


def extract_features(path: Path) -> np.ndarray:
    """Grayscale -> fixed convolution -> ReLU -> pooling -> descriptor."""
    try:
        with Image.open(path) as img:
            if img.width * img.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image is too large")
            gray = ImageOps.fit(img.convert("L"), (SIZE, SIZE))
            pixels = np.asarray(gray, dtype=np.float32) / 255.0
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Cannot read image: {path.name}") from exc
    windows = np.lib.stride_tricks.sliding_window_view(pixels, (3, 3))
    conv = np.einsum("ijxy,kxy->kij", windows, FILTERS)
    active = np.maximum(conv, 0)
    # 30x30 -> 10x10 by average pooling. Include grayscale intensity so a
    # constant black image is not equivalent to a constant white image.
    pooled = active.reshape(3, 10, 3, 10, 3).mean(axis=(2, 4))
    gray_small = pixels.reshape(8, 4, 8, 4).mean(axis=(1, 3))
    return np.concatenate([pooled.ravel(), gray_small.ravel()]).astype(np.float32)


def open_catalog(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE IF NOT EXISTS reference_images (
        id INTEGER PRIMARY KEY, label TEXT NOT NULL, image_path TEXT NOT NULL,
        details TEXT NOT NULL DEFAULT '', features TEXT NOT NULL, thumbnail BLOB NOT NULL
    )""")
    return conn


def add_reference(db: Path, image: Path, label: str, details: str = "") -> int:
    label = label.strip()
    if not label:
        raise ValueError("A label is required")
    features = extract_features(image)
    # Store a small reference thumbnail for the optional visual comparison.
    with Image.open(image) as original:
        thumbnail = ImageOps.contain(original.convert("RGB"), (300, 220))
        data = io.BytesIO()
        thumbnail.save(data, format="PNG")
        thumbnail_bytes = data.getvalue()
    with open_catalog(db) as conn:
        cursor = conn.execute(
            "INSERT INTO reference_images (label, image_path, details, features, thumbnail) VALUES (?, ?, ?, ?, ?)",
            (label, image.name, details.strip(), json.dumps(features.tolist()), thumbnail_bytes),
        )
        return int(cursor.lastrowid)


def match(db: Path, image: Path) -> dict:
    if not db.is_file():
        raise ValueError("No catalogue found. Add at least one reference first.")
    query = extract_features(image)
    with open_catalog(db) as conn:
        rows = conn.execute("SELECT id, label, image_path, details, features, thumbnail FROM reference_images").fetchall()
    if not rows:
        raise ValueError("No reference images. Add at least one reference first.")
    ranked = []
    for row_id, label, path, details, encoded, thumbnail in rows:
        vector = np.asarray(json.loads(encoded), dtype=np.float32)
        if vector.shape != query.shape:
            raise ValueError("Catalogue contains an incompatible descriptor")
        # Root mean squared feature distance, NOT a probability or confidence.
        distance = float(np.sqrt(np.mean((query - vector) ** 2)))
        ranked.append({"id": row_id, "label": label, "reference": path, "details": details, "distance": distance, "thumbnail": thumbnail})
    return min(ranked, key=lambda item: (item["distance"], item["id"]))


def save_comparison(query_path: Path, result: dict, output: Path) -> None:
    """Write a side-by-side comparison, not an AR or 3D visualization."""
    canvas = Image.new("RGB", (660, 300), "white")
    with Image.open(query_path) as source:
        query = ImageOps.contain(source.convert("RGB"), (300, 220))
    with Image.open(io.BytesIO(result["thumbnail"])) as source:
        reference = source.convert("RGB")
    canvas.paste(query, (15, 40))
    canvas.paste(reference, (345, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 10), "Query image", fill="black")
    draw.text((345, 10), "Nearest reference (may be wrong)", fill="black")
    # Arbitrary labels may contain characters unavailable in PIL's default font.
    label = result["label"].encode("ascii", "replace").decode("ascii")
    draw.text((345, 270), label[:38], fill="black")
    canvas.save(output, format="PNG")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Educational local image-matching experiment")
    parser.add_argument("--db", type=Path, default=Path("monuments.sqlite3"))
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add", help="Add a labelled reference image")
    add.add_argument("image", type=Path)
    add.add_argument("--label", required=True)
    add.add_argument("--details", default="", help="Manually supplied description")
    find = sub.add_parser("match", help="Find the nearest stored reference")
    find.add_argument("image", type=Path)
    find.add_argument("--output", type=Path, help="Save a side-by-side comparison PNG")
    args = parser.parse_args(argv)
    try:
        if args.command == "add":
            row_id = add_reference(args.db, args.image, args.label, args.details)
            print(f"Added reference #{row_id} to {args.db}")
        else:
            result = match(args.db, args.image)
            if args.output:
                save_comparison(args.image, result, args.output)
                print("Saved comparison:", args.output)
            print("Nearest reference:", result["label"])
            print("Reference filename:", result["reference"])
            print(f"Feature distance: {result['distance']:.4f} (lower is closer; not confidence)")
            print("Details:", result["details"] or "No details supplied")
            print("Warning: nearest match can be wrong, even for unrelated images.")
        return 0
    except (ValueError, sqlite3.Error, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
