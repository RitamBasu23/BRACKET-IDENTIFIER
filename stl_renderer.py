"""
stl_renderer.py
===============
Renders each bracket STL from 36 camera angles using trimesh + matplotlib,
producing a set of silhouette/depth images suitable for CNN training.

Why 36 angles?
  - 12 azimuth positions × 3 elevation angles
  - Covers the full viewing sphere without redundancy
  - Matches the strategy from the Springer 2023 CAD augmentation paper

Output per bracket:
  renders/<part_id>/angle_00.png ... angle_35.png   (base renders)
  renders/<part_id>/aug_*.png                        (augmented variants)

Image format: 224×224 RGB PNG (ResNet/EfficientNet compatible)
"""

import os
import math
import json
import struct
import numpy as np
from pathlib import Path
from io import BytesIO
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageFilter, ImageEnhance

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE        = 224          # output image size in pixels
N_AZIMUTHS      = 12           # camera positions around the bracket
N_ELEVATIONS    = 3            # elevation angles (low / mid / high)
N_BASE_RENDERS  = N_AZIMUTHS * N_ELEVATIONS   # = 36
AUG_PER_RENDER  = 4           # augmentations per base render → 36×4 = 144 total
RENDER_DPI      = 72           # matplotlib DPI for rendering

# Elevation angles in degrees (from horizontal)
ELEVATION_ANGLES = [15, 35, 60]

# Face colour for 3D rendering
FACE_COLOUR = (0.55, 0.65, 0.75, 0.9)   # steel-blue, slightly transparent
EDGE_COLOUR = (0.2, 0.2, 0.2, 0.3)


def load_stl_mesh(stl_path: str):
    """
    Load an STL binary file and return:
      vertices: np.ndarray [N,3]
      triangles: np.ndarray [M,3,3]  (M triangles, each with 3 xyz vertices)
      center: np.ndarray [3]
      scale: float  (max extent, for normalisation)
    """
    with open(stl_path, "rb") as f:
        f.read(80)                          # header
        n = struct.unpack("<I", f.read(4))[0]
        tris = []
        for _ in range(n):
            raw = f.read(50)
            v1 = struct.unpack("<fff", raw[12:24])
            v2 = struct.unpack("<fff", raw[24:36])
            v3 = struct.unpack("<fff", raw[36:48])
            tris.append([v1, v2, v3])

    triangles = np.array(tris, dtype=np.float32)   # [M, 3, 3]
    vertices  = triangles.reshape(-1, 3)

    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    scale  = float(extent.max())

    # Normalise to unit cube centred at origin
    triangles = (triangles - center) / (scale + 1e-8)

    return triangles


def render_angle(triangles: np.ndarray, azimuth_deg: float,
                 elevation_deg: float, img_size: int = IMG_SIZE) -> Image.Image:
    """
    Render a mesh from a specific camera angle using matplotlib 3D.

    Returns a PIL Image (RGB, img_size × img_size).
    """
    fig = plt.figure(figsize=(2, 2), dpi=img_size // 2)
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()

    # Build Poly3DCollection
    poly = Poly3DCollection(triangles,
                            facecolor=FACE_COLOUR,
                            edgecolor=EDGE_COLOUR,
                            linewidth=0.1)
    ax.add_collection3d(poly)

    # Set axis limits
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_zlim(-0.6, 0.6)

    # Set camera
    ax.view_init(elev=elevation_deg, azim=azimuth_deg)

    # Remove margins
    plt.tight_layout(pad=0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Render to PIL
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=img_size // 2,
                bbox_inches="tight", pad_inches=0,
                facecolor="white")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    img = img.resize((img_size, img_size), Image.LANCZOS)
    plt.close(fig)

    return img


def augment_image(img: Image.Image, aug_idx: int) -> Image.Image:
    """
    Apply a deterministic augmentation based on aug_idx (0-3):
      0 — horizontal flip
      1 — brightness +20%
      2 — slight Gaussian blur (simulates camera defocus)
      3 — brightness -15% + slight rotation (±5°)
    """
    if aug_idx == 0:
        return img.transpose(Image.FLIP_LEFT_RIGHT)

    elif aug_idx == 1:
        return ImageEnhance.Brightness(img).enhance(1.20)

    elif aug_idx == 2:
        return img.filter(ImageFilter.GaussianBlur(radius=0.8))

    elif aug_idx == 3:
        img = ImageEnhance.Brightness(img).enhance(0.85)
        img = img.rotate(5, fillcolor=(255, 255, 255))
        return img

    return img


def render_bracket(stl_path: str, output_dir: str,
                   augment: bool = True,
                   verbose: bool = False) -> dict:
    """
    Render a bracket STL from all 36 angles, optionally augmenting.

    Parameters
    ----------
    stl_path   : path to the structural STL file
    output_dir : directory to save images (created if missing)
    augment    : if True, generate AUG_PER_RENDER augmented images per angle
    verbose    : print progress

    Returns
    -------
    {
        "stl_path": str,
        "output_dir": str,
        "n_base_renders": int,
        "n_augmented": int,
        "image_paths": [str, ...]   (all saved image paths)
    }
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    triangles  = load_stl_mesh(stl_path)
    image_paths = []
    angle_idx   = 0

    for elev in ELEVATION_ANGLES:
        for az_step in range(N_AZIMUTHS):
            azimuth = az_step * (360 // N_AZIMUTHS)

            img = render_angle(triangles, azimuth, elev)

            # Save base render
            fname = f"angle_{angle_idx:02d}_az{azimuth:03d}_el{elev:02d}.png"
            fpath = str(out / fname)
            img.save(fpath)
            image_paths.append(fpath)

            if verbose:
                print(f"  Rendered angle {angle_idx:02d}/{N_BASE_RENDERS-1} "
                      f"(az={azimuth}° el={elev}°) → {fname}")

            # Augmented variants
            if augment:
                for aug_i in range(AUG_PER_RENDER):
                    aug_img = augment_image(img, aug_i)
                    aug_fname = f"aug_{angle_idx:02d}_{aug_i}.png"
                    aug_path  = str(out / aug_fname)
                    aug_img.save(aug_path)
                    image_paths.append(aug_path)

            angle_idx += 1

    n_base = N_BASE_RENDERS
    n_aug  = N_BASE_RENDERS * AUG_PER_RENDER if augment else 0

    return {
        "stl_path":      stl_path,
        "output_dir":    str(out),
        "n_base_renders": n_base,
        "n_augmented":   n_aug,
        "n_total":       n_base + n_aug,
        "image_paths":   image_paths,
    }


def render_all_brackets(db_path: str, renders_root: str,
                        augment: bool = True) -> dict:
    """
    Render all quality-3 brackets found in brackets.json.

    Returns a dict: part_id → render result
    """
    with open(db_path) as f:
        db = json.load(f)

    renders_root = Path(renders_root)
    all_results  = {}

    trainable = [
        (pid, r) for pid, r in db.items()
        if r.get("data_quality", 0) == 3 and r.get("stl") is not None
    ]

    print(f"Rendering {len(trainable)} brackets ...")

    for pid, record in trainable:
        stl_path = record["stl"]["selected"]["path"]
        out_dir  = str(renders_root / pid)

        # Skip if already rendered
        if Path(out_dir).exists() and len(list(Path(out_dir).glob("*.png"))) > 30:
            n = len(list(Path(out_dir).glob("*.png")))
            print(f"  {pid}: already rendered ({n} images) — skipping")
            all_results[pid] = {"output_dir": out_dir, "n_total": n, "skipped": True}
            continue

        print(f"  {pid}: rendering ...", end=" ", flush=True)
        try:
            result = render_bracket(stl_path, out_dir, augment=augment)
            all_results[pid] = result
            print(f"{result['n_total']} images saved")
        except Exception as e:
            print(f"ERROR: {e}")
            all_results[pid] = {"error": str(e)}

    # Save render manifest
    manifest_path = renders_root / "render_manifest.json"
    with open(manifest_path, "w") as f:
        # Save just the summary (not all image_paths to keep file small)
        summary = {
            pid: {k: v for k, v in res.items() if k != "image_paths"}
            for pid, res in all_results.items()
        }
        json.dump(summary, f, indent=2)

    print(f"\nManifest saved: {manifest_path}")
    return all_results


if __name__ == "__main__":
    import sys

    _base        = Path(__file__).parent
    db_path      = str(_base / "data" / "brackets.json")
    renders_root = str(_base / "data" / "renders")

    # Quick test: render just one bracket
    if "--test" in sys.argv:
        print("Test render: 3750590 from 2 angles only ...")
        tri = load_stl_mesh(str(_base / "260410_EXPORT FILES" / "3750590.STL"))
        img = render_angle(tri, azimuth_deg=45, elevation_deg=30)
        test_path = str(_base / "data" / "test_render.png")
        img.save(test_path)
        print(f"Saved test render: {test_path}")
        sys.exit(0)

    # Full render of all brackets
    augment = "--no-aug" not in sys.argv
    render_all_brackets(db_path, renders_root, augment=augment)
