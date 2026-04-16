"""
stl_filter.py
=============
For brackets that have multiple STL files (sub-assembly exports),
this module:
  1. Parses each STL to compute surface area and bounding box
  2. Classifies STLs as FASTENER (area < FASTENER_AREA_THRESHOLD) or STRUCTURAL
  3. Selects the best single structural STL to represent the bracket visually
  4. Returns metadata for all STLs with their classification

The fastener threshold is 1,000 mm² based on analysis of weld nut `6151515`
across multiple assemblies (consistently 736 mm²).
"""

import struct
import math
import os
import glob
from pathlib import Path


FASTENER_AREA_THRESHOLD = 1_000.0  # mm² — anything below is a fastener/small part


def parse_stl_binary(path: str) -> dict:
    """
    Parse a binary STL and return geometric statistics.

    Returns
    -------
    {
        "path": str,
        "filename": str,
        "n_triangles": int,
        "surface_area_mm2": float,
        "bbox_x_mm": float, "bbox_y_mm": float, "bbox_z_mm": float,
        "bbox_volume_mm3": float,
        "is_fastener": bool,
        "centroid": [float, float, float],
    }
    """
    path = str(path)
    with open(path, "rb") as f:
        header = f.read(80)
        n = struct.unpack("<I", f.read(4))[0]

        verts = []
        total_area = 0.0

        for _ in range(n):
            raw = f.read(50)
            if len(raw) < 50:
                break
            v1 = struct.unpack("<fff", raw[12:24])
            v2 = struct.unpack("<fff", raw[24:36])
            v3 = struct.unpack("<fff", raw[36:48])
            verts.extend([v1, v2, v3])

            # Triangle area via cross product
            ab = [v2[i] - v1[i] for i in range(3)]
            ac = [v3[i] - v1[i] for i in range(3)]
            cross = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ]
            total_area += 0.5 * math.sqrt(sum(c * c for c in cross))

    if not verts:
        return {"path": path, "filename": Path(path).name,
                "n_triangles": 0, "surface_area_mm2": 0.0,
                "bbox_x_mm": 0, "bbox_y_mm": 0, "bbox_z_mm": 0,
                "bbox_volume_mm3": 0, "is_fastener": True, "centroid": [0, 0, 0]}

    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]

    bx = round(max(xs) - min(xs), 2)
    by = round(max(ys) - min(ys), 2)
    bz = round(max(zs) - min(zs), 2)
    area = round(total_area, 1)

    centroid = [
        round((max(xs) + min(xs)) / 2, 2),
        round((max(ys) + min(ys)) / 2, 2),
        round((max(zs) + min(zs)) / 2, 2),
    ]

    return {
        "path":             path,
        "filename":         Path(path).name,
        "n_triangles":      n,
        "surface_area_mm2": area,
        "bbox_x_mm":        bx,
        "bbox_y_mm":        by,
        "bbox_z_mm":        bz,
        "bbox_volume_mm3":  round(bx * by * bz, 1),
        "is_fastener":      area < FASTENER_AREA_THRESHOLD,
        "centroid":         centroid,
    }


def select_main_stl(stl_paths: list[str]) -> dict:
    """
    Given a list of STL paths for a single bracket, select the one
    that represents the main structural component.

    Strategy:
      1. Parse all STLs
      2. Exclude fasteners (area < threshold)
      3. Among structural STLs, prefer the one with the largest surface area
         (most likely the main bracket body, not a small sub-component)

    Returns
    -------
    {
        "selected": <stl_dict for main STL>,
        "all_stls": [<stl_dict>, ...],
        "fasteners_excluded": int,
        "structural_candidates": int,
    }
    """
    all_parsed = [parse_stl_binary(p) for p in stl_paths]

    structural = [s for s in all_parsed if not s["is_fastener"]]
    fasteners  = [s for s in all_parsed if s["is_fastener"]]

    if not structural:
        # Fallback: if everything looks like a fastener, take the largest
        selected = max(all_parsed, key=lambda s: s["surface_area_mm2"])
    else:
        # Pick structural STL with largest surface area
        selected = max(structural, key=lambda s: s["surface_area_mm2"])

    return {
        "selected":              selected,
        "all_stls":              all_parsed,
        "fasteners_excluded":    len(fasteners),
        "structural_candidates": len(structural),
    }


def find_stls_for_part(part_id: str, stl_dir: str) -> list[str]:
    """
    Find all STL files in stl_dir that belong to part_id.
    Matches: <part_id>.STL, <part_id>_*.STL, <part_id>-*.STL
    """
    stl_dir = Path(stl_dir)
    patterns = [
        f"{part_id}.STL", f"{part_id}.stl",
        f"{part_id}_*.STL", f"{part_id}_*.stl",
        f"{part_id}-*.STL", f"{part_id}-*.stl",
        f"{part_id}_-_*.STL", f"{part_id}_-_*.stl",
    ]
    found = set()
    for pat in patterns:
        for match in stl_dir.glob(pat):
            found.add(str(match))
    return sorted(found)


if __name__ == "__main__":
    import sys
    stl_dir = str(Path(__file__).parent / "260410_EXPORT FILES")

    test_cases = [
        ("3750590",  ["3750590.STL"]),
        ("3750554R", ["3750554R_-_3750596R-1.STL",
                      "3750554R_-_4654983R-1_3195250-1.STL",
                      "3750554R_-_4654983R-1_6151515-1.STL",
                      "3750554R_-_4654983R-1_6151515-2.STL"]),
        ("3759698L", ["3759698L.STL"]),
        ("3753261R", ["3753261R_-_3753261R-1.STL",
                      "3753261R_-_6151515-1.STL"]),
    ]

    for part_id, filenames in test_cases:
        paths = [os.path.join(stl_dir, f) for f in filenames
                 if os.path.exists(os.path.join(stl_dir, f))]
        if not paths:
            print(f"\n{part_id}: no STL files found")
            continue

        result = select_main_stl(paths)
        sel = result["selected"]
        print(f"\n{'─'*55} {part_id}")
        print(f"  STLs found    : {len(result['all_stls'])}")
        print(f"  Fasteners excl: {result['fasteners_excluded']}")
        print(f"  Structural    : {result['structural_candidates']}")
        print(f"  SELECTED      : {sel['filename']}")
        print(f"    triangles   : {sel['n_triangles']}")
        print(f"    area mm²    : {sel['surface_area_mm2']}")
        print(f"    bbox mm     : {sel['bbox_x_mm']} × {sel['bbox_y_mm']} × {sel['bbox_z_mm']}")
        print(f"    is_fastener : {sel['is_fastener']}")
