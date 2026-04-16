"""
step_extractor.py
=================
Parses a STEP (ISO 10303-21) file and extracts:
  - Entity type histogram (feature vector for GNN node features)
  - Face adjacency graph (nodes = ADVANCED_FACEs, edges = shared EDGE_CURVEs)
  - Scalar geometric summary (face counts by type, bounding stats)

Output: dict ready to be stored in brackets.json
"""

import re
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


# ── Entity types we care about ────────────────────────────────────────────────
FACE_TYPES = {
    "PLANE", "CYLINDRICAL_SURFACE", "CONICAL_SURFACE", "SPHERICAL_SURFACE",
    "TOROIDAL_SURFACE", "B_SPLINE_SURFACE_WITH_KNOTS",
    "SURFACE_OF_REVOLUTION", "SURFACE_OF_LINEAR_EXTRUSION",
}

EDGE_TYPES = {"LINE", "CIRCLE", "ELLIPSE", "B_SPLINE_CURVE_WITH_KNOTS",
              "TRIMMED_CURVE", "EDGE_CURVE"}

# Node feature order (index must stay fixed across all brackets)
NODE_FEATURE_KEYS = [
    "PLANE", "CYLINDRICAL_SURFACE", "CONICAL_SURFACE", "SPHERICAL_SURFACE",
    "TOROIDAL_SURFACE", "B_SPLINE_SURFACE_WITH_KNOTS",
    "SURFACE_OF_REVOLUTION", "SURFACE_OF_LINEAR_EXTRUSION",
]

# Global entity feature vector (graph-level, not per-node)
GLOBAL_FEATURE_KEYS = [
    "total_entities", "ADVANCED_FACE", "CYLINDRICAL_SURFACE", "PLANE",
    "CIRCLE", "B_SPLINE_CURVE_WITH_KNOTS", "EDGE_CURVE", "VERTEX_POINT",
    "ORIENTED_EDGE", "AXIS2_PLACEMENT_3D", "CARTESIAN_POINT",
]


def parse_step(path: str) -> dict:
    """
    Parse a STEP file and return a structured dict.

    Returns
    -------
    {
        "part_id": str,
        "step_path": str,
        "global_features": {entity_type: count, ...},
        "face_graph": {
            "nodes": [{"id": int, "surface_type": str, "feature_vec": [float...]}, ...],
            "edges": [[src_id, dst_id], ...]
        },
        "scalar_summary": {
            "n_faces": int, "n_cylinders": int, "n_planes": int,
            "n_bsplines": int, "n_circles": int, "entity_count": int,
            "cylinder_ratio": float, "bspline_ratio": float
        }
    }
    """
    path = Path(path)
    part_id = path.stem

    with open(path, "r", errors="ignore") as f:
        content = f.read()

    # ── 1. Parse all entity lines: #ID = TYPE(args) ────────────────────────
    entity_map: dict[int, tuple[str, str]] = {}   # id → (type, raw_args)
    pattern = re.compile(
        r"#(\d+)\s*=\s*([A-Z_][A-Z0-9_]*)\s*\(([^;]*)\)\s*;",
        re.DOTALL
    )
    for m in pattern.finditer(content):
        eid = int(m.group(1))
        etype = m.group(2).strip()
        eargs = m.group(3).strip()
        entity_map[eid] = (etype, eargs)

    # ── 2. Global histogram ────────────────────────────────────────────────
    type_counts = Counter(etype for etype, _ in entity_map.values())

    global_features = {k: type_counts.get(k, 0) for k in GLOBAL_FEATURE_KEYS}
    global_features["total_entities"] = len(entity_map)

    # ── 3. Build face adjacency graph ─────────────────────────────────────
    #
    # ADVANCED_FACE → FACE_OUTER_BOUND / FACE_BOUND
    #   → EDGE_LOOP → list of ORIENTED_EDGE
    #     → EDGE_CURVE
    #
    # Two faces are adjacent if they share an EDGE_CURVE.

    # Collect ADVANCED_FACEs and their surface type
    face_ids: list[int] = []
    face_surface_type: dict[int, str] = {}

    for eid, (etype, eargs) in entity_map.items():
        if etype == "ADVANCED_FACE":
            # ADVANCED_FACE(name, [bounds], surface_ref, sense)
            # surface_ref is a #ref to the geometry entity
            refs = re.findall(r"#(\d+)", eargs)
            surface_type = "UNKNOWN"
            for r in refs:
                r = int(r)
                if r in entity_map:
                    st = entity_map[r][0]
                    if st in FACE_TYPES:
                        surface_type = st
                        break
            face_ids.append(eid)
            face_surface_type[eid] = surface_type

    # For each ADVANCED_FACE, collect the set of EDGE_CURVE ids it references
    def get_edge_curves(face_eid: int) -> set[int]:
        """Walk ADVANCED_FACE → bounds → EDGE_LOOP → ORIENTED_EDGE → EDGE_CURVE"""
        result: set[int] = set()
        face_args = entity_map[face_eid][1]
        bound_refs = re.findall(r"#(\d+)", face_args)
        for br in bound_refs:
            br = int(br)
            if br not in entity_map:
                continue
            bt, ba = entity_map[br]
            if bt not in ("FACE_OUTER_BOUND", "FACE_BOUND"):
                continue
            loop_refs = re.findall(r"#(\d+)", ba)
            for lr in loop_refs:
                lr = int(lr)
                if lr not in entity_map:
                    continue
                lt, la = entity_map[lr]
                if lt != "EDGE_LOOP":
                    continue
                oe_refs = re.findall(r"#(\d+)", la)
                for oer in oe_refs:
                    oer = int(oer)
                    if oer not in entity_map:
                        continue
                    ot, oa = entity_map[oer]
                    if ot != "ORIENTED_EDGE":
                        continue
                    ec_refs = re.findall(r"#(\d+)", oa)
                    for ecr in ec_refs:
                        ecr = int(ecr)
                        if ecr in entity_map and entity_map[ecr][0] == "EDGE_CURVE":
                            result.add(ecr)
        return result

    # Map: edge_curve_id → list of face_ids that share it
    edge_to_faces: dict[int, list[int]] = defaultdict(list)
    face_edge_cache: dict[int, set[int]] = {}

    for fid in face_ids:
        ec_set = get_edge_curves(fid)
        face_edge_cache[fid] = ec_set
        for ec in ec_set:
            edge_to_faces[ec].append(fid)

    # Build adjacency list
    adj_pairs: set[tuple[int, int]] = set()
    for ec, faces in edge_to_faces.items():
        if len(faces) == 2:
            a, b = sorted(faces)
            adj_pairs.add((a, b))

    # Remap face IDs to 0-indexed node IDs
    face_index = {fid: i for i, fid in enumerate(face_ids)}

    def feature_vec(surface_type: str) -> list[float]:
        return [1.0 if k == surface_type else 0.0 for k in NODE_FEATURE_KEYS]

    nodes = [
        {
            "id": face_index[fid],
            "step_entity_id": fid,
            "surface_type": face_surface_type[fid],
            "feature_vec": feature_vec(face_surface_type[fid]),
        }
        for fid in face_ids
    ]

    edges = [
        [face_index[a], face_index[b]]
        for a, b in adj_pairs
    ]

    # ── 4. Scalar summary ──────────────────────────────────────────────────
    n_faces = len(face_ids)
    n_cyl = type_counts.get("CYLINDRICAL_SURFACE", 0)
    n_pln = type_counts.get("PLANE", 0)
    n_bsp = type_counts.get("B_SPLINE_CURVE_WITH_KNOTS", 0)
    n_cir = type_counts.get("CIRCLE", 0)

    scalar_summary = {
        "entity_count":   len(entity_map),
        "n_faces":        n_faces,
        "n_cylinders":    n_cyl,
        "n_planes":       n_pln,
        "n_bsplines":     n_bsp,
        "n_circles":      n_cir,
        "n_edges":        len(adj_pairs),
        "cylinder_ratio": round(n_cyl / n_faces, 4) if n_faces else 0.0,
        "bspline_ratio":  round(n_bsp / n_faces, 4) if n_faces else 0.0,
        "plane_ratio":    round(n_pln / n_faces, 4) if n_faces else 0.0,
    }

    return {
        "part_id":        part_id,
        "step_path":      str(path),
        "global_features": global_features,
        "face_graph": {
            "nodes": nodes,
            "edges": edges,
        },
        "scalar_summary": scalar_summary,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python step_extractor.py <file.step>")
        sys.exit(1)

    result = parse_step(sys.argv[1])
    print(f"\nPart:          {result['part_id']}")
    print(f"Entities:      {result['scalar_summary']['entity_count']}")
    print(f"Faces (nodes): {result['scalar_summary']['n_faces']}")
    print(f"Adjacencies:   {result['scalar_summary']['n_edges']} edge pairs")
    print(f"Cylinders:     {result['scalar_summary']['n_cylinders']} ({result['scalar_summary']['cylinder_ratio']*100:.1f}%)")
    print(f"Planes:        {result['scalar_summary']['n_planes']} ({result['scalar_summary']['plane_ratio']*100:.1f}%)")
    print(f"B-splines:     {result['scalar_summary']['n_bsplines']}")
    print(f"Sample nodes:  {result['face_graph']['nodes'][:3]}")
    print(f"Sample edges:  {result['face_graph']['edges'][:5]}")
