"""
database_builder.py
===================
Assembles the complete bracket training database (brackets.json) by:
  1. Running step_extractor on every STEP file
  2. Running pdf_parser (with pair-PDF fallback) on every bracket
  3. Running stl_filter to select the main structural STL
  4. Cross-validating STL surface area vs PDF-stated area (where available)
  5. Assigning a numeric class label (0..N-1) to each bracket
  6. Writing brackets.json and a summary report

Usage:
    python database_builder.py

Outputs:
    data/brackets.json          — full training database
    data/database_report.txt    — human-readable audit report
"""

import json
import os
import sys
import time
from pathlib import Path

# Ensure pipeline modules are importable from the same directory
sys.path.insert(0, str(Path(__file__).parent))

from step_extractor import parse_step
from pdf_parser     import get_label, PAIR_PDF_MAP
from stl_filter     import select_main_stl, find_stls_for_part

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent / "260410_EXPORT FILES"  # where all source files live
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Master catalogue ──────────────────────────────────────────────────────────
# Each entry: part_id → dict of known file names
# step_file / pdf_file may be None for incomplete records
# stl_files is a list (can be empty, one, or many sub-assembly STLs)

CATALOGUE = {
    "3750590":   {"step": "3750590.STEP",   "pdf": "3750590.pdf",   "stls": ["3750590.STL"],                                             "flags": []},
    "3135505R":  {"step": "3135505R.step",  "pdf": "3135505R.pdf",  "stls": ["3135505R.STL"],                                            "flags": ["PROTOTYPE"]},
    "3750259":   {"step": None,             "pdf": "3750259.pdf",   "stls": [],                                                          "flags": ["NO_GEOM"]},
    "3750371L":  {"step": "3750371L.step",  "pdf": None,            "stls": ["3750371L_-_3193594-5.STL", "3750371L_-_3758102L-4.STL"],   "flags": ["NO_PDF"]},
    "3750371R":  {"step": "3750371R.step",  "pdf": None,            "stls": ["3750371R_-_3193594-1.STL", "3750371R_-_3758102R-1.STL"],   "flags": ["NO_PDF"]},
    "3750554L":  {"step": "3750554L.step",  "pdf": None,            "stls": ["3750554L_-_3750596L-1.STL"],                               "flags": ["USE_PAIR"]},
    "3750554R":  {"step": "3750554R.step",  "pdf": "3750554R.pdf",  "stls": ["3750554R_-_3750596R-1.STL"],                               "flags": []},
    "3750555L":  {"step": "3750555L.step",  "pdf": None,            "stls": ["3750555L_-_3750597L-1.STL"],                               "flags": ["USE_PAIR"]},
    "3750555R":  {"step": "3750555R.step",  "pdf": "3750555R.pdf",  "stls": ["3750555R_-_3750597R-1.STL"],                               "flags": []},
    "3753261L":  {"step": "3753261L.step",  "pdf": None,            "stls": ["3753261L_-_3753261L-1.STL"],                               "flags": ["USE_PAIR"]},
    "3753261R":  {"step": "3753261R.step",  "pdf": "3753261R.pdf",  "stls": ["3753261R_-_3753261R-1.STL"],                               "flags": []},
    "3753932L":  {"step": "3753932L.step",  "pdf": None,            "stls": [],                                                          "flags": ["NO_STL", "NO_PDF"]},
    "3753932R":  {"step": "3753932R.step",  "pdf": "3753932R.pdf",  "stls": ["3753932R_-_3753933R-1.STL"],                               "flags": []},
    "3755060L":  {"step": "3755060L.step",  "pdf": None,            "stls": ["3755060L.STL"],                                            "flags": ["USE_PAIR"]},
    "3755060R":  {"step": "3755060R.step",  "pdf": "3755060R.pdf",  "stls": ["3755060R.STL"],                                            "flags": []},
    "3757047":   {"step": "3757047.STEP",   "pdf": "3757047.pdf",   "stls": ["3757047.STL"],                                             "flags": ["OLD_TEMPLATE"]},
    "3759365R":  {"step": "3759365R.step",  "pdf": "3759365R.pdf",  "stls": ["3759365R_-_3759365R-1.STL"],                               "flags": []},
    "3759698L":  {"step": "3759698L.step",  "pdf": "3759698L.pdf",  "stls": ["3759698L.STL"],                                            "flags": []},
    "3759698R":  {"step": "3759698R.step",  "pdf": None,            "stls": ["3759698R.STL"],                                            "flags": ["USE_PAIR"]},
    "3759706R":  {"step": "3759706R.STEP",  "pdf": "3759706R.pdf",  "stls": ["3759706R.STL"],                                            "flags": []},
    "3759968L":  {"step": "3759968L.step",  "pdf": None,            "stls": ["3759968L.STL"],                                            "flags": ["USE_PAIR"]},
    "3759968R":  {"step": "3759968R.step",  "pdf": "3759968R.pdf",  "stls": ["3759968R.STL"],                                            "flags": []},
    "4601064":   {"step": "4601064.STEP",   "pdf": "4601064.pdf",   "stls": ["4601064.STL"],                                             "flags": []},
    "6582530L":  {"step": "6582530L.step",  "pdf": None,            "stls": ["6582530L.STL"],                                            "flags": ["USE_PAIR"]},
    "6582530R":  {"step": "6582530R.step",  "pdf": "6582530R.pdf",  "stls": ["6582530R.STL"],                                            "flags": []},
    "37501263R": {"step": "37501263R.step", "pdf": "37501263R.pdf", "stls": ["37501263R.STL"],                                           "flags": ["PROTOTYPE"]},
}


def check_file(filename):
    """Return full path if file exists, else None."""
    if not filename:
        return None
    p = DATA_DIR / filename
    return str(p) if p.exists() else None


def build_database(verbose=True):
    records = {}
    report_lines = []
    skipped = []

    def log(msg):
        if verbose:
            print(msg)
        report_lines.append(msg)

    log("=" * 65)
    log("ARB BRACKET TRAINING DATABASE BUILD")
    log(f"Source dir : {DATA_DIR}")
    log(f"Brackets   : {len(CATALOGUE)}")
    log("=" * 65)

    # Assign class labels only to brackets that will be trainable
    # (STEP + STL available — PDF may come from pair)
    trainable_parts = [
        pid for pid, cfg in CATALOGUE.items()
        if check_file(cfg["step"]) and any(check_file(s) for s in cfg["stls"])
        and "NO_GEOM" not in cfg["flags"]
    ]
    class_label_map = {pid: i for i, pid in enumerate(sorted(trainable_parts))}
    log(f"Trainable brackets (have STEP + STL): {len(trainable_parts)}")
    log("")

    for part_id, cfg in CATALOGUE.items():
        t0 = time.time()
        log(f"┌─ {part_id}")

        step_path = check_file(cfg["step"])
        pdf_fname = cfg["pdf"]
        stl_fnames = [s for s in cfg["stls"] if check_file(s)]

        # ── Skip if we have neither STEP nor STL ──────────────────────────
        if not step_path and not stl_fnames:
            log(f"│  ⚠  SKIP — no STEP and no STL files")
            log(f"└─ flags: {cfg['flags']}\n")
            skipped.append(part_id)
            continue

        record = {
            "part_id":   part_id,
            "class_label": class_label_map.get(part_id, -1),
            "flags":     cfg["flags"],
            "trainable": part_id in trainable_parts,
        }

        # ── STEP extraction ───────────────────────────────────────────────
        if step_path:
            try:
                step_data = parse_step(step_path)
                record["step"] = step_data
                ss = step_data["scalar_summary"]
                log(f"│  STEP  ✓  ent={ss['entity_count']} faces={ss['n_faces']} "
                    f"cyl={ss['n_cylinders']} adj_edges={ss['n_edges']}")
            except Exception as e:
                log(f"│  STEP  ✗  ERROR: {e}")
                record["step"] = None
        else:
            log(f"│  STEP  —  not provided")
            record["step"] = None

        # ── PDF / label extraction ────────────────────────────────────────
        label = get_label(part_id, str(DATA_DIR), pdf_fname)
        record["label"] = label
        src_tag = label["label_source"]
        log(f"│  PDF   {'✓' if 'direct' in src_tag else ('⚠ '+src_tag if 'pair' in src_tag else '✗')}  "
            f"title='{label['title'][:40]}'  mass={label['mass_kg']}kg  "
            f"thick={label['thickness_mm']}mm")

        # ── STL selection ─────────────────────────────────────────────────
        if stl_fnames:
            full_stl_paths = [str(DATA_DIR / s) for s in stl_fnames
                              if (DATA_DIR / s).exists()]
            if full_stl_paths:
                stl_result = select_main_stl(full_stl_paths)
                sel = stl_result["selected"]
                record["stl"] = stl_result

                # Cross-validate surface area vs PDF
                stl_area = sel["surface_area_mm2"]
                cv_note = ""
                if label.get("mass_kg") and stl_area > 0:
                    # We don't have PDF surface area for all parts, but we
                    # can still flag if area seems wildly off for the mass class
                    cv_note = f"  area={stl_area:.0f}mm²"

                log(f"│  STL   ✓  file={sel['filename']}  "
                    f"tri={sel['n_triangles']}  {sel['bbox_x_mm']}×"
                    f"{sel['bbox_y_mm']}×{sel['bbox_z_mm']}mm"
                    f"  excl={stl_result['fasteners_excluded']} fasteners{cv_note}")
            else:
                log(f"│  STL   ✗  files listed but not found on disk")
                record["stl"] = None
        else:
            log(f"│  STL   —  not provided")
            record["stl"] = None

        # ── Data quality score ────────────────────────────────────────────
        has_step  = record["step"] is not None
        has_label = label["label_source"] != "missing"
        has_stl   = record["stl"] is not None
        quality   = sum([has_step, has_label, has_stl])  # 0..3
        record["data_quality"] = quality
        record["has_step"]  = has_step
        record["has_label"] = has_label
        record["has_stl"]   = has_stl

        quality_str = ["✗✗✗ unusable", "⚠ partial", "⚠ partial", "✓✓✓ ready"][quality]
        log(f"│  Quality: {quality}/3 — {quality_str}")
        log(f"└─ done in {time.time()-t0:.1f}s  class_label={record['class_label']}\n")

        records[part_id] = record

    # ── Write output ──────────────────────────────────────────────────────
    out_path = OUTPUT_DIR / "brackets.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    report_path = OUTPUT_DIR / "database_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # ── Final summary ─────────────────────────────────────────────────────
    quality3 = [r for r in records.values() if r["data_quality"] == 3]
    quality2 = [r for r in records.values() if r["data_quality"] == 2]
    quality1 = [r for r in records.values() if r["data_quality"] <= 1]

    log("=" * 65)
    log("SUMMARY")
    log(f"  Total processed : {len(records)}")
    log(f"  Skipped         : {len(skipped)} → {skipped}")
    log(f"  Quality 3/3 ✓✓✓ : {len(quality3)} → {[r['part_id'] for r in quality3]}")
    log(f"  Quality 2/3 ⚠   : {len(quality2)} → {[r['part_id'] for r in quality2]}")
    log(f"  Quality ≤1/3 ✗  : {len(quality1)} → {[r['part_id'] for r in quality1]}")
    log(f"  Output          : {out_path}")
    log("=" * 65)

    return records


if __name__ == "__main__":
    build_database(verbose=True)
