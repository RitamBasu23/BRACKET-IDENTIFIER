"""
pdf_parser.py
=============
Extracts ground-truth label data from ARB Corporation engineering drawing PDFs.

Handles:
  - Standard ARB title block format (most brackets)
  - Older Jeep template (3757047) — different layout
  - L/R pair fallback: if a bracket has no PDF, inherit from its RH/LH pair
    and flag label_source = "pair:<partner_part_id>"

Output per bracket:
  {
    "part_id": str,
    "pdf_path": str | null,
    "label_source": "direct" | "pair:<id>" | "missing",
    "part_number": str,
    "title": str,
    "material": str,
    "material_code": str,        # e.g. "9002443"
    "thickness_mm": float | null,
    "finish": str,
    "mass_kg": float | null,
    "is_prototype": bool,
    "is_lh_shown": bool,         # True if LH SHOWN (RH OPPOSITE)
    "is_rh_shown": bool,         # True if RH SHOWN (LH OPPOSITE)
    "revision": str,
    "drawing_date": str,
  }
"""

import re
import json
import pdfplumber
from pathlib import Path


# ── Pair PDF fallback map ─────────────────────────────────────────────────────
# Key = part that is missing a PDF, Value = part whose PDF to borrow
PAIR_PDF_MAP = {
    "3750554L": "3750554R",
    "3750555L": "3750555R",
    "3753261L": "3753261R",
    "3755060L": "3755060R",
    "3759698R": "3759698L",
    "3759968L": "3759968R",
    "6582530L": "6582530R",
    # 3750371L and 3750371R have no pair PDF — must be requested from ARB
}


def _extract_text(pdf_path: str) -> str:
    """Extract all text from a PDF, concatenating pages."""
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(p.extract_text() or "" for p in pdf.pages)


def _parse_standard(text: str, part_id: str) -> dict:
    """Parse the standard ARB title block layout."""

    def find(pattern, default=""):
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip() if m else default

    # Part number
    part_number = find(r"PART\s*NO\.?:?\s*([\w\d]+)")

    # Title — try TITLE: first, then DESC 1:
    title = find(r"TITLE:\s*(.+?)(?:\n|$)")
    if not title:
        title = find(r"DESC\s*1:\s*(.+?)(?:\n|$)")
    # Clean up garbled suffix (some PDFs append noise after the title)
    title = re.split(r"\s{3,}|\bPAGE\b|\bMASS\b", title)[0].strip()

    # Material
    material_raw = find(r"MATERIAL:\s*(.+?)(?:\s{2,}|MASS|\n)")
    # Separate material code from description: "9002443 3 mm STEEL" → code="9002443", desc="3 mm STEEL"
    mat_code_match = re.match(r"(\d{7,})\s*(.*)", material_raw)
    mat_code = mat_code_match.group(1) if mat_code_match else ""
    mat_desc = mat_code_match.group(2).strip() if mat_code_match else material_raw

    # Thickness — parse from material description or standalone
    thick_match = re.search(r"(\d+(?:\.\d+)?)\s*mm", mat_desc, re.IGNORECASE)
    thickness_mm = float(thick_match.group(1)) if thick_match else None

    # Finish
    finish = find(r"FINISH:\s*(.+?)(?:\n|$)")
    finish = re.split(r"\s{3,}", finish)[0].strip()

    # Mass — prefer kg, fall back to g
    mass_kg = None
    m_kg = re.search(r"MASS\s*\(kg\):\s*([\d\.]+)", text, re.IGNORECASE)
    m_g  = re.search(r"MASS\s*\(g\):\s*([\d\.]+)", text, re.IGNORECASE)
    if m_kg:
        mass_kg = float(m_kg.group(1))
    elif m_g:
        mass_kg = round(float(m_g.group(1)) / 1000, 4)

    # Prototype flag
    is_prototype = bool(re.search(r"PROTOTYPE\s+RELEASE\s+ONLY", text, re.IGNORECASE))

    # Design under change flag
    is_design_under_change = bool(re.search(r"DESIGN\s+UNDER\s+CHANGE", text, re.IGNORECASE))

    # Handedness
    is_lh_shown = bool(re.search(r"LH\s+SHOWN", text, re.IGNORECASE))
    is_rh_shown = bool(re.search(r"RH\s+SHOWN", text, re.IGNORECASE))

    # Revision — look for "REV. X" or standalone rev letter near part number block
    rev = find(r"\bREV\.?\s+([A-Z]\d*)\s")
    if not rev:
        rev = find(r"REV\s*\n\s*([A-Z])\s")

    # Drawing date (DETAILED date)
    date = find(r"(?:DATE\s+)?(\d{2}/\d{2}/\d{4})")

    return {
        "part_number":      part_number or part_id,
        "title":            title,
        "material":         mat_desc,
        "material_code":    mat_code,
        "thickness_mm":     thickness_mm,
        "finish":           finish,
        "mass_kg":          mass_kg,
        "is_prototype":     is_prototype,
        "is_design_under_change": is_design_under_change,
        "is_lh_shown":      is_lh_shown,
        "is_rh_shown":      is_rh_shown,
        "revision":         rev,
        "drawing_date":     date,
    }


def _parse_jeep_template(text: str, part_id: str) -> dict:
    """
    Parse the older ARB/Jeep drawing template (e.g. 3757047).
    Fields are in different positions — no standard TITLE: label.
    """
    # Part number appears near end as standalone token
    pn = re.search(r"\b(3757\d{3})\b", text)
    part_number = pn.group(1) if pn else part_id

    # Title is near "Part Description:" label
    title = ""
    m = re.search(r"Part\s+Description:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        title = m.group(1).strip()
    if not title:
        # Fallback: look for standalone description text
        m2 = re.search(r"(JK\s+JEEP\s+[\w\s]+)", text, re.IGNORECASE)
        if m2:
            title = m2.group(1).strip()

    # Material — appears as standalone "4MM" or "GAUGE: 4MM"
    mat = ""
    m3 = re.search(r"GAUGE:\s*(\d+)\s*[-–]\s*(\d+)\s*mm", text, re.IGNORECASE)
    if m3:
        mat = f"{m3.group(1)}-{m3.group(2)}mm steel"
    else:
        m4 = re.search(r"\b(\d+)MM\b", text)
        if m4:
            mat = f"{m4.group(1)}mm steel"

    thick_match = re.search(r"(\d+)(?:MM|mm)", mat)
    thickness_mm = float(thick_match.group(1)) if thick_match else None

    finish_parts = []
    if re.search(r"GOLD\s+PASSIVATE", text, re.IGNORECASE):
        finish_parts.append("Gold Passivate")
    if re.search(r"POWDERCOAT\s+BLACK|BLACK\s+POWDER", text, re.IGNORECASE):
        finish_parts.append("Powdercoat Black")
    finish = " + ".join(finish_parts) if finish_parts else "Gold Passivate + Powdercoat Black"

    mass_kg = None
    m_kg = re.search(r"MASS\s*\(kg\):\s*([\d\.]+)", text, re.IGNORECASE)
    if m_kg:
        mass_kg = float(m_kg.group(1))

    return {
        "part_number":      part_number,
        "title":            title or "JK JEEP LIGHT BOLT ON",
        "material":         mat or "4mm steel",
        "material_code":    "",
        "thickness_mm":     thickness_mm,
        "finish":           finish,
        "mass_kg":          mass_kg,
        "is_prototype":     False,
        "is_design_under_change": False,
        "is_lh_shown":      False,
        "is_rh_shown":      False,
        "revision":         "A",
        "drawing_date":     "07/2007",
    }


def parse_pdf(pdf_path: str, part_id: str = None) -> dict:
    """
    Parse a single PDF and return its label dict.
    part_id defaults to the file stem if not provided.
    """
    path = Path(pdf_path)
    if part_id is None:
        part_id = path.stem

    text = _extract_text(str(path))

    # Detect which template to use
    is_jeep = bool(re.search(r"3757047", text)) or "JK JEEP" in text.upper()

    if is_jeep:
        fields = _parse_jeep_template(text, part_id)
    else:
        fields = _parse_standard(text, part_id)

    return {
        "part_id":      part_id,
        "pdf_path":     str(path),
        "label_source": "direct",
        **fields,
    }


def get_label(part_id: str, pdf_dir: str, pdf_filename: str = None) -> dict:
    """
    Get the label for a bracket, using pair-PDF fallback if needed.

    Parameters
    ----------
    part_id      : the bracket part number (e.g. "3750554L")
    pdf_dir      : directory where PDFs live
    pdf_filename : explicit filename override (optional)

    Returns
    -------
    Label dict with label_source indicating how the label was obtained.
    """
    pdf_dir = Path(pdf_dir)

    # Try direct PDF first
    candidates = [pdf_filename] if pdf_filename else [
        f"{part_id}.pdf", f"{part_id}.PDF",
    ]
    for c in candidates:
        p = pdf_dir / c
        if p.exists():
            return parse_pdf(str(p), part_id)

    # Try pair PDF fallback
    if part_id in PAIR_PDF_MAP:
        partner = PAIR_PDF_MAP[part_id]
        for ext in [".pdf", ".PDF"]:
            p = pdf_dir / f"{partner}{ext}"
            if p.exists():
                result = parse_pdf(str(p), partner)
                result["part_id"]      = part_id      # override to correct part
                result["label_source"] = f"pair:{partner}"
                # For L parts borrowing from R, note handedness flip
                result["is_lh_shown"]  = True
                result["is_rh_shown"]  = False
                return result

    # Nothing found
    return {
        "part_id":      part_id,
        "pdf_path":     None,
        "label_source": "missing",
        "part_number":  part_id,
        "title":        "",
        "material":     "",
        "material_code": "",
        "thickness_mm": None,
        "finish":       "",
        "mass_kg":      None,
        "is_prototype": False,
        "is_design_under_change": False,
        "is_lh_shown":  False,
        "is_rh_shown":  False,
        "revision":     "",
        "drawing_date": "",
    }


if __name__ == "__main__":
    import sys
    pdf_dir = str(Path(__file__).parent / "260410_EXPORT FILES")

    test_parts = [
        ("3750590",  "3750590.pdf"),
        ("3750554R", "3750554R.pdf"),
        ("3750554L", None),          # should use pair fallback
        ("3755060R", "3755060R.pdf"),
        ("3757047",  "3757047.pdf"), # jeep template
        ("3759698L", "3759698L.pdf"),
        ("3759698R", None),          # pair fallback from L
        ("3750371L", None),          # no PDF, no pair → missing
    ]

    for part_id, fname in test_parts:
        result = get_label(part_id, pdf_dir, fname)
        print(f"\n{'─'*55} {part_id}")
        print(f"  source   : {result['label_source']}")
        print(f"  title    : {result['title'][:50]}")
        print(f"  material : {result['material']}")
        print(f"  finish   : {result['finish'][:45]}")
        print(f"  mass_kg  : {result['mass_kg']}")
        print(f"  thick_mm : {result['thickness_mm']}")
        print(f"  prototype: {result['is_prototype']}")
