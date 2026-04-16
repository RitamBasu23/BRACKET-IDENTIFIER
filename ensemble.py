"""
ensemble.py
===========
Inference engine that combines all three models into a single ranked
prediction with confidence gating.

Three-model ensemble:
  1. GNN fingerprint lookup  — STEP topology graph → cosine similarity vs DB
  2. Vision CNN fingerprint  — camera image → cosine similarity vs DB  
  3. Metadata classifier     — PDF fields → exact/fuzzy match vs DB

Fusion strategy:
  - Each model votes with a confidence score (0.0–1.0)
  - Weighted sum: GNN ×0.40 + CNN ×0.45 + Metadata ×0.15
  - Top-3 candidates returned with ensemble confidence
  - If best confidence < CONFIDENCE_THRESHOLD → flag for human review

YOLO integration:
  - Optional: run YOLOv8 on camera frame to crop bracket region first
  - Cropped region fed to CNN instead of raw frame
  - Falls back to full frame if YOLO finds no bracket

Usage:
    from ensemble import BracketEnsemble
    engine = BracketEnsemble()
    result = engine.predict_from_image("frame.jpg")
    result = engine.predict_from_query(gnn_vec=..., cnn_vec=..., metadata=...)
"""

import json
import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DB_PATH       = BASE_DIR / "data" / "brackets.json"
GNN_FP_PATH   = BASE_DIR / "models" / "fingerprints.json"
CNN_FP_PATH   = BASE_DIR / "models" / "cnn_fingerprints.json"
CNN_MODEL_PATH = BASE_DIR / "models" / "cnn_best.pt"

# ── Ensemble weights (must sum to 1.0) ────────────────────────────────────────
GNN_WEIGHT      = 0.40
CNN_WEIGHT      = 0.45
METADATA_WEIGHT = 0.15

# ── Confidence gating ─────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.80   # below this → flag for human review
TOP_K = 3                      # always return top-3 candidates


# ── Cosine similarity utility ─────────────────────────────────────────────────

def cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two embedding vectors."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def softmax_scores(raw_scores: dict[str, float]) -> dict[str, float]:
    """Convert raw cosine scores to probability-like values via softmax."""
    parts = list(raw_scores.keys())
    vals  = np.array([raw_scores[p] for p in parts], dtype=np.float32)
    # Temperature-scaled softmax
    vals  = vals * 5.0
    vals  = vals - vals.max()
    exp   = np.exp(vals)
    probs = exp / exp.sum()
    return {p: float(probs[i]) for i, p in enumerate(parts)}


# ── Metadata matcher ──────────────────────────────────────────────────────────

class MetadataMatcher:
    """
    Matches extracted PDF metadata fields against the training database.
    Returns a confidence score per bracket.

    Matching logic:
      - Exact part number match → 1.0 (definitive)
      - Material code match     → +0.4
      - Mass within 10%         → +0.3
      - Thickness match         → +0.3
      Normalised to [0, 1]
    """

    def __init__(self, db_path: str = None):
        with open(db_path or DB_PATH) as f:
            self.db = json.load(f)

        # Pre-extract label fields for fast lookup
        self.labels = {}
        for pid, rec in self.db.items():
            lbl = rec.get("label", {})
            self.labels[pid] = {
                "part_number":   lbl.get("part_number", ""),
                "material_code": lbl.get("material_code", ""),
                "thickness_mm":  lbl.get("thickness_mm"),
                "mass_kg":       lbl.get("mass_kg"),
                "class_label":   rec.get("class_label", -1),
            }

    def score(self, query: dict) -> dict[str, float]:
        """
        Score every bracket against the query metadata dict.

        query keys (all optional):
          part_number, material_code, thickness_mm, mass_kg
        """
        scores = {}

        q_pn   = str(query.get("part_number", "")).strip().upper()
        q_mat  = str(query.get("material_code", "")).strip()
        q_th   = query.get("thickness_mm")
        q_mass = query.get("mass_kg")

        for pid, fields in self.labels.items():
            if fields["class_label"] < 0:
                continue

            s = 0.0
            max_s = 0.0

            # Part number exact match
            max_s += 1.0
            db_pn = str(fields["part_number"]).strip().upper()
            if q_pn and db_pn and q_pn == db_pn:
                s += 1.0

            # Material code
            max_s += 0.4
            if q_mat and fields["material_code"] and q_mat == fields["material_code"]:
                s += 0.4

            # Mass within 10%
            max_s += 0.3
            if q_mass and fields["mass_kg"]:
                ratio = abs(float(q_mass) - float(fields["mass_kg"])) / max(float(fields["mass_kg"]), 1e-6)
                if ratio < 0.10:
                    s += 0.3
                elif ratio < 0.25:
                    s += 0.15

            # Thickness
            max_s += 0.3
            if q_th and fields["thickness_mm"]:
                if abs(float(q_th) - float(fields["thickness_mm"])) < 0.2:
                    s += 0.3

            scores[pid] = s / max_s if max_s > 0 else 0.0

        return scores


# ── Fingerprint database ──────────────────────────────────────────────────────

class FingerprintDB:
    """
    Holds pre-computed embedding vectors from GNN and/or CNN.
    Provides nearest-neighbour lookup via cosine similarity.
    """

    def __init__(self, fp_path: str):
        with open(fp_path) as f:
            raw = json.load(f)
        # Normalise embeddings at load time for fast dot-product lookup
        self.embeddings = {}
        self.class_labels = {}
        for pid, entry in raw.items():
            emb = np.array(entry["embedding"], dtype=np.float32)
            norm = np.linalg.norm(emb)
            self.embeddings[pid] = emb / norm if norm > 1e-8 else emb
            self.class_labels[pid] = entry["class_label"]

        self.parts = list(self.embeddings.keys())
        self.matrix = np.stack([self.embeddings[p] for p in self.parts])  # [N, D]
        print(f"  FingerprintDB loaded: {len(self.parts)} brackets, dim={self.matrix.shape[1]}")

    def query(self, query_vec: list[float]) -> dict[str, float]:
        """
        Returns cosine similarity score for every bracket.
        query_vec: raw embedding (will be normalised internally)
        """
        q = np.array(query_vec, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 1e-8:
            q = q / norm
        sims = self.matrix @ q   # [N]
        # Convert from [-1, 1] to [0, 1]
        sims = (sims + 1.0) / 2.0
        return {self.parts[i]: float(sims[i]) for i in range(len(self.parts))}


# ── YOLO bracket detector (optional) ─────────────────────────────────────────

class YOLOBracketDetector:
    """
    Wraps YOLOv8 to detect and crop bracket regions from a camera frame.

    In production: fine-tune YOLOv8n on rendered bracket images annotated
    with bounding boxes. For now, uses the full frame as fallback.

    To use a real YOLO model:
        detector = YOLOBracketDetector(model_path="models/yolo_bracket.pt")

    Requirements:
        pip install ultralytics
    """

    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.5):
        self.model = None
        self.conf_threshold = conf_threshold

        if model_path and Path(model_path).exists():
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                print(f"  YOLO loaded: {model_path}")
            except ImportError:
                print("  ⚠  ultralytics not installed — YOLO disabled, using full frame")
        else:
            print("  ⚠  No YOLO model path provided — using full frame (no crop)")

    def detect_and_crop(self, image_path: str):
        """
        Run YOLO on image, return cropped PIL Image of best bracket detection.
        Falls back to full image if no detection or YOLO unavailable.

        Returns: (PIL.Image, was_detected: bool, bbox: tuple|None)
        """
        from PIL import Image
        img = Image.open(image_path).convert("RGB")

        if self.model is None:
            return img, False, None

        results = self.model(image_path, conf=self.conf_threshold, verbose=False)

        if not results or len(results[0].boxes) == 0:
            return img, False, None

        # Take the highest-confidence box
        boxes = results[0].boxes
        best_idx = int(boxes.conf.argmax())
        x1, y1, x2, y2 = boxes.xyxy[best_idx].int().tolist()

        # Add 10% padding
        w, h = img.size
        pad_x = int((x2 - x1) * 0.10)
        pad_y = int((y2 - y1) * 0.10)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        cropped = img.crop((x1, y1, x2, y2))
        return cropped, True, (x1, y1, x2, y2)


# ── CNN image encoder (inference only) ───────────────────────────────────────

class CNNEncoder:
    """
    Loads the trained CNN and encodes a single PIL image into an embedding.
    """

    def __init__(self, model_path: str = None):
        import torchvision.models as tv_models
        from torchvision import transforms

        model_path = model_path or str(CNN_MODEL_PATH)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not Path(model_path).exists():
            print(f"  ⚠  CNN model not found at {model_path} — CNN disabled")
            self.model = None
            return

        ckpt = torch.load(model_path, map_location=self.device)
        num_classes = ckpt["num_classes"]

        base = tv_models.resnet18(weights=None)
        base.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(base.fc.in_features, num_classes)
        )
        base.load_state_dict(ckpt["model_state_dict"])
        base.eval()

        # Strip classifier — keep up to avgpool for 512-dim embedding
        self.model = torch.nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            base.avgpool, torch.nn.Flatten()
        ).to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        print(f"  CNN encoder loaded: {model_path}")

    @torch.no_grad()
    def encode(self, pil_image) -> Optional[list[float]]:
        if self.model is None:
            return None
        t = self.transform(pil_image).unsqueeze(0).to(self.device)
        emb = self.model(t).squeeze(0).cpu().tolist()
        return emb


# ── Main ensemble engine ──────────────────────────────────────────────────────

class BracketEnsemble:
    """
    Full inference pipeline:
      camera frame → YOLO crop → CNN encode → ensemble lookup → result

    Initialise once, call predict_from_image() or predict_from_query() per frame.
    """

    def __init__(self,
                 gnn_fp_path:    str = None,
                 cnn_fp_path:    str = None,
                 cnn_model_path: str = None,
                 db_path:        str = None,
                 yolo_path:      str = None):

        print("Initialising BracketEnsemble...")

        # Fingerprint databases
        self.gnn_db = None
        self.cnn_db = None

        gnn_fp = gnn_fp_path or str(GNN_FP_PATH)
        cnn_fp = cnn_fp_path or str(CNN_FP_PATH)

        if Path(gnn_fp).exists():
            self.gnn_db = FingerprintDB(gnn_fp)
        else:
            print(f"  ⚠  GNN fingerprints not found: {gnn_fp}")

        if Path(cnn_fp).exists():
            self.cnn_db = FingerprintDB(cnn_fp)
        else:
            print(f"  ⚠  CNN fingerprints not found: {cnn_fp}")

        # Metadata matcher
        self.metadata = MetadataMatcher(db_path)

        # Optional CNN encoder for live camera inference
        self.cnn_encoder = CNNEncoder(cnn_model_path)

        # Optional YOLO detector
        self.yolo = YOLOBracketDetector(yolo_path)

        # Load bracket titles for readable output
        with open(db_path or DB_PATH) as f:
            db = json.load(f)
        self.titles = {pid: rec.get("label", {}).get("title", pid)
                       for pid, rec in db.items()}

        print("BracketEnsemble ready.\n")

    def _fuse_scores(self,
                     gnn_scores:  Optional[dict[str, float]],
                     cnn_scores:  Optional[dict[str, float]],
                     meta_scores: Optional[dict[str, float]]) -> dict[str, float]:
        """
        Weighted fusion of three score dicts.
        Normalises each active model's scores to [0,1] via softmax first,
        then applies weights. Models that are unavailable are excluded and
        weights redistributed proportionally.
        """
        active = []
        if gnn_scores:  active.append(("gnn",  softmax_scores(gnn_scores),  GNN_WEIGHT))
        if cnn_scores:  active.append(("cnn",  softmax_scores(cnn_scores),  CNN_WEIGHT))
        if meta_scores: active.append(("meta", meta_scores,                  METADATA_WEIGHT))

        if not active:
            return {}

        # Normalise weights of active models
        total_w = sum(w for _, _, w in active)
        all_parts = set()
        for _, scores, _ in active:
            all_parts.update(scores.keys())

        fused = {}
        for pid in all_parts:
            fused[pid] = sum(
                scores.get(pid, 0.0) * (w / total_w)
                for _, scores, w in active
            )
        return fused

    def _top_k(self, fused: dict[str, float], k: int = TOP_K) -> list[dict]:
        """Return top-k results sorted by score."""
        sorted_parts = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        results = []
        for i, (pid, score) in enumerate(sorted_parts[:k]):
            results.append({
                "rank":         i + 1,
                "part_id":      pid,
                "title":        self.titles.get(pid, pid),
                "confidence":   round(score, 4),
                "confidence_pct": round(score * 100, 1),
                "auto_accept":  score >= CONFIDENCE_THRESHOLD,
            })
        return results

    def predict_from_query(self,
                           gnn_vec:  Optional[list[float]] = None,
                           cnn_vec:  Optional[list[float]] = None,
                           metadata: Optional[dict] = None) -> dict:
        """
        Core prediction function. Accepts pre-computed vectors and/or metadata.

        Parameters
        ----------
        gnn_vec  : 64-dim GNN embedding from gnn_train.BracketGNN.embed()
        cnn_vec  : 512-dim CNN embedding from CNNEncoder.encode()
        metadata : dict with keys: part_number, material_code, thickness_mm, mass_kg

        Returns
        -------
        {
            "top3": [{rank, part_id, title, confidence, auto_accept}, ...],
            "best": <top result>,
            "needs_human_review": bool,
            "models_used": [str, ...],
            "inference_ms": float,
        }
        """
        t0 = time.time()

        gnn_scores  = self.gnn_db.query(gnn_vec)   if gnn_vec  and self.gnn_db  else None
        cnn_scores  = self.cnn_db.query(cnn_vec)   if cnn_vec  and self.cnn_db  else None
        meta_scores = self.metadata.score(metadata) if metadata                   else None

        fused = self._fuse_scores(gnn_scores, cnn_scores, meta_scores)

        if not fused:
            return {"error": "No models active — cannot predict",
                    "needs_human_review": True}

        top3   = self._top_k(fused, TOP_K)
        best   = top3[0] if top3 else None
        review = (best is None) or (best["confidence"] < CONFIDENCE_THRESHOLD)

        models_used = []
        if gnn_scores:  models_used.append("GNN")
        if cnn_scores:  models_used.append("CNN")
        if meta_scores: models_used.append("Metadata")

        return {
            "top3":               top3,
            "best":               best,
            "needs_human_review": review,
            "flag_reason":        f"Confidence {best['confidence_pct']}% < {CONFIDENCE_THRESHOLD*100}%" if review and best else None,
            "models_used":        models_used,
            "inference_ms":       round((time.time() - t0) * 1000, 1),
        }

    def predict_from_image(self,
                           image_path: str,
                           metadata:   Optional[dict] = None) -> dict:
        """
        Full pipeline: image path → YOLO crop → CNN encode → ensemble predict.

        image_path : path to camera frame PNG/JPG
        metadata   : optional dict of PDF-extracted fields
        """
        t0 = time.time()

        # Step 1: YOLO crop (if available)
        pil_img, was_detected, bbox = self.yolo.detect_and_crop(image_path)

        # Step 2: CNN encode
        cnn_vec = self.cnn_encoder.encode(pil_img)

        # Step 3: Ensemble predict (no GNN vec available from camera alone)
        result = self.predict_from_query(
            gnn_vec  = None,   # GNN runs offline — not from camera
            cnn_vec  = cnn_vec,
            metadata = metadata,
        )

        result["yolo_detected"] = was_detected
        result["yolo_bbox"]     = bbox
        result["total_ms"]      = round((time.time() - t0) * 1000, 1)
        return result


# ── Demo / test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("BRACKET ENSEMBLE — DEMO")
    print("=" * 60)

    engine = BracketEnsemble()

    # ── Test 1: GNN vector only (simulates offline DB query) ──────────────
    print("\n── Test 1: GNN fingerprint query ──")
    # Load a real GNN fingerprint from the DB and query with it
    with open(GNN_FP_PATH) as f:
        fp_db = json.load(f)

    # Use 3750590 fingerprint as query (should return itself as rank 1)
    test_part = "3750590"
    if test_part in fp_db:
        query_vec = fp_db[test_part]["embedding"]
        result = engine.predict_from_query(gnn_vec=query_vec)
        print(f"Query: {test_part}")
        for r in result["top3"]:
            tick = "✓" if r["part_id"] == test_part else " "
            print(f"  {tick} Rank {r['rank']}: {r['part_id']:<14} "
                  f"{r['title'][:35]:<35} {r['confidence_pct']:>6.1f}%  "
                  f"{'AUTO-ACCEPT' if r['auto_accept'] else 'flag'}")
        print(f"  Models used: {result['models_used']}")
        print(f"  Inference: {result['inference_ms']} ms")
        print(f"  Needs human review: {result['needs_human_review']}")

    # ── Test 2: Metadata only ─────────────────────────────────────────────
    print("\n── Test 2: Metadata-only query ──")
    meta_query = {
        "part_number":   "3753261R",
        "material_code": "9002404",
        "thickness_mm":  4.0,
        "mass_kg":       0.1,
    }
    result2 = engine.predict_from_query(metadata=meta_query)
    print(f"Query: {meta_query}")
    for r in result2["top3"]:
        tick = "✓" if r["part_id"] == "3753261R" else " "
        print(f"  {tick} Rank {r['rank']}: {r['part_id']:<14} "
              f"{r['confidence_pct']:>6.1f}%  "
              f"{'AUTO-ACCEPT' if r['auto_accept'] else 'flag'}")

    # ── Test 3: GNN + Metadata ensemble ──────────────────────────────────
    print("\n── Test 3: GNN + Metadata ensemble ──")
    if test_part in fp_db:
        result3 = engine.predict_from_query(
            gnn_vec  = fp_db["3753261R"]["embedding"],
            metadata = {"material_code": "9002404", "thickness_mm": 4.0},
        )
        for r in result3["top3"]:
            tick = "✓" if r["part_id"] == "3753261R" else " "
            print(f"  {tick} Rank {r['rank']}: {r['part_id']:<14} "
                  f"{r['confidence_pct']:>6.1f}%  "
                  f"{'AUTO-ACCEPT' if r['auto_accept'] else 'flag'}")
        print(f"  Models: {result3['models_used']}  |  {result3['inference_ms']} ms")
