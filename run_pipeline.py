"""
run_pipeline.py
===============
Master script that runs the full ARB bracket identification pipeline
in the correct order, or individual stages on demand.

Stages:
  1. build_db    — parse STEP + PDF + STL → brackets.json
  2. render      — render all STLs from 36 angles → data/renders/
  3. train_gnn   — train GNN on STEP graphs → models/gnn_best.pt
  4. train_cnn   — train CNN on renders → models/cnn_best.pt
  5. gen_yolo    — generate YOLO bounding box annotations from renders
  6. ensemble    — run ensemble demo / inference test

Usage:
  python run_pipeline.py all           # run stages 1-5 in order
  python run_pipeline.py build_db
  python run_pipeline.py render
  python run_pipeline.py train_gnn
  python run_pipeline.py train_cnn
  python run_pipeline.py gen_yolo
  python run_pipeline.py ensemble
  python run_pipeline.py predict --image frame.jpg [--meta '{"part_number":"3750590"}']
"""

import sys
import json
import time
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ── Stage runners ─────────────────────────────────────────────────────────────

def run_build_db():
    print("\n" + "="*60)
    print("STAGE 1: BUILD DATABASE")
    print("="*60)
    from database_builder import build_database
    records = build_database(verbose=True)
    q3 = sum(1 for r in records.values() if r.get("data_quality") == 3)
    print(f"\nDatabase built: {q3} fully-ready brackets in brackets.json")
    return records


def run_render():
    print("\n" + "="*60)
    print("STAGE 2: RENDER STLs (36 angles × 5 augments = 180 images/bracket)")
    print("="*60)
    import stl_renderer
    stl_renderer.render_all()


def run_train_gnn():
    print("\n" + "="*60)
    print("STAGE 3: TRAIN GNN")
    print("="*60)
    from gnn_train import train
    best = train()
    print(f"\nGNN best Top-1: {best:.1f}%")
    return best


def run_train_cnn():
    print("\n" + "="*60)
    print("STAGE 4: TRAIN VISION CNN")
    print("="*60)
    from vision_cnn import train
    best = train()
    print(f"\nCNN best Top-1: {best:.1f}%")
    return best


def run_gen_yolo():
    print("\n" + "="*60)
    print("STAGE 5: GENERATE YOLO ANNOTATIONS")
    print("="*60)
    generate_yolo_annotations()


def run_ensemble_demo():
    print("\n" + "="*60)
    print("STAGE 6: ENSEMBLE DEMO")
    print("="*60)
    import ensemble as ens_module
    # The __main__ block in ensemble.py runs the demo
    import importlib, runpy
    runpy.run_path(str(BASE_DIR / "ensemble.py"))


def run_predict(image_path: str, metadata: dict = None):
    """Run a single prediction from a camera image."""
    from ensemble import BracketEnsemble
    engine = BracketEnsemble()
    result = engine.predict_from_image(image_path, metadata)
    print("\n── PREDICTION RESULT ──")
    print(f"Image: {image_path}")
    print(f"YOLO detected bracket: {result.get('yolo_detected', False)}")
    print(f"Models used: {result.get('models_used', [])}")
    print(f"Inference: {result.get('total_ms', '?')} ms")
    print()
    for r in result.get("top3", []):
        status = "✓ AUTO-ACCEPT" if r["auto_accept"] else "⚠ FLAG"
        print(f"  Rank {r['rank']}: {r['part_id']:<14} {r['title'][:40]:<40} "
              f"{r['confidence_pct']:>6.1f}%  {status}")
    if result.get("needs_human_review"):
        print(f"\n  ⚠ FLAGGED FOR HUMAN REVIEW: {result.get('flag_reason', '')}")
    return result


# ── YOLO annotation generator ─────────────────────────────────────────────────

def generate_yolo_annotations():
    """
    Generate YOLOv8 training annotations from the bracket renders.

    Strategy:
    - Each rendered image is a clean bracket on white background
    - We automatically compute a tight bounding box by finding non-white pixels
    - Write YOLO-format .txt files alongside each image
    - Also generate dataset.yaml for YOLOv8 training

    YOLO format per line: <class_id> <cx> <cy> <w> <h>  (all normalised 0-1)

    To train YOLOv8 after this:
        pip install ultralytics
        yolo train data=data/renders/yolo/dataset.yaml model=yolov8n.pt epochs=50
    """
    import numpy as np
    from PIL import Image

    renders_root = BASE_DIR / "data" / "renders"
    yolo_root    = BASE_DIR / "data" / "renders" / "yolo"
    yolo_root.mkdir(exist_ok=True)

    # Load class map from brackets.json
    with open(BASE_DIR / "data" / "brackets.json") as f:
        db = json.load(f)

    label_map = {
        pid: rec["class_label"]
        for pid, rec in db.items()
        if rec.get("class_label", -1) >= 0 and rec.get("data_quality", 0) >= 2
    }
    num_classes = max(label_map.values()) + 1

    img_paths_train = []
    img_paths_val   = []
    annotation_count = 0
    parts = sorted(label_map.keys())

    # 80/20 split at bracket level
    split_idx = int(len(parts) * 0.8)
    train_parts = set(parts[:split_idx])
    val_parts   = set(parts[split_idx:])

    for pid, class_id in label_map.items():
        bracket_dir = renders_root / pid
        if not bracket_dir.exists():
            continue

        # Only use base renders (not augmented) for YOLO training
        # Augmented images may have flips that confuse the bounding box
        render_imgs = sorted(bracket_dir.glob("angle_*.png"))
        if not render_imgs:
            render_imgs = sorted(bracket_dir.glob("*.png"))[:36]

        for img_path in render_imgs:
            try:
                img = Image.open(img_path).convert("RGB")
                arr = np.array(img)
                W, H = img.size

                # Find non-white pixels to get tight bbox
                # White = all channels > 240
                is_bg = (arr[:, :, 0] > 240) & \
                        (arr[:, :, 1] > 240) & \
                        (arr[:, :, 2] > 240)
                is_fg = ~is_bg

                rows = np.any(is_fg, axis=1)
                cols = np.any(is_fg, axis=0)

                if not rows.any():
                    # All white — skip
                    continue

                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]

                # Add 5% padding
                pad_x = max(1, int((xmax - xmin) * 0.05))
                pad_y = max(1, int((ymax - ymin) * 0.05))
                xmin  = max(0, xmin - pad_x)
                ymin  = max(0, ymin - pad_y)
                xmax  = min(W - 1, xmax + pad_x)
                ymax  = min(H - 1, ymax + pad_y)

                # YOLO normalised coords
                cx = ((xmin + xmax) / 2) / W
                cy = ((ymin + ymax) / 2) / H
                bw = (xmax - xmin) / W
                bh = (ymax - ymin) / H

                # Write annotation alongside image
                label_path = img_path.with_suffix(".txt")
                with open(label_path, "w") as f:
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

                annotation_count += 1

                # Add to train or val list
                rel_path = str(img_path.resolve())
                if pid in train_parts:
                    img_paths_train.append(rel_path)
                else:
                    img_paths_val.append(rel_path)

            except Exception as e:
                print(f"  ⚠ Error annotating {img_path.name}: {e}")

    # Write file lists
    with open(yolo_root / "train.txt", "w") as f:
        f.write("\n".join(img_paths_train))
    with open(yolo_root / "val.txt", "w") as f:
        f.write("\n".join(img_paths_val))

    # Build class names from DB
    class_names = [""] * num_classes
    for pid, cid in label_map.items():
        title = db[pid].get("label", {}).get("title", pid)
        class_names[cid] = f"{pid}_{title[:20].replace(' ','_')}"

    # Write dataset.yaml
    dataset_yaml = f"""# YOLOv8 dataset config — ARB Bracket Detection
# Auto-generated by run_pipeline.py

path: {renders_root.resolve()}
train: {yolo_root.resolve()}/train.txt
val:   {yolo_root.resolve()}/val.txt

nc: {num_classes}
names:
"""
    for i, name in enumerate(class_names):
        dataset_yaml += f"  {i}: '{name or f'class_{i}'}'\n"

    yaml_path = yolo_root / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(dataset_yaml)

    print(f"  Annotations written : {annotation_count}")
    print(f"  Train images        : {len(img_paths_train)}")
    print(f"  Val images          : {len(img_paths_val)}")
    print(f"  Classes             : {num_classes}")
    print(f"  dataset.yaml        : {yaml_path}")
    print(f"\n  To train YOLOv8:")
    print(f"    pip install ultralytics")
    print(f"    yolo train data={yaml_path} model=yolov8n.pt epochs=50 imgsz=224")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ARB Bracket ML Pipeline")
    parser.add_argument("stage", nargs="?", default="all",
                        choices=["all", "build_db", "render", "train_gnn",
                                 "train_cnn", "gen_yolo", "ensemble", "predict"])
    parser.add_argument("--image",  help="Image path for predict stage")
    parser.add_argument("--meta",   help='JSON string of metadata e.g. {"part_number":"3750590"}')
    args = parser.parse_args()

    t_start = time.time()

    if args.stage == "all":
        run_build_db()
        run_render()
        run_train_gnn()
        run_train_cnn()
        run_gen_yolo()
        run_ensemble_demo()

    elif args.stage == "build_db":   run_build_db()
    elif args.stage == "render":     run_render()
    elif args.stage == "train_gnn":  run_train_gnn()
    elif args.stage == "train_cnn":  run_train_cnn()
    elif args.stage == "gen_yolo":   run_gen_yolo()
    elif args.stage == "ensemble":   run_ensemble_demo()
    elif args.stage == "predict":
        if not args.image:
            print("Error: --image required for predict stage")
            sys.exit(1)
        meta = json.loads(args.meta) if args.meta else None
        run_predict(args.image, meta)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
