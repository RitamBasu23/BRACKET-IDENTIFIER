# ARB Bracket Identification Pipeline — Setup & Run Guide

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10 – 3.12 | 3.11 or 3.12 recommended |
| pip | ≥ 23.0 | `pip install --upgrade pip` |
| RAM | ≥ 8 GB | 16 GB recommended for full 15k catalogue |
| Disk | ≥ 10 GB | Renders take ~500 MB per 50 brackets |
| GPU | Optional | CUDA 12.1 for 10× faster CNN training |

---

## 1. Clone / copy the project

```
bracket_pipeline/
├── step_extractor.py      # STEP → face adjacency graph
├── pdf_parser.py          # PDF → metadata labels
├── stl_filter.py          # STL → select main structural part
├── stl_renderer.py        # STL → 36-angle PNG renders
├── database_builder.py    # Assemble brackets.json training DB
├── gnn_train.py           # Train GNN on STEP graphs
├── vision_cnn.py          # Train CNN on rendered images
├── ensemble.py            # Inference engine (GNN + CNN + Metadata + YOLO)
├── run_pipeline.py        # Master entry point
└── requirements.txt
```

Place all ARB source files (`.STEP`, `.STL`, `.pdf`) in a single directory.
The default expected path is:

```
/path/to/arb_files/    ← all .STEP / .STL / .pdf files here
```

Then update `DATA_DIR` in `database_builder.py` to point to that folder:

```python
# database_builder.py, line ~65
DATA_DIR = Path("/path/to/arb_files")
```

---

## 2. Create a virtual environment

```bash
# Create venv (do this once)
python3 -m venv .venv

# Activate — Linux / macOS
source .venv/bin/activate

# Activate — Windows
.venv\Scripts\activate

# Confirm you are in the venv
which python    # should show .venv/bin/python
```

---

## 3. Install dependencies

### Option A — CPU only (works on any machine, no GPU needed)

```bash
pip install --upgrade pip

# Step 1: install PyTorch CPU build first
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu

# Step 2: install torch-geometric (must come AFTER torch)
pip install torch-geometric==2.5.3

# Step 3: install everything else
pip install -r requirements.txt
```

> **Why two steps?** `torch-geometric` needs to detect the installed torch
> version to pick the right wheel. Installing torch first avoids version
> conflicts.

---

### Option B — CUDA (NVIDIA GPU, ~10× faster CNN training)

Check your CUDA version first:
```bash
nvidia-smi    # look for "CUDA Version: 12.x"
```

Then install:
```bash
# Step 1: Upgrade pip
pip install --upgrade pip

# Step 2: Install PyTorch 2.9.1 with CUDA 13.0 (matches your driver exactly)
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130

# Step 3: Install torch-geometric AFTER torch
pip install torch-geometric==2.5.3

# Step 4: Everything else
pip install -r requirements.txt
```

For CUDA 11.8 replace `cu121` with `cu118` in the URL above.

---

### Option C — YOLO support (optional, for real-time bracket detection)

After completing Option A or B:

```bash
pip install ultralytics==8.2.0
```

This adds ~200 MB of dependencies. Only needed if you want YOLO to
auto-crop bracket regions from camera frames.

---

## 4. Verify installation

```bash
python3 -c "
import torch, torchvision, torch_geometric
import pdfplumber, trimesh, matplotlib, numpy, PIL
print('torch:          ', torch.__version__)
print('torchvision:    ', torchvision.__version__)
print('torch_geometric:', torch_geometric.__version__)
print('numpy:          ', numpy.__version__)
print('CUDA available: ', torch.cuda.is_available())
print('All OK')
"
```

Expected output (CPU install):
```
torch:           2.3.1+cpu
torchvision:     0.18.1+cpu
torch_geometric: 2.5.3
numpy:           1.26.4
CUDA available:  False
All OK
```

---

## 5. Run the pipeline

### Run everything end-to-end (recommended first time)

```bash
python3 run_pipeline.py all
```

This runs all 5 stages in order. On the 22-bracket pilot dataset,
total time is approximately:
- Stage 1 (build DB):   ~30 seconds
- Stage 2 (render):     ~5–10 minutes
- Stage 3 (train GNN):  ~1 minute (CPU)
- Stage 4 (train CNN):  ~20–40 minutes (CPU) / ~3 minutes (GPU)
- Stage 5 (gen YOLO):   ~10 seconds

---

### Run individual stages

```bash
# Stage 1: Parse all STEP + PDF + STL files → data/brackets.json
python3 run_pipeline.py build_db

# Stage 2: Render all STLs from 36 angles → data/renders/<part_id>/
python3 run_pipeline.py render

# Stage 3: Train GNN on STEP face graphs → models/gnn_best.pt
python3 run_pipeline.py train_gnn

# Stage 4: Train Vision CNN on renders → models/cnn_best.pt
python3 run_pipeline.py train_cnn

# Stage 5: Generate YOLO bounding-box annotations → data/renders/yolo/
python3 run_pipeline.py gen_yolo

# Stage 6: Run ensemble inference demo
python3 run_pipeline.py ensemble
```

---

### Run a prediction on a single image

```bash
# Basic — camera image only (uses CNN + fingerprint lookup)
python3 run_pipeline.py predict --image path/to/frame.jpg

# With metadata (higher confidence when part number is known)
python3 run_pipeline.py predict \
    --image path/to/frame.jpg \
    --meta '{"part_number":"3750590","material_code":"9002416B","thickness_mm":1.6}'
```

---

### Run individual modules directly

```bash
# Test STEP extraction on a single file
python3 step_extractor.py path/to/bracket.STEP

# Test PDF extraction
python3 pdf_parser.py        # runs test suite on all available PDFs

# Test STL filter (fastener detection)
python3 stl_filter.py        # runs test cases

# Test ensemble engine
python3 ensemble.py          # runs 3 demo queries and prints results
```

---

## 6. Output files

After a full pipeline run, the following files are produced:

```
bracket_pipeline/
├── data/
│   ├── brackets.json              ← master training database
│   ├── database_report.txt        ← data quality audit
│   └── renders/
│       ├── <part_id>/             ← 180 PNG images per bracket
│       │   ├── angle_00_az000_el15.png
│       │   ├── ...
│       │   └── aug_35_4.png
│       └── yolo/
│           ├── dataset.yaml       ← YOLOv8 training config
│           ├── train.txt          ← list of training image paths
│           └── val.txt            ← list of validation image paths
├── models/
│   ├── gnn_best.pt                ← best GNN checkpoint
│   ├── fingerprints.json          ← GNN embedding per bracket (64-dim)
│   ├── cnn_best.pt                ← best CNN checkpoint
│   └── cnn_fingerprints.json      ← CNN embedding per bracket (512-dim)
└── logs/
    ├── gnn_training.json          ← per-epoch GNN metrics
    └── cnn_training.json          ← per-epoch CNN metrics
```

---

## 7. Train YOLO (optional)

After running `gen_yolo`, train YOLOv8 for real-time bracket detection:

```bash
pip install ultralytics==8.2.0

yolo train \
    data=data/renders/yolo/dataset.yaml \
    model=yolov8n.pt \
    epochs=50 \
    imgsz=224 \
    batch=16 \
    project=models/yolo \
    name=bracket_detector
```

The trained model will be at `models/yolo/bracket_detector/weights/best.pt`.
To use it in inference:

```python
from ensemble import BracketEnsemble
engine = BracketEnsemble(yolo_path="models/yolo/bracket_detector/weights/best.pt")
result = engine.predict_from_image("camera_frame.jpg")
```

---

## 8. Common errors

### `ModuleNotFoundError: No module named 'torch_geometric'`
You installed torch-geometric before torch. Fix:
```bash
pip uninstall torch-geometric -y
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric==2.5.3
```

### `No module named 'pdfminer'`
```bash
pip install pdfminer.six==20231228
```

### `RuntimeError: CUDA out of memory`
Reduce batch size in `vision_cnn.py`:
```python
BATCH_SIZE = 4   # default is 8, reduce to 4 or 2
```

### `OSError: cannot open shared object file: libGL.so.1`
Trimesh needs libGL on Linux. Install with:
```bash
sudo apt-get install libgl1-mesa-glx libglib2.0-0
```

### `ValueError: torch.cat(): expected a non-empty list`
The dataset is too small for the train/test split. This is handled
automatically — the script will print a warning and use the training
set as the eval proxy. Add more bracket files to resolve.

### `FileNotFoundError: brackets.json`
Run `build_db` stage before `train_gnn` or `train_cnn`:
```bash
python3 run_pipeline.py build_db
```

### PDF fields extracted as garbled text
Some older ARB drawing templates (e.g. part 3757047) use a non-standard
layout. These are handled by the `_parse_jeep_template()` fallback in
`pdf_parser.py`. For any bracket where the title or material looks wrong,
you can manually override in `database_builder.py` by adding the correct
values to the `CATALOGUE` dict.

---

## 9. Deactivate the virtual environment

```bash
deactivate
```

---

## 10. Project structure summary

| File | Purpose | Runs standalone? |
|---|---|---|
| `step_extractor.py` | STEP → face graph JSON | Yes |
| `pdf_parser.py` | PDF → label dict | Yes |
| `stl_filter.py` | STL → select main part | Yes |
| `stl_renderer.py` | STL → 36-angle PNGs | Yes |
| `database_builder.py` | Assemble brackets.json | Yes |
| `gnn_train.py` | Train GNN | Yes |
| `vision_cnn.py` | Train ResNet-18 CNN | Yes |
| `ensemble.py` | Inference engine | Yes (demo mode) |
| `run_pipeline.py` | Master entry point | Yes (all stages) |
| `requirements.txt` | Python dependencies | — |
