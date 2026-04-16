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
python -m venv .venv

# Activate — Linux / macOS
source .venv/bin/activate

# Activate — Windows
.\.venv\Scripts\Activate.ps1

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

# Step 2: Install PyTorch (GPU-enabled, stable)
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Step 3: Install torch-geometric (must match torch version)
pip install torch-geometric

# Step 4: Install remaining dependencies
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
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
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


