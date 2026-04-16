"""
gnn_train.py
============
Graph Neural Network training scaffold for bracket identification.

Architecture:
  - Input: face adjacency graph from STEP file
    - Node features: 8-dim one-hot surface type vector
    - Edges: face adjacency (shared edge curves)
  - 3 × GCNConv layers with ReLU + BatchNorm
  - Global mean pooling → graph-level embedding
  - MLP classifier → class logits

Training:
  - Loads brackets.json (built by database_builder.py)
  - Uses only quality==3 records with valid class labels
  - 80/20 train/test split (stratified by class)
  - Reports Top-1 and Top-3 accuracy per epoch
  - Saves best model checkpoint to models/gnn_best.pt
  - Saves training log to logs/gnn_training.json
"""

import json
import sys
import time
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_PATH  = BASE_DIR / "data" / "brackets.json"
MODEL_DIR  = BASE_DIR / "models"
LOG_DIR    = BASE_DIR / "logs"
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
HIDDEN_DIM   = 64        # GCN hidden dimension
NUM_LAYERS   = 3         # number of GCN conv layers
DROPOUT      = 0.3
LR           = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS       = 100
BATCH_SIZE   = 4          # small batch for our small dataset
TRAIN_RATIO  = 0.8
SEED         = 42

NODE_FEATURE_DIM = 8     # must match step_extractor.NODE_FEATURE_KEYS length


# ── Data loading ──────────────────────────────────────────────────────────────

def load_graphs(db_path: str) -> list[Data]:
    """
    Load all quality-3 brackets from brackets.json and convert to
    PyTorch Geometric Data objects.
    """
    with open(db_path) as f:
        db = json.load(f)

    graphs = []
    skipped = []

    for part_id, record in db.items():
        # Only use fully quality records with a valid class
        if not record.get("trainable"):
            skipped.append((part_id, "not trainable"))
            continue
        if record.get("data_quality", 0) < 3:
            skipped.append((part_id, f"quality={record.get('data_quality')}"))
            continue
        if record.get("class_label", -1) < 0:
            skipped.append((part_id, "no class label"))
            continue
        if not record.get("step") or not record["step"].get("face_graph"):
            skipped.append((part_id, "no STEP graph"))
            continue

        graph_data = record["step"]["face_graph"]
        nodes = graph_data["nodes"]
        edges = graph_data["edges"]

        if len(nodes) == 0:
            skipped.append((part_id, "0 nodes"))
            continue

        # Node feature matrix [N, 8]
        x = torch.tensor(
            [n["feature_vec"] for n in nodes],
            dtype=torch.float
        )

        # Edge index [2, E] — bidirectional
        if edges:
            ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
            # Make bidirectional
            ei = torch.cat([ei, ei.flip(0)], dim=1)
        else:
            ei = torch.zeros((2, 0), dtype=torch.long)

        label = torch.tensor(record["class_label"], dtype=torch.long)

        # Scalar summary as graph-level extra features
        ss = record["step"]["scalar_summary"]
        scalar = torch.tensor([
            ss["n_faces"],
            ss["n_cylinders"],
            ss["n_planes"],
            ss["n_bsplines"],
            ss["n_circles"],
            ss["cylinder_ratio"],
            ss["plane_ratio"],
            ss["bspline_ratio"],
        ], dtype=torch.float)

        data = Data(x=x, edge_index=ei, y=label)
        data.part_id      = part_id
        data.scalar       = scalar
        data.is_prototype = int("PROTOTYPE" in record.get("flags", []))

        graphs.append(data)

    print(f"Loaded {len(graphs)} graphs, skipped {len(skipped)}")
    for p, reason in skipped:
        print(f"  ⚠  skipped {p}: {reason}")

    return graphs


def train_test_split(graphs: list[Data], train_ratio: float, seed: int):
    """Stratified split by class label."""
    random.seed(seed)

    # Group by class
    class_groups: dict[int, list] = {}
    for g in graphs:
        c = int(g.y.item())
        class_groups.setdefault(c, []).append(g)

    train, test = [], []
    for c, items in class_groups.items():
        random.shuffle(items)
        n_train = max(1, int(len(items) * train_ratio))
        train.extend(items[:n_train])
        test.extend(items[n_train:])

    random.shuffle(train)
    random.shuffle(test)
    return train, test


# ── Model ─────────────────────────────────────────────────────────────────────

class BracketGNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int,
                 num_layers: int = 3, dropout: float = 0.3):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.bns.append(BatchNorm(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(BatchNorm(hidden_dim))

        # Scalar feature branch (8 scalar summary features)
        self.scalar_fc = nn.Linear(8, hidden_dim // 2)

        # Classifier MLP
        combined_dim = hidden_dim + hidden_dim // 2
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.dropout = dropout

    def forward(self, x, edge_index, batch, scalar=None):
        # GCN message passing
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph-level pooling
        graph_emb = global_mean_pool(x, batch)   # [B, hidden_dim]

        # Scalar feature branch
        if scalar is not None:
            scalar_emb = F.relu(self.scalar_fc(scalar))  # [B, hidden_dim//2]
            graph_emb = torch.cat([graph_emb, scalar_emb], dim=-1)
        else:
            # Pad with zeros if no scalar features
            pad = torch.zeros(graph_emb.size(0), self.classifier[0].in_features - graph_emb.size(1),
                              device=graph_emb.device)
            graph_emb = torch.cat([graph_emb, pad], dim=-1)

        return self.classifier(graph_emb)

    def embed(self, x, edge_index, batch):
        """Return graph-level embedding (for fingerprint DB)."""
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        return global_mean_pool(x, batch)


# ── Training loop ─────────────────────────────────────────────────────────────

def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Compute Top-K accuracy."""
    if logits.size(0) == 0:
        return 0.0
    _, top_k = logits.topk(min(k, logits.size(1)), dim=1)
    correct = top_k.eq(labels.view(-1, 1).expand_as(top_k))
    return correct.any(dim=1).float().mean().item()


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        scalar = batch.scalar.view(batch.num_graphs, -1) if hasattr(batch, "scalar") else None

        logits = model(batch.x, batch.edge_index, batch.batch, scalar)
        loss   = F.cross_entropy(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        scalar = batch.scalar.view(batch.num_graphs, -1) if hasattr(batch, "scalar") else None
        logits = model(batch.x, batch.edge_index, batch.batch, scalar)
        all_logits.append(logits)
        all_labels.append(batch.y)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    loss    = F.cross_entropy(all_logits, all_labels).item()
    top1    = topk_accuracy(all_logits, all_labels, 1)
    top3    = topk_accuracy(all_logits, all_labels, 3)

    # Per-class predictions for confusion
    preds   = all_logits.argmax(dim=1)
    correct_mask = preds.eq(all_labels)

    return {
        "loss":  round(loss, 4),
        "top1":  round(top1 * 100, 2),
        "top3":  round(top3 * 100, 2),
        "n":     len(all_labels),
        "correct": int(correct_mask.sum().item()),
    }


def train(db_path: str = None):
    if db_path is None:
        db_path = str(DATA_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────
    graphs = load_graphs(db_path)
    if len(graphs) < 4:
        print(f"ERROR: Only {len(graphs)} graphs — need at least 4 to train.")
        sys.exit(1)

    num_classes = max(int(g.y.item()) for g in graphs) + 1
    print(f"Classes: {num_classes}  |  Graphs: {len(graphs)}")

    train_graphs, test_graphs = train_test_split(graphs, TRAIN_RATIO, SEED)
    # With small datasets all classes may end up in train — evaluate on train as proxy
    if len(test_graphs) == 0:
        print("⚠  No test graphs (dataset too small for split) — using train as eval proxy")
        test_graphs = train_graphs
    print(f"Train: {len(train_graphs)}  |  Test: {len(test_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_graphs,  batch_size=BATCH_SIZE, shuffle=False)

    # ── Build model ───────────────────────────────────────────────────────
    model = BracketGNN(
        in_dim=NODE_FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Training loop ─────────────────────────────────────────────────────
    best_top1 = 0.0
    best_epoch = 0
    log = []

    print(f"\n{'Epoch':>6} {'TrainLoss':>10} {'TestLoss':>10} "
          f"{'Top-1%':>8} {'Top-3%':>8} {'Time':>6}")
    print("─" * 56)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        test_metrics = evaluate(model, test_loader, device, num_classes)
        scheduler.step()

        top1 = test_metrics["top1"]
        top3 = test_metrics["top3"]
        elapsed = time.time() - t0

        marker = " ◄ BEST" if top1 > best_top1 else ""
        print(f"{epoch:>6} {train_loss:>10.4f} {test_metrics['loss']:>10.4f} "
              f"{top1:>8.1f} {top3:>8.1f} {elapsed:>5.1f}s{marker}")

        log_entry = {"epoch": epoch, "train_loss": round(train_loss, 4),
                     **test_metrics}
        log.append(log_entry)

        if top1 > best_top1:
            best_top1 = top1
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "top1": top1,
                "top3": top3,
                "num_classes": num_classes,
                "class_label_map": {g.part_id: int(g.y.item()) for g in graphs},
            }, MODEL_DIR / "gnn_best.pt")

    # ── Save training log ─────────────────────────────────────────────────
    with open(LOG_DIR / "gnn_training.json", "w") as f:
        json.dump({"config": {
            "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS,
            "dropout": DROPOUT, "lr": LR, "epochs": EPOCHS,
            "num_classes": num_classes, "n_train": len(train_graphs),
            "n_test": len(test_graphs),
        }, "log": log}, f, indent=2)

    print(f"\n{'='*56}")
    print(f"Training complete.")
    print(f"Best Top-1: {best_top1:.1f}% at epoch {best_epoch}")
    print(f"Best model: {MODEL_DIR / 'gnn_best.pt'}")
    print(f"Train log:  {LOG_DIR / 'gnn_training.json'}")

    # ── Generate fingerprint database from best model ──────────────────────
    print("\nGenerating fingerprint database from best model...")
    generate_fingerprints(model, graphs, device)

    return best_top1


def generate_fingerprints(model, graphs, device):
    """
    Run every bracket through the GNN encoder and save
    the 128-dim embedding vectors as the fingerprint database.
    """
    model.eval()
    fingerprints = {}

    with torch.no_grad():
        for g in graphs:
            g_dev = g.to(device)
            # batch index = all zeros (single graph)
            batch = torch.zeros(g_dev.x.size(0), dtype=torch.long, device=device)
            emb = model.embed(g_dev.x, g_dev.edge_index, batch)
            fingerprints[g.part_id] = {
                "class_label": int(g.y.item()),
                "embedding": emb.squeeze(0).cpu().tolist(),
            }

    fp_path = MODEL_DIR / "fingerprints.json"
    with open(fp_path, "w") as f:
        json.dump(fingerprints, f, indent=2)

    print(f"Fingerprints saved: {fp_path}  ({len(fingerprints)} brackets)")
    return fingerprints


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else None
    train(db)
