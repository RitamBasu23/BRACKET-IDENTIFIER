"""
vision_cnn.py
=============
Trains a CNN on the multi-angle rendered STL images.

Architecture:
  - EfficientNet-B0 backbone (pretrained on ImageNet, ~5.3M params)
  - Replace final classifier head with a new Linear layer for N bracket classes
  - Fine-tune the full network (not just the head) with a low LR

Why EfficientNet-B0?
  - Strong ImageNet features transfer well to industrial 3D renders
  - Small enough to train on CPU for our pilot dataset
  - 224×224 input matches our render resolution

Training strategy:
  - Load all rendered images with part_id as class label
  - 80/20 train/val split — split at bracket level, not image level
    (so all 180 images of a bracket are in the same split)
  - Data augmentation via torchvision transforms (additional to rendered augments)
  - Reports Top-1 and Top-3 accuracy per epoch
  - Saves best model to models/cnn_best.pt
  - Saves per-bracket embedding vectors to models/cnn_fingerprints.json
"""

import json
import os
import sys
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
DB_PATH      = BASE_DIR / "data" / "brackets.json"
RENDERS_ROOT = BASE_DIR / "data" / "renders"
MODEL_DIR    = BASE_DIR / "models"
LOG_DIR      = BASE_DIR / "logs"
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
IMG_SIZE     = 128
BATCH_SIZE   = 8
EPOCHS       = 30
LR           = 3e-4
WEIGHT_DECAY = 1e-4
TRAIN_RATIO  = 0.8
SEED         = 42


# ── Dataset ───────────────────────────────────────────────────────────────────

class BracketImageDataset(Dataset):
    """
    Loads bracket renders from disk.
    Each sample: (image_tensor, class_label_int)
    """

    TRAIN_TRANSFORMS = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.15),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    VAL_TRANSFORMS = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, samples: list[tuple[str, int]], is_train: bool = True):
        """
        Parameters
        ----------
        samples  : list of (image_path, class_label)
        is_train : use train transforms if True, else val transforms
        """
        self.samples   = samples
        self.transform = self.TRAIN_TRANSFORMS if is_train else self.VAL_TRANSFORMS

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


def build_samples(renders_root: str, db_path: str,
                  train_ratio: float = 0.8, seed: int = 42):
    """
    Scan the renders directory and build train/val sample lists.

    Splits at bracket level: all images for a bracket go to the same split.
    """
    with open(db_path) as f:
        db = json.load(f)

    # Build part_id → class_label map
    label_map = {
        pid: rec["class_label"]
        for pid, rec in db.items()
        if rec.get("class_label", -1) >= 0 and rec.get("data_quality", 0) == 3
    }

    renders_root = Path(renders_root)
    bracket_dirs = sorted([d for d in renders_root.iterdir()
                           if d.is_dir() and d.name in label_map])

    if not bracket_dirs:
        raise RuntimeError(f"No render directories found under {renders_root}")

    random.seed(seed)
    random.shuffle(bracket_dirs)

    n_train = max(1, int(len(bracket_dirs) * train_ratio))
    train_dirs = bracket_dirs[:n_train]
    val_dirs   = bracket_dirs[n_train:]

    # If val is empty (small dataset), use a subset of train images for val
    if not val_dirs:
        val_dirs = train_dirs[:max(1, len(train_dirs) // 5)]

    def dir_to_samples(dirs):
        samples = []
        for d in dirs:
            pid   = d.name
            label = label_map[pid]
            imgs  = sorted(d.glob("*.png"))
            for img_path in imgs:
                samples.append((str(img_path), label))
        return samples

    train_samples = dir_to_samples(train_dirs)
    val_samples   = dir_to_samples(val_dirs)

    num_classes = max(label_map.values()) + 1

    print(f"Brackets — train: {len(train_dirs)}  val: {len(val_dirs)}")
    print(f"Images   — train: {len(train_samples)}  val: {len(val_samples)}")
    print(f"Classes: {num_classes}")

    return train_samples, val_samples, num_classes, label_map


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    """
    ResNet-18 with a custom classification head.
    pretrained=True loads ImageNet weights (recommended; requires internet on first run).
    pretrained=False trains from scratch (faster startup, lower initial accuracy).
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    # Replace final FC: Linear(512 → num_classes) + Dropout
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def get_embedding_model(trained_model: nn.Module) -> nn.Module:
    """Return model without final classifier — outputs 512-dim embedding."""
    # ResNet: remove avgpool onwards → get spatial features, then pool
    emb = nn.Sequential(
        trained_model.conv1,
        trained_model.bn1,
        trained_model.relu,
        trained_model.maxpool,
        trained_model.layer1,
        trained_model.layer2,
        trained_model.layer3,
        trained_model.layer4,
        trained_model.avgpool,
        nn.Flatten(),
    )
    return emb


# ── Training ──────────────────────────────────────────────────────────────────

def topk_acc(logits, labels, k):
    if logits.size(0) == 0:
        return 0.0
    _, topk = logits.topk(min(k, logits.size(1)), dim=1)
    return topk.eq(labels.view(-1, 1).expand_as(topk)).any(dim=1).float().mean().item()


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        all_logits.append(model(imgs).cpu())
        all_labels.append(labels)
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    loss = F.cross_entropy(all_logits, all_labels).item()
    return {
        "loss": round(loss, 4),
        "top1": round(topk_acc(all_logits, all_labels, 1) * 100, 2),
        "top3": round(topk_acc(all_logits, all_labels, 3) * 100, 2),
        "n":    len(all_labels),
    }


def generate_cnn_fingerprints(model, label_map, renders_root, device):
    """
    For each bracket, average all its image embeddings → one fingerprint vector.
    """
    emb_model = get_embedding_model(model).to(device).eval()

    transform = BracketImageDataset.VAL_TRANSFORMS
    renders_root = Path(renders_root)
    fingerprints = {}

    with torch.no_grad():
        for pid, label in sorted(label_map.items()):
            bracket_dir = renders_root / pid
            if not bracket_dir.exists():
                continue

            # Use only base renders (not augmented) for fingerprint
            imgs_paths = sorted(bracket_dir.glob("angle_*.png"))
            if not imgs_paths:
                imgs_paths = sorted(bracket_dir.glob("*.png"))[:36]

            embs = []
            for p in imgs_paths:
                img = Image.open(p).convert("RGB")
                t   = transform(img).unsqueeze(0).to(device)
                emb = emb_model(t).squeeze(0).cpu()
                embs.append(emb)

            if embs:
                mean_emb = torch.stack(embs).mean(dim=0)
                fingerprints[pid] = {
                    "class_label": label,
                    "embedding":   mean_emb.tolist(),
                    "n_images":    len(embs),
                }

    fp_path = MODEL_DIR / "cnn_fingerprints.json"
    with open(fp_path, "w") as f:
        json.dump(fingerprints, f, indent=2)
    print(f"CNN fingerprints saved: {fp_path}  ({len(fingerprints)} brackets)")
    return fingerprints


def train(db_path: str = None, renders_root: str = None):
    db_path      = db_path      or str(DB_PATH)
    renders_root = renders_root or str(RENDERS_ROOT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_s, val_s, num_classes, label_map = build_samples(
        renders_root, db_path, TRAIN_RATIO, SEED
    )

    train_ds = BracketImageDataset(train_s, is_train=True)
    val_ds   = BracketImageDataset(val_s,   is_train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(num_classes, pretrained=True).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: ResNet-18 (scratch)  |  trainable params: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_top1  = 0.0
    best_epoch = 0
    log        = []

    print(f"\n{'Epoch':>6} {'TrainLoss':>10} {'ValLoss':>10} "
          f"{'Top-1%':>8} {'Top-3%':>8} {'Time':>6}")
    print("─" * 56)

    for epoch in range(1, EPOCHS + 1):
        t0         = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_m      = evaluate(model, val_loader, device)
        scheduler.step()

        top1    = val_m["top1"]
        top3    = val_m["top3"]
        elapsed = time.time() - t0

        marker = " ◄ BEST" if top1 > best_top1 else ""
        print(f"{epoch:>6} {train_loss:>10.4f} {val_m['loss']:>10.4f} "
              f"{top1:>8.1f} {top3:>8.1f} {elapsed:>5.1f}s{marker}")

        log.append({"epoch": epoch, "train_loss": round(train_loss, 4), **val_m})

        if top1 > best_top1:
            best_top1  = top1
            best_epoch = epoch
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "top1":             top1,
                "top3":             top3,
                "num_classes":      num_classes,
                "label_map":        label_map,
            }, MODEL_DIR / "cnn_best.pt")

    # ── Save log ───────────────────────────────────────────────────────────
    with open(LOG_DIR / "cnn_training.json", "w") as f:
        json.dump({"config": {
            "model": "ResNet-18 (scratch)",
            "img_size": IMG_SIZE, "batch_size": BATCH_SIZE,
            "epochs": EPOCHS, "lr": LR,
            "num_classes": num_classes,
            "n_train_imgs": len(train_s),
            "n_val_imgs":   len(val_s),
        }, "log": log}, f, indent=2)

    print(f"\n{'='*56}")
    print(f"Training complete.")
    print(f"Best Top-1: {best_top1:.1f}% at epoch {best_epoch}")
    print(f"Best model: {MODEL_DIR / 'cnn_best.pt'}")

    # ── Fingerprints ───────────────────────────────────────────────────────
    print("\nGenerating CNN fingerprints ...")
    generate_cnn_fingerprints(model, label_map, renders_root, device)

    return best_top1


if __name__ == "__main__":
    train()
