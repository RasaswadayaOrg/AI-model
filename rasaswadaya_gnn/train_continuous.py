#!/usr/bin/env python3
"""
Continuous / Incremental Training Runner
=========================================
Trains the GNN on the large dataset and supports:
  1. Fresh full training (100 epochs)
  2. Incremental fine-tuning from a saved checkpoint
  3. Periodic re-training loop (runs every N hours, fine-tuning on latest data)

Usage:
    python train_continuous.py --mode full        # Full 100-epoch train from scratch
    python train_continuous.py --mode finetune    # Resume from checkpoint, 20 more epochs
    python train_continuous.py --mode loop --interval 6  # Loop every 6 hours
"""

import os
import sys
import argparse
import time
import json
import pickle
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import random

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from models.graph_builder import HeterogeneousGraphBuilder
from models.gnn_model import RecommendationModel, train_step, evaluate


# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH  = CHECKPOINT_DIR / "best_model.pt"
LATEST_MODEL_PATH = CHECKPOINT_DIR / "latest_model.pt"
METRICS_PATH     = CHECKPOINT_DIR / "training_metrics.json"


# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(prefer_large: bool = True) -> dict:
    """Load the largest available dataset."""
    candidates = [
        "data/sample_dataset/rasaswadaya_large_dataset.pkl",
        "data/sample_dataset/rasaswadaya_dataset_with_real_artists.json",
        "data/sample_dataset/rasaswadaya_dataset_updated.pkl",
        "data/sample_dataset/rasaswadaya_dataset.pkl",
    ]
    for path in candidates:
        p = Path(path)
        if p.exists():
            print(f"  Loading dataset: {p}")
            if p.suffix == ".pkl":
                with open(p, "rb") as f:
                    return pickle.load(f)
            else:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
    raise FileNotFoundError("No dataset found. Run data/generate_large_dataset.py first.")


def split_edges(data, train_ratio=0.70, val_ratio=0.15):
    edge_splits = {"train": {}, "val": {}, "test": {}}
    for edge_type in [("user","follows","artist")]:
        if edge_type not in data.edge_index_dict:
            continue
        edge_index = data.edge_index_dict[edge_type]
        n = edge_index.size(1)
        perm = torch.randperm(n)
        t = int(train_ratio * n)
        v = int((train_ratio + val_ratio) * n)
        edge_splits["train"][edge_type] = (edge_index[0, perm[:t]],  edge_index[1, perm[:t]])
        edge_splits["val"][edge_type]   = (edge_index[0, perm[t:v]], edge_index[1, perm[t:v]])
        edge_splits["test"][edge_type]  = (edge_index[0, perm[v:]], edge_index[1, perm[v:]])
    return edge_splits


def build_pos_set(data) -> set:
    """All known (user_idx, artist_idx) positive pairs to avoid false negatives."""
    pos_set = set()
    et = ("user", "follows", "artist")
    if et in data.edge_index_dict:
        ei = data.edge_index_dict[et]
        for i in range(ei.size(1)):
            pos_set.add((ei[0, i].item(), ei[1, i].item()))
    return pos_set


def negative_samples(data, pos_edges, pos_set: set = None):
    """
    Corrupted-tail negative sampling.
    Keeps real users (positive src), randomises only the destination artist.
    Retries up to 10 times per sample to avoid false negatives.
    """
    neg = {}
    for et, (ps, pd) in pos_edges.items():
        st, _, dt = et
        n_dst = data[dt].x.size(0)
        neg_dst = torch.randint(0, n_dst, (len(ps),))
        if pos_set:
            for i in range(len(ps)):
                uid = ps[i].item()
                for _ in range(10):
                    if (uid, neg_dst[i].item()) not in pos_set:
                        break
                    neg_dst[i] = torch.randint(0, n_dst, (1,)).item()
        neg[et] = (ps.clone(), neg_dst)  # same users, corrupted artist
    return neg


def load_metrics() -> list:
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return []


def save_metrics(metrics: list):
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)


def print_header(msg):
    print(f"\n{'='*70}\n  {msg}\n{'='*70}")


# ─────────────────────────────────────────────────────────────────────────────
def run_training(
    epochs: int = 100,
    resume_checkpoint: Path = None,
    label: str = "full",
) -> dict:
    """
    Core training loop. Loads data, trains model, saves checkpoints.
    Returns dict of final metrics.
    """
    print_header(f"RASASWADAYA GNN — {label.upper()} TRAINING  ({epochs} epochs)")
    cfg = get_config()
    device = cfg.gnn.device
    print(f"  Device: {device}")

    # ── Dataset ──
    print_header("Loading Dataset")
    dataset = load_dataset()
    meta = dataset.get("metadata", {})
    print(f"  Users:   {len(dataset['users']):,}")
    print(f"  Artists: {len(dataset['artists']):,}")
    print(f"  Events:  {len(dataset['events']):,}")
    follows = dataset.get("interactions", {}).get("follows", [])
    attends = dataset.get("interactions", {}).get("attends", [])
    print(f"  Follows: {len(follows):,}   Attends: {len(attends):,}")

    # ── Graph ──
    print_header("Building Heterogeneous Graph")
    gb = HeterogeneousGraphBuilder(dataset)
    gb.build_graph()
    print(f"  Nodes: {gb.graph.number_of_nodes():,}   Edges: {gb.graph.number_of_edges():,}")
    data = gb.build_pyg_data()

    # ── Edge splits ──
    splits = split_edges(data)
    train_pos = splits["train"]
    val_pos   = splits["val"]
    test_pos  = splits["test"]

    if not train_pos:
        print("  ⚠ No user→artist edges found in PyG data — check dataset.")
        return {}

    pos_set  = build_pos_set(data)
    val_neg  = negative_samples(data, val_pos,  pos_set)
    test_neg = negative_samples(data, test_pos, pos_set)

    n_train = sum(len(v[0]) for v in train_pos.values())
    n_val   = sum(len(v[0]) for v in val_pos.values())
    n_test  = sum(len(v[0]) for v in test_pos.values())
    print(f"  Train edges: {n_train:,}  |  Val: {n_val:,}  |  Test: {n_test:,}")

    # ── Model ──
    print_header("Initialising Model")
    model = RecommendationModel(
        metadata=data.metadata(),
        hidden_channels=cfg.gnn.hidden_channels,
        out_channels=cfg.gnn.out_channels,
        num_layers=cfg.gnn.num_layers,
        model_type=cfg.gnn.model_type,
    ).to(device)

    start_epoch = 1
    if resume_checkpoint and resume_checkpoint.exists():
        print(f"  Resuming from: {resume_checkpoint}")
        model.load_state_dict(torch.load(resume_checkpoint, map_location=device))
        # Infer start epoch from saved metrics
        saved = load_metrics()
        if saved:
            start_epoch = saved[-1].get("epoch", 0) + 1
        print(f"  Resuming at epoch {start_epoch}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.gnn.learning_rate,
        weight_decay=cfg.gnn.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture:  {cfg.gnn.model_type.upper()}")
    print(f"  Hidden/Output: {cfg.gnn.hidden_channels}D / {cfg.gnn.out_channels}D")
    print(f"  Layers:        {cfg.gnn.num_layers}")
    print(f"  Parameters:    {total_params:,}")

    # ── Training loop ──
    print_header("Training Loop")
    print(f"  {'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  {'Val Acc':>9}  {'LR':>10}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*10}  {'-'*9}  {'-'*10}")

    all_metrics = load_metrics()
    best_val_acc = max((m.get("val_acc", 0) for m in all_metrics), default=0.0)
    patience_counter = 0
    training_start = time.time()

    for epoch in range(start_epoch, start_epoch + epochs):
        train_neg = negative_samples(data, train_pos, pos_set)  # fresh negatives each epoch
        loss = train_step(model, data, optimizer, train_pos, train_neg, device)

        if epoch % 10 == 0 or epoch == start_epoch:
            val_loss, val_acc = evaluate(model, data, val_pos, val_neg, device)
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)

            print(f"  {epoch:>6}  {loss:>12.4f}  {val_loss:>10.4f}  {val_acc:>9.4f}  {current_lr:>10.6f}")

            record = {
                "epoch": epoch,
                "train_loss": round(loss, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "lr": round(current_lr, 8),
                "timestamp": datetime.now().isoformat(),
                "label": label,
            }
            all_metrics.append(record)
            save_metrics(all_metrics)

            # Save latest checkpoint every 10 epochs
            torch.save(model.state_dict(), LATEST_MODEL_PATH)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"    ★ New best val_acc: {best_val_acc:.4f} → saved to {BEST_MODEL_PATH}")
            else:
                patience_counter += 1

            if patience_counter >= cfg.training.early_stopping_patience:
                print(f"\n  ⏹  Early stopping at epoch {epoch} (patience={cfg.training.early_stopping_patience})")
                break

    # ── Final test evaluation ──
    print()
    print_header("Final Evaluation on Test Set")
    # Load best model for test eval
    if BEST_MODEL_PATH.exists():
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    test_loss, test_acc = evaluate(model, data, test_pos, test_neg, device)
    elapsed = time.time() - training_start

    print(f"  Best Val Accuracy:  {best_val_acc:.4f}")
    print(f"  Test Loss:          {test_loss:.4f}")
    print(f"  Test Accuracy:      {test_acc:.4f}")
    print(f"  Training time:      {elapsed/60:.1f} min")

    result = {
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "elapsed_seconds": elapsed,
        "total_epochs": epoch,
        "timestamp": datetime.now().isoformat(),
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
def continuous_loop(interval_hours: float = 6.0, finetune_epochs: int = 20):
    """
    Continuously fine-tune the model every `interval_hours` hours.
    Loads the latest checkpoint and runs a short fine-tune pass.
    """
    print_header(f"CONTINUOUS TRAINING LOOP (every {interval_hours}h, {finetune_epochs} epochs each)")
    run = 0
    while True:
        run += 1
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] ── Fine-tune run #{run}")
        checkpoint = LATEST_MODEL_PATH if LATEST_MODEL_PATH.exists() else None
        run_training(epochs=finetune_epochs, resume_checkpoint=checkpoint, label=f"finetune_run{run}")
        print(f"\n  Sleeping {interval_hours}h until next fine-tune…")
        time.sleep(interval_hours * 3600)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Rasaswadaya GNN Continuous Trainer")
    parser.add_argument("--mode", choices=["full","finetune","loop"], default="full",
                        help="full=train from scratch | finetune=resume | loop=continuous")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count (default: 100 for full, 20 for finetune)")
    parser.add_argument("--interval", type=float, default=6.0,
                        help="Hours between loop iterations (loop mode only)")
    args = parser.parse_args()

    if args.mode == "full":
        epochs = args.epochs or 100
        run_training(epochs=epochs, resume_checkpoint=None, label="full")

    elif args.mode == "finetune":
        epochs = args.epochs or 20
        checkpoint = LATEST_MODEL_PATH if LATEST_MODEL_PATH.exists() else BEST_MODEL_PATH
        run_training(epochs=epochs, resume_checkpoint=checkpoint, label="finetune")

    elif args.mode == "loop":
        finetune_epochs = args.epochs or 20
        continuous_loop(interval_hours=args.interval, finetune_epochs=finetune_epochs)


if __name__ == "__main__":
    main()
