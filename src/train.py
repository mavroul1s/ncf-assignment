"""
Training loop for GMF / MLP / NeuMF on MovieLens.

All NCF models are trained with BCEWithLogitsLoss (log loss, equivalent to Eq. 7
of the paper). Negatives are re-sampled every epoch. After each epoch we
evaluate HR@10 / NDCG@10 on the test set (and optionally validation).

The training function is model-agnostic (works for GMF, MLP, NeuMF) and
returns a per-epoch history for plotting Figure-6-style curves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import data as data_mod
from .evaluate import evaluate_hr_ndcg


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    num_negatives: int = 4
    optimizer: str = "adam"      # "adam" or "sgd"; SGD used after pretraining per paper
    eval_every: int = 1          # evaluate every N epochs
    eval_ks: tuple = (10,)
    eval_batch_users: int = 256
    num_workers: int = 0         # keep 0 for reproducibility / portability
    verbose: bool = False


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    epoch_metrics: list[dict] = field(default_factory=list)  # e.g. {'HR@10': ..., 'NDCG@10': ...}


def make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    if cfg.optimizer.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.0,
                               weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"unknown optimizer: {cfg.optimizer}")


def train_ncf(model: nn.Module,
              split: data_mod.DataSplit,
              cfg: TrainConfig,
              device: str = "cpu",
              seed: int = 0,
              use_test_for_eval: bool = True) -> TrainHistory:
    """
    Train a GMF/MLP/NeuMF model. Returns a history of training loss and
    evaluation metrics per epoch.
    """
    model.to(device)
    optim = make_optimizer(model, cfg)
    bce = nn.BCEWithLogitsLoss()
    history = TrainHistory()

    train_ds = data_mod.NCFTrainDataset(split, num_negatives=cfg.num_negatives, seed=seed)

    # Evaluate initial (random-init) performance.
    eval_pairs = split.test_pairs if use_test_for_eval else split.val_pairs
    eval_negs = split.test_negatives if use_test_for_eval else split.val_negatives
    init_metrics = evaluate_hr_ndcg(
        model, eval_pairs, eval_negs, ks=list(cfg.eval_ks),
        device=device, eval_batch_users=cfg.eval_batch_users,
    )
    history.epoch_metrics.append({"epoch": 0, **init_metrics})
    if cfg.verbose:
        print(f"[epoch 0] init metrics: {init_metrics}")

    for epoch in range(1, cfg.epochs + 1):
        train_ds.resample()
        loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=data_mod.collate_batch,
            pin_memory=(device == "cuda"),
        )

        model.train()
        total_loss = 0.0
        total_n = 0
        for users, items, labels in loader:
            users = users.to(device, non_blocking=True)
            items = items.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(users, items)
            loss = bce(logits, labels)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            total_loss += loss.item() * labels.size(0)
            total_n += labels.size(0)

        avg_loss = total_loss / max(total_n, 1)
        history.train_loss.append(avg_loss)

        do_eval = (epoch % cfg.eval_every == 0) or (epoch == cfg.epochs)
        if do_eval:
            metrics = evaluate_hr_ndcg(
                model, eval_pairs, eval_negs, ks=list(cfg.eval_ks),
                device=device, eval_batch_users=cfg.eval_batch_users,
            )
            history.epoch_metrics.append({"epoch": epoch, **metrics})
            if cfg.verbose:
                print(f"[epoch {epoch}] loss={avg_loss:.4f}  {metrics}")

    return history
