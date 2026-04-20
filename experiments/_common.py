"""Utilities shared across task scripts."""
from __future__ import annotations

import argparse
import os
import sys

# Allow `from src...` when scripts are invoked directly from experiments/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import torch

from src.data import load_movielens_100k, default_data_path
from src.models import GMF, MLP, NeuMF, count_trainable_parameters
from src.train import TrainConfig, train_ncf
from src.utils import device_auto, set_seed, ensure_dir

from experiments.config import (
    GMF_EMBED_DIM, MLP_EMBED_DIM, DEFAULT_NUM_LAYERS, DEFAULT_NUM_NEGATIVES,
    BATCH_SIZE, LR, ALPHA_PRETRAIN, FAST, FULL,
)

RESULTS_DIR = os.path.join(_ROOT, "results")
ensure_dir(RESULTS_DIR)


def get_default_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--fast", action="store_true",
                   help="debug mode: few epochs/reps for quick local verification")
    p.add_argument("--device", default=None, help="force device (cpu|cuda)")
    p.add_argument("--data", default=None, help="path to u.data")
    p.add_argument("--output", default=None, help="output CSV path")
    return p


def resolve_mode(args) -> object:
    return FAST if args.fast else FULL


def resolve_device(args) -> str:
    return args.device if args.device else device_auto()


def resolve_data_path(args) -> str:
    return args.data if args.data else default_data_path()


def train_neumf_with_pretraining(split, num_layers: int, device: str, seed: int,
                                 epochs: int, pretrain_epochs: int,
                                 verbose: bool = False):
    """
    Train GMF, MLP, then NeuMF initialized from them (SGD).

    Returns the trained NeuMF model and its final eval metrics.
    """
    set_seed(seed)

    # ---- Step 1: train GMF from scratch ----
    gmf = GMF(split.num_users, split.num_items, embed_dim=GMF_EMBED_DIM)
    cfg_gmf = TrainConfig(epochs=pretrain_epochs, batch_size=BATCH_SIZE, lr=LR,
                          num_negatives=DEFAULT_NUM_NEGATIVES, optimizer="adam",
                          eval_ks=(10,), verbose=verbose)
    train_ncf(gmf, split, cfg_gmf, device=device, seed=seed)

    # ---- Step 2: train MLP from scratch with the requested num_layers ----
    set_seed(seed + 10_000)
    mlp = MLP(split.num_users, split.num_items, embed_dim=MLP_EMBED_DIM,
              num_layers=num_layers)
    cfg_mlp = TrainConfig(epochs=pretrain_epochs, batch_size=BATCH_SIZE, lr=LR,
                          num_negatives=DEFAULT_NUM_NEGATIVES, optimizer="adam",
                          eval_ks=(10,), verbose=verbose)
    train_ncf(mlp, split, cfg_mlp, device=device, seed=seed)

    # ---- Step 3: build NeuMF, load pretrained weights, fine-tune with SGD ----
    set_seed(seed + 20_000)
    neumf = NeuMF(split.num_users, split.num_items,
                  gmf_embed_dim=GMF_EMBED_DIM, mlp_embed_dim=MLP_EMBED_DIM,
                  num_layers=num_layers)
    neumf.load_pretrained_weights(gmf, mlp, alpha=ALPHA_PRETRAIN)
    cfg_neumf = TrainConfig(epochs=epochs, batch_size=BATCH_SIZE, lr=LR,
                            num_negatives=DEFAULT_NUM_NEGATIVES, optimizer="sgd",
                            eval_ks=(10,), verbose=verbose)
    history = train_ncf(neumf, split, cfg_neumf, device=device, seed=seed)
    final = history.epoch_metrics[-1]
    return neumf, final, history


def train_neumf_no_pretraining(split, num_layers: int, device: str, seed: int,
                               epochs: int, num_negatives: int = DEFAULT_NUM_NEGATIVES,
                               eval_ks: tuple = (10,), verbose: bool = False):
    """Train NeuMF from scratch with Adam and return model, final metrics, history."""
    set_seed(seed)
    neumf = NeuMF(split.num_users, split.num_items,
                  gmf_embed_dim=GMF_EMBED_DIM, mlp_embed_dim=MLP_EMBED_DIM,
                  num_layers=num_layers)
    cfg = TrainConfig(epochs=epochs, batch_size=BATCH_SIZE, lr=LR,
                      num_negatives=num_negatives, optimizer="adam",
                      eval_ks=eval_ks, verbose=verbose)
    history = train_ncf(neumf, split, cfg, device=device, seed=seed)
    return neumf, history.epoch_metrics[-1], history


def save_csv(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)) or ".")
    df.to_csv(path, index=False)
    print(f"[saved] {path}  ({len(df)} rows)")
