"""
Task 4 (1.5 pts): Per-epoch curves for (i) training loss, (ii) HR@10, (iii) NDCG@10
                  during training. We plot all three NCF variants (GMF, MLP, NeuMF)
                  as in Figure 6 of the paper.

Each model is trained from scratch using the best parameter setting. We repeat
the experiment `reps` times and the figures will show mean ± std envelope.

Output: long-form CSV with columns
    model, seed, epoch, train_loss, HR@10, NDCG@10
"""
from __future__ import annotations

import os
import time

import pandas as pd

from _common import (
    get_default_parser, resolve_mode, resolve_device, resolve_data_path,
    save_csv, RESULTS_DIR,
)
from experiments.config import (
    GMF_EMBED_DIM, MLP_EMBED_DIM, DEFAULT_NUM_LAYERS,
    DEFAULT_NUM_NEGATIVES, BATCH_SIZE, LR,
)
from src.data import load_movielens_100k
from src.models import GMF, MLP, NeuMF
from src.train import TrainConfig, train_ncf
from src.utils import set_seed


def _train_and_collect(model, split, device, seed, epochs):
    cfg = TrainConfig(
        epochs=epochs, batch_size=BATCH_SIZE, lr=LR,
        num_negatives=DEFAULT_NUM_NEGATIVES, optimizer="adam",
        eval_ks=(10,), verbose=False,
    )
    history = train_ncf(model, split, cfg, device=device, seed=seed)
    # Align: epoch_metrics starts at epoch 0 (init), train_loss starts at epoch 1.
    rows = []
    # Epoch 0: init metrics, loss undefined -> skip / set None
    for em in history.epoch_metrics:
        e = em["epoch"]
        if e == 0:
            train_loss = None
        else:
            train_loss = history.train_loss[e - 1]
        rows.append({
            "epoch": e,
            "train_loss": train_loss,
            "HR@10": em["HR@10"],
            "NDCG@10": em["NDCG@10"],
        })
    return rows


def main():
    parser = get_default_parser("Task 4: training curves")
    args = parser.parse_args()
    mode = resolve_mode(args)
    device = resolve_device(args)
    split = load_movielens_100k(resolve_data_path(args), seed=42)

    print(f"mode: epochs={mode.epochs} reps={mode.reps}  device={device}")
    rows = []
    for seed in range(mode.reps):
        for model_name in ("GMF", "MLP", "NeuMF"):
            set_seed(seed)
            t0 = time.time()
            if model_name == "GMF":
                m = GMF(split.num_users, split.num_items, embed_dim=GMF_EMBED_DIM)
            elif model_name == "MLP":
                m = MLP(split.num_users, split.num_items, embed_dim=MLP_EMBED_DIM,
                        num_layers=DEFAULT_NUM_LAYERS)
            else:
                m = NeuMF(split.num_users, split.num_items,
                          gmf_embed_dim=GMF_EMBED_DIM, mlp_embed_dim=MLP_EMBED_DIM,
                          num_layers=DEFAULT_NUM_LAYERS)
            recs = _train_and_collect(m, split, device, seed, mode.epochs)
            dt = time.time() - t0
            for r in recs:
                rows.append({"model": model_name, "seed": seed, **r})
            print(f"  {model_name} seed={seed} done in {dt:.1f}s  "
                  f"final HR@10={recs[-1]['HR@10']:.4f}")

    df = pd.DataFrame(rows)
    output = args.output or os.path.join(RESULTS_DIR, "task04_training_curves.csv")
    save_csv(df, output)


if __name__ == "__main__":
    main()
