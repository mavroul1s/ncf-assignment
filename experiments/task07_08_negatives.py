"""
Tasks 7 & 8 (0.75 + 0.75 pts): Effect of number of negative samples per positive
                               (1..10, step 1) on HR@10 and NDCG@10 for NeuMF.

Output CSV columns: num_negatives, seed, HR@10, NDCG@10
"""
from __future__ import annotations

import os
import time

import pandas as pd

from _common import (
    get_default_parser, resolve_mode, resolve_device, resolve_data_path,
    train_neumf_no_pretraining, save_csv, RESULTS_DIR,
)
from experiments.config import DEFAULT_NUM_LAYERS
from src.data import load_movielens_100k


def main():
    parser = get_default_parser("Tasks 7 & 8: effect of number of negatives")
    args = parser.parse_args()
    mode = resolve_mode(args)
    device = resolve_device(args)
    split = load_movielens_100k(resolve_data_path(args), seed=42)

    print(f"mode: epochs={mode.epochs} reps={mode.reps}  device={device}")
    rows = []
    for num_neg in range(1, 11):
        for seed in range(mode.reps):
            t0 = time.time()
            _, final, _ = train_neumf_no_pretraining(
                split, num_layers=DEFAULT_NUM_LAYERS, device=device, seed=seed,
                epochs=mode.epochs, num_negatives=num_neg,
            )
            dt = time.time() - t0
            rows.append({
                "num_negatives": num_neg, "seed": seed,
                "HR@10": final["HR@10"], "NDCG@10": final["NDCG@10"],
                "elapsed_sec": dt,
            })
            print(f"  neg={num_neg} seed={seed} "
                  f"HR@10={final['HR@10']:.4f} NDCG@10={final['NDCG@10']:.4f}  ({dt:.1f}s)")

    df = pd.DataFrame(rows)
    output = args.output or os.path.join(RESULTS_DIR, "task07_08_negatives.csv")
    save_csv(df, output)


if __name__ == "__main__":
    main()
