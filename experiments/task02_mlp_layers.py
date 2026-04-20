"""
Task 2 (1 pt): Effect of MLP layers (1..3) on HR@10 for NeuMF
              (i) with pretraining and (ii) without pretraining.

Each of the 2 (pretraining flavors) x 3 (layer counts) = 6 configurations is
repeated `--fast` or full number of times. We report mean and std.

Output CSV columns:
    pretraining, num_layers, seed, HR@10, NDCG@10
"""
from __future__ import annotations

import os
import time

import pandas as pd

from _common import (
    get_default_parser, resolve_mode, resolve_device, resolve_data_path,
    train_neumf_with_pretraining, train_neumf_no_pretraining,
    save_csv, RESULTS_DIR,
)
from src.data import load_movielens_100k


def main():
    parser = get_default_parser("Task 2: HR@10 vs MLP layers with/without pretraining")
    args = parser.parse_args()
    mode = resolve_mode(args)
    device = resolve_device(args)
    data_path = resolve_data_path(args)

    print(f"mode: epochs={mode.epochs} reps={mode.reps}  device={device}")
    split = load_movielens_100k(data_path, seed=42)

    rows = []
    for pretrain in (False, True):
        for num_layers in (1, 2, 3):
            for seed in range(mode.reps):
                t0 = time.time()
                if pretrain:
                    _, final, _ = train_neumf_with_pretraining(
                        split, num_layers=num_layers, device=device, seed=seed,
                        epochs=mode.epochs, pretrain_epochs=mode.pretrain_epochs,
                    )
                else:
                    _, final, _ = train_neumf_no_pretraining(
                        split, num_layers=num_layers, device=device, seed=seed,
                        epochs=mode.epochs,
                    )
                dt = time.time() - t0
                rows.append({
                    "pretraining": pretrain,
                    "num_layers": num_layers,
                    "seed": seed,
                    "HR@10": final["HR@10"],
                    "NDCG@10": final["NDCG@10"],
                    "elapsed_sec": dt,
                })
                print(f"  pretrain={pretrain} layers={num_layers} seed={seed} "
                      f"HR@10={final['HR@10']:.4f} NDCG@10={final['NDCG@10']:.4f}  ({dt:.1f}s)")

    df = pd.DataFrame(rows)
    output = args.output or os.path.join(RESULTS_DIR, "task02_mlp_layers.csv")
    save_csv(df, output)

    # Summary table.
    summary = df.groupby(["pretraining", "num_layers"]).agg(
        HR10_mean=("HR@10", "mean"), HR10_std=("HR@10", "std"),
        NDCG10_mean=("NDCG@10", "mean"), NDCG10_std=("NDCG@10", "std"),
    ).reset_index()
    print("\n=== Task 2 summary ===\n", summary.to_string(index=False))
    summary_path = os.path.join(RESULTS_DIR, "task02_mlp_layers_summary.csv")
    save_csv(summary, summary_path)


if __name__ == "__main__":
    main()
