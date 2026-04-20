"""
Tasks 5 & 6 (0.75 + 0.75 pts): Performance of NeuMF at top-K recommendation for
K = 1..10.

We train NeuMF once per seed with the best parameter setting and evaluate
HR@K and NDCG@K for K = 1..10 on the test set.

Output CSV columns: seed, K, HR@K, NDCG@K
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
from src.evaluate import evaluate_hr_ndcg


def main():
    parser = get_default_parser("Tasks 5 & 6: HR@K, NDCG@K for K=1..10")
    args = parser.parse_args()
    mode = resolve_mode(args)
    device = resolve_device(args)
    split = load_movielens_100k(resolve_data_path(args), seed=42)

    print(f"mode: epochs={mode.epochs} reps={mode.reps}  device={device}")
    ks = list(range(1, 11))
    rows = []
    for seed in range(mode.reps):
        t0 = time.time()
        model, _, _ = train_neumf_no_pretraining(
            split, num_layers=DEFAULT_NUM_LAYERS, device=device, seed=seed,
            epochs=mode.epochs, eval_ks=tuple(ks),
        )
        # Evaluate at all K.
        metrics = evaluate_hr_ndcg(
            model, split.test_pairs, split.test_negatives, ks=ks, device=device,
        )
        for k in ks:
            rows.append({
                "seed": seed, "K": k,
                "HR@K": metrics[f"HR@{k}"],
                "NDCG@K": metrics[f"NDCG@{k}"],
            })
        dt = time.time() - t0
        print(f"  seed={seed} done in {dt:.1f}s  "
              f"HR@10={metrics['HR@10']:.4f} NDCG@10={metrics['NDCG@10']:.4f}")

    df = pd.DataFrame(rows)
    output = args.output or os.path.join(RESULTS_DIR, "task05_06_hr_ndcg_at_k.csv")
    save_csv(df, output)


if __name__ == "__main__":
    main()
