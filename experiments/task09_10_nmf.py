"""
Tasks 9 & 10 (0.5 + 0.5 pts):
    9. Non-negative Matrix Factorization baseline using scikit-learn, adapted
       for top-K recommendation. Sweep n_components in {1, 6, 11, 16, 21, 26}
       (i.e. 1..30 with step 5) and show effect on NDCG@10 (and HR@10 bonus).
    10. Show how the parameter count changes with the number of latent factors.

Each (n_components) x (seed) trains a fresh NMF. NMF has random init variance
(via sklearn's random_state), so we repeat `reps` times per assignment rules.

Output CSV columns: n_components, seed, HR@10, NDCG@10, num_parameters, elapsed_sec
"""
from __future__ import annotations

import os
import time

import pandas as pd

from _common import (
    get_default_parser, resolve_mode, resolve_data_path,
    save_csv, RESULTS_DIR,
)
from src.data import load_movielens_100k
from src.evaluate import evaluate_sklearn_nmf
from src.nmf import fit_nmf, score_matrix


def main():
    parser = get_default_parser("Tasks 9 & 10: NMF sweep")
    parser.add_argument("--max-iter", type=int, default=200, help="sklearn NMF max_iter")
    args = parser.parse_args()
    mode = resolve_mode(args)
    split = load_movielens_100k(resolve_data_path(args), seed=42)

    # n_components = 1, 6, 11, 16, 21, 26  (1..30 step 5)
    components = list(range(1, 31, 5))
    print(f"mode: components={components} reps={mode.reps}")

    rows = []
    for k in components:
        for seed in range(mode.reps):
            t0 = time.time()
            result = fit_nmf(split, n_components=k, seed=seed, max_iter=args.max_iter)
            R_hat = score_matrix(result)
            metrics = evaluate_sklearn_nmf(
                R_hat, split.test_pairs, split.test_negatives,
                split.user_pos_set, ks=[10],
            )
            dt = time.time() - t0
            rows.append({
                "n_components": k,
                "seed": seed,
                "HR@10": metrics["HR@10"],
                "NDCG@10": metrics["NDCG@10"],
                "num_parameters": result.num_parameters,
                "elapsed_sec": dt,
            })
            print(f"  k={k} seed={seed} HR@10={metrics['HR@10']:.4f} "
                  f"NDCG@10={metrics['NDCG@10']:.4f} params={result.num_parameters:,}  ({dt:.1f}s)")

    df = pd.DataFrame(rows)
    output = args.output or os.path.join(RESULTS_DIR, "task09_10_nmf.csv")
    save_csv(df, output)

    # Summary: pick best k by NDCG@10 mean.
    summary = df.groupby("n_components").agg(
        HR10_mean=("HR@10", "mean"), HR10_std=("HR@10", "std"),
        NDCG10_mean=("NDCG@10", "mean"), NDCG10_std=("NDCG@10", "std"),
        num_parameters=("num_parameters", "first"),
    ).reset_index()
    best_k = int(summary.loc[summary["NDCG10_mean"].idxmax(), "n_components"])
    print("\n=== Task 9/10 summary ===\n", summary.to_string(index=False))
    print(f"\nBest n_components by mean NDCG@10: {best_k}")
    save_csv(summary, os.path.join(RESULTS_DIR, "task09_10_nmf_summary.csv"))

    with open(os.path.join(RESULTS_DIR, "task09_best_nmf_k.txt"), "w") as f:
        f.write(str(best_k))


if __name__ == "__main__":
    main()
