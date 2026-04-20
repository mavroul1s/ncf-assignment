"""
Task 11 (0.5 pt): Compare NeuMF (best setting: 3 MLP layers, no pretraining)
                  with NMF (best n_components from Task 9) on HR@10, NDCG@10,
                  and number of parameters.

Reads:
    results/task02_mlp_layers.csv    for NeuMF (layers=3, pretraining=False)
    results/task09_10_nmf.csv        for NMF at best k (from task09_best_nmf_k.txt)

Output: one clean LaTeX-ready summary table.
"""
from __future__ import annotations

import os

import pandas as pd

from _common import get_default_parser, save_csv, RESULTS_DIR
from experiments.config import GMF_EMBED_DIM, MLP_EMBED_DIM, DEFAULT_NUM_LAYERS
from src.data import load_movielens_100k, default_data_path
from src.models import NeuMF, count_trainable_parameters


def main():
    parser = get_default_parser("Task 11: NeuMF vs NMF comparison")
    args = parser.parse_args()

    # ---- Load NeuMF results from Task 2 (layers=3, no pretraining) ----
    task02 = pd.read_csv(os.path.join(RESULTS_DIR, "task02_mlp_layers.csv"))
    neumf_best = task02[(task02["num_layers"] == DEFAULT_NUM_LAYERS) &
                        (task02["pretraining"] == False)]
    if neumf_best.empty:
        raise RuntimeError("Run task02_mlp_layers.py before task11_compare.py")

    # Compute NeuMF parameters from a reference model.
    split = load_movielens_100k(default_data_path(), seed=42)
    m = NeuMF(split.num_users, split.num_items,
              gmf_embed_dim=GMF_EMBED_DIM, mlp_embed_dim=MLP_EMBED_DIM,
              num_layers=DEFAULT_NUM_LAYERS)
    neumf_params = count_trainable_parameters(m)

    # ---- Load NMF results and best k ----
    with open(os.path.join(RESULTS_DIR, "task09_best_nmf_k.txt")) as f:
        best_k = int(f.read().strip())
    task09 = pd.read_csv(os.path.join(RESULTS_DIR, "task09_10_nmf.csv"))
    nmf_best = task09[task09["n_components"] == best_k]
    nmf_params = int(nmf_best["num_parameters"].iloc[0])

    rows = [
        {
            "method": "NeuMF (layers=3, no pretrain)",
            "HR@10 mean": neumf_best["HR@10"].mean(),
            "HR@10 std":  neumf_best["HR@10"].std(ddof=0),
            "NDCG@10 mean": neumf_best["NDCG@10"].mean(),
            "NDCG@10 std":  neumf_best["NDCG@10"].std(ddof=0),
            "num_parameters": neumf_params,
        },
        {
            "method": f"NMF (k={best_k})",
            "HR@10 mean": nmf_best["HR@10"].mean(),
            "HR@10 std":  nmf_best["HR@10"].std(ddof=0),
            "NDCG@10 mean": nmf_best["NDCG@10"].mean(),
            "NDCG@10 std":  nmf_best["NDCG@10"].std(ddof=0),
            "num_parameters": nmf_params,
        },
    ]
    df = pd.DataFrame(rows)
    output = args.output or os.path.join(RESULTS_DIR, "task11_comparison.csv")
    save_csv(df, output)
    print("\n=== Task 11 comparison ===\n", df.to_string(index=False))


if __name__ == "__main__":
    main()
