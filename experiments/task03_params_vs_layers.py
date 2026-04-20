"""
Task 3 (1 pt): Effect of MLP layers (1..3) on the number of weight parameters
              of NeuMF (without pretraining — same arch regardless of pretraining).

This task requires no training; it's a purely structural count. A single run is
sufficient (no repetitions needed), but we produce a single clean CSV for the
report.
"""
from __future__ import annotations

import os

import pandas as pd

from _common import get_default_parser, resolve_data_path, save_csv, RESULTS_DIR
from experiments.config import GMF_EMBED_DIM, MLP_EMBED_DIM
from src.data import load_movielens_100k
from src.models import NeuMF, count_trainable_parameters, neumf_param_breakdown


def main():
    parser = get_default_parser("Task 3: parameter count vs MLP layers")
    args = parser.parse_args()
    split = load_movielens_100k(resolve_data_path(args), seed=42)

    rows = []
    for num_layers in (1, 2, 3):
        model = NeuMF(split.num_users, split.num_items,
                      gmf_embed_dim=GMF_EMBED_DIM, mlp_embed_dim=MLP_EMBED_DIM,
                      num_layers=num_layers)
        total = count_trainable_parameters(model)
        bd = neumf_param_breakdown(model)
        rows.append({
            "num_layers": num_layers,
            "total_parameters": total,
            "gmf_user_embed": bd["gmf_user_embed.weight"],
            "gmf_item_embed": bd["gmf_item_embed.weight"],
            "mlp_user_embed": bd["mlp_user_embed.weight"],
            "mlp_item_embed": bd["mlp_item_embed.weight"],
            "mlp_layer_sizes": str(model.mlp_layer_sizes),
        })
        print(f"layers={num_layers}  params={total:,}  mlp_sizes={model.mlp_layer_sizes}")

    df = pd.DataFrame(rows)
    output = args.output or os.path.join(RESULTS_DIR, "task03_params_vs_layers.csv")
    save_csv(df, output)


if __name__ == "__main__":
    main()
