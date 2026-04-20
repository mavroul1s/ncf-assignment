"""
Task 12 (1.5 pts): Knowledge distillation.

Teacher = NeuMF with the best setting (gmf_emb=8, mlp_emb=32, 3 MLP layers),
         trained from scratch with Adam.
Students = 3 smaller NeuMF models, each distilled with a different KD technique:
    (1) Response-based KD (soft-target BCE, Hinton et al. 2015)
    (2) Feature-based KD  (FitNets hint-MSE, Romero et al. 2014)
    (3) Relation-based KD (RKD pairwise distance, Park et al. 2019)

Each student uses a smaller architecture:
    gmf_embed_dim = 4, mlp_embed_dim = 16, num_layers = 2
resulting in roughly a quarter of the teacher's parameters.

We report per-student mean ± std of HR@10 and NDCG@10 over `reps` seeds, and
the exact number of parameters. Task requires showing that student metrics
are maintained while parameters are reduced.

Output: long CSV (seed, technique, HR@10, NDCG@10, student_params, teacher_params)
        + summary CSV with mean/std per technique.
"""
from __future__ import annotations

import os
import time

import pandas as pd
import torch

from _common import (
    get_default_parser, resolve_mode, resolve_device, resolve_data_path,
    train_neumf_no_pretraining, save_csv, RESULTS_DIR,
)
from experiments.config import (
    GMF_EMBED_DIM, MLP_EMBED_DIM, DEFAULT_NUM_LAYERS,
    DEFAULT_NUM_NEGATIVES, BATCH_SIZE, LR,
)
from src.data import load_movielens_100k
from src.distill import KDConfig, distill_neumf
from src.models import NeuMF, count_trainable_parameters
from src.utils import set_seed


# Student architecture (smaller than teacher).
STUDENT_GMF_DIM = 4
STUDENT_MLP_DIM = 16
STUDENT_NUM_LAYERS = 2


def build_student(split, seed: int) -> NeuMF:
    set_seed(seed + 30_000)
    return NeuMF(
        split.num_users, split.num_items,
        gmf_embed_dim=STUDENT_GMF_DIM,
        mlp_embed_dim=STUDENT_MLP_DIM,
        num_layers=STUDENT_NUM_LAYERS,
    )


def main():
    parser = get_default_parser("Task 12: knowledge distillation")
    args = parser.parse_args()
    mode = resolve_mode(args)
    device = resolve_device(args)
    split = load_movielens_100k(resolve_data_path(args), seed=42)

    print(f"mode: epochs={mode.epochs} reps={mode.reps}  device={device}")

    # Parameter counts (deterministic — same for all seeds).
    teacher_ref = NeuMF(
        split.num_users, split.num_items,
        gmf_embed_dim=GMF_EMBED_DIM, mlp_embed_dim=MLP_EMBED_DIM,
        num_layers=DEFAULT_NUM_LAYERS,
    )
    student_ref = build_student(split, seed=0)
    teacher_params = count_trainable_parameters(teacher_ref)
    student_params = count_trainable_parameters(student_ref)
    param_reduction_pct = 100.0 * (1.0 - student_params / teacher_params)
    print(f"teacher params = {teacher_params:,}")
    print(f"student params = {student_params:,}  "
          f"(reduction: {param_reduction_pct:.1f}%)")

    techniques = ("response", "feature", "relation")
    rows = []

    for seed in range(mode.reps):
        # --- Train a fresh teacher for this seed ---
        t0 = time.time()
        teacher, teacher_final, _ = train_neumf_no_pretraining(
            split, num_layers=DEFAULT_NUM_LAYERS, device=device, seed=seed,
            epochs=mode.epochs,
        )
        print(f"  [seed {seed}] teacher trained in {time.time() - t0:.1f}s  "
              f"HR@10={teacher_final['HR@10']:.4f}")
        # Record the teacher baseline too.
        rows.append({
            "seed": seed, "technique": "teacher",
            "HR@10": teacher_final["HR@10"], "NDCG@10": teacher_final["NDCG@10"],
            "num_parameters": teacher_params,
        })

        # --- Distill three students ---
        for tech in techniques:
            student = build_student(split, seed)
            cfg = KDConfig(
                technique=tech,
                epochs=mode.epochs,
                batch_size=BATCH_SIZE,
                lr=LR,
                num_negatives=DEFAULT_NUM_NEGATIVES,
                alpha=0.5,
                beta=1.0,
                gamma=1.0,
                temperature=2.0,
                eval_ks=(10,),
            )
            t0 = time.time()
            out = distill_neumf(student, teacher, split, cfg, device=device,
                                seed=seed, verbose=False)
            dt = time.time() - t0
            final = out["final_metrics"]
            print(f"    [{tech}] HR@10={final['HR@10']:.4f} "
                  f"NDCG@10={final['NDCG@10']:.4f}  ({dt:.1f}s)")
            rows.append({
                "seed": seed, "technique": tech,
                "HR@10": final["HR@10"], "NDCG@10": final["NDCG@10"],
                "num_parameters": student_params,
            })

    df = pd.DataFrame(rows)
    output = args.output or os.path.join(RESULTS_DIR, "task12_kd.csv")
    save_csv(df, output)

    summary = df.groupby("technique").agg(
        HR10_mean=("HR@10", "mean"), HR10_std=("HR@10", "std"),
        NDCG10_mean=("NDCG@10", "mean"), NDCG10_std=("NDCG@10", "std"),
        num_parameters=("num_parameters", "first"),
    ).reset_index()
    summary["param_reduction_pct"] = (
        100.0 * (1.0 - summary["num_parameters"] / teacher_params)
    ).round(2)
    print("\n=== Task 12 summary ===\n", summary.to_string(index=False))
    save_csv(summary, os.path.join(RESULTS_DIR, "task12_kd_summary.csv"))


if __name__ == "__main__":
    main()
