"""
Generate all report figures from the CSVs in results/.

Outputs PNGs and PDFs to results/figures/. Run AFTER all task scripts complete.

Figures produced:
    fig_task02_hr_vs_layers.pdf     Task 2: HR@10 vs MLP layers, with/without pretrain
    fig_task03_params_vs_layers.pdf Task 3: params vs MLP layers
    fig_task04_curves_loss.pdf      Task 4a: training loss vs epoch
    fig_task04_curves_hr.pdf        Task 4b: HR@10 vs epoch
    fig_task04_curves_ndcg.pdf      Task 4c: NDCG@10 vs epoch
    fig_task05_hr_at_k.pdf          Task 5: HR@K vs K for NeuMF
    fig_task06_ndcg_at_k.pdf        Task 6: NDCG@K vs K for NeuMF
    fig_task07_hr_vs_negatives.pdf  Task 7: HR@10 vs number of negatives
    fig_task08_ndcg_vs_negatives.pdf Task 8: NDCG@10 vs number of negatives
    fig_task09_nmf_ndcg.pdf         Task 9: NDCG@10 vs n_components
    fig_task10_nmf_params.pdf       Task 10: parameters vs n_components
    fig_task11_compare.pdf          Task 11: NeuMF vs NMF bar comparison
    fig_task12_kd.pdf               Task 12: KD student performance vs teacher
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
RESULTS_DIR = os.path.abspath(RESULTS_DIR)
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# --- Consistent style ---
plt.rcParams.update({
    "figure.figsize": (5.2, 3.6),
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "legend.fontsize": 9,
})


def _save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        path = os.path.join(FIG_DIR, f"{stem}.{ext}")
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {stem}")


def _mean_std(df: pd.DataFrame, by: str, metric: str) -> pd.DataFrame:
    return df.groupby(by).agg(
        mean=(metric, "mean"),
        std=(metric, "std"),
    ).reset_index()


# -----------------------------------------------------------------------------
# Task 2 — HR@10 vs MLP layers for NeuMF with/without pretraining.
# -----------------------------------------------------------------------------
def fig_task02():
    path = os.path.join(RESULTS_DIR, "task02_mlp_layers.csv")
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        return
    df = pd.read_csv(path)

    fig, ax = plt.subplots()
    for pre, marker, label in [(False, "o", "without pretraining"),
                               (True,  "s", "with pretraining")]:
        sub = df[df["pretraining"] == pre]
        stats = _mean_std(sub, "num_layers", "HR@10")
        ax.errorbar(stats["num_layers"], stats["mean"], yerr=stats["std"],
                    marker=marker, capsize=3, linewidth=1.5, label=f"NeuMF {label}")
    ax.set_xlabel("MLP layers")
    ax.set_ylabel("HR@10")
    ax.set_xticks([1, 2, 3])
    ax.set_title("HR@10 vs number of MLP layers")
    ax.legend()
    _save(fig, "fig_task02_hr_vs_layers")


# -----------------------------------------------------------------------------
# Task 3 — Parameter count vs MLP layers.
# -----------------------------------------------------------------------------
def fig_task03():
    path = os.path.join(RESULTS_DIR, "task03_params_vs_layers.csv")
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots()
    ax.bar(df["num_layers"].astype(str), df["total_parameters"],
           color="#4c72b0", edgecolor="black", linewidth=0.5)
    for _, row in df.iterrows():
        ax.text(str(row["num_layers"]), row["total_parameters"],
                f"{int(row['total_parameters']):,}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("MLP layers")
    ax.set_ylabel("Trainable parameters")
    ax.set_title("NeuMF parameters vs MLP layers")
    ax.margins(y=0.15)
    _save(fig, "fig_task03_params_vs_layers")


# -----------------------------------------------------------------------------
# Task 4 — Training curves (loss / HR@10 / NDCG@10 vs epoch), one fig each.
# -----------------------------------------------------------------------------
def fig_task04():
    path = os.path.join(RESULTS_DIR, "task04_training_curves.csv")
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        return
    df = pd.read_csv(path)

    def _plot(metric: str, ylabel: str, stem: str, drop_epoch_zero: bool):
        fig, ax = plt.subplots()
        for model, style in [("GMF", ":"), ("MLP", "--"), ("NeuMF", "-")]:
            sub = df[df["model"] == model].copy()
            if drop_epoch_zero:
                sub = sub[sub["epoch"] > 0]
            stats = sub.groupby("epoch").agg(
                mean=(metric, "mean"), std=(metric, "std")
            ).reset_index()
            ax.plot(stats["epoch"], stats["mean"], style, linewidth=1.8, label=model)
            ax.fill_between(stats["epoch"],
                            stats["mean"] - stats["std"],
                            stats["mean"] + stats["std"],
                            alpha=0.15)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        _save(fig, stem)

    _plot("train_loss", "Training loss", "fig_task04_curves_loss", drop_epoch_zero=True)
    _plot("HR@10",      "HR@10",         "fig_task04_curves_hr",   drop_epoch_zero=False)
    _plot("NDCG@10",    "NDCG@10",       "fig_task04_curves_ndcg", drop_epoch_zero=False)


# -----------------------------------------------------------------------------
# Tasks 5 & 6 — HR@K / NDCG@K vs K.
# -----------------------------------------------------------------------------
def fig_task05_06():
    path = os.path.join(RESULTS_DIR, "task05_06_hr_ndcg_at_k.csv")
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        return
    df = pd.read_csv(path)

    for metric, stem, ylabel in [("HR@K",   "fig_task05_hr_at_k",   "HR@K"),
                                  ("NDCG@K", "fig_task06_ndcg_at_k", "NDCG@K")]:
        stats = _mean_std(df, "K", metric)
        fig, ax = plt.subplots()
        ax.errorbar(stats["K"], stats["mean"], yerr=stats["std"],
                    marker="o", capsize=3, linewidth=1.5, color="#c44e52",
                    label="NeuMF")
        ax.set_xlabel("K")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(1, 11))
        ax.set_title(f"{ylabel} vs K (top-K recommendation)")
        ax.legend()
        _save(fig, stem)


# -----------------------------------------------------------------------------
# Tasks 7 & 8 — HR@10 / NDCG@10 vs number of negatives.
# -----------------------------------------------------------------------------
def fig_task07_08():
    path = os.path.join(RESULTS_DIR, "task07_08_negatives.csv")
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        return
    df = pd.read_csv(path)

    for metric, stem in [("HR@10", "fig_task07_hr_vs_negatives"),
                         ("NDCG@10", "fig_task08_ndcg_vs_negatives")]:
        stats = _mean_std(df, "num_negatives", metric)
        fig, ax = plt.subplots()
        ax.errorbar(stats["num_negatives"], stats["mean"], yerr=stats["std"],
                    marker="s", capsize=3, linewidth=1.5, color="#55a868",
                    label="NeuMF")
        ax.set_xlabel("Number of negatives per positive")
        ax.set_ylabel(metric)
        ax.set_xticks(range(1, 11))
        ax.legend()
        _save(fig, stem)


# -----------------------------------------------------------------------------
# Tasks 9 & 10 — NMF sweep.
# -----------------------------------------------------------------------------
def fig_task09_10():
    path = os.path.join(RESULTS_DIR, "task09_10_nmf.csv")
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        return
    df = pd.read_csv(path)

    # Task 9: NDCG@10 vs n_components
    stats = _mean_std(df, "n_components", "NDCG@10")
    fig, ax = plt.subplots()
    ax.errorbar(stats["n_components"], stats["mean"], yerr=stats["std"],
                marker="d", capsize=3, linewidth=1.5, color="#8172b2",
                label="NMF (sklearn)")
    ax.set_xlabel("Latent factors (n_components)")
    ax.set_ylabel("NDCG@10")
    ax.set_title("NMF: NDCG@10 vs latent factors")
    ax.legend()
    _save(fig, "fig_task09_nmf_ndcg")

    # Task 10: params vs n_components
    param_df = df.groupby("n_components")["num_parameters"].first().reset_index()
    fig, ax = plt.subplots()
    ax.plot(param_df["n_components"], param_df["num_parameters"],
            marker="o", linewidth=1.8, color="#8172b2")
    ax.set_xlabel("Latent factors (n_components)")
    ax.set_ylabel("Trainable parameters")
    ax.set_title("NMF: parameters vs latent factors")
    _save(fig, "fig_task10_nmf_params")


# -----------------------------------------------------------------------------
# Task 11 — NeuMF vs NMF comparison.
# -----------------------------------------------------------------------------
def fig_task11():
    path = os.path.join(RESULTS_DIR, "task11_comparison.csv")
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        return
    df = pd.read_csv(path)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    x = np.arange(len(df))
    colors = ["#c44e52", "#8172b2"]

    for ax, metric, title in [
        (axes[0], "HR@10",  "HR@10"),
        (axes[1], "NDCG@10", "NDCG@10"),
    ]:
        means = df[f"{metric} mean"].values
        stds = df[f"{metric} std"].values
        ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(df["method"], rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(title)
        ax.set_title(title)

    axes[2].bar(x, df["num_parameters"], color=colors, edgecolor="black", linewidth=0.5)
    for xi, v in zip(x, df["num_parameters"]):
        axes[2].text(xi, v, f"{int(v):,}", ha="center", va="bottom", fontsize=8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(df["method"], rotation=20, ha="right", fontsize=9)
    axes[2].set_ylabel("Parameters")
    axes[2].set_title("Parameter count")
    axes[2].margins(y=0.15)

    fig.suptitle("NeuMF vs NMF (best settings)")
    fig.tight_layout()
    _save(fig, "fig_task11_compare")


# -----------------------------------------------------------------------------
# Task 12 — KD comparison.
# -----------------------------------------------------------------------------
def fig_task12():
    path = os.path.join(RESULTS_DIR, "task12_kd.csv")
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        return
    df = pd.read_csv(path)
    stats = df.groupby("technique").agg(
        HR10_mean=("HR@10", "mean"), HR10_std=("HR@10", "std"),
        NDCG10_mean=("NDCG@10", "mean"), NDCG10_std=("NDCG@10", "std"),
        num_parameters=("num_parameters", "first"),
    ).reset_index()

    order = ["teacher", "response", "feature", "relation"]
    stats["order"] = stats["technique"].apply(lambda t: order.index(t))
    stats = stats.sort_values("order").reset_index(drop=True)
    labels = [t + f"\n({int(p):,} params)"
              for t, p in zip(stats["technique"], stats["num_parameters"])]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    x = np.arange(len(stats))

    colors = ["#4c72b0", "#c44e52", "#55a868", "#8172b2"]
    axes[0].bar(x, stats["HR10_mean"], yerr=stats["HR10_std"], capsize=4,
                color=colors[:len(stats)], edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("HR@10")
    axes[0].set_title("HR@10 (teacher vs students)")

    axes[1].bar(x, stats["NDCG10_mean"], yerr=stats["NDCG10_std"], capsize=4,
                color=colors[:len(stats)], edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("NDCG@10")
    axes[1].set_title("NDCG@10 (teacher vs students)")

    fig.tight_layout()
    _save(fig, "fig_task12_kd")


def main():
    fig_task02()
    fig_task03()
    fig_task04()
    fig_task05_06()
    fig_task07_08()
    fig_task09_10()
    fig_task11()
    fig_task12()
    print(f"\nAll figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
