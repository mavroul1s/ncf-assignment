"""
Generate LaTeX tables (mean ± std) from CSV results, for inclusion in the report.

Writes to results/tables/*.tex.
"""
from __future__ import annotations

import os

import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
RESULTS_DIR = os.path.abspath(RESULTS_DIR)
TAB_DIR = os.path.join(RESULTS_DIR, "tables")
os.makedirs(TAB_DIR, exist_ok=True)


def _write(latex: str, stem: str) -> None:
    path = os.path.join(TAB_DIR, f"{stem}.tex")
    with open(path, "w") as f:
        f.write(latex)
    print(f"[tab] {stem}")


def _mean_pm_std(m: float, s: float, decimals: int = 4) -> str:
    return f"{m:.{decimals}f} $\\pm$ {s:.{decimals}f}"


# -----------------------------------------------------------------------------
# Task 2 — HR@10 ± std, grid of pretraining × layers.
# -----------------------------------------------------------------------------
def tab_task02():
    path = os.path.join(RESULTS_DIR, "task02_mlp_layers.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)

    summary = df.groupby(["pretraining", "num_layers"]).agg(
        HR10_mean=("HR@10", "mean"), HR10_std=("HR@10", "std"),
        NDCG10_mean=("NDCG@10", "mean"), NDCG10_std=("NDCG@10", "std"),
    ).reset_index()

    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Pretraining & MLP-1 & MLP-2 & MLP-3 \\\\",
        "\\midrule",
    ]
    for pre, label in [(False, "without"), (True, "with")]:
        sub = summary[summary["pretraining"] == pre].sort_values("num_layers")
        hr = [_mean_pm_std(r.HR10_mean, r.HR10_std) for r in sub.itertuples()]
        lines.append(f"HR@10 ({label}) & " + " & ".join(hr) + " \\\\")
    lines.append("\\midrule")
    for pre, label in [(False, "without"), (True, "with")]:
        sub = summary[summary["pretraining"] == pre].sort_values("num_layers")
        nd = [_mean_pm_std(r.NDCG10_mean, r.NDCG10_std) for r in sub.itertuples()]
        lines.append(f"NDCG@10 ({label}) & " + " & ".join(nd) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write("\n".join(lines), "tab_task02_mlp_layers")


# -----------------------------------------------------------------------------
# Task 3 — parameter counts.
# -----------------------------------------------------------------------------
def tab_task03():
    path = os.path.join(RESULTS_DIR, "task03_params_vs_layers.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "MLP layers & 1 & 2 & 3 \\\\",
        "\\midrule",
    ]
    lines.append("Trainable parameters & " +
                 " & ".join(f"{int(v):,}" for v in df['total_parameters']) +
                 " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write("\n".join(lines), "tab_task03_params_vs_layers")


# -----------------------------------------------------------------------------
# Task 9-10 — NMF sweep.
# -----------------------------------------------------------------------------
def tab_task09_10():
    path = os.path.join(RESULTS_DIR, "task09_10_nmf.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    summary = df.groupby("n_components").agg(
        HR10_mean=("HR@10", "mean"), HR10_std=("HR@10", "std"),
        NDCG10_mean=("NDCG@10", "mean"), NDCG10_std=("NDCG@10", "std"),
        num_parameters=("num_parameters", "first"),
    ).reset_index()

    lines = [
        "\\begin{tabular}{rccc}",
        "\\toprule",
        "Latent factors (k) & HR@10 & NDCG@10 & Parameters \\\\",
        "\\midrule",
    ]
    for r in summary.itertuples():
        lines.append(
            f"{int(r.n_components)} & "
            f"{_mean_pm_std(r.HR10_mean, r.HR10_std)} & "
            f"{_mean_pm_std(r.NDCG10_mean, r.NDCG10_std)} & "
            f"{int(r.num_parameters):,} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write("\n".join(lines), "tab_task09_10_nmf")


# -----------------------------------------------------------------------------
# Task 11 — NeuMF vs NMF.
# -----------------------------------------------------------------------------
def tab_task11():
    path = os.path.join(RESULTS_DIR, "task11_comparison.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)

    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method & HR@10 & NDCG@10 & Parameters \\\\",
        "\\midrule",
    ]
    for r in df.itertuples():
        lines.append(
            f"{r.method} & "
            f"{_mean_pm_std(getattr(r, '_2'), getattr(r, '_3'))} & "
            f"{_mean_pm_std(getattr(r, '_4'), getattr(r, '_5'))} & "
            f"{int(r.num_parameters):,} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write("\n".join(lines), "tab_task11_compare")


# -----------------------------------------------------------------------------
# Task 12 — KD.
# -----------------------------------------------------------------------------
def tab_task12():
    path = os.path.join(RESULTS_DIR, "task12_kd.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    summary = df.groupby("technique").agg(
        HR10_mean=("HR@10", "mean"), HR10_std=("HR@10", "std"),
        NDCG10_mean=("NDCG@10", "mean"), NDCG10_std=("NDCG@10", "std"),
        num_parameters=("num_parameters", "first"),
    ).reset_index()

    teacher_params = int(summary.loc[summary["technique"] == "teacher", "num_parameters"].iat[0])

    order = ["teacher", "response", "feature", "relation"]
    summary["order"] = summary["technique"].apply(lambda t: order.index(t))
    summary = summary.sort_values("order")

    pretty = {"teacher": "Teacher (best NeuMF)",
              "response": "Student 1: response-based KD",
              "feature":  "Student 2: feature-based KD (FitNets)",
              "relation": "Student 3: relation-based KD (RKD)"}

    lines = [
        "\\begin{tabular}{lccrr}",
        "\\toprule",
        "Model & HR@10 & NDCG@10 & Parameters & Reduction \\\\",
        "\\midrule",
    ]
    for r in summary.itertuples():
        red = (1.0 - int(r.num_parameters) / teacher_params) * 100.0
        lines.append(
            f"{pretty[r.technique]} & "
            f"{_mean_pm_std(r.HR10_mean, r.HR10_std)} & "
            f"{_mean_pm_std(r.NDCG10_mean, r.NDCG10_std)} & "
            f"{int(r.num_parameters):,} & "
            f"{red:.1f}\\% \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write("\n".join(lines), "tab_task12_kd")


def main():
    tab_task02()
    tab_task03()
    tab_task09_10()
    tab_task11()
    tab_task12()
    print(f"\nAll tables written to {TAB_DIR}")


if __name__ == "__main__":
    main()
