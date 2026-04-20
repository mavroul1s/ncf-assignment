#!/usr/bin/env bash
# Run all experiments end-to-end. Use --fast for local debug.
set -euo pipefail

cd "$(dirname "$0")"

FAST_FLAG=""
if [[ "${1:-}" == "--fast" ]]; then
    FAST_FLAG="--fast"
    echo "=== FAST MODE (debug) ==="
fi

echo "=== Task 2: MLP layers x pretraining ==="
python experiments/task02_mlp_layers.py $FAST_FLAG

echo "=== Task 3: parameter count vs layers ==="
python experiments/task03_params_vs_layers.py

echo "=== Task 4: training curves ==="
python experiments/task04_training_curves.py $FAST_FLAG

echo "=== Tasks 5-6: HR@K, NDCG@K vs K ==="
python experiments/task05_06_at_k.py $FAST_FLAG

echo "=== Tasks 7-8: HR, NDCG vs number of negatives ==="
python experiments/task07_08_negatives.py $FAST_FLAG

echo "=== Tasks 9-10: NMF sweep ==="
python experiments/task09_10_nmf.py $FAST_FLAG

echo "=== Task 11: NeuMF vs NMF comparison ==="
python experiments/task11_compare.py

echo "=== Task 12: knowledge distillation ==="
python experiments/task12_kd.py $FAST_FLAG

echo "=== Generate figures ==="
python experiments/make_figures.py

echo "=== Generate LaTeX tables ==="
python experiments/make_tables.py

echo "=== DONE ==="
echo "Results in results/  (CSVs, figures/, tables/)"
