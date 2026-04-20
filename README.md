# NCF Assignment — Deep Learning and its Applications

Reproduction of the **Neural Collaborative Filtering** paper
([He et al., WWW 2017](https://arxiv.org/abs/1708.05031)) on **MovieLens 100K**
(`u.data`), for the course *Deep Learning and its Applications* (University of
Thessaly).

Per the assignment, every experiment is **repeated 10 times** and results are
reported as mean ± standard deviation.

## Project layout

```
.
├── data/
│   └── u.data                     # MovieLens 100K dataset (100K ratings, 943 users, 1682 items)
├── src/                           # Core library
│   ├── data.py                    # Leave-one-out split + negative sampling
│   ├── models.py                  # GMF, MLP, NeuMF (with pretraining support)
│   ├── evaluate.py                # HR@K, NDCG@K metrics
│   ├── train.py                   # Training loop
│   ├── nmf.py                     # scikit-learn NMF adapted for top-K
│   ├── distill.py                 # 3 knowledge-distillation techniques
│   └── utils.py                   # Seeding, parameter counting, I/O
├── experiments/                   # One script per assignment task
│   ├── config.py                  # Best-setting hyperparameters (Task 1)
│   ├── _common.py                 # Shared CLI / pretraining helper
│   ├── task02_mlp_layers.py       # HR@10 vs MLP layers (with/without pretraining)
│   ├── task03_params_vs_layers.py # Parameter count vs MLP layers
│   ├── task04_training_curves.py  # Loss / HR / NDCG vs epoch (Fig. 6 style)
│   ├── task05_06_at_k.py          # HR@K, NDCG@K for K=1..10 (Fig. 5 style)
│   ├── task07_08_negatives.py     # HR, NDCG vs number of negatives (Fig. 7 style)
│   ├── task09_10_nmf.py           # NMF latent-factor sweep + parameter count
│   ├── task11_compare.py          # NeuMF vs NMF comparison
│   ├── task12_kd.py               # 3 KD techniques (response / feature / relation)
│   ├── make_figures.py            # Produces all figures (PDF + PNG)
│   └── make_tables.py             # Produces LaTeX tables
├── notebooks/
│   └── kaggle_runner.ipynb        # One-click Kaggle GPU driver
├── results/                       # Generated: CSVs, figures/, tables/
├── report/                        # LaTeX report source
├── run_all.sh                     # Run all tasks sequentially
└── requirements.txt
```

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Smoke-test locally (CPU, ~5 min)

```bash
bash run_all.sh --fast
```

This runs every task with a small number of epochs and 2 seeds just to verify
the pipeline works. Outputs go to `results/`.

### 3. Full run (recommended: Kaggle GPU, ~30–60 min)

Either:

- **Kaggle**: open `notebooks/kaggle_runner.ipynb`, enable a GPU, and run all
  cells. Update the `REPO_URL` to your GitHub fork first.
- **Local CPU** (~15–20 h): `bash run_all.sh` (not recommended, but works).

## Configuration

All experiments use the hyperparameters defined in `experiments/config.py`.
The *best setting* (Task 1) was chosen based on the paper's recommendations
and a small pilot study on MovieLens-100K:

| Hyperparameter             | Value |
| -------------------------- | ----- |
| GMF embedding dim          | 8     |
| MLP embedding dim          | 32    |
| MLP hidden layers          | 3 (tower: 64 → 32 → 16 → 8) |
| Optimizer (no pretraining) | Adam, lr = 1e-3 |
| Optimizer (pretraining)    | SGD, lr = 1e-3  |
| Pretraining mix α          | 0.5   |
| Batch size                 | 256   |
| Negative samples / positive | 4    |
| Training epochs            | 20    |
| Repetitions per experiment | 10    |

## Evaluation protocol

Standard leave-one-out evaluation (as in the NCF paper):

- For each user: held-out latest interaction as test, second-latest as validation,
  rest for training.
- At evaluation time: rank the held-out positive item against 99 randomly
  sampled negative items (items the user has *not* interacted with). HR@K and
  NDCG@K are computed on this top-100 list and averaged across users.

## Knowledge distillation (Task 12)

Teacher: best NeuMF (107,761 parameters).
Students: smaller NeuMF — `gmf_emb=4, mlp_emb=16, 2 layers` (~53K parameters,
≈ 50% reduction).

Three distinct distillation techniques are implemented:

1. **Response-based KD** (Hinton et al., 2015) — soft-target BCE with temperature
   scaling, combined with hard-label BCE.
2. **Feature-based KD / FitNets** (Romero et al., 2014) — MSE between the
   student's fused feature (with a learnable linear projection to the teacher's
   dimension) and the teacher's fused feature.
3. **Relation-based KD / RKD** (Park et al., 2019) — Huber loss between the
   pairwise L2-distance matrices of teacher vs projected-student fused features
   (scale-normalized as in the original paper).

## References

- He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.-S. (2017).
  *Neural Collaborative Filtering.* WWW 2017.
  [paper](https://arxiv.org/abs/1708.05031)
- Gou, J., Yu, B., Maybank, S. J., & Tao, D. (2021).
  *Knowledge Distillation: A Survey.* IJCV 2021.
  [paper](https://arxiv.org/abs/2006.05525)
- Hinton, G., Vinyals, O., & Dean, J. (2015).
  *Distilling the Knowledge in a Neural Network.*
  [paper](https://arxiv.org/abs/1503.02531)
- Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta, C., & Bengio, Y.
  (2014). *FitNets: Hints for Thin Deep Nets.*
  [paper](https://arxiv.org/abs/1412.6550)
- Park, W., Kim, D., Lu, Y., & Cho, M. (2019).
  *Relational Knowledge Distillation.* CVPR 2019.
  [paper](https://arxiv.org/abs/1904.05068)
- Reference NCF implementations:
  [hexiangnan/neural_collaborative_filtering](https://github.com/hexiangnan/neural_collaborative_filtering) (TF),
  [guoyang9/NCF](https://github.com/guoyang9/NCF) (PyTorch).
