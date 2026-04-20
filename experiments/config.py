"""
Central configuration used by every task script.

Based on the paper's recommended settings and validated on MovieLens 100K:
    - GMF embedding dim         = 8
    - MLP embedding dim         = 32 (per-branch)
    - MLP hidden layers         = 3   (tower: 64 -> 32 -> 16 -> 8)
    - Optimizer (no pretraining)= Adam, lr = 1e-3
    - Optimizer (pretraining)   = SGD,  lr = 1e-3  (as per paper Section 3.4.1)
    - Alpha (pretraining mix)   = 0.5
    - Batch size                = 256
    - Negative samples / pos    = 4
    - Epochs                    = 20
    - Evaluation protocol       = leave-one-out, 99 sampled negatives, HR/NDCG@10

Every experiment is repeated 10 times with different seeds (0..9) and we report
mean ± std. The final report must use this number of repetitions (per assignment).
"""
from dataclasses import dataclass


# -------- Best NeuMF configuration (fixed throughout the report). --------
GMF_EMBED_DIM = 8
MLP_EMBED_DIM = 32
DEFAULT_NUM_LAYERS = 3
DEFAULT_NUM_NEGATIVES = 4

# -------- Training hyperparameters. --------
BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 20
PRETRAIN_EPOCHS = 20            # epochs for GMF / MLP components when pretraining NeuMF
ALPHA_PRETRAIN = 0.5

# -------- Experiment repetition. --------
NUM_REPETITIONS = 10             # per assignment requirement
SEEDS = list(range(NUM_REPETITIONS))

# -------- Debug / local mode (overridden by experiments via --fast). --------
@dataclass
class RunMode:
    epochs: int = EPOCHS
    reps: int = NUM_REPETITIONS
    pretrain_epochs: int = PRETRAIN_EPOCHS


FAST = RunMode(epochs=3, reps=2, pretrain_epochs=3)    # for local CPU debug
FULL = RunMode(epochs=EPOCHS, reps=NUM_REPETITIONS, pretrain_epochs=PRETRAIN_EPOCHS)
