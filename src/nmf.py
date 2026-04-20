"""
Non-negative Matrix Factorization (NMF) baseline for top-K recommendation.

We use scikit-learn's `NMF` to factorize a user×item binary interaction matrix
R (R[u, i] = 1 if user u has interacted with item i in TRAIN set, else 0).

For evaluation we use the same leave-one-out protocol as NeuMF: rank the held-out
positive item among 99 sampled negatives using the reconstructed score matrix
R_hat = W @ H, where W is (num_users, k) and H is (k, num_items).

Parameter count: the number of free parameters in NMF is k * (num_users + num_items)
(user factors + item factors), which we report for comparison with NeuMF.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import NMF as SkNMF

from .data import DataSplit


@dataclass
class NMFResult:
    n_components: int
    W: np.ndarray           # (num_users, k)
    H: np.ndarray           # (k, num_items)
    num_parameters: int


def fit_nmf(split: DataSplit, n_components: int, seed: int = 0,
            max_iter: int = 200, init: str = "nndsvd") -> NMFResult:
    """
    Fit NMF on a binary interaction matrix built from the training set only.

    Returns the learned factors and the total parameter count.
    """
    num_users = split.num_users
    num_items = split.num_items

    rows = split.train_pairs[:, 0]
    cols = split.train_pairs[:, 1]
    data = np.ones(len(rows), dtype=np.float32)
    R = sp.csr_matrix((data, (rows, cols)), shape=(num_users, num_items))

    model = SkNMF(
        n_components=n_components,
        init=init,
        random_state=seed,
        max_iter=max_iter,
        tol=1e-4,
    )
    W = model.fit_transform(R)
    H = model.components_

    num_parameters = n_components * (num_users + num_items)
    return NMFResult(n_components=n_components, W=W, H=H, num_parameters=num_parameters)


def score_matrix(result: NMFResult) -> np.ndarray:
    """Reconstruct the full prediction matrix R_hat = W @ H."""
    return result.W @ result.H
