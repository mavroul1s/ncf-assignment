"""
Data loading for MovieLens 100K (u.data) for NeuMF experiments.

Format: tab-separated user_id \t item_id \t rating \t timestamp
Size: 100,000 ratings by 943 users on 1682 items. Timestamps are ignored per assignment.

We follow the NCF paper's protocol:
- Implicit feedback: any rating counts as a positive interaction (label = 1).
- Leave-one-out evaluation: for each user, one interaction is held out as the
  test positive, and an additional one as the validation positive.
- For evaluation, we sample 99 negative items (not interacted with) per user and
  rank the test item among them (top-K on a list of 100).
- During training, we sample `num_negatives` negative items per positive
  interaction (default 4, re-sampled every epoch).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class DataSplit:
    """Container for the leave-one-out split of MovieLens 100K."""

    num_users: int
    num_items: int
    train_pairs: np.ndarray  # shape (N_train, 2) of (user, item) positives
    val_pairs: np.ndarray    # shape (num_users, 2): one held-out positive per user
    test_pairs: np.ndarray   # shape (num_users, 2): one held-out positive per user
    val_negatives: np.ndarray   # shape (num_users, 99): 99 neg items per user for val
    test_negatives: np.ndarray  # shape (num_users, 99): 99 neg items per user for test
    user_pos_set: list[set]  # user_pos_set[u] = set of all items user u has interacted with (train ∪ val ∪ test)


def load_movielens_100k(path: str, seed: int = 42) -> DataSplit:
    """
    Load u.data, remap user/item ids to contiguous 0..N-1, perform leave-one-out
    split, and pre-sample evaluation negatives.

    We use timestamp ordering only to pick the *latest* interaction as test and
    the *second latest* as validation (standard leave-one-out protocol).
    """
    rng = np.random.default_rng(seed)

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["user", "item", "rating", "ts"],
        engine="c",
    )

    # Remap to 0..N-1 contiguous ids (u.data is already 1..N but being safe).
    unique_users = np.sort(df["user"].unique())
    unique_items = np.sort(df["item"].unique())
    user2idx = {u: i for i, u in enumerate(unique_users)}
    item2idx = {v: i for i, v in enumerate(unique_items)}
    df["u"] = df["user"].map(user2idx).astype(np.int64)
    df["i"] = df["item"].map(item2idx).astype(np.int64)

    num_users = len(unique_users)
    num_items = len(unique_items)

    # Sort by (user, timestamp) so the latest interaction of each user is last.
    df = df.sort_values(["u", "ts"], kind="mergesort").reset_index(drop=True)

    # Build per-user interaction sets (all positives).
    user_pos_set: list[set] = [set() for _ in range(num_users)]
    for u, i in zip(df["u"].values, df["i"].values):
        user_pos_set[u].add(int(i))

    # Leave-one-out: test = last interaction, val = second-to-last.
    test_pairs = np.zeros((num_users, 2), dtype=np.int64)
    val_pairs = np.zeros((num_users, 2), dtype=np.int64)
    train_records: list[tuple[int, int]] = []

    # Group indices per user (df is already sorted).
    grouped = df.groupby("u", sort=False).indices  # dict: user -> array of row indices

    for u in range(num_users):
        idx = grouped[u]  # sorted by ts ascending
        if len(idx) < 3:
            # Fallback: assignment guarantees ≥20 ratings/user; this shouldn't trigger.
            raise ValueError(f"user {u} has <3 interactions, cannot do leave-one-out")
        test_row = idx[-1]
        val_row = idx[-2]
        test_pairs[u] = (u, df["i"].iat[test_row])
        val_pairs[u] = (u, df["i"].iat[val_row])
        for r in idx[:-2]:
            train_records.append((u, int(df["i"].iat[r])))

    train_pairs = np.array(train_records, dtype=np.int64)

    # Pre-sample 99 negatives per user for val and test (fixed for the run).
    val_negatives = np.zeros((num_users, 99), dtype=np.int64)
    test_negatives = np.zeros((num_users, 99), dtype=np.int64)
    all_items = np.arange(num_items, dtype=np.int64)
    for u in range(num_users):
        pos = user_pos_set[u]
        # Sample without collisions with any positive interaction.
        candidates = np.setdiff1d(all_items, np.fromiter(pos, dtype=np.int64), assume_unique=True)
        # MovieLens 100K has 1682 items; every user has <<1682 positives so this is safe.
        val_negatives[u] = rng.choice(candidates, size=99, replace=False)
        test_negatives[u] = rng.choice(candidates, size=99, replace=False)

    return DataSplit(
        num_users=num_users,
        num_items=num_items,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        val_negatives=val_negatives,
        test_negatives=test_negatives,
        user_pos_set=user_pos_set,
    )


class NCFTrainDataset(Dataset):
    """
    Training dataset that re-samples negatives every epoch.

    Each epoch, for every positive (u, i) we sample `num_negatives` items j
    such that (u, j) is not a training/val/test positive. Labels are 1 for
    positives and 0 for negatives.

    Call `resample()` at the start of every epoch.
    """

    def __init__(self, split: DataSplit, num_negatives: int = 4, seed: int = 0):
        self.split = split
        self.num_negatives = num_negatives
        self.rng = np.random.default_rng(seed)

        self._users = np.empty(0, dtype=np.int64)
        self._items = np.empty(0, dtype=np.int64)
        self._labels = np.empty(0, dtype=np.float32)
        self.resample()

    def resample(self) -> None:
        """Re-sample negatives for a new epoch."""
        pos = self.split.train_pairs  # (N_train, 2)
        n_pos = len(pos)
        k = self.num_negatives
        num_items = self.split.num_items
        user_pos_set = self.split.user_pos_set

        users = np.empty(n_pos * (1 + k), dtype=np.int64)
        items = np.empty(n_pos * (1 + k), dtype=np.int64)
        labels = np.zeros(n_pos * (1 + k), dtype=np.float32)

        # Positives first.
        users[:n_pos] = pos[:, 0]
        items[:n_pos] = pos[:, 1]
        labels[:n_pos] = 1.0

        # Negatives: sample k per positive via rejection (fast for sparse data).
        neg_users = np.repeat(pos[:, 0], k)
        neg_items = self.rng.integers(0, num_items, size=n_pos * k)
        # Rejection loop for collisions with positives.
        for idx in range(len(neg_items)):
            u = neg_users[idx]
            while int(neg_items[idx]) in user_pos_set[u]:
                neg_items[idx] = self.rng.integers(0, num_items)

        users[n_pos:] = neg_users
        items[n_pos:] = neg_items

        # Shuffle.
        perm = self.rng.permutation(len(users))
        self._users = users[perm]
        self._items = items[perm]
        self._labels = labels[perm]

    def __len__(self) -> int:
        return len(self._users)

    def __getitem__(self, idx: int):
        return (
            int(self._users[idx]),
            int(self._items[idx]),
            float(self._labels[idx]),
        )


def collate_batch(batch):
    users = torch.as_tensor([b[0] for b in batch], dtype=torch.long)
    items = torch.as_tensor([b[1] for b in batch], dtype=torch.long)
    labels = torch.as_tensor([b[2] for b in batch], dtype=torch.float32)
    return users, items, labels


def default_data_path() -> str:
    """Best-effort path resolution — works both locally and on Kaggle."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "data", "u.data"),
        "/kaggle/input/movielens-100k/u.data",
        "/kaggle/input/ncf-assignment/u.data",
        "data/u.data",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return os.path.abspath(p)
    raise FileNotFoundError("u.data not found in any default location")
