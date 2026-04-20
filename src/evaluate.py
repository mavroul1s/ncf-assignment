"""
Leave-one-out evaluation: HR@K and NDCG@K.

Following the NCF paper:
- For each user, rank the held-out positive item among 99 sampled negatives.
- HR@K = 1 if the positive item is within the top-K predictions, else 0.
- NDCG@K = 1/log2(rank+1) if positive is within top-K (0-indexed rank +1),
  else 0. Averaged over users.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def _score_user_batch(model: nn.Module, users: torch.Tensor,
                      items: torch.Tensor) -> torch.Tensor:
    """Model scores as 1D tensor of shape (len(users),)."""
    logits = model(users, items)
    return logits


@torch.no_grad()
def evaluate_hr_ndcg(model: nn.Module,
                     pos_pairs: np.ndarray,    # (num_users, 2) [user, pos_item]
                     neg_items: np.ndarray,    # (num_users, 99)
                     ks: list[int] = (10,),
                     device: str = "cpu",
                     eval_batch_users: int = 256) -> dict:
    """
    Compute HR@K and NDCG@K for the leave-one-out protocol.

    Returns a dict like {'HR@10': float, 'NDCG@10': float, ...}.
    """
    model.eval()
    num_users = pos_pairs.shape[0]
    ks = sorted(set(int(k) for k in ks))
    max_k = max(ks)

    hits = {k: 0 for k in ks}
    ndcgs = {k: 0.0 for k in ks}

    for start in range(0, num_users, eval_batch_users):
        end = min(start + eval_batch_users, num_users)
        batch_users = pos_pairs[start:end, 0]    # (B,)
        batch_pos = pos_pairs[start:end, 1]      # (B,)
        batch_neg = neg_items[start:end]         # (B, 99)
        b = end - start

        # Build candidate tensor: [pos, neg1, ..., neg99] per user -> (B, 100)
        candidates = np.concatenate([batch_pos[:, None], batch_neg], axis=1)  # (B, 100)

        users_t = torch.as_tensor(np.repeat(batch_users, 100), dtype=torch.long, device=device)
        items_t = torch.as_tensor(candidates.reshape(-1), dtype=torch.long, device=device)

        scores = _score_user_batch(model, users_t, items_t).view(b, 100).cpu().numpy()

        # Rank: position of positive item among the 100 candidates (desc order).
        # Positive is column 0 — count how many candidates score strictly higher.
        pos_scores = scores[:, 0:1]
        # Tie-breaking: if a negative has the exact same score as positive, we
        # count it as ranked *after* the positive (optimistic). In practice ties
        # are rare for NCF continuous outputs.
        higher = (scores[:, 1:] > pos_scores).sum(axis=1)
        # rank is 0-indexed: 0 means positive is the top prediction.
        ranks = higher

        for k in ks:
            hit = (ranks < k).astype(np.float32)
            hits[k] += hit.sum()
            # NDCG: 1/log2(rank+2) for hits (since rank is 0-indexed, +2 gives log2(pos+1))
            ndcg = np.where(ranks < k, 1.0 / np.log2(ranks + 2), 0.0)
            ndcgs[k] += ndcg.sum()

    result = {}
    for k in ks:
        result[f"HR@{k}"] = float(hits[k] / num_users)
        result[f"NDCG@{k}"] = float(ndcgs[k] / num_users)
    return result


@torch.no_grad()
def evaluate_sklearn_nmf(score_matrix: np.ndarray,
                         pos_pairs: np.ndarray,
                         neg_items: np.ndarray,
                         user_pos_set: list[set],
                         ks: list[int] = (10,)) -> dict:
    """
    Evaluation for a non-neural NMF model that produces a dense score matrix.

    Args:
        score_matrix: shape (num_users, num_items), predicted scores.
        pos_pairs, neg_items: as in `evaluate_hr_ndcg`.
    """
    ks = sorted(set(int(k) for k in ks))
    num_users = pos_pairs.shape[0]
    hits = {k: 0 for k in ks}
    ndcgs = {k: 0.0 for k in ks}

    for idx in range(num_users):
        u = int(pos_pairs[idx, 0])
        pos_item = int(pos_pairs[idx, 1])
        negs = neg_items[idx]  # 99 items
        cand_items = np.concatenate([[pos_item], negs])
        cand_scores = score_matrix[u, cand_items]

        pos_score = cand_scores[0]
        rank = int((cand_scores[1:] > pos_score).sum())  # 0-indexed rank of pos

        for k in ks:
            if rank < k:
                hits[k] += 1
                ndcgs[k] += 1.0 / np.log2(rank + 2)

    return {
        **{f"HR@{k}": hits[k] / num_users for k in ks},
        **{f"NDCG@{k}": ndcgs[k] / num_users for k in ks},
    }
