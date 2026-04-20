"""
Knowledge distillation (KD) for NeuMF.

Teacher: a fully trained NeuMF with the best setting from previous experiments.
Student: a smaller NeuMF (smaller embeddings and/or fewer MLP layers).

We implement three standard KD techniques (see Gou et al., 2021 KD survey,
https://arxiv.org/abs/2006.05525):

1. Response-based KD (Hinton et al., 2015):
   Student mimics the teacher's soft predictions (sigmoid outputs on the
   same (u, i) pairs). We combine with the usual BCE on the hard labels:
       L = alpha * BCE(student_logits, y) + (1 - alpha) * BCE(student_logits, teacher_prob)
   The second term is the binary-classification analogue of Hinton's soft-target
   KL term, with temperature T applied to both sides (via logits/T).

2. Feature-based KD (FitNets, Romero et al., 2014):
   Match an intermediate representation. We MSE-align the student's fused
   feature (concat of GMF element-wise product and MLP last hidden) to a
   projection of the teacher's fused feature. Combined with standard BCE.
       L = alpha * BCE + beta * MSE( proj(student_feature), teacher_feature )

3. Relation-based KD (RKD, Park et al., 2019):
   Preserve pairwise distances between samples. For a training mini-batch,
   compute L2 pairwise distances between teacher fused features, then between
   student fused features (with the same projection as FitNets), and minimize
   the Huber loss between them.
       L = alpha * BCE + gamma * Huber( pdist(s_proj) , pdist(t_feat) )
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import data as data_mod
from .evaluate import evaluate_hr_ndcg
from .models import NeuMF


@dataclass
class KDConfig:
    technique: str = "response"  # "response" | "feature" | "relation"
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    num_negatives: int = 4
    alpha: float = 0.5            # weight on hard BCE
    beta: float = 1.0             # feature-match weight
    gamma: float = 1.0            # relation-match weight
    temperature: float = 2.0      # for response-based KD
    eval_ks: tuple = (10,)


def distill_neumf(student: NeuMF,
                  teacher: NeuMF,
                  split: data_mod.DataSplit,
                  cfg: KDConfig,
                  device: str = "cpu",
                  seed: int = 0,
                  verbose: bool = False) -> dict:
    """
    Train `student` with knowledge distillation from a frozen `teacher`.
    Returns final test-set metrics (HR@K, NDCG@K) and per-epoch history.
    """
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student.to(device)

    # Feature-match projection (needed only if student's fused dim != teacher's).
    t_dim = teacher.gmf_embed_dim + teacher.mlp_last_hidden_dim
    s_dim = student.gmf_embed_dim + student.mlp_last_hidden_dim
    projection: Optional[nn.Linear] = None
    if cfg.technique in ("feature", "relation") and s_dim != t_dim:
        projection = nn.Linear(s_dim, t_dim, bias=False).to(device)
        nn.init.kaiming_uniform_(projection.weight, nonlinearity="linear")

    params = list(student.parameters())
    if projection is not None:
        params += list(projection.parameters())
    optim = torch.optim.Adam(params, lr=cfg.lr)
    bce = nn.BCEWithLogitsLoss()

    train_ds = data_mod.NCFTrainDataset(split, num_negatives=cfg.num_negatives, seed=seed)

    history = {"train_loss": [], "epoch_metrics": []}

    for epoch in range(1, cfg.epochs + 1):
        train_ds.resample()
        loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            collate_fn=data_mod.collate_batch,
            pin_memory=(device == "cuda"),
        )

        student.train()
        total_loss = 0.0
        total_n = 0
        for users, items, labels in loader:
            users = users.to(device, non_blocking=True)
            items = items.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Teacher forward (always).
            with torch.no_grad():
                t_logits = teacher(users, items)
                t_prob = torch.sigmoid(t_logits / cfg.temperature)
                t_feat = teacher.fused_feature(users, items)

            # Student forward.
            s_logits = student(users, items)

            hard_loss = bce(s_logits, labels)

            if cfg.technique == "response":
                # Response-based: match soft probabilities. We apply temperature
                # to both teacher and student logits and use BCE (binary analogue
                # of the Hinton KL term) as the soft-target loss.
                soft_loss = F.binary_cross_entropy_with_logits(
                    s_logits / cfg.temperature, t_prob
                ) * (cfg.temperature ** 2)
                loss = cfg.alpha * hard_loss + (1 - cfg.alpha) * soft_loss

            elif cfg.technique == "feature":
                s_feat = student.fused_feature(users, items)
                if projection is not None:
                    s_feat_proj = projection(s_feat)
                else:
                    s_feat_proj = s_feat
                feat_loss = F.mse_loss(s_feat_proj, t_feat)
                loss = cfg.alpha * hard_loss + cfg.beta * feat_loss

            elif cfg.technique == "relation":
                s_feat = student.fused_feature(users, items)
                if projection is not None:
                    s_feat_proj = projection(s_feat)
                else:
                    s_feat_proj = s_feat
                # Pairwise L2 distances within the batch.
                s_pdist = torch.cdist(s_feat_proj, s_feat_proj, p=2)
                t_pdist = torch.cdist(t_feat, t_feat, p=2)
                # Normalize by their mean (scale-invariant), as in RKD paper.
                s_pdist = s_pdist / (s_pdist.mean() + 1e-12)
                t_pdist = t_pdist / (t_pdist.mean() + 1e-12)
                rel_loss = F.smooth_l1_loss(s_pdist, t_pdist)
                loss = cfg.alpha * hard_loss + cfg.gamma * rel_loss

            else:
                raise ValueError(f"unknown KD technique: {cfg.technique}")

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            total_loss += loss.item() * labels.size(0)
            total_n += labels.size(0)

        avg_loss = total_loss / max(total_n, 1)
        history["train_loss"].append(avg_loss)

        metrics = evaluate_hr_ndcg(
            student, split.test_pairs, split.test_negatives,
            ks=list(cfg.eval_ks), device=device,
        )
        history["epoch_metrics"].append({"epoch": epoch, **metrics})
        if verbose:
            print(f"[KD {cfg.technique} epoch {epoch}] loss={avg_loss:.4f}  {metrics}")

    final = history["epoch_metrics"][-1].copy()
    final.pop("epoch", None)
    return {"final_metrics": final, "history": history}
