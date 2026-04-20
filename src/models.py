"""
GMF, MLP, and NeuMF models (PyTorch).

Architecture follows the NCF paper (He et al., WWW 2017). For an MLP with `num_layers`
layers, the tower halves: the input to the first MLP layer is 2*mlp_embed_dim
(concatenation of user and item embeddings). Each subsequent layer halves.

NeuMF fuses GMF and MLP with independent embeddings and concatenates their last
hidden representations before a final sigmoid layer.

Pretraining: NeuMF can be initialized from trained GMF + MLP checkpoints via
`load_pretrained_weights`. The final output layer weights are mixed per the paper:
    h_neumf = [alpha * h_gmf ; (1-alpha) * h_mlp]
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _init_embedding(module: nn.Embedding, std: float = 0.01) -> None:
    nn.init.normal_(module.weight, mean=0.0, std=std)


def _init_linear(module: nn.Linear) -> None:
    # Kaiming uniform for ReLU layers (paper uses Gaussian with std=0.01 for everything;
    # we follow the common PyTorch NCF convention which trains similarly).
    nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class GMF(nn.Module):
    """Generalized Matrix Factorization under the NCF framework."""

    def __init__(self, num_users: int, num_items: int, embed_dim: int = 8):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.output = nn.Linear(embed_dim, 1)
        self.embed_dim = embed_dim

        _init_embedding(self.user_embed)
        _init_embedding(self.item_embed)
        _init_linear(self.output)

    def interaction(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Return the element-wise product feature (pre-output)."""
        pu = self.user_embed(users)
        qi = self.item_embed(items)
        return pu * qi

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        phi = self.interaction(users, items)
        logits = self.output(phi).squeeze(-1)  # raw logits; BCEWithLogitsLoss
        return logits


class MLP(nn.Module):
    """MLP branch of NCF. `num_layers` hidden layers, tower halving from 2*embed_dim."""

    def __init__(self, num_users: int, num_items: int, embed_dim: int = 32,
                 num_layers: int = 3, dropout: float = 0.0):
        super().__init__()
        assert num_layers >= 1, "MLP needs at least 1 hidden layer"
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)

        # Tower: input = 2*embed_dim, then halves at each layer.
        layers: list[nn.Module] = []
        in_dim = 2 * embed_dim
        self.layer_sizes: list[int] = []
        for l in range(num_layers):
            out_dim = max(in_dim // 2, 1)
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            self.layer_sizes.append(out_dim)
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(in_dim, 1)
        self.last_hidden_dim = in_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        _init_embedding(self.user_embed)
        _init_embedding(self.item_embed)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                _init_linear(m)
        _init_linear(self.output)

    def feature(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        pu = self.user_embed(users)
        qi = self.item_embed(items)
        z = torch.cat([pu, qi], dim=-1)
        return self.mlp(z)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        h = self.feature(users, items)
        return self.output(h).squeeze(-1)


class NeuMF(nn.Module):
    """
    Fusion of GMF and MLP with independent embeddings. Output is a single logit.

    Args:
        num_users, num_items: dataset sizes.
        gmf_embed_dim: embedding dim for the GMF branch (= predictive factor of GMF).
        mlp_embed_dim: embedding dim for the MLP branch. The concatenated input to
            the MLP is 2*mlp_embed_dim, and each layer halves.
        num_layers: number of hidden layers in the MLP branch (>=1).
        dropout: dropout between MLP layers (0 disables).
    """

    def __init__(self, num_users: int, num_items: int,
                 gmf_embed_dim: int = 8, mlp_embed_dim: int = 32,
                 num_layers: int = 3, dropout: float = 0.0):
        super().__init__()
        # GMF branch (no output head — its feature is element-wise product).
        self.gmf_user_embed = nn.Embedding(num_users, gmf_embed_dim)
        self.gmf_item_embed = nn.Embedding(num_items, gmf_embed_dim)

        # MLP branch.
        self.mlp_user_embed = nn.Embedding(num_users, mlp_embed_dim)
        self.mlp_item_embed = nn.Embedding(num_items, mlp_embed_dim)

        layers: list[nn.Module] = []
        in_dim = 2 * mlp_embed_dim
        self.mlp_layer_sizes: list[int] = []
        for l in range(num_layers):
            out_dim = max(in_dim // 2, 1)
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            self.mlp_layer_sizes.append(out_dim)
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

        # Final fusion: concat(GMF element-wise product, MLP last hidden) -> 1 logit.
        self.output = nn.Linear(gmf_embed_dim + in_dim, 1)

        self.gmf_embed_dim = gmf_embed_dim
        self.mlp_embed_dim = mlp_embed_dim
        self.mlp_last_hidden_dim = in_dim
        self.num_layers = num_layers

        _init_embedding(self.gmf_user_embed)
        _init_embedding(self.gmf_item_embed)
        _init_embedding(self.mlp_user_embed)
        _init_embedding(self.mlp_item_embed)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                _init_linear(m)
        _init_linear(self.output)

    # ------------------------------------------------------------------
    # Feature extraction hooks used by knowledge distillation.
    # ------------------------------------------------------------------
    def gmf_feature(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return self.gmf_user_embed(users) * self.gmf_item_embed(items)

    def mlp_feature(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.mlp_user_embed(users), self.mlp_item_embed(items)], dim=-1)
        return self.mlp(z)

    def fused_feature(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.gmf_feature(users, items), self.mlp_feature(users, items)], dim=-1)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        fused = self.fused_feature(users, items)
        return self.output(fused).squeeze(-1)

    # ------------------------------------------------------------------
    # Pretraining weight loading (paper section 3.4.1).
    # ------------------------------------------------------------------
    def load_pretrained_weights(self, gmf: GMF, mlp: MLP, alpha: float = 0.5) -> None:
        """Initialize NeuMF from trained GMF and MLP models."""
        assert gmf.embed_dim == self.gmf_embed_dim, (
            f"GMF embed dim mismatch: {gmf.embed_dim} vs {self.gmf_embed_dim}"
        )
        assert mlp.embed_dim == self.mlp_embed_dim, (
            f"MLP embed dim mismatch: {mlp.embed_dim} vs {self.mlp_embed_dim}"
        )
        assert mlp.num_layers == self.num_layers, (
            f"MLP layer count mismatch: {mlp.num_layers} vs {self.num_layers}"
        )

        # Copy embeddings.
        self.gmf_user_embed.weight.data.copy_(gmf.user_embed.weight.data)
        self.gmf_item_embed.weight.data.copy_(gmf.item_embed.weight.data)
        self.mlp_user_embed.weight.data.copy_(mlp.user_embed.weight.data)
        self.mlp_item_embed.weight.data.copy_(mlp.item_embed.weight.data)

        # Copy MLP hidden layers.
        src_linears = [m for m in mlp.mlp.modules() if isinstance(m, nn.Linear)]
        dst_linears = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)]
        assert len(src_linears) == len(dst_linears)
        for s, d in zip(src_linears, dst_linears):
            d.weight.data.copy_(s.weight.data)
            d.bias.data.copy_(s.bias.data)

        # Fuse output heads: h = [alpha * h_gmf ; (1-alpha) * h_mlp].
        with torch.no_grad():
            w_gmf = gmf.output.weight.data  # shape (1, gmf_embed_dim)
            b_gmf = gmf.output.bias.data
            w_mlp = mlp.output.weight.data  # shape (1, mlp_last_hidden)
            b_mlp = mlp.output.bias.data

            new_w = torch.cat([alpha * w_gmf, (1 - alpha) * w_mlp], dim=1)
            new_b = alpha * b_gmf + (1 - alpha) * b_mlp
            self.output.weight.data.copy_(new_w)
            self.output.bias.data.copy_(new_b)


# ----------------------------------------------------------------------
# Parameter counting helpers.
# ----------------------------------------------------------------------
def count_trainable_parameters(model: nn.Module) -> int:
    """Number of trainable weight parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def neumf_param_breakdown(model: NeuMF) -> dict:
    """Detailed parameter breakdown for reporting."""
    breakdown = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            breakdown[name] = p.numel()
    breakdown["_total"] = sum(breakdown.values())
    return breakdown
