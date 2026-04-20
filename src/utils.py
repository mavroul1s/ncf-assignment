"""Miscellaneous helpers: reproducibility, logging, parameter counting."""
from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_auto() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@contextmanager
def timer(name: str = ""):
    start = time.time()
    yield
    print(f"[timer] {name} took {time.time() - start:.1f}s")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def aggregate_runs(records: list[dict], key: str) -> tuple[float, float]:
    """Return (mean, std) of a numeric key across records."""
    vals = np.asarray([r[key] for r in records], dtype=np.float64)
    return float(vals.mean()), float(vals.std(ddof=0))
