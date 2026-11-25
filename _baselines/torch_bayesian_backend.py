"""Optional PyTorch/CUDA kernels for the Bayesian ranking methods.

The public functions in this module accept and return NumPy arrays. PyTorch is
imported lazily, so the existing CPU-only environment does not depend on it.
All CUDA calculations use float64 to stay close to the NumPy implementations.
"""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Any, Dict, Optional

import numpy as np


@lru_cache(maxsize=None)
def _load_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for CUDA execution. Install a CUDA-enabled "
            "PyTorch build or use --device cpu/auto."
        ) from exc
    return torch


@lru_cache(maxsize=None)
def resolve_device(requested: str = "auto") -> str:
    """Resolve auto/cpu/cuda without making PyTorch a CPU-side dependency."""
    requested = str(requested or "auto").strip().lower()
    if requested == "cpu":
        return "cpu"
    if requested != "auto" and not requested.startswith("cuda"):
        raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'cuda:N'")

    try:
        torch = _load_torch()
    except RuntimeError:
        if requested == "auto":
            return "cpu"
        raise

    if not torch.cuda.is_available():
        if requested == "auto":
            return "cpu"
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    if requested.startswith("cuda:"):
        index = int(requested.split(":", 1)[1])
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {index} does not exist; found {torch.cuda.device_count()} device(s)"
            )
    return "cuda" if requested == "auto" else requested


def cuda_metadata(device: str) -> Dict[str, Any]:
    """Return lightweight backend metadata for result files."""
    actual = resolve_device(device)
    metadata: Dict[str, Any] = {"backend": actual}
    if actual.startswith("cuda"):
        torch = _load_torch()
        index = torch.device(actual).index
        if index is None:
            index = torch.cuda.current_device()
        metadata.update(
            {
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "cuda_device": index,
                "cuda_device_name": torch.cuda.get_device_name(index),
            }
        )
    return metadata


def _tensor(values, torch, device):
    return torch.as_tensor(values, dtype=torch.float64, device=device)


def _numpy(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _elo_change(old_skill, new_skill, scale_factor, torch):
    return scale_factor * torch.max(
        torch.abs(torch.log10(new_skill) - torch.log10(old_skill))
    )


def fit_bayes_bt_cuda(
    wins: np.ndarray,
    initial_skill: np.ndarray,
    *,
    a: float,
    b: float,
    max_updates: int,
    convergence_threshold: float,
    elo_scale_factor: float,
    device: str,
) -> Dict[str, Any]:
    """Run the Bayes-BT MAP updates on CUDA."""
    torch = _load_torch()
    with torch.no_grad():
        wins_t = _tensor(wins, torch, device)
        skill = _tensor(initial_skill, torch, device)
        win_totals = wins_t.sum(dim=1)
        pair_totals = wins_t + wins_t.transpose(0, 1)
        shape = (a - 1) + win_totals
        converged = False

        for iteration in range(max_updates):
            rate = b + (
                pair_totals / (skill[:, None] + skill[None, :])
            ).sum(dim=1)
            new_skill = shape / rate
            change = _elo_change(skill, new_skill, elo_scale_factor, torch)
            skill = new_skill
            if float(change.item()) < convergence_threshold:
                converged = True
                break

        # The original CPU uncertainty code uses rate = shape / final skill.
        posterior_rate = shape / skill
        return {
            "skill": _numpy(skill),
            "posterior_shape": _numpy(shape),
            "posterior_rate": _numpy(posterior_rate),
            "iterations": iteration + 1,
            "converged": converged,
        }


def _original_bbq_expectations(wins, skill, quality, torch):
    ratios = skill[:, None] / (skill[:, None] + skill[None, :])
    quality_expanded = quality[None, None, :]
    bt_probability = quality_expanded * ratios[:, :, None]
    gamma = bt_probability / (bt_probability + (1 - quality_expanded) / 2)
    off_diagonal = (~torch.eye(
        len(skill), dtype=torch.bool, device=skill.device
    ))[:, :, None]
    return gamma * off_diagonal


def fit_original_bbq_cuda(
    wins: np.ndarray,
    initial_skill: np.ndarray,
    initial_quality: np.ndarray,
    *,
    a: float,
    b: float,
    alpha: float,
    beta: float,
    max_updates: int,
    convergence_threshold: float,
    elo_scale_factor: float,
    device: str,
) -> Dict[str, Any]:
    """Run the original BBQ random-response EM updates on CUDA."""
    torch = _load_torch()
    with torch.no_grad():
        wins_t = _tensor(wins, torch, device)
        skill = _tensor(initial_skill, torch, device)
        quality = _tensor(initial_quality, torch, device)
        n = len(skill)
        upper_i, upper_j = torch.triu_indices(n, n, offset=1, device=device)
        quality_denominator = alpha + beta - 2 + (
            wins_t[upper_i, upper_j, :] + wins_t[upper_j, upper_i, :]
        ).sum(dim=0)
        converged = False

        for iteration in range(max_updates):
            gamma = _original_bbq_expectations(wins_t, skill, quality, torch)
            effective = wins_t * gamma
            shape = (a - 1) + effective.sum(dim=(1, 2))
            rate = b + (
                (effective + effective.transpose(0, 1))
                / (skill[:, None, None] + skill[None, :, None])
            ).sum(dim=(1, 2))
            new_skill = shape / rate

            quality_numerator = alpha - 1 + (
                effective[upper_i, upper_j, :]
                + effective[upper_j, upper_i, :]
            ).sum(dim=0)
            new_quality = quality_numerator / quality_denominator
            change = _elo_change(skill, new_skill, elo_scale_factor, torch)
            skill, quality = new_skill, new_quality
            if float(change.item()) < convergence_threshold:
                converged = True
                break

        gamma = _original_bbq_expectations(wins_t, skill, quality, torch)
        effective = wins_t * gamma
        shape = (a - 1) + effective.sum(dim=(1, 2))
        rate = b + (
            (effective + effective.transpose(0, 1))
            / (skill[:, None, None] + skill[None, :, None])
        ).sum(dim=(1, 2))
        return {
            "skill": _numpy(skill),
            "quality": _numpy(quality),
            "posterior_shape": _numpy(shape),
            "posterior_rate": _numpy(rate),
            "iterations": iteration + 1,
            "converged": converged,
        }
