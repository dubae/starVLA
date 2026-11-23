"""Drifting-model loss (Deng et al., "Generative Modeling via Drifting", arXiv 2602.04770).

PyTorch port of the official JAX ``drift_loss`` (https://github.com/lambertae/drifting,
``drift_loss.py``). Generated samples are pushed along the anti-symmetric drift field

    V(x) = E_{y+ ~ p, y- ~ q}[ k~(x, y+) k~(x, y-) (y+ - y-) ],   k(x, y) = exp(-||x - y|| / tau)

where the generated samples themselves are the negatives (y- ~ q). The training target is the
frozen drifted sample ``sg(x + V(x))``; the loss is ``||x - sg(x + V(x))||^2`` in the reference's
scale-normalized coordinates, summed over temperatures with per-temperature force normalization.

Conventions follow the reference exactly:
  * the kernel is normalized with a softmax over targets (row) and over generated samples
    (column), combined by a geometric mean;
  * a generated sample is masked out of its own target set;
  * distances / coordinates are normalized by one scalar ``scale`` = weighted mean pairwise
    distance over the whole call (not per group), and each temperature's force is normalized to
    unit RMS before the temperatures are summed.
"""

from __future__ import annotations

from typing import Sequence

import torch


def _cdist(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """[B, N, D] x [B, M, D] -> [B, N, M] Euclidean distance (clipped like the reference)."""
    xydot = torch.einsum("bnd,bmd->bnm", x, y)
    xnorms = torch.einsum("bnd,bnd->bn", x, x)
    ynorms = torch.einsum("bmd,bmd->bm", y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2 * xydot
    return torch.sqrt(torch.clamp(sq_dist, min=eps))


@torch.no_grad()
def _drifted_goal(old_gen, targets, targets_w, num_gen, num_neg, R_list):
    """Return (goal in scaled coords, coordinate scale, info) — no gradients."""
    B, C_g, S = old_gen.shape
    info = {}
    dist = _cdist(old_gen, targets)  # [B, C_g, C_g + C_n + C_p]
    weighted_dist = dist * targets_w[:, None, :]
    scale = weighted_dist.mean() / targets_w.mean()
    info["scale"] = scale

    scale_inputs = torch.clamp(scale / S**0.5, min=1e-3)  # coordinates of order 1
    old_gen_scaled = old_gen / scale_inputs
    targets_scaled = targets / scale_inputs

    dist_normed = dist / torch.clamp(scale, min=1e-3)
    # A generated sample never attracts / repels itself.
    eye = torch.eye(C_g, device=old_gen.device, dtype=old_gen.dtype)
    block_mask = torch.nn.functional.pad(eye, (0, targets.shape[1] - C_g))
    dist_normed = dist_normed + block_mask[None] * 100.0

    split_idx = num_gen + num_neg
    force_across_R = torch.zeros_like(old_gen_scaled)
    for R in R_list:
        logits = -dist_normed / R
        affinity = torch.softmax(logits, dim=-1)
        aff_transpose = torch.softmax(logits, dim=-2)
        affinity = torch.sqrt(torch.clamp(affinity * aff_transpose, min=1e-6))
        affinity = affinity * targets_w[:, None, :]

        aff_neg = affinity[:, :, :split_idx]
        aff_pos = affinity[:, :, split_idx:]
        r_coeff_neg = -aff_neg * aff_pos.sum(dim=-1, keepdim=True)
        r_coeff_pos = aff_pos * aff_neg.sum(dim=-1, keepdim=True)
        R_coeff = torch.cat([r_coeff_neg, r_coeff_pos], dim=2)

        total_force_R = torch.einsum("biy,byx->bix", R_coeff, targets_scaled)
        total_force_R = total_force_R - R_coeff.sum(dim=-1)[..., None] * old_gen_scaled
        f_norm_val = (total_force_R**2).mean()
        info[f"loss_{R}"] = f_norm_val
        force_across_R = force_across_R + total_force_R / torch.sqrt(torch.clamp(f_norm_val, min=1e-8))

    return old_gen_scaled + force_across_R, scale_inputs, info


def drift_loss(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: torch.Tensor | None = None,
    weight_gen: torch.Tensor | None = None,
    weight_pos: torch.Tensor | None = None,
    weight_neg: torch.Tensor | None = None,
    R_list: Sequence[float] = (0.02, 0.05, 0.2),
):
    """Drifting loss.

    Args:
        gen: [B, C_g, S] generated samples (gradients flow through these).
        fixed_pos: [B, C_p, S] data samples for the same condition.
        fixed_neg: [B, C_n, S] optional extra negatives (e.g. unconditional samples for CFG).
        weight_*: [B, C_*] optional per-sample weights (default 1).
        R_list: kernel temperatures (relative to the mean pairwise distance).

    Returns:
        loss: [B] per-group loss.
        info: dict of scalar diagnostics (``scale`` and ``loss_<R>`` = squared drift norm per R).
    """
    with torch.autocast(device_type=gen.device.type, enabled=False):
        gen = gen.float()
        fixed_pos = fixed_pos.float()
        B, C_g, _ = gen.shape
        if fixed_neg is None:
            fixed_neg = gen.new_zeros(B, 0, gen.shape[-1])
        fixed_neg = fixed_neg.float()
        C_n = fixed_neg.shape[1]
        weight_gen = gen.new_ones(B, C_g) if weight_gen is None else weight_gen.float()
        weight_pos = gen.new_ones(B, fixed_pos.shape[1]) if weight_pos is None else weight_pos.float()
        weight_neg = gen.new_ones(B, C_n) if weight_neg is None else weight_neg.float()

        old_gen = gen.detach()
        targets = torch.cat([old_gen, fixed_neg.detach(), fixed_pos.detach()], dim=1)
        targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)

        goal_scaled, scale_inputs, info = _drifted_goal(old_gen, targets, targets_w, C_g, C_n, tuple(R_list))
        diff = gen / scale_inputs - goal_scaled
        loss = (diff**2).mean(dim=(-1, -2))
    return loss, {k: v.detach() for k, v in info.items()}
