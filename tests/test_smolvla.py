"""Offline checks for the SmolVLA port (no model download needed)."""

import types

import torch
from torch import nn

from starVLA.model.modules.vlm.SmolVLMWithExpert import _vision_embeddings_forward


def _reference_position_ids_4_52(num_patches_per_side: int, nb_h: int, nb_w: int) -> torch.Tensor:
    """Position ids of transformers 4.52 SmolVLMVisionEmbeddings (what SmolVLA was trained with)."""
    boundaries = torch.arange(1 / num_patches_per_side, 1.0, 1 / num_patches_per_side)
    coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_h)
    coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_w)
    bucket_h = torch.bucketize(coords_h, boundaries, right=True)
    bucket_w = torch.bucketize(coords_w, boundaries, right=True)
    return (bucket_h[:, None] * num_patches_per_side + bucket_w).flatten()


def test_vision_position_ids_match_transformers_4_52():
    patch_size, num_patches_per_side = 16, 32
    emb = nn.Module()
    emb.patch_size = patch_size
    emb.num_patches_per_side = num_patches_per_side
    emb.patch_embedding = lambda x: torch.zeros(x.shape[0], 1, x.shape[2] // patch_size, x.shape[3] // patch_size)
    # Identity-like position table: position_embedding(ids) returns the ids themselves.
    emb.position_embedding = nn.Embedding(num_patches_per_side**2, 1)
    emb.position_embedding.weight.data = torch.arange(num_patches_per_side**2, dtype=torch.float32)[:, None]
    forward = types.MethodType(_vision_embeddings_forward, emb)

    pixels = torch.zeros(2, 3, 512, 512)
    ids = forward(pixels, None)[..., 0].long()
    expected = _reference_position_ids_4_52(num_patches_per_side, 32, 32)
    assert torch.equal(ids[0], expected)
    assert torch.equal(ids[1], expected)
    # Sanity: the k-th patch row maps to bucket k (transformers 4.57 maps it to k - 1).
    assert ids[0, num_patches_per_side].item() == num_patches_per_side
