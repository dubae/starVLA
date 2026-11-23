# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SmolVLM backbone + action expert used by SmolVLA.

Vendored from LeRobot (``lerobot/policies/smolvla/smolvlm_with_expert.py``,
lerobot 0.7.0) so that StarVLA does not depend on the ``lerobot`` package
(which pins torch/transformers versions incompatible with this repo).

Differences from upstream (numerically equivalent):
  - the KV cache is a plain ``{layer_idx: {"key_states", "value_states"}}`` dict
    instead of ``transformers.DynamicCache`` (whose API changed across releases);
  - the tokenizer is loaded instead of the full processor (only the tokenizer is used);
  - the SmolVLM vision position ids are computed by a local, version-independent
    function (see ``_vision_embeddings_forward``).

Parameter names are kept identical to upstream so that ``lerobot/smolvla_base``
weights load without key remapping (apart from the leading ``model.`` prefix).
"""

import copy
import types

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageTextToText,
    AutoTokenizer,
    SmolVLMForConditionalGeneration,
)


def apply_rope(x, positions, max_wavelength=10_000):
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)
    radians = radians[..., None, :]

    sin = torch.sin(radians)
    cos = torch.cos(radians)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin
    return res.to(dtype)


def _vision_embeddings_forward(self, pixel_values: torch.Tensor, patch_attention_mask: torch.Tensor) -> torch.Tensor:
    """``SmolVLMVisionEmbeddings.forward`` with the position ids of transformers 4.52 / >=5.

    transformers 4.57 computes the fractional patch coordinates as ``k / n * (1 - 1e-6)``,
    which falls just below the bucket boundary and shifts every patch position by one
    bucket. SmolVLA (``lerobot/smolvla_base``) was trained with the unshifted ids
    (``k / n``), so we pin that behavior here regardless of the installed version.
    """
    batch_size, _, max_im_h, max_im_w = pixel_values.shape
    embeddings = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)

    max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size
    if patch_attention_mask is None:
        patch_attention_mask = torch.ones(
            (batch_size, max_nb_patches_h, max_nb_patches_w), dtype=torch.bool, device=pixel_values.device
        )
    device = patch_attention_mask.device
    boundaries = torch.arange(
        1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side, dtype=torch.float32, device=device
    )
    nb_patches_h = patch_attention_mask[:, :, 0].sum(dim=1)
    nb_patches_w = patch_attention_mask[:, 0, :].sum(dim=1)
    h_indices = torch.arange(max_nb_patches_h, device=device, dtype=torch.float32)
    w_indices = torch.arange(max_nb_patches_w, device=device, dtype=torch.float32)
    coords_h = torch.clamp(h_indices[None, :] / nb_patches_h[:, None], max=1.0 - 1e-6)
    coords_w = torch.clamp(w_indices[None, :] / nb_patches_w[:, None], max=1.0 - 1e-6)
    bucket_h = torch.bucketize(coords_h, boundaries, right=True)
    bucket_w = torch.bucketize(coords_w, boundaries, right=True)
    pos_ids = (bucket_h[:, :, None] * self.num_patches_per_side + bucket_w[:, None, :]).reshape(batch_size, -1)

    flat_mask = patch_attention_mask.view(batch_size, -1)
    position_ids = torch.zeros((batch_size, max_nb_patches_h * max_nb_patches_w), dtype=torch.long, device=device)
    position_ids[flat_mask] = pos_ids[flat_mask]
    return embeddings + self.position_embedding(position_ids.to(self.position_embedding.weight.device))


def _vision_sdpa_backends():
    """Prefer cuDNN attention for the SigLIP encoder (StarVLA speed-up, not in LeRobot).

    The vision attention carries a mask, so SDPA cannot use FlashAttention and falls back
    to the memory-efficient kernel; on H200 the cuDNN backend made a full SmolVLA training
    step ~15% faster with bf16-level differences (grad cosine 0.99998). Unsupported setups
    fall through to the usual backends.
    """
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        backends = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
        return sdpa_kernel(backends, set_priority=True)
    except (ImportError, TypeError):  # torch < 2.6
        import contextlib

        return contextlib.nullcontext()


def get_intermediate_size(hidden_dim, ffn_dim_multiplier=4, multiple_of=256):
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


class SmolVLMWithExpertModel(nn.Module):
    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        load_vlm_weights: bool = True,
        train_expert_only: bool = True,
        freeze_vision_encoder: bool = False,
        attention_mode: str = "self_attn",
        num_expert_layers: int = -1,
        num_vlm_layers: int = -1,
        self_attn_every_n_layers: int = -1,
        expert_width_multiplier: float = 0.5,
    ):
        super().__init__()
        if load_vlm_weights:
            self.vlm = AutoModelForImageTextToText.from_pretrained(model_id, low_cpu_mem_usage=True)
            config = self.vlm.config
        else:
            config = AutoConfig.from_pretrained(model_id)
            self.vlm = SmolVLMForConditionalGeneration(config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        vision_embeddings = self.get_vlm_model().vision_model.embeddings
        vision_embeddings.forward = types.MethodType(_vision_embeddings_forward, vision_embeddings)
        if num_vlm_layers > 0:
            self.get_vlm_model().text_model.layers = self.get_vlm_model().text_model.layers[:num_vlm_layers]
        self.num_vlm_layers = len(self.get_vlm_model().text_model.layers)
        self.config = config

        # Smaller lm expert
        lm_expert_config = copy.deepcopy(config.text_config)
        hidden_size = lm_expert_config.hidden_size
        lm_expert_config.hidden_size = int(hidden_size * expert_width_multiplier)
        lm_expert_config.intermediate_size = get_intermediate_size(int(hidden_size * expert_width_multiplier))
        lm_expert_config.num_hidden_layers = self.num_vlm_layers
        if num_expert_layers > 0:
            assert (
                self.num_vlm_layers % num_expert_layers == 0
            ), f"num_vlm_layers={self.num_vlm_layers} is not a multiple of num_expert_layers={num_expert_layers}"
            lm_expert_config.num_hidden_layers = num_expert_layers
        self.lm_expert = AutoModel.from_config(lm_expert_config)

        self.num_expert_layers = len(self.lm_expert.layers)
        self.self_attn_every_n_layers = self_attn_every_n_layers
        if "cross" in attention_mode:
            # Reshape k/v projections to take the VLM's k/v as input.
            for layer_idx in range(len(self.lm_expert.layers)):
                if self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0:
                    continue
                self.lm_expert.layers[layer_idx].self_attn.k_proj = nn.Linear(
                    config.text_config.num_key_value_heads * config.text_config.head_dim,
                    lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,
                    bias=lm_expert_config.attention_bias,
                )
                self.lm_expert.layers[layer_idx].self_attn.v_proj = nn.Linear(
                    config.text_config.num_key_value_heads * config.text_config.head_dim,
                    lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,
                    bias=lm_expert_config.attention_bias,
                )
        # Remove unused embed_tokens
        self.lm_expert.embed_tokens = None

        self.num_attention_heads = self.config.text_config.num_attention_heads
        self.num_key_value_heads = self.config.text_config.num_key_value_heads

        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_mode = attention_mode
        self.expert_hidden_size = lm_expert_config.hidden_size
        self.set_requires_grad()

    def get_vlm_model(self):
        return self.vlm.model

    def set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()
            for params in self.get_vlm_model().vision_model.parameters():
                params.requires_grad = False
        if self.train_expert_only:
            self.vlm.eval()
            for params in self.vlm.parameters():
                params.requires_grad = False
        else:
            # Freeze parameters that never receive gradients (avoids DDP unused-param errors).
            last_layers = [self.num_vlm_layers - 1]
            if self.num_vlm_layers != self.num_expert_layers and self.num_vlm_layers % self.num_expert_layers == 0:
                last_layers.append(self.num_vlm_layers - 2)
            frozen_layers = ["lm_head", "text_model.norm.weight"]
            for layer in last_layers:
                frozen_layers.append(f"text_model.layers.{layer}.")
            for name, params in self.vlm.named_parameters():
                if any(k in name for k in frozen_layers):
                    params.requires_grad = False
        for name, params in self.lm_expert.named_parameters():
            if "lm_head" in name:
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()
        if self.train_expert_only:
            self.vlm.eval()
        return self

    def embed_image(self, image: torch.Tensor):
        vision_model = self.get_vlm_model().vision_model
        with _vision_sdpa_backends():
            image_hidden_states = vision_model(
                pixel_values=image.to(dtype=vision_model.dtype),
                patch_attention_mask=None,
            ).last_hidden_state
        # Modality projection & resampling
        return self.get_vlm_model().connector(image_hidden_states)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.get_vlm_model().text_model.get_input_embeddings()(tokens)

    def forward_attn_layer(
        self,
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
        use_cache: bool = True,
        fill_kv_cache: bool = True,
        past_key_values=None,
    ):
        query_states, key_states, value_states = [], [], []
        for i, hidden_states in enumerate(inputs_embeds):
            layer = model_layers[i][layer_idx]
            if hidden_states is None or layer is None:
                continue
            hidden_states = layer.input_layernorm(hidden_states)
            hidden_shape = (*hidden_states.shape[:-1], -1, layer.self_attn.head_dim)
            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            query_states.append(layer.self_attn.q_proj(hidden_states).view(hidden_shape))
            key_states.append(layer.self_attn.k_proj(hidden_states).view(hidden_shape))
            value_states.append(layer.self_attn.v_proj(hidden_states).view(hidden_shape))

        # B,L,H,D; concatenate on the token axis
        query_states = torch.cat(query_states, dim=1)
        key_states = torch.cat(key_states, dim=1)
        value_states = torch.cat(value_states, dim=1)
        seq_len = query_states.shape[1]
        if seq_len < position_ids.shape[1]:
            position_ids = position_ids[:, :seq_len]
            attention_mask = attention_mask[:, :seq_len, :seq_len]

        query_states = apply_rope(query_states, position_ids)
        key_states = apply_rope(key_states, position_ids)

        if use_cache:
            if fill_kv_cache:
                past_key_values[layer_idx] = {"key_states": key_states, "value_states": value_states}
            else:
                # Prefix K/V are fixed; append the suffix K/V without mutating the cache.
                key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                value_states = torch.cat([past_key_values[layer_idx]["value_states"], value_states], dim=1)

        att_output = self.eager_attention_forward(
            attention_mask, batch_size, head_dim, query_states, key_states, value_states
        )
        return [att_output], past_key_values

    def forward_cross_attn_layer(
        self,
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
        use_cache: bool = True,
        fill_kv_cache: bool = True,
        past_key_values=None,
    ):
        att_outputs = []
        assert len(inputs_embeds) == 2 or (
            use_cache and past_key_values is not None and not fill_kv_cache
        ), f"Both len(inputs_embeds) == {len(inputs_embeds)} and past_key_values is {past_key_values}"

        if len(inputs_embeds) == 2 and not past_key_values:
            # Prefix attention
            seq_len = inputs_embeds[0].shape[1]
            position_id, expert_position_id = position_ids[:, :seq_len], position_ids[:, seq_len:]
            prefix_attention_mask = attention_mask[:, :seq_len, :seq_len]

            layer = model_layers[0][layer_idx]
            hidden_states = layer.input_layernorm(inputs_embeds[0])
            hidden_shape = (*hidden_states.shape[:-1], -1, layer.self_attn.head_dim)
            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_states = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            query_states = apply_rope(query_state, position_id)
            key_states = apply_rope(key_state, position_id)

            att_output = self.eager_attention_forward(
                prefix_attention_mask, batch_size, head_dim, query_states, key_states, value_states
            )
            att_outputs.append(att_output)
        else:
            expert_position_id = position_ids

        if use_cache and past_key_values is not None and not fill_kv_cache:
            # Cross-attention reads the (post-RoPE) prefix K/V stored during prefill.
            key_states = past_key_values[layer_idx]["key_states"]
            value_states = past_key_values[layer_idx]["value_states"]

        # Expert
        expert_layer = model_layers[1][layer_idx]
        if expert_layer is not None:
            expert_hidden_states = expert_layer.input_layernorm(inputs_embeds[1])
            expert_hidden_shape = (*expert_hidden_states.shape[:-1], -1, expert_layer.self_attn.head_dim)
            expert_hidden_states = expert_hidden_states.to(dtype=expert_layer.self_attn.q_proj.weight.dtype)
            expert_query_state = expert_layer.self_attn.q_proj(expert_hidden_states).view(expert_hidden_shape)

            _key_states = key_states.to(dtype=expert_layer.self_attn.k_proj.weight.dtype).reshape(
                *key_states.shape[:2], -1
            )
            expert_key_states = expert_layer.self_attn.k_proj(_key_states).view(
                *_key_states.shape[:-1], -1, expert_layer.self_attn.head_dim
            )
            _value_states = value_states.to(dtype=expert_layer.self_attn.v_proj.weight.dtype).reshape(
                *value_states.shape[:2], -1
            )
            expert_value_states = expert_layer.self_attn.v_proj(_value_states).view(
                *_value_states.shape[:-1], -1, expert_layer.self_attn.head_dim
            )

            expert_position_id = expert_position_id - torch.min(expert_position_id, dim=1, keepdim=True).values
            expert_attention_mask = attention_mask[:, -inputs_embeds[1].shape[1] :, : expert_key_states.shape[1] :]
            expert_query_states = apply_rope(expert_query_state, expert_position_id)

            att_output = self.eager_attention_forward(
                expert_attention_mask,
                batch_size,
                head_dim,
                expert_query_states,
                expert_key_states,
                expert_value_states,
            )
            att_outputs.append(att_output)
        else:
            att_outputs.append(None)

        return att_outputs, past_key_values

    def get_model_layers(self, models: list) -> list:
        vlm_layers, expert_layers = [], []
        multiple_of = self.num_vlm_layers // self.num_expert_layers
        for i in range(self.num_vlm_layers):
            if multiple_of > 0 and i > 0 and i % multiple_of != 0:
                expert_layer = None
            else:
                expert_layer_index = i // multiple_of if multiple_of > 0 else i
                expert_layer = models[1].layers[expert_layer_index]
            vlm_layers.append(models[0].layers[i])
            expert_layers.append(expert_layer)
        return [vlm_layers, expert_layers]

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: dict | None = None,
        inputs_embeds: list | None = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
    ):
        models = [self.get_vlm_model().text_model, self.lm_expert]
        model_layers = self.get_model_layers(models)
        for hidden_states in inputs_embeds:
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]

        use_cache = bool(use_cache)
        fill_kv_cache = bool(fill_kv_cache)
        if use_cache and past_key_values is None:
            past_key_values = {}

        num_layers = self.num_vlm_layers
        head_dim = self.vlm.config.text_config.head_dim
        for layer_idx in range(num_layers):
            if (
                fill_kv_cache
                or "cross" not in self.attention_mode
                or (self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0)
            ):
                att_outputs, past_key_values = self.forward_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )
            else:
                att_outputs, past_key_values = self.forward_cross_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )
            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = model_layers[i][layer_idx]
                att_output = att_outputs[i] if i < len(att_outputs) else att_outputs[0]  # in case of self_attn
                if hidden_states is not None:
                    if layer is None:
                        outputs_embeds.append(hidden_states)
                        continue
                    end = start + hidden_states.shape[1]
                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start:end])
                    out_emb += hidden_states
                    after_first_residual = out_emb.clone()
                    out_emb = layer.post_attention_layernorm(out_emb)
                    out_emb = layer.mlp(out_emb)
                    out_emb += after_first_residual
                    outputs_embeds.append(out_emb)
                    start = end if len(att_outputs) == 1 else 0
                else:
                    outputs_embeds.append(None)
            inputs_embeds = outputs_embeds

        # final norm
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            outputs_embeds.append(models[i].norm(hidden_states) if hidden_states is not None else None)
        return outputs_embeds, past_key_values

    def eager_attention_forward(self, attention_mask, batch_size, head_dim, query_states, key_states, value_states):
        num_att_heads = self.num_attention_heads
        num_key_value_heads = self.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads

        sequence_length = key_states.shape[1]
        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )
        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        # Attention is upcast to float32 to match the original eager implementation.
        query_states = query_states.to(dtype=torch.float32).transpose(1, 2)
        key_states = key_states.to(dtype=torch.float32).transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5
        att_weights = att_weights.to(dtype=torch.float32)
        big_neg = torch.finfo(att_weights.dtype).min
        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
        probs = nn.functional.softmax(masked_att_weights, dim=-1).to(dtype=value_states.dtype)

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))
        att_output = att_output.permute(0, 2, 1, 3)
        return att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)
