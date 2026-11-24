"""
SmolVLA framework (https://huggingface.co/papers/2506.01844).

SmolVLM2 backbone (first ``num_vlm_layers`` text layers) + a narrower Llama action
expert that interleaves cross-attention to the VLM K/V and self-attention, trained
with flow matching. The model math is a port of LeRobot's ``VLAFlowMatching``
(lerobot 0.7.0); see ``starVLA/model/modules/vlm/SmolVLMWithExpert.py``.

StarVLA-specific glue:
  - inputs follow the StarVLA example schema (``image`` / ``lang`` / ``state`` / ``action``);
    images are resized-with-padding and tokenization happens inside the framework;
  - actions/state are normalized by the StarVLA dataloader (not by LeRobot processors);
  - ``framework.smolvla.pretrained_policy_path`` initializes every weight from a LeRobot
    SmolVLA checkpoint (default ``lerobot/smolvla_base``); leave it empty to start from
    the plain SmolVLM2 weights (``load_vlm_weights``) with a randomly initialized expert.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.vlm.SmolVLMWithExpert import SmolVLMWithExpertModel
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


@dataclass
class SmolVLADefaultConfig:
    name: str = "SmolVLA"
    # `action_model` is read by the policy server (action_horizon) and defines the
    # real action dim; predictions are truncated from max_action_dim to action_dim.
    action_model: dict = field(
        default_factory=lambda: {
            "action_dim": 7,
            "action_horizon": 50,  # == LeRobot `chunk_size`; must match the dataloader chunk
        }
    )
    # Defaults follow LeRobot `SmolVLAConfig` / `lerobot/smolvla_base`.
    smolvla: dict = field(
        default_factory=lambda: {
            "vlm_model_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            "pretrained_policy_path": "lerobot/smolvla_base",  # HF repo id, local dir or .safetensors; "" = none
            "load_vlm_weights": True,  # only used when pretrained_policy_path is empty
            "num_images": -1,  # >0: every example must carry exactly this many views
            "max_state_dim": 32,
            "max_action_dim": 32,
            "resize_imgs_with_padding": [512, 512],  # (width, height)
            "tokenizer_max_length": 48,
            "pad_language_to": "max_length",
            "num_steps": 10,
            "freeze_vision_encoder": True,
            "train_expert_only": True,
            "train_state_proj": True,
            "attention_mode": "cross_attn",
            "prefix_length": 0,
            "num_expert_layers": 0,
            "num_vlm_layers": 16,
            "self_attn_every_n_layers": 2,
            "expert_width_multiplier": 0.75,
            "min_period": 4e-3,
            "max_period": 4.0,
            "add_image_special_tokens": False,
            # Mask out views whose pixels are all zero (a camera slot this embodiment does not
            # have), like LeRobot's padded "empty cameras": the slot keeps its position but the
            # VLM / expert never attend to its tokens.
            "mask_blank_images": False,
            # Training objective: "flow_matching" (LeRobot SmolVLA) or "drift" (Drifting Models,
            # arXiv 2602.04770): the expert becomes a one-step generator a = eps - v(eps, t=1),
            # trained with the drifting loss on raw (normalized) action chunks; inference is 1 NFE.
            "objective": "flow_matching",
            "drift_num_gen": 16,  # generated chunks per observation (they are also the negatives)
            "drift_R_list": [0.02, 0.05, 0.2],  # kernel temperatures
            # torch.compile the VLM / expert passes of the drift objective in training (static
            # shapes; ~25% faster step). Inference always runs eagerly. Needs a C compiler (CC).
            "torch_compile": False,
            # Forward-dynamics (fd) auxiliary loss: a small predictor f(a_gt, o_t) regresses the
            # vision latents (SigLIP + connector, the tokens the VLM consumes) of the frames at the
            # end of the chunk, L2 to a stop-gradient target from the same (trained) encoder.
            # Needs `future_image` in the batch (e.g. mixture robocasa_human50_all_chunk50_future50).
            "fd_loss_weight": 0.0,  # 0 disables the predictor entirely
            "fd_hidden": 512,
            "fd_layers": 4,
            "fd_heads": 8,
        }
    )


def create_sinusoidal_pos_embedding(time: torch.Tensor, dimension: int, min_period: float, max_period: float):
    """Sine-cosine embedding of scalar timesteps (openpi / LeRobot)."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float64, device=time.device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = time[:, None] * scaling_factor[None, :]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """big_vision prefix-LM mask: tokens attend to tokens with cumulative ar-mask <= theirs."""
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def resize_with_pad(img: torch.Tensor, height: int, width: int, pad_value: float = 0.0) -> torch.Tensor:
    """Resize (b, c, h, w) without distortion, padding on the left and top (SmolVLA convention)."""
    cur_height, cur_width = img.shape[2:]
    if cur_height == height and cur_width == width:
        return img
    ratio = max(cur_width / width, cur_height / height)
    resized_height, resized_width = int(cur_height / ratio), int(cur_width / ratio)
    resized_img = F.interpolate(img, size=(resized_height, resized_width), mode="bilinear", align_corners=False)
    pad_height, pad_width = max(0, height - resized_height), max(0, width - resized_width)
    return F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)


def pad_tensor(tensor: torch.Tensor, max_len: int, pad_value=0) -> torch.Tensor:
    b, d = tensor.shape[:2]
    padded = torch.full((b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device)
    padded[:, :d] = tensor
    return padded


class ForwardLatentPredictor(nn.Module):
    """f(a, z_t) -> z_{t+H}: transformer over [current image tokens, action tokens]; the outputs at
    the image-token positions are the predicted future image tokens."""

    def __init__(self, latent_dim: int, action_dim: int, chunk_size: int, hidden: int, layers: int, heads: int,
                 max_img_tokens: int = 1024):
        super().__init__()
        self.latent_in = nn.Linear(latent_dim, hidden)
        self.action_in = nn.Linear(action_dim, hidden)
        self.img_pos = nn.Parameter(torch.zeros(1, max_img_tokens, hidden))
        self.act_pos = nn.Parameter(torch.zeros(1, chunk_size, hidden))
        nn.init.normal_(self.img_pos, std=0.02)
        nn.init.normal_(self.act_pos, std=0.02)
        layer = nn.TransformerEncoderLayer(hidden, heads, 4 * hidden, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(hidden)
        self.latent_out = nn.Linear(hidden, latent_dim)

    def forward(self, z_t: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        n_img = z_t.shape[1]
        img = self.latent_in(z_t.to(self.latent_in.weight.dtype)) + self.img_pos[:, :n_img]
        act = self.action_in(actions.to(self.action_in.weight.dtype)) + self.act_pos[:, : actions.shape[1]]
        h = self.blocks(torch.cat([img, act], dim=1))
        return self.latent_out(self.norm(h[:, :n_img]))


@FRAMEWORK_REGISTRY.register("SmolVLA")
class SmolVLA(baseframework):
    """SmolVLA exposed through the StarVLA framework API."""

    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = merge_framework_config(SmolVLADefaultConfig, config)
        am = self.config.framework.action_model
        sc = self.config.framework.smolvla

        self.action_dim = int(am.action_dim)
        self.chunk_size = int(am.action_horizon)
        self.num_images = int(sc.num_images)
        self.max_state_dim = int(sc.max_state_dim)
        self.max_action_dim = int(sc.max_action_dim)
        self.resize_imgs_with_padding = tuple(int(x) for x in sc.resize_imgs_with_padding)
        self.tokenizer_max_length = int(sc.tokenizer_max_length)
        self.pad_language_to = str(sc.pad_language_to)
        self.num_steps = int(sc.num_steps)
        self.min_period = float(sc.min_period)
        self.max_period = float(sc.max_period)
        self.prefix_length = int(sc.prefix_length)
        self.add_image_special_tokens = bool(sc.add_image_special_tokens)
        self.mask_blank_images = bool(sc.get("mask_blank_images", False))
        self.objective = str(sc.get("objective", "flow_matching"))
        if self.objective not in ("flow_matching", "drift"):
            raise ValueError(f"[SmolVLA] unknown objective {self.objective!r}")
        self.drift_num_gen = int(sc.get("drift_num_gen", 16))
        self.drift_R_list = tuple(float(r) for r in sc.get("drift_R_list", (0.02, 0.05, 0.2)))
        self.torch_compile = bool(sc.get("torch_compile", False))
        self._compiled_fns = {}
        self.fd_loss_weight = float(sc.get("fd_loss_weight", 0.0))
        self._captured_img_embs = None  # set to a list to capture o_t image latents in embed_prefix
        if self.action_dim > self.max_action_dim:
            raise ValueError(f"action_dim={self.action_dim} exceeds max_action_dim={self.max_action_dim}")

        pretrained_policy_path = str(sc.pretrained_policy_path or "")
        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=str(sc.vlm_model_name),
            # Weights are overwritten by the SmolVLA checkpoint when one is given.
            load_vlm_weights=bool(sc.load_vlm_weights) and not pretrained_policy_path,
            train_expert_only=bool(sc.train_expert_only),
            freeze_vision_encoder=bool(sc.freeze_vision_encoder),
            attention_mode=str(sc.attention_mode),
            num_expert_layers=int(sc.num_expert_layers),
            num_vlm_layers=int(sc.num_vlm_layers),
            self_attn_every_n_layers=int(sc.self_attn_every_n_layers),
            expert_width_multiplier=float(sc.expert_width_multiplier),
        )
        vlm_hidden = self.vlm_with_expert.config.text_config.hidden_size
        expert_hidden = self.vlm_with_expert.expert_hidden_size
        self.state_proj = nn.Linear(self.max_state_dim, vlm_hidden)
        self.action_in_proj = nn.Linear(self.max_action_dim, expert_hidden)
        self.action_out_proj = nn.Linear(expert_hidden, self.max_action_dim)
        self.action_time_mlp_in = nn.Linear(expert_hidden * 2, expert_hidden)
        self.action_time_mlp_out = nn.Linear(expert_hidden, expert_hidden)
        for param in self.state_proj.parameters():
            param.requires_grad = bool(sc.train_state_proj)

        tokenizer = self.vlm_with_expert.tokenizer
        fake_image_token = tokenizer.convert_tokens_to_ids("<fake_token_around_image>")
        global_image_token = tokenizer.convert_tokens_to_ids("<global-img>")
        self.global_image_start_token = torch.tensor([fake_image_token, global_image_token], dtype=torch.long)
        self.image_end_token = torch.tensor([fake_image_token], dtype=torch.long)

        if self.fd_loss_weight > 0:
            self.fd_predictor = ForwardLatentPredictor(
                latent_dim=vlm_hidden,
                action_dim=self.action_dim,
                chunk_size=self.chunk_size,
                hidden=int(sc.get("fd_hidden", 512)),
                layers=int(sc.get("fd_layers", 4)),
                heads=int(sc.get("fd_heads", 8)),
            )

        # Building SmolVLM from its config inherits the config's bf16 dtype for some
        # sub-modules; keep fp32 master weights (trainer / server cast as configured).
        self.to(torch.float32)
        if pretrained_policy_path:
            self.load_smolvla_weights(pretrained_policy_path)

    # ------------------------------------------------------------------
    # Pretrained SmolVLA (LeRobot) weights
    # ------------------------------------------------------------------
    def load_smolvla_weights(self, path_or_repo: str):
        from safetensors.torch import load_file

        path = Path(path_or_repo)
        if not path.exists():
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(path_or_repo, allow_patterns=["*.safetensors", "*.json"]))
        if path.is_dir():
            path = path / "model.safetensors"
        state_dict = load_file(str(path))
        # LeRobot wraps everything in `SmolVLAPolicy.model`.
        state_dict = {(k[len("model.") :] if k.startswith("model.") else k): v for k, v in state_dict.items()}

        current = self.state_dict()
        filtered, shape_mismatch = {}, []
        for key, value in state_dict.items():
            if key not in current:
                continue
            if value.shape != current[key].shape:
                shape_mismatch.append((key, tuple(value.shape), tuple(current[key].shape)))
                continue
            # The LeRobot checkpoint stores the VLM in bf16; keep our (fp32) param dtypes so the
            # model is not left in mixed precision (trainer / server cast as configured).
            filtered[key] = value.to(current[key].dtype)
        missing, _ = self.load_state_dict(filtered, strict=False)
        logger.info(
            f"[SmolVLA] loaded {len(filtered)}/{len(current)} tensors from {path} "
            f"(missing={len(missing)}, ignored={len(state_dict) - len(filtered)}, shape_mismatch={len(shape_mismatch)})"
        )
        if shape_mismatch:
            raise ValueError(
                f"[SmolVLA] shape mismatch vs {path} (check framework.smolvla config): {shape_mismatch[:5]}"
            )
        missing = [k for k in missing if not k.startswith("fd_predictor.")]  # new module, not in LeRobot ckpts
        if missing:
            raise ValueError(f"[SmolVLA] checkpoint {path} is missing {len(missing)} tensors, e.g. {missing[:5]}")
        return self

    # ------------------------------------------------------------------
    # Input preparation (StarVLA examples -> tensors)
    # ------------------------------------------------------------------
    @property
    def _device(self):
        return self.action_in_proj.weight.device

    def _prepare_images(self, examples: list[dict]):
        per_example = []
        for example in examples:
            views = to_pil_preserve(example["image"])
            views = list(views) if isinstance(views, (list, tuple)) else [views]
            per_example.append(views)
        num_views = {len(v) for v in per_example}
        if len(num_views) != 1:
            raise ValueError(f"[SmolVLA] all examples in a batch must have the same number of views, got {num_views}")
        num_views = num_views.pop()
        if self.num_images > 0 and num_views != self.num_images:
            raise ValueError(
                f"[SmolVLA] expected {self.num_images} camera views (framework.smolvla.num_images), got {num_views}. "
                "Make the eval client send the same views, in the same order, as training."
            )

        width, height = self.resize_imgs_with_padding
        images, img_masks = [], []
        for view_idx in range(num_views):
            batch = np.stack([np.asarray(views[view_idx].convert("RGB")) for views in per_example])
            img = torch.from_numpy(batch).to(self._device).permute(0, 3, 1, 2).float() / 255.0
            if self.mask_blank_images:
                mask = img.flatten(1).amax(dim=1) > 0  # all-black view -> empty camera slot
            else:
                mask = torch.ones(len(examples), dtype=torch.bool, device=self._device)
            img = resize_with_pad(img, height, width, pad_value=0)
            images.append(img * 2.0 - 1.0)  # SigLIP expects [-1, 1]
            img_masks.append(mask)
        return images, img_masks

    def _prepare_language(self, examples: list[dict]):
        texts = []
        for example in examples:
            text = str(example.get("lang", example.get("language", "")))
            texts.append(text if text.endswith("\n") else f"{text}\n")
        tokens = self.vlm_with_expert.tokenizer(
            texts,
            max_length=self.tokenizer_max_length,
            truncation=True,
            padding=self.pad_language_to,
            padding_side="right",
            return_tensors="pt",
        )
        return tokens["input_ids"].to(self._device), tokens["attention_mask"].to(self._device, dtype=torch.bool)

    def _prepare_state(self, examples: list[dict]):
        states = torch.zeros(len(examples), self.max_state_dim, dtype=torch.float32)
        for idx, example in enumerate(examples):
            state = example.get("state")
            if state is None:
                continue
            state = np.asarray(state, dtype=np.float32)
            state = state[-1] if state.ndim == 2 else state.reshape(-1)
            if state.shape[0] > self.max_state_dim:
                raise ValueError(f"[SmolVLA] state dim {state.shape[0]} exceeds max_state_dim={self.max_state_dim}")
            states[idx, : state.shape[0]] = torch.from_numpy(state)
        return states.to(self._device)

    def _prepare_actions(self, examples: list[dict]):
        actions = torch.zeros(len(examples), self.chunk_size, self.max_action_dim, dtype=torch.float32)
        for idx, example in enumerate(examples):
            action = np.asarray(example["action"], dtype=np.float32)
            if action.shape != (self.chunk_size, self.action_dim):
                raise ValueError(
                    f"[SmolVLA] action shape {action.shape} != (action_horizon={self.chunk_size}, "
                    f"action_dim={self.action_dim}); align framework.action_model with the dataset."
                )
            actions[idx, :, : self.action_dim] = torch.from_numpy(action)
        return actions.to(self._device)

    # ------------------------------------------------------------------
    # VLAFlowMatching (ported from LeRobot)
    # ------------------------------------------------------------------
    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state):
        embs, pad_masks, att_masks = [], [], []
        for img, img_mask in zip(images, img_masks, strict=True):
            bsize = img.shape[0]
            if self.add_image_special_tokens:
                start_token = self.vlm_with_expert.embed_language_tokens(self.global_image_start_token.to(img.device))
                start_token = start_token.unsqueeze(0).expand(bsize, -1, -1)
                embs.append(start_token)
                pad_masks.append(torch.ones(start_token.shape[:2], dtype=torch.bool, device=img.device))
                att_masks += [0] * start_token.shape[1]

            img_emb = self.vlm_with_expert.embed_image(img)
            if self._captured_img_embs is not None:
                self._captured_img_embs.append(img_emb)
            img_emb = img_emb * torch.tensor(img_emb.shape[-1] ** 0.5, dtype=img_emb.dtype, device=img_emb.device)
            num_img_embs = img_emb.shape[1]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

            if self.add_image_special_tokens:
                end_token = self.vlm_with_expert.embed_language_tokens(self.image_end_token.to(img.device))
                end_token = end_token.unsqueeze(0).expand(bsize, -1, -1)
                embs.append(end_token)
                pad_masks.append(torch.ones(end_token.shape[:2], dtype=torch.bool, device=img.device))
                att_masks += [0] * end_token.shape[1]

        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        state_emb = self.state_proj(state.to(self.state_proj.weight.dtype))[:, None, :]
        embs.append(state_emb)
        bsize, device = state_emb.shape[0], state_emb.device
        pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
        # Image and language inputs do not attend to state or actions.
        att_masks += [1]

        embs = torch.cat([e.to(state_emb.dtype) for e in embs], dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=device)[None, :]

        if pad_masks.shape[1] < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)
        return embs, pad_masks, att_masks.expand(bsize, -1)

    def embed_suffix(self, noisy_actions, timestep):
        action_emb = self.action_in_proj(noisy_actions.to(self.action_in_proj.weight.dtype))
        bsize, device, dtype = action_emb.shape[0], action_emb.device, action_emb.dtype
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.vlm_with_expert.expert_hidden_size, self.min_period, self.max_period
        ).to(dtype=dtype)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(action_time_emb)))

        pad_masks = torch.ones(bsize, action_time_emb.shape[1], dtype=torch.bool, device=device)
        # Prefix tokens do not attend to action tokens; actions attend causally block-wise.
        att_masks = torch.ones(bsize, self.chunk_size, dtype=torch.bool, device=device)
        return action_time_emb, pad_masks, att_masks

    def _velocity(self, suffix_out):
        suffix_out = suffix_out[:, -self.chunk_size :]
        return self.action_out_proj(suffix_out.to(self.action_out_proj.weight.dtype)).float()

    def sample_time(self, bsize, device):
        beta = torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
        return (beta.sample((bsize,)) * 0.999 + 0.001).to(device=device, dtype=torch.float32)

    def compute_flow_loss(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):
        if noise is None:
            noise = torch.randn_like(actions)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        t = time[:, None, None]
        x_t = t * noise + (1 - t) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        v_t = self._velocity(suffix_out)
        return F.mse_loss(u_t, v_t, reduction="none")

    # ------------------------------------------------------------------
    # Drifting objective (one-step generator)
    # ------------------------------------------------------------------
    def _maybe_compiled(self, name: str):
        """Bound ``vlm_with_expert`` method, torch.compile-d in training when enabled."""
        fn = getattr(self.vlm_with_expert, name)
        if not (self.torch_compile and self.training and torch.is_grad_enabled()):
            return fn
        if name not in self._compiled_fns:
            self._compiled_fns[name] = torch.compile(fn, dynamic=False)
        return self._compiled_fns[name]

    def _prefill(self, images, img_masks, lang_tokens, lang_masks, state):
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = self._maybe_compiled("forward")(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        return prefix_pad_masks, past_key_values

    def generate_one_step(self, prefix_pad_masks, past_key_values, noise, num_gen: int = 1):
        """a = eps - v(eps, t=1): one Euler step of the flow parameterization (1 NFE).

        The prefix (VLM) is computed once per observation; its K/V are shared by the
        ``num_gen`` noise draws of that observation (noise rows are grouped per observation).
        """
        time = torch.ones(noise.shape[0], dtype=torch.float32, device=noise.device)
        suffix_embs, _, _ = self.embed_suffix(noise, time)
        suffix_out = self._maybe_compiled("forward_expert_grouped")(
            suffix_embs, prefix_pad_masks, past_key_values, num_gen=num_gen
        )
        return noise - self._velocity(suffix_out)

    def compute_drift_loss(self, images, img_masks, lang_tokens, lang_masks, state, actions):
        from starVLA.model.modules.action_model.drift_loss import drift_loss

        bsize, num_gen = actions.shape[0], self.drift_num_gen
        prefix_pad_masks, past_key_values = self._prefill(images, img_masks, lang_tokens, lang_masks, state)
        noise = torch.randn(
            bsize * num_gen, self.chunk_size, self.max_action_dim, dtype=torch.float32, device=actions.device
        )
        gen = self.generate_one_step(prefix_pad_masks, past_key_values, noise, num_gen=num_gen)
        # Drift on the real action dims only, one sample = the whole flattened chunk.
        gen = gen[:, :, : self.action_dim].float().reshape(bsize, num_gen, -1)
        pos = actions[:, :, : self.action_dim].float().reshape(bsize, 1, -1)
        loss, info = drift_loss(gen, pos, R_list=self.drift_R_list)

        with torch.no_grad():
            sq_err = ((gen - pos) ** 2).mean(-1)  # [B, G], per action dim
            metrics = {f"drift/{k}": v for k, v in info.items()}
            metrics["drift/mse_sample"] = sq_err.mean()  # E||a_gen - a_gt||^2 / dim
            metrics["drift/mse_best_of_g"] = sq_err.min(dim=1).values.mean()
            metrics["drift/mse_gen_mean"] = ((gen.mean(1, keepdim=True) - pos) ** 2).mean()
            metrics["drift/gen_std"] = gen.std(dim=1).mean()  # spread across the G draws
            values = torch.stack([v.float() for v in metrics.values()]).tolist()  # one host sync
        return loss.mean(), dict(zip(metrics.keys(), values))

    def compute_fd_loss(self, examples, current_img_embs, actions):
        """L2( sg(z_{t+H}), f(a_gt, z_t) ) with z = embed_image (SigLIP + connector) per camera."""
        future_images, _ = self._prepare_images([{"image": ex["future_image"]} for ex in examples])
        with torch.no_grad():  # stop-gradient target from the same, trained encoder
            target = torch.cat([self.vlm_with_expert.embed_image(img) for img in future_images], dim=1)
        z_t = torch.cat(current_img_embs, dim=1)
        pred = self.fd_predictor(z_t, actions[:, :, : self.action_dim])
        return F.mse_loss(pred.float(), target.float())

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, num_steps=None):
        bsize, device = state.shape[0], state.device
        num_steps = int(num_steps or self.num_steps)
        if noise is None:
            noise = torch.randn(bsize, self.chunk_size, self.max_action_dim, dtype=torch.float32, device=device)
        if self.objective == "drift":
            prefix_pad_masks, past_key_values = self._prefill(images, img_masks, lang_tokens, lang_masks, state)
            return self.generate_one_step(prefix_pad_masks, past_key_values, noise)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )

        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            time = torch.full((bsize,), 1.0 + step * dt, dtype=torch.float32, device=device)
            v_t = self.denoise_step(prefix_pad_masks, past_key_values, x_t, time)
            x_t = x_t + dt * v_t
        return x_t

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size, prefix_len = prefix_pad_masks.shape
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        position_ids = torch.sum(prefix_pad_masks, dim=-1)[:, None] + torch.cumsum(suffix_pad_masks, dim=1) - 1
        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
        )
        return self._velocity(outputs_embeds[1])

    # ------------------------------------------------------------------
    # StarVLA framework API
    # ------------------------------------------------------------------
    def forward(self, examples: list[dict] | None = None, **kwargs) -> dict:
        images, img_masks = self._prepare_images(examples)
        lang_tokens, lang_masks = self._prepare_language(examples)
        state = self._prepare_state(examples)
        actions = self._prepare_actions(examples)
        if self.objective == "drift":
            use_fd = self.fd_loss_weight > 0 and "future_image" in examples[0]
            self._captured_img_embs = [] if use_fd else None
            try:
                loss, metrics = self.compute_drift_loss(images, img_masks, lang_tokens, lang_masks, state, actions)
                if use_fd:
                    fd_loss = self.compute_fd_loss(examples, self._captured_img_embs, actions)
                    metrics["drift/drift_loss"] = loss.item()
                    metrics["fd/loss"] = fd_loss.item()
                    loss = loss + self.fd_loss_weight * fd_loss
            finally:
                self._captured_img_embs = None
            return {"action_loss": loss, "metrics": metrics}
        losses = self.compute_flow_loss(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            noise=kwargs.get("noise"),
            time=kwargs.get("time"),
        )
        # Padded action dims carry no signal; average over the real ones only.
        return {"action_loss": losses[:, :, : self.action_dim].mean()}

    @torch.inference_mode()
    def predict_action(self, examples: list[dict] | dict | None = None, **kwargs) -> dict:
        """Extra kwargs from eval clients (e.g. ``use_ddim``) are ignored; ``num_steps`` overrides Euler steps
        (flow matching only; the drift objective always samples in one step)."""
        if not isinstance(examples, list):
            examples = [examples]
        images, img_masks = self._prepare_images(examples)
        lang_tokens, lang_masks = self._prepare_language(examples)
        state = self._prepare_state(examples)
        actions = self.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise=kwargs.get("noise"),
            num_steps=kwargs.get("num_steps"),
        )
        return {"normalized_actions": actions[:, :, : self.action_dim].float().cpu().numpy()}
