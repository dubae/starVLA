"""Gradient-routing checks for SmolVLA ``vlm_kv_stopgrad`` (StarVLA extension).

With ``vlm_kv_stopgrad=True`` every VLM block hands its K/V to the action expert and
upper-layer gradients must not flow below that hand-off point:

* forward values (the flow-matching loss) are identical to the default model;
* VLM ``q_proj`` never receives gradient (it only feeds the VLM's own attention);
* cutting the expert's use of K_l/V_l removes *all* gradient to LN1/k_proj/v_proj of
  block l and to o_proj/LN2/FFN of block l-1, while block l's FFN still learns
  (from block l+1) — in the default model block l's k_proj keeps learning from above.

Builds a randomly initialised 6-layer SmolVLA from the SmolVLM2 config (needs the
config/tokenizer from the HF cache, no weights). Skipped without CUDA.
"""

import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip("transformers")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from starVLA.model.framework.VLM4A.SmolVLA import SmolVLA  # noqa: E402

NUM_LAYERS, CHUNK = 6, 8


def _build(stopgrad: bool) -> SmolVLA:
    cfg = OmegaConf.create(
        {
            "framework": {
                "name": "SmolVLA",
                "action_model": {"action_dim": 12, "action_horizon": CHUNK},
                "smolvla": {
                    "pretrained_policy_path": "",
                    "load_vlm_weights": False,
                    "num_vlm_layers": NUM_LAYERS,
                    "freeze_vision_encoder": False,
                    "train_expert_only": False,
                    "vlm_kv_stopgrad": stopgrad,
                },
            }
        }
    )
    torch.manual_seed(0)
    return SmolVLA(cfg).cuda()


def _inputs(model: SmolVLA):
    g = torch.Generator(device="cuda").manual_seed(1)
    b = 2
    images = [torch.rand(b, 3, 512, 512, device="cuda", generator=g) * 2 - 1 for _ in range(2)]
    img_masks = [torch.ones(b, dtype=torch.bool, device="cuda") for _ in images]
    tokens = model.vlm_with_expert.tokenizer(["open the drawer\n"] * b, max_length=16, padding="max_length",
                                             truncation=True, return_tensors="pt")
    lang, lang_mask = tokens["input_ids"].cuda(), tokens["attention_mask"].cuda().bool()
    state = torch.randn(b, 32, device="cuda", generator=g)
    actions = torch.randn(b, CHUNK, 32, device="cuda", generator=g)
    noise = torch.randn(b, CHUNK, 32, device="cuda", generator=g)
    time = torch.full((b,), 0.5, device="cuda")
    return images, img_masks, lang, lang_mask, state, actions, noise, time


def _grads(model: SmolVLA, cut_layer: int | None = None) -> dict:
    """Backprop the flow loss; optionally detach the K/V that the expert reads at ``cut_layer``."""
    vwe = model.vlm_with_expert
    current = {"layer": None}
    orig_self, orig_cross, orig_eager = vwe.forward_attn_layer, vwe.forward_cross_attn_layer, vwe.eager_attention_forward

    def track(fn):
        def wrapped(model_layers, inputs_embeds, layer_idx, *a, **k):
            current["layer"] = layer_idx
            return fn(model_layers, inputs_embeds, layer_idx, *a, **k)
        return wrapped

    def eager(mask, bsz, head_dim, q, k, v):
        # the expert's attention call is the one whose queries are the CHUNK action tokens
        if current["layer"] == cut_layer and q.shape[1] == CHUNK:
            k, v = k.detach(), v.detach()
        return orig_eager(mask, bsz, head_dim, q, k, v)

    vwe.forward_attn_layer, vwe.forward_cross_attn_layer = track(orig_self), track(orig_cross)
    vwe.eager_attention_forward = eager
    try:
        model.zero_grad(set_to_none=True)
        loss = model.compute_flow_loss(*_inputs(model)).mean()
        loss.backward()
    finally:
        vwe.forward_attn_layer, vwe.forward_cross_attn_layer, vwe.eager_attention_forward = orig_self, orig_cross, orig_eager
    layers = vwe.get_vlm_model().text_model.layers
    out = {"loss": loss.item()}
    for li, layer in enumerate(layers):
        for name, p in [("q", layer.self_attn.q_proj.weight), ("k", layer.self_attn.k_proj.weight),
                        ("v", layer.self_attn.v_proj.weight), ("ln1", layer.input_layernorm.weight),
                        ("o", layer.self_attn.o_proj.weight), ("ln2", layer.post_attention_layernorm.weight),
                        ("mlp", layer.mlp.down_proj.weight)]:
            out[(li, name)] = 0.0 if p.grad is None else p.grad.abs().sum().item()
    return out


@pytest.fixture(scope="module")
def models():
    return _build(False), _build(True)


def test_forward_identical(models):
    base, sg = models
    sg.load_state_dict(base.state_dict())
    with torch.no_grad():
        a = base.compute_flow_loss(*_inputs(base)).mean()
        b = sg.compute_flow_loss(*_inputs(sg)).mean()
    assert torch.allclose(a, b, rtol=1e-5, atol=1e-6)
    g_base, g_sg = _grads(base), _grads(sg)
    assert abs(g_base["loss"] - g_sg["loss"]) < 1e-5


def test_q_proj_receives_no_gradient(models):
    _, sg = models
    g = _grads(sg)
    for li in range(NUM_LAYERS):
        assert g[(li, "q")] == 0.0
        assert not sg.vlm_with_expert.get_vlm_model().text_model.layers[li].self_attn.q_proj.weight.requires_grad


@pytest.mark.parametrize("cut", [2, 3])  # 2: joint self-attention layer, 3: cross-attention layer
def test_kv_handoff_is_the_only_gradient_path(models, cut):
    base, sg = models
    g = _grads(sg, cut_layer=cut)
    # block `cut`: everything up to its K/V hand-off is cut off
    for name in ("k", "v", "ln1"):
        assert g[(cut, name)] == 0.0, (cut, name)
    # block `cut-1`: the part above its own hand-off only fed block `cut`'s K/V
    for name in ("o", "ln2", "mlp"):
        assert g[(cut - 1, name)] == 0.0, (cut - 1, name)
    # neighbours still learn through their own hand-offs
    assert g[(cut - 1, "k")] > 0 and g[(cut, "mlp")] > 0 and g[(cut + 1, "k")] > 0

    # default model: block `cut` still gets gradient from the layers above
    g_base = _grads(base, cut_layer=cut)
    assert g_base[(cut, "k")] > 0 and g_base[(cut - 1, "mlp")] > 0
