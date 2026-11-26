"""exp/cfg: classifier-free guidance in SmolVLA (small random model, GPU, no weight download)."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _model(p):
    from starVLA.model.framework.VLM4A.SmolVLA import SmolVLA

    cfg = OmegaConf.create({"framework": {"name": "SmolVLA", "action_model": {"action_dim": 12, "action_horizon": 8},
                                          "smolvla": {"num_images": 2, "load_vlm_weights": False, "pretrained_policy_path": None,
                                                      "num_vlm_layers": 2, "train_expert_only": False, "freeze_vision_encoder": False,
                                                      "cfg_drop_prob": p, "resize_imgs_with_padding": [64, 64]}}})
    torch.manual_seed(0)
    return SmolVLA(cfg).cuda().float()


def _examples(seed, n=2):
    r = np.random.default_rng(seed)
    return [{"image": [r.integers(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(2)],
             "lang": ["open the drawer", "pick the cup and place it in the sink please"][i % 2],
             "state": r.normal(size=(1, 32)).astype(np.float32), "action": r.normal(size=(8, 12)).astype(np.float32)} for i in range(n)]


def _inputs(m, ex):
    images, img_masks = m._prepare_images(ex)
    lang, lang_masks = m._prepare_language(ex)
    return images, img_masks, lang, lang_masks, m._prepare_state(ex)


@torch.no_grad()
def test_guidance_is_the_cfg_combination():
    m = _model(0.1).eval()
    args = _inputs(m, _examples(0))
    noise = torch.randn(2, m.chunk_size, m.max_action_dim, device="cuda")
    x = {w: m.sample_actions(*args, noise=noise, num_steps=1, guidance_scale=w) for w in (0.0, 1.0, 2.0, 3.5)}
    # one Euler step: x(w) = noise + dt * (v_null + w (v_cond - v_null)) is affine in w;
    # w = 1 takes the plain (unbatched) path, the others the batched cond/null path
    for w in (2.0, 3.5):
        torch.testing.assert_close(x[w], x[0.0] + w * (x[1.0] - x[0.0]), atol=2e-4, rtol=1e-4)
    assert not torch.allclose(x[0.0], x[1.0])  # null and conditional predictions differ


@torch.no_grad()
def test_null_rows_ignore_images_and_language():
    m = _model(0.1).eval()
    a, b = _inputs(m, _examples(1)), _inputs(m, _examples(2))
    b = (b[0], b[1], b[2], b[3], a[4])  # different images + instructions, same state
    null = torch.ones(2, dtype=torch.bool, device="cuda")
    ea, pa, _ = m.embed_prefix(*a, null_mask=null)
    eb, pb, _ = m.embed_prefix(*b, null_mask=null)
    torch.testing.assert_close(ea, eb)
    assert pa.all() and pb.all()  # null text uses a fixed-length mask (no instruction-length leak)
    ec, _, _ = m.embed_prefix(*a, null_mask=~null)
    assert not torch.allclose(ea, ec)


def test_training_drops_and_trains_null_tokens():
    m = _model(0.1).train()
    out = m.forward(_examples(3, n=4), null_mask=torch.tensor([True, False, True, False], device="cuda"))
    out["action_loss"].backward()
    assert torch.isfinite(out["action_loss"])
    assert m.cfg_null_image.grad is not None and m.cfg_null_image.grad.abs().sum() > 0
    assert m.cfg_null_lang.grad is not None and m.cfg_null_lang.grad.abs().sum() > 0
    torch.manual_seed(0)
    rate = np.mean([(torch.rand(10000, device="cuda") < m.cfg_drop_prob).float().mean().item()])
    assert abs(rate - 0.1) < 0.01


def test_default_model_is_unchanged_and_cfg_loads_plain_checkpoints():
    plain, cfg = _model(0.0), _model(0.1)
    assert not hasattr(plain, "cfg_null_image")
    missing, unexpected = cfg.load_state_dict(plain.state_dict(), strict=False)
    assert sorted(missing) == ["cfg_null_image", "cfg_null_lang"] and not unexpected
