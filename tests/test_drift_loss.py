"""Offline checks for the drifting loss port (starVLA/model/modules/action_model/drift_loss.py)."""

import torch

from starVLA.model.modules.action_model.drift_loss import drift_loss


def test_shapes_and_info():
    gen = torch.randn(4, 8, 6, requires_grad=True)
    pos = torch.randn(4, 1, 6)
    loss, info = drift_loss(gen, pos, R_list=(0.05, 0.2))
    assert loss.shape == (4,)
    assert set(info) == {"scale", "loss_0.05", "loss_0.2"}
    loss.mean().backward()
    assert gen.grad is not None and torch.isfinite(gen.grad).all()


def test_collapsed_samples_move_toward_positive_and_spread():
    torch.manual_seed(0)
    gen = (torch.zeros(1, 16, 4) + 0.01 * torch.randn(1, 16, 4)).requires_grad_()
    pos = torch.full((1, 1, 4), 3.0)
    loss, _ = drift_loss(gen, pos)
    loss.mean().backward()
    step = -gen.grad[0]
    # attraction: the mean update points to the positive
    assert torch.dot(step.mean(0), pos[0, 0] - gen[0].detach().mean(0)) > 0


def test_single_positive_per_condition_learns_two_modes():
    """p(a|c) is bimodal but every condition only ever shows ONE positive (as in VLA data)."""
    torch.manual_seed(0)
    S, B, G = 4, 32, 16
    direction = torch.nn.functional.normalize(torch.ones(S), dim=0)
    net = torch.nn.Sequential(torch.nn.Linear(1 + S, 128), torch.nn.SiLU(), torch.nn.Linear(128, S))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(600):
        c = torch.rand(B, 1) * 2 - 1
        sign = torch.randint(0, 2, (B, 1)) * 2 - 1
        pos = (0.3 * c + 0.5 * sign * direction)[:, None]
        eps = torch.randn(B * G, S)
        gen = net(torch.cat([c.repeat_interleave(G, 0), eps], -1)).view(B, G, S)
        loss, _ = drift_loss(gen, pos)
        opt.zero_grad()
        loss.mean().backward()
        opt.step()
    with torch.no_grad():
        c = torch.zeros(2000, 1)
        proj = net(torch.cat([c, torch.randn(2000, S)], -1)) @ direction
    frac_pos = (proj > 0).float().mean().item()
    assert 0.3 < frac_pos < 0.7, frac_pos  # both modes covered (a regressor would collapse to 0)
    assert proj.abs().mean().item() > 0.3
