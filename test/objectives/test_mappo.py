# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for MultiPPOLoss, MultiAgentGAE, and the ValueNorm family.

MAPPOLoss and IPPOLoss are deprecated aliases; they are covered by
``TestDeprecatedAliases``. All substantive loss tests use ``MultiPPOLoss``
directly.

These tests use synthetic tensordicts so they don't depend on any external
MARL env (VMAS / PettingZoo). They follow the layout pattern from
``test/test_cost.py::TestQMixer`` — per-agent observations under
``("agents", "observation")`` and team-shared reward / done at the root.
"""
from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from torchrl.modules import (
    MultiAgentMLP,
    PopArtValueNorm,
    ProbabilisticActor,
    RunningValueNorm,
    ValueNorm,
)
from torchrl.modules.distributions import NormalParamExtractor, TanhNormal
from torchrl.objectives import IPPOLoss, MAPPOLoss, MultiPPOLoss
from torchrl.objectives.utils import ValueEstimators
from torchrl.objectives.value import GAE, MultiAgentGAE


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _make_actor(n_agents=3, obs_dim=6, action_dim=2, share_params=True):
    backbone = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=2 * action_dim,
            n_agents=n_agents,
            centralized=False,
            share_params=share_params,
        ),
        NormalParamExtractor(),
    )
    module = TensorDictModule(
        backbone,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )
    return ProbabilisticActor(
        module=module,
        in_keys={"loc": ("agents", "loc"), "scale": ("agents", "scale")},
        out_keys=[("agents", "action")],
        distribution_class=TanhNormal,
        return_log_prob=True,
    )


def _make_critic(n_agents=3, obs_dim=6, centralized=True, share_params=True):
    return TensorDictModule(
        MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=1,
            n_agents=n_agents,
            centralized=centralized,
            share_params=share_params,
        ),
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")],
    )


def _make_data(
    B=2, T=10, n_agents=3, obs_dim=6, action_dim=2, per_agent_reward=False, device="cpu"
):
    torch.manual_seed(0)
    obs = torch.randn(B, T, n_agents, obs_dim, device=device)
    next_obs = torch.randn(B, T, n_agents, obs_dim, device=device)
    if per_agent_reward:
        reward_shape = (B, T, n_agents, 1)
    else:
        reward_shape = (B, T, 1)
    reward = torch.randn(*reward_shape, device=device)
    done = torch.zeros(B, T, 1, dtype=torch.bool, device=device)
    terminated = torch.zeros(B, T, 1, dtype=torch.bool, device=device)
    return TensorDict(
        {
            "agents": TensorDict({"observation": obs}, [B, T, n_agents]),
            "next": TensorDict(
                {
                    "agents": TensorDict({"observation": next_obs}, [B, T, n_agents]),
                    "reward": reward,
                    "done": done,
                    "terminated": terminated,
                },
                [B, T],
            ),
        },
        [B, T],
        device=device,
    )


def _attach_action_and_logprob(td: TensorDict, actor: ProbabilisticActor, loss):
    with torch.no_grad():
        sampled = actor(td.clone())
    td[("agents", "action")] = sampled[("agents", "action")]
    td[loss.tensor_keys.sample_log_prob] = sampled[loss.tensor_keys.sample_log_prob]
    return td


# --------------------------------------------------------------------------
# MultiAgentGAE
# --------------------------------------------------------------------------


class TestMultiAgentGAE:
    def test_team_reward_broadcast(self):
        n_agents = 3
        critic = _make_critic(n_agents=n_agents)
        gae = MultiAgentGAE(gamma=0.99, lmbda=0.95, value_network=critic)
        gae.set_keys(value=("agents", "state_value"))
        td = _make_data(n_agents=n_agents, per_agent_reward=False)
        gae(td)
        # advantage matches the critic's per-agent value shape
        assert td[gae.tensor_keys.advantage].shape[-2] == n_agents
        assert td[gae.tensor_keys.advantage].shape[-1] == 1
        assert (
            td[gae.tensor_keys.value_target].shape
            == td[gae.tensor_keys.advantage].shape
        )

    def test_per_agent_reward_passthrough(self):
        n_agents = 3
        critic = _make_critic(n_agents=n_agents, centralized=False)
        gae = MultiAgentGAE(gamma=0.99, lmbda=0.95, value_network=critic)
        gae.set_keys(value=("agents", "state_value"))
        td = _make_data(n_agents=n_agents, per_agent_reward=True)
        gae(td)
        assert td[gae.tensor_keys.advantage].shape[-2] == n_agents

    def test_broadcast_error_on_bad_shape(self):
        gae = MultiAgentGAE(gamma=0.99, lmbda=0.95, value_network=None, agent_dim=-2)
        # value has ndim=4 (B, T, n_agents, 1); a 2-D reward is neither
        # per-agent nor team-shared, so the helper must reject it.
        bad_tensor = torch.zeros(4, 5)
        target = torch.zeros(4, 5, 3, 1)
        with pytest.raises(ValueError, match="expected the reward/done/terminated"):
            gae._broadcast_to_agents(bad_tensor, target, agent_dim=-2)

    def test_value_estimator_enum_registered(self):
        # MAGAE is wired up in default_value_kwargs and the enum.
        from torchrl.objectives.utils import default_value_kwargs

        kw = default_value_kwargs(ValueEstimators.MAGAE)
        assert "gamma" in kw and "lmbda" in kw


# --------------------------------------------------------------------------
# ValueNorm — abstract base + the two concrete implementations
# --------------------------------------------------------------------------


class TestValueNormBase:
    def test_abstract_base_is_not_instantiable(self):
        with pytest.raises(TypeError):
            ValueNorm(shape=1)  # type: ignore[abstract]


class TestPopArtValueNorm:
    def test_running_stats_converge(self):
        torch.manual_seed(0)
        vn = PopArtValueNorm(shape=1)
        x = torch.randn(4096, 1) * 5.0 + 2.0
        for _ in range(200):
            vn.update(x)
        mean, var = vn._running_stats()
        assert abs(mean.item() - 2.0) < 0.2
        assert abs(var.sqrt().item() - 5.0) < 0.5

    def test_denormalize_inverts_normalize(self):
        torch.manual_seed(0)
        vn = PopArtValueNorm(shape=1)
        x = torch.randn(512, 1) * 3.0 + 1.0
        for _ in range(50):
            vn.update(x)
        y = torch.randn(64, 1) * 3.0 + 1.0
        recovered = vn.denormalize(vn.normalize(y))
        torch.testing.assert_close(recovered, y, rtol=1e-4, atol=1e-4)

    def test_bad_shape_raises(self):
        vn = PopArtValueNorm(shape=1)
        with pytest.raises(ValueError, match="trailing shape"):
            vn.update(torch.randn(4, 8))  # trailing 8 != 1


class TestRunningValueNorm:
    def test_running_stats_converge(self):
        """Exact running stats should be very tight even after few updates."""
        torch.manual_seed(0)
        vn = RunningValueNorm(shape=1)
        x = torch.randn(4096, 1) * 5.0 + 2.0
        for _ in range(20):
            vn.update(x)
        assert abs(vn.mean.item() - 2.0) < 0.1
        assert abs(vn._var().sqrt().item() - 5.0) < 0.1

    def test_denormalize_inverts_normalize(self):
        torch.manual_seed(0)
        vn = RunningValueNorm(shape=1)
        for _ in range(10):
            vn.update(torch.randn(256, 1) * 3.0 + 1.0)
        y = torch.randn(64, 1) * 3.0 + 1.0
        recovered = vn.denormalize(vn.normalize(y))
        torch.testing.assert_close(recovered, y, rtol=1e-4, atol=1e-4)

    def test_no_decay(self):
        """RunningValueNorm should not be biased by sample order (no EMA)."""
        torch.manual_seed(0)
        vn = RunningValueNorm(shape=1)
        # Feed two batches with very different scales; running stats should
        # land at the true combined mean rather than getting dominated by
        # whichever batch came last (which is what an EMA would do).
        a = torch.full((1000, 1), 1.0)
        b = torch.full((1000, 1), 5.0)
        vn.update(a)
        vn.update(b)
        # Combined mean of 1000 ones + 1000 fives = 3.0 exactly.
        assert abs(vn.mean.item() - 3.0) < 1e-4


# --------------------------------------------------------------------------
# MultiPPOLoss — centralised (MAPPO) mode
# --------------------------------------------------------------------------


class TestMultiPPOLossCentralized:
    def test_forward_shapes_and_backward(self):
        actor = _make_actor()
        critic = _make_critic(centralized=True)
        loss_mod = MultiPPOLoss(actor, critic, critic_type="centralized")
        loss_mod.set_keys(value=("agents", "state_value"), action=("agents", "action"))

        td = _make_data()
        _attach_action_and_logprob(td, actor, loss_mod)
        out = loss_mod(td)

        for k in ("loss_objective", "loss_entropy", "loss_critic"):
            assert k in out, f"missing key {k}"
            assert out[k].shape == torch.Size(
                []
            ), f"{k} should be scalar, got {out[k].shape}"

        total = out["loss_objective"] + out["loss_entropy"] + out["loss_critic"]
        total.backward()
        actor_grads = [
            p.grad
            for p in loss_mod.actor_network_params.values(True, True)
            if isinstance(p, torch.nn.Parameter) and p.grad is not None
        ]
        critic_grads = [
            p.grad
            for p in loss_mod.critic_network_params.values(True, True)
            if isinstance(p, torch.nn.Parameter) and p.grad is not None
        ]
        assert len(actor_grads) > 0, "actor received no grads"
        assert len(critic_grads) > 0, "critic received no grads"

    def test_critic_type_stored(self):
        actor = _make_actor()
        critic = _make_critic(centralized=True)
        loss_mod = MultiPPOLoss(actor, critic, critic_type="centralized")
        assert loss_mod.critic_type == "centralized"

    def test_invalid_critic_type_raises(self):
        actor = _make_actor()
        critic = _make_critic(centralized=True)
        with pytest.raises(ValueError, match="critic_type must be"):
            MultiPPOLoss(actor, critic, critic_type="global")

    def test_centralized_critic_uses_full_team_obs(self):
        """Perturbing one agent's obs must change every agent's value."""
        actor = _make_actor()
        critic = _make_critic(centralized=True)
        loss_mod = MultiPPOLoss(actor, critic, critic_type="centralized")
        loss_mod.set_keys(value=("agents", "state_value"), action=("agents", "action"))

        td = _make_data()
        with torch.no_grad():
            v_before = critic(td.clone())[("agents", "state_value")]
            td2 = td.clone()
            td2["agents", "observation"][..., 0, :] += 5.0
            v_after = critic(td2)[("agents", "state_value")]

        diff = (v_before - v_after).abs().mean(dim=(0, 1, 3))
        assert diff[1] > 1e-6, "Centralised critic ignored cross-agent obs change"
        assert diff[2] > 1e-6, "Centralised critic ignored cross-agent obs change"

    @pytest.mark.parametrize("share_params", [True, False])
    def test_share_params_modes(self, share_params):
        actor = _make_actor(share_params=share_params)
        critic = _make_critic(centralized=True, share_params=share_params)
        loss_mod = MultiPPOLoss(actor, critic, critic_type="centralized")
        loss_mod.set_keys(value=("agents", "state_value"), action=("agents", "action"))
        td = _make_data()
        _attach_action_and_logprob(td, actor, loss_mod)
        out = loss_mod(td)
        assert out["loss_objective"].shape == torch.Size([])

    def test_value_norm_round_trip(self):
        """With PopArtValueNorm, critic loss should remain bounded across many updates."""
        torch.manual_seed(0)
        actor = _make_actor()
        critic = _make_critic(centralized=True)
        vn = PopArtValueNorm(shape=1)
        loss_mod = MultiPPOLoss(actor, critic, critic_type="centralized", value_norm=vn)
        loss_mod.set_keys(value=("agents", "state_value"), action=("agents", "action"))

        critic_losses = []
        for step in range(8):
            td = _make_data(B=2, T=10)
            td["next", "reward"] *= (step + 1) * 10.0
            _attach_action_and_logprob(td, actor, loss_mod)
            out = loss_mod(td)
            critic_losses.append(out["loss_critic"].item())

        assert max(critic_losses) < 10.0, f"critic loss exploded: {critic_losses}"

    def test_default_value_estimator_is_magae(self):
        actor = _make_actor()
        critic = _make_critic(centralized=True)
        loss_mod = MultiPPOLoss(actor, critic, critic_type="centralized")
        loss_mod.set_keys(value=("agents", "state_value"), action=("agents", "action"))
        loss_mod.make_value_estimator()
        assert loss_mod.value_type == ValueEstimators.MAGAE
        assert isinstance(loss_mod._value_estimator, MultiAgentGAE)

    def test_make_value_estimator_falls_through_for_non_magae(self):
        actor = _make_actor()
        critic = _make_critic(centralized=False)
        loss_mod = MultiPPOLoss(actor, critic, critic_type="centralized")
        loss_mod.set_keys(value=("agents", "state_value"), action=("agents", "action"))
        loss_mod.make_value_estimator(ValueEstimators.GAE)
        assert isinstance(loss_mod._value_estimator, GAE)
        assert not isinstance(loss_mod._value_estimator, MultiAgentGAE)

    # -- bug-fix regression tests ------------------------------------------

    def test_value_norm_not_double_registered(self):
        """``value_norm`` must show up exactly once in ``state_dict()``."""
        actor = _make_actor()
        critic = _make_critic(centralized=True)
        vn = PopArtValueNorm(shape=1)
        loss_mod = MultiPPOLoss(actor, critic, critic_type="centralized", value_norm=vn)
        sd = loss_mod.state_dict()
        vn_keys = [k for k in sd.keys() if "value_norm" in k]
        assert sorted(vn_keys) == [
            "value_norm.debiasing_term",
            "value_norm.running_mean",
            "value_norm.running_mean_sq",
        ]

    def test_value_norm_state_dict_round_trip(self):
        torch.manual_seed(0)
        actor = _make_actor()
        critic = _make_critic(centralized=True)
        vn1 = PopArtValueNorm(shape=1)
        loss_a = MultiPPOLoss(actor, critic, critic_type="centralized", value_norm=vn1)
        for _ in range(20):
            vn1.update(torch.randn(64, 1) * 3.0 + 2.0)

        vn2 = PopArtValueNorm(shape=1)
        loss_b = MultiPPOLoss(
            _make_actor(),
            _make_critic(centralized=True),
            critic_type="centralized",
            value_norm=vn2,
        )
        loss_b.load_state_dict(loss_a.state_dict())
        torch.testing.assert_close(vn2.running_mean, vn1.running_mean)
        torch.testing.assert_close(vn2.running_mean_sq, vn1.running_mean_sq)
        torch.testing.assert_close(vn2.debiasing_term, vn1.debiasing_term)

    def test_value_norm_composes_with_clip_value(self):
        """``clip_value`` must keep working when ``value_norm`` is attached."""
        torch.manual_seed(0)
        actor = _make_actor()
        critic = _make_critic(centralized=True)
        vn = PopArtValueNorm(shape=1)
        loss_mod = MultiPPOLoss(
            actor, critic, critic_type="centralized", value_norm=vn, clip_value=0.2
        )
        loss_mod.set_keys(value=("agents", "state_value"), action=("agents", "action"))
        td = _make_data()
        _attach_action_and_logprob(td, actor, loss_mod)
        out = loss_mod(td)
        assert "value_clip_fraction" in out

    def test_value_norm_separate_losses_detaches_actor_grads(self):
        """``separate_losses=True`` must still detach when ``value_norm`` is on."""
        torch.manual_seed(0)
        actor = _make_actor()
        critic = _make_critic(centralized=True)
        vn = PopArtValueNorm(shape=1)
        loss_mod = MultiPPOLoss(
            actor,
            critic,
            critic_type="centralized",
            value_norm=vn,
            separate_losses=True,
        )
        loss_mod.set_keys(value=("agents", "state_value"), action=("agents", "action"))
        td = _make_data()
        _attach_action_and_logprob(td, actor, loss_mod)
        out = loss_mod(td)
        out["loss_critic"].backward()
        actor_grads = [
            p.grad
            for p in loss_mod.actor_network_params.values(True, True)
            if isinstance(p, torch.nn.Parameter)
        ]
        for g in actor_grads:
            assert g is None or torch.all(
                g == 0
            ), "actor params received critic-loss gradients despite separate_losses=True"

    def test_no_spurious_annotation_warnings_on_instantiation(self):
        """No ``actor_network_params wasn't part of the annotations`` warnings."""
        import warnings as _warnings

        actor = _make_actor()
        critic = _make_critic(centralized=True)
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            MultiPPOLoss(actor, critic, critic_type="centralized")
        annotation_warnings = [
            w for w in caught if "wasn't part of the annotations" in str(w.message)
        ]
        assert not annotation_warnings, (
            f"got annotation warnings during MultiPPOLoss instantiation: "
            f"{[str(w.message) for w in annotation_warnings]}"
        )


# --------------------------------------------------------------------------
# MultiPPOLoss — independent (IPPO) mode
# --------------------------------------------------------------------------


class TestMultiPPOLossIndependent:
    def test_forward_shapes_and_backward(self):
        actor = _make_actor()
        critic = _make_critic(centralized=False)
        loss_mod = MultiPPOLoss(actor, critic, critic_type="independent")
        loss_mod.set_keys(value=("agents", "state_value"), action=("agents", "action"))

        td = _make_data()
        _attach_action_and_logprob(td, actor, loss_mod)
        out = loss_mod(td)

        for k in ("loss_objective", "loss_entropy", "loss_critic"):
            assert out[k].shape == torch.Size([])

        (out["loss_objective"] + out["loss_critic"]).backward()

    def test_critic_type_stored(self):
        actor = _make_actor()
        critic = _make_critic(centralized=False)
        loss_mod = MultiPPOLoss(actor, critic, critic_type="independent")
        assert loss_mod.critic_type == "independent"

    def test_decentralized_critic_ignores_other_agents(self):
        """IPPO critic must depend only on the agent's own observation."""
        critic = _make_critic(centralized=False)
        td = _make_data()
        with torch.no_grad():
            v_before = critic(td.clone())[("agents", "state_value")]
            td2 = td.clone()
            td2["agents", "observation"][..., 0, :] += 5.0
            v_after = critic(td2)[("agents", "state_value")]

        diff = (v_before - v_after).abs().mean(dim=(0, 1, 3))
        assert diff[0] > 1e-6
        assert diff[1] < 1e-6
        assert diff[2] < 1e-6


# --------------------------------------------------------------------------
# Deprecated aliases — MAPPOLoss / IPPOLoss
# --------------------------------------------------------------------------


class TestDeprecatedAliases:
    def test_mappo_loss_emits_deprecation_warning(self):
        import warnings as _warnings

        actor = _make_actor()
        critic = _make_critic(centralized=True)
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            MAPPOLoss(actor, critic)
        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert any("MAPPOLoss" in str(w.message) for w in dep_warnings)
        assert any("MultiPPOLoss" in str(w.message) for w in dep_warnings)

    def test_ippo_loss_emits_deprecation_warning(self):
        import warnings as _warnings

        actor = _make_actor()
        critic = _make_critic(centralized=False)
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            IPPOLoss(actor, critic)
        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert any("IPPOLoss" in str(w.message) for w in dep_warnings)
        assert any("MultiPPOLoss" in str(w.message) for w in dep_warnings)

    def test_mappo_loss_is_multi_ppo_loss(self):
        import warnings as _warnings

        actor = _make_actor()
        critic = _make_critic(centralized=True)
        with _warnings.catch_warnings(record=True):
            _warnings.simplefilter("always")
            loss_mod = MAPPOLoss(actor, critic)
        assert isinstance(loss_mod, MultiPPOLoss)
        assert loss_mod.critic_type == "centralized"

    def test_ippo_loss_is_multi_ppo_loss(self):
        import warnings as _warnings

        actor = _make_actor()
        critic = _make_critic(centralized=False)
        with _warnings.catch_warnings(record=True):
            _warnings.simplefilter("always")
            loss_mod = IPPOLoss(actor, critic)
        assert isinstance(loss_mod, MultiPPOLoss)
        assert loss_mod.critic_type == "independent"

    def test_mappo_loss_forward_still_works(self):
        """Deprecated alias must still produce valid loss tensors."""
        import warnings as _warnings

        actor = _make_actor()
        critic = _make_critic(centralized=True)
        with _warnings.catch_warnings(record=True):
            _warnings.simplefilter("always")
            loss_mod = MAPPOLoss(actor, critic)
        loss_mod.set_keys(value=("agents", "state_value"), action=("agents", "action"))
        td = _make_data()
        _attach_action_and_logprob(td, actor, loss_mod)
        out = loss_mod(td)
        assert out["loss_objective"].shape == torch.Size([])

    def test_ippo_loss_forward_still_works(self):
        import warnings as _warnings

        actor = _make_actor()
        critic = _make_critic(centralized=False)
        with _warnings.catch_warnings(record=True):
            _warnings.simplefilter("always")
            loss_mod = IPPOLoss(actor, critic)
        loss_mod.set_keys(value=("agents", "state_value"), action=("agents", "action"))
        td = _make_data()
        _attach_action_and_logprob(td, actor, loss_mod)
        out = loss_mod(td)
        assert out["loss_objective"].shape == torch.Size([])


if __name__ == "__main__":
    import argparse

    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
