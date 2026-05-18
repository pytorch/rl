# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for MAPPOLoss, IPPOLoss, MultiAgentGAE, and the ValueNorm family.

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
from torchrl.objectives import IPPOLoss, MAPPOLoss
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
# Value-estimator registry
# --------------------------------------------------------------------------


class TestValueEstimatorRegistry:
    def test_all_builtins_registered(self):
        """Every ValueEstimators enum member must have a registry entry."""
        from torchrl.objectives.utils import (
            _VALUE_ESTIMATOR_REGISTRY,
            get_value_estimator_entry,
        )

        for member in ValueEstimators:
            assert member in _VALUE_ESTIMATOR_REGISTRY, f"missing: {member}"
            entry = get_value_estimator_entry(member)
            assert entry.cls is not None
            assert "gamma" in entry.default_kwargs

    def test_string_alias_resolves(self):
        from torchrl.objectives.utils import get_value_estimator_entry

        assert (
            get_value_estimator_entry("gae").cls
            is get_value_estimator_entry(ValueEstimators.GAE).cls
        )
        assert (
            get_value_estimator_entry("magae").cls
            is get_value_estimator_entry(ValueEstimators.MAGAE).cls
        )

    def test_unknown_alias_raises(self):
        from torchrl.objectives.utils import get_value_estimator_entry

        with pytest.raises(KeyError, match="Unknown value estimator alias"):
            get_value_estimator_entry("not_a_real_estimator")

    def test_unknown_type_raises(self):
        from torchrl.objectives.utils import get_value_estimator_entry

        with pytest.raises(TypeError, match="must be a ValueEstimators"):
            get_value_estimator_entry(42)

    def test_register_and_dispatch_custom_estimator(self):
        """Adding a new estimator must not require touching any loss file."""
        from enum import Enum

        from torchrl.objectives.utils import (
            _VALUE_ESTIMATOR_REGISTRY,
            register_value_estimator,
        )
        from torchrl.objectives.value.advantages import GAE

        # We have to extend the enum at runtime for this test. Python's Enum
        # forbids appending, so we monkey-patch the registry directly with a
        # sentinel key — that's the path a third-party custom enum would take.
        class _Custom(Enum):
            FAKE = "fake"

        # Pretend we registered against a "real" entry by abusing _Custom.
        @register_value_estimator(
            _Custom.FAKE, default_kwargs={"gamma": 0.99, "lmbda": 0.5}
        )
        class _MyGAE(GAE):
            pass

        try:
            entry = _VALUE_ESTIMATOR_REGISTRY[_Custom.FAKE]
            assert entry.cls is _MyGAE
            assert entry.default_kwargs == {"gamma": 0.99, "lmbda": 0.5}
        finally:
            _VALUE_ESTIMATOR_REGISTRY.pop(_Custom.FAKE, None)

    def test_default_value_kwargs_reads_registry(self):
        """Back-compat shim must agree with the registry."""
        from torchrl.objectives.utils import (
            _VALUE_ESTIMATOR_REGISTRY,
            default_value_kwargs,
        )

        for member, entry in _VALUE_ESTIMATOR_REGISTRY.items():
            assert default_value_kwargs(member) == entry.default_kwargs


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
# MAPPOLoss
# --------------------------------------------------------------------------


class TestMAPPOLoss:
    def test_forward_shapes_and_backward(self):
        actor = _make_actor()
        critic = _make_critic(centralized=True)
        loss_mod = MAPPOLoss(actor, critic)
        loss_mod.set_keys(value=("agents", "state_value"), action=("agents", "action"))

        td = _make_data()
        _attach_action_and_logprob(td, actor, loss_mod)
        out = loss_mod(td)

        for k in ("loss_objective", "loss_entropy", "loss_critic"):
            assert k in out, f"missing key {k}"
            assert out[k].shape == torch.Size(
                []
            ), f"{k} should be scalar, got {out[k].shape}"

        # Gradients reach both actor and critic.
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

    def test_centralized_critic_uses_full_team_obs(self):
        """Perturbing one agent's obs must change every agent's value."""
        actor = _make_actor()
        critic = _make_critic(centralized=True)
        loss_mod = MAPPOLoss(actor, critic)
        loss_mod.set_keys(value=("agents", "state_value"), action=("agents", "action"))

        td = _make_data()
        with torch.no_grad():
            v_before = critic(td.clone())[("agents", "state_value")]
            td2 = td.clone()
            td2["agents", "observation"][..., 0, :] += 5.0  # perturb agent 0
            v_after = critic(td2)[("agents", "state_value")]

        # Other agents' values must change (centralised critic saw the perturb).
        diff = (v_before - v_after).abs().mean(dim=(0, 1, 3))
        assert diff[1] > 1e-6, "Centralised critic ignored cross-agent obs change"
        assert diff[2] > 1e-6, "Centralised critic ignored cross-agent obs change"

    @pytest.mark.parametrize("share_params", [True, False])
    def test_share_params_modes(self, share_params):
        actor = _make_actor(share_params=share_params)
        critic = _make_critic(centralized=True, share_params=share_params)
        loss_mod = MAPPOLoss(actor, critic)
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
        loss_mod = MAPPOLoss(actor, critic, value_norm=vn)
        loss_mod.set_keys(value=("agents", "state_value"), action=("agents", "action"))

        critic_losses = []
        for step in range(8):
            td = _make_data(B=2, T=10)
            # inflate reward scale over time to stress normalisation
            td["next", "reward"] *= (step + 1) * 10.0
            _attach_action_and_logprob(td, actor, loss_mod)
            out = loss_mod(td)
            critic_losses.append(out["loss_critic"].item())

        # Without ValueNorm a 10x reward inflation would blow critic loss up
        # quadratically; with ValueNorm it should stay roughly bounded.
        assert max(critic_losses) < 10.0, f"critic loss exploded: {critic_losses}"

    def test_default_value_estimator_is_magae(self):
        actor = _make_actor()
        critic = _make_critic(centralized=True)
        loss_mod = MAPPOLoss(actor, critic)
        loss_mod.set_keys(value=("agents", "state_value"), action=("agents", "action"))
        loss_mod.make_value_estimator()
        assert loss_mod.value_type == ValueEstimators.MAGAE
        assert isinstance(loss_mod._value_estimator, MultiAgentGAE)

    def test_make_value_estimator_falls_through_for_non_magae(self):
        """Selecting a non-MAGAE estimator goes through the parent class.

        The downstream call may still fail at runtime if the user feeds
        team-shared reward/done tensors (that is the whole reason
        :class:`MultiAgentGAE` exists), but we want to confirm the dispatch
        actually selects the requested estimator class.
        """
        actor = _make_actor()
        critic = _make_critic(centralized=False)
        loss_mod = MAPPOLoss(actor, critic)
        loss_mod.set_keys(value=("agents", "state_value"), action=("agents", "action"))
        loss_mod.make_value_estimator(ValueEstimators.GAE)
        assert isinstance(loss_mod._value_estimator, GAE)
        assert not isinstance(loss_mod._value_estimator, MultiAgentGAE)


# --------------------------------------------------------------------------
# IPPOLoss
# --------------------------------------------------------------------------


class TestIPPOLoss:
    def test_forward_shapes_and_backward(self):
        actor = _make_actor()
        critic = _make_critic(centralized=False)
        loss_mod = IPPOLoss(actor, critic)
        loss_mod.set_keys(value=("agents", "state_value"), action=("agents", "action"))

        td = _make_data()
        _attach_action_and_logprob(td, actor, loss_mod)
        out = loss_mod(td)

        for k in ("loss_objective", "loss_entropy", "loss_critic"):
            assert out[k].shape == torch.Size([])

        (out["loss_objective"] + out["loss_critic"]).backward()

    def test_decentralized_critic_ignores_other_agents(self):
        """IPPO critic must depend only on the agent's own observation."""
        critic = _make_critic(centralized=False)
        td = _make_data()
        with torch.no_grad():
            v_before = critic(td.clone())[("agents", "state_value")]
            td2 = td.clone()
            td2["agents", "observation"][..., 0, :] += 5.0  # perturb agent 0
            v_after = critic(td2)[("agents", "state_value")]

        diff = (v_before - v_after).abs().mean(dim=(0, 1, 3))
        # Agent 0's value changed.
        assert diff[0] > 1e-6
        # Agents 1, 2 must be unaffected.
        assert diff[1] < 1e-6
        assert diff[2] < 1e-6


if __name__ == "__main__":
    import argparse

    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
