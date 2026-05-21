# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextlib
import functools

import pytest
import torch
from packaging import version

from tensordict import assert_allclose_td, TensorDict
from tensordict.nn import (
    set_composite_lp_aggregate,
    TensorDictModule,
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
)
from torch import nn

from torchrl.envs import GymEnv, InitTracker, SerialEnv
from torchrl.envs.libs.gym import _has_gym
from torchrl.envs.transforms import TransformedEnv
from torchrl.modules import GRUModule, LSTMModule, OneHotCategorical, set_recurrent_mode
from torchrl.modules.models.models import MLP
from torchrl.modules.tensordict_module.actors import ProbabilisticActor
from torchrl.objectives.common import LossModule
from torchrl.objectives.value.advantages import (
    GAE,
    TD0Estimator,
    TD1Estimator,
    TDLambdaEstimator,
    VTrace,
)
from torchrl.objectives.value.functional import (
    generalized_advantage_estimate,
    td0_advantage_estimate,
    td1_advantage_estimate,
    td_lambda_advantage_estimate,
    vec_generalized_advantage_estimate,
    vec_td1_advantage_estimate,
    vec_td_lambda_advantage_estimate,
    vtrace_advantage_estimate,
)
from torchrl.objectives.value.utils import _custom_conv1d, _make_gammas_tensor

from torchrl.testing import (  # noqa
    call_value_nets as _call_value_nets,
    dtype_fixture,
    get_available_devices,
    get_default_devices,
    PENDULUM_VERSIONED,
)

_TORCH_VERSION = version.parse(version.parse(torch.__version__).base_version)


class TestValues:
    @pytest.mark.parametrize(
        "estimator_cls,kwargs",
        [
            (TD0Estimator, {"gamma": 0.9}),
            (TD1Estimator, {"gamma": 0.9}),
            (TDLambdaEstimator, {"gamma": 0.9, "lmbda": 0.95}),
            (GAE, {"gamma": 0.9, "lmbda": 0.95}),
        ],
    )
    @pytest.mark.parametrize("shifted", [False, True])
    def test_value_chunk_size_matches_unchunked(self, estimator_cls, kwargs, shifted):
        torch.manual_seed(0)
        value_net = TensorDictModule(
            nn.Linear(3, 1),
            in_keys=["obs"],
            out_keys=["state_value"],
        )
        td = TensorDict(
            {
                "obs": torch.randn(4, 5, 3),
                "next": {
                    "obs": torch.randn(4, 5, 3),
                    "reward": torch.randn(4, 5, 1),
                    "done": torch.zeros(4, 5, 1, dtype=torch.bool),
                    "terminated": torch.zeros(4, 5, 1, dtype=torch.bool),
                },
            },
            [4, 5],
        )
        td["next", "done"][:, -1] = True
        td["next", "terminated"][:, -1] = True

        unchunked = estimator_cls(
            **kwargs,
            value_network=value_net,
            shifted=shifted,
        )
        chunked = estimator_cls(
            **kwargs,
            value_network=value_net,
            shifted=shifted,
            value_chunk_size=3,
        )

        expected = unchunked(td.clone())
        actual = chunked(td.clone())
        torch.testing.assert_close(actual["advantage"], expected["advantage"])
        torch.testing.assert_close(actual["value_target"], expected["value_target"])

    @pytest.mark.parametrize(
        "estimator_cls,kwargs",
        [
            (TD0Estimator, {"gamma": 0.9}),
            (TD1Estimator, {"gamma": 0.9}),
            (TDLambdaEstimator, {"gamma": 0.9, "lmbda": 0.95}),
            (GAE, {"gamma": 0.9, "lmbda": 0.95}),
        ],
    )
    @pytest.mark.parametrize("shifted", [False, True])
    def test_nan_next_obs_at_done_is_safe(self, estimator_cls, kwargs, shifted):
        """``("next", obs)`` may be ``NaN`` at trajectory ends.

        That happens by design when the buffer is fed by
        :class:`~torchrl.collectors.SyncDataCollector` configured with
        ``compact_obs=True`` and rehydrated via
        :class:`~torchrl.envs.transforms.NextStateReconstructor`. Without the
        sanitizer in
        :meth:`~torchrl.objectives.value.advantages.ValueEstimatorBase._call_value_nets`,
        ``V(NaN) = NaN`` propagates through the TD / GAE kernels because
        ``0 * NaN = NaN`` in IEEE 754 (the canonical ``(1 - done) * V_next``
        masking does not save us). Assert that the advantage and value target
        are finite everywhere, and that the terminated step gets the
        no-bootstrap target ``reward - V(s)``.
        """
        torch.manual_seed(0)
        value_net = TensorDictModule(
            nn.Linear(3, 1, bias=False),
            in_keys=["obs"],
            out_keys=["state_value"],
        )
        B, T, F = 2, 5, 3
        obs = torch.randn(B, T, F)
        next_obs = torch.empty(B, T, F)
        next_obs[:, :-1] = obs[:, 1:]
        next_obs[:, -1] = float("nan")  # mimic NextStateReconstructor at traj end
        done = torch.zeros(B, T, 1, dtype=torch.bool)
        done[:, -1] = True
        reward = torch.ones(B, T, 1)
        td = TensorDict(
            {
                "obs": obs,
                "next": {
                    "obs": next_obs,
                    "reward": reward,
                    "done": done.clone(),
                    "terminated": done.clone(),
                },
            },
            [B, T],
        )

        est = estimator_cls(**kwargs, value_network=value_net, shifted=shifted)
        out = est(td.clone())
        adv = out["advantage"]
        vt = out["value_target"]
        assert torch.isfinite(adv).all(), f"advantage contains non-finite: {adv}"
        assert torch.isfinite(vt).all(), f"value_target contains non-finite: {vt}"

        # At the terminated step the bootstrap is masked out: value_target
        # equals reward (advantage = reward - V(s)). Use the (uncorrupted)
        # raw value net to compute the expected V(s) at the last step.
        with torch.no_grad():
            v_s = value_net(td[..., -1].clone()).get("state_value")
        torch.testing.assert_close(vt[..., -1, :], reward[..., -1, :])
        torch.testing.assert_close(adv[..., -1, :], reward[..., -1, :] - v_s)

    @pytest.mark.parametrize(
        "estimator_cls,kwargs",
        [
            (TD0Estimator, {"gamma": 0.9}),
            (TD1Estimator, {"gamma": 0.9}),
            (TDLambdaEstimator, {"gamma": 0.9, "lmbda": 0.95}),
            (GAE, {"gamma": 0.9, "lmbda": 0.95}),
        ],
    )
    def test_missing_next_obs_shifted_is_safe(self, estimator_cls, kwargs):
        torch.manual_seed(0)
        value_net = TensorDictModule(
            nn.Linear(3, 1, bias=False),
            in_keys=["obs"],
            out_keys=["state_value"],
        )
        B, T, F = 2, 5, 3
        obs = torch.randn(B, T, F)
        done = torch.zeros(B, T, 1, dtype=torch.bool)
        done[:, 2] = True
        done[:, -1] = True
        reward = torch.ones(B, T, 1)
        td_missing_next = TensorDict(
            {
                "obs": obs,
                "next": {
                    "reward": reward,
                    "done": done.clone(),
                    "terminated": done.clone(),
                },
            },
            [B, T],
        )

        next_obs = torch.empty_like(obs)
        next_obs[:, :-1] = obs[:, 1:]
        next_obs[:, -1] = float("nan")
        next_obs[done.expand_as(next_obs)] = float("nan")
        td_reference = td_missing_next.clone()
        td_reference["next", "obs"] = next_obs

        est = estimator_cls(**kwargs, value_network=value_net, shifted=True)
        td_actual = td_missing_next.clone()
        actual = est(td_actual)
        expected = est(td_reference.clone())

        assert td_actual.get(("next", "obs"), default=None) is None
        torch.testing.assert_close(actual["advantage"], expected["advantage"])
        torch.testing.assert_close(actual["value_target"], expected["value_target"])
        assert torch.isfinite(actual["advantage"]).all()
        assert torch.isfinite(actual["value_target"]).all()

    def test_shifted_gae_accepts_noncanonical_strides(self):
        torch.manual_seed(0)
        value_net = TensorDictModule(
            nn.Linear(3, 1, bias=False),
            in_keys=["obs"],
            out_keys=["state_value"],
        )
        B, T, F = 2, 5, 3
        obs = torch.randn(B, T, F)
        done = torch.zeros(B, T, 1, dtype=torch.bool)
        done[:, -1] = True
        reward = torch.ones(B, T, 1)
        td = TensorDict(
            {
                "obs": obs,
                "next": {
                    "obs": torch.randn(B, T, F),
                    "reward": reward,
                    "done": done.clone(),
                    "terminated": done.clone(),
                },
            },
            [B, T],
        ).transpose(0, 1)

        assert not td["obs"].is_contiguous()
        est = GAE(gamma=0.9, lmbda=0.95, value_network=value_net, shifted=True)
        out = est(td)
        assert torch.isfinite(out["advantage"]).all()
        assert torch.isfinite(out["value_target"]).all()

    @pytest.mark.skipif(not _has_gym, reason="requires gym")
    def test_gae_multi_done(self):

        # constants
        batch_size = 10
        seq_size = 5
        n_dims = batch_size
        gamma = 0.99
        lmbda = 0.98

        env = SerialEnv(
            batch_size, [functools.partial(GymEnv, "CartPole-v1")] * batch_size
        )
        obs_size = env.full_observation_spec[env.observation_keys[0]].shape[-1]

        td = env.rollout(seq_size, break_when_any_done=False)
        # make the magic happen: swap dims and create an artificial ndim done state
        done = td["next", "done"].transpose(0, -1)
        terminated = td["next", "terminated"].transpose(0, -1)
        reward = td["next", "reward"].transpose(0, -1)
        td = td[:1]
        td["next", "done"] = done
        td["next", "terminated"] = terminated
        td["next", "reward"] = reward

        critic = TensorDictModule(
            nn.Linear(obs_size, n_dims),
            in_keys=[("observation",)],
            out_keys=[("state_value",)],
        )

        gae_shifted = GAE(gamma=gamma, lmbda=lmbda, value_network=critic, shifted=True)
        gae_no_shifted = GAE(
            gamma=gamma, lmbda=lmbda, value_network=critic, shifted=False
        )

        torch.testing.assert_close(
            gae_shifted(td.clone())["advantage"],
            gae_no_shifted(td.clone())["advantage"],
        )

    @pytest.mark.skipif(not _has_gym, reason="requires gym")
    @pytest.mark.parametrize("module", ["lstm", "gru"])
    @pytest.mark.parametrize("vectorized", [False, True])
    def test_gae_recurrent(self, module, vectorized):
        # Checks that shifted=True and False produce the same advantages
        # when an RNN value net is used, across both vectorized and
        # non-vectorized GAE — vectorized and shifted are orthogonal and
        # should all agree.
        env = SerialEnv(
            2,
            [
                functools.partial(
                    TransformedEnv, GymEnv(PENDULUM_VERSIONED()), InitTracker()
                )
                for _ in range(2)
            ],
        )
        env.set_seed(0)
        torch.manual_seed(0)
        if module == "lstm":
            recurrent_module = LSTMModule(
                input_size=env.observation_spec["observation"].shape[-1],
                hidden_size=64,
                in_keys=["observation", "rs_h", "rs_c"],
                out_keys=["intermediate", ("next", "rs_h"), ("next", "rs_c")],
                python_based=True,
                dropout=0,
            )
        elif module == "gru":
            recurrent_module = GRUModule(
                input_size=env.observation_spec["observation"].shape[-1],
                hidden_size=64,
                in_keys=["observation", "rs_h"],
                out_keys=["intermediate", ("next", "rs_h")],
                python_based=True,
                dropout=0,
            )
        else:
            raise NotImplementedError
        recurrent_module.eval()
        mlp_value = MLP(num_cells=[64], out_features=1)
        value_net = Seq(
            recurrent_module,
            Mod(mlp_value, in_keys=["intermediate"], out_keys=["state_value"]),
        )
        mlp_policy = MLP(num_cells=[64], out_features=1)
        policy_net = Seq(
            recurrent_module,
            Mod(mlp_policy, in_keys=["intermediate"], out_keys=["action"]),
        )
        env = env.append_transform(recurrent_module.make_tensordict_primer())
        vals = env.rollout(1000, policy_net, break_when_any_done=False)
        value_net(vals.copy())

        # Shifted
        gae_shifted = GAE(
            gamma=0.9,
            lmbda=0.99,
            value_network=value_net,
            shifted=True,
            vectorized=vectorized,
        )
        with set_recurrent_mode(True):
            r0 = gae_shifted(vals.copy())
        a0 = r0["advantage"]

        gae = GAE(
            gamma=0.9,
            lmbda=0.99,
            value_network=value_net,
            shifted=False,
            vectorized=vectorized,
            deactivate_vmap=True,
        )
        with pytest.raises(
            NotImplementedError,
            match="This implementation is not supported for torch<2.7",
        ) if torch.__version__ < "2.7" else contextlib.nullcontext():
            with set_recurrent_mode(True):
                r1 = gae(vals.copy())
            a1 = r1["advantage"]
            torch.testing.assert_close(a0, a1)

    def _build_shifted_test_td(self, *, with_internal_done: bool):
        """Build a rollout-shaped tensordict for shifted-mode tests."""
        B, T, obs_dim = 4, 8, 6
        all_obs = torch.randn(B, T + 1, obs_dim)
        obs = all_obs[:, :T].clone()
        next_obs = all_obs[:, 1:].clone()
        reward = torch.randn(B, T, 1)
        done = torch.zeros(B, T, 1, dtype=torch.bool)
        terminated = torch.zeros(B, T, 1, dtype=torch.bool)
        if with_internal_done:
            done[0, 3, 0] = True
            next_obs[0, 3] = torch.randn(obs_dim)
        done[:, -1, 0] = True
        td = TensorDict(
            {
                "observation": obs,
                "next": TensorDict(
                    {
                        "observation": next_obs,
                        "reward": reward,
                        "done": done,
                        "terminated": terminated,
                        "truncated": done & ~terminated,
                    },
                    batch_size=[B, T],
                ),
            },
            batch_size=[B, T],
        )
        td.refine_names(..., "time")
        return td, obs_dim

    def test_gae_shifted_requires_bool(self):
        torch.manual_seed(0)
        _, obs_dim = self._build_shifted_test_td(with_internal_done=True)
        value_net = TensorDictModule(
            nn.Linear(obs_dim, 1),
            in_keys=["observation"],
            out_keys=["state_value"],
        )
        with pytest.raises(ValueError, match="shifted must be a boolean"):
            GAE(
                gamma=0.9,
                lmbda=0.95,
                value_network=value_net,
                shifted="unexpected",
            )

    def test_gae_shifted_true_uses_budgeted_backend(self):
        torch.manual_seed(0)
        td, obs_dim = self._build_shifted_test_td(with_internal_done=True)
        value_net = TensorDictModule(
            nn.Linear(obs_dim, 1),
            in_keys=["observation"],
            out_keys=["state_value"],
        )
        gae = GAE(gamma=0.9, lmbda=0.95, value_network=value_net, shifted=True)
        out = gae(td.copy())
        assert gae.shifted is True
        assert out["shifted_valid"].shape == td.shape

    def _shifted_reference(self, td, shifted_budget=1):
        done = td["next", "done"]
        reset = done.squeeze(-1).clone()
        reset[..., -1] = False
        reset_cs = reset.to(torch.long).cumsum(-1)
        reset_before = torch.cat(
            [reset_cs.new_zeros((*reset.shape[:-1], 1)), reset_cs[..., :-1]], -1
        )
        root_slot = torch.arange(reset.shape[-1], device=reset.device) + reset_before
        valid = root_slot + 1 < reset.shape[-1] + shifted_budget
        next_valid = torch.cat(
            [valid[..., 1:], valid.new_zeros((*valid.shape[:-1], 1))], -1
        )
        last_valid = valid & ~next_valid
        td = td.clone()
        done = done.clone()
        terminated = td["next", "terminated"].clone()
        done[last_valid] = True
        terminated[last_valid] = False
        td["next", "done"] = done
        td["next", "terminated"] = terminated
        return td, valid

    @pytest.mark.parametrize("reset_steps", [[2], [1, 3]])
    @pytest.mark.parametrize("shifted_budget", [1, 2])
    def test_gae_shifted_retained_matches_unshifted(self, reset_steps, shifted_budget):
        torch.manual_seed(0)
        B, T, obs_dim = 2, 6, 6
        all_obs = torch.randn(B, T + 1, obs_dim)
        obs = all_obs[:, :T].clone()
        next_obs = all_obs[:, 1:].clone()
        done = torch.zeros(B, T, 1, dtype=torch.bool)
        for reset_step in reset_steps:
            done[:, reset_step, 0] = True
            next_obs[:, reset_step] = torch.randn(B, obs_dim)
        done[:, -1, 0] = True
        terminated = torch.zeros_like(done)
        reward = torch.randn(B, T, 1)
        td = TensorDict(
            {
                "observation": obs,
                "next": TensorDict(
                    {
                        "observation": next_obs,
                        "reward": reward,
                        "done": done,
                        "terminated": terminated,
                    },
                    [B, T],
                ),
            },
            [B, T],
        )
        value_net = TensorDictModule(
            nn.Linear(obs_dim, 1),
            in_keys=["observation"],
            out_keys=["state_value"],
        )
        ref_td, valid = self._shifted_reference(td, shifted_budget=shifted_budget)
        gae_unshifted = GAE(
            gamma=0.9,
            lmbda=0.95,
            value_network=value_net,
            shifted=False,
        )
        gae_drop = GAE(
            gamma=0.9,
            lmbda=0.95,
            value_network=value_net,
            shifted=True,
            shifted_budget=shifted_budget,
        )
        ref_td = gae_unshifted(ref_td)
        ref = ref_td["advantage"]
        out = gae_drop(td.clone())
        mask = valid.unsqueeze(-1)
        torch.testing.assert_close(out["advantage"][mask], ref[mask])
        torch.testing.assert_close(
            out["value_target"][mask], ref_td["value_target"][mask]
        )
        assert out["shifted_valid"].equal(valid)
        internal_resets = done.squeeze(-1).clone()
        internal_resets[..., -1] = False
        expected_drops = (internal_resets.sum(-1) + 1 - shifted_budget).clamp_min(0)
        assert (~valid).sum() == expected_drops.sum()
        assert (out["advantage"][~mask] == 0).all()

    def test_gae_shifted_without_next_observation(self):
        torch.manual_seed(0)
        B, T, obs_dim = 2, 6, 6
        obs = torch.randn(B, T, obs_dim)
        done = torch.zeros(B, T, 1, dtype=torch.bool)
        done[:, 2, 0] = True
        done[:, -1, 0] = True
        terminated = torch.zeros_like(done)
        reward = torch.randn(B, T, 1)
        td = TensorDict(
            {
                "observation": obs,
                "next": TensorDict(
                    {
                        "reward": reward,
                        "done": done,
                        "terminated": terminated,
                    },
                    [B, T],
                ),
            },
            [B, T],
        )
        value_net = TensorDictModule(
            nn.Linear(obs_dim, 1),
            in_keys=["observation"],
            out_keys=["state_value"],
        )
        out = GAE(
            gamma=0.9,
            lmbda=0.95,
            value_network=value_net,
            shifted=True,
        )(td.clone())
        assert torch.isfinite(out["advantage"]).all()
        assert out["shifted_valid"].shape == td.shape
        assert (~out["shifted_valid"]).sum() == done[..., :-1, :].sum()

    def test_gae_shifted_does_not_keep_stale_valid_mask(self):
        torch.manual_seed(0)
        B, T, obs_dim = 2, 6, 6
        value_net = TensorDictModule(
            nn.Linear(obs_dim, 1),
            in_keys=["observation"],
            out_keys=["state_value"],
        )
        gae = GAE(
            gamma=0.9,
            lmbda=0.95,
            value_network=value_net,
            shifted=True,
        )
        td_with_reset, _ = self._build_shifted_test_td(with_internal_done=True)
        out_with_reset = gae(td_with_reset.clone())
        assert out_with_reset.get("shifted_valid", default=None) is not None

        all_obs = torch.randn(B, T + 1, obs_dim)
        td_without_reset = TensorDict(
            {
                "observation": all_obs[:, :T].clone(),
                "next": TensorDict(
                    {
                        "observation": all_obs[:, 1:].clone(),
                        "reward": torch.randn(B, T, 1),
                        "done": torch.zeros(B, T, 1, dtype=torch.bool),
                        "terminated": torch.zeros(B, T, 1, dtype=torch.bool),
                    },
                    [B, T],
                ),
            },
            [B, T],
        )
        td_without_reset["next", "done"][:, -1, 0] = True
        out_without_reset = gae(td_without_reset)
        assert out_without_reset["shifted_valid"].shape == td_without_reset.shape
        assert out_without_reset["shifted_valid"].all()

    def test_gae_shifted_average_gae_uses_valid_only(self):
        torch.manual_seed(0)
        B, T, obs_dim = 2, 6, 4
        all_obs = torch.randn(B, T + 1, obs_dim)
        done = torch.zeros(B, T, 1, dtype=torch.bool)
        done[:, 2, 0] = True
        done[:, -1, 0] = True
        td = TensorDict(
            {
                "observation": all_obs[:, :T].clone(),
                "next": TensorDict(
                    {
                        "observation": all_obs[:, 1:].clone(),
                        "reward": torch.randn(B, T, 1),
                        "done": done,
                        "terminated": torch.zeros_like(done),
                    },
                    [B, T],
                ),
            },
            [B, T],
        )
        td["next", "observation"][:, 2] = torch.randn(B, obs_dim)
        value_net = TensorDictModule(
            nn.Linear(obs_dim, 1),
            in_keys=["observation"],
            out_keys=["state_value"],
        )
        out = GAE(
            gamma=0.9,
            lmbda=0.95,
            value_network=value_net,
            shifted=True,
            average_gae=True,
        )(td)
        valid = out["shifted_valid"].unsqueeze(-1)
        torch.testing.assert_close(
            out["advantage"][valid].mean(),
            torch.zeros((), dtype=out["advantage"].dtype),
        )

    def test_shifted_valid_masks_loss_reduction(self):
        loss = LossModule()
        loss.reduction = "mean"
        td = TensorDict(
            {"shifted_valid": torch.tensor([True, False, True])},
            [3],
        )
        value = torch.tensor([1.0, 100.0, 3.0])
        torch.testing.assert_close(loss._reduce_loss(value, td), torch.tensor(2.0))

    @pytest.mark.skipif(
        _TORCH_VERSION < version.parse("2.7"),
        reason="GAE shifted recurrent path uses torch.vmap chunked semantics that fall "
        "back to _pseudo_vmap on torch<2.7 (NotImplementedError).",
    )
    @pytest.mark.parametrize("module", ["lstm", "gru"])
    @pytest.mark.parametrize("shifted_budget", [1, 2])
    def test_gae_recurrent_shifted_matches_unshifted_isaac_shape(
        self, module, shifted_budget
    ):
        # Isaac-shaped regression test: recurrent value network, multi-trajectory
        # rollout with truncations every `episode_len` steps (never terminations),
        # and ``compact_obs=False`` semantics — ``("next", obs)`` is populated
        # everywhere, in particular at internal-done positions where it carries
        # the true pre-reset terminal observation (not the post-reset first obs
        # of the new episode).
        #
        # Under these conditions shifted=True should agree with the reference
        # path on retained samples and mark displaced suffix samples invalid.
        torch.manual_seed(0)
        B, T, obs_dim, hidden = 4, 16, 6, 8
        episode_len = 4  # internal truncation every 4 steps
        g = torch.Generator(device="cpu").manual_seed(0)
        all_obs = torch.randn(B, T + 1, obs_dim, generator=g)
        obs = all_obs[:, :T].clone()
        next_obs = all_obs[:, 1:].clone()
        done = torch.zeros(B, T, 1, dtype=torch.bool)
        for t in range(episode_len - 1, T, episode_len):
            done[:, t, 0] = True
            if t < T - 1:
                # Decouple next_obs[t] from obs[t+1]: env returned the true
                # truncation obs, then auto-reset gave a fresh obs[t+1].
                next_obs[:, t] = torch.randn(B, obs_dim, generator=g)
        # Isaac-Ant only ever truncates (max_episode_steps); never terminates.
        terminated = torch.zeros_like(done)
        truncated = done.clone()
        is_init = torch.zeros(B, T, 1, dtype=torch.bool)
        is_init[:, 0, 0] = True
        is_init[:, 1:][done[:, :-1]] = True
        next_is_init = done.clone()
        reward = torch.randn(B, T, 1, generator=g) * 0.1
        td = TensorDict(
            {
                "observation": obs,
                "is_init": is_init,
                "next": TensorDict(
                    {
                        "observation": next_obs,
                        "reward": reward,
                        "done": done,
                        "terminated": terminated,
                        "truncated": truncated,
                        "is_init": next_is_init,
                    },
                    [B, T],
                ),
            },
            [B, T],
        )

        if module == "lstm":
            recurrent_module = LSTMModule(
                input_size=obs_dim,
                hidden_size=hidden,
                num_layers=1,
                in_keys=["observation", "rs_h", "rs_c"],
                out_keys=["intermediate", ("next", "rs_h"), ("next", "rs_c")],
                python_based=True,
                recurrent_backend="pad",
                dropout=0,
            )
        else:
            recurrent_module = GRUModule(
                input_size=obs_dim,
                hidden_size=hidden,
                num_layers=1,
                in_keys=["observation", "rs_h"],
                out_keys=["intermediate", ("next", "rs_h")],
                python_based=True,
                recurrent_backend="pad",
                dropout=0,
            )
        recurrent_module.eval()
        value_net = Seq(
            recurrent_module,
            Mod(
                nn.Linear(hidden, 1), in_keys=["intermediate"], out_keys=["state_value"]
            ),
        )

        gae_unshifted = GAE(
            gamma=0.99,
            lmbda=0.95,
            value_network=value_net,
            shifted=False,
            deactivate_vmap=True,
            average_gae=False,
        )
        ref_td, valid = self._shifted_reference(td, shifted_budget=shifted_budget)
        gae_unshifted_drop_ref = GAE(
            gamma=0.99,
            lmbda=0.95,
            value_network=value_net,
            shifted=False,
            deactivate_vmap=True,
            average_gae=False,
            vectorized=False,
        )
        gae_shifted = GAE(
            gamma=0.99,
            lmbda=0.95,
            value_network=value_net,
            shifted=True,
            shifted_budget=shifted_budget,
            deactivate_vmap=False,
            average_gae=False,
            vectorized=False,
        )
        with set_recurrent_mode(True), torch.no_grad():
            adv_unshifted_drop = gae_unshifted_drop_ref(ref_td.clone())["advantage"]
            out_drop = gae_shifted(td.clone())
        mask = valid.unsqueeze(-1)
        torch.testing.assert_close(
            out_drop["advantage"][mask],
            adv_unshifted_drop[mask],
            atol=2.5e-1,
            rtol=2.5e-1,
        )
        assert out_drop["shifted_valid"].equal(valid)
        internal_resets = done.squeeze(-1).clone()
        internal_resets[..., -1] = False
        expected_drops = (internal_resets.sum(-1) + 1 - shifted_budget).clamp_min(0)
        assert (~valid).sum() == expected_drops.sum()
        assert (out_drop["advantage"][~mask] == 0).all()

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("gamma", [0.1, 0.5, 0.99])
    @pytest.mark.parametrize("lmbda", [0.1, 0.5, 0.99])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [200, 5, 3])
    # @pytest.mark.parametrize("random_gamma,rolling_gamma", [[True, False], [True, True], [False, None]])
    @pytest.mark.parametrize("random_gamma,rolling_gamma", [[False, None]])
    def test_tdlambda(self, device, gamma, lmbda, N, T, random_gamma, rolling_gamma):
        torch.manual_seed(0)

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        terminated = done.clone().bernoulli_(0.1)
        done = done.bernoulli_(0.1) | terminated
        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        if random_gamma:
            gamma = torch.rand_like(reward) * gamma

        next_state_value = torch.cat(
            [state_value[..., 1:, :], torch.randn_like(state_value[..., -1:, :])], -2
        )
        r1 = vec_td_lambda_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
        )
        r2 = td_lambda_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
        )
        r3, *_ = vec_generalized_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )
        torch.testing.assert_close(r3, r2, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(r1, r2, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(r1, r3, rtol=1e-4, atol=1e-4)

        # test when v' is not v from next step (not working with gae)
        next_state_value = torch.randn_like(next_state_value)
        r1 = vec_td_lambda_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
        )
        r2 = td_lambda_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
        )
        torch.testing.assert_close(r1, r2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("gamma", [0.1, 0.99])
    @pytest.mark.parametrize("lmbda", [0.1, 0.99])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [3, 100])
    @pytest.mark.parametrize("feature_dim", [[5], [2, 5]])
    @pytest.mark.parametrize("random_gamma,rolling_gamma", [[False, None]])
    def test_tdlambda_multi(
        self, device, gamma, lmbda, N, T, random_gamma, rolling_gamma, feature_dim
    ):
        torch.manual_seed(0)
        D = feature_dim
        time_dim = -1 - len(D)
        done = torch.zeros(*N, T, *D, device=device, dtype=torch.bool)
        terminated = done.clone().bernoulli_(0.1)
        done = done.bernoulli_(0.1) | terminated
        reward = torch.randn(*N, T, *D, device=device)
        state_value = torch.randn(*N, T, *D, device=device)
        next_state_value = torch.randn(*N, T, *D, device=device)
        if random_gamma:
            gamma = torch.rand_like(reward) * gamma

        r1 = vec_td_lambda_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
            time_dim=time_dim,
        )
        r2 = td_lambda_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
            time_dim=time_dim,
        )
        if len(D) == 2:
            r3 = torch.cat(
                [
                    vec_td_lambda_advantage_estimate(
                        gamma,
                        lmbda,
                        state_value[..., i : i + 1, j],
                        next_state_value[..., i : i + 1, j],
                        reward[..., i : i + 1, j],
                        done=done[..., i : i + 1, j],
                        terminated=terminated[..., i : i + 1, j],
                        rolling_gamma=rolling_gamma,
                        time_dim=-2,
                    )
                    for i in range(D[0])
                    for j in range(D[1])
                ],
                -1,
            ).unflatten(-1, D)
            r4 = torch.cat(
                [
                    td_lambda_advantage_estimate(
                        gamma,
                        lmbda,
                        state_value[..., i : i + 1, j],
                        next_state_value[..., i : i + 1, j],
                        reward[..., i : i + 1, j],
                        done=done[..., i : i + 1, j],
                        terminated=terminated[..., i : i + 1, j],
                        rolling_gamma=rolling_gamma,
                        time_dim=-2,
                    )
                    for i in range(D[0])
                    for j in range(D[1])
                ],
                -1,
            ).unflatten(-1, D)
        else:
            r3 = torch.cat(
                [
                    vec_td_lambda_advantage_estimate(
                        gamma,
                        lmbda,
                        state_value[..., i : i + 1],
                        next_state_value[..., i : i + 1],
                        reward[..., i : i + 1],
                        done=done[..., i : i + 1],
                        terminated=terminated[..., i : i + 1],
                        rolling_gamma=rolling_gamma,
                        time_dim=-2,
                    )
                    for i in range(D[0])
                ],
                -1,
            )
            r4 = torch.cat(
                [
                    td_lambda_advantage_estimate(
                        gamma,
                        lmbda,
                        state_value[..., i : i + 1],
                        next_state_value[..., i : i + 1],
                        reward[..., i : i + 1],
                        done=done[..., i : i + 1],
                        terminated=terminated[..., i : i + 1],
                        rolling_gamma=rolling_gamma,
                        time_dim=-2,
                    )
                    for i in range(D[0])
                ],
                -1,
            )

        torch.testing.assert_close(r4, r2, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(r3, r1, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(r1, r2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("gamma", [0.1, 0.5, 0.99])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [3, 100])
    @pytest.mark.parametrize("random_gamma,rolling_gamma", [[False, None]])
    def test_td1(self, device, gamma, N, T, random_gamma, rolling_gamma):
        torch.manual_seed(0)

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        terminated = done.clone().bernoulli_(0.1)
        done = done.bernoulli_(0.1) | terminated
        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        next_state_value = torch.randn(*N, T, 1, device=device)
        if random_gamma:
            gamma = torch.rand_like(reward) * gamma

        r1 = vec_td1_advantage_estimate(
            gamma,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
        )
        r2 = td1_advantage_estimate(
            gamma,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
        )
        torch.testing.assert_close(r1, r2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("gamma", [0.1, 0.99])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [3, 5])
    @pytest.mark.parametrize("feature_dim", [[5], [2, 5]])
    @pytest.mark.parametrize("random_gamma,rolling_gamma", [[False, None]])
    def test_td1_multi(
        self, device, gamma, N, T, random_gamma, rolling_gamma, feature_dim
    ):
        torch.manual_seed(0)

        D = feature_dim
        time_dim = -1 - len(D)
        done = torch.zeros(*N, T, *D, device=device, dtype=torch.bool)
        terminated = done.clone().bernoulli_(0.1)
        done = done.bernoulli_(0.1) | terminated
        reward = torch.randn(*N, T, *D, device=device)
        state_value = torch.randn(*N, T, *D, device=device)
        next_state_value = torch.randn(*N, T, *D, device=device)
        if random_gamma:
            gamma = torch.rand_like(reward) * gamma

        r1 = vec_td1_advantage_estimate(
            gamma,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
            time_dim=time_dim,
        )
        r2 = td1_advantage_estimate(
            gamma,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
            time_dim=time_dim,
        )
        if len(D) == 2:
            r3 = torch.cat(
                [
                    vec_td1_advantage_estimate(
                        gamma,
                        state_value[..., i : i + 1, j],
                        next_state_value[..., i : i + 1, j],
                        reward[..., i : i + 1, j],
                        done=done[..., i : i + 1, j],
                        terminated=terminated[..., i : i + 1, j],
                        rolling_gamma=rolling_gamma,
                        time_dim=-2,
                    )
                    for i in range(D[0])
                    for j in range(D[1])
                ],
                -1,
            ).unflatten(-1, D)
            r4 = torch.cat(
                [
                    td1_advantage_estimate(
                        gamma,
                        state_value[..., i : i + 1, j],
                        next_state_value[..., i : i + 1, j],
                        reward[..., i : i + 1, j],
                        done=done[..., i : i + 1, j],
                        terminated=terminated[..., i : i + 1, j],
                        rolling_gamma=rolling_gamma,
                        time_dim=-2,
                    )
                    for i in range(D[0])
                    for j in range(D[1])
                ],
                -1,
            ).unflatten(-1, D)
        else:
            r3 = torch.cat(
                [
                    vec_td1_advantage_estimate(
                        gamma,
                        state_value[..., i : i + 1],
                        next_state_value[..., i : i + 1],
                        reward[..., i : i + 1],
                        done=done[..., i : i + 1],
                        terminated=terminated[..., i : i + 1],
                        rolling_gamma=rolling_gamma,
                        time_dim=-2,
                    )
                    for i in range(D[0])
                ],
                -1,
            )
            r4 = torch.cat(
                [
                    td1_advantage_estimate(
                        gamma,
                        state_value[..., i : i + 1],
                        next_state_value[..., i : i + 1],
                        reward[..., i : i + 1],
                        done=done[..., i : i + 1],
                        terminated=terminated[..., i : i + 1],
                        rolling_gamma=rolling_gamma,
                        time_dim=-2,
                    )
                    for i in range(D[0])
                ],
                -1,
            )

        torch.testing.assert_close(r4, r2, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(r3, r1, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(r1, r2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("gamma", [0.99, 0.5, 0.1])
    @pytest.mark.parametrize("lmbda", [0.99, 0.5, 0.1])
    @pytest.mark.parametrize("N", [(1,), (3,), (7, 3)])
    @pytest.mark.parametrize("T", [200, 5, 3])
    @pytest.mark.parametrize("dtype", [torch.float, torch.double])
    @pytest.mark.parametrize("has_done", [False, True])
    def test_gae(self, device, gamma, lmbda, N, T, dtype, has_done):
        torch.manual_seed(0)

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        terminated = done.clone()
        if has_done:
            terminated = terminated.bernoulli_(0.1)
            done = done.bernoulli_(0.1) | terminated
        reward = torch.randn(*N, T, 1, device=device, dtype=dtype)
        state_value = torch.randn(*N, T, 1, device=device, dtype=dtype)
        next_state_value = torch.randn(*N, T, 1, device=device, dtype=dtype)

        r1 = vec_generalized_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )
        r2 = generalized_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )

        torch.testing.assert_close(r1, r2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("N", [(1,), (8,), (7, 3)])
    @pytest.mark.parametrize("dtype", [torch.float, torch.double])
    @pytest.mark.parametrize("has_done", [True, False])
    @pytest.mark.parametrize(
        "gamma_tensor", ["scalar", "tensor", "tensor_single_element"]
    )
    @pytest.mark.parametrize(
        "lmbda_tensor", ["scalar", "tensor", "tensor_single_element"]
    )
    def test_gae_param_as_tensor(
        self, device, N, dtype, has_done, gamma_tensor, lmbda_tensor
    ):
        torch.manual_seed(0)

        gamma = 0.95
        lmbda = 0.90
        T = 200

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        terminated = done.clone()
        if has_done:
            terminated = terminated.bernoulli_(0.1)
            done = done.bernoulli_(0.1) | terminated
        reward = torch.randn(*N, T, 1, device=device, dtype=dtype)
        state_value = torch.randn(*N, T, 1, device=device, dtype=dtype)
        next_state_value = torch.randn(*N, T, 1, device=device, dtype=dtype)

        if gamma_tensor == "tensor":
            gamma_vec = torch.full_like(reward, gamma)
        elif gamma_tensor == "tensor_single_element":
            gamma_vec = torch.as_tensor([gamma], device=device)
        else:
            gamma_vec = gamma

        if lmbda_tensor == "tensor":
            lmbda_vec = torch.full_like(reward, lmbda)
        elif gamma_tensor == "tensor_single_element":
            lmbda_vec = torch.as_tensor([lmbda], device=device)
        else:
            lmbda_vec = lmbda

        r1 = vec_generalized_advantage_estimate(
            gamma_vec,
            lmbda_vec,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )
        r2 = generalized_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )
        torch.testing.assert_close(r1, r2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("gamma", [0.99, 0.5, 0.1])
    @pytest.mark.parametrize("lmbda", [0.99, 0.5, 0.1])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [100, 3])
    @pytest.mark.parametrize("dtype", [torch.float, torch.double])
    @pytest.mark.parametrize("feature_dim", [[5], [2, 5]])
    @pytest.mark.parametrize("has_done", [True, False])
    def test_gae_multidim(
        self, device, gamma, lmbda, N, T, dtype, has_done, feature_dim
    ):
        D = feature_dim
        time_dim = -1 - len(D)

        torch.manual_seed(0)

        done = torch.zeros(*N, T, *D, device=device, dtype=torch.bool)
        terminated = done.clone()
        if has_done:
            terminated = terminated.bernoulli_(0.1)
            done = done.bernoulli_(0.1) | terminated
        reward = torch.randn(*N, T, *D, device=device, dtype=dtype)
        state_value = torch.randn(*N, T, *D, device=device, dtype=dtype)
        next_state_value = torch.randn(*N, T, *D, device=device, dtype=dtype)

        r1 = vec_generalized_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            time_dim=time_dim,
        )
        r2 = generalized_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            time_dim=time_dim,
        )
        if len(D) == 2:
            r3 = [
                vec_generalized_advantage_estimate(
                    gamma,
                    lmbda,
                    state_value[..., i : i + 1, j],
                    next_state_value[..., i : i + 1, j],
                    reward[..., i : i + 1, j],
                    done=done[..., i : i + 1, j],
                    terminated=terminated[..., i : i + 1, j],
                    time_dim=-2,
                )
                for i in range(D[0])
                for j in range(D[1])
            ]
            r4 = [
                generalized_advantage_estimate(
                    gamma,
                    lmbda,
                    state_value[..., i : i + 1, j],
                    next_state_value[..., i : i + 1, j],
                    reward[..., i : i + 1, j],
                    terminated=terminated[..., i : i + 1, j],
                    done=done[..., i : i + 1, j],
                    time_dim=-2,
                )
                for i in range(D[0])
                for j in range(D[1])
            ]
        else:
            r3 = [
                vec_generalized_advantage_estimate(
                    gamma,
                    lmbda,
                    state_value[..., i : i + 1],
                    next_state_value[..., i : i + 1],
                    reward[..., i : i + 1],
                    done=done[..., i : i + 1],
                    terminated=terminated[..., i : i + 1],
                    time_dim=-2,
                )
                for i in range(D[0])
            ]
            r4 = [
                generalized_advantage_estimate(
                    gamma,
                    lmbda,
                    state_value[..., i : i + 1],
                    next_state_value[..., i : i + 1],
                    reward[..., i : i + 1],
                    done=done[..., i : i + 1],
                    terminated=terminated[..., i : i + 1],
                    time_dim=-2,
                )
                for i in range(D[0])
            ]

        list3 = list(zip(*r3))
        list4 = list(zip(*r4))
        r3 = [torch.cat(list3[0], -1), torch.cat(list3[1], -1)]
        r4 = [torch.cat(list4[0], -1), torch.cat(list4[1], -1)]
        if len(D) == 2:
            r3 = [r3[0].unflatten(-1, D), r3[1].unflatten(-1, D)]
            r4 = [r4[0].unflatten(-1, D), r4[1].unflatten(-1, D)]
        torch.testing.assert_close(r2, r4, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(r1, r3, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(r1, r2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("gamma", [0.99, 0.5, 0.1])
    @pytest.mark.parametrize("N", [(1,), (3,), (7, 3)])
    @pytest.mark.parametrize("T", [200, 5, 3])
    @pytest.mark.parametrize("dtype", [torch.float, torch.double])
    @pytest.mark.parametrize("has_done", [False, True])
    def test_vtrace(self, device, gamma, N, T, dtype, has_done):
        torch.manual_seed(0)

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        terminated = done.clone()
        if has_done:
            terminated = terminated.bernoulli_(0.1)
            done = done.bernoulli_(0.1) | terminated
        reward = torch.randn(*N, T, 1, device=device, dtype=dtype)
        state_value = torch.randn(*N, T, 1, device=device, dtype=dtype)
        next_state_value = torch.randn(*N, T, 1, device=device, dtype=dtype)
        log_pi = torch.log(torch.rand(*N, T, 1, device=device, dtype=dtype))
        log_mu = torch.log(torch.rand(*N, T, 1, device=device, dtype=dtype))

        _, value_target = vtrace_advantage_estimate(
            gamma,
            log_pi,
            log_mu,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )

        assert not torch.isnan(value_target).any()
        assert not torch.isinf(value_target).any()

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("gamma", [0.99, 0.5, 0.1])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [100, 3])
    @pytest.mark.parametrize("dtype", [torch.float, torch.double])
    @pytest.mark.parametrize("feature_dim", [[5], [2, 5]])
    @pytest.mark.parametrize("has_done", [True, False])
    def test_vtrace_multidim(self, device, gamma, N, T, dtype, has_done, feature_dim):
        D = feature_dim
        time_dim = -1 - len(D)

        torch.manual_seed(0)

        done = torch.zeros(*N, T, *D, device=device, dtype=torch.bool)
        terminated = done.clone()
        if has_done:
            terminated = terminated.bernoulli_(0.1)
            done = done.bernoulli_(0.1) | terminated
        reward = torch.randn(*N, T, *D, device=device, dtype=dtype)
        state_value = torch.randn(*N, T, *D, device=device, dtype=dtype)
        next_state_value = torch.randn(*N, T, *D, device=device, dtype=dtype)
        log_pi = torch.log(torch.rand(*N, T, *D, device=device, dtype=dtype))
        log_mu = torch.log(torch.rand(*N, T, *D, device=device, dtype=dtype))

        r1 = vtrace_advantage_estimate(
            gamma,
            log_pi,
            log_mu,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            time_dim=time_dim,
        )
        if len(D) == 2:
            r2 = [
                vtrace_advantage_estimate(
                    gamma,
                    log_pi[..., i : i + 1, j],
                    log_mu[..., i : i + 1, j],
                    state_value[..., i : i + 1, j],
                    next_state_value[..., i : i + 1, j],
                    reward[..., i : i + 1, j],
                    terminated=terminated[..., i : i + 1, j],
                    done=done[..., i : i + 1, j],
                    time_dim=-2,
                )
                for i in range(D[0])
                for j in range(D[1])
            ]
        else:
            r2 = [
                vtrace_advantage_estimate(
                    gamma,
                    log_pi[..., i : i + 1],
                    log_mu[..., i : i + 1],
                    state_value[..., i : i + 1],
                    next_state_value[..., i : i + 1],
                    reward[..., i : i + 1],
                    done=done[..., i : i + 1],
                    terminated=terminated[..., i : i + 1],
                    time_dim=-2,
                )
                for i in range(D[0])
            ]

        list2 = list(zip(*r2))
        r2 = [torch.cat(list2[0], -1), torch.cat(list2[1], -1)]
        if len(D) == 2:
            r2 = [r2[0].unflatten(-1, D), r2[1].unflatten(-1, D)]
        torch.testing.assert_close(r1, r2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("gamma", [0.5, 0.99, 0.1])
    @pytest.mark.parametrize("lmbda", [0.1, 0.5, 0.99])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [200, 5, 3])
    @pytest.mark.parametrize("has_done", [True, False])
    def test_tdlambda_tensor_gamma(self, device, gamma, lmbda, N, T, has_done):
        """Tests vec_td_lambda_advantage_estimate against itself with
        gamma being a tensor or a scalar

        """
        torch.manual_seed(0)

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        terminated = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        if has_done:
            terminated = terminated.bernoulli_(0.1)
            done = done.bernoulli_(0.1) | terminated
        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        next_state_value = torch.randn(*N, T, 1, device=device)

        gamma_tensor = torch.full((*N, T, 1), gamma, device=device)
        v1 = vec_td_lambda_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )
        v2 = vec_td_lambda_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )

        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

        # # same with last done being true
        done[..., -1, :] = True  # terminating trajectory
        terminated[..., -1, :] = True  # terminating trajectory
        gamma_tensor[..., -1, :] = 0.0

        v1 = vec_td_lambda_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )
        v2 = vec_td_lambda_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )

        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("gamma", [0.5, 0.99])
    @pytest.mark.parametrize("lmbda", [0.25, 0.99])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [3, 100])
    @pytest.mark.parametrize("F", [1, 4])
    @pytest.mark.parametrize("has_done", [True, False])
    @pytest.mark.parametrize(
        "gamma_tensor", ["scalar", "tensor", "tensor_single_element"]
    )
    @pytest.mark.parametrize("lmbda_tensor", ["scalar", "tensor_single_element"])
    def test_tdlambda_tensor_gamma_single_element(
        self, device, gamma, lmbda, N, T, F, has_done, gamma_tensor, lmbda_tensor
    ):
        """Tests vec_td_lambda_advantage_estimate against itself with
        gamma being a tensor or a scalar

        """
        torch.manual_seed(0)

        done = torch.zeros(*N, T, F, device=device, dtype=torch.bool)
        terminated = torch.zeros(*N, T, F, device=device, dtype=torch.bool)
        if has_done:
            terminated = terminated.bernoulli_(0.1)
            done = done.bernoulli_(0.1) | terminated
        reward = torch.randn(*N, T, F, device=device)
        state_value = torch.randn(*N, T, F, device=device)
        next_state_value = torch.randn(*N, T, F, device=device)

        if gamma_tensor == "tensor":
            gamma_vec = torch.full_like(reward, gamma)
        elif gamma_tensor == "tensor_single_element":
            gamma_vec = torch.as_tensor([gamma], device=device)
        else:
            gamma_vec = gamma

        if gamma_tensor == "tensor_single_element":
            lmbda_vec = torch.as_tensor([lmbda], device=device)
        else:
            lmbda_vec = lmbda

        v1 = vec_td_lambda_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )
        v2 = vec_td_lambda_advantage_estimate(
            gamma_vec,
            lmbda_vec,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )

        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

        # # same with last done being true
        done[..., -1, :] = True  # terminating trajectory
        terminated[..., -1, :] = True  # terminating trajectory

        v1 = vec_td_lambda_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )
        v2 = vec_td_lambda_advantage_estimate(
            gamma_vec,
            lmbda_vec,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )

        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("gamma", [0.5, 0.99, 0.1])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [3, 5, 200])
    @pytest.mark.parametrize("has_done", [True, False])
    def test_td1_tensor_gamma(self, device, gamma, N, T, has_done):
        """Tests vec_td_lambda_advantage_estimate against itself with
        gamma being a tensor or a scalar

        """
        torch.manual_seed(0)

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        terminated = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        if has_done:
            terminated = terminated.bernoulli_(0.1)
            done = done.bernoulli_(0.1) | terminated
        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        next_state_value = torch.randn(*N, T, 1, device=device)

        gamma_tensor = torch.full((*N, T, 1), gamma, device=device)

        v1 = vec_td1_advantage_estimate(
            gamma,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )
        v2 = vec_td1_advantage_estimate(
            gamma_tensor,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )

        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

        # # same with last done being true
        done[..., -1, :] = True  # terminating trajectory
        terminated[..., -1, :] = True  # terminating trajectory
        gamma_tensor[..., -1, :] = 0.0

        v1 = vec_td1_advantage_estimate(
            gamma,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )
        v2 = vec_td1_advantage_estimate(
            gamma_tensor,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )

        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("gamma", [0.5, 0.99, 0.1])
    @pytest.mark.parametrize("lmbda", [0.1, 0.5, 0.99])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [3, 5, 50])
    @pytest.mark.parametrize("has_done", [True, False])
    def test_vectdlambda_tensor_gamma(
        self, device, gamma, lmbda, N, T, dtype_fixture, has_done  # noqa
    ):
        """Tests td_lambda_advantage_estimate against vec_td_lambda_advantage_estimate
        with gamma being a tensor or a scalar

        """

        torch.manual_seed(0)

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        terminated = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        if has_done:
            terminated = terminated.bernoulli_(0.1)
            done = done.bernoulli_(0.1) | terminated
        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        next_state_value = torch.randn(*N, T, 1, device=device)

        gamma_tensor = torch.full((*N, T, 1), gamma, device=device)

        v1 = td_lambda_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )
        v2 = vec_td_lambda_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )

        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

        # same with last done being true
        done[..., -1, :] = True  # terminating trajectory
        terminated[..., -1, :] = True  # terminating trajectory
        gamma_tensor[..., -1, :] = 0.0

        v1 = td_lambda_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )
        v2 = vec_td_lambda_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )

        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("gamma", [0.5, 0.99, 0.1])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [3, 5, 50])
    @pytest.mark.parametrize("has_done", [True, False])
    def test_vectd1_tensor_gamma(
        self, device, gamma, N, T, dtype_fixture, has_done  # noqa
    ):
        """Tests td_lambda_advantage_estimate against vec_td_lambda_advantage_estimate
        with gamma being a tensor or a scalar

        """

        torch.manual_seed(0)

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        terminated = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        if has_done:
            terminated = terminated.bernoulli_(0.1)
            done = done.bernoulli_(0.1) | terminated
        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        next_state_value = torch.randn(*N, T, 1, device=device)

        gamma_tensor = torch.full((*N, T, 1), gamma, device=device)

        v1 = td1_advantage_estimate(
            gamma,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )
        v2 = vec_td1_advantage_estimate(
            gamma_tensor,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )

        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

        # same with last done being true
        done[..., -1, :] = True  # terminating trajectory
        terminated[..., -1, :] = True  # terminating trajectory
        gamma_tensor[..., -1, :] = 0.0

        v1 = td1_advantage_estimate(
            gamma,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )
        v2 = vec_td1_advantage_estimate(
            gamma_tensor,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )

        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("lmbda", [0.1, 0.5, 0.99])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [50, 3])
    @pytest.mark.parametrize("rolling_gamma", [True, False, None])
    @pytest.mark.parametrize("has_done", [True, False])
    @pytest.mark.parametrize("seed", range(1))
    def test_vectdlambda_rand_gamma(
        self, device, lmbda, N, T, rolling_gamma, dtype_fixture, has_done, seed  # noqa
    ):
        """Tests td_lambda_advantage_estimate against vec_td_lambda_advantage_estimate
        with gamma being a random tensor

        """
        torch.manual_seed(seed)

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        terminated = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        if has_done:
            terminated = terminated.bernoulli_(0.1)
            done = done.bernoulli_(0.1) | terminated
        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        next_state_value = torch.randn(*N, T, 1, device=device)

        # avoid low values of gamma
        gamma_tensor = 0.5 + torch.rand_like(next_state_value) / 2

        v1 = td_lambda_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
        )
        if rolling_gamma is False and not done[..., 1:, :][done[..., :-1, :]].all():
            # if a not-done follows a done, then rolling_gamma=False cannot be used
            with pytest.raises(
                NotImplementedError, match="When using rolling_gamma=False"
            ):
                vec_td_lambda_advantage_estimate(
                    gamma_tensor,
                    lmbda,
                    state_value,
                    next_state_value,
                    reward,
                    done=done,
                    terminated=terminated,
                    rolling_gamma=rolling_gamma,
                )
            return
        elif rolling_gamma is False:
            with pytest.raises(
                NotImplementedError, match=r"The vectorized version of TD"
            ):
                vec_td_lambda_advantage_estimate(
                    gamma_tensor,
                    lmbda,
                    state_value,
                    next_state_value,
                    reward,
                    done=done,
                    terminated=terminated,
                    rolling_gamma=rolling_gamma,
                )
            return
        v2 = vec_td_lambda_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
        )
        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [50, 3])
    @pytest.mark.parametrize("rolling_gamma", [True, False, None])
    @pytest.mark.parametrize("has_done", [True, False])
    @pytest.mark.parametrize("seed", range(1))
    def test_vectd1_rand_gamma(
        self, device, N, T, rolling_gamma, dtype_fixture, has_done, seed  # noqa
    ):
        """Tests td_lambda_advantage_estimate against vec_td_lambda_advantage_estimate
        with gamma being a random tensor

        """
        torch.manual_seed(seed)

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        terminated = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        if has_done:
            terminated = terminated.bernoulli_(0.1)
            done = done.bernoulli_(0.1) | terminated
        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        next_state_value = torch.randn(*N, T, 1, device=device)

        # avoid low values of gamma
        gamma_tensor = 0.5 + torch.rand_like(next_state_value) / 2

        v1 = td1_advantage_estimate(
            gamma_tensor,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
        )
        if (
            rolling_gamma is False
            and not terminated[..., 1:, :][terminated[..., :-1, :]].all()
        ):
            # if a not-done follows a done, then rolling_gamma=False cannot be used
            with pytest.raises(
                NotImplementedError, match="When using rolling_gamma=False"
            ):
                vec_td1_advantage_estimate(
                    gamma_tensor,
                    state_value,
                    next_state_value,
                    reward,
                    done=done,
                    terminated=terminated,
                    rolling_gamma=rolling_gamma,
                )
            return
        elif rolling_gamma is False:
            with pytest.raises(
                NotImplementedError, match="The vectorized version of TD"
            ):
                vec_td1_advantage_estimate(
                    gamma_tensor,
                    state_value,
                    next_state_value,
                    reward,
                    done=done,
                    terminated=terminated,
                    rolling_gamma=rolling_gamma,
                )
            return
        v2 = vec_td1_advantage_estimate(
            gamma_tensor,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
        )
        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("gamma", [0.99, "rand"])
    @pytest.mark.parametrize("N", [(3,), (3, 7)])
    @pytest.mark.parametrize("T", [3, 5, 200])
    @pytest.mark.parametrize("rolling_gamma", [True, False])
    def test_custom_conv1d_tensor(self, device, gamma, N, T, rolling_gamma):
        """
        Tests the _custom_conv1d logic against a manual for-loop implementation
        """
        torch.manual_seed(0)

        if gamma == "rand":
            gamma = torch.rand(*N, T, 1, device=device)
            rand_gamma = True
        else:
            gamma = torch.full((*N, T, 1), gamma, device=device)
            rand_gamma = False

        values = torch.randn(*N, 1, T, device=device)
        out = torch.zeros(*N, 1, T, device=device)
        if rand_gamma and not rolling_gamma:
            for i in range(T):
                for j in reversed(range(i, T)):
                    out[..., i] = out[..., i] * gamma[..., i, :] + values[..., j]
        else:
            prev_val = 0.0
            for i in reversed(range(T)):
                prev_val = out[..., i] = prev_val * gamma[..., i, :] + values[..., i]

        gammas = _make_gammas_tensor(gamma, T, rolling_gamma)
        gammas = gammas.cumprod(-2)
        out_custom = _custom_conv1d(values.view(-1, 1, T), gammas).reshape(values.shape)

        torch.testing.assert_close(out, out_custom, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("N", [(3,), (3, 7)])
    @pytest.mark.parametrize("T", [3, 5, 200])
    @pytest.mark.parametrize("rolling_gamma", [True, False])
    def test_successive_traj_tdlambda(self, device, N, T, rolling_gamma):
        """Tests td_lambda_advantage_estimate against vec_td_lambda_advantage_estimate
        with gamma being a random tensor

        """
        torch.manual_seed(0)

        lmbda = torch.rand([]).item()

        terminated = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        terminated[..., T // 2 - 1, :] = 1
        done = terminated.clone()
        done[..., -1, :] = 1

        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        next_state_value = torch.randn(*N, T, 1, device=device)

        # avoid low values of gamma
        gamma_tensor = 0.5 + torch.rand_like(next_state_value) / 2

        v1 = td_lambda_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
        )
        v1a = td_lambda_advantage_estimate(
            gamma_tensor[..., : T // 2, :],
            lmbda,
            state_value[..., : T // 2, :],
            next_state_value[..., : T // 2, :],
            reward[..., : T // 2, :],
            done=done[..., : T // 2, :],
            terminated=terminated[..., : T // 2, :],
            rolling_gamma=rolling_gamma,
        )
        v1b = td_lambda_advantage_estimate(
            gamma_tensor[..., T // 2 :, :],
            lmbda,
            state_value[..., T // 2 :, :],
            next_state_value[..., T // 2 :, :],
            reward[..., T // 2 :, :],
            done=done[..., T // 2 :, :],
            terminated=terminated[..., T // 2 :, :],
            rolling_gamma=rolling_gamma,
        )
        torch.testing.assert_close(v1, torch.cat([v1a, v1b], -2), rtol=1e-4, atol=1e-4)

        if not rolling_gamma:
            with pytest.raises(
                NotImplementedError, match="When using rolling_gamma=False"
            ):
                vec_td_lambda_advantage_estimate(
                    gamma_tensor,
                    lmbda,
                    state_value,
                    next_state_value,
                    reward,
                    done=done,
                    terminated=terminated,
                    rolling_gamma=rolling_gamma,
                )
            return
        v2 = vec_td_lambda_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
        )
        v2a = vec_td_lambda_advantage_estimate(
            gamma_tensor[..., : T // 2, :],
            lmbda,
            state_value[..., : T // 2, :],
            next_state_value[..., : T // 2, :],
            reward[..., : T // 2, :],
            done=done[..., : T // 2, :],
            terminated=terminated[..., : T // 2, :],
            rolling_gamma=rolling_gamma,
        )
        v2b = vec_td_lambda_advantage_estimate(
            gamma_tensor[..., T // 2 :, :],
            lmbda,
            state_value[..., T // 2 :, :],
            next_state_value[..., T // 2 :, :],
            reward[..., T // 2 :, :],
            done=done[..., T // 2 :, :],
            terminated=terminated[..., T // 2 :, :],
            rolling_gamma=rolling_gamma,
        )

        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(v1a, v2a, rtol=1e-4, atol=1e-4)

        torch.testing.assert_close(v1b, v2b, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(v2, torch.cat([v2a, v2b], -2), rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("N", [(3,), (3, 7)])
    @pytest.mark.parametrize("T", [3, 5, 200])
    def test_successive_traj_tdadv(self, device, N, T):
        """Tests td_lambda_advantage_estimate against vec_td_lambda_advantage_estimate
        with gamma being a random tensor

        """
        torch.manual_seed(0)

        # for td0, a done that is not terminated has no effect
        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        done[..., T // 2 - 1, :] = 1
        terminated = done.clone()

        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        next_state_value = torch.randn(*N, T, 1, device=device)

        # avoid low values of gamma
        gamma_tensor = 0.5 + torch.rand_like(next_state_value) / 2

        v1 = td0_advantage_estimate(
            gamma_tensor,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )
        v1a = td0_advantage_estimate(
            gamma_tensor[..., : T // 2, :],
            state_value[..., : T // 2, :],
            next_state_value[..., : T // 2, :],
            reward[..., : T // 2, :],
            done=done[..., : T // 2, :],
            terminated=terminated[..., : T // 2, :],
        )
        v1b = td0_advantage_estimate(
            gamma_tensor[..., T // 2 :, :],
            state_value[..., T // 2 :, :],
            next_state_value[..., T // 2 :, :],
            reward[..., T // 2 :, :],
            done=done[..., T // 2 :, :],
            terminated=terminated[..., T // 2 :, :],
        )
        torch.testing.assert_close(v1, torch.cat([v1a, v1b], -2), rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_default_devices())
    def test_args_kwargs_timedim(self, device):
        torch.manual_seed(0)

        lmbda = 0.95
        N = (2, 3)
        B = (4,)
        T = 20

        terminated = torch.zeros(*N, T, *B, 1, device=device, dtype=torch.bool)
        terminated[..., T // 2 - 1, :, :] = 1
        done = terminated.clone()
        done[..., -1, :, :] = 1

        reward = torch.randn(*N, T, *B, 1, device=device)
        state_value = torch.randn(*N, T, *B, 1, device=device)
        next_state_value = torch.randn(*N, T, *B, 1, device=device)

        # avoid low values of gamma
        gamma = 0.95

        v1 = vec_generalized_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            time_dim=-3,
        )[0]

        v2 = vec_generalized_advantage_estimate(
            gamma=gamma,
            lmbda=lmbda,
            state_value=state_value,
            next_state_value=next_state_value,
            reward=reward,
            done=done,
            terminated=terminated,
            time_dim=-3,
        )[0]

        with pytest.raises(TypeError, match="positional arguments"):
            v3 = vec_generalized_advantage_estimate(
                gamma,
                lmbda,
                state_value,
                next_state_value,
                reward,
                done,
                terminated,
                -3,
            )[0]

        v3 = vec_generalized_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done,
            terminated,
            time_dim=-3,
        )[0]

        v4 = vec_generalized_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done,
            terminated,
            time_dim=2,
        )[0]

        v5 = vec_generalized_advantage_estimate(
            gamma=gamma,
            lmbda=lmbda,
            state_value=state_value,
            next_state_value=next_state_value,
            reward=reward,
            done=done,
            terminated=terminated,
            time_dim=-3,
        )[0]
        torch.testing.assert_close(v1, v2)
        torch.testing.assert_close(v1, v3)
        torch.testing.assert_close(v1, v4)
        torch.testing.assert_close(v1, v5)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("N", [(3,), (3, 7)])
    @pytest.mark.parametrize("T", [3, 5, 200])
    def test_successive_traj_gae(
        self,
        device,
        N,
        T,
    ):
        """Tests td_lambda_advantage_estimate against vec_td_lambda_advantage_estimate
        with gamma being a random tensor

        """
        torch.manual_seed(0)

        lmbda = torch.rand([]).item()

        terminated = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        terminated[..., T // 2 - 1, :] = 1
        done = terminated.clone()
        done[..., -1, :] = 1

        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        next_state_value = torch.randn(*N, T, 1, device=device)

        # avoid low values of gamma
        gamma_tensor = 0.95

        v1 = generalized_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )[0]
        v1a = generalized_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value[..., : T // 2, :],
            next_state_value[..., : T // 2, :],
            reward[..., : T // 2, :],
            done=done[..., : T // 2, :],
            terminated=terminated[..., : T // 2, :],
        )[0]
        v1b = generalized_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value[..., T // 2 :, :],
            next_state_value[..., T // 2 :, :],
            reward[..., T // 2 :, :],
            done=done[..., T // 2 :, :],
            terminated=terminated[..., T // 2 :, :],
        )[0]
        torch.testing.assert_close(v1, torch.cat([v1a, v1b], -2), rtol=1e-4, atol=1e-4)

        v2 = vec_generalized_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
        )[0]
        v2a = vec_generalized_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value[..., : T // 2, :],
            next_state_value[..., : T // 2, :],
            reward[..., : T // 2, :],
            done=done[..., : T // 2, :],
            terminated=terminated[..., : T // 2, :],
        )[0]
        v2b = vec_generalized_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value[..., T // 2 :, :],
            next_state_value[..., T // 2 :, :],
            reward[..., T // 2 :, :],
            done=done[..., T // 2 :, :],
            terminated=terminated[..., T // 2 :, :],
        )[0]
        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(v2, torch.cat([v2a, v2b], -2), rtol=1e-4, atol=1e-4)


class TestAdv:
    @pytest.fixture(scope="class", autouse=True)
    def _composite_log_prob(self):
        setter = set_composite_lp_aggregate(False)
        setter.set()
        yield
        setter.unset()

    @pytest.mark.parametrize(
        "adv,kwargs",
        [
            [GAE, {"lmbda": 0.95}],
            [TD1Estimator, {}],
            [TDLambdaEstimator, {"lmbda": 0.95}],
            [VTrace, {}],
        ],
    )
    def test_dispatch(
        self,
        adv,
        kwargs,
    ):
        value_net = TensorDictModule(
            nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
        )
        if adv is VTrace:
            actor_net = TensorDictModule(
                nn.Linear(3, 4), in_keys=["obs"], out_keys=["logits"]
            )
            actor_net = ProbabilisticActor(
                module=actor_net,
                in_keys=["logits"],
                out_keys=["action"],
                distribution_class=OneHotCategorical,
                return_log_prob=True,
            )
            module = adv(
                gamma=0.98,
                actor_network=actor_net,
                value_network=value_net,
                differentiable=False,
                **kwargs,
            )
            kwargs = {
                "obs": torch.randn(1, 10, 3),
                "action_log_prob": torch.log(torch.rand(1, 10, 1)),
                "next_reward": torch.randn(1, 10, 1, requires_grad=True),
                "next_done": torch.zeros(1, 10, 1, dtype=torch.bool),
                "next_terminated": torch.zeros(1, 10, 1, dtype=torch.bool),
                "next_obs": torch.randn(1, 10, 3),
            }
        else:
            module = adv(
                gamma=0.98,
                value_network=value_net,
                differentiable=False,
                **kwargs,
            )
            kwargs = {
                "obs": torch.randn(1, 10, 3),
                "next_reward": torch.randn(1, 10, 1, requires_grad=True),
                "next_done": torch.zeros(1, 10, 1, dtype=torch.bool),
                "next_terminated": torch.zeros(1, 10, 1, dtype=torch.bool),
                "next_obs": torch.randn(1, 10, 3),
            }
        advantage, value_target = module(**kwargs)
        assert advantage.shape == torch.Size([1, 10, 1])
        assert value_target.shape == torch.Size([1, 10, 1])

    @pytest.mark.parametrize(
        "adv,kwargs",
        [
            [GAE, {"lmbda": 0.95}],
            [TD1Estimator, {}],
            [TDLambdaEstimator, {"lmbda": 0.95}],
            [VTrace, {}],
        ],
    )
    def test_diff_reward(
        self,
        adv,
        kwargs,
    ):
        value_net = TensorDictModule(
            nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
        )
        if adv is VTrace:
            actor_net = TensorDictModule(
                nn.Linear(3, 4), in_keys=["obs"], out_keys=["logits"]
            )
            actor_net = ProbabilisticActor(
                module=actor_net,
                in_keys=["logits"],
                out_keys=["action"],
                distribution_class=OneHotCategorical,
                return_log_prob=True,
            )
            module = adv(
                gamma=0.98,
                actor_network=actor_net,
                value_network=value_net,
                differentiable=True,
                **kwargs,
            )
            td = TensorDict(
                {
                    "obs": torch.randn(1, 10, 3),
                    "action_log_prob": torch.log(torch.rand(1, 10, 1)),
                    "next": {
                        "obs": torch.randn(1, 10, 3),
                        "reward": torch.randn(1, 10, 1, requires_grad=True),
                        "done": torch.zeros(1, 10, 1, dtype=torch.bool),
                        "terminated": torch.zeros(1, 10, 1, dtype=torch.bool),
                    },
                },
                [1, 10],
            )
        else:
            module = adv(
                gamma=0.98,
                value_network=value_net,
                differentiable=True,
                **kwargs,
            )
            td = TensorDict(
                {
                    "obs": torch.randn(1, 10, 3),
                    "next": {
                        "obs": torch.randn(1, 10, 3),
                        "reward": torch.randn(1, 10, 1, requires_grad=True),
                        "done": torch.zeros(1, 10, 1, dtype=torch.bool),
                    },
                },
                [1, 10],
            )
        td = module(td.clone(False))
        # check that the advantage can't backprop to the value params
        td["advantage"].sum().backward()
        for p in value_net.parameters():
            assert p.grad is None or (p.grad == 0).all()
        # check that rewards have a grad
        assert td["next", "reward"].grad.norm() > 0

    @pytest.mark.parametrize(
        "adv,kwargs",
        [
            [GAE, {"lmbda": 0.95}],
            [TD1Estimator, {}],
            [TDLambdaEstimator, {"lmbda": 0.95}],
            [VTrace, {}],
        ],
    )
    @pytest.mark.parametrize("shifted", [True, False])
    def test_non_differentiable(self, adv, shifted, kwargs):
        value_net = TensorDictModule(
            nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
        )

        if adv is VTrace:
            actor_net = TensorDictModule(
                nn.Linear(3, 4), in_keys=["obs"], out_keys=["logits"]
            )
            actor_net = ProbabilisticActor(
                module=actor_net,
                in_keys=["logits"],
                out_keys=["action"],
                distribution_class=OneHotCategorical,
                return_log_prob=True,
            )
            module = adv(
                gamma=0.98,
                actor_network=actor_net,
                value_network=value_net,
                differentiable=False,
                shifted=shifted,
                **kwargs,
            )
            td = TensorDict(
                {
                    "obs": torch.randn(1, 10, 3),
                    "action_log_prob": torch.log(torch.rand(1, 10, 1)),
                    "next": {
                        "obs": torch.randn(1, 10, 3),
                        "reward": torch.randn(1, 10, 1, requires_grad=True),
                        "done": torch.zeros(1, 10, 1, dtype=torch.bool),
                        "terminated": torch.zeros(1, 10, 1, dtype=torch.bool),
                    },
                },
                [1, 10],
                names=[None, "time"],
            )
        else:
            module = adv(
                gamma=0.98,
                value_network=value_net,
                differentiable=False,
                shifted=shifted,
                **kwargs,
            )
            td = TensorDict(
                {
                    "obs": torch.randn(1, 10, 3),
                    "next": {
                        "obs": torch.randn(1, 10, 3),
                        "reward": torch.randn(1, 10, 1, requires_grad=True),
                        "done": torch.zeros(1, 10, 1, dtype=torch.bool),
                    },
                },
                [1, 10],
                names=[None, "time"],
            )
        td = module(td.clone(False))
        assert td["advantage"].is_leaf

    @pytest.mark.parametrize(
        "adv,kwargs",
        [
            [GAE, {"lmbda": 0.95}],
            [TD1Estimator, {}],
            [TDLambdaEstimator, {"lmbda": 0.95}],
            [VTrace, {}],
        ],
    )
    def test_time_dim(self, adv, kwargs, shifted=True):
        value_net = TensorDictModule(
            nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
        )

        if adv is VTrace:
            actor_net = TensorDictModule(
                nn.Linear(3, 4), in_keys=["obs"], out_keys=["logits"]
            )
            actor_net = ProbabilisticActor(
                module=actor_net,
                in_keys=["logits"],
                out_keys=["action"],
                distribution_class=OneHotCategorical,
                return_log_prob=True,
            )
            module_make = functools.partial(
                adv,
                gamma=0.98,
                actor_network=actor_net,
                value_network=value_net,
                differentiable=False,
                shifted=shifted,
                **kwargs,
            )
            td = TensorDict(
                {
                    "obs": torch.randn(1, 10, 3),
                    "action_log_prob": torch.log(torch.rand(1, 10, 1)),
                    "next": {
                        "obs": torch.randn(1, 10, 3),
                        "reward": torch.randn(1, 10, 1, requires_grad=True),
                        "done": torch.zeros(1, 10, 1, dtype=torch.bool),
                        "terminated": torch.zeros(1, 10, 1, dtype=torch.bool),
                    },
                },
                [1, 10],
                names=[None, "time"],
            )
        else:
            module_make = functools.partial(
                adv,
                gamma=0.98,
                value_network=value_net,
                differentiable=False,
                shifted=shifted,
                **kwargs,
            )
            td = TensorDict(
                {
                    "obs": torch.randn(1, 10, 3),
                    "next": {
                        "obs": torch.randn(1, 10, 3),
                        "reward": torch.randn(1, 10, 1, requires_grad=True),
                        "done": torch.zeros(1, 10, 1, dtype=torch.bool),
                    },
                },
                [1, 10],
                names=[None, "time"],
            )

        module_none = module_make(time_dim=None)
        module_0 = module_make(time_dim=0)
        module_1 = module_make(time_dim=1)

        td_none = module_none(td.clone(False))
        td_1 = module_1(td.clone(False))
        td_0 = module_0(td.transpose(0, 1).clone(False))
        assert_allclose_td(td_none, td_1)
        assert_allclose_td(td_none, td_0.transpose(0, 1))

        if adv is not VTrace:
            vt = module_none.value_estimate(td.clone(False))
            vt_patch = module_0.value_estimate(td.clone(False), time_dim=1)
            vt_patch2 = module_0.value_estimate(td.clone(False), time_dim=-1)
            torch.testing.assert_close(vt, vt_patch)

    @pytest.mark.parametrize(
        "adv,kwargs",
        [
            [GAE, {"lmbda": 0.95}],
            [TD1Estimator, {}],
            [TDLambdaEstimator, {"lmbda": 0.95}],
            [VTrace, {}],
        ],
    )
    @pytest.mark.parametrize("has_value_net", [True, False])
    @pytest.mark.parametrize("skip_existing", [True, False, None])
    @pytest.mark.parametrize("shifted", [True, False])
    def test_skip_existing(
        self,
        adv,
        kwargs,
        has_value_net,
        skip_existing,
        shifted,
    ):
        if has_value_net:
            value_net = TensorDictModule(
                lambda x: torch.zeros(*x.shape[:-1], 1),
                in_keys=["obs"],
                out_keys=["state_value"],
            )
        else:
            value_net = None

        if adv is VTrace:
            actor_net = TensorDictModule(
                nn.Linear(3, 4), in_keys=["obs"], out_keys=["logits"]
            )
            actor_net = ProbabilisticActor(
                module=actor_net,
                in_keys=["logits"],
                out_keys=["action"],
                distribution_class=OneHotCategorical,
                return_log_prob=True,
            )
            module = adv(
                gamma=0.98,
                actor_network=actor_net,
                value_network=value_net,
                differentiable=True,
                shifted=shifted,
                skip_existing=skip_existing,
                **kwargs,
            )
            td = TensorDict(
                {
                    "obs": torch.randn(1, 10, 3),
                    "action_log_prob": torch.log(torch.rand(1, 10, 1)),
                    "state_value": torch.ones(1, 10, 1),
                    "next": {
                        "obs": torch.randn(1, 10, 3),
                        "state_value": torch.ones(1, 10, 1),
                        "reward": torch.randn(1, 10, 1, requires_grad=True),
                        "done": torch.zeros(1, 10, 1, dtype=torch.bool),
                        "terminated": torch.zeros(1, 10, 1, dtype=torch.bool),
                    },
                },
                [1, 10],
                names=[None, "time"],
            )
        else:
            module = adv(
                gamma=0.98,
                value_network=value_net,
                differentiable=True,
                shifted=shifted,
                skip_existing=skip_existing,
                **kwargs,
            )
            td = TensorDict(
                {
                    "obs": torch.randn(1, 10, 3),
                    "state_value": torch.ones(1, 10, 1),
                    "next": {
                        "obs": torch.randn(1, 10, 3),
                        "state_value": torch.ones(1, 10, 1),
                        "reward": torch.randn(1, 10, 1, requires_grad=True),
                        "done": torch.zeros(1, 10, 1, dtype=torch.bool),
                    },
                },
                [1, 10],
                names=[None, "time"],
            )
        td = module(td.clone(False))
        if has_value_net and not skip_existing:
            exp_val = 0
        elif has_value_net and skip_existing:
            exp_val = 1
        elif not has_value_net:
            exp_val = 1
        assert (td["state_value"] == exp_val).all()
        # assert (td["next", "state_value"] == exp_val).all()

    @pytest.mark.parametrize("value", ["state_value", "state_value_test"])
    @pytest.mark.parametrize(
        "adv,kwargs",
        [
            [GAE, {"lmbda": 0.95}],
            [TD1Estimator, {}],
            [TDLambdaEstimator, {"lmbda": 0.95}],
            [VTrace, {}],
        ],
    )
    def test_set_keys(self, value, adv, kwargs):
        value_net = TensorDictModule(nn.Linear(3, 1), in_keys=["obs"], out_keys=[value])
        if adv is VTrace:
            actor_net = TensorDictModule(
                nn.Linear(3, 4), in_keys=["obs"], out_keys=["logits"]
            )
            actor_net = ProbabilisticActor(
                module=actor_net,
                in_keys=["logits"],
                out_keys=["action"],
                distribution_class=OneHotCategorical,
                return_log_prob=True,
            )
            module = adv(
                gamma=0.98,
                actor_network=actor_net,
                value_network=value_net,
                **kwargs,
            )
        else:
            module = adv(
                gamma=0.98,
                value_network=value_net,
                **kwargs,
            )
        module.set_keys(value=value)
        assert module.tensor_keys.value == value

        with pytest.raises(KeyError) as excinfo:
            module.set_keys(unknown_key="unknown_value")
            assert "unknown_value not found" in str(excinfo.value)

    @pytest.mark.parametrize(
        "adv,kwargs",
        [
            [GAE, {"lmbda": 0.95}],
            [TD1Estimator, {}],
            [TDLambdaEstimator, {"lmbda": 0.95}],
            [VTrace, {}],
        ],
    )
    def test_set_deprecated_keys(self, adv, kwargs):
        value_net = TensorDictModule(
            nn.Linear(3, 1), in_keys=["obs"], out_keys=["test_value"]
        )

        with pytest.raises(RuntimeError, match="via constructor is deprecated"):
            if adv is VTrace:
                actor_net = TensorDictModule(
                    nn.Linear(3, 4), in_keys=["obs"], out_keys=["logits"]
                )
                actor_net = ProbabilisticActor(
                    module=actor_net,
                    in_keys=["logits"],
                    out_keys=["action"],
                    distribution_class=OneHotCategorical,
                    return_log_prob=True,
                )
                module = adv(
                    gamma=0.98,
                    actor_network=actor_net,
                    value_network=value_net,
                    value_key="test_value",
                    advantage_key="advantage_test",
                    value_target_key="value_target_test",
                    **kwargs,
                )
            else:
                module = adv(
                    gamma=0.98,
                    value_network=value_net,
                    value_key="test_value",
                    advantage_key="advantage_test",
                    value_target_key="value_target_test",
                    **kwargs,
                )
            assert module.tensor_keys.value == "test_value"
            assert module.tensor_keys.advantage == "advantage_test"
            assert module.tensor_keys.value_target == "value_target_test"
