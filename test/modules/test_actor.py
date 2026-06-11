# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse

import numpy as np
import pytest
import torch
from _modules_common import _has_transformers
from tensordict import NonTensorData, NonTensorStack, TensorDict
from tensordict.nn import CompositeDistribution, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

from torch import distributions as dist, nn

from torchrl.data import Bounded, Composite
from torchrl.envs import CatFrames, Compose, InitTracker, SerialEnv, TransformedEnv
from torchrl.modules import (
    DiffusionActor,
    MultiStepActorWrapper,
    ProbabilisticActor,
    SafeModule,
    TanhDelta,
    TanhModule,
    TanhNormal,
    ValueOperator,
)
from torchrl.modules.distributions.utils import safeatanh, safetanh
from torchrl.modules.models import NoisyLazyLinear, NoisyLinear
from torchrl.modules.tensordict_module.actors import (
    ActorValueOperator,
    LMHeadActorValueOperator,
)
from torchrl.modules.vla import LeRobotPolicyWrapper, TinyVLA, VLAWrapperBase

from torchrl.testing import get_default_devices
from torchrl.testing.mocking_classes import CountingEnv, NestedCountingEnv


@pytest.mark.parametrize(
    "log_prob_key",
    [
        None,
        "sample_log_prob",
        ("nested", "sample_log_prob"),
        ("data", "sample_log_prob"),
    ],
)
def test_probabilistic_actor_nested_delta(log_prob_key, nested_dim=5, n_actions=1):
    env = NestedCountingEnv(nested_dim=nested_dim)
    action_spec = Bounded(shape=torch.Size((nested_dim, n_actions)), high=1, low=-1)
    policy_module = TensorDictModule(
        nn.Linear(1, 1), in_keys=[("data", "states")], out_keys=[("data", "param")]
    )
    policy = ProbabilisticActor(
        module=policy_module,
        spec=action_spec,
        in_keys=[("data", "param")],
        out_keys=[("data", "action")],
        distribution_class=TanhDelta,
        distribution_kwargs={
            "low": action_spec.space.low,
            "high": action_spec.space.high,
        },
        log_prob_key=log_prob_key,
        return_log_prob=True,
    )

    td = env.reset()
    td["data", "states"] = td["data", "states"].to(torch.float)
    td_out = policy(td)
    assert td_out["data", "action"].shape == (5, 1)
    if log_prob_key:
        assert td_out[log_prob_key].shape == (5,)
    else:
        assert td_out["data", "action_log_prob"].shape == (5,)

    policy = ProbabilisticActor(
        module=policy_module,
        spec=action_spec,
        in_keys={"param": ("data", "param")},
        out_keys=[("data", "action")],
        distribution_class=TanhDelta,
        distribution_kwargs={
            "low": action_spec.space.low,
            "high": action_spec.space.high,
        },
        log_prob_key=log_prob_key,
        return_log_prob=True,
    )
    td_out = policy(td)
    assert td_out["data", "action"].shape == (5, 1)
    if log_prob_key:
        assert td_out[log_prob_key].shape == (5,)
    else:
        assert td_out["data", "action_log_prob"].shape == (5,)


@pytest.mark.parametrize(
    "log_prob_key",
    [
        None,
        "sample_log_prob",
        ("nested", "sample_log_prob"),
        ("data", "sample_log_prob"),
    ],
)
def test_probabilistic_actor_nested_normal(log_prob_key, nested_dim=5, n_actions=3):
    env = NestedCountingEnv(nested_dim=nested_dim)
    action_spec = Bounded(shape=torch.Size((nested_dim, n_actions)), high=1, low=-1)
    actor_net = nn.Sequential(
        nn.Linear(1, 2),
        NormalParamExtractor(),
    )
    policy_module = TensorDictModule(
        actor_net,
        in_keys=[("data", "states")],
        out_keys=[("data", "loc"), ("data", "scale")],
    )
    policy = ProbabilisticActor(
        module=policy_module,
        spec=action_spec,
        in_keys=[("data", "loc"), ("data", "scale")],
        out_keys=[("data", "action")],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": action_spec.space.low,
            "high": action_spec.space.high,
        },
        log_prob_key=log_prob_key,
        return_log_prob=True,
    )

    td = env.reset()
    td["data", "states"] = td["data", "states"].to(torch.float)
    td_out = policy(td)
    assert td_out["data", "action"].shape == (5, 1)
    if log_prob_key:
        assert td_out[log_prob_key].shape == (5,)
    else:
        assert td_out["data", "action_log_prob"].shape == (5,)

    policy = ProbabilisticActor(
        module=policy_module,
        spec=action_spec,
        in_keys={"loc": ("data", "loc"), "scale": ("data", "scale")},
        out_keys=[("data", "action")],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": action_spec.space.low,
            "high": action_spec.space.high,
        },
        log_prob_key=log_prob_key,
        return_log_prob=True,
    )
    td_out = policy(td)
    assert td_out["data", "action"].shape == (5, 1)
    if log_prob_key:
        assert td_out[log_prob_key].shape == (5,)
    else:
        assert td_out["data", "action_log_prob"].shape == (5,)


class TestProbabilisticActorGenerator:
    """Tests for the ``generator`` kwarg on ``ProbabilisticActor``.

    The actual sampling logic lives in ``tensordict.nn`` and is exhaustively tested there;
    these tests just verify the kwarg threads through ``ProbabilisticActor`` →
    ``SafeProbabilisticModule`` → ``ProbabilisticTensorDictModule``.
    """

    @staticmethod
    def _make_actor(generator=None):
        module = TensorDictModule(
            lambda x: (x, torch.ones_like(x)),
            in_keys=["obs"],
            out_keys=["loc", "scale"],
        )
        return ProbabilisticActor(
            module=module,
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=dist.Normal,
            default_interaction_type="random",
            generator=generator,
        )

    def test_generator_object(self):
        """Two same-seeded Generators must produce identical actions."""
        a1 = self._make_actor(torch.Generator().manual_seed(0))
        a2 = self._make_actor(torch.Generator().manual_seed(0))
        # Set the global RNG to a different state to make sure it's not consulted.
        torch.manual_seed(999)
        s1 = a1(TensorDict(obs=torch.zeros(4)))["action"].clone()
        s2 = a2(TensorDict(obs=torch.zeros(4)))["action"].clone()
        assert torch.equal(s1, s2)

    def test_generator_int_seed(self):
        """Module-level int is shorthand for ``Generator().manual_seed(int)``."""
        a_int = self._make_actor(generator=0)
        a_gen = self._make_actor(generator=torch.Generator().manual_seed(0))
        s_int = a_int(TensorDict(obs=torch.zeros(4)))["action"].clone()
        s_gen = a_gen(TensorDict(obs=torch.zeros(4)))["action"].clone()
        assert torch.equal(s_int, s_gen)

    def test_generator_isolates_global_rng(self):
        """Sampling with a generator must not advance the global RNG."""
        a = self._make_actor(torch.Generator().manual_seed(0))
        torch.manual_seed(1234)
        before = torch.get_rng_state()
        a(TensorDict(obs=torch.zeros(4)))
        after = torch.get_rng_state()
        assert torch.equal(before, after)

    def test_generator_advances_in_place(self):
        a = self._make_actor(torch.Generator().manual_seed(0))
        s1 = a(TensorDict(obs=torch.zeros(4)))["action"].clone()
        s2 = a(TensorDict(obs=torch.zeros(4)))["action"].clone()
        assert not torch.equal(s1, s2)

    def test_generator_td_key_int_writeback(self):
        """Int seed in the input tensordict is treated as a stream-key (JAX-style)."""
        a = self._make_actor(generator="rng")

        def run(seed, n_steps):
            td = TensorDict(obs=torch.zeros(4))
            td["rng"] = NonTensorData(seed)
            samples = []
            for _ in range(n_steps):
                samples.append(a(td)["action"].clone())
            return samples

        traj_a = run(42, 3)
        traj_b = run(42, 3)
        for x, y in zip(traj_a, traj_b):
            assert torch.equal(x, y)
        assert not torch.equal(traj_a[0], traj_a[1])

    def test_generator_td_key_generator_form(self):
        """A Generator placed in the input tensordict is used in place."""
        a = self._make_actor(generator="rng")
        td = TensorDict(obs=torch.zeros(4))
        td["rng"] = NonTensorData(torch.Generator().manual_seed(0))
        s_key = a(td)["action"].clone()
        a_ref = self._make_actor(torch.Generator().manual_seed(0))
        s_ref = a_ref(TensorDict(obs=torch.zeros(4)))["action"].clone()
        assert torch.equal(s_key, s_ref)

    def test_generator_default_unchanged(self):
        """generator=None preserves existing global-RNG behaviour."""
        a = self._make_actor(generator=None)
        torch.manual_seed(0)
        s1 = a(TensorDict(obs=torch.zeros(4)))["action"].clone()
        torch.manual_seed(0)
        s2 = a(TensorDict(obs=torch.zeros(4)))["action"].clone()
        assert torch.equal(s1, s2)


@pytest.mark.parametrize(
    "layer_class",
    [
        NoisyLinear,
        NoisyLazyLinear,
    ],
)
@pytest.mark.parametrize("device", get_default_devices())
def test_noisy(layer_class, device, seed=0):
    torch.manual_seed(seed)
    layer = layer_class(3, 4, device=device, use_exploration_type=False)
    x = torch.randn(10, 3, device=device)
    y1 = layer(x)
    layer.reset_noise()
    y2 = layer(x)
    y3 = layer(x)
    torch.testing.assert_close(y2, y3)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(y1, y2)


class TestTanh:
    def test_errors(self):
        with pytest.raises(
            ValueError, match="in_keys and out_keys should have the same length"
        ):
            TanhModule(in_keys=["a", "b"], out_keys=["a"])
        with pytest.raises(ValueError, match=r"The minimum value \(-2\) provided"):
            spec = Bounded(-1, 1, shape=())
            TanhModule(in_keys=["act"], low=-2, spec=spec)
        with pytest.raises(ValueError, match=r"The maximum value \(-2\) provided to"):
            spec = Bounded(-1, 1, shape=())
            TanhModule(in_keys=["act"], high=-2, spec=spec)
        with pytest.raises(ValueError, match="Got high < low"):
            TanhModule(in_keys=["act"], high=-2, low=-1)

    def test_minmax(self):
        mod = TanhModule(
            in_keys=["act"],
            high=2,
        )
        assert isinstance(mod.act_high, torch.Tensor)
        mod = TanhModule(
            in_keys=["act"],
            low=-2,
        )
        assert isinstance(mod.act_low, torch.Tensor)
        mod = TanhModule(
            in_keys=["act"],
            high=np.ones((1,)),
        )
        assert isinstance(mod.act_high, torch.Tensor)
        mod = TanhModule(
            in_keys=["act"],
            low=-np.ones((1,)),
        )
        assert isinstance(mod.act_low, torch.Tensor)

    @pytest.mark.parametrize("clamp", [True, False])
    def test_boundaries(self, clamp):
        torch.manual_seed(0)
        eps = torch.finfo(torch.float).resolution
        for _ in range(10):
            min, max = (5 * torch.randn(2)).sort()[0]
            mod = TanhModule(in_keys=["act"], low=min, high=max, clamp=clamp)
            assert mod.non_trivial
            td = TensorDict({"act": (2 * torch.rand(100) - 1) * 10}, [])
            mod(td)
            # we should have a good proportion of samples close to the boundaries
            assert torch.isclose(td["act"], max).any()
            assert torch.isclose(td["act"], min).any()
            if not clamp:
                assert (td["act"] <= max + eps).all()
                assert (td["act"] >= min - eps).all()
            else:
                assert (td["act"] < max + eps).all()
                assert (td["act"] > min - eps).all()

    @pytest.mark.parametrize("out_keys", [[("a", "c"), "b"], None])
    @pytest.mark.parametrize("has_spec", [[True, True], [True, False], [False, False]])
    def test_multi_inputs(self, out_keys, has_spec):
        in_keys = [("x", "z"), "y"]
        real_out_keys = out_keys if out_keys is not None else in_keys

        if any(has_spec):
            spec = {}
            if has_spec[0]:
                spec.update({real_out_keys[0]: Bounded(-2.0, 2.0, shape=())})
                low, high = -2.0, 2.0
            if has_spec[1]:
                spec.update({real_out_keys[1]: Bounded(-3.0, 3.0, shape=())})
                low, high = None, None
            spec = Composite(spec)
        else:
            spec = None
            low, high = -2.0, 2.0

        mod = TanhModule(
            in_keys=in_keys,
            out_keys=out_keys,
            low=low,
            high=high,
            spec=spec,
            clamp=False,
        )
        data = TensorDict({in_key: torch.randn(100) * 100 for in_key in in_keys}, [])
        mod(data)
        assert all(out_key in data.keys(True, True) for out_key in real_out_keys)
        eps = torch.finfo(torch.float).resolution

        for out_key in real_out_keys:
            key = out_key if isinstance(out_key, str) else "_".join(out_key)
            low_key = f"{key}_low"
            high_key = f"{key}_high"
            min, max = getattr(mod, low_key), getattr(mod, high_key)
            assert torch.isclose(data[out_key], max).any()
            assert torch.isclose(data[out_key], min).any()
            assert (data[out_key] <= max + eps).all()
            assert (data[out_key] >= min - eps).all()


@pytest.mark.skipif(torch.__version__ < "2.0", reason="torch 2.0 is required")
@pytest.mark.parametrize("use_vmap", [False, True])
@pytest.mark.parametrize("scale", range(10))
def test_tanh_atanh(use_vmap, scale):
    if use_vmap:
        try:
            from torch import vmap
        except ImportError:
            try:
                from functorch import vmap
            except ImportError:
                raise pytest.skip("functorch not found")

    torch.manual_seed(0)
    x = (torch.randn(10, dtype=torch.double) * scale).requires_grad_(True)
    if not use_vmap:
        y = safetanh(x, 1e-6)
    else:
        y = vmap(safetanh, (0, None))(x, 1e-6)

    if not use_vmap:
        xp = safeatanh(y, 1e-6)
    else:
        xp = vmap(safeatanh, (0, None))(y, 1e-6)

    xp.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


class TestDiffusionActor:
    def test_output_shape(self):
        actor = DiffusionActor(action_dim=2, obs_dim=3, num_steps=5)
        td = TensorDict({"observation": torch.randn(4, 3)}, batch_size=[4])
        td = actor(td)
        assert td["action"].shape == torch.Size([4, 2])

    def test_unbatched(self):
        actor = DiffusionActor(action_dim=4, obs_dim=6, num_steps=3)
        td = TensorDict({"observation": torch.randn(6)}, batch_size=[])
        td = actor(td)
        assert td["action"].shape == torch.Size([4])

    def test_custom_in_out_keys(self):
        actor = DiffusionActor(
            action_dim=2,
            obs_dim=3,
            num_steps=3,
            in_keys=["obs"],
            out_keys=["act"],
        )
        assert actor.in_keys == ["obs"]
        assert actor.out_keys == ["act"]
        td = TensorDict({"obs": torch.randn(4, 3)}, batch_size=[4])
        td = actor(td)
        assert td["act"].shape == torch.Size([4, 2])

    def test_custom_score_network(self):
        score_net = nn.Linear(2 + 3 + 1, 2)
        actor = DiffusionActor(
            action_dim=2, obs_dim=3, score_network=score_net, num_steps=3
        )
        td = TensorDict({"observation": torch.randn(4, 3)}, batch_size=[4])
        td = actor(td)
        assert td["action"].shape == torch.Size([4, 2])

    def test_spec_wrapping(self):
        spec = Bounded(low=-1.0, high=1.0, shape=(2,))
        actor = DiffusionActor(action_dim=2, obs_dim=3, num_steps=3, spec=spec)
        assert actor.spec is not None

    def test_gradients_flow(self):
        actor = DiffusionActor(action_dim=2, obs_dim=3, num_steps=3)
        obs = torch.randn(4, 3)
        td = TensorDict({"observation": obs}, batch_size=[4])
        td = actor(td)
        td["action"].sum().backward()
        for p in actor.parameters():
            assert p.grad is not None


@pytest.mark.parametrize("device", get_default_devices())
def test_actorcritic(device):
    common_module = SafeModule(
        module=nn.Linear(3, 4), in_keys=["obs"], out_keys=["hidden"], spec=None
    ).to(device)
    module = SafeModule(nn.Linear(4, 5), in_keys=["hidden"], out_keys=["param"])
    policy_operator = ProbabilisticActor(
        module=module, in_keys=["param"], spec=None, return_log_prob=True
    ).to(device)
    value_operator = ValueOperator(nn.Linear(4, 1), in_keys=["hidden"]).to(device)
    op = ActorValueOperator(
        common_operator=common_module,
        policy_operator=policy_operator,
        value_operator=value_operator,
    ).to(device)
    td = TensorDict(
        source={"obs": torch.randn(4, 3)},
        batch_size=[
            4,
        ],
    ).to(device)
    td_total = op(td.clone())
    policy_op = op.get_policy_operator()
    td_policy = policy_op(td.clone())
    value_op = op.get_value_operator()
    td_value = value_op(td)
    torch.testing.assert_close(td_total.get("action"), td_policy.get("action"))
    torch.testing.assert_close(
        td_total.get("sample_log_prob"), td_policy.get("sample_log_prob")
    )
    torch.testing.assert_close(td_total.get("state_value"), td_value.get("state_value"))

    value_params = set(
        list(op.get_value_operator().parameters()) + list(op.module[0].parameters())
    )
    value_params2 = set(value_op.parameters())
    assert len(value_params.difference(value_params2)) == 0 and len(
        value_params.intersection(value_params2)
    ) == len(value_params)

    policy_params = set(
        list(op.get_policy_operator().parameters()) + list(op.module[0].parameters())
    )
    policy_params2 = set(policy_op.parameters())
    assert len(policy_params.difference(policy_params2)) == 0 and len(
        policy_params.intersection(policy_params2)
    ) == len(policy_params)


@pytest.mark.parametrize("name_map", [True, False])
def test_compound_actor(name_map):
    class Module(nn.Module):
        def forward(self, x):
            return x[..., :3], x[..., 3:6], x[..., 6:]

    module = TensorDictModule(
        Module(),
        in_keys=["x"],
        out_keys=[
            ("params", "normal", "loc"),
            ("params", "normal", "scale"),
            ("params", "categ", "logits"),
        ],
    )
    distribution_kwargs = {
        "distribution_map": {"normal": dist.Normal, "categ": dist.Categorical}
    }
    if name_map:
        distribution_kwargs.update(
            {
                "name_map": {
                    "normal": ("action", "normal"),
                    "categ": ("action", "categ"),
                },
            }
        )
    actor = ProbabilisticActor(
        module,
        in_keys=["params"],
        distribution_class=CompositeDistribution,
        distribution_kwargs=distribution_kwargs,
    )
    if not name_map:
        assert actor.out_keys == module.out_keys + ["normal", "categ"]
    else:
        assert actor.out_keys == module.out_keys + [
            ("action", "normal"),
            ("action", "categ"),
        ]

    data = TensorDict({"x": torch.rand(10)}, [])
    actor(data)
    assert set(data.keys(True, True)) == {
        "categ" if not name_map else ("action", "categ"),
        "normal" if not name_map else ("action", "normal"),
        ("params", "categ", "logits"),
        ("params", "normal", "loc"),
        ("params", "normal", "scale"),
        "x",
    }


@pytest.mark.skipif(not _has_transformers, reason="missing dependencies")
@pytest.mark.parametrize("device", get_default_devices())
def test_lmhead_actorvalueoperator(device):
    from transformers import AutoModelForCausalLM, GPT2Config

    config = GPT2Config(return_dict=False)
    base_model = AutoModelForCausalLM.from_config(config).eval()
    aco = LMHeadActorValueOperator(base_model).to(device)

    # check common
    assert aco.module[0][0].module is base_model.transformer
    assert aco.module[0][1].in_keys == ["x"]
    assert aco.module[0][1].out_keys == ["x"]

    # check actor
    assert aco.module[1].in_keys == ["x"]
    assert aco.module[1].out_keys == ["logits", "action", "action_log_prob"]
    assert aco.module[1][0].module is base_model.lm_head

    # check critic
    assert aco.module[2].in_keys == ["x"]
    assert aco.module[2].out_keys == ["state_value"]
    assert isinstance(aco.module[2].module, nn.Linear)
    assert aco.module[2].module.in_features == base_model.transformer.embed_dim
    assert aco.module[2].module.out_features == 1

    td = TensorDict(
        source={
            "input_ids": torch.randint(50257, (4, 3)),
            "attention_mask": torch.ones((4, 3)),
        },
        batch_size=[
            4,
        ],
        device=device,
    )
    td_total = aco(td.clone())
    policy_op = aco.get_policy_operator()
    td_policy = policy_op(td.clone())
    value_op = aco.get_value_operator()
    td_value = value_op(td)
    torch.testing.assert_close(td_total.get("action"), td_policy.get("action"))
    torch.testing.assert_close(
        td_total.get("sample_log_prob"), td_policy.get("sample_log_prob")
    )
    torch.testing.assert_close(td_total.get("state_value"), td_value.get("state_value"))

    value_params = set(
        list(aco.get_value_operator().parameters()) + list(aco.module[0].parameters())
    )
    value_params2 = set(value_op.parameters())
    assert len(value_params.difference(value_params2)) == 0 and len(
        value_params.intersection(value_params2)
    ) == len(value_params)

    policy_params = set(
        list(aco.get_policy_operator().parameters()) + list(aco.module[0].parameters())
    )
    policy_params2 = set(policy_op.parameters())
    assert len(policy_params.difference(policy_params2)) == 0 and len(
        policy_params.intersection(policy_params2)
    ) == len(policy_params)


class TestBatchedActor:
    def test_batched_actor_exceptions(self):
        time_steps = 5
        actor_base = TensorDictModule(
            lambda x: torch.ones(
                x.shape[0], time_steps, 1, device=x.device, dtype=x.dtype
            ),
            in_keys=["observation_cat"],
            out_keys=["action"],
        )
        with pytest.raises(ValueError, match="Only a single init_key can be passed"):
            MultiStepActorWrapper(actor_base, n_steps=time_steps, init_key=["init_key"])

        batch = 2

        # The second env has frequent resets, the first none
        base_env = SerialEnv(
            batch,
            [lambda: CountingEnv(max_steps=5000), lambda: CountingEnv(max_steps=5)],
        )
        env = TransformedEnv(
            base_env,
            CatFrames(
                N=time_steps,
                in_keys=["observation"],
                out_keys=["observation_cat"],
                dim=-1,
            ),
        )
        actor = MultiStepActorWrapper(actor_base, n_steps=time_steps)
        with pytest.raises(KeyError, match="No init key was passed"):
            env.rollout(2, actor)

        env = TransformedEnv(
            base_env,
            Compose(
                InitTracker(),
                CatFrames(
                    N=time_steps,
                    in_keys=["observation"],
                    out_keys=["observation_cat"],
                    dim=-1,
                ),
            ),
        )
        td = env.rollout(10)[..., -1]["next"]
        actor = MultiStepActorWrapper(actor_base, n_steps=time_steps)
        with pytest.raises(RuntimeError, match="Cannot initialize the wrapper"):
            env.rollout(10, actor, tensordict=td, auto_reset=False)

        actor = MultiStepActorWrapper(actor_base, n_steps=time_steps - 1)
        with pytest.raises(RuntimeError, match="The action's time dimension"):
            env.rollout(10, actor)

    @pytest.mark.parametrize("time_steps", [3, 5])
    def test_batched_actor_simple(self, time_steps):

        batch = 2

        # The second env has frequent resets, the first none
        base_env = SerialEnv(
            batch,
            [lambda: CountingEnv(max_steps=5000), lambda: CountingEnv(max_steps=5)],
        )
        env = TransformedEnv(
            base_env,
            Compose(
                InitTracker(),
                CatFrames(
                    N=time_steps,
                    in_keys=["observation"],
                    out_keys=["observation_cat"],
                    dim=-1,
                ),
            ),
        )

        actor_base = TensorDictModule(
            lambda x: torch.ones(
                x.shape[0], time_steps, 1, device=x.device, dtype=x.dtype
            ),
            in_keys=["observation_cat"],
            out_keys=["action"],
        )
        actor = MultiStepActorWrapper(actor_base, n_steps=time_steps)
        # rollout = env.rollout(100, break_when_any_done=False)
        rollout = env.rollout(50, actor, break_when_any_done=False)
        unique = rollout[0]["observation"].unique()
        predicted = torch.arange(unique.numel())
        assert (unique == predicted).all()
        assert (
            rollout[1]["observation"]
            == (torch.arange(50) % 6).reshape_as(rollout[1]["observation"])
        ).all()


def _make_obs_td(batch=2, h=16, state_dim=5, with_state=True):
    obs = {"image": torch.zeros(batch, 3, h, h, dtype=torch.uint8)}
    if with_state:
        obs["state"] = torch.randn(batch, state_dim)
    data = {
        "observation": obs,
        "language_instruction": NonTensorStack(*[f"task {i}" for i in range(batch)]),
    }
    return TensorDict(data, batch_size=[batch])


class TestVLAWrapperBase:
    def test_token_head_requires_vocab(self):
        with pytest.raises(ValueError, match="vocab_size"):
            VLAWrapperBase(action_dim=2, chunk_size=2, action_head="tokens")

    def test_invalid_action_head(self):
        with pytest.raises(ValueError, match="action_head"):
            VLAWrapperBase(action_dim=2, chunk_size=2, action_head="diffusion")

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            VLAWrapperBase(action_dim=2, chunk_size=2, mode="beam")

    def test_hook_not_implemented(self):
        base = VLAWrapperBase(action_dim=2, chunk_size=2)
        with pytest.raises(NotImplementedError):
            base._predict(TensorDict({}, batch_size=[]))


class TestTinyVLA:
    def test_continuous(self):
        policy = TinyVLA(action_dim=7, chunk_size=4)
        out = policy(_make_obs_td())
        assert out["action_chunk"].shape == torch.Size([2, 4, 7])
        assert policy.out_keys == ["action_chunk"]
        assert ("observation", "image") in policy.in_keys

    def test_tokens(self):
        policy = TinyVLA(
            action_dim=7, chunk_size=4, action_head="tokens", vocab_size=64
        )
        out = policy(_make_obs_td())
        assert out["action_tokens"].shape == torch.Size([2, 4, 7])
        # one sequence-level log-prob per sample (summed over the chunk): the
        # contract PPO-style objectives expect from sample_log_prob
        assert out["log_probs"].shape == torch.Size([2])
        assert (out["action_tokens"] >= 0).all() and (out["action_tokens"] < 64).all()
        dist = policy.get_dist(_make_obs_td())
        assert dist.base_dist.logits.shape == torch.Size([2, 4, 7, 64])
        assert dist.log_prob(out["action_tokens"]).shape == torch.Size([2])

    def test_get_dist_continuous_raises(self):
        policy = TinyVLA(action_dim=3, chunk_size=2)
        with pytest.raises(RuntimeError, match="tokens"):
            policy.get_dist(_make_obs_td())

    def test_no_state(self):
        policy = TinyVLA(action_dim=3, chunk_size=2, use_state=False)
        assert ("observation", "state") not in policy.in_keys
        out = policy(_make_obs_td(with_state=False))
        assert out["action_chunk"].shape == torch.Size([2, 2, 3])

    def test_set_keys(self):
        policy = TinyVLA(action_dim=3, chunk_size=2)
        policy.set_keys(instruction="instr", action_chunk=("pred", "chunk"))
        assert "instr" in policy.in_keys
        assert policy.out_keys == [("pred", "chunk")]
        td = TensorDict(
            {
                "observation": {
                    "image": torch.zeros(2, 3, 16, 16, dtype=torch.uint8),
                    "state": torch.randn(2, 5),
                },
                "instr": NonTensorStack("a", "b"),
            },
            batch_size=[2],
        )
        assert policy(td)["pred", "chunk"].shape == torch.Size([2, 2, 3])

    def test_greedy_deterministic(self):
        policy = TinyVLA(
            action_dim=3, chunk_size=2, action_head="tokens", vocab_size=16
        )
        td = _make_obs_td()
        torch.testing.assert_close(
            policy(td.clone())["action_tokens"], policy(td.clone())["action_tokens"]
        )

    def test_gradient_flow(self):
        policy = TinyVLA(action_dim=3, chunk_size=2)
        out = policy(_make_obs_td())
        out["action_chunk"].sum().backward()
        grads = [p.grad for p in policy.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in grads)


class _DummyChunkPolicy:
    """A stand-in for a LeRobot action-chunk policy."""

    def __init__(self, chunk_size, action_dim, value=0.0):
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.value = value
        self.last_batch = None

    def predict_action_chunk(self, batch):
        self.last_batch = batch
        b = batch["observation.state"].shape[0]
        return torch.full((b, self.chunk_size, self.action_dim), self.value)


class TestLeRobotPolicyWrapper:
    def test_wraps_external_policy(self):
        policy = LeRobotPolicyWrapper(
            _DummyChunkPolicy(4, 7), action_dim=7, chunk_size=4
        )
        out = policy(_make_obs_td())
        assert out["action_chunk"].shape == torch.Size([2, 4, 7])
        assert policy.out_keys == ["action_chunk"]

    def test_builds_lerobot_batch(self):
        dummy = _DummyChunkPolicy(2, 3)
        policy = LeRobotPolicyWrapper(
            dummy, action_dim=3, chunk_size=2, camera_name="top"
        )
        policy(_make_obs_td())
        batch = dummy.last_batch
        assert "observation.images.top" in batch
        assert "observation.state" in batch
        assert batch["task"] == ["task 0", "task 1"]

    def test_predict_fn_override(self):
        called = {}

        def predict_fn(model, batch):
            called["yes"] = True
            b = batch["observation.state"].shape[0]
            return torch.zeros(b, 2, 3)

        policy = LeRobotPolicyWrapper(
            object(), action_dim=3, chunk_size=2, predict_fn=predict_fn
        )
        out = policy(_make_obs_td())
        assert called.get("yes") and out["action_chunk"].shape == torch.Size([2, 2, 3])

    def test_callable_policy(self):
        def model(batch):
            b = batch["observation.state"].shape[0]
            return torch.zeros(b, 2, 3)

        policy = LeRobotPolicyWrapper(model, action_dim=3, chunk_size=2)
        assert policy(_make_obs_td())["action_chunk"].shape == torch.Size([2, 2, 3])

    def test_bad_policy_raises(self):
        policy = LeRobotPolicyWrapper(object(), action_dim=3, chunk_size=2)
        with pytest.raises(TypeError, match="predict_action_chunk"):
            policy(_make_obs_td())

    def test_wrong_chunk_shape_raises(self):
        # a single-step [B, action_dim] output (e.g. LeRobot select_action) is
        # not a chunk and must fail loudly, even when B == chunk_size
        def model(batch):
            b = batch["observation.state"].shape[0]
            return torch.zeros(b, 3)

        policy = LeRobotPolicyWrapper(model, action_dim=3, chunk_size=2)
        with pytest.raises(ValueError, match="action chunk of shape"):
            policy(_make_obs_td())

    def test_from_pretrained_requires_lerobot(self):
        with pytest.raises(ImportError, match="lerobot"):
            LeRobotPolicyWrapper.from_pretrained(
                "fake/repo", action_dim=7, chunk_size=4
            )


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
