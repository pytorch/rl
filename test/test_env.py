# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os.path
from collections import defaultdict

import numpy as np
import pytest
import torch
import yaml
from _utils_internal import get_available_devices
from mocking_classes import (
    DiscreteActionVecMockEnv,
    MockSerialEnv,
    DiscreteActionConvMockEnv,
)
from scipy.stats import chisquare
from torch import nn
from torchrl.data.tensor_specs import (
    OneHotDiscreteTensorSpec,
    MultOneHotDiscreteTensorSpec,
    BoundedTensorSpec,
    NdBoundedTensorSpec,
)
from torchrl.data.tensordict.tensordict import assert_allclose_td, TensorDict
from torchrl.envs import EnvCreator, ObservationNorm
from torchrl.envs import GymEnv
from torchrl.envs.libs.gym import _has_gym
from torchrl.envs.transforms import (
    TransformedEnv,
    Compose,
    ToTensorImage,
    RewardClipping,
)
from torchrl.envs.utils import step_tensordict
from torchrl.envs.vec_env import ParallelEnv, SerialEnv
from torchrl.modules import (
    ActorCriticOperator,
    TensorDictModule,
    ValueOperator,
    Actor,
    MLP,
)

try:
    this_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(this_dir, "configs", "atari.yaml"), "r") as file:
        atari_confs = yaml.load(file, Loader=yaml.FullLoader)
    _atari_found = True
except FileNotFoundError:
    _atari_found = False
    atari_confs = defaultdict(lambda: "")


## TO BE FIXED: DiscreteActionProjection queries a randint on each worker, which leads to divergent results between
## the serial and parallel batched envs
# def _make_atari_env(atari_env):
#     action_spec = GymEnv(atari_env + "-ram-v0").action_spec
#     n_act = action_spec.shape[-1]
#     return lambda **kwargs: TransformedEnv(
#         GymEnv(atari_env + "-ram-v0", **kwargs),
#         DiscreteActionProjection(max_N=18, M=n_act),
#     )
#
#
# @pytest.mark.skipif(
#     "ALE/Pong-v5" not in _get_gym_envs(), reason="no Atari OpenAI Gym env available"
# )
# def test_composite_env():
#     num_workers = 10
#     frameskip = 2
#     create_env_fn = [
#         _make_atari_env(atari_env)
#         for atari_env in atari_confs["atari_envs"][:num_workers]
#     ]
#     kwargs = {"frame_skip": frameskip}
#
#     random_policy = lambda td: td.set(
#         "action", torch.nn.functional.one_hot(torch.randint(18, (*td.batch_size,)), 18)
#     )
#     p = SerialEnv(num_workers, create_env_fn, create_env_kwargs=kwargs)
#     seed = p.set_seed(0)
#     p.reset()
#     torch.manual_seed(seed)
#     rollout1 = p.rollout(max_steps=100, policy=random_policy, auto_reset=False)
#     p.close()
#     del p
#
#     p = ParallelEnv(num_workers, create_env_fn, create_env_kwargs=kwargs)
#     seed = p.set_seed(0)
#     p.reset()
#     torch.manual_seed(seed)
#     rollout0 = p.rollout(max_steps=100, policy=random_policy, auto_reset=False)
#     p.close()
#     del p
#
#     assert_allclose_td(rollout1, rollout0)


@pytest.mark.skipif(not _has_gym, reason="no gym")
@pytest.mark.parametrize("env_name", ["Pendulum-v1", "CartPole-v1"])
@pytest.mark.parametrize("frame_skip", [1, 4])
def test_env_seed(env_name, frame_skip, seed=0):
    env = GymEnv(env_name, frame_skip=frame_skip)
    action = env.action_spec.rand()

    env.set_seed(seed)
    td0a = env.reset()
    td1a = env.step(td0a.clone().set("action", action))

    env.set_seed(seed)
    td0b = env.specs.build_tensordict()
    td0b = env.reset(tensordict=td0b)
    td1b = env.step(td0b.clone().set("action", action))

    assert_allclose_td(td0a, td0b.select(*td0a.keys()))
    assert_allclose_td(td1a, td1b)

    env.set_seed(
        seed=seed + 10,
    )
    td0c = env.reset()
    td1c = env.step(td0c.clone().set("action", action))

    with pytest.raises(AssertionError):
        assert_allclose_td(td0a, td0c.select(*td0a.keys()))
    with pytest.raises(AssertionError):
        assert_allclose_td(td1a, td1c)
    env.close()


@pytest.mark.skipif(not _has_gym, reason="no gym")
@pytest.mark.parametrize("env_name", ["Pendulum-v1", "ALE/Pong-v5"])
@pytest.mark.parametrize("frame_skip", [1, 4])
def test_rollout(env_name, frame_skip, seed=0):
    env = GymEnv(env_name, frame_skip=frame_skip)

    torch.manual_seed(seed)
    np.random.seed(seed)
    env.set_seed(seed)
    env.reset()
    rollout1 = env.rollout(max_steps=100)

    torch.manual_seed(seed)
    np.random.seed(seed)
    env.set_seed(seed)
    env.reset()
    rollout2 = env.rollout(max_steps=100)

    assert_allclose_td(rollout1, rollout2)

    torch.manual_seed(seed)
    env.set_seed(seed + 10)
    env.reset()
    rollout3 = env.rollout(max_steps=100)
    with pytest.raises(AssertionError):
        assert_allclose_td(rollout1, rollout3)
    env.close()


@pytest.mark.parametrize("device", get_available_devices())
def test_rollout_predictability(device):
    env = MockSerialEnv(device=device)
    env.set_seed(100)
    first = 100 % 17
    policy = Actor(torch.nn.Linear(1, 1, bias=False)).to(device)
    for p in policy.parameters():
        p.data.fill_(1.0)
    td_out = env.rollout(policy=policy, max_steps=200)
    assert (
        torch.arange(first, first + 100, device=device)
        == td_out.get("observation").squeeze()
    ).all()
    assert (
        torch.arange(first + 1, first + 101, device=device)
        == td_out.get("next_observation").squeeze()
    ).all()
    assert (
        torch.arange(first + 1, first + 101, device=device)
        == td_out.get("reward").squeeze()
    ).all()
    assert (
        torch.arange(first, first + 100, device=device)
        == td_out.get("action").squeeze()
    ).all()


def _make_envs(
    env_name,
    frame_skip,
    transformed_in,
    transformed_out,
    N,
    selected_keys=None,
    device="cpu",
    kwargs=None,
):
    torch.manual_seed(0)
    if not transformed_in:
        create_env_fn = lambda: GymEnv(env_name, frame_skip=frame_skip, device=device)
    else:
        if env_name == "ALE/Pong-v5":
            create_env_fn = lambda: TransformedEnv(
                GymEnv(env_name, frame_skip=frame_skip, device=device),
                Compose(*[ToTensorImage(), RewardClipping(0, 0.1)]),
            )
        else:
            create_env_fn = lambda: TransformedEnv(
                GymEnv(env_name, frame_skip=frame_skip, device=device),
                Compose(
                    ObservationNorm(keys_in=["next_observation"], loc=0.5, scale=1.1),
                    RewardClipping(0, 0.1),
                ),
            )
    env0 = create_env_fn()
    env_parallel = ParallelEnv(
        N, create_env_fn, selected_keys=selected_keys, create_env_kwargs=kwargs
    )
    env_serial = SerialEnv(
        N, create_env_fn, selected_keys=selected_keys, create_env_kwargs=kwargs
    )
    if transformed_out:
        if env_name == "ALE/Pong-v5":
            t_out = lambda: (
                Compose(*[ToTensorImage(), RewardClipping(0, 0.1)])
                if not transformed_in
                else Compose(
                    *[ObservationNorm(keys_in=["next_pixels"], loc=0, scale=1)]
                )
            )
            env0 = TransformedEnv(
                env0,
                t_out(),
            )
            env_parallel = TransformedEnv(
                env_parallel,
                t_out(),
            )
            env_serial = TransformedEnv(
                env_serial,
                t_out(),
            )
        else:
            t_out = lambda: (
                Compose(
                    ObservationNorm(keys_in=["next_observation"], loc=0.5, scale=1.1),
                    RewardClipping(0, 0.1),
                )
                if not transformed_in
                else Compose(
                    ObservationNorm(keys_in=["next_observation"], loc=1.0, scale=1.0)
                )
            )
            env0 = TransformedEnv(
                env0,
                t_out(),
            )
            env_parallel = TransformedEnv(
                env_parallel,
                t_out(),
            )
            env_serial = TransformedEnv(
                env_serial,
                t_out(),
            )

    return env_parallel, env_serial, env0


class TestParallel:
    @pytest.mark.skipif(not _has_gym, reason="no gym")
    @pytest.mark.parametrize("env_name", ["ALE/Pong-v5", "Pendulum-v1"])
    @pytest.mark.parametrize("frame_skip", [4, 1])
    @pytest.mark.parametrize("transformed_in", [False, True])
    @pytest.mark.parametrize("transformed_out", [False, True])
    def test_parallel_env(
        self, env_name, frame_skip, transformed_in, transformed_out, T=10, N=3
    ):
        env_parallel, env_serial, env0 = _make_envs(
            env_name,
            frame_skip,
            transformed_in=transformed_in,
            transformed_out=transformed_out,
            N=N,
        )

        td = TensorDict(
            source={"action": env0.action_spec.rand((N,))},
            batch_size=[
                N,
            ],
        )
        td1 = env_parallel.step(td)
        assert not td1.is_shared()
        assert "done" in td1.keys()
        assert "reward" in td1.keys()

        with pytest.raises(RuntimeError):
            # number of actions does not match number of workers
            td = TensorDict(
                source={"action": env0.action_spec.rand((N - 1,))}, batch_size=[N - 1]
            )
            td1 = env_parallel.step(td)

        td_reset = TensorDict(
            source={"reset_workers": torch.zeros(N, 1, dtype=torch.bool).bernoulli_()},
            batch_size=[
                N,
            ],
        )
        env_parallel.reset(tensordict=td_reset)

        td = env_parallel.rollout(policy=None, max_steps=T)
        assert (
            td.shape == torch.Size([N, T]) or td.get("done").sum(1).all()
        ), f"{td.shape}, {td.get('done').sum(1)}"
        env_parallel.close()
        # env_serial.close()  # never opened
        env0.close()

    @pytest.mark.skipif(not _has_gym, reason="no gym")
    @pytest.mark.parametrize("env_name", ["Pendulum-v1"])
    @pytest.mark.parametrize("frame_skip", [4, 1])
    @pytest.mark.parametrize("transformed_in", [True, False])
    @pytest.mark.parametrize("transformed_out", [True, False])
    @pytest.mark.parametrize(
        "selected_keys",
        [
            ["action", "observation", "next_observation", "done", "reward"],
            ["hidden", "action", "observation", "next_observation", "done", "reward"],
            None,
        ],
    )
    def test_parallel_env_with_policy(
        self,
        env_name,
        frame_skip,
        transformed_in,
        transformed_out,
        selected_keys,
        T=10,
        N=3,
    ):
        env_parallel, env_serial, env0 = _make_envs(
            env_name,
            frame_skip,
            transformed_in=transformed_in,
            transformed_out=transformed_out,
            N=N,
            selected_keys=selected_keys,
        )

        policy = ActorCriticOperator(
            TensorDictModule(
                spec=None,
                module=nn.LazyLinear(12),
                in_keys=["observation"],
                out_keys=["hidden"],
            ),
            TensorDictModule(
                spec=None,
                module=nn.LazyLinear(env0.action_spec.shape[-1]),
                in_keys=["hidden"],
                out_keys=["action"],
            ),
            ValueOperator(
                module=MLP(out_features=1, num_cells=[]), in_keys=["hidden", "action"]
            ),
        )

        td = TensorDict(
            source={"action": env0.action_spec.rand((N,))},
            batch_size=[
                N,
            ],
        )
        td1 = env_parallel.step(td)
        assert not td1.is_shared()
        assert "done" in td1.keys()
        assert "reward" in td1.keys()

        with pytest.raises(RuntimeError):
            # number of actions does not match number of workers
            td = TensorDict(
                source={"action": env0.action_spec.rand((N - 1,))}, batch_size=[N - 1]
            )
            td1 = env_parallel.step(td)

        td_reset = TensorDict(
            source={"reset_workers": torch.zeros(N, 1, dtype=torch.bool).bernoulli_()},
            batch_size=[
                N,
            ],
        )
        env_parallel.reset(tensordict=td_reset)

        td = env_parallel.rollout(policy=policy, max_steps=T)
        assert (
            td.shape == torch.Size([N, T]) or td.get("done").sum(1).all()
        ), f"{td.shape}, {td.get('done').sum(1)}"
        env_parallel.close()
        # env_serial.close()
        env0.close()

    @pytest.mark.skipif(not _has_gym, reason="no gym")
    @pytest.mark.parametrize(
        "env_name",
        [
            "Pendulum-v1",
            "ALE/Pong-v5",
        ],
    )
    @pytest.mark.parametrize("frame_skip", [4, 1])
    @pytest.mark.parametrize("transformed_in", [False, True])
    @pytest.mark.parametrize("transformed_out", [True, False])
    def test_parallel_env_seed(
        self, env_name, frame_skip, transformed_in, transformed_out
    ):
        env_parallel, env_serial, _ = _make_envs(
            env_name, frame_skip, transformed_in, transformed_out, 5
        )

        out_seed_serial = env_serial.set_seed(0)
        td0_serial = env_serial.reset()
        torch.manual_seed(0)

        td_serial = env_serial.rollout(
            max_steps=10, auto_reset=False, tensordict=td0_serial
        ).contiguous()
        key = "pixels" if "pixels" in td_serial else "observation"
        torch.testing.assert_allclose(
            td_serial[:, 0].get("next_" + key), td_serial[:, 1].get(key)
        )

        out_seed_parallel = env_parallel.set_seed(0)
        td0_parallel = env_parallel.reset()

        torch.manual_seed(0)
        assert out_seed_parallel == out_seed_serial
        td_parallel = env_parallel.rollout(
            max_steps=10, auto_reset=False, tensordict=td0_parallel
        ).contiguous()
        torch.testing.assert_allclose(
            td_parallel[:, :-1].get("next_" + key), td_parallel[:, 1:].get(key)
        )

        assert_allclose_td(td0_serial, td0_parallel)
        assert_allclose_td(td_serial[:, 0], td_parallel[:, 0])  # first step
        assert_allclose_td(td_serial[:, 1], td_parallel[:, 1])  # second step
        assert_allclose_td(td_serial, td_parallel)
        env_parallel.close()
        env_serial.close()

    @pytest.mark.skipif(not _has_gym, reason="no gym")
    def test_parallel_env_shutdown(self):
        env_make = EnvCreator(lambda: GymEnv("Pendulum-v1"))
        env = ParallelEnv(4, env_make)
        env.reset()
        assert not env.is_closed
        env.rand_step()
        assert not env.is_closed
        env.close()
        assert env.is_closed
        env.reset()
        assert not env.is_closed
        env.close()

    @pytest.mark.parametrize("parallel", [True, False])
    def test_parallel_env_custom_method(self, parallel):
        # define env

        if parallel:
            env = ParallelEnv(3, lambda: DiscreteActionVecMockEnv())
        else:
            env = SerialEnv(3, lambda: DiscreteActionVecMockEnv())

        # we must start the environment first
        env.reset()
        assert all(result == 0 for result in env.custom_fun())
        assert all(result == 1 for result in env.custom_attr)
        assert all(result == 2 for result in env.custom_prop)  # to be fixed
        env.close()

    @pytest.mark.skipif(not torch.cuda.device_count(), reason="no cuda to test on")
    @pytest.mark.skipif(not _has_gym, reason="no gym")
    @pytest.mark.parametrize("frame_skip", [4])
    @pytest.mark.parametrize("device", [0])
    @pytest.mark.parametrize("env_name", ["ALE/Pong-v5", "Pendulum-v1"])
    @pytest.mark.parametrize("transformed_in", [True, False])
    @pytest.mark.parametrize("transformed_out", [False, True])
    @pytest.mark.parametrize("open_before", [False, True])
    def test_parallel_env_cast(
        self,
        env_name,
        frame_skip,
        transformed_in,
        transformed_out,
        device,
        open_before,
        T=10,
        N=3,
    ):
        # tests casting to device
        env_parallel, env_serial, env0 = _make_envs(
            env_name,
            frame_skip,
            transformed_in=transformed_in,
            transformed_out=transformed_out,
            N=N,
        )
        if open_before:
            td_cpu = env0.rollout(max_steps=10)
            assert td_cpu.device == torch.device("cpu")
        env0 = env0.to(device)
        assert env0.observation_spec.device == torch.device(device)
        assert env0.action_spec.device == torch.device(device)
        assert env0.reward_spec.device == torch.device(device)
        assert env0.device == torch.device(device)
        td_device = env0.reset()
        assert td_device.device == torch.device(device), env0
        td_device = env0.rand_step()
        assert td_device.device == torch.device(device), env0
        td_device = env0.rollout(max_steps=10)
        assert td_device.device == torch.device(device), env0

        if open_before:
            td_cpu = env_serial.rollout(max_steps=10)
            assert td_cpu.device == torch.device("cpu")
        env_serial = env_serial.to(device)
        assert env_serial.observation_spec.device == torch.device(device)
        assert env_serial.action_spec.device == torch.device(device)
        assert env_serial.reward_spec.device == torch.device(device)
        assert env_serial.device == torch.device(device)
        td_device = env_serial.reset()
        assert td_device.device == torch.device(device), env_serial
        td_device = env_serial.rand_step()
        assert td_device.device == torch.device(device), env_serial
        td_device = env_serial.rollout(max_steps=10)
        assert td_device.device == torch.device(device), env_serial

        if open_before:
            td_cpu = env_parallel.rollout(max_steps=10)
            assert td_cpu.device == torch.device("cpu")
        env_parallel = env_parallel.to(device)
        assert env_parallel.observation_spec.device == torch.device(device)
        assert env_parallel.action_spec.device == torch.device(device)
        assert env_parallel.reward_spec.device == torch.device(device)
        assert env_parallel.device == torch.device(device)
        td_device = env_parallel.reset()
        assert td_device.device == torch.device(device), env_parallel
        td_device = env_parallel.rand_step()
        assert td_device.device == torch.device(device), env_parallel
        td_device = env_parallel.rollout(max_steps=10)
        assert td_device.device == torch.device(device), env_parallel

        env_parallel.close()
        env_serial.close()
        env0.close()

    @pytest.mark.skipif(not _has_gym, reason="no gym")
    @pytest.mark.skipif(not torch.cuda.device_count(), reason="no cuda device detected")
    @pytest.mark.parametrize("frame_skip", [4])
    @pytest.mark.parametrize("device", [0])
    @pytest.mark.parametrize("env_name", ["ALE/Pong-v5", "Pendulum-v1"])
    @pytest.mark.parametrize("transformed_in", [True, False])
    @pytest.mark.parametrize("transformed_out", [True, False])
    def test_parallel_env_device(
        self, env_name, frame_skip, transformed_in, transformed_out, device
    ):
        # tests creation on device
        torch.manual_seed(0)
        N = 3

        env_parallel, env_serial, env0 = _make_envs(
            env_name,
            frame_skip,
            transformed_in=transformed_in,
            transformed_out=transformed_out,
            device=device,
            N=N,
        )

        assert env0.device == torch.device(device)
        out = env0.rollout(max_steps=20)
        assert out.device == torch.device(device)

        assert env_serial.device == torch.device(device)
        out = env_serial.rollout(max_steps=20)
        assert out.device == torch.device(device)

        assert env_parallel.device == torch.device(device)
        out = env_parallel.rollout(max_steps=20)
        assert out.device == torch.device(device)

        env_parallel.close()
        env_serial.close()
        env0.close()

    @pytest.mark.skipif(not _has_gym, reason="no gym")
    @pytest.mark.parametrize("env_name", ["ALE/Pong-v5", "Pendulum-v1"])
    @pytest.mark.parametrize("frame_skip", [4, 1])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_parallel_env_transform_consistency(self, env_name, frame_skip, device):
        env_parallel_in, env_serial_in, env0_in = _make_envs(
            env_name,
            frame_skip,
            transformed_in=True,
            transformed_out=False,
            device=device,
            N=3,
        )
        env_parallel_out, env_serial_out, env0_out = _make_envs(
            env_name,
            frame_skip,
            transformed_in=False,
            transformed_out=True,
            device=device,
            N=3,
        )
        torch.manual_seed(0)
        env_parallel_in.set_seed(0)
        r_in = env_parallel_in.rollout(max_steps=20)
        torch.manual_seed(0)
        env_parallel_out.set_seed(0)
        r_out = env_parallel_out.rollout(max_steps=20)
        assert_allclose_td(r_in, r_out)
        env_parallel_in.close()
        env_parallel_out.close()

        torch.manual_seed(0)
        env_serial_in.set_seed(0)
        r_in = env_serial_in.rollout(max_steps=20)
        torch.manual_seed(0)
        env_serial_out.set_seed(0)
        r_out = env_serial_out.rollout(max_steps=20)
        assert_allclose_td(r_in, r_out)
        env_serial_in.close()
        env_serial_out.close()

        torch.manual_seed(0)
        env0_in.set_seed(0)
        r_in = env0_in.rollout(max_steps=20)
        torch.manual_seed(0)
        env0_out.set_seed(0)
        r_out = env0_out.rollout(max_steps=20)
        assert_allclose_td(r_in, r_out)
        env0_in.close()
        env0_in.close()

    @pytest.mark.parametrize("parallel", [True, False])
    def test_parallel_env_kwargs_set(self, parallel):
        num_env = 3

        def make_make_env():
            def make_transformed_env(seed=None):
                env = DiscreteActionConvMockEnv()
                if seed is not None:
                    env.set_seed(seed)
                return env

            return make_transformed_env

        _class = ParallelEnv if parallel else SerialEnv

        def env_fn1(seed):
            env = _class(
                num_workers=num_env,
                create_env_fn=make_make_env(),
                create_env_kwargs=[{"seed": i} for i in range(seed, seed + num_env)],
            )
            return env

        def env_fn2(seed):
            env = _class(
                num_workers=num_env,
                create_env_fn=make_make_env(),
            )
            env.update_kwargs([{"seed": i} for i in range(seed, seed + num_env)])
            return env

        env1 = env_fn1(100)
        env2 = env_fn2(100)
        env1.start()
        env2.start()
        for c1, c2 in zip(env1.counter, env2.counter):
            assert c1 == c2

        env1.close()
        env2.close()


class TestSpec:
    def test_discrete_action_spec_reconstruct(self):
        torch.manual_seed(0)
        action_spec = OneHotDiscreteTensorSpec(10)

        actions_tensors = [action_spec.rand() for _ in range(10)]
        actions_numpy = [action_spec.to_numpy(a) for a in actions_tensors]
        actions_tensors_2 = [action_spec.encode(a) for a in actions_numpy]
        assert all(
            [(a1 == a2).all() for a1, a2 in zip(actions_tensors, actions_tensors_2)]
        )

        actions_numpy = [int(np.random.randint(0, 10, (1,))) for a in actions_tensors]
        actions_tensors = [action_spec.encode(a) for a in actions_numpy]
        actions_numpy_2 = [action_spec.to_numpy(a) for a in actions_tensors]
        assert all([(a1 == a2) for a1, a2 in zip(actions_numpy, actions_numpy_2)])

    def test_mult_discrete_action_spec_reconstruct(self):
        torch.manual_seed(0)
        action_spec = MultOneHotDiscreteTensorSpec((10, 5))

        actions_tensors = [action_spec.rand() for _ in range(10)]
        actions_numpy = [action_spec.to_numpy(a) for a in actions_tensors]
        actions_tensors_2 = [action_spec.encode(a) for a in actions_numpy]
        assert all(
            [(a1 == a2).all() for a1, a2 in zip(actions_tensors, actions_tensors_2)]
        )

        actions_numpy = [
            np.concatenate(
                [np.random.randint(0, 10, (1,)), np.random.randint(0, 5, (1,))], 0
            )
            for a in actions_tensors
        ]
        actions_tensors = [action_spec.encode(a) for a in actions_numpy]
        actions_numpy_2 = [action_spec.to_numpy(a) for a in actions_tensors]
        assert all([(a1 == a2).all() for a1, a2 in zip(actions_numpy, actions_numpy_2)])

    def test_discrete_action_spec_rand(self):
        torch.manual_seed(0)
        action_spec = OneHotDiscreteTensorSpec(10)

        sample = torch.stack([action_spec.rand() for _ in range(10000)], 0)

        sample_list = sample.argmax(-1)
        sample_list = list([sum(sample_list == i).item() for i in range(10)])
        assert chisquare(sample_list).pvalue > 0.1

        sample = action_spec.to_numpy(sample)
        sample = [sum(sample == i) for i in range(10)]
        assert chisquare(sample).pvalue > 0.1

    def test_mult_discrete_action_spec_rand(self):
        torch.manual_seed(0)
        ns = (10, 5)
        N = 100000
        action_spec = MultOneHotDiscreteTensorSpec((10, 5))

        actions_tensors = [action_spec.rand() for _ in range(10)]
        actions_numpy = [action_spec.to_numpy(a) for a in actions_tensors]
        actions_tensors_2 = [action_spec.encode(a) for a in actions_numpy]
        assert all(
            [(a1 == a2).all() for a1, a2 in zip(actions_tensors, actions_tensors_2)]
        )

        sample = np.stack(
            [action_spec.to_numpy(action_spec.rand()) for _ in range(N)], 0
        )
        assert sample.shape[0] == N
        assert sample.shape[1] == 2
        assert sample.ndim == 2, f"found shape: {sample.shape}"

        sample0 = sample[:, 0]
        sample_list = list([sum(sample0 == i) for i in range(ns[0])])
        assert chisquare(sample_list).pvalue > 0.1

        sample1 = sample[:, 1]
        sample_list = list([sum(sample1 == i) for i in range(ns[1])])
        assert chisquare(sample_list).pvalue > 0.1

    def test_bounded_rand(self):
        spec = BoundedTensorSpec(-3, 3)
        sample = torch.stack([spec.rand() for _ in range(100)])
        assert (-3 <= sample).all() and (3 >= sample).all()

    def test_ndbounded_shape(self):
        spec = NdBoundedTensorSpec(-3, 3 * torch.ones(10, 5), shape=[10, 5])
        sample = torch.stack([spec.rand() for _ in range(100)], 0)
        assert (-3 <= sample).all() and (3 >= sample).all()
        assert sample.shape == torch.Size([100, 10, 5])


@pytest.mark.skipif(not _has_gym, reason="no gym")
def test_seed():
    torch.manual_seed(0)
    env1 = GymEnv("Pendulum-v1")
    env1.set_seed(0)
    state0_1 = env1.reset()
    state1_1 = env1.step(state0_1.set("action", env1.action_spec.rand()))

    torch.manual_seed(0)
    env2 = GymEnv("Pendulum-v1")
    env2.set_seed(0)
    state0_2 = env2.reset()
    state1_2 = env2.step(state0_2.set("action", env2.action_spec.rand()))

    assert_allclose_td(state0_1, state0_2)
    assert_allclose_td(state1_1, state1_2)

    env1.set_seed(0)
    torch.manual_seed(0)
    rollout1 = env1.rollout(max_steps=30)

    env2.set_seed(0)
    torch.manual_seed(0)
    rollout2 = env2.rollout(max_steps=30)

    torch.testing.assert_allclose(
        rollout1["observation"][1:], rollout1["next_observation"][:-1]
    )
    torch.testing.assert_allclose(
        rollout2["observation"][1:], rollout2["next_observation"][:-1]
    )
    torch.testing.assert_allclose(rollout1["observation"], rollout2["observation"])


@pytest.mark.parametrize("keep_other", [True, False])
@pytest.mark.parametrize("exclude_reward", [True, False])
@pytest.mark.parametrize("exclude_done", [True, False])
@pytest.mark.parametrize("exclude_action", [True, False])
@pytest.mark.parametrize("has_out", [True, False])
def test_steptensordict(
    keep_other, exclude_reward, exclude_done, exclude_action, has_out
):
    torch.manual_seed(0)
    tensordict = TensorDict(
        {
            "ledzep": torch.randn(4, 2),
            "next_ledzep": torch.randn(4, 2),
            "reward": torch.randn(4, 1),
            "done": torch.zeros(4, 1, dtype=torch.bool),
            "beatles": torch.randn(4, 1),
            "action": torch.randn(4, 2),
        },
        [4],
    )
    next_tensordict = TensorDict({}, [4]) if has_out else None
    out = step_tensordict(
        tensordict,
        keep_other=keep_other,
        exclude_reward=exclude_reward,
        exclude_done=exclude_done,
        exclude_action=exclude_action,
        next_tensordict=next_tensordict,
    )
    assert "ledzep" in out.keys()
    assert out["ledzep"] is tensordict["next_ledzep"]
    if keep_other:
        assert "beatles" in out.keys()
        assert out["beatles"] is tensordict["beatles"]
    else:
        assert "beatles" not in out.keys()
    if not exclude_reward:
        assert "reward" in out.keys()
        assert out["reward"] is tensordict["reward"]
    else:
        assert "reward" not in out.keys()
    if not exclude_action:
        assert "action" in out.keys()
        assert out["action"] is tensordict["action"]
    else:
        assert "action" not in out.keys()
    if not exclude_done:
        assert "done" in out.keys()
        assert out["done"] is tensordict["done"]
    else:
        assert "done" not in out.keys()
    if has_out:
        assert out is next_tensordict


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
