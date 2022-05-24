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
from mocking_classes import DiscreteActionVecMockEnv, MockSerialEnv
from scipy.stats import chisquare
from torch import nn
from torchrl.data.tensor_specs import (
    OneHotDiscreteTensorSpec,
    MultOneHotDiscreteTensorSpec,
    BoundedTensorSpec,
    NdBoundedTensorSpec,
)
from torchrl.data.tensordict.tensordict import assert_allclose_td, TensorDict
from torchrl.envs import EnvCreator
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
from torchrl.modules import ActorCriticOperator, TDModule, ValueOperator, Actor

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
    policy = Actor(torch.nn.Linear(1, 1, bias=False))
    for p in policy.parameters():
        p.data.fill_(1.0)
    td_out = env.rollout(policy=policy, max_steps=200)
    assert (torch.arange(100, 200) == td_out.get("observation").squeeze()).all()
    assert (torch.arange(101, 201) == td_out.get("next_observation").squeeze()).all()
    assert (torch.arange(101, 201) == td_out.get("reward").squeeze()).all()
    assert (torch.arange(100, 200) == td_out.get("action").squeeze()).all()


def _make_envs(env_name, frame_skip, transformed, N, selected_keys=None):
    torch.manual_seed(0)
    if not transformed:
        create_env_fn = lambda: GymEnv(env_name, frame_skip=frame_skip)
    else:
        if env_name == "ALE/Pong-v5":
            create_env_fn = lambda: TransformedEnv(
                GymEnv(env_name, frame_skip=frame_skip),
                Compose(*[ToTensorImage(), RewardClipping(0, 0.1)]),
            )
        else:
            create_env_fn = lambda: TransformedEnv(
                GymEnv(env_name, frame_skip=frame_skip),
                Compose(*[RewardClipping(0, 0.1)]),
            )
    env0 = create_env_fn()
    env_parallel = ParallelEnv(N, create_env_fn, selected_keys=selected_keys)
    env_serial = SerialEnv(N, create_env_fn, selected_keys=selected_keys)
    return env_parallel, env_serial, env0


@pytest.mark.skipif(not _has_gym, reason="no gym")
@pytest.mark.parametrize("env_name", ["ALE/Pong-v5", "Pendulum-v1"])
@pytest.mark.parametrize("frame_skip", [4, 1])
@pytest.mark.parametrize("transformed", [True, False])
def test_parallel_env(env_name, frame_skip, transformed, T=10, N=5):
    env_parallel, env_serial, env0 = _make_envs(env_name, frame_skip, transformed, N)

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


@pytest.mark.skipif(not _has_gym, reason="no gym")
@pytest.mark.parametrize("env_name", ["Pendulum-v1"])
@pytest.mark.parametrize("frame_skip", [4, 1])
@pytest.mark.parametrize("transformed", [True, False])
@pytest.mark.parametrize(
    "selected_keys",
    [
        ["action", "observation", "next_observation", "done", "reward"],
        ["hidden", "action", "observation", "next_observation", "done", "reward"],
        None,
    ],
)
def test_parallel_env_with_policy(
    env_name, frame_skip, transformed, selected_keys, T=10, N=5
):
    env_parallel, env_serial, env0 = _make_envs(
        env_name, frame_skip, transformed, N, selected_keys
    )

    policy = ActorCriticOperator(
        TDModule(
            spec=None,
            module=nn.LazyLinear(12),
            in_keys=["observation"],
            out_keys=["hidden"],
        ),
        TDModule(
            spec=None,
            module=nn.LazyLinear(env0.action_spec.shape[-1]),
            in_keys=["hidden"],
            out_keys=["action"],
        ),
        ValueOperator(module=nn.LazyLinear(1), in_keys=["hidden"]),
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


@pytest.mark.skipif(not _has_gym, reason="no gym")
@pytest.mark.parametrize("env_name", ["ALE/Pong-v5", "Pendulum-v1"])
@pytest.mark.parametrize("frame_skip", [4, 1])
@pytest.mark.parametrize(
    "transformed",
    [
        False,
        True,
    ],
)
def test_parallel_env_seed(env_name, frame_skip, transformed):
    env_parallel, env_serial, env0 = _make_envs(env_name, frame_skip, transformed, 5)

    out_seed_serial = env_serial.set_seed(0)
    env_serial.reset()
    td0_serial = env_serial.current_tensordict
    torch.manual_seed(0)

    td_serial = env_serial.rollout(max_steps=10, auto_reset=False).contiguous()
    key = "pixels" if "pixels" in td_serial else "observation"
    torch.testing.assert_allclose(
        td_serial[:, 0].get("next_" + key), td_serial[:, 1].get(key)
    )

    out_seed_parallel = env_parallel.set_seed(0)
    env_parallel.reset()
    td0_parallel = env_parallel.current_tensordict

    torch.manual_seed(0)
    assert out_seed_parallel == out_seed_serial
    td_parallel = env_parallel.rollout(max_steps=10, auto_reset=False).contiguous()
    torch.testing.assert_allclose(
        td_parallel[:, 0].get("next_" + key), td_parallel[:, 1].get(key)
    )

    assert_allclose_td(td0_serial, td0_parallel)
    assert_allclose_td(td_serial[:, 0], td_parallel[:, 0])  # first step
    assert_allclose_td(td_serial[:, 1], td_parallel[:, 1])  # second step
    assert_allclose_td(td_serial, td_parallel)
    env_parallel.close()
    env_serial.close()
    env0.close()


@pytest.mark.skipif(not _has_gym, reason="no gym")
def test_parallel_env_shutdown():
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
def test_parallel_env_custom_method(parallel):
    # define env

    if parallel:
        env = ParallelEnv(3, lambda: DiscreteActionVecMockEnv())
    else:
        env = SerialEnv(3, lambda: DiscreteActionVecMockEnv())

    # we must start the environment first
    env.reset()
    assert all(result == 0 for result in env.custom_fun())
    assert all(result == 1 for result in env.custom_attr)
    assert all(result == 2 for result in env.custom_prop)
    env.close()


@pytest.mark.skipif(not _has_gym, reason="no gym")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="no cuda device detected")
@pytest.mark.parametrize("env_name", ["ALE/Pong-v5", "Pendulum-v1"])
@pytest.mark.parametrize("frame_skip", [4, 1])
@pytest.mark.parametrize("transformed", [True, False])
@pytest.mark.parametrize("device", [0, "cuda:0"])
def test_parallel_env_device(env_name, frame_skip, transformed, device):
    torch.manual_seed(0)
    N = 5
    if not transformed:
        create_env_fn = lambda: GymEnv("ALE/Pong-v5", frame_skip=frame_skip)
    else:
        if env_name == "ALE/Pong-v5":
            create_env_fn = lambda: TransformedEnv(
                GymEnv(env_name, frame_skip=frame_skip),
                Compose(*[ToTensorImage(), RewardClipping(0, 0.1)]),
            )
        else:
            create_env_fn = lambda: TransformedEnv(
                GymEnv(env_name, frame_skip=frame_skip),
                Compose(*[RewardClipping(0, 0.1)]),
            )
    env_parallel = ParallelEnv(N, create_env_fn, device=device)
    out = env_parallel.rollout(max_steps=20)
    env_parallel.close()


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


@pytest.mark.skipif(not _has_gym, reason="no gym")
def test_current_tensordict():
    torch.manual_seed(0)
    env = GymEnv("Pendulum-v1")
    env.set_seed(0)
    tensordict = env.reset()
    assert_allclose_td(tensordict, env.current_tensordict)
    tensordict = env.step(
        TensorDict(source={"action": env.action_spec.rand()}, batch_size=[])
    )
    assert_allclose_td(step_tensordict(tensordict), env.current_tensordict)


# TODO: test for frame-skip

if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
