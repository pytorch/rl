# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import functools
import gc
import os.path
import re
from collections import defaultdict
from functools import partial
from sys import platform

import numpy as np
import pytest
import torch
import yaml

from _utils_internal import (
    _make_envs,
    CARTPOLE_VERSIONED,
    check_rollout_consistency_multikey_env,
    decorate_thread_sub_func,
    get_default_devices,
    HALFCHEETAH_VERSIONED,
    PENDULUM_VERSIONED,
    PONG_VERSIONED,
    rand_reset,
)
from mocking_classes import (
    ActionObsMergeLinear,
    AutoResetHeteroCountingEnv,
    AutoResettingCountingEnv,
    ContinuousActionConvMockEnv,
    ContinuousActionConvMockEnvNumpy,
    ContinuousActionVecMockEnv,
    CountingBatchedEnv,
    CountingEnv,
    CountingEnvCountPolicy,
    DiscreteActionConvMockEnv,
    DiscreteActionConvMockEnvNumpy,
    DiscreteActionVecMockEnv,
    DummyModelBasedEnvBase,
    EnvWithDynamicSpec,
    HeterogeneousCountingEnv,
    HeterogeneousCountingEnvPolicy,
    MockBatchedLockedEnv,
    MockBatchedUnLockedEnv,
    MockSerialEnv,
    MultiKeyCountingEnv,
    MultiKeyCountingEnvPolicy,
    NestedCountingEnv,
)
from packaging import version
from tensordict import (
    assert_allclose_td,
    dense_stack_tds,
    LazyStackedTensorDict,
    TensorDict,
)
from tensordict.nn import TensorDictModuleBase
from tensordict.utils import _unravel_key_to_tuple
from torch import nn

from torchrl.collectors import MultiSyncDataCollector, SyncDataCollector
from torchrl.data.tensor_specs import (
    CompositeSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs import (
    CatFrames,
    CatTensors,
    DoubleToFloat,
    EnvBase,
    EnvCreator,
    ParallelEnv,
    SerialEnv,
)
from torchrl.envs.batched_envs import _stackable
from torchrl.envs.gym_like import default_info_dict_reader
from torchrl.envs.libs.dm_control import _has_dmc, DMControlEnv
from torchrl.envs.libs.gym import _has_gym, gym_backend, GymEnv, GymWrapper
from torchrl.envs.transforms import Compose, StepCounter, TransformedEnv
from torchrl.envs.transforms.transforms import AutoResetEnv, AutoResetTransform
from torchrl.envs.utils import (
    _StepMDP,
    _terminated_or_truncated,
    check_env_specs,
    check_marl_grouping,
    make_composite_from_td,
    MarlGroupMapType,
    step_mdp,
)
from torchrl.modules import Actor, ActorCriticOperator, MLP, SafeModule, ValueOperator
from torchrl.modules.tensordict_module import WorldModelWrapper

gym_version = None
if _has_gym:
    try:
        import gymnasium as gym
    except ModuleNotFoundError:
        import gym

    gym_version = version.parse(gym.__version__)

try:
    this_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(this_dir, "configs", "atari.yaml"), "r") as file:
        atari_confs = yaml.load(file, Loader=yaml.FullLoader)
    _atari_found = True
except FileNotFoundError:
    _atari_found = False
    atari_confs = defaultdict(lambda: "")

IS_OSX = platform == "darwin"
IS_WIN = platform == "win32"
if IS_WIN:
    mp_ctx = "spawn"
else:
    mp_ctx = "fork"

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
@pytest.mark.parametrize("env_name", [PENDULUM_VERSIONED, CARTPOLE_VERSIONED])
@pytest.mark.parametrize("frame_skip", [1, 4])
def test_env_seed(env_name, frame_skip, seed=0):
    env_name = env_name()
    env = GymEnv(env_name, frame_skip=frame_skip)
    action = env.action_spec.rand()

    env.set_seed(seed)
    td0a = env.reset()
    td1a = env.step(td0a.clone().set("action", action))

    env.set_seed(seed)
    td0b = env.fake_tensordict()
    td0b = env.reset(tensordict=td0b)
    td1b = env.step(td0b.exclude("next").clone().set("action", action))

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
@pytest.mark.parametrize("env_name", [PENDULUM_VERSIONED, PONG_VERSIONED])
@pytest.mark.parametrize("frame_skip", [1, 4])
def test_rollout(env_name, frame_skip, seed=0):
    if env_name is PONG_VERSIONED and version.parse(
        gym_backend().__version__
    ) < version.parse("0.19"):
        # Then 100 steps in pong are not sufficient to detect a difference
        pytest.skip("can't detect difference in gym rollout with this gym version.")

    env_name = env_name()
    env = GymEnv(env_name, frame_skip=frame_skip)

    torch.manual_seed(seed)
    np.random.seed(seed)
    env.set_seed(seed)
    env.reset()
    rollout1 = env.rollout(max_steps=100)
    assert rollout1.names[-1] == "time"

    torch.manual_seed(seed)
    np.random.seed(seed)
    env.set_seed(seed)
    env.reset()
    rollout2 = env.rollout(max_steps=100)
    assert rollout2.names[-1] == "time"

    assert_allclose_td(rollout1, rollout2)

    torch.manual_seed(seed)
    env.set_seed(seed + 10)
    env.reset()
    rollout3 = env.rollout(max_steps=100)
    with pytest.raises(AssertionError):
        assert_allclose_td(rollout1, rollout3)
    env.close()


def test_rollout_set_truncated():
    env = ContinuousActionVecMockEnv()
    with pytest.raises(RuntimeError, match="set_truncated was set to True"):
        env.rollout(max_steps=10, set_truncated=True, break_when_any_done=False)
    env.add_truncated_keys()
    r = env.rollout(max_steps=10, set_truncated=True, break_when_any_done=False)
    assert r.shape == torch.Size([10])
    assert r[..., -1]["next", "truncated"].all()
    assert r[..., -1]["next", "done"].all()


@pytest.mark.parametrize("max_steps", [1, 5])
def test_rollouts_chaining(max_steps, batch_size=(4,), epochs=4):
    # CountingEnv is done at max_steps + 1, so to emulate it being done at max_steps, we feed max_steps=max_steps - 1
    env = CountingEnv(max_steps=max_steps - 1, batch_size=batch_size)
    policy = CountingEnvCountPolicy(
        action_spec=env.action_spec, action_key=env.action_key
    )

    input_td = env.reset()
    for _ in range(epochs):
        rollout_td = env.rollout(
            max_steps=max_steps,
            policy=policy,
            auto_reset=False,
            break_when_any_done=False,
            tensordict=input_td,
        )
        assert (env.count == max_steps).all()
        input_td = step_mdp(
            rollout_td[..., -1],
            keep_other=True,
            exclude_action=False,
            exclude_reward=True,
            reward_keys=env.reward_keys,
            action_keys=env.action_keys,
            done_keys=env.done_keys,
        )


@pytest.mark.parametrize("device", get_default_devices())
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
        == td_out.get(("next", "observation")).squeeze()
    ).all()
    assert (
        torch.arange(first + 1, first + 101, device=device)
        == td_out.get(("next", "reward")).squeeze()
    ).all()
    assert (
        torch.arange(first, first + 100, device=device)
        == td_out.get("action").squeeze()
    ).all()


@pytest.mark.skipif(not _has_gym, reason="no gym")
@pytest.mark.parametrize("env_name", [PENDULUM_VERSIONED])
@pytest.mark.parametrize("frame_skip", [1])
@pytest.mark.parametrize("truncated_key", ["truncated", "done"])
@pytest.mark.parametrize("parallel", [False, True])
def test_rollout_reset(
    env_name, frame_skip, parallel, truncated_key, maybe_fork_ParallelEnv, seed=0
):
    env_name = env_name()
    envs = []
    for horizon in [20, 30, 40]:
        envs.append(
            lambda horizon=horizon: TransformedEnv(
                GymEnv(env_name, frame_skip=frame_skip),
                StepCounter(horizon, truncated_key=truncated_key),
            )
        )
    if parallel:
        env = maybe_fork_ParallelEnv(3, envs)
    else:
        env = SerialEnv(3, envs)
    env.set_seed(100)
    out = env.rollout(100, break_when_any_done=False)
    assert out.names[-1] == "time"
    assert out.shape == torch.Size([3, 100])
    assert (
        out[..., -1]["step_count"].squeeze().cpu() == torch.tensor([19, 9, 19])
    ).all()
    assert (
        out[..., -1]["next", "step_count"].squeeze().cpu() == torch.tensor([20, 10, 20])
    ).all()
    assert (
        out["next", truncated_key].squeeze().sum(-1) == torch.tensor([5, 3, 2])
    ).all()


class TestModelBasedEnvBase:
    @staticmethod
    def world_model():
        return WorldModelWrapper(
            SafeModule(
                ActionObsMergeLinear(5, 4),
                in_keys=["hidden_observation", "action"],
                out_keys=["hidden_observation"],
            ),
            SafeModule(
                nn.Linear(4, 1),
                in_keys=["hidden_observation"],
                out_keys=["reward"],
            ),
        )

    @pytest.mark.parametrize("device", get_default_devices())
    def test_mb_rollout(self, device, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        world_model = self.world_model()
        mb_env = DummyModelBasedEnvBase(
            world_model, device=device, batch_size=torch.Size([10])
        )
        check_env_specs(mb_env)
        rollout = mb_env.rollout(max_steps=100)
        expected_keys = {
            ("next", key)
            for key in (*mb_env.observation_spec.keys(), "reward", "done", "terminated")
        }
        expected_keys = expected_keys.union(
            set(mb_env.input_spec["full_action_spec"].keys())
        )
        expected_keys = expected_keys.union(
            set(mb_env.input_spec["full_state_spec"].keys())
        )
        expected_keys = expected_keys.union({"done", "terminated", "next"})
        assert set(rollout.keys(True)) == expected_keys
        assert rollout[("next", "hidden_observation")].shape == (10, 100, 4)

    @pytest.mark.parametrize("device", get_default_devices())
    def test_mb_env_batch_lock(self, device, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        world_model = WorldModelWrapper(
            SafeModule(
                ActionObsMergeLinear(5, 4),
                in_keys=["hidden_observation", "action"],
                out_keys=["hidden_observation"],
            ),
            SafeModule(
                nn.Linear(4, 1),
                in_keys=["hidden_observation"],
                out_keys=["reward"],
            ),
        )
        mb_env = DummyModelBasedEnvBase(
            world_model, device=device, batch_size=torch.Size([10])
        )
        assert not mb_env.batch_locked

        with pytest.raises(RuntimeError, match="batch_locked is a read-only property"):
            mb_env.batch_locked = False
        td = mb_env.reset()
        td["action"] = mb_env.action_spec.rand()
        td_expanded = td.unsqueeze(-1).expand(10, 2).reshape(-1).to_tensordict()
        mb_env.step(td)

        with pytest.raises(
            RuntimeError,
            match=re.escape("Expected a tensordict with shape==env.batch_size"),
        ):
            mb_env.step(td_expanded)

        mb_env = DummyModelBasedEnvBase(
            world_model, device=device, batch_size=torch.Size([])
        )
        assert not mb_env.batch_locked

        with pytest.raises(RuntimeError, match="batch_locked is a read-only property"):
            mb_env.batch_locked = False
        td = mb_env.reset()
        td["action"] = mb_env.action_spec.rand()
        td_expanded = td.expand(2)
        mb_env.step(td)
        # we should be able to do a step with a tensordict that has been expended
        mb_env.step(td_expanded)


class TestParallel:
    @pytest.mark.skipif(
        not torch.cuda.device_count(), reason="No cuda device detected."
    )
    @pytest.mark.parametrize("parallel", [True, False])
    @pytest.mark.parametrize("hetero", [True, False])
    @pytest.mark.parametrize("pdevice", [None, "cpu", "cuda"])
    @pytest.mark.parametrize("edevice", ["cpu", "cuda"])
    @pytest.mark.parametrize("bwad", [True, False])
    def test_parallel_devices(
        self, parallel, hetero, pdevice, edevice, bwad, maybe_fork_ParallelEnv
    ):
        if parallel:
            cls = maybe_fork_ParallelEnv
        else:
            cls = SerialEnv
        if not hetero:
            env = cls(
                2, lambda: ContinuousActionVecMockEnv(device=edevice), device=pdevice
            )
        else:
            env1 = lambda: ContinuousActionVecMockEnv(device=edevice)
            env2 = lambda: TransformedEnv(ContinuousActionVecMockEnv(device=edevice))
            env = cls(2, [env1, env2], device=pdevice)

        r = env.rollout(2, break_when_any_done=bwad)
        if pdevice is not None:
            assert env.device.type == torch.device(pdevice).type
            assert r.device.type == torch.device(pdevice).type
            assert all(
                item.device.type == torch.device(pdevice).type
                for item in r.values(True, True)
            )
        else:
            assert env.device.type == torch.device(edevice).type
            assert r.device.type == torch.device(edevice).type
            assert all(
                item.device.type == torch.device(edevice).type
                for item in r.values(True, True)
            )
        if parallel:
            assert (
                env.shared_tensordict_parent.device.type == torch.device(edevice).type
            )

    @pytest.mark.parametrize("start_method", [None, mp_ctx])
    def test_serial_for_single(self, maybe_fork_ParallelEnv, start_method):
        env = ParallelEnv(
            1,
            ContinuousActionVecMockEnv,
            serial_for_single=True,
            mp_start_method=start_method,
        )
        assert isinstance(env, SerialEnv)
        env = ParallelEnv(1, ContinuousActionVecMockEnv, mp_start_method=start_method)
        assert isinstance(env, ParallelEnv)
        env = ParallelEnv(
            2,
            ContinuousActionVecMockEnv,
            serial_for_single=True,
            mp_start_method=start_method,
        )
        assert isinstance(env, ParallelEnv)

    @pytest.mark.parametrize("num_parallel_env", [1, 10])
    @pytest.mark.parametrize("env_batch_size", [[], (32,), (32, 1)])
    def test_env_with_batch_size(
        self, num_parallel_env, env_batch_size, maybe_fork_ParallelEnv
    ):
        env = MockBatchedLockedEnv(device="cpu", batch_size=torch.Size(env_batch_size))
        env.set_seed(1)
        parallel_env = maybe_fork_ParallelEnv(num_parallel_env, lambda: env)
        assert parallel_env.batch_size == (num_parallel_env, *env_batch_size)

    @pytest.mark.skipif(not _has_dmc, reason="no dm_control")
    @pytest.mark.parametrize("env_task", ["stand,stand,stand", "stand,walk,stand"])
    @pytest.mark.parametrize("share_individual_td", [True, False])
    def test_multi_task_serial_parallel(
        self, env_task, share_individual_td, maybe_fork_ParallelEnv
    ):
        tasks = env_task.split(",")
        if len(tasks) == 1:
            single_task = True

            def env_make():
                return DMControlEnv("humanoid", tasks[0])

        elif len(set(tasks)) == 1 and len(tasks) == 3:
            single_task = True
            env_make = [lambda: DMControlEnv("humanoid", tasks[0])] * 3
        else:
            single_task = False
            env_make = [
                lambda task=task: DMControlEnv("humanoid", task) for task in tasks
            ]

        env_serial = SerialEnv(3, env_make, share_individual_td=share_individual_td)
        env_serial.start()
        assert env_serial._single_task is single_task
        env_parallel = maybe_fork_ParallelEnv(
            3, env_make, share_individual_td=share_individual_td
        )
        env_parallel.start()
        assert env_parallel._single_task is single_task

        env_serial.set_seed(0)
        torch.manual_seed(0)
        td_serial = env_serial.rollout(max_steps=50)

        env_parallel.set_seed(0)
        torch.manual_seed(0)
        td_parallel = env_parallel.rollout(max_steps=50)

        assert_allclose_td(td_serial, td_parallel)

    @pytest.mark.skipif(not _has_dmc, reason="no dm_control")
    def test_multitask(self, maybe_fork_ParallelEnv):
        env1 = DMControlEnv("humanoid", "stand")
        env1_obs_keys = list(env1.observation_spec.keys())
        env2 = DMControlEnv("humanoid", "walk")
        env2_obs_keys = list(env2.observation_spec.keys())

        assert len(env1_obs_keys)
        assert len(env2_obs_keys)

        def env1_maker():
            return TransformedEnv(
                DMControlEnv("humanoid", "stand"),
                Compose(
                    CatTensors(env1_obs_keys, "observation_stand", del_keys=False),
                    CatTensors(env1_obs_keys, "observation"),
                    DoubleToFloat(
                        in_keys=["observation_stand", "observation"],
                        in_keys_inv=["action"],
                    ),
                ),
            )

        def env2_maker():
            return TransformedEnv(
                DMControlEnv("humanoid", "walk"),
                Compose(
                    CatTensors(env2_obs_keys, "observation_walk", del_keys=False),
                    CatTensors(env2_obs_keys, "observation"),
                    DoubleToFloat(
                        in_keys=["observation_walk", "observation"],
                        in_keys_inv=["action"],
                    ),
                ),
            )

        env = maybe_fork_ParallelEnv(2, [env1_maker, env2_maker])
        # env = SerialEnv(2, [env1_maker, env2_maker])
        assert not env._single_task

        td = env.rollout(10, return_contiguous=False)
        assert "observation_walk" not in td.keys()
        assert "observation_walk" in td[1].keys()
        assert "observation_walk" not in td[0].keys()
        assert "observation_stand" in td[0].keys()
        assert "observation_stand" not in td[1].keys()
        assert "observation_walk" in td[:, 0][1].keys()
        assert "observation_walk" not in td[:, 0][0].keys()
        assert "observation_stand" in td[:, 0][0].keys()
        assert "observation_stand" not in td[:, 0][1].keys()

    @pytest.mark.skipif(not _has_gym, reason="no gym")
    @pytest.mark.parametrize(
        "env_name", [PENDULUM_VERSIONED, CARTPOLE_VERSIONED]
    )  # 1226: faster execution
    @pytest.mark.parametrize("frame_skip", [4])  # 1226: faster execution
    @pytest.mark.parametrize(
        "transformed_in,transformed_out", [[True, True], [False, False]]
    )  # 1226: faster execution
    def test_parallel_env(
        self, env_name, frame_skip, transformed_in, transformed_out, T=10, N=3
    ):
        env_name = env_name()
        env_parallel, env_serial, _, env0 = _make_envs(
            env_name,
            frame_skip,
            transformed_in=transformed_in,
            transformed_out=transformed_out,
            N=N,
        )
        td = TensorDict(source={"action": env0.action_spec.rand((N,))}, batch_size=[N])
        td1 = env_parallel.step(td)
        assert not td1.is_shared()
        assert ("next", "done") in td1.keys(True)
        assert ("next", "reward") in td1.keys(True)

        with pytest.raises(RuntimeError):
            # number of actions does not match number of workers
            td = TensorDict(
                source={"action": env0.action_spec.rand((N - 1,))},
                batch_size=[N - 1],
            )
            _ = env_parallel.step(td)

        td_reset = TensorDict(source=rand_reset(env_parallel), batch_size=[N])
        env_parallel.reset(tensordict=td_reset)

        # check that interruption occured because of max_steps or done
        td = env_parallel.rollout(policy=None, max_steps=T)
        assert td.shape == torch.Size([N, T]) or td.get(("next", "done")).sum(1).any()
        env_parallel.close()
        # env_serial.close()  # never opened
        env0.close()

    @pytest.mark.skipif(not _has_gym, reason="no gym")
    @pytest.mark.parametrize("env_name", [PENDULUM_VERSIONED])
    @pytest.mark.parametrize("frame_skip", [4])  # 1226: faster execution
    @pytest.mark.parametrize(
        "transformed_in,transformed_out", [[True, True], [False, False]]
    )  # 1226: faster execution
    def test_parallel_env_with_policy(
        self,
        env_name,
        frame_skip,
        transformed_in,
        transformed_out,
        T=10,
        N=3,
    ):
        env_name = env_name()
        env_parallel, env_serial, _, env0 = _make_envs(
            env_name,
            frame_skip,
            transformed_in=transformed_in,
            transformed_out=transformed_out,
            N=N,
        )

        policy = ActorCriticOperator(
            SafeModule(
                spec=None,
                module=nn.LazyLinear(12),
                in_keys=["observation"],
                out_keys=["hidden"],
            ),
            SafeModule(
                spec=None,
                module=nn.LazyLinear(env0.action_spec.shape[-1]),
                in_keys=["hidden"],
                out_keys=["action"],
            ),
            ValueOperator(
                module=MLP(out_features=1, num_cells=[]), in_keys=["hidden", "action"]
            ),
        )

        td = TensorDict(source={"action": env0.action_spec.rand((N,))}, batch_size=[N])
        td1 = env_parallel.step(td)
        assert not td1.is_shared()
        assert ("next", "done") in td1.keys(True)
        assert ("next", "reward") in td1.keys(True)

        with pytest.raises(RuntimeError):
            # number of actions does not match number of workers
            td = TensorDict(
                source={"action": env0.action_spec.rand((N - 1,))},
                batch_size=[N - 1],
            )
            _ = env_parallel.step(td)

        td_reset = TensorDict(source=rand_reset(env_parallel), batch_size=[N])
        env_parallel.reset(tensordict=td_reset)

        td = env_parallel.rollout(policy=policy, max_steps=T)
        assert (
            td.shape == torch.Size([N, T]) or td.get("done").sum(1).all()
        ), f"{td.shape}, {td.get('done').sum(1)}"
        env_parallel.close()

        # env_serial.close()
        env0.close()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.parametrize("heterogeneous", [False, True])
    def test_transform_env_transform_no_device(
        self, heterogeneous, maybe_fork_ParallelEnv
    ):
        # Tests non-regression on 1865
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(), StepCounter(max_steps=3)
            )

        if heterogeneous:
            make_envs = [EnvCreator(make_env), EnvCreator(make_env)]
        else:
            make_envs = make_env
        penv = maybe_fork_ParallelEnv(2, make_envs)
        r = penv.rollout(6, break_when_any_done=False)
        assert r.shape == (2, 6)
        try:
            env = TransformedEnv(penv)
            r = env.rollout(6, break_when_any_done=False)
            assert r.shape == (2, 6)
        finally:
            penv.close()

    @pytest.mark.skipif(not _has_gym, reason="no gym")
    @pytest.mark.parametrize(
        "env_name",
        [PENDULUM_VERSIONED],
    )  # PONG_VERSIONED])  # 1226: efficiency
    @pytest.mark.parametrize("frame_skip", [4])
    @pytest.mark.parametrize(
        "transformed_in,transformed_out", [[True, True], [False, False]]
    )  # 1226: effociency
    @pytest.mark.parametrize("static_seed", [False, True])
    def test_parallel_env_seed(
        self, env_name, frame_skip, transformed_in, transformed_out, static_seed
    ):
        env_name = env_name()
        env_parallel, env_serial, _, _ = _make_envs(
            env_name, frame_skip, transformed_in, transformed_out, 5
        )
        out_seed_serial = env_serial.set_seed(0, static_seed=static_seed)
        if static_seed:
            assert out_seed_serial == 0
        td0_serial = env_serial.reset()
        torch.manual_seed(0)

        td_serial = env_serial.rollout(
            max_steps=10, auto_reset=False, tensordict=td0_serial
        ).contiguous()
        key = "pixels" if "pixels" in td_serial.keys() else "observation"
        torch.testing.assert_close(
            td_serial[:, 0].get(("next", key)), td_serial[:, 1].get(key)
        )

        out_seed_parallel = env_parallel.set_seed(0, static_seed=static_seed)
        if static_seed:
            assert out_seed_serial == 0
        td0_parallel = env_parallel.reset()

        torch.manual_seed(0)
        assert out_seed_parallel == out_seed_serial
        td_parallel = env_parallel.rollout(
            max_steps=10, auto_reset=False, tensordict=td0_parallel
        ).contiguous()
        torch.testing.assert_close(
            td_parallel[:, :-1].get(("next", key)), td_parallel[:, 1:].get(key)
        )
        assert_allclose_td(td0_serial, td0_parallel)
        assert_allclose_td(td_serial[:, 0], td_parallel[:, 0])  # first step
        assert_allclose_td(td_serial[:, 1], td_parallel[:, 1])  # second step
        assert_allclose_td(td_serial, td_parallel)
        env_parallel.close()
        env_serial.close()

    @pytest.mark.skipif(not _has_gym, reason="no gym")
    def test_parallel_env_shutdown(self, maybe_fork_ParallelEnv):
        env_make = EnvCreator(lambda: GymEnv(PENDULUM_VERSIONED()))
        env = maybe_fork_ParallelEnv(4, env_make)
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
    def test_parallel_env_custom_method(self, parallel, maybe_fork_ParallelEnv):
        # define env

        if parallel:
            env = maybe_fork_ParallelEnv(2, lambda: DiscreteActionVecMockEnv())
        else:
            env = SerialEnv(2, lambda: DiscreteActionVecMockEnv())

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
    @pytest.mark.parametrize(
        "env_name", [PENDULUM_VERSIONED]
    )  # 1226: Skip PONG for efficiency
    @pytest.mark.parametrize(
        "transformed_in,transformed_out,open_before",
        [  # 1226: efficiency
            [True, True, True],
            [True, True, False],
            [False, False, True],
        ],
    )
    def test_parallel_env_cast(
        self,
        env_name,
        frame_skip,
        transformed_in,
        transformed_out,
        device,
        open_before,
        N=3,
    ):
        env_name = env_name()
        # tests casting to device
        env_parallel, env_serial, _, env0 = _make_envs(
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
        observation_spec = env_serial.observation_spec.clone()
        done_spec = env_serial.done_spec.clone()
        reward_spec = env_serial.reward_spec.clone()
        action_spec = env_serial.action_spec.clone()
        state_spec = env_serial.state_spec.clone()
        env_serial = env_serial.to(device)
        assert env_serial.observation_spec.device == torch.device(device)
        assert env_serial.action_spec.device == torch.device(device)
        assert env_serial.reward_spec.device == torch.device(device)
        assert env_serial.device == torch.device(device)
        assert env_serial.observation_spec == observation_spec.to(device)
        assert env_serial.action_spec == action_spec.to(device)
        assert env_serial.reward_spec == reward_spec.to(device)
        assert env_serial.done_spec == done_spec.to(device)
        assert env_serial.state_spec == state_spec.to(device)
        td_device = env_serial.reset()
        assert td_device.device == torch.device(device), env_serial
        td_device = env_serial.rand_step()
        assert td_device.device == torch.device(device), env_serial
        td_device = env_serial.rollout(max_steps=10)
        assert td_device.device == torch.device(device), env_serial

        if open_before:
            td_cpu = env_parallel.rollout(max_steps=10)
            assert td_cpu.device == torch.device("cpu")
        observation_spec = env_parallel.observation_spec.clone()
        done_spec = env_parallel.done_spec.clone()
        reward_spec = env_parallel.reward_spec.clone()
        action_spec = env_parallel.action_spec.clone()
        state_spec = env_parallel.state_spec.clone()
        env_parallel = env_parallel.to(device)
        assert env_parallel.observation_spec.device == torch.device(device)
        assert env_parallel.action_spec.device == torch.device(device)
        assert env_parallel.reward_spec.device == torch.device(device)
        assert env_parallel.device == torch.device(device)
        assert env_parallel.observation_spec == observation_spec.to(device)
        assert env_parallel.action_spec == action_spec.to(device)
        assert env_parallel.reward_spec == reward_spec.to(device)
        assert env_parallel.done_spec == done_spec.to(device)
        assert env_parallel.state_spec == state_spec.to(device)
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
    @pytest.mark.parametrize("env_name", [PENDULUM_VERSIONED])  # 1226: efficiency
    @pytest.mark.parametrize(
        "transformed_in,transformed_out",
        [  # 1226
            [True, True],
            [False, False],
        ],
    )
    def test_parallel_env_device(
        self, env_name, frame_skip, transformed_in, transformed_out, device
    ):
        env_name = env_name()
        # tests creation on device
        torch.manual_seed(0)
        N = 3

        env_parallel, env_serial, _, env0 = _make_envs(
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
    @pytest.mark.flaky(reruns=3, reruns_delay=1)
    @pytest.mark.parametrize(
        "env_name", [PENDULUM_VERSIONED]
    )  # 1226: No pong for efficiency
    @pytest.mark.parametrize("frame_skip", [4])
    @pytest.mark.parametrize(
        "device",
        [torch.device("cuda:0") if torch.cuda.device_count() else torch.device("cpu")],
    )
    def test_parallel_env_transform_consistency(self, env_name, frame_skip, device):
        env_name = env_name()
        env_parallel_in, env_serial_in, _, env0_in = _make_envs(
            env_name,
            frame_skip,
            transformed_in=True,
            transformed_out=False,
            device=device,
            N=3,
        )
        env_parallel_out, env_serial_out, _, env0_out = _make_envs(
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
    def test_parallel_env_kwargs_set(self, parallel, maybe_fork_ParallelEnv):
        num_env = 2

        def make_make_env():
            def make_transformed_env(seed=None):
                env = DiscreteActionConvMockEnv()
                if seed is not None:
                    env.set_seed(seed)
                return env

            return make_transformed_env

        _class = maybe_fork_ParallelEnv if parallel else SerialEnv

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

    @pytest.mark.parametrize("batch_size", [(32, 5), (4,), (1,), ()])
    @pytest.mark.parametrize("n_workers", [2, 1])
    def test_parallel_env_reset_flag(
        self, batch_size, n_workers, maybe_fork_ParallelEnv, max_steps=3
    ):
        torch.manual_seed(1)
        env = maybe_fork_ParallelEnv(
            n_workers, lambda: CountingEnv(max_steps=max_steps, batch_size=batch_size)
        )
        env.set_seed(1)
        action = env.action_spec.rand()
        action[:] = 1
        for i in range(max_steps):
            td = env.step(
                TensorDict(
                    {"action": action}, batch_size=env.batch_size, device=env.device
                )
            )
            assert (td["next", "done"] == 0).all()
            assert (td["next"]["observation"] == i + 1).all()

        td = env.step(
            TensorDict({"action": action}, batch_size=env.batch_size, device=env.device)
        )
        assert (td["next", "done"] == 1).all()
        assert (td["next"]["observation"] == max_steps + 1).all()

        td_reset = TensorDict(
            rand_reset(env), batch_size=env.batch_size, device=env.device
        )
        td_reset.update(td.get("next").exclude("reward"))
        reset = td_reset["_reset"]
        td_reset = env.reset(td_reset)
        env.close()

        assert (td_reset["done"][reset] == 0).all()
        assert (td_reset["observation"][reset] == 0).all()
        assert (td_reset["done"][~reset] == 1).all()
        assert (td_reset["observation"][~reset] == max_steps + 1).all()

    @pytest.mark.parametrize("nested_obs_action", [True, False])
    @pytest.mark.parametrize("nested_done", [True, False])
    @pytest.mark.parametrize("nested_reward", [True, False])
    @pytest.mark.parametrize("env_type", ["serial", "parallel"])
    def test_parallel_env_nested(
        self,
        nested_obs_action,
        nested_done,
        nested_reward,
        env_type,
        maybe_fork_ParallelEnv,
        n_envs=2,
        batch_size=(32,),
        nested_dim=5,
        rollout_length=3,
        seed=1,
    ):
        env_fn = lambda: NestedCountingEnv(
            nest_done=nested_done,
            nest_reward=nested_reward,
            nest_obs_action=nested_obs_action,
            batch_size=batch_size,
            nested_dim=nested_dim,
        )

        if env_type == "serial":
            env = SerialEnv(n_envs, env_fn)
        else:
            env = maybe_fork_ParallelEnv(n_envs, env_fn)

        try:
            env.set_seed(seed)

            batch_size = (n_envs, *batch_size)

            td = env.reset()
            assert td.batch_size == batch_size
            if nested_done or nested_obs_action:
                assert td["data"].batch_size == (*batch_size, nested_dim)
            if not nested_done and not nested_reward and not nested_obs_action:
                assert "data" not in td.keys()

            policy = CountingEnvCountPolicy(env.action_spec, env.action_key)
            td = env.rollout(rollout_length, policy)
            assert td.batch_size == (*batch_size, rollout_length)
            if nested_done or nested_obs_action:
                assert td["data"].batch_size == (
                    *batch_size,
                    rollout_length,
                    nested_dim,
                )
            if nested_reward or nested_done or nested_obs_action:
                assert td["next", "data"].batch_size == (
                    *batch_size,
                    rollout_length,
                    nested_dim,
                )
            if not nested_done and not nested_reward and not nested_obs_action:
                assert "data" not in td.keys()
                assert "data" not in td["next"].keys()

            if nested_obs_action:
                assert "observation" not in td.keys()
                assert (td[..., -1]["data", "states"] == 2).all()
            else:
                assert ("data", "states") not in td.keys(True, True)
                assert (td[..., -1]["observation"] == 2).all()
        finally:
            try:
                env.close()
                del env
            except Exception:
                pass


@pytest.mark.parametrize("batch_size", [(), (2,), (32, 5)])
def test_env_base_reset_flag(batch_size, max_steps=3):
    torch.manual_seed(0)
    env = CountingEnv(max_steps=max_steps, batch_size=batch_size)
    env.set_seed(1)

    action = env.action_spec.rand()
    action[:] = 1

    for i in range(max_steps):
        td = env.step(
            TensorDict({"action": action}, batch_size=env.batch_size, device=env.device)
        )
        assert (td["next", "done"] == 0).all()
        assert (td["next", "observation"] == i + 1).all()

    td = env.step(
        TensorDict({"action": action}, batch_size=env.batch_size, device=env.device)
    )
    assert (td["next", "done"] == 1).all()
    assert (td["next", "observation"] == max_steps + 1).all()

    td_reset = TensorDict(rand_reset(env), batch_size=env.batch_size, device=env.device)
    td_reset.update(td.get("next").exclude("reward"))
    reset = td_reset["_reset"]
    td_reset = env.reset(td_reset)

    assert (td_reset["done"][reset] == 0).all()
    assert (td_reset["observation"][reset] == 0).all()
    assert (td_reset["done"][~reset] == 1).all()
    assert (td_reset["observation"][~reset] == max_steps + 1).all()


@pytest.mark.skipif(not _has_gym, reason="no gym")
def test_seed():
    torch.manual_seed(0)
    env1 = GymEnv(PENDULUM_VERSIONED())
    env1.set_seed(0)
    state0_1 = env1.reset()
    state1_1 = env1.step(state0_1.set("action", env1.action_spec.rand()))

    torch.manual_seed(0)
    env2 = GymEnv(PENDULUM_VERSIONED())
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

    torch.testing.assert_close(
        rollout1["observation"][1:], rollout1[("next", "observation")][:-1]
    )
    torch.testing.assert_close(
        rollout2["observation"][1:], rollout2[("next", "observation")][:-1]
    )
    torch.testing.assert_close(rollout1["observation"], rollout2["observation"])


@pytest.mark.filterwarnings("error")
class TestStepMdp:
    @pytest.mark.parametrize("keep_other", [True, False])
    @pytest.mark.parametrize("exclude_reward", [True, False])
    @pytest.mark.parametrize("exclude_done", [True, False])
    @pytest.mark.parametrize("exclude_action", [True, False])
    @pytest.mark.parametrize("has_out", [True, False])
    @pytest.mark.parametrize("lazy_stack", [False, True])
    def test_steptensordict(
        self,
        keep_other,
        exclude_reward,
        exclude_done,
        exclude_action,
        has_out,
        lazy_stack,
    ):
        torch.manual_seed(0)
        tensordict = TensorDict(
            {
                "reward": torch.randn(4, 1),
                "done": torch.zeros(4, 1, dtype=torch.bool),
                "ledzep": torch.randn(4, 2),
                "next": {
                    "ledzep": torch.randn(4, 2),
                    "reward": torch.randn(4, 1),
                    "done": torch.zeros(4, 1, dtype=torch.bool),
                },
                "beatles": torch.randn(4, 1),
                "action": torch.randn(4, 2),
            },
            [4],
        )
        if lazy_stack:
            # let's spice this a little bit
            tds = tensordict.unbind(0)
            tds[0]["this", "one"] = torch.zeros(2)
            tds[1]["but", "not", "this", "one"] = torch.ones(2)
            tds[0]["next", "this", "one"] = torch.ones(2) * 2
            tensordict = LazyStackedTensorDict.lazy_stack(tds, 0)
        next_tensordict = TensorDict({}, [4]) if has_out else None
        if has_out and lazy_stack:
            next_tensordict = LazyStackedTensorDict.lazy_stack(
                next_tensordict.unbind(0), 0
            )
        out = step_mdp(
            tensordict.lock_(),
            keep_other=keep_other,
            exclude_reward=exclude_reward,
            exclude_done=exclude_done,
            exclude_action=exclude_action,
            next_tensordict=next_tensordict,
        )
        assert "ledzep" in out.keys()
        if lazy_stack:
            assert (out["ledzep"] == tensordict["next", "ledzep"]).all()
            assert (out[0]["this", "one"] == 2).all()
            if keep_other:
                assert (out[1]["but", "not", "this", "one"] == 1).all()
        else:
            assert out["ledzep"] is tensordict["next", "ledzep"]
        if keep_other:
            assert "beatles" in out.keys()
            if lazy_stack:
                assert (out["beatles"] == tensordict["beatles"]).all()
            else:
                assert out["beatles"] is tensordict["beatles"]
        else:
            assert "beatles" not in out.keys()
        if not exclude_reward:
            assert "reward" in out.keys()
            if lazy_stack:
                assert (out["reward"] == tensordict["next", "reward"]).all()
            else:
                assert out["reward"] is tensordict["next", "reward"]
        else:
            assert "reward" not in out.keys()
        if not exclude_action:
            assert "action" in out.keys()
            if lazy_stack:
                assert (out["action"] == tensordict["action"]).all()
            else:
                assert out["action"] is tensordict["action"]
        else:
            assert "action" not in out.keys()
        if not exclude_done:
            assert "done" in out.keys()
            if lazy_stack:
                assert (out["done"] == tensordict["next", "done"]).all()
            else:
                assert out["done"] is tensordict["next", "done"]
        else:
            assert "done" not in out.keys()
        if has_out:
            assert out is next_tensordict

    @pytest.mark.parametrize("keep_other", [True, False])
    @pytest.mark.parametrize("exclude_reward", [True, False])
    @pytest.mark.parametrize("exclude_done", [False, True])
    @pytest.mark.parametrize("exclude_action", [False, True])
    @pytest.mark.parametrize(
        "envcls",
        [
            ContinuousActionVecMockEnv,
            CountingBatchedEnv,
            CountingEnv,
            NestedCountingEnv,
            CountingBatchedEnv,
            HeterogeneousCountingEnv,
            DiscreteActionConvMockEnv,
        ],
    )
    def test_step_class(
        self,
        envcls,
        keep_other,
        exclude_reward,
        exclude_done,
        exclude_action,
    ):
        torch.manual_seed(0)
        env = envcls()

        tensordict = env.rand_step(env.reset())
        out_func = step_mdp(
            tensordict.lock_(),
            keep_other=keep_other,
            exclude_reward=exclude_reward,
            exclude_done=exclude_done,
            exclude_action=exclude_action,
            done_keys=env.done_keys,
            action_keys=env.action_keys,
            reward_keys=env.reward_keys,
        )
        step_func = _StepMDP(
            env,
            keep_other=keep_other,
            exclude_reward=exclude_reward,
            exclude_done=exclude_done,
            exclude_action=exclude_action,
        )
        out_cls = step_func(tensordict)
        assert (out_func == out_cls).all()

    @pytest.mark.parametrize("nested_obs", [True, False])
    @pytest.mark.parametrize("nested_action", [True, False])
    @pytest.mark.parametrize("nested_done", [True, False])
    @pytest.mark.parametrize("nested_reward", [True, False])
    @pytest.mark.parametrize("nested_other", [True, False])
    @pytest.mark.parametrize("exclude_reward", [True, False])
    @pytest.mark.parametrize("exclude_done", [True, False])
    @pytest.mark.parametrize("exclude_action", [True, False])
    @pytest.mark.parametrize("keep_other", [True, False])
    def test_nested(
        self,
        nested_obs,
        nested_action,
        nested_done,
        nested_reward,
        nested_other,
        exclude_reward,
        exclude_done,
        exclude_action,
        keep_other,
    ):
        td_batch_size = (4,)
        nested_batch_size = (4, 3)
        nested_key = ("data",)
        td = TensorDict(
            {
                nested_key: TensorDict({}, nested_batch_size),
                "next": {
                    nested_key: TensorDict({}, nested_batch_size),
                },
            },
            td_batch_size,
        )
        reward_key = "reward"
        if nested_reward:
            reward_key = nested_key + (reward_key,)
        done_key = "done"
        if nested_done:
            done_key = nested_key + (done_key,)
        action_key = "action"
        if nested_action:
            action_key = nested_key + (action_key,)
        obs_key = "state"
        if nested_obs:
            obs_key = nested_key + (obs_key,)
        other_key = "other"
        if nested_other:
            other_key = nested_key + (other_key,)

        td[reward_key] = torch.zeros(*nested_batch_size, 1)
        td[done_key] = torch.zeros(*nested_batch_size, 1)
        td[obs_key] = torch.zeros(*nested_batch_size, 1)
        td[action_key] = torch.zeros(*nested_batch_size, 1)
        td[other_key] = torch.zeros(*nested_batch_size, 1)

        td["next", reward_key] = torch.ones(*nested_batch_size, 1)
        td["next", done_key] = torch.ones(*nested_batch_size, 1)
        td["next", obs_key] = torch.ones(*nested_batch_size, 1)

        input_td = td

        td = step_mdp(
            td.lock_(),
            exclude_reward=exclude_reward,
            exclude_done=exclude_done,
            exclude_action=exclude_action,
            reward_keys=reward_key,
            done_keys=done_key,
            action_keys=action_key,
            keep_other=keep_other,
        )
        td_nested_keys = td.keys(True, True)
        td_keys = td.keys()

        assert td.batch_size == input_td.batch_size
        # Obs will always be present
        assert obs_key in td_nested_keys
        # Nested key should not be present in this specific conditions
        if (
            (exclude_done or not nested_done)
            and (exclude_reward or not nested_reward)
            and (exclude_action or not nested_action)
            and not nested_obs
            and ((not keep_other) or (keep_other and not nested_other))
        ):
            assert nested_key[0] not in td_keys
        else:  # Nested key is present
            assert not td[nested_key] is input_td["next", nested_key]
            assert not td[nested_key] is input_td[nested_key]
            assert td[nested_key].batch_size == nested_batch_size
        # If we exclude everything we are left with just obs
        if exclude_done and exclude_reward and exclude_action and not keep_other:
            if nested_obs:
                assert len(td_nested_keys) == 1 and list(td_nested_keys)[0] == obs_key
            else:
                assert len(td_nested_keys) == 1 and list(td_nested_keys)[0] == obs_key
        # Key-wise exclusions
        if not exclude_reward:
            assert reward_key in td_nested_keys
            assert (td[reward_key] == 1).all()
        else:
            assert reward_key not in td_nested_keys
        if not exclude_action:
            assert action_key in td_nested_keys
            assert (td[action_key] == 0).all()
        else:
            assert action_key not in td_nested_keys
        if not exclude_done:
            assert done_key in td_nested_keys
            assert (td[done_key] == 1).all()
        else:
            assert done_key not in td_nested_keys
        if keep_other:
            assert other_key in td_nested_keys, other_key
            assert (td[other_key] == 0).all()
        else:
            assert other_key not in td_nested_keys

    @pytest.mark.parametrize("nested_other", [True, False])
    @pytest.mark.parametrize("exclude_reward", [True, False])
    @pytest.mark.parametrize("exclude_done", [True, False])
    @pytest.mark.parametrize("exclude_action", [True, False])
    @pytest.mark.parametrize("keep_other", [True, False])
    def test_nested_partially(
        self,
        nested_other,
        exclude_reward,
        exclude_done,
        exclude_action,
        keep_other,
    ):
        # General
        td_batch_size = (4,)
        nested_batch_size = (4, 3)
        nested_key = ("data",)
        reward_key = "reward"
        done_key = "done"
        action_key = "action"
        obs_key = "state"
        other_key = "beatles"
        if nested_other:
            other_key = nested_key + (other_key,)

        # Nested only in root
        td = TensorDict(
            {
                nested_key: TensorDict({}, nested_batch_size),
                "next": {},
            },
            td_batch_size,
        )

        td[reward_key] = torch.zeros(*nested_batch_size, 1)
        td[done_key] = torch.zeros(*nested_batch_size, 1)
        td[obs_key] = torch.zeros(*nested_batch_size, 1)
        td[action_key] = torch.zeros(*nested_batch_size, 1)
        td[other_key] = torch.zeros(*nested_batch_size, 1)

        td["next", reward_key] = torch.zeros(*nested_batch_size, 1)
        td["next", done_key] = torch.zeros(*nested_batch_size, 1)
        td["next", obs_key] = torch.zeros(*nested_batch_size, 1)

        td = step_mdp(
            td.lock_(),
            exclude_reward=exclude_reward,
            exclude_done=exclude_done,
            exclude_action=exclude_action,
            reward_keys=reward_key,
            done_keys=done_key,
            action_keys=action_key,
            keep_other=keep_other,
        )
        td_keys_nested = td.keys(True, True)
        td_keys = td.keys()
        if keep_other:
            if nested_other:
                assert nested_key[0] in td_keys
                assert td[nested_key].batch_size == nested_batch_size
            else:
                assert nested_key[0] not in td_keys
            assert (td[other_key] == 0).all()
        else:
            assert other_key not in td_keys_nested

        # Nested only in next
        td = TensorDict(
            {
                "next": {nested_key: TensorDict({}, nested_batch_size)},
            },
            td_batch_size,
        )
        td[reward_key] = torch.zeros(*nested_batch_size, 1)
        td[done_key] = torch.zeros(*nested_batch_size, 1)
        td[obs_key] = torch.zeros(*nested_batch_size, 1)
        td[action_key] = torch.zeros(*nested_batch_size, 1)

        td["next", other_key] = torch.zeros(*nested_batch_size, 1)
        td["next", reward_key] = torch.zeros(*nested_batch_size, 1)
        td["next", done_key] = torch.zeros(*nested_batch_size, 1)
        td["next", obs_key] = torch.zeros(*nested_batch_size, 1)

        td = step_mdp(
            td.lock_(),
            exclude_reward=exclude_reward,
            exclude_done=exclude_done,
            exclude_action=exclude_action,
            reward_keys=reward_key,
            done_keys=done_key,
            action_keys=action_key,
            keep_other=keep_other,
        )
        td_keys = td.keys()

        if nested_other:
            assert nested_key[0] in td_keys
            assert td[nested_key].batch_size == nested_batch_size
        else:
            assert nested_key[0] not in td_keys
        assert (td[other_key] == 0).all()

    @pytest.mark.parametrize("het_action", [True, False])
    @pytest.mark.parametrize("het_done", [True, False])
    @pytest.mark.parametrize("het_reward", [True, False])
    @pytest.mark.parametrize("het_other", [True, False])
    @pytest.mark.parametrize("het_obs", [True, False])
    @pytest.mark.parametrize("exclude_reward", [True, False])
    @pytest.mark.parametrize("exclude_done", [True, False])
    @pytest.mark.parametrize("exclude_action", [True, False])
    @pytest.mark.parametrize("keep_other", [True, False])
    def test_heterogeenous(
        self,
        het_action,
        het_done,
        het_reward,
        het_other,
        het_obs,
        exclude_reward,
        exclude_done,
        exclude_action,
        keep_other,
    ):
        td_batch_size = 4
        nested_dim = 3
        nested_batch_size = (td_batch_size, nested_dim)
        nested_key = ("data",)

        reward_key = "reward"
        nested_reward_key = nested_key + (reward_key,)
        done_key = "done"
        nested_done_key = nested_key + (done_key,)
        action_key = "action"
        nested_action_key = nested_key + (action_key,)
        obs_key = "state"
        nested_obs_key = nested_key + (obs_key,)
        other_key = "beatles"
        nested_other_key = nested_key + (other_key,)

        tds = []
        for i in range(1, nested_dim + 1):
            tds.append(
                TensorDict(
                    {
                        nested_key: TensorDict(
                            {
                                reward_key: torch.zeros(
                                    td_batch_size, i if het_reward else 1
                                ),
                                done_key: torch.zeros(
                                    td_batch_size, i if het_done else 1
                                ),
                                action_key: torch.zeros(
                                    td_batch_size, i if het_action else 1
                                ),
                                obs_key: torch.zeros(
                                    td_batch_size, i if het_obs else 1
                                ),
                                other_key: torch.zeros(
                                    td_batch_size, i if het_other else 1
                                ),
                            },
                            [td_batch_size],
                        ),
                        "next": {
                            nested_key: TensorDict(
                                {
                                    reward_key: torch.ones(
                                        td_batch_size, i if het_reward else 1
                                    ),
                                    done_key: torch.ones(
                                        td_batch_size, i if het_done else 1
                                    ),
                                    obs_key: torch.ones(
                                        td_batch_size, i if het_obs else 1
                                    ),
                                },
                                [td_batch_size],
                            ),
                        },
                    },
                    [td_batch_size],
                )
            )
        lazy_td = LazyStackedTensorDict.lazy_stack(tds, dim=1)

        td = step_mdp(
            lazy_td.lock_(),
            exclude_reward=exclude_reward,
            exclude_done=exclude_done,
            exclude_action=exclude_action,
            reward_keys=nested_reward_key,
            done_keys=nested_done_key,
            action_keys=nested_action_key,
            keep_other=keep_other,
        )
        td_nested_keys = td.keys(True, True)
        td_keys = td.keys()
        for i in range(nested_dim):
            if het_obs:
                assert td[..., i][nested_obs_key].shape == (td_batch_size, i + 1)
            else:
                assert td[..., i][nested_obs_key].shape == (td_batch_size, 1)
            assert (td[..., i][nested_obs_key] == 1).all()
        if exclude_reward:
            assert nested_reward_key not in td_keys
        else:
            for i in range(nested_dim):
                if het_reward:
                    assert td[..., i][nested_reward_key].shape == (td_batch_size, i + 1)
                else:
                    assert td[..., i][nested_reward_key].shape == (td_batch_size, 1)
                assert (td[..., i][nested_reward_key] == 1).all()
        if exclude_done:
            assert nested_done_key not in td_keys
        else:
            for i in range(nested_dim):
                if het_done:
                    assert td[..., i][nested_done_key].shape == (td_batch_size, i + 1)
                else:
                    assert td[..., i][nested_done_key].shape == (td_batch_size, 1)
                assert (td[..., i][nested_done_key] == 1).all()
        if exclude_action:
            assert nested_action_key not in td_keys
        else:
            for i in range(nested_dim):
                if het_action:
                    assert td[..., i][nested_action_key].shape == (td_batch_size, i + 1)
                else:
                    assert td[..., i][nested_action_key].shape == (td_batch_size, 1)
                assert (td[..., i][nested_action_key] == 0).all()
        if not keep_other:
            assert nested_other_key not in td_keys
        else:
            for i in range(nested_dim):
                if het_other:
                    assert td[..., i][nested_other_key].shape == (td_batch_size, i + 1)
                else:
                    assert td[..., i][nested_other_key].shape == (td_batch_size, 1)
                assert (td[..., i][nested_other_key] == 0).all()

    @pytest.mark.parametrize("serial", [False, True])
    def test_multi_purpose_env(self, serial):
        # Tests that even if it's validated, the same env can be used within a collector
        # and independently of it.
        if serial:
            env = SerialEnv(2, ContinuousActionVecMockEnv)
        else:
            env = ContinuousActionVecMockEnv()
        env.rollout(10)
        assert env._step_mdp.validate(None)
        c = SyncDataCollector(
            env, env.rand_action, frames_per_batch=10, total_frames=20
        )
        for data in c:  # noqa: B007
            pass
        assert ("collector", "traj_ids") in data.keys(True)
        assert env._step_mdp.validate(None)
        env.rollout(10)

        # An exception will be raised when the collector sees extra keys
        if serial:
            env = SerialEnv(2, ContinuousActionVecMockEnv)
        else:
            env = ContinuousActionVecMockEnv()
        c = SyncDataCollector(
            env, env.rand_action, frames_per_batch=10, total_frames=20
        )
        for data in c:  # noqa: B007
            pass


@pytest.mark.parametrize("device", get_default_devices())
def test_batch_locked(device):
    env = MockBatchedLockedEnv(device)
    assert env.batch_locked

    with pytest.raises(RuntimeError, match="batch_locked is a read-only property"):
        env.batch_locked = False
    td = env.reset()
    td["action"] = env.action_spec.rand()
    td_expanded = td.expand(2).clone()
    _ = env.step(td)

    with pytest.raises(
        RuntimeError, match="Expected a tensordict with shape==env.batch_size, "
    ):
        env.step(td_expanded)


@pytest.mark.parametrize("device", get_default_devices())
def test_batch_unlocked(device):
    env = MockBatchedUnLockedEnv(device)
    assert not env.batch_locked

    with pytest.raises(RuntimeError, match="batch_locked is a read-only property"):
        env.batch_locked = False
    td = env.reset()
    td["action"] = env.action_spec.rand()
    td_expanded = td.expand(2).clone()
    td = env.step(td)

    env.step(td_expanded)


@pytest.mark.parametrize("device", get_default_devices())
def test_batch_unlocked_with_batch_size(device):
    env = MockBatchedUnLockedEnv(device, batch_size=torch.Size([2]))
    assert not env.batch_locked

    with pytest.raises(RuntimeError, match="batch_locked is a read-only property"):
        env.batch_locked = False

    td = env.reset()
    td["action"] = env.action_spec.rand()
    td_expanded = td.expand(2, 2).reshape(-1).to_tensordict()
    td = env.step(td)

    with pytest.raises(RuntimeError, match="Expected a tensordict with shape"):
        env.step(td_expanded)


class TestInfoDict:
    @pytest.mark.skipif(not _has_gym, reason="no gym")
    @pytest.mark.skipif(
        gym_version is None or gym_version < version.parse("0.20.0"),
        reason="older versions of half-cheetah do not have 'x_position' info key.",
    )
    @pytest.mark.parametrize("device", get_default_devices())
    def test_info_dict_reader(self, device, seed=0):
        try:
            import gymnasium as gym
        except ModuleNotFoundError:
            import gym

        env = GymWrapper(gym.make(HALFCHEETAH_VERSIONED()), device=device)
        env.set_info_dict_reader(
            default_info_dict_reader(
                ["x_position"],
                spec=CompositeSpec(
                    x_position=UnboundedContinuousTensorSpec(
                        dtype=torch.float64, shape=()
                    )
                ),
            )
        )

        assert "x_position" in env.observation_spec.keys()
        assert isinstance(
            env.observation_spec["x_position"], UnboundedContinuousTensorSpec
        )

        tensordict = env.reset()
        tensordict = env.rand_step(tensordict)

        x_position_data = tensordict["next", "x_position"]
        assert env.observation_spec["x_position"].is_in(x_position_data), (
            x_position_data.shape,
            x_position_data.dtype,
            env.observation_spec["x_position"],
        )

        for spec in (
            {"x_position": UnboundedContinuousTensorSpec((), dtype=torch.float64)},
            # None,
            CompositeSpec(
                x_position=UnboundedContinuousTensorSpec((), dtype=torch.float64),
                shape=[],
            ),
            [UnboundedContinuousTensorSpec((), dtype=torch.float64)],
        ):
            env2 = GymWrapper(gym.make("HalfCheetah-v4"))
            env2.set_info_dict_reader(
                default_info_dict_reader(["x_position"], spec=spec)
            )

            tensordict2 = env2.reset()
            tensordict2 = env2.rand_step(tensordict2)
            data = tensordict2[("next", "x_position")]
            assert env2.observation_spec["x_position"].is_in(data), (
                data.dtype,
                data.device,
                data.shape,
                env2.observation_spec["x_position"],
            )

    @pytest.mark.skipif(not _has_gym, reason="no gym")
    @pytest.mark.skipif(
        gym_version is None or gym_version < version.parse("0.20.0"),
        reason="older versions of half-cheetah do not have 'x_position' info key.",
    )
    @pytest.mark.parametrize("device", get_default_devices())
    def test_auto_register(self, device, maybe_fork_ParallelEnv):
        try:
            import gymnasium as gym
        except ModuleNotFoundError:
            import gym

        # env = GymWrapper(gym.make(HALFCHEETAH_VERSIONED()), device=device)
        # check_env_specs(env)
        # env.set_info_dict_reader()
        # with pytest.raises(
        #     AssertionError, match="The keys of the specs and data do not match"
        # ):
        #     check_env_specs(env)

        env = GymWrapper(gym.make(HALFCHEETAH_VERSIONED()), device=device)
        env = env.auto_register_info_dict()
        check_env_specs(env)

        # check that the env can be executed in parallel
        penv = maybe_fork_ParallelEnv(
            2,
            lambda: GymWrapper(
                gym.make(HALFCHEETAH_VERSIONED()), device=device
            ).auto_register_info_dict(),
        )
        senv = maybe_fork_ParallelEnv(
            2,
            lambda: GymWrapper(
                gym.make(HALFCHEETAH_VERSIONED()), device=device
            ).auto_register_info_dict(),
        )
        try:
            torch.manual_seed(0)
            penv.set_seed(0)
            rolp = penv.rollout(10)
            torch.manual_seed(0)
            senv.set_seed(0)
            rols = senv.rollout(10)
            assert_allclose_td(rolp, rols)
        finally:
            penv.close()
            del penv
            senv.close()
            del senv


def test_make_spec_from_td():
    data = TensorDict(
        {
            "obs": torch.randn(3),
            "action": torch.zeros(2, dtype=torch.int),
            "next": {
                "obs": torch.randn(3),
                "reward": torch.randn(1),
                "done": torch.zeros(1, dtype=torch.bool),
            },
        },
        [],
    )
    spec = make_composite_from_td(data)
    assert (spec.zero() == data.zero_()).all()
    for key, val in data.items(True, True):
        assert val.dtype is spec[key].dtype


@pytest.mark.parametrize("group_type", list(MarlGroupMapType))
def test_marl_group_type(group_type):
    agent_names = ["agent"]
    check_marl_grouping(group_type.get_group_map(agent_names), agent_names)

    agent_names = ["agent", "agent"]
    with pytest.raises(ValueError):
        check_marl_grouping(group_type.get_group_map(agent_names), agent_names)

    agent_names = ["agent_0", "agent_1"]
    check_marl_grouping(group_type.get_group_map(agent_names), agent_names)

    agent_names = []
    with pytest.raises(ValueError):
        check_marl_grouping(group_type.get_group_map(agent_names), agent_names)


@pytest.mark.skipif(not torch.cuda.device_count(), reason="No cuda device")
class TestConcurrentEnvs:
    """Concurrent parallel envs on multiple procs can interfere."""

    class Policy(TensorDictModuleBase):
        in_keys = []
        out_keys = ["action"]

        def __init__(self, spec):
            super().__init__()
            self.spec = spec

        def forward(self, tensordict):
            tensordict.set("action", self.spec["action"].zero() + 1)
            return tensordict

    @staticmethod
    def main_penv(j, q=None):
        device = "cpu" if not torch.cuda.device_count() else "cuda:0"
        n_workers = 1
        env_p = ParallelEnv(
            n_workers,
            [
                lambda i=i: CountingEnv(i, device=device)
                for i in range(j, j + n_workers)
            ],
        )
        env_s = SerialEnv(
            n_workers,
            [
                lambda i=i: CountingEnv(i, device=device)
                for i in range(j, j + n_workers)
            ],
        )
        spec = env_p.action_spec
        policy = TestConcurrentEnvs.Policy(CompositeSpec(action=spec.to(device)))
        N = 10
        r_p = []
        r_s = []
        for _ in range(N):
            with torch.no_grad():
                r_p.append(env_s.rollout(100, break_when_any_done=False, policy=policy))
                r_s.append(env_p.rollout(100, break_when_any_done=False, policy=policy))

        td_equals = torch.stack(r_p) == torch.stack(r_s)
        if td_equals.all():
            if q is not None:
                q.put(("passed", j))
            else:
                pass
        else:
            if q is not None:
                s = ""
                for key, item in td_equals.items(True, True):
                    if not item.all():
                        s = s + f"\t{key}"
                q.put((f"failed: {s}", j))
            else:
                raise RuntimeError()

    @staticmethod
    def main_collector(j, q=None):
        device = "cpu" if not torch.cuda.device_count() else "cuda:0"
        N = 10
        n_workers = 1
        make_envs = [
            lambda i=i: CountingEnv(i, device=device) for i in range(j, j + n_workers)
        ]
        spec = make_envs[0]().action_spec
        policy = TestConcurrentEnvs.Policy(CompositeSpec(action=spec))
        collector = MultiSyncDataCollector(
            make_envs,
            policy,
            frames_per_batch=n_workers * 100,
            total_frames=N * n_workers * 100,
            storing_device=device,
            device=device,
        )
        single_collectors = [
            SyncDataCollector(
                make_envs[i](),
                policy,
                frames_per_batch=n_workers * 100,
                total_frames=N * n_workers * 100,
                storing_device=device,
                device=device,
            )
            for i in range(n_workers)
        ]
        iter_collector = iter(collector)
        iter_single_collectors = [iter(sc) for sc in single_collectors]

        r_p = []
        r_s = []
        for _ in range(N):
            with torch.no_grad():
                r_p.append(next(iter_collector).clone())
                r_s.append(torch.cat([next(sc) for sc in iter_single_collectors]))

        collector.shutdown()
        for sc in single_collectors:
            sc.shutdown()
        del collector
        del single_collectors
        r_p = torch.stack(r_p).contiguous()
        r_s = torch.stack(r_s).contiguous()
        td_equals = r_p == r_s

        if td_equals.all():
            if q is not None:
                q.put(("passed", j))
            else:
                pass
        else:
            if q is not None:
                s = ""
                for key, item in td_equals.items(True, True):
                    if not item.all():
                        s = s + f"\t{key}"
                q.put((f"failed: {s}", j))
            else:
                raise RuntimeError()

    @pytest.mark.parametrize("nproc", [3, 1])
    def test_mp_concurrent(self, nproc):
        if nproc == 1:
            self.main_penv(3)
            self.main_penv(6)
            self.main_penv(9)
        else:
            from torch import multiprocessing as mp

            q = mp.Queue(3)
            ps = []
            try:
                for k in range(3, 10, 3):
                    p = mp.Process(target=type(self).main_penv, args=(k, q))
                    ps.append(p)
                    p.start()
                for _ in range(3):
                    msg, j = q.get(timeout=100)
                    assert msg == "passed", j
            finally:
                for p in ps:
                    p.join()

    @pytest.mark.parametrize("nproc", [3, 1])
    def test_mp_collector(self, nproc):
        if nproc == 1:
            self.main_collector(3)
            self.main_collector(6)
            self.main_collector(9)
        else:
            from torch import multiprocessing as mp

            q = mp.Queue(3)
            ps = []
            try:
                for j in range(3, 10, 3):
                    p = mp.Process(target=type(self).main_collector, args=(j, q))
                    ps.append(p)
                    p.start()
                for _ in range(3):
                    msg, j = q.get(timeout=100)
                    assert msg == "passed", j
            finally:
                for p in ps:
                    p.join(timeout=2)


class TestNestedSpecs:
    @pytest.mark.parametrize("envclass", ["CountingEnv", "NestedCountingEnv"])
    def test_nested_env(self, envclass):
        if envclass == "CountingEnv":
            env = CountingEnv()
        elif envclass == "NestedCountingEnv":
            env = NestedCountingEnv()
        else:
            raise NotImplementedError
        reset = env.reset()
        assert not isinstance(env.reward_spec, CompositeSpec)
        for done_key in env.done_keys:
            assert (
                env.full_done_spec[done_key]
                == env.output_spec[("full_done_spec", *_unravel_key_to_tuple(done_key))]
            )
        assert (
            env.reward_spec
            == env.output_spec[
                ("full_reward_spec", *_unravel_key_to_tuple(env.reward_key))
            ]
        )
        if envclass == "NestedCountingEnv":
            for done_key in env.done_keys:
                assert done_key in (("data", "done"), ("data", "terminated"))
            assert env.reward_key == ("data", "reward")
            assert ("data", "done") in reset.keys(True)
            assert ("data", "states") in reset.keys(True)
            assert ("data", "reward") not in reset.keys(True)
        for done_key in env.done_keys:
            assert done_key in reset.keys(True)
        assert env.reward_key not in reset.keys(True)

        next_state = env.rand_step()
        if envclass == "NestedCountingEnv":
            assert ("next", "data", "done") in next_state.keys(True)
            assert ("next", "data", "states") in next_state.keys(True)
            assert ("next", "data", "reward") in next_state.keys(True)
        for done_key in env.done_keys:
            assert ("next", *_unravel_key_to_tuple(done_key)) in next_state.keys(True)
        assert ("next", *_unravel_key_to_tuple(env.reward_key)) in next_state.keys(True)

    @pytest.mark.parametrize("batch_size", [(), (32,), (32, 1)])
    def test_nested_env_dims(self, batch_size, nested_dim=5, rollout_length=3):
        env = NestedCountingEnv(batch_size=batch_size, nested_dim=nested_dim)

        td_reset = env.reset()
        assert td_reset.batch_size == batch_size
        assert td_reset["data"].batch_size == (*batch_size, nested_dim)

        td = env.rand_action()
        assert td.batch_size == batch_size
        assert td["data"].batch_size == (*batch_size, nested_dim)

        td = env.rand_action(td_reset)
        assert td.batch_size == batch_size
        assert td["data"].batch_size == (*batch_size, nested_dim)

        td = env.rand_step(td)
        assert td.batch_size == batch_size
        assert td["data"].batch_size == (*batch_size, nested_dim)
        assert td["next", "data"].batch_size == (*batch_size, nested_dim)

        td = env.rand_step()
        assert td.batch_size == batch_size
        assert td["data"].batch_size == (*batch_size, nested_dim)
        assert td["next", "data"].batch_size == (*batch_size, nested_dim)

        td = env.rand_step(td_reset)
        assert td.batch_size == batch_size
        assert td["data"].batch_size == (*batch_size, nested_dim)
        assert td["next", "data"].batch_size == (*batch_size, nested_dim)

        td = env.rollout(rollout_length)
        assert td.batch_size == (*batch_size, rollout_length)
        assert td["data"].batch_size == (*batch_size, rollout_length, nested_dim)
        assert td["next", "data"].batch_size == (
            *batch_size,
            rollout_length,
            nested_dim,
        )

        policy = CountingEnvCountPolicy(env.action_spec, env.action_key)
        td = env.rollout(rollout_length, policy)
        assert td.batch_size == (*batch_size, rollout_length)
        assert td["data"].batch_size == (*batch_size, rollout_length, nested_dim)
        assert td["next", "data"].batch_size == (
            *batch_size,
            rollout_length,
            nested_dim,
        )

    @pytest.mark.parametrize("batch_size", [(), (32,), (32, 1)])
    @pytest.mark.parametrize(
        "nest_done,has_root_done", [[False, False], [True, False], [True, True]]
    )
    def test_nested_reset(self, nest_done, has_root_done, batch_size):
        env = NestedCountingEnv(
            nest_done=nest_done, has_root_done=has_root_done, batch_size=batch_size
        )
        for reset_key, done_keys in zip(env.reset_keys, env.done_keys_groups):
            if isinstance(reset_key, str):
                for done_key in done_keys:
                    assert isinstance(done_key, str)
            else:
                for done_key in done_keys:
                    assert done_key[:-1] == reset_key[:-1]
        env.rollout(100)
        env.rollout(100, break_when_any_done=False)


class TestHeteroEnvs:
    @pytest.mark.parametrize("batch_size", [(), (32,), (1, 2)])
    def test_reset(self, batch_size):
        env = HeterogeneousCountingEnv(batch_size=batch_size)
        env.reset()

    @pytest.mark.parametrize("batch_size", [(), (32,), (1, 2)])
    def test_rand_step(self, batch_size):
        env = HeterogeneousCountingEnv(batch_size=batch_size)
        td = env.reset()
        assert (td["lazy"][..., 0]["tensor_0"] == 0).all()
        td = env.rand_step()
        assert (td["next", "lazy"][..., 0]["tensor_0"] == 1).all()
        td = env.rand_step()
        assert (td["next", "lazy"][..., 1]["tensor_1"] == 2).all()

    @pytest.mark.parametrize("batch_size", [(), (2,), (2, 1)])
    @pytest.mark.parametrize("rollout_steps", [1, 2, 5])
    def test_rollout(self, batch_size, rollout_steps, n_lazy_dim=3):
        env = HeterogeneousCountingEnv(batch_size=batch_size)
        td = env.rollout(rollout_steps, return_contiguous=False)
        td = dense_stack_tds(td)

        assert isinstance(td, TensorDict)
        assert td.batch_size == (*batch_size, rollout_steps)

        assert isinstance(td["lazy"], LazyStackedTensorDict)
        assert td["lazy"].shape == (*batch_size, rollout_steps, n_lazy_dim)
        assert td["lazy"].stack_dim == len(td["lazy"].batch_size) - 1

        assert (td[..., -1]["next", "state"] == rollout_steps).all()
        assert (td[..., -1]["next", "lazy", "camera"] == rollout_steps).all()
        assert (
            td["lazy"][(0,) * len(batch_size)][..., 0]["tensor_0"].squeeze(-1)
            == torch.arange(rollout_steps)
        ).all()

    @pytest.mark.parametrize("batch_size", [(), (2,), (2, 1)])
    @pytest.mark.parametrize("rollout_steps", [1, 2, 5])
    @pytest.mark.parametrize("count", [True, False])
    def test_rollout_policy(self, batch_size, rollout_steps, count):
        env = HeterogeneousCountingEnv(batch_size=batch_size)
        policy = HeterogeneousCountingEnvPolicy(
            env.input_spec["full_action_spec"], count=count
        )
        td = env.rollout(rollout_steps, policy=policy, return_contiguous=False)
        td = dense_stack_tds(td)
        for i in range(env.n_nested_dim):
            if count:
                agent_obs = td["lazy"][(0,) * len(batch_size)][..., i][f"tensor_{i}"]
                for _ in range(i + 1):
                    agent_obs = agent_obs.mean(-1)
                assert (agent_obs == torch.arange(rollout_steps)).all()
                assert (td["lazy"][..., i]["action"] == 1).all()
            else:
                assert (td["lazy"][..., i]["action"] == 0).all()

    @pytest.mark.parametrize("batch_size", [(1, 2)])
    @pytest.mark.parametrize("env_type", ["serial", "parallel"])
    @pytest.mark.parametrize("break_when_any_done", [False, True])
    def test_vec_env(
        self, batch_size, env_type, break_when_any_done, rollout_steps=4, n_workers=2
    ):
        env_fun = lambda: HeterogeneousCountingEnv(batch_size=batch_size)
        if env_type == "serial":
            vec_env = SerialEnv(n_workers, env_fun)
        else:
            vec_env = ParallelEnv(n_workers, env_fun)
        vec_batch_size = (n_workers,) + batch_size
        # check_env_specs(vec_env, return_contiguous=False)
        policy = HeterogeneousCountingEnvPolicy(vec_env.input_spec["full_action_spec"])
        vec_env.reset()
        td = vec_env.rollout(
            rollout_steps,
            policy=policy,
            return_contiguous=False,
            break_when_any_done=break_when_any_done,
        )
        td = dense_stack_tds(td)
        for i in range(env_fun().n_nested_dim):
            agent_obs = td["lazy"][(0,) * len(vec_batch_size)][..., i][f"tensor_{i}"]
            for _ in range(i + 1):
                agent_obs = agent_obs.mean(-1)
            assert (agent_obs == torch.arange(rollout_steps)).all()
            assert (td["lazy"][..., i]["action"] == 1).all()


@pytest.mark.parametrize("seed", [0])
class TestMultiKeyEnvs:
    @pytest.mark.parametrize("batch_size", [(), (2,), (2, 1)])
    @pytest.mark.parametrize("rollout_steps", [1, 5])
    @pytest.mark.parametrize("max_steps", [2, 5])
    def test_rollout(self, batch_size, rollout_steps, max_steps, seed):
        env = MultiKeyCountingEnv(batch_size=batch_size, max_steps=max_steps)
        policy = MultiKeyCountingEnvPolicy(full_action_spec=env.action_spec)
        td = env.rollout(rollout_steps, policy=policy)
        torch.manual_seed(seed)
        check_rollout_consistency_multikey_env(td, max_steps=max_steps)

    @pytest.mark.parametrize("batch_size", [(), (2,), (2, 1)])
    @pytest.mark.parametrize("rollout_steps", [5])
    @pytest.mark.parametrize("env_type", ["serial", "parallel"])
    @pytest.mark.parametrize("max_steps", [2, 5])
    def test_parallel(
        self,
        batch_size,
        rollout_steps,
        env_type,
        max_steps,
        seed,
        maybe_fork_ParallelEnv,
        n_workers=2,
    ):
        torch.manual_seed(seed)
        env_fun = lambda: MultiKeyCountingEnv(
            batch_size=batch_size, max_steps=max_steps
        )
        if env_type == "serial":
            vec_env = SerialEnv(n_workers, env_fun)
        else:
            vec_env = maybe_fork_ParallelEnv(n_workers, env_fun)

        # check_env_specs(vec_env)
        policy = MultiKeyCountingEnvPolicy(
            full_action_spec=vec_env.input_spec["full_action_spec"]
        )
        vec_env.reset()
        td = vec_env.rollout(
            rollout_steps,
            policy=policy,
        )
        check_rollout_consistency_multikey_env(td, max_steps=max_steps)


@pytest.mark.parametrize(
    "envclass",
    [
        ContinuousActionConvMockEnv,
        ContinuousActionConvMockEnvNumpy,
        ContinuousActionVecMockEnv,
        CountingBatchedEnv,
        CountingEnv,
        DiscreteActionConvMockEnv,
        DiscreteActionConvMockEnvNumpy,
        DiscreteActionVecMockEnv,
        partial(
            DummyModelBasedEnvBase, world_model=TestModelBasedEnvBase.world_model()
        ),
        MockBatchedLockedEnv,
        MockBatchedUnLockedEnv,
        MockSerialEnv,
        NestedCountingEnv,
        HeterogeneousCountingEnv,
        MultiKeyCountingEnv,
    ],
)
def test_mocking_envs(envclass):
    env = envclass()
    env.set_seed(100)
    reset = env.reset()
    _ = env.rand_step(reset)
    check_env_specs(env, seed=100, return_contiguous=False)


class TestTerminatedOrTruncated:
    @pytest.mark.parametrize("done_key", ["done", "terminated", "truncated"])
    def test_root_prevail(self, done_key):
        _spec = DiscreteTensorSpec(2, shape=(), dtype=torch.bool)
        spec = CompositeSpec({done_key: _spec, ("agent", done_key): _spec})
        data = TensorDict({done_key: [False], ("agent", done_key): [True, False]}, [])
        assert not _terminated_or_truncated(data)
        assert not _terminated_or_truncated(data, full_done_spec=spec)
        data = TensorDict({done_key: [True], ("agent", done_key): [True, False]}, [])
        assert _terminated_or_truncated(data)
        assert _terminated_or_truncated(data, full_done_spec=spec)

    def test_terminated_or_truncated_nospec(self):
        done_shape = (2, 1)
        nested_done_shape = (2, 3, 1)
        data = TensorDict(
            {"done": torch.zeros(*done_shape, dtype=torch.bool)}, done_shape[0]
        )
        assert not _terminated_or_truncated(data, write_full_false=True)
        assert data["_reset"].shape == done_shape
        assert not _terminated_or_truncated(data, write_full_false=False)
        assert data.get("_reset", None) is None

        data = TensorDict(
            {
                ("agent", "done"): torch.zeros(*nested_done_shape, dtype=torch.bool),
                ("nested", "done"): torch.ones(*nested_done_shape, dtype=torch.bool),
            },
            [done_shape[0]],
        )
        assert _terminated_or_truncated(data)
        assert data["agent", "_reset"].shape == nested_done_shape
        assert data["nested", "_reset"].shape == nested_done_shape

        data = TensorDict(
            {
                "done": torch.zeros(*done_shape, dtype=torch.bool),
                ("nested", "done"): torch.zeros(*nested_done_shape, dtype=torch.bool),
            },
            [done_shape[0]],
        )
        assert not _terminated_or_truncated(data, write_full_false=False)
        assert data.get("_reset", None) is None
        assert data.get(("nested", "_reset"), None) is None
        assert not _terminated_or_truncated(data, write_full_false=True)
        assert data["_reset"].shape == done_shape
        assert data["nested", "_reset"].shape == nested_done_shape

        data = TensorDict(
            {
                "terminated": torch.zeros(*done_shape, dtype=torch.bool),
                "truncated": torch.ones(*done_shape, dtype=torch.bool),
                ("nested", "terminated"): torch.zeros(
                    *nested_done_shape, dtype=torch.bool
                ),
            },
            [done_shape[0]],
        )
        assert _terminated_or_truncated(data, write_full_false=False)
        assert data["_reset"].shape == done_shape
        assert data["nested", "_reset"].shape == nested_done_shape
        assert data["_reset"].all()
        assert not data["nested", "_reset"].any()

    def test_terminated_or_truncated_spec(self):
        done_shape = (2, 1)
        nested_done_shape = (2, 3, 1)
        spec = CompositeSpec(
            done=DiscreteTensorSpec(2, shape=done_shape, dtype=torch.bool),
            shape=[
                2,
            ],
        )
        data = TensorDict(
            {"done": torch.zeros(*done_shape, dtype=torch.bool)}, [done_shape[0]]
        )
        assert not _terminated_or_truncated(
            data, write_full_false=True, full_done_spec=spec
        )
        assert data["_reset"].shape == done_shape
        assert not _terminated_or_truncated(
            data, write_full_false=False, full_done_spec=spec
        )
        assert data.get("_reset", None) is None

        spec = CompositeSpec(
            {
                ("agent", "done"): DiscreteTensorSpec(
                    2, shape=nested_done_shape, dtype=torch.bool
                ),
                ("nested", "done"): DiscreteTensorSpec(
                    2, shape=nested_done_shape, dtype=torch.bool
                ),
            },
            shape=[nested_done_shape[0]],
        )
        data = TensorDict(
            {
                ("agent", "done"): torch.zeros(*nested_done_shape, dtype=torch.bool),
                ("nested", "done"): torch.ones(*nested_done_shape, dtype=torch.bool),
            },
            [nested_done_shape[0]],
        )
        assert _terminated_or_truncated(data, full_done_spec=spec)
        assert data["agent", "_reset"].shape == nested_done_shape
        assert data["nested", "_reset"].shape == nested_done_shape

        data = TensorDict(
            {
                ("agent", "done"): torch.zeros(*nested_done_shape, dtype=torch.bool),
                ("nested", "done"): torch.zeros(*nested_done_shape, dtype=torch.bool),
            },
            [nested_done_shape[0]],
        )
        assert not _terminated_or_truncated(
            data, write_full_false=False, full_done_spec=spec
        )
        assert data.get(("agent", "_reset"), None) is None
        assert data.get(("nested", "_reset"), None) is None
        assert not _terminated_or_truncated(
            data, write_full_false=True, full_done_spec=spec
        )
        assert data["agent", "_reset"].shape == nested_done_shape
        assert data["nested", "_reset"].shape == nested_done_shape

        spec = CompositeSpec(
            {
                "truncated": DiscreteTensorSpec(2, shape=done_shape, dtype=torch.bool),
                "terminated": DiscreteTensorSpec(2, shape=done_shape, dtype=torch.bool),
                ("nested", "terminated"): DiscreteTensorSpec(
                    2, shape=nested_done_shape, dtype=torch.bool
                ),
            },
            shape=[2],
        )
        data = TensorDict(
            {
                "terminated": torch.zeros(*done_shape, dtype=torch.bool),
                "truncated": torch.ones(*done_shape, dtype=torch.bool),
                ("nested", "terminated"): torch.zeros(
                    *nested_done_shape, dtype=torch.bool
                ),
            },
            [done_shape[0]],
        )
        assert _terminated_or_truncated(
            data, write_full_false=False, full_done_spec=spec
        )
        assert data["_reset"].shape == done_shape
        assert data["nested", "_reset"].shape == nested_done_shape
        assert data["_reset"].all()
        assert not data["nested", "_reset"].any()


class TestLibThreading:
    @pytest.mark.skipif(
        IS_OSX,
        reason="setting different threads across workers can randomly fail on OSX.",
    )
    def test_num_threads(self):
        from torchrl.envs import batched_envs

        _run_worker_pipe_shared_mem_save = batched_envs._run_worker_pipe_shared_mem
        batched_envs._run_worker_pipe_shared_mem = decorate_thread_sub_func(
            batched_envs._run_worker_pipe_shared_mem, num_threads=3
        )
        num_threads = torch.get_num_threads()
        try:
            env = ParallelEnv(
                2, ContinuousActionVecMockEnv, num_sub_threads=3, num_threads=7
            )
            # We could test that the number of threads isn't changed until we start the procs.
            # Even though it's unlikely that we have 7 threads, we still disable this for safety
            # assert torch.get_num_threads() != 7
            env.rollout(3)
            assert torch.get_num_threads() == 7
        finally:
            # reset vals
            batched_envs._run_worker_pipe_shared_mem = _run_worker_pipe_shared_mem_save
            torch.set_num_threads(num_threads)

    @pytest.mark.skipif(
        IS_OSX,
        reason="setting different threads across workers can randomly fail on OSX.",
    )
    def test_auto_num_threads(self, maybe_fork_ParallelEnv):
        init_threads = torch.get_num_threads()

        try:
            env3 = maybe_fork_ParallelEnv(3, ContinuousActionVecMockEnv)
            env3.rollout(2)

            assert torch.get_num_threads() == max(1, init_threads - 3)

            env2 = maybe_fork_ParallelEnv(2, ContinuousActionVecMockEnv)
            env2.rollout(2)

            assert torch.get_num_threads() == max(1, init_threads - 5)

            env2.close()
            del env2
            gc.collect()

            assert torch.get_num_threads() == max(1, init_threads - 3)

            env3.close()
            del env3
            gc.collect()

            assert torch.get_num_threads() == init_threads
        finally:
            torch.set_num_threads(init_threads)


def test_run_type_checks():
    env = ContinuousActionVecMockEnv()
    env._run_type_checks = False
    check_env_specs(env)
    env._run_type_checks = True
    check_env_specs(env)
    env.output_spec.unlock_()
    # check type check on done
    env.output_spec["full_done_spec", "done"].dtype = torch.int
    with pytest.raises(TypeError, match="expected done.dtype to"):
        check_env_specs(env)
    env.output_spec["full_done_spec", "done"].dtype = torch.bool
    # check type check on reward
    env.output_spec["full_reward_spec", "reward"].dtype = torch.int
    with pytest.raises(TypeError, match="expected"):
        check_env_specs(env)
    env.output_spec["full_reward_spec", "reward"].dtype = torch.float
    # check type check on obs
    env.output_spec["full_observation_spec", "observation"].dtype = torch.float16
    with pytest.raises(TypeError):
        check_env_specs(env)


@pytest.mark.skipif(not torch.cuda.device_count(), reason="No cuda device found.")
@pytest.mark.parametrize("break_when_any_done", [True, False])
def test_auto_cast_to_device(break_when_any_done):
    env = ContinuousActionVecMockEnv(device="cpu")
    policy = Actor(
        nn.Linear(
            env.observation_spec["observation"].shape[-1],
            env.action_spec.shape[-1],
            device="cuda:0",
        ),
        in_keys=["observation"],
    )
    with pytest.raises(RuntimeError):
        env.rollout(10, policy)
    torch.manual_seed(0)
    env.set_seed(0)
    rollout0 = env.rollout(
        100, policy, auto_cast_to_device=True, break_when_any_done=break_when_any_done
    )
    torch.manual_seed(0)
    env.set_seed(0)
    rollout1 = env.rollout(
        100,
        policy.cpu(),
        auto_cast_to_device=False,
        break_when_any_done=break_when_any_done,
    )
    assert_allclose_td(rollout0, rollout1)


@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("share_individual_td", [True, False])
def test_backprop(device, maybe_fork_ParallelEnv, share_individual_td):
    # Tests that backprop through a series of single envs and through a serial env are identical
    # Also tests that no backprop can be achieved with parallel env.
    class DifferentiableEnv(EnvBase):
        def __init__(self, device):
            super().__init__(device=device)
            self.observation_spec = CompositeSpec(
                observation=UnboundedContinuousTensorSpec(3, device=device),
                device=device,
            )
            self.action_spec = CompositeSpec(
                action=UnboundedContinuousTensorSpec(3, device=device), device=device
            )
            self.reward_spec = CompositeSpec(
                reward=UnboundedContinuousTensorSpec(1, device=device), device=device
            )
            self.seed = 0

        def _set_seed(self, seed):
            self.seed = seed
            return seed

        def _reset(self, tensordict):
            td = self.observation_spec.zero().update(self.done_spec.zero())
            td["observation"] = (
                td["observation"].clone() + self.seed % 10
            ).requires_grad_()
            return td

        def _step(self, tensordict):
            action = tensordict.get("action")
            obs = (tensordict.get("observation") + action) / action.norm()
            return TensorDict(
                {
                    "reward": action.sum().unsqueeze(0),
                    **self.full_done_spec.zero(),
                    "observation": obs,
                },
                batch_size=[],
            )

    torch.manual_seed(0)
    policy = Actor(torch.nn.Linear(3, 3, device=device))
    env0 = DifferentiableEnv(device=device)
    seed = env0.set_seed(0)
    env1 = DifferentiableEnv(device=device)
    env1.set_seed(seed)
    r0 = env0.rollout(10, policy)
    r1 = env1.rollout(10, policy)
    r = torch.stack([r0, r1])
    g = torch.autograd.grad(r["next", "reward"].sum(), policy.parameters())

    def make_env(seed, device=device):
        env = DifferentiableEnv(device=device)
        env.set_seed(seed)
        return env

    serial_env = SerialEnv(
        2,
        [functools.partial(make_env, seed=0), functools.partial(make_env, seed=seed)],
        device=device,
        share_individual_td=share_individual_td,
    )
    if share_individual_td:
        r_serial = serial_env.rollout(10, policy)
    else:
        with pytest.raises(RuntimeError, match="Cannot update a view of a tensordict"):
            r_serial = serial_env.rollout(10, policy)
        return

    g_serial = torch.autograd.grad(
        r_serial["next", "reward"].sum(), policy.parameters()
    )
    torch.testing.assert_close(g, g_serial)

    p_env = maybe_fork_ParallelEnv(
        2,
        [functools.partial(make_env, seed=0), functools.partial(make_env, seed=seed)],
        device=device,
    )
    try:
        r_parallel = p_env.rollout(10, policy)
        assert not r_parallel.exclude("action").requires_grad
    finally:
        p_env.close()


@pytest.mark.skipif(not _has_gym, reason="Gym required for this test")
def test_non_td_policy():
    env = GymEnv("CartPole-v1", categorical_action_encoding=True)

    class ArgMaxModule(nn.Module):
        def forward(self, values):
            return values.argmax(-1)

    policy = nn.Sequential(
        nn.Linear(env.observation_spec["observation"].shape[-1], env.action_spec.n),
        ArgMaxModule(),
    )
    env.rollout(10, policy)
    env = SerialEnv(2, lambda: GymEnv("CartPole-v1", categorical_action_encoding=True))
    env.rollout(10, policy)


@pytest.mark.skipif(IS_WIN, reason="fork not available on windows 10")
def test_parallel_another_ctx():
    from torch import multiprocessing as mp

    sm = mp.get_start_method()
    if sm == "spawn":
        other_sm = "fork"
    else:
        other_sm = "spawn"
    env = ParallelEnv(2, ContinuousActionVecMockEnv, mp_start_method=other_sm)
    try:
        assert env.rollout(3) is not None
        assert env._workers[0]._start_method == other_sm
    finally:
        try:
            env.close()
            del env
        except RuntimeError:
            pass


@pytest.mark.skipif(not _has_gym, reason="gym not found")
def test_single_task_share_individual_td():
    cartpole = CARTPOLE_VERSIONED()
    env = SerialEnv(2, lambda: GymEnv(cartpole))
    assert not env.share_individual_td
    assert env._single_task
    env.rollout(2)
    assert isinstance(env.shared_tensordict_parent, TensorDict)

    env = SerialEnv(2, lambda: GymEnv(cartpole), share_individual_td=True)
    assert env.share_individual_td
    assert env._single_task
    env.rollout(2)
    assert isinstance(env.shared_tensordict_parent, LazyStackedTensorDict)

    env = SerialEnv(2, [lambda: GymEnv(cartpole)] * 2)
    assert not env.share_individual_td
    assert env._single_task
    env.rollout(2)
    assert isinstance(env.shared_tensordict_parent, TensorDict)

    env = SerialEnv(2, [lambda: GymEnv(cartpole)] * 2, share_individual_td=True)
    assert env.share_individual_td
    assert env._single_task
    env.rollout(2)
    assert isinstance(env.shared_tensordict_parent, LazyStackedTensorDict)

    env = SerialEnv(2, [EnvCreator(lambda: GymEnv(cartpole)) for _ in range(2)])
    assert not env.share_individual_td
    assert not env._single_task
    env.rollout(2)
    assert isinstance(env.shared_tensordict_parent, TensorDict)

    env = SerialEnv(
        2,
        [EnvCreator(lambda: GymEnv(cartpole)) for _ in range(2)],
        share_individual_td=True,
    )
    assert env.share_individual_td
    assert not env._single_task
    env.rollout(2)
    assert isinstance(env.shared_tensordict_parent, LazyStackedTensorDict)

    # Change shape: makes results non-stackable
    env = SerialEnv(
        2,
        [
            EnvCreator(lambda: GymEnv(cartpole)),
            EnvCreator(
                lambda: TransformedEnv(
                    GymEnv(cartpole), CatFrames(N=4, dim=-1, in_keys=["observation"])
                )
            ),
        ],
    )
    assert env.share_individual_td
    assert not env._single_task
    env.rollout(2)
    assert isinstance(env.shared_tensordict_parent, LazyStackedTensorDict)

    with pytest.raises(ValueError, match="share_individual_td=False"):
        SerialEnv(
            2,
            [
                EnvCreator(lambda: GymEnv(cartpole)),
                EnvCreator(
                    lambda: TransformedEnv(
                        GymEnv(cartpole),
                        CatFrames(N=4, dim=-1, in_keys=["observation"]),
                    )
                ),
            ],
            share_individual_td=False,
        )


def test_stackable():
    # Tests the _stackable util
    stack = [TensorDict({"a": 0}, []), TensorDict({"b": 1}, [])]
    assert not _stackable(*stack), torch.stack(stack)
    stack = [TensorDict({"a": [0]}, []), TensorDict({"a": 1}, [])]
    assert not _stackable(*stack)
    stack = [TensorDict({"a": [0]}, []), TensorDict({"a": [1]}, [])]
    assert _stackable(*stack)
    stack = [TensorDict({"a": [0]}, []), TensorDict({"a": [1], "b": {}}, [])]
    assert _stackable(*stack)
    stack = [TensorDict({"a": {"b": [0]}}, []), TensorDict({"a": {"b": [1]}}, [])]
    assert _stackable(*stack)
    stack = [TensorDict({"a": {"b": [0]}}, []), TensorDict({"a": {"b": 1}}, [])]
    assert not _stackable(*stack)
    stack = [TensorDict({"a": "a string"}, []), TensorDict({"a": "another string"}, [])]
    assert _stackable(*stack)


class TestAutoReset:
    def test_auto_reset(self):
        policy = lambda td: td.set(
            "action", torch.ones((*td.shape, 1), dtype=torch.int64)
        )

        env = AutoResettingCountingEnv(4, auto_reset=True)
        assert isinstance(env, TransformedEnv) and isinstance(
            env.transform, AutoResetTransform
        )
        r = env.rollout(20, policy, break_when_any_done=False)
        assert r.shape == torch.Size([20])
        assert r["next", "done"].sum() == 4
        assert (r["next", "observation"][r["next", "done"].squeeze()] == -1).all(), r[
            "next", "observation"
        ][r["next", "done"].squeeze()]
        assert (
            r[..., 1:]["observation"][r[..., :-1]["next", "done"].squeeze()] == 0
        ).all()
        r = env.rollout(20, policy, break_when_any_done=True)
        assert r["next", "done"].sum() == 1
        assert not r["done"].any()

    def test_auto_reset_transform(self):
        policy = lambda td: td.set(
            "action", torch.ones((*td.shape, 1), dtype=torch.int64)
        )
        env = TransformedEnv(
            AutoResettingCountingEnv(4, auto_reset=True), StepCounter()
        )
        assert isinstance(env, TransformedEnv) and isinstance(
            env.base_env.transform, AutoResetTransform
        )
        r = env.rollout(20, policy, break_when_any_done=False)
        assert r.shape == torch.Size([20])
        assert r["next", "done"].sum() == 4
        assert (r["next", "observation"][r["next", "done"].squeeze()] == -1).all()
        assert (
            r[..., 1:]["observation"][r[..., :-1]["next", "done"].squeeze()] == 0
        ).all()
        r = env.rollout(20, policy, break_when_any_done=True)
        assert r["next", "done"].sum() == 1
        assert not r["done"].any()

    def test_auto_reset_serial(self):
        policy = lambda td: td.set(
            "action", torch.ones((*td.shape, 1), dtype=torch.int64)
        )
        env = SerialEnv(
            2, functools.partial(AutoResettingCountingEnv, 4, auto_reset=True)
        )
        r = env.rollout(20, policy, break_when_any_done=False)
        assert r.shape == torch.Size([2, 20])
        assert r["next", "done"].sum() == 8
        assert (r["next", "observation"][r["next", "done"].squeeze()] == -1).all()
        assert (
            r[..., 1:]["observation"][r[..., :-1]["next", "done"].squeeze()] == 0
        ).all()
        r = env.rollout(20, policy, break_when_any_done=True)
        assert r["next", "done"].sum() == 2
        assert not r["done"].any()

    def test_auto_reset_serial_hetero(self):
        policy = lambda td: td.set(
            "action", torch.ones((*td.shape, 1), dtype=torch.int64)
        )
        env = SerialEnv(
            2,
            [
                functools.partial(AutoResettingCountingEnv, 4, auto_reset=True),
                functools.partial(AutoResettingCountingEnv, 5, auto_reset=True),
            ],
        )
        r = env.rollout(20, policy, break_when_any_done=False)
        assert r.shape == torch.Size([2, 20])
        assert (r["next", "observation"][r["next", "done"].squeeze()] == -1).all()
        assert (
            r[..., 1:]["observation"][r[..., :-1]["next", "done"].squeeze()] == 0
        ).all()
        assert not r["done"].any()

    def test_auto_reset_parallel(self):
        policy = lambda td: td.set(
            "action", torch.ones((*td.shape, 1), dtype=torch.int64)
        )
        env = ParallelEnv(
            2,
            functools.partial(AutoResettingCountingEnv, 4, auto_reset=True),
            mp_start_method=mp_ctx,
        )
        r = env.rollout(20, policy, break_when_any_done=False)
        assert r.shape == torch.Size([2, 20])
        assert r["next", "done"].sum() == 8
        assert (r["next", "observation"][r["next", "done"].squeeze()] == -1).all()
        assert (
            r[..., 1:]["observation"][r[..., :-1]["next", "done"].squeeze()] == 0
        ).all()
        r = env.rollout(20, policy, break_when_any_done=True)
        assert r["next", "done"].sum() == 2
        assert not r["done"].any()

    def test_auto_reset_parallel_hetero(self):
        policy = lambda td: td.set(
            "action", torch.ones((*td.shape, 1), dtype=torch.int64)
        )
        env = ParallelEnv(
            2,
            [
                functools.partial(AutoResettingCountingEnv, 4, auto_reset=True),
                functools.partial(AutoResettingCountingEnv, 5, auto_reset=True),
            ],
            mp_start_method=mp_ctx,
        )
        r = env.rollout(20, policy, break_when_any_done=False)
        assert r.shape == torch.Size([2, 20])
        assert (r["next", "observation"][r["next", "done"].squeeze()] == -1).all()
        assert (
            r[..., 1:]["observation"][r[..., :-1]["next", "done"].squeeze()] == 0
        ).all()
        assert not r["done"].any()

    def test_auto_reset_heterogeneous_env(self):
        torch.manual_seed(0)
        env = TransformedEnv(
            AutoResetHeteroCountingEnv(4, auto_reset=True), StepCounter()
        )

        def policy(td):
            return td.update(
                env.full_action_spec.zero().apply(lambda x: x.bernoulli_(0.5))
            )

        assert isinstance(env.base_env, AutoResetEnv) and isinstance(
            env.base_env.transform, AutoResetTransform
        )
        check_env_specs(env)
        r = env.rollout(40, policy, break_when_any_done=False)
        assert (r["next", "lazy", "step_count"] - 1 == r["lazy", "step_count"]).all()
        done = r["next", "lazy", "done"].squeeze(-1)[:-1]
        assert (
            r["next", "lazy", "step_count"][1:][~done]
            == r["next", "lazy", "step_count"][:-1][~done] + 1
        ).all()
        assert (
            r["next", "lazy", "step_count"][1:][done]
            != r["next", "lazy", "step_count"][:-1][done] + 1
        ).all()
        done_split = r["next", "lazy", "done"].unbind(1)
        lazy_slit = r["next", "lazy"].unbind(1)
        lazy_roots = r["lazy"].unbind(1)
        for lazy, lazy_root, done in zip(lazy_slit, lazy_roots, done_split):
            assert lazy["lidar"][done.squeeze()].isnan().all()
            assert not lazy["lidar"][~done.squeeze()].isnan().any()
            assert (lazy_root["lidar"][1:][done[:-1].squeeze()] == 0).all()


class TestEnvWithDynamicSpec:
    def test_dynamic_rollout(self):
        env = EnvWithDynamicSpec()
        with pytest.raises(
            RuntimeError,
            match="The environment specs are dynamic. Call rollout with return_contiguous=False",
        ):
            rollout = env.rollout(4)
        rollout = env.rollout(4, return_contiguous=False)
        check_env_specs(env, return_contiguous=False)

    @pytest.mark.skipif(not _has_gym, reason="requires gym to be installed")
    @pytest.mark.parametrize("penv", [SerialEnv, ParallelEnv])
    def test_batched_nondynamic(self, penv):
        # Tests not using buffers in batched envs
        env_buffers = penv(
            3,
            lambda: GymEnv(CARTPOLE_VERSIONED(), device=None),
            use_buffers=True,
            mp_start_method=mp_ctx if penv is ParallelEnv else None,
        )
        env_buffers.set_seed(0)
        torch.manual_seed(0)
        rollout_buffers = env_buffers.rollout(
            20, return_contiguous=True, break_when_any_done=False
        )
        del env_buffers
        gc.collect()

        env_no_buffers = penv(
            3,
            lambda: GymEnv(CARTPOLE_VERSIONED(), device=None),
            use_buffers=False,
            mp_start_method=mp_ctx if penv is ParallelEnv else None,
        )
        env_no_buffers.set_seed(0)
        torch.manual_seed(0)
        rollout_no_buffers = env_no_buffers.rollout(
            20, return_contiguous=True, break_when_any_done=False
        )
        del env_no_buffers
        gc.collect()
        assert_allclose_td(rollout_buffers, rollout_no_buffers)

    @pytest.mark.parametrize("break_when_any_done", [False, True])
    def test_batched_dynamic(self, break_when_any_done):
        list_of_envs = [EnvWithDynamicSpec(i + 4) for i in range(3)]
        dummy_rollouts = [
            env.rollout(
                20, return_contiguous=False, break_when_any_done=break_when_any_done
            )
            for env in list_of_envs
        ]
        t = min(dr.shape[0] for dr in dummy_rollouts)
        dummy_rollouts = TensorDict.maybe_dense_stack([dr[:t] for dr in dummy_rollouts])
        del list_of_envs

        # Tests not using buffers in batched envs
        env_no_buffers = SerialEnv(
            3,
            [lambda i=i + 4: EnvWithDynamicSpec(i) for i in range(3)],
            use_buffers=False,
        )
        env_no_buffers.set_seed(0)
        torch.manual_seed(0)
        rollout_no_buffers_serial = env_no_buffers.rollout(
            20, return_contiguous=False, break_when_any_done=break_when_any_done
        )
        del env_no_buffers
        gc.collect()
        assert_allclose_td(
            dummy_rollouts.exclude("action"),
            rollout_no_buffers_serial.exclude("action"),
        )

        env_no_buffers = ParallelEnv(
            3,
            [lambda i=i + 4: EnvWithDynamicSpec(i) for i in range(3)],
            use_buffers=False,
            mp_start_method=mp_ctx,
        )
        env_no_buffers.set_seed(0)
        torch.manual_seed(0)
        rollout_no_buffers_parallel = env_no_buffers.rollout(
            20, return_contiguous=False, break_when_any_done=break_when_any_done
        )
        del env_no_buffers
        gc.collect()

        assert_allclose_td(
            dummy_rollouts.exclude("action"),
            rollout_no_buffers_parallel.exclude("action"),
        )
        assert_allclose_td(rollout_no_buffers_serial, rollout_no_buffers_parallel)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
