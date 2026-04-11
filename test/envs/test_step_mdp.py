# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import pytest
import torch

from _envs_common import _has_gym, _has_mujoco, gym_version
from packaging import version
from tensordict import assert_allclose_td, LazyStackedTensorDict, TensorDict

from torchrl.collectors import Collector
from torchrl.data.tensor_specs import Composite, Unbounded
from torchrl.envs import SerialEnv, set_gym_backend
from torchrl.envs.gym_like import default_info_dict_reader
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs.utils import (
    _StepMDP,
    check_env_specs,
    check_marl_grouping,
    MarlGroupMapType,
    step_mdp,
)
from torchrl.testing import get_default_devices, HALFCHEETAH_VERSIONED
from torchrl.testing.mocking_classes import (
    ContinuousActionVecMockEnv,
    CountingBatchedEnv,
    CountingEnv,
    DiscreteActionConvMockEnv,
    HeterogeneousCountingEnv,
    NestedCountingEnv,
)


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
        next_tensordict = TensorDict(batch_size=[4]) if has_out else None
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

    @pytest.mark.parametrize(
        "envcls",
        [
            ContinuousActionVecMockEnv,
            CountingBatchedEnv,
            CountingEnv,
        ],
    )
    def test_step_class_out_reuse(self, envcls):
        torch.manual_seed(0)
        env = envcls()
        tensordict = env.rand_step(env.reset())

        step_func = _StepMDP(env, exclude_action=False)
        result_no_out = step_func(tensordict.clone())
        out_buf = result_no_out.clone()
        out_buf_id = id(out_buf)

        result_with_out = step_func(tensordict.clone(), out=out_buf)
        assert id(result_with_out) == out_buf_id
        assert (result_no_out == result_with_out).all()

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
                nested_key: TensorDict(batch_size=nested_batch_size),
                "next": {
                    nested_key: TensorDict(batch_size=nested_batch_size),
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
                nested_key: TensorDict(batch_size=nested_batch_size),
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
                "next": {nested_key: TensorDict(batch_size=nested_batch_size)},
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
        (td_batch_size, nested_dim)
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
        td.keys(True, True)
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
        env.set_spec_lock_()
        env.rollout(10)
        c = Collector(env, env.rand_action, frames_per_batch=10, total_frames=20)
        for data in c:  # noqa: B007
            pass
        assert ("collector", "traj_ids") in data.keys(True)
        env.rollout(10)

        # An exception will be raised when the collector sees extra keys
        if serial:
            env = SerialEnv(2, ContinuousActionVecMockEnv)
        else:
            env = ContinuousActionVecMockEnv()
        c = Collector(env, env.rand_action, frames_per_batch=10, total_frames=20)
        for data in c:  # noqa: B007
            pass


class TestInfoDict:
    @pytest.mark.skipif(not _has_gym, reason="no gym")
    @pytest.mark.skipif(
        gym_version is None or gym_version < version.parse("0.20.0"),
        reason="older versions of half-cheetah do not have 'x_position' info key.",
    )
    @pytest.mark.parametrize("device", get_default_devices())
    def test_info_dict_reader(self, device, seed=0):
        if not _has_mujoco:
            pytest.skip(
                "MuJoCo not available (missing mujoco); skipping MuJoCo gym test."
            )
        try:
            import gymnasium as gym
        except ModuleNotFoundError:
            import gym
        set_gym_backend(gym).set()

        env = GymWrapper(gym.make(HALFCHEETAH_VERSIONED()), device=device)
        env.set_info_dict_reader(
            default_info_dict_reader(
                ["x_position"],
                spec=Composite(x_position=Unbounded(dtype=torch.float64, shape=())),
            )
        )

        assert "x_position" in env.observation_spec.keys()
        assert isinstance(env.observation_spec["x_position"], Unbounded)

        tensordict = env.reset()
        tensordict = env.rand_step(tensordict)

        x_position_data = tensordict["next", "x_position"]
        assert env.observation_spec["x_position"].is_in(x_position_data), (
            x_position_data.shape,
            x_position_data.dtype,
            env.observation_spec["x_position"],
        )

        for spec in (
            {"x_position": Unbounded((), dtype=torch.float64)},
            # None,
            Composite(
                x_position=Unbounded((), dtype=torch.float64),
                shape=[],
            ),
            [Unbounded((), dtype=torch.float64)],
        ):
            if not _has_mujoco:
                pytest.skip(
                    "MuJoCo not available (missing mujoco); skipping MuJoCo gym test."
                )
            env2 = GymWrapper(gym.make("HalfCheetah-v5"))
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
        if not _has_mujoco:
            pytest.skip(
                "MuJoCo not available (missing mujoco); skipping MuJoCo gym test."
            )
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
