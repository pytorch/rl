# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import copy
from functools import partial

import pytest
import torch
from tensordict import assert_allclose_td, LazyStackedTensorDict, TensorDict
from torch import nn

from torchrl.collectors import Collector
from torchrl.envs.libs.vmas import _has_vmas, VmasEnv, VmasWrapper
from torchrl.envs.utils import check_env_specs, MarlGroupMapType
from torchrl.modules import SafeModule
from torchrl.testing import get_available_devices, rand_reset


@pytest.mark.skipif(not _has_vmas, reason="vmas not installed")
class TestVmas:
    @pytest.mark.parametrize("scenario_name", VmasWrapper.available_envs)
    @pytest.mark.parametrize("continuous_actions", [True, False])
    def test_all_vmas_scenarios(self, scenario_name, continuous_actions):
        env = VmasEnv(
            scenario=scenario_name,
            continuous_actions=continuous_actions,
            num_envs=4,
        )
        env.set_seed(0)
        env.check_env_specs()
        env.rollout(10, break_when_any_done=False)
        env.check_env_specs()
        env.close()

    @pytest.mark.parametrize(
        "scenario_name", ["simple_reference", "waterfall", "flocking", "discovery"]
    )
    def test_vmas_seeding(self, scenario_name):
        final_seed = []
        tdreset = []
        tdrollout = []
        rollout_length = 10

        def create_env():
            return VmasEnv(
                scenario=scenario_name,
                num_envs=4,
            )

        env = create_env()
        td_actions = [env.action_spec.rand() for _ in range(rollout_length)]

        for _ in range(2):
            env = create_env()
            td_actions_buffer = copy.deepcopy(td_actions)

            def policy(td, actions=td_actions_buffer):
                return actions.pop(0)

            final_seed.append(env.set_seed(0))
            tdreset.append(env.reset())
            tdrollout.append(env.rollout(max_steps=rollout_length, policy=policy))
            env.close()
            del env
        assert final_seed[0] == final_seed[1]
        assert_allclose_td(*tdreset)
        assert_allclose_td(*tdrollout)

    @pytest.mark.parametrize(
        "batch_size", [(), (12,), (12, 2), (12, 3), (12, 3, 1), (12, 3, 4)]
    )
    @pytest.mark.parametrize("scenario_name", VmasWrapper.available_envs)
    def test_vmas_batch_size_error(self, scenario_name, batch_size):
        num_envs = 12
        if len(batch_size) > 1:
            with pytest.raises(
                TypeError,
                match="Batch size used in constructor is not compatible with vmas.",
            ):
                _ = VmasEnv(
                    scenario=scenario_name,
                    num_envs=num_envs,
                    batch_size=batch_size,
                )
        elif len(batch_size) == 1 and batch_size != (num_envs,):
            with pytest.raises(
                TypeError,
                match="Batch size used in constructor does not match vmas batch size.",
            ):
                _ = VmasEnv(
                    scenario=scenario_name,
                    num_envs=num_envs,
                    batch_size=batch_size,
                )
        else:
            env = VmasEnv(
                scenario=scenario_name,
                num_envs=num_envs,
                batch_size=batch_size,
            )
            env.close()

    @pytest.mark.parametrize("num_envs", [1, 20])
    @pytest.mark.parametrize("n_agents", [1, 5])
    @pytest.mark.parametrize(
        "scenario_name",
        ["simple_reference", "simple_tag", "waterfall", "flocking", "discovery"],
    )
    def test_vmas_batch_size(self, scenario_name, num_envs, n_agents):
        torch.manual_seed(0)
        n_rollout_samples = 5
        env = VmasEnv(
            scenario=scenario_name,
            num_envs=num_envs,
            n_agents=n_agents,
            group_map=MarlGroupMapType.ALL_IN_ONE_GROUP,
        )
        env.set_seed(0)
        tdreset = env.reset()
        tdrollout = env.rollout(
            max_steps=n_rollout_samples,
            return_contiguous=False if env.het_specs else True,
        )

        env.close()

        if env.het_specs:
            assert isinstance(tdreset["agents"], LazyStackedTensorDict)
        else:
            assert isinstance(tdreset["agents"], TensorDict)

        assert tdreset.batch_size == (num_envs,)
        assert tdreset["agents"].batch_size == (num_envs, env.n_agents)
        if not env.het_specs:
            assert tdreset["agents", "observation"].shape[1] == env.n_agents
        assert tdreset["done"].shape[1] == 1

        assert tdrollout.batch_size == (num_envs, n_rollout_samples)
        assert tdrollout["agents"].batch_size == (
            num_envs,
            n_rollout_samples,
            env.n_agents,
        )
        if not env.het_specs:
            assert tdrollout["agents", "observation"].shape[2] == env.n_agents
        assert tdrollout["next", "agents", "reward"].shape[2] == env.n_agents
        assert tdrollout["agents", "action"].shape[2] == env.n_agents
        assert tdrollout["done"].shape[2] == 1
        del env

    @pytest.mark.parametrize("num_envs", [1, 20])
    @pytest.mark.parametrize("n_agents", [1, 5])
    @pytest.mark.parametrize("continuous_actions", [True, False])
    @pytest.mark.parametrize(
        "scenario_name",
        ["simple_reference", "simple_tag", "waterfall", "flocking", "discovery"],
    )
    def test_vmas_spec_rollout(
        self, scenario_name, num_envs, n_agents, continuous_actions
    ):
        import vmas

        vmas_env = VmasEnv(
            scenario=scenario_name,
            num_envs=num_envs,
            n_agents=n_agents,
            continuous_actions=continuous_actions,
        )
        vmas_wrapped_env = VmasWrapper(
            vmas.make_env(
                scenario=scenario_name,
                num_envs=num_envs,
                n_agents=n_agents,
                continuous_actions=continuous_actions,
            )
        )
        for env in [vmas_env, vmas_wrapped_env]:
            env.set_seed(0)
            check_env_specs(env, return_contiguous=False if env.het_specs else True)
            env.close()

    @pytest.mark.parametrize("num_envs", [1, 20])
    @pytest.mark.parametrize("scenario_name", VmasWrapper.available_envs)
    def test_vmas_repr(self, scenario_name, num_envs):
        env = VmasEnv(
            scenario=scenario_name,
            num_envs=num_envs,
        )
        assert str(env) == (
            f"{VmasEnv.__name__}(num_envs={num_envs}, n_agents={env.n_agents},"
            f" batch_size={torch.Size((num_envs,))}, device={env.device}) (scenario={scenario_name})"
        )
        env.close()

    @pytest.mark.parametrize("num_envs", [1, 10])
    @pytest.mark.parametrize("n_workers", [1, 3])
    @pytest.mark.parametrize("continuous_actions", [True, False])
    @pytest.mark.parametrize(
        "scenario_name", ["simple_reference", "waterfall", "flocking", "discovery"]
    )
    def test_vmas_parallel(
        self,
        scenario_name,
        num_envs,
        n_workers,
        continuous_actions,
        maybe_fork_ParallelEnv,
        n_agents=5,
        n_rollout_samples=3,
    ):
        torch.manual_seed(0)

        def make_vmas():
            env = VmasEnv(
                scenario=scenario_name,
                num_envs=num_envs,
                n_agents=n_agents,
                continuous_actions=continuous_actions,
            )
            env.set_seed(0)
            return env

        env = maybe_fork_ParallelEnv(n_workers, make_vmas)
        tensordict = env.rollout(max_steps=n_rollout_samples)

        assert tensordict.shape == torch.Size(
            [n_workers, list(env.num_envs)[0], n_rollout_samples]
        )
        env.close()

    @pytest.mark.parametrize("num_envs", [1, 2])
    @pytest.mark.parametrize("n_workers", [1, 3])
    @pytest.mark.parametrize(
        "scenario_name", ["simple_reference", "waterfall", "flocking", "discovery"]
    )
    def test_vmas_reset(
        self,
        scenario_name,
        num_envs,
        n_workers,
        maybe_fork_ParallelEnv,
        n_agents=5,
        n_rollout_samples=3,
        max_steps=3,
    ):
        torch.manual_seed(0)

        def make_vmas():
            env = VmasEnv(
                scenario=scenario_name,
                num_envs=num_envs,
                n_agents=n_agents,
                max_steps=max_steps,
            )
            env.set_seed(0)
            return env

        env = maybe_fork_ParallelEnv(n_workers, make_vmas)
        tensordict = env.rollout(max_steps=n_rollout_samples)

        assert (
            tensordict["next", "done"]
            .sum(
                tuple(range(tensordict.batch_dims, tensordict["next", "done"].ndim)),
                dtype=torch.bool,
            )[..., -1]
            .all()
        )

        td_reset = TensorDict(
            rand_reset(env), batch_size=env.batch_size, device=env.device
        )
        # it is good practice to have a "complete" input tensordict for reset
        for done_key in env.done_keys:
            td_reset.set(done_key, tensordict[..., -1].get(("next", done_key)))
        reset = td_reset["_reset"]
        tensordict = env.reset(td_reset)

        assert not tensordict.get("done")[reset].any()
        assert tensordict.get("done")[~reset].all()
        env.close()

    @pytest.mark.skipif(len(get_available_devices()) < 2, reason="not enough devices")
    @pytest.mark.parametrize("first", [0, 1])
    @pytest.mark.parametrize(
        "scenario_name", ["simple_reference", "waterfall", "flocking", "discovery"]
    )
    def test_to_device(self, scenario_name: str, first: int):
        torch.manual_seed(0)
        devices = get_available_devices()

        def make_vmas():
            env = VmasEnv(
                scenario=scenario_name,
                num_envs=7,
                n_agents=3,
                seed=0,
                device=devices[first],
            )
            return env

        env = make_vmas()

        assert env.rollout(max_steps=3).device == devices[first]

        env.to(devices[1 - first])

        assert env.rollout(max_steps=3).device == devices[1 - first]
        env.close()

    @pytest.mark.parametrize("n_envs", [1, 4])
    @pytest.mark.parametrize("n_workers", [1, 2])
    @pytest.mark.parametrize("n_agents", [1, 3])
    def test_collector(
        self, n_envs, n_workers, n_agents, maybe_fork_ParallelEnv, frames_per_batch=80
    ):
        torch.manual_seed(1)
        env_fun = partial(
            VmasEnv,
            scenario="flocking",
            num_envs=n_envs,
            n_agents=n_agents,
            max_steps=7,
        )

        env = maybe_fork_ParallelEnv(n_workers, env_fun)

        n_actions_per_agent = env.full_action_spec[env.action_key].shape[-1]
        n_observations_per_agent = env.observation_spec["agents", "observation"].shape[
            -1
        ]

        policy = SafeModule(
            nn.Linear(
                n_observations_per_agent,
                n_actions_per_agent,
            ),
            in_keys=[("agents", "observation")],
            out_keys=[env.action_key],
            spec=env.full_action_spec[env.action_key],
            safe=True,
        )
        ccollector = Collector(
            create_env_fn=env,
            policy=policy,
            frames_per_batch=frames_per_batch,
            total_frames=1000,
            device="cpu",
        )

        for i, _td in enumerate(ccollector):
            if i == 1:
                break
        ccollector.shutdown()

        td_batch = (n_workers, n_envs, frames_per_batch // (n_workers * n_envs))
        agents_td_batch = td_batch + (n_agents,)

        assert _td.shape == td_batch
        assert _td["next"].shape == td_batch
        assert _td["agents"].shape == agents_td_batch
        assert _td["agents", "info"].shape == agents_td_batch
        assert _td["next", "agents"].shape == agents_td_batch
        assert _td["next", "agents", "info"].shape == agents_td_batch
        assert _td["collector"].shape == td_batch

        assert _td[env.action_key].shape == agents_td_batch + (n_actions_per_agent,)
        assert _td["agents", "observation"].shape == agents_td_batch + (
            n_observations_per_agent,
        )
        assert _td["next", "agents", "observation"].shape == agents_td_batch + (
            n_observations_per_agent,
        )
        assert _td["next", env.reward_key].shape == agents_td_batch + (1,)
        for done_key in env.done_keys:
            assert _td[done_key].shape == td_batch + (1,)
            assert _td["next", done_key].shape == td_batch + (1,)

        assert env.reward_key not in _td.keys(True, True)
        assert env.action_key not in _td["next"].keys(True, True)

    def test_collector_heterogeneous(self, n_envs=10, frames_per_batch=20):
        env = VmasEnv(
            scenario="simple_tag",
            num_envs=n_envs,
            group_map=MarlGroupMapType.ALL_IN_ONE_GROUP,
        )
        torch.manual_seed(1)

        ccollector = Collector(
            create_env_fn=env,
            policy=None,
            frames_per_batch=frames_per_batch,
            total_frames=1000,
            device="cpu",
        )

        for i, _td in enumerate(ccollector):
            if i == 1:
                break
        ccollector.shutdown()

        td_batch = (n_envs, frames_per_batch // n_envs)
        agents_td_batch = td_batch + (env.n_agents,)

        assert _td.shape == td_batch
        assert _td["next"].shape == td_batch
        assert _td["agents"].shape == agents_td_batch
        assert _td["next", "agents"].shape == agents_td_batch
        assert _td["collector"].shape == td_batch
        assert _td["next", env.reward_key].shape == agents_td_batch + (1,)
        for done_key in env.done_keys:
            assert _td[done_key].shape == td_batch + (1,)
            assert _td["next", done_key].shape == td_batch + (1,)

        assert env.reward_key not in _td.keys(True, True)
        assert env.action_key not in _td["next"].keys(True, True)

    @pytest.mark.parametrize("n_agents", [1, 5])
    def test_grouping(self, n_agents, scenario_name="dispersion", n_envs=2):
        env = VmasEnv(
            scenario=scenario_name,
            num_envs=n_envs,
            n_agents=n_agents,
        )
        env = VmasEnv(
            scenario=scenario_name,
            num_envs=n_envs,
            n_agents=n_agents,
            # Put each agent in a group with its name
            group_map={
                agent_name: [agent_name] for agent_name in reversed(env.agent_names)
            },
        )

        # Check that when setting the action for a specific group, it is reflected to the right agent in the backend
        for group in env.group_map.keys():
            env.reset()
            action = env.full_action_spec.zero()
            action.set((group, "action"), action.get((group, "action")) + 1.0)
            prev_pos = {agent.name: agent.state.pos.clone() for agent in env.agents}
            env.step(action)
            pos = {agent.name: agent.state.pos.clone() for agent in env.agents}
            for agent_name in env.agent_names:
                if agent_name == group:
                    assert (pos[agent_name] > prev_pos[agent_name]).all()
                else:
                    assert (pos[agent_name] == prev_pos[agent_name]).all()
