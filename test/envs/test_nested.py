# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import gc

import pytest
import torch
from tensordict import (
    assert_allclose_td,
    dense_stack_tds,
    LazyStackedTensorDict,
    TensorDict,
)
from tensordict.utils import _unravel_key_to_tuple

from torchrl.data.tensor_specs import Categorical, Composite
from torchrl.envs import EnvBase, ParallelEnv, SerialEnv, TransformedEnv
from torchrl.envs.transforms.transforms import InitTracker
from torchrl.envs.utils import _terminated_or_truncated
from torchrl.testing import check_rollout_consistency_multikey_env
from torchrl.testing.mocking_classes import (
    CountingEnv,
    CountingEnvCountPolicy,
    HeterogeneousCountingEnv,
    HeterogeneousCountingEnvPolicy,
    MockNestedResetEnv,
    MultiKeyCountingEnv,
    MultiKeyCountingEnvPolicy,
    NestedCountingEnv,
)


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
        if envclass == "NestedCountingEnv":
            assert isinstance(env.reward_spec, Composite)
        else:
            assert not isinstance(env.reward_spec, Composite)
        for done_key in env.done_keys:
            assert (
                env.full_done_spec[done_key]
                == env.output_spec[("full_done_spec", *_unravel_key_to_tuple(done_key))]
            )
        if envclass == "NestedCountingEnv":
            assert (
                env.full_reward_spec[env.reward_key]
                == env.output_spec[
                    ("full_reward_spec", *_unravel_key_to_tuple(env.reward_key))
                ]
            )
        else:
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

        policy = CountingEnvCountPolicy(
            env.full_action_spec[env.action_key], env.action_key
        )
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

    @pytest.mark.parametrize("done_at_root", [True, False])
    def test_nested_partial_resets(self, maybe_fork_ParallelEnv, done_at_root):
        def make_env(num_steps):
            return MockNestedResetEnv(num_steps, done_at_root)

        def manual_rollout(env: EnvBase, num_steps: int):
            steps = []
            td = env.reset()
            for _ in range(num_steps):
                td, next_td = env.step_and_maybe_reset(td)
                steps.append(td)
                td = next_td
            return TensorDict.stack(steps)

        # NOTE: we expect the env[0] to reset after 4 steps, env[1] to reset after 6 steps.
        parallel_env = maybe_fork_ParallelEnv(
            2,
            create_env_fn=make_env,
            create_env_kwargs=[{"num_steps": i} for i in [4, 6]],
        )
        transformed_env = TransformedEnv(
            base_env=maybe_fork_ParallelEnv(
                2,
                create_env_fn=make_env,
                create_env_kwargs=[{"num_steps": i} for i in [4, 6]],
            ),
            transform=InitTracker(),
        )

        parallel_td = manual_rollout(parallel_env, 6)

        transformed_td = manual_rollout(transformed_env, 6)

        # We expect env[0] to have been reset and executed 2 steps.
        # We expect env[1] to have just been reset (0 steps).
        assert parallel_env._counter() == [2, 0]
        assert transformed_env._counter() == [2, 0]
        if done_at_root:
            assert parallel_env._simple_done
            assert transformed_env._simple_done
            # We expect each env to have reached a done state once.
            assert parallel_td["next", "done"].sum().item() == 2
            assert transformed_td["next", "done"].sum().item() == 2
            assert_allclose_td(transformed_td, parallel_td, intersection=True)
        else:
            assert not parallel_env._simple_done
            assert not transformed_env._simple_done

            assert ("next", "done") not in parallel_td
            assert ("next", "done") not in transformed_td
            assert parallel_td["next", "agent_1", "done"].sum().item() == 2
            assert transformed_td["next", "agent_1", "done"].sum().item() == 2
            assert_allclose_td(transformed_td, parallel_td, intersection=True)

        assert transformed_env._counter() == [2, 0]


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
        gc.collect()
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
        torch.manual_seed(seed)
        env = MultiKeyCountingEnv(batch_size=batch_size, max_steps=max_steps)
        policy = MultiKeyCountingEnvPolicy(full_action_spec=env.full_action_spec)
        td = env.rollout(rollout_steps, policy=policy)
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


class TestTerminatedOrTruncated:
    @pytest.mark.parametrize("done_key", ["done", "terminated", "truncated"])
    def test_root_prevail(self, done_key):
        _spec = Categorical(2, shape=(), dtype=torch.bool)
        spec = Composite({done_key: _spec, ("agent", done_key): _spec})
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
        spec = Composite(
            done=Categorical(2, shape=done_shape, dtype=torch.bool),
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

        spec = Composite(
            {
                ("agent", "done"): Categorical(
                    2, shape=nested_done_shape, dtype=torch.bool
                ),
                ("nested", "done"): Categorical(
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

        spec = Composite(
            {
                "truncated": Categorical(2, shape=done_shape, dtype=torch.bool),
                "terminated": Categorical(2, shape=done_shape, dtype=torch.bool),
                ("nested", "terminated"): Categorical(
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
