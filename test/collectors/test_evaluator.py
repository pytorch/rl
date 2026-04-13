# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import threading
import time
from collections import OrderedDict

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.collectors import Evaluator
from torchrl.envs import SerialEnv, TransformedEnv
from torchrl.envs.transforms import RewardSum, StepCounter
from torchrl.testing.mocking_classes import ContinuousActionVecMockEnv
from torchrl.weight_update import WeightStrategy


def _make_env():
    return ContinuousActionVecMockEnv()


def _make_policy(env=None):
    if env is None:
        env = _make_env()
    obs_dim = env.observation_spec["observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]
    return TensorDictModule(
        nn.Linear(obs_dim, act_dim),
        in_keys=["observation"],
        out_keys=["action"],
    )


class TestEvaluatorSync:
    def test_evaluate_basic(self):
        env = _make_env()
        policy = _make_policy(env)
        evaluator = Evaluator(env, policy, max_steps=50)
        try:
            metrics = evaluator.evaluate(step=0)
            assert "eval/reward" in metrics
            assert "eval/episode_length" in metrics
            assert isinstance(metrics["eval/reward"], float)
        finally:
            evaluator.shutdown()

    def test_evaluate_with_weights(self):
        env = _make_env()
        policy = _make_policy(env)
        train_policy = _make_policy(env)
        evaluator = Evaluator(env, policy, max_steps=50)
        try:
            weights = Evaluator.extract_weights(train_policy)
            metrics = evaluator.evaluate(weights=weights, step=0)
            assert "eval/reward" in metrics
        finally:
            evaluator.shutdown()

    def test_evaluate_with_module_weights(self):
        env = _make_env()
        policy = _make_policy(env)
        train_policy = _make_policy(env)
        evaluator = Evaluator(env, policy, max_steps=50)
        try:
            metrics = evaluator.evaluate(weights=train_policy, step=0)
            assert "eval/reward" in metrics
        finally:
            evaluator.shutdown()

    def test_custom_log_prefix(self):
        env = _make_env()
        policy = _make_policy(env)
        evaluator = Evaluator(env, policy, max_steps=50, log_prefix="test_eval")
        try:
            metrics = evaluator.evaluate(step=0)
            assert "test_eval/reward" in metrics
            assert "test_eval/episode_length" in metrics
        finally:
            evaluator.shutdown()

    def test_custom_metrics_fn(self):
        env = _make_env()
        policy = _make_policy(env)

        def my_metrics(td):
            return {"num_steps": td.shape[-1]}

        evaluator = Evaluator(env, policy, max_steps=50, metrics_fn=my_metrics)
        try:
            metrics = evaluator.evaluate(step=0)
            assert "eval/num_steps" in metrics
        finally:
            evaluator.shutdown()

    def test_callback(self):
        env = _make_env()
        policy = _make_policy(env)
        results = []

        def cb(metrics, step):
            results.append((metrics, step))

        evaluator = Evaluator(env, policy, max_steps=50, callback=cb)
        try:
            evaluator.evaluate(step=42)
            assert len(results) == 1
            assert results[0][1] == 42
            assert "eval/reward" in results[0][0]
        finally:
            evaluator.shutdown()

    def test_policy_factory(self):
        evaluator = Evaluator(
            _make_env,
            policy_factory=lambda env: _make_policy(env),
            max_steps=50,
        )
        try:
            metrics = evaluator.evaluate(step=0)
            assert "eval/reward" in metrics
        finally:
            evaluator.shutdown()

    def test_multiple_evals(self):
        env = _make_env()
        policy = _make_policy(env)
        evaluator = Evaluator(env, policy, max_steps=50)
        try:
            for i in range(3):
                metrics = evaluator.evaluate(step=i)
                assert "eval/reward" in metrics
        finally:
            evaluator.shutdown()

    def test_extract_weights(self):
        policy = _make_policy()
        weights = Evaluator.extract_weights(policy)
        assert isinstance(weights, TensorDict)
        assert weights.device == torch.device("cpu")

    def test_step_provenance_in_result(self):
        env = _make_env()
        policy = _make_policy(env)
        evaluator = Evaluator(env, policy, max_steps=50)
        try:
            metrics = evaluator.evaluate(step=123)
            assert metrics["eval/step"] == 123
        finally:
            evaluator.shutdown()


class TestEvaluatorAsync:
    def test_trigger_and_wait(self):
        env = _make_env()
        policy = _make_policy(env)
        evaluator = Evaluator(env, policy, max_steps=50)
        try:
            evaluator.trigger_eval(step=0)
            result = evaluator.wait(timeout=30)
            assert result is not None
            assert "eval/reward" in result
        finally:
            evaluator.shutdown()

    def test_trigger_and_poll(self):
        env = _make_env()
        policy = _make_policy(env)
        evaluator = Evaluator(env, policy, max_steps=50)
        try:
            evaluator.trigger_eval(step=0)
            # poll with generous timeout
            result = evaluator.poll(timeout=30)
            assert result is not None
            assert "eval/reward" in result
        finally:
            evaluator.shutdown()

    def test_poll_nonblocking_returns_none(self):
        env = _make_env()
        policy = _make_policy(env)
        evaluator = Evaluator(env, policy, max_steps=500)
        try:
            evaluator.trigger_eval(step=0)
            # Immediately poll - might be None
            result = evaluator.poll(timeout=0)
            # Either None or valid
            if result is not None:
                assert "eval/reward" in result
        finally:
            evaluator.shutdown()

    def test_fire_and_forget(self):
        env = _make_env()
        policy = _make_policy(env)
        evaluator = Evaluator(env, policy, max_steps=200)
        try:
            evaluator.trigger_eval(step=0)
            evaluator.trigger_eval(step=1)
            result = evaluator.wait(timeout=30)
            assert result is not None
            assert "eval/reward" in result
        finally:
            evaluator.shutdown()

    def test_async_with_weights(self):
        env = _make_env()
        policy = _make_policy(env)
        train_policy = _make_policy(env)
        evaluator = Evaluator(env, policy, max_steps=50)
        try:
            evaluator.trigger_eval(weights=train_policy, step=0)
            result = evaluator.wait(timeout=30)
            assert result is not None
            assert "eval/reward" in result
        finally:
            evaluator.shutdown()

    def test_async_callback(self):
        env = _make_env()
        policy = _make_policy(env)
        results = []

        def cb(metrics, step):
            results.append((metrics, step))

        evaluator = Evaluator(env, policy, max_steps=50, callback=cb)
        try:
            evaluator.trigger_eval(step=7)
            evaluator.wait(timeout=30)
            assert len(results) >= 1
            assert results[0][1] == 7
        finally:
            evaluator.shutdown()

    def test_multiple_async_evals(self):
        env = _make_env()
        policy = _make_policy(env)
        evaluator = Evaluator(env, policy, max_steps=50)
        try:
            for i in range(3):
                evaluator.trigger_eval(step=i)
                result = evaluator.wait(timeout=30)
                assert result is not None
                assert "eval/reward" in result
        finally:
            evaluator.shutdown()

    def test_shutdown_no_thread_leak(self):
        env = _make_env()
        policy = _make_policy(env)
        evaluator = Evaluator(env, policy, max_steps=50)
        evaluator.trigger_eval(step=0)
        evaluator.wait(timeout=30)
        evaluator.shutdown()
        time.sleep(0.5)
        for t in threading.enumerate():
            assert (
                not t.is_alive() or "eval" not in t.name.lower()
            ), f"Thread {t.name} still alive after shutdown"

    def test_step_provenance_in_result(self):
        env = _make_env()
        policy = _make_policy(env)
        evaluator = Evaluator(env, policy, max_steps=50)
        try:
            evaluator.trigger_eval(step=999)
            result = evaluator.wait(timeout=30)
            assert result is not None
            assert result["eval/step"] == 999
        finally:
            evaluator.shutdown()

    def test_shutdown_with_inflight_eval(self):
        """Shutdown while an eval is still running should not hang."""
        env = _make_env()
        policy = _make_policy(env)
        evaluator = Evaluator(env, policy, max_steps=5000)
        evaluator.trigger_eval(step=0)
        # Don't wait for completion -- shut down immediately
        t0 = time.time()
        evaluator.shutdown(timeout=5.0)
        elapsed = time.time() - t0
        assert elapsed < 10.0, f"Shutdown took {elapsed:.1f}s, expected < 10s"


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires 2+ CUDA devices",
)
class TestEvaluatorMultiDevice:
    def test_eval_on_dedicated_device(self):
        """Training policy on cuda:0, eval on cuda:1."""
        train_device = torch.device("cuda:0")
        eval_device = torch.device("cuda:1")

        env = ContinuousActionVecMockEnv(device=eval_device)
        eval_policy = _make_policy(env).to(eval_device)
        train_policy = _make_policy(env).to(train_device)

        evaluator = Evaluator(env, eval_policy, max_steps=50, device=eval_device)
        try:
            # Pass training weights from cuda:0 -- should be moved to cuda:1
            metrics = evaluator.evaluate(weights=train_policy, step=0)
            assert "eval/reward" in metrics
            assert isinstance(metrics["eval/reward"], float)
        finally:
            evaluator.shutdown()

    def test_async_eval_on_dedicated_device(self):
        """Async eval on a different device from training."""
        train_device = torch.device("cuda:0")
        eval_device = torch.device("cuda:1")

        env = ContinuousActionVecMockEnv(device=eval_device)
        eval_policy = _make_policy(env).to(eval_device)
        train_policy = _make_policy(env).to(train_device)

        evaluator = Evaluator(env, eval_policy, max_steps=50, device=eval_device)
        try:
            evaluator.trigger_eval(weights=train_policy, step=0)
            result = evaluator.wait(timeout=30)
            assert result is not None
            assert "eval/reward" in result
        finally:
            evaluator.shutdown()


class TestEvaluatorPending:
    def test_pending_false_initially(self):
        env = _make_env()
        policy = _make_policy(env)
        evaluator = Evaluator(env, policy, max_steps=50)
        try:
            assert not evaluator.pending
        finally:
            evaluator.shutdown()

    def test_pending_during_async_eval(self):
        env = _make_env()
        policy = _make_policy(env)
        evaluator = Evaluator(env, policy, max_steps=500)
        try:
            evaluator.trigger_eval(step=0)
            assert evaluator.pending
            result = evaluator.wait(timeout=30)
            assert result is not None
            assert not evaluator.pending
        finally:
            evaluator.shutdown()

    def test_pending_cleared_by_poll(self):
        env = _make_env()
        policy = _make_policy(env)
        evaluator = Evaluator(env, policy, max_steps=50)
        try:
            evaluator.trigger_eval(step=0)
            result = evaluator.poll(timeout=30)
            assert result is not None
            assert not evaluator.pending
        finally:
            evaluator.shutdown()

    def test_pending_cleared_by_shutdown(self):
        env = _make_env()
        policy = _make_policy(env)
        evaluator = Evaluator(env, policy, max_steps=5000)
        evaluator.trigger_eval(step=0)
        assert evaluator.pending
        evaluator.shutdown(timeout=5.0)
        assert not evaluator.pending


class TestEvaluatorLazyInit:
    """Tests for lazy env/policy creation when using factories with thread backend."""

    def test_sync_eval_with_factories(self):
        evaluator = Evaluator(
            _make_env,
            policy_factory=lambda env: _make_policy(env),
            max_steps=50,
        )
        try:
            metrics = evaluator.evaluate(step=0)
            assert "eval/reward" in metrics
        finally:
            evaluator.shutdown()

    def test_async_eval_with_factories(self):
        evaluator = Evaluator(
            _make_env,
            policy_factory=lambda env: _make_policy(env),
            max_steps=50,
        )
        try:
            evaluator.trigger_eval(step=0)
            result = evaluator.wait(timeout=30)
            assert result is not None
            assert "eval/reward" in result
        finally:
            evaluator.shutdown()

    def test_lazy_init_deferred_to_worker_thread(self):
        """Verify env/policy are created in the worker thread, not the constructor."""
        creation_thread_names = []

        def tracked_env_factory():
            creation_thread_names.append(threading.current_thread().name)
            return _make_env()

        def tracked_policy_factory(env):
            creation_thread_names.append(threading.current_thread().name)
            return _make_policy(env)

        evaluator = Evaluator(
            tracked_env_factory,
            policy_factory=tracked_policy_factory,
            max_steps=50,
        )
        # Nothing created yet
        assert len(creation_thread_names) == 0

        try:
            evaluator.trigger_eval(step=0)
            result = evaluator.wait(timeout=30)
            assert result is not None
            # Both env and policy created on the worker thread, not MainThread
            assert len(creation_thread_names) == 2
            for name in creation_thread_names:
                assert (
                    name != "MainThread"
                ), f"Expected creation on worker thread, got {name}"
        finally:
            evaluator.shutdown()

    def test_lazy_init_with_weights(self):
        """Weights are applied after lazy creation."""
        train_policy = _make_policy()
        evaluator = Evaluator(
            _make_env,
            policy_factory=lambda env: _make_policy(env),
            max_steps=50,
        )
        try:
            evaluator.trigger_eval(weights=train_policy, step=0)
            result = evaluator.wait(timeout=30)
            assert result is not None
            assert "eval/reward" in result
        finally:
            evaluator.shutdown()


class TestEvaluatorProcess:
    """Tests for the process-based backend.

    Note: the ``spawn`` mp context requires picklable callables, so we
    use top-level functions (``_make_env``, ``_make_policy``) rather than
    lambdas.
    """

    def test_sync_eval(self):
        evaluator = Evaluator(
            _make_env,
            policy_factory=_make_policy,
            max_steps=50,
            backend="process",
        )
        try:
            metrics = evaluator.evaluate(step=0)
            assert "eval/reward" in metrics
            assert "eval/episode_length" in metrics
            assert isinstance(metrics["eval/reward"], float)
        finally:
            evaluator.shutdown()

    def test_async_trigger_and_wait(self):
        evaluator = Evaluator(
            _make_env,
            policy_factory=_make_policy,
            max_steps=50,
            backend="process",
        )
        try:
            evaluator.trigger_eval(step=0)
            assert evaluator.pending
            result = evaluator.wait(timeout=30)
            assert result is not None
            assert "eval/reward" in result
            assert not evaluator.pending
        finally:
            evaluator.shutdown()

    def test_async_trigger_and_poll(self):
        evaluator = Evaluator(
            _make_env,
            policy_factory=_make_policy,
            max_steps=50,
            backend="process",
        )
        try:
            evaluator.trigger_eval(step=0)
            result = evaluator.poll(timeout=30)
            assert result is not None
            assert "eval/reward" in result
        finally:
            evaluator.shutdown()

    def test_with_weights(self):
        train_policy = _make_policy()
        evaluator = Evaluator(
            _make_env,
            policy_factory=_make_policy,
            max_steps=50,
            backend="process",
        )
        try:
            metrics = evaluator.evaluate(weights=train_policy, step=0)
            assert "eval/reward" in metrics
        finally:
            evaluator.shutdown()

    def test_multiple_evals(self):
        evaluator = Evaluator(
            _make_env,
            policy_factory=_make_policy,
            max_steps=50,
            backend="process",
        )
        try:
            for i in range(3):
                evaluator.trigger_eval(step=i)
                result = evaluator.wait(timeout=30)
                assert result is not None
                assert "eval/reward" in result
        finally:
            evaluator.shutdown()

    def test_step_provenance(self):
        evaluator = Evaluator(
            _make_env,
            policy_factory=_make_policy,
            max_steps=50,
            backend="process",
        )
        try:
            metrics = evaluator.evaluate(step=456)
            assert metrics["eval/step"] == 456
        finally:
            evaluator.shutdown()

    def test_callback(self):
        results = []

        def cb(metrics, step):
            results.append((metrics, step))

        evaluator = Evaluator(
            _make_env,
            policy_factory=_make_policy,
            max_steps=50,
            backend="process",
            callback=cb,
        )
        try:
            evaluator.evaluate(step=42)
            assert len(results) == 1
            assert results[0][1] == 42
        finally:
            evaluator.shutdown()

    def test_requires_callable_env(self):
        env = _make_env()
        with pytest.raises(ValueError, match="callable"):
            Evaluator(
                env,
                policy_factory=_make_policy,
                max_steps=50,
                backend="process",
            )

    def test_requires_policy_factory(self):
        with pytest.raises(ValueError, match="policy_factory"):
            Evaluator(
                _make_env,
                policy=_make_policy(),
                max_steps=50,
                backend="process",
            )


def _make_batched_env(num_envs=4, max_steps=5):
    """Create a batched env with StepCounter + RewardSum for testing."""
    return TransformedEnv(
        SerialEnv(
            num_envs,
            lambda: TransformedEnv(
                ContinuousActionVecMockEnv(),
                StepCounter(max_steps=max_steps),
            ),
        ),
        RewardSum(),
    )


def _make_batched_env_no_reward_sum(num_envs=4, max_steps=5):
    """Batched env without RewardSum (fallback path)."""
    return SerialEnv(
        num_envs,
        lambda: TransformedEnv(
            ContinuousActionVecMockEnv(),
            StepCounter(max_steps=max_steps),
        ),
    )


class TestEvaluatorBatchedMetrics:
    """Tests for batched env metric extraction with collector backend."""

    def test_batched_env_reports_num_episodes(self):
        """With RewardSum, evaluator should count completed episodes."""
        env = _make_batched_env(num_envs=4, max_steps=5)
        policy = _make_policy(env)
        evaluator = Evaluator(env, policy, max_steps=20, num_trajectories=5)
        try:
            metrics = evaluator.evaluate(step=0)
            assert "eval/num_episodes" in metrics
            assert metrics["eval/num_episodes"] == 5
            assert "eval/reward" in metrics
            assert "eval/reward_std" in metrics
        finally:
            evaluator.shutdown()

    def test_num_trajectories_controls_episode_count(self):
        """num_trajectories determines how many complete episodes are collected."""
        env = _make_batched_env(num_envs=4, max_steps=5)
        policy = _make_policy(env)
        evaluator = Evaluator(env, policy, max_steps=20, num_trajectories=3)
        try:
            metrics = evaluator.evaluate(step=0)
            assert metrics["eval/num_episodes"] == 3
        finally:
            evaluator.shutdown()

    def test_single_env_with_collector(self):
        """Single env works with the collector backend."""
        env = _make_env()
        policy = _make_policy(env)
        evaluator = Evaluator(env, policy, max_steps=100, num_trajectories=2)
        try:
            metrics = evaluator.evaluate(step=0)
            assert "eval/reward" in metrics
            assert metrics["eval/num_episodes"] == 2
        finally:
            evaluator.shutdown()

    def test_fallback_without_reward_sum(self):
        """Without RewardSum, falls back to summing raw rewards per trajectory."""
        env = _make_batched_env_no_reward_sum(num_envs=4, max_steps=5)
        policy = _make_policy(env)
        evaluator = Evaluator(env, policy, max_steps=20, num_trajectories=3)
        try:
            metrics = evaluator.evaluate(step=0)
            assert "eval/reward" in metrics
            assert isinstance(metrics["eval/reward"], float)
        finally:
            evaluator.shutdown()

    def test_episode_length_from_step_count(self):
        """With StepCounter, episode_length should come from step_count at done."""
        env = _make_batched_env(num_envs=4, max_steps=5)
        policy = _make_policy(env)
        evaluator = Evaluator(env, policy, max_steps=20, num_trajectories=4)
        try:
            metrics = evaluator.evaluate(step=0)
            # StepCounter caps at 5, so episode_length should be ~5
            assert 1 <= metrics["eval/episode_length"] <= 6
        finally:
            evaluator.shutdown()

    def test_batched_async_metrics(self):
        """Async eval with batched env produces correct metrics."""
        env = _make_batched_env(num_envs=4, max_steps=5)
        policy = _make_policy(env)
        evaluator = Evaluator(env, policy, max_steps=20, num_trajectories=4)
        try:
            evaluator.trigger_eval(step=0)
            result = evaluator.wait(timeout=30)
            assert result is not None
            assert result["eval/num_episodes"] == 4
            assert "eval/reward_std" in result
        finally:
            evaluator.shutdown()


class TestEvaluatorErrors:
    def test_no_policy_raises(self):
        with pytest.raises(ValueError, match="policy.*must be provided"):
            Evaluator(_make_env, max_steps=50)

    def test_both_policy_and_factory_raises(self):
        with pytest.raises(ValueError, match="not both"):
            Evaluator(
                _make_env,
                policy=_make_policy(),
                policy_factory=lambda env: _make_policy(env),
                max_steps=50,
            )

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            Evaluator(_make_env, policy=_make_policy(), max_steps=50, backend="invalid")


class TestEvaluatorWeightsDict:
    """Tests for the new weights_dict multi-model weight sync API."""

    def test_weights_dict_policy_only(self):
        """weights_dict={"policy": module} should work same as weights=module."""
        env = _make_env()
        policy = _make_policy(env)
        train_policy = _make_policy(env)
        evaluator = Evaluator(env, policy, max_steps=50)
        try:
            metrics = evaluator.evaluate(weights_dict={"policy": train_policy}, step=0)
            assert "eval/reward" in metrics
        finally:
            evaluator.shutdown()

    def test_weights_backward_compat(self):
        """Old weights=module API still works alongside new weights_dict."""
        env = _make_env()
        policy = _make_policy(env)
        train_policy = _make_policy(env)
        evaluator = Evaluator(env, policy, max_steps=50)
        try:
            metrics = evaluator.evaluate(weights=train_policy, step=0)
            assert "eval/reward" in metrics
        finally:
            evaluator.shutdown()

    def test_weights_and_weights_dict_merged(self):
        """weights goes to 'policy' key if not already in weights_dict."""
        env = _make_env()
        policy = _make_policy(env)
        train_policy = _make_policy(env)
        evaluator = Evaluator(env, policy, max_steps=50)
        try:
            # weights_dict doesn't have "policy", so weights fills it
            metrics = evaluator.evaluate(weights=train_policy, weights_dict={}, step=0)
            assert "eval/reward" in metrics
        finally:
            evaluator.shutdown()

    def test_async_weights_dict(self):
        """trigger_eval accepts weights_dict."""
        env = _make_env()
        policy = _make_policy(env)
        train_policy = _make_policy(env)
        evaluator = Evaluator(env, policy, max_steps=50)
        try:
            evaluator.trigger_eval(weights_dict={"policy": train_policy}, step=0)
            result = evaluator.wait(timeout=30)
            assert result is not None
            assert "eval/reward" in result
        finally:
            evaluator.shutdown()


class _ModuleWithExtraState(nn.Module):
    """Test module that defines get_extra_state/set_extra_state."""

    def __init__(self, dim: int = 4):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
        self.count = torch.tensor(0)

    def forward(self, x):
        return self.linear(x)

    def get_extra_state(self) -> OrderedDict:
        return OrderedDict(
            running_mean=self.running_mean.clone(),
            running_var=self.running_var.clone(),
            count=self.count.clone(),
        )

    def set_extra_state(self, state: OrderedDict) -> None:
        self.running_mean = state["running_mean"]
        self.running_var = state["running_var"]
        self.count = state["count"]


class TestWeightStrategyExtraState:
    """Tests for WeightStrategy tensordict mode capturing get_extra_state."""

    def test_extract_captures_extra_state(self):
        """extract_weights with tensordict mode captures get_extra_state."""
        module = _ModuleWithExtraState(dim=4)
        module.running_mean.fill_(42.0)
        module.count.fill_(100)

        strategy = WeightStrategy(extract_as="tensordict")
        weights = strategy.extract_weights(module)
        assert "__extra_state__" in weights.keys()
        extra = weights["__extra_state__"]
        assert torch.allclose(extra["running_mean"], torch.tensor(42.0).expand(4))
        assert extra["count"].item() == 100

    def test_extract_no_extra_state(self):
        """Modules without get_extra_state should not have __extra_state__."""
        module = nn.Linear(4, 4)
        strategy = WeightStrategy(extract_as="tensordict")
        weights = strategy.extract_weights(module)
        assert "__extra_state__" not in weights.keys()

    def test_apply_restores_extra_state(self):
        """apply_weights with __extra_state__ calls set_extra_state."""
        src = _ModuleWithExtraState(dim=4)
        src.running_mean.fill_(99.0)
        src.running_var.fill_(2.0)
        src.count.fill_(50)

        dst = _ModuleWithExtraState(dim=4)
        assert dst.running_mean.sum().item() == 0.0

        strategy = WeightStrategy(extract_as="tensordict")
        weights = strategy.extract_weights(src)
        strategy.apply_weights(dst, weights, inplace=True)

        assert torch.allclose(dst.running_mean, torch.tensor(99.0).expand(4))
        assert torch.allclose(dst.running_var, torch.tensor(2.0).expand(4))
        assert dst.count.item() == 50

    def test_apply_extra_state_outofplace(self):
        """apply_weights out-of-place also restores extra_state."""
        src = _ModuleWithExtraState(dim=4)
        src.running_mean.fill_(7.0)
        src.count.fill_(3)

        dst = _ModuleWithExtraState(dim=4)

        strategy = WeightStrategy(extract_as="tensordict")
        weights = strategy.extract_weights(src)
        strategy.apply_weights(dst, weights, inplace=False)

        assert torch.allclose(dst.running_mean, torch.tensor(7.0).expand(4))
        assert dst.count.item() == 3

    def test_state_dict_mode_already_captures_extra(self):
        """state_dict mode should already capture extra_state via PyTorch."""
        module = _ModuleWithExtraState(dim=4)
        module.running_mean.fill_(5.0)

        strategy = WeightStrategy(extract_as="state_dict")
        weights = strategy.extract_weights(module)
        # state_dict() includes extra_state under a special key
        assert isinstance(weights, dict)

    def test_roundtrip_preserves_params_and_extra(self):
        """Full roundtrip: extract -> apply preserves both params and extra."""
        src = _ModuleWithExtraState(dim=4)
        nn.init.constant_(src.linear.weight, 3.0)
        nn.init.constant_(src.linear.bias, 1.0)
        src.running_mean.fill_(10.0)
        src.count.fill_(25)

        dst = _ModuleWithExtraState(dim=4)
        nn.init.constant_(dst.linear.weight, 0.0)
        nn.init.constant_(dst.linear.bias, 0.0)

        strategy = WeightStrategy(extract_as="tensordict")
        weights = strategy.extract_weights(src)
        strategy.apply_weights(dst, weights, inplace=True)

        # Check params were updated
        assert torch.allclose(
            dst.linear.weight, torch.tensor(3.0).expand_as(dst.linear.weight)
        )
        assert torch.allclose(
            dst.linear.bias, torch.tensor(1.0).expand_as(dst.linear.bias)
        )
        # Check extra_state was updated
        assert torch.allclose(dst.running_mean, torch.tensor(10.0).expand(4))
        assert dst.count.item() == 25


class TestEvaluatorProcessBackendAsMultiCollector:
    """Tests that backend='process' uses MultiSyncCollector internally."""

    def test_process_backend_sync_eval(self):
        """Process backend sync eval works via MultiSyncCollector."""
        evaluator = Evaluator(
            _make_env,
            policy_factory=_make_policy,
            max_steps=50,
            backend="process",
        )
        try:
            metrics = evaluator.evaluate(step=0)
            assert "eval/reward" in metrics
            assert "eval/episode_length" in metrics
        finally:
            evaluator.shutdown()

    def test_process_backend_async_eval(self):
        """Process backend async eval works via MultiSyncCollector."""
        evaluator = Evaluator(
            _make_env,
            policy_factory=_make_policy,
            max_steps=50,
            backend="process",
        )
        try:
            evaluator.trigger_eval(step=0)
            result = evaluator.wait(timeout=30)
            assert result is not None
            assert "eval/reward" in result
        finally:
            evaluator.shutdown()

    def test_process_backend_with_weights(self):
        """Process backend with weight transfer works."""
        train_policy = _make_policy()
        evaluator = Evaluator(
            _make_env,
            policy_factory=_make_policy,
            max_steps=50,
            backend="process",
        )
        try:
            metrics = evaluator.evaluate(weights=train_policy, step=0)
            assert "eval/reward" in metrics
        finally:
            evaluator.shutdown()

    def test_process_backend_requires_callable(self):
        """Process backend requires env to be a callable."""
        env = _make_env()
        with pytest.raises(ValueError, match="callable"):
            Evaluator(
                env,
                policy_factory=_make_policy,
                max_steps=50,
                backend="process",
            )

    def test_process_backend_requires_policy_factory(self):
        """Process backend requires policy_factory."""
        with pytest.raises(ValueError, match="policy_factory"):
            Evaluator(
                _make_env,
                policy=_make_policy(),
                max_steps=50,
                backend="process",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
