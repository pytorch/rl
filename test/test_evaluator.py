# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import threading
import time

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.collectors import Evaluator
from torchrl.testing.mocking_classes import ContinuousActionVecMockEnv


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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
