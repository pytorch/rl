# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

import torchrl.envs.libs.openenv as openenv_mod
import torchrl.envs.llm.libs.openenv as openenv_chat_mod
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData
from torchrl.data import LazyStackStorage, ReplayBuffer
from torchrl.data.llm import History
from torchrl.envs.libs.openenv import OpenEnvEnv, OpenEnvWrapper
from torchrl.envs.llm import OpenEnvChatEnv
from torchrl.envs.utils import check_env_specs
from torchrl.modules.llm.policies.common import ChatHistory
from torchrl.objectives.llm.grpo import MCAdvantage

_has_omegaconf = importlib.util.find_spec("omegaconf") is not None


@dataclass
class _StepResult:
    observation: object
    reward: float | None = None
    done: bool = False


class _TextAction:
    model_fields = {"text": object()}

    def __init__(self, text):
        self.text = text


class _ObservationModel:
    def model_dump(self):
        return {"prompt": ["nested", {"value": 1}], "reward": 2.0, "done": True}


class _Config(dict):
    def __init__(self, data):
        super().__init__((key, self._convert(value)) for key, value in data.items())

    @classmethod
    def _convert(cls, value):
        if isinstance(value, dict):
            return cls(value)
        return value

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as err:
            raise AttributeError(name) from err


class _SyncOpenEnv:
    def __init__(self):
        self.connected = False
        self.closed = False
        self.seed_value = None
        self.count = 0
        self.last_action = None

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.closed = True
        self.connected = False

    def seed(self, seed):
        self.seed_value = seed

    def state(self):
        return {"count": self.count}

    def reset(self):
        self.count = 0
        return _StepResult({"prompt": "say hello"})

    def step(self, action):
        self.last_action = action
        self.count += 1
        return _StepResult(
            {"prompt": f"turn {self.count}", "seen": getattr(action, "text", action)},
            reward=float(self.count),
            done=self.count >= 2,
        )


class _AsyncLikeOpenEnv:
    def __init__(self):
        self.sync_called = False
        self.sync_env = _SyncOpenEnv()

    def sync(self):
        self.sync_called = True
        return self.sync_env


def _make_chat_policy(text, *, use_full=False):
    def policy(td):
        prompt = td["history"].prompt
        response = History(
            role=["assistant"],
            content=[text],
            tool_calls=[None],
            tool_responses=[None],
            batch_size=(1,),
        ).unsqueeze(-1)
        chat = ChatHistory._from_tensordict(TensorDict({}, batch_size=(1,)))
        chat.prompt = prompt
        if use_full:
            chat.full = prompt.extend(response, inplace=False, dim=-1)
        else:
            chat.response = response
        return TensorDict({"history": chat}, batch_size=(1,))

    return policy


class TestOpenEnvGeneric:
    def test_missing_dependency_error(self, monkeypatch):
        monkeypatch.setattr(openenv_mod, "_has_openenv", False)
        with pytest.raises(ImportError, match="openenv python package"):
            OpenEnvEnv("missing-env")

    def test_sync_conversion_reset_step_state_seed_close(self):
        async_env = _AsyncLikeOpenEnv()
        env = OpenEnvWrapper(env=async_env, include_state=True)
        assert async_env.sync_called
        assert async_env.sync_env.connected

        env.set_seed(123)
        assert async_env.sync_env.seed_value == 123

        td = env.reset()
        assert td["observation"] == {"prompt": "say hello"}
        assert "reward" not in td.keys()
        assert td["state"] == {"count": 0}

        td["action"] = NonTensorData("hello", batch_size=[])
        out = env.step(td)
        assert out["next", "reward"].item() == 1.0
        assert out["next", "done"].item() is False
        assert async_env.sync_env.last_action == "hello"
        assert out["next", "state"] == {"count": 1}

        env.close()
        assert async_env.sync_env.closed

    def test_action_cls_mapping_and_observation_done_fallback(self):
        class ObservationDoneEnv(_SyncOpenEnv):
            def step(self, action):
                self.last_action = action
                return SimpleNamespace(observation={"done": True, "reward": 3.0})

        raw_env = ObservationDoneEnv()
        env = OpenEnvWrapper(
            env=raw_env,
            action_cls=_TextAction,
            return_observation_dict=True,
        )
        td = env.reset()
        td["action"] = NonTensorData({"text": "from-dict"}, batch_size=[])
        out = env.step(td)
        assert isinstance(raw_env.last_action, _TextAction)
        assert raw_env.last_action.text == "from-dict"
        assert out["next", "reward"].item() == 3.0
        assert out["next", "done"].item() is True

    def test_model_dump_observation_and_nested_values(self):
        class PydanticObservationEnv(_SyncOpenEnv):
            def reset(self):
                return _StepResult(_ObservationModel())

            def step(self, action):
                self.last_action = action
                return SimpleNamespace(observation=_ObservationModel())

        env = OpenEnvWrapper(env=PydanticObservationEnv(), return_observation_dict=True)
        td = env.reset()
        assert td["observation"] == {
            "prompt": ["nested", {"value": 1}],
            "reward": 2.0,
            "done": True,
        }
        td["action"] = NonTensorData("hello", batch_size=[])
        out = env.step(td)
        assert out["next", "observation"]["prompt"][1]["value"] == 1
        assert out["next", "reward"].item() == 2.0
        assert out["next", "done"].item() is True

    def test_check_env_specs_and_rollout(self):
        env = OpenEnvWrapper(env=_SyncOpenEnv())
        check_env_specs(env)
        rollout = env.rollout(3, break_when_any_done=False, return_contiguous=False)
        assert ("next", "observation") in rollout.keys(True)


class TestOpenEnvChat:
    @pytest.mark.parametrize("use_full", [False, True])
    def test_history_response_to_typed_action(self, use_full):
        raw_env = _SyncOpenEnv()
        env = OpenEnvChatEnv(
            env=raw_env,
            action_cls=_TextAction,
            history_content_adapter=lambda obs: obs["prompt"],
        )
        td = env.reset()
        assert td["history"].prompt.content[0][0] == "say hello"
        assert td["query"] == "say hello"
        out = env.step(_make_chat_policy('{"text": "hello"}', use_full=use_full)(td))
        assert isinstance(raw_env.last_action, _TextAction)
        assert raw_env.last_action.text == "hello"
        assert out["next", "query"] == "say hello"
        assert out["next", "history"].prompt.role[0][-1] == "user"
        assert out["next", "history"].prompt.content[0][-1] == "turn 1"

    def test_rand_step_check_env_specs_and_rollout(self):
        env = OpenEnvChatEnv(
            env=_SyncOpenEnv(), history_content_adapter=lambda obs: obs["prompt"]
        )
        env.rand_step(env.reset())
        env.check_env_specs(break_when_any_done="both", return_contiguous=False)
        rollout = env.rollout(
            2,
            policy=_make_chat_policy("hello"),
            break_when_any_done=False,
            return_contiguous=False,
        )
        assert ("next", "history") in rollout.keys(True)
        assert "query" in rollout.keys()
        assert ("next", "reward") in rollout.keys(True)


class TestOpenEnvGRPO:
    def test_make_env_openenv_with_local_fixture(self, monkeypatch):
        if not _has_omegaconf:
            pytest.skip("omegaconf is required to import the GRPO recipe helpers")
        pytest.importorskip("transformers")
        pytest.importorskip("openenv")
        spec = importlib.util.spec_from_file_location(
            "grpo_utils", Path("sota-implementations/grpo/grpo_utils.py")
        )
        grpo_utils = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(grpo_utils)

        monkeypatch.setattr(grpo_utils, "get_tokenizer", lambda cfg: object())
        monkeypatch.setattr(
            openenv_chat_mod.AutoEnv,
            "from_env",
            staticmethod(lambda name, **kwargs: _SyncOpenEnv()),
        )
        monkeypatch.setattr(
            openenv_mod.AutoAction,
            "from_env",
            staticmethod(lambda name: _TextAction),
        )
        cfg = _Config(
            {
                "env": {
                    "dataset": "openenv",
                    "num_envs": 1,
                    "repeats": 2,
                    "reasoning": True,
                    "max_steps": 2,
                    "openenv": {
                        "name": "local-fixture",
                        "env_kwargs": {},
                        "reward_threshold": 0.0,
                        "return_observation_dict": True,
                        "history_content_adapter": None,
                        "action_adapter": None,
                        "observation_adapter": None,
                        "action_cls": None,
                        "system_prompt": None,
                        "max_steps": 2,
                    },
                }
            }
        )
        env = grpo_utils.make_env(cfg, single_env=True)
        td = env.reset()
        assert "history" in td.keys()
        assert "query" in td.keys()
        response = History(
            role=["assistant"],
            content=['{"text": "hello"}'],
            tool_calls=[None],
            tool_responses=[None],
            batch_size=(1,),
        ).unsqueeze(-1)
        chat = ChatHistory._from_tensordict(TensorDict({}, batch_size=(1,)))
        chat.prompt = td["history"].prompt
        chat.response = response
        td = td.clone()
        td["history"] = chat
        out = env.step(td)
        assert out["next", "reward"].item() == 1.0
        assert out["next", "done"].item() is False

    def test_mcadvantage_smoke_for_openenv_rollouts(self):
        def collect_rollout():
            env = OpenEnvChatEnv(
                env=_SyncOpenEnv(), history_content_adapter=lambda obs: obs["prompt"]
            )
            rollout = env.rollout(
                2,
                policy=_make_chat_policy("hello"),
                break_when_any_done=False,
                return_contiguous=False,
            )
            rollout = rollout.squeeze(0)
            return rollout

        rb = ReplayBuffer(storage=LazyStackStorage(4))
        advantage = MCAdvantage(
            grpo_size=2, prompt_key="query", trajectory_return="sum"
        )
        assert advantage.inv(collect_rollout()) is None
        out = advantage.inv(collect_rollout())
        rb.extend(out)
        sample = rb.sample(2)
        assert "advantage" in sample.keys()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
