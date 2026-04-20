# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools

import json
import os
import re
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING

import numpy as np

import torch
from tensordict import NestedKey, NonTensorData, TensorDict, TensorDictBase
from tensordict.tensorclass import is_non_tensor

from torchrl._utils import logger as torchrl_logger
from torchrl.data import Choice, Composite, NonTensor
from torchrl.data.llm import History
from torchrl.envs import ConditionalSkip, GymWrapper, Transform, TransformedEnv

if TYPE_CHECKING:
    import mlgym
    import transformers

# Inv transforms:
#  Transforms to apply prior to pass the model output to the env


@contextmanager
def _temp_cwd_mlgym():
    """Temporarily change the current working directory to mlgym."""
    import mlgym

    path = Path(mlgym.__spec__.submodule_search_locations[0]).parent
    old_pwd = os.getcwd()
    os.chdir(str(path))
    # sys.path.insert(-1, "mlgym")
    try:
        yield
    finally:
        # sys.path.pop()
        os.chdir(old_pwd)


class MLGymBaseTransform(Transform):
    """Base class for all MLGym transforms."""

    @property
    def config(self):
        return self.parent.base_env.config

    @property
    def system_args(self):
        return {
            "command_docs": self.config.tools_handler.command_docs,
            **self.config.tools_handler.env_variables,
        }

    @property
    def task_args(self):
        # Placeholder
        task_args = getattr(self, "_task_args", None)
        if task_args is None:
            return self.parent.base_env.task.args
        return task_args

    @task_args.setter
    def task_args(self, task_args):
        self._task_args = task_args

    @property
    def name(self):
        return "torchrl"

    @property
    def state_command(self):
        return self.config.state_command.name

    @property
    def agent_args(self):
        return self.parent.base_env.agent_args

    @property
    def model_name(self) -> Literal["human", "human_thought"]:
        return self.agent_args.model.model_name


#######################################################
# Forward transforms: Format the env output


# Transform #0: Resets the env
class ResetModule(MLGymBaseTransform):
    """Runs setup pipeline and enables multi-resets.

    The reset method reads the 'system' initial input from the config and parses it to a History
    object.

    """

    response_key: NestedKey = "text_response"

    def __init__(self):
        super().__init__(in_keys=[], out_keys=["history"])

    @_temp_cwd_mlgym()
    def _reset_env_preprocess(self, tensordict: TensorDictBase) -> TensorDictBase:
        base_env = self.parent.base_env._env
        if tensordict is not None and "task" in tensordict:
            import gymnasium as gym

            task = tensordict["task"]
            torchrl_logger.info(f"Resetting with {task=}")
            if is_non_tensor(task):
                task = task.data
            task_id, agent_args = _TASK_IDS[task]
            try:
                base_env.close()
            except Exception:
                torchrl_logger.info(f"Failed to close {base_env=}")
            base_env = gym.make(
                f"mlgym/{task}",
                devices=["cpu_0"],
            ).unwrapped
            base_env.config = agent_args.config
            self.parent.base_env.set_env(base_env)
        base_env.reset_container()
        base_env.communicate(f"cd {Path(base_env.task_workspace).parent}")
        return tensordict

    @_temp_cwd_mlgym()
    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        # TODO: what to do with this?
        # reset model stats
        # self.model.reset_stats(init_model_stats)
        # env = self.parent.base_env._env

        env = self.parent.base_env._env
        self.set_environment_vars(env, self.config.env_variables)

        system_msg = self.config.system_template.format(
            **self.system_args, **asdict(self.task_args)
        )
        # self.logger.log(self._default_logging_level, f"SYSTEM ({self.name})\n{system_msg}")
        history = History(
            role="system",
            content=system_msg,  # agent=self.name,
            batch_size=(1,),
            device=self.parent.device,
        )
        tensordict_reset["history"] = history

        return tensordict_reset

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        # Placeholder
        if "history" not in next_tensordict:
            if "local_history" in tensordict:
                local_history = tensordict["local_history"]
            else:
                local_history = None
            history = tensordict["history"]
            if local_history is not None:
                history = history.append(local_history, inplace=False)
                tensordict["history"] = history
            next_tensordict["history"] = history
        return next_tensordict

    def set_environment_vars(
        self, env: MLGymWrapper, env_variables: dict[str, Any]
    ) -> None:
        commands_to_execute = (
            [self.config.state_command.code]
            +  # [code for code in self.config.util_functions] +
            # [command.code for command in self.config._commands] +
            [f"{k}={v}" for k, v in env_variables.items()]
        )
        commands = "\n".join(commands_to_execute)
        try:
            output = env.communicate(commands)
            if env.returncode != 0:
                msg = f"Nonzero return code: {env.returncode}\nOutput: {output}"
                raise RuntimeError(msg)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            raise e
        command_files = []
        for file in self.config.command_files:
            datum = {}
            with open(file) as f:
                contents = f.read()
            datum["contents"] = contents
            filename = Path(file).name
            if not contents.strip().startswith("#!"):
                if filename.endswith(".sh"):
                    # files are sourced, so they are not executable
                    datum["name"] = Path(file).name
                    datum["type"] = "source_file"
                elif filename.startswith("_"):
                    # files are sourced, so they are not executable
                    datum["name"] = Path(file).name
                    datum["type"] = "utility"
                else:
                    msg = (
                        f"Non-shell script file {file} does not start with shebang.\n"
                        "Either add a shebang (#!) or change the file extension to .sh if you want to source it.\n"
                        "You can override this behavior by adding an underscore to the file name (e.g. _utils.py)."
                    )
                    raise ValueError(msg)
            else:
                # scripts are made executable
                datum["name"] = Path(file).name.rsplit(".", 1)[0]
                datum["type"] = "script"
            command_files.append(datum)
        # TODO: implement add commands method in environment
        env.add_commands(command_files)

    def transform_observation_spec(self, observation_spec: Composite) -> Composite:
        observation_spec["history"] = History.default_spec()
        return observation_spec

    def transform_action_spec(self, action_spec: Composite) -> Composite:
        if isinstance(action_spec, Composite):
            action_spec[self.response_key] = self.transform_action_spec(
                action_spec[self.response_key]
            )
            return action_spec
        # make the "random" action just a choice between innocuous bash commands
        return Choice(
            [
                NonTensor(example_data="ls -rtlh", shape=action_spec.shape),
                NonTensor(example_data="pwd", shape=action_spec.shape),
            ]
        )

    def transform_state_spec(self, state_spec: Composite) -> Composite:
        state_spec["history"] = History.default_spec()
        return state_spec


class TaskSampler(Transform):
    """A sampler for tasks in a certain task set."""

    def __init__(self, tasks: list[str]):
        super().__init__()
        self.tasks = tasks

    def transform_observation_spec(self, observation_spec: Composite) -> Composite:
        observation_spec["task"] = NonTensor(example_data="<a task>", shape=())
        return observation_spec

    @_temp_cwd_mlgym()
    def _reset_env_preprocess(
        self, tensordict: TensorDictBase | None
    ) -> TensorDictBase:
        if tensordict is None:
            tensordict = TensorDict(batch_size=self.parent.batch_size)
        # Sample a task
        task = np.random.choice(self.tasks)
        tensordict["task"] = NonTensorData(task)
        self._current_task = task
        return tensordict

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        next_tensordict["task"] = self._current_task
        return next_tensordict


# Transform #1: env -> state
class ReadState(MLGymBaseTransform):
    """Reads current state and writes it as a parsable str in the tensordict."""

    # from mlgym/agent/base.py:BaseAgent:forward_model
    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        base_mlgym_env = self.parent.base_env  # getattr is forwarded

        command = self.state_command
        state = base_mlgym_env.communicate(command) if self.state_command else None

        next_tensordict["state"] = state
        return next_tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        # tensordict_reset.setdefault("message", NonTensorData(""))
        # tensordict_reset.setdefault("state", NonTensorData(""))
        return self._step(tensordict_reset, tensordict_reset)

    def transform_observation_spec(self, observation_spec):
        observation_spec.set(
            "state",
            NonTensor(
                example_data="a string",
                device=observation_spec.device,
                shape=observation_spec.shape,
            ),
        )
        return observation_spec


# Transform #2: state -> message
class StateToMessage(MLGymBaseTransform):
    """Parses the string using json to a given template.

    Requires:
        - a 'state' key from the ReadState transform
        - an 'observation' key from the base environment

    """

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        base_mlgym_env = self.parent.base_env  # getattr is forwarded
        observation = tensordict["observation"]
        state = tensordict["state"]
        config = self.config

        current_step = base_mlgym_env.current_step
        max_steps = base_mlgym_env.max_steps
        try:
            state_vars = json.loads(state)
        except json.JSONDecodeError as e:
            msg = f"State {state!r} is not valid json. This is an internal error, please report it."
            raise ValueError(msg) from e
        # add step information to state_vars
        state_vars["current_step"] = current_step
        state_vars["remaining_steps"] = max_steps - current_step

        # FIXME: we don't need to do this, we have our own observation space
        # Determine observation template based on what prior observation was

        history: History = tensordict["history"]
        if history[..., -1].role == "system":
            # Show task template if prev. obs. was initial system message
            templates = [config.task_template]
            if config.strategy_template is not None:
                templates.append(config.strategy_template)
        elif observation is None or observation.strip() == "":
            # Show no output template if observation content was empty
            assert config.next_step_no_output_template is not None  # linting
            templates = [config.next_step_no_output_template]
        else:
            # Show standard output template if there is observation content
            assert config.next_step_template is not None  # linting
            templates = [config.next_step_template]

        # Format selected template(s) with information
        messages = []
        assert self.task_args is not None
        for template in templates:
            messages.append(
                template.format(
                    **asdict(self.task_args),
                    **self.system_args,
                    **state_vars,
                    observation=(observation if observation is not None else ""),
                    # missing forwarded_vars because no attempts
                ),
            )

        message = "\n".join(messages)
        next_tensordict["message"] = message
        # model query hooks here
        return next_tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        # tensordict_reset.setdefault("message", NonTensorData(""))
        # tensordict_reset.setdefault("state", NonTensorData(""))
        return self._step(tensordict_reset, tensordict_reset)

    def transform_observation_spec(self, observation_spec):
        observation_spec.set(
            "message",
            NonTensor(
                example_data="a string",
                device=observation_spec.device,
                shape=observation_spec.shape,
            ),
        )
        return observation_spec


# Transform #3: Append message to history
class MessageToHistory(MLGymBaseTransform):
    """Parses the message string to a History object, then reparses the history to a complete message.

    .. seealso:: HistoryToMessage

    """

    def __init__(self):
        super().__init__(in_keys=["message", "history"], out_keys=["history", "chat"])

    # from mlgym/agent/base.py:BaseAgent:local_history
    # from mlgym/agent/base.py:BaseAgent:_append_history
    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        # From PrepareDataForModel
        message: str = next_tensordict["message"]
        # from mlgym/agent/base.py:BaseAgent:forward_model
        history = tensordict["history"]
        cur_history = History(
            role="user", content=message, batch_size=(), device=self.parent.device
        )
        # This is the basic thing our transform does: append the history to the existing one.
        # (We should be able to extend the lazy stack directly)
        history = history.append(cur_history, inplace=False)

        next_tensordict["history"] = history
        return next_tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._step(tensordict_reset, tensordict_reset)


# Inverse transforms:
#  Format the action from the model for the env


class TemplateTransform(MLGymBaseTransform):
    """A transform to apply the chat template to the History."""

    response_key: NestedKey = "text_response"
    prompt_key: NestedKey = "text"

    # alternative to DummyFormat, wip
    def __init__(
        self,
        in_keys=None,
        out_keys=None,
        in_keys_inv=None,
        out_keys_inv=None,
        tokenizer=None,
        chat_template_name: Literal["chatml_format"] | None = None,
        continue_final_message: bool = False,
        tokenize: bool = False,
        return_tensors: str = "pt",
        return_dict: bool = False,
        padding: bool | str = False,
        truncation: bool | str = False,
    ):
        super().__init__(
            in_keys=["history"] if in_keys is None else in_keys,
            out_keys=[self.prompt_key] if out_keys is None else out_keys,
            in_keys_inv=[self.prompt_key, self.response_key]
            if in_keys_inv is None
            else in_keys_inv,
            # TODO: we should not use the response key here but another dedicated entry, like "action_parsed"
            out_keys_inv=[self.response_key] if out_keys_inv is None else out_keys_inv,
        )
        self.chat_template_name = chat_template_name
        self.tokenizer = tokenizer
        self.tokenize = tokenize
        self.continue_final_message = continue_final_message
        self.return_tensors = return_tensors
        self.return_dict = return_dict
        self.padding = padding
        self.truncation = truncation

    def transform_observation_spec(self, observation_spec: Composite):
        observation_spec[self.prompt_key] = NonTensor(
            example_data="<some chat string>",
            shape=observation_spec.shape,
            device=observation_spec.device,
        )
        return observation_spec

    @property
    def _chat_template(self):
        chat_template = None
        if self.chat_template_name:
            from torchrl.data.llm.datatypes.chat import _CHAT_TEMPLATES

            chat_template = _CHAT_TEMPLATES[self.chat_template_name]
        elif self.tokenizer.chat_template is not None:
            chat_template = self.tokenizer.chat_template
        elif chat_template is None:
            raise ValueError("Failed to determine chat template.")
        return chat_template

    def _apply_transform(self, history: History) -> NonTensorData:
        if self.tokenizer is None:
            raise RuntimeError("Cannot apply chat template without a tokenizer.")
        result = history.apply_chat_template(
            tokenizer=self.tokenizer,
            add_generation_prompt=True,
            chat_template=self._chat_template,
            continue_final_message=self.continue_final_message,
            tokenize=self.tokenize,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=self.return_tensors,
        )
        return result

    def _reset(self, tensordict, tensordict_reset):
        return self._call(tensordict_reset)

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.in_keys_inv:
            prompt = tensordict[self.prompt_key]
            response = tensordict[self.response_key]
            if isinstance(prompt, list):
                action = [
                    prompt + response for prompt, response in zip(prompt, response)
                ]
            else:
                action = prompt + response
            try:
                history, action = self._inv_apply_transform(action)
                tensordict["local_history"] = history
                tensordict[self.response_key] = action
            except RuntimeError as e:
                if "Expected assistant role" in str(e):
                    tensordict["local_history"] = History(role="assistant", content="")
                    tensordict[self.response_key] = ""
        return tensordict

    def _inv_apply_transform(self, action):
        if self.tokenize:
            action = self.tokenizer.decode(action)

        if not isinstance(action, (str, list)):
            action = action.data
            history, action = self._inv_apply_transform(action)
            action = NonTensorData(
                action, batch_size=action.batch_size, device=action.device
            )
            return history, action

        history = History.from_text(
            action,
            # chat_template=self._chat_template,
        )[..., -1]
        if history.role != "assistant":
            raise RuntimeError(f"Expected assistant role, got {history.role=}")
        action = history.get("content")
        return history, action


class IsolateCodeBlock(MLGymBaseTransform):
    """A transform that isolates the code block in the action generated by the LLM.

    Optionally, wrongly formatted actions are assigned a negative reward.
    """

    response_key: NestedKey = "text_response"

    def __init__(self, reward_wrong_format: float | None = None):
        super().__init__(
            in_keys_inv=[self.response_key], out_keys_inv=[self.response_key]
        )
        from mlgym.agent.parsing import ThoughtActionParser

        self.parser = ThoughtActionParser()
        self.reward_wrong_format = reward_wrong_format
        self._assign_reward = False

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        torchrl_logger.info("inv call with IsolateCodeBlock")
        action = tensordict[self.response_key]
        # if we didn't find an action, the action is empty
        if not action:
            torchrl_logger.info(
                "Did not find a suitable action, skipping the call to step."
            )
            tensordict["retry"] = torch.ones(tensordict.shape, dtype=torch.bool)
            self._assign_reward = True
        else:
            from mlgym.exceptions import FormatError

            try:
                action = self._inv_apply_transform(action)
                tensordict[self.response_key] = action
                torchrl_logger.info(f"Code block: {action}")
                tensordict["retry"] = torch.zeros(tensordict.shape, dtype=torch.bool)
                self._assign_reward = False
            except FormatError:
                tensordict["retry"] = torch.ones(tensordict.shape, dtype=torch.bool)
                self._assign_reward = True
        return tensordict

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self._assign_reward:
            torchrl_logger.info(
                f"Assigning penalty for unsuitable action: {self.reward_wrong_format}"
            )
            if self.reward_wrong_format is not None:
                tensordict[self.parent.reward_key] += self.reward_wrong_format
        return tensordict

    def _inv_apply_transform(self, action):
        if not isinstance(action, (str, list)):
            return NonTensorData(
                self._inv_apply_transform(action.tolist()),
                batch_size=action.batch_size,
                device=action.device,
            )
        if isinstance(action, list):
            return [self._inv_apply_transform(action) for action in action]
        thought, action = self.parser(action, None)
        return action


class EvaluationOutputParser:
    """Parser for the reward transform in MLGym.

    .. seealso:: :class:`~torchrl.envs.llm.libs.mlgym.MLGymRewardAssignment`

    """

    def __init__(self):
        # Regular expressions to match the required fields
        self.patterns = {
            "submission_artefact_path": r"valid submission artefact at (.*)\.",
            "baseline_score": r"Baseline Score: \{'Score': (.*)\}",
            "evaluation_score": r"Evaluation Score: \{'Score': (.*)\}",
            "current_step": r"\(Current Step: (\d+),",
            "remaining_steps": r"Remaining Steps: (\d+)\)",
            "open_file": r"\(Open file: (.*)\)",
            "current_directory": r"\(Current directory: (.*)\)",
        }

    def __call__(self, output_string):

        parsed_data = {}

        for key, pattern in self.patterns.items():
            match = re.search(pattern, output_string)
            if match:
                parsed_data[key] = match.group(1).strip()
        if "baseline_score" in parsed_data:
            parsed_data["baseline_score"] = float(parsed_data["baseline_score"])

        if "evaluation_score" in parsed_data:
            parsed_data["evaluation_score"] = float(parsed_data["evaluation_score"])
        if "current_step" in parsed_data:
            parsed_data["current_step"] = int(parsed_data["current_step"])
        if "remaining_steps" in parsed_data:
            parsed_data["remaining_steps"] = int(parsed_data["remaining_steps"])

        return parsed_data


class MLGymRewardAssignment(MLGymBaseTransform):
    """Reward assignment through parsing of the last item in history.

    By default, the :class:`~torchrl.envs.llm.libs.mlgym.EvaluationOutputParser` class is used as parser.

    """

    def __init__(self):
        super().__init__(in_keys=["reward", "history"], out_keys=["reward"])
        self.parser = EvaluationOutputParser()

    def _call(self, tensordict):
        history = tensordict.get("history")
        if history is None:
            raise KeyError(f"History is missing in tensordict {tensordict}")
        if history.ndim != 1:
            raise ValueError(f"History shape must be 1D, got {history.shape}")
        content = history[-1].content
        torchrl_logger.info(f"Parsing reward from: {content}")
        parsed = self.parser(content)
        reward = parsed.get("evaluation_score", 0.0) - parsed.get("baseline_score", 0.0)
        torchrl_logger.info(f"Parsed reward: {reward}")
        tensordict["reward"] = tensordict["reward"] + reward
        return tensordict


class _add_info_to_reset:
    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs), {}


class _add_truncated_to_step:
    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func

    @_temp_cwd_mlgym()
    def __call__(self, *args, **kwargs):
        obs, r, done, info = self.func(*args, **kwargs)
        return obs, r, done, False, info


class MLGymWrapper(GymWrapper):
    """A thin wrapper for MLGym environments.

    This specialized :class:`~torchrl.envs.GymWrapper` subclass defines the observation space with `observation=NonTensor()`
    and the action space with `text_response=NonTensor()`, according to the :class:`~torchrl.envs.llm.ChatEnv` API.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_action_spec = Composite(
            text_response=NonTensor(example_data="<a string>", shape=())
        )
        self.full_observation_spec = Composite(
            observation=NonTensor(example_data="<a string>", shape=())
        )
        self.set_env()

    def set_env(self, env: Any = None):
        if env is not None:
            self._env = env
        self._patch_reset()
        self._patch_step()

    def _patch_reset(self):
        if not isinstance(self._env.reset, _add_info_to_reset):
            self._env.reset = _add_info_to_reset(self._env.reset)

    def _patch_step(self):
        if not isinstance(self._env.reset, _add_truncated_to_step):
            self._env.step = _add_truncated_to_step(self._env.step)

    @_temp_cwd_mlgym()
    def _reset(
        self, tensordict: TensorDictBase | None = None, **kwargs
    ) -> TensorDictBase:
        return super()._reset(tensordict=tensordict, **kwargs)


_TASK_IDS = {}


def get_args(
    task: Literal["prisonersDilemma"] = "prisonersDilemma",
) -> tuple[
    mlgym.environment.env.EnvironmentArguments,  # noqa
    mlgym.agent.base.AgentArguments,  # noqa
]:  # noqa
    """Parse command line arguments and return a ScriptArguments object.

    Args:
        args: Optional list of arguments to parse. If not provided, uses sys.argv.
    """
    import mlgym.environment.registration  # noqa
    from mlgym import CONFIG_DIR
    from mlgym.agent.base import AgentArguments
    from mlgym.backend.base import ModelArguments
    from mlgym.environment.env import EnvironmentArguments
    from mlgym.environment.registration import register_task

    environment_args = EnvironmentArguments(
        task_config_path=f"tasks/{task}.yaml",
        max_steps=10,
        seed=42,
        container_type="podman",
        verbose=False,
        aliases_file="docker/aliases.sh",
    )

    agent_args = AgentArguments(
        # placeholder
        model=ModelArguments(""),
        # Despite using torchrl as an agent, we still need the agent config - see StateToMessage parser
        agent_config_path=CONFIG_DIR / "agents" / "default.yaml",
    )

    register_task(environment_args)

    _TASK_IDS[task] = (environment_args.task.id, agent_args)

    return environment_args, agent_args


def make_mlgym(
    *,
    task: Literal["prisonersDilemma"] | None = None,
    tasks: list[Literal["prisonersDilemma"]] | None = None,
    tokenizer: transformers.AutoTokenizer | str | None = None,  # noqa
    device="cpu",
    reward_wrong_format: float | None = None,
) -> TransformedEnv:
    """Wraps an MLGymEnv in a TorchRL Environment.

    The appended transforms will make sure that the data is formatted for the LLM during (for the outputs of `env.step`)
    and for the MLGym API (for inputs to `env.step`).

    Keyword Args:
        task (str): The task to wrap. Exclusive with `tasks` argument.

            .. note:: The correct format is simply the task name, e.g., `"prisonersDilemma"`.

        tasks (List[str]): The tasks available for the env. Exclusive with `task` argument.

            .. note:: The correct format is simply the task name, e.g., `"prisonersDilemma"`.

        tokenizer (transformers.AutoTokenizer or str, optional): A transformer that tokenizes the data.
            If a string is passed, it will be converted to a `transformers.AutoTokenizer`.
        device (str, optional): The device to set to the env. Defaults to "cpu".
        reward_wrong_format (float, optional): The reward (negative penalty) for wrongly formatted actions.
            Defaults to `None` (no penalty).

    """
    import gymnasium as gym

    if isinstance(tokenizer, str):
        import transformers

        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)

    with _temp_cwd_mlgym():

        if task and not tasks:
            environment_args, agent_args = get_args(task=task)
        elif tasks and not task:
            for task in tasks:
                environment_args, agent_args = get_args(task=task)
        else:
            raise ValueError(
                f"Either task or tasks should be provided, not both and not none. Got {task=} and {tasks=}."
            )

        base_env = gym.make(
            f"mlgym/{_TASK_IDS[task][0]}",
            devices=["cpu_0"],
        ).unwrapped
        # we need the env to have access to the config
        base_env.config = agent_args.config
        env = TransformedEnv(
            MLGymWrapper(base_env, auto_reset=False, device=device), auto_unwrap=False
        )

        env.append_transform(ConditionalSkip(lambda td: td["retry"]))
        env.append_transform(IsolateCodeBlock(reward_wrong_format=reward_wrong_format))

        env.append_transform(ResetModule())
        if tasks:
            # Add a task sampler
            env.append_transform(TaskSampler(tasks))
        env.append_transform(ReadState())
        env.append_transform(StateToMessage())
        env.append_transform(MessageToHistory())
        env.append_transform(TemplateTransform(tokenizer=tokenizer))
        env.append_transform(MLGymRewardAssignment())
        # # We want the env to have a batch-size of (1,) because it will be easier to interact with
        # #  LLMs
        # env.append_transform(BatchSizeTransform(batch_size=(1,)))
        return env
