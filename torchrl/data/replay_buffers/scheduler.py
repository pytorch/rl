# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
import torch
from torchrl.data.replay_buffers.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import Sampler


class ParameterScheduler(ABC):
    """Scheduler to adjust the value of a given parameter of a replay buffer's sampler.

    Scheduler can for example be used to alter the alpha and beta values in the PrioritizedSampler.

    Args:
        obj (ReplayBuffer or Sampler): the replay buffer or sampler whose sampler to adjust
        param_name (str): the name of the attribute to adjust, e.g. `beta` to adjust the beta parameter
        min_value (Union[int, float], optional): a lower bound for the parameter to be adjusted
            Defaults to `None`.
        max_value (Union[int, float], optional): an upper bound for the parameter to be adjusted
            Defaults to `None`.

    """

    def __init__(
        self,
        obj: ReplayBuffer | Sampler,
        param_name: str,
        min_value: int | float | None = None,
        max_value: int | float | None = None,
    ):
        if not isinstance(obj, (ReplayBuffer, Sampler)):
            raise TypeError(
                f"ParameterScheduler only supports Sampler class. Pass either `ReplayBuffer` or `Sampler` object. Got {type(obj)} instead."
            )
        self.sampler = obj.sampler if isinstance(obj, ReplayBuffer) else obj
        self.param_name = param_name
        self._min_val = min_value or float("-inf")
        self._max_val = max_value or float("inf")
        if not hasattr(self.sampler, self.param_name):
            raise ValueError(
                f"Provided class {type(obj).__name__} does not have an attribute {param_name}"
            )
        initial_val = getattr(self.sampler, self.param_name)
        if isinstance(initial_val, torch.Tensor):
            initial_val = initial_val.clone()
            self.backend = torch
        else:
            self.backend = np
        self.initial_val = initial_val
        self._step_cnt = 0

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in ``self.__dict__`` which
        is not the sampler.
        """
        sd = dict(self.__dict__)
        del sd["sampler"]
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Load the scheduler's state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def step(self):
        self._step_cnt += 1
        # Apply the step function
        new_value = self._step()
        # clip value to specified range
        new_value_clipped = self.backend.clip(new_value, self._min_val, self._max_val)
        # Set the new value of the parameter dynamically
        setattr(self.sampler, self.param_name, new_value_clipped)

    @abstractmethod
    def _step(self):
        ...


class LambdaScheduler(ParameterScheduler):
    """Sets a parameter to its initial value times a given function.

    Similar to :class:`~torch.optim.LambdaLR`.

    Args:
        obj (ReplayBuffer or Sampler): the replay buffer whose sampler to adjust (or the sampler itself).
        param_name (str): the name of the attribute to adjust, e.g. `beta` to adjust the
            beta parameter.
        lambda_fn (Callable[[int], float]): A function which computes a multiplicative factor given an integer
            parameter ``step_count``.
        min_value (Union[int, float], optional): a lower bound for the parameter to be adjusted
            Defaults to `None`.
        max_value (Union[int, float], optional): an upper bound for the parameter to be adjusted
            Defaults to `None`.

    """

    def __init__(
        self,
        obj: ReplayBuffer | Sampler,
        param_name: str,
        lambda_fn: Callable[[int], float],
        min_value: int | float | None = None,
        max_value: int | float | None = None,
    ):
        super().__init__(obj, param_name, min_value, max_value)
        self.lambda_fn = lambda_fn

    def _step(self):
        return self.initial_val * self.lambda_fn(self._step_cnt)


class LinearScheduler(ParameterScheduler):
    """A linear scheduler for gradually altering a parameter in an object over a given number of steps.

    This scheduler linearly interpolates between the initial value of the parameter and a final target value.

    Args:
        obj (ReplayBuffer or Sampler): the replay buffer whose sampler to adjust (or the sampler itself).
        param_name (str): the name of the attribute to adjust, e.g. `beta` to adjust the
            beta parameter.
        final_value (number): The final value that the parameter will reach after the
            specified number of steps.
        num_steps (number, optional): The total number of steps over which the parameter
            will be linearly altered.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming sampler uses initial beta = 0.6
        >>> # beta = 0.7   if step  == 1
        >>> # beta = 0.8   if step  == 2
        >>> # beta = 0.9   if step  == 3
        >>> # beta = 1.0   if step  >= 4
        >>> scheduler = LinearScheduler(sampler, param_name='beta', final_value=1.0, num_steps=4)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
        self,
        obj: ReplayBuffer | Sampler,
        param_name: str,
        final_value: int | float,
        num_steps: int,
    ):
        super().__init__(obj, param_name)
        if isinstance(self.initial_val, torch.Tensor):
            # cast to same type as initial value
            final_value = torch.tensor(final_value).to(self.initial_val)
        self.final_val = final_value
        self.num_steps = num_steps
        self._delta = (self.final_val - self.initial_val) / self.num_steps

    def _step(self):
        # Nit: we should use torch.where instead than if/else here to make the scheduler compatible with compile
        #  without graph breaks
        if self._step_cnt < self.num_steps:
            return self.initial_val + (self._delta * self._step_cnt)
        else:
            return self.final_val


class StepScheduler(ParameterScheduler):
    """A step scheduler that alters a parameter after every n steps using either multiplicative or additive changes.

    The scheduler can apply:
    1. Multiplicative changes: `new_val = curr_val * gamma`
    2. Additive changes: `new_val = curr_val + gamma`

    Args:
        obj (ReplayBuffer or Sampler): the replay buffer whose sampler to adjust (or the sampler itself).
        param_name (str): the name of the attribute to adjust, e.g. `beta` to adjust the
            beta parameter.
        gamma (int or float, optional): The value by which to adjust the parameter,
            either in a multiplicative or additive way.
        n_steps (int, optional): The number of steps after which the parameter should be altered.
            Defaults to 1.
        mode (str, optional): The mode of scheduling. Can be either `'multiplicative'` or `'additive'`.
            Defaults to `'multiplicative'`.
        min_value (int or float, optional): a lower bound for the parameter to be adjusted.
            Defaults to `None`.
        max_value (int or float, optional): an upper bound for the parameter to be adjusted.
            Defaults to `None`.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming sampler uses initial beta = 0.6
        >>> # beta = 0.6   if  0 <= step < 10
        >>> # beta = 0.7   if 10 <= step < 20
        >>> # beta = 0.8   if 20 <= step < 30
        >>> # beta = 0.9   if 30 <= step < 40
        >>> # beta = 1.0   if 40 <= step
        >>> scheduler = StepScheduler(sampler, param_name='beta', gamma=0.1, mode='additive', max_value=1.0)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
        self,
        obj: ReplayBuffer | Sampler,
        param_name: str,
        gamma: int | float = 0.9,
        n_steps: int = 1,
        mode: str = "multiplicative",
        min_value: int | float | None = None,
        max_value: int | float | None = None,
    ):

        super().__init__(obj, param_name, min_value, max_value)
        self.gamma = gamma
        self.n_steps = n_steps
        self.mode = mode
        if mode == "additive":
            operator = self.backend.add
        elif mode == "multiplicative":
            operator = self.backend.multiply
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Choose 'multiplicative' or 'additive'."
            )
        self.operator = operator

    def _step(self):
        """Applies the scheduling logic to alter the parameter value every `n_steps`."""
        # Check if the current step count is a multiple of n_steps
        current_val = getattr(self.sampler, self.param_name)
        # Nit: we should use torch.where instead than if/else here to make the scheduler compatible with compile
        #  without graph breaks
        if self._step_cnt % self.n_steps == 0:
            return self.operator(current_val, self.gamma)
        else:
            return current_val


class SchedulerList:
    """Simple container abstracting a list of schedulers."""

    def __init__(self, schedulers: list[ParameterScheduler]) -> None:
        if isinstance(schedulers, ParameterScheduler):
            schedulers = [schedulers]
        self.schedulers = schedulers

    def append(self, scheduler: ParameterScheduler):
        self.schedulers.append(scheduler)

    def step(self):
        for scheduler in self.schedulers:
            scheduler.step()
