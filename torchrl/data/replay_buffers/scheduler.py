import numpy as np
from typing import Callable, Dict, Any

from .replay_buffers import ReplayBuffer
from .samplers import Sampler


class ParameterScheduler:

    """Scheduler to adjust the value of a given parameter of a replay buffer's sampler, e.g. the 
    alpha and beta values in the PrioritizedSampler.

    Args:
        rb (ReplayBuffer): the replay buffer whose sampler to adjust
        param_name (str): the name of the attribute to adjust, e.g. `beta` to adjust the beta parameter
        min_value (Union[int, float], optional): a lower bound for the parameter to be adjusted
            Defaults to None.
        max_value (Union[int, float], optional): an upper bound for the parameter to be adjusted
            Defaults to None

    """

    def __init__(
        self, 
        obj: ReplayBuffer | Sampler, 
        param_name: str,
        min_value: int | float = None, 
        max_value: int | float = None
    ):
        if not isinstance(obj, ReplayBuffer) and not isinstance(obj, Sampler):
            raise TypeError(
                f"ParameterScheduler only supports Sampler class. Pass either ReplayBuffer or Sampler object. Got {type(obj)}"
            )
        self.sampler = obj.sampler if isinstance(obj, ReplayBuffer) else obj
        self.param_name = param_name
        self._min_val = min_value
        self._max_val = max_value
        if not hasattr(self.sampler, self.param_name):
            raise ValueError(f"Provided class {obj.__name__} does not have an attribute {param_name}")
        self.initial_val = getattr(self.sampler, self.param_name)
        self._step_cnt = 0

    def state_dict(self):
        """Return the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "sampler"
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
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
        new_value_clipped = np.clip(new_value, a_min=self._min_val, a_max=self._max_val)
        # Set the new value of the parameter dynamically
        setattr(self.sampler, self.param_name, new_value_clipped)

    def _step(self):
        raise NotImplementedError


class LambdaScheduler(ParameterScheduler):
    """Similar to torch.optim.LambdaLR, this class sets a parameter to its initial value
    times a given function. 

    Args:
        obj (ReplayBuffer | Sampler): the replay buffer whose sampler to adjust (or the sampler itself)
        param_name (str): the name of the attribute to adjust, e.g. `beta` to adjust the 
            beta parameter
        lambda_fn (function): A function which computes a multiplicative factor given an integer 
            parameter step_count
        min_value (Union[int, float], optional): a lower bound for the parameter to be adjusted
            Defaults to None.
        max_value (Union[int, float], optional): an upper bound for the parameter to be adjusted
            Defaults to None
    
    """

    def __init__(
        self, 
        obj: ReplayBuffer | Sampler, 
        param_name: str, 
        lambda_fn: Callable[[int], float],
        min_value: int | float = None, 
        max_value: int | float = None
    ):
        super().__init__(obj, param_name, min_value, max_value)
        self.lambda_fn = lambda_fn

    def _step(self):
        return self.initial_val * self.lambda_fn(self._step_cnt)



class LinearScheduler(ParameterScheduler):
    """A linear scheduler for gradually altering a parameter in an object over a given number of steps.
    This scheduler linearly interpolates between the initial value of the parameter and a final target value.
    
    Args:
        obj (ReplayBuffer | Sampler): the replay buffer whose sampler to adjust (or the sampler itself)
        param_name (str): the name of the attribute to adjust, e.g. `beta` to adjust the 
            beta parameter
        final_value (Union[int, float]): The final value that the parameter will reach after the 
            specified number of steps.
        num_steps (Union[int, float], optional): The total number of steps over which the parameter 
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
        num_steps: int
    ):
        super().__init__(obj, param_name)
        self.final_val = final_value
        self.num_steps = num_steps
        self._delta = (self.final_val - self.initial_val) / self.num_steps

    def _step(self):
        if self._step_cnt < self.num_steps:
            return self.initial_val + (self._delta * self._step_cnt)
        else:
            return self.final_val



class StepScheduler(ParameterScheduler):
    """
    A step scheduler that alters a parameter after every n steps using either multiplicative or additive changes.
    
    The scheduler can apply:
    1. Multiplicative changes: `new_val = curr_val * gamma`
    2. Additive changes: `new_val = curr_val + gamma`
    
    Args:
        obj (ReplayBuffer | Sampler): the replay buffer whose sampler to adjust (or the sampler itself)
        param_name (str): the name of the attribute to adjust, e.g. `beta` to adjust the 
            beta parameter
        gamma (int | float, optional): The value by which to adjust the parameter, 
            either multiplicatively or additive
        n_steps (int, optional): The number of steps after which the parameter should be altered.
            Defaults to 1
        mode (str, optional): The mode of scheduling. Can be either 'multiplicative' or 'additive'.
            Defaults to 'multiplicative'
        min_value (int | float, optional): a lower bound for the parameter to be adjusted
            Defaults to None.
        max_value (int | float, optional): an upper bound for the parameter to be adjusted
            Defaults to None 

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming sampler uses initial beta = 0.6
        >>> # beta = 0.6   if step  <  10
        >>> # beta = 0.7   if step  == 10
        >>> # beta = 0.8   if step  == 20
        >>> # beta = 0.9   if step  == 30
        >>> # beta = 1.0   if step  >= 40
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
        min_value: int | float = None, 
        max_value: int | float = None
    ):

        super().__init__(obj, param_name, min_value, max_value)
        self.gamma = gamma
        self.n_steps = n_steps
        if mode == "additive": 
            operator = np.add
        elif mode == "multiplicative":
            operator = np.multiply
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Choose 'multiplicative' or 'additive'.")
        self.operator = operator

    def _step(self):
        """Applies the scheduling logic to alter the parameter value every `n_steps`."""
        # Check if the current step count is a multiple of n_steps
        current_val = getattr(self.sampler, self.param_name)
        if self._step_cnt % self.n_steps == 0:
            return self.operator(current_val, self.gamma)
        else:
            return current_val


class SchedulerList:
    def __init__(self, scheduler: list[ParameterScheduler]) -> None:
        if isinstance(scheduler, ParameterScheduler):
            scheduler = [scheduler]
        self.scheduler = scheduler

    def append(self, scheduler: ParameterScheduler):
        self.scheduler.append(scheduler)

    def step(self):
        for scheduler in self.scheduler:
            scheduler.step()