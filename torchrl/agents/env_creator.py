from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Dict, Optional

import torch

from torchrl.data.tensordict.tensordict import _TensorDict
from torchrl.data.utils import CloudpickleWrapper
from torchrl.envs.common import _EnvClass

__all__ = ["EnvCreator"]


class EnvCreator:
    """Environment creator class.

    EnvCreator is a generic environment creator class that can substitute
    lambda functions when creating environments in multiprocessing contexts.
    If the environment created on a subprocess must share information with the
    main process (e.g. for the VecNorm transform), EnvCreator will pass the
    pointers to the tensordicts in shared memory to each process such that
    all of them are synchronised.

    Args:
        create_env_fn (callable): a callable that returns an _EnvClass
            instance.
        create_env_kwargs (dict, optional): the kwargs of the env creator.
        share_memory (bool, optional): if False, the resulting tensordict
            from the environment won't be placed in shared memory.

    Examples:
        >>> from torchrl.envs import GymEnv
        >>> from torchrl.data import VecNorm, TransformedEnv
        >>> from torchrl.agents import EnvCreator
        >>> from torch import multiprocessing as mp
        >>> env_fn = lambda: TransformedEnv(GymEnv("Pendulum-v1"), VecNorm())
        >>> env_creator = EnvCreator(env_fn)
        >>> def test_env1(env_creator):
        ...    env = env_creator()
        ...    for _ in range(10):
        ...        env.rand_step()
        ...        if env.is_done:
        ...            env.reset()
        ...    print("env 1: ", env.current_tensordict.get("next_observation"))
        >>> def test_env2(env_creator):
        ...    env = env_creator()
        ...    time.sleep(5)
        ...    print("env 2: ", env.current_tensordict.get("next_observation"))
        >>> ps = []
        >>> p1 = mp.Process(target=test_env1, args=(env_creator,))
        >>> p1.start()
        >>> ps.append(p1)
        >>> p2 = mp.Process(target=test_env2, args=(env_creator,))
        >>> p2.start()
        >>> ps.append(p1)
        >>> for p in ps:
        ...     p.join()

    """

    def __init__(
        self,
        create_env_fn: Callable[..., _EnvClass],
        create_env_kwargs: Optional[Dict] = None,
        share_memory: bool = True,
    ) -> None:
        if not isinstance(create_env_fn, EnvCreator):
            self.create_env_fn = CloudpickleWrapper(create_env_fn)
        else:
            self.create_env_fn = create_env_fn

        self.create_env_kwargs = (
            create_env_kwargs if isinstance(create_env_kwargs, dict) else dict()
        )
        self.initialized = False
        self._share_memory = share_memory
        self.init_()

    def share_memory(self, state_dict: OrderedDict) -> None:
        for key, item in list(state_dict.items()):
            if isinstance(item, (_TensorDict,)):
                if not item.is_shared():
                    print(f"{self.env_type}: sharing mem of {item}")
                    item.share_memory_()
                else:
                    print(
                        f"{self.env_type}: {item} is already shared"
                    )  # , deleting key')
                    del state_dict[key]
            elif isinstance(item, OrderedDict):
                self.share_memory(item)
            elif isinstance(item, torch.Tensor):
                del state_dict[key]

    def init_(self) -> EnvCreator:
        shadow_env = self.create_env_fn(**self.create_env_kwargs)
        shadow_env.reset()
        shadow_env.rand_step()
        self.env_type = type(shadow_env)
        self._transform_state_dict = shadow_env.state_dict()
        if self._share_memory:
            self.share_memory(self._transform_state_dict)
        self.initialized = True
        return self

    def __call__(self) -> _EnvClass:
        if not self.initialized:
            raise RuntimeError("EnvCreator must be initialized before being called.")
        env = self.create_env_fn(**self.create_env_kwargs)
        env.load_state_dict(self._transform_state_dict, strict=False)
        return env

    def state_dict(self, destination: Optional[OrderedDict] = None) -> OrderedDict:
        if self._transform_state_dict is None:
            return destination if destination is not None else OrderedDict()
        if destination is not None:
            destination.update(self._transform_state_dict)
            return destination
        return self._transform_state_dict

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        if self._transform_state_dict is not None:
            for key, item in state_dict.items():
                item_to_update = self._transform_state_dict[key]
                item_to_update.copy_(item)

    def __repr__(self) -> str:
        substr = ", ".join(
            [f"{key}: {type(item)}" for key, item in self.create_env_kwargs]
        )
        return f"EnvCreator({self.create_env_fn}({substr}))"


def env_creator(fun: Callable) -> EnvCreator:
    return EnvCreator(fun)
