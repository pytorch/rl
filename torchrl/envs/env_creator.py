# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import OrderedDict
from multiprocessing.sharedctypes import Synchronized
from typing import Callable

import torch
from tensordict import TensorDictBase
from torchrl._utils import logger as torchrl_logger
from torchrl.data.utils import CloudpickleWrapper
from torchrl.envs.common import EnvBase, EnvMetaData


class EnvCreator:
    """Environment creator class.

    EnvCreator is a generic environment creator class that can substitute
    lambda functions when creating environments in multiprocessing contexts.
    If the environment created on a subprocess must share information with the
    main process (e.g. for the VecNorm transform), EnvCreator will pass the
    pointers to the tensordicts in shared memory to each process such that
    all of them are synchronised.

    Args:
        create_env_fn (callable): a callable that returns an EnvBase
            instance.
        create_env_kwargs (dict, optional): the kwargs of the env creator.
        share_memory (bool, optional): if False, the resulting tensordict
            from the environment won't be placed in shared memory.
        **kwargs: additional keyword arguments to be passed to the environment
            during construction.

    Examples:
        >>> # We create the same environment on 2 processes using VecNorm
        >>> # and check that the discounted count of observations matches on
        >>> # both workers, even if one has not executed any step
        >>> import time
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> from torchrl.envs.transforms import VecNorm, TransformedEnv
        >>> from torchrl.envs import EnvCreator
        >>> from torch import multiprocessing as mp
        >>> env_fn = lambda: TransformedEnv(GymEnv("Pendulum-v1"), VecNorm())
        >>> env_creator = EnvCreator(env_fn)
        >>>
        >>> def test_env1(env_creator):
        ...     env = env_creator()
        ...     tensordict = env.reset()
        ...     for _ in range(10):
        ...         env.rand_step(tensordict)
        ...         if tensordict.get(("next", "done")):
        ...             tensordict = env.reset(tensordict)
        ...     print("env 1: ", env.transform._td.get(("next", "observation_count")))
        >>>
        >>> def test_env2(env_creator):
        ...     env = env_creator()
        ...     time.sleep(5)
        ...     print("env 2: ", env.transform._td.get(("next", "observation_count")))
        >>>
        >>> if __name__ == "__main__":
        ...     ps = []
        ...     p1 = mp.Process(target=test_env1, args=(env_creator,))
        ...     p1.start()
        ...     ps.append(p1)
        ...     p2 = mp.Process(target=test_env2, args=(env_creator,))
        ...     p2.start()
        ...     ps.append(p1)
        ...     for p in ps:
        ...         p.join()
        env 1:  tensor([11.9934])
        env 2:  tensor([11.9934])
    """

    def __init__(
        self,
        create_env_fn: Callable[..., EnvBase],
        create_env_kwargs: dict | None = None,
        share_memory: bool = True,
        **kwargs,
    ) -> None:
        if not isinstance(create_env_fn, (EnvCreator, CloudpickleWrapper)):
            self.create_env_fn = CloudpickleWrapper(create_env_fn)
        else:
            self.create_env_fn = create_env_fn

        self.create_env_kwargs = kwargs
        if isinstance(create_env_kwargs, dict):
            self.create_env_kwargs.update(create_env_kwargs)
        self.initialized = False
        self._meta_data = None
        self._share_memory = share_memory
        self.init_()

    def make_variant(self, **kwargs) -> EnvCreator:
        """Creates a variant of the EnvCreator, pointing to the same underlying metadata but with different keyword arguments during construction.

        This can be useful with transforms that share a state, like :class:`~torchrl.envs.TrajCounter`.

        Examples:
            >>> from torchrl.envs import GymEnv
            >>> env_creator_pendulum = EnvCreator(GymEnv, env_name="Pendulum-v1")
            >>> env_creator_cartpole = env_creator_pendulum.make_variant(env_name="CartPole-v1")

        """
        # Copy self
        out = type(self).__new__(type(self))
        out.__dict__.update(self.__dict__)
        out.create_env_kwargs.update(kwargs)
        return out

    def share_memory(self, state_dict: OrderedDict) -> None:
        for key, item in list(state_dict.items()):
            if isinstance(item, (TensorDictBase,)):
                if not item.is_shared():
                    item.share_memory_()
                else:
                    torchrl_logger.info(
                        f"{self.env_type}: {item} is already shared"
                    )  # , deleting key'val)
                    del state_dict[key]
            elif isinstance(item, OrderedDict):
                self.share_memory(item)
            elif isinstance(item, torch.Tensor):
                del state_dict[key]

    @property
    def meta_data(self) -> EnvMetaData:
        if self._meta_data is None:
            raise RuntimeError(
                "meta_data is None in EnvCreator. " "Make sure init_() has been called."
            )
        return self._meta_data

    @meta_data.setter
    def meta_data(self, value: EnvMetaData):
        self._meta_data = value

    @staticmethod
    def _is_mp_value(val):

        return isinstance(val, (Synchronized,)) and hasattr(val, "_obj")

    @classmethod
    def _find_mp_values(cls, env_or_transform, values, prefix=()):
        from torchrl.envs.transforms.transforms import Compose, TransformedEnv

        if isinstance(env_or_transform, EnvBase) and isinstance(
            env_or_transform, TransformedEnv
        ):
            cls._find_mp_values(
                env_or_transform.transform,
                values=values,
                prefix=prefix + ("transform",),
            )
            cls._find_mp_values(
                env_or_transform.base_env, values=values, prefix=prefix + ("base_env",)
            )
        elif isinstance(env_or_transform, Compose):
            for i, t in enumerate(env_or_transform.transforms):
                cls._find_mp_values(t, values=values, prefix=prefix + (i,))
        for k, v in env_or_transform.__dict__.items():
            if cls._is_mp_value(v):
                values.append((prefix + (k,), v))
        return values

    def init_(self) -> EnvCreator:
        shadow_env = self.create_env_fn(**self.create_env_kwargs)
        tensordict = shadow_env.reset()
        shadow_env.rand_step(tensordict)
        self.env_type = type(shadow_env)
        self._transform_state_dict = shadow_env.state_dict()
        # Extract any mp.Value object from the env
        self._mp_values = self._find_mp_values(shadow_env, values=[])

        if self._share_memory:
            self.share_memory(self._transform_state_dict)
        self.initialized = True
        self.meta_data = EnvMetaData.metadata_from_env(shadow_env)
        shadow_env.close()
        del shadow_env
        return self

    @classmethod
    def _set_mp_value(cls, env, key, value):
        if len(key) > 1:
            if isinstance(key[0], int):
                return cls._set_mp_value(env[key[0]], key[1:], value)
            else:
                return cls._set_mp_value(getattr(env, key[0]), key[1:], value)
        else:
            setattr(env, key[0], value)

    def __call__(self, **kwargs) -> EnvBase:
        if not self.initialized:
            raise RuntimeError("EnvCreator must be initialized before being called.")
        kwargs.update(self.create_env_kwargs)  # create_env_kwargs precedes
        env = self.create_env_fn(**kwargs)
        if self._mp_values:
            for k, v in self._mp_values:
                self._set_mp_value(env, k, v)
        env.load_state_dict(self._transform_state_dict, strict=False)
        return env

    def state_dict(self) -> OrderedDict:
        if self._transform_state_dict is None:
            return OrderedDict()
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
    """Helper function to call `EnvCreator`."""
    return EnvCreator(fun)


def get_env_metadata(env_or_creator: EnvBase | Callable, kwargs: dict | None = None):
    """Retrieves a EnvMetaData object from an env."""
    if isinstance(env_or_creator, (EnvBase,)):
        return EnvMetaData.metadata_from_env(env_or_creator)
    elif not isinstance(env_or_creator, EnvBase) and not isinstance(
        env_or_creator, EnvCreator
    ):
        # then env is a creator
        if kwargs is None:
            kwargs = {}
        env = env_or_creator(**kwargs)
        return EnvMetaData.metadata_from_env(env)
    elif isinstance(env_or_creator, EnvCreator):
        if not (
            kwargs == env_or_creator.create_env_kwargs
            or kwargs is None
            or len(kwargs) == 0
        ):
            raise RuntimeError(
                "kwargs mismatch between EnvCreator and the kwargs provided to get_env_metadata:"
                f"got EnvCreator.create_env_kwargs={env_or_creator.create_env_kwargs} and "
                f"kwargs = {kwargs}"
            )
        return env_or_creator.meta_data.clone()
    else:
        raise NotImplementedError(
            f"env of type {type(env_or_creator)} is not supported by get_env_metadata."
        )
