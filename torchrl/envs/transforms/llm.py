# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import warnings
from collections import deque
from collections.abc import Mapping
from copy import copy, deepcopy
from typing import Any, Callable, Iterable, Literal

import torch
from tensordict import lazy_stack, NestedKey, TensorDict, TensorDictBase, unravel_key
from tensordict.nn import ProbabilisticTensorDictModule, TensorDictParams
from tensordict.utils import is_seq_of_nested_key
from torch import nn

from torchrl.data.tensor_specs import Composite, Unbounded
from torchrl.envs.common import EnvBase
from torchrl.envs.transforms.transforms import TensorDictPrimer, Transform
from torchrl.envs.transforms.utils import _set_missing_tolerance, _stateless_param
from torchrl.envs.utils import make_composite_from_td


def as_nested_tensor(list_of_tensordicts: list[TensorDictBase]) -> TensorDictBase:
    """Stacks a list of tensordicts into a single tensordict with nested tensors.

    Args:
        list_of_tensordicts (list[TensorDictBase]): A list of tensordicts to stack.

    Returns:
        TensorDictBase: A tensordict with nested tensors.

    """

    def _as_nested_tensor(*list_of_tensors):
        return torch.nested.as_nested_tensor(list_of_tensors, layout=torch.jagged)

    batch_size = list(list_of_tensordicts[0].shape)
    batch_size.insert(0, len(list_of_tensordicts))
    return list_of_tensordicts[0].apply(
        _as_nested_tensor, *list_of_tensordicts[1:], batch_size=batch_size
    )


def as_padded_tensor(
    list_of_tensordicts: list[[TensorDictBase]], dim=0, stack_dim: int = 0
) -> TensorDictBase:
    """Stacks a list of tensordicts into a single tensordict with padded tensors.

    Args:
        list_of_tensordicts (list[[TensorDictBase]]): A list of tensordicts to stack.
        dim (int, optional): The dimension along which to pad. Defaults to 0.
        stack_dim (int, optional): The dimension along which to stack. Defaults to 0.

    Returns:
        TensorDictBase: A tensordict with padded tensors.
    """

    def _stack_tensors(*list_of_tensors):
        if dim < 0:
            raise ValueError("dim must be >= 0")
        max_length = max([t.size(dim) for t in list_of_tensors])

        def pad_tensor(tensor):
            padding_length = max_length - tensor.size(dim)
            shape = [
                s if i != dim else padding_length for i, s in enumerate(tensor.shape)
            ]
            return torch.cat((tensor.new_zeros(shape), tensor), dim=dim)

        return torch.stack([pad_tensor(t) for t in list_of_tensors], dim=stack_dim)

    batch_size = list(list_of_tensordicts[0].shape)
    batch_size.insert(dim, len(list_of_tensordicts))
    result = list_of_tensordicts[0].apply(
        _stack_tensors, *list_of_tensordicts[1:], batch_size=batch_size
    )
    return result


class DataLoadingPrimer(TensorDictPrimer):
    """A primer that loads data from a dataloader and converts it into a tensordict using ``stack_method``.

    Args:
        dataloader (Iterable[Dict[str, Any]]): The dataloader to load data from.
            During collection, we will attempt to convert it into a tensordict using :func:`~tensordict.from_dict` or a
            similar function.
            It is assumed that the elements retrieved from the dataloader come in batches along the first dimension
            of every tensor, unless `dataloader.batch_size=0`.
            The dataloader must yield mappable data structures (e.g., dictionaries).

    Keyword Args:
        primers (Composite | None, optional): The primers to use for each key in the dataloader. Defaults to None.
        stack_method (Callable[[Any], Any] | Literal["as_nested_tensor", "as_padded_tensor"], optional): The method to
            use for stacking the data. Defaults to ``maybe_dense_stack``.
        repeats (int, optional): How many times the same sample needs to appear successively. This can be useful in
            situations like GRPO where a single prompt is used multiple times to estimate the advantage using Monte-Carlo
            samples (rather than an advantage module).
        batch_size (int, torch.Size or None): the batch-size of the data delivered by the transform.
            This is somewhat unrelated to the batch-size of the dataloader, in the sense that this number may or may
            not match the DL's batch size.
            If left empty, the batch-size is inferred from `dataloader.batch_size` if that attribute exists. If not,
            an empty batch-size will be used (`torch.Size([])`).

            .. note:: The batch-size of the Primer must match the batch-size of the parent environment (typically a
                wrapper around :class:`~torchrl.envs.LLMEnv`).

        group_repeats (bool, optional): if ``True``, the batch-size is multiplied by the number of repeats such that
            all repeats are grouped in a single batch collected from the buffer. Defaults to ``False``.

    Attributes:
        dataloader (Iterable[Any]): The dataloader to load data from.
        endless_dataloader (Iterable[Any]): An endless iterator over the dataloader.
        stack_method (Callable[[Any], Any]): The method to use for stacking the data.

    .. seealso:: :class:`~torchrl.envs.LLMEnv` and :class:`~torchrl.envs.LLMEnv.from_dataloader`.

    Example of a dataloader yielding strings:
        >>> import random
        >>> import string
        >>> import tensordict as td
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import Unbounded
        >>> from torchrl.envs import DataLoadingPrimer, LLMEnv
        >>> td.set_capture_non_tensor_stack(False).set()
        >>> class DummyDataLoader:
        ...     '''A dummy dataloader that generates random strings.'''
        ...     def __init__(self, batch_size: int = 0):
        ...         self.batch_size = batch_size
        ...     def generate_random_string(self, length: int = 10) -. str:
        ...         '''Generate a random string of a given length.'''
        ...         return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))
        ...     def __iter__(self):
        ...         return self
        ...     def __next__(self):
        ...         if self.batch_size == 0:
        ...             return self.generate_random_string()
        ...         else:
        ...             return [self.generate_random_string() for _ in range(self.batch_size)]
        >>> # Create an LLM environment with string-to-string input/output.
        >>> env = LLMEnv(from_text=True)
        >>> # Append a DataLoadingPrimer to the environment.
        >>> env = env.append_transform(
        >>>     DataLoadingPrimer(
        >>>         dataloader=DummyDataLoader(),
        >>>         example_data="a string!",
        >>>     )
        >>> )
        >>> # Test the environment.
        >>> print(env.rand_action(TensorDict()))
        TensorDict(
            fields={
                action: NonTensorData(data=a string, batch_size=torch.Size([]), device=None)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> print(env.rollout(3))
        TensorDict(
            fields={
                action: NonTensorStack(
                    ['a string', 'a string', 'a string'],
                    batch_size=torch.Size([3]),
                    device=None),
                done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: NonTensorStack(
                            ['zxwvupirska string', 'zxwvupirska stringa string...,
                            batch_size=torch.Size([3]),
                            device=None),
                        terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([3]),
                    device=None,
                    is_shared=False),
                observation: NonTensorStack(
                    ['zxwvupirsk', 'zxwvupirska string', 'zxwvupirska ...,
                    batch_size=torch.Size([3]),
                    device=None),
                terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
        >>> # Roll out the environment with a specific initial state.
        >>> init_state = env.reset(TensorDict(batch_size=[3]))
        >>> print(env.rollout(3, auto_reset=False, tensordict=init_state))
        TensorDict(
            fields={
                action: NonTensorStack(
                    [['a string', 'a string', 'a string'], ['a string'...,
                    batch_size=torch.Size([3, 3]),
                    device=None),
                done: Tensor(shape=torch.Size([3, 3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([3, 3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: NonTensorStack(
                            [[array(['nngcmflsana string', 'vrrbnhzpmga string...,
                            batch_size=torch.Size([3, 3]),
                            device=None),
                        terminated: Tensor(shape=torch.Size([3, 3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([3, 3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([3, 3]),
                    device=None,
                    is_shared=False),
                observation: NonTensorStack(
                    [['nngcmflsan', array(['nngcmflsana string', 'vrrb...,
                    batch_size=torch.Size([3, 3]),
                    device=None),
                terminated: Tensor(shape=torch.Size([3, 3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([3, 3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([3, 3]),
            device=None,
            is_shared=False)

    Example of dataloader yielding tensors:
        >>> import random
        >>> import string
        >>>
        >>> import tensordict as td
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import Unbounded
        >>> from torchrl.envs import DataLoadingPrimer, LLMEnv
        >>>
        >>> td.set_capture_non_tensor_stack(False).set()
        >>>
        >>>
        >>> class DummyTensorDataLoader:
        ...     '''A dummy dataloader that generates tensors of random int64 values.'''
        ...
        ...     def __init__(self, batch_size: int = 0, max_length: int = 10, padding: bool = False):
        ...         '''
        ...         Args:
        ...             batch_size (int, optional): The batch size of the generated tensors. Defaults to 0.
        ...             max_length (int, optional): The maximum length of the generated tensors. Defaults to 10.
        ...             padding (bool, optional): Whether to pad the tensors to the maximum length. Defaults to False.
        ...         '''
        ...         self.batch_size = batch_size
        ...         self.max_length = max_length
        ...         self.padding = padding
        ...
        ...     def generate_random_tensor(self) -. torch.Tensor:
        ...         '''Generate a tensor of random int64 values.'''
        ...         length = random.randint(1, self.max_length)
        ...         return torch.tensor([random.randint(0, 100) for _ in range(length)], dtype=torch.int64)
        ...
        ...     def pad_tensor(self, tensor: torch.Tensor) -. torch.Tensor:
        ...         '''Pad a tensor to the maximum length.'''
        ...         padding_length = self.max_length - len(tensor)
        ...         return torch.cat((torch.zeros(padding_length, dtype=torch.int64), tensor))
        ...
        ...     def __iter__(self):
        ...         return self
        ...
        ...     def __next__(self):
        ...         if self.batch_size == 0:
        ...             tensor = self.generate_random_tensor()
        ...             return self.pad_tensor(tensor) if self.padding else tensor
        ...         else:
        ...             tensors = [self.generate_random_tensor() for _ in range(self.batch_size)]
        ...             if self.padding:
        ...                 tensors = [self.pad_tensor(tensor) for tensor in tensors]
        ...                 return torch.stack(tensors)
        ...             else:
        ...                 return tensors
        >>>
        >>> # Create an LLM environment with non-string input/output and append a DataLoadingPrimer.
        >>> env = LLMEnv(from_text=False)
        >>> env = env.append_transform(
        >>>     DataLoadingPrimer(
        >>>         dataloader=DummyTensorDataLoader(),
        >>>         data_specs=[Unbounded(shape=(-1,), dtype=torch.int64)],
        >>>     )
        >>> )
        >>> print(env.rand_action(TensorDict()))
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.int64, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> print(env.rollout(3))
        LazyStackedTensorDict(
            fields={
                action: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: LazyStackedTensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([3, -1]), device=cpu, dtype=torch.int64, is_shared=False),
                        terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    exclusive_fields={
                    },
                    batch_size=torch.Size([3]),
                    device=None,
                    is_shared=False,
                    stack_dim=0),
                observation: Tensor(shape=torch.Size([3, -1]), device=cpu, dtype=torch.int64, is_shared=False),
                terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            exclusive_fields={
            },
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False,
            stack_dim=0)
        >>> # Create an LLM environment with padded tensor input/output and append a DataLoadingPrimer.
        >>> env = LLMEnv(from_text=False)
        >>> env = env.append_transform(
        >>>     DataLoadingPrimer(
        >>>         dataloader=DummyTensorDataLoader(padding=True),
        >>>         data_specs=[Unbounded(shape=(-1,), dtype=torch.int64)],
        >>>         stack_method="as_padded_tensor",
        >>>     )
        >>> )
        >>> print(env.rollout(3, auto_reset=False, tensordict=env.reset(TensorDict(batch_size=[3]))))
        LazyStackedTensorDict(
            fields={
                action: Tensor(shape=torch.Size([3, 3, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                done: Tensor(shape=torch.Size([3, 3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: LazyStackedTensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([3, 3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([3, 3, -1]), device=cpu, dtype=torch.int64, is_shared=False),
                        terminated: Tensor(shape=torch.Size([3, 3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([3, 3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    exclusive_fields={
                    },
                    batch_size=torch.Size([3, 3]),
                    device=None,
                    is_shared=False,
                    stack_dim=1),
                observation: Tensor(shape=torch.Size([3, 3, -1]), device=cpu, dtype=torch.int64, is_shared=False),
                terminated: Tensor(shape=torch.Size([3, 3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([3, 3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            exclusive_fields={
            },
            batch_size=torch.Size([3, 3]),
            device=None,
            is_shared=False,
            stack_dim=1)

    """

    def __init__(
        self,
        dataloader: Iterable[dict[str, Any]],
        *,
        primers: Composite | None = None,
        stack_method: Callable[[Any], Any]
        | Literal["as_nested_tensor", "as_padded_tensor"] = None,
        batch_size: int | torch.Size | None = None,
        repeats: int | None = None,
        device: torch.device | None = None,
        group_repeats: bool = False,
    ):
        self.dataloader = dataloader
        if repeats is None:
            repeats = 0
        self.repeats = repeats

        # Determine batch-size
        #  We must distinguish the batch-size of the DL and the batch size of the transform.
        #  We may want more or less elements than the DL and the logic is slightly different so we
        #  allow to recompose batches on the fly. If the DL has a batch-size, every element will be
        #  unbound and stored in a queue. Otherwise, we get as many elements from the DL to fulfill
        #  the required batch-size.
        #
        #  If the batch-size is passed, we will stack as many elements as necessary to fulfill this.
        #  If not, we try to get it from the dataloader. Contrary to the dataloader, we will always
        #  deliver the same batch-size (we create an infinite dataloader and reset when it's done),
        #  whereas DLs with drop_last=False may return batches of different sizes.
        #
        # If the batch size passed to the transform is empty (torch.Size(())) or 0, we will consider that
        #  the batch-size is determined on-the-fly.
        #
        # A batch-size of 0 in the dataloader means no batch-size.
        #
        # If needed, the various repeats can be grouped in a single batch through group_repeats.
        #
        # If auto_batch_size is on, we call auto_batch_size=True when doing TensorDict.from_dict:
        #  That way we get a tensordict of the right batch-size.
        # If the dataloader has no batch-size, we're not sure that we can determine the batch-size
        #  automatically so we will consider that each element in the DL has a batch-size of 0 (ie,
        #  a single non-batched element is returned at a time).

        if batch_size is None:
            batch_size = getattr(dataloader, "batch_size", torch.Size([]))
        if batch_size == 0:
            batch_size = torch.Size(())
        if not isinstance(batch_size, (list, tuple)):
            batch_size = (batch_size,)
        batch_size = torch.Size(batch_size)
        auto_batch_size = getattr(dataloader, "batch_size", 1) != 0

        if len(batch_size) > 1:
            raise ValueError(
                f"batch_size can only be 0 or 1D, got batch_size={batch_size}."
            )

        # We deliver all the repeats in the same batch
        if repeats and group_repeats:
            if batch_size == torch.Size([]):
                batch_size = torch.Size((repeats,))
            else:
                batch_size = torch.Size([batch_size[0] * repeats])

        self._queue = deque()
        self.auto_batch_size = auto_batch_size
        self.batch_size = batch_size
        self.endless_dataloader = self._endless_iter(self.dataloader)

        if stack_method is None:
            stack_method = lazy_stack
        elif stack_method == "as_nested_tensor":
            stack_method = as_nested_tensor
        elif stack_method == "as_padded_tensor":
            stack_method = as_padded_tensor
        elif not callable(stack_method):
            raise ValueError(f"Unknown stack_method={stack_method}")
        self.stack_method = stack_method

        if primers is None:
            # We can get the primer from the dataloader itself
            data = self._load_from_dataloader()
            primers = make_composite_from_td(data, dynamic_shape=True)
            if batch_size:
                primers = primers.expand(batch_size)
            self._queue.insert(0, data)
            self.data_keys = list(primers.keys(True, True))
        else:
            self.data_keys = list(primers.keys(True, True))

        super().__init__(
            primers=primers,
            default_value=self._load_from_dataloader,
            reset_key=None,
            expand_specs=None,
            single_default_value=True,
            call_before_env_reset=True,
            device=device,
        )
        self._reset_key = "_reset"

    @classmethod
    def _endless_iter(self, obj):
        while True:
            yield from obj

    def _load_from_dataloader(self, reset: torch.Tensor | None = None):
        """Loads a single element from the dataloader, or alternatively from the buffer.

        If `reset` is passed, then one element per reset will be loaded.
        """
        if reset is not None:
            if not reset.any():
                raise RuntimeError("reset must have at least one True value.")
            if reset.ndim > 0:
                loaded = [self._load_from_dataloader() for _ in range(reset.sum())]
                return self.stack_method(loaded)

        primers = getattr(self, "primers", None)
        if primers is not None:
            device = self.primers.device
        else:
            device = None

        if len(self._queue) > 0:
            result = self._queue.popleft()
            if result.device != device:
                result = result.to(device)
            return result

        data = next(self.endless_dataloader)
        # Some heuristic here:
        # if data is a map, assume its keys match the keys in spec
        # TODO: one could rename the keys too
        if isinstance(data, Mapping):
            out = TensorDict.from_dict(
                data,
                auto_batch_size=self.auto_batch_size,
                batch_dims=int(bool(self.auto_batch_size or self.batch_size)),
                device=device,
            )
        else:
            raise TypeError(
                "Data loader must return a mapping that can be automatically cast to a tensordict. Check that you have "
                "the appropriate collate_fn in your dataloader to do so."
            )
        if not out.ndim:
            out = out.unsqueeze(0)
        self._queue.extend(
            [d for d in out.unbind(0) for _ in range(max(1, self.repeats))]
        )
        out = self._queue.popleft()
        return out

    def set_container(self, container: Transform | EnvBase) -> None:
        result = super().set_container(container)
        # Check batch size
        parent = getattr(self, "parent", None)
        if (
            self.batch_size is not None
            and parent is not None
            and parent.batch_size != self.batch_size
        ):
            warnings.warn(
                f"The parent env has a different batch size than the {type(self).__name__} transform."
            )
        return result

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(primers={self.primers}, dataloader={self.dataloader})"


class KLRewardTransform(Transform):
    """A transform to add a KL[pi_current||pi_0] correction term to the reward.

    This transform is used to constrain the policy to remain close to its original
    configuration which limits overfitting when fine-tuning using RLHF.

    Args:
        actor (ProbabilisticTensorDictModule): a probabilistic actor. It must
            have the following features: it must have a set of input (``in_keys``)
            and output keys (``out_keys``). It must have a ``get_dist`` method
            that outputs the distribution of the action.
        coef (:obj:`float`): the coefficient of the KL term. Defaults to ``1.0``.
        in_keys (str or list of str/tuples of str): the input key where the
            reward should be fetched. Defaults to ``"reward"``.
        out_keys (str or list of str/tuples of str): the output key where the
            reward should be written. Defaults to ``"reward"``.
        requires_grad (bool, optional): if ``True``, the frozen parameters will
            consist of differentiable clones of the original params.
            Defaults to ``False``.

    .. note:: If the parameters are not differentiable (default), they will *not*
        follow the module when dtype or device casting operations will be called
        (such as :meth:`cuda`, :meth:`to` etc.). When ``requires_grad=True``,
        casting operations will work as expected.

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> from torchrl.envs import TransformedEnv
        >>> from tensordict.nn import TensorDictModule as Mod, NormalParamExtractor
        >>> from torchrl.modules import ProbabilisticActor
        >>> from tensordict import TensorDict
        >>> from torchrl.modules.distributions import TanhNormal
        >>> from torch import nn
        >>> base_env = GymEnv("Pendulum-v1")
        >>> n_obs = base_env.observation_spec["observation"].shape[-1]
        >>> n_act = base_env.action_spec.shape[-1]
        >>> module = Mod(
        ...     nn.Sequential(nn.Linear(n_obs, n_act * 2), NormalParamExtractor()),
        ...     in_keys=["observation"],
        ...     out_keys=["loc", "scale"],
        ... )
        >>> actor = ProbabilisticActor(
        ...     module,
        ...     in_keys=["loc", "scale"],
        ...     distribution_class=TanhNormal,
        ...     return_log_prob=True,
        ... )
        >>> transform = KLRewardTransform(actor, out_keys="reward_kl")
        >>> env = TransformedEnv(base_env, transform)
        >>> with torch.no_grad():
        ...     # modify the actor parameters
        ...     _ = TensorDict(dict(actor.named_parameters()), []).apply_(lambda x: x.data.copy_(x.data + 1))
        ...     td = env.rollout(3, actor)
        >>> # check that rewards have been modified
        >>> assert (td.get(("next", "reward")) != td.get(("next", "reward_kl"))).all()

    .. note:: Because the KL formula is not always available and the parameters of the
      original distribution may not have been recorded, we use a stochastic estimate
      of the KL divergence.

    """

    DEFAULT_IN_KEYS = ["reward"]

    def __init__(
        self,
        actor: ProbabilisticTensorDictModule,
        coef=1.0,
        in_keys=None,
        out_keys=None,
        requires_grad=False,
        # TODO: adapt this to new API
        log_prob_key: NestedKey = "sample_log_prob",
        action_key: NestedKey | None = None,
        functional: bool | None = None,
        device: torch.device | None = None,
    ):
        if in_keys is None:
            in_keys = self.DEFAULT_IN_KEYS
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        if not is_seq_of_nested_key(self.in_keys) or not is_seq_of_nested_key(
            self.out_keys
        ):
            raise ValueError(
                f"invalid in_keys / out_keys:\nin_keys={self.in_keys} \nout_keys={self.out_keys}"
            )
        if len(self.in_keys) != 1 or len(self.out_keys) != 1:
            raise ValueError(
                f"Only one in_key/out_key is allowed, got in_keys={self.in_keys}, out_keys={self.out_keys}."
            )
        self._out_keys = [unravel_key(out_key) for out_key in self._out_keys]

        # update the in_keys for dispatch etc
        self.in_keys = self.in_keys + actor.in_keys
        self.in_keys = [unravel_key(in_key) for in_key in self.in_keys]

        if functional is None:
            from torchrl.modules.llm import CategoricalSequential

            functional = not isinstance(actor, CategoricalSequential)
        self.functional = functional
        # check that the model has parameters
        if functional:
            params = TensorDict.from_module(actor)
            with params.apply(
                _stateless_param, device="meta", filter_empty=False
            ).to_module(actor):
                # copy a stateless actor
                self.__dict__["functional_actor"] = deepcopy(actor)

            # we need to register these params as buffer to have `to` and similar
            # methods work properly

            def _make_detached_param(x):

                if isinstance(x, nn.Parameter):
                    # we need an nn.Parameter since some modules (RNN) require nn.Parameters
                    return nn.Parameter(x.data.clone(), requires_grad=requires_grad)
                elif x.requires_grad:
                    raise ValueError(
                        "Encountered a value that requires gradients but is not an nn.Parameter instance."
                    )
                return x.clone()

            self.frozen_params = params.apply(_make_detached_param, filter_empty=False)
            if requires_grad:
                # includes the frozen params/buffers in the module parameters/buffers
                self.frozen_params = TensorDictParams(
                    self.frozen_params, no_convert=True
                )

        else:
            self.__dict__["functional_actor"] = actor

        # self._buffers["actor_params"] = params.clone().detach()

        self.device = device
        self.action_key = action_key

        # find the sample log-prob key
        self.sample_log_prob_key = log_prob_key

        def find_sample_log_prob(module):
            if hasattr(module, "log_prob_key"):
                self.sample_log_prob_key = module.log_prob_key

        self.functional_actor.apply(find_sample_log_prob)

        if not isinstance(coef, torch.Tensor):
            coef = torch.as_tensor(coef)
        self.register_buffer("coef", coef)

    def set_container(self, container: Transform | EnvBase) -> None:
        result = super().set_container(container)
        if self.action_key is None:
            parent = getattr(self, "parent", None)
            if parent is not None:
                action_keys = parent.action_keys
                if len(action_keys) != 1:
                    raise ValueError(
                        f"More than one action_key found. Please pass the `action_key` argument directly to {type(self).__name__}."
                    )
                action_key = action_keys[0]
                self.action_key = action_key
        return result

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._step(tensordict_reset, tensordict_reset)
        return tensordict_reset

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        # run the actor on the tensordict
        action_key = self.action_key
        if action_key is None:
            raise ValueError(
                f"action_key is required. Please set a parent for the {type(self).__name__} to recover the action keys automatically, "
                f"or pass the action_key argument directly to {type(self).__name__} constructor."
            )
        action = tensordict.get(action_key, None)
        if action is None:
            if not self.missing_tolerance:
                raise RuntimeError(
                    f"Action with key {action_key} not found data {tensordict}"
                )
            # being called after reset or without action, skipping
            if self.out_keys[0] != "reward" and self.parent is not None:
                next_tensordict.set(self.out_keys[0], self.parent.reward_spec.zero())
            return next_tensordict

        if self.device is not None:
            action = action.to(self.device)

        if self.functional:
            with self.frozen_params.to_module(self.functional_actor):
                dist = self.functional_actor.get_dist(tensordict.clone(False))
            # get the log_prob given the original model
            log_prob = dist.log_prob(action)
        elif hasattr(self.functional_actor, "log_prob"):
            if self.device is not None:
                td_device = tensordict.to(self.device)
            else:
                td_device = tensordict
            log_prob = self.functional_actor.log_prob(
                td_device, as_nested_tensor=True, layout=torch.strided
            )
        else:
            log_prob = self.functional_actor(tensordict).get(self.sample_log_prob_key)

        reward_key = self.in_keys[0]
        reward = next_tensordict.get(reward_key)
        curr_log_prob = tensordict.get(
            self.sample_log_prob_key, as_nested_tensor=True, layout=torch.strided
        )
        log_prob = log_prob.to(curr_log_prob.device)
        # We want the log-probs to have a similar dim to the reward
        curr_log_prob = curr_log_prob.unsqueeze(-1)
        log_prob = log_prob.unsqueeze(-1)

        # we use the unbiased consistent estimator of the KL: log_p(x) - log_q(x) when x ~ p(x)
        if not reward.is_nested and log_prob.is_nested:
            reward = torch.nested.nested_tensor(
                [rew.expand(lp.shape) for rew, lp in zip(reward, log_prob)],
                layout=torch.strided,
            )
        if log_prob[0].shape != curr_log_prob[0].shape:
            # Don't check shapes if nested
            raise ValueError(
                f"the log-probability tensor shapes must match, got cur_log_prob.shape={curr_log_prob[0].shape} and log_prob.shape={log_prob[0].shape}."
            )
        if reward is not None and reward.ndim != curr_log_prob.ndim:
            raise ValueError(
                "The number of dimensions of reward must be the same as the number of dimensions of the KL "
                f"term. Got ndim={reward.ndim} and {curr_log_prob.ndim} respectively."
            )
        kl = curr_log_prob - log_prob
        if reward is None:
            reward = 0
        next_tensordict.set(self.out_keys[0], reward + self.coef * kl)
        return next_tensordict

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        next_td = tensordict.pop("next")
        next_td = self._step(tensordict, next_td)
        return tensordict.set("next", next_td)

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        in_key = unravel_key(self.in_keys[0])
        out_key = unravel_key(self.out_keys[0])

        if in_key == "reward" and out_key == "reward":
            parent = self.parent

            reward_keys = parent.reward_keys
            if len(reward_keys) == 1:
                reward_key = reward_keys[0]
            elif "reward" in reward_keys:
                reward_key = "reward"
            else:
                raise KeyError("Couln't find the reward key.")
            shape = output_spec["full_reward_spec"][reward_key].shape
            if len(shape) > 2:
                # For LLMs, the shape of the reward is (batch, -1, 1)
                shape = (*shape[:-2], -1, 1)
            reward_spec = Unbounded(
                device=output_spec.device,
                shape=shape,
            )
            output_spec["full_reward_spec"] = Composite(
                {reward_key: reward_spec},
                shape=output_spec["full_reward_spec"].shape,
            )
        elif in_key == "reward":
            # TODO: we should at least allow to make this a component of the reward specs, to avoid a call during reset
            parent = self.parent
            reward_spec = output_spec["full_reward_spec"][parent.reward_key]

            shape = reward_spec.shape
            if len(shape) > 2:
                # For LLMs, the shape of the reward is (batch, -1, 1)
                shape = (*shape[:-2], -1, 1)
            reward_spec = reward_spec.clone()
            reward_spec.shape = torch.Size(shape)

            # then we need to populate the output keys
            observation_spec = output_spec["full_observation_spec"]
            observation_spec[out_key] = reward_spec
        else:
            observation_spec = output_spec["full_observation_spec"]
            reward_spec = observation_spec[in_key]

            shape = reward_spec.shape
            shape = (*shape[:-2], -1, 1)
            reward_spec = reward_spec.clone()
            reward_spec.shape = torch.Size(shape)

            # then we need to populate the output keys
            observation_spec[out_key] = reward_spec
        return output_spec
