# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import warnings
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal

import torch
from tensordict import is_tensor_collection, lazy_stack, TensorDict, TensorDictBase

from torchrl.data.tensor_specs import Composite
from torchrl.envs.common import EnvBase
from torchrl.envs.transforms.transforms import TensorDictPrimer, Transform
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
            If a dataloader_factory is provided, it will be used to create a fresh dataloader and this argument can be
            omitted.

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
        dataloader_factory (Callable[[], Iterable[dict[str, Any]]], optional): A callable that returns a dataloader.
            This allows for explicit resource control and avoids serialization issues.

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
        ...             padding (bool, optional): Whether to pad the tensors to the maximum length. Defaults to `False`.
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
        dataloader: Iterable[dict[str, Any]] | None = None,
        *,
        dataloader_factory: Callable[[], Iterable[dict[str, Any]]] | None = None,
        primers: Composite | None = None,
        stack_method: Callable[[Any], Any]
        | Literal["as_nested_tensor", "as_padded_tensor"]
        | None = None,
        batch_size: int | torch.Size | None = None,
        repeats: int | None = None,
        device: torch.device | None = None,
        group_repeats: bool = False,
    ):
        # Validate arguments: exactly one of dataloader or dataloader_factory must be provided
        if dataloader is not None and dataloader_factory is not None:
            raise ValueError(
                "Cannot provide both 'dataloader' and 'dataloader_factory'. Choose one."
            )
        if dataloader is None and dataloader_factory is None:
            raise ValueError(
                "Must provide exactly one of 'dataloader' or 'dataloader_factory'."
            )

        # Initialize dataloader from factory if provided
        if dataloader_factory is not None:
            self.dataloader = dataloader_factory()
            self.dataloader_factory = dataloader_factory
        else:
            self.dataloader = dataloader
            self.dataloader_factory = None

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
            if isinstance(batch_size, int):
                batch_size_tuple = (batch_size,)
            elif isinstance(batch_size, torch.Size):
                batch_size_tuple = tuple(batch_size)
            else:
                batch_size_tuple = (batch_size,)
        else:
            batch_size_tuple = batch_size
        batch_size = torch.Size(batch_size_tuple)
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
            primers = make_composite_from_td(
                data, dynamic_shape=True, unsqueeze_null_shapes=False
            )
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

    def reset_dataloader(self):
        """Reset the dataloader.

        This is useful when the dataloader is not infinite and we want to reset it.
        If a dataloader_factory was provided, it will be used to create a fresh dataloader.

        Returns:
            self: The transform itself.
        """
        self._queue.clear()
        if self.dataloader_factory is not None:
            # Create a fresh dataloader from the factory
            self.dataloader = self.dataloader_factory()
        self.endless_dataloader = self._endless_iter(self.dataloader)
        return self

    @classmethod
    def _endless_iter(self, obj):
        while True:
            yield from obj

    _device: torch.device | None = None

    @property
    def device(self) -> torch.device | None:
        if self._device is None:
            primers = getattr(self, "primers", None)
            if primers is not None:
                device = self.primers.device
            else:
                parent = getattr(self, "parent", None)
                if parent is not None:
                    device = getattr(parent, "device", None)
                else:
                    device = None
            self._device = device
        return self._device

    @device.setter
    def device(self, device: torch.device | None):
        self._device = device

    def _load_from_dataloader(self, reset: torch.Tensor | None = None):
        """Loads a single element from the dataloader, or alternatively from the buffer.

        If `reset` is passed, then one element per reset will be loaded.
        """
        device = self.device

        if reset is not None:
            if not reset.any():
                raise RuntimeError("reset must have at least one True value.")
            if reset.ndim > 0:
                loaded = [
                    self._load_from_dataloader().to(device) for _ in range(reset.sum())
                ]
                return self.stack_method(loaded)

        if len(self._queue) > 0:
            result = self._queue.popleft()
            if result.device != device:
                result = result.to(device)
            return result

        data = next(self.endless_dataloader)
        # Some heuristic here:
        # if data is a map, assume its keys match the keys in spec
        # TODO: one could rename the keys too
        if is_tensor_collection(data):
            out = data
        elif isinstance(data, Mapping):
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

    def _update_primers_batch_size(self, parent_batch_size):
        """Update the primers to match the parent's batch size.

        This method is called remotely to ensure the remote actor's primers
        have the correct batch dimensions.
        """
        if hasattr(self.primers, "expand"):
            # Expand primers to match the parent batch size
            if self.primers.shape != parent_batch_size:
                self.primers = self.primers.expand(parent_batch_size)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(primers={self.primers}, dataloader={self.dataloader})"

    def set_attr(self, name, value):
        """Set attribute on the remote actor or locally."""
        setattr(self, name, value)


class RayDataLoadingPrimer(Transform):
    """A :class:`~torchrl.envs.llm.transforms.dataloading.DataLoadingPrimer` that creates a single actor that can be shared by multiple environments.

    This class creates a Ray remote actor from DataLoadingPrimer that can be shared across multiple workers.
    All method calls are delegated to the remote actor, ensuring that multiple environments iterate over
    the same shared dataloader.

    Keyword Args:
        dataloader: A dataloader object to be used directly. Ray will handle serialization.
        dataloader_factory: A callable that returns a dataloader. This allows for explicit
            resource control and avoids serialization issues.
        num_cpus (int, optional): Number of CPUs to allocate to the Ray actor.
            Defaults to the dataloader's num_workers if available, otherwise 1.
        num_gpus (int, optional): Number of GPUs to allocate to the Ray actor. Defaults to 0.
        **kwargs: Additional keyword arguments to pass to DataLoadingPrimer.

    Note:
        Exactly one of `dataloader` or `dataloader_factory` must be provided.

    Examples:
        >>> # Option 1: Using a dataloader factory for explicit resource control
        >>> def create_dataloader():
        ...     return torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4)
        >>> primer1 = RayDataLoadingPrimer(dataloader_factory=create_dataloader, num_cpus=4)
        >>> primer2 = RayDataLoadingPrimer(dataloader_factory=create_dataloader, num_cpus=4)  # Same shared actor

        >>> # Option 2: Pass dataloader directly (Ray handles serialization)
        >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4)
        >>> primer1 = RayDataLoadingPrimer(dataloader=dataloader)  # num_cpus=4 inferred from num_workers
        >>> primer2 = RayDataLoadingPrimer(dataloader=dataloader)  # Same shared actor
    """

    def __init__(
        self,
        *,
        dataloader=None,
        dataloader_factory=None,
        num_cpus=None,
        num_gpus=0,
        **kwargs,
    ):

        # Import ray here to avoid requiring it as a dependency
        try:
            import ray
        except ImportError:
            raise ImportError(
                "Ray is required for RayDataLoadingPrimer. Install with: pip install ray"
            )
        self._ray = ray

        super().__init__(
            in_keys=kwargs.get("in_keys", []), out_keys=kwargs.get("out_keys", [])
        )

        # Validate arguments: exactly one of dataloader or dataloader_factory must be provided
        if dataloader is not None and dataloader_factory is not None:
            raise ValueError(
                "Cannot provide both 'dataloader' and 'dataloader_factory'. Choose one."
            )
        if dataloader is None and dataloader_factory is None:
            raise ValueError(
                "Must provide exactly one of 'dataloader' or 'dataloader_factory'."
            )

        # Infer num_cpus from dataloader if not specified and dataloader is provided
        if num_cpus is None:
            if dataloader is not None:
                num_cpus = getattr(dataloader, "num_workers", 1)
            elif dataloader_factory is not None:
                # Create a temporary dataloader to infer num_cpus, then discard it
                temp_dataloader = dataloader_factory()
                num_cpus = getattr(temp_dataloader, "num_workers", 1)
                del temp_dataloader  # Clean up
            else:
                num_cpus = 1

            if num_cpus == 0:  # Common case for single-threaded dataloaders
                num_cpus = 1

        # Create the remote DataLoadingPrimer with resource specifications
        RemoteDataLoadingPrimer = ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)(
            DataLoadingPrimer
        )

        primers = kwargs.get("primers", None)
        if hasattr(primers, "device"):
            self._device = primers.device
        else:
            self._device = None  # Initialize device tracking
        if hasattr(primers, "cpu"):
            primers = primers.cpu()
        if primers is not None:
            kwargs["primers"] = primers

        # Create the shared actor, passing factory or dataloader as appropriate
        if dataloader_factory is not None:
            actor = RemoteDataLoadingPrimer.remote(
                dataloader_factory=dataloader_factory, **kwargs
            )
        else:
            actor = RemoteDataLoadingPrimer.remote(dataloader=dataloader, **kwargs)

        self._actor = actor

    # Upward-facing methods (container management) - handled locally, not delegated to remote actor
    def set_container(self, container: Transform | EnvBase) -> None:
        """Set the container for this transform. This is handled locally."""
        result = super().set_container(container)

        # After setting the container locally, provide batch size information to the remote actor
        # This ensures the remote actor has the right batch size for proper shape handling
        if self.parent is not None:
            parent_batch_size = self.parent.batch_size

            # Set the batch size directly on the remote actor to override its initialization
            self._ray.get(self._actor.set_attr.remote("batch_size", parent_batch_size))

            # Also disable validation on the remote actor since we'll handle consistency locally
            self._ray.get(self._actor.set_attr.remote("_validated", True))

        return result

    def reset_parent(self) -> None:
        """Reset the parent. This is handled locally."""
        return super().reset_parent()

    @property
    def container(self) -> EnvBase | None:
        """Returns the env containing the transform. This is handled locally."""
        return super().container

    @property
    def parent(self) -> EnvBase | None:
        """Returns the parent env of the transform. This is handled locally."""
        return super().parent

    @property
    def base_env(self):
        """Returns the base environment. This traverses the parent chain locally."""
        return (
            getattr(self.parent, "base_env", None) if self.parent is not None else None
        )

    # Explicit method delegation for DataLoadingPrimer methods
    def reset_dataloader(self):
        """Reset the dataloader."""
        return self._ray.get(self._actor.reset_dataloader.remote())

    def _load_from_dataloader(self, reset=None):
        """Load data from the dataloader."""
        result = self._ray.get(self._actor._load_from_dataloader.remote(reset))
        # Cast to local device if one is set
        if hasattr(self, "_device") and self._device is not None:
            result = result.to(self._device)

        # Ensure proper batch dimensions: if result is scalar (ndim=0), unsqueeze to add batch dim
        # This matches the behavior in the original DataLoadingPrimer._load_from_dataloader
        if hasattr(result, "ndim") and not result.ndim:
            result = result.unsqueeze(0)

        return result

    def __repr__(self):
        """String representation."""
        try:
            if hasattr(self, "_actor") and self._actor is not None:
                return self._ray.get(self._actor.__repr__.remote())
            else:
                return "RayDataLoadingPrimer(actor=None)"
        except Exception:
            return f"RayDataLoadingPrimer(actor={getattr(self, '_actor', 'None')})"

    # Properties - access via generic attribute getter since Ray doesn't support direct property access
    @property
    def device(self):
        """Get device property."""
        return self._ray.get(self._actor.__getattribute__.remote("device"))

    @device.setter
    def device(self, value):
        """Set device property."""
        self._ray.get(self._actor.set_attr.remote("device", value))

    @property
    def dataloader(self):
        """Get dataloader property."""
        return self._ray.get(self._actor.__getattribute__.remote("dataloader"))

    @property
    def endless_dataloader(self):
        """Get endless_dataloader property."""
        return self._ray.get(self._actor.__getattribute__.remote("endless_dataloader"))

    @property
    def stack_method(self):
        """Get stack_method property."""
        return self._ray.get(self._actor.__getattribute__.remote("stack_method"))

    @property
    def repeats(self):
        """Get repeats property."""
        return self._ray.get(self._actor.__getattribute__.remote("repeats"))

    @property
    def data_keys(self):
        """Get data_keys property."""
        return self._ray.get(self._actor.__getattribute__.remote("data_keys"))

    @property
    def primers(self):
        """Get primers property."""
        return self._ray.get(self._actor.__getattribute__.remote("primers"))

    @primers.setter
    def primers(self, value):
        """Set primers property."""
        self._ray.get(self._actor.set_attr.remote("primers", value))

    # TensorDictPrimer methods
    def init(self, tensordict):
        """Initialize."""
        return self._ray.get(self._actor.init.remote(tensordict))

    def _reset_func(self, tensordict, tensordict_reset):
        """Reset function."""
        result = self._ray.get(
            self._actor._reset_func.remote(tensordict.cpu(), tensordict_reset.cpu())
        )
        # Cast to local device if one is set
        if hasattr(self, "_device") and self._device is not None:
            result = result.to(self._device)

        # Handle batch size expansion locally since remote actor lacks parent context
        # This mimics the batch expansion logic from TensorDictPrimer._reset_func
        if (
            self.parent
            and self.parent.batch_locked
            and hasattr(result, "apply")  # Check if it's a TensorDict-like object
        ):
            # Ensure result has proper batch dimensions to match parent
            expected_batch_size = self.parent.batch_size
            if result.batch_size != expected_batch_size:
                # Expand result to match expected batch size
                result = result.expand(expected_batch_size)

        return result

    # Override _reset to ensure proper delegation
    def _reset(self, tensordict, tensordict_reset):
        """Reset method for TensorDictPrimer."""
        result = self._ray.get(
            self._actor._reset.remote(tensordict.cpu(), tensordict_reset.cpu())
        )
        # Cast to local device if one is set
        if hasattr(self, "_device") and self._device is not None:
            result = result.to(self._device)
        return result

    def _reset_env_preprocess(self, tensordict):
        """Reset environment preprocess - crucial for call_before_env_reset=True."""
        result = self._ray.get(
            self._actor._reset_env_preprocess.remote(tensordict.cpu())
        )
        # Cast to local device if one is set
        if hasattr(self, "_device") and self._device is not None and result is not None:
            result = result.to(self._device)
        return result

    # Transform methods
    def close(self):
        """Close the transform."""
        return self._ray.get(self._actor.close.remote())

    def _apply_transform(self, obs):
        """Apply transform."""
        result = self._ray.get(self._actor._apply_transform.remote(obs))
        # Cast to local device if one is set
        if hasattr(self, "_device") and self._device is not None and result is not None:
            result = result.to(self._device)
        return result

    def _call(self, next_tensordict):
        """Call method."""
        result = self._ray.get(self._actor._call.remote(next_tensordict.cpu()))
        # Cast to local device if one is set
        if hasattr(self, "_device") and self._device is not None and result is not None:
            result = result.to(self._device)
        return result

    def forward(self, tensordict):
        """Forward pass."""
        result = self._ray.get(self._actor.forward.remote(tensordict.cpu()))
        # Cast to local device if one is set
        if hasattr(self, "_device") and self._device is not None and result is not None:
            result = result.to(self._device)
        return result

    def _inv_apply_transform(self, state):
        """Inverse apply transform."""
        result = self._ray.get(self._actor._inv_apply_transform.remote(state.cpu()))
        # Cast to local device if one is set
        if hasattr(self, "_device") and self._device is not None and result is not None:
            result = result.to(self._device)
        return result

    def _inv_call(self, tensordict):
        """Inverse call."""
        result = self._ray.get(self._actor._inv_call.remote(tensordict.cpu()))
        # Cast to local device if one is set
        if hasattr(self, "_device") and self._device is not None and result is not None:
            result = result.to(self._device)
        return result

    def inv(self, tensordict):
        """Inverse."""
        result = self._ray.get(self._actor.inv.remote(tensordict.cpu()))
        # Cast to local device if one is set
        if hasattr(self, "_device") and self._device is not None and result is not None:
            result = result.to(self._device)
        return result

    def _step(self, tensordict, next_tensordict):
        """Step method."""
        result = self._ray.get(
            self._actor._step.remote(tensordict.cpu(), next_tensordict.cpu())
        )
        # Cast to local device if one is set
        if hasattr(self, "_device") and self._device is not None and result is not None:
            result = result.to(self._device)
        return result

    def transform_env_device(self, device):
        """Transform environment device."""
        return self._ray.get(self._actor.transform_env_device.remote(device))

    def transform_env_batch_size(self, batch_size):
        """Transform environment batch size."""
        return self._ray.get(self._actor.transform_env_batch_size.remote(batch_size))

    def transform_output_spec(self, output_spec):
        """Transform output spec."""
        return self._ray.get(self._actor.transform_output_spec.remote(output_spec))

    def transform_input_spec(self, input_spec):
        """Transform input spec."""
        return self._ray.get(self._actor.transform_input_spec.remote(input_spec))

    def transform_observation_spec(self, observation_spec):
        """Transform observation spec."""
        return self._ray.get(
            self._actor.transform_observation_spec.remote(observation_spec)
        )

    def transform_reward_spec(self, reward_spec):
        """Transform reward spec."""
        return self._ray.get(self._actor.transform_reward_spec.remote(reward_spec))

    def transform_done_spec(self, done_spec):
        """Transform done spec."""
        return self._ray.get(self._actor.transform_done_spec.remote(done_spec))

    def transform_action_spec(self, action_spec):
        """Transform action spec."""
        return self._ray.get(self._actor.transform_action_spec.remote(action_spec))

    def transform_state_spec(self, state_spec):
        """Transform state spec."""
        return self._ray.get(self._actor.transform_state_spec.remote(state_spec))

    def dump(self, **kwargs):
        """Dump method."""
        return self._ray.get(self._actor.dump.remote(**kwargs))

    def clone(self):
        """Clone the transform."""
        # Use the parent's clone method to properly copy all Transform attributes
        new_instance = super().clone()
        # Then copy our specific Ray attributes to share the same actor
        new_instance._actor = self._actor
        new_instance._ray = self._ray
        new_instance._device = getattr(self, "_device", None)  # Copy device state
        return new_instance

    def empty_cache(self):
        """Empty cache."""
        super().empty_cache()
        return self._ray.get(self._actor.empty_cache.remote())

    def set_missing_tolerance(self, mode=False):
        """Set missing tolerance."""
        return self._ray.get(self._actor.set_missing_tolerance.remote(mode))

    @property
    def missing_tolerance(self):
        """Get missing tolerance."""
        return self._ray.get(self._actor.missing_tolerance.remote())

    @property
    def primers(self):
        """Get primers."""
        return self._ray.get(self._actor.__getattribute__.remote("primers"))

    @primers.setter
    def primers(self, value):
        """Set primers."""
        self.__dict__["_primers"] = value
        if hasattr(self, "_actor"):
            self._ray.get(self._actor.set_attr.remote("primers", value))

    def to(self, *args, **kwargs):
        """Move to device."""
        # Parse the device from args/kwargs like torch does
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )
        if device is not None:
            self._device = device
        # Don't delegate to remote actor - just register device locally
        return super().to(*args, **kwargs)

    # Properties that should be accessed from the remote actor
    @property
    def in_keys(self):
        """Get in_keys property."""
        return self._ray.get(self._actor.__getattribute__.remote("in_keys"))

    @in_keys.setter
    def in_keys(self, value):
        """Set in_keys property."""
        self.__dict__["_in_keys"] = value
        if hasattr(self, "_actor"):
            self._ray.get(self._actor.set_attr.remote("in_keys", value))

    @property
    def out_keys(self):
        """Get out_keys property."""
        return self._ray.get(self._actor.__getattribute__.remote("out_keys"))

    @out_keys.setter
    def out_keys(self, value):
        """Set out_keys property."""
        self.__dict__["_out_keys"] = value
        if hasattr(self, "_actor"):
            self._ray.get(self._actor.set_attr.remote("out_keys", value))

    @property
    def in_keys_inv(self):
        """Get in_keys_inv property."""
        return self._ray.get(self._actor.__getattribute__.remote("in_keys_inv"))

    @in_keys_inv.setter
    def in_keys_inv(self, value):
        """Set in_keys_inv property."""
        self.__dict__["_in_keys_inv"] = value
        if hasattr(self, "_actor"):
            self._ray.get(self._actor.set_attr.remote("in_keys_inv", value))

    @property
    def out_keys_inv(self):
        """Get out_keys_inv property."""
        return self._ray.get(self._actor.__getattribute__.remote("out_keys_inv"))

    @out_keys_inv.setter
    def out_keys_inv(self, value):
        """Set out_keys_inv property."""
        self.__dict__["_out_keys_inv"] = value
        if hasattr(self, "_actor"):
            self._ray.get(self._actor.set_attr.remote("out_keys_inv", value))

    # Generic attribute access for any remaining attributes
    def __getattr__(self, name):
        """Get attribute from the remote actor.

        This method should only be called for attributes that don't exist locally
        and should be delegated to the remote actor (inward-facing).

        Outward-facing attributes (parent, container, base_env, etc.) should be handled
        by the Transform base class and never reach this method.
        """
        # Upward-facing attributes that should never be delegated to remote actor
        upward_attrs = {"parent", "container", "base_env", "_parent", "_container"}

        if name in upward_attrs:
            # These should be handled by the local Transform implementation
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

        # Only delegate to remote actor if we're sure this is an inward-facing attribute
        # and the actor is properly initialized
        actor = self.__dict__.get("_actor", None)
        if actor is None:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

        # Only delegate specific DataLoadingPrimer methods/attributes to the remote actor
        # This is a whitelist approach to be more conservative
        delegated_methods = {
            # DataLoadingPrimer methods that should be called on the remote actor
            "_call",
            "_reset",
            "_inv_call",
            "forward",
            "inv",
            "_apply_transform",
            "_inv_apply_transform",
            "_reset_func",
            "init",  # TensorDictPrimer specific methods
            "primers",
            "dataloader",  # Properties
            # Add other specific methods that should be delegated as needed
        }

        if name in delegated_methods:
            try:
                result = self._ray.get(getattr(actor, name).remote())
                # If it's a method, wrap it to make remote calls
                if callable(result):
                    return lambda *args, **kwargs: self._ray.get(
                        getattr(actor, name).remote(*args, **kwargs)
                    )
                return result
            except (AttributeError, TypeError):
                # If that fails, it might be a callable method
                try:
                    remote_method = getattr(actor, name)
                    return lambda *args, **kwargs: self._ray.get(
                        remote_method.remote(*args, **kwargs)
                    )
                except AttributeError:
                    pass

        # If not in our whitelist, don't delegate to remote actor
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name, value):
        """Set attribute on the remote actor or locally."""
        # Local attributes that should never be delegated to remote actor
        local_attrs = {
            "_actor",
            "_ray",
            "_parent",
            "_container",
            "_missing_tolerance",
            "_in_keys",
            "_out_keys",
            "_in_keys_inv",
            "_out_keys_inv",
            "in_keys",
            "out_keys",
            "in_keys_inv",
            "out_keys_inv",
            "_modules",
            "_parameters",
            "_buffers",
            "_device",
        }

        if name in local_attrs:
            super().__setattr__(name, value)
        else:
            # Try to set on remote actor for other attributes
            try:
                if hasattr(self, "_actor") and self._actor is not None:
                    self._ray.get(self._actor.set_attr.remote(name, value))
                else:
                    super().__setattr__(name, value)
            except Exception:
                # Fall back to local setting for attributes that can't be set remotely
                super().__setattr__(name, value)
