# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import warnings

import torch
from tensordict.utils import expand_as_right


def get_primers_from_module(module, warn=True):
    """Get all tensordict primers from all submodules of a module.

    This method is useful for retrieving primers from modules that are contained within a
    parent module.

    Args:
        module (torch.nn.Module): The parent module.
        warn (bool, optional): if ``True``, a warning is raised when no primers
            are found. Defaults to ``True``.

    Returns:
        TensorDictPrimer: A TensorDictPrimer Transform.

    Example:
        >>> from torchrl.modules.utils import get_primers_from_module
        >>> from torchrl.modules import GRUModule, MLP
        >>> from tensordict.nn import TensorDictModule, TensorDictSequential
        >>> # Define a GRU module
        >>> gru_module = GRUModule(
        ...     input_size=10,
        ...     hidden_size=10,
        ...     num_layers=1,
        ...     in_keys=["input", "recurrent_state", "is_init"],
        ...     out_keys=["features", ("next", "recurrent_state")],
        ... )
        >>> # Define a head module
        >>> head = TensorDictModule(
        ...     MLP(
        ...         in_features=10,
        ...         out_features=10,
        ...         num_cells=[],
        ...     ),
        ...     in_keys=["features"],
        ...     out_keys=["output"],
        ... )
        >>> # Create a sequential model
        >>> model = TensorDictSequential(gru_module, head)
        >>> # Retrieve primers from the model
        >>> primers = get_primers_from_module(model)
        >>> print(primers)

        TensorDictPrimer(primers=Composite(
            recurrent_state: UnboundedContinuous(
                shape=torch.Size([1, 10]),
                space=None,
                device=cpu,
                dtype=torch.float32,
                domain=continuous), device=None, shape=torch.Size([])), default_value={'recurrent_state': 0.0}, random=None)

    """
    primers = []

    def make_primers(submodule):
        if hasattr(submodule, "make_tensordict_primer"):
            primers.append(submodule.make_tensordict_primer())

    module.apply(make_primers)
    if not primers:
        if warn:
            warnings.warn("No primers found in the module.")
        return
    elif len(primers) == 1:
        return primers[0]
    else:
        from torchrl.envs.transforms import Compose

        return Compose(*primers)


def get_env_transforms_from_module(module, init_key="is_init"):
    """Return all :class:`~torchrl.envs.transforms.TransformedEnv` transforms needed for a recurrent module.

    Composes :class:`~torchrl.envs.transforms.InitTracker` (writes
    ``is_init=True`` at episode resets) with
    :class:`~torchrl.envs.transforms.TensorDictPrimer` (initialises hidden
    states). Pass the result directly to
    :class:`~torchrl.envs.transforms.TransformedEnv`.

    Args:
        module (torch.nn.Module): A module that may contain recurrent
            submodules (e.g. :class:`~torchrl.modules.LSTMModule` or
            :class:`~torchrl.modules.GRUModule`).
        init_key (str, optional): the key used by
            :class:`~torchrl.envs.transforms.InitTracker` to mark episode
            starts. Must match the ``is_init`` key expected by the recurrent
            module. Defaults to ``"is_init"``.

    Returns:
        A :class:`~torchrl.envs.transforms.Compose` of
        ``[InitTracker, TensorDictPrimer]`` when the module contains recurrent
        submodules, or a bare :class:`~torchrl.envs.transforms.InitTracker`
        otherwise.

    Example:
        >>> from torchrl.modules import GRUModule
        >>> from torchrl.modules.utils import get_env_transforms_from_module
        >>> gru = GRUModule(
        ...     input_size=4, hidden_size=8, num_layers=1,
        ...     in_keys=["obs", "recurrent_state", "is_init"],
        ...     out_keys=["features", ("next", "recurrent_state")],
        ... )
        >>> transforms = get_env_transforms_from_module(gru)
        >>> # TransformedEnv(base_env, transforms)
    """
    # Local import: torchrl.envs imports from torchrl.modules, so a
    # module-top import here creates a circular import. TYPE_CHECKING
    # wouldn't help — we need the runtime classes.
    from torchrl.envs.transforms import Compose, InitTracker

    primer = get_primers_from_module(module, warn=False)
    tracker = InitTracker(init_key=init_key)
    if primer is None:
        return tracker
    return Compose(tracker, primer)


def _maybe_append_env_transforms_from_module(
    env,
    module,
    init_key: str = "is_init",
    *,
    require_primer: bool = False,
):
    """Append recurrent env transforms required by ``module`` if absent."""
    if not hasattr(module, "apply"):
        return env
    primer = get_primers_from_module(module, warn=False)
    if require_primer and primer is None:
        return env

    # Local import: torchrl.envs imports torchrl.modules at module import time.
    from torchrl.envs.transforms import Compose, InitTracker, TensorDictPrimer
    from torchrl.envs.transforms.transforms import TransformedEnv

    def _flatten(transform):
        if transform is None:
            return []
        if isinstance(transform, Compose):
            result = []
            for item in transform:
                result.extend(_flatten(item))
            return result
        return [transform]

    existing = []
    if isinstance(env, TransformedEnv):
        existing = _flatten(env.transform)

    has_init_tracker = any(
        isinstance(transform, InitTracker) and transform.init_key == init_key
        for transform in existing
    )
    existing_primer_keys = set()
    for transform in existing:
        if isinstance(transform, TensorDictPrimer):
            existing_primer_keys.update(transform.primers.keys(True, True))

    transforms = []
    if not has_init_tracker:
        transforms.append(InitTracker(init_key=init_key))

    if primer is not None:
        primer_transforms = [
            transform
            for transform in _flatten(primer)
            if not (
                isinstance(transform, TensorDictPrimer)
                and set(transform.primers.keys(True, True)).issubset(
                    existing_primer_keys
                )
            )
        ]
        transforms.extend(primer_transforms)

    if not transforms:
        return env
    return env.append_transform(Compose(*transforms))


def _unpad_tensors(tensors, mask, as_nested: bool = True) -> torch.Tensor:
    shape = tensors.shape[2:]
    mask = expand_as_right(mask.bool(), tensors)
    nelts = mask.sum(-1)
    while nelts.dim() > 1:
        nelts = nelts.sum(-1)
    vals = [t.view(-1, *shape) for t in tensors[mask].split(nelts.tolist(), dim=0)]
    if as_nested:
        return torch.nested.as_nested_tensor(vals)
    return vals
