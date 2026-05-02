# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import warnings

import torch
from tensordict.utils import expand_as_right


def get_primers_from_module(module, warn=True, strict=True):
    """Get all tensordict primers from all submodules of a module.

    This method is useful for retrieving primers from modules that are contained within a
    parent module.

    Args:
        module (torch.nn.Module): The parent module.
        warn (bool, optional): if ``True``, a warning is raised when no primers
            are found. Defaults to ``True``.
        strict (bool, optional): if ``True`` (default), exceptions raised by
            ``make_tensordict_primer()`` propagate. If ``False``, failures are
            caught per-submodule and a ``UserWarning`` lists the offending
            module types; primers from sibling submodules are still returned.
            Set to ``False`` from the collector dry-run path so that a single
            conditionally-built primer (e.g.
            :class:`~torchrl.modules.ConsistentDropoutModule` without
            ``input_shape``) doesn't drop primers from other submodules
            (e.g. a sibling :class:`~torchrl.modules.LSTMModule`).

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
    failures: list[str] = []

    def make_primers(submodule):
        if not hasattr(submodule, "make_tensordict_primer"):
            return
        try:
            primers.append(submodule.make_tensordict_primer())
        except Exception as e:
            if strict:
                raise
            failures.append(f"{type(submodule).__name__}: {e}")

    module.apply(make_primers)
    if failures:
        warnings.warn(
            "Could not auto-wire tensordict primers from these submodules: "
            + "; ".join(sorted(set(failures)))
            + ". They need explicit configuration (e.g. `input_shape=`) to "
            "build a primer. Continuing without them; wire manually if needed.",
            stacklevel=2,
        )
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


def _compute_missing_env_transforms(
    env,
    module,
    init_key: str = "is_init",
):
    """Return the list of env transforms ``module`` needs that ``env`` lacks.

    Pure read — does not mutate ``env``. See
    :func:`_maybe_append_env_transforms_from_module` for the full description
    of the detection rules and limitations; this function is the dry-run
    counterpart used by callers that need to decide whether to apply the
    transforms or warn the user.
    """
    if not hasattr(module, "apply"):
        # Policy factory or other plain Callable: no submodules to walk.
        return []

    # Walking submodules calls every ``make_tensordict_primer()`` we find. Some
    # implementations (e.g. ConsistentDropoutModule without ``input_shape``)
    # raise when they can't build a primer without further user input.
    # ``strict=False`` keeps primers from sibling submodules and warns naming
    # the offending module type, instead of aborting the whole walk.
    primer = get_primers_from_module(module, warn=False, strict=False)

    # Local import: torchrl.envs imports torchrl.modules at module import time.
    from torchrl.envs.transforms import Compose, InitTracker, TensorDictPrimer

    spec_keys: set = set()
    for spec_attr in ("full_observation_spec", "full_state_spec"):
        spec = getattr(env, spec_attr, None)
        if spec is None:
            continue
        try:
            spec_keys.update(spec.keys(True, True))
        except (RuntimeError, AttributeError):
            pass

    def _has_init_leaf(keys, key):
        for k in keys:
            if k == key:
                return True
            if isinstance(k, tuple) and k and k[-1] == key:
                return True
        return False

    transforms = []
    if not _has_init_leaf(spec_keys, init_key):
        transforms.append(InitTracker(init_key=init_key))

    if primer is not None:

        def _flatten(transform):
            if transform is None:
                return []
            if isinstance(transform, Compose):
                result = []
                for item in transform:
                    result.extend(_flatten(item))
                return result
            return [transform]

        primer_transforms = [
            t
            for t in _flatten(primer)
            if not (
                isinstance(t, TensorDictPrimer)
                and set(t.primers.keys(True, True)).issubset(spec_keys)
            )
        ]
        transforms.extend(primer_transforms)

    return transforms


def _maybe_append_env_transforms_from_module(
    env,
    module,
    init_key: str = "is_init",
):
    """Append recurrent env transforms required by ``module`` if absent.

    Detection is spec-based: we ask the env's ``full_observation_spec`` and
    ``full_state_spec`` whether ``init_key`` (or any leaf matching it, for
    multi-agent setups with nested init keys) and the keys produced by each
    discovered ``TensorDictPrimer`` are already present. This is correct in
    the presence of :class:`~torchrl.envs.BatchedEnv` /
    :class:`~torchrl.envs.SerialEnv` where transforms may live inside child
    envs and not be visible from the top-level transform stack. The helper is
    idempotent — calling it twice on the same env is a no-op.

    .. note::
        Limitations:

        * If a user has manually attached an :class:`InitTracker` with a
          *renamed* ``init_key``, this helper won't recognise it and may add
          a duplicate. Pass the same custom ``init_key`` here to avoid that,
          or wire transforms manually.
        * If ``module`` is a policy factory (a ``Callable`` that produces a
          policy on demand), we cannot inspect it without instantiating it.
          Auto-wrapping is skipped — pass ``policy=`` to the env constructor
          with an already-built policy, or attach transforms manually with
          :func:`get_env_transforms_from_module`.
    """
    transforms = _compute_missing_env_transforms(env, module, init_key)
    if not transforms:
        return env
    from torchrl.envs.transforms import Compose

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
