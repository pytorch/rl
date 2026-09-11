# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
import torch.nn.functional as F

from tensordict import TensorDictBase, unravel_key, unravel_key_list
from tensordict.nn import dispatch, TensorDictModuleBase as ModuleBase
from torch import nn

from torchrl._utils import is_compiling
from torchrl.data.tensor_specs import Composite
from torchrl.modules.tensordict_module.rnn import recurrent_mode

if TYPE_CHECKING:
    from torchrl.envs import TensorDictPrimer


def positions_from_is_init(is_init: torch.Tensor) -> torch.Tensor:
    """Compute per-token positions within each episode segment of a window.

    Positions restart at ``0`` on every ``is_init`` flag. The first step of
    the window is always treated as position ``0``, so callers must pass
    episode-aligned windows: :class:`TransformerModule` validates that every
    row of a training window starts with ``is_init=True``.

    Args:
        is_init (torch.Tensor): a boolean tensor of shape ``[*batch, T]``
            marking the first step of each episode.

    Returns:
        A ``torch.long`` tensor of shape ``[*batch, T]`` holding the position
        of each step within its episode segment.

    Examples:
        >>> is_init = torch.tensor([[True, False, True, False]])
        >>> positions_from_is_init(is_init)
        tensor([[0, 1, 0, 1]])
    """
    if is_init.dtype is not torch.bool:
        raise ValueError(f"is_init must be a boolean tensor, got {is_init.dtype}.")
    init = is_init.clone()
    init[..., 0] = True
    idx = torch.arange(is_init.shape[-1], device=is_init.device).expand_as(init)
    last_reset = torch.cummax(idx * init, dim=-1).values
    return idx - last_reset


def segment_causal_mask_from_is_init(is_init: torch.Tensor) -> torch.Tensor:
    """Build a block-diagonal causal attention mask from ``is_init`` flags.

    Entry ``[..., i, j]`` is ``True`` (attend) iff ``j <= i`` and steps ``i``
    and ``j`` belong to the same episode segment, so attention never crosses
    an episode boundary within a training window.

    Args:
        is_init (torch.Tensor): a boolean tensor of shape ``[*batch, T]``
            marking the first step of each episode.

    Returns:
        A boolean tensor of shape ``[*batch, T, T]`` where ``True`` means
        "may attend".

    Examples:
        >>> is_init = torch.tensor([[False, True]])
        >>> segment_causal_mask_from_is_init(is_init)
        tensor([[[ True, False],
                 [False,  True]]])
    """
    if is_init.dtype is not torch.bool:
        raise ValueError(f"is_init must be a boolean tensor, got {is_init.dtype}.")
    segment = is_init.long().cumsum(dim=-1)
    same_segment = segment.unsqueeze(-1) == segment.unsqueeze(-2)
    t = is_init.shape[-1]
    causal = torch.ones(t, t, dtype=torch.bool, device=is_init.device).tril()
    return same_segment & causal


class _TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        device=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.norm1 = nn.LayerNorm(hidden_size, device=device)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False, device=device)
        self.out_proj = nn.Linear(hidden_size, hidden_size, device=device)
        self.norm2 = nn.LayerNorm(hidden_size, device=device)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, dim_feedforward, device=device),
            nn.GELU(),
            nn.Linear(dim_feedforward, hidden_size, device=device),
            nn.Dropout(dropout),
        )

    def _qkv(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, t, _ = h.shape
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        shape = (batch, t, self.num_heads, self.head_dim)
        return (
            q.view(shape).transpose(1, 2),
            k.view(shape).transpose(1, 2),
            v.view(shape).transpose(1, 2),
        )

    def forward(
        self,
        h: torch.Tensor,
        attn_mask: torch.Tensor,
        cache_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q, k, v = self._qkv(self.norm1(h))
        if cache_kv is not None:
            cache_k, cache_v = cache_kv
            batch = torch.arange(h.shape[0], device=h.device)
            cache_k[batch, :, positions] = k.squeeze(2).detach().to(cache_k.dtype)
            cache_v[batch, :, positions] = v.squeeze(2).detach().to(cache_v.dtype)
            k, v = cache_k.to(q.dtype), cache_v.to(q.dtype)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        h = h + self.out_proj(attn.transpose(1, 2).flatten(-2))
        return h + self.mlp(self.norm2(h))


class CausalTransformer(nn.Module):
    """A causal transformer backbone with matching windowed and cached-step semantics.

    This is the reference implementation of the temporal-transformer backbone
    contract consumed by :class:`~torchrl.modules.TransformerModule`:

    - ``forward(features, positions, mask=None, kv_cache=None) -> (out, kv_cache)``
    - ``new_kv_cache(batch_size, device=None) -> kv_cache``
    - ``reset_kv_cache(kv_cache, mask) -> kv_cache``

    together with ``num_layers``, ``num_heads``, ``head_dim`` and
    ``max_seq_len`` attributes. The cache object is opaque to the module: the
    backbone decides its layout, dtype and device and how a reset clears the
    rows selected by a boolean mask over the batch. Any module honoring that
    contract can be used in its place, including adapters over an inference
    engine that keeps the cache in its own representation.

    Two execution paths share the same parameters and produce the same
    outputs: a window path processing ``[B, T]`` at once under a causal mask
    (training), and a cached-step path attending against a fixed-shape
    key/value cache (collection). Positions are always explicit inputs, which
    is what keeps the two paths consistent across episode resets.

    The reference cache is a ``(k, v)`` pair of shape ``[B, num_layers,
    num_heads, max_seq_len, head_dim]`` allocated in the dtype of the
    projection weights, so a module converted to ``bfloat16`` or ``float64``
    gets a matching cache. Under autocast the projected keys and values are
    cast to the cache dtype on write and the cache to the query dtype on
    read. Cached entries are detached: the cached-step path is inference
    only.

    Args:
        input_size (int): number of input features.
        hidden_size (int): dimension of the residual stream. Must be divisible
            by ``num_heads``.
        num_layers (int, optional): number of transformer blocks. Defaults to
            ``1``.

    Keyword Args:
        num_heads (int): number of attention heads.
        max_seq_len (int): maximum episode length; sets the positional
            embedding table and the cache size. Episodes longer than this
            raise an error (sliding-window semantics are deliberately not
            implemented).
        dim_feedforward (int, optional): hidden dimension of the per-block
            MLP. Defaults to ``4 * hidden_size``.
        dropout (float, optional): dropout probability in the block MLPs.
            Defaults to ``0.0``.
        device (torch.device, optional): device to build the parameters on.

    Examples:
        >>> import torch
        >>> net = CausalTransformer(3, 16, 2, num_heads=4, max_seq_len=10)
        >>> features = torch.randn(2, 5, 3)
        >>> positions = torch.arange(5).expand(2, 5)
        >>> out, _ = net(features, positions)
        >>> out.shape
        torch.Size([2, 5, 16])
        >>> cache = net.new_kv_cache(2)
        >>> step, cache = net(features[:, :1], positions[:, :1], kv_cache=cache)
        >>> torch.allclose(step, out[:, :1], atol=1e-6)
        True
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        *,
        num_heads: int,
        max_seq_len: int,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        device=None,
    ):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads "
                f"({num_heads})."
            )
        if dim_feedforward is None:
            dim_feedforward = 4 * hidden_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_seq_len = max_seq_len
        self.in_proj = nn.Linear(input_size, hidden_size, device=device)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size, device=device)
        self.blocks = nn.ModuleList(
            [
                _TransformerBlock(
                    hidden_size, num_heads, dim_feedforward, dropout, device=device
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size, device=device)

    def new_kv_cache(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate an empty key/value cache for ``batch_size`` streams.

        Args:
            batch_size (int): number of concurrent streams (environments).

        Keyword Args:
            device (torch.device, optional): where to allocate the cache.
                Defaults to the device of the projection weights.
            dtype (torch.dtype, optional): dtype of the cache. Pass the
                compute dtype under autocast so cached keys and values are
                stored as the projections produce them, without a conversion
                on every step. Defaults to the dtype of the projection
                weights.

        Returns:
            A ``(k, v)`` tuple of zero tensors of shape ``[batch_size,
            num_layers, num_heads, max_seq_len, head_dim]``.
        """
        weight = self.blocks[0].qkv.weight
        shape = (
            batch_size,
            self.num_layers,
            self.num_heads,
            self.max_seq_len,
            self.head_dim,
        )
        device = weight.device if device is None else device
        dtype = weight.dtype if dtype is None else dtype
        return (
            torch.zeros(shape, dtype=dtype, device=device),
            torch.zeros(shape, dtype=dtype, device=device),
        )

    @staticmethod
    def reset_kv_cache(
        kv_cache: tuple[torch.Tensor, torch.Tensor], mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Clear the cache rows of the streams selected by ``mask``.

        Args:
            kv_cache (tuple of torch.Tensor): a cache from :meth:`new_kv_cache`.
            mask (torch.Tensor): a boolean tensor of shape ``[batch_size]``;
                ``True`` rows are zeroed in place.

        Returns:
            The same ``(k, v)`` tuple.
        """
        mask = mask.view(-1, 1, 1, 1, 1)
        for cache in kv_cache:
            cache.masked_fill_(mask, 0)
        return kv_cache

    def _check_positions(self, positions: torch.Tensor) -> None:
        if not is_compiling() and positions.max() >= self.max_seq_len:
            raise RuntimeError(
                f"Episode length exceeded max_seq_len={self.max_seq_len}. "
                "Increase max_seq_len or truncate episodes; sliding-window "
                "attention is not implemented."
            )

    def forward(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Run the backbone over a window or a single cached step.

        Args:
            features (torch.Tensor): ``[B, T, input_size]`` inputs. ``T`` must
                be ``1`` when ``kv_cache`` is provided.
            positions (torch.Tensor): ``[B, T]`` integer positions of each
                step within its episode.
            mask (torch.Tensor, optional): ``[B, T, T]`` boolean mask
                (``True`` = attend) for the window path; defaults to a plain
                causal mask. Ignored on the cached-step path, where validity
                is derived from ``positions``.
            kv_cache (tuple of torch.Tensor, optional): a cache from
                :meth:`new_kv_cache`. Providing it selects the cached-step
                path; the cache is updated in place at ``positions``.

        Returns:
            A tuple ``(out, kv_cache)`` with ``out`` of shape
            ``[B, T, hidden_size]`` and ``kv_cache`` the updated cache on the
            cached-step path (``None`` on the window path).
        """
        self._check_positions(positions)
        h = self.in_proj(features) + self.pos_emb(positions)
        if kv_cache is None:
            t = features.shape[1]
            if mask is None:
                mask = (
                    torch.ones(t, t, dtype=torch.bool, device=features.device)
                    .tril()
                    .expand(features.shape[0], t, t)
                )
            attn_mask = mask.unsqueeze(-3)
            for block in self.blocks:
                h = block(h, attn_mask)
            return self.norm(h), None
        if features.shape[1] != 1:
            raise ValueError(
                "The cached-step path expects a single step (T==1), got "
                f"T={features.shape[1]}. Pass kv_cache=None to process a "
                "window."
            )
        cache_k, cache_v = kv_cache
        positions = positions.squeeze(-1)
        valid = torch.arange(
            self.max_seq_len, device=features.device
        ) <= positions.view(-1, 1)
        attn_mask = valid.view(-1, 1, 1, self.max_seq_len)
        for layer, block in enumerate(self.blocks):
            h = block(
                h,
                attn_mask,
                cache_kv=(cache_k[:, layer], cache_v[:, layer]),
                positions=positions,
            )
        return self.norm(h), (cache_k, cache_v)


_BACKBONE_ATTRIBUTES = ("num_layers", "num_heads", "head_dim", "max_seq_len")
_BACKBONE_METHODS = ("new_kv_cache", "reset_kv_cache")


def _autocast_dtype(device: torch.device) -> torch.dtype | None:
    """Return the active autocast dtype for ``device``, or ``None`` when disabled."""
    device_type = device.type
    try:
        enabled = torch.is_autocast_enabled(device_type)
        return torch.get_autocast_dtype(device_type) if enabled else None
    except TypeError:
        if device_type == "cuda":
            return (
                torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else None
            )
        if device_type == "cpu":
            return (
                torch.get_autocast_cpu_dtype()
                if torch.is_autocast_cpu_enabled()
                else None
            )
        return None


class TransformerModule(ModuleBase):
    """A TensorDict wrapper turning a causal transformer into a temporal policy module.

    The transformer analogue of :class:`~torchrl.modules.LSTMModule`: the same
    network runs either over a full ``[B, T]`` window (training) or one step
    at a time against a key/value cache (collection), with matching outputs.
    The execution path is selected by the
    :class:`~torchrl.modules.set_recurrent_mode` context manager, exactly as
    for the recurrent modules.

    With the reference :class:`CausalTransformer`, no state travels in the tensordict. The
    key/value cache is inference state owned by the module instance: it is
    allocated by the backbone on the first cached step, indexed by batch
    position (one stream per environment of the batch), cleared wherever
    ``is_init`` is set (sourced from :class:`~torchrl.envs.InitTracker`),
    invalidated when the parameters change (in place or swapped for other
    tensors), and released by :meth:`reset_cache`. Copies and pickled
    instances start with an empty cache. Rollouts and replay buffers hold
    observations and features only, never a cache; the training path reads
    ``is_init`` to rebuild positions and a block-diagonal causal mask over the
    window.

    An explicit-state backbone such as :class:`~torchrl.modules.GTrXL` can expose
    ``forward_state(features, state, is_init, *, recurrent) -> (out, next_state)``
    and a ``state_spec`` :class:`~torchrl.data.Composite`. The wrapper passes
    flattened environment batches with shape ``[B, T]`` and preserves the
    returned state container. Use ``in_keys=[features, state, "is_init"]`` and
    ``out_keys=[out, ("next", state)]`` (including nested state keys), and attach
    :meth:`make_tensordict_primer`. This path stores state in trajectories and
    never reads or updates the module-owned cache. In recurrent mode the
    backbone must initialize from stored state at the window start and each
    ``is_init`` boundary, including boundaries inserted by a slice sampler.
    For compact replay, a ``[B]`` record can instead contain input features
    ``[B, T, F]``, ``is_init`` markers ``[B, T, 1]`` and one state with batch
    ``[B]``. In recurrent mode the backbone receives this single initial state
    and returns one final state. Output features have shape ``[B, T, H]``.
    With GTrXL this runs parallel masked attention and ``is_init`` marks real
    episode resets only. Sample whole records to keep state and observations
    aligned; a slice sampler cannot recover missing intermediate carries.

    The episode-alignment and ``max_seq_len`` restrictions below concern the
    reference module-owned-cache path.

    Args:
        input_size (int, optional): number of input features. Unused if
            ``transformer`` is passed.
        hidden_size (int, optional): dimension of the transformer's residual
            stream. Unused if ``transformer`` is passed.
        num_layers (int, optional): number of transformer blocks. Defaults to
            ``1``. Unused if ``transformer`` is passed.

    Keyword Args:
        num_heads (int, optional): number of attention heads. Required unless
            ``transformer`` is passed.
        max_seq_len (int, optional): maximum episode length (positional table
            and cache size). Required unless ``transformer`` is passed.
        dim_feedforward (int, optional): per-block MLP width. Defaults to
            ``4 * hidden_size``.
        dropout (float, optional): dropout probability. Defaults to ``0.0``.
        transformer (nn.Module, optional): a pre-built backbone honoring the
            contract described in :class:`~torchrl.modules.CausalTransformer`
            (``forward``, ``new_kv_cache`` and ``reset_kv_cache`` plus the
            ``num_layers``, ``num_heads``, ``head_dim`` and ``max_seq_len``
            attributes). Exclusive with the size arguments.
        in_key (NestedKey, optional): the input value key. Exclusive with
            ``in_keys``.
        in_keys (list of NestedKey, optional): the input value key, optionally
            followed by ``"is_init"``. Defaults to ``[in_key, "is_init"]``.
        out_key (NestedKey, optional): the output value key. Exclusive with
            ``out_keys``.
        out_keys (list of NestedKey, optional): a one-element list with the
            output value key. Defaults to ``[out_key]``.
        device (torch.device, optional): device to build the parameters on.
        default_recurrent_mode (bool, optional): the recurrent mode when not
            overridden by the :class:`~torchrl.modules.set_recurrent_mode`
            context manager. Defaults to ``False``.
        validate_windows (bool, optional): whether the module-owned-cache window path checks that
            every row starts with ``is_init=True`` and raises otherwise. The
            check is data-dependent: under :func:`torch.compile` it costs one
            graph break, and ``fullgraph=True`` rejects it at compile time.
            Pass ``False`` to compile the window path as one graph, in which
            case the caller is responsible for episode-aligned windows.
            Defaults to ``True``.

    .. note::
        The cache is discarded whenever the parameters change. Parameter
        edits in place or swapped parameter tensors are detected on the next
        eager step; TorchRL's weight-synchronization paths (collectors and
        the inference server) call :meth:`mark_weight_update` explicitly,
        which also covers compiled modules and updates that write through
        ``.data``. Call :meth:`mark_weight_update` (or :meth:`reset_cache`)
        yourself after updating the parameters by any other means.

    .. note::
        The batch position is the stream identity of the cached-step path:
        a module instance must see the same environments in the same order
        on every call, which is what a collector over a batched environment
        provides. Use one instance per collector (or per collector worker)
        and call :meth:`reset_cache` before reusing an instance with another
        environment. Batches whose composition changes between calls, such as
        the partial batches of an asynchronous collector, need a stream-keyed
        cache and are not supported by this module yet.

    .. note::
        Training windows must be episode-aligned: every row must start with
        ``is_init=True``, which is what complete-trajectory sampling
        provides. A window that starts mid-episode raises a ``ValueError``
        rather than silently recomputing the prefix from position ``0``.

    .. note::
        Episodes longer than ``max_seq_len`` raise an error; sliding-window
        attention is deliberately out of scope.

    Examples:
        >>> import torch
        >>> from tensordict.nn import TensorDictModule, TensorDictSequential
        >>> from torch import nn
        >>> from torchrl.envs import GymEnv, InitTracker, TransformedEnv
        >>> from torchrl.modules import TransformerModule, set_recurrent_mode
        >>> env = TransformedEnv(GymEnv("Pendulum-v1"), InitTracker())
        >>> module = TransformerModule(
        ...     input_size=env.observation_spec["observation"].shape[-1],
        ...     hidden_size=16,
        ...     num_layers=2,
        ...     num_heads=4,
        ...     max_seq_len=200,
        ...     in_key="observation",
        ...     out_key="embed",
        ... )
        >>> policy = TensorDictSequential(
        ...     module,
        ...     TensorDictModule(nn.Linear(16, 1), in_keys=["embed"], out_keys=["action"]),
        ... )
        >>> rollout = env.rollout(10, policy)
        >>> rollout["embed"].shape
        torch.Size([10, 16])
        >>> "transformer_state" in rollout.keys()
        False
        >>> with set_recurrent_mode(True):
        ...     window = module(rollout.exclude("embed").clone())
        >>> torch.allclose(window["embed"], rollout["embed"], atol=1e-5)
        True
    """

    DEFAULT_IN_KEYS = ["is_init"]

    def __init__(
        self,
        input_size: int | None = None,
        hidden_size: int | None = None,
        num_layers: int = 1,
        *,
        num_heads: int | None = None,
        max_seq_len: int | None = None,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        transformer: nn.Module | None = None,
        in_key=None,
        in_keys=None,
        out_key=None,
        out_keys=None,
        device=None,
        default_recurrent_mode: bool | None = None,
        validate_windows: bool = True,
    ):
        super().__init__()
        if transformer is not None:
            if input_size is not None or hidden_size is not None:
                raise ValueError(
                    "A transformer instance cannot be passed along with size "
                    "arguments."
                )
            explicit_state = callable(getattr(transformer, "forward_state", None))
            for attr in () if explicit_state else _BACKBONE_ATTRIBUTES:
                if not hasattr(transformer, attr):
                    raise ValueError(
                        "The transformer backbone must expose a "
                        f"{attr!r} attribute; see CausalTransformer for the "
                        "backbone contract."
                    )
            for method in () if explicit_state else _BACKBONE_METHODS:
                if not callable(getattr(transformer, method, None)):
                    raise ValueError(
                        "The transformer backbone must implement "
                        f"{method!r}; see CausalTransformer for the backbone "
                        "contract."
                    )
        else:
            if input_size is None or hidden_size is None:
                raise ValueError("input_size and hidden_size must be passed.")
            if num_heads is None or max_seq_len is None:
                raise ValueError("num_heads and max_seq_len must be passed.")
            transformer = CausalTransformer(
                input_size,
                hidden_size,
                num_layers,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                device=device,
            )
        if not ((in_key is None) ^ (in_keys is None)):
            raise ValueError(
                f"Either in_keys or in_key must be specified but not both or "
                f"none. Got {in_keys} and {in_key} respectively."
            )
        elif in_key:
            in_keys = [in_key, *self.DEFAULT_IN_KEYS]
        if not ((out_key is None) ^ (out_keys is None)):
            raise ValueError(
                f"Either out_keys or out_key must be specified but not both "
                f"or none. Got {out_keys} and {out_key} respectively."
            )
        elif out_key:
            out_keys = [out_key]
        in_keys = unravel_key_list(in_keys)
        out_keys = unravel_key_list(out_keys)
        self._explicit_state = callable(getattr(transformer, "forward_state", None))
        if self._explicit_state:
            if len(in_keys) == 2 and in_keys[-1] != "is_init":
                in_keys = [*in_keys, "is_init"]
            if len(in_keys) != 3 or in_keys[-1] != "is_init" or len(out_keys) != 2:
                raise ValueError(
                    "Explicit state requires in_keys=[features, state, 'is_init'] "
                    "and out_keys=[features, ('next', state)]."
                )
            state_key = in_keys[1]
            state_key = (state_key,) if isinstance(state_key, str) else state_key
            if out_keys[1] != ("next", *state_key):
                raise ValueError(
                    "The state output key must be ('next', *state_input_key)."
                )
        if (
            not isinstance(in_keys, (tuple, list))
            or (
                len(in_keys) != 1
                and not (len(in_keys) == 2 and in_keys[-1] == "is_init")
            )
            and not self._explicit_state
        ):
            raise ValueError(
                "TransformerModule expects 1 input: a value (and potentially "
                f"an 'is_init' marker). Got in_keys {in_keys} instead."
            )
        if not self._explicit_state and (
            not isinstance(out_keys, (tuple, list)) or len(out_keys) != 1
        ):
            raise ValueError(
                "TransformerModule expects 1 output: a value. Got out_keys "
                f"{out_keys} instead."
            )
        self.transformer = transformer
        if "is_init" not in in_keys:
            in_keys = in_keys + ["is_init"]
        self.in_keys = in_keys
        self.out_keys = out_keys
        self._recurrent_mode = default_recurrent_mode
        self.validate_windows = validate_windows
        self._kv_cache: Any = None
        self._positions: torch.Tensor | None = None
        self._cache_dtype: torch.dtype | None = None
        self._weights_version: tuple[tuple[int, int], ...] | None = None

    def make_tensordict_primer(self) -> TensorDictPrimer | None:
        """Initialize explicit recurrent state using the backbone's composite spec.

        Module-owned caches do not require an environment primer.
        """
        if not self._explicit_state:
            return None
        # EnvBase imports the module package before transforms are initialized.
        from torchrl.envs import TensorDictPrimer

        return TensorDictPrimer(
            Composite({unravel_key(self.in_keys[1]): self.transformer.state_spec}),
            expand_specs=True,
        )

    @property
    def recurrent_mode(self):
        rm = recurrent_mode()
        if rm is None:
            return bool(self._recurrent_mode)
        return rm

    @recurrent_mode.setter
    def recurrent_mode(self, value):
        raise RuntimeError(
            "recurrent_mode cannot be changed in-place. Please use the "
            "set_recurrent_mode context manager."
        )

    def reset_cache(self) -> None:
        """Release the key/value cache and the position counters.

        The next cached step allocates a fresh cache for the batch it sees.
        Call this before reusing the module with a different environment or
        collector.
        """
        self._kv_cache = None
        self._positions = None
        self._cache_dtype = None
        self._weights_version = None

    def mark_weight_update(self) -> None:
        """Discard the cache after a weight update.

        TorchRL's weight-synchronization paths (collectors and the inference
        server) call this through :func:`torchrl._utils.mark_weight_update`
        once new weights are applied, so every stream restarts instead of
        attending to keys and values computed with the previous weights. Call
        it yourself after updating the parameters by other means.

        Explicit caller-owned state is preserved. Stored activations may be
        stale relative to the new weights, as with other recurrent replay.
        """
        self.reset_cache()

    def __getstate__(self):
        """Pickle and copy the module without its cache: copies start empty."""
        state = dict(super().__getstate__())
        state["_kv_cache"] = None
        state["_positions"] = None
        state["_cache_dtype"] = None
        state["_weights_version"] = None
        return state

    def _current_weights_version(self) -> tuple[tuple[int, int], ...]:
        """Identify the parameter tensors and their in-place modification counters."""
        return tuple(
            (p.data_ptr(), int(p._version)) for p in self.transformer.parameters()
        )

    def _restart_mask(self, value: torch.Tensor) -> torch.Tensor:
        """Return the mask of streams whose cache must restart for this batch.

        A fresh cache is allocated when none exists, when the batch size or
        device changed, or when the parameters changed, in place or by being
        swapped for other tensors: cached keys and values computed with
        previous weights would otherwise be mixed with the current
        projections.
        """
        batch_size = value.shape[0]
        positions = self._positions
        cache_dtype = _autocast_dtype(value.device)
        stale = (
            positions is None
            or positions.shape[0] != batch_size
            or positions.device != value.device
            or cache_dtype != self._cache_dtype
        )
        if not stale and not is_compiling():
            version = self._current_weights_version()
            stale = version != self._weights_version
        if stale:
            self._kv_cache = self.transformer.new_kv_cache(
                batch_size, device=value.device, dtype=cache_dtype
            )
            self._positions = torch.zeros(
                batch_size, dtype=torch.long, device=value.device
            )
            self._cache_dtype = cache_dtype
            if not is_compiling():
                self._weights_version = self._current_weights_version()
            return torch.ones(batch_size, dtype=torch.bool, device=value.device)
        return torch.zeros(batch_size, dtype=torch.bool, device=value.device)

    @dispatch
    def forward(self, tensordict: TensorDictBase):
        """Run the transformer, honouring ``is_init`` for state resets.

        For the module-owned cache, with ``recurrent_mode=False`` one step is processed against the
        module's cache, whose rows are cleared where ``is_init`` is set; this
        path is inference only and runs under :func:`torch.no_grad`. With
        ``recurrent_mode=True``, a full ``(B, T)`` window is processed under a
        block-diagonal causal mask built from ``is_init``; the cache is
        neither read nor written, and gradients flow through the window.
        Explicit-state backbones receive the stored state in both modes;
        only their single-step collection path runs under ``torch.no_grad``.
        """
        shape = tensordict.shape
        value = tensordict.get(self.in_keys[0])
        if (
            self._explicit_state
            and self.recurrent_mode
            and value.ndim == tensordict.ndim + 2
        ):
            # Dense replay can store [B] records containing [B, T, F] features
            # and a single [B] carry. Time is a feature dimension of the outer
            # record, not a batch dimension shared by its initial state.
            flat = tensordict.reshape(-1)
            out, next_state = self.transformer.forward_state(
                flat.get(self.in_keys[0]),
                flat.get(self.in_keys[1]),
                flat.get("is_init").squeeze(-1),
                recurrent=True,
            )
            tensordict.set(self.out_keys[0], out.reshape(*shape, *out.shape[1:]))
            tensordict.set(self.out_keys[1], next_state.reshape(shape))
            return tensordict
        if self.recurrent_mode:
            td_ndim = tensordict.ndim
            if td_ndim == 0:
                raise ValueError(
                    "TransformerModule(recurrent_mode=True) requires the "
                    "input tensordict to have at least one batch dim (time). "
                    "Got a 0-d tensordict."
                )
            elif td_ndim == 1:
                tensordict_shaped = tensordict.unsqueeze(0)
            elif td_ndim == 2:
                tensordict_shaped = tensordict
            else:
                tensordict_shaped = tensordict.flatten(0, -2)
        else:
            tensordict_shaped = tensordict.reshape(-1).unsqueeze(-1)

        is_init = tensordict_shaped["is_init"].squeeze(-1)
        value = tensordict_shaped.get(self.in_keys[0])

        if self._explicit_state:
            state = tensordict_shaped.get(self.in_keys[1])
            if self.recurrent_mode:
                out, next_state = self.transformer.forward_state(
                    value, state, is_init, recurrent=True
                )
            else:
                with torch.no_grad():
                    out, next_state = self.transformer.forward_state(
                        value, state, is_init, recurrent=False
                    )
            tensordict_shaped.set(self.out_keys[1], next_state)
        elif self.recurrent_mode:
            if self.validate_windows and not is_init[..., 0].all():
                raise ValueError(
                    "TransformerModule(recurrent_mode=True) expects "
                    "episode-aligned windows: every row must start with "
                    "is_init=True. Sample complete trajectories (for instance "
                    "with a SliceSampler over episode boundaries) or include "
                    "the beginning of the episode in the window."
                )
            positions = positions_from_is_init(is_init)
            mask = segment_causal_mask_from_is_init(is_init)
            out, _ = self.transformer(value, positions, mask=mask)
        else:
            with torch.no_grad():
                init = is_init.reshape(-1) | self._restart_mask(value)
                kv_cache = self.transformer.reset_kv_cache(self._kv_cache, init)
                positions = self._positions.masked_fill(init, 0)
                out, kv_cache = self.transformer(
                    value, positions.unsqueeze(-1), kv_cache=kv_cache
                )
            self._kv_cache = kv_cache
            self._positions = positions + 1
        tensordict_shaped.set(self.out_keys[0], out)

        if shape != tensordict_shaped.shape or tensordict_shaped is not tensordict:
            tensordict.update(tensordict_shaped.reshape(shape))
        return tensordict
