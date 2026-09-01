# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch
import torch.nn.functional as F

from tensordict import TensorDictBase, unravel_key_list
from tensordict.base import NO_DEFAULT
from tensordict.nn import dispatch, TensorDictModuleBase as ModuleBase
from tensordict.utils import expand_as_right
from torch import nn

from torchrl._utils import is_compiling
from torchrl.data.tensor_specs import Unbounded
from torchrl.modules.tensordict_module.rnn import recurrent_mode


def positions_from_is_init(is_init: torch.Tensor) -> torch.Tensor:
    """Compute per-token positions within each episode segment of a window.

    Positions restart at ``0`` on every ``is_init`` flag. The first step of
    the window is always treated as position ``0``: like the recurrent
    modules, a window is assumed to start at the beginning of a trajectory.

    Args:
        is_init (torch.Tensor): a boolean tensor of shape ``[*batch, T]``
            marking the first step of each episode.

    Returns:
        A ``torch.long`` tensor of shape ``[*batch, T]`` holding the position
        of each step within its episode segment.

    Examples:
        >>> is_init = torch.tensor([[False, False, True, False]])
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
    ) -> None:
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
            cache_k[batch, :, positions] = k.squeeze(2)
            cache_v[batch, :, positions] = v.squeeze(2)
            k, v = cache_k, cache_v
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        h = h + self.out_proj(attn.transpose(1, 2).flatten(-2))
        return h + self.mlp(self.norm2(h))


class CausalTransformer(nn.Module):
    """A causal transformer backbone with matching windowed and cached-step semantics.

    This is the reference implementation of the temporal-transformer backbone
    contract consumed by :class:`~torchrl.modules.TransformerModule`:

        ``forward(features, positions, mask=None, kv_cache=None) -> (out, kv_cache)``

    Any module honoring that signature and exposing ``num_layers``,
    ``num_heads``, ``head_dim`` and ``max_seq_len`` attributes can be used in
    its place.

    Two execution paths share the same parameters and produce the same
    outputs: a window path processing ``[B, T]`` at once under a causal mask
    (training), and a cached-step path attending against a fixed-shape
    key/value cache (collection). Positions are always explicit inputs, which
    is what keeps the two paths consistent across episode resets.

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
    ) -> None:
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
            kv_cache (tuple of torch.Tensor, optional): ``(k, v)`` caches of
                shape ``[B, num_layers, num_heads, max_seq_len, head_dim]``.
                Providing them selects the cached-step path. The caches passed
                in are not modified; updated copies are returned.

        Returns:
            A tuple ``(out, kv_cache)`` with ``out`` of shape
            ``[B, T, hidden_size]`` and ``kv_cache`` the updated ``(k, v)``
            tuple on the cached-step path (``None`` on the window path).
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
        cache_k, cache_v = (c.clone() for c in kv_cache)
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


class TransformerModule(ModuleBase):
    """A TensorDict wrapper turning a causal transformer into a temporal policy module.

    The transformer analogue of :class:`~torchrl.modules.LSTMModule`: the same
    network runs either over a full ``[B, T]`` window (training) or one step
    at a time against a fixed-shape key/value cache carried in the tensordict
    (collection), with matching outputs. The execution path is selected by the
    :class:`~torchrl.modules.set_recurrent_mode` context manager, exactly as
    for the recurrent modules.

    State transport follows the recurrent-module pattern: the cache and
    position entries under ``"transformer_state"`` travel in the tensordict,
    are declared to the environment through :meth:`make_tensordict_primer`,
    and are zeroed wherever ``is_init`` is set (sourced from
    :class:`~torchrl.envs.InitTracker`). In recurrent (window) mode, episode
    boundaries are handled with a block-diagonal causal mask and restarting
    positions instead, so no cache is read or written.

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
            (the ``forward`` signature plus ``num_layers``, ``num_heads``,
            ``head_dim`` and ``max_seq_len`` attributes). Exclusive with the
            size arguments.
        in_key (NestedKey, optional): the input value key. Exclusive with
            ``in_keys``.
        in_keys (list of NestedKey, optional): the input value key followed by
            the three state keys (k, v, pos). Defaults to
            ``[in_key, ("transformer_state", "k"), ("transformer_state", "v"),
            ("transformer_state", "pos")]``.
        out_key (NestedKey, optional): the output value key. Exclusive with
            ``out_keys``.
        out_keys (list of NestedKey, optional): the output value key followed
            by the three next-state keys. Defaults to
            ``[out_key, ("next", "transformer_state", "k"), ("next",
            "transformer_state", "v"), ("next", "transformer_state", "pos")]``.
        device (torch.device, optional): device to build the parameters on.
        default_recurrent_mode (bool, optional): the recurrent mode when not
            overridden by the :class:`~torchrl.modules.set_recurrent_mode`
            context manager. Defaults to ``False``.

    .. note::
        Unlike :class:`~torchrl.modules.LSTMModule`, only the single-step path
        writes the ``("next", ...)`` state entries: the cache is inference
        state, per-step copies of it would be prohibitively large, and
        training consumes stored windows, not caches.

    .. note::
        Episodes longer than ``max_seq_len`` raise an error; sliding-window
        attention is deliberately out of scope.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.envs import GymEnv, InitTracker, TransformedEnv
        >>> from torchrl.modules import TransformerModule
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
        >>> env = env.append_transform(module.make_tensordict_primer())
        >>> td = env.reset()
        >>> td = module(td)
        >>> td["embed"].shape
        torch.Size([16])
    """

    DEFAULT_IN_KEYS = [
        ("transformer_state", "k"),
        ("transformer_state", "v"),
        ("transformer_state", "pos"),
    ]
    DEFAULT_OUT_KEYS = [
        ("next", "transformer_state", "k"),
        ("next", "transformer_state", "v"),
        ("next", "transformer_state", "pos"),
    ]

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
    ) -> None:
        super().__init__()
        if transformer is not None:
            if input_size is not None or hidden_size is not None:
                raise ValueError(
                    "A transformer instance cannot be passed along with size "
                    "arguments."
                )
            for attr in ("num_layers", "num_heads", "head_dim", "max_seq_len"):
                if not hasattr(transformer, attr):
                    raise ValueError(
                        "The transformer backbone must expose a "
                        f"{attr!r} attribute; see CausalTransformer for the "
                        "backbone contract."
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
            out_keys = [out_key, *self.DEFAULT_OUT_KEYS]
        in_keys = unravel_key_list(in_keys)
        out_keys = unravel_key_list(out_keys)
        if not isinstance(in_keys, (tuple, list)) or (
            len(in_keys) != 4 and not (len(in_keys) == 5 and in_keys[-1] == "is_init")
        ):
            raise ValueError(
                "TransformerModule expects 4 inputs: a value, the k and v "
                "caches and a position counter (and potentially an 'is_init' "
                f"marker). Got in_keys {in_keys} instead."
            )
        if not isinstance(out_keys, (tuple, list)) or len(out_keys) != 4:
            raise ValueError(
                "TransformerModule expects 4 outputs: a value, the k and v "
                f"caches and a position counter. Got out_keys {out_keys} "
                "instead."
            )
        self.transformer = transformer
        if "is_init" not in in_keys:
            in_keys = in_keys + ["is_init"]
        self.in_keys = in_keys
        self.out_keys = out_keys
        self._recurrent_mode = default_recurrent_mode

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

    def make_tensordict_primer(self):
        """Makes a tensordict primer for the environment.

        A :class:`~torchrl.envs.TensorDictPrimer` object ensures that the
        cache and position entries are registered in the environment specs,
        so batched and parallel environments carry them between steps. See
        :meth:`torchrl.modules.LSTMModule.make_tensordict_primer` for the
        rationale; the mechanics are identical.

        Examples:
            >>> from torchrl.envs import GymEnv, InitTracker, TransformedEnv
            >>> from torchrl.modules import TransformerModule
            >>> env = TransformedEnv(GymEnv("Pendulum-v1"), InitTracker())
            >>> module = TransformerModule(
            ...     input_size=3, hidden_size=16, num_heads=4, max_seq_len=200,
            ...     in_key="observation", out_key="embed")
            >>> env = env.append_transform(module.make_tensordict_primer())
        """
        from torchrl.envs.transforms.transforms import TensorDictPrimer

        def make_tuple(key):
            if isinstance(key, tuple):
                return key
            return (key,)

        for in_key, out_key in zip(self.in_keys[1:4], self.out_keys[1:4]):
            if make_tuple(out_key) != ("next", *make_tuple(in_key)):
                raise RuntimeError(
                    "make_tensordict_primer is supposed to work with "
                    "in_keys/out_keys that have compatible names, ie. the "
                    "out_keys should be named after ('next', <in_key>). Got "
                    f"in_keys={self.in_keys} and out_keys={self.out_keys} "
                    "instead."
                )
        transformer = self.transformer
        cache_shape = (
            transformer.num_layers,
            transformer.num_heads,
            transformer.max_seq_len,
            transformer.head_dim,
        )
        return TensorDictPrimer(
            {
                self.in_keys[1]: Unbounded(shape=cache_shape),
                self.in_keys[2]: Unbounded(shape=cache_shape),
                self.in_keys[3]: Unbounded(shape=(1,), dtype=torch.long),
            },
            expand_specs=True,
        )

    def _init_cache(self, value: torch.Tensor) -> torch.Tensor:
        transformer = self.transformer
        return value.new_zeros(
            value.shape[0],
            transformer.num_layers,
            transformer.num_heads,
            transformer.max_seq_len,
            transformer.head_dim,
        )

    @dispatch
    def forward(self, tensordict: TensorDictBase):
        """Run the transformer, honouring ``is_init`` for state resets.

        With ``recurrent_mode=False``, one step is processed against the
        cache carried in the tensordict (zeroed where ``is_init`` is set) and
        the updated state is written under the ``("next", ...)`` keys. With
        ``recurrent_mode=True``, a full ``(B, T)`` window is processed under
        a block-diagonal causal mask built from ``is_init``; no cache is read
        or written.
        """
        defaults = [NO_DEFAULT, None, None, None]
        shape = tensordict.shape
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
        value, cache_k, cache_v, pos = (
            tensordict_shaped.get(key, default)
            for key, default in zip(self.in_keys, defaults)
        )

        if self.recurrent_mode:
            positions = positions_from_is_init(is_init)
            mask = segment_causal_mask_from_is_init(is_init)
            out, _ = self.transformer(value, positions, mask=mask)
            tensordict_shaped.set(self.out_keys[0], out)
        else:
            if cache_k is None:
                cache_k, cache_v = self._init_cache(value), self._init_cache(value)
            else:
                cache_k, cache_v = cache_k.squeeze(1), cache_v.squeeze(1)
            if pos is None:
                pos = value.new_zeros(value.shape[0], 1, dtype=torch.long)
            else:
                pos = pos.squeeze(-1)
            init = is_init.view(-1)
            cache_k, cache_v, pos = (
                t.masked_fill(expand_as_right(init, t), 0)
                for t in (cache_k, cache_v, pos)
            )
            out, (cache_k, cache_v) = self.transformer(
                value, pos, kv_cache=(cache_k, cache_v)
            )
            tensordict_shaped.set(self.out_keys[0], out)
            tensordict_shaped.set(self.out_keys[1], cache_k.unsqueeze(1))
            tensordict_shaped.set(self.out_keys[2], cache_v.unsqueeze(1))
            tensordict_shaped.set(self.out_keys[3], (pos + 1).unsqueeze(-1))

        if shape != tensordict_shaped.shape or tensordict_shaped is not tensordict:
            tensordict.update(tensordict_shaped.reshape(shape))
        return tensordict
