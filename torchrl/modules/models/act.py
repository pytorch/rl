# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ACTModel(nn.Module):
    """Action Chunking with Transformers (ACT) backbone.

    Implements the model from *Learning Fine-Grained Bimanual Manipulation
    with Low-Cost Hardware* (`Zhao et al., 2023
    <https://arxiv.org/abs/2304.13705>`_).

    The model operates in two modes:

    **Training** — supply ``action_chunk``.  A CVAE encoder maps
    ``(observation, action_chunk)`` to a style latent ``z``; the Transformer
    decoder then reconstructs the full chunk conditioned on ``z``.

    **Inference** — omit ``action_chunk`` (or pass ``None``).  The latent
    defaults to ``z = 0`` (the prior mean), and the decoder produces the
    action chunk from the observation alone.

    The architecture uses a standard :class:`~torch.nn.TransformerEncoder`
    as the CVAE encoder and a DETR-style
    :class:`~torch.nn.TransformerDecoder` for action prediction.  The encoder
    inputs receive sinusoidal positional encodings; the decoder uses learned
    action queries (one per chunk timestep).

    Args:
        obs_dim (int): Proprioceptive / state observation dimension.
        action_dim (int): Action dimension.
        chunk_size (int): Number of actions predicted per forward pass (T).
        hidden_dim (int, optional): Transformer hidden dimension.
            Default: ``256``.
        nheads (int, optional): Number of attention heads.  Default: ``8``.
        num_encoder_layers (int, optional): CVAE encoder depth.
            Default: ``4``.
        num_decoder_layers (int, optional): Action decoder depth.
            Default: ``7``.
        latent_dim (int, optional): CVAE latent dimension.  Default: ``32``.
        dropout (float, optional): Dropout probability.  Default: ``0.1``.
        dim_feedforward (int, optional): Feedforward dimension inside each
            Transformer layer.  Default: ``hidden_dim * 4``.

    Shape:
        - ``observation``: ``(..., obs_dim)``
        - ``action_chunk``: ``(..., chunk_size, action_dim)`` — only during
          training
        - Returned ``action_pred``: ``(..., chunk_size, action_dim)``
        - Returned ``mu``, ``log_var``: ``(..., latent_dim)``

    Examples:
        >>> import torch
        >>> from torchrl.modules.models import ACTModel
        >>> model = ACTModel(obs_dim=14, action_dim=7, chunk_size=100)
        >>> obs = torch.randn(4, 14)
        >>> chunk = torch.randn(4, 100, 7)
        >>> # Training mode (encoder active)
        >>> action_pred, mu, log_var = model(obs, chunk)
        >>> action_pred.shape
        torch.Size([4, 100, 7])
        >>> # Inference mode (z = 0)
        >>> action_pred, mu, log_var = model(obs)
        >>> mu.eq(0).all().item()
        True
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dim: int = 256,
        nheads: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 7,
        latent_dim: int = 32,
        dropout: float = 0.1,
        dim_feedforward: int | None = None,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        if dim_feedforward is None:
            dim_feedforward = hidden_dim * 4

        # ── Shared observation projection ────────────────────────────────
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)

        # ── CVAE encoder (training only) ─────────────────────────────────
        # Input sequence: [CLS, action_0, ..., action_{T-1}, obs]
        # CLS token output → z_mu, z_log_var
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.cvae_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_encoder_layers
        )
        # Sinusoidal pos-enc covering [CLS + T actions + 1 obs] = T+2 tokens
        self.register_buffer(
            "enc_pos",
            _sinusoidal_pos_enc(2 + chunk_size, hidden_dim),
            persistent=False,
        )
        # Project CLS output → (mu, log_var)
        self.latent_proj = nn.Linear(hidden_dim, latent_dim * 2)
        # Project sampled z → hidden_dim for the decoder
        self.latent_out_proj = nn.Linear(latent_dim, hidden_dim)

        # ── DETR-style action decoder ─────────────────────────────────────
        # Learned queries: one per action timestep in the chunk
        self.action_queries = nn.Embedding(chunk_size, hidden_dim)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            dec_layer, num_layers=num_decoder_layers
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)

    # ─────────────────────────────────────────────────────────────────────
    def forward(
        self,
        observation: torch.Tensor,
        action_chunk: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict an action chunk from an observation.

        Args:
            observation (Tensor): ``(..., obs_dim)`` proprioceptive state.
            action_chunk (Tensor | None): ``(..., chunk_size, action_dim)``
                expert actions.  Pass during training; ``None`` at inference.

        Returns:
            Tuple of:
            - ``action_pred`` ``(..., chunk_size, action_dim)``
            - ``mu`` ``(..., latent_dim)`` — CVAE mean (``0`` at inference)
            - ``log_var`` ``(..., latent_dim)`` — CVAE log-variance (``0`` at
              inference)
        """
        batch = observation.shape[:-1]
        flat_b = math.prod(batch)

        obs_flat = observation.reshape(flat_b, self.obs_dim)
        obs_tok = self.obs_proj(obs_flat).unsqueeze(1)  # (B, 1, D), shared

        if action_chunk is not None:
            # ── CVAE encoder ──────────────────────────────────────────────
            acts_flat = action_chunk.reshape(flat_b, self.chunk_size, self.action_dim)

            cls_tok = self.cls_embed.weight.unsqueeze(0).expand(flat_b, -1, -1)
            act_tok = self.action_proj(acts_flat)  # (B, T, D)

            # [CLS | actions | obs] — total length T+2
            enc_in = torch.cat([cls_tok, act_tok, obs_tok], dim=1)
            enc_in = enc_in + self.enc_pos[: enc_in.size(1)]

            enc_out = self.cvae_encoder(enc_in)  # (B, T+2, D)
            cls_out = enc_out[:, 0]  # (B, D)

            params = self.latent_proj(cls_out)  # (B, 2*latent_dim)
            mu, log_var = params.chunk(2, dim=-1)
            z = _reparameterise(mu, log_var)  # (B, latent_dim)
        else:
            # ── Prior (inference): allocate three independent zero tensors
            # so callers can mutate one without affecting the others.
            mu = obs_flat.new_zeros(flat_b, self.latent_dim)
            log_var = obs_flat.new_zeros(flat_b, self.latent_dim)
            z = obs_flat.new_zeros(flat_b, self.latent_dim)

        # ── Transformer decoder ───────────────────────────────────────────
        z_tok = self.latent_out_proj(z).unsqueeze(1)  # (B, 1, D)
        memory = torch.cat([obs_tok, z_tok], dim=1)  # (B, 2, D)

        queries = self.action_queries.weight.unsqueeze(0).expand(flat_b, -1, -1)
        dec_out = self.transformer_decoder(queries, memory)  # (B, T, D)
        action_pred = self.action_head(dec_out)  # (B, T, action_dim)

        # Restore original batch shape
        action_pred = action_pred.reshape(*batch, self.chunk_size, self.action_dim)
        mu = mu.reshape(*batch, self.latent_dim)
        log_var = log_var.reshape(*batch, self.latent_dim)

        return action_pred, mu, log_var


# ── Module-level helpers ──────────────────────────────────────────────────────


def _reparameterise(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Sample z ~ N(mu, exp(log_var)) using the reparameterisation trick."""
    std = (0.5 * log_var).exp()
    return mu + std * torch.randn_like(std)


def _sinusoidal_pos_enc(length: int, dim: int) -> torch.Tensor:
    """Return a ``(length, dim)`` sinusoidal positional encoding tensor."""
    if dim % 2:
        raise ValueError(f"_sinusoidal_pos_enc requires an even `dim`, got {dim}.")
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    enc = torch.zeros(length, dim)
    enc[:, 0::2] = torch.sin(pos * div)
    enc[:, 1::2] = torch.cos(pos * div)
    return enc
