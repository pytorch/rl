# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Pretrained MicroDuck locomotion policy.

:class:`MicroDuckPolicy` is the network architecture of the published
``torchrl/microduck-skills`` walker checkpoints, packaged as a
:class:`~tensordict.nn.TensorDictModule` with matching in/out keys. It reads
proprioception (``observation``), the active task id (``task_id``), the GRU
carry (``recurrent_state``), and the episode-start marker (``is_init``); it
writes joint-space actions plus the next recurrent state.

Use :meth:`MicroDuckPolicy.from_pretrained` to download and instantiate the
published walker in one call.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from tensordict.nn import (
    NormalParamExtractor,
    TensorDictModule,
    TensorDictSequential,
)
from torch import nn

from torchrl.modules.distributions import TanhNormal
from torchrl.modules.tensordict_module.actors import ProbabilisticActor
from torchrl.modules.tensordict_module.rnn import GRUModule

if TYPE_CHECKING:
    from torchrl.envs.custom.mujoco.microduck import MicroDuckTask

_has_huggingface_hub = importlib.util.find_spec("huggingface_hub") is not None

RECURRENT_STATE_KEY = "recurrent_state"


class _TaskConditionedEncoder(nn.Module):
    """Observation encoder conditioned on the task index.

    The observation carries the command but no other task parameter, so a
    learned embedding of the task index tells the policy which task of the
    library the env is in (for instance jumping, whose command is zero).
    """

    def __init__(
        self,
        observation_dim: int,
        num_tasks: int,
        hidden_size: int,
        *,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.observation = nn.Linear(observation_dim, hidden_size, device=device)
        self.task = nn.Embedding(num_tasks, hidden_size, device=device)

    def forward(self, observation: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        return torch.tanh(
            self.observation(observation) + self.task(task_id.squeeze(-1))
        )


class _GaussianHead(nn.Module):
    """Plain Gaussian policy head for training from scratch.

    The mean starts near zero, which is the ``STAND`` pose, and the
    state-independent exploration scale starts at ``initial_policy_scale``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_actions: int,
        *,
        initial_policy_scale: float,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.loc = nn.Linear(hidden_size, num_actions, device=device)
        nn.init.orthogonal_(self.loc.weight, gain=0.01)
        nn.init.zeros_(self.loc.bias)
        self.scale = nn.Parameter(torch.zeros(num_actions, device=device))
        self.param_extractor = NormalParamExtractor(
            scale_mapping=f"biased_softplus_{initial_policy_scale}"
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        loc = self.loc(features)
        return self.param_extractor(torch.cat((loc, self.scale.expand_as(loc)), -1))


class MicroDuckPolicy:
    """Factory for the published MicroDuck locomotion policy.

    Builds a :class:`~torchrl.modules.ProbabilisticActor` whose backbone is a
    task-conditioned observation encoder plus a single-layer GRU, topped by a
    Gaussian head over the MicroDuck joint targets. Mirrors the architecture
    used to train the released walkers.

    Use :meth:`from_pretrained` to download the published walker from
    Hugging Face in one call; use :meth:`from_checkpoint` for a local
    ``*.ckpt`` produced by ``examples.microduck.train_skills``. Both return
    the frozen actor together with its ordered task library (as a stacked
    :class:`~torchrl.envs.MicroDuckTask`).

    Args:
        hidden_size: hidden width of the GRU and the policy head.
        num_tasks: number of skills the walker was trained on; drives the task
            embedding table.
        observation_dim: width of the MicroDuck proprioceptive observation.
        num_actions: joint count of the deployed robot.
        initial_policy_scale: exploration standard deviation at the start of
            training; the published walker uses ``1.0``.
        device: target device for the policy's parameters.
        action_low / action_high: per-action bounds. The published walker was
            trained with a TanhNormal over ``[-1, 1]`` and relies on
            ``action_scale`` in the wrapped environment to denormalize.

    Examples:
        >>> from torchrl.modules import MicroDuckPolicy
        >>> policy = MicroDuckPolicy(
        ...     hidden_size=32, num_tasks=2, observation_dim=56, num_actions=14
        ... )
        >>> type(policy.actor).__name__
        'ProbabilisticActor'

        Download the published checkpoint and its task library:

        >>> from torchrl.modules import MicroDuckPolicy  # doctest: +SKIP
        >>> walker, skill_tasks, action_scale = MicroDuckPolicy.from_pretrained()  # doctest: +SKIP
        >>> walker  # doctest: +SKIP
        ProbabilisticActor(...)
    """

    DEFAULT_REPO_ID = "torchrl/microduck-skills"
    DEFAULT_FILENAME = "walker.ckpt"
    DEFAULT_REVISION = "4191d7d25c4fd58a5c6e6395fcf8217459fdd073"

    def __init__(
        self,
        hidden_size: int,
        num_tasks: int,
        observation_dim: int,
        num_actions: int,
        *,
        initial_policy_scale: float = 1.0,
        device: torch.device | str = "cpu",
        action_low: float = -1.0,
        action_high: float = 1.0,
    ):
        if initial_policy_scale <= 0:
            raise ValueError("initial_policy_scale must be positive.")
        self.hidden_size = int(hidden_size)
        self.num_tasks = int(num_tasks)
        self.observation_dim = int(observation_dim)
        self.num_actions = int(num_actions)
        self.device = torch.device(device)

        embed = TensorDictModule(
            _TaskConditionedEncoder(
                observation_dim,
                num_tasks,
                hidden_size,
                device=self.device,
            ),
            in_keys=["observation", "task_id"],
            out_keys=["embed"],
        )
        gru = GRUModule(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            in_keys=["embed", RECURRENT_STATE_KEY, "is_init"],
            out_keys=["features", ("next", RECURRENT_STATE_KEY)],
            device=self.device,
        )
        actor_head = TensorDictModule(
            _GaussianHead(
                hidden_size,
                num_actions,
                initial_policy_scale=initial_policy_scale,
                device=self.device,
            ),
            in_keys=["features"],
            out_keys=["loc", "scale"],
        )
        backbone = TensorDictSequential(embed, gru)
        self.actor = ProbabilisticActor(
            module=TensorDictSequential(backbone, actor_head),
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={"low": action_low, "high": action_high},
            return_log_prob=True,
        )

    @classmethod
    def from_config(
        cls,
        policy_kwargs: Mapping[str, Any],
        *,
        num_tasks: int,
        observation_dim: int,
        num_actions: int,
        device: torch.device | str = "cpu",
    ) -> MicroDuckPolicy:
        """Build from a ``policy_kwargs`` mapping."""
        kwargs = dict(policy_kwargs)
        policy_head = kwargs.pop("policy_head", "gaussian")
        if policy_head != "gaussian":
            raise ValueError(
                f"MicroDuckPolicy supports only the 'gaussian' head; got "
                f"{policy_head!r}. The 'gait-residual' head is example-side."
            )
        return cls(
            num_tasks=num_tasks,
            observation_dim=observation_dim,
            num_actions=num_actions,
            device=device,
            **kwargs,
        )

    @staticmethod
    def _checkpoint_payload(
        source: str | Path | Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if isinstance(source, Mapping):
            return source
        # Import lazily to avoid torchrl.render re-entering torchrl.envs while
        # torchrl.modules is still being initialized.
        from torchrl.render import load_checkpoint

        path = Path(source).expanduser().resolve()
        return load_checkpoint(path, weights_only=True)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path | Mapping[str, Any],
        *,
        device: torch.device | str = "cpu",
        freeze: bool = True,
    ) -> tuple[ProbabilisticActor, MicroDuckTask, float]:
        """Rebuild from a training checkpoint produced by ``examples.microduck.train_skills``.

        The checkpoint carries the ordered task library, the architecture
        arguments (``policy_kwargs``), and the env-side action scale.
        """
        # Import lazily to avoid a cycle between torchrl.envs and torchrl.modules.
        from torchrl.envs.custom.mujoco.microduck import MicroDuckEnv

        payload = cls._checkpoint_payload(checkpoint)
        policy_kwargs = dict(payload.get("policy_kwargs", {}))
        config = payload.get("config", {})
        task_specs = config.get("env", {}).get("tasks", [])
        if not isinstance(task_specs, list):
            raise TypeError(
                "MicroDuckPolicy could not read the ordered task list from "
                "config.env.tasks."
            )
        tasks = torch.stack(
            [
                getattr(MicroDuckEnv, spec["preset"])(
                    **{k: v for k, v in spec.items() if k != "preset"}
                )
                for spec in task_specs
            ]
        )
        policy = cls.from_config(
            policy_kwargs,
            num_tasks=len(tasks),
            observation_dim=MicroDuckEnv.OBSERVATION_DIM,
            num_actions=MicroDuckEnv.NUM_JOINTS,
            device=device,
        )
        policy.actor.load_state_dict(payload["model_state_dict"])
        if freeze:
            policy.actor.eval().requires_grad_(False)
        action_scale = config.get("env", {}).get("action_scale", 1.0)
        return policy.actor, tasks, action_scale

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str | None = None,
        *,
        filename: str | None = None,
        revision: str | None = None,
        device: torch.device | str = "cpu",
        freeze: bool = True,
        download: bool = True,
        **hub_kwargs: Any,
    ) -> tuple[ProbabilisticActor, MicroDuckTask, float]:
        """Download the published walker and rebuild it.

        Args:
            repo_id: Hugging Face repo. Defaults to ``"torchrl/microduck-skills"``.
            filename: checkpoint filename. Defaults to ``"walker.ckpt"``.
            revision: pinned commit hash. Defaults to the published six-skill
                walker revision.
            device: device for the rebuilt actor.
            freeze: load in eval() mode and disable gradients.
            download: must be ``True`` to fetch from the hub; pass a local
                path via :meth:`from_checkpoint` otherwise.
            **hub_kwargs: extra ``huggingface_hub.hf_hub_download`` kwargs.

        Returns:
            ``(actor, skill_tasks, action_scale)``.
        """
        if not download:
            raise ValueError(
                "MicroDuckPolicy.from_pretrained fetches the checkpoint from "
                "Hugging Face. Pass download=True, or use "
                "MicroDuckPolicy.from_checkpoint(path) for a local file."
            )
        if not _has_huggingface_hub:
            raise ImportError(
                "huggingface_hub is required to load MicroDuckPolicy from the "
                "hub. Install it with `pip install huggingface-hub`."
            )
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=repo_id or cls.DEFAULT_REPO_ID,
            filename=filename or cls.DEFAULT_FILENAME,
            revision=revision or cls.DEFAULT_REVISION,
            **hub_kwargs,
        )
        return cls.from_checkpoint(path, device=device, freeze=freeze)
