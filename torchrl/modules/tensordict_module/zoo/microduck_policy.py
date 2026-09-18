# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Task-conditioned policy architecture for MicroDuck skills."""

from __future__ import annotations

import hashlib
import importlib.util
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch
from tensordict.nn import (
    NormalParamExtractor,
    TensorDictModule,
    TensorDictModuleBase,
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
    """Encode proprioception together with the active task index."""

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
    """Gaussian joint-target head used by the published skill policies."""

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


class MicroDuckSkillPolicy(ProbabilisticActor):
    """Recurrent policy shared by a library of MicroDuck skills.

    The policy combines a task-conditioned observation encoder, a single-layer
    GRU and a Gaussian head over normalized joint targets. It is an ordinary
    :class:`~torchrl.modules.ProbabilisticActor`: ``task_id`` selects the skill
    embedding, while ``recurrent_state`` and ``is_init`` make its memory
    explicit in the input TensorDict.

    Args:
        hidden_size: Hidden width of the GRU and policy head.
        num_tasks: Number of skills indexed by the task embedding.
        observation_dim: Width of the proprioceptive observation.
        num_actions: Number of robot joints controlled by the policy.
        initial_policy_scale: Initial exploration standard deviation.
        device: Device for the policy parameters.
        action_low: Lower normalized action bound.
        action_high: Upper normalized action bound.

    Examples:
        >>> from torchrl.modules.tensordict_module.zoo import MicroDuckSkillPolicy
        >>> skill_policy = MicroDuckSkillPolicy(
        ...     hidden_size=32,
        ...     num_tasks=2,
        ...     observation_dim=56,
        ...     num_actions=14,
        ... )
        >>> type(skill_policy).__name__
        'MicroDuckSkillPolicy'

        Pair a trained policy with its ordered task library before deployment:

        >>> import torch
        >>> from torchrl.envs import MicroDuckEnv
        >>> from torchrl.modules.tensordict_module.zoo import MicroDuckSkills
        >>> task_library = torch.stack([
        ...     MicroDuckEnv.standing_task(),
        ...     MicroDuckEnv.tracking_task(0.2),
        ... ])
        >>> skills = MicroDuckSkills(skill_policy, task_library, action_scale=1.0)

    .. seealso::
        :class:`MicroDuckSkills` packages the policy with the task metadata
        needed for deployment; :class:`~torchrl.envs.MicroDuckEnv` is the
        joint-level environment used to train it; and
        :class:`~torchrl.modules.GRUModule` provides its recurrent core.
    """

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
        embed = TensorDictModule(
            _TaskConditionedEncoder(
                observation_dim,
                num_tasks,
                hidden_size,
                device=device,
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
            device=device,
        )
        actor_head = TensorDictModule(
            _GaussianHead(
                hidden_size,
                num_actions,
                initial_policy_scale=initial_policy_scale,
                device=device,
            ),
            in_keys=["features"],
            out_keys=["loc", "scale"],
        )
        backbone = TensorDictSequential(embed, gru)
        super().__init__(
            # Preserve the nested backbone used by the published checkpoints.
            module=TensorDictSequential(backbone, actor_head),
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={"low": action_low, "high": action_high},
            return_log_prob=True,
        )
        self.hidden_size = int(hidden_size)
        self.num_tasks = int(num_tasks)
        self.observation_dim = int(observation_dim)
        self.num_actions = int(num_actions)

    @classmethod
    def from_config(
        cls,
        policy_kwargs: Mapping[str, Any],
        *,
        num_tasks: int,
        observation_dim: int,
        num_actions: int,
        device: torch.device | str = "cpu",
    ) -> MicroDuckSkillPolicy:
        """Build the policy architecture recorded in checkpoint metadata."""
        kwargs = dict(policy_kwargs)
        policy_head = kwargs.pop("policy_head", "gaussian")
        if policy_head != "gaussian":
            raise ValueError(
                "MicroDuckSkillPolicy supports only the 'gaussian' head; got "
                f"{policy_head!r}."
            )
        return cls(
            num_tasks=num_tasks,
            observation_dim=observation_dim,
            num_actions=num_actions,
            device=device,
            **kwargs,
        )


@dataclass
class MicroDuckSkills:
    """A deployable MicroDuck skill policy and its environment metadata.

    ``policy`` maps a task-conditioned MicroDuck observation to normalized
    joint targets. ``task_library`` preserves the exact meaning and order of
    its task embeddings. ``action_scale`` records how the joint targets were
    applied during training. Keeping the three together prevents a high-level
    environment from silently pairing a policy with incompatible task ids or
    motor scaling.

    Args:
        policy: Trained task-conditioned TensorDict policy.
        task_library: Ordered, stacked :class:`~torchrl.envs.MicroDuckTask`.
        action_scale: Environment-side joint-target scale used during training.

    Examples:
        Download the pinned published skills and pass the resulting object to
        the high-level environment rather than unpacking policy metadata:

        >>> from torchrl.modules.tensordict_module.zoo import MicroDuckSkills
        >>> skills = MicroDuckSkills.from_pretrained()  # doctest: +SKIP
        >>> skill_policy = skills.policy  # doctest: +SKIP
        >>> task_library = skills.task_library  # doctest: +SKIP

    .. seealso::
        :class:`MicroDuckSkillPolicy` is the neural policy stored here;
        :class:`~torchrl.envs.MicroDuckTask` describes one row of the ordered
        task library; and :class:`~torchrl.envs.MicroDuckEnv` supplies the
        joint-level training dynamics.
    """

    policy: TensorDictModuleBase
    task_library: MicroDuckTask
    action_scale: float

    DEFAULT_REPO_ID = "torchrl/microduck-skills"
    DEFAULT_FILENAME = "walker.ckpt"
    DEFAULT_REVISION = "4191d7d25c4fd58a5c6e6395fcf8217459fdd073"

    @staticmethod
    def _checkpoint_payload(
        source: str | Path | Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if isinstance(source, Mapping):
            return source
        # Runtime import avoids modules -> render -> envs -> modules while
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
        sha256: str | None = None,
    ) -> MicroDuckSkills:
        """Rebuild the policy and deployment metadata from a checkpoint.

        Args:
            checkpoint: Local checkpoint path or an already loaded payload.
            device: Device for the rebuilt policy.
            freeze: Load in evaluation mode and disable gradients.
            sha256: Expected SHA-256 digest for a path. This cannot be used
                with an already loaded payload.
        """
        # Runtime import avoids envs importing the modules package while the
        # module-zoo namespace is being initialized.
        from torchrl.envs.custom.mujoco.microduck import MicroDuckEnv

        if sha256 is not None:
            if isinstance(checkpoint, Mapping):
                raise TypeError("sha256 cannot verify an already loaded payload.")
            path = Path(checkpoint).expanduser().resolve()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != sha256.lower():
                raise ValueError(
                    f"The checkpoint {path} has SHA-256 {digest}, expected {sha256}."
                )
        payload = cls._checkpoint_payload(checkpoint)
        config = payload.get("config") or {}
        env_config = config.get("env") or {}
        task_specs = env_config.get("tasks", [])
        if not isinstance(task_specs, list):
            raise TypeError(
                "MicroDuckSkills could not read the ordered task list from "
                "config.env.tasks."
            )
        if not task_specs:
            raise ValueError("MicroDuckSkills requires at least one task.")
        task_library = torch.stack(
            [
                getattr(MicroDuckEnv, spec["preset"])(
                    **{key: value for key, value in spec.items() if key != "preset"}
                )
                for spec in task_specs
            ]
        )
        policy = MicroDuckSkillPolicy.from_config(
            payload.get("policy_kwargs") or {},
            num_tasks=len(task_library),
            observation_dim=MicroDuckEnv.OBSERVATION_DIM,
            num_actions=MicroDuckEnv.NUM_JOINTS,
            device=device,
        )
        policy.load_state_dict(payload["model_state_dict"])
        if freeze:
            policy.eval().requires_grad_(False)
        return cls(
            policy=policy,
            task_library=task_library,
            action_scale=float(env_config.get("action_scale", 1.0)),
        )

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str | None = None,
        *,
        filename: str | None = None,
        revision: str | None = None,
        device: torch.device | str = "cpu",
        freeze: bool = True,
        sha256: str | None = None,
        **hub_kwargs: Any,
    ) -> MicroDuckSkills:
        """Download the pinned published skills and rebuild them.

        Args:
            repo_id: Hugging Face repository. Defaults to
                ``"torchrl/microduck-skills"``.
            filename: Checkpoint path in the repository. Defaults to the
                historical ``"walker.ckpt"`` artifact name.
            revision: Immutable repository revision. Defaults to the published
                six-skill policy revision.
            device: Device for the rebuilt policy.
            freeze: Load in evaluation mode and disable gradients.
            sha256: Expected checkpoint digest. This is useful in addition to
                an immutable Hub revision when reproducing published results.
            **hub_kwargs: Extra arguments for
                :func:`huggingface_hub.hf_hub_download`.

        Returns:
            A :class:`MicroDuckSkills` object containing the frozen policy,
            ordered task library and action scale.
        """
        if not _has_huggingface_hub:
            raise ImportError(
                "huggingface_hub is required to load MicroDuckSkills from the "
                "hub. Install it with `pip install huggingface-hub`."
            )
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=repo_id or cls.DEFAULT_REPO_ID,
            filename=filename or cls.DEFAULT_FILENAME,
            revision=revision or cls.DEFAULT_REVISION,
            **hub_kwargs,
        )
        return cls.from_checkpoint(path, device=device, freeze=freeze, sha256=sha256)
