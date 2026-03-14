from collections.abc import Sequence

import torch

from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule

from torchrl.envs import (
    EnvBase,
    GymEnv,
    ModelBasedEnvBase,
    RewardSum,
    StepCounter,
    TransformedEnv,
)


def make_env(
    env_name: str, device: str | torch.device, from_pixels: bool = False
) -> TransformedEnv:
    """Creates the transformed environment."""
    env = TransformedEnv(
        GymEnv(env_name, pixels_only=False, from_pixels=from_pixels, device=device)
    )
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    return env


class ImaginedEnv(ModelBasedEnvBase):
    def __init__(
        self,
        world_model_module: TensorDictModule,
        base_env: EnvBase,
        batch_size: int | torch.Size | Sequence[int] | None = None,
        **kwargs
    ) -> None:
        if batch_size is not None:
            self.batch_size = (
                torch.Size(batch_size)
                if not isinstance(batch_size, torch.Size)
                else batch_size
            )
        elif len(base_env.batch_size) == 0:
            self.batch_size = torch.Size([1])
        else:
            self.batch_size = base_env.batch_size

        super().__init__(
            world_model_module,
            device=base_env.device,
            batch_size=self.batch_size,
            **kwargs
        )

        self.observation_spec = base_env.observation_spec.expand(
            self.batch_size
        ).clone()
        self.action_spec = base_env.action_spec.expand(self.batch_size).clone()
        self.reward_spec = base_env.reward_spec.expand(self.batch_size).clone()
        self.done_spec = base_env.done_spec.expand(self.batch_size).clone()

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = self.world_model(tensordict)

        reward = torch.zeros(*tensordict.shape, 1, device=self.device)
        done = torch.zeros(*tensordict.shape, 1, dtype=torch.bool, device=self.device)
        out = TensorDict(
            {
                "observation": tensordict.get("next_observation"),
                "reward": reward,
                "done": done,
                "terminated": done.clone(),
            },
            tensordict.shape,
        )
        return out

    def _reset(
        self, tensordict: TensorDictBase | None = None, **kwargs
    ) -> TensorDictBase:
        if tensordict is None:
            tensordict = TensorDict({}, batch_size=self.batch_size, device=self.device)

        if (
            tensordict.get(("observation", "var"), None) is not None
            and tensordict.get(("observation", "mean"), None) is not None
        ):
            return tensordict.copy()

        obs = tensordict.get("observation", None)
        if obs is None:
            obs = self.observation_spec.rand(shape=self.batch_size).get("observation")
        if obs.ndim == 1:
            obs = obs.expand(self.batch_size, -1)

        obs = obs.to(self.device)
        B, D = obs.shape

        out = TensorDict(
            {
                ("observation", "mean"): obs,
                ("observation", "var"): torch.zeros(
                    B, D, D, dtype=obs.dtype, device=self.device
                ),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

        out.set("done", torch.zeros(B, 1, dtype=torch.bool, device=self.device))
        out.set("terminated", torch.zeros(B, 1, dtype=torch.bool, device=self.device))

        return out
