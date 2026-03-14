import torch
from torchrl.envs import GymEnv, RewardSum, StepCounter, TransformedEnv


def make_env(
    env_name: str, device: str | torch.device, from_pixels: bool = False
) -> TransformedEnv:
    """Creates the transformed environment for PILCO experiments."""
    env = TransformedEnv(
        GymEnv(env_name, pixels_only=False, from_pixels=from_pixels, device=device)
    )
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    return env
