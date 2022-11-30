def test_imports():
    from torchrl.data import (
        PrioritizedReplayBuffer,
        ReplayBuffer,
        TensorSpec,
    )  # noqa: F401
    from torchrl.envs import Transform, TransformedEnv  # noqa: F401
    from torchrl.envs.gym_like import GymLikeEnv  # noqa: F401
    from torchrl.modules import SafeModule  # noqa: F401
    from torchrl.objectives.common import LossModule  # noqa: F401
