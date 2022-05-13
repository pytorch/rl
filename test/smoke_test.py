def test_imports():
    from torchrl.data import (
        ReplayBuffer,
        PrioritizedReplayBuffer,
        TensorDict,
        TensorSpec,
    )
    from torchrl.envs import GymLikeEnv
    from torchrl.envs import TransformedEnv, Transform
    from torchrl.modules import TDModule
    from torchrl.objectives.costs.common import _LossModule
