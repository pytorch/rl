def test_imports():
    from torchrl.data import (
        ReplayBuffer,
        PrioritizedReplayBuffer,
        TensorDict,
        TensorSpec,
    )
    from torchrl.envs import TransformedEnv, Transform
    from torchrl.envs.gym_like import GymLikeEnv
    from torchrl.modules import TensorDictModule
    from torchrl.objectives.costs.common import _LossModule
