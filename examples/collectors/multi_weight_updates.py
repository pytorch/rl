"""Example of updating weights of several models at once in a multiprocessed data collector.

This example demonstrates:
1. Using different weight sync schemes for different models
2. Updating the policy (via pipes with MultiProcessWeightSyncScheme)
3. Updating Ray-based transforms in env and replay buffer (via RayModuleTransformScheme)
4. Atomic multi-model weight updates using weights_dict

Note:
- Ray actors are shared across all workers, so RayModuleTransformScheme uses a
  single transport rather than per-worker pipes.
- When using transform_factory with a replay buffer, delayed_init automatically defaults
  to True for proper serialization in multiprocessing contexts.
- extend_buffer defaults to True in all collectors, extending the buffer with entire
  rollouts rather than individual frames for better compatibility with postprocessing.
"""

from functools import partial

import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from torchrl.collectors import MultiSyncCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms.module import ModuleTransform
from torchrl.weight_update import MultiProcessWeightSyncScheme


def make_module():
    # A module that transforms the observations
    return TensorDictModule(
        nn.Linear(3, 3), in_keys=["observation"], out_keys=["observation"]
    )


def policy_factory():
    # A module that produces the actions
    return TensorDictModule(
        nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"]
    )


def make_env():
    env_module = ModuleTransform(
        module_factory=make_module, inverse=False, no_grad=True
    )
    return GymEnv("Pendulum-v1").append_transform(env_module)


def main():
    rb = ReplayBuffer(
        storage=LazyTensorStorage(10000, shared_init=True),
        transform_factory=partial(
            ModuleTransform,
            module_factory=make_module,
            inverse=True,
            no_grad=True,
        ),
        # delayed_init automatically defaults to True when transform_factory is provided
    )

    policy = policy_factory()

    weight_sync_schemes = {
        "policy": MultiProcessWeightSyncScheme(strategy="state_dict"),
        "replay_buffer.transform[0].module": MultiProcessWeightSyncScheme(
            strategy="tensordict"
        ),
        "env.transform[0].module": MultiProcessWeightSyncScheme(strategy="tensordict"),
    }

    collector = MultiSyncCollector(
        create_env_fn=[make_env, make_env],
        policy_factory=policy_factory,
        total_frames=2000,
        max_frames_per_traj=50,
        frames_per_batch=200,
        init_random_frames=-1,
        device="cpu",
        storing_device="cpu",
        weight_sync_schemes=weight_sync_schemes,
        replay_buffer=rb,
        local_init_rb=True,
        # extend_buffer=True is the default for MultiSyncCollector
    )

    policy_weights = TensorDict.from_module(policy).data
    env_module_weights = TensorDict.from_module(make_module()).data
    rb_module_weights = TensorDict.from_module(make_module()).data

    for i, _data in enumerate(collector):
        env_module_weights.zero_()
        rb_module_weights.zero_()
        policy_weights.zero_()

        collector.update_policy_weights_(
            weights_dict={
                "policy": policy_weights,
                "env.transform[0].module": env_module_weights,
                "replay_buffer.transform[0].module": rb_module_weights,
            }
        )

        assert len(rb) == i * 200 + 200

        if i >= 10:
            break

    collector.shutdown()


if __name__ == "__main__":
    main()
