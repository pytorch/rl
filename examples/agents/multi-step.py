# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Example of a dummy multi-step agent.

A multi-step actor predicts a macro (or an action sequence) and executes it regardless of the observations
coming in the meantime.

The core component of this example is the `MultiStepActorWrapper` class.

`MultiStepActorWrapper` handles the calls to the actor when the macro has run out of actions or
when the environment has been reset (which is indicated by the InitTracker transform).

"""

import torch.nn
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq
from torchrl.envs import (
    CatFrames,
    Compose,
    GymEnv,
    InitTracker,
    SerialEnv,
    TransformedEnv,
)
from torchrl.modules.tensordict_module.actors import MultiStepActorWrapper

time_steps = 6
n_obs = 4
n_action = 2
batch = 5


# Transforms a CatFrames in a stack of frames
def reshape_cat(data: torch.Tensor):
    return data.unflatten(-1, (time_steps, n_obs))


# an actor that reads `time_steps` frames and outputs one action per frame
# (actions are conditioned on the observation of `time_steps` in the past)
actor_base = Seq(
    Mod(reshape_cat, in_keys=["obs_cat"], out_keys=["obs_cat_reshape"]),
    Mod(
        torch.nn.Linear(n_obs, n_action),
        in_keys=["obs_cat_reshape"],
        out_keys=["action"],
    ),
)
# Wrap the actor to dispatch the actions
actor = MultiStepActorWrapper(actor_base, n_steps=time_steps)

env = TransformedEnv(
    SerialEnv(batch, lambda: GymEnv("CartPole-v1")),
    Compose(
        InitTracker(),
        CatFrames(N=time_steps, in_keys=["observation"], out_keys=["obs_cat"], dim=-1),
    ),
)

print(env.rollout(100, policy=actor, break_when_any_done=False))
