# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""An actor with a causal-transformer backbone.

The transformer counterpart of ``recurrent_actor.py``: a
:class:`~torchrl.modules.TransformerModule` turns observations into features
that a Q-value head consumes. Two things differ from the recurrent modules and
are the point of this example:

- Nothing about the transformer's state travels in the tensordict. The
  key/value cache is inference state owned by the module instance, cleared
  where ``is_init`` is set and released with ``reset_cache()``, so there is no
  primer to attach and rollouts stay as small as the observations they carry.
- Training runs over episode-aligned ``[B, T]`` windows under
  :class:`~torchrl.modules.set_recurrent_mode`. The window path recovers the
  episode boundaries from ``is_init`` and produces the same features the
  policy computed one step at a time during collection.
"""
from collections import OrderedDict

import torch
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq

from torchrl.envs import Compose, GymEnv, InitTracker, StepCounter, TransformedEnv
from torchrl.modules import MLP, QValueModule, set_recurrent_mode, TransformerModule

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# InitTracker provides the ``is_init`` flag the transformer reads to start a
# fresh stream at every episode boundary. No other environment-side wiring is
# needed: unlike the recurrent modules there is no TensorDictPrimer to append.
env = TransformedEnv(
    GymEnv("CartPole-v1", device=device),
    Compose(StepCounter(), InitTracker()),
)

# ``max_seq_len`` bounds the episode length: it sizes the positional table and
# the cache. CartPole-v1 truncates at 500 steps.
transformer = TransformerModule(
    input_size=env.observation_spec["observation"].shape[-1],
    hidden_size=64,
    num_layers=2,
    num_heads=4,
    max_seq_len=500,
    in_key="observation",
    out_key="embed",
    device=device,
)

mlp = MLP(out_features=env.action_spec.shape[-1], num_cells=[64], device=device)
mlp[-1].bias.data.fill_(0.0)
head = Mod(mlp, in_keys=["embed"], out_keys=["action_value"])
qval = QValueModule(spec=env.action_spec)

policy = Seq(OrderedDict(transformer=transformer, head=head, qval=qval))

# Collection: the policy is called once per step and attends to its cache.
# ``break_when_any_done=False`` lets the rollout span several episodes; each
# reset raises ``is_init`` and the corresponding stream restarts.
rollout = env.rollout(300, policy, break_when_any_done=False)
episodes = int(rollout["is_init"].sum())
print(f"collected {rollout.numel()} steps over {episodes} episodes")
print(sorted(rollout.keys()))  # no transformer state, only observations and outputs

# Training: the same rollout processed as one window. Every row of a window
# must start with ``is_init=True`` (a rollout from a reset does); inside the
# window a block-diagonal causal mask keeps attention within each episode. The
# features match the ones collected step by step, which is what makes a loss
# computed on windows consistent with the acting policy.
with set_recurrent_mode(True):
    window = policy(rollout.exclude("embed", "action_value").clone())
max_diff = (window["embed"] - rollout["embed"]).abs().max().item()
print(f"max |window - step| features: {max_diff:.2e}")
assert torch.allclose(window["embed"], rollout["embed"], atol=1e-4)

# The window path carries gradients; the cached-step path is inference only.
loss = window["action_value"].square().mean()
loss.backward()
assert transformer.transformer.in_proj.weight.grad is not None

# Lifecycle: after an optimizer step the cache holds keys and values computed
# with stale weights. TorchRL's collectors and inference server call
# ``mark_weight_update()`` for you when they apply new weights; do it yourself
# when updating the parameters by other means, and call ``reset_cache()``
# before reusing the module instance with another environment.
transformer.mark_weight_update()
