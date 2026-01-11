# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
This code exemplifies how a composite actor can be built.

The actor has two components: a categorical and a normal distributions.

We use a ProbabilisticActor and explicitly pass it the key-map that we want through a 'name_map'
argument.

"""

import torch
from tensordict import TensorDict
from tensordict.nn import CompositeDistribution, TensorDictModule
from torch import distributions as d, nn

from torchrl.modules import ProbabilisticActor


class Module(nn.Module):
    def forward(self, x):
        return x[..., :3], x[..., 3:6], x[..., 6:]


module = TensorDictModule(
    Module(),
    in_keys=["x"],
    out_keys=[
        ("params", "normal", "loc"),
        ("params", "normal", "scale"),
        ("params", "categ", "logits"),
    ],
)
actor = ProbabilisticActor(
    module,
    in_keys=["params"],
    distribution_class=CompositeDistribution,
    distribution_kwargs={
        "distribution_map": {"normal": d.Normal, "categ": d.Categorical},
        "name_map": {"normal": ("action", "normal"), "categ": ("action", "categ")},
    },
)
print(actor.out_keys)

data = TensorDict({"x": torch.rand(10)}, [])
module(data)
print(actor(data))


# TODO:
#  1. Use ("action", "action0") + ("action", "action1") vs ("agent0", "action") + ("agent1", "action")
#  2. Must multi-head require an action_key to be a list of keys (I guess so)
#  3. Using maps in the Actor
