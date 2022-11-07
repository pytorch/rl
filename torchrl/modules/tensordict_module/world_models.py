# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from torchrl.modules.tensordict_module import TensorDictModule, TensorDictSequential


class WorldModelWrapper(TensorDictSequential):
    """World model wrapper.

    This module wraps together a transition model and a reward model.
    The transition model is used to predict an imaginary world state.
    The reward model is used to predict the reward of the imagined transition.

    Args:
        transition_model (TensorDictModule): a transition model that generates a new world states.
        reward_model (TensorDictModule): a reward model, that reads the world state and returns a reward.

    """

    def __init__(
        self,
        transition_model: TensorDictModule,
        reward_model: TensorDictModule,
    ):
        super().__init__(
            transition_model,
            reward_model,
        )

    def get_transition_model_operator(self) -> TensorDictSequential:
        """Returns a transition operator that maps either an observation to a world state or a world state to the next world state."""
        return self.module[0]

    def get_reward_operator(self) -> TensorDictSequential:
        """Returns a reward operator that maps a world state to a reward."""
        return self.module[1]
