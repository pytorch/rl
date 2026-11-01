# WorldModelWrapper

*class*torchrl.modules.WorldModelWrapper(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/tensordict_module/world_models.html#WorldModelWrapper)

World model wrapper.

This module wraps together a transition model and a reward model.
The transition model is used to predict an imaginary world state.
The reward model is used to predict the reward of the imagined transition.

Parameters:

- **transition_model** (*TensorDictModule*) - a transition model that generates a new world states.
- **reward_model** (*TensorDictModule*) - a reward model, that reads the world state and returns a reward.

get_reward_operator() → [TensorDictModule](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule)[[source]](../../_modules/torchrl/modules/tensordict_module/world_models.html#WorldModelWrapper.get_reward_operator)

Returns a reward operator that maps a world state to a reward.

get_transition_model_operator() → [TensorDictModule](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule)[[source]](../../_modules/torchrl/modules/tensordict_module/world_models.html#WorldModelWrapper.get_transition_model_operator)

Returns a transition operator that maps either an observation to a world state or a world state to the next world state.