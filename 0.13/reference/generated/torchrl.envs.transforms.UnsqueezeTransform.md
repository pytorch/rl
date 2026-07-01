# UnsqueezeTransform

*class*torchrl.envs.transforms.UnsqueezeTransform(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/transforms/_observation.html#UnsqueezeTransform)

Inserts a dimension of size one at the specified position.

Parameters:

**dim** (*int*) - dimension to unsqueeze. Must be negative (or allow_positive_dim
must be turned on).

Keyword Arguments:

- **allow_positive_dim** (*bool**,**optional*) - if `True`, positive dimensions are accepted.
UnsqueezeTransform` will map these to the n^th feature dimension
(ie n^th dimension after batch size of parent env) of the input tensor,
independently of the tensordict batch size (ie positive dims may be
dangerous in contexts where tensordict of different batch dimension
are passed).
Defaults to False, ie. non-negative dimensions are not permitted.
- **in_keys** ([*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**NestedKeys*) - input entries (read).
- **out_keys** ([*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**NestedKeys*) - input entries (write). Defaults to `in_keys` if
not provided.
- **in_keys_inv** ([*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**NestedKeys*) - input entries (read) during `inv` calls.
- **out_keys_inv** ([*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**NestedKeys*) - input entries (write) during `inv` calls.
Defaults to `in_keys_in` if not provided.

transform_action_spec(*action_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_observation.html#UnsqueezeTransform.transform_action_spec)

Transforms the action spec such that the resulting spec matches transform mapping.

Parameters:

**action_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_observation_spec(*observation_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_observation.html#UnsqueezeTransform.transform_observation_spec)

Transforms the observation spec such that the resulting spec matches transform mapping.

Parameters:

**observation_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_reward_spec(*reward_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_observation.html#UnsqueezeTransform.transform_reward_spec)

Transforms the reward spec such that the resulting spec matches transform mapping.

Parameters:

**reward_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_state_spec(*state_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_observation.html#UnsqueezeTransform.transform_state_spec)

Transforms the state spec such that the resulting spec matches transform mapping.

Parameters:

**state_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform