# LeRobotPolicyWrapper

*class*torchrl.modules.vla.LeRobotPolicyWrapper(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/vla/wrappers.html#LeRobotPolicyWrapper)

Expose an external (LeRobot-style) policy as a TorchRL VLA policy.

This adapts a pretrained action-chunk policy - such as a LeRobot
`PreTrainedPolicy` (ACT, Diffusion Policy, SmolVLA, pi0, ...) - to the
canonical VLA key contract (see [`VLAWrapperBase`](torchrl.modules.vla.VLAWrapperBase.html#torchrl.modules.vla.VLAWrapperBase)),
so an off-the-shelf checkpoint can be evaluated or fine-tuned inside the
TorchRL stack. On `forward()` it builds a LeRobot-style batch dict from
the canonical observation keys (`observation.state`,
`observation.images.<camera>`, `task`), calls the wrapped policy, and
writes the returned continuous action chunk under
`("vla_action", "chunk")`.

The wrapped object can be any callable / module that maps a LeRobot batch
dict to an action chunk of shape `[B, chunk_size, action_dim]`; by default
the wrapper tries the policy's `predict_action_chunk`, `select_action`
then `forward` methods (override with `predict_fn` for a specific API).

Parameters:

**policy** - the wrapped policy (a callable / `nn.Module` that returns an
action chunk given a LeRobot batch dict).

Keyword Arguments:

- **action_dim** (*int*) - the dimensionality of a single action.
- **chunk_size** (*int*) - the action-chunk horizon.
- **predict_fn** (*Callable**,**optional*) - a `(policy, batch) -> chunk` callable
overriding the default policy-call dispatch.
- **camera_name** (*str*) - the LeRobot camera name to use for the image key
(`observation.images.<camera_name>`). Defaults to `"image"`.
- **use_state** (*bool*) - whether to forward the proprioceptive state.
Defaults to `True`.
- **instruction_key** (*image_key**,**state_key**,*) - canonical input keys.

Warning

Loading a real LeRobot checkpoint (`from_pretrained()`) requires the
optional `lerobot` package and targets its documented API; that path
is **best-effort / not exercised in CI**. The key mapping and base-class
integration are tested with a stand-in policy.

Note

Only the continuous chunk head is supported - external policies emit
continuous chunks, not TorchRL action-token logits.

Examples

```
>>> import torch
>>> from tensordict import NonTensorStack, TensorDict
>>> from torchrl.modules.vla import LeRobotPolicyWrapper
>>> class DummyPolicy:
... def predict_action_chunk(self, batch):
... b = batch["observation.state"].shape[0]
... return torch.zeros(b, 4, 7)
>>> policy = LeRobotPolicyWrapper(DummyPolicy(), action_dim=7, chunk_size=4)
>>> td = TensorDict(
... {
... "observation": {
... "image": torch.zeros(2, 3, 16, 16),
... "state": torch.zeros(2, 5),
... },
... "language_instruction": NonTensorStack("pick", "place"),
... },
... batch_size=[2],
... )
>>> policy(td)["vla_action", "chunk"].shape
torch.Size([2, 4, 7])
```

*classmethod*from_pretrained(*repo_id: str*, ***, *action_dim: int*, *chunk_size: int*, ***kwargs*) → LeRobotPolicyWrapper[[source]](../../_modules/torchrl/modules/vla/wrappers.html#LeRobotPolicyWrapper.from_pretrained)

Load a pretrained LeRobot policy and wrap it (requires `lerobot`).