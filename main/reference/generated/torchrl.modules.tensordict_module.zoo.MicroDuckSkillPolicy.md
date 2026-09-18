# MicroDuckSkillPolicy

*class*torchrl.modules.tensordict_module.zoo.MicroDuckSkillPolicy(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/tensordict_module/zoo/microduck_policy.html#MicroDuckSkillPolicy)

Recurrent policy shared by a library of MicroDuck skills.

The policy combines a task-conditioned observation encoder, a single-layer
GRU and a Gaussian head over normalized joint targets. It is an ordinary
`ProbabilisticActor`: `task_id` selects the skill
embedding, while `recurrent_state` and `is_init` make its memory
explicit in the input TensorDict.

Parameters:

- **hidden_size** - Hidden width of the GRU and policy head.
- **num_tasks** - Number of skills indexed by the task embedding.
- **observation_dim** - Width of the proprioceptive observation.
- **num_actions** - Number of robot joints controlled by the policy.
- **initial_policy_scale** - Initial exploration standard deviation.
- **device** - Device for the policy parameters.
- **action_low** - Lower normalized action bound.
- **action_high** - Upper normalized action bound.

Examples

```
>>> from torchrl.modules.tensordict_module.zoo import MicroDuckSkillPolicy
>>> skill_policy = MicroDuckSkillPolicy(
... hidden_size=32,
... num_tasks=2,
... observation_dim=56,
... num_actions=14,
... )
>>> type(skill_policy).__name__
'MicroDuckSkillPolicy'
```

Pair a trained policy with its ordered task library before deployment:

```
>>> import torch
>>> from torchrl.envs import MicroDuckEnv
>>> from torchrl.modules.tensordict_module.zoo import MicroDuckSkills
>>> task_library = torch.stack([
... MicroDuckEnv.standing_task(),
... MicroDuckEnv.tracking_task(0.2),
... ])
>>> skills = MicroDuckSkills(skill_policy, task_library, action_scale=1.0)
```

See also

[`MicroDuckSkills`](torchrl.modules.tensordict_module.zoo.MicroDuckSkills.html#torchrl.modules.tensordict_module.zoo.MicroDuckSkills) packages the policy with the task metadata
needed for deployment; [`MicroDuckEnv`](torchrl.envs.MicroDuckEnv.html#torchrl.envs.MicroDuckEnv) is the
joint-level environment used to train it; and
[`GRUModule`](torchrl.modules.GRUModule.html#torchrl.modules.GRUModule) provides its recurrent core.

*classmethod*from_config(*policy_kwargs: Mapping[str, Any]*, ***, *num_tasks: int*, *observation_dim: int*, *num_actions: int*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str = 'cpu'*) → MicroDuckSkillPolicy[[source]](../../_modules/torchrl/modules/tensordict_module/zoo/microduck_policy.html#MicroDuckSkillPolicy.from_config)

Build the policy architecture recorded in checkpoint metadata.