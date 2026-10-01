# MicroDuckSkills

*class*torchrl.modules.tensordict_module.zoo.MicroDuckSkills(*policy: TensorDictModuleBase*, *task_library: [MicroDuckTask](torchrl.envs.MicroDuckTask.html#torchrl.envs.MicroDuckTask)*, *action_scale: float*)[[source]](../../_modules/torchrl/modules/tensordict_module/zoo/microduck_policy.html#MicroDuckSkills)

A deployable MicroDuck skill policy and its environment metadata.

`policy` maps a task-conditioned MicroDuck observation to normalized
joint targets. `task_library` preserves the exact meaning and order of
its task embeddings. `action_scale` records how the joint targets were
applied during training. Keeping the three together prevents a high-level
environment from silently pairing a policy with incompatible task ids or
motor scaling.

Parameters:

- **policy** - Trained task-conditioned TensorDict policy.
- **task_library** - Ordered, stacked [`MicroDuckTask`](torchrl.envs.MicroDuckTask.html#torchrl.envs.MicroDuckTask).
- **action_scale** - Environment-side joint-target scale used during training.

Examples

Download the pinned published skills and pass the resulting object to
the high-level environment rather than unpacking policy metadata:

```
>>> from torchrl.modules.tensordict_module.zoo import MicroDuckSkills
>>> skills = MicroDuckSkills.from_pretrained() 
>>> skill_policy = skills.policy 
>>> task_library = skills.task_library
```

Promote compatible joint-level dynamics to a high-level environment
without separating the policy from that metadata:

```
>>> from torchrl.envs import MicroDuckSkillEnv
>>> base_env = make_microduck_game_env( 
... action_scale=skills.action_scale
... )
>>> env = MicroDuckSkillEnv.from_env( 
... base_env, skills, control_steps_per_decision=5
... )
```

See also

[`MicroDuckSkillPolicy`](torchrl.modules.tensordict_module.zoo.MicroDuckSkillPolicy.html#torchrl.modules.tensordict_module.zoo.MicroDuckSkillPolicy) is the neural policy stored here;
[`MicroDuckTask`](torchrl.envs.MicroDuckTask.html#torchrl.envs.MicroDuckTask) describes one row of the ordered
task library; [`MicroDuckEnv`](torchrl.envs.MicroDuckEnv.html#torchrl.envs.MicroDuckEnv) supplies the
joint-level training dynamics; and
[`MicroDuckSkillEnv`](torchrl.envs.MicroDuckSkillEnv.html#torchrl.envs.MicroDuckSkillEnv) deploys the complete artifact.

*classmethod*from_checkpoint(*checkpoint: str | Path | Mapping[str, Any]*, ***, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str = 'cpu'*, *freeze: bool = True*, *sha256: str | None = None*) → MicroDuckSkills[[source]](../../_modules/torchrl/modules/tensordict_module/zoo/microduck_policy.html#MicroDuckSkills.from_checkpoint)

Rebuild the policy and deployment metadata from a checkpoint.

Parameters:

- **checkpoint** - Local checkpoint path or an already loaded payload.
- **device** - Device for the rebuilt policy.
- **freeze** - Load in evaluation mode and disable gradients.
- **sha256** - Expected SHA-256 digest for a path. This cannot be used
with an already loaded payload.

*classmethod*from_pretrained(*repo_id: str | None = None*, ***, *filename: str | None = None*, *revision: str | None = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str = 'cpu'*, *freeze: bool = True*, *sha256: str | None = None*, ***hub_kwargs: Any*) → MicroDuckSkills[[source]](../../_modules/torchrl/modules/tensordict_module/zoo/microduck_policy.html#MicroDuckSkills.from_pretrained)

Download the pinned published skills and rebuild them.

Parameters:

- **repo_id** - Hugging Face repository. Defaults to
`"torchrl/microduck-skills"`.
- **filename** - Checkpoint path in the repository. Defaults to the
historical `"walker.ckpt"` artifact name.
- **revision** - Immutable repository revision. Defaults to the published
six-skill policy revision.
- **device** - Device for the rebuilt policy.
- **freeze** - Load in evaluation mode and disable gradients.
- **sha256** - Expected checkpoint digest. This is useful in addition to
an immutable Hub revision when reproducing published results.
- ****hub_kwargs** - Extra arguments for
`huggingface_hub.hf_hub_download()`.

Returns:

A `MicroDuckSkills` object containing the frozen policy,
ordered task library and action scale.