# Task-specific module zoo

The module zoo contains task-specific policies composed from TorchRL's general
TensorDict modules. These classes are intentionally not exported from the
top-level `torchrl.modules` namespace.

| [`MicroDuckSkillPolicy`](generated/torchrl.modules.tensordict_module.zoo.MicroDuckSkillPolicy.html#torchrl.modules.tensordict_module.zoo.MicroDuckSkillPolicy)(*args, **kwargs) | Recurrent policy shared by a library of MicroDuck skills. |
| --- | --- |
| [`MicroDuckSkills`](generated/torchrl.modules.tensordict_module.zoo.MicroDuckSkills.html#torchrl.modules.tensordict_module.zoo.MicroDuckSkills)(policy, task_library, ...) | A deployable MicroDuck skill policy and its environment metadata. |