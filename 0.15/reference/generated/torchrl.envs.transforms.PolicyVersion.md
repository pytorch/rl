# PolicyVersion

*class*torchrl.envs.transforms.PolicyVersion(*version_type: type | ~typing.Literal['uuid'*, *'int'] = <class 'int'>*)[[source]](../../_modules/torchrl/envs/transforms/policy_version.html#PolicyVersion)

A transform that keeps track of the version of the policy.

This transform is used to track policy versions during training, particularly in asynchronous
settings where policy weights are updated periodically. It works with TorchRL collectors to
ensure data collection and training remain in sync.

The version can be either a UUID (string) or an integer counter. When passed to a collector as
its `track_policy_version` argument, the version is automatically incremented each time the
policy weights are updated.

Parameters:

**version_type** - The type of versioning to use. Can be either:
- str or "uuid": Uses UUID4 for versions (good for distributed systems)
- int or "int": Uses incrementing integers (good for debugging)

Examples

```
>>> from torchrl.envs.transforms import PolicyVersion
>>> policy_version = PolicyVersion(version_type="int")
>>> policy_version.version
0
>>> policy_version.increment_version()
>>> policy_version.version
1
```

increment_version() → None[[source]](../../_modules/torchrl/envs/transforms/policy_version.html#PolicyVersion.increment_version)

Increment the version number.

This is called automatically by collectors when policy weights are updated.
Can also be called manually if needed.

transform_observation_spec(*spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*) → [Composite](torchrl.data.Composite.html#torchrl.data.Composite)[[source]](../../_modules/torchrl/envs/transforms/policy_version.html#PolicyVersion.transform_observation_spec)

Update the environment spec to include the version field.

Parameters:

**spec** - The environment spec to update

Returns:

Updated spec including the version field

*property*version*: str | int*

The current version of the policy.