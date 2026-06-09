# register_gym_spec_conversion

torchrl.envs.register_gym_spec_conversion(*spec_type*)[[source]](../../_modules/torchrl/envs/libs/gym.html#register_gym_spec_conversion)

Decorator to register a conversion function for a specific spec type.

The method must have the following signature:

```
>>> @register_gym_spec_conversion("spec.name")
... def convert_specname(
... spec,
... dtype=None,
... device=None,
... categorical_action_encoding=None,
... remap_state_to_observation=None,
... batch_size=None,
... ):
```

where gym(nasium).spaces.spec.name is the location of the spec in gym.

If the spec type is accessible, this will also work:

```
>>> @register_gym_spec_conversion(SpecType)
... def convert_specname(
... spec,
... dtype=None,
... device=None,
... categorical_action_encoding=None,
... remap_state_to_observation=None,
... batch_size=None,
... ):
```

..note:: The wrapped function can be simplified, and unused kwargs can be wrapped in **kwargs.