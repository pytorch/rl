# CheckpointOptions

*class*torchrl.checkpoint.CheckpointOptions(*save_args: tuple[Any, ...] | None = None*, *save_kwargs: Mapping[str, Any] | None = None*, *load_args: tuple[Any, ...] | None = None*, *load_kwargs: Mapping[str, Any] | None = None*)[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#CheckpointOptions)

Arguments forwarded to a component's serialization methods.

Operation-level options are merged over registration-time options. Keyword
arguments are shallow-merged and explicitly supplied positional arguments
replace the registration-time tuple.

Parameters:

- **save_args** - Positional arguments forwarded after the checkpoint path for
`dump` adapters, or to `state_dict` for state-dict adapters.
- **save_kwargs** - Keyword arguments forwarded during saving.
- **load_args** - Positional arguments forwarded after the checkpoint path for
`load` adapters, or after the state mapping for state-dict adapters.
- **load_kwargs** - Keyword arguments forwarded during restoration.

Examples

```
>>> from torchrl.checkpoint import CheckpointOptions
>>> options = CheckpointOptions(save_kwargs={"compression": "zstd"})
>>> options.save_kwargs["compression"]
'zstd'
```

merged(*override: CheckpointOptions | None*) → CheckpointOptions[[source]](../../_modules/torchrl/checkpoint/_checkpoint.html#CheckpointOptions.merged)

Return these options with an operation-level override applied.