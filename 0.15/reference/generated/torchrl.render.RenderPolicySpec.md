# RenderPolicySpec

*class*torchrl.render.RenderPolicySpec(*ckpt_path: Path*, *checkpoint: Any | None*, *checkpoint_hash: str | None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*, *env_specs: Any | None*, *policy_kwargs: dict[str, Any]*, *config: [RenderConfig](torchrl.render.RenderConfig.html#torchrl.render.RenderConfig)*)[[source]](../../_modules/torchrl/render/config.html#RenderPolicySpec)

Context object passed to policy factories.

Parameters:

- **ckpt_path** - Local checkpoint path.
- **checkpoint** - Loaded checkpoint payload, if loading succeeded.
- **checkpoint_hash** - SHA256 checkpoint hash.
- **device** - Device requested for policy inference.
- **env_specs** - Optional environment specs exposed by the environment.
- **policy_kwargs** - Extra keyword arguments supplied by the user.
- **config** - Full render configuration.

Examples

```
>>> from pathlib import Path
>>> from torchrl.render import RenderConfig, RenderPolicySpec
>>> cfg = RenderConfig("policy.pt", "p:make", "e:make", max_steps=2)
>>> spec = RenderPolicySpec(Path("policy.pt"), None, None, cfg.device, None, {}, cfg)
>>> spec.ckpt_path.name
'policy.pt'
```