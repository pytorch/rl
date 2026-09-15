# resume_config

torchrl.checkpoint.resume_config(*cfg: DictConfig*, *checkpoint_path: str | Path*, ***, *overrides: [Sequence](torchrl.data.Sequence.html#torchrl.data.Sequence)[str] | None = None*) → DictConfig[[source]](../../_modules/torchrl/checkpoint/_hydra.html#resume_config)

Return the configuration of a run resumed from `checkpoint_path`.

The configuration saved with the checkpoint under its `config` component
is the base and `overrides` are applied on top, so `resume=<path>` alone
rebuilds the original run while `resume=<path> collector.total_frames=...`
extends it. Interpolations survive because recipes save the configuration
unresolved. Config-group overrides such as `logger@logger=csv`, deletions
(`~key`) and bare flags cannot be applied to a saved configuration and
are ignored with a warning. When the checkpoint holds no `config`
component, `cfg` is returned unchanged with a warning.

Parameters:

- **cfg** (*DictConfig*) - the configuration composed for the current run.
- **checkpoint_path** (*str**or**Path*) - the checkpoint being resumed.
- **overrides** ([*Sequence*](torchrl.data.Sequence.html#torchrl.data.Sequence)*[**str**]**,**optional*) - `key=value` overrides applied over
the saved configuration. Defaults to the task overrides of the
current Hydra run, or none outside a Hydra application.

Returns:

The configuration to run.

Examples

```
>>> import tempfile
>>> from omegaconf import OmegaConf 
>>> from torchrl.checkpoint import Checkpoint, resume_config
>>> saved = {"budget": 100, "trainer": {"total_frames": "${budget}"}}
>>> with tempfile.TemporaryDirectory() as tmpdir: 
... path = Checkpoint(config=saved).save(f"{tmpdir}/checkpoint")
... cfg = resume_config(
... OmegaConf.create({"budget": 5}), path, overrides=["budget=200"]
... )
>>> cfg.trainer.total_frames 
200
```