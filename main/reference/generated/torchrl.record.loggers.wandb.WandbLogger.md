# WandbLogger

torchrl.record.loggers.wandb.WandbLogger(**args*, *use_ray_service=False*, ***kwargs*)[[source]](../../_modules/torchrl/record/loggers/wandb.html#WandbLogger)

Wrapper for the wandb logger.

The keyword arguments are mainly based on the `wandb.init()` kwargs.
See the doc [here](https://docs.wandb.ai/ref/python/init).

Parameters:

- **exp_name** (*str*) - The name of the experiment.
- **offline** (*bool**,**optional*) - if `True`, the logs will be stored locally
only. Defaults to `False`.
- **save_dir** (*path**,**optional*) - the directory where to save data. Exclusive with
`log_dir`.
- **log_dir** (*path**,**optional*) - the directory where to save data. Exclusive with
`save_dir`.
- **id** (*str**,**optional*) - A unique ID for this run, used for resuming.
It must be unique in the project, and if you delete a run you can't reuse the ID.
- **project** (*str**,**optional*) - The name of the project where you're sending
the new run. If the project is not specified, the run is put in
an `"Uncategorized"` project.
- **log_env_packages** (*bool**,**optional*) - if `True`, logs the Python runtime,
installed package versions, and editable source locations under
`wandb.config["env"]`. Defaults to `True`.

Keyword Arguments:

- **fps** (*int**,**optional*) - Number of frames per second when recording videos. Defaults to `30`.
- ****kwargs** - Extra keyword arguments for `wandb.init`. See relevant page for
more info.