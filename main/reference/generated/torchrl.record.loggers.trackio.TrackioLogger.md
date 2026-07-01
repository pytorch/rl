# TrackioLogger

torchrl.record.loggers.trackio.TrackioLogger(**args*, *use_ray_service=False*, ***kwargs*)[[source]](../../_modules/torchrl/record/loggers/trackio.html#TrackioLogger)

Wrapper for the trackio logger.

Parameters:

- **exp_name** (*str*) - The name of the experiment.
- **project** (*str*) - The name of the project.

Keyword Arguments:

- **fps** (*int**,**optional*) - Number of frames per second when recording videos. Defaults to `30`.
- ****kwargs** - Extra keyword arguments for `trackio.init`.