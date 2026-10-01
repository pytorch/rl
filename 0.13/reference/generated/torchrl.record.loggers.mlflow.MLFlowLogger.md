# MLFlowLogger

torchrl.record.loggers.mlflow.MLFlowLogger(**args*, *use_ray_service=False*, ***kwargs*)[[source]](../../_modules/torchrl/record/loggers/mlflow.html#MLFlowLogger)

Wrapper for the mlflow logger.

Parameters:

- **exp_name** (*str*) - The name of the experiment.
- **tracking_uri** (*str*) - A tracking URI to a datastore that supports MLFlow.
Since MLFlow 3.10, filesystem tracking backends (e.g. `./mlruns`)
are no longer supported and a database backend such as
`sqlite:///path/to/mlflow.db` must be used. See the
[MLflow migration guide](https://mlflow.org/docs/latest/self-hosting/migrate-from-file-store).

Keyword Arguments:

- **artifact_location** (*str**,**optional*) - Location used to store run artifacts
(videos, models, ...). When `None` (default), MLFlow uses its
default artifact location. When `tracking_uri` is a filesystem
URI, it is also used as `artifact_location` for backward
compatibility.
- **fps** (*int**,**optional*) - Number of frames per second when recording videos. Defaults to `30`.