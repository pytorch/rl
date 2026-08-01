# CSVLogger

torchrl.record.loggers.csv.CSVLogger(**args*, *use_ray_service=False*, ***kwargs*)[[source]](../../_modules/torchrl/record/loggers/csv.html#CSVLogger)

A minimal-dependency CSV logger.

Parameters:

- **exp_name** (*str*) - The name of the experiment.
- **log_dir** (*str**or**Path**,**optional*) - where the experiment should be saved.
Defaults to `<cur_dir>/csv_logs`.
- **video_format** (*str**,**optional*) - how videos should be saved when calling `add_video()`. Must be one of
`"pt"` (video saved as a video_<tag>_<step>.pt file with torch.save),
`"memmap"` (video saved as a video_<tag>_<step>.memmap file with [`MemoryMappedTensor`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.MemoryMappedTensor.html#tensordict.MemoryMappedTensor)),
`"mp4"` (video saved as a video_<tag>_<step>.mp4 file, requires torchcodec to be installed).
Defaults to `"pt"`.
- **video_fps** (*int**,**optional*) - the video frames-per-seconds if video_format="mp4". Defaults to 30.