# set_video_decoder_cache_size

*class*torchrl.data.set_video_decoder_cache_size(*maxsize: int*)[[source]](../../_modules/torchrl/data/video.html#set_video_decoder_cache_size)

Sets the maximum number of open torchcodec decoders cached per process.

The cache is keyed by `(source, stream, device)`; least-recently-used
decoders are evicted (and closed) once the limit is exceeded.

Parameters:

**maxsize** (*int*) - the maximum number of decoders to keep open per process.