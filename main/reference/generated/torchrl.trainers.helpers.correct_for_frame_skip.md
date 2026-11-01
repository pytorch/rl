# correct_for_frame_skip

torchrl.trainers.helpers.correct_for_frame_skip(*cfg: DictConfig*) → DictConfig[[source]](../../_modules/torchrl/trainers/helpers/envs.html#correct_for_frame_skip)

Correct the arguments for the input frame_skip, by dividing all the arguments that reflect a count of frames by the frame_skip.

This is aimed at avoiding unknowingly over-sampling from the environment, i.e. targeting a total number of frames
of 1M but actually collecting frame_skip * 1M frames.

Parameters:

**cfg** (*DictConfig*) - DictConfig containing some frame-counting argument, including:
"max_frames_per_traj", "total_frames", "frames_per_batch", "record_frames", "annealing_frames",
"init_random_frames", "init_env_steps"

Returns:

the input DictConfig, modified in-place.