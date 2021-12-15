from torchrl.data.transforms import NoopResetEnv, ToTensorImage, Resize, GrayScale, CatFrames, ObservationNorm, \
    TransformedEnv, Compose
from torchrl.envs import ParallelEnv, GymEnv
from torchrl.record.recorder import VideoRecorder, TensorDictRecorder


def _cat_transform():
    transforms = [
        CatFrames(keys=["next_observation_pixels"]),
        ObservationNorm(loc=-1.0, scale=2.0, keys=["next_observation_pixels"], observation_spec_key="pixels"),
    ]
    return transforms


def make_env(env_name, catframes=True, video_tag="", writer=None, random_noop=True, **kwargs):
    env = GymEnv(env_name, **kwargs)
    transforms = [
        NoopResetEnv(env, random=random_noop),
        ToTensorImage(keys=["next_observation_pixels"]),
        Resize(84, 84, keys=["next_observation_pixels"]),
        GrayScale(keys=["next_observation_pixels"]),
    ]
    if catframes:
        transforms += _cat_transform()
    if len(video_tag):
        transforms = [
            VideoRecorder(
                writer=writer,
                tag=f"{video_tag}_{env_name}_video",
            ),
            TensorDictRecorder(f"{video_tag}_{env_name}"),
            *transforms,
        ]
    env = TransformedEnv(env, Compose(*transforms), )
    return env


def make_parallel_env(env_names, n_processes, **kwargs):
    env_fns = [lambda **kwargs: make_env(env_name, catframes=False, **kwargs) for env_name in env_names]
    p_env = ParallelEnv(n_processes, env_fns, kwargs)
    p_env = TransformedEnv(p_env, Compose(*_cat_transform()), )
    return p_env
