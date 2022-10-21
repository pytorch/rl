from torchrl.envs.libs.gym import GymEnv

try:
    import habitat.utils.gym_definitions  # noqa

    _has_habitat = True
except ImportError:
    _has_habitat = False


class HabitatEnv(GymEnv):
    pass
