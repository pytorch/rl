from torchrl.envs.libs.gym import GymEnv

try:
    import habitat
    import habitat.utils.gym_definitions  # noqa

    _has_habitat = True
except ImportError:
    _has_habitat = False


class HabitatEnv(GymEnv):
    """A wrapper for habitat envs.

    This class currently serves as placeholder and compatibility security.
    It behaves exactly like the GymEnv wrapper.

    """

    pass
