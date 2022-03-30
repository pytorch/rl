from env import SCEnv
from torchrl.envs import TransformedEnv, ObservationNorm

if __name__ == "__main__":
    # create an env
    env = SCEnv("8m")

    # reset
    td = env.reset()
    print("tensordict after reset: ")
    print(td)

    # apply a sequence of transforms
    env = TransformedEnv(env, ObservationNorm(0, 1, standard_normal=True))

    #
