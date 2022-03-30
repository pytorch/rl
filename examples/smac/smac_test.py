from env import SCEnv
from examples.smac.policy import MaskedLogitPolicy
from torchrl.envs import TransformedEnv, ObservationNorm
from torchrl.modules import ProbabilisticTDModule, OneHotCategorical
from torch import nn

if __name__ == "__main__":
    # create an env
    env = SCEnv("8m")

    # reset
    td = env.reset()
    print("tensordict after reset: ")
    print(td)

    # apply a sequence of transforms
    env = TransformedEnv(env, ObservationNorm(0, 1, standard_normal=True))

    # Get policy
    policy = nn.LazyLinear(env.action_spec.shape[-1])
    policy_wrap = MaskedLogitPolicy(policy)
    policy_td_module = ProbabilisticTDModule(
        module=policy_wrap,
        spec=None,
        in_keys=["observation", "available_actions"],
        out_keys=["action"],
        distribution_class=OneHotCategorical,
        save_dist_params=True,
    )

    # Test the policy
    policy_td_module(td)
    print(td)

    # check that an ation can be performed in the env with this
    env.step(td)
    print(td)
