import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs import EnvCreator, ParallelEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import ProbabilisticActor
from torchrl.modules.distributions import TanhNormal
from torchrl.modules.models import DTActor
from torchrl.modules.models.decision_transformer import DecisionTransformer


def env_maker(env_name, frame_skip=1, device="cpu", from_pixels=False):
    return GymEnv(
        env_name, "run", device=device, frame_skip=frame_skip, from_pixels=from_pixels
    )


def env_factory(num_workers):
    """Creates an instance of the environment."""

    # 1.2 Create env vector
    vec_env = ParallelEnv(
        create_env_fn=EnvCreator(lambda: env_maker(env_name="Pendulum-v1")),
        num_workers=num_workers,
    )

    return vec_env


# Sanity check
test_env = env_factory(num_workers=1)
action_spec = test_env.action_spec

in_keys = ["observation", "action", "returns_to_go", "timesteps", "padding_mask"]
transformer = DecisionTransformer(
    state_dim=5, action_dim=2, hidden_size=512, max_ep_len=1000, ordering=False
)

actor_head = DTActor(action_dim=2)

actor_net = torch.nn.ModuleList([transformer, actor_head])

dist_class = TanhNormal
dist_kwargs = {
    "min": -1.0,
    "max": 1.0,
    "tanh_loc": False,
}

actor_module = TensorDictModule(
    actor_net,
    in_keys=in_keys,
    out_keys=["loc", "scale", "hidden_state"],
)
actor = ProbabilisticActor(
    spec=action_spec,
    in_keys=["loc", "scale", "hidden_state"],
    out_keys=["action", "log_prob", "hidden_state"],
    module=actor_module,
    distribution_class=dist_class,
    distribution_kwargs=dist_kwargs,
    default_interaction_mode="random",
    cache_dist=True,
    return_log_prob=False,
)

print(transformer)

observation = torch.rand(1, 20, 5)
action = torch.rand(1, 20, 2)
reward_to_go = torch.rand(1, 20, 1)
padding_mask = torch.ones(1, 20, 1)
timesteps = torch.arange(1, 21).unsqueeze(0).unsqueeze(-1)

td = TensorDict(
    {
        "observation": observation,
        "action": action,
        "returns_to_go": reward_to_go,
        "padding_mask": padding_mask,
        "timesteps": timesteps,
    },
    batch_size=1,
)
