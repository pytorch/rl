import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.modules.distributions import OneHotCategorical
from torchrl.modules.tensordict_module.actors import ProbabilisticActor
from torchrl.objectives.value.advantages import GAE, VTrace

value_net = TensorDictModule(nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"])
actor_net = TensorDictModule(nn.Linear(3, 4), in_keys=["obs"], out_keys=["logits"])
actor_net = ProbabilisticActor(
    module=actor_net,
    in_keys=["logits"],
    out_keys=["action"],
    distribution_class=OneHotCategorical,
    return_log_prob=True,
)
vtrace_module = VTrace(
    gamma=0.98,
    value_network=value_net,
    actor_network=actor_net,
    differentiable=False,
)
gae_module = GAE(
    gamma=0.98,
    lmbda=0.95,
    value_network=value_net,
    differentiable=False,
)

obs, next_obs = torch.randn(2, 1, 10, 3)
reward = torch.randn(1, 10, 1)
done = torch.zeros(1, 10, 1, dtype=torch.bool)
terminated = torch.zeros(1, 10, 1, dtype=torch.bool)
sample_log_prob = torch.randn(1, 10, 1)
tensordict = TensorDict(
    {
        "obs": obs,
        "done": done,
        "terminated": terminated,
        "sample_log_prob": sample_log_prob,
        "next": {
            "obs": next_obs,
            "reward": reward,
            "done": done,
            "terminated": terminated,
        },
    },
    batch_size=[1, 10],
)
advantage, value_target = gae_module(
    obs=obs,
    reward=reward,
    done=done,
    next_obs=next_obs,
    terminated=terminated,
    sample_log_prob=sample_log_prob,
)
advantage, value_target = vtrace_module(
    obs=obs,
    reward=reward,
    done=done,
    next_obs=next_obs,
    terminated=terminated,
    sample_log_prob=sample_log_prob,
)
