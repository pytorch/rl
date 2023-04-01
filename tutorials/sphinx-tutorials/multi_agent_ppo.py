import time
from collections import defaultdict

import numpy as np
import torch
import wandb
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss, DDPGLoss, ValueEstimators
from torchrl.record.loggers.wandb import WandbLogger


def rendering_callback(env, td):
    env.frames.append(env_test.render(mode="rgb_array", agent_index_focus=None))


class AgentNet(nn.Module):
    def __init__(self, n_outputs, n_agents, device, share_params):
        super().__init__()
        self.n_agents = n_agents
        self.n_outputs = n_outputs
        self.share_params = share_params

        self.agent_networks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LazyLinear(256, device=device),
                    nn.Tanh(),
                    nn.LazyLinear(256, device=device),
                    nn.Tanh(),
                    nn.LazyLinear(n_outputs, device=device),
                )
                for _ in range(self.n_agents if self.share_params else 1)
            ]
        )
        # self.share_init_hetero_networks()

    def forward(self, *inputs):
        if len(inputs) > 1:
            tensor = torch.cat([*inputs], -1)
        else:
            tensor = inputs[0]

        if self.share_params:
            logits = torch.stack(
                [net(tensor[..., i, :]) for i, net in enumerate(self.agent_networks)],
                dim=-2,
            )
        else:
            logits = self.agent_networks[0](tensor)

        assert not logits.isnan().any()

        return logits

    def share_init_hetero_networks(self):
        for child in self.children():
            assert isinstance(child, nn.ModuleList)
            for agent_index, agent_model in enumerate(child.children()):
                if agent_index == 0:
                    state_dict = agent_model.state_dict()
                else:
                    agent_model.load_state_dict(state_dict)


if __name__ == "__main__":
    device = "cpu" if not torch.has_cuda else "cuda:0"
    vmas_device = "cpu"

    test = False

    lr = 5e-5
    max_grad_norm = 40.0

    vmas_envs = 640 if not test else 1
    max_steps = 200
    frames_per_batch = vmas_envs * max_steps
    # For a complete training, bring the number of frames up to 1M
    n_iters = 500
    total_frames = frames_per_batch * n_iters

    sub_batch_size = (
        4096 if not test else 10
    )  # cardinality of the sub-samples gathered from the current data in the inner loop
    num_epochs = 45 if not test else 2  # optimisation steps per batch of data collected
    clip_epsilon = (
        0.2  # clip value for PPO loss: see the equation in the intro for more context.
    )
    gamma = 0.9
    lmbda = 0.9
    entropy_eps = 0

    het_actor = False
    het_critic = False

    scenario_args = {"n_agents": 4, "same_goal": 1, "collision_reward": -0.5}

    env = VmasEnv(
        scenario_name="transport",
        num_envs=vmas_envs,
        continuous_actions=True,
        max_steps=max_steps,
        device=vmas_device,
        # Scenario kwargs
        **scenario_args,
    )
    env_test = VmasEnv(
        scenario_name="transport",
        num_envs=1,
        continuous_actions=True,
        max_steps=max_steps,
        device=vmas_device,
        # Scenario kwargs
        **scenario_args,
    )

    # print("observation_spec:", env.observation_spec)
    # print("reward_spec:", env.reward_spec)
    # print("input_spec:", env.input_spec)
    # print("action_spec (as defined by input_spec):", env.action_spec)
    #
    # rollout = env.rollout(3)
    # print("rollout of three steps:", rollout)
    # print("Shape of the rollout TensorDict:", rollout.batch_size)

    actor_net = nn.Sequential(
        AgentNet(
            2 * env.action_spec.shape[-1], env.n_agents, device, share_params=het_actor
        ),
        NormalParamExtractor(),
    )

    policy_module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
    )

    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": env.action_spec.space.minimum,
            "max": env.action_spec.space.maximum,
        },
        return_log_prob=True,
        # we'll need the log-prob for the numerator of the importance weights
    )

    module = AgentNet(1, env.n_agents, device, share_params=het_critic)
    # value_module = ValueOperator(
    #     module=module,
    #     in_keys=["observation", "action"],
    # )
    value_module = ValueOperator(
        module=module,
        in_keys=["observation"],
    )

    value_module(policy_module(env.reset().to(device)))

    # print("Running policy:", policy_module(env.reset()))
    # print("Running value:", value_module(env.reset()))

    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
        device=device,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch),
        sampler=SamplerWithoutReplacement(),
        batch_size=sub_batch_size,
        collate_fn=lambda x: x,
    )

    loss_module = ClipPPOLoss(
        actor=policy_module,
        critic=value_module,
        advantage_key="advantage",
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        # these keys match by default but we set this for completeness
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
        normalize_advantage=False,
    )
    # loss_module = DDPGLoss(actor_network=policy_module, value_network=value_module)

    # loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)
    loss_module.make_value_estimator(ValueEstimators.GAE, gamma=gamma, lmbda=lmbda)

    optim = torch.optim.Adam(loss_module.parameters(), lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optim, total_frames // frames_per_batch, 0.0
    # )

    logs = defaultdict(list)
    logger = WandbLogger(exp_name="mult_ppo_tutorial", project="torchrl")
    sampling_start = time.time()

    # for i in range(n_iters):
    for i, tensordict_data in enumerate(collector):
        print(f"Iteration {i}")
        # print(tensordict_data)
        print(f"Sampling took {time.time() - sampling_start}")
        tensordict_data = tensordict_data.to(device)

        # Permute
        del tensordict_data["collector"]
        tensordict_data.batch_size = [*tensordict_data.batch_size, env.n_agents]
        tensordict_data = tensordict_data.permute(0, 2, 1)
        with torch.no_grad():
            loss_module.value_estimator(
                tensordict_data, params=loss_module.critic_params.detach()
            )
        tensordict_data = tensordict_data.permute(0, 2, 1)
        tensordict_data.batch_size = tensordict_data.batch_size[:-1]

        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)
        training_start = time.time()
        for _ in range(num_epochs):
            # print(f"Inner epoch {j}")

            for _ in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata.to(device))

                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    # loss_vals["loss_actor"]
                    # + loss_vals["loss_value"]
                    # + loss_vals["loss_entropy"]
                )

                # Optimization: backward, grad clipping and optim step
                loss_value.backward()
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

        logger.log_scalar(
            "reward", tensordict_data["next", "reward"].mean().item(), step=i
        )
        print(f"Training took: {time.time() - training_start}")

        if i % 1 == 0:
            with torch.no_grad():
                env_test.frames = []
                env_test.rollout(
                    max_steps=max_steps,
                    policy=policy_module,
                    callback=rendering_callback,
                    auto_cast_to_device=True,
                )
                vid = np.transpose(env_test.frames, (0, 3, 1, 2))
                logger.experiment.log(
                    {"video": wandb.Video(vid, fps=1 / env_test.world.dt, format="mp4")}
                ),
        sampling_start = time.time()
