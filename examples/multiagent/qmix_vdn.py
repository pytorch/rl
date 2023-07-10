import time

import torch

import wandb
from models.mlp import MultiAgentMLP
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import EGreedyWrapper, QValueModule, SafeSequential
from torchrl.modules.models.multiagent import QMixer, VDNMixer
from torchrl.objectives import SoftUpdate, ValueEstimators
from torchrl.objectives.multiagent.qmixer import QMixerLoss
from torchrl.record.loggers import generate_exp_name
from torchrl.record.loggers.wandb import WandbLogger
from utils.logging import log_evaluation, log_training


def rendering_callback(env, td):
    env.frames.append(env.render(mode="rgb_array", agent_index_focus=None))


def train(seed):
    # Device
    training_device = "cpu" if not torch.has_cuda else "cuda:0"
    vmas_device = training_device

    # Seeding
    seed = seed
    torch.manual_seed(seed)

    # Log
    log = False

    # Sampling
    frames_per_batch = 60_000  # Frames sampled each sampling iteration
    max_steps = 100
    vmas_envs = frames_per_batch // max_steps
    n_iters = 500  # Number of sampling/training iterations
    total_frames = frames_per_batch * n_iters
    memory_size = frames_per_batch

    scenario_name = "balance"
    env_config = {
        "n_agents": 3,
    }

    config = {
        # QMIX
        "mixer_type": "qmix",
        # RL
        "gamma": 0.9,
        "seed": seed,
        # Sampling,
        "frames_per_batch": frames_per_batch,
        "max_steps": max_steps,
        "vmas_envs": vmas_envs,
        "n_iters": n_iters,
        "total_frames": total_frames,
        "memory_size": memory_size,
        "vmas_device": vmas_device,
        # Training
        "num_epochs": 45,  # optimization steps per batch of data collected
        "minibatch_size": 4096,  # size of minibatches used in each epoch
        "lr": 5e-5,
        "max_grad_norm": 40.0,
        "training_device": training_device,
        # Target
        "tau": 0.005,
        # Evaluation
        "evaluation_interval": 20,
        "evaluation_episodes": 200,
    }

    model_config = {
        "shared_parameters": True,
    }

    # Create env and env_test
    env = VmasEnv(
        scenario=scenario_name,
        num_envs=vmas_envs,
        continuous_actions=False,
        max_steps=max_steps,
        device=vmas_device,
        seed=seed,
        # Scenario kwargs
        **env_config,
    )

    env_test = VmasEnv(
        scenario=scenario_name,
        num_envs=config["evaluation_episodes"],
        continuous_actions=False,
        max_steps=max_steps,
        device=vmas_device,
        seed=seed,
        # Scenario kwargs
        **env_config,
    )
    env_config.update({"n_agents": env.n_agents, "scenario_name": scenario_name})

    # Policy
    net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=env.action_spec.space.n,
        n_agents=env.n_agents,
        centralised=False,
        share_params=model_config["shared_parameters"],
        device=training_device,
        depth=2,
        num_cells=256,
        activation_class=nn.Tanh,
    )
    module = TensorDictModule(
        net, in_keys=[("agents", "observation")], out_keys=[("agents", "action_value")]
    )
    value_module = QValueModule(
        action_value_key=("agents", "action_value"),
        out_keys=[
            env.action_key,
            ("agents", "action_value"),
            ("agents", "chosen_action_value"),
        ],
        spec=env.unbatched_action_spec,
        action_space=None,
    )
    qnet = SafeSequential(module, value_module)

    qnet_explore = EGreedyWrapper(
        qnet,
        eps_init=0.3,
        eps_end=0,
        annealing_num_steps=int(total_frames * (1 / 2)),
        action_key=env.action_key,
        spec=env.action_spec,
    )

    if config["mixer_type"] == "qmix":
        mixer = TensorDictModule(
            module=QMixer(
                state_shape=env.unbatched_observation_spec[
                    "agents", "observation"
                ].shape,
                mixing_embed_dim=32,
                n_agents=env.n_agents,
                device=training_device,
            ),
            in_keys=[("agents", "chosen_action_value"), ("agents", "observation")],
            out_keys=["chosen_action_value"],
        )
    elif config["mixer_type"] == "vdn":
        mixer = TensorDictModule(
            module=VDNMixer(
                n_agents=env.n_agents,
                device=training_device,
            ),
            in_keys=[("agents", "chosen_action_value")],
            out_keys=["chosen_action_value"],
        )
    else:
        raise ValueError("Mixer type not in the example")

    with set_exploration_type(ExplorationType.RANDOM):
        qnet_explore(env.reset().to(training_device))

    collector = SyncDataCollector(
        env,
        qnet_explore,
        device=vmas_device,
        storing_device=training_device,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(memory_size, device=training_device),
        sampler=SamplerWithoutReplacement(),
        batch_size=config["minibatch_size"],
    )

    loss_module = QMixerLoss(qnet, mixer, delay_value=True)
    loss_module.set_keys(
        action_value=("agents", "action_value"),
        local_value=("agents", "chosen_action_value"),
        global_value="chosen_action_value",
        action=env.action_key,
    )
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=config["gamma"])
    target_net_updater = SoftUpdate(loss_module, eps=1 - config["tau"])

    optim = torch.optim.Adam(loss_module.parameters(), config["lr"])

    # Logging
    if log:
        config.update({"model": model_config, "env": env_config})
        model_name = ("Het" if not model_config["shared_parameters"] else "") + config[
            "mixer_type"
        ].upper()
        logger = WandbLogger(
            exp_name=generate_exp_name(env_config["scenario_name"], model_name),
            project=f"torchrl_{env_config['scenario_name']}",
            group=model_name,
            save_code=True,
            config=config,
        )
        wandb.run.log_code(".")

    total_time = 0
    total_frames = 0
    sampling_start = time.time()
    for i, tensordict_data in enumerate(collector):
        print(f"\nIteration {i}")

        sampling_time = time.time() - sampling_start
        print(f"Sampling took {sampling_time}")

        # Remove agent dimension from reward
        tensordict_data["next", "reward"] = tensordict_data[
            "next", env.reward_key
        ].mean(-2)
        del tensordict_data["next", env.reward_key]

        current_frames = tensordict_data.numel()
        total_frames += current_frames
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        training_tds = []
        training_start = time.time()
        for _ in range(config["num_epochs"]):
            for _ in range(frames_per_batch // config["minibatch_size"]):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)
                training_tds.append(loss_vals.detach())

                loss_value = loss_vals["loss"]

                loss_value.backward()

                total_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), config["max_grad_norm"]
                )
                training_tds[-1]["grad_norm"] = total_norm.mean()

                optim.step()
                optim.zero_grad()
                target_net_updater.step()

        qnet_explore.step(frames=current_frames)  # Update exploration annealing
        collector.update_policy_weights_()

        training_time = time.time() - training_start
        print(f"Training took: {training_time}")

        iteration_time = sampling_time + training_time
        total_time += iteration_time
        training_tds = torch.stack(training_tds)

        # More logs
        if log:
            log_training(
                logger,
                training_tds,
                tensordict_data,
                sampling_time,
                training_time,
                total_time,
                i,
                current_frames,
                total_frames,
            )

        if (
            config["evaluation_episodes"] > 0
            and i % config["evaluation_interval"] == 0
            and log
        ):
            evaluation_start = time.time()
            with torch.no_grad() and set_exploration_type(ExplorationType.MEAN):
                env_test.frames = []
                rollouts = env_test.rollout(
                    max_steps=max_steps,
                    policy=qnet,
                    callback=rendering_callback,
                    auto_cast_to_device=True,
                    break_when_any_done=False,
                    # We are running vectorized evaluation we do not want it to stop when just one env is done
                )

                evaluation_time = time.time() - evaluation_start
                print(f"Evaluation took: {evaluation_time}")

                log_evaluation(
                    logger,
                    rollouts,
                    env_test,
                    evaluation_time,
                )

        if log:
            logger.experiment.log({}, commit=True)
        sampling_start = time.time()
    wandb.finish()


if __name__ == "__main__":
    for seed in [0]:
        train(seed)
