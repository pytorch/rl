import torch
from torch import nn
from tensordict.nn import TensorDictModule
from torchrl.objectives import PPOLoss
from torchrl.envs.libs.gym import GymEnv
from torchrl.collectors import SyncDataCollector
from torchrl.modules import OneHotCategorical, ProbabilisticActor
from torchrl.distributed_gradient_collector import DistributedGradientCollector, get_weights_and_grad
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement


if __name__ == "__main__":

    # 1. Create environment factory
    def env_maker():
        return GymEnv("CartPole-v1", device="cpu")

    # 2. Define models
    test_env = env_maker()
    action_spec = test_env.action_spec
    obs_spec = test_env.observation_spec
    n = action_spec.space.n
    actor = ProbabilisticActor(
        module=TensorDictModule(nn.Linear(4, n), in_keys=["observation"], out_keys=["logits"]),
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=OneHotCategorical,
        return_log_prob=True,
        spec=action_spec)
    critic = TensorDictModule(nn.Linear(4, 1), in_keys=["observation"], out_keys=["state_value"])

    # 3. Define collector
    collector = SyncDataCollector(
        env_maker,
        actor,
        total_frames=1000,
        frames_per_batch=200,
    )

    # 4. Define data buffer
    sampler = SamplerWithoutReplacement()
    buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(200),
        sampler=sampler,
        batch_size=200,
    )

    # 5. Define objective
    loss_module = PPOLoss(actor, critic)

    # 6. Define gradient collector
    gradient_collector = DistributedGradientCollector(
        model=actor,
        num_workers=2,
        collector=collector,
        loss_module=loss_module,
        data_buffer=buffer,
    )

    # 7. Define optimizer
    optim = torch.optim.Adam(
        loss_module.parameters(),
        lr=0.01,
    )

    # Sample batches until reaching total_frames
    counter = 0
    num_frames = 0
    local_weights, local_grads = get_weights_and_grad(loss_module)
    for remote_grad in gradient_collector:
        import ipdb; ipdb.set_trace()
        local_grads.zero_()  # Just to be sure, should not be necessary
        local_grads.update_(remote_grad)
        optim.step()
        gradient_collector.update_policy_weights_(local_weights)
