from torch import nn
from tensordict.nn import TensorDictModule
from torchrl.objectives import PPOLoss
from torchrl.envs.libs.gym import GymEnv
from torchrl.collectors import SyncDataCollector
from torchrl.modules import OneHotCategorical, ProbabilisticActor
from torchrl.distributed_gradient_collector import DistributedGradientCollector


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
    distributed_collector = SyncDataCollector(
        env_maker,
        actor,
        total_frames=1000,
        frames_per_batch=200,
    )

    # 4. Define objective
    loss_module = PPOLoss(actor, critic)

    # Sample batches until reaching total_frames
    counter = 0
    num_frames = 0
    for batch in distributed_collector:

        loss = loss_module(batch)
        loss_sum = loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"]
        loss_sum.backward()

        counter += 1
        num_frames += batch.shape.numel()
        print(f"batch {counter}, total frames {num_frames}")