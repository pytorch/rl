import sys
from torch import nn
from torch import multiprocessing as mp
from tensordict.nn import TensorDictModule
from torchrl.objectives import PPOLoss
from torchrl.envs.libs.gym import GymEnv
from torchrl.collectors import SyncDataCollector
from torchrl.objectives.value.advantages import GAE
from torchrl.modules import OneHotCategorical, ProbabilisticActor


def run_collector(mp_collector, mp_advantage, mp_loss,  qout):
    for batch in mp_collector:
        try:
            batch = mp_advantage(batch)
            loss = mp_loss(batch)
            qout.put("succeeded")
            print('succeeded')
        except AttributeError:  #  AttributeError: 'PPOLoss' object has no attribute 'actor_params'
            qout.put("failed")
            print('failed')


if __name__ == "__main__":

    def env_maker():
        return GymEnv("CartPole-v1", device="cpu")

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

    collector = SyncDataCollector(
        env_maker,
        actor,
        total_frames=200,
        frames_per_batch=200,
    )

    advantage_module = GAE(
        gamma=0.99,
        lmbda=0.95,
        value_network=critic,
        average_gae=True,
        advantage_key="advantage",
    )
    loss_module = PPOLoss(actor=actor, critic=critic)

    qout = mp.Queue(1)
    p = mp.Process(target=run_collector, args=(collector, advantage_module, loss_module, qout))
    p.start()
    assert qout.get() == "succeeded"
    p.join()

