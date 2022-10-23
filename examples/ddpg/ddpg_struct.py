import hydra
from hydra.utils import instantiate
from torchrl.envs import TransformedEnv, ParallelEnv, Compose


@hydra.main(version_base=None, config_path="../configs", config_name="ddpg")
def my_app(cfg):
    instantiate_cfg = instantiate(cfg)
    env = instantiate_cfg.env
    print("env:", env)
    env = TransformedEnv(ParallelEnv(4, env), Compose(*instantiate_cfg.transforms))
    print(env.reset())


if __name__ == "__main__":
    my_app()
