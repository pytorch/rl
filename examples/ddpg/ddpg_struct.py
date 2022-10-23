from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None, config_path="../configs", config_name="ddpg")
def my_app(cfg):
    instantiate_cfg = instantiate(cfg)
    env = instantiate_cfg.env
    print("env:", env)

if __name__ == "__main__":
    my_app()
