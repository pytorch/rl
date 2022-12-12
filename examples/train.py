import torch
import hydra
from hydra.utils import instantiate
from torchrl.envs import TransformedEnv, Compose
from torchrl.modules.tensordict_module.actors import ActorValueOperator


@hydra.main(version_base=None, config_path="configs", config_name="ppo.yaml")
def my_app(cfg):

    device = (
        torch.device("cpu")
        if torch.cuda.device_count() == 0
        else torch.device("cuda:0")
    )

    # 1. Define environment --------------------------------------------------------------------------------------------

    instantiate_cfg = instantiate(cfg)

    # Get environment function
    base_env = instantiate_cfg.env.base_env

    # Create env vector
    vec_env = instantiate_cfg.env.vec_env(create_env_fn=base_env)

    # Apply transformations to vec env
    transformed_vec_env = TransformedEnv(vec_env, Compose(*instantiate_cfg.env.transforms))

    # 2. Define model --------------------------------------------------------------------------------------------------

    instantiate_cfg = instantiate(cfg)

    # Get environment function
    base_env = instantiate_cfg.env.base_env

    # Apply transformations - this env can be reused later as a recorder
    transformed_env = TransformedEnv(base_env(), Compose(*instantiate_cfg.env.transforms))

    # Define Model
    model = instantiate_cfg.model(proof_environment=transformed_env, device=device)

    # Close Transformed Env
    transformed_env.close()

    # 3. Ddefine Collector ---------------------------------------------------------------------------------------------

    # Define collector
    collector = instantiate_cfg.collector(create_env_fn=transformed_vec_env, policy=model["policy"])

    # 4. Define Replay Buffer (if required) ----------------------------------------------------------------------------

    if "replay_buffer" in instantiate_cfg.keys():
        replay_buffer = instantiate_cfg.replay_buffer()
    else:
        replay_buffer = None

    # 5. Define Objective ----------------------------------------------------------------------------------------------

    if "advantage" in instantiate_cfg.objective.keys():
        advantage_module = instantiate_cfg.objective.advantage(
            gamma=instantiate_cfg.objective.loss.keywords["gamma"],
            value_network=model["critic"])
        model["advantage_module"] = advantage_module
    else:
        advantage_module = None

    loss_kwargs = {}
    for k, v in instantiate_cfg.objective.loss.keywords.items():
        if v == "MISSING":
            if k in model.keys():
                loss_kwargs[k] = model[k]

    objective = instantiate_cfg.objective.loss(**loss_kwargs)

    # 6. Define target network updater (if required) -------------------------------------------------------------------

    if "target_net_updater" in instantiate_cfg.objective.keys():
        target_net_updater = nstantiate_cfg.objective.target_net_updater(
            loss_module=objective)
    else:
        target_net_updater = None

    # 7. Define Logger -------------------------------------------------------------------------------------------------

    logger = instantiate_cfg.logger()

    # 8. Define Recorder -----------------------------------------------------------------------------------------------

    if "recorder" in instantiate_cfg.keys():
        transformed_env.append_transform(
            instantiate_cfg.recorder.recorder(logger=logger))
        recorder_obj = instantiate_cfg.recorder.recorder_hook(
            recorder=transformed_env,
            policy_exploration=model["policy"],
            frame_skip=instantiate_cfg.env.base_env.keywords["frame_skip"] or 0,
        )

    else:
        recorder_obj = None

    # 9. Define Trainer ------------------------------------------------------------------------------------------------

    trainer = instantiate_cfg.trainer(
        collector=collector,
        loss_module=objective,
        recorder=recorder_obj,
        target_net_updater=target_net_updater,
        policy_exploration=model["policy"],
        replay_buffer=replay_buffer,
        logger=logger,
        frame_skip=instantiate_cfg.env.base_env.keywords["frame_skip"] or 0,
    )

    ### 10. Train

    # trainer.train()


if __name__ == "__main__":
    my_app()