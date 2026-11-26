"""HIL-SERL in simulation: SAC with demos, interventions and a success reward.

Luo et al., "Precise and Dexterous Robotic Manipulation via Human-in-the-Loop
Reinforcement Learning", https://arxiv.org/abs/2410.21845.

Robosuite Lift stands in for the robot. A scripted controller plays the
human: it takes over when the policy gets stuck or wanders off. Each batch is
half from the demo buffer and half from the online buffer, and intervened
steps are stored in both. The reward comes from the simulator
(reward.source=oracle) or from a trained success classifier
(reward.source=classifier, see train_reward_classifier.py).

With a classifier, reward.false_positives_out saves the states where it
claimed a success that did not happen. HIL-SERL fixes a classifier by adding
such states as negatives and retraining it (classifier.extra_negatives), then
training the policy again, a few rounds if needed. A person reviews the
claimed successes on a robot; here the simulator's success flag does.
"""
from __future__ import annotations

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import DictConfig
from tensordict import TensorDict
from torchrl._utils import get_available_device, timeit
from torchrl.record.loggers import generate_exp_name, get_logger
from utils import (
    evaluate,
    make_collector,
    make_demos,
    make_env,
    make_loss,
    make_optimizers,
    make_policy,
    make_replay_buffer,
    make_sac_agent,
    make_update,
    to_replay,
)

torch.set_float32_matmul_precision("high")


@hydra.main(version_base="1.3", config_path="", config_name="config")
def main(cfg: DictConfig):
    device = (
        torch.device(cfg.network.device)
        if cfg.network.device
        else get_available_device()
    )
    replay_device = torch.device(cfg.replay.device) if cfg.replay.device else device
    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)

    classifier = None
    if cfg.reward.source == "classifier":
        classifier = torch.load(cfg.reward.classifier_path, weights_only=False)
        classifier.eval()
    elif cfg.reward.source != "oracle":
        raise ValueError(f"Unknown reward source {cfg.reward.source!r}.")

    eval_env = make_env(cfg)
    actor, qvalue = make_sac_agent(cfg, eval_env, device)
    loss, target_updater = make_loss(cfg, actor, qvalue)
    update = make_update(cfg, loss, target_updater, *make_optimizers(cfg, loss))

    replay = make_replay_buffer(cfg, device=replay_device)
    replay[0].extend(make_demos(cfg, classifier=classifier).to(replay_device))

    collector = make_collector(
        cfg, make_policy(actor, cfg), device, classifier=classifier
    )
    logger = get_logger(
        cfg.logger.backend,
        logger_name=cfg.logger.log_dir,
        experiment_name=generate_exp_name("HIL-SERL", cfg.logger.exp_name),
    )

    num_updates = cfg.collector.frames_per_batch * cfg.optim.utd_ratio
    collected = 0
    intervened = 0
    claims = 0
    update_count = 0
    false_positives = []
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)
    try:
        for batch in collector:
            with timeit("extend"):
                replay.extend(
                    to_replay(
                        batch, end_on_success=cfg.reward.end_episode_on_success
                    ).to(replay_device)
                )
            collected += batch.numel()
            intervened += int(batch["intervention"].sum())
            pbar.update(batch.numel())
            if classifier is not None:
                claimed = batch["next", "success_pred"].reshape(-1)
                wrong = claimed & ~batch["next", "success"].reshape(-1)
                claims += int(claimed.sum())
                false_positives.append(
                    batch["next", "observation"].reshape(batch.numel(), -1)[wrong]
                )

            losses = []
            if collected >= cfg.optim.training_starts:
                with timeit("train"):
                    for _ in range(num_updates):
                        sample = replay.sample().to(device)
                        out = update(sample, update_count % cfg.optim.cta_ratio == 0)
                        update_count += 1
                        if cfg.replay.demo_prioritized:
                            sample.set("td_error", out["td_error"])
                            replay.update_tensordict_priority(sample)
                        losses.append(out.exclude("td_error"))
                collector.update_policy_weights_()

            metrics = {
                "intervention_rate": batch["intervention"].float().mean(),
                "human_seconds": intervened / cfg.env.control_freq,
            }
            if classifier is not None:
                metrics["claims"] = claims
                metrics["false_claims"] = sum(len(obs) for obs in false_positives)
            done = batch["next", "done"].squeeze(-1)
            if done.any():
                metrics["train_success"] = (
                    batch["next", "success"].squeeze(-1)[done].float().mean()
                )
                metrics["train_predicted_success"] = (
                    batch["next", "terminated"].squeeze(-1)[done].float().mean()
                )
            if losses:
                metrics.update(torch.stack(losses).mean().to_dict())
            if collected % cfg.logger.eval_every < cfg.collector.frames_per_batch:
                with timeit("eval"):
                    result = evaluate(
                        eval_env,
                        actor,
                        cfg.logger.eval_episodes,
                        cfg.env.max_steps,
                        classifier=classifier,
                    )
                metrics.update({f"eval_{k}": v for k, v in result.items()})
            metrics.update(timeit.todict(prefix="time"))
            for key, value in metrics.items():
                logger.log_scalar(key, float(value), step=collected)
    finally:
        collector.shutdown()
        eval_env.close()
        pbar.close()
        if classifier is not None and cfg.reward.false_positives_out:
            observation = torch.cat(false_positives).cpu()
            TensorDict(observation=observation, batch_size=[len(observation)]).memmap(
                cfg.reward.false_positives_out
            )


if __name__ == "__main__":
    main()
