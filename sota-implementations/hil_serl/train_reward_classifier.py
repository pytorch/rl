"""Trains the HIL-SERL success classifier on labelled Lift states.

HIL-SERL trains its success classifier on about 200 successes and 1000
failures recorded by a person. Two sources are supported:

- classifier.source=human uses the states of recorded human demos (for
  instance the robomimic proficient and multi-human Lift sets), labelled by
  the success they reached. Held-out states come from each file's valid
  split, and the classifier is also scored on scripted and random rollouts,
  which look more like a learning policy's data.
- classifier.source=synthetic runs the scripted controller (with noise) and
  random actions, and labels every state with the simulator.

Rollouts use classifier.collect_env (the full-pose controller and hard
resets by default) and last classifier.episode_steps steps, whatever the
training env uses, so the classifier sees the same states however the policy
is set up.

Both keep the paper's 1 to 5 ratio, and states just before a success count as
failures. The classifier is saved with its error rates.

A policy learns to reach states the classifier wrongly calls successes.
hil_serl.py saves them (reward.false_positives_out), and
classifier.extra_negatives adds every saved state to the next training set as
a failure. Repeating this for a few rounds is how HIL-SERL fixes a classifier.

Usage::

    python train_reward_classifier.py demos.path=/path/to/lift_ph.hdf5 \\
        classifier.out=classifier.pt
    python hil_serl.py reward.source=classifier \\
        reward.classifier_path=classifier.pt reward.false_positives_out=fp_0
    python train_reward_classifier.py classifier.out=classifier_1.pt \\
        "classifier.extra_negatives=[fp_0]"
"""
from __future__ import annotations

import json

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl._utils import logger as torchrl_logger
from torchrl.envs.transforms import Compose, DoubleToFloat, StepCounter, TransformedEnv
from utils import (
    _has_h5py,
    DEMO_OBS_KEYS,
    make_base_env,
    ScriptedLiftController,
    SuccessClassifier,
)


class NoisyController(nn.Module):
    """The scripted controller with added noise. A noise level of 2 is close to random."""

    def __init__(self, noise_std: float, full_pose: bool):
        super().__init__()
        self.controller = ScriptedLiftController(full_pose=full_pose)
        self.noise_std = noise_std

    def forward(self, observation):
        action = self.controller(observation)
        return (action + self.noise_std * torch.randn_like(action)).clamp(-1.0, 1.0)


def collect(cfg, episodes: int, noise_std: float, steps: int) -> TensorDict:
    """Records states with their true success labels and their distance to the first success."""
    env = TransformedEnv(
        make_base_env(cfg), Compose(DoubleToFloat(), StepCounter(max_steps=steps))
    )
    policy = TensorDictModule(
        NoisyController(noise_std, full_pose=not cfg.env.position_only),
        in_keys=["observation"],
        out_keys=["action"],
    )
    records = []
    for _ in range(episodes):
        rollout = env.rollout(steps, policy, break_when_any_done=False)
        success = rollout["next", "success"].squeeze(-1)
        first = int(success.nonzero()[0]) if success.any() else steps
        steps_to_success = first - torch.arange(rollout.shape[0])
        records.append(
            TensorDict(
                {
                    "observation": rollout["next", "observation"],
                    "label": success,
                    "steps_to_success": steps_to_success,
                },
                [rollout.shape[0]],
            )
        )
    env.close()
    return torch.cat(records)


def human_states(paths, split: str) -> TensorDict:
    """Reads the states of recorded human demos with their success labels.

    Uses the robomimic train or valid split of each file, so held-out states
    come from demos the classifier never saw.
    """
    if not _has_h5py:
        raise ImportError("Reading human demos needs h5py.")
    import h5py

    records = []
    for path in paths:
        with h5py.File(path, "r") as f:
            names = [name.decode() for name in f["mask"][split][()]]
            for name in names:
                demo = f["data"][name]
                next_obs = torch.cat(
                    [
                        torch.as_tensor(demo["next_obs"][key][()])
                        for key in DEMO_OBS_KEYS
                    ],
                    -1,
                ).float()
                success = torch.as_tensor(demo["rewards"][()]) > 0
                first = int(success.nonzero()[0]) if success.any() else len(success)
                records.append(
                    TensorDict(
                        {
                            "observation": next_obs,
                            "label": success,
                            "steps_to_success": first - torch.arange(len(success)),
                        },
                        [len(success)],
                    )
                )
    return torch.cat(records)


def subsample(
    data: TensorDict, positives: int, negatives: int, near_miss: float, generator
):
    """Picks the training set: successes, near misses and other failures."""
    pos = data[data["label"]]
    neg = data[~data["label"]]
    is_near = (neg["steps_to_success"] > 0) & (neg["steps_to_success"] <= 5)
    near, far = neg[is_near], neg[~is_near]
    num_near = min(int(near_miss * negatives), len(near))

    def pick(td, n):
        n = min(n, len(td))
        return td[torch.randperm(len(td), generator=generator)[:n]]

    return torch.cat(
        [pick(pos, positives), pick(near, num_near), pick(far, negatives - num_near)]
    )


def error_rates(classifier: SuccessClassifier, data: TensorDict) -> dict:
    with torch.no_grad():
        _, pred = classifier(data["observation"])
    pred, label = pred.squeeze(-1), data["label"]
    return {
        "accuracy": float((pred == label).float().mean()),
        "false_positive_rate": float(pred[~label].float().mean()),
        "false_negative_rate": float((~pred[label]).float().mean()),
        "num_positives": int(label.sum()),
        "num_negatives": int((~label).sum()),
    }


@hydra.main(version_base="1.3", config_path="", config_name="config")
def main(cfg: DictConfig):
    ccfg = cfg.classifier
    torch.manual_seed(ccfg.seed)
    generator = torch.Generator().manual_seed(ccfg.seed)
    steps = ccfg.episode_steps
    collect_cfg = OmegaConf.merge(cfg, {"env": ccfg.collect_env})

    def dataset(episodes):
        return torch.cat(
            [
                collect(collect_cfg, episodes, ccfg.controller_noise, steps),
                collect(collect_cfg, max(episodes // 4, 1), ccfg.random_noise, steps),
            ]
        )

    if ccfg.source == "human":
        pool = human_states(ccfg.human_paths, "train")
        held_out = {"held_out": human_states(ccfg.human_paths, "valid")}
        held_out["rollouts"] = dataset(ccfg.held_out_episodes)
    elif ccfg.source == "synthetic":
        pool = dataset(ccfg.train_episodes)
        held_out = {"held_out": dataset(ccfg.held_out_episodes)}
    else:
        raise ValueError(f"Unknown classifier source {ccfg.source!r}.")
    train = subsample(
        pool, ccfg.positives, ccfg.negatives, ccfg.near_miss_share, generator
    )
    if ccfg.extra_negatives:
        extra = torch.cat(
            [
                TensorDict.load_memmap(path).to_tensordict()
                for path in ccfg.extra_negatives
            ]
        )
        extra["label"] = torch.zeros(len(extra), dtype=torch.bool)
        extra["steps_to_success"] = torch.zeros(len(extra), dtype=torch.long)
        held_out["extra_negatives"] = extra
        train = torch.cat([train, extra])

    classifier = SuccessClassifier(train["observation"].shape[-1])
    optimizer = torch.optim.Adam(classifier.parameters(), lr=ccfg.lr)
    positive_weight = (~train["label"]).sum() / train["label"].sum().clamp_min(1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    for _ in range(ccfg.epochs):
        for batch in train[torch.randperm(len(train), generator=generator)].split(64):
            loss = loss_fn(
                classifier.logits(batch["observation"]).squeeze(-1),
                batch["label"].float(),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    report = {"source": ccfg.source, "train": error_rates(classifier, train)}
    report.update(
        {name: error_rates(classifier, data) for name, data in held_out.items()}
    )
    torch.save(classifier, ccfg.out)
    with open(f"{ccfg.out}.json", "w") as f:
        json.dump(report, f, indent=2)
    torchrl_logger.info(f"Saved {ccfg.out}: {json.dumps(report)}")


if __name__ == "__main__":
    main()
