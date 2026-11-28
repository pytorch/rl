# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools
import importlib.util
import os

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import InteractionType, TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn, optim
from torchrl._utils import compile_with_warmup, logger as torchrl_logger
from torchrl.collectors import AsyncCollector, Collector
from torchrl.data import (
    Binary,
    Bounded,
    LazyTensorStorage,
    PrioritizedSampler,
    ReplayBufferEnsemble,
    TensorDictReplayBuffer,
    TensorDictRoundRobinWriter,
    Unbounded,
)
from torchrl.envs.libs.gym import default_info_dict_reader, GymWrapper
from torchrl.envs.transforms import (
    Compose,
    DoubleToFloat,
    InitTracker,
    ModuleTransform,
    RewardSum,
    StepCounter,
    SuccessReward,
    TerminateTransform,
    Transform,
    TransformedEnv,
)
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import (
    InterventionModule,
    MLP,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)
from torchrl.objectives import SACLoss, SoftUpdate

_has_robosuite = importlib.util.find_spec("robosuite") is not None
_has_h5py = importlib.util.find_spec("h5py") is not None
_has_gymnasium = importlib.util.find_spec("gymnasium") is not None

OBS_KEYS = ("object-state", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
DEMO_OBS_KEYS = ("object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
CUBE_POS = slice(0, 3)
EEF_POS = slice(10, 13)
GRIPPER_QPOS = slice(17, 19)

REPLAY_KEYS = (
    "observation",
    "action",
    "policy_action",
    "intervention",
    "step_count",
    ("collector", "traj_ids"),
    ("next", "observation"),
    ("next", "reward"),
    ("next", "done"),
    ("next", "terminated"),
    ("next", "truncated"),
    ("next", "success"),
    ("next", "success_prob"),
)
ORACLE_KEYS = (("next", "success"),)


def _gym_success_wrapper():
    if not _has_gymnasium:
        raise ImportError("The HIL-SERL recipe needs gymnasium.")
    import gymnasium

    class LiftSuccessInfo(gymnasium.Wrapper):
        """Adds the simulator's success check to ``info`` so TorchRL can read it."""

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            info["success"] = np.array([self.unwrapped._check_success()], dtype=bool)
            return obs, reward, terminated, truncated, info

    return LiftSuccessInfo


def make_base_env(cfg, device="cpu"):
    """Robosuite Lift with the same four observation groups the robomimic demos store.

    Importing robosuite switches MUJOCO_GL to egl; the original value is put
    back so that collector worker processes start with the same backend.
    """
    if not _has_robosuite:
        raise ImportError(
            "The HIL-SERL recipe needs robosuite (pip install robosuite)."
        )
    gl_backend = os.environ.get("MUJOCO_GL")
    import robosuite
    from robosuite.wrappers import GymWrapper as SuiteGymWrapper

    if gl_backend is not None:
        os.environ["MUJOCO_GL"] = gl_backend

    suite_env = robosuite.make(
        cfg.env.task,
        robots=cfg.env.robot,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=False,
        ignore_done=True,
        control_freq=cfg.env.control_freq,
        horizon=10**9,
        hard_reset=cfg.env.hard_reset,
    )
    gym_env = _gym_success_wrapper()(SuiteGymWrapper(suite_env, keys=list(OBS_KEYS)))
    env = GymWrapper(gym_env, device=device)
    env.set_info_dict_reader(
        default_info_dict_reader(
            ["success"], spec={"success": Binary(n=1, shape=(1,), dtype=torch.bool)}
        )
    )
    if cfg.env.position_only:
        env = TransformedEnv(env, PositionGripperActions())
    return env


class PositionGripperActions(Transform):
    """Gives the policy a 4-D action: a position change and a gripper command.

    Rotation is held at zero. Lift does not need rotation, and HIL-SERL also
    narrows the action space per task. The robot still receives the full
    7-D action.
    """

    def __init__(self):
        super().__init__(in_keys_inv=["action"], out_keys_inv=["action"])

    def _inv_apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [action[..., :3], action.new_zeros(*action.shape[:-1], 3), action[..., 3:]],
            -1,
        )

    def transform_input_spec(self, input_spec):
        input_spec = input_spec.clone()
        spec = input_spec["full_action_spec", "action"]
        input_spec["full_action_spec", "action"] = Bounded(
            -1.0,
            1.0,
            shape=(*spec.shape[:-1], 4),
            dtype=spec.dtype,
            device=spec.device,
        )
        return input_spec


class EpisodeSuccess(Transform):
    """Records whether the simulator reported a success at any step of the episode.

    The training success rate reads it at the end of each episode, so it
    counts the same thing as the evaluation whether or not a success ends the
    episode.
    """

    def __init__(self):
        super().__init__(in_keys=["success"], out_keys=["episode_success"])

    def _reset(self, tensordict, tensordict_reset):
        tensordict_reset.set(
            "episode_success", self.parent.full_observation_spec["success"].zero()
        )
        return tensordict_reset

    def _step(self, tensordict, next_tensordict):
        next_tensordict.set(
            "episode_success",
            tensordict.get("episode_success") | next_tensordict.get("success"),
        )
        return next_tensordict

    def transform_observation_spec(self, observation_spec):
        observation_spec["episode_success"] = observation_spec["success"].clone()
        return observation_spec


def _suite_env(env):
    while isinstance(env, TransformedEnv):
        env = env.base_env
    return env._env.unwrapped


class SuccessClassifier(nn.Module):
    """Predicts task success from the state: an MLP and a threshold."""

    def __init__(self, obs_dim: int, hidden: int = 256, threshold: float = 0.5):
        super().__init__()
        self.net = MLP(
            in_features=obs_dim,
            out_features=1,
            num_cells=[hidden, hidden],
            activation_class=nn.ReLU,
        )
        self.register_buffer("threshold", torch.tensor(float(threshold)))
        self.register_buffer("temperature", torch.tensor(1.0))

    def logits(self, observation):
        return self.net(observation)

    def forward(self, observation):
        prob = torch.sigmoid(self.net(observation) / self.temperature)
        return prob, prob > self.threshold


def classifier_module(classifier: SuccessClassifier) -> TensorDictModule:
    return TensorDictModule(
        classifier,
        in_keys=["observation"],
        out_keys=["success_prob", "success_pred"],
    )


def _add_classifier_specs(spec):
    spec = spec.clone()
    spec["success_prob"] = Unbounded(shape=(*spec.shape, 1))
    spec["success_pred"] = Binary(n=1, shape=(*spec.shape, 1), dtype=torch.bool)
    return spec


def make_env(
    cfg,
    classifier: SuccessClassifier | None = None,
    device="cpu",
    *,
    end_on_success: bool = True,
):
    """Lift with a success reward from the simulator or from a classifier.

    The classifier, when given, decides the reward and, with end_on_success,
    when the episode ends. Without it the episode runs to max_steps, and
    to_replay marks the successful steps as terminal instead. The simulator's
    own success flag is always kept in the data, for evaluation and for
    simulated label checks.
    """
    transforms = [DoubleToFloat(), InitTracker()]
    if classifier is None:
        success_key = "success"
    else:
        success_key = "success_pred"
        transforms.append(
            ModuleTransform(
                module=classifier_module(classifier),
                no_grad=True,
                observation_spec_transform=_add_classifier_specs,
            )
        )
    transforms += [SuccessReward(success_key=success_key), EpisodeSuccess()]
    if end_on_success:
        transforms.append(TerminateTransform(lambda next_td: next_td.get(success_key)))
    transforms += [StepCounter(max_steps=cfg.env.max_steps), RewardSum()]
    env = TransformedEnv(make_base_env(cfg, device=device), Compose(*transforms))
    env.set_seed(cfg.env.seed)
    return env


class ScriptedLiftController(nn.Module):
    """A hand-written Lift controller: move above the cube, go down, close, lift.

    It returns a position change and a gripper command between -1 and 1, or the
    full 7-D action with zero rotation when full_pose is True. The arm holds
    still while the fingers close, and lifts while they still hold the cube.
    """

    def __init__(
        self,
        gain: float = 20.0,
        hover: float = 0.06,
        lift: float = 0.2,
        near: float = 0.02,
        closed_width: float = 0.032,
        in_hand: float = 0.03,
        min_width: float = 0.01,
        grasp_depth: float = 0.01,
        full_pose: bool = False,
    ):
        super().__init__()
        self.gain = gain
        self.hover = hover
        self.lift = lift
        self.near = near
        self.closed_width = closed_width
        self.in_hand = in_hand
        self.min_width = min_width
        self.grasp_depth = grasp_depth
        self.full_pose = full_pose

    def forward(self, observation):
        cube = observation[..., CUBE_POS]
        eef = observation[..., EEF_POS]
        offset = cube - eef
        xy_far = offset[..., :2].norm(dim=-1, keepdim=True) > 0.015
        near = offset.norm(dim=-1, keepdim=True) < self.near
        width = observation[..., GRIPPER_QPOS][..., :1]
        closed = (width < self.closed_width) & (width > self.min_width)
        in_hand = (eef[..., 2:] - cube[..., 2:]) < self.in_hand
        holding = closed & ~xy_far & in_hand
        grasping = near & ~xy_far & ~holding
        hover_target = cube + cube.new_tensor([0.0, 0.0, self.hover])
        lift_target = eef + eef.new_tensor([0.0, 0.0, self.lift])
        target = torch.where(
            xy_far,
            hover_target,
            torch.where(
                holding,
                lift_target,
                torch.where(
                    grasping, eef, cube - cube.new_tensor([0.0, 0.0, self.grasp_depth])
                ),
            ),
        )
        move = ((target - eef) * self.gain).clamp(-1.0, 1.0)
        grip = torch.where(grasping | holding, 1.0, -1.0)
        if self.full_pose:
            return torch.cat([move, torch.zeros_like(move), grip], -1)
        return torch.cat([move, grip], -1)


class StallIntervener(nn.Module):
    """A simulated human who takes over when the policy gets stuck or wanders off.

    The policy is stuck when the gripper barely moves for stall_steps steps in a
    row, and has wandered off when the gripper is more than workspace_radius
    from the cube. A takeover starts reaction_delay steps after either is
    noticed and lasts at most max_takeover steps. Noise is added to the
    controller's actions so the help is imperfect. The takeover state resets at
    the start of each episode, and is rebuilt if the batch shape or device
    changes. It is kept out of the module's buffers so collector weight
    updates never copy it.
    """

    def __init__(
        self,
        controller: nn.Module,
        *,
        stall_steps: int = 10,
        stall_speed: float = 0.002,
        workspace_radius: float = 0.35,
        max_takeover: int = 20,
        noise_std: float = 0.1,
        reaction_delay: int = 0,
        enabled: bool = True,
    ):
        super().__init__()
        self.controller = controller
        self.stall_steps = stall_steps
        self.stall_speed = stall_speed
        self.workspace_radius = workspace_radius
        self.max_takeover = max_takeover
        self.noise_std = noise_std
        self.reaction_delay = reaction_delay
        self.enabled = enabled
        self.last_eef = None
        self.stalled = None
        self.takeover_left = None
        self.delay_left = None

    def forward(self, observation, is_init):
        eef = observation[..., EEF_POS]
        batch = observation.shape[:-1]
        if (
            self.last_eef is None
            or self.last_eef.shape != eef.shape
            or self.last_eef.device != eef.device
        ):
            self.last_eef = eef.clone()
            self.stalled = torch.zeros(batch, dtype=torch.long, device=eef.device)
            self.takeover_left = torch.zeros_like(self.stalled)
            self.delay_left = torch.zeros_like(self.stalled)
        reset = is_init.reshape(batch)
        speed = (eef - self.last_eef).norm(dim=-1)
        self.stalled = torch.where(
            reset | (speed >= self.stall_speed), 0, self.stalled + 1
        )
        outside = (observation[..., CUBE_POS] - eef).norm(
            dim=-1
        ) > self.workspace_radius
        trigger = (self.stalled >= self.stall_steps) | outside
        waiting = self.delay_left > 0
        noticed = (self.takeover_left == 0) & ~waiting & trigger
        self.delay_left = torch.where(
            reset | ~(noticed | waiting),
            0,
            torch.where(noticed, self.reaction_delay, self.delay_left - 1),
        )
        start = (noticed | waiting) & (self.delay_left == 0)
        self.takeover_left = torch.where(
            reset,
            0,
            torch.where(
                start,
                self.max_takeover,
                (self.takeover_left - 1).clamp_min(0),
            ),
        )
        active = self.takeover_left > 0
        if not self.enabled:
            active = torch.zeros_like(active)
        self.stalled = torch.where(active, 0, self.stalled)
        self.last_eef = eef.clone()
        action = self.controller(observation)
        action = (action + self.noise_std * torch.randn_like(action)).clamp(-1.0, 1.0)
        return action, active


def make_intervener(cfg) -> TensorDictModule:
    icfg = cfg.intervention
    return TensorDictModule(
        StallIntervener(
            ScriptedLiftController(full_pose=not cfg.env.position_only),
            stall_steps=icfg.stall_steps,
            stall_speed=icfg.stall_speed,
            workspace_radius=icfg.workspace_radius,
            max_takeover=icfg.max_takeover,
            noise_std=icfg.noise_std,
            reaction_delay=icfg.reaction_delay,
            enabled=icfg.enabled,
        ),
        in_keys=["observation", "is_init"],
        out_keys=[("intervener", "action"), ("intervener", "active")],
    )


def make_sac_agent(cfg, env, device):
    """Builds the SAC actor and critic, with LayerNorm as in the HIL-SERL code."""
    action_spec = env.action_spec_unbatched.to(device)
    hidden = list(cfg.network.hidden_sizes)
    if len(set(hidden)) != 1:
        raise ValueError("LayerNorm MLPs here expect equal hidden sizes.")
    actor_net = nn.Sequential(
        MLP(
            num_cells=hidden,
            out_features=2 * action_spec.shape[-1],
            activation_class=nn.ReLU,
            norm_class=nn.LayerNorm,
            norm_kwargs={"normalized_shape": hidden[0]},
            device=device,
        ),
        NormalParamExtractor(scale_mapping="biased_softplus_1.0", scale_lb=1e-4),
    )
    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=["loc", "scale"],
        module=TensorDictModule(
            actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
        ),
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": action_spec.space.low,
            "high": action_spec.space.high,
            "tanh_loc": False,
        },
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    )
    qvalue = ValueOperator(
        in_keys=["action", "observation"],
        module=MLP(
            num_cells=hidden,
            out_features=1,
            activation_class=nn.ReLU,
            norm_class=nn.LayerNorm,
            norm_kwargs={"normalized_shape": hidden[0]},
            device=device,
        ),
    )
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = env.fake_tensordict().to(device)
        actor(td)
        qvalue(td)
    return actor, qvalue


def make_loss(cfg, actor, qvalue):
    """Builds the SAC loss and its target updater.

    The loss creates its target-entropy buffer on first use; it is read here
    so the buffer exists before the update is compiled.
    """
    loss = SACLoss(
        actor_network=actor,
        qvalue_network=qvalue,
        num_qvalue_nets=cfg.optim.num_qvalue_nets,
        loss_function="l2",
        delay_actor=False,
        delay_qvalue=True,
        alpha_init=cfg.optim.alpha_init,
        backup_entropy=cfg.optim.backup_entropy,
    )
    loss.make_value_estimator(gamma=cfg.optim.gamma)
    loss.target_entropy
    leaked = [key for key in ORACLE_KEYS if key in loss.in_keys]
    if leaked:
        raise RuntimeError(f"The loss reads oracle keys {leaked}.")
    return loss, SoftUpdate(loss, tau=cfg.optim.tau)


def make_optimizers(cfg, loss):
    critic = optim.Adam(
        loss.qvalue_network_params.flatten_keys().values(), lr=cfg.optim.lr
    )
    actor = optim.Adam(
        loss.actor_network_params.flatten_keys().values(), lr=cfg.optim.lr
    )
    alpha = optim.Adam([loss.log_alpha], lr=cfg.optim.lr)
    return critic, actor, alpha


def make_update(cfg, loss, target_updater, critic_optim, actor_optim, alpha_optim):
    """Builds one SAC update step. The actor and temperature update only when asked.

    The HIL-SERL code updates the critic cta_ratio times for each actor update.
    A behavior-cloning term on intervened and demo rows can be added with
    optim.bc_weight; it is off by default. Only the loss and its backward pass
    are compiled, because two compiled update functions with CUDA graphs
    overwrite each other's outputs. Returns the losses, the temperature and the
    TD errors.
    """

    bc_weight = cfg.optim.bc_weight

    def grads(batch):
        out = loss(batch)
        total = out["loss_qvalue"] + out["loss_actor"] + out["loss_alpha"]
        if bc_weight:
            intervened = batch.get("intervention").reshape(-1).float()
            dist = loss.actor_network.get_dist(batch.select("observation"))
            log_prob = dist.log_prob(batch.get("action").clamp(-0.999, 0.999))
            loss_bc = -(log_prob * intervened).sum() / intervened.sum().clamp_min(1)
            total = total + bc_weight * loss_bc
        total.backward()
        out = out.select("loss_qvalue", "loss_actor", "alpha").detach()
        out.set("td_error", batch.get("td_error").detach())
        return out

    if cfg.compile.compile:
        grads = compile_with_warmup(grads, mode=cfg.compile.mode, warmup=1)

    def update(batch, update_actor: bool):
        torch.compiler.cudagraph_mark_step_begin()
        out = grads(batch).clone()
        critic_optim.step()
        if update_actor:
            actor_optim.step()
            alpha_optim.step()
        for optimizer in (critic_optim, actor_optim, alpha_optim):
            optimizer.zero_grad(set_to_none=True)
        target_updater.step()
        return out

    return update


def make_collector(cfg, policy, device, classifier: SuccessClassifier | None = None):
    """A collector in this process, or in a separate one with collector.async_collection.

    The separate process steps the simulator while this one trains, like
    HIL-SERL's asynchronous actor and learner. Call update_policy_weights_
    after training to send it the new weights.
    """
    kwargs = {
        "frames_per_batch": cfg.collector.frames_per_batch,
        "total_frames": cfg.collector.total_frames,
    }
    end_on_success = cfg.reward.end_episode_on_success
    if cfg.collector.async_collection:
        return AsyncCollector(
            functools.partial(make_env, cfg, classifier, end_on_success=end_on_success),
            policy,
            device=device,
            env_device=torch.device("cpu"),
            **kwargs,
        )
    return Collector(
        make_env(cfg, classifier=classifier, end_on_success=end_on_success),
        policy,
        policy_device=device,
        **kwargs,
    )


def make_policy(actor, cfg) -> TensorDictSequential:
    return TensorDictSequential(actor, InterventionModule(make_intervener(cfg)))


def make_replay_buffer(cfg, device="cpu") -> ReplayBufferEnsemble:
    """A demo buffer and an online buffer. Each batch is half from each.

    Each buffer samples with its own generator seeded from env.seed, on the
    device its sampler draws on. Prefetching is turned off on MPS, where
    sampling from a background thread aborts the process.
    """
    rcfg = cfg.replay
    prefetch = rcfg.prefetch
    if prefetch and torch.device(device).type == "mps":
        torchrl_logger.warning("replay.prefetch is not supported on MPS; using 0.")
        prefetch = 0
    demo_sampler = (
        PrioritizedSampler(rcfg.demo_capacity, alpha=rcfg.alpha, beta=rcfg.beta)
        if rcfg.demo_prioritized
        else None
    )
    demo_kwargs = {"sampler": demo_sampler} if demo_sampler is not None else {}
    demo = TensorDictReplayBuffer(
        storage=LazyTensorStorage(rcfg.demo_capacity, device=device),
        writer=TensorDictRoundRobinWriter(track_generations=True),
        **demo_kwargs,
    )
    online = TensorDictReplayBuffer(
        storage=LazyTensorStorage(rcfg.online_capacity, device=device),
        writer=TensorDictRoundRobinWriter(track_generations=True),
    )
    replay = ReplayBufferEnsemble(
        demo,
        online,
        routing_key="route",
        routing_mode="mask",
        fractions=[rcfg.demo_fraction, 1.0 - rcfg.demo_fraction],
        batch_size=cfg.optim.batch_size,
        prefetch=prefetch or None,
    )
    demo_device = "cpu" if demo_sampler is not None else device
    demo.set_rng(torch.Generator(device=demo_device).manual_seed(cfg.env.seed))
    online.set_rng(torch.Generator(device=device).manual_seed(cfg.env.seed + 1))
    return replay


def to_replay(batch: TensorDictBase, *, end_on_success: bool = True) -> TensorDictBase:
    """Keeps the keys the buffers store and marks intervened steps for both buffers.

    When the episode did not end on success (end_on_success=False), the steps
    with a success reward are marked terminal here, so the critic still
    treats a success as the end of the task.
    """
    batch = batch.reshape(-1).select(*REPLAY_KEYS, strict=False)
    if not end_on_success:
        success = batch.get(("next", "reward")) > 0
        batch.set(("next", "terminated"), success)
        batch.set(("next", "done"), success | batch.get(("next", "truncated")))
    if ("next", "success_prob") not in batch.keys(True):
        batch.set(("next", "success_prob"), batch.get(("next", "success")).float())
    intervention = batch.get("intervention").reshape(-1)
    batch.set("route", torch.stack([intervention, torch.ones_like(intervention)], -1))
    return batch


def _demo_names(data) -> list[str]:
    """The demo names in recorded order: demo_0, demo_1, ..., demo_10."""
    return sorted(data.keys(), key=lambda name: int(name.split("_")[-1]))


def load_demos(
    path: str,
    num_demos: int,
    *,
    classifier: SuccessClassifier | None = None,
    position_only: bool = False,
):
    """Loads robomimic Lift demos, each cut at its first success.

    Rewards and episode ends come from the same source as the online data: the
    simulator when no classifier is given, the classifier otherwise. Demo
    trajectory ids are negative so they never clash with collected ones.
    """
    if not _has_h5py:
        raise ImportError("Loading robomimic demos needs h5py.")
    import h5py

    episodes = []
    with h5py.File(path, "r") as f:
        data = f["data"]
        for k, name in enumerate(_demo_names(data)[:num_demos]):
            demo = data[name]
            obs = torch.cat(
                [torch.as_tensor(demo["obs"][key][()]) for key in DEMO_OBS_KEYS], -1
            ).float()
            next_obs = torch.cat(
                [torch.as_tensor(demo["next_obs"][key][()]) for key in DEMO_OBS_KEYS],
                -1,
            ).float()
            action = torch.as_tensor(demo["actions"][()]).float()
            if position_only:
                action = action[..., [0, 1, 2, 6]]
            success = torch.as_tensor(demo["rewards"][()]).bool().unsqueeze(-1)
            if success.any():
                length = int(success.squeeze(-1).nonzero()[0]) + 1
            else:
                length = action.shape[0]
            T = length
            episodes.append(
                TensorDict(
                    {
                        "observation": obs[:T],
                        "action": action[:T],
                        "policy_action": action[:T],
                        "intervention": torch.ones(T, 1, dtype=torch.bool),
                        "step_count": torch.arange(T).unsqueeze(-1),
                        ("collector", "traj_ids"): torch.full((T,), -(k + 1)),
                        ("next", "observation"): next_obs[:T],
                        ("next", "success"): success[:T],
                    },
                    [T],
                )
            )
    demos = torch.cat(episodes)
    return relabel(demos, classifier)


@torch.no_grad()
def collect_scripted_demos(
    cfg, num_demos: int, *, classifier: SuccessClassifier | None = None
):
    """Records demos with the scripted controller, in the same format as load_demos.

    Used when no robomimic file is given, for example in CI.
    """
    env = make_env(cfg)
    policy = TensorDictModule(
        ScriptedLiftController(full_pose=not cfg.env.position_only),
        in_keys=["observation"],
        out_keys=["action"],
    )
    episodes = []
    for k in range(num_demos):
        rollout = env.rollout(cfg.env.max_steps, policy, break_when_any_done=True)
        T = rollout.shape[0]
        episodes.append(
            TensorDict(
                {
                    "observation": rollout["observation"],
                    "action": rollout["action"],
                    "policy_action": rollout["action"],
                    "intervention": torch.ones(T, 1, dtype=torch.bool),
                    "step_count": torch.arange(T).unsqueeze(-1),
                    ("collector", "traj_ids"): torch.full((T,), -(k + 1)),
                    ("next", "observation"): rollout["next", "observation"],
                    ("next", "success"): rollout["next", "success"],
                    ("next", "truncated"): rollout["next", "truncated"],
                },
                [T],
            )
        )
    env.close()
    return relabel(torch.cat(episodes), classifier)


def make_demos(cfg, classifier: SuccessClassifier | None = None):
    """Loads robomimic demos if a path is set, and records scripted demos otherwise."""
    if cfg.demos.path is None:
        return collect_scripted_demos(cfg, cfg.demos.num_demos, classifier=classifier)
    check_obs_alignment(cfg, cfg.demos.path)
    return load_demos(
        cfg.demos.path,
        cfg.demos.num_demos,
        classifier=classifier,
        position_only=cfg.env.position_only,
    )


@torch.no_grad()
def relabel(data: TensorDictBase, classifier: SuccessClassifier | None):
    """Sets the reward and episode-end flags from the chosen success source."""
    if classifier is None:
        pred = data.get(("next", "success"))
        prob = pred.float()
    else:
        device = next(classifier.parameters()).device
        prob, pred = classifier(data.get(("next", "observation")).to(device))
        prob, pred = prob.to(data.device or "cpu"), pred.to(data.device or "cpu")
    data.set(("next", "success_prob"), prob)
    data.set(("next", "reward"), pred.float())
    data.set(("next", "terminated"), pred.clone())
    truncated = data.get(("next", "truncated"), torch.zeros_like(pred))
    data.set(("next", "truncated"), truncated)
    data.set(("next", "done"), pred | truncated)
    return data


@torch.no_grad()
def evaluate(
    env,
    actor,
    episodes: int,
    max_steps: int,
    classifier: SuccessClassifier | None = None,
) -> TensorDictBase:
    """Runs deterministic episodes and averages their true success and length.

    The predicted success is the classifier's verdict on the episode when one
    is given, and the simulator's otherwise.
    """
    results = []
    for _ in range(episodes):
        with set_exploration_type(ExplorationType.DETERMINISTIC):
            rollout = env.rollout(
                max_steps, actor, break_when_any_done=True, auto_cast_to_device=True
            )
        if classifier is None:
            predicted = rollout["next", "terminated"]
        else:
            with torch.no_grad():
                _, predicted = classifier(rollout["next", "observation"])
        results.append(
            TensorDict(
                {
                    "success": rollout["next", "success"].any().float(),
                    "length": torch.tensor(float(rollout.shape[0])),
                    "predicted_success": predicted.any().float(),
                },
                [],
            )
        )
    return torch.stack(results).mean()


def check_obs_alignment(
    cfg, path: str, num_demos: int = 3, atol: float = 1e-6
) -> float:
    """Checks that the simulator reproduces the demos' recorded observations."""
    if not _has_h5py:
        raise ImportError("Checking demo alignment needs h5py.")
    import h5py

    env = make_base_env(cfg)
    suite = _suite_env(env)
    worst = 0.0
    with h5py.File(path, "r") as f:
        data = f["data"]
        for name in _demo_names(data)[:num_demos]:
            demo = data[name]
            suite.reset_from_xml_string(suite.edit_model_xml(demo.attrs["model_file"]))
            suite.sim.reset()
            suite.sim.set_state_from_flattened(demo["states"][0])
            suite.sim.forward()
            obs = suite._get_observations(force_update=True)
            env_obs = torch.cat([torch.as_tensor(obs[key]) for key in OBS_KEYS])
            demo_obs = torch.cat(
                [torch.as_tensor(demo["obs"][key][0]) for key in DEMO_OBS_KEYS]
            )
            worst = max(worst, float((env_obs - demo_obs).abs().max()))
    env.close()
    if worst > atol:
        raise RuntimeError(
            f"Demo and environment observations differ by {worst}; check the "
            "robosuite version and observation keys."
        )
    return worst
