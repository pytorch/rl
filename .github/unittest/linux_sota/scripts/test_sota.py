# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from tensordict.nn import composite_lp_aggregate
from torchrl.checkpoint import Checkpoint, resolve_checkpoint_path

# Check that we're using the new behavior
assert (
    not composite_lp_aggregate()
), "Composite LP must be set to False. Run this test with COMPOSITE_LP_AGGREGATE=0"

commands = {
    "fql": """python sota-implementations/fql/fql.py \
  device=cpu dataset.name=null dataset.random_frames=64 \
  env.max_episode_steps=20 \
  optim.offline_steps=10 optim.online_steps=12 optim.batch_size=16 \
  network.width=32 network.depth=2 network.num_steps=3 \
  evaluation.interval=10 evaluation.episodes=1 log_interval=1 \
  hydra.run.dir=$SOTA_LOG_DIR/fql
""",
    "dqn_trainer_resume": """python sota-implementations/dqn_trainer/train.py \
  collector.total_frames=2000 \
  collector.frames_per_batch=1000 \
  collector.init_random_frames=1000 \
  trainer.optim_steps_per_batch=2 \
  trainer.progress_bar=false \
  hydra.run.dir=outputs/sota_dqn_trainer \
&& python sota-implementations/dqn_trainer/train.py \
  resume=outputs/sota_dqn_trainer/checkpoints \
  collector.total_frames=3000 \
  trainer.progress_bar=false \
  hydra.run.dir=outputs/sota_dqn_trainer_resumed
""",
    "sac_resume": """python sota-implementations/sac/sac.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=2 \
  optim.batch_size=10 \
  optim.utd_ratio=1 \
  replay_buffer.size=120 \
  env.name=Pendulum-v1 \
  logger.backend= \
  checkpoint.interval=16 \
  hydra.run.dir=outputs/sota_sac \
&& python sota-implementations/sac/sac.py \
  resume=outputs/sota_sac/checkpoints \
  collector.total_frames=80 \
  hydra.run.dir=outputs/sota_sac_resumed
""",
    "td3_resume": """python sota-implementations/td3/td3.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  optim.batch_size=10 \
  collector.frames_per_batch=16 \
  collector.num_workers=4 \
  collector.env_per_collector=2 \
  logger.mode=offline \
  env.name=Pendulum-v1 \
  logger.backend= \
  checkpoint.interval=16 \
  hydra.run.dir=outputs/sota_td3 \
&& python sota-implementations/td3/td3.py \
  resume=outputs/sota_td3/checkpoints \
  collector.total_frames=80 \
  hydra.run.dir=outputs/sota_td3_resumed
""",
    "ddpg_resume": """python sota-implementations/ddpg/ddpg.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  optim.batch_size=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=2 \
  optim.utd_ratio=1 \
  replay_buffer.size=120 \
  env.name=Pendulum-v1 \
  logger.backend= \
  checkpoint.interval=16 \
  hydra.run.dir=outputs/sota_ddpg \
&& python sota-implementations/ddpg/ddpg.py \
  resume=outputs/sota_ddpg/checkpoints \
  collector.total_frames=80 \
  hydra.run.dir=outputs/sota_ddpg_resumed
""",
    "vla_grpo": """python sota-implementations/vla_grpo/vla-grpo.py \
  collector.groups_per_iter=2 \
  collector.group_size=2 \
  collector.total_iters=3 \
  loss.mini_batch_size=8 \
  logger.backend= \
  logger.eval_iter=2 \
  logger.eval_episodes=4 \
  checkpoint.save_iter=2
""",
    "reward_model_training": """python sota-implementations/reward_model_training/reward_model.py \
  model.name= \
  data.dataset_name= \
  data.synthetic_size=32 \
  data.batch_size=8 \
  data.max_length=32 \
  optim.max_iters=3 \
  logger.eval_iter=2 \
  logger.eval_iters=1 \
  logger.log_interval=1 \
  logger.backend= \
  export.save_iter=2
""",
    "diffusion_bc": """python sota-implementations/diffusion_bc/diffusion_bc.py \
  optim.gradient_steps=55 \
  replay_buffer.dataset= \
  replay_buffer.demo_episodes=5 \
  env.name=Pendulum-v1 \
  env.max_episode_steps=200 \
  network.num_steps=5 \
  logger.backend= \
  logger.eval_iter=50
""",
    "td3_bc": """python sota-implementations/td3_bc/td3_bc.py \
  optim.gradient_steps=55 \
  logger.backend=
""",
    "impala_single_node": """python sota-implementations/impala/impala_single_node.py \
  collector.total_frames=80 \
  collector.frames_per_batch=20 \
  collector.num_workers=1 \
  logger.backend= \
  env.backend=gymnasium \
  logger.test_interval=10
""",
    "ppo_mujoco": """python sota-implementations/ppo/ppo_mujoco.py \
  env.env_name=HalfCheetah-v4 \
  env.max_episode_steps=20 \
  collector.total_frames=40 \
  collector.frames_per_batch=20 \
  loss.mini_batch_size=10 \
  loss.ppo_epochs=2 \
  optim.device=cpu \
  logger.backend=csv \
  logger.video=False \
  logger.test_interval=10 \
  logger.num_test_episodes=1 \
  hydra.run.dir=$SOTA_LOG_DIR/ppo_mujoco
""",
    "rnd_mujoco": """python sota-implementations/rnd/rnd_mujoco.py \
  env.env_name=HalfCheetah-v4 \
  collector.total_frames=40 \
  collector.frames_per_batch=20 \
  loss.mini_batch_size=10 \
  loss.ppo_epochs=2 \
  logger.backend= \
  logger.test_interval=40 \
  logger.num_test_episodes=1
""",
    "ppo_atari": """python sota-implementations/ppo/ppo_atari.py \
  collector.total_frames=80 \
  collector.frames_per_batch=20 \
  loss.mini_batch_size=20 \
  loss.ppo_epochs=2 \
  logger.backend= \
  env.backend=gymnasium \
  logger.test_interval=10
""",
    "ddpg": """python sota-implementations/ddpg/ddpg.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  optim.batch_size=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=2 \
  optim.utd_ratio=1 \
  replay_buffer.size=120 \
  env.name=Pendulum-v1 \
  logger.backend=
""",
    "a2c_mujoco": """python sota-implementations/a2c/a2c_mujoco.py \
  env.env_name=HalfCheetah-v4 \
  collector.total_frames=40 \
  collector.frames_per_batch=20 \
  loss.mini_batch_size=10 \
  logger.backend= \
  logger.test_interval=40
""",
    "a2c_atari": """python sota-implementations/a2c/a2c_atari.py \
  collector.total_frames=80 \
  collector.frames_per_batch=20 \
  loss.mini_batch_size=20 \
  logger.backend= \
  env.backend=gymnasium \
  logger.test_interval=40
""",
    "dqn_atari": """python sota-implementations/dqn/dqn_atari.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  buffer.batch_size=10 \
  loss.num_updates=1 \
  logger.backend= \
  env.backend=gymnasium \
  buffer.buffer_size=120
""",
    "discrete_cql_online": """python sota-implementations/cql/discrete_cql_online.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  optim.batch_size=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=2 \
  replay_buffer.size=120 \
  logger.backend=
""",
    "discrete_cql_offline": """python sota-implementations/cql/discrete_cql_offline.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=2 \
  replay_buffer.batch_size=10 \
  logger.backend=
""",
    "redq": """python sota-implementations/redq/redq.py \
  num_workers=4 \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=2 \
  buffer.batch_size=10 \
  optim.steps_per_batch=1 \
  logger.video=True \
  logger.record_frames=4 \
  buffer.size=120 \
  logger.backend=
""",
    "sac": """python sota-implementations/sac/sac.py \
  env.max_episode_steps=20 \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=1 \
  optim.batch_size=10 \
  optim.utd_ratio=1 \
  replay_buffer.size=120 \
  env.name=Pendulum-v1 \
  logger.backend=csv \
  logger.video=False \
  logger.eval_iter=16 \
  hydra.run.dir=$SOTA_LOG_DIR/sac
""",
    "tqc": """python sota-implementations/tqc/tqc.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=2 \
  collector.device= \
  optim.batch_size=10 \
  optim.utd_ratio=1 \
  replay_buffer.size=120 \
  env.name=Pendulum-v1 \
  network.device= \
  logger.backend=
""",
    "discrete_sac": """python sota-implementations/discrete_sac/discrete_sac.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=1 \
  optim.batch_size=10 \
  optim.utd_ratio=1 \
  optim.batch_size=10 \
  optim.utd_ratio=1 \
  replay_buffer.size=120 \
  env.name=CartPole-v1 \
  logger.backend=
""",
    "crossq": """python sota-implementations/crossq/crossq.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=2 \
  collector.device= \
  optim.batch_size=10 \
  optim.utd_ratio=1 \
  replay_buffer.size=120 \
  env.name=Pendulum-v1 \
  network.device= \
  logger.backend=
""",
    "td3": """python sota-implementations/td3/td3.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  optim.batch_size=10 \
  collector.frames_per_batch=16 \
  collector.num_workers=4 \
  collector.env_per_collector=2 \
  logger.mode=offline \
  env.name=Pendulum-v1 \
  logger.backend=
""",
    "iql_online": """python sota-implementations/iql/iql_online.py \
  collector.total_frames=48 \
  optim.batch_size=10 \
  collector.frames_per_batch=16 \
  env.train_num_envs=2 \
  logger.mode=offline \
  logger.backend=
""",
    "discrete_iql": """python sota-implementations/iql/discrete_iql.py \
  collector.total_frames=48 \
  optim.batch_size=10 \
  collector.frames_per_batch=16 \
  env.train_num_envs=2 \
  logger.mode=offline \
  logger.backend=
""",
    "cql_online": """python sota-implementations/cql/cql_online.py \
  collector.total_frames=48 \
  optim.batch_size=10 \
  collector.frames_per_batch=16 \
  env.train_num_envs=2 \
  logger.mode=offline \
  logger.backend=
""",
    "gail": """python sota-implementations/gail/gail.py \
  ppo.collector.total_frames=48 \
  replay_buffer.batch_size=16 \
  ppo.loss.mini_batch_size=10 \
  ppo.collector.frames_per_batch=16 \
  logger.mode=offline \
  logger.backend=
""",
    "ddpg-single": """python sota-implementations/ddpg/ddpg.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  optim.batch_size=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=1 \
  optim.utd_ratio=1 \
  replay_buffer.size=120 \
  env.name=Pendulum-v1 \
  logger.backend=
""",
    "redq-single": """python sota-implementations/redq/redq.py \
  num_workers=2 \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=1 \
  buffer.batch_size=10 \
  optim.steps_per_batch=1 \
  logger.video=True \
  logger.record_frames=4 \
  buffer.size=120 \
  logger.backend=
""",
    "iql_online-single": """python sota-implementations/iql/iql_online.py \
  collector.total_frames=48 \
  optim.batch_size=10 \
  collector.frames_per_batch=16 \
  env.train_num_envs=1 \
  logger.mode=offline \
  logger.backend=
""",
    "cql_online-single": """python sota-implementations/cql/cql_online.py \
  collector.total_frames=48 \
  optim.batch_size=10 \
  collector.frames_per_batch=16 \
  collector.env_per_collector=1 \
  logger.mode=offline \
  logger.backend=
""",
    "td3-single": """python sota-implementations/td3/td3.py \
  collector.total_frames=48 \
  collector.init_random_frames=10 \
  collector.frames_per_batch=16 \
  collector.num_workers=2 \
  collector.env_per_collector=1 \
  logger.mode=offline \
  optim.batch_size=10 \
  env.name=Pendulum-v1 \
  logger.backend=
""",
    "mappo_ippo": """python sota-implementations/multiagent/mappo_ippo.py \
  collector.n_iters=2 \
  collector.frames_per_batch=200 \
  train.num_epochs=3 \
  train.minibatch_size=100 \
  logger.backend=
""",
    "maddpg_iddpg": """python sota-implementations/multiagent/maddpg_iddpg.py \
  collector.n_iters=2 \
  collector.frames_per_batch=200 \
  train.num_epochs=3 \
  train.minibatch_size=100 \
  logger.backend=
""",
    "iql_marl": """python sota-implementations/multiagent/iql.py \
  collector.n_iters=2 \
  collector.frames_per_batch=200 \
  train.num_epochs=3 \
  train.minibatch_size=100 \
  logger.backend=
""",
    "qmix_vdn": """python sota-implementations/multiagent/qmix_vdn.py \
  collector.n_iters=2 \
  collector.frames_per_batch=200 \
  train.num_epochs=3 \
  train.minibatch_size=100 \
  logger.backend=
""",
    "coma": """python sota-implementations/multiagent/coma.py \
  collector.n_iters=2 \
  collector.frames_per_batch=200 \
  train.num_epochs=3 \
  train.minibatch_size=100 \
  logger.backend=
""",
    "marl_sac": """python sota-implementations/multiagent/sac.py \
  collector.n_iters=2 \
  collector.frames_per_batch=200 \
  train.num_epochs=3 \
  train.minibatch_size=100 \
  logger.backend=
""",
    "bandits": """python sota-implementations/bandits/dqn.py --n_steps=100 --dataset=synthetic
""",
    "dreamer": """python sota-implementations/dreamer/dreamer.py \
  optimization.total_optim_steps=2 \
  optimization.log_every=1 \
  optimization.compile.enabled=False \
  collector.init_random_frames=32 \
  collector.frames_per_batch=200 \
  collector.num_collectors=1 \
  env.n_parallel_envs=1 \
  logger.eval_every=1000000 \
  logger.video=False \
  logger.backend=csv \
  replay_buffer.buffer_size=120 \
  replay_buffer.batch_size=8 \
  replay_buffer.batch_length=8 \
  replay_buffer.prefetch=1 \
  networks.rssm_hidden_dim=17
""",
    "dreamer_v3": """python sota-implementations/dreamer_v3/train.py \
  collector.total_frames=400 \
  collector.frames_per_batch=200 \
  replay_buffer.batch_size=2 \
  replay_buffer.seq_len=4 \
  replay_buffer.warmup_factor=1 \
  optimization.updates_per_batch=1 \
  optimization.compile=off \
  logger.eval_every=200 \
  logger.eval_episodes=1 \
  logger.output_plot= \
  networks.hidden_dim=8 \
  networks.encoder_layers=1 \
  networks.decoder_layers=1 \
  networks.reward_layers=1 \
  networks.actor_layers=1 \
  networks.value_layers=1 \
  networks.num_categoricals=2 \
  networks.num_classes=2 \
  networks.num_reward_bins=11 \
  networks.num_value_bins=11 \
  networks.rnn_hidden_dim=8
""",
}

_OFFLINE_DATASETS = {
    "gail": "halfcheetah-expert-v2",
    "td3_bc": "halfcheetah-medium-v2",
}


def _write_synthetic_d4rl_dataset(root: Path, dataset_id: str) -> None:
    generator = torch.Generator().manual_seed(0)
    size = 512
    done = torch.zeros(size, 1, dtype=torch.bool)
    truncated = torch.zeros_like(done)
    dataset = TensorDict(
        {
            "observation": torch.randn(size, 17, generator=generator),
            "action": torch.randn(size, 6, generator=generator).tanh(),
            "reward": torch.randn(size, 1, generator=generator),
            "done": done,
            "terminated": done.clone(),
            "truncated": truncated,
            "next": {
                "observation": torch.randn(size, 17, generator=generator),
                "reward": torch.randn(size, 1, generator=generator),
                "done": done.clone(),
                "terminated": done.clone(),
                "truncated": truncated.clone(),
            },
        },
        [size],
    )
    dataset.memmap_(root / ".cache" / "torchrl" / "d4rl" / dataset_id)


# CI sharding: the smoke list runs as SOTA_NUM_SHARDS parallel jobs, each
# selecting an interleaved slice of the sorted command list via SOTA_SHARD
# (1-based). Interleaving keeps the heavy neighbors (dreamer/dreamer_v3) on
# different shards. Both variables unset (the local default) runs everything.
_num_shards = int(os.environ.get("SOTA_NUM_SHARDS", "1"))
_shard = os.environ.get("SOTA_SHARD")
if _num_shards > 1:
    if _shard is None:
        raise RuntimeError("SOTA_NUM_SHARDS is set but SOTA_SHARD is not.")
    _shard_index = int(_shard) - 1
    if not 0 <= _shard_index < _num_shards:
        raise RuntimeError(
            f"SOTA_SHARD={_shard} is out of range for SOTA_NUM_SHARDS={_num_shards}."
        )
    commands = {
        algo: command
        for index, (algo, command) in enumerate(sorted(commands.items()))
        if index % _num_shards == _shard_index
    }


def run_command(command):
    # Get the current coverage settings
    cov_settings = os.environ.get("COVERAGE_PROCESS_START")
    if cov_settings:
        # If coverage is enabled, run the command with coverage
        command = f"coverage run --parallel-mode {command}"
    process = subprocess.Popen(
        command,
        shell=True,
        cwd=Path(__file__).parent.parent.parent.parent.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    while True:
        output = process.stdout.readline()
        if output == "" and process.poll() is not None:
            break
        if output:
            print(output.strip())  # noqa: T201
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


@pytest.mark.parametrize("algo", list(commands))
def test_commands(algo, monkeypatch, tmp_path):
    dataset_id = _OFFLINE_DATASETS.get(algo)
    if dataset_id is not None:
        monkeypatch.setenv("HOME", str(tmp_path))
        _write_synthetic_d4rl_dataset(tmp_path, dataset_id)
    if algo in {"ppo_mujoco", "sac", "fql"}:
        monkeypatch.setenv("SOTA_LOG_DIR", str(tmp_path))
    run_command(commands[algo])
    if algo in {"ppo_mujoco", "sac", "fql"}:
        scalar_roots = list(tmp_path.rglob("scalars"))
        assert len(scalar_roots) == 1
        scalar_names = {
            path.relative_to(scalar_roots[0]).as_posix()
            for path in scalar_roots[0].rglob("*.csv")
        }
        expected_training_metric = {
            "ppo_mujoco": "training/loss_objective.csv",
            "sac": "training/q_loss.csv",
            "fql": "training/loss_actor.csv",
        }[algo]
        assert expected_training_metric in scalar_names
        expected_evaluation_metric = "return" if algo == "fql" else "reward"
        assert f"evaluation/{expected_evaluation_metric}.csv" in scalar_names
        assert scalar_names
        assert all(
            name.startswith(("training/", "evaluation/", "timing/"))
            for name in scalar_names
        )


@pytest.fixture
def fql_recipe():
    directory = Path(__file__).resolve().parents[4] / "sota-implementations" / "fql"
    spec = importlib.util.spec_from_file_location(
        "fql_recipe_utils", directory / "utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return directory, module


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_fql_evaluation(fql_recipe, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    directory, recipe = fql_recipe
    cfg = OmegaConf.load(directory / "config.yaml")
    cfg.device = device
    cfg.dataset.name = None
    cfg.dataset.random_frames = 8
    cfg.env.max_episode_steps = 4
    cfg.evaluation.episodes = 2
    cfg.network.width = 8
    cfg.network.depth = 1
    _, env, eval_env = recipe.make_data_and_envs(cfg)
    policy, _, _ = recipe.make_agent(cfg, env, torch.device(device))
    try:
        eval_env.set_seed(12)
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state() if device == "cuda" else None
        expected = torch.stack(
            [
                eval_env.rollout(4, policy, auto_cast_to_device=True)[
                    "next", "reward"
                ].sum()
                for _ in range(2)
            ]
        ).mean()
        eval_env.set_seed(12)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)
        metrics = recipe.evaluate(policy, eval_env, cfg)
        torch.testing.assert_close(metrics["evaluation/return"], expected)
        assert torch.equal(torch.get_rng_state(), cpu_state)
        if cuda_state is not None:
            assert torch.equal(torch.cuda.get_rng_state(), cuda_state)
        stop = SimpleNamespace(requested=False)

        def request_stop(module, args, result):
            stop.requested = True

        handle = policy.register_forward_hook(request_stop)
        try:
            assert not recipe.evaluate(policy, eval_env, cfg, stop=stop).keys()
        finally:
            handle.remove()
    finally:
        env.close()
        eval_env.close()


def fql_command(directory, output, *overrides):
    return [
        sys.executable,
        str(directory / "fql.py"),
        "device=cpu",
        "dataset.name=null",
        "dataset.random_frames=16",
        "network.width=8",
        "network.depth=1",
        "optim.batch_size=4",
        "optim.offline_steps=3",
        "evaluation.interval=3",
        "evaluation.episodes=1",
        "env.max_episode_steps=2",
        "log_interval=1",
        f"hydra.run.dir={output}",
        *overrides,
    ]


def test_fql_checkpoint_resume(fql_recipe, tmp_path):
    directory, _ = fql_recipe
    uninterrupted, resumed = tmp_path / "full", tmp_path / "resumed"
    for output, overrides in (
        (uninterrupted, ()),
        (resumed, ("optim.offline_steps=1",)),
        (resumed, (f"resume={resumed / 'checkpoints'}",)),
    ):
        result = subprocess.run(
            fql_command(directory, output, *overrides),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    full_path = resolve_checkpoint_path(uninterrupted / "checkpoints")
    resumed_path = resolve_checkpoint_path(resumed / "checkpoints")
    for path in (full_path, resumed_path):
        assert Checkpoint.manifest(path)["metadata"]["optim_steps"] == 3
        optimizer = Checkpoint.read_component(path, "optimizer")
        assert {int(state["step"]) for state in optimizer["state"].values()} == {3}
    full_weights = TensorDict(Checkpoint.read_component(full_path, "loss_module"), [])
    resumed_weights = TensorDict(
        Checkpoint.read_component(resumed_path, "loss_module"), []
    )
    torch.testing.assert_close(full_weights, resumed_weights, rtol=0, atol=0)
    steps = [
        int(line.split(",")[0])
        for line in (resumed / "fql/scalars/training/loss_actor.csv")
        .read_text()
        .splitlines()
    ]
    assert steps == [1, 2, 3]


def test_fql_interruption_checkpoint(fql_recipe, tmp_path):
    directory, _ = fql_recipe
    output = tmp_path / "interrupted"
    with (tmp_path / "recipe.log").open("w") as log:
        process = subprocess.Popen(
            fql_command(directory, output, "optim.offline_steps=100000"),
            stdout=log,
            stderr=log,
        )
        try:
            scalar = output / "fql/scalars/training/loss_actor.csv"
            deadline = time.monotonic() + 30
            while not scalar.exists():
                assert process.poll() is None, (tmp_path / "recipe.log").read_text()
                assert time.monotonic() < deadline, "Training produced no scalar logs"
                time.sleep(0.05)
            process.terminate()
            assert process.wait(timeout=30) == 130, (
                tmp_path / "recipe.log"
            ).read_text()
            checkpoint = resolve_checkpoint_path(output / "checkpoints")
            step = Checkpoint.manifest(checkpoint)["metadata"]["optim_steps"]
            assert 0 < step < 100000
            optimizer = Checkpoint.read_component(checkpoint, "optimizer")
            assert {int(state["step"]) for state in optimizer["state"].values()} == {
                step
            }
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
