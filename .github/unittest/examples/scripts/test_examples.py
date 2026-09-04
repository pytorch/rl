# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).parents[4]
EXAMPLES_DIR = ROOT_DIR / "examples"

pytestmark = pytest.mark.gpu


@dataclass(frozen=True)
class WarningFilter:
    value: str
    reason: str


@dataclass(frozen=True)
class ExampleSpec:
    name: str
    path: str
    argv: tuple[str, ...] = ()
    cwd: str = "."
    timeout: int = 300
    env: dict[str, str] = field(default_factory=dict)
    warning_filters: tuple[WarningFilter, ...] = ()


RUNNABLE_EXAMPLES = (
    ExampleSpec("agent-composite-actor", "examples/agents/composite_actor.py"),
    ExampleSpec("agent-composite-ppo", "examples/agents/composite_ppo.py"),
    ExampleSpec("agent-multi-step", "examples/agents/multi-step.py"),
    ExampleSpec("agent-recurrent", "examples/agents/recurrent_actor.py"),
    ExampleSpec(
        "collector-async-batched",
        "examples/collectors/async_batched_collector.py",
    ),
    ExampleSpec(
        "collector-device",
        "examples/collectors/collector_device.py",
        ("--smoke",),
    ),
    ExampleSpec(
        "collector-weight-sync",
        "examples/collectors/weight_sync_collectors.py",
        ("--smoke",),
    ),
    ExampleSpec(
        "distributed-generic",
        "examples/distributed/collectors/single_machine/generic.py",
        (
            "--smoke",
            "--num_workers=1",
            "--num_nodes=1",
            "--frames_per_batch=16",
            "--total_frames=16",
            "--backend=gloo",
            "--env=CartPole-v1",
        ),
        env={"CUDA_VISIBLE_DEVICES": ""},
    ),
    ExampleSpec(
        "distributed-rpc",
        "examples/distributed/collectors/single_machine/rpc.py",
        (
            "--smoke",
            "--num_workers=1",
            "--num_nodes=1",
            "--frames_per_batch=16",
            "--total_frames=16",
            "--env=CartPole-v1",
        ),
        env={"CUDA_VISIBLE_DEVICES": ""},
    ),
    ExampleSpec(
        "distributed-sync",
        "examples/distributed/collectors/single_machine/sync.py",
        (
            "--smoke",
            "--num_workers=1",
            "--num_nodes=1",
            "--frames_per_batch=16",
            "--total_frames=16",
            "--backend=gloo",
            "--env=CartPole-v1",
        ),
        env={"CUDA_VISIBLE_DEVICES": ""},
    ),
    ExampleSpec(
        "distributed-ray-dqn",
        "examples/distributed/ray_dqn_trainer.py",
    ),
    ExampleSpec(
        "env-compile-step-reset",
        "examples/envs/benchmark_compile_step_and_maybe_reset.py",
        (
            "--batch-size=8",
            "--steps=2",
            "--repeats=1",
            "--warmup=1",
            "--compile-warmup=1",
            "--device=cpu",
            "--compile-modes=default",
        ),
    ),
    ExampleSpec("env-async-info", "examples/envs/gym-async-info-reader.py"),
    ExampleSpec(
        "env-async-info-wrapper",
        "examples/envs/gym-async-info-reader.py",
        ("--use_wrapper",),
    ),
    ExampleSpec("env-gym-conversion", "examples/envs/gym_conversion_examples.py"),
    ExampleSpec("llm-tool-service", "examples/llm/tool_service_example.py"),
    ExampleSpec(
        "mujoco-cube-bowl",
        "examples/mujoco_macros/cube_bowl_macros.py",
        ("--smoke", "--max-rollouts=1"),
        cwd="examples/mujoco_macros",
    ),
    ExampleSpec(
        "mujoco-humanoid",
        "examples/mujoco_macros/humanoid_macros.py",
        ("--smoke", "--max-rollouts=1"),
        cwd="examples/mujoco_macros",
    ),
    ExampleSpec(
        "mujoco-satellite",
        "examples/mujoco_macros/satellite_macros.py",
        ("--smoke", "--max-rollouts=1", "--max-macros=2", "--fixed-attitudes"),
        cwd="examples/mujoco_macros",
    ),
    ExampleSpec(
        "multiagent-mappo",
        "examples/multiagent/mappo_vmas.py",
        (
            "--algo=mappo",
            "--frames=16",
            "--frames_per_batch=16",
            "--minibatch_size=8",
            "--max_steps=8",
            "--epochs=1",
        ),
    ),
    ExampleSpec(
        "multiagent-ippo",
        "examples/multiagent/mappo_vmas.py",
        (
            "--algo=ippo",
            "--frames=16",
            "--frames_per_batch=16",
            "--minibatch_size=8",
            "--max_steps=8",
            "--epochs=1",
        ),
    ),
    ExampleSpec(
        "replay-buffer-catframes",
        "examples/replay-buffers/catframes-in-buffer.py",
    ),
    ExampleSpec(
        "replay-buffer-checkpoint",
        "examples/replay-buffers/checkpoint.py",
    ),
    ExampleSpec(
        "replay-buffer-filter-trajectories",
        "examples/replay-buffers/filter-imcomplete-trajs.py",
    ),
    ExampleSpec(
        "replay-buffer-recurrent-slices",
        "examples/replay-buffers/recurrent_slice_sampler_pipeline.py",
    ),
    ExampleSpec(
        "rlhf-transformer",
        "examples/rlhf/train.py",
        (
            "smoke=true",
            "sys.device=cpu",
            "sys.dtype=float32",
            "sys.compile=false",
            "data.batch_size=2",
            "data.block_size=8",
            "train.max_iters=1",
            "train.gradient_accumulation_steps=1",
            "train.decay_lr=false",
            "io.eval_interval=2",
            "io.log_interval=1",
            "hydra.run.dir={tmp_path}",
        ),
        cwd="examples/rlhf",
    ),
    ExampleSpec(
        "rlhf-reward",
        "examples/rlhf/train_reward.py",
        (
            "smoke=true",
            "sys.device=cpu",
            "sys.dtype=float32",
            "sys.compile=false",
            "data.batch_size=2",
            "data.block_size=8",
            "train.max_iters=1",
            "train.decay_lr=false",
            "io.eval_interval=2",
            "io.log_interval=1",
            "hydra.run.dir={tmp_path}",
        ),
        cwd="examples/rlhf",
    ),
    ExampleSpec(
        "rlhf-ppo",
        "examples/rlhf/train_rlhf.py",
        (
            "smoke=true",
            "sys.device=cpu",
            "sys.ref_device=cpu",
            "sys.dtype=float32",
            "sys.compile=false",
            "data.batch_size=2",
            "data.block_size=8",
            "train.max_epochs=1",
            "train.decay_lr=false",
            "train.ppo.episode_length=2",
            "train.ppo.ppo_batch_size=2",
            "train.ppo.ppo_num_epochs=1",
            "train.ppo.num_rollouts_per_epoch=2",
            "io.logger=csv",
            "io.eval_interval=2",
            "hydra.run.dir={tmp_path}",
        ),
        cwd="examples/rlhf",
    ),
    ExampleSpec(
        "satellite-sac",
        "examples/satellite/sac_per.py",
        (
            "--num-envs=4",
            "--max-steps=4",
            "--frame-skip=1",
            "--frames-per-env=4",
            "--total-iters=1",
            "--buffer-size=32",
            "--batch-size=4",
            "--no-prb",
            "--gradient-steps=1",
            "--init-random-frames-per-env=0",
            "--device=cpu",
            "--buffer-device=cpu",
            "--hidden",
            "16",
            "--no-obs-norm",
            "--no-eval",
            "--no-wandb",
        ),
    ),
    ExampleSpec(
        "satellite-sac-async",
        "examples/satellite/sac_per_fully_async.py",
        (
            "--num-envs=4",
            "--max-steps=4",
            "--frame-skip=1",
            "--frames-per-env=4",
            "--total-iters=1",
            "--buffer-size=32",
            "--batch-size=4",
            "--no-prb",
            "--gradient-steps=1",
            "--min-random-horizon=1",
            "--init-random-frames-per-env=0",
            "--init-buffer-size=4",
            "--device=cpu",
            "--buffer-device=cpu",
            "--hidden",
            "16",
            "--no-obs-norm",
            "--no-eval",
            "--no-wandb",
        ),
    ),
    ExampleSpec(
        "services-distributed",
        "examples/services/distributed_services.py",
    ),
    ExampleSpec(
        "services-multiprocess",
        "examples/services/multi_service_multiprocess.py",
        ("--steps=2", "--batch-size=2", "--log-dir={tmp_path}/services-mp"),
    ),
    ExampleSpec(
        "services-ray",
        "examples/services/multi_service_ray.py",
        ("--steps=2", "--batch-size=2", "--log-dir={tmp_path}/services-ray"),
    ),
    ExampleSpec(
        "services-single-process",
        "examples/services/multi_service_single_process.py",
        ("--steps=2", "--batch-size=2", "--log-dir={tmp_path}/services-local"),
    ),
    ExampleSpec(
        "services-ray-collector",
        "examples/services/ray_collector_services.py",
    ),
)


EXCLUDED_EXAMPLES = {
    "examples/collectors/isaaclab_rnn_ppo_memory.py": "requires Isaac Lab",
    "examples/collectors/isaaclab_rnn_ppo_memory_utils.py": "Isaac Lab helper module",
    "examples/collectors/multi_weight_updates.py": "requires unsupported nested process and Ray transform synchronization",
    "examples/collectors/mp_collector_mps.py": "requires macOS MPS",
    "examples/collectors/profile_isaaclab_collector.py": "requires Isaac Lab",
    "examples/collectors/profile_mujoco_playground_collector.py": "requires MuJoCo Playground",
    "examples/distributed/collectors/multi_nodes/delayed_dist.py": "requires a multi-node scheduler",
    "examples/distributed/collectors/multi_nodes/delayed_rpc.py": "requires a multi-node scheduler",
    "examples/distributed/collectors/multi_nodes/generic.py": "requires Slurm",
    "examples/distributed/collectors/multi_nodes/ray_buffer_infra.py": "multi-node helper module",
    "examples/distributed/collectors/multi_nodes/ray_collect.py": "requires a Ray cluster",
    "examples/distributed/collectors/multi_nodes/ray_train.py": "requires a Ray cluster",
    "examples/distributed/collectors/multi_nodes/rpc.py": "requires Slurm",
    "examples/distributed/collectors/multi_nodes/sync.py": "requires Slurm",
    "examples/distributed/replay_buffers/distributed_replay_buffer.py": "interactive manual multi-rank example",
    "examples/distributed/replay_buffers/ray_buffer.py": "downloads a large model and dataset",
    "examples/llm/python_mcp_tool.py": "support module without a standalone entry point",
    "examples/llm/web_search_tool.py": "requires live external web services",
    "examples/memmap/memmap_speed_distributed.py": "manual multi-rank benchmark",
    "examples/memmap/memmap_td_distributed.py": "manual multi-rank benchmark",
    "examples/microduck/ppo_mjlab.py": "requires MicroDuck, MJLab, and CUDA",
    "examples/microduck/heuristic_gait.py": "requires the external MicroDuck MJCF assets",
    "examples/microduck/ppo_mujoco.py": "requires the external MicroDuck MJCF assets",
    "examples/mujoco_macros/_viewer.py": "viewer helper module",
    "examples/replay-buffers/compressed_replay_buffer.py": "requires nvCOMP and Atari assets",
    "examples/replay-buffers/compressed_replay_buffer_checkpoint.py": "requires nvCOMP",
    "examples/rlhf/_smoke.py": "shared RLHF smoke-data helper",
    "examples/rlhf/models/actor_critic.py": "RLHF model helper module",
    "examples/rlhf/models/reward.py": "RLHF model helper module",
    "examples/rlhf/models/transformer.py": "RLHF model helper module",
    "examples/rlhf/utils.py": "RLHF helper module",
    "examples/satellite/_utils.py": "satellite example helper module",
    "examples/services/multi_service_utils.py": "service example helper module",
    "examples/video/video-from-dataset.py": "requires the OpenX dataset",
}


class ExampleRunError(RuntimeError):
    pass


def validate_manifest(
    discovered: set[str],
    runnable: tuple[ExampleSpec, ...],
    excluded: dict[str, str],
) -> list[str]:
    errors = []
    runnable_paths = [spec.path for spec in runnable]
    runnable_names = [spec.name for spec in runnable]
    duplicate_commands = sorted(
        command
        for command, count in Counter(
            (spec.path, spec.argv, spec.cwd) for spec in runnable
        ).items()
        if count > 1
    )
    if duplicate_commands:
        errors.append(f"duplicate runnable commands: {duplicate_commands}")
    duplicate_names = sorted(
        name for name, count in Counter(runnable_names).items() if count > 1
    )
    if duplicate_names:
        errors.append(f"duplicate runnable names: {duplicate_names}")
    missing_commands = sorted(
        spec.name or "<unnamed>" for spec in runnable if not spec.path.strip()
    )
    if missing_commands:
        errors.append(f"runnable entries without commands: {missing_commands}")
    undocumented_warning_filters = sorted(
        f"{spec.name}: {warning_filter.value}"
        for spec in runnable
        for warning_filter in spec.warning_filters
        if not warning_filter.reason.strip()
    )
    if undocumented_warning_filters:
        errors.append(
            "warning filters without reasons: " f"{undocumented_warning_filters}"
        )
    warning_env_overrides = sorted(
        spec.name for spec in runnable if "PYTHONWARNINGS" in spec.env
    )
    if warning_env_overrides:
        errors.append(
            "PYTHONWARNINGS overrides must use warning_filters: "
            f"{warning_env_overrides}"
        )
    runnable_set = set(runnable_paths)
    overlap = runnable_set & excluded.keys()
    if overlap:
        errors.append(f"both runnable and excluded: {sorted(overlap)}")
    missing_reasons = sorted(
        path for path, reason in excluded.items() if not reason.strip()
    )
    if missing_reasons:
        errors.append(f"exclusions without reasons: {missing_reasons}")
    classified = runnable_set | excluded.keys()
    unclassified = discovered - classified
    if unclassified:
        errors.append(f"unclassified examples: {sorted(unclassified)}")
    stale = classified - discovered
    if stale:
        errors.append(f"manifest entries without files: {sorted(stale)}")
    return errors


def select_shard(
    specs: tuple[ExampleSpec, ...], num_shards: int, shard: int
) -> tuple[ExampleSpec, ...]:
    if num_shards < 1:
        raise ValueError("EXAMPLES_NUM_SHARDS must be at least 1")
    if not 1 <= shard <= num_shards:
        raise ValueError(f"EXAMPLES_SHARD={shard} is outside the range 1..{num_shards}")
    return tuple(
        spec
        for index, spec in enumerate(sorted(specs, key=lambda item: item.name))
        if index % num_shards == shard - 1
    )


def run_example(spec: ExampleSpec, tmp_path: Path) -> None:
    substitutions = {"tmp_path": str(tmp_path)}
    script = spec.path if spec.path.startswith("-") else str(ROOT_DIR / spec.path)
    argv = [script, *(arg.format(**substitutions) for arg in spec.argv)]
    command = [sys.executable, *argv]
    env = os.environ.copy()
    mujoco_gl = "egl" if sys.platform.startswith("linux") else "glfw"
    python_path = env.get("PYTHONPATH")
    env.update(
        {
            "MPLBACKEND": "Agg",
            "MUJOCO_GL": mujoco_gl,
            "PYOPENGL_PLATFORM": mujoco_gl,
            "PYTHONPATH": os.pathsep.join(
                item for item in (str(ROOT_DIR), python_path) if item
            ),
            "SDL_VIDEODRIVER": "dummy",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_MODE": "disabled",
            "PYTHONWARNINGS": ",".join(
                (
                    "error::FutureWarning",
                    *(warning_filter.value for warning_filter in spec.warning_filters),
                )
            ),
        }
    )
    env.update({key: value.format(**substitutions) for key, value in spec.env.items()})
    process = subprocess.Popen(
        command,
        cwd=ROOT_DIR / spec.cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=spec.timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        tail = "\n".join(output.splitlines()[-80:])
        raise ExampleRunError(
            f"{spec.name} timed out after {spec.timeout}s\n{tail}"
        ) from None
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        raise
    if process.returncode:
        tail = "\n".join(output.splitlines()[-80:])
        raise ExampleRunError(
            f"{spec.name} failed with exit code {process.returncode}\n{tail}"
        )


def test_manifest_covers_examples() -> None:
    discovered = {
        path.relative_to(ROOT_DIR).as_posix()
        for path in EXAMPLES_DIR.rglob("*.py")
        if path.name != "__init__.py"
    }
    errors = validate_manifest(discovered, RUNNABLE_EXAMPLES, EXCLUDED_EXAMPLES)
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize(
    ("discovered", "runnable", "excluded", "expected"),
    [
        ({"new.py"}, (), {}, "unclassified examples"),
        (
            set(),
            (ExampleSpec("gone", "gone.py"),),
            {},
            "manifest entries without files",
        ),
        (
            {"both.py"},
            (ExampleSpec("both", "both.py"),),
            {"both.py": "reason"},
            "both runnable and excluded",
        ),
        ({"skip.py"}, (), {"skip.py": ""}, "exclusions without reasons"),
        (
            {"duplicate.py"},
            (
                ExampleSpec("first", "duplicate.py"),
                ExampleSpec("second", "duplicate.py"),
            ),
            {},
            "duplicate runnable commands",
        ),
        (
            {"first.py", "second.py"},
            (
                ExampleSpec("duplicate", "first.py"),
                ExampleSpec("duplicate", "second.py"),
            ),
            {},
            "duplicate runnable names",
        ),
        (
            set(),
            (ExampleSpec("no-command", ""),),
            {},
            "runnable entries without commands",
        ),
        (
            {"warning.py"},
            (
                ExampleSpec(
                    "undocumented-warning",
                    "warning.py",
                    warning_filters=(WarningFilter("ignore::UserWarning", ""),),
                ),
            ),
            {},
            "warning filters without reasons",
        ),
        (
            {"warning-env.py"},
            (
                ExampleSpec(
                    "warning-env",
                    "warning-env.py",
                    env={"PYTHONWARNINGS": "ignore"},
                ),
            ),
            {},
            "PYTHONWARNINGS overrides must use warning_filters",
        ),
    ],
)
def test_manifest_rejects_invalid_classification(
    discovered: set[str],
    runnable: tuple[ExampleSpec, ...],
    excluded: dict[str, str],
    expected: str,
) -> None:
    assert any(
        expected in error for error in validate_manifest(discovered, runnable, excluded)
    )


def test_shards_are_complete_and_disjoint() -> None:
    first = select_shard(RUNNABLE_EXAMPLES, 2, 1)
    second = select_shard(RUNNABLE_EXAMPLES, 2, 2)
    first_names = {spec.name for spec in first}
    second_names = {spec.name for spec in second}
    assert not first_names & second_names
    assert first_names | second_names == {spec.name for spec in RUNNABLE_EXAMPLES}
    with pytest.raises(ValueError, match="outside the range"):
        select_shard(RUNNABLE_EXAMPLES, 2, 3)
    with pytest.raises(ValueError, match="at least 1"):
        select_shard(RUNNABLE_EXAMPLES, 0, 1)


@pytest.mark.parametrize(
    "code",
    (
        "raise SystemExit(2)",
        "import warnings; warnings.warn('deprecated', FutureWarning)",
    ),
)
def test_run_example_reports_process_failures(tmp_path: Path, code: str) -> None:
    spec = ExampleSpec("intentional-failure", "-c", (code,))
    with pytest.raises(ExampleRunError, match="failed with exit code"):
        run_example(spec, tmp_path)


def test_run_example_terminates_timeouts(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "child.pid"
    code = (
        "import pathlib, subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid)); "
        "time.sleep(30)"
    )
    spec = ExampleSpec(
        "intentional-timeout",
        "-c",
        (code,),
        timeout=1,
    )
    with pytest.raises(ExampleRunError, match="timed out"):
        run_example(spec, tmp_path)
    child_pid = int(child_pid_path.read_text())
    for _ in range(100):
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    else:
        pytest.fail(f"child process {child_pid} survived timeout cleanup")


_NUM_SHARDS = int(os.environ.get("EXAMPLES_NUM_SHARDS", "1"))
_SHARD = int(os.environ.get("EXAMPLES_SHARD", "1"))
SHARD_EXAMPLES = select_shard(RUNNABLE_EXAMPLES, _NUM_SHARDS, _SHARD)


@pytest.mark.parametrize("spec", SHARD_EXAMPLES, ids=lambda spec: spec.name)
def test_example(spec: ExampleSpec, tmp_path: Path) -> None:
    run_example(spec, tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
