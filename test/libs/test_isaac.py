# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import io
import itertools
import os
import queue as queue_lib
import time
import traceback
from functools import partial

import pytest
import torch
import torch.distributed as dist
import torchrl.testing.env_helper
from tensordict import assert_allclose_td
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq
from torch import multiprocessing as mp

from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import Collector, Evaluator
from torchrl.collectors.distributed import RayCollector
from torchrl.data import LazyMemmapStorage, RayReplayBuffer, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import InitTracker, StepCounter, TransformedEnv, VecNormV2
from torchrl.envs.utils import check_env_specs
from torchrl.modules import LSTMModule, MLP
from torchrl.testing import get_default_devices
from torchrl.testing.env_helper import (
    _isaac_app_launcher_init,
    make_isaac_env,
    make_isaac_policy,
)

_has_isaac = importlib.util.find_spec("isaacgym") is not None

if _has_isaac:
    import isaacgym  # noqa
    import isaacgymenvs  # noqa
    from torchrl.envs.libs.isaacgym import IsaacGymEnv

_has_isaaclab = importlib.util.find_spec("isaaclab") is not None
_has_ray = importlib.util.find_spec("ray") is not None

# Ray / process backends pickle the factory callables and import them by
# qualified name in the child / actor process.  Locally-defined helpers in
# this test module live at ``test_isaac.*`` which is not importable inside a
# fresh Ray actor.  We therefore bind arguments onto library-level functions
# (reachable from anywhere via ``torchrl.testing.env_helper``) using
# ``functools.partial``.
_isaac_env_maker = partial(make_isaac_env, init_app=False)
_isaac_env_maker_cuda1 = partial(
    make_isaac_env, init_app=False, device=torch.device("cuda:1")
)
# ``make_isaac_policy`` already accepts ``env=None`` for the probe path, so
# it can be used directly as a ``policy_factory``.
_isaac_policy_maker = make_isaac_policy
_isaac_policy_maker_cuda1 = partial(make_isaac_policy, device=torch.device("cuda:1"))


class CountingVecNormV2(VecNormV2):
    def __init__(self, *args, **kwargs):
        self.step_calls = 0
        self.reset_calls = 0
        super().__init__(*args, **kwargs)

    def _step(self, tensordict, next_tensordict):
        self.step_calls += 1
        return super()._step(tensordict, next_tensordict)

    def _reset(self, tensordict, tensordict_reset):
        self.reset_calls += 1
        return super()._reset(tensordict, tensordict_reset)


def _isaaclab_raw_env(env):
    while isinstance(env, TransformedEnv):
        env = env.base_env
    return env._env.unwrapped


def _force_isaaclab_next_step_done(env):
    raw_env = _isaaclab_raw_env(env)
    if not hasattr(raw_env, "episode_length_buf") or not hasattr(
        raw_env, "max_episode_length"
    ):
        pytest.skip("Isaac Lab env does not expose episode length buffers.")
    raw_env.episode_length_buf.zero_()
    raw_env.episode_length_buf.reshape(-1)[0] = raw_env.max_episode_length - 1


def _isaaclab_rollout_keys(rollout):
    keys = [
        "action",
        "policy",
        "done",
        "terminated",
        "truncated",
        ("next", "policy"),
        ("next", "reward"),
        ("next", "done"),
        ("next", "terminated"),
        ("next", "truncated"),
    ]
    return [key for key in keys if key in rollout.keys(True, True)]


def _isaaclab_rollout_in_process(
    seed: int, *, native_autoreset: bool, max_steps: int = 4
):
    counter = itertools.count()

    def seeded_random_policy(tensordict):
        torch.manual_seed(seed + next(counter))
        return env.full_action_spec.rand_update(tensordict)

    os.environ["ACCEPT_EULA"] = "Y"
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    os.environ["PRIVACY_CONSENT"] = "Y"
    env = make_isaac_env(native_autoreset=native_autoreset)
    try:
        env.set_seed(seed, static_seed=True)
        torch.manual_seed(seed)
        tensordict = env.reset()
        _force_isaaclab_next_step_done(env)
        rollout = env.rollout(
            max_steps,
            policy=seeded_random_policy,
            auto_reset=False,
            tensordict=tensordict,
            break_when_any_done=False,
            return_contiguous=True,
        )
        return rollout.select(*_isaaclab_rollout_keys(rollout)).cpu()
    finally:
        env.close()


def _isaaclab_rollout_worker(queue, seed: int, native_autoreset: bool, max_steps: int):
    try:
        buffer = io.BytesIO()
        torch.save(
            _isaaclab_rollout_in_process(
                seed, native_autoreset=native_autoreset, max_steps=max_steps
            ),
            buffer,
        )
        queue.put(
            (
                "succeeded",
                buffer.getvalue(),
            )
        )
    except BaseException:
        queue.put(("failed", traceback.format_exc()))
        raise


def _isaaclab_rollout(seed: int, *, native_autoreset: bool, max_steps: int = 4):
    queue = mp.Queue(1)
    proc = mp.Process(
        target=_isaaclab_rollout_worker,
        args=(queue, seed, native_autoreset, max_steps),
    )
    try:
        proc.start()
        deadline = time.monotonic() + 300
        while True:
            try:
                status, result = queue.get(timeout=1)
                break
            except queue_lib.Empty:
                if not proc.is_alive():
                    proc.join()
                    pytest.fail(
                        f"Isaac Lab rollout worker exited with code {proc.exitcode} "
                        "without returning a rollout."
                    )
                if time.monotonic() > deadline:
                    proc.terminate()
                    proc.join(timeout=30)
                    pytest.fail("Isaac Lab rollout worker timed out.")
        proc.join(timeout=30)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=30)
            pytest.fail("Isaac Lab rollout worker did not exit after returning.")
        if status != "succeeded":
            pytest.fail(f"Isaac Lab rollout worker failed:\n{result}")
        if proc.exitcode:
            pytest.fail(f"Isaac Lab rollout worker exited with code {proc.exitcode}.")
        return torch.load(io.BytesIO(result), weights_only=False)
    finally:
        queue.close()
        proc.join()


def _isaaclab_native_rollout(seed: int, max_steps: int = 4):
    return _isaaclab_rollout(seed, native_autoreset=True, max_steps=max_steps)


def _isaaclab_transition_keys(rollout):
    return _isaaclab_rollout_keys(rollout)


@pytest.mark.skipif(not _has_isaac, reason="IsaacGym not found")
@pytest.mark.parametrize(
    "task",
    [
        "AllegroHand",
        # "AllegroKuka",
        # "AllegroKukaTwoArms",
        # "AllegroHandManualDR",
        # "AllegroHandADR",
        "Ant",
        # "Anymal",
        # "AnymalTerrain",
        # "BallBalance",
        # "Cartpole",
        # "FactoryTaskGears",
        # "FactoryTaskInsertion",
        # "FactoryTaskNutBoltPick",
        # "FactoryTaskNutBoltPlace",
        # "FactoryTaskNutBoltScrew",
        # "FrankaCabinet",
        # "FrankaCubeStack",
        "Humanoid",
        # "HumanoidAMP",
        # "Ingenuity",
        # "Quadcopter",
        # "ShadowHand",
        "Trifinger",
    ],
)
@pytest.mark.parametrize("num_envs", [10, 20])
@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("from_pixels", [False])
class TestIsaacGym:
    @classmethod
    def _run_on_proc(cls, q, task, num_envs, device, from_pixels):
        try:
            env = IsaacGymEnv(
                task=task, num_envs=num_envs, device=device, from_pixels=from_pixels
            )
            check_env_specs(env)
            q.put(("succeeded!", None))
        except Exception as err:
            q.put(("failed!", err))
            raise err

    def test_env(self, task, num_envs, device, from_pixels):
        q = mp.Queue(1)
        self._run_on_proc(q, task, num_envs, device, from_pixels)
        proc = mp.Process(
            target=self._run_on_proc, args=(q, task, num_envs, device, from_pixels)
        )
        try:
            proc.start()
            msg, error = q.get()
            if msg != "succeeded!":
                raise error
        finally:
            q.close()
            proc.join()

    #
    # def test_collector(self, task, num_envs, device):
    #     env = IsaacGymEnv(task=task, num_envs=num_envs, device=device)
    #     collector = Collector(
    #         env,
    #         policy=SafeModule(nn.LazyLinear(out_features=env.observation_spec['obs'].shape[-1]), in_keys=["obs"], out_keys=["action"]),
    #         frames_per_batch=20,
    #         total_frames=-1
    #     )
    #     for c in collector:
    #         assert c.shape == torch.Size([num_envs, 20])
    #         break


@pytest.mark.skipif(not _has_isaaclab, reason="Isaaclab not found")
class TestIsaacLab:
    @pytest.fixture(scope="class")
    def env(self):
        env = torchrl.testing.env_helper.make_isaac_env()
        try:
            yield env
        finally:
            torchrl_logger.info("Closing IsaacLab env...")
            env.close()
            torchrl_logger.info("Closed")

    def test_isaaclab(self, env):
        assert env.batch_size == (4096,)
        assert env._is_batched
        torchrl_logger.info("Checking env specs...")
        env.check_env_specs(break_when_any_done="both")
        torchrl_logger.info("Check succeeded!")

    def test_isaaclab_rb(self, env):
        env = env.append_transform(StepCounter())
        rb = ReplayBuffer(
            storage=LazyTensorStorage(100_000, ndim=2),
            sampler=SliceSampler(num_slices=5),
            batch_size=20,
        )
        r = env.rollout(20, break_when_any_done=False)
        rb.extend(r)
        # check that rb["step_count"].flatten() is made of sequences of 4 consecutive numbers
        flat_ranges = rb.sample()["step_count"]
        flat_ranges = flat_ranges.view(-1, 4)
        flat_ranges = flat_ranges - flat_ranges[:, :1]  # subtract baseline
        flat_ranges = flat_ranges.flatten()
        arange = torch.arange(flat_ranges.numel(), device=flat_ranges.device) % 4
        assert (flat_ranges == arange).all()

    def test_isaac_collector(self, env):
        col = Collector(
            env, env.rand_action, frames_per_batch=1000, total_frames=100_000_000
        )
        try:
            for data in col:
                assert data.shape == (4096, 1)
                break
        finally:
            # We must do that, otherwise `__del__` calls `shutdown` and the next test will fail
            col.shutdown(close_env=False)

    @pytest.fixture(scope="function")
    def clean_ray(self):
        import ray

        # Clean up any existing process group from previous tests
        if dist.is_initialized():
            dist.destroy_process_group()

        ray.shutdown()
        ray.init(ignore_reinit_error=True)
        yield
        ray.shutdown()

        # Clean up process group after test
        if dist.is_initialized():
            dist.destroy_process_group()

    @pytest.mark.skipif(not _has_ray, reason="Ray not found")
    @pytest.mark.parametrize("use_rb", [False, True], ids=["rb_false", "rb_true"])
    @pytest.mark.parametrize("num_collectors", [1, 4], ids=["1_col", "4_col"])
    def test_isaaclab_ray_collector(self, env, use_rb, clean_ray, num_collectors):
        # Create replay buffer if requested
        replay_buffer = None
        if use_rb:
            replay_buffer = RayReplayBuffer(
                # We place the storage on memmap to make it shareable
                storage=partial(LazyMemmapStorage, 10_000, ndim=2),
                ray_init_config={"num_cpus": 4},
            )

        col = RayCollector(
            [torchrl.testing.env_helper.make_isaac_env] * num_collectors,
            env.full_action_spec.rand_update,
            frames_per_batch=8192,
            total_frames=65536,
            replay_buffer=replay_buffer,
            num_collectors=num_collectors,
            collector_kwargs={
                "trust_policy": True,
                "no_cuda_sync": True,
                "extend_buffer": True,
            },
        )

        try:
            if use_rb:
                # When replay buffer is provided, collector yields None and populates buffer
                for i, data in enumerate(col):
                    # Data is None when using replay buffer
                    assert data is None, "Expected None when using replay buffer"

                    # Check replay buffer is being populated
                    if i >= 0:
                        # Wait for buffer to have enough data to sample
                        if len(replay_buffer) >= 32:
                            sample = replay_buffer.sample(32)
                            assert sample.batch_size == (32,)
                            # Check that we have meaningful data (not all zeros/nans)
                            assert sample["policy"].isfinite().any()
                            assert sample["action"].isfinite().any()
                            # Check shape is correct for Isaac Lab env (should have batch dim from env)
                            assert len(sample.shape) == 1

                    # Only collect a few batches for the test
                    if i >= 2:
                        break

                # Verify replay buffer has data
                assert len(replay_buffer) > 0, "Replay buffer should not be empty"
                # Test that we can sample multiple times
                for _ in range(5):
                    sample = replay_buffer.sample(16)
                    assert sample.batch_size == (16,)
                    assert sample["policy"].isfinite().any()

            else:
                # Without replay buffer, collector yields data normally
                collected_frames = 0
                for i, data in enumerate(col):
                    assert (
                        data is not None
                    ), "Expected data when not using replay buffer"
                    # Check the data shape matches the batch size
                    assert (
                        data.numel() >= 1000
                    ), f"Expected at least 1000 frames, got {data.numel()}"
                    collected_frames += data.numel()

                    # Only collect a few batches for the test
                    if i >= 2:
                        break

                # Verify we collected some data
                assert collected_frames > 0, "No frames were collected"

        finally:
            # Clean shutdown
            col.shutdown()
            if use_rb:
                replay_buffer.close()

    @pytest.mark.skipif(not _has_ray, reason="Ray not found")
    @pytest.mark.parametrize("num_collectors", [1, 4], ids=["1_col", "4_col"])
    def test_isaaclab_ray_collector_start(self, env, clean_ray, num_collectors):
        rb = RayReplayBuffer(
            storage=partial(LazyTensorStorage, 100_000, ndim=2),
            ray_init_config={"num_cpus": 4},
        )
        col = RayCollector(
            [torchrl.testing.env_helper.make_isaac_env] * num_collectors,
            env.full_action_spec.rand_update,
            frames_per_batch=8192,
            total_frames=65536,
            trust_policy=True,
            replay_buffer=rb,
            num_collectors=num_collectors,
        )
        col.start()
        try:
            time_waiting = 0
            while time_waiting < 30:
                if len(rb) >= 4096:
                    break
                time.sleep(0.1)
                time_waiting += 0.1
            else:
                raise RuntimeError("Timeout waiting for data")
            sample = rb.sample(4096)
            assert sample.batch_size == (4096,)
            assert sample["policy"].isfinite().any()
            assert sample["action"].isfinite().any()
        finally:
            col.shutdown()
            rb.close()

    def test_isaaclab_reset(self, env):
        # Make a rollout that will stop as soon as a trajectory reaches a done state
        r = env.rollout(1_000_000)

        # Check that done obs are None
        assert not r["next", "policy"][r["next", "done"].squeeze(-1)].isfinite().any()

    def test_isaaclab_native_autoreset_vecnorm_step_and_maybe_reset(self):
        env = make_isaac_env(native_autoreset=True)
        try:
            obs_keys = list(env.full_observation_spec.keys(True, True))
            obs_key = "policy" if "policy" in obs_keys else obs_keys[0]
            vecnorm = CountingVecNormV2(
                in_keys=[obs_key],
                reduce_batch_dims=True,
            )
            env = TransformedEnv(env, vecnorm)
            td = env.reset()
            initial_step_calls = vecnorm.step_calls
            initial_call_count = vecnorm.step_calls + vecnorm.reset_calls

            isaac_env = env
            while isinstance(isaac_env, TransformedEnv):
                isaac_env = isaac_env.base_env
            raw_env = isaac_env._env.unwrapped
            if not hasattr(raw_env, "episode_length_buf") or not hasattr(
                raw_env, "max_episode_length"
            ):
                pytest.skip("Isaac Lab env does not expose episode length buffers.")
            raw_env.episode_length_buf.fill_(raw_env.max_episode_length - 1)

            td = env.rand_action(td)
            td, td_ = env.step_and_maybe_reset(td)

            assert td["next", "done"].any()
            done = td["next", "done"].squeeze(-1)
            assert torch.isnan(td["next", obs_key][done]).all()
            assert not td_["done"].any()
            assert td_[obs_key][done].isfinite().all()
            assert vecnorm.step_calls == initial_step_calls + 1
            assert vecnorm.step_calls + vecnorm.reset_calls == initial_call_count + 1
        finally:
            env.close()

    @pytest.mark.parametrize("seed", [0, 1])
    def test_isaaclab_native_autoreset_rollout_seeded(self, seed):
        rollout0 = _isaaclab_native_rollout(seed)
        rollout1 = _isaaclab_native_rollout(seed)

        assert rollout0["next", "done"].any()
        assert_allclose_td(rollout0, rollout1, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("seed", [0, 1])
    def test_isaaclab_native_autoreset_rollout_matches_default_signals(self, seed):
        rollout = _isaaclab_rollout(seed, native_autoreset=False)
        rollout_native = _isaaclab_rollout(seed, native_autoreset=True)
        keys = _isaaclab_transition_keys(rollout_native)

        assert rollout_native["next", "done"].any()
        done = rollout["next", "done"].squeeze(-1)
        done_native = rollout_native["next", "done"].squeeze(-1)
        assert torch.isnan(rollout["next", "policy"][done]).all()
        assert torch.isnan(rollout_native["next", "policy"][done_native]).all()
        assert_allclose_td(
            rollout.select(*keys),
            rollout_native.select(*keys),
            rtol=1e-6,
            atol=1e-6,
        )

    @pytest.mark.parametrize("seed", [0, 1])
    def test_isaaclab_native_autoreset_rollout_reset_obs_continuity(self, seed):
        rollout = _isaaclab_native_rollout(seed)
        done = rollout["next", "done"].squeeze(-1)

        assert done.any()
        assert not rollout["done"].any()
        if done[..., :-1].any():
            assert torch.isnan(
                rollout["next", "policy"][..., :-1, :][done[..., :-1]]
            ).all()
            assert rollout["policy"][..., 1:, :][done[..., :-1]].isfinite().all()

    def test_isaaclab_lstm(self, env):
        """Test that LSTM/RNN works with pre-vectorized IsaacLab environments (Issue #1493).

        This test verifies that TensorDictPrimer correctly expands hidden state specs
        to match the environment's batch size when using vectorized environments like IsaacLab.
        """
        # Create a fresh env with InitTracker (required for LSTM)
        test_env = TransformedEnv(env, InitTracker())

        # Get observation size from the env
        obs_size = test_env.observation_spec["policy"].shape[-1]
        action_size = test_env.action_spec.shape[-1]

        # Create LSTM module
        lstm = LSTMModule(
            input_size=obs_size,
            hidden_size=32,
            in_keys=["policy", "recurrent_state_h", "recurrent_state_c"],
            out_keys=[
                "lstm_out",
                ("next", "recurrent_state_h"),
                ("next", "recurrent_state_c"),
            ],
        )

        # Add the primer transform - this was the original failure point in Issue #1493
        primer = lstm.make_tensordict_primer()
        test_env = test_env.append_transform(primer)

        # Verify reset works (this was the original failure point)
        td = test_env.reset()
        assert "recurrent_state_h" in td.keys()
        assert "recurrent_state_c" in td.keys()
        # Hidden states should have shape (num_envs, num_layers, hidden_size)
        assert td["recurrent_state_h"].shape[0] == env.batch_size[0]
        assert td["recurrent_state_c"].shape[0] == env.batch_size[0]

        # Create a simple policy using the LSTM and move to correct device
        policy = Seq(
            lstm,
            Mod(
                MLP(in_features=32, out_features=action_size, num_cells=[]),
                in_keys=["lstm_out"],
                out_keys=["action"],
            ),
        ).to(env.device)

        # Verify rollout works with the LSTM policy
        rollout = test_env.rollout(10, policy=policy, break_when_any_done=False)
        assert rollout.shape[0] == env.batch_size[0]
        assert rollout.shape[1] == 10
        # Verify recurrent states are carried through the rollout
        assert ("next", "recurrent_state_h") in rollout.keys(True)
        assert ("next", "recurrent_state_c") in rollout.keys(True)


@pytest.mark.skipif(not _has_isaaclab, reason="Isaaclab not found")
class TestIsaacLabEvaluator:
    """End-to-end coverage for the Evaluator with Isaac Lab.

    Isaac Lab has two quirks that stress every Evaluator backend:
      * ``AppLauncher`` must run before ``import torch`` in whatever process
        hosts the env, so the ``process`` / ``ray`` backends must expose an
        ``init_fn`` hook.
      * The env is GPU-native and pinned to a CUDA device, so device
        assignment must be exercised on a dedicated GPU.
    """

    @pytest.fixture(scope="class")
    def env(self):
        env = make_isaac_env()
        try:
            yield env
        finally:
            torchrl_logger.info("Closing IsaacLab env (evaluator tests)...")
            env.close()
            torchrl_logger.info("Closed")

    @pytest.fixture(scope="function")
    def clean_ray(self):
        import ray

        if dist.is_initialized():
            dist.destroy_process_group()
        ray.shutdown()
        ray.init(ignore_reinit_error=True)
        yield
        ray.shutdown()
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_evaluator_thread_backend(self, env):
        """Thread backend with an eagerly-constructed env + policy.

        This is the simplest case: ``AppLauncher`` has already run in the
        main process via the ``env`` fixture, so the daemon thread spawned
        by the Evaluator inherits the ready-to-use Isaac state.
        """
        policy = make_isaac_policy(env)
        evaluator = Evaluator(
            env=env,
            policy=policy,
            num_trajectories=1,
            max_steps=32,
            backend="thread",
        )
        try:
            result = evaluator.evaluate(step=0)
            assert "eval/reward" in result
            assert "eval/episode_length" in result
            assert torch.isfinite(torch.as_tensor(result["eval/reward"]))
        finally:
            evaluator.shutdown()

    def test_evaluator_process_backend(self):
        """Process backend: AppLauncher must be started in the child via init_fn.

        Uses ``_isaac_app_launcher_init`` as ``init_fn`` and a factory that
        skips the in-process AppLauncher (``init_app=False``).  Exercises
        the newly plumbed ``init_fn`` support in ``_ThreadEvalBackend`` →
        ``MultiSyncCollector`` → ``_main_async_collector``.
        """
        evaluator = Evaluator(
            env=_isaac_env_maker,
            policy_factory=_isaac_policy_maker,
            num_trajectories=1,
            max_steps=32,
            backend="process",
            init_fn=_isaac_app_launcher_init,
        )
        try:
            result = evaluator.evaluate(step=0)
            assert "eval/reward" in result
            assert "eval/episode_length" in result
            assert torch.isfinite(torch.as_tensor(result["eval/reward"]))
        finally:
            evaluator.shutdown()

    @pytest.mark.skipif(not _has_ray, reason="Ray not found")
    def test_evaluator_ray_backend(self, clean_ray):
        """Ray backend: the canonical Isaac-friendly path.

        The Ray actor process is fresh, so ``init_fn`` runs before any torch
        import and the env / policy are built inside the actor.
        """
        evaluator = Evaluator(
            env=_isaac_env_maker,
            policy_factory=_isaac_policy_maker,
            num_trajectories=1,
            max_steps=32,
            backend="ray",
            init_fn=_isaac_app_launcher_init,
            num_gpus=1,
        )
        try:
            result = evaluator.evaluate(step=0)
            assert "eval/reward" in result
            assert "eval/episode_length" in result
            assert torch.isfinite(torch.as_tensor(result["eval/reward"]))
        finally:
            evaluator.shutdown()

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="Needs 2+ CUDA devices to test dedicated-device eval",
    )
    @pytest.mark.skipif(not _has_ray, reason="Ray not found")
    def test_evaluator_dedicated_device(self, clean_ray):
        """Run Isaac + policy on cuda:1 while the main process keeps cuda:0.

        Reserved GPU 0 for training; evaluation runs on GPU 1.  Uses Ray
        backend so that (a) ``init_fn`` fires before torch is imported in
        the actor and (b) the actor gets its own CUDA context on the target
        device.
        """
        # ``num_gpus=2`` so the actor gets both cuda:0 and cuda:1 visible and
        # can place Isaac + policy on cuda:1 explicitly.  With ``num_gpus=1``
        # Ray would map a single physical GPU onto the actor's cuda:0 and
        # cuda:1 would be 'invalid device ordinal'.
        evaluator = Evaluator(
            env=_isaac_env_maker_cuda1,
            policy_factory=_isaac_policy_maker_cuda1,
            num_trajectories=1,
            max_steps=32,
            backend="ray",
            init_fn=_isaac_app_launcher_init,
            num_gpus=2,
        )
        try:
            result = evaluator.evaluate(step=0)
            assert "eval/reward" in result
            assert torch.isfinite(torch.as_tensor(result["eval/reward"]))
        finally:
            evaluator.shutdown()
