from __future__ import annotations

import contextlib
import queue
from collections.abc import Callable
from functools import partial
from multiprocessing import connection, queues
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase

from torchrl import logger as torchrl_logger
from torchrl._utils import timeit, VERBOSE
from torchrl.collectors._base import BaseCollector, ProfileConfig
from torchrl.collectors._constants import (
    _MAX_IDLE_COUNT,
    _MIN_TIMEOUT,
    _TIMEOUT,
    DEFAULT_EXPLORATION_TYPE,
)
from torchrl.collectors._single import Collector

from torchrl.collectors.utils import (
    _cast,
    _make_policy_factory,
    _map_to_cpu_if_needed,
    _TrajectoryPool,
)
from torchrl.data import ReplayBuffer
from torchrl.envs import EnvBase, EnvCreator
from torchrl.envs.utils import ExplorationType
from torchrl.weight_update import WeightSyncScheme


class _WorkerProfiler:
    """Helper class for profiling worker rollouts.

    Manages the PyTorch profiler lifecycle for a worker process,
    handling warmup, active profiling, and trace export.
    """

    def __init__(
        self,
        profile_config: ProfileConfig,
        worker_idx: int,
    ):
        self.config = profile_config
        self.worker_idx = worker_idx
        self.rollout_count = 0
        self._profiler = None
        self._stopped = False
        self._active = False

        # Check if this worker should be profiled
        if not self.config.should_profile_worker(worker_idx):
            return

        # Set up profiler schedule
        # - skip_first: warmup rollouts (profiler runs but data discarded)
        # - wait: 0 (no wait between cycles)
        # - warmup: 0 (we handle warmup via skip_first)
        # - active: num_rollouts - warmup_rollouts
        # - repeat: 1 (single profiling cycle)
        active_rollouts = self.config.num_rollouts - self.config.warmup_rollouts
        profiler_schedule = torch.profiler.schedule(
            skip_first=self.config.warmup_rollouts,
            wait=0,
            warmup=0,
            active=active_rollouts,
            repeat=1,
        )

        # Get activities
        activities = self.config.get_activities()
        if not activities:
            torchrl_logger.warning(
                f"Worker {worker_idx}: No profiler activities available. Profiling disabled."
            )
            return

        # Determine trace handler
        if self.config.on_trace_ready is not None:
            on_trace_ready = self.config.on_trace_ready
        else:
            save_path = self.config.get_save_path(worker_idx)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            def on_trace_ready(prof, save_path=save_path):
                prof.export_chrome_trace(str(save_path))
                torchrl_logger.info(
                    f"Worker {worker_idx}: Profiling trace saved to {save_path}"
                )

        self._profiler = torch.profiler.profile(
            activities=activities,
            schedule=profiler_schedule,
            on_trace_ready=on_trace_ready,
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.profile_memory,
            with_stack=self.config.with_stack,
            with_flops=self.config.with_flops,
        )
        self._active = True

    def start(self) -> None:
        """Start the profiler."""
        if self._profiler is not None and not self._stopped:
            self._profiler.start()
            torchrl_logger.info(
                f"Worker {self.worker_idx}: Profiling started. "
                f"Will profile rollouts {self.config.warmup_rollouts} to {self.config.num_rollouts - 1}."
            )

    def step(self) -> bool:
        """Step the profiler after a rollout.

        Returns:
            True if profiling is complete.
        """
        if self._profiler is None or self._stopped:
            return False

        self.rollout_count += 1
        self._profiler.step()

        # Check if profiling is complete
        if self.rollout_count >= self.config.num_rollouts:
            self.stop()
            return True

        return False

    def stop(self) -> None:
        """Stop the profiler and export trace."""
        if self._profiler is not None and not self._stopped:
            self._profiler.stop()
            self._stopped = True
            torchrl_logger.info(
                f"Worker {self.worker_idx}: Profiling complete after {self.rollout_count} rollouts."
            )

    @property
    def is_active(self) -> bool:
        """Check if profiling is active."""
        return self._active and not self._stopped

    @contextlib.contextmanager
    def profile_rollout(self):
        """Context manager for profiling a single rollout."""
        if self._profiler is not None and not self._stopped:
            with torch.profiler.record_function(f"worker_{self.worker_idx}_rollout"):
                yield
        else:
            yield


def _main_async_collector(
    pipe_child: connection.Connection,
    queue_out: queues.Queue,
    create_env_fn: EnvBase | EnvCreator | Callable[[], EnvBase],  # noqa: F821
    create_env_kwargs: dict[str, Any],
    policy: Callable[[TensorDictBase], TensorDictBase],
    max_frames_per_traj: int,
    frames_per_batch: int,
    reset_at_each_iter: bool,
    storing_device: torch.device | str | int | None,
    env_device: torch.device | str | int | None,
    policy_device: torch.device | str | int | None,
    idx: int = 0,
    exploration_type: ExplorationType = DEFAULT_EXPLORATION_TYPE,
    reset_when_done: bool = True,
    verbose: bool = VERBOSE,
    interruptor=None,
    set_truncated: bool = False,
    use_buffers: bool | None = None,
    replay_buffer: ReplayBuffer | None = None,
    extend_buffer: bool = True,
    traj_pool: _TrajectoryPool = None,
    trust_policy: bool = False,
    compile_policy: bool = False,
    cudagraph_policy: bool = False,
    no_cuda_sync: bool = False,
    policy_factory: Callable | None = None,
    collector_class: type | Callable[[], BaseCollector] | None = None,
    postproc: Callable[[TensorDictBase], TensorDictBase] | None = None,
    weight_sync_schemes: dict[str, WeightSyncScheme] | None = None,
    worker_idx: int | None = None,
    init_random_frames: int | None = None,
    profile_config: ProfileConfig | None = None,
) -> None:
    if collector_class is None:
        collector_class = Collector
    # init variables that will be cleared when closing
    collected_tensordict = data = next_data = data_in = inner_collector = dc_iter = None

    # Make a policy-factory out of the policy
    policy_factory = partial(
        _make_policy_factory,
        policy=policy,
        policy_factory=policy_factory,
        weight_sync_scheme=weight_sync_schemes.get("policy")
        if weight_sync_schemes
        else None,
        worker_idx=worker_idx,
        pipe=pipe_child,
    )
    policy = None
    # Store the original init_random_frames for run_free mode logic
    original_init_random_frames = (
        init_random_frames if init_random_frames is not None else 0
    )
    try:
        collector_class._ignore_rb = extend_buffer
        inner_collector = collector_class(
            create_env_fn,
            create_env_kwargs=create_env_kwargs,
            policy=policy,
            policy_factory=policy_factory,
            total_frames=-1,
            max_frames_per_traj=max_frames_per_traj,
            frames_per_batch=frames_per_batch,
            reset_at_each_iter=reset_at_each_iter,
            postproc=postproc,
            split_trajs=False,
            storing_device=storing_device,
            policy_device=policy_device,
            env_device=env_device,
            exploration_type=exploration_type,
            reset_when_done=reset_when_done,
            return_same_td=replay_buffer is None,
            interruptor=interruptor,
            set_truncated=set_truncated,
            use_buffers=use_buffers,
            replay_buffer=replay_buffer,
            extend_buffer=extend_buffer,
            traj_pool=traj_pool,
            trust_policy=trust_policy,
            compile_policy=compile_policy,
            cudagraph_policy=cudagraph_policy,
            no_cuda_sync=no_cuda_sync,
            # We don't pass the weight sync scheme as only the sender has the weight sync scheme within.
            # weight_sync_schemes=weight_sync_schemes,
            worker_idx=worker_idx,
            # init_random_frames is passed; inner collector will use _should_use_random_frames()
            # which checks replay_buffer.write_count when replay_buffer is provided
            init_random_frames=init_random_frames,
        )
        # Set up weight receivers for worker process using the standard register_scheme_receiver API.
        # This properly initializes the schemes on the receiver side and stores them in _receiver_schemes.
        if weight_sync_schemes:
            inner_collector.register_scheme_receiver(weight_sync_schemes)

        use_buffers = inner_collector._use_buffers
        if verbose:
            torchrl_logger.debug("Sync data collector created")

        # Set up profiler for this worker if configured
        worker_profiler = None
        if profile_config is not None:
            worker_profiler = _WorkerProfiler(profile_config, worker_idx)
            if worker_profiler.is_active:
                worker_profiler.start()

        dc_iter = iter(inner_collector)
        j = 0
        pipe_child.send("instantiated")
    except Exception as e:
        # Send error information to main process
        # We send a dict with the exception info so we can recreate it in the main process
        import traceback

        error_info = {
            "error": True,
            "exception_type": type(e).__name__,
            "exception_module": type(e).__module__,
            "exception_msg": str(e),
            "traceback": traceback.format_exc(),
        }
        try:
            pipe_child.send(error_info)
        except Exception:
            # If pipe is broken, nothing we can do
            pass
        return

    has_timed_out = False
    counter = 0
    run_free = False
    while True:
        _timeout = _TIMEOUT if not has_timed_out else 1e-3
        if not run_free and pipe_child.poll(_timeout):
            counter = 0
            try:
                data_in, msg = pipe_child.recv()
                if verbose:
                    torchrl_logger.debug(f"mp worker {idx} received {msg}")
            except EOFError:
                raise
        elif not run_free:
            if verbose:
                torchrl_logger.debug(f"poll failed, j={j}, worker={idx}")
            # default is "continue" (after first iteration)
            # this is expected to happen if queue_out reached the timeout, but no new msg was waiting in the pipe
            # in that case, the main process probably expects the worker to continue collect data
            if has_timed_out:
                counter = 0
                # has_timed_out is True if the process failed to send data, which will
                # typically occur if main has taken another batch (i.e. the queue is Full).
                # In this case, msg is the previous msg sent by main, which will typically be "continue"
                # If it's not the case, it is not expected that has_timed_out is True.
                if msg not in ("continue", "continue_random"):
                    raise RuntimeError(f"Unexpected message after time out: msg={msg}")
            else:
                # if has_timed_out is False, then the time out does not come from the fact that the queue is Full.
                # this means that our process has been waiting for a command from main in vain, while main was not
                # receiving data.
                # This will occur if main is busy doing something else (e.g. computing loss etc).

                counter += _timeout
                if verbose:
                    torchrl_logger.debug(f"mp worker {idx} has counter {counter}")
                if counter >= (_MAX_IDLE_COUNT * _TIMEOUT):
                    raise RuntimeError(
                        f"This process waited for {counter} seconds "
                        f"without receiving a command from main. Consider increasing the maximum idle count "
                        f"if this is expected via the environment variable MAX_IDLE_COUNT "
                        f"(current value is {_MAX_IDLE_COUNT})."
                        f"\nIf this occurs at the end of a function or program, it means that your collector has not been "
                        f"collected, consider calling `collector.shutdown()` before ending the program."
                    )
                continue
        else:
            # placeholder, will be checked after
            msg = "continue"
        if msg == "run_free":
            run_free = True
            msg = "continue"
        if run_free:
            # Capture shutdown / update / seed signal, but continue should not be expected
            if pipe_child.poll(1e-4):
                data_in, msg = pipe_child.recv()
                if msg == "continue":
                    # Switch back to run_free = False
                    run_free = False
                if msg == "pause":
                    queue_out.put((idx, "paused"), timeout=_TIMEOUT)
                    while not pipe_child.poll(1e-2):
                        continue
                    data_in, msg = pipe_child.recv()
                    if msg != "restart":
                        raise RuntimeError(f"Expected msg='restart', got {msg=}")
                    msg = "continue"
            else:
                data_in = None
                # In run_free mode, determine msg based on replay_buffer.write_count for random frames
                if (
                    replay_buffer is not None
                    and original_init_random_frames > 0
                    and replay_buffer.write_count < original_init_random_frames
                ):
                    msg = "continue_random"
                else:
                    msg = "continue"
        # Note: Weight updates are handled by background threads in weight sync schemes.
        # The scheme's background receiver thread listens for "receive" instructions.

        if msg == "enable_profile":
            # Handle profile configuration sent after worker startup
            if worker_profiler is None or not worker_profiler.is_active:
                worker_profiler = _WorkerProfiler(data_in, worker_idx)
                if worker_profiler.is_active:
                    worker_profiler.start()
            pipe_child.send((j, "profile_enabled"))
            has_timed_out = False
            continue

        if msg == "update":
            # Legacy - weight updater
            with timeit(f"worker/{idx}/update") as update_timer:
                torchrl_logger.debug(
                    f"mp worker {idx}: Received weight update request..."
                )
                inner_collector.update_policy_weights_(policy_weights=data_in)
                torchrl_logger.debug(
                    f"mp worker {idx}: Weight update completed in {update_timer.elapsed():.3f}s"
                )
            pipe_child.send((j, "updated"))
            has_timed_out = False
            continue

        # Note: Weight updates are now handled by background threads in the weight sync schemes.
        # The scheme's background receiver thread listens for "receive" instructions and
        # applies weights automatically. No explicit message handling needed here.

        if msg in ("continue", "continue_random"):
            # When in run_free mode with a replay_buffer, the inner collector uses
            # _should_use_random_frames() which checks replay_buffer.write_count.
            # So we don't override init_random_frames. Otherwise, we use the message
            # to control whether random frames are used.
            if not run_free or replay_buffer is None:
                if msg == "continue_random":
                    inner_collector.init_random_frames = float("inf")
                else:
                    inner_collector.init_random_frames = -1

            # Debug logging for rollout timing
            # Use profiler context if profiling is active
            profile_ctx = (
                worker_profiler.profile_rollout()
                if worker_profiler is not None and worker_profiler.is_active
                else contextlib.nullcontext()
            )
            with profile_ctx:
                with timeit(f"worker/{idx}/rollout") as rollout_timer:
                    torchrl_logger.debug(
                        f"mp worker {idx}: Starting rollout (j={j})..."
                    )
                    next_data = next(dc_iter)
                    torchrl_logger.debug(
                        f"mp worker {idx}: Rollout completed in {rollout_timer.elapsed():.3f}s, "
                        f"frames={next_data.numel() if hasattr(next_data, 'numel') else 'N/A'}"
                    )

            # Step the profiler after each rollout
            if worker_profiler is not None and worker_profiler.is_active:
                worker_profiler.step()
            if pipe_child.poll(_MIN_TIMEOUT):
                # in this case, main send a message to the worker while it was busy collecting trajectories.
                # In that case, we skip the collected trajectory and get the message from main. This is faster than
                # sending the trajectory in the queue until timeout when it's never going to be received.
                continue

            if replay_buffer is not None:
                if extend_buffer:
                    next_data.names = None
                    replay_buffer.extend(next_data)

                if run_free:
                    continue

                try:
                    queue_out.put((idx, j), timeout=_TIMEOUT)
                    if verbose:
                        torchrl_logger.debug(f"mp worker {idx} successfully sent data")
                    j += 1
                    has_timed_out = False
                    continue
                except queue.Full:
                    has_timed_out = True
                    continue

            if j == 0 or not use_buffers:
                collected_tensordict = next_data
                if (
                    storing_device is not None
                    and collected_tensordict.device != storing_device
                ):
                    raise RuntimeError(
                        f"expected device to be {storing_device} but got {collected_tensordict.device}"
                    )
                if use_buffers:
                    # If policy and env are on cpu, we put in shared mem,
                    # if policy is on cuda and env on cuda, we are fine with this
                    # If policy is on cuda and env on cpu (or opposite) we put tensors that
                    # are on cpu in shared mem.
                    MPS_ERROR = (
                        "tensors on mps device cannot be put in shared memory. Make sure "
                        "the shared device (aka storing_device) is set to CPU."
                    )
                    if collected_tensordict.device is not None:
                        # placeholder in case we need different behaviors
                        if collected_tensordict.device.type in ("cpu",):
                            collected_tensordict.share_memory_()
                        elif collected_tensordict.device.type in ("mps",):
                            raise RuntimeError(MPS_ERROR)
                        elif collected_tensordict.device.type == "cuda":
                            collected_tensordict.share_memory_()
                        else:
                            raise NotImplementedError(
                                f"Device {collected_tensordict.device} is not supported in multi-collectors yet."
                            )
                    else:
                        # make sure each cpu tensor is shared - assuming non-cpu devices are shared
                        def cast_tensor(x, MPS_ERROR=MPS_ERROR):
                            if x.device.type in ("cpu",):
                                x.share_memory_()
                            if x.device.type in ("mps",):
                                RuntimeError(MPS_ERROR)

                        collected_tensordict.apply(cast_tensor, filter_empty=True)
                data = (collected_tensordict, idx)
            else:
                if next_data is not collected_tensordict:
                    raise RuntimeError(
                        "Collector should return the same tensordict modified in-place."
                    )
                data = idx  # flag the worker that has sent its data
            try:
                queue_out.put((data, j), timeout=_TIMEOUT)
                if verbose:
                    torchrl_logger.debug(f"mp worker {idx} successfully sent data")
                j += 1
                has_timed_out = False
                continue
            except queue.Full:
                if verbose:
                    torchrl_logger.debug(f"mp worker {idx} has timed out")
                has_timed_out = True
                continue

        if msg == "seed":
            data_in, static_seed = data_in
            new_seed = inner_collector.set_seed(data_in, static_seed=static_seed)
            torch.manual_seed(data_in)
            np.random.seed(data_in)
            pipe_child.send((new_seed, "seeded"))
            has_timed_out = False
            continue

        elif msg == "reset":
            inner_collector.reset()
            pipe_child.send((j, "reset"))
            continue

        elif msg == "state_dict":
            from torch.utils._pytree import tree_map

            state_dict = inner_collector.state_dict()
            # Map exotic devices (MPS, NPU, etc.) to CPU for multiprocessing compatibility
            # CPU and CUDA tensors are already shareable and don't need conversion BUT we need to clone the CUDA tensors in case they were sent from main (cannot send cuda tensors back and forth)
            state_dict = tree_map(_map_to_cpu_if_needed, state_dict)
            state_dict = TensorDict(state_dict)
            state_dict = state_dict.clone().apply(_cast, state_dict).to_dict()
            pipe_child.send((state_dict, "state_dict"))
            has_timed_out = False
            continue

        elif msg == "load_state_dict":
            state_dict = data_in
            inner_collector.load_state_dict(state_dict)
            del state_dict
            pipe_child.send((j, "loaded"))
            has_timed_out = False
            continue

        elif msg == "getattr_policy":
            attr_name = data_in
            try:
                result = getattr(inner_collector.policy, attr_name)
                pipe_child.send((result, "getattr_policy"))
            except AttributeError as e:
                pipe_child.send((e, "getattr_policy"))
            has_timed_out = False
            continue

        elif msg == "getattr_env":
            attr_name = data_in
            try:
                result = getattr(inner_collector.env, attr_name)
                pipe_child.send((result, "getattr_env"))
            except AttributeError as e:
                pipe_child.send((e, "getattr_env"))
            has_timed_out = False
            continue

        elif msg == "close":
            # Stop profiler if active
            if worker_profiler is not None and worker_profiler.is_active:
                worker_profiler.stop()
            del collected_tensordict, data, next_data, data_in
            inner_collector.shutdown()
            del inner_collector, dc_iter
            pipe_child.send("closed")
            if verbose:
                torchrl_logger.debug(f"collector {idx} closed")
            break

        else:
            raise Exception(f"Unrecognized message {msg}")
