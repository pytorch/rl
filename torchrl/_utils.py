# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import collections
import functools
import inspect
import logging
import math
import os
import pickle
import sys
import threading
import time
import traceback
import warnings
from collections.abc import Callable
from contextlib import nullcontext
from functools import wraps
from textwrap import indent
from typing import Any, cast, TypeVar

import numpy as np
import torch

from pyvers import implement_for  # noqa: F401
from tensordict import unravel_key
from tensordict.utils import NestedKey
from torch import multiprocessing as mp, Tensor

try:
    from torch.compiler import is_compiling
except ImportError:
    from torch._dynamo import is_compiling


def strtobool(val: Any) -> bool:
    """Convert a string representation of truth to a boolean.

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values are 'n', 'no', 'f', 'false', 'off', and '0'.
    Raises ValueError if 'val' is anything else.
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    if val in ("n", "no", "f", "false", "off", "0"):
        return False
    raise ValueError(f"Invalid truth value {val!r}")


LOGGING_LEVEL = os.environ.get("RL_LOGGING_LEVEL", "INFO")
logger = logging.getLogger("torchrl")
logger.setLevel(LOGGING_LEVEL)
logger.propagate = False
# Clear existing handlers
while logger.hasHandlers():
    logger.removeHandler(logger.handlers[0])
stream_handlers = {
    "stdout": sys.stdout,
    "stderr": sys.stderr,
}
TORCHRL_CONSOLE_STREAM = os.getenv("TORCHRL_CONSOLE_STREAM")
stream_handler = stream_handlers.get(TORCHRL_CONSOLE_STREAM, sys.stdout)


# Create colored handler
class _CustomFormatter(logging.Formatter):
    def format(self, record):
        # Format the initial part in green
        green_format = "\033[92m%(asctime)s [%(name)s][%(levelname)s]\033[0m"
        # Format the message part
        message_format = "%(message)s"
        # End marker in green
        end_marker = "\033[92m [END]\033[0m"
        # Combine all parts
        formatted_message = logging.Formatter(
            green_format + indent(message_format, " " * 4) + end_marker
        ).format(record)

        return formatted_message


console_handler = logging.StreamHandler(stream=stream_handler)
console_handler.setFormatter(_CustomFormatter())
logger.addHandler(console_handler)

console_handler.setLevel(LOGGING_LEVEL)
logger.debug(f"Logging level: {logger.getEffectiveLevel()}")

VERBOSE = strtobool(os.environ.get("VERBOSE", str(logger.isEnabledFor(logging.DEBUG))))
_os_is_windows = sys.platform == "win32"
RL_WARNINGS = strtobool(os.environ.get("RL_WARNINGS", "1"))
if RL_WARNINGS:
    warnings.filterwarnings("once", category=DeprecationWarning, module="torchrl")

BATCHED_PIPE_TIMEOUT = float(os.environ.get("BATCHED_PIPE_TIMEOUT", "10000.0"))

_TORCH_DTYPES = (
    torch.bfloat16,
    torch.bool,
    torch.complex128,
    torch.complex32,
    torch.complex64,
    torch.float16,
    torch.float32,
    torch.float64,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.int8,
    torch.qint32,
    torch.qint8,
    torch.quint4x2,
    torch.quint8,
    torch.uint8,
)
if hasattr(torch, "uint16"):
    _TORCH_DTYPES = _TORCH_DTYPES + (torch.uint16,)
if hasattr(torch, "uint32"):
    _TORCH_DTYPES = _TORCH_DTYPES + (torch.uint32,)
if hasattr(torch, "uint64"):
    _TORCH_DTYPES = _TORCH_DTYPES + (torch.uint64,)
_STR_DTYPE_TO_DTYPE = {str(dtype): dtype for dtype in _TORCH_DTYPES}
_STRDTYPE2DTYPE = _STR_DTYPE_TO_DTYPE
_DTYPE_TO_STR_DTYPE = {
    dtype: str_dtype for str_dtype, dtype in _STR_DTYPE_TO_DTYPE.items()
}
_DTYPE2STRDTYPE = _STR_DTYPE_TO_DTYPE


class timeit:
    """A dirty but easy to use decorator for profiling code.

    Args:
        name (str): The name of the timer.

    Examples:
        >>> from torchrl import timeit
        >>> @timeit("my_function")
        >>> def my_function():
            ...
        >>> my_function()
        >>> with timeit("my_other_function"):
        ...     my_other_function()
        >>> timeit.print()  # prints the state of the timer for each function
    """

    _REG = {}

    def __init__(self, name):
        self.name = name

    def __call__(self, fn: Callable) -> Callable:
        @wraps(fn)
        def decorated_fn(*args, **kwargs):
            with self:
                out = fn(*args, **kwargs)
                return out

        return decorated_fn

    def __enter__(self) -> None:
        self.t0 = time.time()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        t = time.time() - self.t0
        val = self._REG.setdefault(self.name, [0.0, 0.0, 0])

        count = val[2]
        N = count + 1
        val[0] = val[0] * (count / N) + t / N
        val[1] += t
        val[2] = N

    @staticmethod
    def print(prefix: str | None = None) -> str:  # noqa: T202
        """Prints the state of the timer.

        Args:
            prefix (str): The prefix to add to the keys. If `None`, no prefix is added.

        Returns:
            the string printed using the logger.
        """
        keys = list(timeit._REG)
        keys.sort()
        string = []
        for name in keys:
            strings = []
            if prefix:
                strings.append(prefix)
            strings.append(
                f"{name} took {timeit._REG[name][0] * 1000:4.4f} msec (total = {timeit._REG[name][1]: 4.4f} sec since last reset)."
            )
            string.append(" -- ".join(strings))
            logger.info(string[-1])
        return "\n".join(string)

    _printevery_count = 0

    @classmethod
    def printevery(
        cls,
        num_prints: int,
        total_count: int,
        *,
        prefix: str | None = None,
        erase: bool = False,
    ) -> None:
        """Prints the state of the timer at regular intervals.

        Args:
            num_prints (int): The number of times to print the state of the timer, given the total_count.
            total_count (int): The total number of times to print the state of the timer.
            prefix (str): The prefix to add to the keys. If `None`, no prefix is added.
            erase (bool): If True, erase the timer after printing. Default is `False`.

        """
        interval = max(1, total_count // num_prints)
        if cls._printevery_count % interval == 0:
            cls.print(prefix=prefix)
            if erase:
                cls.erase()
        cls._printevery_count += 1

    @classmethod
    def todict(
        cls, percall: bool = True, prefix: str | None = None
    ) -> dict[str, float]:
        """Convert the timer to a dictionary.

        Args:
            percall (bool): If True, return the average time per call.
            prefix (str): The prefix to add to the keys.
        """

        def _make_key(key):
            if prefix:
                return f"{prefix}/{key}"
            return key

        if percall:
            return {_make_key(key): val[0] for key, val in cls._REG.items()}
        return {_make_key(key): val[1] for key, val in cls._REG.items()}

    @staticmethod
    def erase():
        """Erase the timer.

        .. seealso:: :meth:`reset`
        """
        for k in timeit._REG:
            timeit._REG[k] = [0.0, 0.0, 0]

    @classmethod
    def reset(cls):
        """Reset the timer.

        .. seealso:: :meth:`erase`
        """
        cls.erase()


def _check_for_faulty_process(processes):
    terminate = False
    for p in processes:
        if not p._closed and not p.is_alive():
            terminate = True
            for _p in processes:
                _p: mp.Process
                if not _p._closed and _p.is_alive():
                    try:
                        _p.terminate()
                    except Exception:
                        _p.kill()
                    finally:
                        time.sleep(0.1)
                        _p.close()
            if terminate:
                break
    if terminate:
        raise RuntimeError(
            "At least one process failed. Check for more infos in the log."
        )


def seed_generator(seed):
    """A seed generator function.

    Given a seeding integer, generates a deterministic next seed to be used in a
    seeding sequence.

    Args:
        seed (int): initial seed.

    Returns: Next seed of the chain.

    """
    max_seed_val = (
        2**32 - 1
    )  # https://discuss.pytorch.org/t/what-is-the-max-seed-you-can-set-up/145688
    rng = np.random.default_rng(seed)
    seed = int.from_bytes(rng.bytes(8), "big")
    return seed % max_seed_val


class KeyDependentDefaultDict(collections.defaultdict):
    """A key-dependent default dict.

    Examples:
        >>> my_dict = KeyDependentDefaultDict(lambda key: "foo_" + key)
        >>> print(my_dict["bar"])
        foo_bar
    """

    def __init__(self, fun):
        self.fun = fun
        super().__init__()

    def __missing__(self, key):
        value = self.fun(key)
        self[key] = value
        return value


def prod(sequence):
    """General prod function, that generalised usage across math and np.

    Created for multiple python versions compatibility).

    """
    if hasattr(math, "prod"):
        return math.prod(sequence)
    else:
        return int(np.prod(sequence))


def get_binary_env_var(key):
    """Parses and returns the binary environment variable value.

    If not present in environment, it is considered `False`.

    Args:
        key (str): name of the environment variable.
    """
    val = os.environ.get(key, "False")
    if val in ("0", "False", "false"):
        val = False
    elif val in ("1", "True", "true"):
        val = True
    else:
        raise ValueError(
            f"Environment variable {key} should be in 'True', 'False', '0' or '1'. "
            f"Got {val} instead."
        )
    return val


class _Dynamic_CKPT_BACKEND:
    """Allows CKPT_BACKEND to be changed on-the-fly."""

    backends = ["torch", "torchsnapshot"]

    def _get_backend(self):
        backend = os.environ.get("CKPT_BACKEND", "torch")
        if backend == "torchsnapshot":
            try:
                import torchsnapshot  # noqa: F401
            except ImportError as err:
                raise ImportError(
                    f"torchsnapshot not found, but the backend points to this library. "
                    f"Consider installing torchsnapshot or choose another backend (available backends: {self.backends})"
                ) from err
        return backend

    def __getattr__(self, item):
        return getattr(self._get_backend(), item)

    def __eq__(self, other):
        return self._get_backend() == other

    def __ne__(self, other):
        return self._get_backend() != other

    def __repr__(self):
        return self._get_backend()


_CKPT_BACKEND = _Dynamic_CKPT_BACKEND()


def accept_remote_rref_invocation(func):
    """Decorator that allows a method to be invoked remotely.

    Passes the `rpc.RRef` associated with the remote object construction as first argument in place of the object reference.

    """

    @wraps(func)
    def unpack_rref_and_invoke_function(self, *args, **kwargs):
        # windows does not know torch._C._distributed_rpc.PyRRef
        if not _os_is_windows and isinstance(self, torch._C._distributed_rpc.PyRRef):
            self = self.local_value()
        return func(self, *args, **kwargs)

    return unpack_rref_and_invoke_function


def accept_remote_rref_udf_invocation(decorated_class):
    """Class decorator that applies `accept_remote_rref_invocation` to all public methods."""
    # ignores private methods
    for name in dir(decorated_class):
        method = getattr(decorated_class, name, None)
        if method is None:
            continue
        if callable(method) and not name.startswith("_"):
            setattr(decorated_class, name, accept_remote_rref_invocation(method))
    return decorated_class


# We copy this from torch as older versions do not have it
# see torch.utils._contextlib

# Extra utilities for working with context managers that should have been
# in the standard library but are not

# Used for annotating the decorator usage of _DecoratorContextManager (e.g.,
# 'no_grad' and 'enable_grad').
# See https://mypy.readthedocs.io/en/latest/generics.html#declaring-decorators
FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)


def _wrap_generator(ctx_factory, func):
    """Wrap each generator invocation with the context manager factory.

    The input should be a function that returns a context manager,
    not a context manager itself, to handle one-shot context managers.
    """

    @functools.wraps(func)
    def generator_context(*args, **kwargs):
        gen = func(*args, **kwargs)

        # Generators are suspended and unsuspended at `yield`, hence we
        # make sure the grad mode is properly set every time the execution
        # flow returns into the wrapped generator and restored when it
        # returns through our `yield` to our caller (see PR #49017).
        try:
            # Issuing `None` to a generator fires it up
            with ctx_factory():
                response = gen.send(None)

            while True:
                try:
                    # Forward the response to our caller and get its next request
                    request = yield response

                except GeneratorExit:
                    # Inform the still active generator about its imminent closure
                    with ctx_factory():
                        gen.close()
                    raise

                except BaseException:
                    # Propagate the exception thrown at us by the caller
                    with ctx_factory():
                        response = gen.throw(*sys.exc_info())

                else:
                    # Pass the last request to the generator and get its response
                    with ctx_factory():
                        response = gen.send(request)

        # We let the exceptions raised above by the generator's `.throw` or
        # `.send` methods bubble up to our caller, except for StopIteration
        except StopIteration as e:
            # The generator informed us that it is done: take whatever its
            # returned value (if any) was and indicate that we're done too
            # by returning it (see docs for python's return-statement).
            return e.value

    return generator_context


def context_decorator(ctx, func):
    """Context decorator.

    Like contextlib.ContextDecorator, but:

    1. Is done by wrapping, rather than inheritance, so it works with context
       managers that are implemented from C and thus cannot easily inherit from
       Python classes
    2. Wraps generators in the intuitive way (c.f. https://bugs.python.org/issue37743)
    3. Errors out if you try to wrap a class, because it is ambiguous whether
       or not you intended to wrap only the constructor

    The input argument can either be a context manager (in which case it must
    be a multi-shot context manager that can be directly invoked multiple times)
    or a callable that produces a context manager.
    """
    if callable(ctx) and hasattr(ctx, "__enter__"):
        raise RuntimeError(
            f"Passed in {ctx} is both callable and also a valid context manager "
            "(has __enter__), making it ambiguous which interface to use.  If you "
            "intended to pass a context manager factory, rewrite your call as "
            "context_decorator(lambda: ctx()); if you intended to pass a context "
            "manager directly, rewrite your call as context_decorator(lambda: ctx)"
        )

    if not callable(ctx):

        def ctx_factory():
            return ctx

    else:
        ctx_factory = ctx

    if inspect.isclass(func):
        raise RuntimeError(
            "Cannot decorate classes; it is ambiguous whether only the "
            "constructor or all methods should have the context manager applied; "
            "additionally, decorating a class at definition-site will prevent "
            "use of the identifier as a conventional type.  "
            "To specify which methods to decorate, decorate each of them "
            "individually."
        )

    if inspect.isgeneratorfunction(func):
        return _wrap_generator(ctx_factory, func)

    @functools.wraps(func)
    def decorate_context(*args, **kwargs):
        with ctx_factory():
            return func(*args, **kwargs)

    return decorate_context


class _DecoratorContextManager:
    """Allow a context manager to be used as a decorator."""

    def __call__(self, orig_func: F) -> F:
        if inspect.isclass(orig_func):
            warnings.warn(
                "Decorating classes is deprecated and will be disabled in "
                "future versions. You should only decorate functions or methods. "
                "To preserve the current behavior of class decoration, you can "
                "directly decorate the `__init__` method and nothing else."
            )
            func = cast(F, lambda *args, **kwargs: orig_func(*args, **kwargs))
        else:
            func = orig_func

        return cast(F, context_decorator(self.clone, func))

    def __enter__(self) -> None:
        raise NotImplementedError

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        raise NotImplementedError

    def clone(self):
        # override this method if your children class takes __init__ parameters
        return self.__class__()


def get_trace():
    """A simple debugging util to spot where a function is being called."""
    traceback.print_stack()


class _ProcessNoWarn(mp.Process):
    """A private Process class that shuts down warnings on the subprocess and controls the number of threads in the subprocess."""

    @wraps(mp.Process.__init__)
    def __init__(self, *args, num_threads=None, _start_method=None, **kwargs):
        import torchrl

        self.filter_warnings_subprocess = torchrl.filter_warnings_subprocess
        self.num_threads = num_threads
        if _start_method is not None:
            self._start_method = _start_method
        super().__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        if self.num_threads is not None:
            torch.set_num_threads(self.num_threads)
        if self.filter_warnings_subprocess:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return mp.Process.run(self, *args, **kwargs)
        return mp.Process.run(self, *args, **kwargs)


def print_directory_tree(path, indent="", display_metadata=True):
    """Prints the directory tree starting from the specified path.

    Args:
        path (str): The path of the directory to print.
        indent (str): The current indentation level for formatting.
        display_metadata (bool): if ``True``, metadata of the dir will be
            displayed too.

    """
    if display_metadata:

        def get_directory_size(path="."):
            total_size = 0

            for dirpath, _, filenames in os.walk(path):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(file_path)

            return total_size

        def format_size(size):
            # Convert size to a human-readable format
            for unit in ["B", "KB", "MB", "GB", "TB"]:
                if size < 1024.0:
                    return f"{size:.2f} {unit}"
                size /= 1024.0

        total_size_bytes = get_directory_size(path)
        formatted_size = format_size(total_size_bytes)
        logger.info(f"Directory size: {formatted_size}")

    if os.path.isdir(path):
        logger.info(indent + os.path.basename(path) + "/")
        indent += "    "
        for item in os.listdir(path):
            print_directory_tree(
                os.path.join(path, item), indent=indent, display_metadata=False
            )
    else:
        logger.info(indent + os.path.basename(path))


def _ends_with(key, match):
    if isinstance(key, str):
        return key == match
    return key[-1] == match


def _replace_last(key: NestedKey, new_ending: str) -> NestedKey:
    if isinstance(key, str):
        return new_ending
    else:
        return key[:-1] + (new_ending,)


def _append_last(key: NestedKey, new_suffix: str) -> NestedKey:
    key = unravel_key(key)
    if isinstance(key, str):
        return key + new_suffix
    else:
        return key[:-1] + (key[-1] + new_suffix,)


class _rng_decorator(_DecoratorContextManager):
    """Temporarily sets the seed and sets back the rng state when exiting."""

    def __init__(self, seed, device=None):
        self.seed = seed
        self.device = device
        self.has_cuda = torch.cuda.is_available()

    def __enter__(self):
        self._get_state()
        torch.manual_seed(self.seed)

    def _get_state(self):
        if self.has_cuda:
            if self.device is None:
                self._state = (torch.random.get_rng_state(), torch.cuda.get_rng_state())
            else:
                self._state = (
                    torch.random.get_rng_state(),
                    torch.cuda.get_rng_state(self.device),
                )

        else:
            self._state = torch.random.get_rng_state()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.has_cuda:
            torch.random.set_rng_state(self._state[0])
            if self.device is not None:
                torch.cuda.set_rng_state(self._state[1], device=self.device)
            else:
                torch.cuda.set_rng_state(self._state[1])

        else:
            torch.random.set_rng_state(self._state)


def _can_be_pickled(obj):
    try:
        pickle.dumps(obj)
        return True
    except (pickle.PickleError, AttributeError, TypeError):
        return False


def _make_ordinal_device(device: torch.device):
    if device is None:
        return device
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", index=torch.cuda.current_device())
    if device.type == "mps" and device.index is None:
        return torch.device("mps", index=0)
    return device


class _ContextManager:
    def __init__(self):
        self._mode: Any | None = None
        self._lock = threading.Lock()

    def get_mode(self) -> Any | None:
        cm = self._lock if not is_compiling() else nullcontext()
        with cm:
            return self._mode

    def set_mode(self, type: Any | None) -> None:
        cm = self._lock if not is_compiling() else nullcontext()
        with cm:
            self._mode = type


def _standardize(
    input: Tensor,
    exclude_dims: tuple[int] = (),
    mean: Tensor | None = None,
    std: Tensor | None = None,
    eps: float | None = None,
):
    """Standardizes the input tensor with the possibility of excluding specific dims from the statistics.

    Useful when processing multi-agent data to keep the agent dimensions independent.

    Args:
        input (Tensor): the input tensor to be standardized.
        exclude_dims (Tuple[int]): dimensions to exclude from the statistics, can be negative. Default: ().
        mean (Tensor): a mean to be used for standardization. Must be of shape broadcastable to input. Default: None.
        std (Tensor): a standard deviation to be used for standardization. Must be of shape broadcastable to input. Default: None.
        eps (:obj:`float`): epsilon to be used for numerical stability. Default: float32 resolution.

    """
    if eps is None:
        if input.dtype.is_floating_point:
            eps = torch.finfo(torch.float).resolution
        else:
            eps = 1e-6

    len_exclude_dims = len(exclude_dims)
    if not len_exclude_dims:
        if mean is None:
            mean = input.mean()
        else:
            # Assume dtypes are compatible
            mean = torch.as_tensor(mean, device=input.device)
        if std is None:
            std = input.std()
        else:
            # Assume dtypes are compatible
            std = torch.as_tensor(std, device=input.device)
        return (input - mean) / std.clamp_min(eps)

    input_shape = input.shape
    exclude_dims = [
        d if d >= 0 else d + len(input_shape) for d in exclude_dims
    ]  # Make negative dims positive

    if len(set(exclude_dims)) != len_exclude_dims:
        raise ValueError("Exclude dims has repeating elements")
    if any(dim < 0 or dim >= len(input_shape) for dim in exclude_dims):
        raise ValueError(
            f"exclude_dims={exclude_dims} provided outside bounds for input of shape={input_shape}"
        )
    if len_exclude_dims == len(input_shape):
        warnings.warn(
            "_standardize called but all dims were excluded from the statistics, returning unprocessed input"
        )
        return input

    included_dims = tuple(d for d in range(len(input_shape)) if d not in exclude_dims)
    if mean is None:
        mean = torch.mean(input, keepdim=True, dim=included_dims)
    if std is None:
        std = torch.std(input, keepdim=True, dim=included_dims)
    return (input - mean) / std.clamp_min(eps)


@wraps(torch.compile)
def compile_with_warmup(*args, warmup: int = 1, **kwargs):
    """Compile a model with warm-up.

    This function wraps :func:`~torch.compile` to add a warm-up phase. During the warm-up phase,
    the original model is used. After the warm-up phase, the model is compiled using
    `torch.compile`.

    Args:
        *args: Arguments to be passed to `torch.compile`.
        warmup (int): Number of calls to the model before compiling it. Defaults to 1.
        **kwargs: Keyword arguments to be passed to `torch.compile`.

    Returns:
        A callable that wraps the original model. If no model is provided, returns a
        lambda function that takes a model as input and returns the wrapped model.

    Notes:
        If no model is provided, this function returns a lambda function that can be
        used to wrap a model later. This allows for delayed compilation of the model.

    Example:
        >>> model = torch.nn.Linear(5, 3)
        >>> compiled_model = compile_with_warmup(model, warmup=10)
        >>> # First 10 calls use the original model
        >>> # After 10 calls, the model is compiled and used
    """
    if len(args):
        model = args[0]
        args = ()
    else:
        model = kwargs.pop("model", None)
    if model is None:
        return lambda model: compile_with_warmup(model, warmup=warmup, **kwargs)
    else:
        count = -1
        compiled_model = model

        @wraps(model)
        def count_and_compile(*model_args, **model_kwargs):
            nonlocal count
            nonlocal compiled_model
            count += 1
            if count == warmup:
                compiled_model = torch.compile(model, *args, **kwargs)
            return compiled_model(*model_args, **model_kwargs)

        return count_and_compile


# auto unwrap control
_DEFAULT_AUTO_UNWRAP = True
_AUTO_UNWRAP = os.environ.get("AUTO_UNWRAP_TRANSFORMED_ENV")


class set_auto_unwrap_transformed_env(_DecoratorContextManager):
    """A context manager or decorator to control whether TransformedEnv should automatically unwrap nested TransformedEnv instances.

    Args:
        mode (bool): Whether to automatically unwrap nested :class:`~torchrl.envs.TransformedEnv`
            instances. If ``False``, :class:`~torchrl.envs.TransformedEnv` will not unwrap nested instances.
            Defaults to ``True``.

    .. note:: Until v0.9, this will raise a warning if :class:`~torchrl.envs.TransformedEnv` are nested
        and the value is not set explicitly (`auto_unwrap=True` default behavior).
        You can set the value of :func:`~torchrl.envs.auto_unwrap_transformed_env`
        through:

        - The ``AUTO_UNWRAP_TRANSFORMED_ENV`` environment variable;
        - By setting ``torchrl.set_auto_unwrap_transformed_env(val: bool).set()`` at the
          beginning of your script;
        - By using ``torchrl.set_auto_unwrap_transformed_env(val: bool)`` as a context
          manager or a decorator.

    .. seealso:: :class:`~torchrl.envs.TransformedEnv`

    Examples:
        >>> with set_auto_unwrap_transformed_env(False):
        ...     env = TransformedEnv(TransformedEnv(env))
        ...     assert not isinstance(env.base_env, TransformedEnv)
        >>> @set_auto_unwrap_transformed_env(False)
        ... def my_function():
        ...     env = TransformedEnv(TransformedEnv(env))
        ...     assert not isinstance(env.base_env, TransformedEnv)
        ...     return env

    """

    def __init__(self, mode: bool) -> None:
        super().__init__()
        self.mode = mode

    def clone(self) -> set_auto_unwrap_transformed_env:
        # override this method if your children class takes __init__ parameters
        return type(self)(self.mode)

    def __enter__(self) -> None:
        self.set()

    def set(self) -> None:
        global _AUTO_UNWRAP
        self._old_mode = _AUTO_UNWRAP
        _AUTO_UNWRAP = bool(self.mode)
        # we do this such that sub-processes see the same lazy op than the main one
        os.environ["AUTO_UNWRAP_TRANSFORMED_ENV"] = str(_AUTO_UNWRAP)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global _AUTO_UNWRAP
        _AUTO_UNWRAP = self._old_mode
        os.environ["AUTO_UNWRAP_TRANSFORMED_ENV"] = str(_AUTO_UNWRAP)


def auto_unwrap_transformed_env(allow_none=False):
    """Get the current setting for automatically unwrapping TransformedEnv instances.

    Args:
        allow_none (bool, optional): If True, returns ``None`` if no setting has been
            specified. Otherwise, returns the default setting. Defaults to ``False``.

    seealso: :func:`~torchrl.set_auto_unwrap_transformed_env`

    Returns:
        bool or None: The current setting for automatically unwrapping TransformedEnv
            instances.
    """
    global _AUTO_UNWRAP  # noqa: F824
    if _AUTO_UNWRAP is None and allow_none:
        return None
    elif _AUTO_UNWRAP is None:
        return _DEFAULT_AUTO_UNWRAP
    return strtobool(_AUTO_UNWRAP) if isinstance(_AUTO_UNWRAP, str) else _AUTO_UNWRAP


def safe_is_current_stream_capturing():
    """A safe proxy to torch.cuda.is_current_stream_capturing."""
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception as error:
        warnings.warn(
            f"torch.cuda.is_current_stream_capturing() exited unexpectedly with the error message {error=}. "
            f"Returning False by default."
        )
        return False


@classmethod
def as_remote(cls, remote_config: dict[str, Any] | None = None):
    """Creates an instance of a remote ray class.

    Args:
        cls (Python Class): class to be remotely instantiated.
        remote_config (dict): the quantity of CPU cores to reserve for this class.

    Returns:
        A function that creates ray remote class instances.
    """
    import ray

    if remote_config is None:
        remote_config = {}

    remote_collector = ray.remote(**remote_config)(cls)
    remote_collector.is_remote = True
    return remote_collector


def get_ray_default_runtime_env() -> dict[str, Any]:
    """Get the default Ray runtime environment configuration for TorchRL.

    This function returns a runtime environment configuration that excludes
    large directories and files that should not be uploaded to Ray workers.
    This helps prevent issues with Ray's working_dir size limits (512MB default).

    Returns:
        dict: A dictionary containing the default runtime_env configuration with
            excludes for common large directories.

    Examples:
        >>> import ray
        >>> from torchrl._utils import get_ray_default_runtime_env
        >>> ray_init_config = {"num_cpus": 4}
        >>> ray_init_config["runtime_env"] = get_ray_default_runtime_env()
        >>> ray.init(**ray_init_config)

    Note:
        The excludes list includes:
        - Virtual environments (.venv/, venv/, etc.)
        - Test files and caches
        - Documentation builds
        - Benchmarks
        - Examples and tutorials
        - CI/CD configurations
        - IDE configurations

    """
    return {
        "excludes": [
            ".venv/",
            "venv/",
            "env/",
            "ENV/",
            "env.bak/",
            "venv.bak/",
            "test/",
            "tests/",
            "docs/",
            "benchmarks/",
            "tutorials/",
            "examples/",
            ".github/",
            ".pytest_cache/",
            ".mypy_cache/",
            ".ruff_cache/",
            "__pycache__/",
            "*.pyc",
            "*.pyo",
            "*.egg-info/",
            ".idea/",
            ".vscode/",
            "dev/",
            "main/",
            "*.html",
        ]
    }


def merge_ray_runtime_env(ray_init_config: dict[str, Any]) -> dict[str, Any]:
    """Merge user-provided ray_init_config with default runtime_env excludes.

    This function ensures that the default TorchRL runtime_env excludes are applied
    to prevent large directories from being uploaded to Ray workers, while preserving
    any user-provided configuration.

    Args:
        ray_init_config (dict): The ray init configuration dictionary to merge.

    Returns:
        dict: The merged configuration with default runtime_env excludes applied.

    Examples:
        >>> from torchrl._utils import merge_ray_runtime_env
        >>> ray_init_config = {"num_cpus": 4}
        >>> ray_init_config = merge_ray_runtime_env(ray_init_config)
        >>> ray.init(**ray_init_config)

    """
    default_runtime_env = get_ray_default_runtime_env()
    runtime_env = ray_init_config.get("runtime_env")

    # Handle None or missing runtime_env
    if runtime_env is None:
        runtime_env = {}
        ray_init_config["runtime_env"] = runtime_env
    elif not isinstance(runtime_env, dict):
        runtime_env = dict(runtime_env)
        ray_init_config["runtime_env"] = runtime_env

    # Merge excludes lists
    excludes = runtime_env.get("excludes", [])
    runtime_env["excludes"] = list(set(default_runtime_env["excludes"] + excludes))

    # Ensure env_vars exists
    if "env_vars" not in runtime_env:
        runtime_env["env_vars"] = {}
    elif not isinstance(runtime_env["env_vars"], dict):
        runtime_env["env_vars"] = dict(runtime_env["env_vars"])

    return ray_init_config


def rl_warnings():
    """Checks the status of the RL_WARNINGS env varioble."""
    return RL_WARNINGS
