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
import time
import traceback
import warnings
from copy import copy
from distutils.util import strtobool
from functools import wraps
from importlib import import_module
from typing import Any, Callable, cast, Dict, TypeVar, Union

import numpy as np
import torch
from packaging.version import parse
from tensordict import unravel_key

from tensordict.utils import NestedKey
from torch import multiprocessing as mp

LOGGING_LEVEL = os.environ.get("RL_LOGGING_LEVEL", "INFO")
logger = logging.getLogger("torchrl")
logger.setLevel(getattr(logging, LOGGING_LEVEL))
# Disable propagation to the root logger
logger.propagate = False
# Remove all attached handlers
while logger.hasHandlers():
    logger.removeHandler(logger.handlers[0])
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(name)s][%(levelname)s] %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

VERBOSE = strtobool(os.environ.get("VERBOSE", "0"))
_os_is_windows = sys.platform == "win32"
RL_WARNINGS = strtobool(os.environ.get("RL_WARNINGS", "1"))
if RL_WARNINGS:
    warnings.simplefilter("once", DeprecationWarning)

BATCHED_PIPE_TIMEOUT = float(os.environ.get("BATCHED_PIPE_TIMEOUT", "10000.0"))


class timeit:
    """A dirty but easy to use decorator for profiling code."""

    _REG = {}

    def __init__(self, name):
        self.name = name

    def __call__(self, fn):
        @wraps(fn)
        def decorated_fn(*args, **kwargs):
            with self:
                out = fn(*args, **kwargs)
                return out

        return decorated_fn

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        t = time.time() - self.t0
        val = self._REG.setdefault(self.name, [0.0, 0.0, 0])

        count = val[2]
        N = count + 1
        val[0] = val[0] * (count / N) + t / N
        val[1] += t
        val[2] = N

    @staticmethod
    def print(prefix=None):  # noqa: T202
        keys = list(timeit._REG)
        keys.sort()
        for name in keys:
            strings = []
            if prefix:
                strings.append(prefix)
            strings.append(
                f"{name} took {timeit._REG[name][0] * 1000:4.4} msec (total = {timeit._REG[name][1]} sec)"
            )
            logger.info(" -- ".join(strings))

    @classmethod
    def todict(cls, percall=True):
        if percall:
            return {key: val[0] for key, val in cls._REG.items()}
        return {key: val[1] for key, val in cls._REG.items()}

    @staticmethod
    def erase():
        for k in timeit._REG:
            timeit._REG[k] = [0.0, 0.0, 0]


def _check_for_faulty_process(processes):
    terminate = False
    for p in processes:
        if not p.is_alive():
            terminate = True
            for _p in processes:
                if _p.is_alive():
                    _p.terminate()
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


class implement_for:
    """A version decorator that checks the version in the environment and implements a function with the fitting one.

    If specified module is missing or there is no fitting implementation, call of the decorated function
    will lead to the explicit error.
    In case of intersected ranges, last fitting implementation is used.

    This wrapper also works to implement different backends for a same function (eg. gym vs gymnasium,
    numpy vs jax-numpy etc).

    Args:
        module_name (str or callable): version is checked for the module with this
            name (e.g. "gym"). If a callable is provided, it should return the
            module.
        from_version: version from which implementation is compatible. Can be open (None).
        to_version: version from which implementation is no longer compatible. Can be open (None).

    Keyword Args:
        class_method (bool, optional): if ``True``, the function will be written as a class method.
            Defaults to ``False``.

    Examples:
        >>> @implement_for("gym", "0.13", "0.14")
        >>> def fun(self, x):
        ...     # Older gym versions will return x + 1
        ...     return x + 1
        ...
        >>> @implement_for("gym", "0.14", "0.23")
        >>> def fun(self, x):
        ...     # More recent gym versions will return x + 2
        ...     return x + 2
        ...
        >>> @implement_for(lambda: import_module("gym"), "0.23", None)
        >>> def fun(self, x):
        ...     # More recent gym versions will return x + 2
        ...     return x + 2
        ...
        >>> @implement_for("gymnasium")
        >>> def fun(self, x):
        ...     # If gymnasium is to be used instead of gym, x+3 will be returned
        ...     return x + 3
        ...

        This indicates that the function is compatible with gym 0.13+, but doesn't with gym 0.14+.
    """

    # Stores pointers to fitting implementations: dict[func_name] = func_pointer
    _implementations = {}
    _setters = []
    _cache_modules = {}

    def __init__(
        self,
        module_name: Union[str, Callable],
        from_version: str = None,
        to_version: str = None,
        *,
        class_method: bool = False,
    ):
        self.module_name = module_name
        self.from_version = from_version
        self.to_version = to_version
        self.class_method = class_method
        implement_for._setters.append(self)

    @staticmethod
    def check_version(version: str, from_version: str | None, to_version: str | None):
        version = parse(".".join([str(v) for v in parse(version).release]))
        return (from_version is None or version >= parse(from_version)) and (
            to_version is None or version < parse(to_version)
        )

    @staticmethod
    def get_class_that_defined_method(f):
        """Returns the class of a method, if it is defined, and None otherwise."""
        out = f.__globals__.get(f.__qualname__.split(".")[0], None)
        return out

    @classmethod
    def get_func_name(cls, fn):
        # produces a name like torchrl.module.Class.method or torchrl.module.function
        fn_str = str(fn).split(".")
        if fn_str[0].startswith("<bound method "):
            first = fn_str[0][len("<bound method ") :]
        elif fn_str[0].startswith("<function "):
            first = fn_str[0][len("<function ") :]
        else:
            raise RuntimeError(f"Unkown func representation {fn}")
        last = fn_str[1:]
        if last:
            first = [first]
            last[-1] = last[-1].split(" ")[0]
        else:
            last = [first.split(" ")[0]]
            first = []
        return ".".join([fn.__module__] + first + last)

    def _get_cls(self, fn):
        cls = self.get_class_that_defined_method(fn)
        if cls is None:
            # class not yet defined
            return
        if cls.__class__.__name__ == "function":
            cls = inspect.getmodule(fn)
        return cls

    def module_set(self):
        """Sets the function in its module, if it exists already."""
        prev_setter = type(self)._implementations.get(self.get_func_name(self.fn), None)
        if prev_setter is not None:
            prev_setter.do_set = False
        type(self)._implementations[self.get_func_name(self.fn)] = self
        cls = self.get_class_that_defined_method(self.fn)
        if cls is not None:
            if cls.__class__.__name__ == "function":
                cls = inspect.getmodule(self.fn)
        else:
            # class not yet defined
            return
        if self.class_method:
            setattr(cls, self.fn.__name__, classmethod(self.fn))
        else:
            setattr(cls, self.fn.__name__, self.fn)

    @classmethod
    def import_module(cls, module_name: Union[Callable, str]) -> str:
        """Imports module and returns its version."""
        if not callable(module_name):
            module = cls._cache_modules.get(module_name, None)
            if module is None:
                if module_name in sys.modules:
                    sys.modules[module_name] = module = import_module(module_name)
                else:
                    cls._cache_modules[module_name] = module = import_module(
                        module_name
                    )
        else:
            module = module_name()
        return module.__version__

    _lazy_impl = collections.defaultdict(list)

    def _delazify(self, func_name):
        out = None
        for local_call in implement_for._lazy_impl[func_name]:
            out = local_call()
        return out

    def __call__(self, fn):
        # function names are unique
        self.func_name = self.get_func_name(fn)
        self.fn = fn
        implement_for._lazy_impl[self.func_name].append(self._call)

        @wraps(fn)
        def _lazy_call_fn(*args, **kwargs):
            # first time we call the function, we also do the replacement.
            # This will cause the imports to occur only during the first call to fn

            result = self._delazify(self.func_name)(*args, **kwargs)
            return result

        if self.class_method:
            return classmethod(_lazy_call_fn)

        return _lazy_call_fn

    def _call(self):

        # If the module is missing replace the function with the mock.
        fn = self.fn
        func_name = self.func_name
        implementations = implement_for._implementations

        @wraps(fn)
        def unsupported(*args, **kwargs):
            raise ModuleNotFoundError(
                f"Supported version of '{func_name}' has not been found."
            )

        self.do_set = False
        # Return fitting implementation if it was encountered before.
        if func_name in implementations:
            try:
                # check that backends don't conflict
                version = self.import_module(self.module_name)
                if self.check_version(version, self.from_version, self.to_version):
                    if VERBOSE:
                        module = import_module(self.module_name)
                        warnings.warn(
                            f"Got multiple backends for {func_name}. "
                            f"Using the last queried ({module} with version {version})."
                        )
                    self.do_set = True
                if not self.do_set:
                    return implementations[func_name].fn
            except ModuleNotFoundError:
                # then it's ok, there is no conflict
                return implementations[func_name].fn
        else:
            try:
                version = self.import_module(self.module_name)
                if self.check_version(version, self.from_version, self.to_version):
                    self.do_set = True
            except ModuleNotFoundError:
                return unsupported
        if self.do_set:
            self.module_set()
            return fn
        return unsupported

    @classmethod
    def reset(cls, setters_dict: Dict[str, implement_for] = None):
        """Resets the setters in setter_dict.

        ``setter_dict`` is a copy of implementations. We just need to iterate through its
        values and call :meth:`~.module_set` for each.

        """
        if VERBOSE:
            logger.info("resetting implement_for")
        if setters_dict is None:
            setters_dict = copy(cls._implementations)
        for setter in setters_dict.values():
            setter.module_set()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"module_name={self.module_name}({self.from_version, self.to_version}), "
            f"fn_name={self.fn.__name__}, cls={self._get_cls(self.fn)}, is_set={self.do_set})"
        )


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
        method = getattr(decorated_class, name)
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
