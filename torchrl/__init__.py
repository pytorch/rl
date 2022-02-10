from warnings import warn

from torch import multiprocessing as mp

from ._extension import _init_extension

_init_extension()

# if not HAS_OPS:
#     print("could not load C++ libraries")

try:
    mp.set_start_method("spawn")
except RuntimeError as err:
    if str(err).startswith("context has already been set"):
        mp_start_method = mp.get_start_method()
        if mp_start_method != "spawn":
            warn(
                f"failed to set start method to spawn, and current start method for mp is {mp_start_method}."
            )
