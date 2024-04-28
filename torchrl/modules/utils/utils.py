import warnings

from torchrl.envs import Compose


def get_primers_from_module(module):
    """Get all tensordict primers from all submodules of a module."""
    primers = []

    def make_primers(submodule):
        if hasattr(submodule, "make_tensordict_primer"):
            primers.append(submodule.make_tensordict_primer())

    module.apply(make_primers)
    if not primers:
        raise warnings.warn("No primers found in the module.")
    elif len(primers) == 1:
        return primers[0]
    else:
        return Compose(primers)
