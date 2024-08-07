# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings


def get_primers_from_module(module):
    """Get all tensordict primers from all submodules of a module.

    This method is useful for retrieving primers from modules that are contained within a
    parent module.

    Args:
        module (torch.nn.Module): The parent module.

    Returns:
        TensorDictPrimer: A TensorDictPrimer Transform.

    Example:
        >>> from torchrl.modules.utils import get_primers_from_module
        >>> from torchrl.modules import GRUModule, MLP
        >>> from tensordict.nn import TensorDictModule, TensorDictSequential
        >>> # Define a GRU module
        >>> gru_module = GRUModule(
        ...     input_size=10,
        ...     hidden_size=10,
        ...     num_layers=1,
        ...     in_keys=["input", "recurrent_state", "is_init"],
        ...     out_keys=["features", ("next", "recurrent_state")],
        ... )
        >>> # Define a head module
        >>> head = TensorDictModule(
        ...     MLP(
        ...         in_features=10,
        ...         out_features=10,
        ...         num_cells=[],
        ...     ),
        ...     in_keys=["features"],
        ...     out_keys=["output"],
        ... )
        >>> # Create a sequential model
        >>> model = TensorDictSequential(gru_module, head)
        >>> # Retrieve primers from the model
        >>> primers = get_primers_from_module(model)
        >>> print(primers)

        TensorDictPrimer(primers=Composite(
            recurrent_state: UnboundedContinuous(
                shape=torch.Size([1, 10]),
                space=None,
                device=cpu,
                dtype=torch.float32,
                domain=continuous), device=None, shape=torch.Size([])), default_value={'recurrent_state': 0.0}, random=None)

    """
    primers = []

    def make_primers(submodule):
        if hasattr(submodule, "make_tensordict_primer"):
            primers.append(submodule.make_tensordict_primer())

    module.apply(make_primers)
    if not primers:
        warnings.warn("No primers found in the module.")
        return
    elif len(primers) == 1:
        return primers[0]
    else:
        from torchrl.envs.transforms import Compose

        return Compose(*primers)
