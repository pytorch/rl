# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn

from torchrl.data.tensor_specs import Composite
from torchrl.modules.tensordict_module.common import SafeModule


class SafeSequential(TensorDictSequential, SafeModule):
    """A safe sequence of TensorDictModules.

    Similarly to :obj:`nn.Sequence` which passes a tensor through a chain of mappings that read and write a single tensor
    each, this module will read and write over a tensordict by querying each of the input modules.
    When calling a :obj:`TensorDictSequencial` instance with a functional module, it is expected that the parameter lists (and
    buffers) will be concatenated in a single list.

    Args:
         modules (iterable of TensorDictModules): ordered sequence of TensorDictModule instances to be run sequentially.
         partial_tolerant (bool, optional): if ``True``, the input tensordict can miss some of the input keys.
            If so, the only module that will be executed are those who can be executed given the keys that
            are present.
            Also, if the input tensordict is a lazy stack of tensordicts AND if partial_tolerant is ``True`` AND if the
            stack does not have the required keys, then SafeSequential will scan through the sub-tensordicts
            looking for those that have the required keys, if any.

    TensorDictSequence supports functional, modular and vmap coding:
    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import Composite, Unbounded
        >>> from torchrl.modules import TanhNormal, SafeSequential, TensorDictModule, NormalParamExtractor
        >>> from torchrl.modules.tensordict_module import SafeProbabilisticModule
        >>> td = TensorDict({"input": torch.randn(3, 4)}, [3,])
        >>> spec1 = Composite(hidden=Unbounded(4), loc=None, scale=None)
        >>> net1 = nn.Sequential(torch.nn.Linear(4, 8), NormalParamExtractor())
        >>> module1 = TensorDictModule(net1, in_keys=["input"], out_keys=["loc", "scale"])
        >>> td_module1 = SafeProbabilisticModule(
        ...     module=module1,
        ...     spec=spec1,
        ...     in_keys=["loc", "scale"],
        ...     out_keys=["hidden"],
        ...     distribution_class=TanhNormal,
        ...     return_log_prob=True,
        ... )
        >>> spec2 = Unbounded(8)
        >>> module2 = torch.nn.Linear(4, 8)
        >>> td_module2 = TensorDictModule(
        ...    module=module2,
        ...    spec=spec2,
        ...    in_keys=["hidden"],
        ...    out_keys=["output"],
        ...    )
        >>> td_module = SafeSequential(td_module1, td_module2)
        >>> params = TensorDict.from_module(td_module)
        >>> with params.to_module(td_module):
        ...     td_module(td)
        >>> print(td)
        TensorDict(
            fields={
                hidden: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                input: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                loc: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                output: Tensor(torch.Size([3, 8]), dtype=torch.float32),
                sample_log_prob: Tensor(torch.Size([3, 1]), dtype=torch.float32),
                scale: Tensor(torch.Size([3, 4]), dtype=torch.float32)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
        >>> # The module spec aggregates all the input specs:
        >>> print(td_module.spec)
        Composite(
            hidden: UnboundedContinuous(
                shape=torch.Size([4]), space=None, device=cpu, dtype=torch.float32, domain=continuous),
            loc: None,
            scale: None,
            output: UnboundedContinuous(
                shape=torch.Size([8]), space=None, device=cpu, dtype=torch.float32, domain=continuous))

    In the vmap case:
        >>> from torch import vmap
        >>> params = params.expand(4, *params.shape)
        >>> td_vmap = vmap(td_module, (None, 0))(td, params)
        >>> print(td_vmap)
        TensorDict(
            fields={
                hidden: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                input: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                loc: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                output: Tensor(torch.Size([4, 3, 8]), dtype=torch.float32),
                sample_log_prob: Tensor(torch.Size([4, 3, 1]), dtype=torch.float32),
                scale: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32)},
            batch_size=torch.Size([4, 3]),
            device=None,
            is_shared=False)

    """

    module: nn.ModuleList

    def __init__(
        self,
        *modules: TensorDictModule,
        partial_tolerant: bool = False,
    ):
        self.partial_tolerant = partial_tolerant

        in_keys, out_keys = self._compute_in_and_out_keys(modules)

        spec = Composite()
        for module in modules:
            try:
                spec.update(module.spec)
            except AttributeError:
                spec.update(Composite({key: None for key in module.out_keys}))

        super(TensorDictSequential, self).__init__(
            spec=spec,
            module=nn.ModuleList(list(modules)),
            in_keys=in_keys,
            out_keys=out_keys,
        )
