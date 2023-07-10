from typing import Optional
import warnings

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import make_functional, TensorDictModuleBase
from torch import nn


class EnsembleModule(TensorDictModuleBase):
    """Module that wraps a module and repeats it to form an ensemble.

    Args:
        module (nn.Module): The nn.module to duplicate and wrap.
        num_copies (int): The number of copies of module to make.
        parameter_init_function (Callable): A function that takes a module copy and initializes its parameters.
        expand_input (bool): Whether to expand the input TensorDict to match the number of copies. This should be
            True unless you are chaining ensemble modules together, e.g. EnsembleModule(cnn) -> EnsembleModule(mlp).
            If False, EnsembleModule(mlp) will expect the previous module(s) to have already expanded the input.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules import EnsembleModule
        >>> from tensordict import TensorDict
        >>> net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2))
        >>> mod = TensorDictModule(net, in_keys=['a'], out_keys=['b'])
        >>> ensemble = EnsembleModule(mod, num_copies=3)
        >>> data = TensorDict({'a': torch.randn(10, 4)}, batch_size=[10])
        >>> ensemble(data)
        TensorDict(
            fields={
                a: Tensor(shape=torch.Size([3, 10, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                b: Tensor(shape=torch.Size([3, 10, 2]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([3, 10]),
            device=None,
            is_shared=False)

    To stack EnsembleModules together, we should be mindful of turning off `expand_input` from the second module and on.
    
    Examples:
        >>> import torch
        >>> from tensordict.nn import TensorDictModule, TensorDictSequential
        >>> from torchrl.modules import EnsembleModule
        >>> from tensordict import TensorDict
        >>> module = TensorDictModule(torch.nn.Linear(2,3), in_keys=['bork'], out_keys=['dork'])
        >>> next_module = TensorDictModule(torch.nn.Linear(3,1), in_keys=['dork'], out_keys=['spork'])
        >>> e0 = EnsembleModule(module, num_copies=4, expand_input=True)
        >>> e1 = EnsembleModule(next_module, num_copies=4, expand_input=False)
        >>> seq = TensorDictSequential(e0, e1)
        >>> data = TensorDict({'bork': torch.randn(5,2)}, batch_size=[5])
        >>> seq(data)
        TensorDict(
            fields={
                bork: Tensor(shape=torch.Size([4, 5, 2]), device=cpu, dtype=torch.float32, is_shared=False),
                dork: Tensor(shape=torch.Size([4, 5, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                spork: Tensor(shape=torch.Size([4, 5, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([4, 5]),
            device=None,
            is_shared=False)
    """

    def __init__(
        self,
        module: TensorDictModuleBase,
        num_copies: int,
        expand_input: bool = True,
    ):
        super().__init__()
        self.in_keys = module.in_keys
        self.out_keys = module.out_keys
        params_td = make_functional(module).expand(num_copies).to_tensordict()

        self.module = module
        self.params_td = params_td
        self.ensemble_parameters = nn.ParameterList(
            list(self.params_td.values(True, True))
        )
        if expand_input:
            self.vmapped_forward = torch.vmap(self.module, (None, 0))
        else:
            self.vmapped_forward = torch.vmap(self.module, 0)

        self.reset_parameters_recursive(self.params_td)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        return self.vmapped_forward(tensordict, self.params_td)

    def reset_parameters_recursive(self, parameters: TensorDictBase = None) -> TensorDictBase:
        """Resets the parameters of all the copies of the module.

        Args:
            parameters (TensorDict): A TensorDict of parameters for self.module. The batch dimension(s) of the tensordict
                denote the number of module copies to reset.

        Returns:
            A TensorDict of pointers to the reset parameters.
        """
        if parameters is None:
            raise ValueError(
                "Ensembles are functional and require passing a TensorDict of parameters to reset_parameters_recursive"
            )
        if parameters.ndim:
            params_pointers = []
            for params_copy in parameters.unbind(0):
                self.reset_parameters_recursive(params_copy)
                params_pointers.append(params_copy)
            return torch.stack(params_pointers, -1)
        else:
            # In case the user has added other neural networks to the EnsembleModule
            # besides those in self.module
            child_mods = [
                mod
                for name, mod in self.named_children()
                if name != "module" and name != "ensemble_parameters"
            ]
            if child_mods:
                warnings.warn(
                    "EnsembleModule.reset_parameters_recursive() only resets parameters of self.module, but other parameters were detected. These parameters will not be reset."
                )
            # Reset all self.module descendant parameters
            return self.module.reset_parameters_recursive(parameters)
