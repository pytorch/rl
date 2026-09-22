# SimplicialNormalization

*class*torchrl.modules.SimplicialNormalization(*dim: int*)[[source]](../../_modules/torchrl/modules/models/tdmpc2.html#SimplicialNormalization)

Apply softmax independently to fixed-size feature simplices.

Parameters:

**dim** (*int*) - Number of features in each simplex. The last input
dimension must be divisible by `dim`.

forward(*x: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/models/tdmpc2.html#SimplicialNormalization.forward)

Normalize the last dimension of `x` in groups of `dim`.