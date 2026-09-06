# set_recurrent_matmul_precision

torchrl.modules.set_recurrent_matmul_precision(*mode: str | None*) → None[[source]](../../_modules/torchrl/modules/tensordict_module/_rnn_precision.html#set_recurrent_matmul_precision)

Set the process-global precision for the triton RNN backend.

Parameters:

**mode** - One of `"ieee"`, `"tf32"`, `"tf32x3"`, `"fast"`,
`"high-prec"`, `"auto"` or `None`. `"auto"` and `None`
clear the override and fall back to
[`torch.get_float32_matmul_precision()`](https://docs.pytorch.org/docs/stable/generated/torch.get_float32_matmul_precision.html#torch.get_float32_matmul_precision) (modulated by the
`TORCHRL_RNN_PRECISION` env var if set). `"fast"` and
`"high-prec"` are stored symbolically and resolve to a concrete
mode at every kernel call based on the active CUDA device.

The setting is read at every triton GRU/LSTM call, so changes take
effect immediately. Per-module `recurrent_matmul_precision=` kwargs
still override this global value.