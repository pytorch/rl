# set_recurrent_mode

*class*torchrl.modules.set_recurrent_mode(*mode: bool | Literal['recurrent', 'sequential'] | None = True*)[[source]](../../_modules/torchrl/modules/tensordict_module/rnn.html#set_recurrent_mode)

Context manager for setting RNNs recurrent mode.

Parameters:

**mode** (*bool**,**"recurrent"**or**"sequential"*) - the recurrent mode to be used within the context manager.
"recurrent" leads to mode=True and "sequential" leads to mode=False.
An RNN executed with recurrent_mode "on" assumes that the data comes in time batches, otherwise
it is assumed that each data element in a tensordict is independent of the others.
The default value of this context manager is `True`.
The default recurrent mode is `None`, i.e., the default recurrent mode of the RNN is used
(see [`LSTMModule`](torchrl.modules.LSTMModule.html#torchrl.modules.LSTMModule) and [`GRUModule`](torchrl.modules.GRUModule.html#torchrl.modules.GRUModule) constructors).

See also

recurrent_mode`.

Note

All of TorchRL methods are decorated with `set_recurrent_mode(True)` by default.

When to use which mode:

- **Sequential** (default, `mode=False`): inside collectors, where
the policy is called step-by-step and the hidden state from the
previous step is fed back through the tensordict.
- **Recurrent** (`mode=True`): inside loss / advantage computation
(e.g. GAE) where a full `(B, T, ...)` batch is replayed and you
want the RNN to process the time dim in a single call. This is the
mode that activates the multi-trajectory split inside
[`LSTMModule.forward()`](torchrl.modules.LSTMModule.html#torchrl.modules.LSTMModule.forward).

See the [Recurrent state lifecycle](../recurrent_state_lifecycle.html#ref-recurrent-state-lifecycle)
guide for a full walkthrough of when each mode fires.