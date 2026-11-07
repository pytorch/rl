# SelectKeys

*class*torchrl.trainers.SelectKeys(*keys: Sequence[str]*)[[source]](../../_modules/torchrl/trainers/trainers.html#SelectKeys)

Selects keys in a TensorDict batch.

Parameters:

**keys** (*iterable**of**strings*) - keys to be selected in the tensordict.

Examples

```
>>> trainer = make_trainer()
>>> key1 = "first key"
>>> key2 = "second key"
>>> td = TensorDict(
... {
... key1: torch.randn(3),
... key2: torch.randn(3),
... },
... [],
... )
>>> trainer.register_op("batch_process", SelectKeys([key1]))
>>> td_out = trainer._process_batch_hook(td)
>>> assert key1 in td_out.keys()
>>> assert key2 not in td_out.keys()
```

register(*trainer*, *name='select_keys'*) → None[[source]](../../_modules/torchrl/trainers/trainers.html#SelectKeys.register)

Registers the hook in the trainer at a default location.

Parameters:

- **trainer** ([*Trainer*](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)) - the trainer where the hook must be registered.
- **name** (*str*) - the name of the hook.

Note

To register the hook at another location than the default, use
`register_op()`.