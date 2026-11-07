# TargetNetUpdaterHook

*class*torchrl.trainers.TargetNetUpdaterHook(*target_params_updater: TargetNetUpdater*)[[source]](../../_modules/torchrl/trainers/trainers.html#TargetNetUpdaterHook)

A hook for target parameters update.

Examples

```
>>> # define a loss module
>>> loss_module = SACLoss(actor_network, qvalue_network)
>>> # define a target network updater
>>> target_net_updater = SoftUpdate(loss_module)
>>> # define a target network updater hook
>>> target_net_updater_hook = TargetNetUpdaterHook(target_net_updater)
>>> # register the target network updater hook
>>> trainer.register_op("post_optim", target_net_updater_hook)
```

register(*trainer: [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)*, *name: str*)[[source]](../../_modules/torchrl/trainers/trainers.html#TargetNetUpdaterHook.register)

Registers the hook in the trainer at a default location.

Parameters:

- **trainer** ([*Trainer*](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)) - the trainer where the hook must be registered.
- **name** (*str*) - the name of the hook.

Note

To register the hook at another location than the default, use
`register_op()`.